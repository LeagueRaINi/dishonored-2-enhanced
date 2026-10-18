//! Bindless material textures: every decoded map uploaded in its stored format (BC7 sRGB
//! colour maps decode to linear in the sampler, BC5 SNORM normals, BC4/R8 masks) into one
//! shader-visible SRV heap. Slot 0 is a 1x1 white RGBA8 texture (the fallback for missing
//! or rejected maps); `textures[i]` lives in slot `i + 1`. Kernels clamp every index to
//! the slot count (`count`) before indexing the table.

use std::mem::ManuallyDrop;

use void_assets::texture::{Mip, TextureData};
use void_formats::bim::Format;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use crate::gpu::Gpu;

/// DXGI format and storage block geometry (block edge in texels, bytes per block).
pub fn dxgi_format(f: Format) -> (DXGI_FORMAT, u32, u32) {
    match f {
        Format::R8Unorm => (DXGI_FORMAT_R8_UNORM, 1, 1),
        Format::R16Unorm => (DXGI_FORMAT_R16_UNORM, 1, 2),
        Format::A8Unorm => (DXGI_FORMAT_A8_UNORM, 1, 1),
        Format::Rg8Unorm => (DXGI_FORMAT_R8G8_UNORM, 1, 2),
        Format::Rg8Snorm => (DXGI_FORMAT_R8G8_SNORM, 1, 2),
        Format::Rgba8Unorm => (DXGI_FORMAT_R8G8B8A8_UNORM, 1, 4),
        Format::Rgba8UnormSrgb => (DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, 1, 4),
        Format::Bc4Unorm => (DXGI_FORMAT_BC4_UNORM, 4, 8),
        Format::Bc4Snorm => (DXGI_FORMAT_BC4_SNORM, 4, 8),
        Format::Bc5Unorm => (DXGI_FORMAT_BC5_UNORM, 4, 16),
        Format::Bc5Snorm => (DXGI_FORMAT_BC5_SNORM, 4, 16),
        Format::Bc6hUf16 => (DXGI_FORMAT_BC6H_UF16, 4, 16),
        Format::Bc6hSf16 => (DXGI_FORMAT_BC6H_SF16, 4, 16),
        Format::Bc7Unorm => (DXGI_FORMAT_BC7_UNORM, 4, 16),
        Format::Bc7UnormSrgb => (DXGI_FORMAT_BC7_UNORM_SRGB, 4, 16),
    }
}

/// Checks a mip chain against the D3D12 layout (halving extents, block-aligned top mip,
/// exact byte sizes) before anything is allocated.
pub fn validate_mips(format: Format, mips: &[Mip]) -> Result<(), String> {
    let (_, block, block_bytes) = dxgi_format(format);
    let top = mips.first().ok_or("no mips")?;
    if block > 1 && (top.width % block != 0 || top.height % block != 0) {
        return Err(format!(
            "{}x{} top mip is not block aligned",
            top.width, top.height
        ));
    }
    for (i, m) in mips.iter().enumerate() {
        let expect = ((top.width >> i).max(1), (top.height >> i).max(1));
        if (m.width, m.height) != expect {
            return Err(format!(
                "mip {i}: {}x{}, expected {}x{}",
                m.width, m.height, expect.0, expect.1
            ));
        }
        let bytes = m.width.div_ceil(block) as usize
            * block_bytes as usize
            * m.height.div_ceil(block) as usize;
        if m.data.len() != bytes {
            return Err(format!(
                "mip {i} ({}x{}): {} bytes, expected {bytes}",
                m.width,
                m.height,
                m.data.len()
            ));
        }
    }
    Ok(())
}

/// Mean of the decoded texels of a BC4/BC5/BC7 mip, per channel in the format's own
/// range (unorm [0, 1], snorm [-1, 1]); `None` for other formats.
fn block_mean(format: Format, m: &Mip) -> Option<[f32; 4]> {
    let (_, block, block_bytes) = dxgi_format(format);
    if block != 4 {
        return None;
    }
    let mut sum = [0f64; 4];
    let mut n = 0f64;
    for b in m.data.chunks_exact(block_bytes as usize) {
        let mut px = [[0f32; 4]; 16];
        match format {
            Format::Bc4Unorm | Format::Bc4Snorm => {
                let mut r = [0f32; 16];
                bcdec_rs::bc4_float(b, &mut r, 4, format == Format::Bc4Snorm);
                for (p, v) in px.iter_mut().zip(r) {
                    *p = [v, 0.0, 0.0, 1.0];
                }
            }
            Format::Bc5Unorm | Format::Bc5Snorm => {
                let mut rg = [0f32; 32];
                bcdec_rs::bc5_float(b, &mut rg, 8, format == Format::Bc5Snorm);
                for (k, p) in px.iter_mut().enumerate() {
                    *p = [rg[2 * k], rg[2 * k + 1], 0.0, 1.0];
                }
            }
            Format::Bc7Unorm | Format::Bc7UnormSrgb => {
                let mut rgba = [0u8; 64];
                bcdec_rs::bc7(b, &mut rgba, 16);
                for (k, p) in px.iter_mut().enumerate() {
                    *p = std::array::from_fn(|c| f32::from(rgba[4 * k + c]) / 255.0);
                }
            }
            _ => return None,
        }
        for p in px {
            for c in 0..4 {
                sum[c] += f64::from(p[c]);
            }
            n += 1.0;
        }
    }
    (n > 0.0).then(|| sum.map(|s| (s / n) as f32))
}

/// One block of constant colour `v` (format range as [`block_mean`]).
fn constant_block(format: Format, v: [f32; 4]) -> Option<Vec<u8>> {
    let unorm = |x: f32| (x.clamp(0.0, 1.0) * 255.0).round() as u8;
    let snorm = |x: f32| ((x.clamp(-1.0, 1.0) * 127.0).round() as i8) as u8;
    // BC4 with red0 == red1 and all indices 0 decodes to red0 everywhere.
    let bc4 = |r: u8| [r, r, 0, 0, 0, 0, 0, 0];
    Some(match format {
        Format::Bc4Unorm => bc4(unorm(v[0])).to_vec(),
        Format::Bc4Snorm => bc4(snorm(v[0])).to_vec(),
        Format::Bc5Unorm => [bc4(unorm(v[0])), bc4(unorm(v[1]))].concat(),
        Format::Bc5Snorm => [bc4(snorm(v[0])), bc4(snorm(v[1]))].concat(),
        Format::Bc7Unorm | Format::Bc7UnormSrgb => {
            // Mode 6: 7-bit RGBA endpoints + one p-bit each, all indices 0 -> endpoint 0
            // = (c7 << 1) | p0 (p0 = 0: even values, error <= 1/255).
            let mut bits: u128 = 1 << 6;
            let mut pos = 7;
            for &x in &v {
                let c7 = u128::from(unorm(x) >> 1);
                bits |= c7 << pos;
                bits |= c7 << (pos + 7);
                pos += 14;
            }
            bits.to_le_bytes().to_vec()
        }
        _ => return None,
    })
}

/// Completes a block-compressed mip chain down to 1x1: full chains are stored down to one
/// block row or column (O: e.g. 64x64 -> 4x4 in 5 mips, 32x64 -> 4x8). The added levels
/// hold the last stored mip's mean colour, so the coarsest LOD (triangles without UVs,
/// `scene::LOD_BIAS_NO_UV`) reads the map's mean instead of a corner texel. Chains whose
/// last level is larger than one block on both axes are left as stored: `$nomips` images
/// (one 64..2048 texel level, e.g. `roots_lod2_green_d$nomips`) are sampled at their only
/// level by the engine; a mean-colour level below them turned every LOD > 0 into a flat
/// colour (and alpha-tested cards into mean alpha).
pub fn complete_chain(format: Format, mips: &[Mip]) -> Option<Vec<Mip>> {
    let last = mips.last()?;
    if (last.width == 1 && last.height == 1) || last.width.min(last.height) > 4 {
        return None;
    }
    let (_, block, _) = dxgi_format(format);
    let blk = constant_block(format, block_mean(format, last)?)?;
    let mut out = mips.to_vec();
    let (mut w, mut h) = (last.width, last.height);
    while w > 1 || h > 1 {
        w = (w >> 1).max(1);
        h = (h >> 1).max(1);
        let blocks = (w.div_ceil(block) * h.div_ceil(block)) as usize;
        out.push(Mip {
            width: w,
            height: h,
            data: blk.repeat(blocks),
        });
    }
    Some(out)
}

/// Index of the first stored mip whose longer edge is at most `cap`, else the last one (the
/// decoder's rule, `void_assets::texture::load`): a cap drops the mips before it.
pub fn cap_start(mips: &[Mip], cap: u32) -> usize {
    mips.iter()
        .position(|m| m.width.max(m.height) <= cap)
        .unwrap_or(mips.len().saturating_sub(1))
}

/// Drops the mips above `cap` (the same chain as decoding at `cap`; single-level images
/// keep their only level).
pub fn cap_mips(t: &mut TextureData, cap: u32) {
    let s = cap_start(&t.mips, cap);
    t.mips.drain(..s);
}

/// Levels of the chain [`complete_chain`] uploads for `mips` (without building it).
fn uploaded_levels(format: Format, mips: &[Mip]) -> usize {
    let Some(last) = mips.last() else {
        return 0;
    };
    if (last.width == 1 && last.height == 1)
        || last.width.max(last.height) > 4
        || constant_block(format, [0.0; 4]).is_none()
    {
        return mips.len();
    }
    mips.len() + last.width.max(last.height).ilog2() as usize
}

/// Bytes of `t` uploaded with the cap `cap`: (device allocation by
/// `GetResourceAllocationInfo`, staging data). (0, 0) for textures [`TextureSet::upload`]
/// rejects. CPU only (a size query on the device).
pub fn texture_bytes(gpu: &Gpu, t: &TextureData, cap: u32) -> (u64, u64) {
    let mips = &t.mips[cap_start(&t.mips, cap)..];
    if validate_mips(t.format, mips).is_err() {
        return (0, 0);
    }
    let (format, _, _) = dxgi_format(t.format);
    let top = &mips[0];
    let desc = tex2d_desc(
        format,
        top.width,
        top.height,
        uploaded_levels(t.format, mips) as u16,
    );
    // SAFETY: plain size query for a valid descriptor.
    let alloc = unsafe {
        gpu.device
            .GetResourceAllocationInfo(0, std::slice::from_ref(&desc))
    }
    .SizeInBytes;
    (alloc, mips.iter().map(|m| m.data.len() as u64).sum())
}

pub struct TextureSet {
    pub(crate) heap: ID3D12DescriptorHeap,
    pub gpu_start: D3D12_GPU_DESCRIPTOR_HANDLE,
    cpu_start: D3D12_CPU_DESCRIPTOR_HANDLE,
    increment: u32,
    _resources: Vec<ID3D12Resource>,
    /// Slots in the heap (textures + the white fallback + the reserve).
    pub count: u32,
    /// First slot of the reserve for textures added later ([`TextureSet::upload_slot`]).
    pub reserve_start: u32,
    /// Sum of the textures' allocation sizes (`GetResourceAllocationInfo`).
    pub bytes: u64,
    /// Textures replaced by the white fallback: (name, reason).
    pub rejected: Vec<(String, String)>,
    pub upload_submits: usize,
    /// Mip chains completed to 1x1 by [`complete_chain`].
    pub completed_chains: usize,
}

fn tex2d_desc(format: DXGI_FORMAT, w: u32, h: u32, mips: u16) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: u64::from(w),
        Height: h,
        DepthOrArraySize: 1,
        MipLevels: mips,
        Format: format,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        ..Default::default()
    }
}

fn upload_desc(size: u64) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size.max(256),
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    }
}

fn committed(
    gpu: &Gpu,
    heap: D3D12_HEAP_TYPE,
    desc: &D3D12_RESOURCE_DESC,
    state: D3D12_RESOURCE_STATES,
) -> Result<ID3D12Resource, String> {
    let mut res: Option<ID3D12Resource> = None;
    // SAFETY: valid descriptors and out pointer.
    unsafe {
        gpu.device.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES {
                Type: heap,
                ..Default::default()
            },
            D3D12_HEAP_FLAG_NONE,
            desc,
            state,
            None,
            &mut res,
        )
    }
    .map_err(|e| format!("CreateCommittedResource: {e}{}", gpu.removed_reason()))?;
    res.ok_or_else(|| "no resource".into())
}

/// Creates the texture and its staging buffer and records the copies into the open list.
/// Returns (texture, staging, allocation bytes).
fn record(
    gpu: &Gpu,
    format: DXGI_FORMAT,
    block: u32,
    block_bytes: u32,
    mips: &[Mip],
) -> Result<(ID3D12Resource, ID3D12Resource, u64), String> {
    let top = &mips[0];
    let n = mips.len() as u32;
    let desc = tex2d_desc(format, top.width, top.height, n as u16);
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); n as usize];
    let mut rows = vec![0u32; n as usize];
    let mut total = 0u64;
    // SAFETY: output arrays hold `n` entries.
    unsafe {
        gpu.device.GetCopyableFootprints(
            &desc,
            0,
            n,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            None,
            Some(&mut total),
        );
    }
    // SAFETY: plain size query for a valid descriptor.
    let alloc = unsafe {
        gpu.device
            .GetResourceAllocationInfo(0, std::slice::from_ref(&desc))
    }
    .SizeInBytes;
    let tex = committed(
        gpu,
        D3D12_HEAP_TYPE_DEFAULT,
        &desc,
        D3D12_RESOURCE_STATE_COPY_DEST,
    )?;
    let up = committed(
        gpu,
        D3D12_HEAP_TYPE_UPLOAD,
        &upload_desc(total),
        D3D12_RESOURCE_STATE_GENERIC_READ,
    )?;
    let mut ptr = std::ptr::null_mut();
    // SAFETY: the staging buffer holds `total` bytes; every row written lies inside its
    // subresource footprint (offset + row * pitch + row bytes <= total), and the source
    // rows were size-checked by `validate_mips`.
    unsafe {
        up.Map(0, None, Some(&mut ptr))
            .map_err(|e| format!("Map: {e}"))?;
        for (i, m) in mips.iter().enumerate() {
            let fp = footprints[i];
            let row = m.width.div_ceil(block) as usize * block_bytes as usize;
            let pitch = fp.Footprint.RowPitch as usize;
            for r in 0..rows[i] as usize {
                std::ptr::copy_nonoverlapping(
                    m.data.as_ptr().add(r * row),
                    ptr.cast::<u8>().add(fp.Offset as usize + r * pitch),
                    row,
                );
            }
        }
        up.Unmap(0, None);
    }
    for (i, fp) in footprints.iter().enumerate() {
        let dst = D3D12_TEXTURE_COPY_LOCATION {
            pResource: ManuallyDrop::new(Some(tex.clone())),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: i as u32,
            },
        };
        let src = D3D12_TEXTURE_COPY_LOCATION {
            pResource: ManuallyDrop::new(Some(up.clone())),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: *fp,
            },
        };
        // SAFETY: both resources outlive the submission (the caller keeps `up`).
        unsafe { gpu.list.CopyTextureRegion(&dst, 0, 0, 0, &src, None) };
        drop(ManuallyDrop::into_inner(dst.pResource));
        drop(ManuallyDrop::into_inner(src.pResource));
    }
    gpu.transition_resource(
        &tex,
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
    );
    Ok((tex, up, alloc))
}

impl TextureSet {
    /// The shader-visible CBV/SRV/UAV heap holding the `count` texture SRVs from
    /// `gpu_start` (bind it with `SetDescriptorHeaps` before using the table).
    pub fn heap(&self) -> &ID3D12DescriptorHeap {
        &self.heap
    }

    fn write_srv(
        &self,
        gpu: &Gpu,
        slot: u32,
        res: &ID3D12Resource,
        format: DXGI_FORMAT,
        mips: u32,
    ) {
        let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: format,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV {
                    MipLevels: mips,
                    ..Default::default()
                },
            },
        };
        let h = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: self.cpu_start.ptr + slot.min(self.count - 1) as usize * self.increment as usize,
        };
        // SAFETY: the slot is clamped below `count`; the handle lies inside the heap.
        unsafe { gpu.device.CreateShaderResourceView(res, Some(&desc), h) };
    }

    /// Uploads `t` (blocking: one submit, after which no work of `gpu` is in flight) and
    /// then points reserve slot `slot` at it, so no executing dispatch sees the descriptor
    /// change (a reserve slot is referenced by no material before this call). Returns the
    /// texture (keep it while a material uses the slot) and its allocation bytes. An
    /// invalid mip chain is an error (the slot stays white).
    pub fn upload_slot(
        &self,
        gpu: &mut Gpu,
        slot: u32,
        t: &TextureData,
    ) -> Result<(ID3D12Resource, u64), String> {
        if slot < self.reserve_start || slot >= self.count {
            return Err(format!(
                "texture slot {slot} outside the reserve {}..{}",
                self.reserve_start, self.count
            ));
        }
        let (format, block, block_bytes) = dxgi_format(t.format);
        validate_mips(t.format, &t.mips).map_err(|e| format!("{}: {e}", t.name))?;
        let full = complete_chain(t.format, &t.mips);
        let mips = full.as_deref().unwrap_or(&t.mips);
        validate_mips(t.format, mips).map_err(|e| format!("{}: completed chain: {e}", t.name))?;
        let (tex, up, bytes) = match record(gpu, format, block, block_bytes, mips) {
            Ok(r) => r,
            Err(e) => {
                let _ = gpu.discard();
                return Err(e);
            }
        };
        gpu.submit()?;
        drop(up);
        self.write_srv(gpu, slot, &tex, format, mips.len() as u32);
        Ok((tex, bytes))
    }

    /// Points reserve slot `slot` back at the white texture (once no material uses it).
    pub fn clear_slot(&self, gpu: &Gpu, slot: u32) {
        if slot >= self.reserve_start && slot < self.count {
            self.write_srv(
                gpu,
                slot,
                &self._resources[0],
                DXGI_FORMAT_R8G8B8A8_UNORM,
                1,
            );
        }
    }

    /// Uploads `textures` (slot `i + 1`), submitting whenever the pending staging data reach
    /// the batch size: at most `batch_bytes`, scaled by the measured GPU time of the previous
    /// submit towards [`crate::accel::SUBMIT_TARGET_MS`] (short submits; staging memory
    /// bounded). `reserve` more slots after them show the white texture until
    /// [`TextureSet::upload_slot`] fills them. On an error the open list is discarded (it
    /// references staging buffers that are freed on return).
    pub fn upload(
        gpu: &mut Gpu,
        textures: &[TextureData],
        batch_bytes: u64,
        reserve: u32,
    ) -> Result<Self, String> {
        let count = textures.len() as u32 + 1 + reserve;
        let heap: ID3D12DescriptorHeap =
            // SAFETY: valid descriptor.
            unsafe {
                gpu.device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: count,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })
            }
            .map_err(|e| format!("SRV heap ({count}): {e}"))?;
        // SAFETY: plain queries on a live device/heap.
        let (inc, cpu0, gpu_start) = unsafe {
            (
                gpu.device
                    .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV),
                heap.GetCPUDescriptorHandleForHeapStart(),
                heap.GetGPUDescriptorHandleForHeapStart(),
            )
        };
        let mut set = Self {
            heap,
            gpu_start,
            cpu_start: cpu0,
            increment: inc,
            _resources: Vec::with_capacity(textures.len() + 1),
            count,
            reserve_start: textures.len() as u32 + 1,
            bytes: 0,
            rejected: Vec::new(),
            upload_submits: 0,
            completed_chains: 0,
        };
        let white = [Mip {
            width: 1,
            height: 1,
            data: vec![255; 4],
        }];
        let mut staging: Vec<ID3D12Resource> = Vec::new();
        let r = (|| -> Result<(), String> {
            let mut pending = 0u64;
            let mut batch = batch_bytes.max(1 << 20);
            let (wt, wst, wbytes) = record(gpu, DXGI_FORMAT_R8G8B8A8_UNORM, 1, 4, &white)?;
            staging.push(wst);
            set.write_srv(gpu, 0, &wt, DXGI_FORMAT_R8G8B8A8_UNORM, 1);
            set.bytes += wbytes;
            set._resources.push(wt.clone());
            for (i, t) in textures.iter().enumerate() {
                let slot = i as u32 + 1;
                let (format, block, block_bytes) = dxgi_format(t.format);
                if let Err(e) = validate_mips(t.format, &t.mips) {
                    set.rejected.push((t.name.clone(), e));
                    set.write_srv(gpu, slot, &wt, DXGI_FORMAT_R8G8B8A8_UNORM, 1);
                    continue;
                }
                let full = complete_chain(t.format, &t.mips);
                set.completed_chains += usize::from(full.is_some());
                let mips = full.as_deref().unwrap_or(&t.mips);
                validate_mips(t.format, mips)
                    .map_err(|e| format!("{}: completed chain: {e}", t.name))?;
                let (tex, up, bytes) = record(gpu, format, block, block_bytes, mips)?;
                pending += mips.iter().map(|m| m.data.len() as u64).sum::<u64>();
                staging.push(up);
                set.write_srv(gpu, slot, &tex, format, mips.len() as u32);
                set.bytes += bytes;
                set._resources.push(tex);
                if pending >= batch {
                    gpu.submit()?;
                    set.upload_submits += 1;
                    staging.clear();
                    batch = crate::accel::rescale(
                        pending,
                        batch,
                        gpu.last_submit_ms(),
                        1 << 20,
                        batch_bytes.max(1 << 20),
                    );
                    pending = 0;
                }
            }
            for slot in set.reserve_start..count {
                set.write_srv(gpu, slot, &wt, DXGI_FORMAT_R8G8B8A8_UNORM, 1);
            }
            gpu.submit()?;
            set.upload_submits += 1;
            Ok(())
        })();
        if let Err(e) = r {
            let _ = gpu.discard();
            return Err(e);
        }
        drop(staging);
        Ok(set)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_blocks_decode_to_their_colour() {
        let v = [0.2, 0.6, 0.9, 1.0];
        let b7 = constant_block(Format::Bc7UnormSrgb, v).unwrap();
        let mut rgba = [0u8; 64];
        bcdec_rs::bc7(&b7, &mut rgba, 16);
        for k in 0..16 {
            for c in 0..4 {
                let want = (v[c] * 255.0).round();
                assert!((f32::from(rgba[4 * k + c]) - want).abs() <= 1.0, "{rgba:?}");
            }
        }
        let b5 = constant_block(Format::Bc5Snorm, [-0.5, 0.25, 0.0, 1.0]).unwrap();
        let mut rg = [0f32; 32];
        bcdec_rs::bc5_float(&b5, &mut rg, 8, true);
        assert!(
            rg.chunks(2)
                .all(|p| (p[0] + 0.5).abs() < 0.01 && (p[1] - 0.25).abs() < 0.01)
        );
    }

    #[test]
    fn chain_completed_to_one_texel() {
        let top = Mip {
            width: 8,
            height: 4,
            data: vec![0; 2 * 8],
        };
        let last = Mip {
            width: 4,
            height: 2,
            data: vec![0; 8],
        };
        let full = complete_chain(Format::Bc4Unorm, &[top, last]).unwrap();
        assert_eq!(
            full.iter().map(|m| (m.width, m.height)).collect::<Vec<_>>(),
            vec![(8, 4), (4, 2), (2, 1), (1, 1)]
        );
        assert!(validate_mips(Format::Bc4Unorm, &full).is_ok());
    }

    #[test]
    fn chains_ending_above_one_block_stay_as_stored() {
        let only = Mip {
            width: 64,
            height: 64,
            data: vec![0; 16 * 16 * 8],
        };
        assert!(complete_chain(Format::Bc4Unorm, &[only]).is_none());
    }

    fn bc4_chain(top: u32, last: u32) -> Vec<Mip> {
        let mut v = Vec::new();
        let mut s = top;
        while s >= last {
            v.push(Mip {
                width: s,
                height: s,
                data: vec![0; (s.div_ceil(4) * s.div_ceil(4) * 8) as usize],
            });
            s /= 2;
        }
        v
    }

    #[test]
    fn caps_drop_leading_mips_like_the_decoder() {
        let mut t = TextureData {
            name: "t".into(),
            format: Format::Bc4Unorm,
            mips: bc4_chain(2048, 4),
        };
        assert_eq!(cap_start(&t.mips, 2048), 0);
        assert_eq!(cap_start(&t.mips, 512), 2);
        // 512..4 (8 levels) completed to 1x1: 4x4 -> 2x2 -> 1x1.
        assert_eq!(uploaded_levels(t.format, &t.mips[2..]), 8 + 2);
        cap_mips(&mut t, 512);
        assert_eq!(t.mips[0].width, 512);
        assert!(validate_mips(t.format, &t.mips).is_ok());
        // A single stored level stays whatever the cap.
        let only = bc4_chain(1024, 1024);
        assert_eq!(cap_start(&only, 256), 0);
        assert_eq!(uploaded_levels(Format::Bc4Unorm, &only), 1);
    }
}
