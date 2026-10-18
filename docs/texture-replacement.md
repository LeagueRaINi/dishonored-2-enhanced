# Texture replacement (and resource overrides) for the original renderer

Status: implemented 2026-10-17 (§6; in-game behaviour not yet verified). Executable `b671f824…6180`, image base
0x140000000, IDB `void-rs/local/ida/Dishonored2.exe.i64`. Labels: **O** observed in the
IDB (decompiled/disassembled code), **H** inferred, **U** unknown. Functions that were still
`sub_*` have been renamed in the IDB to the names used here (old address kept in the
tables).

d2enhanced: `d2enhanced/re/NOTES.md` has nothing on the image/file path (bounded grep:
`texture|bimage|CreateTexture2D|fs_|loose|override|image`; the only hit was
`image_lodbias`). Every claim below comes from this IDB, so there was no d2enhanced claim
to corroborate.

## Summary

- All image bytes (the base `.bimage7` file and every streamed `.bimage7_mipN` part)
  reach the engine as an `idFile` returned by **`idFileSystemLocal::OpenFileRead`**
  (`0x140161850`, vtable slot +248) (O). The same function serves decls (materials),
  `.bmodel`/md6 models, anims, probes and fonts (O, see §4). Hooking it is enough to
  substitute any resource by name, which makes it the recommended hook point.
- The shipped build has **no loose-file override** (O, §2). Loose directories
  (`<fs_basepath>/base/`, `<fs_savepath>/base/`) are searched only after the archive
  container, and they are not searched at all while the resource callback is installed
  (game init and map load).
- Every engine texture is created with `CreateTexture2D(desc, pInitialData = NULL)` (O),
  then filled with `UpdateSubresource1`. Streaming creates a larger texture and copies
  the old mips with `CopySubresourceRegion` (O). Because no initial data is passed,
  ReShade/Special K-style hash-on-create texture injection does not apply to this game
  (§3.4).
- Recommended mechanism: the `OpenFileRead` detour returns an engine `idFile_Memory`
  that holds a BIM synthesized from a user DDS, or an exact user BIM. Replacements may
  change the resolution and mip count, but must keep the texture type, faces, format
  class and flags. Write a replacement fully resident (all mips in the base file) by
  default; the streamer then never touches it.
- Implemented (§5, §6): BIM writer, DDS reader/writer, shared validation, `void-tool
  texmod`, and the `OpenFileRead` hook in void-inject (`texture_mods = 1`). No BC encoder
  is needed: inputs are already-compressed DDS files.

## 1. Image load path

### 1.1 Call chain (O unless marked)

```
material / decl references image name (e.g. models/.../foo_d.tga)
  -> image name without extension unless it carries `$` options (material.md)
idImage resource load (idImage vtable 0x141B1CCA8, slot +88)
  idImage_LoadResourceInternal                     0x1402EA140
    names starting '_' -> generated image (no file)
    arkTextureStreamer_Unregister(&streamer, img)    0x14030DD00
    free old GPU resource (0x1402F4910)
    idImage_GetResourceFileName (vtable +56)         0x1402E9920
      -> sub_14005E890: "generated/image/<name>" + "." + off_1422AB920[dword_1432EA098]
         ("bimage7"; archive names per void-assets texture.rs, e.g.
          generated/image/models/weapons/sword_player/sword_player_d.bimage7,
          x_d$nomips.tga.bimage7)
    idResource_OpenFile(img, &file, name, 0, ctx)    0x14005F0D0
      idFileSystemLocal_OpenFileRead(fs, name, useCallback=1, 0)   0x140161850  <== HOOK
        if useCallback && fs+0xC0 callback: return callback(name)   (exclusive, no fallback)
          ResourceOpenCallback_BackgroundLoader      0x140069B20 (map load)
            -> idBackgroundLoader_GetFile            0x14006B850 (sequential, 10-entry window)
               -> else idResourceManagerLocal_OpenFileUncached (rm vtable +120) 0x140065390
          ResourceOpenCallback_Preloaded             0x140068040 (game init)
            -> preload table qword_142BC5450, else OpenFileUncached
        else: normalise name (lowercase, '\'->'/', strip leading '/'), then containers in
              priority order (fs+0xE0 list, count fs+0xE8):
              GameResources (arkGameFileContainer, prio 1000) -> OpenFileUncached (archives)
              UserFolder (arkDiskFileContainer <savepath>/base/, prio 450)
              GameFolder (arkDiskFileContainer <basepath>/base/, prio 400)
      if the returned file is not a memory file, copy it into an idFile_Memory
    idBinaryImage(name) on stack
    idBinaryImage_LoadFromGeneratedFile(bin, file)   0x1402D2740
      magic 0x184D4942 "BIM\x18"; idImageOpts_Read 0x1402EA840 (version must be 19)
      reads mips_in_file x faces records; keeps level >= skip, rebases levels
    if opts.flags & 0x4000: copy opts, register data with the image manager
      (sub_1402ECC80 / sub_1402EB370), no GPU texture (CPU-side image, H)
    else idImage_UploadBinaryImage(img, bin)        0x1402E9FF0
      idImage_AllocImage(img, ctx, &opts, 0)          0x1402F4060
        D3D11_CreateTexture2D_NoInitData 0x1402932F0:
          ID3D11Device(qword_1432D4F00)->CreateTexture2D(desc, NULL, &img+304)
          (3D: device vtable +48 CreateTexture3D, also NULL initial data)
      per record while record dims <= texture dims:
        idImage_SubImageUpload 0x1402F5500 -> idImage_SubImageUpload2D 0x1402F50C0
          USAGE_DEFAULT: ctx->UpdateSubresource1(tex, sub, box, data, pitch, 0, D3D11_COPY_DISCARD)
                         (context vtable +928 = index 116)
          USAGE_DYNAMIC: Map(WRITE_DISCARD) / memcpy rows / Unmap (vtable +112/+120)
      if resident mips (+88) < total mips (+100): arkTextureStreamer_Register 0x14030DBF0
```

Streaming (O for the functions; thread names from RTTI vtables):

```
arkTextureStreamerSubLoaderThread::Run = arkTextureStreamerSubLoaderThread_Run 0x14030E840
  pops (image, n) from qword_1432F0210 queue; level = img+100 - n
  name = strip_ext(GetResourceFileName(img)) + ".bimage%d_mip%d" (dword_1432EA098 = 7, level)
  OpenFileRead(fs, name, 1, 0)                                     <== same HOOK
  reads the 20-byte record header (level, face, w, h, size); queues a 320-byte request
  ("arkTextureStreamer: Could not open file '%s'" on failure)
arkTextureStreamerLoaderThread (vtable 0x141B31608) -> arkTextureStreamer_ReadMipPayloads 0x14030F0F0
  allocates the payload (TAG_RENDERER) and reads it from the request's file
render pass ARK_RENDERPASS_TEXTURE_STREAMER (sub_14034FD00) -> arkTextureStreamer_ApplyMips 0x14030D1E0
  new opts = old opts with resident mips + 1, width*2, height*2
  check: request size == new_w * new_h * bpp / 8, else "arkTextureStreamer: Skipping mip
         for texture '%s'"
  idImage_GrowForStreamedMip 0x1402F4990: CreateTexture2D(new desc, NULL) and
         CopySubresourceRegion (context vtable +368) of every old mip/face, new texture -> img+312
  upload the new top mip: UpdateSubresource1 in row slices (time-sliced when the renderer
         flag at sub_140373680(&unk_1432FC580)+8000 is set, H: r_textureUploadTimesliceMs)
         or in one call (sub_1402E8BD0)
  sub_1402F5530 swaps img+312 -> img+304 and releases the old texture
eligibility: arkTextureStreamer_CanStreamNextMip 0x14030CCE0 (screen-coverage test,
  then mip KB <= r_textureStreamingMaxTextureSizeKB, default 131072)
```

Threads: `OpenFileRead` runs on the resource-loading thread (U: whether this is the main
thread or a job thread), on `arkTextureStreamerSubLoaderThread` (O) and wherever decls
and models are read. Initial uploads use the context object in `qword_14397A248`, which
three renderer setup functions (`0x14035EA10`, `0x140362460`, `0x14036F510`) set from
`qword_1432D4EF0` (H: immediate-context wrapper). Streamed mips are applied in a render
job (O: pass registration). The detour therefore has to be thread-safe.

### 1.2 Layouts

`idImageOpts`, 76 bytes, at `idImage+52` and at `idBinaryImage+52`, read by
`idImageOpts_Read` from the BIM header (O):

| opts | idImage | BIM file | meaning |
| --- | --- | --- | --- |
| +0 | +52 | 0x08 | texture type (0 2D, 1 3D, 2 cube) |
| +4 | +56 | 0x0C/0x10/0x14 | packed format word: bits 0-5 format, 6-9 numeric, 10-12 file 0x14 (1 in all samples; the exe format table has 2 for A8, so H: channel kind); canonicalised by `sub_1402C2EF0`; `>> 13` = DXGI format |
| +8/+12 | +60/+64 | copy of 0x18/0x1C | **GPU texture width/height = largest resident mip** |
| +16/+20 | +68/+72 | 0x18/0x1C | first mip in the base file |
| +24/+28 | +76/+80 | 0x20/0x24 | full width/height |
| +32 | +84 | 0x28 | depth |
| +36/+40 | +88/+92 | copy of 0x2C | **resident mip count = GPU MipLevels**; streaming increments it |
| +44 | +96 | 0x2C | mips in the base file |
| +48 | +100 | 0x30 | total mips |
| +52 | +104 | 0x34 | faces |
| +56 | +108 | - | MSAA sample count (desc SampleDesc.Count) |
| +60 | +112 | 0x38 (u32) | passed to the upload as an out-parameter; purpose U |
| +64 | +116 | - | byte: create the resource with numeric 1 (H: typeless, for dual views) |
| +68 | +120 | - | usage (0 default, 2 dynamic, 3 staging) |
| +72 | +124 | 0x3C (u16) | flags: 0x80 = keep all mips in the base file (no streaming; the writer `idBinaryImage_WriteGeneratedFile` 0x1402D3090 also forces this for `fonts*`, the loader for `maps*probe*`), 0x4000 = CPU-side image path, 0x1000 = keep texture on realloc, 0x40 = UAV, 0x400, 0x200 (passed to create), 1 = GENERATE_MIPS |
| +74 | +126 | - | bind bits: 1 -> SRV (8), 2 -> RT (0x20), 4 -> DS (0x40) |

Other `idImage` fields (O for the accesses, the names are H):

| offset | content |
| --- | --- |
| +0 | vtable `??_7idImage@@6B@` 0x141B1CCA8 (slot +56 GetResourceFileName, +88 LoadResourceInternal) |
| +8 | name (idStr data pointer), e.g. `models/weapons/sword_player/sword_player_d` |
| +48 | copied from `idBinaryImage+48` (U) |
| +128..+147 | optional extra opts block (AllocImage `a4`) |
| +152 | view/format descriptor block (`sub_1402BCF90`), +256 its key |
| +264 | view cache released on realloc (`sub_1402F4730`, H) |
| +304 | `ID3D11Texture2D*` / `ID3D11Texture3D*` (released through vtable +16) |
| +312 | grown texture while a streamed mip is being uploaded |
| +320 | streaming lock word, +324 bit 0 = request pending |

`idBinaryImage` (stack object): +0 name idStr, +52 opts, +128 pointer to 40-byte mip
records `{u32 level, u32 face, u32 w, u32 h, u32 size, pad, +24 data*}`, +136 count (low
24 bits), +140 capacity. `idFile_Memory` (360 bytes, ctor `idFile_Memory_ctor`
0x14015AC60, vtable `??_7idFile_Memory@@6B@`): +8 name, +288 data, +304 mode (0 read),
+312/+320/+328 length, +344 cursor, +352 byte "owns data", +296 free callback (default
`sub_14015BCA0`). The destructor `sub_14015B5B0` calls `(+296)(file)` when +352 != 0 and
then `sub_1400F1520` (engine operator delete) when deleting. `idFile_Memory_SetData`
0x14015CA70 sets data and length and clears +352.

### 1.3 Streaming facts that constrain replacements (O)

- The GPU texture always has the size of the largest resident mip: it is created from the
  base file's first record and replaced by a 2x larger texture for each streamed mip.
- A streamed mip is accepted only if it is exactly twice the current top mip in both
  axes and its size is `w * h * bits_per_pixel / 8`. BC formats (engine formats 14-20)
  pad the texture to multiples of 4. A chain that does not halve exactly (non-power-of-two)
  makes the streamer log "Skipping mip", and that mip is not applied.
- The engine splits images into streamed parts only for 2D, single-face images without flag
  0x80 and outside `fonts`. Cubes, 3D textures and arrays always keep every mip in the base
  file.
- Relevant cvars (defaults from the constructors): `r_imagesMipmapsStreamingEnable` 1,
  `r_textureStreamerMemoryBudgetMB` -1 (auto, with `r_vramMiscSizeEstimationMB` 2560),
  `r_textureStreamingMaxTextureSizeKB` 131072, `r_textureStreamingPerFrameBudgetKB` 4096,
  `r_textureUploadTimesliceMs` 3, `r_freezeTextureStreaming` 0, `r_showStreamingStats`
  (1/2/3, 3 = VRAM occupancy), `image_useCompression` 1, `r_textureDetailsPreset` 3.
- U: how `r_textureDetailsPreset` caps mips (the loader's skip count only triggers when
  total < mips in file).

## 2. Loose files and alternative search paths

Bounded search: all `fs_*`/`ark_fs_*` strings and their cvar constructors, `idFileSystemLocal`
Init (`sub_140160610`), `OpenFileRead`, the container classes, `idResourceManagerLocal`
Init/InitResources, and every writer of the resource mode `dword_1432C7ED4`.

- Resource mode (`idLib::SetProduction`, setter `sub_1401666A0`; O): 2 =
  `PROD_PRODUCTION`, always set by `idResourceManagerLocal` Init (`sub_140065180`), which
  then calls `InitResources("")`. The game-init window uses 3 (`sub_140062CE0`), which
  `sub_14004AF90` puts back to 2. 0 = `PROD_DEVELOPMENT` is only reached inside
  `idResourceManagerLocal_InitResources` (0x1400658B0) when a *named* resource set (DLC,
  caller `sub_1400D9990` "Loading DLC %s") fails to parse its master index. No cvar or
  command-line switch reaches it.
- Resource callback (`idFileSystemLocal_SetResourceOpenCallback`, fs vtable +40 -> fs+0xC0;
  O): installed during game init (`sub_140062CE0`), map load (`sub_1400661E0`, cleared after
  "seconds spent loading resources") and the map/world loads in `sub_14023A5D0` and
  `sub_1402960D0`. While it is set, `OpenFileRead(…, 1, …)` returns the callback's result
  without consulting any container.
- Containers (O): `GameResources` (`arkGameFileContainer`, priority 1000; its open is a
  tail call to `OpenFileUncached`), `UserFolder` and `GameFolder` (`arkDiskFileContainer`,
  priorities 450/400, `<fs_savepath>/base/` and `<fs_basepath>/base/`), and DLC disk
  containers (410). `containerSearch_byPriority` keeps the list in descending priority (O:
  comparator `sub_140162CF0`). Archives are therefore searched first, and a loose file is
  found only for a name absent from the index, outside callback windows. That makes it a
  fallback, not an override.
- `fs_debug` (int at cvar+40 = `dword_1432C7118`) makes `OpenFileRead` log "Can't find %s"
  (O). `dword_1432C7B98` sets every container's "[%s] loaded file '%s'" verbose flag (+877);
  which cvar it belongs to is U. `fs_generatedPath` ("generated"), `ark_fs_resourceCache`
  ("voidResourceCache"), `ark_fs_pk5_filelist`, `fs_arbitraryZipSupport` and
  `r_looseRenderProg` exist. Bounded: only their constructors were read, none was found on the
  production open path. `ResourceCache` (fs vtable +384 lookup) is used to delete a cached copy
  of a zero-filled file (`idResource_OpenFile`).
- F005 (patch-index precedence): `InitResources` parses `master.index` (`sub_140068C30`) and
  builds 176-byte container records. Which index file of a group wins was not traced, so
  F005 stays D.

**Conclusion (O):** the engine itself cannot be told to prefer loose files. Overrides need a
hook.

## 3. Recommended mechanism

### 3.1 Hook point: `idFileSystemLocal::OpenFileRead` (0x140161850)

Signature (O): `idFile* __fastcall(idFileSystemLocal* fs, const char* name, bool
useResourceCallback, u8 flags)`. Prologue `48 8B C4 57 41 56 41 57 48 83 EC 50` (no
RIP-relative bytes; check it before `MinHook::create_hook`, as `hook.rs` does).

Why this point:

- It is the single choke point for base images, streamed parts, decls (`ReadFile`
  0x1401622E0 -> `OpenFileRead`), models, anims, probes and fonts (O, §4). Hooking it covers
  both callback windows and container mode.
- Byte-level substitution keeps every engine invariant: the engine parses the replacement
  with its own loader, sizes its own D3D11 texture and runs its own streaming. Nothing
  depends on D3D11 object identity.
- Gap (O): `idFileResource` (generic `File` resources, vtable slot 11 `sub_1400590F0`) calls
  `OpenFileUncached` directly in mode 3. This does not affect images, decls or models.

Detour logic:

1. Normalise `name` like the engine does (lowercase, `\` -> `/`, strip leading `/`, strip an
   absolute prefix up to `/base/` like `idBackgroundLoader_GetFile`). Look it up in an
   immutable `HashMap<String, Override>` built at startup. No hit: tail-call the original.
2. Hit while a resource callback is installed (`*(fs+0xC0) != 0`) and the name exists in
   the archive index: call the original first and destroy its result (`(**file)(file, 1)`).
   The background loader consumes entries in order, so skipping one may leave its buffer
   unconsumed (H, see risks); the bytes were already read by the loader, so this costs
   little. Outside callback windows (gameplay streaming), and for synthesized names absent
   from the index, skip the original to avoid an uncached archive read.
3. Build the replacement file:
   `obj = sub_1400F13F0(360)` (engine operator new; O: the same allocator pairing as
   `idImage_LoadResourceInternal`/destructor) -> `idFile_Memory_ctor(obj, name)` ->
   `idFile_Memory_SetData(obj, ptr, len)` -> `*(obj+296) = our_free; *(u8*)(obj+352) = 1`.
   `our_free(file)` releases our buffer, which is a leaked `Box<[u8]>`; it reads pointer
   +288 and length +312. The engine then owns the file and frees it through our callback,
   so no engine heap ABI is needed for the data.
4. Log once per name: resource name, source path, original vs replacement size, BIM header
   (dims, format, mips in file / total), thread id, and whether it called through.

### 3.2 What may differ from the original

| property | allowed? | reason |
| --- | --- | --- |
| resolution | yes | the texture is created from the replacement header; budgets are dynamic |
| total mips | yes | GPU MipLevels = resident count; `_mipN` names are generated from the header |
| mips in base file | yes; **default all** | resident = total: the streamer never registers the image (O: `idImage_UploadBinaryImage` registers only when +88 < +100) |
| format class (format + numeric) | **same as original** | shaders expect the channel layout and colour space: BC5 SNORM normals reconstruct Z, BC7 SRGB vs UNORM decides gamma |
| texture type / faces / depth | same | cube and 3D paths differ; the material binds by type |
| flags (0x3C) and 0x14 / 0x38 fields | copied from the original | 0x80 / 0x4000 change load paths; meaning of 0x38 is U |
| non-power-of-two | only with every mip in the base file | the streamer rejects chains that do not halve exactly |

Streamed replacements, which are optional (for packs too large to keep resident): write the
base file with the smallest `k` mips and parts `_mip0 … _mip(total-k-1)`, each one
`{level, face 0, w, h, size}` + data with level `N` = `full >> N`. The detour must serve
part names that the original index lacks (higher-resolution replacements have more parts).
The streamer opens parts by name only (O), so no index entry is needed. Constraints:
power-of-two dims and every streamed level >= 4x4 (BC).

### 3.3 File layout and identity

```
<mods_dir>/                              (void-inject.ini; default void-rs/mods)
  textures/<image name>.dds              convenience layer, e.g.
      textures/models/weapons/sword_player/sword_player_d.dds
        -> generated/image/models/weapons/sword_player/sword_player_d.bimage7
  override/<exact resource name>         expert layer, any kind, served byte-for-byte, e.g.
      override/generated/image/....bimage7, override/generated/image/....bimage7_mip0,
      override/generated/decls/material/....decl
```

- The `.dds` layer is converted to a fully resident BIM on each request (header rewrite
  and record chunking, no recompression). Settings keys in `void-inject.ini`:
  `texture_mods = 1`, `mods_dir = <dir>`.
- Identity: every mod is validated on its first request against **every** archive copy of
  the original (all groups holding the name): type/faces/depth, format class, header fields
  0x14 and 0x38..0x3E, and the engine rules of §1.3. Invalid mods are logged once and never
  served. Duplicate targets: `override/` wins over `textures/`. Only names present in the
  archive index can be replaced, except streamed parts of a raw `override/` base.
- The hook addresses are protected by exact prologue-byte checks of the four engine
  functions used (as the boot-video patch), because the hook is installed in DllMain
  before the executable hash is computed (§6).

### 3.4 Comparison with a D3D11-level approach

ReShade and Special K texture mods hash `pInitialData` in `CreateTexture2D`. Here (O):

- `pInitialData` is always NULL (`xor r8d, r8d` in `D3D11_CreateTexture2D_NoInitData`), so
  there is nothing to hash at creation.
- Data arrives later in per-subresource `UpdateSubresource1` calls (row-sliced for streamed
  mips) or in `Map` writes. The texture is recreated, with `CopySubresourceRegion`, each time
  a mip streams in, so a D3D object's identity does not even survive streaming.
- The engine has already fixed size, format and mip count, so a D3D11 hook could only swap
  same-shape data. It would also hash every upload on an already CPU-bound frame (7.9 ms
  CPU vs 6.0 ms GPU).

The D3D11 layer remains useful only for diagnostics: dumping, identifying textures by
`img+304` while hooking `idImage_AllocImage`, or textures that exist only at runtime
(`_`-prefixed generated images, render targets). For files, the `OpenFileRead`
substitution is preferable in every case.

## 4. Other resource types through the same path (bounded)

O: callers of `idResource_OpenFile`: binary resources with magic check (`sub_140002F00`),
fonts (`sub_140079100`), anims (`sub_140240350`), `sub_140256E60`, md6 models
(`sub_14025AA00`), `.bmodel` static models (`sub_14026A5A0`, rejects version changes),
`.bprobe` (`sub_1402B84C0`), images. Decls in production go through
`idFileSystemLocal_ReadFile` -> `OpenFileRead` (`sub_140011A20`, "couldn't read production
file", text parsed with idLexer). The same detour can therefore override materials
(`docs/formats/material.md` syntax) and models with the `override/` layer.

- Material decls: changing parameters or stages of an existing material is in reach.
  Referencing image names absent from the index is U (the resource manager decides what to
  load from `.mapresources` ids; unknown names may never be requested or may error in
  `ResolveCodeResources`).
- Models: the format must match the loader version exactly. Collision (`.hkx`/
  `.bphysworld`), world geometry (`.bwm`) and Umbra (`.tome`) are separate files.
  `.bwm`/`.tome` are opened in `sub_14023A5D0` inside a callback window (path via fs: H).
- Not covered: `idFileResource` in mode 3 (§3.1).

## 5. Code

- `void-formats/src/bim.rs`: parser (record count now follows the engine: 3D images store
  one record per depth slice of every mip, cubes stored with `faces = 1` carry 6 faces;
  sizes are checked for all types), writer (`Image::to_bytes`, `Header::to_bytes`,
  `Mip::to_bytes` = streamed part), `Image::from_dds` (fully resident, template header
  fields kept, chain cut at the engine's `max_mips`), `check_replacement` (§3.2/§1.3 rules),
  `Format::fields`, `max_mips`, `mip_dim`.
- `void-formats/src/dds.rs`: DDS reader/writer. DX10 header (DXGI 28/29, 49/51, 56, 61,
  65, 80/81, 83/84, 95/96, 98/99), legacy FourCC `ATI1`/`BC4U`/`BC4S`/`ATI2`/`BC5U`/`BC5S`,
  legacy masks RGBA8 (R,G,B,A order), L8, A8. Rejects BC1-BC3 (absent from the engine's
  format table), typeless formats, cubes, arrays, volumes, truncated and trailing data.
- `void-formats/src/texmod.rs`: mods-dir layout and scan, engine-style name normalisation,
  validation (`check`) and served bytes (`materialize`) behind an `Originals` trait;
  constant-colour test textures (BC7 mode 6, BC4/BC5, uncompressed). Shared by the tool
  and the hook.
- `void-tool texmod`: `check --mods <dir>`, `import <dds> --image <name> --mods <dir>`,
  `export --image <name> --out <file.dds>` (original base + streamed parts as one DDS),
  `test-pattern --image <name> --mods <dir> [--color R,G,B,A] [--per-mip]`,
  `roundtrip [--group --name <mapresources>]`. `<name>` is an image name
  (`models/weapons/sword_player/sword_player_d`) or the resource name.
- `void-inject/src/texmods.rs`: the `OpenFileRead` detour of §3.1; settings
  `texture_mods` (default 0: no hook) and `mods_dir`.
- `void-assets/src/texture.rs` is unchanged (decoders for the in-process renderer).

## 6. Implementation and verification (2026-10-17)

Install time: the hook is installed from DllMain (`ffi.rs` -> `texmods::install_early`),
right after the boot-video patch. `initialize` runs on a thread that starts only after
the loader lock is released and first hashes the 41 MB executable, while the main thread
goes on into WinMain, whose filesystem and game init already open resources. Nothing
orders the two, so an install from `initialize` can miss game-init loads. DllMain work is
kept cheap: a directory listing of the mods dir, prologue-byte checks of
`OpenFileRead` (0x140161850), operator new (0x1400F13F0), `idFile_Memory` ctor
(0x14015AC60) and SetData (0x14015CA70), and the MinHook install. The archive index is
opened on the first request of a modded name (only the modded names are indexed), and
each mod is validated once, on its first request. Messages before the log exists are
queued and written by `initialize` (`texmods::report`).

Detour (per call): normalise the name; names without a mod go straight to the original.
For a valid mod: build the bytes, allocate the `idFile_Memory` with the engine's
operator new, construct it, `SetData`, set the free callback (+296) and owns flag (+352)
so the engine releases our buffer through `free_buffer`; inside a callback window
(`fs+0xC0 != 0`, resource callback requested) and for indexed names, call the original
first and destroy its file; return ours. One log line per name (summary, size,
call-through, thread). Panics fall back to the original.

Offline results (`cargo run --release -p void-tool -- --game <root> texmod ...`):

- `roundtrip` (tutorial_p.mapresources, group 0): 3,722 images (1,491 base files, 2,231
  streamed parts), every one rewritten byte-identically (3,198,049,226 bytes), and all
  1,491 originals pass `check_replacement` against themselves (the rules accept what the
  game ships, including cubes and 3D colour cubes). Before the record-count fix three 3D
  images (two flipbooks, one colour cube) failed to parse.
- Unit tests: BIM writer round trip, DDS -> BIM -> parse round trip, engine mip limit,
  incompatible replacements (format class, BC size < 4, non-halving streamed chain,
  CPU-image resize, changed unmodelled header field), DDS edge cases (legacy FourCC
  BC4/BC5, DX10, BC1-3 rejection, typeless, truncation at every header boundary, trailing
  data, cube/array), BC7/BC5 constant blocks, mods-dir scan/check/materialize.
- `export` -> `import` of `sword_player_d` validates; importing it over the BC5 SNORM
  normal map is refused ("DDS is Bc7UnormSrgb, the original is Bc5Snorm").

Smoke-test mod: `void-rs/local/mods-test/textures/models/weapons/sword_player/sword_player_d.dds`,
magenta BC7 sRGB 2048x2048 with all 10 mips resident (original: 7 streamed parts + 3 base
mips). It is the player's folding blade, held in first person in every mission (e.g. Edge
of the World, the benchmark map); written by
`void-tool texmod test-pattern --image models/weapons/sword_player/sword_player_d --mods local/mods-test`
(`--per-mip` colours each mip level instead). `texmod check --mods local/mods-test`: 1 ok.
To try it: `texture_mods = 1` and `mods_dir = void-rs/local/mods-test` in
`void-rs/local/inject/void-inject.ini`; expected: a magenta blade, one `texmods: served
...` log line for `sword_player_d`, no `_mipN` requests for it.

Not verified (needs a game run): the hook in practice (DllMain install, callback-window
call-through, `free_buffer` on destruction), visibility of the test mod, and whether
anything other than `OpenFileRead` ever reads the replaced name. In-game checks: the
void-inject log; `r_showStreamingStats 3`; no "Skipping mip" / "Could not open file" in
the engine console; `fs_debug 1` for names that fail. Not done: the viewer overlay (old
step 4) and streamed (split) DDS replacements (§3.2, not needed while replacements stay
resident).

## 7. Risks

- Background loader (H): if the detour skips an entry, the sequential loader may keep its
  buffer or wait ("BGL perf : More than 1 second elapsed without consuming a file"). The
  call-through in §3.1 step 2 avoids this; it has not been verified in game.
- Heap ownership (O for the mechanism): the `idFile_Memory` object must come from the engine
  allocator (`sub_1400F13F0`), because the destructor deletes it with `sub_1400F1520`. Data
  freed through our +296 callback, never by the engine allocator.
- Thread safety: the detour runs on loader, streamer and main threads. Use an immutable map
  after startup and no locks held across file I/O.
- Semantics: wrong format class (SRGB vs UNORM, BC5 SNORM normals, BC4 masks) renders wrong
  without errors. The validator enforces the original's class.
- VRAM: higher-resolution resident textures reduce the streaming budget for everything else
  (auto budget = VRAM - 2560 MB, H); large resident packs can cause pop-in elsewhere.
- CPU images (flag 0x4000): data feeds a CPU-side image (H). Keep their dims.
- Mode-3 `idFileResource` bypass (§3.1) and unknown new-name behaviour for decls (§4).
- Game updates: the executable hash gate disables the hook. Mods whose original-hash no
  longer matches are reported.

## Unknowns

- Thread of `idImage_LoadResourceInternal` and whether `qword_14397A248` is always the
  immediate context.
- Meaning of BIM field 0x38 (opts +60) and of flag bits 0x200/0x400/0x1000 (only seen as
  create-path conditions).
- How `r_textureDetailsPreset` reduces texture resolution.
- Whether the background loader really stalls when an entry is skipped (call-through
  avoids the question).
- Container search order is inferred from the comparator; whether a loose
  `base/<name>` is ever reached for indexed names was not tested.
