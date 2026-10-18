# FSR 4 integration: engine-level design

Replaces the engine TAA pass of Dishonored 2 (exe SHA-256 `b671f824...6180`, image base
0x140000000) with FSR 4 (native AA and upscale), from `void-inject`, using engine structures
instead of D3D11 state / constant-buffer shadows where possible.

Evidence tags: **O** observed (IDB disassembly/pseudocode of `void-rs/local/ida/Dishonored2.exe.i64`,
reflection tables `void-rs/local/reflect/*.tsv`, or a measurement named in the text), **H** inferred,
**U** unknown. "d2e" = lead from `d2enhanced/re/NOTES.md` or `d2enhanced/src/*`; each use states
whether it was corroborated here. Names such as `arkRenderPass` / engine context are ours unless they
come from reflection or strings.

## Engine TAA site

### The pass and its job struct

- `PostFX_TemporalAA_and_Post` `0x140327270` is the callback of render pass
  `ARK_RENDERPASS_POSTPROC`, registered in `sub_14034FE00` at `0x140350E20` (`sub_1402C2960(slot
  renderer+0x570, "ARK_RENDERPASS_POSTPROC", count 1, mask arg 2, flags 0, callback)`) followed by
  `sub_1403492F0` (pass parameter decl, hash `0xA4719E7A` = "default"). **O** (d2e "one job / one
  deferred ctx": corroborated, see Command-list flow).
- Pass entries are 288 bytes (`sub_1402C29A0` writes them in steps of 288; source path string
  `voidengine\core\render\arkRenderPass.h`). Fields used here (**O** from the functions named):

| Offset | Meaning | Evidence |
| --- | --- | --- |
| +0x48 / +0x50 | read list `{u32 hash, pad, arkImageView*}` x count (low 24 bits) | lookups in `0x140327270` |
| +0x70 | `idRenderView*` | `mov r15,[r14+70h]` `0x1403272AD` (d2e "view = job+0x70": corroborated) |
| +0x80 / +0x88 | write list, same layout | lookups in `0x140327270` |
| +0xC0 / +0xC8 | dependent passes | `sub_1402C72B0` |
| +0xD0 / +0xD4 | pending / dependency counters | `sub_1402C5C80`, `sub_1402C5C10` |
| +0xD8 | callback `bool(*)(pass*)` (true = run again) | `sub_1402C5D50`, `sub_1402C29A0` |
| +0xF0 | worker-thread mask (ctor default 0x7FFFFFFF) | `sub_1402C56E0`, `sub_1402C7AC0` |
| +0xF4 | flags: 0x10 CPU-only (no context), 0x20 record on the immediate context at submit, 0x100 skipped this frame, 0x200 second ready queue, 0x400 needs submit | `sub_1402C7AC0`, `sub_1402C77B0`, `sub_1402C5F60`, writers in `sub_140351780` |
| +0x100 | state (2 queued, 3 recorded, 4 done) | `sub_1402C77B0`, `sub_1402C72B0` |
| +0x108 | engine render context (below) | `unknown_libname_22` `0x1402C5C60` = `return *(a1+0x108)` (d2e "engine ctx = job+0x108": corroborated) |
| +0x110 | byte: context owned by the pass (not pooled) | `sub_1402C5DD0`, `sub_1402C5C90` |
| +0x118 | `ID3D11CommandList*` recorded for the pass | `sub_1402C5D50`, `sub_1402C5F60` |

- Resource names are FNV-1a 32 over the name **including the terminating NUL** (`sub_1402FC600`
  hashes "srgbBuffer" as 11 bytes). **O**. Resolved for this pass (hashes matched against the
  exe's strings, **O**):

| List | Hash | Name | Role in TAA |
| --- | --- | --- | --- |
| write | 0x8F93AB75 | hdrBuffer | `viewColorMap` (TAA input) |
| write | 0x8E999AED | hdrBuffer2 | `taaResult` (TAA output) |
| write | 0x3337E277 / 0x0D35680E | temporalAAHistory0 / 1 | history ping-pong |
| write | 0xBB9F8C8F | depthAlphaDiv2 | later (its size is used by sub_140325D80) |
| write | 0xBCA5C3F1 / 0xF92B2904 / 0x72A1B068 | srgbBuffer / floatBackBuffer / exrBuffer | tonemap / capture targets |
| read | 0xFFD54B9F | motionVectors | `motionVectors` |
| read | 0x21181874 | decompressedDepth | `fx/depthFull` (bound for the whole pass) |
| read | 0x1BBE2D1A | distoVectors | motion blur (`sub_140326930` arg) |
| read | 0x7A833082 | depthWaterStencilView | looked up, result unused |

### TAA block inside `0x140327270` (all **O**)

Registers in the function: `r14` = pass, `rsi` = engine context, `r15` = view, `rbx` = view+0x39D0
(`m_renderRV`), `r13` = `taaResult` view.

| Address | Call | Effect |
| --- | --- | --- |
| 0x1403274CD | `sub_140310750(ctx+0x10, fx/depthFull, decompressedDepth)` | depth param for the pass |
| 0x1403274E2 | `sub_1402BC610(ctx, view+0x7878)` | viewport = `m_unscaledViewport` |
| 0x140327517 | `sub_1402BC4B0(ctx, view+0x78A8 or +0x7768)` | scissor = `m_unscaledScissor` (or unscaled subrect if view byte +14900 bit0) |
| 0x14032752B / 0x140327535 | skip if view byte +14900 bit0 or +14903 bit2 | TAA off; then `sub_140362450(rs,1)` at 0x14032783F sets the history-reset counter to 60 unless bit0 |
| 0x140327542 | `GetId(rs)` (`0x14035EB00` = `return *(rs+0xCC)`, misnamed library match) & 1 | history parity: read = History0/1 swapped by frame parity |
| 0x14032765E | `taaDitherAndViewportSize` = (m_viewport.x2-x1+1, y2-y1+1) | render size |
| 0x140327690 | `taaTexSize` = 1/(taaResult.m_width * m_previousResolutionScale.x), same for y | (d2e said history size; it is the `taaResult` view size, same dimensions) |
| 0x1403276E3 | `taaAmount` = clamp(int `dword_1432F88E8` * 0.01, 0.01, 0.2); 1.0 if reset counter > 0 | current-frame weight |
| 0x1403276FF | `sub_1402BB0E0(ctx, historyRead)` = `ClearRenderTargetView` if reset counter > 0 | history clear |
| 0x140327753 .. 0x1403277A6 | `taaHistory_read`, `taaHistory_write`, `viewColorMap`=hdrBuffer, `motionVectors`, `taaResult`=hdrBuffer2 | `sub_140310750` param stores |
| 0x1403277AF | `sub_140325420(ctx+0x10)` | tonemapping params/coefficients (forward + inverse) from `env/tonemapping/*` |
| 0x1403277F0 | `Ctx_BindProgram(material+0xA0 program, ctx, ctx+0x10, 0, 6, ..)` | material `g_mtrTAA_R11G11B10` / `g_mtrTAA_R16G16B16` if `dword_14397A6F8 != 0` (d2e: corroborated) |
| **0x140327817** | `call Ctx_Dispatch` (`E8 44 3C F9 FF`) with `edx = (vpW+15)/16`, `r8d = (vpH+15)/16`, `r9d = 1` | **the TAA dispatch** |
| 0x14032781F | `sub_1402BCD00(ctx)` = `CSSetUnorderedAccessViews(0, 8, null)` | UAV unbind |
| 0x140327824 / 0x14032782B | `v113 = taaResult; v10 = hdrBuffer` | output becomes the scene buffer |

- Engine dispatch wrapper: `Ctx_Dispatch` `0x1402BB460` = `mov rcx,[rcx]; mov rax,[rcx]; jmp
  [rax+148h]`, i.e. `ID3D11DeviceContext::Dispatch(ctx->d3d, x, y, z)` (vtable slot 41). Signature
  `void Ctx_Dispatch(engineCtx*, u32 x, u32 y, u32 z)`; the stack argument the caller also writes is
  unused. 25 call sites; the TAA one is 0x140327817. **O**
- `sub_14035ED60` (reset query) decrements `rs+8` and returns old value > 0; it is called twice per
  TAA frame (0x1403276C8, 0x1403276EF). `rs` = `unk_1432FC580` (`idRenderSystemLocal`, vtable
  0x141B31648; `sub_140362450` is its vtable slot 100). **O**. Only writer of the counter found:
  `sub_140362450` (call sites: the TAA skip path; a `call [reg+320h]` scan of 0x140000000-0x140A00000
  found no other caller of slot 100). The engine TAA therefore does **not** reset on teleports. **O**
  for the bounded scan, **U** outside it.

### What the rest of the function expects after the dispatch

- `hdrBuffer2` (`taaResult`) holds the anti-aliased scene in the **same stored units as
  `hdrBuffer`** (d2e shader analysis: taaresult is written back after inverse tonemap and exposure
  undo; not re-verified here, **H** from d2e), covering at least `m_viewport` rounded up to 16 px
  tiles. It is then: debug draw target (view byte +16952 bit3), motion-blur source/destination
  ping-pong with `hdrBuffer` (`sub_1403262D0` / `sub_140326930` x2), render target for stage-3
  screen effects (`sub_14031DA70(pass,3)` at 0x140327B08), input of exposure/bloom/tonemap
  (`sub_1403257A0`, `sub_140325910`, `sub_140325D80`, `sub_140326AF0`). **O** (call graph).
- `hdrBuffer` is free scratch after TAA (`v10`). **O**
- Histories: only TAA reads/writes `temporalAAHistory0/1` (pass write list; no other use in this
  function, **O**; other functions not searched, **U**). With FSR active they can stay stale. When
  handing back to the engine TAA, set `*(int*)(rs+8) >= 2` (both checks decrement) so the engine
  clears history and uses `taaAmount` 1.0. **H**
- No D3D11 state set by the TAA block is relied upon later: every later section re-sets viewport
  (`sub_1402BC610`), scissor (`sub_1402BC4B0`), render target (`sub_1402BC340`) and binds its
  material (`sub_1402BB470` -> `sub_1402BBFA0` state objects, `Ctx_BindProgram` shaders/resources).
  The engine context does cache some state (see Command-list flow): a replacement that **changes
  D3D11 state at record time** must restore it or invalidate the cache. **O**

### Proposed record-time hook

Patch the 5-byte call at **0x140327817** (`E8 44 3C F9 FF`, verify before writing) to a thunk with
the same register arguments (`rcx` = engine context, `edx`/`r8d`/`r9d` = groups) that also passes
`r14` (the pass). The thunk:

1. Reads the already-resolved inputs from the engine context's parameter block (layouts in Inputs):
   `viewColorMap`, `motionVectors`, `taaResult`, `fx/depthFull`, `PostFX/Luminance/buffAutoExposure`,
   exposure index, `useCompressedHdrBuffers`, `PostFX/Luminance/customEVBias`, white levels; and the
   view (`pass+0x70`) fields.
2. Queues an FSR job keyed by the pass pointer and the frame counter `*(u32*)(rs+0xCC)`.
3. Does **not** call `Ctx_Dispatch` (FSR active) or tail-calls it (FSR inactive / job rejected).
   Nothing else is recorded on the deferred context, so no state is disturbed.

Equivalent alternative: MinHook `PostFX_TemporalAA_and_Post` entry to stash the pass pointer in a
thread-local, plus the call-site patch (no `r14` access needed). Hooking `Ctx_Dispatch` itself would
need a return-address check (0x14032781C).

TAA must be enabled for the site to run (view byte +14903 bit2 comes from `r_skipTemporalAA`, d2e
"Graphics menu", not re-verified, **H**): force `r_skipTemporalAA 0` / `r_AAMode -1` as d2e does, or
also patch the skip branches 0x14032752B/0x140327535.

## Inputs

### Parameter handles and the parameter block (**O**)

- A parameter global (e.g. `qword_1422B3500`) points to a parameter declaration whose `int` at
  **+0x48** is its slot index. Records `{const char* name; ...; decl* handle @+0x10}` (32 B), e.g.
  name at 0x1422B34F0 -> handle 0x1422B3500 ("taaResult"). (d2e said handle at +0x20 of the record;
  it is +0x10 from the name pointer.)
- Engine render context (76,816 B, `0x12C10`; pooled array at job system +7960): `+0x0`
  `ID3D11DeviceContext1*`; **parameter block at +0x10**: slot `i` occupies 16 bytes at
  `ctx + 0x10 + 16*(i+1)` (`sub_140310750`, `sub_1403105C0`):
  - texture/image parameter: `{arkImageView*, 0}` (`sub_140310750`),
  - buffer parameter: `{0, arkBufferView*}` (`sub_140310720`),
  - scalars/vectors/matrices: values inline (`sub_140310780` float, `sub_1403109E0` vec2,
    `sub_140310A80` vec4, `sub_140310890` 4x4 stored transposed by `sub_14010B430`).
  - Read: `slot(ctx, h) = ctx + 0x10 + 16 * (*(int*)(*(u64*)h + 0x48) + 1)`.
- `Ctx_BindProgram` `0x1402C3630` builds the D3D11 bindings from that block per stage record
  (`sub_1402BDC60(program+40, stage)`, u16 ranges +2..+4 texture SRVs, +4..+6 texture UAVs, +6..+8
  samplers, +8..+10 buffer SRVs, +10..+12 constant buffers, +12..+14 buffer UAVs; parameter list at
  +24); stage 5 = CS (`program+200`). Constant buffers: b0 `perFrameCB` (`qword_14229C1A8`), b1
  `perViewCB` (`qword_14229C308` = view+30648), b2 the per-instance buffer built from the parameter
  block by `sub_1402C2B00` (**H**: b2 = d2e's PerInstanceCB). So every value d2e decodes from
  PerInstanceCB is a parameter-block slot at the hook.

### Object layouts to reach D3D11 objects

`arkImageView` (104 B; members 0-40 from reflection **O**, views from code **O**):

| Offset | Field |
| --- | --- |
| 0x00 | `idImage* m_Image` |
| 0x08 | `arkGFXFormat_t m_format` (DXGI format = `(v >> 13) & 0xFF`; `(v >> 10) & 7` = 4/6 depth SRV remap: low 6 bits 2->56, 3->41, 22->46 (R24_UNORM_X8), 23->21; 5 = stencil: 22->47, 23->22) |
| 0x0C / 0x10 | `m_mipStart` / `m_mipsCount` |
| 0x14 / 0x18 | `m_arraySize` start / count |
| 0x1C / 0x20 | `m_width` / `m_height` (of `m_mipStart`) |
| 0x24 / 0x28 | `m_forceAsArray` / `m_viewHash` |
| 0x30 | `ID3D11ShaderResourceView*` (lazy, `sub_1402BD380`) |
| 0x58 | `ID3D11UnorderedAccessView*` (lazy, `sub_1402BD5A0`) |
| 0x60 | `ID3D11RenderTargetView*` (lazy, `sub_1402BD230`) |

`idImage`: `m_opts` (`idImageOpts`) at +52: `m_type` +0x34, `m_format` +0x38, `m_curWidth` +0x3C,
`m_curHeight` +0x40, `m_depth` +0x54, `m_curNumLevels` +0x58, `m_arraySize` +0x68, `m_numSamples`
+0x6C (reflection **O**); `m_DefaultView` +152; **`ID3D11Resource*` at +0x130** (read by every
view creator, **O**; not reflected).

`arkBufferView` (40 B): `m_pBuffer` +0, `m_firstElement` +8, `m_numElements` +12, `m_format` +16
(reflection **O**); SRV cache +0x18 (`sub_1402D36D0`, BUFFEREX), UAV cache +0x20 (`sub_1402D37C0`)
(**O** code). `arkBuffer`: `ID3D11Buffer*` at +0 (`**view`, **O** code), `m_ByteWidth` +12,
`m_StructureByteStride` +16, `m_StructuredBufferMaxElements` +20 (reflection **O**).

At the hook the views needed by TAA already exist (`Ctx_BindProgram` at 0x1403277F0 created them).

### FSR inputs: engine source vs d2enhanced source

| FSR input | Engine-level source at 0x140327817 | d2enhanced source | Status |
| --- | --- | --- | --- |
| color | slot `viewColorMap` (0x1422B25D8) = `hdrBuffer` view; texture `idImage+0x130`; size `m_width/m_height` (full display size, rendering in top-left subrect) | CS t2 SRV | engine path **O**; d2e runtime format R11G11B10_FLOAT (26) display-size: not re-measured |
| output | slot `taaResult` (0x1422B3500) = `hdrBuffer2`; UAV `view+0x58` | CS u1 UAV | **O** |
| motion vectors | slot `motionVectors` (0x1422B27E0); read list 0xFFD54B9F | CS t1 SRV | **O**; d2e: R16G16_FLOAT full size, viewport-UV units, cur-prev, scale (-renderW,-renderH): not re-verified (shader) |
| depth | slot `fx/depthFull` (0x1422B1F18) = `decompressedDepth` (read list 0x21181874), set at 0x1403274CD | t0 of depth-downsample CS d1df844725b21ff5 (R24G8, SRV R24_UNORM_X8) | **O** binding; same texture as d2e's: **H**. Create own SRV with the remap above if `view+0x30` is a different format |
| exposure | slot `PostFX/Luminance/buffAutoExposure` (handle 0x1422B28E0, `{0, arkBufferView*}`), set in `SetPerViewShaderParms` 0x140358A4B from `frameData+632 -> +24` (`arkStructuredBuffer<autoExposure_t>`, RTTI string at 0x1422FD6E0) | CS t3 SRV | **O**; element 32 B and field order (EngineLuminanceFactor first) from d2e (runtime/RDEF), not re-verified |
| exposure index | slot `PostFX/Luminance/exposureIndex` (0x1422B2900, uint2) | PerInstanceCB +128 `.y` | see Auto-exposure |
| exposure scalars | slots `PostFX/Luminance/customEVBias` (0x1422B2920), `PostFX/Luminance/white_level` (0x1422B2960), `env/tonemapping/white_level` (0x1422B2F80), `useCompressedHdrBuffers` (0x1422FA798 = `dword_14397A6F8 != 1`, set in `SetPerViewShaderParms`) | PerInstanceCB/PerViewCB shadows | **O** slots; mapping white_level -> d2e `view_white_level` / `env_white_level`: **H** |
| render size | view `m_viewport` (+30816) extents `x2-x1+1, y2-y1+1` (exactly what TAA uses); also `m_renderWidth/Height` +30792/+30796 | PerInstanceCB `taaditherandviewportsize.zw` | **O** |
| display size | view `m_unscaledViewport` (+30840) extents; texture `m_width/m_height` | texture desc | **O** |
| jitter | view `m_subPixelJitter` +30800 (prev +30808); details in Jitter | PerViewCB `jittervectors` | **O** (offsets reflection + `SetPerViewShaderParms`) |
| projection / near / FOV | `m_projectionMatrix` +29584 (engine memory is the transpose of the CB: CB [14] = near -> engine [11] = +29628, CB [11] = -1 -> engine [14]); `m_renderRV.fov_y_RADIANS` +14960 (fov_x +14956) | PerViewCB projection @176 | offsets **O**; near = 0.01, reversed-Z infinite from d2e runtime (**H** here) |
| previous matrices | `m_previousProjectionMatrix` +29712, `m_previousViewMatrix` +29904, `m_previousViewProjectionMatrix` +30096; current VP +29968, inverse VP +30032 | PerViewCB | **O** |
| frame number | `*(u32*)(rs+0xCC)` = `dword_1432FC64C` (TAA parity, jitter index `& 7`); `m_previousFrameNumber` +30644 equals it after `idRenderView_Setup` | PerViewCB `framenumber` @792 | **O** |
| reset | see below | frame-number gap / generation | |
| frame time | `m_renderRV.m_deltaTimes[5]` (`durationTime_t`, float seconds) at view+14848; `SetPerViewShaderParms` uses [1] (+14852) or [3] (+14860, if `m_useWorldTime` +14804) | QPC between dispatches | offsets **O**; index meaning **H**. QPC between FSR dispatches is equally valid |

Reset / camera cut sources:

- `m_justTeleported` (view+48380, **O** reflection) is set to 1 by the view constructor
  `sub_140373950` (0x140373A81) and **consumed and cleared in `idRenderView_Setup` 0x140374E40**
  (test 0x1403760A6, previous origin/view matrices copied from current, cleared at 0x14037624B with
  `r13 = 0` from 0x140375D3D). It is always false at the TAA site. **O** (other writers searched in
  0x140200000-0x140400000 only). Capture it at `idRenderView_Setup` entry if a cut signal is wanted.
- Engine TAA reset counter `rs+8` > 0 (TAA re-enabled). **O**
- Ours: first frame, frame counter not consecutive for this pass, render/display size change, FSR
  context recreated, mode switch. **H** (design)

### Auto-exposure (question 5)

- Buffer: `PostFX/Luminance/buffAutoExposure` = `arkStructuredBuffer<autoExposure_t>` held by the
  current frame data (`sub_14035EC00(rs)` = `*(rs+54728) + 976 * *(int*)(rs+54744)`, buffer object at
  `frameData+632 -> +24`), bound for every pass by `SetPerViewShaderParms`. **O**
- Writer: `sub_140325910`, called directly by `PostFX_TemporalAA_and_Post` at **0x140327C28**, after
  TAA in the same pass (gated by view bytes +14916 bit0 / +14918 bit1): three CS dispatches (histogram from
  `texDownsampledHDR`, histogram read, then material `PostFX/Luminance/autoExposure`
  `qword_1422B2860`); between them it sets `exposureIndex = (c, 1-c, 0, 0)` with `c = *(int*)(frameData+640)`
  (0x140325C04) and flips `frameData+640`. **O**
- So **TAA reads the exposure produced by the previous frame's POSTPROC**; nothing writes the buffer
  before TAA in the same frame. **O** (call order inside the pass; no other writer searched, **U**).
- Which element the autoExposure CS writes (`.x` or `.y`) is not visible in the exe (**U**; needs its
  DXBC). TAA and tonemap use `.y` (d2e shader analysis). If the CS writes `.y = 1-c`, the latest
  value at the next frame's TAA is element `*(int*)(frameData+640)` (after the flip). **H**
- Caveat: `exposureIndex` is set only in `sub_140325910`, after TAA, in the **pooled** context's
  parameter block. TAA's value is whatever the last POSTPROC that ran on that worker context left
  (possibly several frames old, so possibly the other ping-pong element). d2e saw `.y = 0` in its
  sample. Using `frameData+640` directly is the engine-level fix. **H**
- Format: structured buffer, stride 32 (d2e runtime; `arkBuffer.m_StructureByteStride` at +16 gives
  it at run time), field 0 `EngineLuminanceFactor` (d2e). The FSR exposure value d2e computes:
  `exp(-customEVBias) * ELF * (compressedHdr ? ELF * viewWhite : 1)` (pack.hlsl) on the GPU into a
  1x1 R32_FLOAT; FSR upscale mode rejects a missing exposure resource unless AUTO_EXPOSURE (d2e).

## Command-list flow

### Recording (**O** unless stated)

- Scheduler ("job system") = `rs + 6325184` (`sub_140373680`). Deferred mode = byte `jobsys+8000`,
  from cvar `r_schedulerUseDeferredCtx` (value `dword_1432D5378`, default "1", flags 0x4002
  command-line only; -1 = auto by `DriverCommandLists`, resolved in D3D init `sub_1403694A0`).
  `r_schedulerThreadingMode` (`dword_1432D52F8`, default 2) = GPU passes on all workers.
- Context pool: `sub_1402C7580` creates one engine context per worker (`jobsys+7960`, count
  `jobsys+7968`), each with `ID3D11Device1::CreateDeferredContext1` (`sub_1402BBA80`, device vtable
  +352) in deferred mode, else wrapping the immediate context (`sub_1402BBAB0`). Count = `jobsys+7968`
  (where it is sized was not traced, **U**); contexts are indexed by worker index in
  `sub_1402C7AC0`. d2e saw 8 deferred contexts at run time (**H**: one per worker).
- Worker loop `sub_1402C8490` (job function): picks a ready pass (`sub_1402C7AC0`), which assigns
  the worker's pooled context (`sub_1402C5E60`: `pass+0x108 = ctx`, `SetPerViewShaderParms(ctx+0x10,
  view)`, unbinds GS/HS/DS, all SRVs/UAVs/RTVs, VS/PS/CS/input layout, clears the context's shader
  caches), then records with `sub_1402C5D50` (call 0x1402C85B6): runs the callback
  (`PostFX_TemporalAA_and_Post`) and **`FinishCommandList(FALSE, &list)` at 0x1402BBA4B** (in
  `sub_1402BBA20`, called from 0x1402C5DAD) -> `pass+0x118`. One command list per pass per frame:
  the whole post chain (TAA, motion blur, stage-3 effects, exposure, bloom, tonemap, stage 4) is one
  list on one worker's deferred context. (d2e "one job / one deferred ctx": corroborated.)
- Completion `sub_1402C77B0` (0x1402C85C6): sets flag 0x400, releases the pooled context
  (`sub_1402C5C90` -> `sub_1402BBB50` clears the context's state-cache bits), pushes the pass to the
  in-order submit ring (`sub_1402C6DA0`, ring at `jobsys+4248/+4316/+4320`) once its counter allows,
  and tries to submit (`sub_1402C72B0`).

### Submission (**O**)

- `sub_1402C72B0` (callers 0x1402C7874 in `sub_1402C77B0`, 0x1402C7C00 in `sub_1402C7AC0`,
  0x1402C857D in the worker loop): **try-lock `jobsys+7988`** (one submitter at a time), pops the
  ring in order, and for passes with 0x400 calls **`sub_1402C5F60(pass)`** (0x1402C739D).
- `sub_1402C5F60`: if the pass has flag 0x20, runs the callback now on the immediate engine context
  `qword_1432D4EF0`; else takes the **immediate-context spinlock** `sub_14036A6F0(rs)` (rs+6532172,
  enabled by byte rs+6532170), **`ExecuteCommandList(immediate qword_1432D4EF8, pass+0x118, FALSE)`
  at 0x1402C6067** (`FF 90 D0 01 00 00`), unlocks `sub_14036B4E0(rs)`, releases the list.
- So the submit thread is **whichever worker wins the try-lock**, not a fixed thread. Only other
  `ExecuteCommandList` in 0x140280000-0x140380000: loading-context path `sub_14035EA10` 0x14035EAB0.
- The AMD "replay through the immediate context" behaviour (d2e) happens inside that
  `ExecuteCommandList`; it is runtime behaviour, not checked statically.

### Engine context state caches (relevant to any record-time D3D11 work, **O**)

| Offset | Cache |
| --- | --- |
| 0x12AD0 | worker index |
| 0x12AD4 | byte: context is not recording (no Finish) |
| 0x12AD8 | valid bits: bit0 render states, bit1 viewport, bit2 scissor (cleared on release) |
| 0x12B04 | stencil ref (byte) |
| 0x12B08 / 0x12B20 | viewport / scissor rect copies |
| 0x12B70 / 0x12B78 / 0x12B80 | RS / blend / depth-stencil state last set (`sub_1402BBFA0`) |
| 0x12B88 | input layout last set (`Ctx_BindProgram`) |
| 0x12B90..0x12BA0 | VS / PS / CS (cleared at pass start) |

`sub_1402BBFA0` skips `RSSetState`/`OMSetBlendState`/`OMSetDepthStencilState` when bit0 is set and
the cached pointer matches; `Ctx_BindProgram` skips `IASetInputLayout` on a match.

### Is splitting the list at TAA safe? (d2enhanced v2)

d2e records a pack pass, calls `FinishCommandList(TRUE)` on the deferred context at the TAA dispatch
(part A), records unpack at the start of part B; the engine's own `FinishCommandList` (0x1402BBA4B)
then yields part B into `pass+0x118`; at the immediate `ExecuteCommandList(part B)` it executes A,
signals, flushes, runs FSR on D3D12, waits, then lets B run.

- With **`TRUE`** the deferred context state is preserved for part B, so the engine caches above stay
  truthful: safe. With `FALSE` the D3D11 state would reset while bit0/caches still claim RS/blend/
  DS/input layout/viewport are bound: later post draws could skip state and render wrong. **O**
  (cache logic) / **H** (consequence).
- Anything the injected work binds must be restored (d2e uses state guards) or the cache invalidated:
  zero `ctx+0x12AD8` and the pointers at 0x12B70..0x12B88 so the next material/program bind sets all
  state. **H** (follows from the code above).
- After the TAA dispatch the engine itself re-binds viewport, scissor, RT, programs and resources for
  every following section; nothing else persists across the split point. **O**

### Engine-level alternative without a split (recommended)

TAA is the **first GPU work of the POSTPROC list** (only parameter stores, viewport/scissor and, on
reset, a history clear precede it, **O**), and all its inputs are produced by earlier passes
(`hdrBuffer`, `motionVectors`, `decompressedDepth` are in the pass read/write lists, so the frame
graph orders their producers first, **H**) or by the previous frame (exposure, **O**). FSR can
therefore run on the immediate context **just before the POSTPROC command list is executed**:

1. **Record time** (worker thread): call-site hook at **0x140327817** as described in Engine TAA site:
   capture inputs, queue `FsrJob{pass, frame, resources, view snapshot}`, skip `Ctx_Dispatch`.
   No D3D11 command is recorded, so no split and no state/cache concerns.
2. **Submit time**: MinHook **`sub_1402C5F60`** (entry, `rcx` = pass). If `pass+0xD8 ==
   0x140327270` and a job for `(pass, frame)` is queued: take the engine immediate lock
   (`sub_14036A6F0(rs)`; it is a non-reentrant spinlock, so release it with `sub_14036B4E0(rs)` before
   calling the original), record pack -> `Signal` -> `Flush` on the immediate context, D3D12 `Wait`
   + `ffxDispatch` + `Signal`, D3D11 `Wait`, unpack into `hdrBuffer2` (and, in upscale mode, any
   display-size motion-vector expansion), then call the original, which executes the POSTPROC list.
   `ExecuteCommandList` clears the immediate state before and after a list (D3D11 contract), so our
   immediate-context state cannot leak into the engine's lists. (Alternative: patch the
   `call [rax+1D0h]` at 0x1402C6067, already inside the lock.)

Advantages over the d2e split: no `FinishCommandList`/`ExecuteCommandList` D3D11 hooks or list
pointer matching (d2e has to erase stale entries for reused list addresses), no dependence on the
AMD replay, works the same on DXVK and on drivers with native command lists, and our own pre-TAA
passes (SSGI/SSR, edge repair) also run on the immediate context at that point.

Rejected alternative: setting flag 0x20 on the POSTPROC pass makes the engine run the whole post
chain on the immediate context at submit (`sub_1402C5F60`), but that path skips
`SetPerViewShaderParms` and the per-pass state reset of `sub_1402C5E60` (only done for pooled
contexts) and moves all post recording onto the serial submit path. **O** (paths) / **H** (cost).

### Risks

- The submit hook runs on whichever worker submits; serialise FSR with our own mutex and keep the
  D3D12 queue/allocators thread-agnostic. **O** (varying thread)
- Cross-API GPU waits inside submission: d2e runs exactly this fence pattern successfully, but
  void-inject's former takeover (removed 2026-10-17), run 3 (STATUS E047), hit a TDR with per-frame D3D11 waits inside the replay
  (cause not isolated). Keep a no-FSR fallback and watch for device removal.
- One POSTPROC pass instance per renderer is registered (count 1, **O**); whether secondary
  renderers/views (render-to-texture, menus) also run it is **U**. Key jobs by pass pointer and frame
  counter and drop unmatched jobs.
- Assumes a pass is not re-recorded before its previous list is submitted (single `pass+0x118`
  slot, **O**; no overlap **H**).
- TAA must be on (skip branches 0x14032752B/0x140327535).
- `exposureIndex` staleness on pooled contexts (Auto-exposure).
- When FSR is turned off, re-arm the engine TAA history reset (`rs+8`).
- With `r_schedulerUseDeferredCtx 0` (see `performance-baseline.md`, CPU submission) passes
  record directly on the immediate context and `sub_1402C5F60` never executes a list: then
  run FSR inline in the record-time thunk at 0x140327817 (the immediate context is current,
  single GPU-pass worker) instead of the submit-time hook.

## Jitter

Shader evidence in this section comes from disassembling (in memory, `d3dcompiler_47`
`D3DDisassemble`) the 8,438 PS / 6,195 VS DXBC dumps in `d2enhanced/captures/shaders/` (read only).
Pass order evidence comes from the RenderDoc dumps `void-rs/local/rdc/A_original/f16159|f6210|f7805/events.tsv`
(original renderer, native 3440x1440, no scaling; event ids "eid" below are from f16159).

### Engine path (all **O**, IDB)

- `idRenderView_Setup` `0x140374E40` is only called by `idRenderView_SetupScaled` `0x140374B60`
  (0x140374B91). It takes `w = rect.x2-x1+1`, `h = rect.y2-y1+1` from the rect `SetupScaled` got
  from `RenderAllViews`, i.e. the **unscaled** viewport (scaling happens afterwards in `SetupScaled`).
  d2e NOTES line 262 ("unscaled viewport size"): corroborated. d2e lines 59/152 ("fraction of render
  size") are only true at scale 1: the divisor is the unscaled size.
- Order in `Setup`: `m_previousSubPixelJitter (+30808) = m_subPixelJitter (+30800)` **unconditionally**
  (0x140375AA2), then `idRenderView_BuildJitteredProjection(&m_renderRV, &m_projectionMatrix, zNear,
  _, &m_subPixelJitter, w, h)` at 0x140375AE4. `zNear` = `r_znear` (`dword_14393740C`), x0.25 if
  `renderView_t.cramZNear` (+164). On a new frame (`rs+0xCC != m_previousFrameNumber`) the previous
  view / view-projection matrices are copied first, and `m_previousProjectionMatrix` takes the old
  projection (0x140375542), so previous matrices carry the previous jitter.
- `BuildJitteredProjection` `0x1403742B0`: explicit projection if `renderView_t.useExplicitProjectionMatrix`
  (+320). Else, if renderView byte +103 bit 2: offsets `r_subpixelXoffset/w`, `r_subpixelYoffset/h` plus
  optional LCG noise scaled by `r_jitter` / `r_jitterXFactor` / `r_jitterYFactor`, and **`m_subPixelJitter`
  is not written** (screenshot path). Default path: `i = *(u32*)(rs+0xCC) & 7` (0x14037442F..0x14037443B;
  `rs` = `0x1432FC580`, counter incremented once per frame by `inc [rcx+0CCh]` at 0x140361A4C in
  `sub_1403619D0`), `jit = table[i] * (1/w, 1/h)`, stored to `m_subPixelJitter`. The frustum is shifted
  by `2*tan(fov/2)*zNear*jit` per axis and built by `sub_1401099E0` -> `sub_1401098E0` (off-centre
  perspective). The function has **no TAA-dependent branch**: views jitter whether TAA is on or not.
- Parameter `subpixelOffset` (`qword_1422FD248`, `cb_subpixeloffset` @160) =
  `(r_subpixelXoffset/w, -r_subpixelYoffset/h, r_jitterXFactor, r_jitterYFactor)`: static user offsets,
  not the TAA jitter.
- `g_taaJitterTable8` `0x141B35F10`: bytes read from the IDB match d2e lines 56-58 exactly
  ((0.0625,-0.1875) (-0.0625,0.1875) (0.3125,0.0625) (-0.1875,-0.3125) (-0.3125,0.3125) (-0.4375,0.0625)
  (0.1875,0.4375) (0.4375,-0.4375)). Corroborated. Its only xref is 0x140374441.
- Other caller of `BuildJitteredProjection`: wrapper `sub_140374680` (same zNear logic), called from
  game code `sub_140890C50` / `sub_140890DD0` (purpose **U**). It would also read a patched table.
- `SetPerViewShaderParms` `0x1403586A0`: `jitterRelativeToPreviousFrame` (`qword_14229C468`, cb @752)
  = `2*(prev - cur)` (0x140358919..0x140358931); `jitterVectors` (`qword_14229C448`, cb @464) =
  `(cur.xy, prev.xy)` (0x140358974..0x140358996). d2e line 63: corroborated. The values reach the GPU
  through `PerViewCB_Upload` `0x1403590F0` (renamed here), called once per view per frame from
  `RenderSingleView` (`sub_140351780`, 0x140351F65); see Render scale.

### Motion vectors are jitter-free (**O**, shader code)

- 202 PS compute velocity as `(cur.xy/cur.w - prev.xy/prev.w - cb_jitterrelativetopreviousframe) *
  (0.5, -0.5)`, e.g. `ps_885d5177f134b382` (7 instructions) and `ps_104ebfab300484c8` (prev position =
  world position x `cb_previousviewprojectionmatrix`, static geometry). All 203 PS that use
  `cb_jitterrelativetopreviousframe` (RDEF not `[unused]`) are such velocity writers; no PS divides two
  interpolated clip positions without the subtraction (regex over all 8,438 PS). The VS (e.g. skinned
  `vs_030c30375eb3b1fb`) output current clip from `cb_viewprojectionmatrix` (jittered) and previous from
  `cb_previousviewprojectionmatrix` (previous jitter).
- NDC jitter is `-2*jit` (below), so the jitter part of `cur_ndc - prev_ndc` is `2*(prev - cur)` =
  `jitterRelativeToPreviousFrame`: the subtraction cancels it exactly. d2e line 134 ("likely
  jitter-free", **H** there): corroborated by code. That exactly these shaders run in the velocity
  prepass is **H** (capture f16159: prepass PS bound with PerViewCB only, eids 261-4622).
- Units: viewport-UV delta, current minus previous, y down. FSR `motionVectorScale = (-renderW,
  -renderH)` (d2e line 133: corroborated by the shader).
- Because the cancellation uses the view's own `m_previousSubPixelJitter`/`m_subPixelJitter`, any
  jitter sequence fed through the engine table stays cancelled.

### Sign and units

- `PerViewCB` projection memory `[8] = 2*jit.x`, `[9] = 2*jit.y`, `[11] = -1` (d2e runtime) is the
  off-centre term of the frustum above (**O** for the formula, values **H**). With `w_clip = -z_view`,
  `ndc += -2*jit`. The image is drawn into the scaled `m_viewport` (`rW = x2-x1+1` of +30816), so the
  shift in render pixels is `(-jit.x*rW, +jit.y*rH)` (y flipped). With `jit = entry / W` (unscaled
  width): **FSR jitter = `(-jit.x*rW, +jit.y*rH)` = `(-entry.x*rW/W, +entry.y*rH/H)`**. **H** (math);
  agrees with d2e's measured default signs -1/+1 (NOTES lines 155, 222): corroborated.
- Inverse, to make the engine render FSR phase `(jx, jy)` (render pixels): `entry = (-jx*W/rW, +jy*H/rH)`.
  At native AA `entry = (-jx, +jy)`. This matches d2e line 263 (Halton / scale) up to the sign
  convention they apply in the shader inputs.

### Recommended feed for an N-phase FSR sequence

`N = ffxQueryDescUpscaleGetJitterPhaseCount(renderW, displayW)`, offsets from
`ffxQueryDescUpscaleGetJitterOffset(index, N)`. Typical N: 8 (native), 18 (1.5x), 32 (2x), 72 (3x).

1. **Table fill, no code patch (recommended).** `VirtualProtect` the 64-byte table RW once (after the
   SHA gate). Each frame, before the frame's views are set up, write the same entry into all 8 slots:
   `entry = (-jx*W/rW, +jy*H/rH)` for phase `k = frame % N`. The index `rs+0xCC & 7` then no longer
   matters. A natural place is a MinHook on `idRenderView_SetupScaled` entry filtered on
   `a3 == &g_resolutionScale` (0x1422FD9F0): it has the unscaled rect (`a2`) and the scale (`a3`), so
   `rW = floor(W*s + 0.5)` can be computed exactly as `ScaleScreenRect` will, and it runs before
   `Setup` for that view in the same frame.
2. **FSR reads the jitter back from the view** at the TAA hook: `jit = m_subPixelJitter` (+30800),
   `rW/rH` from `m_viewport` -> `(-jit.x*rW, +jit.y*rH)`, `jitterPhaseCount = N`. This stays correct
   even if the phase bookkeeping and the recorded frame drift apart (worker-thread recording), because
   the value used to build that frame's projection is what is read.

Rejected: patching `and eax,7` / the `lea` of the table at 0x14037443B/0x140374441 to a mod table works
only for power-of-two N and changes code bytes; replacing `BuildJitteredProjection` (MinHook) means
re-implementing the frustum, `sub_1401099E0` call and the `subpixelOffset` parameter.

Risks: other views that call `Setup`/`sub_140374680` in the same frame get the same jitter (harmless
without TAA, **H**). The screenshot path (renderView byte +103 bit 2) leaves `m_subPixelJitter` stale:
treat it as no-jitter/reset. If `Setup` ran twice for one view in one frame, `prev = cur` (0x140375AA2
is unconditional) and that frame's MV cancellation would be off by one phase (**H**, not observed).

## Render scale and post chain

### Engine resolution scaling (corroboration, **O**)

- `ResScale_ApplyCvars` `0x140377540` (called from `sub_1403591B0` at 0x14035928C): `prev = cur`; if
  `rs_enable` (int 0x14397BEE8, object 0x14397BEC0) is 0 -> `(1,1)`; else `cur = controller value`
  (`g_resolutionScale+16`), overridden per axis by `rs_forceFractionX/Y` (floats 0x14397BFEC /
  0x14397C06C, objects 0x14397BFC0 / 0x14397C040) when in (0,1]; `rs_oscillate` (0x14397C0E8) blends
  with `sin(t * rs_oscillateFrequency)` (0x14397C16C). d2e lines 66-70: corroborated. d2e "rs_enable is
  a bitmask" (line 291): not visible here, this function only tests `!= 0` (**U** for the controller).
- `RenderAllViews` `0x14035DC60` calls `idRenderView_SetupScaled(view, rect, &g_resolutionScale, 0)` at
  0x14035E06A for world views; the two views of the `v5 == 2` path use `(1,1)` (0x14035DDA7, 0x14035DDDC).
  Other `SetupScaled` caller: `sub_140366590` (**U**).
- `SetupScaled` `0x140374B60`: `Setup(unscaled)`; `m_unscaledViewport`/`m_unscaledScissor` = rect;
  unscaled subrects (30568/30616) = subrect argument or rect; `m_viewport = m_scissor =
  ScaleScreenRect(rect, scale)`; `m_renderWidth/Height` = scaled extents; scaled subrects (30544/30592)
  only when a subrect is passed; `m_previousResolutionScale = old m_resolutionScale`;
  `m_resolutionScale = scale`. d2e line 73-74: corroborated.
- `ScaleScreenRect` `0x140377440`: `x1' = floor(x1*s+0.5)`, `w' = floor(w*s+0.5)` (d2e line 241:
  corroborated). Shaders see the raw float through `cb_resolutionscale`, hence d2e's snap
  `s = round(W*s)/W` per axis is required.
- Other readers of `g_resolutionScale`: HBAO+ setup `sub_140305620` (`min(sx, sy)`), `com_showFPS`
  overlay `sub_14004D840`, `sub_14034F840` (**U**). Readers of `m_previousResolutionScale` (+30920):
  only `PostFX_TemporalAA_and_Post` (`taaTexSize`) and `SetPerViewShaderParms` (parameter
  `prevResolutionScale`), scan of 0x140280000-0x140390000.

### Frame pass order around TAA

GPU order from the three captures (identical structure; f16159 eids), attributed to engine passes via
the pass table `sub_14034FE00` (callbacks below) and resource-name hashes found in each callback (**O**
for both; the eid-to-pass attribution is **H** where marked):

| eids | Work | Engine pass |
| --- | --- | --- |
| 261-4622 | depth + velocity prepass (R16G16_FLOAT + D24S8), incl. alpha-tested `ro_fx_*` particles (4408, 4422) and skinned/viewmodel models | `ARK_RENDERPASS_DEPTHPASS` `sub_1402FD8D0` |
| 4668-4700 | depth copies + "Depth buffer hack" fix-up (`unhackDepth`, `oDepth = depth * cb_modeldepthhack`) | inside `sub_1402FCFF0` (0x1402FD406..) |
| 9402-9416 | distortion particles into 1720x720 RGBA16F with half-res depth | `ARK_RENDERPASS_DISTORTION` `sub_1403222F0` (writes `distoVectors`, DSV `depthAlphaDiv2`) **H** |
| 10227-11012 | transparent / fx geometry depth only, full then half res | `ARK_RENDERPASS_DEPTHALPHA_LIT` `sub_1403222A0` (`depthAlpha`, `depthAlphaDiv2`) **H** |
| 19788-24000 | forward opaque (tlf), emissive/lightmap (22842), **sky** (23084: RT0 HDR + RT1 R8 `skyCloudMaskBuffer`), SSS MRT (23779), unlit (23869), hair (24000) | `ARK_RENDERPASS_FORWARD_WORLD/DYNAMIC` |
| 24028-24087 | sun disk, half-res distance fog, volumetric light scattering, blur | `ARK_RENDERPASS_FOG` `sub_140309430` |
| 24134-29335 | water depth, ocean colour (339 tiles), full-res distance-fog composite | `ARK_RENDERPASS_RENDER_WATER` / fog |
| 29400-29796 | transparent lit surfaces, hair and particles into OIT MRT (RGBA16F + R8 / RGBA16F + R16G16_FLOAT) | `ARK_RENDERPASS_EMISSIVE` `sub_14031F220` (sets `OIT/Enable`, writes `AccumulationBuffer`, `OpacityBuffer`) / `ALPHA_LIT` **H** |
| 29814 | OIT resolve onto HDR (`ps_8c57a4fa8f20fa0d`) | `ARK_RENDERPASS_ALPHA_LIT` `sub_14031DCE0` (binds `OIT/*` at 0x14031EE5D, `r_OIT != 0`) |
| 29841-29989 | depth of field (near fuzziness, `dofdatamap` RG8, gather) | `ARK_RENDERPASS_DOF` `sub_140315450` |
| 30008 | TAA dispatch | `ARK_RENDERPASS_POSTPROC` |
| 30038-30495 | distortion (2 draws), bloom/exposure, lens flares, colour cube, tonemap, `RENDER_UPSCALE` | `POSTPROC`, `POST_TONEMAP` `sub_140323E70` |

**DoF runs before TAA** (3/3 captures). d2e's list of render-scaled DoF inputs read "after the switch"
(NOTES 317-319) does not apply to a switch placed after the TAA dispatch: `ARK_RENDERPASS_DOF` is a
separate pass recorded with the real view. Whether a second, post-TAA DoF variant exists (the frame
graph wires DoF inputs twice, string refs at 0x1403561A0 and 0x140357365 in `sub_140351780`) is **U**.
The DoF "refraction map" RG16F@3440 is most likely the OIT `OpacityBuffer` (the DoF wiring at
0x14035616D reads `OpacityBuffer`; it is MRT1 RG16F in the 29461+ draws) **H**.

### Every scaled read after the TAA dispatch

CPU reads of the view (all **O**, `r15`/`r13`/`rsi`/`rax` = view):

| Address | Function | Field | Use |
| --- | --- | --- | --- |
| 0x140327859 / 0x14032787E | `PostFX_TemporalAA_and_Post` | `m_viewport` / `m_scissor` (or 30592 subrect) | debug draw (view byte +16952 bit3) |
| 0x1403279D3 / 0x1403279F8 | same | `m_viewport` / `m_scissor` | before the distortion pass |
| 0x140327ABA / 0x140327ADF | same | `m_viewport` / `m_scissor` | stage-3 screen effects `sub_14031DA70(pass,3)` (full-screen material list per stage at `pass+56`, 144-byte entries) |
| 0x140327EBA | same | `m_resolutionScale` | `exrBuffer` capture rect (`byte_1439A529C`, debug) |
| 0x1403269A3 | `PostFX_DistortionVectors` `0x140326930` (renamed; d2e "motion blur") | `m_resolutionScale` | `ScaleScreenRect(target, scale)` viewport |
| 0x1403260A3 / 0x1403261EB | `sub_140325D80` (lens flares / bloom chain) | `m_resolutionScale.x/.y` | blur step = texel x scale |
| 0x140326547 | `sub_1403263E0` (half-res chain before tonemap) | `m_resolutionScale` | `ScaleScreenRect(half size, scale)` |
| 0x140326ED1 | `sub_140326AF0` (tonemap) | `m_resolutionScale` | `ScaleScreenRect(srgbBuffer, scale)` viewport/scissor |
| 0x140324074..0x1403240E2, 0x140324415..0x14032453C, 0x1403246E0..0x14032473F | `sub_140323E70` (`POST_TONEMAP`) | `m_viewport`, `m_scissor`, subrects, `perViewCB` / portal CB | post-tonemap surface draws `sub_140319900` (d2e: in-world UI), FXAA (`r_AAMode`, unscaled viewport), `RENDER_UPSCALE` |
| 0x1403215D8..0x140321627 | `sub_140321590` | same | `r_showAlphaDebug` only |

Not scale-dependent on the CPU: exposure downsample `sub_1403257A0` (full mip rects, the shader
scales UVs), auto-exposure `sub_140325910` (texture sizes), `PostFX_LensDistortionCheap` `0x1403262D0`
(`r_useCheapLensFlareDistortion`, full target rect), stage 3/4 `sub_14031DA70` (uses the viewport set
by the caller).

GPU reads: `cb_resolutionscale` (PerViewCB @776) is used by 2,976 dumped shaders, including the post
shaders above. PerViewCB is filled **once per view per frame** by `PerViewCB_Upload` `0x1403590F0`
(`sub_1402AF360(view+30648, immediate ctx, params qword_1432F3538)`: Map/WRITE_DISCARD + copy of each
variable from the global parameter block), and the same function fills the **portal CB
`m_portalViewConstantBuffer` (view+30720)** from the same parameters (matrices replaced only when
view byte +14900 bit 0). Every bind of view+30720 in the exe is under `+14900 & 1` (checked in
`sub_1402FCD90`, `sub_1402FD8F0`, `sub_140311AA0`, `sub_14031F220`, `sub_140321860`, `sub_140323E70`),
and that bit also skips TAA (0x14032752B). So for any view that reaches FSR, view+30720 is a byte
copy of PerViewCB that the engine uploads every frame and never binds. **O**

Render-scaled textures read after TAA (pass read lists, hashes FNV-1a incl. NUL, **O**):

| Resource | Reader | Problem at display res |
| --- | --- | --- |
| `distoVectors` 0x1BBE2D1A (half res, render-scaled subrect; written by `DISTORTION` and by the camera radial-blur PS `ca04fcea5ec50dfa`) | `PostFX_DistortionVectors` x2 (bound as parameter `motionVectors`; PS `35468971580c3d71`: `ro_motionvectors` + `ro_viewcolormap`) | sampled with display UVs: cut-off at right/bottom (d2e NOTES 315-316 "motion blur velocity": same texture, corroborated by hash + bindings) |
| `decompressedDepth` 0x21181874 as `fx/depthFull` for the whole POSTPROC pass (0x1403274CD) | stage-3/4 effect materials, debug draw | depth sampled at wrong place (which effects read it: **U**) |
| `zBuffer` 0xBA729A59 (DSV) | `POST_TONEMAP` surface draws (`sub_140319900`) | depth test against render-scaled depth |
| `depthAlphaDiv2` 0xBB9F8C8F | `POSTPROC` tail | only its size (`/2` temp target): fine |

(d2e's hash/role table in Engine TAA site lists `distoVectors` as "motion blur": it is the screen
distortion pass; motion vectors are not read after TAA.)

### Proposed switch to display resolution (engine level)

1. **PerViewCB twin = engine portal CB.** MinHook `PerViewCB_Upload` `0x1403590F0`. For the FSR view
   (`(view[14900] & 1) == 0`, upscaling active): `sub_1402AF360(view+30648, *qword_1432D4EF0,
   qword_1432F3538)`; set parameter `resolutionScale` (`qword_14229C508`) in `qword_1432F3538` to
   `(1,1)`; `sub_1402AF360(view+30720, ...)`; restore the parameter. Identical to the original apart
   from the twin's `resolutionscale` (the original's matrix block only runs for bit 0). Otherwise call
   the original.
2. **Private view copy for the two post passes.** MinHook the `POSTPROC` callback
   `PostFX_TemporalAA_and_Post` `0x140327270`: copy the real view (`pass+0x70`, `idRenderView` is
   48,384 B by reflection) into a per-frame slot (ring of 3, keyed by `rs+0xCC`), set `pass+0x70` to the
   copy, call the original, restore `pass+0x70`. The copy still holds scaled values, so the TAA site
   hook (0x140327817) captures render-res inputs unchanged.
3. **Switch at the TAA site** (in the 0x140327817 thunk, after queuing FSR): in the copy set
   `m_viewport = m_unscaledViewport`, `m_scissor = m_unscaledScissor`, `m_resolutionScale =
   m_previousResolutionScale = (1,1)`, and copy the 72-byte object at real view+30720 over copy+30648;
   in the pass's parameter block (`ctx+0x10`) store `perViewCB` (`qword_14229C308`) = real view+30720
   and `resolutionScale` / `prevResolutionScale` = `(1,1)`. Everything after the dispatch then runs at
   display size on `hdrBuffer2` (FSR output) / `hdrBuffer` (scratch), both full-size textures.
4. MinHook the `POST_TONEMAP` callback `sub_140323E70`: point `pass+0x70` at the same (switched) copy
   and store the same three parameters; restore after. Its explicit `perViewCB = view+30648` binds now
   pick the twin; `RENDER_UPSCALE` becomes a 1:1 copy.
5. Before `PostFX_DistortionVectors` (i.e. at the same thunk, after FSR): resample the render-scaled
   subrect of `distoVectors` (half of `ScaleScreenRect(full, s)`) to the full texture via a temp copy
   (one half-res pass on the pass's deferred context; restore state / invalidate the context caches
   as described in Command-list flow).

Moving parts: 3 function hooks + the existing TAA-site thunk, one 48 KB memcpy per frame, one small
resample. No per-draw D3D11 constant-buffer swapping, no `ID3D11Device` hooks, no shader patching.
The real view is never modified, so passes recording concurrently on other workers are unaffected.
Compared with d2e (private view copy + own PerViewCB twin swapped per draw + in-place stretching of
several intermediates): same idea, but the twin is uploaded by the engine and selected through the
engine's own parameter, and only one intermediate needs stretching because DoF runs before TAA.

### What still breaks / risks

- Post-tonemap surface draws (d2e: in-world UI) depth-test against render-scaled `zBuffer`. Fix: resample depth with
  an `SV_Depth` full-screen pass (the engine's own `unhackDepth` PS `ps_a36b057ea52d03e6` is the
  pattern) into a display-size D24S8, or in place via a copy, before the GUI draws; stencil is not
  preserved that way (**H**: GUI stencil use **U**). d2e's "DoF DSV cannot be stretched with a colour
  pass" is solved the same way, and does not arise for DoF here (before TAA).
- `fx/depthFull` (`decompressedDepth`) for stage-3/4 effect materials was render-scaled. Done
  2026-10-17: resampled in place at display size right after the FSR pack (which keeps the
  render-scaled copy FSR needs), before the POSTPROC list executes; targets without a
  depth-stencil binding go through a depth-bindable scratch copy (which effects read it: **U**).
- Other post-TAA consumers of render-scaled targets not in the two pass read lists (e.g. effect
  materials sampling `maskBuffer` or `darkVisionAccumulationBuffer`) are **U**; d2e's `rescale.cpp`
  observation list is the runtime check to repeat.
- The shallow copy shares internal pointers (occlusion queries, parameter blocks) with the real view;
  POSTPROC/POST_TONEMAP only read them (**H**, d2e ran the same copy approach).
- Bloom/exposure now process display-size images: slightly more GPU time (native-scale costs from
  the capture: exposure+bloom+lens-flare chain eids 30080-30438 ~0.2 ms, tonemap 0.07 ms, upscale copy 0.08 ms).

## Mip bias

### `image_lodbias` (**O** within the stated search, conclusion **H**)

- Registered by `sub_1413698A0`: object 0x1432EA470, default "0", flags 0x20004. The only read of its
  value is `idImageManager_ApplySamplerCvars` `0x1402EB830` (0x1402EB9B5), which stores it at
  `imageManager (off_1422ABA40) + 104`; `image_anisotropy` goes to +100 (clamped to >= 1). It is
  re-run every frame by `idImageManager_CheckSamplerCvars` `0x1402EB9F0` (from `sub_1403619D0`
  0x140361A5E) when one of five sampler cvars has the modified flag 0x40000.
- **No reader of `imageManager+104` was found**: xrefs to the cvar object fields; all 58 functions that
  reference `off_1422ABA40` decompiled and searched for `+104` / float index 26; float loads from
  `[reg+68h]` in those functions (only DoF parameters in `sub_140313980`).
- Sampler creation for materials/render parms: `idTextureSamplerManagerLocal_CreateSampler`
  `0x140378280` (vtable slot 6 at 0x141B37698): `D3D11_SAMPLER_DESC.MipLODBias = samplerOptions+20`
  (the decl's `MipLODBias`, parser `sub_1402D9300`, which also sets option bit 0x10000 for
  `forceNoMipLodBias`), `MaxAnisotropy = imageManager+100` for non-point filters, then
  `ID3D11Device::CreateSamplerState` (device `qword_1432D4F00`, vtable +0xB8). Samplers are cached by an
  option key (`sub_1403778A0`, "sampler_%d_..."). Other `CreateSamplerState` calls: D3D init
  `sub_1403694A0` (one sampler), Iggy GDraw `sub_1402DFD40` (UI).
- So `image_lodbias` has **no effect** in this build, for existing or new samplers; `image_anisotropy`
  only affects samplers created after the change (cached objects are kept). d2e's
  `image_lodbias = log2(scale) - 1` override (NOTES 395-397, "unverified") is most likely a no-op.

### Which samplers

Material shaders use a handful of global named samplers (bind counts over the 8,438 PS dumps):
`smp_trilinearsampler` 6,830, `smp_anisotropicsampler` 4,408, `smp_bilinearsampler` 4,166,
`smp_shadowhardwarepcfsampler` 3,735, `smp_iblsampler` 3,711, `smp_bilinearclamp` 2,727,
`smp_linearclamp` 2,638, `smp_anisotropicsamplerhq` 1,950, `smp_pointsampler` 1,017,
`smp_decalsampler` 357. **O**. Bias candidates: the anisotropic ones, `smp_decalsampler`, and
`smp_trilinearsampler` if its textures are material textures (texture-to-sampler pairing per shader
**U**). Never: comparison, point, clamp-only post samplers, `forceNoMipLodBias` samplers.

### Recommendation

- Engine level: MinHook `CreateSampler` `0x140378280` and add `b = log2(renderW/displayW) - 1` (FSR 2/3
  guidance; FSR 4-specific guidance **U**) to `options+20` when the mip filter is linear/anisotropic,
  addressing is wrap, no comparison, and bit 0x10000 is clear. The bias is part of the cache key, so
  biased and unbiased samplers do not collide. Install before material load (DllMain, after the SHA
  gate) so every sampler goes through it.
- Runtime scale changes: `sub_1403778A0` allocates an 8-byte cell per sampler (+48) and
  `CreateSamplerState` writes the D3D object into it; `Ctx_BindProgram` loads each sampler as
  `*(slot value)` (0x1402C3A53 `mov rax,[r13+rcx*8+10h]`, 0x1402C3A58 `mov rax,[rax]`) at every bind.
  Recording `(cell, options)` in the hook and replacing the cell's sampler with a re-biased one (release
  the old one after the frames in flight) re-biases without per-draw hooks. **O** (allocation, the
  load through the slot pointer) / **H** (that the slot holds this cell's address).
- D3D-level `ID3D11Device::CreateSamplerState` hook is the fallback; it cannot see `forceNoMipLodBias`
  or the sampler's role.

## Reactive and transparency inputs

### Transparent / composited work between the opaque pass and TAA

From the pass order table (Render scale): forward-blended decals and emissive/lightmapped surfaces
are part of the forward pass (same depth/MVs as their surface); then sun disk, half-res fog and
volumetric scattering, ocean (opaque, with its own velocity pass), full-res distance-fog composite;
then **all transparent lit surfaces, hair and particles go through weighted-blended OIT**
(`r_OIT`, default "1", `dword_1432F71D8`): accumulation in `EMISSIVE` (`sub_14031F220`, `OIT/Enable`,
writes `AccumulationBuffer` 0x82A3E954 / `OpacityBuffer` 0x01208B70), resolve in `ALPHA_LIT`
(`sub_14031DCE0`, 0x14031EE5D: `OIT/Accumulation`, `OIT/Opacity`, `OIT/HairAccumulation`,
`OIT/HairOpacity`); then DoF. Distortion (`distoVectors`) is applied **after** TAA. **O** (code,
capture) / **H** (eid attribution).

### Existing engine targets

| Target | Content | Use for FSR |
| --- | --- | --- |
| `_OPACITY_BUFFER_%p` (`OpacityBuffer`, view-size, cleared to (1,0,0,1), MRT1 RG16F in the particle/translucent draws) | `.x` = revealage of OIT layers (resolve `ps_8c57a4fa8f20fa0d`: `o0.w = hairOpacity.x * opacity.x`, colour weight `1 - opacity.x`) | **coverage = 1 - revealage**: exact transparency mask for everything drawn through OIT |
| `_HAIR_OPACITY_BUFFER_%p` (R8, same clear) | revealage of OIT hair | multiply in |
| `_ACCUMULATION_BUFFER_%p` (RGBA16F) | weighted premultiplied colour sum | optional luminance weighting of the mask |
| `_DepthAlphaBuffer_%p` (`depthAlpha` 0x834D9376, D24S8) | opaque depth plus the nearest transparent/fx depth (DEPTHALPHA_LIT, depth-only alpha-tested draws) | binary "transparent in front" = `depthAlpha != decompressedDepth` |
| `_DistortionVectors_%p` (`distoVectors`, half res) | screen distortion field (+ camera radial blur) | not needed: applied after FSR |
| `_maskBuffer_%p` (`maskingSurf` surfaces; read by DISTORTION and DoF) | **U** | none yet |
| `skyCloudMaskBuffer` (R8, MRT1 of the sky draw 23084) | sky/cloud mask | sky MV fix (below) |

Formats: engine format codes from `sub_14036D0C0` (OPACITY 0x5C5, HAIR_OPACITY 0x541 = same code as
the R8 `skyCloudMask`), clear colours `xmmword_142294E60` = (1,0,0,1) for both opacity buffers. DXGI
formats from the capture MRTs; the code-to-format mapping is **H**.

### Proposed reactive / transparency-and-composition masks

- Build both in our pre-FSR pack pass from the engine targets, at render resolution (same top-left
  subrect as colour): `tc = saturate(1 - opacity.x * hairOpacity.x)`; `reactive = min(0.9, k * tc)`
  (k tunable, optionally weighted by accumulation luminance relative to the scene colour). The buffers
  still hold this frame's values at the TAA site (next clear is next frame's EMISSIVE) **H**. They are
  reachable engine-level as image views from the EMISSIVE/ALPHA_LIT pass write/read lists (hashes
  above) or by capturing the SRVs bound with the resolve PS.
- FSR 4 support for these optional inputs (whether the ML upscaler uses reactive/T&C masks) is **U**:
  measure with and without.
- Not covered by the OIT mask: surfaces ALPHA_LIT draws **after** the resolve (`v219 > 0 || v202` list
  in `sub_14031DCE0`; none in the three captures), `EMISSIVE_ONLY` (`sub_140323210`) and `DARKVISION`
  (`sub_14031F960`, own `darkVisionAccumulationBuffer`) when active, and `r_OIT 0`. For those, a colour
  difference mask is the generic fallback: copy `hdrBuffer` at the `EMISSIVE` callback entry
  (`CopyResource` on the pass's deferred context changes no pipeline state) and compute
  `saturate(|after - before| / max(after, before, eps))` in the pack pass.
- Fog, sun disk and the ocean need no mask initially: depth-attached, smooth, or with their own MVs.

### Sky and viewmodel motion vectors

- **Sky**: drawn in the forward pass (eid 23084), not in the velocity prepass (no sky shader among the
  R16G16_FLOAT draws of f16159), and `_MotionVectors_%p` is created with clear value (0,0,1,0)
  (`sub_14036D0C0`; the velocity PS write `zw = (1,0)` too) -> sky MVs are 0 and sky depth is the reversed-Z far value 0 (renderer-replacement:
  "sky dome drawn at infinity"). **H** (runtime values not re-measured). Under camera rotation FSR then
  treats the sky as screen-static (smearing of clouds/stars; d2e FSR debug "purple speckle on sky",
  NOTES 166-167, consistent). Fix in the pack pass for `depth == 0` (or `skyCloudMaskBuffer` coverage):
  `d = invVP * (ndc.x, ndc.y, 0, 1)` (a direction, w ~ 0 with the infinite projection), `p =
  prevVP * d`, `mv = (ndc - p.xy/p.w - jitterRelativeToPreviousFrame) * (0.5, -0.5)` using
  view+30032 / +30096 (or PerViewCB @480 / @240), i.e. the same convention as the engine's velocity PS.
  Implemented 2026-10-17 (`fsr.rs` pack shader, `fsr_sky_motion = 1`): the engine's matrices are
  read raw, so the multiplication order and the jitter term are not assumed. While the camera
  moves (view matrix differs from the previous one), the pack shader evaluates all six
  variants (`mul(M, x)` / `mul(x, M)` x no / minus / plus `jitterRel`) on every 16th geometry
  pixel against the engine's own vectors and counts inliers (< 0.5 px), read back with
  `DO_NOT_WAIT` copies. `sky_decision` locks the variant once it explains >= 50 % of the samples and
  beats the other multiplication order by 1.5x (>= 30 moving frames, >= 2000 samples), or gives up
  after 900 moving frames (sky vectors stay 0). It then fills pixels with depth 0 and vector 0.
  The log line `fsr: sky motion vectors on: ...` reports the decision and the inlier ratios.
  Offline (`fsr_offline`): both a row-vector and a transposed camera are recognised after 31
  frames; sky vectors within 0.027 px of the analytic value (R16G16 precision).
- **First-person viewmodel**: drawn in the prepass with the normal skinned velocity path (correct
  per-object MVs incl. sway, consistent with d2e's debug view) and with `clip.z *= cb_modeldepthhack`
  (PerInstanceCB `cb2[7].w` in `vs_030c30375eb3b1fb`; parameter `modelDepthHack` set per model from
  renderModel+6328 by `sub_140227C70`, `r_playerDepthHack` default "0.0"). The DEPTHPASS then runs
  "Depth buffer hack" (`sub_1402FCFF0`: copies + `unhackDepth/depthView` PS writing `oDepth = depth *
  cb_modeldepthhack`), which presumably restores comparable depth for later passes **H**. No reactive
  mask for the viewmodel by default; if FSR ghosts at weapon edges, compare the depth FSR receives at
  viewmodel pixels with world depth (whether the unhack covers all of them is **U**).

## Implementation status

(2026-10-17, JitterScaleRE; `void-inject` `fsr.rs` + `fsr_scale.rs`. Offline-verified only;
not yet run in the game.)

### Settings (`void-inject.ini`)

| Key | Default | Effect |
| --- | --- | --- |
| `fsr` | 0 | 0 at startup = no site patched, no hook installed. Live: turning it on installs FSR while the game runs (below); turning it off deactivates FSR like F6 (everything stays installed, inert) |
| `fsr_mode` | `native` | `upscale` = render at `fsr_scale`, FSR writes display size, post chain at display size |
| `fsr_scale` | 0.67 | per-axis render scale (0.33..1), render size rounded to multiples of 8 px |
| `fsr_mip_bias` | `auto` | upscale only: `log2(fsr_scale) - 1`, or a number (-4..4) |
| `fsr_masks` | 1 | reactive + transparency/composition masks from the OIT buffers |
| `fsr_reactive` | 0.9 | reactive = transparency coverage x this |
| `fsr_depth_resample` | 1 | upscale: resample render-scaled depth (`fx/depthFull` after the FSR pack; the depth buffer before the post-tonemap surface draws) with stencil via `SV_StencilRef` |
| `fsr_sky_motion` | 1 | sky motion vectors from the camera matrices (convention measured in game) |

Every `fsr*` key is live (`settings::KEYS`, overlay tag `live`). The only reason keys used to
need a restart was that hooks were installed conditionally at startup; now every FSR hook is
installed together with FSR and each is inert unless its feature is active for the frame:

| Key | How it applies live |
| --- | --- |
| `fsr` | on: `fsr::tick` (main thread, every frame, `hook.rs` observe) calls `fsr::install` if needed: MinHook hooks (MinHook suspends the other threads itself), then both call sites through `mem::patch_code`, which suspends every other thread (Toolhelp snapshot), checks that no thread's RIP lies strictly inside the replaced bytes (else resumes, waits 1 ms, retries up to 200 times), writes, flushes the instruction cache and resumes; no heap allocation while threads are suspended. A failed run-time install is not retried. off: `set_active(false)` (TAA history reset re-armed) |
| `fsr_mode`, `fsr_scale` | `g_resolutionScale` written per frame in `SetupScaled` (upscale: snapped `fsr_scale`; native: 1, so engine dynamic resolution never scales an FSR frame); the post-chain switch only happens when the view is actually scaled; history reset by the render-size change (`dispatch_fsr`); the FFX context (max render = max upscale = display, `DYNAMIC_RESOLUTION`) covers every render size up to display without re-creation; toast `FSR upscale 0.67 (2304x968)` / `FSR native AA (3440x1440)` |
| `fsr_masks` | read per frame (CLEAR/ASYNC_CLEAR hooks always installed) |
| `fsr_mip_bias` (and the auto bias from `fsr_scale`) | live sampler swap (Mip bias below); toast `FSR mip bias -1.58 on N samplers` / `FSR mip bias off` |
| `fsr_auto_exposure` | `fsr::tick` sees the running context's flag differ, clears the ready size: the next TAA frame re-creates the context (same path as a resize), engine TAA meanwhile; toast |

### Hooks (all installed by `fsr::install`, each prologue compared with the analysed build)

| Site | Kind | First bytes checked | Feature |
| --- | --- | --- | --- |
| 0x140327817 `call Ctx_Dispatch` | call redirect (thread-suspending patch) | `E8 44 3C F9 FF` | TAA site: capture, inline run without command lists, post-chain switch |
| 0x1402C6067 `call [rax+1D0h]` | call redirect (thread-suspending patch) | `FF 90 D0 01 00 00` | submit: FSR before POSTPROC, depth resample before POST_TONEMAP |
| 0x140374B60 `idRenderView_SetupScaled` | MinHook | `48 89 5C 24 08 48 89 74 24 10 48 89 7C 24 18 55` | jitter table, render scale, `m_justTeleported` |
| 0x1403590F0 `PerViewCB_Upload` | MinHook (acts in upscale) | `40 53 48 83 EC 20 4C 8B 05 3B A4 F9 02 48 8B DA` | portal CB as PerViewCB twin with `resolutionScale = 1` |
| 0x140327270 POSTPROC callback | MinHook (acts in upscale) | `48 8B C4 55 53 56 57 41 54 41 55 41 56 41 57 48` | private view copy |
| 0x140323E70 POST_TONEMAP callback | MinHook (acts in upscale) | same | switched copy, parameters, depth job |
| 0x140378280 `CreateSampler` | MinHook | `48 8B C4 55 57 41 56 48 8D 68 A1 48 81 EC A0 00` | registers sampler cells (live mip bias) |
| 0x1403110E0 CLEAR / 0x140310FB0 ASYNC_CLEAR | MinHook (acts with masks) | `48 89 5C 24 18 48 89 6C 24 20 56 57 41 54 41 56` / `48 89 5C 24 08 48 89 6C 24 10 48 89 74 24 18 48` | `OpacityBuffer` (0x01208B70) / `hairOpacity` (0x20293876) from the write lists |

The jitter table page (0x141B35F10, first 8 bytes checked = entry 0) is made read-write once.
Any failed hook disables only its feature; missing post-chain hooks disable upscale (native AA).

### What runs

- **Jitter (all modes):** in `SetupScaled` for the world view (`scale == &g_resolutionScale`),
  while FSR is running for that display size: phase `rs+0xCC mod N` of the provider's sequence
  (cached by the runtime from `ffxQuery` for (render, display) widths; FFX 3.x Halton formula
  until then), written into all 8 table entries as `(-jx*W/rW, +jy*H/rH)`. `capture` reads the
  resulting `m_subPixelJitter`, so FSR always gets the jitter the projection used. FSR stopped
  -> original table restored. Native AA now uses FSR's 8-phase Halton sequence instead of the
  engine table.
- **Render scale:** written into `g_resolutionScale` in the same hook (after
  `ResScale_ApplyCvars` ran for the frame), not through `rs_*` cvars: `rs_enable` /
  `rs_forceFraction*` are read for the scale only by `ResScale_ApplyCvars`, but also by the menu,
  the Ansel path `sub_1408B0160` and the controller `sub_140376C80`; the direct write is exact per
  frame (no 30-frame override period, no menu interaction). Render size = `round(W*s/8)*8`: integer
  rects keep `ScaleScreenRect` and `cb_resolutionscale` identical, and multiples of 8 keep the
  half/quarter/eighth targets' scaled rects whole (the engine's own controller steps in 8-px units,
  `sub_140376C80`, **O**). Scaling applies only while FSR is running; F6 off = native engine TAA.
- **Post-chain switch (upscale):** as designed in "Proposed switch": twin CB in
  `PerViewCB_Upload`; POSTPROC runs on a ring copy (3 x 48,384 B); at the TAA site the frame is
  taken only if the pass runs on a copy and the twin of (view, `m_previousFrameNumber`) was filled
  this frame (else engine TAA, which upscales itself). After taking it: copy rects unscaled, scale
  and previous scale 1, `m_renderWidth/Height` = display, copy+30648 = twin object; pass
  parameters `perViewCB` = real view+30720, `resolutionScale` / `prevResolutionScale` = 1, and the
  two viewport parameters `SetPerViewShaderParms` derives from `m_renderWidth/Height`
  (`sub_140310670`, slots at block +74704 / +74708, **O**) rewritten for the display size.
  POST_TONEMAP uses the switched copy of the same (view, frame) and the same parameters.
- **FSR output:** `upscale_size` = display, unpack over the whole display target; pack repairs the
  bright last column/row in upscale (d2e "Bright edge"). `distoVectors` (POSTPROC read list
  0x1BBE2D1A) is stretched from its scaled rect to full size by CS + `CopyResource` after FSR
  (no graphics state touched).
- **Masks:** `tc = saturate(1 - opacity.x * hairOpacity.x)`, `reactive = tc * fsr_reactive`,
  R8 shared textures passed as `transparency_composition` / `reactive`; absent OIT buffers ->
  no masks.
- **Camera cut:** `m_justTeleported` seen at `SetupScaled` entry (before `idRenderView_Setup`
  consumes it) -> `reset` for that view frame; counted in the periodic log line.
- **Mip bias (live):** layout **O** (IDB, comments at 0x1422FDA20 / 0x1403785C7 / 0x1402C3A58):
  the sampler manager `idTextureSamplerManagerLocal` (static 0x1422FDA20) keeps an `idList` at +8
  (array +8, count low 24 bits +16) of 56-byte records `{+0 name, +8 samplerOptions_t (bits +8,
  MipLODBias +28), +48 cell}`; the *cell* is a separate 8-byte heap slot holding the
  `ID3D11SamplerState*` (allocated by `sub_1403778A0`, filled by `CreateSamplerState`).
  `CreateSampler` returns the cell (0x1403782F4 lookup hit, 0x1403785C7 new); parameter blocks
  store it (`idParmBlock::Load` `sub_140343090`, find-by-name slot 3); `Ctx_BindProgram` binds
  `*cell` at every bind (0x1402C3A53/0x1402C3A58). Records are only appended (vtable has no
  erase; destructor at exit), so cells live for the whole run. Implementation
  (`fsr_scale::samplers`): the `CreateSampler` hook passes the options through unchanged (the
  engine's lookup compares them) and registers the returned cell with its options bits and
  bias; at run-time install the manager list is walked once (each cell checked to point at an
  object whose vtable is in d3d11.dll). Each frame the wanted bias (upscale and FSR running:
  `fsr_mip_bias` or `log2(render/display) - 1`; else 0) is applied once it has been stable for 8
  frames: every qualifying sampler (linear/anisotropic mip, U wrap, no comparison, no
  `forceNoMipLodBias`) is re-created from its current `GetDesc` with `MipLODBias = options bias
  + bias` and swapped into the cell by an 8-byte compare-exchange; the cell owns one reference
  as the engine's object did, swapped-out objects are released 8 frames later (a recording
  thread may have read the old pointer just before the swap; command lists keep references to
  what they bind). F6 off / native mode restore the engine values (bias 0 = re-created with the
  options bias). Samplers outside the manager (Iggy UI, D3D init's default sampler) are not
  touched.
- **Scheduler without command lists** (`r_schedulerUseDeferredCtx 0` / -1 without driver
  command lists): the TAA site sees the immediate context and runs the same `execute_job` inline,
  then zeroes the engine context's cached input layout / VS / PS / CS (0x12B88..0x12BA7); the depth
  resample runs inline from the POST_TONEMAP hook under `SwapDeviceContextState`.
- **Main menu = native AA** (2026-10-18): the user saw the two menu characters twice, offset,
  while the camera followed the mouse, only with upscale (native AA fine). Log of that session
  (`local/inject/void-inject.log`, session 1792291317): map `maps/campaign/menu/menu_p`, one
  view (models 245, fov (0.802 0.351)), `first frame 65 ... (native)` then `jitter sequence
  2304->3440` and once `POST_TONEMAP recorded without a switched POSTPROC for its frame`
  (**O**). Root cause **U** (candidates, not separated without a game run: a menu frame whose
  render scale was written in `SetupScaled` but whose POSTPROC was not taken, so the post chain
  ran at render scale; or history/jitter computed for a different render size than the menu
  view used). Fix: `fsr_scale::update_menu` reads the map name of the first view's render world
  once per world change (`rt::live::map_name`, world +132328); while it starts with
  `maps/campaign/menu/`, `upscale_mode()` is false, so the menu renders at native size with
  FSR native AA and its mip bias is 0. Log `fsr: main-menu scene: native AA while it is shown`,
  toast `FSR native AA in the main menu (WxH)`; upscale resumes when a game map loads.
- **Value check and guard (white-flash diagnostics, 2026-10-18; `fsr_check.rs`):** user saw an
  occasional full-screen white flash with upscale 0.67 + SSR temporal + RT reflections (FSR GPU
  max 74 ms once). Guard: the pack replaces non-finite colour channels with 0 and clamps to
  [0, 65000] before FSR (`Sanitize` in `PACK_HLSL`), the unpack does the same on FSR's output;
  the SSR and RT-reflection composites have their own guards (commit d1cc247 of the
  pre-release history). Check: per FSR frame a compute shader (8x8 groups of 4x4 texels, group-shared
  reduction, raw 64-byte buffer) counts NaN / Inf / > 65000 and the largest finite value at
  four stages: 0 engine colour before SSR (only when SSR runs), 1 colour after SSR (pack input),
  2 FSR's shared colour input after the RT-reflection composite (D3D12), 3 FSR's output before
  the unpack. The result is copied to a 4-deep staging ring and read with
  `D3D11_MAP_FLAG_DO_NOT_WAIT` frames later (never stalls; a full ring skips the frame). Logged:
  `fsr: value check frame N: [stage: NaN a Inf b huge c max m] ...` with the first bad stage
  marked `(first)` (each of the first 50 bad frames, then every 100th), a per-1200-frame
  summary `fsr: value check: ... (by stage [..]), max per stage [..]`, every FSR dispatch over
  20 ms (`fsr: GPU spike X ms in the dispatch of FSR frame N`), every CPU fence wait over 20 ms
  (`fsr: CPU waited X ms for D3D12 fence V`), every queued job replaced before its submission
  (`fsr: frame N job dropped ...`: that frame's `taaResult` was not written). Fence timeouts
  (2 s) already disable FSR with `fsr: frame N failed: GPU fence ... not reached`.
- **Motion-vector coverage and debug views (2026-10-18, `fsr_motion.rs`; E080).** User: in
  FFX's debug view (Shift+F6, upper-left = motion vectors) only things within ~3-5 m move
  (own hands / sword); NPCs walking farther away look still, camera standing still.
  Evidence (**O**, raw engine vectors: d2enhanced's dumps `captures/dump{0..15}_motion_rg16f_*`
  are written by its pack, which copies `motionVectors` unchanged, `shaders/pack.hlsl`
  `o_motion[id] = g_motion[id]`; linear depth `z = near / d`, near 0.01, engine units taken as
  metres **H**): dump 0 (3440x1440, camera almost still: static facades 0.006 px median)
  holds per-object vectors for NPCs at 10.7 m (0.39 px median), 12.7 m (0.43), 35 m (0.66,
  max 2.1), 57 m (0.07, max 0.12) and 97 m (0.20, max 0.33), for the ocean surface out to
  ~90 m (0.3-1.1 px) and for tree foliage at ~24 m (wind, 0.01-0.03 px); hands / sword 1-5 px.
  So the velocity prepass is not distance-limited in that capture, and the "no motion" look is
  the linear scale of FFX's view: a walking NPC at 30-100 m moves 0.07-0.7 px per frame.
  Bounded negative search (IDB strings matching velocity / motion vector / motionblur /
  previous skinning): no render cvar that limits velocity output by distance or LOD (only
  `r_skipCameraMotionBlur`, `PostFX_uberMotionVector`, `_MotionVectors_%p`, gameplay
  velocities). Engine gaps that remain **H**: blended particles / smoke / fog volumes are drawn
  after the velocity prepass (pass table above: only alpha-tested `ro_fx_*` particles are in
  it), so they carry the vectors of what is behind them; the OIT masks (`fsr_masks`) are what
  limits their ghosting. Changes: Shift+F6 cycles the debug view `off -> FFX -> motion
  vectors` (overlay FSR tab, Expert controls: combo with the legend); "motion vectors" draws
  the vectors FSR received (after the sky fill) over the back buffer at present: brightness =
  `log2(1 + px/0.02) / log2(1 + 32/0.02)` of the per-frame length in render pixels (0.1 px ->
  0.24, 1 px -> 0.53), hue = direction, black = 0, legend bar along the bottom with ticks at
  0.1, 1, 10 px (`local/fsr/mv_view_dump0.png`: the same mapping applied to dump 0; distant
  NPCs and foliage visible). Statistic: on static-camera frames (view matrix unchanged), one in
  8, a compute shader counts per depth band (0-5, 5-20, 20-80, > 80 m, sky) the pixels whose
  engine vector exceeds 0.25 render px (DO_NOT_WAIT read-back), logged every 10 s as `fsr:
  engine motion vectors, static camera (N frames sampled): 0-5 m x % of n px > 0.25 px, ...`.
  No camera-motion fill (user: camera motion is fine; dumps 12-15 walking forward: 0.28 px <
  5 m, 0.12 px 5-20 m, 0.03 px 20-80 m, i.e. proportional to 1/distance).
- **"Faint glowy outline" candidates (user report).** Resolved 2026-10-18: the rim is
  vanilla (also without our DLL, every preset, AA off) and comes from the game's DOF-combine
  sharpening (docs/lighting-options.md 5b, E082; fix `dof_sharpen`). The FSR-path list below
  stays as the A/B list for FSR-specific edge artefacts. In order of likelihood for a
  *bright* rim, each with a live toggle:
  1. RCAS sharpening (`fsr_sharpness`, default 0.15, native and upscale): sharpening
     overshoots at high-contrast edges into a bright rim on the bright side. A/B
     `fsr_sharpness = 0`. **H**
  2. Upscale only: post-TAA effects read `fx/depthFull` that our resample fills by
     nearest-neighbour from render resolution (`ps_depth`, `Load` of the scaled texel), so depth
     silhouettes sit up to one render texel (1.5 display px at 0.67) off FSR's reconstructed
     colour edge; fog / haze / depth-tested post draws then apply the background's (brighter)
     fog to the object's edge pixels. A/B `fsr_mode = native`, then `fsr_depth_resample = 0`.
     **H**
  3. Our reflections: SSR / RT composite at depth discontinuities (reflections of the
     background bleeding onto the silhouette, the 3x3 reuse of RT reflections). A/B `ssr = 0`,
     `rt_reflections = 0`. **H**
  4. Exposure: FSR gets the engine's exposure (`EngineLuminanceFactor` path) and
     `pre_exposure = 1`; a mismatch with the exposure the colour was stored with shifts FSR's
     history clamping and can leave bright fringes on moving high-contrast edges. A/B
     `fsr_auto_exposure = 1`. **H**
  5. Sky vector fill at sky / geometry borders: only sky pixels (depth 0, engine vector 0)
     change; a wrong convention would show as a sky-coloured fringe around objects against the
     sky while turning (the menu decision had 0.95 inliers). A/B `fsr_sky_motion = 0`. **H**
  6. Reactive / T&C masks (`fsr_reactive` 0.9): less history over particles / glass gives
     aliasing and flicker there rather than a glow. A/B `fsr_masks = 0`. **H**
  Not a cause: missing object vectors at distance (refuted above).

### Offline verification (`cargo run -p void-inject --release --example fsr_offline -- 3440x1440 64 0.67`)

```
fsr: jitter sequence 3440->3440: 8 phases from the provider (formula 8), first Some([0.0, -0.16666666])
fsr: jitter sequence 2304->3440: 17 phases from the provider (formula 17), first Some([0.0, -0.16666666])
fsr: 128 frames (2 resets, 0 camera cuts), 4.1.1 GPU ms p50 1.805 p99 2.545 max 2.605 (n 125)
native: 64 frames in 184.9 ms CPU (inline + deferred-list styles); red mean 0.4961 (input 0.6919), min 0.0493, non-finite 0; deferred list saw the FSR output: true; resets 1
upscale 2304x968 -> 3440x1440 (scale 0.67): 64 frames in 141.7 ms CPU; red mean 0.5830 (render-region input 0.6922), min 0.0493, non-finite 0; bottom-right quarter mean 0.5820 min 0.0493
masks: transparency texels > 0.5 = 77400 (expected 77400); distoVectors stretch: right column 1152 (last source column 1152), middle 576.5; depth resample: 4953600 of 4953600 texels at 0.5; resets total 2
```

Also 2560x1080 at 0.5 (1280x544): same checks pass. Provider phase counts equal the FFX 3.x formula
(8, 17). Verified offline: pack/fence/FSR/unpack at render < display, whole-display unpack (bottom
right filled), masks, the stretch, the depth resample (`SV_Depth` under a swapped context state),
ordering before a deferred command list, the table-entry round trip (unit tests). Not verifiable
offline: every engine-memory read/write, the hooks, the sampler options layout, the switch itself.

Live settings (2026-10-17, `fsr_offline -- 3440x1440 32 0.67`): the same runtime runs native then
upscale (render-size change -> history reset, no context re-creation); the live mip bias on two
stand-in cells: `live mip bias: wrap sampler 0.25 -> -1.25 (new object true) -> 0.25; clamp
sampler 0.25 (untouched); applied Some((-1.5, 1)), restored Some((0.0, 1)): OK`. Unit test
`mem::tests::patch_code_is_safe_while_another_thread_calls_through_the_site`: 101 retargetings
of a `call rel32` while another thread calls through it in a loop; the caller only ever sees one
of the two targets. `settings::tests::every_fsr_key_is_live`.

Value check (2026-10-18, same command): 4 native frames with 4 NaN, 4 Inf and 4 65504 texels in
the colour input: `value check: 4 frames read back, last bad frame 199: [engine colour before
reflections: NaN 0 Inf 0 huge 0 max 0.0] [colour after SSR (FSR pack input) (first): NaN 4 Inf
4 huge 4 max 65504.0] [FSR colour input after RT reflections: NaN 0 Inf 0 huge 0 max 64992.0]
[FSR output: NaN 0 Inf 0 huge 0 max 24672.0]; final output non-finite 0: OK` (65000 stored as
FP16 64992; the RT composite is not recorded offline, so stage 2 shows the pack guard only).

Motion diagnostics (2026-10-18, same command): synthetic engine vectors in four depth-band
columns (2, 10, 40, 200 m, top quarter sky; half of each column 0.5 px, a quarter 0.1 px):
`motion statistics: 0-5 m 33.33 % of 928800 px > 0.25 px, 5-20 m 33.33 %, 20-80 m 33.33 %,
>80 m 33.33 %, sky 0.00 % of 1238400 px (expected match); debug view brightness 0.5 px 0.443
(expected 0.442), 0.1 px 0.243 (0.243), zero 0.000, legend middle 0.502: OK`.

### Known gaps

- Sky motion vectors depend on the in-game decision: if the log reports `sky motion vectors
  off`, no convention matched the engine's vectors and the sky stays at 0 as before.
- Stencil is resampled with `SV_StencilRef` (device reports `PSSpecifiedStencilRefSupported`;
  offline on the RX 7900 XTX: yes); without it the stencil stays as rendered.
- `distoVectors` units after the stretch (UV of the render rect vs. scaled texture UV) are **U**;
  wrong units show as distortion strength off by the scale factor.
- If POST_TONEMAP ever records before its frame's POSTPROC switch, it stays at render scale
  (logged once).
- A failed FSR dispatch in deferred mode is detected at submit, after the switch was recorded:
  one frame shows the render-size fallback copy in the top-left, then FSR is disabled.
- Live mip bias assumes the engine never writes a cell after creation (none found: only
  `CreateSamplerState` in `CreateSampler` writes it) and that no thread holds a raw sampler
  pointer longer than the 8-frame grace (binds read the cell each time).
- Turning `fsr` on at run time while a POSTPROC list is being recorded: that frame keeps the
  engine TAA (the call sites are patched between recordings; FSR takes the next frame).
- Main menu: upscale is off there by design (map-name test); the double-image cause itself is
  **U**. Another scene with the same symptom would need its own test (none known).
- White flash: cause **U**. The guards remove NaN / Inf / > 65000 at our outputs; a flash
  caused by finite but very bright values (e.g. an exposure jump) or by a frame whose
  `taaResult` was never written would still show, and is then identified by the log lines
  listed under "Value check and guard".

### In-game checklist

1. `fsr = 1`, `fsr_mode = native`, Anti-aliasing TAA. Log: `fsr: hooks: SetupScaled 0x140374b60,
   PerViewCB_Upload ..., POSTPROC ..., POST_TONEMAP ..., CreateSampler ..., CLEAR ..., ASYNC_CLEAR
   ...; 0 existing samplers tracked`, `fsr: ready, provider 4.1.1`, `fsr: jitter sequence
   3440->3440: 8 phases from the provider`, `fsr: first frame ...: ... masks true (hair ...)`. Look
   for: no shimmer on static edges (jitter correct), Shift+F6 debug view with masks over
   particles/glass, F6 A/B against engine TAA.
2. `fsr_mode = upscale`: toast `FSR upscale 0.67 (2304x968)`, log `fsr: mip bias -1.578 set on N
   of M tracked samplers`, first frame `render
   2304x968, display 3440x1440 (upscale) ... distoVectors true`, jitter `2304->3440: 17 phases`. No
   `upscale switch not possible` / `POST_TONEMAP recorded without a switched POSTPROC` lines after
   the first frames. Look at: full-screen image (no top-left crop, no zoom), bloom/lens flare
   placement, sprint/crouch blur and heat-haze distortion reaching the right/bottom edges, in-world
   UI markers occluded correctly by geometry, texture sharpness vs native, Dark Vision.
3. Camera cuts (load, teleport power): the periodic `fsr: ... camera cuts` counter rises.
4. `+set r_schedulerUseDeferredCtx 0`: same as 1-2 (inline path), no `FSR disabled` line.
5. Live switching (overlay F2, FSR tab; every row tagged `live`): `fsr_mode` native <-> upscale
   and `fsr_scale` moves: toast `FSR upscale 0.67 (2304x968)`, log `fsr: jitter sequence ...`,
   no crop/zoom glitch beyond one frame, `resets` count rises by one per change. Mip bias:
   toast `FSR mip bias -1.58 on N samplers` (log `fsr: mip bias ... set on N of M tracked
   samplers`, M ~ the number of decl samplers), texture sharpness changes without reload; back
   to native: `FSR mip bias off`. `fsr_auto_exposure`: toast, log `re-creating the FSR
   context`, then `fsr: ready`. Start with `fsr = 0`, enable it in the overlay: toast `FSR 4
   installed and on`, log `fsr: hooks: ...; N existing samplers tracked`, `call sites
   redirected`, no hitch beyond a frame; disable: `FSR off (engine TAA; hooks stay installed,
   inert)`.
6. `fsr_masks = 0` A/B on particles, glass, hair.
7. Main menu with `fsr_mode = upscale`: log `fsr: main-menu scene: native AA while it is shown`
   (first frame `render 3440x1440, display 3440x1440 (native)`, no `jitter sequence 2304->3440`
   while in the menu; toast `FSR native AA in the main menu (3440x1440)` only when returning to
   the menu from a map); moving the mouse shows each character once. Loading a map: `fsr: left
   the main-menu scene`, toast `FSR upscale 0.67 (2304x968)`.
8. White flash capture (upscale 0.67, `ssr = 1` temporal, RT reflections on): play until a flash,
   note the wall-clock time, then quit and keep `local/inject/void-inject.log`. Lines near that
   time: `fsr: value check frame N` (which stage is `(first)`), `fsr: GPU spike`, `fsr: CPU
   waited`, `fsr: frame N job dropped`, `fsr: frame N failed`, and the 1200-frame
   `fsr: value check: ...` summary. No such line at the flash = finite values and no skipped
   frame: then repeat with RT reflections off, then SSR off, to bisect.
9. Motion vectors: Shift+F6 twice (or the overlay combo) = "motion vectors". Standing still,
   walking NPCs at 20-100 m must show as dim coloured figures (brightness ~0.2-0.5), static
   geometry black, hands bright. Log every 10 s while standing still: `fsr: engine motion
   vectors, static camera ...`; report the percentages per band with NPCs in view. Glowy
   outline: A/B in this order and note which one removes it: F6, `fsr_sharpness = 0`,
   `fsr_mode = native`, `fsr_depth_resample = 0`, `ssr = 0`, `rt_reflections = 0`,
   `fsr_auto_exposure = 1`, `fsr_sky_motion = 0`, `fsr_masks = 0`; also whether it shows
   standing still or only while moving.
