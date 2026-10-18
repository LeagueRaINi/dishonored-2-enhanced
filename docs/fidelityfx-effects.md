# AMD FidelityFX SDK 2.3 and further effects for G2 (plan, 2026-10-17)

User decision (2026-10-17): stay on FidelityFX SDK 2.3 (no older SDK). Effects the SDK does not
provide are our own implementations.

## 1. What SDK 2.3 provides on this machine (measured)

Throwaway probe `local/tools/ffxprobe` (signed DLLs from
`d2enhanced/third_party/FidelityFX-SDK/Kits/FidelityFX/signedbin`, `ffxQuery` version list
and `ffxCreateContext` per effect; RX 7900 XTX, driver 32.0.32015.2008, highest shader model
6.8):

| Effect | Versions offered | Context creation | Use for us |
| --- | --- | --- | --- |
| Upscaler | 4.1.1, 3.1.5, 2.3.4 | (in use) | G1, done |
| Frame generation | 3.1.6 | not tried | frame-rate gain while CPU-bound (§3 P1) |
| Denoiser (FSR Ray Regeneration) | none | fails, `NO_PROVIDER` for AO, dominant-light and indirect signals | not available on RDNA3 (documented: RX 9000 or later) |
| Radiance cache | 0.9.0 | OK with the default backend; forcing WMMA fails (`ERROR_PARAMETER`) | only with a path tracer (§3 Q7) |

d2enhanced used only the upscaler from the SDK (`src/fsr.cpp`). Its SSR/SSGI are custom
shaders (`re/SSGI_SSR_REVIEW_2026-10-15.md`).

## 2. Integration rules

- **Cross-API hand-offs cost CPU.** Every D3D11 -> D3D12 hand-off costs CPU on the critical
  path: FSR's copy + `Signal` is 0.2-0.5 ms p50 (E058, E060). Effects composited at the FSR
  point (S2) share that hand-off. Mid-frame effects (S1, before `FORWARD_WORLD`) are written as
  D3D11 compute and recorded into the engine pass's own deferred context: no sync.
- **Replacing an engine pass removes its cost.** A pass with flag `0x100` (pass+244) is skipped
  by `RenderPass_Record` but stays in the dependency graph (E061). A replacement sets the
  flag through a patch-registry toggle. Passes we only add to stay on: CSM shadows, probes,
  IBL.
- Every effect: setting + overlay toggle, A/B benchmark (GPU ms per effect from the profiler),
  debug view, and no change when off.

## 3. Candidate effects

Value: visual or frame-time gain on this machine. Cost/effort: S days, M 1-3 weeks, L more.

### Performance

| Id | Effect | Mechanism | Value | Effort | Risk / blocker |
| --- | --- | --- | --- | --- | --- |
| P1 | FSR 3.1.6 frame generation | FFX frame-interpolation swap chain on our D3D12 queue; the game's final image (and HUD-less colour before UI) + depth + motion vectors from the FSR splice; our overlay drawn after | ~1.6-1.9x presented fps when CPU-bound [H] | L | Presentation moves from the game's D3D11 swap chain to D3D12; latency (needs P2); UI/HUD-less separation |
| P2 | Latency reduction for P1 | AMD Anti-Lag 2 SDK [unverified: D3D11 support and RDNA3 driver support to be checked] or our own frame limiter before input sampling | needed for P1 to feel right | S-M | separate AMD SDK, not part of FFX 2.3 |
| P3 | Dynamic render scale | FSR scale (live since fbb2b9d) driven by GPU time to a frame-time target | holds fps in GPU-heavy spots | S | deferred (user, 2026-10-17: not needed now) |
| P4 | Submission order | `patch.shadows_submit_early` (on by default, -0.47 ms, E062); no further edge with a measurable gain (E062) | done | — | — |
| P5 | Hitch analysis | per-frame spikes (p99/p99.9) against streaming/shader-creation events in the profiler | smoother play | S-M | cause unknown until measured |

### Image quality

| Id | Effect | Mechanism | Value | Effort | Risk / blocker |
| --- | --- | --- | --- | --- | --- |
| Q1 | Ambient occlusion replacing HBAO+ (B1) | **done 2026-10-18 (E063/E064), default:** `ao.rs` detours HBAO+'s `RenderAO` and records a visibility-bitmask AO (Therrien 2023, XeGTAO structure) into the same deferred context and target; HBAO+ does not run (`ao_mode = void`) | stronger, stable contact AO without halos; in game 0.08 ms vs HBAO+ ~0.22 ms (high, half res, scale 0.67) | M | no temporal filter needed (fixed 4x4 pattern + blur) |
| Q2 | Contact shadows (B2) | **done 2026-10-18 (E066), default on after the user's in-game check:** `contact.rs` hooks the SCREENSHADOWS callback and writes `.y = min(CSM, contact)` after the engine's blur | grounded characters/props, fine detail the 2048² cascades miss; also masks the cascade flicker the user saw; 0.32 ms native / 0.15 ms at 0.67 (high, offline) | S-M | thin-geometry halos (thickness heuristic); local lights need shader access |
| Q3 | Shadow settings beyond Ultra (A2-A6) | CSM 4096, longer cascade fars, 2048 local maps, no partial slice updates | sharper, longer shadows | S | GPU/CPU cost; atlas eviction risk (A5) |
| Q4 | Soft shadows (PCSS-style) | blocker search + variable filter on the CSM in the screen-shadow pass (sun) | penumbrae that widen with distance | M | needs the cascade layout and sampling code of `SCREENSHADOWS` (RE) |
| Q5 | Screen-space reflections (B3) | **implemented 2026-10-18 (E067, E071, E072; offline-verified, in-game test pending):** `ssr.rs` at the TAA site (engine TAA: POSTPROC deferred list; FSR: immediate context before FSR's pack), Hi-Z march, fixed-pattern GGX + spatial resolve, premultiplied confidence, Fresnel blend over the engine colour (misses keep the engine's own IBL); per-pixel normal, roughness and F0 from our DXBC patch of the 3,037 lit forward shaders (`ssr_material = auto`, reflections.md 2b), depth normals + a user surface model where none | wet streets, floors, metal; half res medium 0.31 ms native / 0.14 ms at 0.67 (offline, morning machine state) | done | blended lit draws (decals) and load-time patch cost to check in game |
| Q6 | Screen-space GI bounce (B4) | own visibility-bitmask SSGI at S2 | colour bleeding, darker crevices | M-L | same data blocker; off-screen light missing |
| Q7 | Ray-traced GI / shadows / reflections (option C) | DXR on the decoded static scene; radiance cache 0.9.0 (available here) only as the secondary-ray terminator of a path tracer it is trained by (reflections.md section 3); RT sun shadows | largest lighting upgrade | L (reflections with engine-matching hit shading L+) | no DXR code exists yet (checkpoint 9282529 searched); skinned characters undecoded; entity sync must be incremental (E044: 47 ms full walk); no FFX denoiser on RDNA3; first spike: offline BLAS/TLAS of eotw + 1 ray/pixel timings (E069) |
| Q8 | Volumetric lighting | froxel volume lit by the sun CSM and local lights, composited before `FOG` / at S2 | Karnaca's dusty light shafts | M-L | interaction with the engine fog pass |
| Q9 | HDR output | scRGB/HDR10 swap chain, own tone mapper instead of `POST_TONEMAP`, UI composited at paper white | large on HDR displays | M | UI path, game's colour grading must be re-applied |
| Q10 | Specular/geometric anti-aliasing, debanding | filtered roughness in the forward shaders; dither before quantisation in post | less shimmer/banding | S-M | shader access (DXBC or replacement shaders) |
| Q11 | Texture and material mods (G3) | already implemented, in-game test pending | community upgrades | — | — |

## 4. Order

1. P4 done (E062); P3 deferred.
2. Q1 AO done offline (in-game A/B pending), then Q2 contact shadows: same S1 splice, proves the D3D11
   in-pass route.
3. Q3 settings A/B (cheap) and Q4 if the screen-shadow pass allows it.
4. P1 frame generation feasibility spike (presentation path), with P2.
5. Material data source done (DXBC material target, E072), Q5 done offline; Q6 next (needs albedo:
   one more capture, `ro_diffusemap` as d2enhanced, or the radiance itself).
6. Q8, Q9, Q7 as separate projects.
