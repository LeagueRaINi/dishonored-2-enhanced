# Development guidelines

These rules apply to people and coding agents working on this repository.

## Direction

- Goal: better performance and image quality in the installed game, with a manageable
  scope. In order of preference: engine settings, targeted fixes, then replacing
  individual render passes. Replacing the whole renderer and removing D3D11 are not goals.
- Native D3D11 only. Do not depend on DXVK. The AMD D3D11 driver has no driver command
  lists (docs/STATUS.md E052), so address CPU cost through engine options or our own code.
- Rust. Prefer proven crates over hand-rolled code (e.g. `iced-x86` for assembling stubs,
  egui for UI). Add a narrow C++ bridge only if a specific integration needs one.

## Pass replacements

- Hook where the engine identifies the pass (call sites, view/job structures from the
  reflection tables), not by guessing from D3D11 state.
- Every replacement has a settings key that turns it off. With the feature off, no hook or
  patch is installed. Each one also has an A/B comparison against the original pass and
  profiler coverage.
- Code patches verify the exact original bytes before writing. Unsafe code stays narrow,
  documented and behind checked interfaces.
- Unknown resource types and engine states must produce useful diagnostics. Never
  substitute placeholders silently.
- Keep original-behaviour compatibility separate from intentional enhancements.

## Performance work

- Measure before changing anything (docs/performance-baseline.md). Record the build,
  GPU/driver, resolution, render scale, graphics settings, frame cap, VSync and active mods.
- Use a stationary camera plus a reproducible camera route, warm up first, and report
  frame-time distributions over repeated runs. Averages alone are not enough.
- Measure CPU and GPU separately. GPU timers are asynchronous: read them frames later,
  without flushes, and handle disjoint results. CPU time spent inside a deferred-context
  call is not GPU time.
- Separate CPU limits, GPU limits, frame caps and synchronisation stalls. Never claim an
  optimisation from one scene or one run.

## Evidence

- Label findings as observed (O), inferred (H) or unknown (U). Record provenance for each:
  executable hash, function or asset location, the tool used and a reproduction method.
  Keep the ledger in docs/STATUS.md and move larger topics into focused documents.
- Never call a subsystem complete because the generated code compiles.
- Recover contracts and implement bounded systems. Do not translate decompiled code
  wholesale.

## The game installation

- `Dishonored2.exe`, `base/`, game configuration and saves are reference inputs. Never
  modify them. Record and revert any temporary substitution.
- Generated files, extraction output, captures and analysis databases go under the
  ignored `local/`. Build output goes to `target/`.
- Never commit game assets, extracted scripts or shaders, executables (patched or not),
  saves or decompiler databases.
- Reusable tools take the game path as an argument. Never hard-code a machine's path.
- Launch the game only for experiments with a stated target, purpose, capture method and
  stopping condition.

## Checks

For Rust changes run:

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets --release -- -D warnings
cargo test --workspace --release
```

Also run the offline harness of the area you changed (`crates/void-inject/examples`,
`crates/void-rt/examples`). Use synthetic fixtures for parser edge cases and local game
files for integration checks. Report the commands that ran and their limits. Never claim
that a check you did not run passed.
