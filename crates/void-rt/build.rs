//! Precompiles every compute kernel of `src/photo.hlsl` and `src/rt.hlsl` to DXIL with the
//! DirectX Shader Compiler (`dxc.exe`), so the in-process path (`ingame`, the game's DLL) needs
//! no `dxcompiler.dll` at run time. Kernels are found as the function following each
//! `[numthreads(` line; flags match `dxc::Dxc::compile` (`-T cs_6_5 -HV 2021 -O3`).
//!
//! DXC location: env `VOID_DXC_DIR` (a directory with `dxc.exe` and `dxil.dll`), else the
//! newest Windows SDK `bin/<version>/x64` holding both. `dxil.dll` must sit next to `dxc.exe`:
//! without it the output is unsigned and `CreateComputePipelineState` rejects it.
//!
//! Output: `$OUT_DIR/<entry>.dxil` and `$OUT_DIR/dxil.rs` (`DXIL: &[(&str, &[u8])]`, included
//! by `src/ingame.rs`). Reuse for other crates: copy this file, change `SOURCES`.

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

const SOURCES: &[&str] = &["src/photo.hlsl", "src/rt.hlsl", "src/dynamic.hlsl"];

fn find_dxc() -> Result<PathBuf, String> {
    let ok = |d: &Path| d.join("dxc.exe").is_file() && d.join("dxil.dll").is_file();
    if let Some(d) = std::env::var_os("VOID_DXC_DIR").map(PathBuf::from) {
        return if ok(&d) {
            Ok(d)
        } else {
            Err(format!(
                "VOID_DXC_DIR={} does not contain dxc.exe and dxil.dll",
                d.display()
            ))
        };
    }
    let root = std::env::var_os("ProgramFiles(x86)")
        .map(PathBuf::from)
        .ok_or("ProgramFiles(x86) not set")?
        .join("Windows Kits")
        .join("10")
        .join("bin");
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(&root)
        .map_err(|e| format!("{}: {e}", root.display()))?
        .filter_map(|e| Some(e.ok()?.path().join("x64")))
        .filter(|d| ok(d))
        .collect();
    dirs.sort();
    dirs.pop().ok_or_else(|| {
        format!(
            "no dxc.exe + dxil.dll under {}\\<version>\\x64",
            root.display()
        )
    })
}

/// Entry points: the function name on the line after each `[numthreads(`.
fn entries(src: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut lines = src.lines();
    while let Some(l) = lines.next() {
        if !l.trim_start().starts_with("[numthreads(") {
            continue;
        }
        if let Some(name) = lines
            .next()
            .and_then(|n| n.trim_start().strip_prefix("void "))
            .and_then(|n| n.split('(').next())
        {
            out.push(name.trim().to_owned());
        }
    }
    out
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=VOID_DXC_DIR");
    for s in SOURCES {
        println!("cargo:rerun-if-changed={s}");
    }
    let dxc = match find_dxc() {
        Ok(d) => d,
        Err(e) => panic!(
            "void-rt build: DXC not found ({e}). Install the Windows SDK (it ships \
             bin/<version>/x64/dxc.exe + dxil.dll) or set VOID_DXC_DIR to a directory holding \
             dxc.exe and dxil.dll (DirectXShaderCompiler release)."
        ),
    };
    let out = PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR"));
    let manifest = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").expect("manifest dir"));
    let mut jobs: Vec<(String, PathBuf, Child)> = Vec::new();
    for s in SOURCES {
        let path = manifest.join(s);
        let src = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("void-rt build: {}: {e}", path.display()));
        for e in entries(&src) {
            if jobs.iter().any(|j| j.0 == e) {
                panic!("void-rt build: kernel {e} defined twice");
            }
            let obj = out.join(format!("{e}.dxil"));
            let child = Command::new(dxc.join("dxc.exe"))
                .args([
                    "-nologo", "-E", &e, "-T", "cs_6_5", "-HV", "2021", "-O3", "-Fo",
                ])
                .arg(&obj)
                .arg(&path)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap_or_else(|err| panic!("void-rt build: {}: {err}", dxc.display()));
            jobs.push((e, obj, child));
        }
    }
    let mut table = String::from(
        "/// Precompiled DXIL per kernel entry (build.rs, cs_6_5).\npub static DXIL: &[(&str, &[u8])] = &[\n",
    );
    for (e, obj, child) in jobs {
        let o = child
            .wait_with_output()
            .unwrap_or_else(|err| panic!("void-rt build: dxc {e}: {err}"));
        if !o.status.success() || !obj.is_file() {
            panic!(
                "void-rt build: dxc failed for {e}:\n{}{}",
                String::from_utf8_lossy(&o.stdout),
                String::from_utf8_lossy(&o.stderr)
            );
        }
        table.push_str(&format!(
            "    ({e:?}, include_bytes!(concat!(env!(\"OUT_DIR\"), \"/{e}.dxil\"))),\n"
        ));
    }
    table.push_str("];\n");
    std::fs::write(out.join("dxil.rs"), table).expect("write dxil.rs");
}
