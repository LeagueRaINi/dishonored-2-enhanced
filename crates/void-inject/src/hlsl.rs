//! Runtime HLSL compilation (`D3DCompile`) for the module's own shaders.

use std::ffi::c_void;

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::core::{PCSTR, s};

fn blob_bytes(b: &ID3DBlob) -> &[u8] {
    // SAFETY: the blob owns `GetBufferSize()` bytes at `GetBufferPointer()` for its lifetime.
    unsafe { std::slice::from_raw_parts(b.GetBufferPointer().cast::<u8>(), b.GetBufferSize()) }
}

/// Compiles the HLSL source `src` (entry point `entry`, profile `target`) to bytecode;
/// the error carries the compiler messages.
pub fn compile(src: &str, entry: PCSTR, target: PCSTR) -> Result<Vec<u8>, String> {
    let mut code: Option<ID3DBlob> = None;
    let mut errs: Option<ID3DBlob> = None;
    // SAFETY: source pointer/length describe `src`; out pointers are valid.
    let r = unsafe {
        D3DCompile(
            src.as_ptr().cast::<c_void>(),
            src.len(),
            s!("void_inject"),
            None,
            None,
            entry,
            target,
            0,
            0,
            &mut code,
            Some(&mut errs),
        )
    };
    if let Err(e) = r {
        return Err(format!(
            "D3DCompile: {e} {}",
            errs.as_ref()
                .map(|b| String::from_utf8_lossy(blob_bytes(b)).into_owned())
                .unwrap_or_default()
        ));
    }
    code.as_ref()
        .map(|b| blob_bytes(b).to_vec())
        .ok_or_else(|| "no shader".into())
}
