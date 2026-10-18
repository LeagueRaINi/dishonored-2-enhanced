//! HLSL -> DXIL through the DirectX Shader Compiler, loaded at run time from an explicit
//! directory (`dxcompiler.dll` + `dxil.dll`; the Windows SDK ships both). Shader model 6.5
//! is required for `RayQuery`.

use std::ffi::c_void;
use std::path::{Path, PathBuf};

use windows::Win32::Foundation::HMODULE;
use windows::Win32::Graphics::Direct3D::Dxc::{
    CLSID_DxcCompiler, DXC_CP_UTF8, DXC_OUT_ERRORS, DXC_OUT_OBJECT, DxcBuffer,
    DxcCreateInstanceProc, IDxcBlob, IDxcBlobUtf8, IDxcCompiler3, IDxcIncludeHandler, IDxcResult,
};
use windows::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};
use windows::core::{HSTRING, Interface, PCWSTR, s};

pub struct Dxc {
    compiler: IDxcCompiler3,
    pub dir: PathBuf,
    /// Keeps the DLLs loaded for the compiler's lifetime.
    _modules: [HMODULE; 2],
}

/// Newest `Windows Kits/10/bin/<version>/x64` holding both DLLs (`None` if absent).
pub fn find_windows_sdk() -> Option<PathBuf> {
    let root = std::env::var_os("ProgramFiles(x86)")
        .map(PathBuf::from)?
        .join("Windows Kits")
        .join("10")
        .join("bin");
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(root)
        .ok()?
        .filter_map(|e| Some(e.ok()?.path().join("x64")))
        .filter(|d| d.join("dxcompiler.dll").is_file() && d.join("dxil.dll").is_file())
        .collect();
    dirs.sort();
    dirs.pop()
}

impl Dxc {
    pub fn load(dir: &Path) -> Result<Self, String> {
        let load = |name: &str| -> Result<HMODULE, String> {
            let p = dir.join(name);
            // SAFETY: loading a DLL by full path; its initialisation is the vendor's.
            unsafe { LoadLibraryW(&HSTRING::from(p.as_os_str())) }
                .map_err(|e| format!("{}: {e}", p.display()))
        };
        // dxil.dll first: dxcompiler.dll later loads it by module name for signing and finds
        // the already loaded module.
        let dxil = load("dxil.dll")?;
        let dxcompiler = load("dxcompiler.dll")?;
        // SAFETY: symbol lookup in a module we hold.
        let proc = unsafe { GetProcAddress(dxcompiler, s!("DxcCreateInstance")) }
            .ok_or("dxcompiler.dll: no DxcCreateInstance")?;
        // SAFETY: DxcCreateInstance has the documented `DxcCreateInstanceProc` signature.
        let create: DxcCreateInstanceProc = unsafe { std::mem::transmute(proc) };
        let create = create.ok_or("DxcCreateInstance is null")?;
        let mut ptr: *mut c_void = std::ptr::null_mut();
        // SAFETY: valid CLSID/IID and out pointer; on success `ptr` owns one reference.
        unsafe { create(&CLSID_DxcCompiler, &IDxcCompiler3::IID, &mut ptr) }
            .ok()
            .map_err(|e| format!("DxcCreateInstance: {e}"))?;
        // SAFETY: `ptr` is a live IDxcCompiler3 whose reference we take over.
        let compiler = unsafe { IDxcCompiler3::from_raw(ptr) };
        Ok(Self {
            compiler,
            dir: dir.to_owned(),
            _modules: [dxil, dxcompiler],
        })
    }

    /// Compiles `entry` of `source` for `profile`; returns the DXIL container.
    pub fn compile(&self, source: &str, entry: &str, profile: &str) -> Result<Vec<u8>, String> {
        let args: Vec<HSTRING> = ["-E", entry, "-T", profile, "-HV", "2021", "-O3"]
            .iter()
            .map(|a| HSTRING::from(*a))
            .collect();
        let argv: Vec<PCWSTR> = args.iter().map(|a| PCWSTR(a.as_ptr())).collect();
        let buf = DxcBuffer {
            Ptr: source.as_ptr().cast(),
            Size: source.len(),
            Encoding: DXC_CP_UTF8.0,
        };
        // SAFETY: `buf` and `argv` outlive the call.
        let result: IDxcResult = unsafe {
            self.compiler
                .Compile(&buf, Some(&argv), None::<&IDxcIncludeHandler>)
        }
        .map_err(|e| format!("{entry}: Compile: {e}"))?;
        let mut errors: Option<IDxcBlobUtf8> = None;
        // SAFETY: querying an output of a live result.
        let _ = unsafe { result.GetOutput(DXC_OUT_ERRORS, std::ptr::null_mut(), &mut errors) };
        let messages = errors
            .map(|e| {
                // SAFETY: the blob owns `GetStringLength` bytes at `GetStringPointer`.
                unsafe {
                    String::from_utf8_lossy(std::slice::from_raw_parts(
                        e.GetStringPointer().0,
                        e.GetStringLength(),
                    ))
                    .into_owned()
                }
            })
            .unwrap_or_default();
        // SAFETY: plain status query.
        let status = unsafe { result.GetStatus() }.map_err(|e| e.to_string())?;
        if status.is_err() {
            return Err(format!("{entry}: compilation failed:\n{messages}"));
        }
        let mut object: Option<IDxcBlob> = None;
        // SAFETY: as above.
        unsafe { result.GetOutput(DXC_OUT_OBJECT, std::ptr::null_mut(), &mut object) }
            .map_err(|e| format!("{entry}: no object: {e}"))?;
        let object = object.ok_or_else(|| format!("{entry}: no object"))?;
        // SAFETY: the blob owns `GetBufferSize` bytes at `GetBufferPointer`.
        Ok(unsafe {
            std::slice::from_raw_parts(
                object.GetBufferPointer().cast::<u8>(),
                object.GetBufferSize(),
            )
        }
        .to_vec())
    }
}
