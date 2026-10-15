@echo off
rem Builds the patched DXVK (third_party\dxvk, release 3.1.1 + patches\dxvk-3.1.1-d3d12-interop.patch) with MSVC and
rem copies d3d11.dll/dxgi.dll to d2enhanced\dxvk\. Needs meson + ninja (.venv) and glslang (third_party\glslang_dl).
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
set ROOT=%~dp0
set PATH=%ROOT%.venv\Scripts;%ROOT%third_party\glslang_dl\x\bin;%PATH%
cd /d "%ROOT%third_party\dxvk"

rem The interop patch, once (a reverse check that succeeds means it is applied already).
git apply --reverse --check --ignore-whitespace "%ROOT%patches\dxvk-3.1.1-d3d12-interop.patch" >nul 2>&1 || (
    git apply --ignore-whitespace "%ROOT%patches\dxvk-3.1.1-d3d12-interop.patch" || exit /b 1
    echo Applied the D3D12 interop patch.
)

if not exist build-msvc (
    meson setup build-msvc --buildtype=release --backend=ninja -Db_ndebug=true || exit /b 1
)
ninja -C build-msvc src/d3d11/d3d11.dll src/dxgi/dxgi.dll || exit /b 1
if not exist "%ROOT%dxvk" mkdir "%ROOT%dxvk"
copy /y build-msvc\src\d3d11\d3d11.dll "%ROOT%dxvk\d3d11.dll" >nul || exit /b 1
copy /y build-msvc\src\dxgi\dxgi.dll "%ROOT%dxvk\dxgi.dll" >nul || exit /b 1
echo Built DXVK into %ROOT%dxvk
