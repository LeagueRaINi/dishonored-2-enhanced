@echo off
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
cd /d "%~dp0"
if not exist build mkdir build

rem Helper and overlay shaders are compiled here, never at runtime.
python -I tools\compile_shaders.py "%~dp0." || exit /b 1

rem Third-party code (MinHook, Dear ImGui) is compiled once per build with its own warning level.
set MH=third_party\minhook\src
set IG=third_party\imgui
cl /nologo /O2 /MT /W3 /c %MH%\buffer.c %MH%\hook.c %MH%\trampoline.c %MH%\hde\hde64.c /Fobuild\ || exit /b 1
cl /nologo /std:c++17 /O2 /MT /EHsc /W3 /DUNICODE /D_UNICODE /I%IG% /c ^
   %IG%\imgui.cpp %IG%\imgui_draw.cpp %IG%\imgui_tables.cpp %IG%\imgui_widgets.cpp %IG%\backends\imgui_impl_win32.cpp ^
   /Fobuild\ || exit /b 1

rem The mod itself (latest C++ standard: C++23 and the C++26 parts MSVC implements).
cl /nologo /std:c++latest /permissive- /O2 /MT /EHsc /W4 /wd4100 /DUNICODE /D_UNICODE /D_WINDOWS ^
   /Ithird_party\minhook\include /I%IG% /I%IG%\backends ^
   src\core.cpp src\config.cpp src\proxy.cpp src\game.cpp src\cbuffers.cpp src\shader_info.cpp src\capture.cpp ^
   src\console.cpp src\hooks.cpp src\fsr.cpp src\hdr.cpp src\rescale.cpp src\ssgi.cpp src\material.cpp src\shader_patch.cpp src\overlay.cpp src\ui.cpp src\counter.cpp ^
   src\imgui_impl_dx11_precompiled.cpp ^
   build\buffer.obj build\hook.obj build\trampoline.obj build\hde64.obj ^
   build\imgui.obj build\imgui_draw.obj build\imgui_tables.obj build\imgui_widgets.obj build\imgui_impl_win32.obj ^
   /Fobuild\ /Febuild\d3d11.dll ^
   /link /DLL /DEF:src\d3d11.def d3d11.lib d3d12.lib dxgi.lib user32.lib shell32.lib dinput8.lib dxguid.lib || exit /b 1
echo Built build\d3d11.dll
