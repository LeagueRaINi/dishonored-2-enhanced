@echo off
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
cd /d "%~dp0"
if not exist build\ui_preview mkdir build\ui_preview
rem The precompiled shaders the UI shares with the mod (src\shaders_generated.h).
python -I tools\compile_shaders.py "%~dp0." || exit /b 1
set IG=third_party\imgui
rem Isolated objects/exe; never links game hooks or loads the game's proxy DLLs.
cl /nologo /std:c++latest /permissive- /O2 /MT /EHsc /W3 /DUNICODE /D_UNICODE ^
   /Isrc /I%IG% /I%IG%\backends ^
   tools\ui_preview.cpp src\ui.cpp src\counter.cpp src\config.cpp src\core.cpp ^
   src\imgui_impl_dx11_precompiled.cpp ^
   %IG%\imgui.cpp %IG%\imgui_draw.cpp %IG%\imgui_tables.cpp %IG%\imgui_widgets.cpp %IG%\backends\imgui_impl_win32.cpp ^
   /Fobuild\ui_preview\ /Febuild\ui_preview\ui_preview.exe ^
   /link d3d11.lib dxgi.lib user32.lib gdi32.lib shell32.lib ole32.lib windowscodecs.lib dwmapi.lib || exit /b 1
echo Built build\ui_preview\ui_preview.exe
