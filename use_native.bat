@echo off
rem Switch the game back to native D3D11 (removes DXVK). The mod stays active.
cd /d "%~dp0.."
tasklist /fi "imagename eq Dishonored2.exe" | find /i "Dishonored2.exe" >nul && (echo Close the game first. & pause & exit /b 1)
if exist dxgi.dll del dxgi.dll
if exist dxvk.conf del dxvk.conf
copy /y d2enhanced\build\d3d11.dll d3d11.dll >nul
echo Game now uses native D3D11 + mod.
