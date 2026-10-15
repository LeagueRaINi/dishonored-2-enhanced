@echo off
rem Switch the game to DXVK (D3D11 on Vulkan). The mod stays active and chain-loads DXVK.
rem Undo with use_native.bat.
cd /d "%~dp0.."
tasklist /fi "imagename eq Dishonored2.exe" | find /i "Dishonored2.exe" >nul && (echo Close the game first. & pause & exit /b 1)
copy /y d2enhanced\build\d3d11.dll d3d11.dll >nul
copy /y d2enhanced\dxvk\dxgi.dll dxgi.dll >nul
copy /y d2enhanced\dxvk\dxvk.conf dxvk.conf >nul
echo Game now uses DXVK + mod.
