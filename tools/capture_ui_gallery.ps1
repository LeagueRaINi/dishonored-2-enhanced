param(
    [switch]$Build,
    [ValidateRange(640, 7680)][int]$Width = 1600,
    [ValidateRange(480, 4320)][int]$Height = 1000,
    [ValidateRange(0.75, 2.0)][double]$Scale = 1.25
)
$ErrorActionPreference = 'Stop'
$modRoot = Split-Path -Parent $PSScriptRoot
$previewExe = Join-Path $modRoot 'build\ui_preview\ui_preview.exe'
if ($Build -or !(Test-Path -LiteralPath $previewExe)) {
    & (Join-Path $modRoot 'build_ui_preview.bat')
    if ($LASTEXITCODE -ne 0) { throw 'Preview build failed' }
}
$gallery = Join-Path $modRoot 'build\ui_preview\gallery'
$scaleArg = $Scale.ToString([System.Globalization.CultureInfo]::InvariantCulture)
$pages = @('upscaling', 'lighting', 'display', 'colour', 'performance', 'console', 'log', 'diagnostics')
for ($page = 0; $page -lt $pages.Count; ++$page) {
    & $previewExe --hidden --capture (Join-Path $gallery ($pages[$page] + '.png')) --page $page --width $Width --height $Height --scale $scaleArg
    if ($LASTEXITCODE -ne 0) { throw "Capture failed: $($pages[$page])" }
}
for ($mode = 0; $mode -lt 6; ++$mode) {
    & $previewExe --hidden --capture (Join-Path $gallery "counter-$mode.png") --page 4 --mode $mode --scenario 1 --width $Width --height $Height --scale $scaleArg
    if ($LASTEXITCODE -ne 0) { throw "Counter capture failed: $mode" }
}
Write-Output "Gallery: $gallery"
