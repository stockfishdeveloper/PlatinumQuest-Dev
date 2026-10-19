param([string]$OutputDirectory = (Join-Path $env:TEMP 'pq_checks/marble_sim'))
# Builds nav/native/marble_sim.cpp into marble_sim.dll (same compiler settings as the sweep kernel: /O2 /fp:strict).
$ErrorActionPreference = 'Stop'
$setup = 'C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/Common7/Tools/VsDevCmd.bat'
if (-not (Test-Path -LiteralPath $setup)) { throw 'Visual Studio Build Tools not found.' }
$source = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot 'marble_sim.cpp')).Path
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$out = (Resolve-Path -LiteralPath $OutputDirectory).Path
$script = Join-Path $out 'build.cmd'
$body = @"
@echo off
call "$setup" -arch=x64 -host_arch=x64 >nul
if errorlevel 1 exit /b 1
cd /d "$out"
cl /nologo /LD /O2 /fp:strict /std:c++17 /EHsc "$source" /Fomarble_sim.obj /Femarble_sim.dll
"@
[IO.File]::WriteAllText($script, $body, [Text.UTF8Encoding]::new($false))
& cmd.exe /c $script
if ($LASTEXITCODE -ne 0) { throw "marble_sim build failed: $LASTEXITCODE" }
Get-FileHash -LiteralPath (Join-Path $out 'marble_sim.dll') -Algorithm SHA256
