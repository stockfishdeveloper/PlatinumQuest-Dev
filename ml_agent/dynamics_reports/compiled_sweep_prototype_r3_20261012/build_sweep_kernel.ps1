param([string]$OutputDirectory = (Join-Path $env:TEMP 'pq_checks/compiled_sweep_20261012'))
$ErrorActionPreference = 'Stop'
$taskCompilerSetup = 'C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/Common7/Tools/VsDevCmd.bat'
if (-not (Test-Path -LiteralPath $taskCompilerSetup)) { throw 'Existing Visual Studio Build Tools not found.' }
$taskKernelSource = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot 'sweep_kernel.cpp')).Path
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$taskOutputRoot = (Resolve-Path -LiteralPath $OutputDirectory).Path
$taskBuildScript = Join-Path $taskOutputRoot 'build.cmd'
$taskBuildBody = @"
@echo off
call "$taskCompilerSetup" -arch=x64 -host_arch=x64
if errorlevel 1 exit /b 1
cd /d "$taskOutputRoot"
cl /nologo /LD /O2 /fp:strict /std:c++17 "$taskKernelSource" /Fosweep_kernel.obj /Fesweep_kernel.dll
"@
[IO.File]::WriteAllText($taskBuildScript,$taskBuildBody,[Text.UTF8Encoding]::new($false))
& cmd.exe /c $taskBuildScript
if ($LASTEXITCODE -ne 0) { throw "Independent sweep prototype build failed: $LASTEXITCODE" }
Get-FileHash -LiteralPath (Join-Path $taskOutputRoot 'sweep_kernel.dll') -Algorithm SHA256
