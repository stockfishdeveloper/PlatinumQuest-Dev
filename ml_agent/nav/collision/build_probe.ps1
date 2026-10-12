param(
    [string]$EngineRoot = 'C:/Users/doug/src/OpenPQ-TGEMIT-mbx'
)
$ErrorActionPreference = 'Stop'
$taskWorkspace = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$taskEngineRoot = (Resolve-Path -LiteralPath $EngineRoot).Path
$taskMarbleDir = Join-Path $taskEngineRoot 'engine/game/marble'
$taskInclude = Join-Path $taskMarbleDir 'aiGeometryExport.inc'
$taskSource = Join-Path $taskMarbleDir 'marble.cc'
$taskProbe = Join-Path $taskWorkspace 'Marble Blast Platinum/marbleblast_geometry_probe.exe'
if (Get-Process -Name 'marbleblast_geometry_probe' -ErrorAction SilentlyContinue) {
    throw 'Close geometry probe sessions before rebuilding their executable.'
}
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'engine_geometry.inc') -Destination $taskInclude
$taskCode = [IO.File]::ReadAllText($taskSource)
if (-not $taskCode.Contains('#include "aiGeometryExport.inc"')) {
    [IO.File]::AppendAllText($taskSource, "`n// Read-only geometry probe instrumentation.`n#include `"aiGeometryExport.inc`"`n", [Text.UTF8Encoding]::new($false))
}
$taskVS = 'C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VsDevCmd.bat'
if (-not (Test-Path -LiteralPath $taskVS)) {
    $taskVS = 'C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/Common7/Tools/VsDevCmd.bat'
}
$taskCmake = 'C:/Program Files/CMake/bin/cmake.exe'
$taskRuntime = Join-Path $env:TEMP 'pq_checks/collision_maps_20261011'
New-Item -ItemType Directory -Path $taskRuntime -Force | Out-Null
$taskCommand = Join-Path $taskRuntime 'build_probe.cmd'
$taskBody = "@echo off`r`ncall `"$taskVS`" -arch=x64 -host_arch=x64`r`n`"$taskCmake`" --build `"$taskEngineRoot/build`" --target TGEMit_Demo -j 4`r`n"
[IO.File]::WriteAllText($taskCommand, $taskBody, [Text.UTF8Encoding]::new($false))
& cmd.exe /c $taskCommand
if ($LASTEXITCODE -ne 0) { throw "Geometry probe build failed: $LASTEXITCODE" }
Copy-Item -LiteralPath (Join-Path $taskEngineRoot 'game/TGEMit_Demo.exe') -Destination $taskProbe
$taskDso = Join-Path $taskWorkspace 'Marble Blast Platinum/platinum/client/scripts/ai/mlAgent.cs.dso'
if (Test-Path -LiteralPath $taskDso) { Remove-Item -LiteralPath $taskDso }
Write-Output "Built separate geometry probe: $taskProbe"
