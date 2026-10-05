# Watch the model play at 1x with a free camera (2026-10-04). Starts the model's runner in watch mode
# (nav.real_run: real time, every frame drawn) and the repo build with -aifreecam, so the mouse and the left/right
# arrows turn the view without touching the model's steering (client/scripts/ai/watch/freeCam.cs). Mouse clicks
# do nothing while it plays; the keyboard still acts on the marble, and Esc pauses the round.
# Usage: .\watch_model.ps1 [map]    default KingOfTheMarble_Hunt, e.g. .\watch_model.ps1 Skatium_Hunt
# The map needs a terrain map in terrain_maps\ (python generate_terrain_map.py makes one).
# Output: logs\learned_nav\rr_watch_<map>.txt (one line per round). Stop: close the game window; this script then
# stops the runner.
param([string]$Map = "KingOfTheMarble_Hunt")
$ml = $PSScriptRoot
$pq = [System.IO.Path]::GetFullPath((Join-Path $ml "..\Marble Blast Platinum"))
$py = "C:\Users\doug\AppData\Local\Programs\Python\Python39\python.exe"
$port = 9961
$out = Join-Path $ml "logs\learned_nav\rr_watch_$Map.txt"
# The runner's settings go to its process only; the old values come back right after it starts.
$set = @{ NAV_WATCH = "1"; NAV_SPEED = "1"; NAV_VIEW_SUBSTEPS = "4"; NAV_ROUNDS = "50"; NAV_PORT = "$port";
          NAV_MAP = $Map; NAV_CKPT = "models/nav/nav_latest.pth"; NAV_TOUR = "walk"; NAV_TAG = "watch_$Map";
          NAV_TRACE = "logs/nav/real_trace_watch_$Map.csv"; CUDA_VISIBLE_DEVICES = "-1" }
$old = @{}
foreach ($k in $set.Keys) {
    $old[$k] = [Environment]::GetEnvironmentVariable($k, "Process")
    [Environment]::SetEnvironmentVariable($k, $set[$k], "Process")
}
try {
    $runner = Start-Process -FilePath $py -ArgumentList @("-u", "-m", "nav.real_run") -WorkingDirectory $ml `
        -RedirectStandardOutput $out -RedirectStandardError "$out.err" -PassThru -WindowStyle Hidden
} finally {
    foreach ($k in $set.Keys) { [Environment]::SetEnvironmentVariable($k, $old[$k], "Process") }
}
for ($i = 0; $i -lt 60; $i++) {
    if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) { break }
    Start-Sleep -Seconds 1
}
if (-not (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)) {
    Write-Host "the runner did not start listening on port $port; see $out.err"
    Stop-Process -Id $runner.Id -Force -ErrorAction SilentlyContinue
    exit 1
}
$game = Start-Process -FilePath (Join-Path $pq "marbleblast_mbx.exe") -WorkingDirectory $pq -PassThru `
    -ArgumentList @("-autotrain", $Map, "-aiport", "$port", "-offline", "-aifreecam")
Write-Host ("watching {0}: runner pid {1} (log {2}), game pid {3}. Close the game window to stop." -f $Map, $runner.Id, $out, $game.Id)
Wait-Process -Id $game.Id
Stop-Process -Id $runner.Id -Force -ErrorAction SilentlyContinue
Write-Host "game closed; runner stopped"
