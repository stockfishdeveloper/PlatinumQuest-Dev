# Real-time planner play (2026-10-19): starts the planner server (nav.marble_dynamics.realtime) and the game with the
# free camera. Then, in the game window: Play > Hunt > pick a map and start it. The planner loads that map (the game
# holds a few seconds the first time a map is chosen), plays the Ready/Set countdown at 1x and drives from GO in real
# time; later rounds and other maps work the same way. The mouse turns and tilts the view only (the left/right arrow
# keys turn it too); do not press the other keys while it plays (they act on the marble).
# Maps it can plan: the ten pool maps (collision and terrain data in ml_agent/collision_maps and terrain_maps).
# Output: logs\realtime\play_realtime.txt (rounds, points, planner timing), play_realtime_game.txt (the game's
# console); per-round journals in dynamics_reports\realtime\<date>\.
# Stop: close the game window, then stop the server (its process id is printed below).
$ml = $PSScriptRoot
$repo = [System.IO.Path]::GetFullPath((Join-Path $ml ".."))
$pq = Join-Path $repo "Marble Blast Platinum"
$py = "C:\Users\doug\AppData\Local\Programs\Python\Python39\python.exe"
$logs = Join-Path $ml "logs\realtime"; New-Item -ItemType Directory -Force $logs | Out-Null
$out = Join-Path $logs "play_realtime.txt"
if (Get-NetTCPConnection -LocalPort 8888 -State Listen -ErrorAction SilentlyContinue) {
    Write-Host "port 8888 is already in use (another runner, e.g. play_live.ps1 or an earlier play_realtime.ps1); stop it first"
    exit 1
}
$env:PYTHONPATH = $ml
$server = Start-Process -FilePath $py -ArgumentList @("-u", "-m", "nav.marble_dynamics.realtime") -WorkingDirectory $repo `
    -RedirectStandardOutput $out -RedirectStandardError (Join-Path $logs "play_realtime.err") -PassThru -WindowStyle Hidden
for ($i = 0; $i -lt 60; $i++) {
    if (Get-NetTCPConnection -LocalPort 8888 -State Listen -ErrorAction SilentlyContinue) { break }
    Start-Sleep -Seconds 1
}
if (-not (Get-NetTCPConnection -LocalPort 8888 -State Listen -ErrorAction SilentlyContinue)) {
    Write-Host "the planner server did not start listening on port 8888; see $logs\play_realtime.err"
    exit 1
}
# marbleblast_realtime.exe (2026-10-19): the pool exe plus real-time frames (each 16 ms frame waits for its own 16 ms of
# wall clock, so the picture moves every frame; REALTIMESTEP in mlAgent.cs) and the mouse's up/down tilting the view only
$game = Start-Process -FilePath (Join-Path $pq "marbleblast_realtime.exe") -ArgumentList @("-offline", "-aifreecam") -WorkingDirectory $pq `
    -RedirectStandardOutput (Join-Path $logs "play_realtime_game.txt") -RedirectStandardError (Join-Path $logs "play_realtime_game.err") -PassThru
Write-Host ("real-time play: server pid {0} (log {1}), game pid {2}. In the game: Play > Hunt > a map." -f $server.Id, $out, $game.Id)
