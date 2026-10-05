# Live play against people (production, 2026-10-04): starts the model's runner (nav.live_play) and the repo build
# with -ailive and -aifreecam. Then, in the game window: log in as the model's account, join or host a King of the
# Marble server, and start the round. The agent plays the Ready/Set countdown and drives from GO. On this machine
# the mouse and the left/right arrows turn the view only (client/scripts/ai/watch/freeCam.cs) and mouse clicks do
# nothing in game; do not press the other keys while it plays (they act on the marble). Settings (checkpoint, map,
# port) are the defaults in nav/live_play.py.
# Output: logs\learned_nav\live_play.txt (rounds, and [live] lines saying why the agent waits).
# Stop: close the game window, then stop the runner (the last line printed here shows its process id).
# Training is unaffected: start_training.ps1 never passes -ailive.
$ml = $PSScriptRoot
$pq = [System.IO.Path]::GetFullPath((Join-Path $ml "..\Marble Blast Platinum"))
$py = "C:\Users\doug\AppData\Local\Programs\Python\Python39\python.exe"
$out = Join-Path $ml "logs\learned_nav\live_play.txt"
$runner = Start-Process -FilePath $py -ArgumentList @("-u", "-m", "nav.live_play") -WorkingDirectory $ml `
    -RedirectStandardOutput $out -RedirectStandardError "$out.err" -PassThru -WindowStyle Hidden
for ($i = 0; $i -lt 60; $i++) {
    if (Get-NetTCPConnection -LocalPort 8888 -State Listen -ErrorAction SilentlyContinue) { break }
    Start-Sleep -Seconds 1
}
if (-not (Get-NetTCPConnection -LocalPort 8888 -State Listen -ErrorAction SilentlyContinue)) {
    Write-Host "the runner did not start listening on port 8888; see $out.err"
    exit 1
}
$game = Start-Process -FilePath (Join-Path $pq "marbleblast_mbx.exe") -ArgumentList @("-ailive", "-aifreecam") -WorkingDirectory $pq -PassThru
Write-Host ("live play: runner pid {0} (log {1}), game pid {2}. Log in, join or host a KOTM server, start the round." -f $runner.Id, $out, $game.Id)
