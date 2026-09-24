# real_run.ps1 -- the TRANSFER TEST: play real Hunt rounds where the goals are the game's own
# gems, not synthetic waypoints.
#
# Everything the navigator has trained on is a terrain-sampled waypoint reached from a teleport,
# with arrival judged by our own radius check. This plays the actual game instead: gems spawn
# where the game puts them, play is continuous, and a pickup is whatever the game scores.
# It trains nothing and writes no checkpoint.
#
#   .\real_run.ps1                                       # KOTM, 1 round, nav_latest
#   .\real_run.ps1 -Map FlatIslands_Hunt -Rounds 3
#   .\real_run.ps1 -Ckpt models\nav\nav_005750.pth
#   .\real_run.ps1 -ValueWeight 1.0                      # prefer higher-value (yellow) gems
#
# IMPORTANT: stop training first. The 8 GB GPU has no room for a 9th game instance alongside the
# trainer's PPO update (each instance ~450 MiB, the update ~1.9 GB) -- the same constraint
# eval_heldout.ps1 carries. This script refuses to start while the trainer is running.

param(
    [string]$Map = "KingOfTheMarble_Hunt",
    [int]$Rounds = 1,
    [string]$Ckpt = "",                                  # default: models/nav/nav_latest.pth
    [double]$ValueWeight = 0,
    [int]$Port = 8920,                                   # clear of the training ports 8888..8895
    [switch]$Force,                                      # run even if the trainer is up (may OOM)
    [string]$Exe = "marbleblast_mbx.exe"
)
$ml = $PSScriptRoot
$pq = [System.IO.Path]::GetFullPath((Join-Path $ml "..\Marble Blast Platinum"))
$exePath = Join-Path $pq $Exe
if (-not (Test-Path $exePath)) { Write-Error "not found: $exePath"; exit 1 }
$py = "C:\Users\doug\AppData\Local\Programs\Python\Python39\python.exe"

$trainer = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'nav.train_nav' })
if ($trainer.Count -gt 0 -and -not $Force) {
    Write-Error ("the trainer is running (pid {0}). Stop it first, or pass -Force to risk a GPU OOM." -f $trainer[0].ProcessId)
    exit 1
}

$npz = Join-Path $ml "terrain_maps\terrain_$Map.npz"
if (-not (Test-Path $npz)) { Write-Error "no terrain map ($npz); run: python generate_terrain_map.py $Map"; exit 1 }

# READY AT GO (user rule, 2026-09-21): Python is the socket SERVER and the game dials us 100 ms
# after "GO!". Start Python FIRST, wait for its port to be LISTENING, and only then launch the
# game, so the bridge is answered on the first dial and the marble rolls on the first tick.
# The old order (game, blind Start-Sleep 30, then Python) meant the round began with nobody
# listening and the marble sat on the pad.
. (Join-Path $ml "nav_ready.ps1")

$env:NAV_PORT = "$Port"
$env:NAV_ROUNDS = "$Rounds"
$env:NAV_VALUE_WEIGHT = "$ValueWeight"
$env:NAV_MAP = $Map                    # pre-build the terrain grid off the critical path
if ($Ckpt -ne "") { $env:NAV_CKPT = (Resolve-Path $Ckpt).Path }
$out = Join-Path $ml "logs\nav\real_run_$Map.txt"
Write-Host ("[{0}] starting the navigator on port {1} (it binds before anything else)" -f (Get-Date).ToString("HH:mm:ss"), $Port)
$p = Start-Process -FilePath $py -ArgumentList @("-u", "-m", "nav.real_run") -WorkingDirectory $ml -RedirectStandardOutput $out -RedirectStandardError "$out.err" -PassThru -WindowStyle Hidden
if (-not (Wait-NavPort -Port $Port -TimeoutSec 120)) {
    Write-Error "navigator never opened port $Port; not launching the game"
    if (-not $p.HasExited) { $p.Kill() }
    exit 1
}
Write-Host ("[{0}] launching {1} on port {2}" -f (Get-Date).ToString("HH:mm:ss"), $Map, $Port)
$g = Start-Process -FilePath $exePath -ArgumentList @("-autotrain", $Map, "-aiport", "$Port") -WorkingDirectory $pq -PassThru
Write-Host "  game pid $($g.Id); it will be answered the instant it reaches GO"
$timeoutMs = 300000 * $Rounds + 180000
if (-not $p.WaitForExit($timeoutMs)) { Write-Host "  run timed out"; $p.Kill() }
Write-Host ("  navigator exit code {0}" -f $p.ExitCode)

Write-Host ""
Get-Content $out -ErrorAction SilentlyContinue | ForEach-Object { "  $_" }
$err = Get-Content "$out.err" -ErrorAction SilentlyContinue | Select-Object -Last 6
if ($err) { Write-Host "  stderr tail:"; $err | ForEach-Object { "    $_" } }
if (-not $g.HasExited) { Stop-Process -Id $g.Id -Force }
Write-Host ("[{0}] done. JSON: logs\nav\real_run_{1}_*.json" -f (Get-Date).ToString("HH:mm:ss"), $Map)
