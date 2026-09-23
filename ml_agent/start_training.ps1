# start_training.ps1 -- bring up training in the ONLY correct order.
#
# READY AT GO (user rule, 2026-09-21, no exceptions): the trainer starts first so its 8 worker
# sockets are bound and listening, then the game loop launches the instances. Python is the socket
# SERVER (nav/env.py:51-53) and each game dials its port 100 ms after "GO!" (mlAgent.cs:677-686),
# so launching games first leaves every marble idling on its start pad until the workers come up.
#
#   .\start_training.ps1
#   .\start_training.ps1 -Missions "FlatIslands_Hunt,KingOfTheMarble_Hunt" -Split "4,4"
param(
    [string]$Missions = "KingOfTheMarble_Hunt,FlatIslands_Hunt",   # 2026-09-22: FlatGem dropped (saturated at 93-95 % of human, unchanged by every change today)
    [string]$Split = "7,1",        # 2026-09-22 20:40: 7 KOTM / 1 Islands (HANDOFF 28.13). Islands stays: only map with dense jump edges.
                                   # (superseded) 4,3,1 -> 2,5,1 at the operator request: KOTM is the
                                   # scored map and the behaviours HANDOFF section 28 targets (edge-cell
                                   # gems, 4 u ring pairs, the centre block entry) only occur there.
                                   # FlatGem kept at 2 as the no-edges pacing control.
                                   # (earlier note) 2026-09-21: FlatGemTraining added as the SPEED map (flat, no
                                   # holes, a recorded human baseline of 9.92 u/s between pickups
                                   # and 2.80 s/gem to aim at). KOTM keeps a strong share because
                                   # it is the map we are scored on. One Islands instance is kept
                                   # purely so gap-crossing does not decay: it is the only map with
                                   # a meaningful number of jump edges (1708 vs KOTM's 228).
                                   # Block-assigned: inst 0-3 FlatGem, 4-6 KOTM, 7 Islands.
    [int]$Instances = 8,
    [int]$Port0 = 8888
)
$ml = $PSScriptRoot
$py = "C:\Users\doug\AppData\Local\Programs\Python\Python39\python.exe"
. (Join-Path $ml "nav_ready.ps1")
function Stamp { (Get-Date).ToString("HH:mm:ss") }

# refuse to stack a second trainer or a second loop on top of a live one
$live = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'nav\.train_nav' })
if ($live.Count -gt 0) { Write-Error ("a trainer is already running (pid {0}); stop it first" -f $live[0].ProcessId); exit 1 }

Write-Host ("[{0}] starting the trainer (binds {1}..{2} before any game launches)" -f (Stamp), $Port0, ($Port0 + $Instances - 1))
$t = Start-Process -FilePath $py -ArgumentList @("-u", "-m", "nav.train_nav") -WorkingDirectory $ml -RedirectStandardOutput (Join-Path $ml "logs\nav\stdout.txt") -RedirectStandardError (Join-Path $ml "logs\nav\stderr.txt") -WindowStyle Hidden -PassThru
Write-Host "  trainer pid $($t.Id)"

if (-not (Wait-NavPorts -Port0 $Port0 -Count $Instances -TimeoutSec 240)) {
    Write-Warning "not every worker port opened; the games will retry, but check logs\nav\stderr.txt"
}

Write-Host ("[{0}] launching {1} game instance(s): {2} split {3}" -f (Stamp), $Instances, $Missions, $Split)
Start-Process -FilePath "powershell.exe" -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $ml "run_game_loop.ps1"), "-Mission", $Missions, "-Split", $Split, "-Instances", "$Instances") -WorkingDirectory $ml -WindowStyle Hidden | Out-Null
Write-Host ("[{0}] up. Every instance is answered on its first dial at GO." -f (Stamp))
