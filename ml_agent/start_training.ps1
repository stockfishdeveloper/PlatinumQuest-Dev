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
    [string]$Missions = "FlatIslands_Hunt,KingOfTheMarble_Hunt",
    [string]$Split = "4,4",
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
