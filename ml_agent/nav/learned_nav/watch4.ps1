# Watch the stage 4 planner take a floating gem at 1x (nav/learned_nav/watch4.py). Viewing only.
#   powershell -ExecutionPolicy Bypass -File nav\learned_nav\watch4.ps1                  # kotmjump_p0
#   powershell -ExecutionPolicy Bypass -File nav\learned_nav\watch4.ps1 -Map kotmjump_p2
# Two games: the planner plays each trial in a minimized game, and the visible game replays it at real speed.
# Output: logs\learned_nav\watch4.txt. Close the visible game window (or kill the python process) to stop.
param([string]$Map = "kotmjump_p0", [int]$Port = 9071)
$ml = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$pq = [System.IO.Path]::GetFullPath((Join-Path $ml "..\Marble Blast Platinum"))
$py = "C:\Users\doug\AppData\Local\Programs\Python\Python39\python.exe"
. (Join-Path $ml "nav_ready.ps1")
$out = Join-Path $ml "logs\learned_nav\watch4.txt"
$Port2 = $Port + 1
$p = Start-Process -FilePath $py -ArgumentList @("-u", "-m", "nav.learned_nav.watch4", "--port", "$Port", "--port2", "$Port2", "--map", $Map) -WorkingDirectory $ml -RedirectStandardOutput $out -RedirectStandardError "$out.err" -PassThru -WindowStyle Hidden
if (-not (Wait-NavPort -Port $Port2 -TimeoutSec 120)) { Write-Error "watch4.py never opened port $Port2 (see $out.err)"; exit 1 }
$g2 = Start-Process -FilePath (Join-Path $pq "marbleblast_mbx.exe") -ArgumentList @("-autotrain", $Map, "-aiport", "$Port2") -WorkingDirectory $pq -PassThru -WindowStyle Minimized
if (-not (Wait-NavPort -Port $Port -TimeoutSec 180)) { Write-Error "watch4.py never opened port $Port (see $out.err)"; exit 1 }
$g = Start-Process -FilePath (Join-Path $pq "marbleblast_mbx.exe") -ArgumentList @("-autotrain", $Map, "-aiport", "$Port") -WorkingDirectory $pq -PassThru
Write-Host ("watch4: python pid {0}, planning game pid {1} (minimized), visible game pid {2}; results in {3}" -f $p.Id, $g2.Id, $g.Id, $out)
