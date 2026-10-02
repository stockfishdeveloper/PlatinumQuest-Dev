# Watch a whole kotmjump round at 1x: navigator plus planner (nav/learned_nav/watch5.py). Viewing only.
#   powershell -ExecutionPolicy Bypass -File nav\learned_nav\watch5.ps1
# The planner's jump sequences play in slow motion (it needs ~0.3-0.4 s a decision). Output: logs\learned_nav\watch5.txt.
# Close the game window (or kill the python process) to stop.
param([int]$Port = 9071, [int]$Rounds = 1)
$ml = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$pq = [System.IO.Path]::GetFullPath((Join-Path $ml "..\Marble Blast Platinum"))
$py = "C:\Users\doug\AppData\Local\Programs\Python\Python39\python.exe"
. (Join-Path $ml "nav_ready.ps1")
$out = Join-Path $ml "logs\learned_nav\watch5.txt"
$p = Start-Process -FilePath $py -ArgumentList @("-u", "-m", "nav.learned_nav.watch5", "--port", "$Port", "--rounds", "$Rounds") -WorkingDirectory $ml -RedirectStandardOutput $out -RedirectStandardError "$out.err" -PassThru -WindowStyle Hidden
if (-not (Wait-NavPort -Port $Port -TimeoutSec 180)) { Write-Error "watch5.py never opened port $Port (see $out.err)"; exit 1 }
$g = Start-Process -FilePath (Join-Path $pq "marbleblast_mbx.exe") -ArgumentList @("-autotrain", "kotmjump", "-aiport", "$Port") -WorkingDirectory $pq -PassThru
Write-Host ("watch5: python pid {0}, game pid {1}; log in {2}" -f $p.Id, $g.Id, $out)
