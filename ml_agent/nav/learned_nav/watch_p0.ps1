# Watch the P0 learned jump chooser at 1x on the drill map (nav/learned_nav/watch.py). Viewing only.
#   powershell -ExecutionPolicy Bypass -File nav\learned_nav\watch_p0.ps1            # held-out feasible starts
#   powershell -ExecutionPolicy Bypass -File nav\learned_nav\watch_p0.ps1 -Fresh     # brand-new random starts
# Output: logs\nav\watch_p0.txt. Close the game window (or kill the python process) to stop.
param([switch]$Fresh, [int]$Port = 8961)
$ml = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$pq = [System.IO.Path]::GetFullPath((Join-Path $ml "..\Marble Blast Platinum"))
$py = "C:\Users\doug\AppData\Local\Programs\Python\Python39\python.exe"
. (Join-Path $ml "nav_ready.ps1")
$pargs = @("-u", "-m", "nav.learned_nav.watch", "--port", "$Port")
if ($Fresh) { $pargs += "--fresh" }
$out = Join-Path $ml "logs\nav\watch_p0.txt"
$p = Start-Process -FilePath $py -ArgumentList $pargs -WorkingDirectory $ml -RedirectStandardOutput $out -RedirectStandardError "$out.err" -PassThru -WindowStyle Hidden
if (-not (Wait-NavPort -Port $Port -TimeoutSec 120)) { Write-Error "watch.py never opened port $Port (see $out.err)"; exit 1 }
$g = Start-Process -FilePath (Join-Path $pq "marbleblast_mbx.exe") -ArgumentList @("-autotrain", "kotmjump_p0", "-aiport", "$Port") -WorkingDirectory $pq -PassThru
Write-Host ("watch_p0: python pid {0}, game pid {1}; choices and outcomes in {2}" -f $p.Id, $g.Id, $out)
