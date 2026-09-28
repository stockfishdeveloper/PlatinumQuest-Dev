# Watch "where will it land" at 1x (nav/learned_nav/watch3.py). Viewing only.
#   powershell -ExecutionPolicy Bypass -File nav\learned_nav\watch3.ps1                     # Gems Ahoy (never trained on)
#   powershell -ExecutionPolicy Bypass -File nav\learned_nav\watch3.ps1 -Map Acropolis2_Hunt_phys
# Output: logs\nav\watch3.txt. Close the game window (or kill the python process) to stop.
param([string]$Map = "GemsAhoy_Hunt_phys", [int]$Port = 9061)
$ml = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$pq = [System.IO.Path]::GetFullPath((Join-Path $ml "..\Marble Blast Platinum"))
$py = "C:\Users\doug\AppData\Local\Programs\Python\Python39\python.exe"
. (Join-Path $ml "nav_ready.ps1")
$out = Join-Path $ml "logs\nav\watch3.txt"
$p = Start-Process -FilePath $py -ArgumentList @("-u", "-m", "nav.learned_nav.watch3", "--port", "$Port", "--map", $Map) -WorkingDirectory $ml -RedirectStandardOutput $out -RedirectStandardError "$out.err" -PassThru -WindowStyle Hidden
if (-not (Wait-NavPort -Port $Port -TimeoutSec 120)) { Write-Error "watch3.py never opened port $Port (see $out.err)"; exit 1 }
$g = Start-Process -FilePath (Join-Path $pq "marbleblast_mbx.exe") -ArgumentList @("-autotrain", $Map, "-aiport", "$Port") -WorkingDirectory $pq -PassThru
Write-Host ("watch3: python pid {0}, game pid {1}; results in {2}" -f $p.Id, $g.Id, $out)
