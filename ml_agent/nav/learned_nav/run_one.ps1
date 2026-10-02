# Run one learned_nav module with its own game on one port (stage 6b launcher; a single-game version of
# stage5b/run_maps.ps1). Example:
#   powershell -ExecutionPolicy Bypass -File nav\learned_nav\run_one.ps1 -Module nav.learned_nav.crossdrill -Map kotmjump_p0 -Port 9621 -Tag v0_dev -ArgStr "--set dev --tag v0"
param([string]$Module, [string]$Map, [int]$Port, [string]$Tag = "run", [int]$TimeoutSec = 7200, [string]$ArgStr = "")
$ml = "C:\Users\doug\OneDrive\Documents\GitHub\PlatinumQuest-Dev\ml_agent"
$pq = [System.IO.Path]::GetFullPath((Join-Path $ml "..\Marble Blast Platinum"))
$py = "C:\Users\doug\AppData\Local\Programs\Python\Python39\python.exe"
. (Join-Path $ml "nav_ready.ps1")
$out = Join-Path $ml "logs\learned_nav\${Tag}_$Map.txt"
$pargs = @("-u", "-m", $Module, "--port", "$Port", "--map", $Map)
if ($ArgStr) { $pargs += $ArgStr.Split(" ") }
$p = Start-Process -FilePath $py -ArgumentList $pargs -WorkingDirectory $ml -RedirectStandardOutput $out -RedirectStandardError "$out.err" -PassThru -WindowStyle Hidden
$null = Wait-NavPort -Port $Port -TimeoutSec 60 -Quiet
$g = Start-Process -FilePath (Join-Path $pq "marbleblast_mbx.exe") -ArgumentList @("-autotrain", $Map, "-aiport", "$Port") -WorkingDirectory $pq -PassThru
Write-Host ("{0}: python {1} game {2} -> {3}" -f $Tag, $p.Id, $g.Id, $out)
$deadline = (Get-Date).AddSeconds($TimeoutSec)
while ((Get-Date) -lt $deadline -and -not $p.HasExited) { Start-Sleep 3 }
if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue; Write-Host ("{0}: TIMEOUT" -f $Tag) }
else { Write-Host ("{0}: {1}" -f $Tag, (Get-Content $out -Tail 1)) }
if (-not $g.HasExited) { $null = $g.CloseMainWindow(); Start-Sleep 1; if (-not $g.HasExited) { Stop-Process -Id $g.Id -Force -ErrorAction SilentlyContinue } }
