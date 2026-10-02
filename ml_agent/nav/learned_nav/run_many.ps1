# Run N copies of a learned_nav module on one map, each with its own game and port (stage 6b gate launcher).
# Tags are <Tag>0..<Tag>N-1; each copy gets --port P --map M --tag <TagK> [extra args]. Waits for all copies.
#   powershell -ExecutionPolicy Bypass -File nav\learned_nav\run_many.ps1 -Module nav.learned_nav.hybrid -Map KingOfTheMarble_Hunt -N 8 -Port0 9701 -Tag g1_hyb -ArgStr "--rounds 1 --memory current --shortcuts 1 --rescue 1"
param([string]$Module, [string]$Map, [int]$N = 8, [int]$Port0 = 9701, [string]$Tag = "run", [int]$TimeoutSec = 5400, [string]$ArgStr = "", [int]$StaggerSec = 6)
$ml = "C:\Users\doug\OneDrive\Documents\GitHub\PlatinumQuest-Dev\ml_agent"
$pq = [System.IO.Path]::GetFullPath((Join-Path $ml "..\Marble Blast Platinum"))
$py = "C:\Users\doug\AppData\Local\Programs\Python\Python39\python.exe"
. (Join-Path $ml "nav_ready.ps1")
$jobs = @()
for ($k = 0; $k -lt $N; $k++) {
    $port = $Port0 + $k; $t = "$Tag$k"
    $out = Join-Path $ml "logs\learned_nav\${t}_$Map.txt"
    $pargs = @("-u", "-m", $Module, "--port", "$port", "--map", $Map, "--tag", $t)
    if ($ArgStr) { $pargs += $ArgStr.Split(" ") }
    $p = Start-Process -FilePath $py -ArgumentList $pargs -WorkingDirectory $ml -RedirectStandardOutput $out -RedirectStandardError "$out.err" -PassThru -WindowStyle Hidden
    $null = Wait-NavPort -Port $port -TimeoutSec 90 -Quiet
    $g = Start-Process -FilePath (Join-Path $pq "marbleblast_mbx.exe") -ArgumentList @("-autotrain", $Map, "-aiport", "$port") -WorkingDirectory $pq -PassThru
    Write-Host ("{0}: python {1} game {2}" -f $t, $p.Id, $g.Id)
    $jobs += [pscustomobject]@{ Tag = $t; Py = $p; Game = $g; Out = $out }
    Start-Sleep $StaggerSec
}
$deadline = (Get-Date).AddSeconds($TimeoutSec)
while ((Get-Date) -lt $deadline -and ($jobs | Where-Object { -not $_.Py.HasExited })) { Start-Sleep 5 }
foreach ($j in $jobs) {
    if (-not $j.Py.HasExited) { Stop-Process -Id $j.Py.Id -Force -ErrorAction SilentlyContinue; Write-Host ("{0}: TIMEOUT" -f $j.Tag) }
    else { Write-Host ("{0}: {1}" -f $j.Tag, (Get-Content $j.Out -Tail 1)) }
    if (-not $j.Game.HasExited) { Stop-Process -Id $j.Game.Id -Force -ErrorAction SilentlyContinue }
}
