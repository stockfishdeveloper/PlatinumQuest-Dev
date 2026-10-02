# Run a learned_nav module once per map, $Batch games at a time; each module gets --port P --map M [extra args].
param([string]$Module, [string[]]$Maps, [int]$Batch = 4, [int]$Port0 = 8991, [string]$Tag = "verify", [int]$TimeoutSec = 900, [string]$ArgStr = "")
$Maps = @($Maps | ForEach-Object { $_ -split "," } | Where-Object { $_ })
$ml = "C:\Users\doug\OneDrive\Documents\GitHub\PlatinumQuest-Dev\ml_agent"
$pq = [System.IO.Path]::GetFullPath((Join-Path $ml "..\Marble Blast Platinum"))
$py = "C:\Users\doug\AppData\Local\Programs\Python\Python39\python.exe"
. (Join-Path $ml "nav_ready.ps1")
$logdir = Join-Path $ml "logs\learned_nav"
for ($i = 0; $i -lt $Maps.Count; $i += $Batch) {
    $jobs = @()
    for ($k = 0; $k -lt $Batch -and ($i + $k) -lt $Maps.Count; $k++) {
        $m = $Maps[$i + $k]; $port = $Port0 + $k
        $out = Join-Path $logdir "${Tag}_$m.txt"
        $pargs = @("-u", "-m", $Module, "--port", "$port", "--map", $m)
        if ($ArgStr) { $pargs += $ArgStr.Split(" ") }
        $p = Start-Process -FilePath $py -ArgumentList $pargs -WorkingDirectory $ml -RedirectStandardOutput $out -RedirectStandardError "$out.err" -PassThru -WindowStyle Hidden
        $null = Wait-NavPort -Port $port -TimeoutSec 60 -Quiet
        $g = Start-Process -FilePath (Join-Path $pq "marbleblast_mbx.exe") -ArgumentList @("-autotrain", $m, "-aiport", "$port") -WorkingDirectory $pq -PassThru
        $jobs += [pscustomobject]@{ Map = $m; Py = $p; Game = $g; Out = $out }
    }
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline -and ($jobs | Where-Object { -not $_.Py.HasExited })) { Start-Sleep 3 }
    foreach ($j in $jobs) {
        if (-not $j.Py.HasExited) { Stop-Process -Id $j.Py.Id -Force -ErrorAction SilentlyContinue; Write-Host ("{0}: TIMEOUT" -f $j.Map) }
        else { Write-Host ("{0}: {1}" -f $j.Map, (Get-Content $j.Out -Tail 1)) }
        if (-not $j.Game.HasExited) { $null = $j.Game.CloseMainWindow(); Start-Sleep 1; if (-not $j.Game.HasExited) { Stop-Process -Id $j.Game.Id -Force -ErrorAction SilentlyContinue } }
    }
}
