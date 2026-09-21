# run_dirskip.ps1 -- inference-time test of proposal 1: scale the dir_head's goal-bearing path.
#
# Plays one real round per scale with the scaling applied at inference only. Nothing is trained,
# no checkpoint is written. Each round's trace is saved separately so aim concentration, speed and
# gems can be compared per condition.
#
#   .\run_dirskip.ps1                       # scales 1 (control), 10, 30
#   .\run_dirskip.ps1 -Scales 1,10,30 -Rounds 1
param(
    [string]$Map = "FlatGemTraining_Hunt",
    [int]$Rounds = 1,
    [string]$Ckpt = "models\nav\nav_eval_flatgem_1208.pth",
    [double[]]$Scales = @(1, 10, 30)
)
$ml = $PSScriptRoot
$py = "C:\Users\doug\AppData\Local\Programs\Python\Python39\python.exe"
. (Join-Path $ml "nav_ready.ps1")
$pq = [System.IO.Path]::GetFullPath((Join-Path $ml "..\Marble Blast Platinum"))

foreach ($s in $Scales) {
    Write-Host ("[{0}] === NAV_DIRSKIP = {1}" -f (Get-Date).ToString("HH:mm:ss"), $s)
    $procs = @(Get-CimInstance Win32_Process | Where-Object { ($_.Name -eq 'python.exe' -and ($_.CommandLine -match 'nav\.' -or $_.CommandLine -match 'dirskip')) -or $_.Name -match 'marbleblast' })
    foreach ($p in $procs) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 7

    $env:NAV_PORT = "8920"; $env:NAV_ROUNDS = "$Rounds"; $env:NAV_MAP = $Map
    $env:NAV_CKPT = (Join-Path $ml $Ckpt); $env:NAV_DIRSKIP = "$s"
    Remove-Item Env:NAV_WATCH, Env:NAV_VIEW_SUBSTEPS, Env:NAV_MARK, Env:NAV_SMOOTH, Env:NAV_FORCE_THROTTLE -ErrorAction SilentlyContinue
    $out = Join-Path $ml ("logs\nav\dirskip_{0}.txt" -f $s)
    $p = Start-Process -FilePath $py -ArgumentList @("-u", "dirskip_scale.py") -WorkingDirectory $ml -RedirectStandardOutput $out -RedirectStandardError "$out.err" -PassThru -WindowStyle Hidden
    if (-not (Wait-NavPort -Port 8920 -TimeoutSec 120)) { Write-Error "port never opened"; $p.Kill(); continue }
    $g = Start-Process -FilePath (Join-Path $pq "marbleblast_mbx.exe") -ArgumentList @("-autotrain", $Map, "-aiport", "8920") -WorkingDirectory $pq -PassThru
    if (-not $p.WaitForExit(600000)) { Write-Host "  timed out"; $p.Kill() }
    if (-not $g.HasExited) { Stop-Process -Id $g.Id -Force }
    Copy-Item (Join-Path $ml "logs\nav\real_trace.csv") (Join-Path $ml ("logs\nav\trace_dirskip_{0}.csv" -f $s)) -Force -ErrorAction SilentlyContinue
    Get-Content $out -ErrorAction SilentlyContinue | Where-Object { $_ -match "dirskip|round 1:" } | ForEach-Object { "    $_" }
}
$procs = @(Get-CimInstance Win32_Process | Where-Object { ($_.Name -eq 'python.exe' -and ($_.CommandLine -match 'nav\.' -or $_.CommandLine -match 'dirskip')) -or $_.Name -match 'marbleblast' })
foreach ($p in $procs) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }
Remove-Item Env:NAV_DIRSKIP -ErrorAction SilentlyContinue
Write-Host ("[{0}] all scales complete" -f (Get-Date).ToString("HH:mm:ss"))
