# smooth_ab.ps1 -- does holding a heading longer actually make the marble faster?
#
# THE QUESTION (2026-09-21). The step-size probe showed the marble reaches 21.8 u/s on a constant
# held input, and that a 64 ms decision delivers 99.6 % of what four 16 ms ticks deliver, so the
# decision rate is NOT costing thrust. Yet the agent averages 8.4 u/s and peaks at 16.6, while the
# human averages 10.0 and peaks at 21.0. The agent holds a heading within 30 deg for a median of
# ONE decision; the human holds for four.
#
# This runs the same checkpoint at several values of NAV_SMOOTH, an EMA on the commanded direction
# (1.0 = raw policy output, lower = the command is forced to persist), and measures speed and gems.
# If persistence is causal, speed rises as SMOOTH falls. If gems fall at the same time, persistence
# buys speed at the cost of accuracy, which is the trade worth knowing about.
#
# This is a MEASUREMENT, not a proposed fix: ACTION_SMOOTH was removed from training on 2026-09-20
# precisely because imposing it from outside is a crutch. The point is to find out whether the
# jitter is causal before spending another experiment on it.
param(
    [string]$Map = "FlatGemTraining_Hunt",
    [int]$Rounds = 3,
    [string]$Ckpt = "models\nav\nav_eval_flatgem_1208.pth",
    [double[]]$Smooth = @(1.0, 0.5, 0.3)
)
$ml = $PSScriptRoot
. (Join-Path $ml "nav_ready.ps1")
$out = Join-Path $ml "logs\nav\smooth_ab.txt"
"" | Set-Content $out
foreach ($s in $Smooth) {
    Write-Host ("[{0}] === NAV_SMOOTH = {1}" -f (Get-Date).ToString("HH:mm:ss"), $s)
    $procs = @(Get-CimInstance Win32_Process | Where-Object { ($_.Name -eq 'python.exe' -and $_.CommandLine -match 'nav\.') -or $_.Name -match 'marbleblast' })
    foreach ($p in $procs) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 6
    Remove-Item Env:NAV_WATCH, Env:NAV_VIEW_SUBSTEPS, Env:NAV_MARK -ErrorAction SilentlyContinue
    $env:NAV_SMOOTH = "$s"
    & (Join-Path $ml "real_run.ps1") -Map $Map -Rounds $Rounds -Ckpt (Join-Path $ml $Ckpt) | Out-Null
    Copy-Item (Join-Path $ml "logs\nav\real_trace.csv") (Join-Path $ml ("logs\nav\trace_smooth_{0}.csv" -f $s)) -Force
    $line = Select-String -Path (Join-Path $ml ("logs\nav\real_run_{0}.txt" -f $Map)) -Pattern "'gems':" -AllMatches
    Add-Content $out ("NAV_SMOOTH=" + $s)
    $line | ForEach-Object { Add-Content $out ("  " + $_.Line) }
    Write-Host ("[{0}]   done, trace saved" -f (Get-Date).ToString("HH:mm:ss"))
}
$procs = @(Get-CimInstance Win32_Process | Where-Object { ($_.Name -eq 'python.exe' -and $_.CommandLine -match 'nav\.') -or $_.Name -match 'marbleblast' })
foreach ($p in $procs) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }
Remove-Item Env:NAV_SMOOTH -ErrorAction SilentlyContinue
Write-Host ("[{0}] all conditions complete" -f (Get-Date).ToString("HH:mm:ss"))
