# eval_both.ps1 -- stop training, snapshot, and play 8 real rounds on BOTH FlatGem and KOTM.
#
# eval_cycle.ps1 evaluates one map (KOTM) and restarts training afterwards. This one evaluates
# the two maps the DIR_GOAL_GAIN experiment is judged on and leaves training DOWN, because
# restarting is the operator's call, not the script's.
#
#   .\eval_both.ps1 -Tag dirgain
#   .\eval_both.ps1 -Tag dirgain -Rounds 8
param(
    [string]$Tag = "dirgain",
    [int]$Rounds = 8,
    [string]$Ckpt = "",                                   # default: snapshot models\nav\nav_latest.pth now
    [string[]]$Maps = @("FlatGemTraining_Hunt", "KingOfTheMarble_Hunt")
)
$ml = $PSScriptRoot
function Stamp { (Get-Date).ToString("HH:mm:ss") }

# ---- 1. stop everything that trains (the loop FIRST, or it relaunches the games)
$loop = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'powershell' -and $_.CommandLine -match 'run_game_loop' })
foreach ($p in $loop) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }
$procs = @(Get-CimInstance Win32_Process | Where-Object {
    ($_.Name -eq 'python.exe' -and ($_.CommandLine -match 'nav.train_nav' -or $_.CommandLine -match 'nav.vec_worker')) -or $_.Name -match 'marbleblast' })
foreach ($p in $procs) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 6
Write-Host ("[{0}] stopped {1} loop + {2} training processes" -f (Stamp), $loop.Count, $procs.Count)

# ---- 2. snapshot the checkpoint under test, ONCE, so both maps judge the same weights
if ($Ckpt -eq "") {
    $Ckpt = Join-Path $ml ("models\nav\nav_eval_{0}_{1}.pth" -f $Tag, (Get-Date).ToString("HHmm"))
    Copy-Item (Join-Path $ml "models\nav\nav_latest.pth") $Ckpt
}
Write-Host ("[{0}] checkpoint under test: {1}" -f (Stamp), (Split-Path $Ckpt -Leaf))

# ---- 3. evaluate each map in turn
Remove-Item Env:NAV_WATCH, Env:NAV_VIEW_SUBSTEPS, Env:NAV_MARK, Env:NAV_VIEWYAW, Env:NAV_CKPT, Env:NAV_FORCE_THROTTLE, Env:NAV_SMOOTH, Env:NAV_DIRSKIP -ErrorAction SilentlyContinue
foreach ($map in $Maps) {
    Write-Host ""
    Write-Host ("[{0}] ===== {1}: {2} rounds" -f (Stamp), $map, $Rounds)
    & (Join-Path $ml "real_run.ps1") -Map $map -Rounds $Rounds -Ckpt $Ckpt | Out-Null
    $out = Join-Path $ml ("logs\nav\real_run_{0}.txt" -f $map)
    $pts = @(Select-String -Path $out -Pattern "'points': ([0-9.]+)" -AllMatches | ForEach-Object { $_.Matches } | ForEach-Object { [double]$_.Groups[1].Value })
    $gem = @(Select-String -Path $out -Pattern "'gems': ([0-9]+)" -AllMatches | ForEach-Object { $_.Matches } | ForEach-Object { [int]$_.Groups[1].Value })
    if ($pts.Count -gt 0) {
        $mp = ($pts | Measure-Object -Average).Average
        $mg = if ($gem.Count -gt 0) { ($gem | Measure-Object -Average).Average } else { 0 }
        Write-Host ("[{0}] {1}: points {2} -> mean {3:N1} | gems mean {4:N2}" -f (Stamp), $map, ($pts -join ","), $mp, $mg)
        Add-Content (Join-Path $ml "logs\nav\overnight_notes.txt") ("[{0}] EVAL {1} {2} ({3}): points {4} -> mean {5:N1}, gems mean {6:N2} (gems-only steering, DIR_GOAL_GAIN=30)" -f (Get-Date).ToString("HH:mm"), $Tag, $map, (Split-Path $Ckpt -Leaf), ($pts -join ","), $mp, $mg)
    } else {
        Write-Host ("[{0}] {1}: NO ROUNDS PARSED -- see {2}" -f (Stamp), $map, $out)
    }
}
Write-Host ""
Write-Host ("[{0}] both maps done. Training is still DOWN; restart is the operator's call." -f (Stamp))
