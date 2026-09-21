# eval_cycle.ps1 -- stop training, evaluate a snapshot over N real KOTM rounds, restart training.
#
# The 8 GB GPU cannot hold the trainer's PPO update beside a 9th game instance, so a real-round
# evaluation means stopping the game loop (it would relaunch anything killed), the trainer, its
# workers and the 8 games; running nav/real_run.py; then bringing the loop and the trainer back.
#
#   .\eval_cycle.ps1 -Tag edge                    # snapshot nav_latest -> nav_eval_edge_<HHMM>.pth, 8 rounds
#   .\eval_cycle.ps1 -Tag edge -Rounds 8 -NoRestart
#
# Real-run steering is GEMS-ONLY (nav/real_run.py DIRECT_GEM default on, 2026-09-20 23:50).
param(
    [string]$Tag = "eval",
    [int]$Rounds = 8,
    [string]$Ckpt = "",                                   # default: snapshot models\nav\nav_latest.pth now
    [string]$Split = "4,4",
    [string]$Missions = "FlatIslands_Hunt,KingOfTheMarble_Hunt",
    [switch]$NoRestart
)
$ml = $PSScriptRoot
$py = "C:\Users\doug\AppData\Local\Programs\Python\Python39\python.exe"
function Stamp { (Get-Date).ToString("HH:mm:ss") }

# ---- 1. stop everything that trains (loop first, or it relaunches the games)
$loop = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'powershell' -and $_.CommandLine -match 'run_game_loop' })
foreach ($p in $loop) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }
$procs = @(Get-CimInstance Win32_Process | Where-Object {
    ($_.Name -eq 'python.exe' -and ($_.CommandLine -match 'nav.train_nav' -or $_.CommandLine -match 'nav.vec_worker')) -or $_.Name -match 'marbleblast' })
foreach ($p in $procs) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 5
Write-Host ("[{0}] stopped {1} loop + {2} training processes" -f (Stamp), $loop.Count, $procs.Count)

# ---- 2. snapshot the checkpoint under test
if ($Ckpt -eq "") {
    $Ckpt = Join-Path $ml ("models\nav\nav_eval_{0}_{1}.pth" -f $Tag, (Get-Date).ToString("HHmm"))
    Copy-Item (Join-Path $ml "models\nav\nav_latest.pth") $Ckpt
    Write-Host ("[{0}] snapshot {1}" -f (Stamp), $Ckpt)
}

# ---- 3. evaluate (real_run.ps1 launches one game on port 8920 and closes it afterwards)
Remove-Item Env:NAV_WATCH, Env:NAV_VIEW_SUBSTEPS, Env:NAV_MARK, Env:NAV_VIEWYAW, Env:NAV_CKPT -ErrorAction SilentlyContinue
& (Join-Path $ml "real_run.ps1") -Rounds $Rounds -Ckpt $Ckpt
$out = Join-Path $ml "logs\nav\real_run_KingOfTheMarble_Hunt.txt"
$pts = @(Select-String -Path $out -Pattern "'points': ([0-9.]+)" -AllMatches | ForEach-Object { $_.Matches } | ForEach-Object { [double]$_.Groups[1].Value })
if ($pts.Count -gt 0) {
    $mean = ($pts | Measure-Object -Average).Average
    Write-Host ("[{0}] EVAL {1}: rounds {2} -> mean {3:N1}" -f (Stamp), $Tag, ($pts -join ","), $mean)
    Add-Content (Join-Path $ml "logs\nav\overnight_notes.txt") ("[{0}] EVAL {1} ({2}): {3} -> mean {4:N1} (gems-only steering)" -f (Get-Date).ToString("HH:mm"), $Tag, (Split-Path $Ckpt -Leaf), ($pts -join ","), $mean)
}

# ---- 4. restart the loop and the trainer
if (-not $NoRestart) {
    # READY AT GO (user rule, 2026-09-21): the TRAINER starts first so its 8 worker sockets are
    # bound and listening, and only then does the game loop launch the instances. The old order
    # (loop, blind Start-Sleep 20, trainer) meant each game reached GO with nobody on its port
    # and its marble idled until the workers came up.
    . (Join-Path $ml "nav_ready.ps1")
    Start-Process -FilePath $py -ArgumentList @("-u", "-m", "nav.train_nav") -WorkingDirectory $ml -RedirectStandardOutput (Join-Path $ml "logs\nav\stdout.txt") -RedirectStandardError (Join-Path $ml "logs\nav\stderr.txt") -WindowStyle Hidden | Out-Null
    if (-not (Wait-NavPorts -Port0 8888 -Count 8 -TimeoutSec 180)) {
        Write-Warning "not all worker ports came up; launching the games anyway (they retry)"
    }
    Start-Process -FilePath "powershell.exe" -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $ml "run_game_loop.ps1"), "-Mission", $Missions, "-Split", $Split, "-Instances", "8") -WorkingDirectory $ml -WindowStyle Hidden | Out-Null
    Write-Host ("[{0}] trainer up and listening, then game loop restarted" -f (Stamp))
}
