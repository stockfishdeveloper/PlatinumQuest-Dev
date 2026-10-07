# Keep this specific points-training run alive. Resumes nav_latest after a crash/hang.
# Put a STOP file in logs/nav/<RunTag>/ to stop this supervisor and its owned processes.
param([string]$RunTag = 'points_20261005')
$ErrorActionPreference = 'Stop'
$ml = $PSScriptRoot
$py = 'C:\Users\doug\AppData\Local\Programs\Python\Python39\python.exe'
$runDir = Join-Path $ml "logs/nav/$RunTag"
New-Item -ItemType Directory -Path $runDir -Force | Out-Null
. (Join-Path $ml 'nav_ready.ps1')
if (Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match '-m nav\.train_nav(?:\s|$)' }) {
    throw 'A trainer already exists; refusing to stack another run.'
}
$env:NAV_CKPT = Join-Path $ml 'models/nav/nav_latest.pth'
$env:NAV_CHECKPOINT_PREFIX = "nav_$RunTag"
$env:NAV_INSTANCES = '8'
$env:NAV_DRILL_PLAN = 'none'   # 2026-10-05 22:15 (log 40.58): all eight games play full rounds while the navigator relearns the
                               # corrected map ('none' matches no worker; an empty value would DELETE the variable in PowerShell
                               # and the worker default 0:4,1:4,2:4,3:4 would apply). Before: '0:4,1:4,2:4,3:4'
$env:NAV_DRILL4_STARTS = Join-Path $ml 'datasets/ss_drill/starts_replay_dev.json'
$env:NAV_TOUR = 'walk'
$env:NAV_REAL_GEMS = '1'
$env:NAV_LIVE_RECORD = '1'
$env:NAV_CRITIC_WARMUP_UPDATES = '20'
$env:NAV_DRILL_EVAL = '0'
$env:NAV_DRILL_NO_USE = '0'
$env:NAV_NO_SAVE = '0'
$env:NAV_TRAIN_WATCH = '0'
$env:NAV_OBS_MS = '64'
$env:NAV_SPEED = '3'
$env:NAV_RENDER_EVERY = '100'
$stdout = Join-Path $ml 'logs/nav/stdout.txt'
$stderr = Join-Path $ml 'logs/nav/stderr.txt'
$stopFile = Join-Path $runDir 'STOP'
$trainer = $null; $gameLoop = $null

function Write-RunLog([string]$message) {
    $line = "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] $message"
    Write-Host $line
    Add-Content -LiteralPath (Join-Path $runDir 'supervisor.log') -Value $line
}

function Stop-OwnedProcesses {
    if ($gameLoop -and -not $gameLoop.HasExited) { Stop-Process -Id $gameLoop.Id -ErrorAction SilentlyContinue }
    # The loop owns only these eight training ports. Evaluation/collection games are excluded.
    $children = @(Get-CimInstance Win32_Process | Where-Object {
        ($_.Name -eq 'marbleblast_mbx.exe' -and $_.CommandLine -match '-aiport (888[89]|889[0-5])(?:\s|$)') -or
        ($trainer -and $_.Name -eq 'python.exe' -and $_.ParentProcessId -eq $trainer.Id -and $_.CommandLine -match 'nav\.vec_worker')
    })
    foreach ($child in $children) { Stop-Process -Id $child.ProcessId -ErrorAction SilentlyContinue }
    if ($trainer -and -not $trainer.HasExited) { Stop-Process -Id $trainer.Id -ErrorAction SilentlyContinue }
}

$attempt = 0
try {
    while (-not (Test-Path -LiteralPath $stopFile)) {
        $attempt++
        $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
        foreach ($file in @($stdout, $stderr)) {
            if (Test-Path -LiteralPath $file) {
                Copy-Item -LiteralPath $file -Destination (Join-Path $runDir ("{0}_{1}" -f $stamp, (Split-Path $file -Leaf)))
            }
        }
        Write-RunLog "Attempt ${attempt}: resuming nav_latest, workers by NAV_DRILL_PLAN=$env:NAV_DRILL_PLAN."
        $trainer = Start-Process $py -ArgumentList @('-u','-m','nav.train_nav') -WorkingDirectory $ml -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
        if (-not (Wait-NavPorts -Port0 8888 -Count 8 -TimeoutSec 120)) {
            Write-RunLog 'Worker startup failed; preserving logs and retrying the same run.'
            Stop-OwnedProcesses
            Start-Sleep -Seconds 30
            continue
        }
        $gameLoop = Start-Process 'powershell.exe' -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',('"' + (Join-Path $ml 'run_game_loop.ps1') + '"'),'-Mission','KingOfTheMarble_Hunt','-Split','8','-Instances','8') -WorkingDirectory $ml -RedirectStandardOutput (Join-Path $runDir "games_$stamp.out") -RedirectStandardError (Join-Path $runDir "games_$stamp.err") -WindowStyle Hidden -PassThru
        @{supervisor=$PID; trainer=$trainer.Id; game_loop=$gameLoop.Id; attempt=$attempt; started=$stamp; checkpoint=$env:NAV_CKPT} |
            ConvertTo-Json | Set-Content (Join-Path $runDir 'pids.json')
        Write-RunLog "Trainer $($trainer.Id), game loop $($gameLoop.Id)."
        while (-not $trainer.HasExited -and -not (Test-Path -LiteralPath $stopFile)) {
            Start-Sleep -Seconds 30
            $age = ((Get-Date) - (Get-Item -LiteralPath $stdout).LastWriteTime).TotalSeconds
            if ($gameLoop.HasExited -or $age -gt 600) {
                Write-RunLog "Run stalled (stdout age $([int]$age)s, game loop exited=$($gameLoop.HasExited)); recovering from latest checkpoint."
                break
            }
        }
        Stop-OwnedProcesses
        if (-not (Test-Path -LiteralPath $stopFile)) {
            Write-RunLog 'Trainer exited or hung; restarting the same experiment in 15 seconds.'
            Start-Sleep -Seconds 15
        }
    }
} finally {
    Stop-OwnedProcesses
    Write-RunLog 'Supervisor ended. Owned training processes closed; latest saved checkpoint retained.'
}
