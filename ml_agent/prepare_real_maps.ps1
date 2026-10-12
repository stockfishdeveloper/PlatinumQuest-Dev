# Preparation is the default. Actual training requires a reviewed config and -StartTraining.
param([ValidateSet('base','clusters','clusters_reset','A2','B2')][string]$Checkpoint = 'clusters_reset', [switch]$StartTraining, [double]$PotentialK = -1, [string]$DrillPlan = 'none', [string]$DrillStarts = '',
      [ValidateSet('legacy','discounted')][string]$ShapingMode = 'legacy',
      [string]$RunTag = 'real_maps_20261009', [double]$MaxWallMinutes = 0,
      [ValidateSet('handling','judgment')][string]$PracticeMode = 'judgment', [string]$PracticeMission = '', [string]$SourceCheckpoint = '',
      [ValidateRange(0,2147483647)][int]$StopUpdate = 0,
      [ValidateRange(1,2147483647)][int]$Seed = 1)
$ErrorActionPreference = 'Stop'
if ([double]::IsNaN($PotentialK) -or [double]::IsInfinity($PotentialK)) { throw 'PotentialK must be finite.' }
$ml = $PSScriptRoot
$configPath = Join-Path $ml 'configs/real_maps_20261009.json'
$config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
$blockMinutes = if ($MaxWallMinutes -gt 0) { $MaxWallMinutes } else { $config.initial_training_minutes }
if ([double]::IsNaN($blockMinutes) -or [double]::IsInfinity($blockMinutes) -or $blockMinutes -le 0) { throw 'A finite positive training duration is required.' }
if ($PracticeMission -and ($DrillPlan -ne '0:4' -or -not $DrillStarts)) { throw 'A pinned practice mission requires one stage-4 worker (0:4) and an explicit fixture pool.' }
$source = Join-Path $ml $config.checkpoints.$Checkpoint
if ($SourceCheckpoint) { $source = [System.IO.Path]::GetFullPath($SourceCheckpoint) }
$runDir = Join-Path $ml "logs/nav/$RunTag"
$outputDir = Join-Path $ml "models/nav/$RunTag"
$stopFile = Join-Path $runDir 'STOP'
if (-not (Test-Path -LiteralPath $source)) { throw "Missing checkpoint: $source" }
if ($config.instances -ne 8 -or $config.initial_training_minutes -le 0) { throw 'Expected eight workers and a finite training block.' }
if ($StartTraining -and -not $config.training_approved) { throw "Training remains unapproved in $configPath. Review the handoff before changing training_approved." }
New-Item -ItemType Directory -Path $runDir -Force | Out-Null
New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
$profile = $config.profiles.($config.training_profile)
$settings = @{
    NAV_OUTPUT_DIR=$outputDir; NAV_DIRECT_GEM='0'; NAV_ESCAPE_POWERUPS='1';
    NAV_SHAPING_MODE=$ShapingMode; NAV_PRACTICE_MODE=$PracticeMode; NAV_PRACTICE_MISSION=$PracticeMission;
    NAV_USE_EPS='0.02';
    NAV_STOP_UPDATE="$StopUpdate";
    NAV_SEED="$Seed";
    NAV_PRACTICE_WORKER='0'; NAV_PRACTICE_RECORD_DIR=(Join-Path $runDir 'held_starts');
    NAV_PRACTICE_CAPTURE='held'; NAV_HELD_CAPTURE_MS='8000'; NAV_HELD_CAPTURE_LIMIT='200';
    NAV_COMPETITIVE="$([int][bool]$profile.competitive)"; NAV_RESCUE_S="$($profile.rescue_s)";
    NAV_JUMP_FORCE_P='0.03'; NAV_JUMP_EXPLORE_P='0.01'; NAV_JUMP_EXPLORE_BONUS='0';
    NAV_POTENTIAL_K=$(if ($PotentialK -ge 0) { "$PotentialK" } else { '0.05' }); NAV_USE_DAMP='3'; NAV_FREEZE_USE_HEAD='0';   # 2026-10-11 12:45 (operator "do it", log 40.97): use damping 3 replaces the frozen use head (40.96); was NAV_USE_DAMP 0   # 2026-10-09 21:40: potential shaping (recovery.potential_term), -PotentialK 0 = points-only control block   # operator 2026-10-09 16:30: forced stall samples stay at 3 % (with the 3 u clearance of part 4); evaluations run them at 0
    NAV_JUMP_DET_THRESHOLD='0.5'; NAV_JUMP_ON_PRIOR='0'; NAV_STUCK_S='0';
    NAV_DRILL_PLAN=$DrillPlan; NAV_ROTATION_STATE_DIR=(Join-Path (Join-Path (Join-Path $env:TEMP 'pq_checks') $RunTag) 'rotation')
}
if ($DrillStarts) {
    $settings.NAV_DRILL4_STARTS = [System.IO.Path]::GetFullPath($DrillStarts)
    if (-not (Test-Path -LiteralPath $settings.NAV_DRILL4_STARTS)) { throw 'Practice fixture file is missing.' }
    if ($PracticeMission) {
        $pool = @(Get-Content -LiteralPath $settings.NAV_DRILL4_STARTS -Raw | ConvertFrom-Json)
        if (@($pool | Where-Object { $_.mission -eq $PracticeMission }).Count -eq 0) { throw 'Practice pool contains no fixtures for the pinned mission.' }
    }
}
@{ checkpoint=$source; source_sha256=(Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash;
   practice_starts_sha256=$(if ($DrillStarts) { (Get-FileHash -LiteralPath $settings.NAV_DRILL4_STARTS -Algorithm SHA256).Hash } else { $null });
   maps=$config.maps; environment=$settings; max_wall_minutes=$blockMinutes;
   rounds_per_rotation=$config.rounds_per_rotation; training_started=[bool]$StartTraining } |
    ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $runDir 'prepared.json')
if (-not $StartTraining) {
    New-Item -ItemType File -Path $stopFile -Force | Out-Null
    Write-Host "Prepared $runDir. STOP is present. No trainer or game was started."
    return
}
# Snapshot the reviewed starting weights. All later saves/resumes stay in this run's directory.
$startingSnapshot = Join-Path $outputDir 'starting_checkpoint.pth'
if (-not (Test-Path -LiteralPath $startingSnapshot)) { Copy-Item -LiteralPath $source -Destination $startingSnapshot }
if ((Get-FileHash -LiteralPath $startingSnapshot).Hash -ne (Get-FileHash -LiteralPath $source).Hash) { throw 'Starting checkpoint differs; choose a fresh RunTag for a different source.' }
foreach ($name in $settings.Keys) { Set-Item -LiteralPath "Env:$name" -Value $settings[$name] }
Remove-Item -LiteralPath $stopFile -ErrorAction SilentlyContinue
& (Join-Path $ml 'supervise_points_training.ps1') -RunTag $RunTag -Missions ($config.maps -join ',') -Split '' `
    -Checkpoint $startingSnapshot -RotateEveryRounds $config.rounds_per_rotation -MaxWallMinutes $blockMinutes -DrillPlan $DrillPlan -DrillStarts $DrillStarts
