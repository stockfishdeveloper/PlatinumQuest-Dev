# Explicit finite 1x review, no trainer. Python warms before the game is launched.
param([string]$Mission = 'Cragmire_Hunt', [ValidateSet('base','clusters','clusters_reset')][string]$Checkpoint = 'clusters_reset',
      [ValidateSet('assisted','competitive')][string]$Profile = 'assisted', [switch]$ShowWindow, [int]$Seed = 0)
$ErrorActionPreference = 'Stop'
$ml = $PSScriptRoot
$config = Get-Content (Join-Path $ml 'configs/real_maps_20261009.json') -Raw | ConvertFrom-Json
if (@(Get-Process marbleblast_mbx -ErrorAction SilentlyContinue).Count) { throw 'Close existing game instances before the review.' }
if ($Mission -notin $config.maps) { throw 'Choose one of the ten prepared maps.' }
$output = Join-Path $env:TEMP ('pq_checks/review_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
New-Item -ItemType Directory -Path $output -Force | Out-Null
$settings = @{
    NAV_CKPT=(Join-Path $ml $config.checkpoints.$Checkpoint); NAV_MAP=$Mission; NAV_PORT='9211';
    NAV_ROUNDS='1'; NAV_TAG="review_$Profile"; NAV_TOUR='walk'; NAV_DIRECT_GEM='0';
    NAV_ESCAPE_POWERUPS='1'; NAV_COMPETITIVE="$([int][bool]$config.profiles.$Profile.competitive)";
    NAV_RESCUE_S="$($config.profiles.$Profile.rescue_s)"; NAV_STUCK_S='0';
    NAV_JUMP_FORCE_P='0'; NAV_JUMP_EXPLORE_P='0'; NAV_JUMP_EXPLORE_BONUS='0';
    NAV_JUMP_ON_PRIOR='0'; NAV_JUMP_DET_THRESHOLD='0.5'; NAV_SAMPLE='0'; NAV_NO_USE='0';
    NAV_FORCE_USE='0'; NAV_WATCH='1'; NAV_VIEW_SUBSTEPS='4'; NAV_SPEED='1'; NAV_NO_SAVE='1';
    NAV_TORCH_THREADS='1'; NAV_ESCAPE_RECORD_DIR=''; NAV_EVAL_OUTPUT_DIR=$output;
    NAV_TRACE=(Join-Path $output 'trace.csv'); CUDA_VISIBLE_DEVICES='-1'
}
foreach ($key in @('NAV_SS_ROUTE','NAV_SS_TOUR','NAV_SS_PREP','NAV_SS_FOLLOW','NAV_THROUGH_GEM','NAV_SS_COMMAND_AIM','NAV_SS_YAW_FIX','NAV_GEO_ORDER')) { $settings[$key] = '0' }
foreach ($key in $settings.Keys) { Set-Item -LiteralPath "Env:$key" -Value $settings[$key] }
$py = 'C:/Users/doug/AppData/Local/Programs/Python/Python39/python.exe'
$out = Join-Path $output 'stdout.txt'; $err = Join-Path $output 'stderr.txt'
$runner = $null; $game = $null
try {
    $runner = Start-Process $py -ArgumentList @('-u','-m','nav.real_run') -WorkingDirectory $ml -WindowStyle Hidden -RedirectStandardOutput $out -RedirectStandardError $err -PassThru
    $deadline = (Get-Date).AddSeconds(120)
    while (-not ((Get-Content -LiteralPath $out -Raw -ErrorAction SilentlyContinue) -match 'warm-up: model ready')) {
        if ($runner.HasExited -or (Get-Date) -gt $deadline) { throw "Runner startup failed. See $err" }
        Start-Sleep -Milliseconds 200
    }
    $exe = Join-Path $ml '../Marble Blast Platinum/marbleblast_mbx.exe'
    $style = if ($ShowWindow) { 'Normal' } else { 'Hidden' }
    # -aifreecam ALWAYS for 1x viewing (operator): the camera moves freely without touching the model.
    # No -aiseed unless asked (operator 2026-10-09 20:40): a fixed seed gave the same spawn and the same first gem
    # group every round; training never seeds, so a review should not either. -Seed N reproduces a baseline round.
    $gameArgs = @('-autotrain',$Mission,'-aiport','9211','-offline','-aifreecam','-aicompetitive',$settings.NAV_COMPETITIVE)
    if ($Seed -gt 0) { $gameArgs += @('-aiseed', "$Seed") }
    $game = Start-Process $exe -WorkingDirectory (Split-Path $exe) -ArgumentList $gameArgs -WindowStyle $style -PassThru
    Write-Host "1x review running; output $output. One round, no training."
    if (-not $runner.WaitForExit(900000)) { throw 'Review timed out.' }
    if ($runner.ExitCode -ne 0) { throw "Review failed. See $err" }
} finally {
    foreach ($proc in @($game,$runner)) { if ($proc -and -not $proc.HasExited) { Stop-Process -Id $proc.Id -ErrorAction SilentlyContinue } }
    Write-Host "Review artifacts: $output"
}
