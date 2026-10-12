# Finite follow-up: wait for a bounded pilot, freeze its candidate, then evaluate.
param([Parameter(Mandatory=$true)][string]$RunTag,
      [Parameter(Mandatory=$true)][int]$LauncherPID,
      [Parameter(Mandatory=$true)][string]$BaselineDir,
      [Parameter(Mandatory=$true)][string]$HoldoutStarts, [string]$BaselinePractice = '', [string]$CompareRun = '',
      [string]$PracticeResources = '1,0')
$ErrorActionPreference = 'Stop'
$ml = $PSScriptRoot
$baseline = [System.IO.Path]::GetFullPath($BaselineDir)
$holdout = [System.IO.Path]::GetFullPath($HoldoutStarts)
$basePracticePath = if ($BaselinePractice) { [System.IO.Path]::GetFullPath($BaselinePractice) } else { Join-Path (Split-Path $baseline) 'sj_holdout_A2.json' }
$py = 'C:\Users\doug\AppData\Local\Programs\Python\Python39\python.exe'
$runDir = Join-Path $ml "logs/nav/$RunTag"
$modelDir = Join-Path $ml "models/nav/$RunTag"
$review = Join-Path $runDir 'review'
if (Test-Path -LiteralPath $review) { throw 'Review directory already exists; preserve earlier evidence.' }
Write-Output "Waiting for finite pilot launcher $LauncherPID."
if ($LauncherPID -gt 0 -and (Get-Process -Id $LauncherPID -ErrorAction SilentlyContinue)) {
    Wait-Process -Id $LauncherPID -Timeout 2700
}
if (-not (Test-Path -LiteralPath (Join-Path $runDir 'STOP'))) { throw 'Pilot did not finish with STOP present.' }
# Stop-Process can return before Windows has finished closing the game processes.
# Allow bounded cleanup latency; never kill another session's game to run a review.
$cleanupDeadline = (Get-Date).AddSeconds(30)
while (@(Get-Process marbleblast_mbx -ErrorAction SilentlyContinue).Count -and (Get-Date) -lt $cleanupDeadline) {
    Start-Sleep -Milliseconds 500
}
if (@(Get-Process marbleblast_mbx -ErrorAction SilentlyContinue).Count) { throw 'Games still exist after cleanup wait; refusing evaluation overlap.' }
$latest = Join-Path $modelDir 'nav_latest.pth'
if (-not (Test-Path -LiteralPath $latest)) { throw 'No trained checkpoint was saved.' }
$prepared = Get-Content -LiteralPath (Join-Path $runDir 'prepared.json') -Raw | ConvertFrom-Json
if ((Get-FileHash -LiteralPath $prepared.checkpoint).Hash -ne $prepared.source_sha256) { throw 'Protected training source changed.' }
New-Item -ItemType Directory -Path $review | Out-Null
$candidate = Join-Path $modelDir 'review_checkpoint.pth'
if (Test-Path -LiteralPath $candidate) { throw 'Review candidate already exists.' }
Copy-Item -LiteralPath $latest -Destination $candidate
$candidateHash = (Get-FileHash -LiteralPath $candidate).Hash
Set-Location -LiteralPath $ml
& $py -u -m nav.powerup_practice_eval --ckpt $candidate --starts $holdout --resources $PracticeResources --n-per-map 0 --port 9232 --out (Join-Path $review 'practice.json')
if ($LASTEXITCODE -ne 0) { throw 'Practice evaluation process failed.' }
$practice = Get-Content -LiteralPath (Join-Path $review 'practice.json') -Raw | ConvertFrom-Json
if ($practice.summary.failed.Count -gt 0) { throw 'Practice fixtures failed; inspect retained evidence.' }
& $py -u -m nav.phase_r --baseline --rounds 2 --instances 8 --profiles assisted --checkpoint-file $candidate --output (Join-Path $review 'full_maps')
if ($LASTEXITCODE -ne 0) { throw 'Full-map evaluation failed.' }
if ((Get-FileHash -LiteralPath $candidate).Hash -ne $candidateHash) { throw 'Frozen evaluation candidate changed.' }
$base = Get-Content -LiteralPath (Join-Path $baseline 'summary.json') -Raw | ConvertFrom-Json
$current = Get-Content -LiteralPath (Join-Path $review 'full_maps/summary.json') -Raw | ConvertFrom-Json
$basePractice = Get-Content -LiteralPath $basePracticePath -Raw | ConvertFrom-Json
function Mean-Window($data, [string]$branch, [string]$metric) {
    $values = @($data.rows | Where-Object { $_.status -eq 'ok' -and $_.branch -eq $branch } | ForEach-Object { $_.$metric } | Where-Object { $null -ne $_ })
    if ($values.Count) { return ($values | Measure-Object -Average).Average }
    return $null
}
$windows = @{}
foreach ($branch in @('no_use','handling','learned_det','learned_use_sample')) {
    $windows[$branch] = @{
        baseline_points=(Mean-Window $basePractice $branch 'points'); candidate_points=(Mean-Window $practice $branch 'points');
        baseline_falls=(Mean-Window $basePractice $branch 'falls'); candidate_falls=(Mean-Window $practice $branch 'falls');
        candidate_use_probability=(Mean-Window $practice $branch 'learned_use_probability_mean')
    }
}
$baseMean = ($base.results | ForEach-Object { $_.mean.points } | Measure-Object -Average).Average
$currentMean = ($current.results | ForEach-Object { $_.mean.points } | Measure-Object -Average).Average
$baseKotm = @($base.results | Where-Object { $_.map -eq 'KingOfTheMarble_Hunt' })[0].mean.points
$currentKotm = @($current.results | Where-Object { $_.map -eq 'KingOfTheMarble_Hunt' })[0].mean.points
$summary = @{
    run=$RunTag; source_sha256=$prepared.source_sha256; candidate=$candidate; candidate_sha256=$candidateHash;
    update=$current.results[0].update; baseline_mean_points=$baseMean; candidate_mean_points=$currentMean;
    baseline_kotm=$baseKotm; candidate_kotm=$currentKotm; practice=$windows;
    retention_screen_pass=($currentMean -ge .9*$baseMean -and $currentKotm -ge .9*$baseKotm);
    note='Two rounds per map; named practice starts retain source-round groups. Screening estimates, not proof of skill or transfer.'
}
if ($CompareRun) {
    $otherPath = Join-Path $ml "logs/nav/$CompareRun/review/comparison.json"
    $other = Get-Content -LiteralPath $otherPath -Raw | ConvertFrom-Json
    $summary.control_comparison = @{
        other_run=$CompareRun; other_update=$other.update;
        same_source=($prepared.source_sha256 -eq $other.source_sha256);
        same_final_update=($summary.update -eq $other.update);
        mean_points_delta=($currentMean - $other.candidate_mean_points);
        kotm_delta=($currentKotm - $other.candidate_kotm);
        interpretation='One run per arm; equal update/source checks do not establish statistical significance.'
    }
}
$summary | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $review 'comparison.json')
Write-Output ($summary | ConvertTo-Json -Depth 6)
Add-Content -LiteralPath (Join-Path $ml 'docs/HANDOFF_POWERUP_LOOP_2026-10-10.md') -Value "`n## Automatic review completed $((Get-Date).ToString('HH:mm:ss')) MDT`nRun $RunTag, update $($summary.update): ten-map mean $currentMean versus A2 $baseMean; KOTM $currentKotm versus $baseKotm. Details: logs/nav/$RunTag/review/comparison.json. This is a screening result; continuation requires inspecting the retained map/window evidence.`n"
if ($CompareRun) {
    $comparison = $summary.control_comparison
    Add-Content -LiteralPath (Join-Path $ml 'docs/HANDOFF_POWERUP_LOOP_2026-10-10.md') -Value "Compared with ${CompareRun}: same source=$($comparison.same_source), same final update=$($comparison.same_final_update), mean points delta=$($comparison.mean_points_delta), KOTM delta=$($comparison.kotm_delta). No automatic promotion; both learning and retention evidence must be assessed.`n"
}
