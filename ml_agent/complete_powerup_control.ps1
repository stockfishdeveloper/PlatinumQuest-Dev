# After the ordinary control review, lock and screen its fresh development states.
param([Parameter(Mandatory=$true)][int]$ReviewPID,
      [Parameter(Mandatory=$true)][ValidatePattern('^[A-Za-z0-9_]+$')][string]$RunTag)
$ErrorActionPreference = 'Stop'
$ml = $PSScriptRoot
$py = 'C:\Users\doug\AppData\Local\Programs\Python\Python39\python.exe'
$run = Join-Path $ml "logs/nav/$RunTag"
$handoff = Join-Path $ml 'docs/HANDOFF_POWERUP_LOOP_2026-10-10.md'
try {
    if (Get-Process -Id $ReviewPID -ErrorAction SilentlyContinue) { Wait-Process -Id $ReviewPID -Timeout 3300 }
    $comparison = Get-Content -LiteralPath (Join-Path $run 'review/comparison.json') -Raw | ConvertFrom-Json
    if (-not $comparison.control_comparison.same_source -or -not $comparison.control_comparison.same_final_update) {
        throw 'Control comparison is not matched; inspect before continuing.'
    }
    $deadline = (Get-Date).AddSeconds(30)
    while (@(Get-Process marbleblast_mbx -ErrorAction SilentlyContinue).Count -and (Get-Date) -lt $deadline) {
        Start-Sleep -Milliseconds 500
    }
    if (@(Get-Process marbleblast_mbx -ErrorAction SilentlyContinue).Count) { throw 'Games remain; no screening overlap allowed.' }
    Set-Location -LiteralPath $ml
    $baseline = Join-Path $ml 'logs/nav/powerup_handling_20261010/baseline_A2/escape_practice'
    $next = Join-Path $run 'review/next_curriculum'
    & $py -u -m nav.prepare_powerup_curriculum --dev (Join-Path $baseline 'starts_dev.json') `
        --holdout (Join-Path $baseline 'starts_eval.json') --recordings (Join-Path $run 'held_starts') --out $next
    if ($LASTEXITCODE -ne 0) { throw 'Fresh curriculum preparation failed.' }
    $selection = Get-Content -LiteralPath (Join-Path $next 'selection.json') -Raw | ConvertFrom-Json
    Add-Content -LiteralPath $handoff -Value "`n## Fresh control curriculum locked $((Get-Date).ToString('HH:mm:ss')) MDT`nDevelopment $($selection.dev.n) starts / $($selection.dev.source_rounds) source rounds; holdout $($selection.holdout.n) starts / $($selection.holdout.source_rounds) rounds. Geometry-only selection, holdout outcomes uninspected. Artifacts: logs/nav/$RunTag/review/next_curriculum.`n"
    & $py -u -m nav.powerup_practice_eval --ckpt (Join-Path $ml 'models/nav/real_maps_20261010_A2/nav_latest.pth') `
        --starts (Join-Path $next 'dev_candidates.json') --n-per-map 0 --branches no_use,handling `
        --resources 1,2,5 --port 9234 --out (Join-Path $next 'dev_screen_A2.json')
    if ($LASTEXITCODE -ne 0) { throw 'Fresh development screening process failed.' }
    $screen = Get-Content -LiteralPath (Join-Path $next 'dev_screen_A2.json') -Raw | ConvertFrom-Json
    $failed = @($screen.summary.failed).Count
    Add-Content -LiteralPath $handoff -Value "Fresh development screening completed: $($screen.summary.attempts) windows, $failed failures. Preserved paired outcomes in next_curriculum/dev_screen_A2.json. No policy promotion or additional practice training; inspect useful cases and control retention before choosing the next mixture.`n"
    if ($failed) { throw 'Fresh fixture failures retained; inspect before training.' }
} catch {
    Add-Content -LiteralPath $handoff -Value "`nControl follow-up stopped $((Get-Date).ToString('HH:mm:ss')) MDT: $($_.Exception.Message). No further training launched; inspect retained follow-up logs.`n"
    throw
}
