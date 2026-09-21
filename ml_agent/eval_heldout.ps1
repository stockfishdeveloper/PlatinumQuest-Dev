# eval_heldout.ps1 -- run the deterministic navigator evaluation across several maps.
#
# For each map: launch ONE built-engine instance on a spare port, wait for it to reach its round,
# run `python -m nav.eval_nav` against it (60 seeded segments, greedy policy, no sampling), then
# close it. Results go to logs/nav/eval_<map>_<ckpt>.json and a summary is printed at the end.
#
#   .\eval_heldout.ps1                                   # the three maps the policy has never trained on
#   .\eval_heldout.ps1 -Maps "KingOfTheMarble_Hunt"       # anything else
#   .\eval_heldout.ps1 -Ckpt models\nav\nav_003300.pth    # a specific checkpoint
#
# IMPORTANT: stop training first. The 8 GB GPU has no room for a 9th game instance alongside the
# trainer's PPO update (each instance holds ~450 MiB, the update needs ~1.9 GB), and eval on a
# contended GPU is slow and not comparable. Typical cost: ~3 min per map.

param(
    [string[]]$Maps = @("FlatWithJump_Hunt", "JumpOnly_Hunt", "FlatGemTraining_Hunt"),
    [string]$Ckpt = "",                                  # default: models/nav/nav_latest.pth
    [int]$Port = 8920,                                   # clear of the training ports 8888..8895
    [string]$Exe = "marbleblast_mbx.exe"
)
$ml = $PSScriptRoot
$pq = [System.IO.Path]::GetFullPath((Join-Path $ml "..\Marble Blast Platinum"))
$exePath = Join-Path $pq $Exe
if (-not (Test-Path $exePath)) { Write-Error "not found: $exePath"; exit 1 }
$py = "C:\Users\doug\AppData\Local\Programs\Python\Python39\python.exe"
. (Join-Path $ml "nav_ready.ps1")        # Wait-NavPort: ready-at-GO helpers

$results = @()
foreach ($map in $Maps) {
    Write-Host ("[{0}] === {1}" -f (Get-Date).ToString("HH:mm:ss"), $map)
    $npz = Join-Path $ml "terrain_maps\terrain_$map.npz"
    if (-not (Test-Path $npz)) { Write-Host "  no terrain map ($npz); run: python generate_terrain_map.py $map"; continue }

    # READY AT GO (user rule, 2026-09-21): Python binds first, we wait for the port to be
    # listening, and only then does the game launch -- so the bridge is answered on its first
    # dial at GO instead of the marble idling while Python starts up.
    $env:NAV_PORT = "$Port"
    $env:NAV_MAP = $map                    # pre-build the terrain grid off the critical path
    if ($Ckpt -ne "") { $env:NAV_CKPT = (Resolve-Path $Ckpt).Path }
    $out = Join-Path $ml "logs\nav\eval_run_$map.txt"
    $p = Start-Process -FilePath $py -ArgumentList @("-m", "nav.eval_nav") -WorkingDirectory $ml -RedirectStandardOutput $out -RedirectStandardError "$out.err" -PassThru -WindowStyle Hidden
    if (-not (Wait-NavPort -Port $Port -TimeoutSec 120)) {
        Write-Host "  evaluator never opened port $Port; skipping $map"
        if (-not $p.HasExited) { $p.Kill() }
        $results += [PSCustomObject]@{ Map = $map; Result = "FAILED (port)" }
        continue
    }
    $g = Start-Process -FilePath $exePath -ArgumentList @("-autotrain", $map, "-aiport", "$Port") -WorkingDirectory $pq -PassThru
    Write-Host "  game pid $($g.Id) on port $Port; it will be answered the instant it reaches GO"
    if (-not $p.WaitForExit(600000)) { Write-Host "  eval timed out after 10 min"; $p.Kill() }

    $line = Get-Content $out -ErrorAction SilentlyContinue | Where-Object { $_ -match "arrive" } | Select-Object -Last 1
    if ($line) { Write-Host "  $line"; $results += [PSCustomObject]@{ Map = $map; Result = $line.Trim() } }
    else {
        Write-Host "  no result line; stderr tail:"
        Get-Content "$out.err" -ErrorAction SilentlyContinue | Select-Object -Last 4 | ForEach-Object { "    $_" }
        $results += [PSCustomObject]@{ Map = $map; Result = "FAILED" }
    }
    if (-not $g.HasExited) { Stop-Process -Id $g.Id -Force }
    Start-Sleep -Seconds 5
}

Write-Host ""
Write-Host "=== held-out summary ==="
foreach ($r in $results) { "{0,-24} {1}" -f $r.Map, $r.Result }
