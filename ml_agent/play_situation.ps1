# Play a captured situation yourself (2026-10-29): stops the 1x play (the planner server on port 8888, its search workers
# and the game), starts record_situation.py on the situation and the game straight into its map with the normal camera
# (you steer). In the game: during Ready/Set the marble waits where the planner's was, with the same gems and powerup;
# at GO it moves off with the same speed and spin, and you play on. Every new round sets the same moment up again.
# The situation file comes from: python ml_agent/tools/situation.py capture   (newest real-time journal, its last decision
# or --seconds-back S). Without -Situation the newest file in ml_agent\situations is used.
# Output: ml_agent\demos\demo_<stamp>.npz + .situation.json; recorder log logs\realtime\record_situation.txt.
# Stop: close the game window, then stop the recorder (its process id is printed below).
param([string]$Situation = "")
$ml = $PSScriptRoot
$repo = [System.IO.Path]::GetFullPath((Join-Path $ml ".."))
$pq = Join-Path $repo "Marble Blast Platinum"
$py = "C:\Users\doug\AppData\Local\Programs\Python\Python39\python.exe"
$logs = Join-Path $ml "logs\realtime"; New-Item -ItemType Directory -Force $logs | Out-Null
if (-not $Situation) {
    $newest = Get-ChildItem (Join-Path $ml "situations") -Filter *.json -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -Last 1
    if (-not $newest) { Write-Host "no situation file in $ml\situations (capture one with tools\situation.py capture)"; exit 1 }
    $Situation = $newest.FullName
}
$sit = Get-Content $Situation -Raw | ConvertFrom-Json
# stop the 1x play: whatever listens on port 8888 (the planner server or a recorder) with its child processes, and the game
$listen = Get-NetTCPConnection -LocalPort 8888 -State Listen -ErrorAction SilentlyContinue
foreach ($c in @($listen)) {
    if ($c -and $c.OwningProcess) {
        Get-CimInstance Win32_Process -Filter ("ParentProcessId=" + $c.OwningProcess) | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
        Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue
    }
}
Get-Process marbleblast_realtime -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
for ($i = 0; $i -lt 20 -and (Get-NetTCPConnection -LocalPort 8888 -State Listen -ErrorAction SilentlyContinue); $i++) { Start-Sleep -Milliseconds 500 }
$env:PYTHONPATH = $ml
$out = Join-Path $logs "record_situation.txt"
$rec = Start-Process -FilePath $py -ArgumentList @("-u", "record_situation.py", $Situation) -WorkingDirectory $ml `
    -RedirectStandardOutput $out -RedirectStandardError (Join-Path $logs "record_situation.err") -PassThru -WindowStyle Hidden
for ($i = 0; $i -lt 60; $i++) {
    if (Get-NetTCPConnection -LocalPort 8888 -State Listen -ErrorAction SilentlyContinue) { break }
    Start-Sleep -Seconds 1
}
if (-not (Get-NetTCPConnection -LocalPort 8888 -State Listen -ErrorAction SilentlyContinue)) {
    Write-Host "the recorder did not start listening on port 8888; see $logs\record_situation.err"
    exit 1
}
$game = Start-Process -FilePath (Join-Path $pq "marbleblast_realtime.exe") -ArgumentList @("-offline", "-autotrain", $sit.map) -WorkingDirectory $pq `
    -RedirectStandardOutput (Join-Path $logs "record_situation_game.txt") -RedirectStandardError (Join-Path $logs "record_situation_game.err") -PassThru
Write-Host ("situation {0} decision {1}: recorder pid {2} (log {3}), game pid {4}. At GO the marble moves off as the planner's did: play it your way." -f $sit.map, $sit.decision, $rec.Id, $out, $game.Id)
