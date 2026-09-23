#!/bin/bash
# 2026-09-22 operator instruction: one hour after the 12:4x restart, read KOTM sgem. Above 1.7 -> stop
# training (kill watchdog, loop, trainer, workers, games) and leave it down. Otherwise -> continue.
cd "c:/Users/doug/OneDrive/Documents/GitHub/PlatinumQuest-Dev/ml_agent"
echo "$(date +%H:%M:%S) sleeping 3600 s"
sleep 3600
f=$(ls -t logs/nav/train_nav_*.log | head -1)
sgem=$(grep "MAPS" "$f" | tail -1 | grep -o "KingOfTheMarble_Hunt: [^|]*" | grep -o "sgem=[0-9.]*" | cut -d= -f2)
upd=$(grep -o "NAV upd=[0-9,]*" "$f" | tail -1)
echo "$(date +%H:%M:%S) 1-hour check: KOTM sgem=$sgem at $upd (threshold 1.7)"
if [ -z "$sgem" ] || awk -v v="$sgem" 'BEGIN{exit !(v > 1.7)}'; then
  echo "$(date +%H:%M:%S) DECISION: above 1.7 -> stopping training"
  powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { (\$_.Name -match 'bash' -and \$_.CommandLine -match 'game_watchdog') -or (\$_.Name -match 'powershell' -and \$_.CommandLine -match 'run_game_loop') -or (\$_.Name -eq 'python.exe' -and (\$_.CommandLine -match 'nav.train_nav' -or \$_.CommandLine -match 'nav.vec_worker')) -or \$_.Name -match 'marbleblast' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"
  sleep 5
  echo "$(date +%H:%M:%S) training STOPPED; games left: $(powershell.exe -NoProfile -Command "(Get-Process marbleblast_mbx -ErrorAction SilentlyContinue).Count")"
else
  echo "$(date +%H:%M:%S) DECISION: at or below 1.7 -> training continues"
fi
