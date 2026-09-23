#!/bin/bash
# 2026-09-22 operator instruction: at 12:00 read KOTM sgem (seconds between pickups, training side).
# If it is still above 1.7 -> stop training, run one 8-round KOTM eval, leave training DOWN.
# Otherwise -> leave training (and the eval waiter / watchdog) running.
cd "c:/Users/doug/OneDrive/Documents/GitHub/PlatinumQuest-Dev/ml_agent"
target=$(date -d "12:00" +%s); now=$(date +%s); wait=$(( target - now ))
echo "$(date +%H:%M:%S) sleeping $wait s until 12:00"
[ "$wait" -gt 0 ] && sleep "$wait"
f=$(ls -t logs/nav/train_nav_*.log | head -1)
sgem=$(grep "MAPS" "$f" | tail -1 | grep -o "KingOfTheMarble_Hunt: [^|]*" | grep -o "sgem=[0-9.]*" | cut -d= -f2)
upd=$(grep -o "NAV upd=[0-9,]*" "$f" | tail -1)
echo "$(date +%H:%M:%S) 12:00 check: KOTM sgem=$sgem at $upd (threshold 1.7)"
if [ -z "$sgem" ] || awk -v v="$sgem" 'BEGIN{exit !(v > 1.7)}'; then
  echo "$(date +%H:%M:%S) DECISION: sgem above 1.7 -> stopping training, killing the waiter and watchdog, running the eval"
  powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { \$_.Name -match 'bash' -and \$_.CommandLine -match 'wait_and_eval3|game_watchdog' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File ./eval_cycle.ps1 -Tag noon -Rounds 8 -NoRestart
  echo "$(date +%H:%M:%S) noon eval finished; training left DOWN"
else
  echo "$(date +%H:%M:%S) DECISION: sgem at or below 1.7 -> training continues untouched"
fi
