#!/bin/bash
# wait until the trainer log shows update >= TARGET, then run the 8-round KOTM eval cycle
cd "c:/Users/doug/OneDrive/Documents/GitHub/PlatinumQuest-Dev/ml_agent"
TARGET=14250
while true; do
  f=$(ls -t logs/nav/train_nav_*.log | head -1)
  upd=$(grep -o "NAV upd=[0-9,]*" "$f" | tail -1 | tr -d ',' | sed 's/NAV upd=//')
  echo "$(date +%H:%M:%S) latest update ${upd:-none} in $f"
  if [ -n "$upd" ] && [ "$upd" -ge "$TARGET" ]; then break; fi
  sleep 300
done
echo "$(date +%H:%M:%S) reached $upd, running eval_cycle -Tag timeprice -Rounds 8"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ./eval_cycle.ps1 -Tag timeprice -Rounds 8
echo "$(date +%H:%M:%S) eval cycle finished"
