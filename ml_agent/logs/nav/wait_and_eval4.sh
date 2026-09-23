#!/bin/bash
# align run (HANDOFF 28.11): eval at update 14555 (edgefix1) and 15255 (edgefix2). One process only.
cd "c:/Users/doug/OneDrive/Documents/GitHub/PlatinumQuest-Dev/ml_agent"
for spec in "15575 align1" "16175 align2"; do
  set -- $spec; TARGET=$1; TAG=$2
  while true; do
    f=$(ls -t logs/nav/train_nav_*.log | head -1)
    upd=$(grep -o "NAV upd=[0-9,]*" "$f" | tail -1 | tr -d ',' | sed 's/NAV upd=//')
    echo "$(date +%H:%M:%S) latest update ${upd:-none} (target $TARGET)"
    if [ -n "$upd" ] && [ "$upd" -ge "$TARGET" ]; then break; fi
    sleep 300
  done
  echo "$(date +%H:%M:%S) reached $upd, running eval_cycle -Tag $TAG -Rounds 8"
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File ./eval_cycle.ps1 -Tag $TAG -Rounds 8
  echo "$(date +%H:%M:%S) eval cycle $TAG finished"
  sleep 120
done
