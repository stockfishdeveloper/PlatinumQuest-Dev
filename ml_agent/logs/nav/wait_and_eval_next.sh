#!/bin/bash
# Overnight 2026-09-22/23 (operator: stay on the jump track, goal 150 on KOTM): 8-round KOTM eval every
# STEP updates, training restarts itself after each (eval_cycle.ps1). Tags next1, next2, ...
cd "c:/Users/doug/OneDrive/Documents/GitHub/PlatinumQuest-Dev/ml_agent"
TARGET=18895; STEP=600; N=1
while true; do
  while true; do
    f=$(ls -t logs/nav/train_nav_*.log | head -1)
    upd=$(grep -o "NAV upd=[0-9,]*" "$f" | tail -1 | tr -d ',' | sed 's/NAV upd=//')
    echo "$(date +%H:%M:%S) latest update ${upd:-none} (target $TARGET)"
    if [ -n "$upd" ] && [ "$upd" -ge "$TARGET" ]; then break; fi
    sleep 300
  done
  echo "$(date +%H:%M:%S) reached $upd, running eval_cycle -Tag next$N -Rounds 8"
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File ./eval_cycle.ps1 -Tag next$N -Rounds 8
  echo "$(date +%H:%M:%S) eval cycle jump$N finished"
  cp logs/nav/real_trace.csv "logs/nav/real_trace_next${N}.csv"
  N=$((N+1)); TARGET=$((TARGET+STEP)); sleep 120
done
