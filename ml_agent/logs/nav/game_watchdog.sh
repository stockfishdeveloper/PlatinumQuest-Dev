#!/bin/bash
# 2026-09-22: relaunch the game instances when training throughput halves (HANDOFF 28.8).
# Symptom: wall_s per update 16 -> 35 with w_game 21 -> 135 ms and no CPU/GPU load; every instance's
# rtf drops to ~2.0. Killing the games (the loop relaunches them, workers reconnect) restores 16 s.
cd "c:/Users/doug/OneDrive/Documents/GitHub/PlatinumQuest-Dev/ml_agent"
slow=0
while true; do
  sleep 120
  f=$(ls -t logs/nav/train_nav_*.log 2>/dev/null | head -1)
  [ -z "$f" ] && continue
  w=$(grep -o "wall_s=[0-9]*" "$f" | tail -1 | cut -d= -f2)
  age=$(( $(date +%s) - $(stat -c %Y "$f") ))
  if [ -n "$w" ] && [ "$w" -ge 36 ] && [ "$age" -lt 300 ]; then slow=$((slow+1)); else slow=0; fi
  echo "$(date +%H:%M:%S) wall_s=${w:-?} slow=$slow"
  if [ "$slow" -ge 3 ]; then
    echo "$(date +%H:%M:%S) throughput halved for 3 readings: relaunching the game instances"
    powershell.exe -NoProfile -Command "Get-Process marbleblast_mbx -ErrorAction SilentlyContinue | Stop-Process -Force"
    slow=0; sleep 240
  fi
done
