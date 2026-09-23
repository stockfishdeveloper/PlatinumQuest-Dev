#!/bin/bash
# 2026-09-23 08:57 operator: keep 1x watch games running on the jump5 checkpoint until told to stop.
cd "c:/Users/doug/OneDrive/Documents/GitHub/PlatinumQuest-Dev/ml_agent"
export NAV_WATCH=1 NAV_VIEW_SUBSTEPS=4
while true; do
  # wait for any running real_run to finish (the 2-round run already in progress)
  while powershell.exe -NoProfile -Command "if (@(Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'python.exe' -and \$_.CommandLine -match 'nav.real_run' }).Count -gt 0) { exit 0 } else { exit 1 }"; do sleep 10; done
  echo "$(date +%H:%M:%S) launching a 200-round 1x watch run"
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File ./real_run.ps1 -Map KingOfTheMarble_Hunt -Rounds 200 -Ckpt models/nav/nav_best_next1_18900.pth
  echo "$(date +%H:%M:%S) watch run exited; relaunching in 10 s"
  sleep 10
done
