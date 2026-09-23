#!/bin/bash
# 2026-09-23 07:45 operator: if the jump5 eval is not serious progress, stop training. Gate: mean >= 125.
cd "c:/Users/doug/OneDrive/Documents/GitHub/PlatinumQuest-Dev/ml_agent"
while true; do
  line=$(grep "EVAL jump5" logs/nav/overnight_notes.txt | tail -1)
  [ -n "$line" ] && break
  sleep 60
done
mean=$(echo "$line" | grep -o "mean [0-9.]*" | cut -d' ' -f2)
echo "$(date +%H:%M:%S) jump5: $line"
if awk -v v="$mean" 'BEGIN{exit !(v >= 125)}'; then
  echo "$(date +%H:%M:%S) GATE: mean $mean >= 125 -> training continues"
else
  echo "$(date +%H:%M:%S) GATE: mean $mean < 125 -> stopping training"
  sleep 90    # let eval_cycle's own restart finish so nothing relaunches after the kill
  powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { (\$_.Name -match 'bash' -and (\$_.CommandLine -match 'game_watchdog\.sh' -or \$_.CommandLine -match 'wait_and_eval_loop\.sh')) -or (\$_.Name -match 'powershell' -and \$_.CommandLine -match 'run_game_loop') -or (\$_.Name -eq 'python.exe' -and (\$_.CommandLine -match 'nav.train_nav' -or \$_.CommandLine -match 'nav.vec_worker')) -or \$_.Name -match 'marbleblast' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"
  sleep 5
  echo "$(date +%H:%M:%S) training STOPPED; games left: $(powershell.exe -NoProfile -Command "(Get-Process marbleblast_mbx -ErrorAction SilentlyContinue).Count")"
fi
