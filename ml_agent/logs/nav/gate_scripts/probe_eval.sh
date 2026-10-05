#!/bin/bash
# The learned use preference where the approval mask allows a Super Speed kick (log 40.33): runs nav.ss_drill_eval through
# use_probe.py (one game window) and prints "USE PROBE: ... p_learn mean / median / p90 / max | share > 0.5".
# usage: bash probe_eval.sh <ckpt> <tag> [port=9971] [ss_drill_eval flags, e.g. --stage 2 --no-use --n 80]
ML="c:/Users/doug/OneDrive/Documents/GitHub/PlatinumQuest-Dev/ml_agent"
PQ="c:/Users/doug/OneDrive/Documents/GitHub/PlatinumQuest-Dev/Marble Blast Platinum"
PY="C:/Users/doug/AppData/Local/Programs/Python/Python39/python.exe"
cd "$ML" || exit 1
PROBE="$ML/logs/nav/gate_scripts/use_probe.py"; CKPT=$1; TAG=$2; PORT=${3:-9971}; shift 3 2>/dev/null; EXTRA="$*"
CUDA_VISIBLE_DEVICES=-1 "$PY" -u "$PROBE" --ckpt "$CKPT" --port "$PORT" --tag "$TAG" $EXTRA > "logs/nav/ss_drill_eval_$TAG.txt" 2>&1 &
pid=$!
for i in $(seq 1 90); do
  powershell.exe -NoProfile -Command "if (Get-NetTCPConnection -LocalPort $PORT -State Listen -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }" && break
  sleep 1
done
(cd "$PQ" && ./marbleblast_mbx.exe -autotrain KingOfTheMarble_Hunt -aiport "$PORT" > /dev/null 2>&1 &)
wait $pid
powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -match '-aiport $PORT' -and \$_.ProcessId -ne \$PID } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"
grep "USE PROBE\|DRILL EVAL\|Traceback\|Error" "logs/nav/ss_drill_eval_$TAG.txt" | tail -n 3
