#!/bin/bash
# Deterministic Super Speed drill evaluation with one game window (log 40.26): bash ss_eval.sh <ckpt> <tag> [port=9971] [n=211]
ML="c:/Users/doug/OneDrive/Documents/GitHub/PlatinumQuest-Dev/ml_agent"
PQ="c:/Users/doug/OneDrive/Documents/GitHub/PlatinumQuest-Dev/Marble Blast Platinum"
PY="C:/Users/doug/AppData/Local/Programs/Python/Python39/python.exe"
cd "$ML" || exit 1
CKPT=$1; TAG=$2; PORT=${3:-9971}; shift 3 2>/dev/null; EXTRA="$*"     # e.g. --stage 2 --no-use
CUDA_VISIBLE_DEVICES=-1 "$PY" -u -m nav.ss_drill_eval --ckpt "$CKPT" --port "$PORT" --tag "$TAG" $EXTRA > "logs/nav/ss_drill_eval_$TAG.txt" 2>&1 &
pid=$!
for i in $(seq 1 90); do
  powershell.exe -NoProfile -Command "if (Get-NetTCPConnection -LocalPort $PORT -State Listen -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }" && break
  sleep 1
done
(cd "$PQ" && ./marbleblast_mbx.exe -autotrain KingOfTheMarble_Hunt -aiport "$PORT" > /dev/null 2>&1 &)
wait $pid
powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -match '-aiport $PORT' -and \$_.ProcessId -ne \$PID } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"
grep "DRILL EVAL\|Traceback\|Error" "logs/nav/ss_drill_eval_$TAG.txt" | tail -n 3
