#!/bin/bash
# Light gate beside the trainer: nav/real_run.py on the CPU (no CUDA memory), whole-spawn chooser, one game per arm.
# usage: rr_gate.sh <rounds> <port> <ckpt> <tag> [<port> <ckpt> <tag> ...]
# Results: logs/nav/real_run_KingOfTheMarble_Hunt_<ckpt>_<tag>.json, stdout logs/learned_nav/rr_<tag>.txt
ML="c:/Users/doug/OneDrive/Documents/GitHub/PlatinumQuest-Dev/ml_agent"
PQ="c:/Users/doug/OneDrive/Documents/GitHub/PlatinumQuest-Dev/Marble Blast Platinum"
PY="C:/Users/doug/AppData/Local/Programs/Python/Python39/python.exe"
cd "$ML" || exit 1
R=$1; shift
pids=(); ports=()
while [ $# -ge 3 ]; do
  port=$1; ckpt=$2; tag=$3; shift 3
  CUDA_VISIBLE_DEVICES=-1 NAV_ROUNDS=$R NAV_PORT=$port NAV_CKPT=$ckpt NAV_TAG=$tag NAV_TOUR=walk \
    NAV_TRACE="logs/nav/real_trace_$tag.csv" "$PY" -u -m nav.real_run > "logs/learned_nav/rr_$tag.txt" 2>&1 &
  pids+=($!); ports+=($port)
  for i in $(seq 1 90); do
    powershell.exe -NoProfile -Command "if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }" && break
    sleep 1
  done
  (cd "$PQ" && ./marbleblast_mbx.exe -autotrain KingOfTheMarble_Hunt -aiport $port > /dev/null 2>&1 &)
  echo "$(date +%H:%M:%S) $tag: python $! port $port ckpt $ckpt"
  sleep 6
done
for p in "${pids[@]}"; do wait $p; done
for port in "${ports[@]}"; do
  powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -match '-aiport $port' -and \$_.ProcessId -ne \$PID } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"
done
echo "$(date +%H:%M:%S) done"
