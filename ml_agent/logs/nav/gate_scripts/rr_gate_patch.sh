#!/bin/bash
# rr_gate.sh with inference-time patches (log 40.33-40.38): runs rr_patch.py instead of nav.real_run. Env: PATCH_RES=off
# (residual zeroed) | post (residual only after a kick); PATCH_LAM=<u/s> (Super Speed cap); PATCH_LEVEL=<dz> (the level
# floor check, now built into nav/obs.py as SS_LEVEL_DZ). Other env (NAV_NO_USE, NAV_FORCE_USE) passes through.
# usage: rr_gate_patch.sh <rounds> <port> <ckpt> <tag> [<port> <ckpt> <tag> ...]
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
    NAV_TRACE="logs/nav/real_trace_$tag.csv" "$PY" -u "$ML/logs/nav/gate_scripts/rr_patch.py" > "logs/learned_nav/rr_$tag.txt" 2>&1 &
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
