#!/bin/bash
# Many deterministic KOTM rounds of one checkpoint beside the trainer (log 40.24): batches of rr_gate.sh (CPU real_run, two
# games of 4 rounds), until a round reaches STOP_AT points or MAX_BATCHES batches ran. One summary line per batch, plus a
# "ROUND >= 170" line per high round (the loop model's monitor watches for them). Traces of batches whose best round is
# under KEEP_AT are deleted (1.2 MB per 4 rounds).
# usage: bash logs/nav/gate_scripts/many_rounds.sh <ckpt> <tag_prefix> [max_batches=200] [stop_at=176] [keep_at=170]
# env passes through (NAV_NO_USE=1 for checkpoints from before the use head).
ML="c:/Users/doug/OneDrive/Documents/GitHub/PlatinumQuest-Dev/ml_agent"
cd "$ML" || exit 1
CKPT=$1; PFX=$2; MAXB=${3:-200}; STOP=${4:-176}; KEEP=${5:-170}
for b in $(seq 1 "$MAXB"); do
  ta="${PFX}${b}a"; tb="${PFX}${b}b"
  bash logs/nav/gate_scripts/rr_gate.sh 4 9961 "$CKPT" "$ta" 9962 "$CKPT" "$tb" > /dev/null 2>&1
  pts=$(grep -h "  round [0-9]*:" "logs/learned_nav/rr_$ta.txt" "logs/learned_nav/rr_$tb.txt" 2>/dev/null | sed -E "s/.*'points': ([0-9.]+).*/\1/" | tr '\n' ' ')
  best=$(echo "$pts" | tr ' ' '\n' | grep -v '^$' | sort -n | tail -1)
  echo "$(date +%H:%M:%S) batch $b ($ta/$tb): ${pts}best ${best:-none}"
  for t in "$ta" "$tb"; do
    grep -h "  round [0-9]*:" "logs/learned_nav/rr_$t.txt" 2>/dev/null | while read -r line; do
      p=$(echo "$line" | sed -E "s/.*'points': ([0-9.]+).*/\1/")
      if [ "${p%.*}" -ge 170 ]; then echo "$(date +%H:%M:%S) ROUND >= 170: $t $(echo "$line" | cut -c1-160)"; fi
    done
  done
  if [ -z "$best" ] || [ "${best%.*}" -lt "$KEEP" ]; then rm -f "logs/nav/real_trace_$ta.csv" "logs/nav/real_trace_$tb.csv"; fi
  if [ -n "$best" ] && [ "${best%.*}" -ge "$STOP" ]; then echo "$(date +%H:%M:%S) STOP: a round of $best points (>= $STOP)"; break; fi
  sleep 5
done
echo "$(date +%H:%M:%S) many_rounds finished"
