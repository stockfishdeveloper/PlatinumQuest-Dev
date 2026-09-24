#!/bin/bash
# Every 15 min: KOTM training-round score summary from the GAME lines (HANDOFF 28.25), plus crashes.
cd "c:/Users/doug/OneDrive/Documents/GitHub/PlatinumQuest-Dev/ml_agent"
while true; do
  f=$(ls -t logs/nav/train_nav_*.log | head -1)
  grep -E "Traceback|Error|assert" "$f" | tail -2
  python - "$f" <<'PY'
import re, sys
f = sys.argv[1]
rx = re.compile(r'\[(\d\d:\d\d:\d\d)\] \[(\d+)\] GAME map=(\S+) points=([\d.]+) gems=(\d+) falls=(\d+) rtf=([\d.]+)')
g = [m.groups() for m in (rx.search(l) for l in open(f, encoding='utf-8', errors='replace')) if m]
k = [x for x in g if x[2].startswith('King') and int(x[4]) <= 130]   # drop merged double rounds
upd = re.findall(r'NAV upd=(\d+)', open(f, encoding='utf-8', errors='replace').read())
def s(rows):
    p = [float(r[3]) for r in rows]; fl = [int(r[5]) for r in rows]
    return f'n={len(p)} mean={sum(p)/len(p):.1f} max={max(p):.0f} min={min(p):.0f} falls/rd={sum(fl)/len(fl):.1f}' if p else 'n=0'
print(f'upd {upd[-1] if upd else "?"} KOTM all: {s(k)} | last 14: {s(k[-14:])} | last time {k[-1][0] if k else "-"}')
PY
  sleep 900
done
