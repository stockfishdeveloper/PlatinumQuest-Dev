#!/bin/bash
# Every 15 min: KOTM training-round score summary from the GAME lines (HANDOFF 28.25), plus crashes.
# Overnight 2026-09-23: also snapshots nav_latest.pth whenever the last-50-round mean sets a new high
# (models/nav/nav_night_<upd>_<mean>.pth, state in logs/nav/night_best.json) so the morning has candidates.
cd "c:/Users/doug/OneDrive/Documents/GitHub/PlatinumQuest-Dev/ml_agent"
while true; do
  f=$(ls -t logs/nav/train_nav_*.log | head -1)
  grep -E "Traceback|Error|assert" "$f" | tail -2
  alive=$(powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -match 'nav\.train_nav' }).Count" | tr -d '\r')
  games=$(powershell.exe -NoProfile -Command "@(Get-Process marbleblast_mbx -ErrorAction SilentlyContinue).Count" | tr -d '\r')
  python - "$f" "$alive" "$games" <<'PY'
import re, sys, json, os, shutil
f, alive, games = sys.argv[1], sys.argv[2], sys.argv[3]
rx = re.compile(r'\[(\d\d:\d\d:\d\d)\] \[(\d+)\] GAME map=(\S+) points=([\d.]+) gems=(\d+) falls=(\d+) rtf=([\d.]+)')
txt = open(f, encoding='utf-8', errors='replace').read()
g = [m.groups() for m in rx.finditer(txt)]
k = [x for x in g if x[2].startswith('King') and int(x[4]) <= 130]   # drop merged double rounds
upd = re.findall(r'NAV upd=(\d+)', txt)
def s(rows):
    p = [float(r[3]) for r in rows]; fl = [int(r[5]) for r in rows]
    return f'n={len(p)} mean={sum(p)/len(p):.1f} max={max(p):.0f} min={min(p):.0f} falls/rd={sum(fl)/len(fl):.1f}' if p else 'n=0'
u = upd[-1] if upd else '?'
print(f'upd {u} trainer={alive} games={games} KOTM all: {s(k)} | last 14: {s(k[-14:])} | last 50: {s(k[-50:])} | last time {k[-1][0] if k else "-"}')
if len(k) >= 50:
    m50 = sum(float(r[3]) for r in k[-50:]) / 50
    st = {}
    try:
        st = json.load(open('logs/nav/night_best.json'))
    except Exception:
        pass
    if m50 > st.get('best50', 0.0) + 0.5:
        dst = f'models/nav/nav_night_{u}_{m50:.0f}.pth'
        shutil.copy('models/nav/nav_latest.pth', dst)
        st = {'best50': m50, 'update': u, 'file': dst, 'falls50': sum(int(r[5]) for r in k[-50:]) / 50}
        json.dump(st, open('logs/nav/night_best.json', 'w'))
        print(f'NEW 50-ROUND HIGH {m50:.1f} at upd {u}: snapshot {dst}')
PY
  sleep 900
done
