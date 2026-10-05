"""Super Speed Stage 1 drill statistics (log 40.26) from the DRILL lines of logs/nav/stdout.txt, by time blocks:
intended gem taken (hit1) and how fast, the gem after it (hit2), falls, timeouts, segment return.
    python logs/nav/gate_scripts/drill_stats.py [stdout.txt] [n_blocks=6]
"""
import re, sys
import numpy as np
path = sys.argv[1] if len(sys.argv) > 1 else 'logs/nav/stdout.txt'
nb = int(sys.argv[2]) if len(sys.argv) > 2 else 6
rx = re.compile(r'^\[(\d\d:\d\d):\d\d\] \[(\d+)\] DRILL v0=([\d.]+) hit1=(\d) t1=([\d.]+) hit2=(\d) t2=([\d.]+) out=(\w+) ret=([-\d.]+)')
rows = [m.groups() for m in (rx.match(l) for l in open(path, encoding='utf-8', errors='replace')) if m]
if not rows:
    print('no DRILL lines'); sys.exit()
print('drills %d (%s - %s)' % (len(rows), rows[0][0], rows[-1][0]))
print('%-11s %5s %6s %8s %6s %8s %6s %7s %6s' % ('time', 'n', 'hit1%', 't1 med', 'hit2%', 't2 med', 'fell%', 'tmout%', 'ret'))
size = max(1, -(-len(rows) // nb))
for i in range(0, len(rows), size):
    b = rows[i:i + size]; n = len(b)
    h1 = [float(r[4]) for r in b if r[3] == '1']; h2 = [float(r[6]) for r in b if r[5] == '1']
    print('%-5s-%-5s %5d %5.0f%% %7.2fs %5.0f%% %7.2fs %5.0f%% %6.0f%% %6.1f' % (
        b[0][0], b[-1][0], n, 100.0 * len(h1) / n, np.median(h1) if h1 else float('nan'), 100.0 * len(h2) / n,
        np.median(h2) if h2 else float('nan'), 100.0 * sum(r[7] == 'fell' for r in b) / n,
        100.0 * sum(r[7] == 'timeout' for r in b) / n, np.mean([float(r[8]) for r in b])))
# fast hits: the intended gem within 1.0 s of the start (the kick's direct pass; the demo reached it in 0.8-2.2 s)
fast = [r for r in rows[-min(len(rows), 400):]]
print('last %d: intended gem within 1.0 s %.0f%%, within 1.5 s %.0f%%' % (
    len(fast), 100.0 * sum(r[3] == '1' and float(r[4]) <= 1.0 for r in fast) / len(fast),
    100.0 * sum(r[3] == '1' and float(r[4]) <= 1.5 for r in fast) / len(fast)))
