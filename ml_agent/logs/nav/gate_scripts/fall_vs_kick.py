# Falls in real_run traces by the time since the last Super Speed kick (speed jump >= 10 u/s in one decision).
# python fall_vs_kick.py <label> trace.csv [...]
import csv, sys
label = sys.argv[1]
b = {'<3s': 0, '3-10s': 0, '>10s': 0, 'no kick before': 0}
rounds = 0; kicks = 0
for f in sys.argv[2:]:
    rows = list(csv.DictReader(open(f)))
    rounds += len(set(r['round'] for r in rows))
    last_kick = None; prev = None; fell_prev = 0
    for r in rows:
        if prev is not None and prev['round'] != r['round']:
            last_kick = None
        if prev is not None and prev['round'] == r['round'] and float(r['speed']) - float(prev['speed']) >= 10.0:
            last_kick = int(r['dec']); kicks += 1
        fell = int(r['fell'])
        if fell and not fell_prev:
            if last_kick is None:
                b['no kick before'] += 1
            else:
                dt = (int(r['dec']) - last_kick) * 0.064
                b['<3s' if dt < 3 else ('3-10s' if dt < 10 else '>10s')] += 1
        fell_prev = fell; prev = r
print('%-14s rounds %d kicks %d | falls per round: ' % (label, rounds, kicks) + ', '.join('%s %.2f' % (k, v / max(rounds, 1)) for k, v in b.items()))
