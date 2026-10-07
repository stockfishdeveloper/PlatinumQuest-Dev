"""Offline check of nav.ss_two_gem on recorded kicks (2026-10-05; diagnostic, no game).

For each kick: does the planner's prediction for the aim actually used match what happened (both gems taken without
turning back; a fall within 3.2 s), how often a safe two-gem path existed, and how far its aim is from the one used.
Usage: python -m nav.ss_two_gem_offline <rounds.jsonl glob> ... [--out report.json]
"""
import argparse, glob, json, math, os
import numpy as np
from nav.terrain import TerrainMap
from nav.ss_floor_aim_offline import load_trace
from nav import ss_two_gem as S


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def observed(rows, rnd, D, T, N):
    P = lambda r: np.array([float(r['x']), float(r['y'])])
    V = lambda r: np.array([float(r['vx']), float(r['vy'])])
    picks = [k for k in range(D + 1, D + 120) if (rnd, k) in rows and rows[(rnd, k)]['gem'] not in ('0', '')]
    near = lambda k, G: (rnd, k) in rows and np.linalg.norm(P(rows[(rnd, k)]) - G) < 1.6
    kT = next((k for k in picks if near(k, T)), None)
    kN = next((k for k in picks if N is not None and near(k, N)), None)
    rev_T = kT is not None and any(float(V(rows[(rnd, k)]) @ (T - P(rows[(rnd, k)]))) < 0
                                   for k in range(D + 2, kT) if (rnd, k) in rows)
    rev_N = kT is not None and kN is not None and any(float(V(rows[(rnd, k)]) @ (N - P(rows[(rnd, k)]))) < 0
                                                      for k in range(kT + 1, kN) if (rnd, k) in rows)
    both = (kT is not None and (kT - D) * .064 <= 1.5 and kN is not None and (kN - kT) * .064 <= 1.6
            and not rev_T and not rev_N)
    fell = any(rows.get((rnd, k), {}).get('fell') == '1' for k in range(D + 1, D + 50))
    return both, fell, (None if kT is None else (kT - D) * .064), (None if kN is None else (kN - D) * .064)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('jsonl', nargs='+')
    ap.add_argument('--map', default='KingOfTheMarble_Hunt')
    ap.add_argument('--out')
    args = ap.parse_args()
    terrain = TerrainMap(TerrainMap.resolve(args.map))
    files = sorted({f for g in args.jsonl for f in glob.glob(g)})
    out = []
    for jf in files:
        rows = load_trace(jf[:-len('.rounds.jsonl')])
        for rec in (json.loads(l) for l in open(jf) if l.strip()):
            rnd = rec['round']
            for ev in rec.get('ss_events', []):
                D = ev['decision']; r0 = rows.get((rnd, D))
                if r0 is None:
                    continue
                T = np.array([float(r0['tx']), float(r0['ty'])])
                N = np.array([float(r0['nx']), float(r0['ny'])])
                N = N if np.any(N) else None
                p = np.array(ev['position'][:2]); z = float(ev['position'][2])
                v0 = np.array(ev['velocity_before'][:2]); w0 = np.array(ev['spin_before'])
                aim = unit(np.array(ev['velocity_after'][:2]) - v0)
                pr = S.rollout(terrain, p, z, v0, w0, aim[None], T, N)
                pred_both = bool(np.isfinite(pr['t2'][0]) and pr['off'][0] > pr['t2'][0])
                pred_off = bool(pr['off'][0] < 3.2)
                obs_both, obs_fell, oT, oN = observed(rows, rnd, D, T, N)
                pl = S.plan(terrain, p, z, v0, w0, T, N) if N is not None else None
                used = math.degrees(math.atan2(aim[1], aim[0]) - math.atan2(T[1] - p[1], T[0] - p[0]))
                used = (used + 180) % 360 - 180
                out.append({'file': os.path.basename(jf), 'round': rnd, 'decision': D, 'has_next': N is not None,
                            'line_deg': (None if N is None else math.degrees(math.acos(max(-1, min(1, float(unit(T - p) @ unit(N - T))))))),
                            'speed_before': float(np.linalg.norm(v0)), 'instant': float(np.linalg.norm(v0 + S.KICK * aim)),
                            'pred_both': pred_both, 'pred_off': pred_off, 'pred_t1': float(pr['t1'][0]), 'pred_t2': float(pr['t2'][0]),
                            'obs_both': obs_both, 'obs_fell': obs_fell, 'obs_t1': oT, 'obs_t2': oN,
                            'used_delta_deg': used, 'plan': pl})
    n = len(out)
    def cm(a, b):
        return {'tt': sum(o[a] and o[b] for o in out), 'tf': sum(o[a] and not o[b] for o in out),
                'ft': sum(not o[a] and o[b] for o in out), 'ff': sum(not o[a] and not o[b] for o in out)}
    withN = [o for o in out if o['has_next']]
    planned = [o for o in withN if o['plan']]
    summ = {'kicks': n, 'with_next_gem': len(withN),
            'both: predicted(actual aim) x observed': cm('pred_both', 'obs_both'),
            'fall: predicted off-floor x observed fall': cm('pred_off', 'obs_fell'),
            'plan_both_exists': sum(o['plan']['both'] for o in planned),
            'plan_delta_deg_vs_gem1_line': {'p10_50_90': np.percentile([o['plan']['delta_deg'] for o in planned], [10, 50, 90]).round(1).tolist()} if planned else None,
            'used_delta_deg_vs_gem1_line': {'p10_50_90': np.percentile([o['used_delta_deg'] for o in out], [10, 50, 90]).round(1).tolist()},
            'obs_both_when_plan_both': (sum(o['obs_both'] for o in planned if o['plan']['both']), sum(o['plan']['both'] for o in planned)),
            'obs_both_when_not_plan_both': (sum(o['obs_both'] for o in planned if not o['plan']['both']), sum(not o['plan']['both'] for o in planned)),
            'obs_fell_when_plan_both': (sum(o['obs_fell'] for o in planned if o['plan']['both']), sum(o['plan']['both'] for o in planned)),
            'obs_fell_when_not_plan_both': (sum(o['obs_fell'] for o in planned if not o['plan']['both']), sum(not o['plan']['both'] for o in planned))}
    print(json.dumps(summ, indent=1))
    if args.out:
        with open(args.out, 'w') as f:
            json.dump({'summary': summ, 'kicks': out, 'files': files}, f, indent=1, default=float)


if __name__ == '__main__':
    main()
