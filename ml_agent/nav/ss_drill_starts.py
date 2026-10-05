"""Super Speed curriculum starts (log 40.26) from the agent's OWN training traces: every real kick (horizontal velocity
changed by >= KICK_DV u/s in one decision, on the floor, no teleport) gives
    post: the state one decision after the kick (Stage 1: steer and brake into the gem, then continue)
    pre:  the state PRE_DEC decisions before the kick, a Super Speed still held (Stage 2: use or not)
with the intended gem (the waypoint at the kick) and the gem after it (the next distinct waypoint). Spin is the rolling
spin of the pre-kick velocity (the trace has no spin). Written as JSON, split dev / eval by a hash of (file, step).
    python -m nav.ss_drill_starts [trace.csv ...]   (default: the 10-03 traces from 18:40 on, the post-pickup rule)
"""
import glob, json, math, os, sys, zlib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, 'datasets', 'ss_drill')
KICK_DV = 15.0          # u/s change of the horizontal velocity in one decision
PRE_DEC = 3             # decisions before the kick for the Stage 2 start
R_MARBLE = 0.19
COLS = ['step', 'time_left_s', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'on_floor', 'gx', 'gy', 'inst']


def starts_from(path):
    df = pd.read_csv(path, usecols=COLS)
    out = []
    for inst, g in df.groupby('inst', sort=False):
        a = g.to_numpy(dtype=np.float64)
        st, tl, x, y, z, vx, vy, vz, fl, gx, gy = (a[:, i] for i in range(11))
        dvh = np.hypot(np.diff(vx), np.diff(vy))
        dp = np.hypot(np.diff(x), np.diff(y))
        dt = tl[:-1] - tl[1:]
        cand = np.nonzero((dvh >= KICK_DV) & (dp < 3.0) & (dt > 0) & (dt < 0.2) & (fl[:-1] > 0.5) & (fl[1:] > 0.5))[0]
        for k in cand:
            j = k + 1                                     # first decision after the kick
            if j + 2 >= len(a) or k - PRE_DEC < 0:
                continue
            tx, ty = gx[j], gy[j]
            if abs(gx[k] - tx) > 0.5 or abs(gy[k] - ty) > 0.5:
                continue                                  # the waypoint switched at the kick: not a post-pickup kick
            sp = math.hypot(vx[j], vy[j]); dx, dy = tx - x[j], ty - y[j]; d = math.hypot(dx, dy)
            if not (8.0 <= sp <= 32.0 and 2.0 <= d <= 25.0):
                continue
            if (vx[j] * dx + vy[j] * dy) / (sp * d) < math.cos(math.radians(45)):
                continue                                  # the kick did not point at the intended gem
            fx = fy = None
            for m in range(j + 1, min(len(a), j + 120)):
                if dt[m - 1] <= 0 or dt[m - 1] > 0.2 if m - 1 < len(dt) else True:
                    break                                 # round end / teleport: stop looking
                if math.hypot(gx[m] - tx, gy[m] - ty) > 0.5:
                    fx, fy = gx[m], gy[m]; break
            if fx is None or math.hypot(fx - tx, fy - ty) > 30.0:
                continue
            pk = k - PRE_DEC
            if math.hypot(gx[pk] - tx, gy[pk] - ty) > 0.5:
                pre_target = (gx[pk], gy[pk])             # the gem about to be picked up before the kick (40.27)
            else:
                pre_target = (tx, ty)                     # already past that pickup
            pre_goals = []                                # 40.27: every waypoint from the pre start to the kick, in order
            for m in range(pk, j + 1):
                if not pre_goals or math.hypot(gx[m] - pre_goals[-1][0], gy[m] - pre_goals[-1][1]) > 0.5:
                    pre_goals.append([float(gx[m]), float(gy[m])])
            # rolling spin of the pre-kick velocity: v = r (wy, -wx) -> w = (-vy / r, vx / r, 0)
            wx, wy = -vy[k] / R_MARBLE, vx[k] / R_MARBLE
            out.append({'src': os.path.basename(path), 'step': int(st[j]), 'inst': int(inst),
                        'post': [x[j], y[j], z[j], vx[j], vy[j], vz[j], wx, wy, 0.0],
                        'pre': [x[pk], y[pk], z[pk], vx[pk], vy[pk], vz[pk], -vy[pk] / R_MARBLE, vx[pk] / R_MARBLE, 0.0],
                        'pre_target': [float(pre_target[0]), float(pre_target[1])], 'pre_goals': pre_goals, 'pre_dec': PRE_DEC,
                        'target': [float(tx), float(ty)], 'next': [float(fx), float(fy)],
                        'speed_before': math.hypot(vx[k], vy[k]), 'speed_after': sp,
                        'turn_deg': math.degrees(math.acos(max(-1.0, min(1.0, (vx[k] * vx[j] + vy[k] * vy[j]) /
                                                                         max(1e-6, math.hypot(vx[k], vy[k]) * sp)))))})
    return out


def live_starts(paths, min_speed=8.0, max_speed=32.0):
    """40.40: live-recorded fires (vec_worker LIVE_RECORD: logs/nav/live_starts_*.jsonl) -> starts with the ACTUAL spin
    (RAW_SPIN) in the pre and post states, the GRU warm-up history, and the chain of gems picked up in the 12 s after
    the kick. 'pre' is LIVE_PRE_DEC decisions before the use (a Super Speed held), 'post' right after the kick."""
    from nav.protocol import RAW_POS, RAW_VEL, RAW_SPIN
    out = []
    for p in paths:
        for line in open(p, encoding='utf-8'):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            pre, post = r['pre_raw'], r['post_raw']
            def state(raw):
                return [float(q) for q in raw[RAW_POS]] + [float(q) for q in raw[RAW_VEL]] + [float(q) for q in raw[RAW_SPIN]]
            sp_after = math.hypot(post[RAW_VEL][0], post[RAW_VEL][1]); sp_before = math.hypot(pre[RAW_VEL][0], pre[RAW_VEL][1])
            if not (min_speed <= sp_after <= max_speed) or not r.get('chain'):
                continue
            tx, ty = r['target'][0], r['target'][1]
            nxt = r['next'] if r.get('next') else r['chain'][0]
            vb = pre[RAW_VEL]; va = post[RAW_VEL]
            turn = math.degrees(math.acos(max(-1.0, min(1.0, (vb[0] * va[0] + vb[1] * va[1]) / max(1e-6, sp_before * sp_after)))))
            out.append({'id': r['id'], 'mission': r.get('mission', 'KingOfTheMarble_Hunt'), 'src': os.path.basename(p), 'step': int(r['fire_step']),
                        'pre': state(pre), 'post': state(post), 'pre_target': [float(r['pre_goal'][0]), float(r['pre_goal'][1])],
                        'pre_goals': [[float(r['pre_goal'][0]), float(r['pre_goal'][1])]] + ([[float(tx), float(ty)]] if math.hypot(r['pre_goal'][0] - tx, r['pre_goal'][1] - ty) > 0.5 else []),
                        'pre_dec': 3, 'target': [float(tx), float(ty)], 'next': [float(nxt[0]), float(nxt[1])],
                        'chain': [[float(a), float(b)] for a, b in r['chain']], 'chain_t': r.get('chain_t', []), 'falls_live': r.get('falls', 0),
                        'hist': r.get('hist', []), 'speed_before': sp_before, 'speed_after': sp_after, 'turn_deg': turn})
    return out


def main_live(paths):
    allst = live_starts(paths)
    os.makedirs(OUT, exist_ok=True)
    dev, ev = [], []
    for s in allst:
        (ev if zlib.crc32(s['id'].encode()) % 4 == 0 else dev).append(s)
    json.dump(dev, open(os.path.join(OUT, 'starts_live_dev.json'), 'w')); json.dump(ev, open(os.path.join(OUT, 'starts_live_eval.json'), 'w'))
    sp = [s['speed_after'] for s in allst]; ch = [len(s['chain']) for s in allst]
    print(f'live starts: {len(allst)} fires from {len(paths)} file(s) -> dev {len(dev)} / eval {len(ev)}; speed after median '
          f'{np.median(sp) if sp else 0:.1f} u/s; chain gems median {np.median(ch) if ch else 0:.0f}; '
          f'with a fall in the 12 s {sum(1 for s in allst if s["falls_live"]) / max(len(allst), 1) * 100:.0f} %')


def main(paths):
    allst = []
    for p in paths:
        s = starts_from(p)
        print(f'{os.path.basename(p)}: {len(s)} kicks')
        allst += s
    os.makedirs(OUT, exist_ok=True)
    dev, ev = [], []
    for s in allst:
        (ev if zlib.crc32(f"{s['src']}:{s['step']}".encode()) % 5 == 0 else dev).append(s)
    for name, lst in (('starts_dev.json', dev), ('starts_eval.json', ev)):
        tmp = os.path.join(OUT, name + '.new')
        with open(tmp, 'w') as f:
            json.dump(lst, f)
        os.replace(tmp, os.path.join(OUT, name))
    sa = np.array([s['speed_after'] for s in allst]); tu = np.array([s['turn_deg'] for s in allst])
    print(f'total {len(allst)} (dev {len(dev)}, eval {len(ev)}); speed after the kick median {np.median(sa):.1f} '
          f'(p10 {np.percentile(sa, 10):.1f}, p90 {np.percentile(sa, 90):.1f}); turn median {np.median(tu):.0f} deg')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--live':
        files = sys.argv[2:] or sorted(glob.glob(os.path.join(HERE, 'logs', 'nav', 'live_starts_*.jsonl')))
        main_live(files); sys.exit(0)
    paths = sys.argv[1:] or sorted(p for p in glob.glob(os.path.join(HERE, 'logs', 'nav', 'trace_20261003_*.csv'))
                                   if os.path.basename(p) >= 'trace_20261003_184036.csv')
    main(paths)
