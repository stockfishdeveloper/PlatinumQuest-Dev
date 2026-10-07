"""Offline check of a spin-aware Super Speed aim on recorded kicks (2026-10-05, diagnostic: no game, no training).

For every recorded fire on supported flat floor:
1. Model check: replay the kick with nav.ss_floor_model from the recorded state, the recorded joysticks and the
   reconstructed camera yaw, and compare with the observed path.
2. Aim comparison: under one common input model ("steer straight at the target", the bridge's diagonal camera), the
   path of the aim actually used vs the aim the floor model picks to reach the target soonest (or pass closest).

Timing (HANDOFF_SUPERSPEED_2026-10-05 section 7): an event's 'decision' D is the state C_D before the kick; the kick
happens on C_D -> C_D+1, driven by the joystick recorded on trace row D; its direction is the command aim recorded on
row D (or the latest earlier row that sent the use). Spin at C_k is row k+1's spin_before; the event carries the
full-precision C_D velocity and spin.

Usage: python -m nav.ss_floor_aim_offline <rounds.jsonl glob> [--out report.json]
"""
import argparse, glob, json, math, os
import numpy as np
from nav.ss_floor_model import step_batch

DT = 0.064
SUB = 0.008                 # integration and path sampling step
KICK = 25.0                 # u/s added along the kick yaw (engine SuperSpeedVelocity)
R_PICK = 0.85               # u, horizontal: planner PICK_R0
HOLD_DEC = 3                # decisions the kick yaw overrides the camera after the last use send (12 ticks)
HORIZON = 24                # decisions (1.536 s)
SQ2 = math.sqrt(2.0)


def load_trace(path):
    rows = {}
    with open(path) as f:
        hdr = f.readline().strip().split(',')
        for line in f:
            vals = line.strip().split(',')
            if len(vals) == len(hdr):
                d = dict(zip(hdr, vals))
                rows[(int(d['round']), int(d['dec']))] = d
    return rows


def js_world(row):
    return np.array([float(row['right']) - float(row['left']), float(row['fwd']) - float(row['back'])])


def steer_yaw(u):
    """The bridge's camera yaw for a world command: the command lands on the camera diagonal (joystick.py)."""
    return math.pi / 4 - math.atan2(u[1], u[0]) if np.any(u) else 0.0


def simulate(p0, v0, w0, aims, inputs, yaws, target):
    """Batch rollout for N candidate kick directions. aims (N, 2) unit; inputs/yaws: callables (k, p) -> (N, 2) / (N,)
    for decision k = 0..HORIZON-1 (k = 0 is the kick step). Returns first reach time (s, inf if none), min distance,
    and positions at decisions 4, 8, 16, 24."""
    n = len(aims)
    p = np.repeat(p0[None], n, 0).astype(float)
    v = np.repeat(v0[None], n, 0).astype(float) + KICK * aims
    w = np.repeat(w0[None], n, 0).astype(float)
    reach = np.full(n, np.inf); dmin = np.linalg.norm(p - target, axis=1)
    snaps = {}
    for k in range(HORIZON):
        u = inputs(k, p); yaw = yaws(k, p)
        for s in range(int(round(DT / SUB))):
            # step_batch's yaw arithmetic is element-wise, so a per-candidate (N,) yaw array broadcasts row by row
            p, v, w = step_batch(p, v, w, u, np.asarray(yaw, dtype=float), dt=SUB)
            d = np.linalg.norm(p - target, axis=1)
            t = (k * DT) + (s + 1) * SUB
            reach = np.where((d <= R_PICK) & np.isinf(reach), t, reach)
            dmin = np.minimum(dmin, d)
        if k + 1 in (4, 8, 16, 24):
            snaps[k + 1] = p.copy()
    return reach, dmin, snaps


def kick_aim(rows, rnd, dec):
    """The kick direction: the command aim on row dec, else the latest earlier sending row (the latched aim)."""
    for k in range(dec, dec - 4, -1):
        r = rows.get((rnd, k))
        if r is not None and r['use_sent'] == '1':
            a = np.array([float(r['command_aim_x']), float(r['command_aim_y'])])
            if np.linalg.norm(a) > 0.5:
                return a / np.linalg.norm(a), k
    return None, None


def supported(rows, rnd, dec, z0):
    for k in range(dec, dec + HORIZON + 1):
        r = rows.get((rnd, k))
        if r is None or r['on_floor'] != '1' or r['fell'] != '0' or abs(float(r['z']) - z0) > 0.3:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('jsonl', nargs='+')
    ap.add_argument('--out')
    args = ap.parse_args()
    files = sorted({f for g in args.jsonl for f in glob.glob(g)})
    out, excluded = [], {'no_trace_row': 0, 'no_aim': 0, 'unsupported': 0, 'no_target': 0}
    for jf in files:
        rows = load_trace(jf[:-len('.rounds.jsonl')])
        for rec in (json.loads(l) for l in open(jf) if l.strip()):
            for ev in rec.get('ss_events', []):
                rnd, D = rec['round'], ev['decision']
                if (rnd, D) not in rows:
                    excluded['no_trace_row'] += 1; continue
                aim, aim_row = kick_aim(rows, rnd, D)
                if aim is None:
                    excluded['no_aim'] += 1; continue
                if not ev.get('target'):
                    excluded['no_target'] += 1; continue
                p0 = np.array(ev['position'][:2]); z0 = ev['position'][2]
                if not supported(rows, rnd, D, z0):
                    excluded['unsupported'] += 1; continue
                v0 = np.array(ev['velocity_before'][:2]); w0 = np.array(ev['spin_before'])
                tgt = np.array(ev['target'][:2])
                kick_yaw = math.atan2(aim[0], aim[1])

                # 1. model check with recorded joysticks; camera yaw = kick yaw for HOLD_DEC decisions, then steering
                rec_u = [js_world(rows[(rnd, D + k)]) for k in range(HORIZON)]
                _, _, snaps = simulate(p0, v0, w0, aim[None],
                                       lambda k, p: np.repeat(rec_u[k][None], len(p), 0),
                                       lambda k, p: np.full(len(p), kick_yaw if k < HOLD_DEC else steer_yaw(rec_u[k])), tgt)
                err = {k: float(np.linalg.norm(snaps[k][0] - np.array([float(rows[(rnd, D + k)]['x']), float(rows[(rnd, D + k)]['y'])])))
                       for k in snaps}
                obs_path = np.array([[float(rows[(rnd, D + k)]['x']), float(rows[(rnd, D + k)]['y'])] for k in range(1, HORIZON + 1)])
                obs_dmin = float(np.min(np.linalg.norm(obs_path - tgt, axis=1)))

                # 2. common input model: steer straight at the target; the kick yaw holds the camera for HOLD_DEC
                base = math.atan2(aim[1], aim[0])
                angs = base + np.radians(np.arange(-45, 46, 1.0))
                cand = np.stack([np.cos(angs), np.sin(angs)], 1)
                cyaw = np.arctan2(cand[:, 0], cand[:, 1])

                def steer_u(k, p):
                    d = tgt[None] - p
                    n = np.linalg.norm(d, axis=1, keepdims=True)
                    return SQ2 * d / np.maximum(n, 1e-6)

                def steer_y(k, p):
                    if k < HOLD_DEC:
                        return cyaw.copy()
                    u = steer_u(k, p)
                    return np.pi / 4 - np.arctan2(u[:, 1], u[:, 0])

                reach, dmin, _ = simulate(p0, v0, w0, cand, steer_u, steer_y, tgt)
                i0 = 45                                                          # the aim actually used
                key = np.where(np.isfinite(reach), reach, 10.0 + dmin)
                ib = int(np.argmin(key))
                vr = v0 + KICK * cand[ib]
                out.append({'file': os.path.basename(jf), 'round': rnd, 'decision': D, 'aim_row': aim_row,
                            'speed_before': float(np.linalg.norm(v0)), 'target_dist': float(np.linalg.norm(tgt - p0)),
                            'model_err': err, 'observed_min_dist': obs_dmin,
                            'actual': {'reach_s': float(reach[i0]), 'min_dist': float(dmin[i0])},
                            'best': {'reach_s': float(reach[ib]), 'min_dist': float(dmin[ib]),
                                     'delta_deg': float(ib - i0), 'instant_speed': float(np.linalg.norm(vr))}})
    n = len(out)
    summ = {'events': n, 'excluded': excluded}
    if n:
        for k in (4, 8, 16, 24):
            e = [o['model_err'][k] for o in out]
            summ[f'model_err_{k}dec'] = {'mean': float(np.mean(e)), 'p90': float(np.percentile(e, 90))}
        ra = np.array([o['actual']['reach_s'] for o in out]); rb = np.array([o['best']['reach_s'] for o in out])
        summ['reach_actual'] = int(np.isfinite(ra).sum()); summ['reach_best'] = int(np.isfinite(rb).sum())
        both = np.isfinite(ra) & np.isfinite(rb)
        summ['reach_time_saved_s_when_both'] = float(np.mean(ra[both] - rb[both])) if both.any() else None
        summ['observed_within_pick'] = int(sum(o['observed_min_dist'] <= R_PICK for o in out))
        summ['predicted_actual_within_pick'] = summ['reach_actual']
        summ['delta_deg'] = {'mean': float(np.mean([o['best']['delta_deg'] for o in out])),
                             'mean_abs': float(np.mean([abs(o['best']['delta_deg']) for o in out]))}
        summ['min_dist_actual_mean'] = float(np.mean([o['actual']['min_dist'] for o in out]))
        summ['min_dist_best_mean'] = float(np.mean([o['best']['min_dist'] for o in out]))
    print(json.dumps(summ, indent=1))
    if args.out:
        with open(args.out, 'w') as f:
            json.dump({'summary': summ, 'events': out, 'files': files, 'r_pick': R_PICK, 'kick': KICK,
                       'hold_dec': HOLD_DEC, 'horizon_dec': HORIZON}, f, indent=1)


if __name__ == '__main__':
    main()
