"""Deterministic Super Speed drill evaluation (log 40.26-40.27): the held-out starts in order, no variations, the same
worker code as training (nav/vec_worker.py drill mode), the policy's mean action. One game on --port.
    stage 1 (after the kick):   python -m nav.ss_drill_eval --ckpt <pth> --port 9971 --tag d1
    stage 2 (before it, a Super Speed held; --no-use for the matched comparison without the kick):
                                python -m nav.ss_drill_eval --ckpt <pth> --stage 2 [--no-use] --port 9971 --tag d2
then start: marbleblast_mbx.exe -autotrain KingOfTheMarble_Hunt -aiport <port>
Writes logs/nav/ss_drill_eval_<tag>.json and prints: the kick's gem (within 1.0 / 1.5 s, at all), the gem after it,
falls, the time from the start to the gem after (the whole manoeuvre), fires (stage 2).
"""
import argparse, json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True); ap.add_argument('--port', type=int, default=9971); ap.add_argument('--tag', default='eval')
ap.add_argument('--n', type=int, default=0, help='drills (default: every eval start once)')
ap.add_argument('--map', default='KingOfTheMarble_Hunt', help='accepted for run_one.ps1; the starts carry their mission')
ap.add_argument('--stage', type=int, default=1); ap.add_argument('--no-use', action='store_true'); ap.add_argument('--sample', action='store_true')
ap.add_argument('--force-brake', type=int, default=0, help='diagnostic: brake (full thrust against the velocity) for the first K '
                'decisions of every drill (40.32: does braking along the kick line pay?)')
ap.add_argument('--force-use', action='store_true', help='stages 2-3: fire at the first decision the approval mask allows (the '
                'manoeuvre with the current control, independent of the learned decision); compare with --no-use')
args = ap.parse_args()
ev = os.path.join('datasets', 'ss_drill', {1: 'starts_eval.json', 2: 'starts2_eval.json', 3: 'starts3_eval.json', 4: 'starts_live_eval.json'}[args.stage])
os.environ['NAV_DRILL_PLAN'] = f'0:{args.stage}'; os.environ['NAV_DRILL_EVAL'] = '1'
os.environ[{1: 'NAV_DRILL_STARTS', 2: 'NAV_DRILL2_STARTS', 3: 'NAV_DRILL3_STARTS', 4: 'NAV_DRILL4_STARTS'}[args.stage]] = ev
os.environ['NAV_DRILL_NO_USE'] = '1' if args.no_use else '0'
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
from nav.vec_worker import InstanceWorker, DRILL_MAX_SPEED     # noqa: E402
from nav.model import NavActorCritic                           # noqa: E402

n_starts = sum(1 for s in json.load(open(ev)) if s['speed_after'] <= DRILL_MAX_SPEED)
N = args.n or n_starts
ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
model = NavActorCritic(); model.load_state_dict(ck['model']); model.eval()
w = InstanceWorker(0, args.port, 12345, float(ck.get('arrive_r', 0.65)), float(ck.get('arrive_dz', 0.6)))
print(f'drill eval stage {args.stage}{" no-use" if args.no_use else ""}: {os.path.basename(args.ckpt)} (update {ck.get("update")}), '
      f'{"sampled" if args.sample else "deterministic"}, {N} drills, port {args.port}', flush=True)
w.connect()
h = model.initial_state(1, 'cpu')
rx = re.compile(r'DRILL[234]? v0=([\d.]+) (?:fired=(\d) tf=([\d.]+) pk=(\d+) )?hit1=(\d) t1=([\d.]+) hit2=(\d) t2=([\d.]+) out=(\w+) ret=([-\d.]+)'
                r'(?: id=(\S+) picks=(\d+) tpicks=(\S*) falls=(\d+))?')
res = []; t0 = time.time(); k_in = 0; warmed = False
while len(res) < N:
    with torch.no_grad():
        o = model.act(torch.as_tensor(w.crop).unsqueeze(0), torch.as_tensor(w.vec).unsqueeze(0), h, deterministic=not args.sample)
    a = o['action_game'][0].tolist() + o['mean_dir'][0].tolist()
    h = o['h_next']
    if k_in < args.force_brake:
        a[4] = 1.0
    if args.force_use:
        with torch.no_grad():
            a[5] = 1.0 if float(model.use_prior(torch.as_tensor(w.vec).unsqueeze(0))[0]) > 0.5 else 0.0
    k_in += 1
    rep = w.step(a)
    if rep.get('new_segment'):
        k_in = 0
        if rep.get('warm'):                      # 40.40: a live start: warm the GRU with the recorded observations
            h = model.initial_state(1, 'cpu')
            with torch.no_grad():
                for c_w, v_w in rep['warm']:
                    h = model.core(torch.as_tensor(c_w).unsqueeze(0), torch.as_tensor(v_w).unsqueeze(0), h)
            warmed = True
    for line in rep.get('log', []):
        m = rx.search(line)
        if m:
            g = m.groups()
            res.append({'v0': float(g[0]), 'fired': g[1] == '1', 'tf': float(g[2] or 0), 'hit1': g[4] == '1', 't1': float(g[5]),
                        'hit2': g[6] == '1', 't2': float(g[7]), 'out': g[8], 'ret': float(g[9]),
                        'id': g[10] or '', 'picks': int(g[11] or 0), 'tpicks': [float(t) for t in (g[12] or '').split(',') if t],
                        'falls': int(g[13] or 0)})
    if rep.get('new_segment') and not warmed:
        h = model.initial_state(1, 'cpu')
    warmed = False
w.env.close()
n = len(res)
h1 = [r['t1'] for r in res if r['hit1']]
both = [r['t1'] + r['t2'] for r in res if r['hit1'] and r['hit2']]
out = {'ckpt': args.ckpt, 'update': ck.get('update'), 'stage': args.stage, 'no_use': args.no_use, 'n': n, 'sample': args.sample,
       'hit1_1s': sum(1 for r in res if r['hit1'] and r['t1'] <= 1.0) / n,
       'hit1_15s': sum(1 for r in res if r['hit1'] and r['t1'] <= 1.5) / n,
       'hit1': len(h1) / n, 't1_median': float(np.median(h1)) if h1 else None,
       'hit2': sum(1 for r in res if r['hit2']) / n, 't_both_median': float(np.median(both)) if both else None,
       'fell': sum(1 for r in res if r['out'] == 'fell') / n, 'timeout': sum(1 for r in res if r['out'] == 'timeout') / n,
       'fired': sum(1 for r in res if r['fired']) / n, 'ret': float(np.mean([r['ret'] for r in res])),
       'picks_mean': float(np.mean([r['picks'] for r in res])), 'falls_mean': float(np.mean([r['falls'] for r in res])),
       't_first_median': float(np.median([r['tpicks'][0] for r in res if r['tpicks']])) if any(r['tpicks'] for r in res) else None,
       'wall_s': round(time.time() - t0, 1), 'per_drill': res}
json.dump(out, open(os.path.join('logs', 'nav', f'ss_drill_eval_{args.tag}.json'), 'w'), indent=1)
if args.stage == 4:
    print('WINDOW EVAL %s: n %d (%s) | pickups in 12 s mean %.2f | falls mean %.2f | first pickup median %.2f s | fired %.0f%% | return %.2f | %.0f s' % (
        args.tag, n, 'no-use' if args.no_use else ('force-use' if args.force_use else 'learned'), out['picks_mean'], out['falls_mean'],
        out['t_first_median'] or -1, 100 * out['fired'], out['ret'], out['wall_s']), flush=True)
print('DRILL EVAL %s: stage %d%s n %d | kick gem within 1.0 s %.0f%%, 1.5 s %.0f%%, at all %.0f%% (median %.2f s) | gem after %.0f%% | '
      'both, median time %.2f s | fell %.0f%% | timeout %.0f%% | fired %.0f%% | return %.1f | %.0f s' % (
          args.tag, args.stage, ' no-use' if args.no_use else '', n, 100 * out['hit1_1s'], 100 * out['hit1_15s'], 100 * out['hit1'],
          out['t1_median'] or -1, 100 * out['hit2'], out['t_both_median'] or -1, 100 * out['fell'], 100 * out['timeout'],
          100 * out['fired'], out['ret'], out['wall_s']), flush=True)
