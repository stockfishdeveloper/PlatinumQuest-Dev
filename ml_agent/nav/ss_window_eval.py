"""Matched real-game windows. Restores once per id/branch, never trains a model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from nav.replay import WINDOW_MS, load_starts, paired_summary, REWARD_VERSION


def run(args):
    os.environ['NAV_DRILL_PLAN'] = '0:4'
    os.environ['NAV_DRILL_EVAL'] = '1'
    os.environ['NAV_LIVE_RECORD'] = '0'
    os.environ['NAV_DRILL_NO_USE'] = '0'
    os.environ.setdefault('NAV_TOUR', 'walk')
    import torch
    from nav.vec_worker import InstanceWorker
    from nav.model import NavActorCritic
    torch.set_num_threads(1)
    starts = load_starts(args.starts, args.window_ms)
    if args.n:
        if args.n > len(starts):
            raise ValueError('--n exceeds unique fixture count; repeated ids are not allowed')
        starts = starts[:args.n]
    branches = args.branches.split(',')
    allowed = {'base', 'no_use', 'force', 'learned', 'fire_now', 'wait_now'}
    if not branches or len(set(branches)) != len(branches) or not set(branches) <= allowed:
        raise ValueError('invalid or duplicate branches')
    models = {}
    for key, file in [('trained', args.ckpt), ('base', args.base_ckpt)]:
        ck = torch.load(file, map_location='cpu', weights_only=False)
        m = NavActorCritic(); m.load_state_dict(ck['model']); m.eval()
        models[key] = m
    w = InstanceWorker(0, args.port, 12345, .65, .6)
    w.eval_starts = starts
    w.replay_window_ms = args.window_ms
    rows = []
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {'schema': 2, 'reward_version': REWARD_VERSION, 'window_ms': args.window_ms,
                'checkpoint': args.ckpt, 'base_checkpoint': args.base_ckpt,
                'checkpoint_sha256': hashlib.sha256(Path(args.ckpt).read_bytes()).hexdigest(),
                'base_checkpoint_sha256': hashlib.sha256(Path(args.base_ckpt).read_bytes()).hexdigest(),
                'starts_sha256': hashlib.sha256(Path(args.starts).read_bytes()).hexdigest(),
                'branches': branches, 'requested_ids': [s['id'] for s in starts]}

    def save():
        ref = 'no_use' if 'no_use' in branches else ('wait_now' if 'wait_now' in branches else branches[0])
        output.write_text(json.dumps({**manifest, 'rows': rows, 'summary': paired_summary(rows, ref)}, indent=2))

    try:
        w.connect()
        if w.mission != args.map:
            raise ValueError(f'expected mission {args.map}, got {w.mission}')
        for s in starts:
            for branch in branches:
                m = models['base' if branch == 'base' else 'trained']
                try:
                    w.begin_replay(s)
                    h = m.initial_state(1, 'cpu')
                    with torch.no_grad():
                        for c, v in w.warm or []:
                            h = m.core(torch.as_tensor(c)[None], torch.as_tensor(v)[None], h)
                    w.warm = None
                    decision = 0
                    while True:
                        with torch.no_grad():
                            o = m.act(torch.as_tensor(w.crop)[None], torch.as_tensor(w.vec)[None], h, deterministic=True)
                            approved = float(m.use_prior(torch.as_tensor(w.vec)[None])[0]) > .5
                        a = o['action_game'][0].tolist() + o['mean_dir'][0].tolist()
                        if branch in ('base', 'no_use') or (branch == 'wait_now' and decision == 0):
                            a[5] = 0
                        elif branch == 'force' or (branch == 'fire_now' and decision == 0):
                            a[5] = float(approved)
                        rep = w.step(a); h = o['h_next']; decision += 1
                        if rep['skip']:
                            raise RuntimeError('unexpected reset during replay')
                        if rep['done']:
                            row = dict(w.last_drill_result)
                            if row['elapsed_ms'] != args.window_ms:
                                raise RuntimeError(f'window length mismatch: {row["elapsed_ms"]} ms')
                            break
                    row['branch'] = branch
                except (ValueError, RuntimeError) as e:
                    row = {'id': s['id'], 'branch': branch, 'status': 'failed', 'reason': str(e)}
                row['split_group'] = s['split_group']
                rows.append(row)
                print(json.dumps(row, separators=(',', ':')), flush=True)
                save()
    finally:
        w.env.close()
    ref = 'no_use' if 'no_use' in branches else ('wait_now' if 'wait_now' in branches else branches[0])
    print(json.dumps(paired_summary(rows, ref), indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--base-ckpt', default='models/nav/nav_v10_28897.pth')
    p.add_argument('--starts', default='datasets/ss_drill/starts_replay_eval.json')
    p.add_argument('--branches', default='base,no_use,force,learned')
    p.add_argument('--window-ms', type=int, default=WINDOW_MS)
    p.add_argument('--n', type=int, default=0)
    p.add_argument('--port', type=int, default=9978)
    p.add_argument('--map', default='KingOfTheMarble_Hunt')
    p.add_argument('--out', default='logs/nav/ss_window_eval.json')
    args = p.parse_args()
    if args.window_ms <= 0 or args.window_ms % 64:
        p.error('window must be a positive multiple of 64 simulated milliseconds')
    run(args)


if __name__ == '__main__':
    main()
