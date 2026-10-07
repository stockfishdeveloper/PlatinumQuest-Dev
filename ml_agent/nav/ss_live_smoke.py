"""Collect schema-2 Super Speed approaches using inference only (no training).

Start one separate -autotrain game on --port after launching this listener.
Run nav.ss_drill_starts --live on the printed JSONL path after collecting enough rounds.
"""
import argparse
import os
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port', type=int, default=9975)
    p.add_argument('--map', default='KingOfTheMarble_Hunt')
    p.add_argument('--steps', type=int, default=6000)
    p.add_argument('--seed', type=int, default=777)
    p.add_argument('--worker-id', type=int, default=0, help='distinct log suffix for parallel collectors')
    p.add_argument('--ckpt', default='models/nav/nav_v10_28897.pth')
    p.add_argument('--prepare', action='store_true', help='use the diagnostic approach controller for state coverage only')
    p.add_argument('--force-use', action='store_true', help='fire only where the existing mask approves')
    p.add_argument('--deterministic', action='store_true')
    p.add_argument('--lookbacks', default='24', help='capture these decision counts before each fire, e.g. 24,4')
    args = p.parse_args()
    os.environ['NAV_DRILL_PLAN'] = '0:0'
    os.environ['NAV_LIVE_RECORD'] = '1'
    os.environ['NAV_DRILL_NO_USE'] = '0'
    os.environ['NAV_LIVE_LOOKBACKS'] = args.lookbacks
    os.environ['NAV_COLLECTION_BEHAVIOR'] = f'prepare={int(args.prepare)},force={int(args.force_use)},deterministic={int(args.deterministic)}'
    os.environ.setdefault('NAV_TOUR', 'walk')
    import torch
    from nav.vec_worker import InstanceWorker
    from nav.model import NavActorCritic
    from nav.ss_prepare import prepare_action
    torch.set_num_threads(1); torch.manual_seed(args.seed)
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    model = NavActorCritic(); model.load_state_dict(ck['model']); model.eval()
    w = InstanceWorker(args.worker_id, args.port, args.seed, .65, .6)
    print(f'INFERENCE COLLECTION: {args.ckpt}, {args.steps} decisions, no optimizer; '
          f'{os.environ["NAV_COLLECTION_BEHAVIOR"]}; lookbacks={args.lookbacks}', flush=True)
    t0 = time.time()
    try:
        w.connect()
        if w.mission != args.map:
            raise ValueError(f'expected {args.map}, got {w.mission}')
        h = model.initial_state(1, 'cpu')
        for k in range(args.steps):
            with torch.no_grad():
                v = torch.as_tensor(w.vec)[None]
                o = model.act(torch.as_tensor(w.crop)[None], v, h, deterministic=args.deterministic)
                a = o['action_game'][0].tolist()
                if args.force_use:
                    a[5] = float(model.use_prior(v)[0] > .5)
            if args.prepare and w.pending_since is None:
                a, _ = prepare_action(a, w.terrain, w.env.msg.obs, w.vec)
            rep = w.step(a + o['mean_dir'][0].tolist())
            h = model.initial_state(1, 'cpu') if rep.get('new_segment') else o['h_next']
            for line in rep.get('log', []):
                if 'GAME map' in line:
                    print(line, flush=True)
            if (k + 1) % 500 == 0:
                print(f'{k + 1} decisions, recorded={w.live_written}, rejected={w.live_rejected}', flush=True)
        if w.live_file is None:
            raise RuntimeError('no eligible approaches collected; collect more sampled rounds')
        print(f'COLLECTED {w.live_written} starts in {time.time() - t0:.0f}s: {w.live_file.name}', flush=True)
        print('Convert with: python -m nav.ss_drill_starts --live <JSONL paths>', flush=True)
        print('Keep collecting if either round-grouped split is empty. Never copy dev into eval.', flush=True)
    finally:
        if w.live_file is not None:
            w.live_file.close()
        w.env.close()


if __name__ == '__main__':
    main()
