"""Deterministic navigator evaluation on the map the game is running.

    python -m nav.eval_nav                 (game launched with -autotrain <Map>; uses models/nav/nav_latest.pth)
    NAV_CKPT=models/nav/nav_000100.pth python -m nav.eval_nav

Runs N_SEGMENTS segments from fixed (seeded) start/goal pairs with the greedy
policy (no sampling) and reports arrival %, falls per 100 u, mean speed and
time per 10 u. Writes logs/nav/eval_<map>_<ckpt>.json. These are the milestone
numbers; training-curve values are not.
"""
import os
import sys
import json
import numpy as np
import torch
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from terrain_obs import TerrainMap                                   # noqa: E402
from nav.protocol import RAW_VEL                                     # noqa: E402
from nav.env import HuntEnv                                          # noqa: E402
from nav.terrain import TerrainGrid                                  # noqa: E402
from nav.obs import ObsBuilder, NAV_OBS_VERSION                      # noqa: E402
from nav.model import NavActorCritic, action_to_joystick             # noqa: E402
from nav.waypoints import SegmentManager, RoundOver                  # noqa: E402

PORT = 8888
GAME_SPEED = 3
N_SEGMENTS = 60
EVAL_SEED = 12345
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ckpt = os.environ.get('NAV_CKPT', os.path.join(HERE, 'models', 'nav', 'nav_latest.pth'))
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(ckpt, map_location=dev)
    assert ck.get('obs_version') == NAV_OBS_VERSION, ck.get('obs_version')
    model = NavActorCritic().to(dev); model.load_state_dict(ck['model']); model.eval()
    env = HuntEnv(PORT, speed=GAME_SPEED); env.connect()
    mission = env.info['mission']
    terrain = TerrainGrid(TerrainMap.resolve(mission))
    rng = np.random.default_rng(EVAL_SEED)
    segs = SegmentManager(terrain, rng, log=lambda s: None)
    obs_b = ObsBuilder(terrain)
    print(f'eval {os.path.basename(ckpt)} (update {ck.get("update")}) on {mission}: {N_SEGMENTS} segments, seed {EVAL_SEED}')
    results = []
    while len(results) < N_SEGMENTS:
        try:
            goal = segs.begin(env, teleport=True)
        except RoundOver:
            env.wait_new_round(); continue
        obs_b.reset(); h = model.initial_state(1, dev)
        crop, vec, on_floor = obs_b.build(env.msg.obs, goal)
        while True:
            out = model.act(torch.as_tensor(crop, device=dev).unsqueeze(0), torch.as_tensor(vec, device=dev).unsqueeze(0), h, deterministic=True)
            a = out['action_game'][0].tolist(); vel = env.msg.obs[RAW_VEL]
            msg, info = env.step(action_to_joystick(a[0], a[1], a[2], a[3], a[4], float(vel[0]), float(vel[1])))
            if info['round_ended'] or info['reconnected']:
                segs.abandon(); env.wait_new_round(); break
            r, done, outcome = segs.step(msg.obs[:3], info['fell'], not on_floor, False, env.time_left_s())
            h = out['h_next']
            crop, vec, on_floor = obs_b.build(env.msg.obs, goal)
            if done:
                results.append(segs.history[-1]); n = len(results)
                if n % 10 == 0:
                    print(f'  {n}/{N_SEGMENTS}: {segs.stats(last=n)}')
                break
    s = segs.stats(last=N_SEGMENTS)
    t10 = np.mean([x['decisions'] * 0.064 / max(x['travelled'], 1e-3) * 10 for x in results if x['outcome'] == 'arrived']) if s['arrive_pct'] > 0 else float('nan')
    out = {'map': mission, 'ckpt': os.path.basename(ckpt), 'update': ck.get('update'), 'n': len(results), 'seed': EVAL_SEED,
           'arrival_pct': s['arrive_pct'], 'falls_per_100u': s['falls_per_100u'], 'speed': s['speed'], 'timeout_pct': s['timeout_pct'],
           't_per_10u_s': t10, 'when': datetime.now().isoformat()}
    os.makedirs(os.path.join(HERE, 'logs', 'nav'), exist_ok=True)
    p = os.path.join(HERE, 'logs', 'nav', f'eval_{mission}_{os.path.splitext(os.path.basename(ckpt))[0]}.json')
    json.dump(out, open(p, 'w'), indent=1)
    print('EVAL', json.dumps(out)); print('wrote', p)
    env.close()


if __name__ == '__main__':
    main()
