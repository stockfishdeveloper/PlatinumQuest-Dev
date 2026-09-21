"""Play a REAL Hunt round: goals come from the game's actual gems, not synthetic waypoints.

This is the transfer test. Everything the navigator has been trained on is a waypoint sampled
from the terrain grid, reached from a teleport, with arrival judged by our own radius check. A
real round has none of that: gems spawn where the game puts them, play is continuous, and a
pickup is whatever the game says it is (gemDelta). Nothing here trains or changes the policy.

    NAV_PORT=8920 python -m nav.real_run              (spare port; training holds 8888..8895)
    NAV_CKPT=models/nav/nav_005750.pth NAV_PORT=8920 python -m nav.real_run
    NAV_ROUNDS=3 NAV_PORT=8920 python -m nav.real_run

Gem positions come from RAW_GEMS as (dx, dy, dz) from the marble. They are described as
camera-relative, but mlAgent.cs pins $AIObserver::ForceYaw = 0, so the observation frame IS the
world frame and the world position is simply marble + delta. Only the nearest 5 are sent, which
is about one gem group -- there is no long-range view, so this cannot plan past the current
cluster. That limit is the point of measuring.

Reports gems/min, falls per 100 u, speed and how often no gem was visible, against the human
demo. Writes logs/nav/real_run_<map>_<ckpt>.json.
"""
import os
import sys
import json
import math
import numpy as np
import torch
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from terrain_obs import TerrainMap                                   # noqa: E402
from nav.protocol import RAW_GEMS, RAW_VEL                           # noqa: E402
from nav.env import HuntEnv                                          # noqa: E402
from nav.terrain import TerrainGrid                                  # noqa: E402
from nav.obs import ObsBuilder, NAV_OBS_VERSION                      # noqa: E402
from nav.model import NavActorCritic, action_to_joystick             # noqa: E402

PORT = int(os.environ.get('NAV_PORT', '8920'))
GAME_SPEED = 3
ROUNDS = int(os.environ.get('NAV_ROUNDS', '1'))
ABSENT = -500.0                # RAW_GEMS pads missing slots with value/dist <= -500
STICKY_TOL = 0.75              # u: a gem within this of the current target counts as the same gem
SWITCH_GAIN = 0.60             # only abandon the current target for one at most this fraction of
                               # its distance -- without hysteresis the choice flaps between two
                               # near-equal gems every tick and the navigator is handed a new
                               # heading each decision
VALUE_WEIGHT = float(os.environ.get('NAV_VALUE_WEIGHT', '0'))   # 0 = pure nearest; >0 prefers
                               # higher-value gems (yellow), cost = dist / value**VALUE_WEIGHT

# Human demo reference, measured from demos/demo_20260914_214854.npz (68,208 ticks at 16 ms =
# 18.2 min over 6 rounds, 682 pickups, 861 points -> 1.26 points per gem). Hunt is scored on
# POINTS, so gemDelta is a point delta, not a gem count; both are tracked separately below.
HUMAN = {'gems_per_min': 37.5, 'points_per_min': 47.3, 'speed': 8.05, 'falls_per_100u': 0.046}
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def visible_gems(raw):
    """Real gems as (world_x, world_y, world_z, value, dist), nearest first, absent slots dropped."""
    px, py, pz = float(raw[0]), float(raw[1]), float(raw[2])
    g = np.asarray(raw[RAW_GEMS], dtype=np.float64).reshape(5, 5)
    out = []
    for dx, dy, dz, value, dist in g:
        if value <= ABSENT or dist <= ABSENT:
            continue
        out.append((px + dx, py + dy, pz + dz, float(value), float(dist)))
    out.sort(key=lambda t: t[4])
    return out


def choose(gems, current):
    """Pick the gem to chase, and the one after it for the next-gem observation block.

    Sticky: keep the current target while it is still on the list, unless another is much closer
    (SWITCH_GAIN). Returns (target, next_target), either may be None.
    """
    if not gems:
        return None, None

    def cost(g):
        if VALUE_WEIGHT > 0 and g[3] > 0:
            return g[4] / (g[3] ** VALUE_WEIGHT)
        return g[4]

    ranked = sorted(gems, key=cost)
    best = ranked[0]
    nxt = ranked[1] if len(ranked) > 1 else None
    if current is None:
        return best, nxt
    still = [g for g in gems if math.hypot(g[0] - current[0], g[1] - current[1]) < STICKY_TOL
             and abs(g[2] - current[2]) < 2.0]
    if not still:
        return best, nxt                       # the gem we were chasing is gone (collected/expired)
    held = still[0]
    if cost(best) < SWITCH_GAIN * cost(held):
        return best, nxt
    other = [g for g in ranked if g is not held]
    return held, (other[0] if other else None)


def main():
    ckpt = os.environ.get('NAV_CKPT', os.path.join(HERE, 'models', 'nav', 'nav_latest.pth'))
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(ckpt, map_location=dev)
    assert ck.get('obs_version') == NAV_OBS_VERSION, f'checkpoint is {ck.get("obs_version")}'
    model = NavActorCritic().to(dev); model.load_state_dict(ck['model']); model.eval()

    env = HuntEnv(PORT, speed=GAME_SPEED); env.connect()
    mission = env.info['mission']
    terrain = TerrainGrid(TerrainMap.resolve(mission))
    obs_b = ObsBuilder(terrain)
    print(f'real run: {os.path.basename(ckpt)} (update {ck.get("update")}) on {mission}, '
          f'{ROUNDS} round(s), port {PORT}')

    rounds = []
    trace = open(os.path.join(HERE, 'logs', 'nav', 'real_trace.csv'), 'w', buffering=1)
    trace.write('round,dec,x,y,z,vx,vy,speed,on_floor,tx,ty,tdist,nvis,gem,fell,fwd,back,left,right,jump\n')
    for r in range(ROUNDS):
        obs_b.reset(); h = model.initial_state(1, dev)
        target = None
        gems = falls = decisions = blind = 0
        points = 0.0
        travelled = 0.0
        last = np.array(env.msg.obs[:3], dtype=np.float64)
        t_start = env.time_left_s()          # recorded only; see the note on mins below
        while True:
            vis = visible_gems(env.msg.obs)
            target, nxt = choose(vis, target)
            if target is None:
                blind += 1
                # nothing in the 5-gem window: hold still rather than wander off an edge
                msg, info = env.step((0.0, 0.0, 0.0, 0.0, 0))
                if info['round_ended'] or info['reconnected']:
                    break
                decisions += 1
                continue
            goal = (target[0], target[1], target[2])
            ngoal = (nxt[0], nxt[1], nxt[2]) if nxt else None
            crop, vec, on_floor = obs_b.build(env.msg.obs, goal, ngoal)
            out = model.act(torch.as_tensor(crop, device=dev).unsqueeze(0),
                            torch.as_tensor(vec, device=dev).unsqueeze(0), h, deterministic=True)
            a = out['action_game'][0].tolist(); vel = env.msg.obs[RAW_VEL]
            msg, info = env.step(action_to_joystick(a[0], a[1], a[2], a[3], a[4],
                                                    float(vel[0]), float(vel[1])))
            h = out['h_next']
            if info['round_ended'] or info['reconnected']:
                break
            decisions += 1
            p = np.array(msg.obs[:3], dtype=np.float64)
            step_d = float(np.linalg.norm(p[:2] - last[:2]))
            if step_d < 1.5:                      # ignore respawn / teleport jumps
                travelled += step_d
            last = p
            js = action_to_joystick(a[0], a[1], a[2], a[3], a[4], float(vel[0]), float(vel[1]))
            trace.write(f'{r+1},{decisions},{p[0]:.2f},{p[1]:.2f},{p[2]:.2f},{msg.obs[3]:.2f},{msg.obs[4]:.2f},'
                        f'{math.hypot(float(msg.obs[3]), float(msg.obs[4])):.2f},{int(on_floor)},'
                        f'{target[0]:.1f},{target[1]:.1f},'
                        f'{math.hypot(target[0]-p[0], target[1]-p[1]):.2f},{len(vis)},'
                        f'{info["gem_delta"]:.0f},{int(info["fell"])},'
                        f'{js[0]:.2f},{js[1]:.2f},{js[2]:.2f},{js[3]:.2f},{js[4]}\n')
            if info['gem_delta'] > 0:
                gems += 1                         # one pickup...
                points += float(info['gem_delta'])   # ...worth this many points (red 1, yellow more)
                target = None                     # collected: pick the next one
            if info['fell']:
                falls += 1
                target = None
                obs_b.reset(); h = model.initial_state(1, dev)   # respawn = a fresh start
        # NOT from the round clock: at round end it already reads the NEXT round's full time, so
        # (t_start - time_left) is 0. Decisions are the reliable clock -- one per 64 ms of sim.
        mins = max(decisions * 0.064 / 60.0, 1e-6)
        row = {'round': r + 1, 'gems': gems, 'points': round(points, 1), 'minutes': round(mins, 2),
               'decisions': decisions,
               'gems_per_min': round(gems / mins, 1), 'points_per_min': round(points / mins, 1),
               'falls': falls,
               'falls_per_100u': round(100.0 * falls / max(travelled, 1e-6), 3),
               'travelled_u': round(travelled, 1),
               'speed': round(travelled / max(decisions * 0.064, 1e-6), 2),
               'blind_pct': round(100.0 * blind / max(decisions, 1), 1)}
        rounds.append(row)
        print(f'  round {r+1}: {row}')
        if r + 1 < ROUNDS:
            env.wait_new_round()

    trace.close()
    agg = {k: round(float(np.mean([x[k] for x in rounds])), 3)
           for k in ('gems_per_min', 'points_per_min', 'falls_per_100u', 'speed', 'blind_pct')}
    out = {'map': mission, 'ckpt': os.path.basename(ckpt), 'update': ck.get('update'),
           'rounds': rounds, 'mean': agg, 'human': HUMAN,
           'vs_human': {'gems_per_min': round(agg['gems_per_min'] / HUMAN['gems_per_min'], 2),
                        'points_per_min': round(agg['points_per_min'] / HUMAN['points_per_min'], 2),
                        'speed': round(agg['speed'] / HUMAN['speed'], 2),
                        'falls_ratio': round(agg['falls_per_100u'] / HUMAN['falls_per_100u'], 1)},
           'value_weight': VALUE_WEIGHT, 'when': datetime.now().isoformat()}
    os.makedirs(os.path.join(HERE, 'logs', 'nav'), exist_ok=True)
    p = os.path.join(HERE, 'logs', 'nav',
                     f'real_run_{mission}_{os.path.splitext(os.path.basename(ckpt))[0]}.json')
    json.dump(out, open(p, 'w'), indent=1)
    print('REAL RUN', json.dumps(out['mean']), 'vs human', json.dumps(out['vs_human']))
    print('wrote', p)
    env.close()


if __name__ == '__main__':
    main()
