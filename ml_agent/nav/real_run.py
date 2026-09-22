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
import time
import json
import math
import numpy as np
import torch
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from terrain_obs import TerrainMap                                   # noqa: E402
from nav.protocol import RAW_GEMS, RAW_VEL                           # noqa: E402
from nav.env import HuntEnv, OBS_MS                                  # noqa: E402
from nav.terrain import TerrainGrid                                  # noqa: E402
from nav.obs import ObsBuilder, NAV_OBS_VERSION, VEC_DIM             # noqa: E402
from nav.terrain import CROP_SHAPE                                   # noqa: E402
from nav.model import NavActorCritic, action_to_joystick             # noqa: E402

PORT = int(os.environ.get('NAV_PORT', '8920'))
GAME_SPEED = int(os.environ.get('NAV_SPEED', '3'))   # NAV_SPEED=1 to watch it play in real time
FORCE_THROTTLE = os.environ.get('NAV_FORCE_THROTTLE', '0') == '1'   # peg output strength to 100 %:
                               # the policy still chooses DIRECTION, but throttle is pinned at 1.0
                               # whatever it asks for. Note the model already runs at a measured
                               # mean throttle of 0.99 (THROTTLE_FLOOR = 0.90 caps how low it can
                               # go), so this is expected to be close to a null; it is run to find
                               # out whether the residual 1 % is hiding anything.
ROUNDS = int(os.environ.get('NAV_ROUNDS', '1'))
ABSENT = -500.0                # RAW_GEMS pads missing slots with value/dist <= -500
STICKY_TOL = 0.75              # u: a gem within this of the current target counts as the same gem
SWITCH_GAIN = 0.60             # only abandon the current target for one at most this fraction of
                               # its distance -- without hysteresis the choice flaps between two
                               # near-equal gems every tick and the navigator is handed a new
                               # heading each decision
H_RESET = os.environ.get('NAV_H_RESET', 'never')   # 'never' | 'pickup' | '<N>' decisions.
                               # Training resets the recurrent state every segment (~230 decisions);
                               # a real round runs ~2800 unbroken, far outside that distribution. The
                               # marble moves at 2.2 u/s in real rounds vs 4.1-4.9 in training, so this
                               # is the prime suspect. 'pickup' mimics a training segment boundary.
SMOOTH = float(os.environ.get('NAV_SMOOTH', '1.0'))   # EMA on the commanded DIRECTION.
                               # 1.0 = raw policy output, which is where this should stay.
                               # STALE RATIONALE, KEPT FOR HISTORY: this used to read "the policy
                               # flips its heading a median 39.5 deg every decision and sustains a
                               # driving direction for only 1 decision (0.06 s); the human demo
                               # changes heading 0.4 deg and sustains 10 decisions (0.64 s)".
                               # BOTH HALVES ARE NOW WRONG. The human figure was later remeasured
                               # at 2.67 deg MEAN over 60,836 demo decisions (the 0.4 was an
                               # error), and after DIR_GOAL_GAIN = 30 the agent holds its heading
                               # inside 20 deg for a median of 14 decisions = 0.90 s against the
                               # human's 46 ticks = 0.74 s, i.e. it now holds a line LONGER than
                               # the human. Sustain is no longer a defect and smoothing is not the
                               # fix for anything. See HANDOFF section 23 for what the real
                               # acceleration defect is (thrust 57-63 deg off the marble's own
                               # velocity in the 9-18 u band, where the human holds 25-31 deg).
VIEW_SUBSTEPS = int(os.environ.get('NAV_VIEW_SUBSTEPS', '1'))   # WATCH only: split each 64 ms
                               # decision into this many sim slices so motion renders smoothly.
                               # 1 = exactly as trained (jagged, 15.6 Hz). 4 = ~62 Hz, viewing only.
WATCH = os.environ.get('NAV_WATCH', '0') == '1'    # real-time viewing. Lockstep + FIXEDSTEP mean the
                               # sim advances as fast as we reply, and set_speed also sends
                               # RENDEREVERY 100 -- so NAV_SPEED alone does NOT give real time.
                               # WATCH renders every frame and paces the loop to 64 ms/decision,
                               # leaving physics and decisions bit-identical to a fast run.
MARK_GOAL = os.environ.get('NAV_MARK', '0') == '1'   # NAV_MARK=1 drops a black marker gem on the
                               # current goal so a human watching can see what it is steering at
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


def snap_to_walkable(terrain, gx, gy, gz, radius=3):
    """Real gems can sit on cells our terrain grid calls non-walkable (thin ledges, bevels, or
    just rasterisation error at 1 u). Handing the navigator such a goal makes it stall: it reads
    the goal as beyond an edge and refuses to close. Measured cost, 2026-09-19: one such gem ate
    1165 decisions -- 42 % of a round -- and was never collected, and because a group must be
    cleared before the next spawns, it blocks the round outright.

    So path to the nearest cell the grid DOES call walkable. The marble ends up adjacent to the
    gem and the game's own pickup radius finishes the job. Returns the goal to steer at.
    """
    j, i = terrain.cell_of(gx, gy)
    if terrain.in_walk_grid(j, i) and terrain.walkable[j, i]:
        return (gx, gy, gz), False
    best = None
    for dj in range(-radius, radius + 1):
        for di in range(-radius, radius + 1):
            jj, ii = j + dj, i + di
            if not terrain.in_walk_grid(jj, ii) or not terrain.walkable[jj, ii]:
                continue
            wx, wy = float(terrain.wxs[ii]), float(terrain.wys[jj])
            d = math.hypot(wx - gx, wy - gy)
            if best is None or d < best[0]:
                best = (d, wx, wy, float(terrain.walk_top[jj, ii]))
    if best is None:
        return (gx, gy, gz), False
    return (best[1], best[2], best[3]), True


DIRECT_GEM = os.environ.get('NAV_DIRECT_GEM', '1') == '1'   # default ON since 2026-09-20 23:50 by
                               # user instruction: the waypoint is the gem's exact position, always.
                               # Routing corner points and snap_to_walkable are bypassed. The
                               # 2026-09-19 measurement below is why routing existed; the cost of
                               # removing it is to be re-measured on the current policy.
                               # Historical note on the OFF setting:
                               # Steering straight at the gem is right ONLY when the line to it is
                               # clear; path_waypoint already returns the gem itself in that case.
                               # With routing removed entirely the marble drives at gems that sit
                               # behind holes, stops at the rim 2-6 u out and jitters there until
                               # the round ends -- measured 2026-09-19: 714 s of stalling across
                               # 8 rounds, 50 % of all decisions, target never changing because the
                               # unreachable gem stays the nearest one. Score 62.1 -> 27.4.
                               # The TRAINING goal pool is a separate thing and does use exact gem
                               # coordinates (nav/terrain.py gem_spawns).
GAP_EXCLUDE_U = 6.0            # between groups: spawn points within this of the marble are the group
                               # it just cleared and are left out of the centroid it heads for
WAYPOINT_HOLD_U = 2.5          # keep steering at the SAME waypoint until this close to it, so the
                               # string-pulled point cannot snap forward mid-approach and hand the
                               # policy a new heading (see the hold logic in main)
GEM_SNAP_U = 1.5               # a routing waypoint this close to the gem IS the gem. Path points are
                               # walk-cell centres and KOTM's gems sit on cell corners, so the last
                               # path cell is 0.71 u from the gem; handing that over put the marker
                               # (and the steering) beside the gem instead of on it (2026-09-20).
LOOKAHEAD_U = 12.0             # how far along the path the navigator is aimed. Training goals were
                               # 8-40 u (first gem) and 6-22 u (group links), so this keeps the goal
                               # in-distribution instead of pointing at something 30 u away.


def line_clear(terrain, ax, ay, bx, by, step=0.35):
    """True if every sample on the straight line ax,ay -> bx,by is walkable ground."""
    d = math.hypot(bx - ax, by - ay)
    n = max(int(d / step), 1)
    for k in range(n + 1):
        x = ax + (bx - ax) * k / n
        y = ay + (by - ay) * k / n
        j, i = terrain.cell_of(x, y)
        if not terrain.in_walk_grid(j, i) or not terrain.walkable[j, i]:
            return False
    return True


def path_waypoint(terrain, mx, my, field, gem, lookahead=LOOKAHEAD_U):
    """Aim the navigator at the furthest point on the path it can reach IN A STRAIGHT LINE.

    The navigator only receives a direction and distance to its goal plus local terrain rays --
    it has no route. Walking `lookahead` units along the Dijkstra path is not enough, because the
    path curves around a hole while the straight line to its end still crosses that hole: measured
    2026-09-19, the marble sat at (-13.5,18.3) with a goal at (-23.7,13.5) whose straight line hit
    the void 2.3 u ahead, and oscillated there for 2234 decisions.

    So walk the path, then string-pull: take the FURTHEST path point with clear line of sight from
    the marble. The heading handed to the policy is then always traversable ground.
    """
    j, i = terrain.cell_of(mx, my)
    if not terrain.in_walk_grid(j, i) or not np.isfinite(field[j, i]):
        c = terrain._nearest_walkable(j, i, radius=3)
        if c is None or not np.isfinite(field[c]):
            return (gem[0], gem[1], gem[2]), False
        j, i = c
    if field[j, i] <= terrain.walk_res * 2 or line_clear(terrain, mx, my, gem[0], gem[1]):
        return (gem[0], gem[1], gem[2]), False        # the gem itself is directly reachable

    path = []
    travelled = 0.0
    for _ in range(int((lookahead * 3) / terrain.walk_res) + 8):
        best = None
        for dj in (-1, 0, 1):
            for di in (-1, 0, 1):
                if dj == 0 and di == 0:
                    continue
                jj, ii = j + dj, i + di
                if not terrain.in_walk_grid(jj, ii) or not terrain.walkable[jj, ii]:
                    continue
                f = field[jj, ii]
                if np.isfinite(f) and (best is None or f < best[0]):
                    best = (f, jj, ii, math.hypot(dj, di) * terrain.walk_res)
        if best is None or best[0] >= field[j, i]:
            break
        travelled += best[3]
        j, i = best[1], best[2]
        path.append((float(terrain.wxs[i]), float(terrain.wys[j]), float(terrain.walk_top[j, i])))
        if field[j, i] <= terrain.walk_res or travelled >= lookahead * 3:
            break
    if not path:
        return (gem[0], gem[1], gem[2]), False
    # string-pull: furthest point on the path that is a straight clear shot from here
    pick = None
    for wp in path:
        if math.hypot(wp[0] - mx, wp[1] - my) > lookahead:
            break
        if line_clear(terrain, mx, my, wp[0], wp[1]):
            pick = wp
    if pick is None:
        pick = path[0]                                # nothing visible: take the next path cell
    if math.hypot(pick[0] - gem[0], pick[1] - gem[1]) <= GEM_SNAP_U:
        return (gem[0], gem[1], gem[2]), False        # beside the gem: aim at the gem itself
    return pick, True


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
    # THE LISTENING SOCKET OPENS FIRST, before the checkpoint, CUDA or the terrain grid.
    # Python is the SERVER and the game is the client: the game dials us 100 ms after "GO!"
    # (mlAgent.cs:677-686), so if the port is not already open the marble sits on the start pad
    # until we bind. Constructing HuntEnv binds and listens (nav/env.py:51-53); connect() only
    # accepts. Ready-at-GO is a hard rule for every run path (2026-09-21).
    env = HuntEnv(PORT, speed=GAME_SPEED)
    ckpt = os.environ.get('NAV_CKPT', os.path.join(HERE, 'models', 'nav', 'nav_latest.pth'))
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(ckpt, map_location=dev)
    assert ck.get('obs_version') == NAV_OBS_VERSION, f'checkpoint is {ck.get("obs_version")}'
    model = NavActorCritic().to(dev); model.load_state_dict(ck['model']); model.eval()

    # Everything expensive happens BEFORE we connect. The first model.act() pays CUDA context
    # creation and cuDNN algorithm selection -- seconds of wall clock -- and if that lands after
    # the connect, the lockstep safety timeout (5 s) fires and the game free-runs while the
    # marble sits there. NAV_MAP lets the terrain grid and its Dijkstra graph be built up front
    # too; without it we fall back to asking the game and rebuilding.
    pre_map = os.environ.get('NAV_MAP', '')
    terrain = obs_b = None
    if pre_map:
        terrain = TerrainGrid(TerrainMap.resolve(pre_map)); obs_b = ObsBuilder(terrain)
    _t0 = time.perf_counter()
    with torch.no_grad():                     # warm the kernels on a throwaway observation
        model.act(torch.zeros(1, *CROP_SHAPE, device=dev),
                  torch.zeros(1, VEC_DIM, device=dev),
                  model.initial_state(1, dev), deterministic=True)
    print(f'warm-up: model ready in {time.perf_counter() - _t0:.2f} s'
          + (f', terrain "{pre_map}" pre-built' if pre_map else ''))

    env.connect()                      # already bound and listening: this only accepts
    if WATCH:
        env.control('RENDEREVERY 1')       # draw every frame so a human can actually see it
        vy = os.environ.get('NAV_VIEWYAW', '0')   # fixed view yaw in radians; 'off' = follow the marble camera
        env.control(f'VIEWYAW {vy}')       # stable view while the marble camera rotates for force (engine: Marble.setViewYaw)
        if VIEW_SUBSTEPS > 1:
            # The sim normally advances in ONE 64 ms jump per decision (gAIFixedStepMs), i.e. 15.6
            # motion updates a second -- the renderer just redraws that frozen state, which is why
            # a 1250 fps counter still looks jagged. Split the same 64 ms into VIEW_SUBSTEPS
            # slices and repeat the action across them: identical decisions every 64 ms, but the
            # motion renders at 64/VIEW_SUBSTEPS ms. VIEWING ONLY -- the integration step differs
            # from training, so do not measure with this on.
            env.control(f'FIXEDSTEP {max(1, OBS_MS // VIEW_SUBSTEPS)}')
        print(f'WATCH mode: every frame drawn, {VIEW_SUBSTEPS} sim slice(s) per 64 ms decision, real time')
    mission = env.info['mission']
    if terrain is None or pre_map != mission:
        if pre_map and pre_map != mission:
            print(f'NAV_MAP was "{pre_map}" but the game is on "{mission}"; rebuilding terrain')
        terrain = TerrainGrid(TerrainMap.resolve(mission)); obs_b = ObsBuilder(terrain)
    print(f'real run: {os.path.basename(ckpt)} (update {ck.get("update")}) on {mission}, '
          f'{ROUNDS} round(s), port {PORT}')

    rounds = []
    trace = open(os.path.join(HERE, 'logs', 'nav', 'real_trace.csv'), 'w', buffering=1)
    trace.write('round,dec,x,y,z,vx,vy,speed,on_floor,tx,ty,tdist,nx,ny,gx,gy,nvis,gem,fell,fwd,back,left,right,jump\n')
    for r in range(ROUNDS):
        obs_b.reset(); h = model.initial_state(1, dev)
        target = None
        gems = falls = decisions = blind = n_snapped = n_pathed = 0
        marked = None
        smooth_dir = None
        field = None; field_key = None; held_goal = None
        gap_goal = None                  # where to head while no gem is on the map (held per gap)
        points = 0.0
        travelled = 0.0
        last = np.array(env.msg.obs[:3], dtype=np.float64)
        t_start = env.time_left_s()          # recorded only; see the note on mins below
        next_tick = [time.perf_counter()]

        def paced_step(js):
            """One sim slice, paced to wall clock. Pacing must be PER SLICE: sleeping once per
            decision and then firing all VIEW_SUBSTEPS slices back to back renders them as a
            burst followed by a stall, which looks exactly as jagged as no smoothing at all."""
            out = env.step(js)
            if WATCH:
                next_tick[0] += 0.064 / max(VIEW_SUBSTEPS, 1)
                slack = next_tick[0] - time.perf_counter()
                if slack > 0:
                    time.sleep(slack)
                else:
                    next_tick[0] = time.perf_counter()
            return out

        while True:
            vis = visible_gems(env.msg.obs)
            target, nxt = choose(vis, target)
            if target is not None:
                gap_goal = None
            else:
                # No gem on the map: the group is cleared and the next has not spawned yet.
                #
                # THIS PATH IS NOW NEARLY DEAD, AND THE NUMBERS BELOW WERE A BUG, NOT THE GAME.
                # The quoted "KOTM: median 1.15 s, 33 times a run, 10.3 % of all decisions" and
                # FlatGem's 31 % were caused by the observer reading a stale client-side cache
                # (HANDOFF section 22, $AIObserver::GemSource). The server replaces a collected gem
                # synchronously, so these windows were almost entirely fictional. With the fix,
                # blind_pct measured 0.0 on all 16 rounds across KOTM and FlatGem. Kept because a
                # genuine spawn gap is still possible, but if blind_pct is ever non-trivial again,
                # suspect the observation pipeline FIRST rather than tuning this fallback. Two earlier
                # fallbacks were both wrong. Commanding ZERO left the marble dead for the gap.
                # Steering at the NEAREST spawn point sent it back to the spot it had just
                # cleared: it reversed, drove onto empty floor, and was doing 3.5 u/s in a random
                # direction when the group landed a median 14.7 u away (run mean 5.96 u/s).
                # The game puts the next group FAR away on purpose: getCenterGems blocks
                # 2 * radiusFromGem (30 u here) round the last centre and falls back to the
                # furthest sampled gem. Measured on 33 gaps, the marble's own heading at the
                # pickup pointed 92.9 deg from the next target, the centroid of the remaining
                # spawn points pointed 13.5 deg from it, and 5 u toward that centroid gained
                # 4.4 u on the new target. So head for that centroid and HOLD it until gems
                # reappear, which also keeps this to one Dijkstra field per gap.
                blind += 1
                if getattr(terrain, 'gem_spawns', None):
                    mx, my = float(env.msg.obs[0]), float(env.msg.obs[1])
                    if gap_goal is None:
                        cand = [q for q in terrain.gem_spawns
                                if math.hypot(q[0] - mx, q[1] - my) > GAP_EXCLUDE_U] or list(terrain.gem_spawns)
                        cx = sum(q[0] for q in cand) / len(cand)
                        cy = sum(q[1] for q in cand) / len(cand)
                        near = min(cand, key=lambda q: math.hypot(q[0] - cx, q[1] - cy))
                        g, _ = snap_to_walkable(terrain, cx, cy, near[2])   # nearest floor to the centroid
                        j, i = terrain.cell_of(g[0], g[1])
                        if terrain.in_walk_grid(j, i) and terrain.walkable[j, i]:
                            gap_goal = (g[0], g[1], float(terrain.walk_top[j, i]))
                        else:                         # no floor within 3 cells: nearest real spawn point
                            gap_goal = (near[0], near[1], near[2])
                    target = (gap_goal[0], gap_goal[1], gap_goal[2], 1.0,
                              math.hypot(gap_goal[0] - mx, gap_goal[1] - my))
                    nxt = None
                else:
                    msg, info = paced_step((0.0, 0.0, 0.0, 0.0, 0))
                    if info['round_ended'] or info['reconnected']:
                        break
                    decisions += 1
                    continue
            if DIRECT_GEM:
                gem_goal, snapped = (target[0], target[1], target[2]), False
            else:
                gem_goal, snapped = snap_to_walkable(terrain, target[0], target[1], target[2])
            if snapped:
                n_snapped += 1
            key = (round(gem_goal[0], 2), round(gem_goal[1], 2))
            if key != field_key:                          # Dijkstra is expensive: one per target
                field = terrain.goal_field(gem_goal[0], gem_goal[1]); field_key = key
                held_goal = None                          # new target: the old waypoint is stale
            mx, my = float(env.msg.obs[0]), float(env.msg.obs[1])
            # HOLD the waypoint until it is reached or no longer reachable in a straight line.
            # Recomputing the string-pulled point every decision let it snap forward whenever
            # line-of-sight cleared round a corner, and the policy reads that snap as a heading
            # change: steering swung 84.6 deg on decisions where the goal jumped >3 u, against
            # 63.8 deg otherwise. A waypoint should be a fixed point you drive at.
            if DIRECT_GEM:
                goal, pathed = gem_goal, False     # the goal IS the gem, never a point beside it
            elif (held_goal is not None
                    and math.hypot(held_goal[0] - mx, held_goal[1] - my) > WAYPOINT_HOLD_U
                    and line_clear(terrain, mx, my, held_goal[0], held_goal[1])):
                goal, pathed = held_goal, True
            else:
                goal, pathed = path_waypoint(terrain, mx, my, field, gem_goal)
                held_goal = goal if pathed else None
            if pathed:
                n_pathed += 1
            ngoal = ((nxt[0], nxt[1], nxt[2]) if DIRECT_GEM else
                     snap_to_walkable(terrain, nxt[0], nxt[1], nxt[2])[0]) if nxt else None
            if MARK_GOAL:
                if len(vis) > 0:
                    # the goal IS a gem (DIRECT_GEM): the marker sits exactly on it
                    if marked is None or math.hypot(goal[0]-marked[0], goal[1]-marked[1]) > 0.25:
                        env.mark(goal[0], goal[1], goal[2])
                        marked = goal
                elif marked is not None:
                    env.mark_off()                        # no gem on the map: draw nothing anywhere
                    marked = None
            crop, vec, on_floor = obs_b.build(env.msg.obs, goal, ngoal)
            out = model.act(torch.as_tensor(crop, device=dev).unsqueeze(0),
                            torch.as_tensor(vec, device=dev).unsqueeze(0), h, deterministic=True)
            a = out['action_game'][0].tolist(); vel = env.msg.obs[RAW_VEL]
            if SMOOTH < 1.0:
                n = math.hypot(a[0], a[1])
                if n > 1e-6:
                    ux, uy = a[0] / n, a[1] / n
                    if smooth_dir is None:
                        smooth_dir = (ux, uy)
                    else:
                        sx = (1 - SMOOTH) * smooth_dir[0] + SMOOTH * ux
                        sy = (1 - SMOOTH) * smooth_dir[1] + SMOOTH * uy
                        sn = math.hypot(sx, sy)
                        smooth_dir = (sx / sn, sy / sn) if sn > 1e-6 else smooth_dir
                    a[0], a[1] = smooth_dir
            if FORCE_THROTTLE:
                a[2] = 1.0                        # direction is the policy's, strength is pegged
            js_cmd = action_to_joystick(a[0], a[1], a[2], a[3], a[4], float(vel[0]), float(vel[1]))
            msg, info = paced_step(js_cmd)
            for _ in range(VIEW_SUBSTEPS - 1):     # same action across the remaining slices
                msg, inf2 = paced_step(js_cmd)
                info['gem_delta'] += inf2['gem_delta']
                info['fell'] = info['fell'] or inf2['fell']
                info['round_ended'] = info['round_ended'] or inf2['round_ended']
                info['reconnected'] = info['reconnected'] or inf2['reconnected']
                if info['round_ended'] or info['reconnected']:
                    break
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
                        f'{math.hypot(target[0]-p[0], target[1]-p[1]):.2f},'
                        f'{(nxt[0] if nxt else 0):.1f},{(nxt[1] if nxt else 0):.1f},'
                        f'{goal[0]:.1f},{goal[1]:.1f},{len(vis)},'
                        f'{info["gem_delta"]:.0f},{int(info["fell"])},'
                        f'{js[0]:.2f},{js[1]:.2f},{js[2]:.2f},{js[3]:.2f},{js[4]}\n')
            if info['gem_delta'] > 0:
                gems += 1                         # one pickup...
                points += float(info['gem_delta'])   # ...worth this many points (red 1, yellow more)
                target = None                     # collected: pick the next one
                if H_RESET == 'pickup':
                    obs_b.reset(); h = model.initial_state(1, dev)
            elif H_RESET.isdigit() and decisions % int(H_RESET) == 0:
                obs_b.reset(); h = model.initial_state(1, dev)
            if info['fell']:
                falls += 1
                target = None
                obs_b.reset(); h = model.initial_state(1, dev)   # respawn = a fresh start
        # NOT from the round clock: at round end it already reads the NEXT round's full time, so
        # (t_start - time_left) is 0. Decisions are the reliable clock -- one per 64 ms of sim.
        # OBS_MS, not a hardcoded 0.064: at a finer decision rate the decision count doubles
        # and a hardcoded step made speed and gems/min read HALF their true value (found
        # 2026-09-20 during the 32 ms eval, where 5.78 u/s was reported as 2.89).
        mins = max(decisions * (OBS_MS / 1000.0) / 60.0, 1e-6)
        row = {'round': r + 1, 'gems': gems, 'points': round(points, 1), 'minutes': round(mins, 2),
               'decisions': decisions,
               'gems_per_min': round(gems / mins, 1), 'points_per_min': round(points / mins, 1),
               'falls': falls,
               'falls_per_100u': round(100.0 * falls / max(travelled, 1e-6), 3),
               'travelled_u': round(travelled, 1),
               'speed': round(travelled / max(decisions * (OBS_MS / 1000.0), 1e-6), 2),
               'blind_pct': round(100.0 * blind / max(decisions, 1), 1),
               'snapped_pct': round(100.0 * n_snapped / max(decisions, 1), 1),
               'pathed_pct': round(100.0 * n_pathed / max(decisions, 1), 1),
               'smooth': SMOOTH}
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
           'value_weight': VALUE_WEIGHT, 'h_reset': H_RESET, 'when': datetime.now().isoformat()}
    os.makedirs(os.path.join(HERE, 'logs', 'nav'), exist_ok=True)
    p = os.path.join(HERE, 'logs', 'nav',
                     f'real_run_{mission}_{os.path.splitext(os.path.basename(ckpt))[0]}.json')
    json.dump(out, open(p, 'w'), indent=1)
    print('REAL RUN', json.dumps(out['mean']), 'vs human', json.dumps(out['vs_human']))
    print('wrote', p)
    env.close()


if __name__ == '__main__':
    main()
