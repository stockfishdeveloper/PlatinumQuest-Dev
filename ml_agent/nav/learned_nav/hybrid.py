"""Stage 5 (design M4): the PPO navigator drives, the planner takes over for floating gems and hands back.

    python -m nav.learned_nav.hybrid --port 9311 --map kotmjump --rounds 2 --memory current --tag hyb_current
    (then: marbleblast_mbx.exe -autotrain kotmjump -aiport 9311)

Ownership. The navigator (nav_latest.pth, played exactly as nav/real_run.py plays it: its gem chooser, the gem as the
goal, the next gem, the stuck-breaker, mean actions) drives while its target has a floor under it. When its target is
a FLOATING gem (no floor within 1.5 u below) the planner (planner.py) owns the marble: approach, run-up, jump, and after
the pickup the landing, until the marble has been on a floor LAND_HOLD decisions; then control goes back. Both always
see the real last reply (the game applies each reply one decision late).
Memory at handback (the design's comparison): 'current' keeps running the navigator on every observation while the
planner drives, its actions discarded, so its recurrent state is current when it takes over; 'reset' starts it from
a fresh state at handback.
Per round (logs/learned_nav/rounds/<tag>_<map>.jsonl): points, gems floor / floating, falls with their owner, falls
within 3 s after a handback, handovers, groups cleared, planning time.

Stage 5b additions (log section 33), each an arm switch for comparisons:
* --rescue 1 (default): the planner takes a floor gem after RESCUE_AFTER stuck-breaker firings on it;
* --shortcuts 1 (default): when the navigator's floor gem detours round a hole, the planner is consulted and takes
  over if a jump plan beats the navigator's fitted time (route.Route.legs) by SHORTCUT_MARGIN_S; 2 = consult and log
  only. After a planner pickup it rolls on toward the navigator's next gem and hands back once rolling toward it
  clear of edges (HANDBACK_*);
* --route 1 (off): jump legs the greedy chooser would not take (route.py); cost ~5 points in 8 rounds, off;
* --guard 1 (off): the fall guard (guard.py); harmful in its round (6 falls), off.
Also logged per round: every consultation (probe_log), every leg from gem choice to pickup (legs), the last 25
decisions before each fall (fall_log).
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
from nav.learned_nav.geometry import Geometry                                     # noqa: E402
from nav.learned_nav.session import Session, RoundOver, R                         # noqa: E402
from nav.learned_nav import planner as PL                                        # noqa: E402
from nav.learned_nav import seeds as SEEDS                                       # noqa: E402
from nav.learned_nav import drills as DR                                         # noqa: E402
from nav.learned_nav import route as RT                                          # noqa: E402
from nav.learned_nav import dynamics3 as D3                                      # noqa: E402
from nav.learned_nav import guard as GD                                          # noqa: E402
from nav.learned_nav.rounds import is_floating, same_gem, OUT_DIR, CONT_MAX, LAND_HOLD   # noqa: E402
from nav.gems import visible_gems, plan_tour                                     # noqa: E402
from nav.protocol import (RAW_POS, RAW_VEL, RAW_SPIN, NOOP_ACTION, RAW_POW_HELD, RAW_POW_BLAST, RAW_POW_SPECIAL,   # noqa: E402
                          RAW_POW_HELI_LEFT, RAW_POW_BOUNCE_LEFT, RAW_POW_SHOCK_LEFT)
from nav.learned_nav import powerup_physics as PW                                # noqa: E402
from nav.joystick import action_to_joystick as js_of                             # noqa: E402

AFTER_HANDBACK_S = 3.0       # a fall this soon after the navigator takes back over counts against the handover
HANDBACK_SPEED = 4.0         # u/s: the planner hands back once the marble is this slow (or at CONT_MAX); the first
                             # KOTM shortcut round handed back fast marbles next to holes: 7 falls within 3 s of handbacks
HANDBACK_CLEAR = 1.0         # ... or at any speed once it rolls toward the navigator's next gem (the walking distance to
HANDBACK_DOWNHILL = 0.3      # it drops this much over the next 0.5 u along the velocity) on a continuation plan that
                             # stays this far from every edge (braking to 4 u/s cost 6-25 decisions a shortcut)
RESCUE_AFTER = 2             # stuck-breaker firings on the same gem before the planner takes that (floor) gem over
                             # (2026-09-29: one reset round lost 140 s with the navigator stuck on a floor gem)
FLOOR_BROAD = 160            # the planner's broad proposals for a floor gem (as rounds.py)
RESCUE_MAX_S = 15.0          # a rescue the planner has not finished by then goes back to the navigator
# Shortcuts (the KOTM lever: the human makes 12-15 gap jumps a round, the navigator ~1.6): when the navigator's floor
# gem is within reach of a jump but its walking route detours round a hole, the planner is asked for a jump route
SHORTCUT_MAX_U = 18.0        # straight-line distance to the gem at most this (14 missed the human's 14.1 u crossings)
SHORTCUT_GAIN_U = 2.0        # the walking route (the planner's fine walking field) at least this much longer
SHORTCUT_MARGIN_S = 0.3      # the planner's pickup must come this much sooner than the navigator's predicted time
                             # (route.Route.legs: the navigator's time fitted on its KOTM legs)
SHORT_TIME_W = 0.015         # the planner's time weight among pickup plans on shortcut and route legs (planner.TIME_W)
SHORT_P_SAT = 0.6            # ... and its ranking saturation there (planner.P_SAT): stage 6b, the planner braked from 10 to
                             # 6 u/s before crossings for a 0.9 plan over a 0.6 one; P above 0.6 ranks by time alone
SHORT_FLIGHT_CHECK = 1.0     # ... and the flight-head launch check there (planner.flight_check; 0 = off, crossing drill v2)
PROBE_EVERY = 8              # decisions between two consultations for the same gem
SHORTCUT_ABORT = 8           # decisions without a pickup plan (on the floor) before a shortcut goes back
CONSULT_DEC = 8              # (3 until 10-01; log 35.6: crossings INTO the centre need 3-7 decisions before a plan appears) stage 6b (log 34.5): the planner is consulted by DRIVING for this many decisions right at the
                             # pickup, its state kept between them (a single cold question found a plan 1 time in 37; the
                             # crossing drill's continuous planning finds it at decision 1-2); no plan by then: hand back
CONSULT_AT_DEC = 2           # ... and only when the leg started this recently (a pickup state, what the drill measured)
BIG_GAIN_U = float('inf')    # OFF (log 36.5: g6r 141, g6s 147 against the navigator's 154; the planner's takeovers on
                             # big-detour legs were slower than the navigator's own legs with the oracle, and the
                             # continuations after its failed consults fell 1-5 times a round). 4.0 = a leg whose walk
                             # is this much longer than the line is a BIG detour (log 36.4): consulted
                             # at any decision of the leg where the heading is within SHORTCUT_MAX_ANGLE (not only at
                             # the pickup), re-consulted every PROBE_EVERY decisions, window CONSULT_DEC_BIG
CONSULT_DEC_BIG = 16         # decisions the planner gets on a big-detour consult (run-up + the plan: 1 s; 24 in g6r)
EXIT_TO_NEXT = True          # a crossing plan is charged the next leg's turn (planner.exit_dir toward the navigator's next gem)
ORACLE = os.environ.get('NAV_PLANNER_OFF') != '1'   # the planner as a jump oracle for the navigator (no takeover): log 35.4.
                             # NAV_PLANNER_OFF=1 (with --shortcuts 0 --rescue 0): the navigator alone, same chooser and
                             # checkpoint, for the A/B against the hybrid (operator 10-02, log 37.6)
ORACLE_P = 0.75              # P(success) of the best "jump now" program needed to set the navigator's jump bit
ORACLE_MIN_SPEED = 8.0       # u/s (g6m: 0.5 and 6 u/s gave 9.6 marginal hops a round, 152.0)
ORACLE_LIP_U = 3.0           # the void must begin within this along the velocity
ORACLE_ANGLE = 40.0          # degrees between the heading and the line to the gem
ACROSS_CONSULT = False       # consult on the gem across a hole the marble points at, even when not the nearest: OFF (log
                             # 35.2: 11 retargets a round, 70 % of the consults failed, falls x3, 137.9 points)
ACROSS_MAX_ANGLE = 20.0      # ... within this of the heading (drill: aligned crossings gain 0.3-0.5 s, 30-60 deg ~0)
HANDBACK_FAST = 9.0          # u/s: the instant handback after a failed consult needs the marble at most this fast (g6i: handed
                             # back clear but at 10-13 u/s, a fall within 3 s of a handback every round)
CONT_HARD_MAX = 90           # decisions after a planner pickup after which it hands back whatever the speed (5.8 s)
POW_ORACLE = True            # the planner as a powerup oracle for the navigator (POWERUP_PLAN phase 3, log 39)
POLICY_USE = True            # the navigator's own use bit (action element 5, phase 4, log 40) fires too
POW_ORACLE_EVERY = 4         # decisions between two questions (a question costs ~100 ms)
POW_GAIN = 0.10              # the use's ranked score (p_succ saturated minus time_w x t_close) must beat no-use by this
POW_MAX_U = 30.0             # the gem at most this far
POW_LIP_U = 3.0              # a void beginning within this ahead = a crossing question (Super Jump / blast vs the jump)
POW_RUN_ANGLE = 30.0         # a Super Speed run: heading within this of the line to the gem ...
POW_RUN_U = 8.0              # ... and the gem at least this far over continuous floor
ORDER_TOUR = 'walk'          # 'walk': the trainer's whole-spawn chooser (gems.plan_tour, walk-only fields; training parity);
                             # True: route.Route.order (fitted walk/jump leg times; g6v 139.2, g6w walk-only 139.0 against
                             # greedy 149.9: the policy falls on routes it was not trained on, as in log 28.50); False:
                             # the greedy chooser (operator 10-01: "grossly inefficient routes through some spawns"; log 37)
HANDBACK_STOP_ROOM = True    # after a planner pickup (or a failed consult) hand back on the first floor decision where the
                             # marble could stop before the first void along its velocity (A_STOP + the margin); the
                             # continuation only lasts while it could not (operator, 1x viewing 10-01: use the navigator)
HANDBACK_STOP_MARGIN = 0.5   # u added to the stopping distance
HANDBACK_ANY_DIR = True      # after a planner pickup hand back as soon as the continuation is clear and on the floor, whatever
                             # the direction (the downhill-toward-the-next-gem rule held the marble for 10-25 decisions)
SETUPS = False               # setup orders (the planner drives the leg before a crossing): OFF. Gate g6e (8 rounds): the
                             # planner's floor leg to the first gem took 2.0-2.9 s where the navigator walks it in 1.3 s,
                             # 2.4 a round, a net loss; the crossings themselves paid (10 shortcut legs 1.5-2.4 s vs ~2.5)
SHORTCUT_MAX_ANGLE = 60.0    # degrees between the heading and the line to the gem (the walk needs the turn too; the consult's
                             # own time check decides: drill 30-60 deg tied the walk, beyond that it lost)
                             # (drill v1: within 30 deg 1.66-1.73 s vs 2.0 s walking; 30-60 deg 2.3 s)
ROUTE_ABORT = 16             # ... before a route jump leg goes back (it starts with an approach, not a pickup plan)
CONT_SLOW = 12               # continuation decisions left (of CONT_MAX) when the planner stops rolling on toward the next
                             # gem and slows down for a slow handback instead


def _near(a, b, tol=0.5):
    return abs(a[0] - b[0]) < tol and abs(a[1] - b[1]) < tol


def _count(stuck, gem):
    for q in stuck:
        if _near(q[0], gem):
            return q[1]
    return 0


def _bump(stuck, gem):
    for q in stuck:
        if _near(q[0], gem):
            q[1] += 1
            return
    stuck.append([tuple(gem[:2]), 1])
STUCK_S, STUCK_U, STUCK_DETOUR = 3.0, 1.0, 24     # nav/real_run.py's stuck-breaker (defaults there)


def pow_state(ob, use_hist):
    """The planner's powerup dict from the raw observation (protocol.RAW_POW_*) and the uses sent in the last two
    decisions (they fire two decisions after they are sent)."""
    h = list(use_hist) + [(0, 0.0, 0)] * (2 - len(use_hist))
    prev2, prev1 = h[-2], h[-1]
    code = lambda u: 2 if u[2] else (1 if u[0] else 0)
    return {'held': int(ob[RAW_POW_HELD]) if len(ob) > RAW_POW_HELD else 0,
            'meter': float(ob[RAW_POW_BLAST]) if len(ob) > RAW_POW_BLAST else 0.0,
            'special': bool(ob[RAW_POW_SPECIAL] > 0.5) if len(ob) > RAW_POW_SPECIAL else False,
            'heli_left': float(ob[RAW_POW_HELI_LEFT]) if len(ob) > RAW_POW_HELI_LEFT else 0.0,
            'bounce_left': float(ob[RAW_POW_BOUNCE_LEFT]) if len(ob) > RAW_POW_BOUNCE_LEFT else 0.0,
            'shock_left': float(ob[RAW_POW_SHOCK_LEFT]) if len(ob) > RAW_POW_SHOCK_LEFT else 0.0,
            'use_p': code(prev2), 'use_yaw_p': prev2[1], 'use_c': code(prev1), 'use_yaw_c': prev1[1]}


def play(port, map_name, n_rounds, memory, tag, log, shortcuts=1, rescue=True, route_on=False, guard_on=False):
    import torch
    from terrain_obs import TerrainMap
    from nav.terrain import TerrainGrid, CROP_SHAPE
    from nav.obs import ObsBuilder, NAV_OBS_VERSION, VEC_DIM, ss_aim
    from nav.model import NavActorCritic, action_to_joystick
    from nav.real_run import choose
    g = Geometry(map_name)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck_path = os.environ.get('NAV_CKPT') or os.path.join(HERE, 'models', 'nav', 'nav_latest.pth')   # NAV_CKPT: a snapshot to gate (log 37)
    ck = torch.load(ck_path, map_location=dev, weights_only=False)
    assert ck.get('obs_version') == NAV_OBS_VERSION
    model = NavActorCritic().to(dev); model.load_state_dict(ck['model']); model.eval()
    terrain = TerrainGrid(TerrainMap.resolve(map_name)); obs_b = ObsBuilder(terrain)
    tour_fields = {}

    def walk_dist(a, gq):                       # as vec_worker.walk_dist: walk-only field per gem position
        kq = (round(gq[0], 1), round(gq[1], 1))
        f = tour_fields.get(kq)
        if f is None:
            f = tour_fields[kq] = terrain.goal_field(gq[0], gq[1], jumps=False)
        return terrain.dist_at(f, float(a[0]), float(a[1]), (gq[0], gq[1]))
    planner = PL.Planner(g, (0.0, 0.0, 0.0), seeds=None)
    route = RT.Route(g)
    guard = GD.Guard(planner)
    floating = lambda q: is_floating(g, q[:3])
    with torch.no_grad():
        model.act(torch.zeros(1, *CROP_SHAPE, device=dev), torch.zeros(1, VEC_DIM, device=dev),
                  model.initial_state(1, dev), deterministic=True)
    s = Session(port, map_name, g, log=log)
    log(f'hybrid: navigator update {ck.get("update")}, memory at handback "{memory}", map {map_name}, '
        f'shortcuts {shortcuts}, rescue {int(rescue)}, route {int(route_on)}, guard {int(guard_on)}')
    seed_cache = {}
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f'{tag}_{map_name}.jsonl')
    done = 0
    while done < n_rounds:
        st = {'map': map_name, 'tag': tag, 'memory': memory, 'navigator_update': ck.get('update'), 'points': 0.0,
              'gems_floor': 0, 'gems_float': 0, 'falls_nav': 0, 'falls_planner': 0, 'falls_after_handback': 0,
              'handovers': 0, 'groups': [], 'decisions': 0, 'planner_decisions': 0, 'ms': [], 'stuck_breaks': 0,
              'rescues': 0, 'shortcuts': 0, 'shortcut_aborts': 0, 'probes': 0, 'handbacks_rolling': 0,
              'route_legs': 0, 'route_aborts': 0, 'fall_log': [], 'guard_checks': 0, 'guard_overrides': 0,
              'guard_log': [],
              'probe_log': [], 'legs': [],    # every consultation; every leg from gem choice to pickup (who drove)
              'pickups': [], 'setups': 0,     # the state right after every pickup (stage 6b: crossing drill starts)
              'consult_seen': 0, 'consult_skip': {'far': 0, 'angle': 0, 'nogain': 0}, 'consult_cont': 0, 'consult_back': 0, 'big_consults': 0, 'handbacks_stoproom': 0,
              'pow_asked': 0, 'pow_uses': 0, 'pow_ms': [], 'pow_log': [], 'policy_uses': 0, 'policy_use_log': [],
              'jumps_planner': 0, 'jumps_nav': 0, 'jumps_suppressed': 0, 'across': 0,
              'oracle_asked': 0, 'oracle_jumps': 0, 'oracle_ms': [], 'oracle_log': []}
        try:
            s.ready()
            obs_b.reset(); h = model.initial_state(1, dev)
            target = None; owner = 'nav'; cont = 0; landed_n = 0; handback_i = -10 ** 9
            ptarget = None; group_key = None; group_t0 = 0
            pos_hist = []; stuck_until = -1; stuck_on = []; rescue_i = None
            shortcut = False; probe_next = 0; no_plan = 0; walk_fields = {}; leg = None; last_info = None
            trial_i0 = None; trial_t_nav = 0.0; trial_walk = 0.0; trial_big = False     # a consult in progress (CONSULT_DEC)
            oracle_gem = None                                        # the gem the oracle's planner target was set to
            use_now = 0; yaw_now = None; blast_now = 0; use_hist = []; pow_next = 0   # powerup oracle state (log 39)
            setup = None                                             # (gem A, gem B): walk to A, then jump to B
            setup_i0 = None                                          # decision the planner took the setup's first leg
            cont_i0 = 0                                              # decision the current continuation started
            route_leg = False; route_block = set(); recent = []        # (x, y, z, vx, vy, vz, reply...) per decision
            recent_plans = []                                          # (i, kind, p_succ, jump, t_close) of planner decisions

            def walk_to(gem, x, y):
                gk = tuple(np.round(gem[:2], 0))
                if gk not in walk_fields:
                    walk_fields[gk] = terrain.goal_field(gem[0], gem[1], jumps=False)
                return terrain.dist_at(walk_fields[gk], float(x), float(y), (gem[0], gem[1]))
            Uc, Jc = PL.reply_vector(NOOP_ACTION + (0.0,)); Up, Jp = Uc.copy(), Jc
            msg = s.env.msg
            i = 0
            while True:
                ob = msg.obs
                p = np.asarray(ob[RAW_POS], float); v = np.asarray(ob[RAW_VEL], float); w = np.asarray(ob[RAW_SPIN], float)
                tel = msg.extra
                lv = g.floor_below(p[0], p[1], p[2])
                gap = (p[2] - R - lv) if lv is not None else 9.0
                sup = bool(tel is not None and tel[2] > 0)
                airborne = (not sup) and gap > 0.5
                vis = visible_gems(ob)
                key = tuple(sorted((round(q[0], 1), round(q[1], 1)) for q in vis))
                if vis and group_key is None:
                    group_key = key; group_t0 = i
                if ORDER_TOUR == 'walk' and vis:
                    # the TRAINER's whole-spawn chooser (nav/gems.plan_tour over walk-only terrain fields, exactly
                    # vec_worker.pick with NAV_TOUR=walk): what the policy is being fine-tuned on (log 37)
                    target, nxt = plan_tour(vis, target, ob[0:2], ob[3:5], dist=walk_dist)
                elif ORDER_TOUR and vis:
                    # whole-spawn order (operator 10-01, log 37): walking/jumping leg times over every order of the
                    # visible gems, instead of the greedy nearest-with-momentum chooser the policy was trained on
                    target, nxt, oinfo = route.order(p, v, vis, target, floating=floating)
                    if target is None:
                        target, nxt = choose(vis, target, pos=ob[0:2], vel=ob[3:5], terrain=terrain, mfield=None)
                else:
                    target, nxt = choose(vis, target, pos=ob[0:2], vel=ob[3:5], terrain=terrain, mfield=None)
                if target is not None and (leg is None or not _near(leg['gem'], target)):
                    leg = {'gem': [round(float(c), 2) for c in target[:3]], 'i0': i,
                           'walk0': round(walk_to(target, p[0], p[1]), 2),
                           'straight0': round(math.hypot(target[0] - p[0], target[1] - p[1]), 2),
                           'speed0': round(math.hypot(v[0], v[1]), 2), 'floating': bool(is_floating(g, target[:3])),
                           'planner': owner == 'planner', 'shortcut': False}
                # ---- ownership
                if owner == 'nav' and target is not None and is_floating(g, target[:3]):
                    owner = 'planner'; st['handovers'] += 1; rescue_i = None
                    ptarget = np.asarray(target[:3], float)
                    kk = tuple(np.round(ptarget, 2))
                    if kk not in seed_cache:
                        seed_cache[kk] = SEEDS.for_gem(map_name, ptarget)
                    planner.set_target(ptarget, seed_cache[kk])
                    planner.floor_ref = DR.floor_ref_of(g, ptarget); planner.n_broad = PL.N_BROAD
                if (rescue and owner == 'nav' and target is not None and RESCUE_AFTER > 0
                        and _count(stuck_on, target) >= RESCUE_AFTER):
                    # the navigator is stuck on this (floor) gem: the planner takes it, then hands back
                    owner = 'planner'; st['rescues'] += 1
                    ptarget = np.asarray(target[:3], float)
                    planner.set_target(ptarget, None)
                    planner.floor_ref = DR.floor_ref_of(g, ptarget); planner.n_broad = FLOOR_BROAD
                    stuck_on = [q for q in stuck_on if not _near(q[0], target)]; rescue_i = i
                if owner == 'planner' and rescue_i is not None and cont == 0 and (i - rescue_i) * 0.064 > RESCUE_MAX_S:
                    owner = 'nav'; handback_i = i; ptarget = None; rescue_i = None
                    if setup_i0 is not None:
                        st['probe_log'][-1].update({'ok': False}); setup = None; setup_i0 = None; planner.exit_dir = None
                # after a planner pickup: roll on toward the navigator's next (floor) gem, then hand back (continue_cost)
                if (owner == 'planner' and cont > CONT_SLOW and target is not None and not is_floating(g, target[:3])
                        and not (planner.cont_next and same_gem(planner.gem, target[:3]))):
                    fr = min(planner.floor_ref, DR.floor_ref_of(g, np.asarray(target[:3], float)))
                    planner.set_target(np.asarray(target[:3], float), None, cont_next=True); planner.floor_ref = fr
                    last_info = None
                # a jump leg the greedy chooser would not take (route.py): the planner drives it from here
                if route_on and owner == 'nav' and cont == 0 and target is not None and not airborne:
                    J, rinfo = route.choose(p, v, vis, target, floating=floating)
                    if J is not None and RT.Route.key(J) not in route_block:
                        target = J
                        owner = 'planner'; shortcut = True; route_leg = True; ptarget = np.asarray(J[:3], float)
                        planner.set_target(ptarget, None); planner.floor_ref = DR.floor_ref_of(g, ptarget)
                        planner.n_broad = PL.N_BROAD; planner.time_w = SHORT_TIME_W; planner.p_sat = SHORT_P_SAT; planner.flight_check = SHORT_FLIGHT_CHECK; planner.approach_line = True
                        planner.land_steer = True
                        st['route_legs'] += 1; no_plan = 0; rescue_i = None
                        leg = {'gem': [round(float(c), 2) for c in J[:3]], 'i0': i, 'walk0': round(walk_to(J, p[0], p[1]), 2),
                               'straight0': round(math.hypot(J[0] - p[0], J[1] - p[1]), 2),
                               'speed0': round(math.hypot(v[0], v[1]), 2), 'floating': False, 'planner': True,
                               'shortcut': True, 'route': True, 'take_i': i, 't_route': rinfo}
                probe = None
                # a BIG detour (the walk round the holes much longer than the line, e.g. the ring -> centre legs the
                # operator watched the hybrid walk, log 36.4): consulted whatever the heading and again every
                # PROBE_EVERY decisions along the leg, with a longer window for the planner's approach and turn
                big = (leg is not None and leg.get('walk0') is not None and np.isfinite(leg['walk0'])
                       and leg['walk0'] - leg['straight0'] >= BIG_GAIN_U and leg['straight0'] <= SHORTCUT_MAX_U)
                if (shortcuts and owner == 'nav' and target is not None and not airborne and i >= probe_next
                        and not is_floating(g, target[:3]) and leg is not None and (i - leg['i0'] <= CONSULT_AT_DEC or big)):
                    # the crossing is usually to the SECOND-nearest gem (the human's order is chosen around the jump,
                    # log 33.3): a jump-first order that beats the greedy order by ROUTE_GAIN_S (route.py, crossing
                    # times fitted on the drill, turn cost included) names the candidate; otherwise the greedy target
                    J, rinfo = route.choose(p, v, vis, target, floating=floating)
                    if J is None and setup is not None and setup[0] is None and any(same_gem(setup[1], q[:3]) for q in vis):
                        J = next(q for q in vis if same_gem(setup[1], q[:3]))       # the setup's jump gem, after A
                    setup = None
                    if SETUPS and rinfo and rinfo.get('setup') is not None and J is None:
                        setup = (np.asarray(target[:3], float), np.asarray(rinfo['setup'][:3], float)); st['setups'] += 1
                        # two-leg plan: the planner drives to A rewarded for leaving it along the A -> B line at speed
                        tgA = setup[0]; uu = setup[1][:2] - tgA[:2]; uu = uu / max(1e-6, np.linalg.norm(uu))
                        planner.set_target(tgA, None); planner.floor_ref = DR.floor_ref_of(g, tgA); planner.n_broad = FLOOR_BROAD
                        planner.exit_dir = (float(uu[0]), float(uu[1])); planner.time_w = SHORT_TIME_W; planner.p_sat = SHORT_P_SAT
                        owner = 'planner'; ptarget = tgA; rescue_i = i; no_plan = 0; setup_i0 = i
                        if leg is not None:
                            leg['planner'] = True; leg['setup'] = True; leg['take_i'] = i
                        st['probe_log'].append({'i': i, 'x': round(float(p[0]), 2), 'y': round(float(p[1]), 2),
                                                'speed': round(math.hypot(v[0], v[1]), 2), 'angle': None,
                                                'gem': [round(float(c), 2) for c in tgA], 'jump_gem': [round(float(c), 2) for c in setup[1]],
                                                'straight': round(math.hypot(tgA[0] - p[0], tgA[1] - p[1]), 2), 'walk': None,
                                                't_nav': round(rinfo['t_greedy'], 3), 't_setup': round(rinfo['t_setup'], 3), 'kind': 'setup', 'ok': None})
                    if J is None and ACROSS_CONSULT:
                        # the gem across a hole that the marble already points at, even when it is not the nearest
                        # (the human's crossings go to the second-nearest gem, log 33.3): the planner's own time
                        # against the walk to it decides, as for any consult
                        best = None
                        for q in vis:
                            if is_floating(g, q[:3]) or same_gem(q[:3], target[:3]):
                                continue
                            stq = math.hypot(q[0] - p[0], q[1] - p[1])
                            if stq > SHORTCUT_MAX_U or stq < 6.0:
                                continue
                            spq = math.hypot(v[0], v[1])
                            if spq < 0.5:
                                continue
                            lq = math.atan2(q[1] - p[1], q[0] - p[0])
                            aq = abs(math.degrees((math.atan2(v[1], v[0]) - lq + math.pi) % (2 * math.pi) - math.pi))
                            if aq > ACROSS_MAX_ANGLE:
                                continue
                            gapq = route.jump_gap(p[0], p[1], q)
                            if gapq is None or gapq < 3.0:
                                continue
                            if best is None or aq < best[0]:
                                best = (aq, q)
                        if best is not None:
                            J = best[1]; st['across'] += 1
                    if J is not None and not same_gem(J[:3], target[:3]):
                        target = J
                        leg = {'gem': [round(float(c), 2) for c in J[:3]], 'i0': i, 'walk0': round(walk_to(J, p[0], p[1]), 2),
                               'straight0': round(math.hypot(J[0] - p[0], J[1] - p[1]), 2),
                               'speed0': round(math.hypot(v[0], v[1]), 2), 'floating': False, 'planner': True,
                               'shortcut': False, 'route': True, 't_route': rinfo}
                    straight = math.hypot(target[0] - p[0], target[1] - p[1])
                    spd0 = math.hypot(v[0], v[1])
                    line = math.atan2(target[1] - p[1], target[0] - p[0])
                    ang = abs(math.degrees((math.atan2(v[1], v[0]) - line + math.pi) % (2 * math.pi) - math.pi)) if spd0 > 0.5 else 0.0
                    st['consult_seen'] += 1
                    if straight > SHORTCUT_MAX_U:
                        st['consult_skip']['far'] += 1
                    elif ang > SHORTCUT_MAX_ANGLE:
                        st['consult_skip']['angle'] += 1
                    # gate g6r (log 36.5): big consults from any heading made the planner turn and run up from slow,
                    # misaligned states (shortcut legs 2.6-2.8 s against the navigator's own 1.4-2.0 s with the
                    # oracle) and roll into holes (1-3 planner falls a round, ~140 points). The heading gate stays;
                    # a big leg is consulted whenever the navigator's own path aligns with the line (any decision)
                    if straight <= SHORTCUT_MAX_U and ang <= SHORTCUT_MAX_ANGLE:
                        tg3 = np.asarray(target[:3], float); fr = DR.floor_ref_of(g, tg3)
                        planner.set_target(tg3, None); planner.floor_ref = fr; planner.n_broad = PL.N_BROAD
                        planner.time_w = SHORT_TIME_W; planner.p_sat = SHORT_P_SAT; planner.flight_check = SHORT_FLIGHT_CHECK; planner.approach_line = True; planner.land_steer = True
                        if EXIT_TO_NEXT and nxt is not None:
                            # charge the crossing plan for the turn the leg AFTER it will need (planner.exit_bonus, the
                            # next-leg turn time): a plan that lands slower or aligned for the next gem wins over one
                            # that flies through at 13 u/s and has to stop and come back (operator, 1x viewing 09-30)
                            uu = np.asarray(nxt[:2], float) - tg3[:2]
                            if np.linalg.norm(uu) > 1e-6:
                                uu = uu / np.linalg.norm(uu); planner.exit_dir = (float(uu[0]), float(uu[1]))
                        walk = float(planner.walk_at(np.array([p[0]]), np.array([p[1]]), fr)[0])
                        if np.isfinite(walk) and walk - straight >= SHORTCUT_GAIN_U:
                            # consult by driving: the planner owns the marble for CONSULT_DEC decisions with its state
                            # kept; a pickup plan that jumps and beats the navigator's fitted time commits the shortcut
                            st['probes'] += 1
                            t_nav = route.legs(p[0], p[1], math.atan2(v[1], v[0]) if spd0 > 0.5 else 0.0,
                                               spd0 if spd0 > 0.5 else 0.0, target)[0]
                            owner = 'planner'; shortcut = True; ptarget = tg3; no_plan = 0; rescue_i = None
                            trial_i0 = i; trial_t_nav = t_nav; trial_walk = walk; last_info = None; trial_big = big
                            if big:
                                st['big_consults'] += 1
                            st['probe_log'].append({'i': i, 'x': round(float(p[0]), 2), 'y': round(float(p[1]), 2),
                                                    'speed': round(spd0, 2), 'angle': round(ang, 1),
                                                    'gem': [round(float(c), 2) for c in target[:3]],
                                                    'straight': round(straight, 2), 'walk': round(walk, 2),
                                                    't_nav': round(t_nav, 3), 'kind': 'trial', 'ok': None})
                        else:
                            st['consult_skip']['nogain'] += 1
                            probe_next = i + PROBE_EVERY
                if owner == 'planner' and shortcut and trial_i0 is not None and cont == 0:
                    # the consult: judged on the previous decision's plan (last_info)
                    li = last_info
                    if li is not None and li['kind'] == 'pickup' and li['jump_at'] >= 0 and li['t_close'] * 0.064 + SHORTCUT_MARGIN_S < trial_t_nav:
                        st['shortcuts'] += 1; trial_i0 = None
                        st['probe_log'][-1].update({'ok': True, 'found_at': i, 't_close': li['t_close'], 'p_succ': round(li.get('p_succ', 0.0), 3)})
                        if leg is not None:
                            leg['shortcut'] = True; leg['take_i'] = i; leg['walk_take'] = round(trial_walk, 2); leg['t_close_plan'] = li['t_close']
                    elif i - trial_i0 >= (CONSULT_DEC_BIG if trial_big else CONSULT_DEC) or airborne:
                        st['probe_log'][-1].update({'ok': False, 'kind': (li['kind'] if li is not None else 'none'),
                                                    't_close': (li['t_close'] if li is not None else -1)})
                        # no crossing: not an instant handback (gate g6d: a fall within 3 s of a handback every round,
                        # the marble handed back at ~10 u/s beside the hole) but the planner's safe continuation toward
                        # the navigator's gem (cont_next), handed back when slow or rolling clear (HANDBACK_*)
                        shortcut = False; trial_i0 = None; probe_next = i + PROBE_EVERY
                        if leg is not None and leg.get('route'):
                            target = None; leg = None                  # back to the navigator's own choice next decision
                        if (li is not None and li['safe'] and not li['fell'] and li['clear'] >= HANDBACK_CLEAR
                                and not airborne and sup and math.hypot(v[0], v[1]) <= HANDBACK_FAST):
                            # the state is clear of edges and on the floor: hand back at once (the continuation braked
                            # to 4 u/s first and cost ~1.5 s per failed consult, gate g6f)
                            owner = 'nav'; handback_i = i; ptarget = None; st['consult_back'] += 1
                        else:
                            cont = CONT_MAX; landed_n = 0; planner.cont_next = False; st['consult_cont'] += 1; cont_i0 = i
                if owner == 'planner' and shortcut and trial_i0 is None and cont == 0 and not airborne:
                    no_plan = no_plan + 1 if planner.last_kind != 'pickup' else 0
                    if no_plan >= (ROUTE_ABORT if route_leg else SHORTCUT_ABORT):   # no jump route: back to walking
                        if route_leg:
                            route_block.add(RT.Route.key(ptarget)); st['route_aborts'] += 1; target = None
                        else:
                            st['shortcut_aborts'] += 1
                        owner = 'nav'; shortcut = False; route_leg = False; handback_i = i; ptarget = None
                        probe_next = i + 4 * PROBE_EVERY
                if owner == 'planner' and cont == 0 and ptarget is not None and not any(same_gem(ptarget, q[:3]) for q in vis):
                    ptarget = None                                     # the gem went without our pickup (should not happen)
                    owner = 'nav'; handback_i = i; rescue_i = None; shortcut = False; route_leg = False
                if owner == 'planner' and cont > 0:
                    cont -= 1
                    landed_n = landed_n + 1 if (sup and gap < 0.35) else 0
                    spd = math.hypot(v[0], v[1])
                    rolling_on = False
                    if (HANDBACK_STOP_ROOM and landed_n >= 1 and sup and not airborne and lv is not None
                            and (i - cont_i0) >= 1):
                        # operator, 1x viewing 10-01: the planner kept the marble long after its pickup (the
                        # continuation ran to the next gem); the navigator is the better driver, so hand back the
                        # moment the marble is on the floor and could STOP before the first void along its velocity
                        # (A_STOP braking plus a margin): the physics decide, not the continuation's rollout
                        d_void = 10.0
                        if spd > 0.5:
                            ux, uy = v[0] / spd, v[1] / spd
                            dq = np.arange(0.25, 10.01, 0.25)
                            lvs, _ = D3.level_below(g, p[0] + dq * ux, p[1] + dq * uy, np.full(len(dq), p[2] + 0.3))
                            void = ~(np.isfinite(lvs) & (lvs >= lv - 0.6))
                            if void.any():
                                d_void = float(dq[int(np.argmax(void))])
                        if spd * spd / (2.0 * PL.A_STOP) + HANDBACK_STOP_MARGIN <= d_void:
                            rolling_on = True; landed_n = max(landed_n, LAND_HOLD); st['handbacks_stoproom'] += 1
                    if (not rolling_on and landed_n >= LAND_HOLD and planner.cont_next and spd > 0.5
                            and last_info is not None):
                        # rolling toward the next gem on a continuation that keeps clear of every edge: hand back at speed
                        ux, uy = v[0] / spd, v[1] / spd
                        w0, w1 = planner.walk_at(np.array([p[0], p[0] + 0.5 * ux]), np.array([p[1], p[1] + 0.5 * uy]),
                                                 planner.floor_ref)
                        rolling_on = bool((HANDBACK_ANY_DIR or (np.isfinite(w0) and np.isfinite(w1) and w1 <= w0 - HANDBACK_DOWNHILL))
                                          and last_info.get('kind') == 'continue' and last_info['safe']
                                          and not last_info['fell'] and last_info['clear'] >= HANDBACK_CLEAR)
                    if cont == CONT_SLOW and planner.cont_next:
                        planner.cont_next = False                      # not handed back yet: slow down (CONT_SPEED_COST)
                    if cont == 0 and spd > HANDBACK_SPEED and not rolling_on and (i - cont_i0) < CONT_HARD_MAX:
                        cont = 4                                       # not slow and not clear yet: keep braking (g6g: the
                                                                       # cap handed the marble back at 11-16 u/s, falls followed)
                    if (landed_n >= LAND_HOLD and (spd <= HANDBACK_SPEED or rolling_on)) or cont == 0:
                        if st['legs'] and st['legs'][-1].get('pick_i') is not None and 'back_i' not in st['legs'][-1]:
                            st['legs'][-1]['back_i'] = i; st['legs'][-1]['back_speed'] = round(spd, 2)
                            st['legs'][-1]['back_rolling'] = rolling_on
                        st['handbacks_rolling'] += int(rolling_on)
                        owner = 'nav'; cont = 0; handback_i = i; ptarget = None; rescue_i = None; shortcut = False
                        route_leg = False
                        if memory == 'reset':
                            obs_b.reset(); h = model.initial_state(1, dev)
                # ---- the navigator (always run in 'current' memory mode; only when it owns otherwise)
                js_nav = None
                if owner == 'nav' or memory == 'current':
                    if target is not None:
                        goal = (target[0], target[1], target[2])
                        ngoal = (nxt[0], nxt[1], nxt[2]) if nxt else None
                        if setup is not None and setup[0] is not None and same_gem(setup[0], target[:3]):
                            ngoal = (float(setup[1][0]), float(setup[1][1]), float(setup[1][2]))   # exit toward the jump gem
                        mx, my = float(p[0]), float(p[1]); pos_hist.append((mx, my))
                        nwin = int(round(STUCK_S / 0.064))
                        if (owner == 'nav' and i >= stuck_until and len(pos_hist) > nwin
                                and math.hypot(mx - pos_hist[-1 - nwin][0], my - pos_hist[-1 - nwin][1]) < STUCK_U):
                            obs_b.reset(); h = model.initial_state(1, dev); stuck_until = i + STUCK_DETOUR
                            st['stuck_breaks'] += 1
                            _bump(stuck_on, target)
                        crop, vec, _ = obs_b.build(ob, goal, ngoal)
                        with torch.no_grad():
                            o = model.act(torch.as_tensor(crop, device=dev).unsqueeze(0), torch.as_tensor(vec, device=dev).unsqueeze(0),
                                          h, deterministic=not (i < stuck_until))
                        h = o['h_next']
                        a = o['action_game'][0].tolist()
                        js_nav = tuple(action_to_joystick(a[0], a[1], a[2], a[3], a[4], float(v[0]), float(v[1])))
                        if POLICY_USE and owner == 'nav' and len(a) > 5 and a[5] > 0.5:
                            pwn = pow_state(ob, use_hist)
                            if pwn['held'] > 0:
                                use_now = 1; yaw_now = math.atan2(*ss_aim(vec)) if pwn['held'] == 2 else None   # V9 physics aim
                            elif pwn['special'] or pwn['meter'] >= PW.BLAST_REQUIRED:
                                blast_now = 1
                            if use_now or blast_now:
                                st['policy_uses'] += 1
                                if len(st['policy_use_log']) < 200:
                                    st['policy_use_log'].append([i, pwn['held'], round(pwn['meter'], 2), round(float(p[0]), 1), round(float(p[1]), 1)])
                    else:
                        js_nav = tuple(js_of(0.0, 0.0, 1.0, 0, 1, v[0], v[1]))     # no gem in view: brake and wait
                if (ORACLE and owner == 'nav' and js_nav is not None and target is not None and not airborne and sup
                        and not is_floating(g, target[:3]) and math.hypot(v[0], v[1]) >= ORACLE_MIN_SPEED and lv is not None):
                    # the planner as a jump oracle (log 35.4): the target lies across a void on the line, the lip is within
                    # ORACLE_LIP_U ahead along the velocity, the heading is within ORACLE_ANGLE of the line: ask the model
                    # whether a jump NOW takes the gem and lands safely; if so, set the jump bit on the navigator's action
                    spd = math.hypot(v[0], v[1]); line = math.atan2(target[1] - p[1], target[0] - p[0])
                    ang = abs(math.degrees((math.atan2(v[1], v[0]) - line + math.pi) % (2 * math.pi) - math.pi))
                    straight = math.hypot(target[0] - p[0], target[1] - p[1])
                    if ang <= ORACLE_ANGLE and 4.0 <= straight <= SHORTCUT_MAX_U:
                        ux, uy = v[0] / spd, v[1] / spd
                        dd = np.arange(0.25, ORACLE_LIP_U + 0.01, 0.25)
                        lvs, _ = D3.level_below(g, p[0] + dd * ux, p[1] + dd * uy, np.full(len(dd), p[2] + 0.3))
                        void = ~(np.isfinite(lvs) & (lvs >= lv - 0.6))
                        if void.any() and not void[0]:
                            st['oracle_asked'] += 1
                            tg3 = np.asarray(target[:3], float)
                            if oracle_gem is None or not same_gem(oracle_gem, tg3):
                                planner.set_target(tg3, None); planner.floor_ref = DR.floor_ref_of(g, tg3); oracle_gem = tg3
                            t0o = time.perf_counter()
                            po, tco, ppo = planner.jump_oracle(p, v, w, tel, Uc, Jc, Up, Jp, planner.floor_ref)
                            st['oracle_ms'].append(1000.0 * (time.perf_counter() - t0o))
                            if po >= ORACLE_P:
                                js_nav = tuple(js_nav[:4]) + (1,) + tuple(js_nav[5:]); st['oracle_jumps'] += 1
                                st['oracle_log'].append({'i': i, 'speed': round(spd, 2), 'angle': round(ang, 1), 'straight': round(straight, 2),
                                                         'p': round(po, 3), 't_close': tco, 'gem': [round(float(c), 2) for c in tg3]})
                # POWERUP ORACLE (POWERUP_PLAN phase 3, log 39): holding a Super Jump / Super Speed / Helicopter, or with
                # the blast meter usable, ask the planner every POW_ORACLE_EVERY decisions whether firing NOW (aimed at
                # the gem, a few variants) beats carrying on; if so set the use bit (and the yaw) on the navigator's
                # action. The navigator keeps driving; the bridge latches the yaw over the two-decision key lag.
                pw = pow_state(ob, use_hist)
                if (POW_ORACLE and owner == 'nav' and js_nav is not None and target is not None and not airborne and sup
                        and not is_floating(g, target[:3]) and i >= pow_next
                        and (pw['held'] in (1, 2, 5) or pw['special'] or pw['meter'] >= PW.BLAST_REQUIRED)):
                    straight = math.hypot(target[0] - p[0], target[1] - p[1])
                    spd = math.hypot(v[0], v[1]); line = math.atan2(target[1] - p[1], target[0] - p[0])
                    ang = abs(math.degrees((math.atan2(v[1], v[0]) - line + math.pi) % (2 * math.pi) - math.pi)) if spd > 0.5 else 180.0
                    # when is a question worth 200 ms (g8a asked 560 times a round, used once): (a) a void begins
                    # within POW_LIP_U ahead along the velocity and the heading is within ORACLE_ANGLE of the line (a
                    # crossing: a Super Jump / blast flies further than the jump), (b) a Super Speed held, the heading
                    # within POW_RUN_ANGLE of the line and the line to the gem all floor (a run)
                    ask = False
                    if 3.0 <= straight <= POW_MAX_U and spd > 1.0 and lv is not None:
                        ux, uy = v[0] / spd, v[1] / spd
                        dd = np.arange(0.25, POW_LIP_U + 0.01, 0.25)
                        lvs, _ = D3.level_below(g, p[0] + dd * ux, p[1] + dd * uy, np.full(len(dd), p[2] + 0.3))
                        void = ~(np.isfinite(lvs) & (lvs >= lv - 0.6))
                        if ang <= ORACLE_ANGLE and void.any() and not void[0]:
                            ask = True
                        elif pw['held'] == 2 and ang <= POW_RUN_ANGLE and straight >= POW_RUN_U:
                            m = int(straight / 0.5) + 1; tt = np.linspace(0.0, 1.0, m)
                            lvl, _ = D3.level_below(g, p[0] + tt * (target[0] - p[0]), p[1] + tt * (target[1] - p[1]), np.full(m, p[2] + 0.3))
                            ask = bool((np.isfinite(lvl) & (lvl >= lv - 0.6)).all())
                    if ask:
                        pow_next = i + POW_ORACLE_EVERY
                        st['pow_asked'] += 1
                        tg3 = np.asarray(target[:3], float)
                        if oracle_gem is None or not same_gem(oracle_gem, tg3):
                            planner.set_target(tg3, None); planner.floor_ref = DR.floor_ref_of(g, tg3); oracle_gem = tg3
                        t0o = time.perf_counter()
                        gain, best = planner.use_oracle(p, v, w, tel, Uc, Jc, Up, Jp, planner.floor_ref, pw)
                        st['pow_ms'].append(1000.0 * (time.perf_counter() - t0o))
                        if best is not None and gain >= POW_GAIN and best['p_succ'] >= ORACLE_P:
                            if best['use'] == 1:
                                use_now = 1; yaw_now = best['yaw'] if pw['held'] == 2 else None
                            else:
                                blast_now = 1
                            st['pow_uses'] += 1
                            st['pow_log'].append({'i': i, 'held': pw['held'], 'meter': round(pw['meter'], 2), 'use': best['use'],
                                                  'yaw': round(best['yaw'], 2), 'gain': round(gain, 3), 'p': round(best['p_succ'], 3),
                                                  't_close': best['t_close'], 'ref_p': round(best['ref_p'], 3), 'ref_t': best['ref_t'],
                                                  'straight': round(straight, 2), 'gem': [round(float(c), 2) for c in tg3]})
                if (guard_on and owner == 'nav' and js_nav is not None and target is not None and not airborne
                        and lv is not None and guard.near_void(p, v)):
                    # fall guard (guard.py): keep the navigator's action only if the marble can still stay up after it
                    st['guard_checks'] += 1
                    js_nav, over = guard.check(p, v, w, Uc, Jc, Up, Jp, js_nav, target, float(lv))
                    if over:
                        st['guard_overrides'] += 1
                        st['guard_log'].append([i, round(float(p[0]), 2), round(float(p[1]), 2), round(math.hypot(v[0], v[1]), 2)])
                if owner == 'planner' and leg is not None:
                    leg['planner'] = True
                if owner == 'planner':
                    if probe is not None:
                        js, info = probe
                    else:
                        js, info = planner.plan(p, v, w, tel, Uc, Jc, Up, Jp, airborne, cont > 0, planner.floor_ref)
                        st['ms'].append(info['ms'])
                        if js[4] and not airborne and (info['kind'] != 'pickup' or trial_i0 is not None) and not (rescue_i is not None and info.get('escape_jump')):
                            # no jump outside a committed pickup plan (g6g: 5.5 planner jumps a round for 2 crossings:
                            # approach and continuation hops on open floor, the operator's "needless jumps")
                            js = tuple(js[:4]) + (0,) + tuple(js[5:]); st['jumps_suppressed'] += 1
                    last_info = info
                    st['planner_decisions'] += 1
                    recent_plans.append([i, info['kind'][0] if info['kind'] else '-', round(float(info.get('p_succ', 0.0)), 2), int(js[4]),
                                         round(float(info.get('t_close', -1)))]); recent_plans = recent_plans[-30:]
                else:
                    js = js_nav
                js = tuple(float(a) for a in js[:4]) + (int(js[4]), float(js[5]) if len(js) > 5 else 0.0)
                if js[4] and not airborne:
                    st['jumps_planner' if owner == 'planner' else 'jumps_nav'] += 1    # needless jumps? (operator, 09-30)
                recent.append(list(p) + list(v) + list(w) + list(js)); recent = recent[-30:]
                msg, sinfo = s.step(js, use_pow=use_now, pow_yaw=yaw_now, use_blast=blast_now)
                use_hist.append((use_now, yaw_now if yaw_now is not None else 0.0, blast_now)); use_hist = use_hist[-2:]
                use_now = 0; yaw_now = None; blast_now = 0
                Up, Jp = Uc, Jc
                Uc, Jc = PL.reply_vector(js)
                i += 1
                if sinfo['gem_delta'] > 0:
                    st['points'] += float(sinfo['gem_delta'])
                    new = [q[:3] for q in visible_gems(msg.obs)]
                    gone = [q[:3] for q in vis if not any(same_gem(q[:3], r) for r in new)]
                    if gone and is_floating(g, gone[0]):
                        st['gems_float'] += 1
                    else:
                        st['gems_floor'] += 1
                    if gone:
                        po = msg.obs
                        st['pickups'].append({'i': i, 'owner': owner, 'gem': [round(float(c), 2) for c in gone[0]],
                                              'p': [round(float(c), 3) for c in po[RAW_POS]],
                                              'v': [round(float(c), 3) for c in po[RAW_VEL]],
                                              'w': [round(float(c), 3) for c in po[RAW_SPIN]],
                                              'gems': [[round(float(c), 2) for c in q[:3]] for q in new]})
                    if owner == 'planner' and ptarget is not None and gone and same_gem(ptarget, gone[0]):
                        cont = CONT_MAX; landed_n = 0; cont_i0 = i     # its gem: land (if flying) and settle, then hand back
                        if trial_i0 is not None:                       # the gem taken while still consulting: counts
                            st['shortcuts'] += 1; trial_i0 = None; st['probe_log'][-1].update({'ok': True, 'found_at': i})
                    if setup is not None and setup[0] is not None and gone and same_gem(setup[0], gone[0]):
                        setup = (None, setup[1]); planner.exit_dir = None
                        if owner == 'planner' and st['probe_log'] and st['probe_log'][-1].get('kind') == 'setup':
                            st['probe_log'][-1].update({'ok': True, 'found_at': i, 'pick_speed': round(math.hypot(msg.obs[RAW_VEL][0], msg.obs[RAW_VEL][1]), 2)})
                            cont = 0; landed_n = 0; rescue_i = None; setup_i0 = None
                            # straight into the crossing consult on B from this state (the leg is created next decision
                            # by the generic code; the consult block needs owner == 'nav')
                            owner = 'nav'; handback_i = i; ptarget = None; probe_next = 0
                    elif setup is not None and gone:
                        setup = None
                    if leg is not None and gone and _near(leg['gem'], gone[0]):
                        leg['pick_i'] = i; leg['t'] = round((i - leg['i0']) * 0.064, 3)
                        leg['pick_speed'] = round(math.hypot(msg.obs[RAW_VEL][0], msg.obs[RAW_VEL][1]), 2)
                        st['legs'].append(leg)
                    leg = None
                    target = None
                    route_block = set()                                # a pickup changes every order
                if sinfo['fell']:
                    st['falls_planner' if owner == 'planner' else 'falls_nav'] += 1
                    st['fall_log'].append({'i': i, 'owner': owner, 'target': [round(float(c), 2) for c in target[:3]] if target is not None else None,
                                           'recent': [[round(float(c), 2) for c in q] for q in recent[-25:]],
                                           'plans': [q for q in recent_plans if q[0] >= i - 25]})
                    if owner == 'nav' and (i - handback_i) * 0.064 <= AFTER_HANDBACK_S:
                        st['falls_after_handback'] += 1
                    s.recover(); msg = s.env.msg
                    owner = 'nav'; cont = 0; ptarget = None; target = None; leg = None
                    shortcut = False; route_leg = False; rescue_i = None      # (shortcut stayed set after a fall before)
                    trial_i0 = None; setup = None; setup_i0 = None; planner.exit_dir = None
                    obs_b.reset(); h = model.initial_state(1, dev); pos_hist = []
                    Uc, Jc = PL.reply_vector(NOOP_ACTION + (0.0,)); Up, Jp = Uc.copy(), Jc
                # a new group: more gems in view than before (positions are rebuilt from the marble's position and
                # the offset, so comparing them rounded flickers: the first version counted 12 groups in 13 s)
                if len(visible_gems(msg.obs)) > len(vis) and group_key is not None:
                    st['groups'].append(round((i - group_t0) * 0.064, 2)); group_t0 = i
                st['decisions'] = i
                if i % 300 == 0:
                    log(f'  t {i * 0.064:.0f} s: points {st["points"]:.0f}, floor {st["gems_floor"]}, floating '
                        f'{st["gems_float"]}, falls nav {st["falls_nav"]} planner {st["falls_planner"]}, '
                        f'groups {len(st["groups"])}, owner {owner}')
        except RoundOver:
            pass
        oms = st.pop('oracle_ms')
        st['oracle_ms_mean'] = float(np.mean(oms)) if oms else 0.0
        ms = st.pop('ms')
        st['plan_ms_mean'] = float(np.mean(ms)) if ms else 0.0
        st['time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        with open(out, 'a') as f:
            f.write(json.dumps(st) + '\n')
        done += 1
        log(f'round {done}: points {st["points"]:.0f}, gems floor {st["gems_floor"]} floating {st["gems_float"]}, '
            f'falls nav {st["falls_nav"]} planner {st["falls_planner"]} (after handback {st["falls_after_handback"]}), '
            f'handovers {st["handovers"]}, groups {len(st["groups"])} {st["groups"]}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, required=True)
    ap.add_argument('--map', default='kotmjump')
    ap.add_argument('--rounds', type=int, default=2)
    ap.add_argument('--memory', choices=('current', 'reset'), default='current')
    ap.add_argument('--tag', default='hybrid')
    ap.add_argument("--shortcuts", type=int, default=1)       # 0 off, 1 on, 2 consult and log only
    ap.add_argument('--rescue', type=int, default=1)
    ap.add_argument('--route', type=int, default=0)          # 1: jump legs the greedy chooser would not take (route.py)
    ap.add_argument('--guard', type=int, default=0)          # 1: the fall guard (guard.py)
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    log_f = open(os.path.join(OUT_DIR, f'{a.tag}_{a.map}.log'), 'a', buffering=1)

    def log(m):
        line = f'[{time.strftime("%H:%M:%S")}] {m}'
        print(line, flush=True); log_f.write(line + '\n')
    play(a.port, a.map, a.rounds, a.memory, a.tag, log, shortcuts=a.shortcuts, rescue=bool(a.rescue),
         route_on=bool(a.route), guard_on=bool(a.guard))


if __name__ == '__main__':
    main()
