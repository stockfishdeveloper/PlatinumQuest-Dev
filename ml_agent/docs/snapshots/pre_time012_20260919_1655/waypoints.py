"""Waypoint task: segments (start -> goal), the map-free reward, outcomes.

A segment starts from wherever the marble is (or from a random walkable spot,
teleported, TELEPORT_P of the time), draws a reachable goal 8-40 u away, and
ends on arrival, on a fall (the game's OOB flag; the game respawns the
marble), on a timeout, or when the round ends.

Reward per decision (64 ms), all map-independent:
    PROGRESS * (path_dist_prev - path_dist_now)   path distance from the goal's Dijkstra field,
                                                   so going around a hole counts as progress
  + ARRIVE   on arrival (horizontal < ARRIVE_R, |dz| < ARRIVE_DZ)
  - FALL     on OOB
  - TIME     every decision
  - AIR      every decision the marble is airborne
"""
import math
import time
import numpy as np

from nav.protocol import NOOP_ACTION

PROGRESS = 1.0
ARRIVE = 10.0
FALL = 10.0                    # was 20: twice the arrival bonus made the policy freeze and brake 73 % of
                               # the time on King of the Marble rather than move (2026-09-17)
TIME = 0.05                    # was 0.02: on King of the Marble the policy sat still (timeouts) rather
                               # than risk a fall; a 20 s timeout must cost more than a fall (2026-09-17)
AIR = 0.1                      # per decision airborne BEYOND the grace period below
AIR_GRACE = 16                 # ~1 s: a purposeful jump is free; tumbling / falling still costs
                               # (2026-09-17: charging every airborne decision taught the policy that
                               # jumping never pays, so it never jumped gaps)
JUMP_TAKEOFF = 0.4             # per jump COMMANDED WHILE ON THE FLOOR (an actual takeoff; a jump
                               # pressed in mid-air does nothing in the engine and is not charged).
                               # 2026-09-18: on King of the Marble the policy jumped on 32 % of
                               # decisions and was airborne 51 % of the time with the gap prior
                               # active on only 0.5 % -- compulsive bouncing, not gap crossing.
                               # Airborne = almost no control, so it fell: segments with a
                               # below-median jump rate arrived 96 % of the time, above-median 11 %.
                               # AIR alone could not stop it because 71 % of airborne stretches are
                               # shorter than AIR_GRACE and cost nothing. Charging the takeoff
                               # instead of the airtime keeps a real gap jump cheap (one fee that
                               # the progress reward covers) while bouncing pays on every hop.
BRAKE = 0.05                   # 0.1 -> 0.05 on 2026-09-19 03:05. With braking re-enabled the policy used
                               # it on only 0.6 % of decisions (human: 14.3 %): its head was frozen while
                               # the action was disabled, and at 0.1 the immediate cost outweighs a benefit
                               # (not falling) that arrives seconds later and rarely, so PPO was pushing
                               # usage DOWN from the 4.7 % prior rather than discovering the payoff.
                               # 0.05 keeps it non-free (the 2026-09-17 collapse happened when it was free)
                               # while making exploration affordable.
                               # per decision with the brake held: on King of the Marble the policy
                               # settled into braking 52 % of the time (1 u/s, 5 % arrivals) because
                               # braking was free (2026-09-17)
ARRIVE_R = 1.5                 # initial horizontal arrival radius (u, marble centre to waypoint)
ARRIVE_DZ = 2.0                # initial vertical tolerance
# Measured gem pickup (2026-09-17, FlatGemTraining, marble resting beside a gem): collected at
# 0.6 u centre-to-centre, not at 0.7 -> the real hitbox is ~0.65 u (marble radius 0.2 + gem box).
ARRIVE_R_FINAL = 0.65
ARRIVE_DZ_FINAL = 0.8
# Ratchet: whenever the rolling arrival rate is >= ARRIVE_TIGHTEN_AT the radius shrinks by
# ARRIVE_STEP (never grows), so the requirement ends up as strict as a real gem pickup.
ARRIVE_TIGHTEN_AT = 75.0       # percent, over the last 300 segments of the current map
ARRIVE_STEP = 0.15
ARRIVE_MIN_SEGMENTS = 150      # segments on the current map before the ratchet may act
TIMEOUT_DECISIONS = 312        # 20 s at 64 ms
PROGRESS_CLIP = 3.0            # per decision, guards against teleports / respawns
TRAVEL_CLIP = 1.5              # u per decision counted as travel (8 u/s = 0.5 u); respawn jumps are not travel
TELEPORT_P = 1.0               # every segment starts with a verified teleport: it is the only way to clear the
                               # spin the game leaves on a respawned or arriving marble (2026-09-17)
GOAL_DMIN, GOAL_DMAX = 8.0, 40.0
TELEPORT_Z = 0.6               # above the floor
RESPAWN_WAIT_DECISIONS = 120   # ~8 s of sim time before we start forcing teleports
OFFMAP_FALL_DECISIONS = 4      # ~0.25 s below the lowest floor / outside the grid = a fall. The game's
                               # OOB flag is unreliable for a second fall soon after a first one, so the
                               # env decides itself and forces the respawn. RESPAWN only reaches the
                               # client marble while it is still ABOVE the out-of-bounds plane (about
                               # 10 u below the floor); once the marble is in the OOB state the client
                               # keeps falling regardless, so respawn early (2026-09-17).
POST_RESPAWN_TICKS = 20        # the game ignores TELEPORT while the respawned marble is still dropping in
FORCE_RESPAWN_DECISIONS = 30   # re-send RESPAWN every this many decisions while still off the map
REST_WAIT_TICKS = 90           # max ticks to wait for the marble to come to rest before a teleport

# --- gem groups (2026-09-19) --------------------------------------------------------------
GEM_GROUP_MIN, GEM_GROUP_MAX = 4, 8   # real hunt gems spawn in groups of this size
GROUP_LINK_DMIN, GROUP_LINK_DMAX = 6.0, 22.0   # spacing between consecutive gems in a group
TIMEOUT_PER_GEM = 312          # 20 s of budget per gem still to collect (was a flat TIMEOUT_DECISIONS)
FALL_CONTINUE = True           # a fall respawns and CONTINUES the group instead of ending it, because
                               # that is what happens in a real round: the gems stay on the map and
                               # the fall costs time, not the group (2026-09-19)
MAX_FALLS_PER_SEGMENT = 3      # past this the segment ends as 'fell', so a marble that cannot stop
                               # falling does not burn the whole round in one group


class RoundOver(Exception):
    """The round ended while a segment was being set up."""


class Segment:
    """One gem GROUP. `goal` is the gem currently being chased; `remaining` the rest of the group.
    Collecting a gem does NOT end the segment and does NOT reset the recurrent state -- the marble
    keeps its momentum and heads for the next one, which is the whole point of grouping."""
    __slots__ = ('goal', 'field', 'path_len', 'prev_d', 'decisions', 'travelled', 'last_pos', 'outcome',
                 'start', 'offmap', 'remaining', 'collected', 'group_size', 'pickup_speeds', 'falls')

    def __init__(self, goal, field, path_len, start, remaining=()):
        self.goal = goal; self.field = field; self.path_len = path_len; self.start = start
        self.prev_d = None; self.decisions = 0; self.travelled = 0.0; self.last_pos = None; self.outcome = None
        self.offmap = 0
        self.remaining = list(remaining)      # gems still to collect after `goal`
        self.collected = 0
        self.group_size = 1 + len(self.remaining)
        self.pickup_speeds = []               # horizontal speed at each pickup: the control metric
        self.falls = 0                        # falls SURVIVED in this group (see FALL_CONTINUE)


class SegmentManager:
    def __init__(self, terrain, rng, log=print, arrive_r=ARRIVE_R, arrive_dz=ARRIVE_DZ):
        self.terrain = terrain
        self.rng = rng
        self.log = log
        self.seg = None
        self.history = []          # recent outcomes: dicts
        self.last_outcome = None
        self.arrive_r = float(arrive_r)
        self.arrive_dz = float(arrive_dz)
        self.pending_mark = None       # (x, y, z) the worker should show as the current target
        self.pending_respawn = False   # set by step() when a fall needs a respawn mid-group

    def take_respawn(self):
        r = self.pending_respawn; self.pending_respawn = False
        return r

    def recover(self, env):
        """Put the marble back on the map after a mid-group fall and keep the SAME goal.
        The gems are still there, exactly as in a real round; only prev_d and the travel anchor
        move, because the marble did."""
        s = self.seg
        waited = 0
        while not self._on_map(env.pos()):
            if waited % FORCE_RESPAWN_DECISIONS == 0:
                env.control('RESPAWN')
                if env.round_ended or env.reconnected:
                    raise RoundOver()
                self._step_checked(env, POST_RESPAWN_TICKS)
            self._step_checked(env, 1)
            waited += 1
            if waited > RESPAWN_WAIT_DECISIONS * 2:
                raise RoundOver()                 # cannot get back on the map: let the caller reset
        self._settle(env)
        x, y, z = (float(v) for v in env.pos())
        gx, gy, _ = s.goal
        s.prev_d = self.terrain.dist_at(s.field, x, y, (gx, gy))   # field is from the goal: still valid
        s.last_pos = np.array([x, y, z])
        s.offmap = 0

    def next_goal(self):
        """The gem the greedy order will hand over after the current one, or None on the last gem.
        Nearest remaining to where the marble WILL be, i.e. to the current goal."""
        s = self.seg
        if s is None or not s.remaining:
            return None
        gx, gy, _ = s.goal
        i = int(np.argmin([math.hypot(g[0] - gx, g[1] - gy) for g in s.remaining]))
        return (s.remaining[i][0], s.remaining[i][1], s.remaining[i][2])

    def take_mark(self):
        m = self.pending_mark; self.pending_mark = None
        return m

    def _sample_group(self, x, y):
        """A chain of 4-8 reachable gems: the first GOAL_DMIN..GOAL_DMAX from the marble, each
        later one GROUP_LINK_DMIN..GROUP_LINK_DMAX from its predecessor, like a real gem cluster.
        Returns [(gx, gy, gz, path_len), ...] or None if even the first cannot be placed."""
        first = self.terrain.sample_goal(x, y, self.rng, GOAL_DMIN, GOAL_DMAX)
        if first is None:
            return None
        group = [first]
        n = int(self.rng.integers(GEM_GROUP_MIN, GEM_GROUP_MAX + 1))
        px, py = first[0], first[1]
        for _ in range(n - 1):
            g = self.terrain.sample_goal(px, py, self.rng, GROUP_LINK_DMIN, GROUP_LINK_DMAX)
            if g is None:
                break
            group.append(g); px, py = g[0], g[1]
        return group

    def _advance_goal(self, x, y):
        """Pick the nearest remaining gem (greedy stand-in for the planner) and retarget onto it.
        Recomputes the Dijkstra field and resets prev_d so the goal switch awards no progress."""
        s = self.seg
        i = int(np.argmin([math.hypot(g[0] - x, g[1] - y) for g in s.remaining]))
        gx, gy, gz, path_len = s.remaining.pop(i)
        s.goal = (gx, gy, gz); s.path_len = path_len
        s.field = self.terrain.goal_field(gx, gy)
        s.prev_d = self.terrain.dist_at(s.field, x, y, (gx, gy))
        self.pending_mark = (gx, gy, gz)

    def maybe_tighten(self):
        """Shrink the arrival radius one step if the policy is reliably arriving. Returns True if it did."""
        s = self.stats()
        if s['segments'] >= ARRIVE_MIN_SEGMENTS and s['arrive_pct'] >= ARRIVE_TIGHTEN_AT and self.arrive_r > ARRIVE_R_FINAL + 1e-6:
            self.arrive_r = max(ARRIVE_R_FINAL, self.arrive_r - ARRIVE_STEP)
            self.arrive_dz = max(ARRIVE_DZ_FINAL, self.arrive_dz - ARRIVE_STEP)
            self.history = []      # the stat must be re-earned at the new radius
            self.log(f'ARRIVE radius tightened to {self.arrive_r:.2f} u (dz {self.arrive_dz:.2f})')
            return True
        return False

    def begin(self, env, teleport=None):
        """Start a segment. Returns the goal (x, y, z). Teleports first with probability TELEPORT_P
        (or always/never if `teleport` is given)."""
        do_tp = self.rng.random() < TELEPORT_P if teleport is None else teleport
        # After an arrival always teleport: a waypoint near an edge reached at speed otherwise
        # carried the marble over the edge on momentum and booked a false fall against the next
        # segment (2026-09-17). Chained waypoints along a path are the planner's job later.
        if self.last_outcome == 'arrived':
            do_tp = True
        t_begin = time.perf_counter(); ticks_begin = env.ticks
        # A fresh observation (after a fall the last message carried the pre-fall edge
        # position), then wait until the marble is back on the map: while it is still
        # falling / being respawned the game ignores TELEPORT.
        self._step_checked(env, 2)
        waited = 0
        while not self._on_map(env.pos()):
            if waited % FORCE_RESPAWN_DECISIONS == 0:
                # falling: respawn on the server right away (works mid-fall; the game's own
                # OOB respawn is unreliable after a recent fall and disabled at round end)
                if waited > 0:
                    self.log(f'marble still off the map after {waited} decisions at {np.round(env.pos(), 1)} '
                             f'(time left {env.time_left_s():.0f}s); forcing a respawn again')
                p_before = np.round(env.pos(), 1)
                env.control('RESPAWN')
                if env.round_ended or env.reconnected:
                    raise RoundOver()
                p_after = np.round(env.pos(), 1)
                self._step_checked(env, POST_RESPAWN_TICKS)
                if not self._on_map(env.pos()):
                    self.log(f'RESPAWN did not take: before {p_before} -> next obs {p_after} -> after {POST_RESPAWN_TICKS} ticks '
                             f'{np.round(env.pos(), 1)} (time left {env.time_left_s():.0f}s)')
            self._step_checked(env, 1)
            waited += 1
        # A marble that is not rolling (fresh respawn drop-in, or after a teleport) must be
        # settled on the floor before the segment starts: segments that began while the
        # respawned marble was still dropping/bouncing failed 73 % vs 27 % (2026-09-17).
        # A rolling marble is NOT held here (it would roll off uncontrolled).
        self._settle(env)
        g = None
        for attempt in range(6):
            if do_tp or attempt > 0:
                sx, sy, sz = self.terrain.sample_start(self.rng)
                ok = self._teleport_checked(env, sx, sy, sz + TELEPORT_Z)
                if not ok:
                    self.log(f'teleport to {(round(sx, 1), round(sy, 1))} ignored; marble at {np.round(env.pos(), 1)}')
            if do_tp or attempt > 0:
                self._settle(env)
            x, y, z = (float(v) for v in env.pos())
            g = self._sample_group(x, y)
            if g is not None:
                break
            self.log(f'no reachable goal from {(round(x, 1), round(y, 1), round(z, 1))}; teleporting')
        if g is None:
            raise RuntimeError('no reachable goal after 6 teleports; check the terrain map')
        gx, gy, gz, path_len = g[0]
        env.mark(gx, gy, gz)
        dt = time.perf_counter() - t_begin
        if dt > 5.0:
            self.log(f'segment start took {dt:.0f} s wall / {env.ticks - ticks_begin} ticks (teleport {do_tp}, waited {waited})')
        field = self.terrain.goal_field(gx, gy)
        self.seg = Segment((gx, gy, gz), field, path_len, (x, y, z), remaining=g[1:])
        self.seg.prev_d = self.terrain.dist_at(field, x, y, (gx, gy))
        self.seg.last_pos = np.array([x, y, z])
        return self.seg.goal

    def _settle(self, env, max_ticks=REST_WAIT_TICKS):
        """If the marble is not rolling, wait until it rests on the floor (bounces finished)."""
        for _ in range(max_ticks):
            p = env.pos(); v = env.vel()
            if float(np.hypot(v[0], v[1])) > 1.0:
                return                                   # rolling: do not hold it uncontrolled
            floor = self.terrain.floor_z(float(p[0]), float(p[1]), float(p[2]))
            if abs(float(v[2])) < 0.05 and abs(float(p[2]) - floor) < 0.6:
                return
            self._step_checked(env, 1)

    def _teleport_checked(self, env, x, y, z, tries=3):
        """Teleport and verify; the game drops TELEPORT for a few ticks after a respawn."""
        for _ in range(tries):
            env.teleport(x, y, z)
            if env.round_ended or env.reconnected:
                raise RoundOver()
            p = env.pos()
            if abs(p[0] - x) < 1.0 and abs(p[1] - y) < 1.0:
                return True
            self._step_checked(env, POST_RESPAWN_TICKS)
        return False

    def _step_checked(self, env, n):
        for _ in range(n):
            env.step(NOOP_ACTION, repeat=1)
            if env.round_ended or env.reconnected:
                raise RoundOver()

    def abandon(self):
        """Drop the current segment without recording an outcome (round ended)."""
        self.seg = None
        self.last_outcome = None

    def _on_map(self, pos):
        x, y, z = (float(v) for v in pos)
        return self.terrain.contains(x, y) and z >= self.terrain.z_floor_min - 1.0

    def step(self, pos, fell, airborne, round_ended, elapsed_s=0.0, braked=False, airborne_decisions=0,
             jumped=False, vel=(0.0, 0.0)):
        """Reward and (done, outcome) for the decision that led to `pos`. `airborne` is the flag
        for this decision, `airborne_decisions` how many consecutive decisions it has been airborne,
        `jumped` whether a jump was commanded (charged only when it was a real takeoff, i.e. the
        marble was on the floor)."""
        s = self.seg
        self._elapsed = elapsed_s
        self._vel = vel
        x, y, z = (float(v) for v in pos)
        gx, gy, gz = s.goal
        s.decisions += 1
        p = np.array([x, y, z])
        if not fell and not airborne:
            s.travelled += min(float(np.linalg.norm(p[:2] - s.last_pos[:2])), TRAVEL_CLIP)
        s.last_pos = p
        d = self.terrain.dist_at(s.field, x, y, (gx, gy))
        progress = max(-PROGRESS_CLIP, min(PROGRESS_CLIP, s.prev_d - d))
        s.prev_d = d
        air_cost = AIR if (airborne and airborne_decisions > AIR_GRACE) else 0.0
        takeoff_cost = JUMP_TAKEOFF if (jumped and not airborne) else 0.0
        r = PROGRESS * progress - TIME - air_cost - takeoff_cost - (BRAKE if braked else 0.0)
        done, outcome = False, None
        # off the map (below the lowest floor / outside the grid) without the game's OOB flag:
        # after OFFMAP_FALL_DECISIONS decisions count it as a fall ourselves
        if not self._on_map(p) or z < self.terrain.z_floor_min - 3.0:
            s.offmap += 1
        else:
            s.offmap = 0
        if not fell and s.offmap >= OFFMAP_FALL_DECISIONS:
            fell = True
            self.log(f'fall without OOB flag: {s.offmap} decisions off the map at {np.round(p, 1)} (time left {self._elapsed:.0f}s)')
        if fell:
            r -= FALL
            s.falls += 1
            if FALL_CONTINUE and s.falls < MAX_FALLS_PER_SEGMENT:
                self.pending_respawn = True       # the worker respawns us; the group is NOT lost
            else:
                done, outcome = True, 'fell'
        elif math.hypot(gx - x, gy - y) < self.arrive_r and abs(gz - z) < self.arrive_dz:
            r += ARRIVE
            s.collected += 1
            s.pickup_speeds.append(float(math.hypot(float(self._vel[0]), float(self._vel[1]))))
            if s.remaining:
                # keep going: no teleport, no reset, momentum carries into the next gem
                self._advance_goal(x, y)
            else:
                done, outcome = True, 'arrived'          # whole group collected
        elif s.decisions >= TIMEOUT_PER_GEM * (1 + len(s.remaining)):
            done, outcome = True, 'timeout'
        elif round_ended:
            done, outcome = True, 'round'
        if done:
            s.outcome = outcome
            self.last_outcome = outcome
            self.history.append({'outcome': outcome, 'decisions': s.decisions, 'path_len': s.path_len,
                                 'travelled': s.travelled, 'speed': s.travelled / (s.decisions * 0.064),
                                 'collected': s.collected, 'group_size': s.group_size, 'falls': s.falls,
                                 'pickup_speed': (sum(s.pickup_speeds) / len(s.pickup_speeds)) if s.pickup_speeds else 0.0})
            if len(self.history) > 2000:
                self.history = self.history[-2000:]
        return r, done, outcome

    def stats(self, last=300):
        h = [x for x in self.history[-last:] if x['outcome'] != 'round']
        if not h:
            return {'segments': 0, 'arrive_pct': 0.0, 'falls_per_100u': 0.0, 'speed': 0.0, 'timeout_pct': 0.0,
                    'gems_pct': 0.0, 'gems_per_group': 0.0, 'pickup_speed': 0.0}
        n = len(h)
        trav = sum(x['travelled'] for x in h)
        # falls no longer end a segment (FALL_CONTINUE), so count them, not 'fell' outcomes
        falls = sum(x.get('falls', 1 if x['outcome'] == 'fell' else 0) for x in h)
        picked = sum(x.get('collected', 0) for x in h)
        offered = sum(x.get('group_size', 1) for x in h)
        psp = [x['pickup_speed'] for x in h if x.get('pickup_speed')]
        return {'segments': n,
                'gems_pct': 100.0 * picked / max(offered, 1),          # gems collected of gems offered
                'gems_per_group': picked / n,
                'pickup_speed': (sum(psp) / len(psp)) if psp else 0.0,  # control metric: lower = under control
                'arrive_pct': 100.0 * sum(1 for x in h if x['outcome'] == 'arrived') / n,
                'falls_per_100u': 100.0 * falls / max(trav, 1.0),
                'speed': trav / max(sum(x['decisions'] for x in h) * 0.064, 1e-6),
                'timeout_pct': 100.0 * sum(1 for x in h if x['outcome'] == 'timeout') / n}
