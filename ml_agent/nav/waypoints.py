"""Waypoint task: segments (start -> goal), the map-free reward, outcomes.

A segment starts from wherever the marble is (or from a random walkable spot,
teleported, TELEPORT_P of the time), draws a reachable goal 8-40 u away, and
ends on arrival, on a fall (the game's OOB flag; the game respawns the
marble), on a timeout, or when the round ends.

Reward per decision (64 ms), all map-independent:
    PROGRESS * (path_dist_prev - path_dist_now)   path distance from the goal's Dijkstra field,
                                                   so going around a hole counts as progress
  + ARRIVE   on arrival (horizontal < ARRIVE_R, |dz| < ARRIVE_DZ)
  - FALL     on OOB (and after the respawn no progress is paid until the marble is back inside
               the distance it fell from: the trip back is not new ground, 2026-09-21)
  - TIME     every decision
  - AIR      every decision the marble is airborne
"""
import math
import time
import numpy as np

from nav.protocol import NOOP_ACTION
from nav.terrain import JUMP_GAP, JUMP_DROP, JUMP_RISE   # edge_time_cost's jump exemption

PROGRESS = 1.0
ARRIVE = 10.0
GEM_SPEED_BONUS = 8.0          # paid ON COLLECTION, scaled by how quickly THIS gem was reached:
                               # GEM_SPEED_BONUS * max(0, 1 - gem_decisions / GEM_SPEED_REF).
                               # Pure outcome pressure -- nothing about direction or throttle.
                               # human pace ~24 decisions -> 4.8 | agent median 36 -> 3.0 | 60+ -> 0.
                               # Closing to human pace is worth ~+1.8 a gem, ~11 a group against ~60
                               # for the group. Cannot be farmed by abandoning a gem: only paid on
                               # collection. Raising flat TIME instead made speed WORSE (see TIME).
GEM_SPEED_REF = 60             # decisions at which the bonus reaches zero (~3.8 s)
CARRY = 0.0                    # DISABLED 2026-09-20 11:55 after a null result. Paid ON COLLECTION for
                               # momentum pointing at the NEXT gem. Two calibrations and 160 updates
                               # moved the measured arrival alignment not at all (carry held 1.1-1.2
                               # against a human 2.28), while falls rose 0.95 -> 1.28 and speed fell
                               # 4.8 -> 4.7. Best read on why: the bonus fires once, at the instant of
                               # pickup, for a property that can only be produced by shaping the whole
                               # approach 10-20 decisions earlier, and every decision in between gets
                               # identical feedback. If retried, make it DENSE (a small per-decision
                               # payment for alignment during the approach) rather than a lump sum.
                               # Set to 0.0 rather than deleted so the carry= metric keeps reporting.
                               # back to 6.0 from 12.0 once CARRY_REF was fixed. The doubling was
                               # compensating for a flat response whose real cause was the wrong
                               # reference; with REF = 2.5, CARRY = 12 would pay ~11 at human
                               # carry, equal to ARRIVE, and invite trading arrivals for speed.
                               # At 6.0 human-level carry pays ~5.5, about half of ARRIVE.
                               # CARRY * min(1, max(0, v . unit(next - pos)) / CARRY_REF).
                               # GEM_SPEED_BONUS pays for the time a gem TOOK; this pays for the
                               # speed the marble still HAS when it touches it, which is the
                               # measured defect: the agent floors at 4.46 u/s at every pickup
                               # where the human floors at 7.37 and never drops below ~6.3.
                               # Worth ~2.25 a gem now, ~5.55 at human carry, against ARRIVE 10
                               # and an expected fall cost of ~0.57 a gem. Only the component
                               # toward the next gem counts, so speed in a useless direction
                               # earns nothing, and it is paid only on collection so it cannot
                               # be farmed by overshooting. The last gem of a group is excluded
                               # (no next gem to carry into).
CARRY_REF = 2.5                # u/s at which the carry bonus saturates. MEASURED on the human demo
                               # 2026-09-20 11:30, 682 pickups: mean arrival angle to the next gem
                               # 72 deg, mean carry 2.28 u/s, only 19 % of pickups within 45 deg.
                               # The human does NOT arrive aimed at the next gem either; they
                               # arrive FASTER and somewhat better aimed. carry = speed*cos(angle)
                               # captures both: human 7.4*cos(72) = 2.29, agent 5.4*cos(90) = 0.
                               # This was 8.0, from misreading the human's SPEED at 1-2 u from the
                               # gem (7.37 u/s) as their ALIGNED speed. That put the entire
                               # achievable range in the bottom 29 % of the curve and cut the
                               # gradient to a third, which is why 96 updates moved nothing.
CARRY_FLOOR = -0.5             # the term is TWO-SIDED: arriving aimed AWAY from the next gem is
                               # charged, down to CARRY * CARRY_FLOOR. Revised 2026-09-20 11:00
                               # after 96 updates of the one-sided version produced no movement
                               # at all (measured arrival angle to the next gem: 89 deg -> 91 deg
                               # on KOTM, i.e. perpendicular and uncorrelated). Two faults:
                               # (1) clamping at zero paid ~1.6 a gem for a coin flip, because a
                               # symmetric spread clamped at zero averages positive, so most of
                               # the term was a constant with no gradient in it; (2) one-sided
                               # meant 180 deg scored the same as 90 deg, so nothing pushed the
                               # policy out of the bad half of the distribution. CARRY also went
                               # 6.0 -> 12.0, pre-registered before the 90-update check.
EDGE_K = 1.0                   # RESTORED 09:05 on 2026-09-21. I set this to 0.0 at 07:50 on the
                               # suspicion that it was buying falls with speed, but the user stopped
                               # training two minutes later so that ablation NEVER RAN. 1.0 is the
                               # value that was in the configuration which actually halved KOTM
                               # falls100 overnight (0.73 -> 0.42), and the earlier "edge term does
                               # nothing" readings were all taken while the fall-credit bug
                               # (section 17) still made falling profitable. Do not zero it again
                               # without an eval behind it.
                               # (superseded note from the ablation: the fall-credit bug (section 17) was the
                               # real cause of the falls, and with it fixed the 8-round eval showed
                               # falls 0.65 -> 0.39 (t = -2.95) but speed 6.69 -> 6.36 (t = -2.59).
                               # The edge term charges 15.7 % of KOTM floor decisions, so it is the
                               # prime suspect for that speed cost. 0.0 disables it while keeping the
                               # code and the metric; restore 1.0 if falls100 rises above ~0.55.
                               # (v2 at 00:52 on 2026-09-21) (v1: 0.5 then 1.5, both left KOTM falls100
                               # flat at 0.73 because the charge began at 0.6 s, AFTER the policy's
                               # own median 0.47 s reaction point). Per decision, at a drop with no
                               # stopping time left; the integrated charge over a 1.2 s approach at
                               # 6 u/s is ~9, i.e. about one FALL, paid progressively and avoidably.
                               # Dense shaping for the fall mechanism measured on 10,225 KOTM training
                               # falls: 87 % were the marble ROLLING off an edge (not jumping) with the
                               # command a median 81 deg off the velocity 0.5 s earlier, i.e. turning
                               # too late. FALL alone is one sparse hit per ~312 decisions and falls100
                               # was rising (0.64 -> 0.73) under it. Charge = EDGE_K * (1 - t/EDGE_T_SAFE)
                               # where t = distance to the first non-walkable cell along the velocity,
                               # divided by speed. A run-up at a JUMPABLE gap is exempt (see
                               # edge_time_cost), so gap jumping on Islands is untouched.
EDGE_T_SAFE = 1.2              # s of time-to-edge below which the charge starts. 0.6 -> 1.2 in v2: the
                               # policy was first turning away at a median 0.47 s (2.8 u at 6 u/s,
                               # less than a 90 deg turn needs), so the signal must start earlier
EDGE_LOOK = 9.0                # u: how far ahead along the velocity to look for a drop (>= T_SAFE * 7 u/s)
EDGE_GOAL_LATERAL = 1.5        # u: a goal within this of the velocity ray and BEFORE the drop makes the
                               # approach a pickup, not a shortcut: no charge (1 % of falls, 15 % of v1 charges)
EDGE_V_MIN = 1.5               # u/s: slower than this the marble stops within a cell; no charge
FALL = 25.0                    # 10 -> 25 at 01:12 on 2026-09-21. Measured on the gems-only real rounds: a
                               # fall costs ~7.5 s of respawn and recovery = ~55-60 reward units of
                               # forgone gems, against 10 + ~6 TIME charged here, so shortcuts with a
                               # few % fall risk were RATIONAL under the old price and three settings of
                               # the edge-time term (330 updates) could not move falls100 (0.73-0.79).
                               # Braking is no longer free (BRAKE 0.05, BRAKE_SUPPRESS), which is what
                               # made 20 collapse on 2026-09-17. Revert to 15 if speed drops > 10 % or
                               # the brake share passes 20 % of decisions.
                               # (older note) was 20: twice the arrival bonus made the policy freeze and brake 73 % of
                               # the time on King of the Marble rather than move (2026-09-17)
TIME = 0.05                    # 0.12 reverted 2026-09-19 evening: raising it to push speed did the
                               # OPPOSITE on KOTM (speed 4.40 -> 4.11, gems/grp 5.44 -> 5.10 over 49
                               # updates, then flat). A constant per-decision cost mostly shifts the
                               # value baseline; it is a weak lever on pace. Do not simply retry it.
                               # (Original note: was 0.02 -- on King of the Marble the policy sat still.)
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
TURN_COST = 0.0                # per decision: TURN_COST * (1 - cos(angle between this commanded
                               # direction and the last)). 0 for holding a line, 2*TURN_COST for a
                               # reversal. The policy swings its commanded heading a measured
                               # 40 deg every 64 ms where the human swings 2.67 deg MEAN (measured
                               # 2026-09-20 over 60,836 demo decisions; median 0.0, p90 4.2, p99 45.0,
                               # and 54.7 % of human decisions change heading by EXACTLY zero because
                               # the key is held). The often-quoted 0.4 deg was wrong. The human shape
                               # is what this term rewards: hold a line for free, pay rarely for a real
                               # turn -- their mean is carried by the 4.6 % of decisions above 10 deg.
                               # Sampling noise
                               # explains less than half of that, so the MEAN output is oscillating
                               # ~36 deg per decision. It therefore cannot sustain thrust (mid-leg
                               # speed 7.3 u/s against a human 10.6) and cannot hold an approach
                               # geometry (which is why CARRY was a null at any calibration).
                               # Per ~230-decision segment against a group worth ~78: current
                               # jitter 8.1, human-like 0.001, six real 90 deg turns 0.9. Jitter is
                               # paid 230 times a segment and a genuine turn a handful, so the cost
                               # falls on twitching rather than turning. See HANDOFF section 8.
                               # REVERTED TO 0.0 on 2026-09-20 14:50. It worked as designed and the
                               # design was wrong. Reaching a gem REQUIRES turning toward it, so
                               # charging for turning made the policy turn less, and the cheapest way
                               # to turn less while still hitting a 0.65 u target is TO GO SLOWER: a
                               # slower marble needs less heading change per decision to track the
                               # same curve. Pickup speed fell monotonically with every raise --
                               # 5.19 -> 4.98 -> 4.82 -> 4.70 -> 4.64 -- and pickup speed is exactly
                               # the floor each leg accelerates from. Against that, section 10 measures
                               # jitter as worth only 10-12 % of speed, and at 0.6 the mean-direction
                               # jitter did not move at all (27.2 -> 27.5 over 72 updates).
                               # 0.15 -> 0.3 on 2026-09-20 13:10: at 0.15 the jitter fell only
                               # 40.4 -> 38.7 deg over 92 updates, monotone but far too slow, while
                               # arrivals held at 86 % and falls stayed flat. The gradient works and
                               # nothing was being harmed, so the term was simply too small: ~8 per
                               # segment against a group worth ~78, where GEM_SPEED_BONUS (which did
                               # move behaviour) is worth up to 48. At 0.3 a jittering segment
                               # carries ~16, about a fifth of the group.
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


def edge_time_cost(terrain, x, y, vx, vy, goal=None):
    """EDGE_K * (1 - time_to_edge / EDGE_T_SAFE), clipped to [0, EDGE_K]. Zero when no drop lies
    within EDGE_LOOK along the velocity, when the marble is slower than EDGE_V_MIN, when the
    drop is a jumpable gap (walkable landing within JUMP_GAP past it, height within the walk
    graph's jump limits) so a run-up for a jump is never charged, or when `goal` (x, y) lies on
    the velocity ray BEFORE the drop (a pickup approach, not a shortcut across a hole)."""
    v = math.hypot(vx, vy)
    if v < EDGE_V_MIN:
        return 0.0
    ux, uy = vx / v, vy / v
    step = terrain.walk_res * 0.5
    j0, i0 = terrain.cell_of(x, y)
    here_z = terrain.walk_top[j0, i0] if (terrain.in_walk_grid(j0, i0) and terrain.walkable[j0, i0]) else None
    d_edge = None
    n = int(EDGE_LOOK / step)
    for k in range(1, n + 1):
        d = k * step
        j, i = terrain.cell_of(x + ux * d, y + uy * d)
        if not terrain.in_walk_grid(j, i) or not terrain.walkable[j, i]:
            d_edge = d
            break
    if d_edge is None:
        return 0.0
    if goal is not None:
        # the gem sits between the marble and the drop: this is the approach to a pickup
        along = (float(goal[0]) - x) * ux + (float(goal[1]) - y) * uy
        lateral = abs(-(float(goal[0]) - x) * uy + (float(goal[1]) - y) * ux)
        if 0.0 < along < d_edge and lateral < EDGE_GOAL_LATERAL:
            return 0.0
    # jumpable gap ahead? look past the drop for a landing cell within JUMP_GAP
    if here_z is not None:
        m = int(JUMP_GAP / step)
        for k in range(1, m + 1):
            d = d_edge + k * step
            j, i = terrain.cell_of(x + ux * d, y + uy * d)
            if not terrain.in_walk_grid(j, i):
                break
            if terrain.walkable[j, i]:
                dz = float(terrain.walk_top[j, i]) - float(here_z)
                if -JUMP_DROP <= dz <= JUMP_RISE:
                    return 0.0                      # a real gap the walk graph would jump: run-up is free
                break
    t = d_edge / v
    if t >= EDGE_T_SAFE:
        return 0.0
    return EDGE_K * (1.0 - t / EDGE_T_SAFE)


class RoundOver(Exception):
    """The round ended while a segment was being set up."""


class Segment:
    """One gem GROUP. `goal` is the gem currently being chased; `remaining` the rest of the group.
    Collecting a gem does NOT end the segment and does NOT reset the recurrent state -- the marble
    keeps its momentum and heads for the next one, which is the whole point of grouping."""
    __slots__ = ('fall_mark', 'goal', 'field', 'path_len', 'prev_d', 'decisions', 'travelled', 'last_pos', 'outcome',
                 'start', 'offmap', 'remaining', 'collected', 'group_size', 'pickup_speeds', 'falls',
                 'gem_decisions', 'carry_vals', 'turn_degs')

    def __init__(self, goal, field, path_len, start, remaining=()):
        self.goal = goal; self.field = field; self.path_len = path_len; self.start = start
        self.prev_d = None; self.decisions = 0; self.travelled = 0.0; self.last_pos = None; self.outcome = None
        self.offmap = 0
        self.remaining = list(remaining)      # gems still to collect after `goal`
        self.collected = 0
        self.group_size = 1 + len(self.remaining)
        self.pickup_speeds = []               # horizontal speed at each pickup: the control metric
        self.carry_vals = []                  # speed toward the NEXT gem at each pickup (CARRY)
        self.turn_degs = []                   # commanded heading change per decision, degrees
        self.falls = 0                        # falls SURVIVED in this group (see FALL_CONTINUE)
        self.fall_mark = None                 # path distance at the last on-map decision before a
                                              # fall: no progress credit until back inside it
        self.gem_decisions = 0                # decisions spent on the CURRENT gem, for the speed bonus


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
        self._prev_cmd = None          # last commanded direction, for TURN_COST

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
        s.fall_mark = None                        # new goal, new field: the mark no longer applies
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
        self._prev_cmd = None      # a teleport makes the previous heading meaningless
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
             jumped=False, vel=(0.0, 0.0), cmd_dir=None):
        """Reward and (done, outcome) for the decision that led to `pos`. `airborne` is the flag
        for this decision, `airborne_decisions` how many consecutive decisions it has been airborne,
        `jumped` whether a jump was commanded (charged only when it was a real takeoff, i.e. the
        marble was on the floor)."""
        s = self.seg
        self._elapsed = elapsed_s
        self._vel = vel
        # cost of swinging the commanded heading since the last decision
        turn_cost = 0.0
        if cmd_dir is not None:
            n = math.hypot(float(cmd_dir[0]), float(cmd_dir[1]))
            if n > 1e-6:
                u = (float(cmd_dir[0]) / n, float(cmd_dir[1]) / n)
                if self._prev_cmd is not None:
                    dot = max(-1.0, min(1.0, u[0] * self._prev_cmd[0] + u[1] * self._prev_cmd[1]))
                    turn_cost = TURN_COST * (1.0 - dot)
                    s.turn_degs.append(math.degrees(math.acos(dot)))
                self._prev_cmd = u
        x, y, z = (float(v) for v in pos)
        gx, gy, gz = s.goal
        s.decisions += 1
        s.gem_decisions += 1
        p = np.array([x, y, z])
        if not fell and not airborne:
            s.travelled += min(float(np.linalg.norm(p[:2] - s.last_pos[:2])), TRAVEL_CLIP)
        s.last_pos = p
        d_before = s.prev_d                       # last known distance, for the fall mark below
        d = self.terrain.dist_at(s.field, x, y, (gx, gy))
        if s.fall_mark is not None:
            # Back from a respawn: the game put the marble a median ~17 u away, and paying the
            # trip back as progress made a fall net POSITIVE at FALL = 10 (found 2026-09-21).
            # Nothing is earned until the marble is back inside the distance it fell from.
            if d >= s.fall_mark:
                progress = 0.0
            else:
                progress = max(-PROGRESS_CLIP, min(PROGRESS_CLIP, s.fall_mark - d))
                s.fall_mark = None
        else:
            progress = max(-PROGRESS_CLIP, min(PROGRESS_CLIP, s.prev_d - d))
        s.prev_d = d
        air_cost = AIR if (airborne and airborne_decisions > AIR_GRACE) else 0.0
        takeoff_cost = JUMP_TAKEOFF if (jumped and not airborne) else 0.0
        edge_cost = 0.0 if (airborne or fell) else edge_time_cost(self.terrain, x, y, float(vel[0]), float(vel[1]), goal=(gx, gy))
        r = PROGRESS * progress - TIME - air_cost - takeoff_cost - (BRAKE if braked else 0.0) - turn_cost - edge_cost
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
            if s.fall_mark is None and np.isfinite(d_before):
                s.fall_mark = float(d_before)     # earliest mark wins if it falls again on the way back
            if FALL_CONTINUE and s.falls < MAX_FALLS_PER_SEGMENT:
                self.pending_respawn = True       # the worker respawns us; the group is NOT lost
            else:
                done, outcome = True, 'fell'
        elif math.hypot(gx - x, gy - y) < self.arrive_r and abs(gz - z) < self.arrive_dz:
            r += ARRIVE + GEM_SPEED_BONUS * max(0.0, 1.0 - s.gem_decisions / float(GEM_SPEED_REF))
            s.gem_decisions = 0                   # the next gem is timed from here
            s.collected += 1
            s.pickup_speeds.append(float(math.hypot(float(self._vel[0]), float(self._vel[1]))))
            if s.remaining:
                # keep going: no teleport, no reset, momentum carries into the next gem
                self._advance_goal(x, y)
                # ...and pay for the part of that momentum already aimed at the new gem. Computed
                # AFTER the retarget so it uses the gem actually chosen, not the one just taken.
                nx, ny = s.goal[0] - x, s.goal[1] - y
                nd = math.hypot(nx, ny)
                if nd > 1e-6:
                    v_toward = (float(self._vel[0]) * nx + float(self._vel[1]) * ny) / nd
                    s.carry_vals.append(v_toward)          # RAW, signed: the honest metric
                    r += CARRY * max(CARRY_FLOOR, min(1.0, v_toward / CARRY_REF))
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
                                 'pickup_speed': (sum(s.pickup_speeds) / len(s.pickup_speeds)) if s.pickup_speeds else 0.0,
                                 'carry_speed': (sum(s.carry_vals) / len(s.carry_vals)) if s.carry_vals else 0.0,
                                 'turn_deg': (sum(s.turn_degs) / len(s.turn_degs)) if s.turn_degs else 0.0})
            if len(self.history) > 2000:
                self.history = self.history[-2000:]
        return r, done, outcome

    def stats(self, last=300):
        h = [x for x in self.history[-last:] if x['outcome'] != 'round']
        if not h:
            return {'segments': 0, 'arrive_pct': 0.0, 'falls_per_100u': 0.0, 'speed': 0.0, 'timeout_pct': 0.0,
                    'gems_pct': 0.0, 'gems_per_group': 0.0, 'pickup_speed': 0.0, 'carry_speed': 0.0,
                    'turn_deg': 0.0}
        n = len(h)
        trav = sum(x['travelled'] for x in h)
        # falls no longer end a segment (FALL_CONTINUE), so count them, not 'fell' outcomes
        falls = sum(x.get('falls', 1 if x['outcome'] == 'fell' else 0) for x in h)
        picked = sum(x.get('collected', 0) for x in h)
        offered = sum(x.get('group_size', 1) for x in h)
        psp = [x['pickup_speed'] for x in h if x.get('pickup_speed')]
        csp = [x['carry_speed'] for x in h if x.get('carry_speed') is not None]
        tdg = [x['turn_deg'] for x in h if x.get('turn_deg')]
        return {'segments': n,
                'gems_pct': 100.0 * picked / max(offered, 1),          # gems collected of gems offered
                'gems_per_group': picked / n,
                'pickup_speed': (sum(psp) / len(psp)) if psp else 0.0,  # control metric: lower = under control
                'carry_speed': (sum(csp) / len(csp)) if csp else 0.0,   # what CARRY pays for: higher = better
                'turn_deg': (sum(tdg) / len(tdg)) if tdg else 0.0,      # what TURN_COST pays for: LOWER = better (human 0.4)
                'arrive_pct': 100.0 * sum(1 for x in h if x['outcome'] == 'arrived') / n,
                'falls_per_100u': 100.0 * falls / max(trav, 1.0),
                'speed': trav / max(sum(x['decisions'] for x in h) * 0.064, 1e-6),
                'timeout_pct': 100.0 * sum(1 for x in h if x['outcome'] == 'timeout') / n}
