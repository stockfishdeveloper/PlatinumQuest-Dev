"""Spawn value (2026-11-04, operator: "it thinks 2 seconds ahead ... optimize collecting entire spawns ... save the Blast for the
platform").

The planner's forecast covers 3 s and its end state was priced only by the walking distance to the tour's next gem, so nothing
past that horizon counted: a plan that spent the Blast on a floor gem under a high platform scored the same as one that kept it.
Here every candidate is priced by the model TIME TO CLEAR THE WHOLE SPAWN from its end state:

    cost = SV_W_U_PER_S * (t_end + rest_time(end state, gems not yet collected, Blast meter left)) - 100 per gem the forecast collects

rest_time walks the spawn plan's order (spawn_order.best_order: a dynamic programme over the SV_MAX_GEMS nearest gems, legs on the
walking grid where it has them, turn and climb charges fitted to the operator's demos) from the candidate's end state with the
meter the candidate leaves: a leg up a rise a jump cannot clear needs the Blast (crossings.needed_meter), and if the meter will not
have refilled by then the model waits for it (BLAST_CHARGE_S per full meter, capped at SV_NO_BLAST_ALT_S: another way up). A
candidate that fires the Blast for nothing therefore pays the wait it causes later; one that fires it for the climb pays nothing
extra. The order is rebuilt when the set of visible gems changes; the first gem of the order is the controller's target.

Lobby-valid: visible gems and the observed meter only. Nothing map-specific: legs come from the walking grid and the mesh.
"""
import math
import numpy as np

from nav.marble_dynamics import spawn_order as SO
from nav.routing import goal_key

_PHYS = None


def _phys():
    """(needed_meter, JUMP_RISE_U, BLAST_CHARGE_S), imported late (crossings / climbs import search)."""
    global _PHYS
    if _PHYS is None:
        from nav.marble_dynamics.crossings import needed_meter, JUMP_RISE_U
        from nav.marble_dynamics.climbs import BLAST_CHARGE_S
        _PHYS = (needed_meter, JUMP_RISE_U, BLAST_CHARGE_S)
    return _PHYS

SV_ON = True              # the spawn plan prices every candidate (search.assess) and names the target (ContinuousController)
SV_W_U_PER_S = 8.         # cost u per model second (the old end cost was a walking distance; the marble travels ~8 u/s)
SV_MAX_GEMS = 10          # the dynamic programme orders this many nearest gems (35 ms); the rest follow greedily
SV_PICK_CREDIT = 100.     # per gem the forecast collects (the exact forecast beats the model)
SV_EVAL_REPLIES = 0       # the end state is read at this reply (0 = the candidate's own end: sv1 read every plan at reply 24 and a fast
                          # plan's pickup at reply 30 earned nothing; Bowl / KOTM / Cube lost 10-28 with slow 'accelerate_then_brake' plans)
SV_TARGET = False         # the order's first gem is the controller's target: OFF (KOTM seed 2: cost only 157, control 155, with the
                          # retarget 134-138; the tour + physics order choice keep the target, the spawn value prices every plan)
SV_COST = True            # the spawn value prices the candidates (ablation switch)
SV_AT_PICKUP = True       # sv4: value a collecting plan at its last pickup, not at its open-loop end
SV_APPROACH_MAX = 18.     # the first leg keeps the marble's speed up to this (sv2: 12; a launch at 20 u/s covers its leg at 20)
SV_ACCEL = 8.             # u/s2: the roll gains speed at this rate up to the leg speed (measured 11 from rest, ~5 above 9 u/s)
SV_TURN_KEEP_DEG = 60.    # a turn sharper than this at a gem loses speed (cos of the angle, at least SV_TURN_KEEP_MIN of it)
SV_TURN_KEEP_MIN = .3
SV_SWITCH_S = 1.0         # a rebuilt order (same gems) replaces the old one only when quicker from here by this
SV_CRUISE_U_S = 10.       # long legs run up to this ...
SV_SPEED_GAIN = .6        # ... gaining this much speed per u of leg length beyond SV_SHORT_U over the in-spawn speed
SV_SHORT_U = 3.
SV_CLIMB_S = 1.0          # a Blast climb itself (spawn_order.SO_BIG_CLIMB_S)
SV_NO_BLAST_ALT_S = 8.    # a climb without the meter waits at most this long (another way up, an edge hit, a detour)
SV_REPLAN_U = 8.          # the order is also rebuilt when the marble is this far from where it was planned
SV_OFF_GRID_FACTOR = 1.5  # a leg the walking grid has no distance for: this times the straight line ...
SV_OFF_GRID_SLACK_U = 2.  # ... plus this
SV_STOP_U_S2 = 0.         # OFF 2026-11-05 01:00: both stop models LOST Sprawl ~10 a round on 3-4 of 4 seeds (stop: angle rule only,
                          # +1.2 overall = noise; stop2: braking physics on every leg, -2.0 overall, Cragmire -12): any price on a fast
                          # arrival costs more Super Speed launches than the overshoots cost. 9 = the tested value.
                          # 2026-11-04 (operator: the Sprawl overshoots): a leg entered faster than SV_STOP_MIN_U_S with the gem more
                          # than SV_STOP_DEG behind the heading is priced as a stop (this deceleration, u/s2: 18 u/s stopped in 21 u,
                          # 16 in 10 u on Sprawl) and a fresh leg from the stop point; 0 = off (the old cos-of-the-angle speed keep)
SV_STOP_MIN_U_S = 12.
SV_OVER_U_S = 3.           # stop2 (2026-11-05): a leg entered more than this above its leg speed must brake (else the old model)
SV_STOP_DEG = 90.


def leg_speed(L):
    return min(SV_CRUISE_U_S, SO.SO_SPEED + SV_SPEED_GAIN * max(0., L - SV_SHORT_U))


def leg_time(L, v0, v):
    """(seconds, exit speed) over L u entered at v0 u/s, accelerating at SV_ACCEL up to v (a faster entry keeps its speed)."""
    if L <= 1e-6: return 0., v0
    if v0 >= v: return L / v0, v0
    d_acc = (v * v - v0 * v0) / (2. * SV_ACCEL)
    if d_acc >= L:
        v_end = math.sqrt(v0 * v0 + 2. * SV_ACCEL * L); return (v_end - v0) / SV_ACCEL, v_end
    return (v - v0) / SV_ACCEL + (L - d_acc) / v, v


class SpawnPlan:
    """The visible gems in the order best_order found, with a per-pair leg table."""

    def __init__(self, terrain, gems, items, pos, vel, previous=None):
        self.terrain = terrain; self.gems = [tuple(map(float, g[:3])) + tuple(g[3:]) for g in gems]
        self.keys = frozenset(goal_key(g) for g in self.gems); self.items = items; self.origin = np.asarray(pos[:3], dtype=float)
        self._routes = {}; self._legs = {}; self.kept = False
        self.order = self.best_order(pos, vel) if len(self.gems) > 1 else list(self.gems)
        self.keys_order = [goal_key(g) for g in self.order]
        if previous is not None and previous.keys == self.keys and len(previous.order) == len(self.order):
            # the marble moved (same gems): the old order stays unless the new one is SV_SWITCH_S quicker from here
            t_new = self.rest_time(pos, vel, 1.)
            new_order, new_keys = self.order, self.keys_order
            self.order, self.keys_order = list(previous.order), list(previous.keys_order)
            if self.rest_time(pos, vel, 1.) > t_new + SV_SWITCH_S: self.order, self.keys_order = new_order, new_keys
            else: self.kept = True

    def best_order(self, pos, vel):
        """The order with the least model time by the SAME model rest_time uses (dynamic programme over the SV_MAX_GEMS nearest
        gems carrying the previous gem for the turn and the entry speed; the rest follow nearest-next)."""
        gems = list(self.gems); p0 = np.asarray(pos[:3], dtype=float); sp = math.hypot(float(vel[0]), float(vel[1]))
        gems.sort(key=lambda g: math.hypot(g[0] - p0[0], g[1] - p0[1]))
        core, rest = gems[:SV_MAX_GEMS], gems[SV_MAX_GEMS:]
        n = len(core); P = np.array([g[:2] for g in core], dtype=float); Z = np.array([g[2] for g in core], dtype=float)
        Lw = np.zeros((n, n)); exit_v = np.zeros((n, n))   # leg length a -> b (walking) and the cruise speed of that leg
        for a_ in range(n):
            for b in range(n):
                if a_ != b: Lw[a_, b] = self.pair_leg(goal_key(core[a_]), core[a_], core[b]); exit_v[a_, b] = leg_speed(Lw[a_, b])
        first_L = np.array([self.leg(p0, core[j]) for j in range(n)])
        hd0 = (float(vel[0]), float(vel[1])) if sp >= SO.SO_TURN_MIN_SPEED else None
        def climb_s(z0, z1):
            rise = float(z1) - float(z0)
            if rise >= _phys()[1]: return SV_CLIMB_S + .5   # a Blast climb (the meter wait is priced in rest_time, not here)
            return SO.SO_CLIMB_S if rise >= SO.SO_STEP_U else 0.
        # first[j]: from the marble; T[p, a, c]: at gem a coming from p (n = the marble), the turn and the leg a -> c
        first = np.zeros(n); v_first = np.zeros(n)
        for j in range(n):
            ang = SO._angle(hd0, P[j] - p0[:2]) if hd0 is not None else 0.
            v0 = min(SV_APPROACH_MAX, sp) * (max(SV_TURN_KEEP_MIN, math.cos(math.radians(ang))) if ang > SV_TURN_KEEP_DEG else 1.)
            dt, v_end = leg_time(first_L[j], v0, leg_speed(first_L[j]))
            first[j] = dt + (SO.SO_TURN_S_PER_90 * ang / 90. if hd0 is not None else 0.) + climb_s(p0[2], Z[j]); v_first[j] = v_end
        T = np.zeros((n + 1, n, n))
        for p_ in range(n + 1):
            for a_ in range(n):
                if p_ == a_: continue
                head = P[a_] - (P[p_] if p_ < n else p0[:2]); v_in = exit_v[p_, a_] if p_ < n else v_first[a_]
                for c in range(n):
                    if c == a_: continue
                    ang = SO._angle(head, P[c] - P[a_])
                    v0 = v_in * (max(SV_TURN_KEEP_MIN, math.cos(math.radians(ang))) if ang > SV_TURN_KEEP_DEG else 1.)
                    dt, _ = leg_time(Lw[a_, c], v0, exit_v[a_, c])
                    T[p_, a_, c] = dt + SO.SO_TURN_S_PER_90 * ang / 90. + climb_s(Z[a_], Z[c])
        INF = 1e18; full = (1 << n) - 1
        dp = np.full((1 << n, n, n + 1), INF); bp = np.full((1 << n, n, n + 1), -1, dtype=np.int16)
        for j in range(n): dp[1 << j, j, n] = first[j]
        for mask in range(1, full + 1):
            for last in range(n):
                if not mask >> last & 1: continue
                row = dp[mask, last]
                if row.min() >= INF: continue
                tot = row[:, None] + T[:, last, :]
                pmin = tot.argmin(axis=0); cand = tot[pmin, np.arange(n)]
                for nx in range(n):
                    if mask >> nx & 1: continue
                    m2 = mask | 1 << nx
                    if cand[nx] < dp[m2, nx, last]: dp[m2, nx, last] = cand[nx]; bp[m2, nx, last] = pmin[nx]
        flat = dp[full].reshape(-1); k = int(flat.argmin()); last, prev = divmod(k, n + 1)
        order = []; mask = full
        while True:
            order.append(last)
            if prev == n: break
            pp = int(bp[mask, last, prev]); mask ^= 1 << last; last, prev = prev, pp
        out = [core[i] for i in order[::-1]]
        while rest:
            q = out[-1]; j = min(range(len(rest)), key=lambda i: math.hypot(rest[i][0] - q[0], rest[i][1] - q[1])); out.append(rest.pop(j))
        return out

    def route(self, g):
        from nav.marble_dynamics.search import cached_route
        k = goal_key(g); r = self._routes.get(k)
        if r is None:
            r = self._routes[k] = cached_route(self.terrain, np.asarray(g[:3], dtype=float))
        return r

    def dist(self, a, g):
        """Walking-grid distance from point a to gem g (inf off the grid)."""
        try: return float(self.route(g).distance(np.asarray(a[:3], dtype=float)))
        except Exception: return math.inf

    def leg(self, a, g):
        """Model distance of the leg point a -> gem g."""
        straight = math.hypot(float(g[0]) - float(a[0]), float(g[1]) - float(a[1]))
        return SO._leg(self.dist(a, g), straight, float(g[2]) - float(a[2]))

    def pair_leg(self, ka, a, g):
        k = (ka, goal_key(g)); L = self._legs.get(k)
        if L is None: L = self._legs[k] = self.leg(a, g)
        return L

    def rest_time(self, pos, vel, meter, t0=0., collected=()):
        """Model seconds to collect the order's remaining gems from (pos, vel) at time t0 with the Blast meter `meter`."""
        rem = [g for g, k in zip(self.order, self.keys_order) if k not in collected]
        if not rem: return 0.
        needed_meter, JUMP_RISE_U, BLAST_CHARGE_S = _phys()
        sp = math.hypot(float(vel[0]), float(vel[1])); t = 0.; m = float(meter)
        px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
        hd = (float(vel[0]), float(vel[1])) if sp >= SO.SO_TURN_MIN_SPEED else None
        prev_key = None
        for i, g in enumerate(rem):
            d = (g[0] - px, g[1] - py); L = math.hypot(*d)
            Lw = self.leg((px, py, pz), g) if prev_key is None else self.pair_leg(prev_key, (px, py, pz), g)
            ang = SO._angle(hd, d) if hd is not None else 0.
            if i == 0: v0 = min(SV_APPROACH_MAX, sp)
            v_in = sp if i == 0 else v0; v_leg = leg_speed(Lw)
            if SV_STOP_U_S2 and hd is not None and ang > SV_STOP_DEG and v_in > SV_STOP_MIN_U_S:
                # 2026-11-04: too fast to turn back: the marble brakes to a stop along its heading, then rolls to the gem from
                # there (the overshoot the Sprawl cases showed: 10-20 u past the spawn, 2-5 s lost)
                hn = math.hypot(*hd); ux, uy = hd[0] / hn, hd[1] / hn
                stop_d = v_in * v_in / (2. * SV_STOP_U_S2); dt_stop = v_in / SV_STOP_U_S2
                sx, sy = px + ux * stop_d, py + uy * stop_d
                Lw = self.leg((sx, sy, pz), g) if prev_key is None else max(self.pair_leg(prev_key, (px, py, pz), g), math.hypot(g[0] - sx, g[1] - sy))
                dt, v0 = leg_time(Lw, 0., leg_speed(Lw)); dt += dt_stop + SO.SO_TURN_S_PER_90 * ang / 90.
            elif SV_STOP_U_S2 and v_in > v_leg + SV_OVER_U_S:
                # 2026-11-05 stop2: entered faster than the leg allows: brake to the leg speed; when the gem comes before the
                # marble is slow enough it is passed at speed, the marble stops and rolls back (stop: a fast entry was assumed
                # to keep its 18 u/s through a 2 u leg and every leg after it)
                if ang > SV_TURN_KEEP_DEG: v_in *= max(SV_TURN_KEEP_MIN, math.cos(math.radians(ang)))
                d_brake = max(0., v_in * v_in - v_leg * v_leg) / (2. * SV_STOP_U_S2)
                if d_brake <= Lw:
                    dt = (v_in - v_leg) / SV_STOP_U_S2 + (Lw - d_brake) / v_leg; v0 = v_leg
                else:
                    stop_d = v_in * v_in / (2. * SV_STOP_U_S2); back = max(0., stop_d - Lw)
                    dt_back, v0 = leg_time(back, 0., leg_speed(back))
                    dt = v_in / SV_STOP_U_S2 + dt_back + SO.SO_TURN_S_PER_90 * 2.
                if hd is not None: dt += SO.SO_TURN_S_PER_90 * ang / 90.
            else:
                if ang > SV_TURN_KEEP_DEG: v0 *= max(SV_TURN_KEEP_MIN, math.cos(math.radians(ang)))
                dt, v0 = leg_time(Lw, v0, leg_speed(Lw))
                if hd is not None: dt += SO.SO_TURN_S_PER_90 * ang / 90.
            rise = float(g[2]) - pz
            if rise >= JUMP_RISE_U:
                need = needed_meter(rise); m_at = min(1., m + (t + dt) / BLAST_CHARGE_S)
                wait = min(SV_NO_BLAST_ALT_S, (need - m_at) * BLAST_CHARGE_S) if m_at < need else 0.
                dt += SV_CLIMB_S + wait; m = max(0., m_at - need) - (t + dt) / BLAST_CHARGE_S   # spent: the refill below restarts from it
                v0 = 2.   # the climb lands the marble: little speed carried
            elif rise >= SO.SO_STEP_U: dt += SO.SO_CLIMB_S
            t += dt; hd = d if L > 1e-6 else hd; px, py, pz = g[0], g[1], g[2]; prev_key = goal_key(g)
        return t


def update_plan(engine, terrain, dynamics, raw, state, skipped=()):
    """The engine's spawn plan for this decision: rebuilt when the visible gem set changes or the marble has moved SV_REPLAN_U."""
    from nav.marble_dynamics.global_goals import observed_gems
    gems = [g for g in observed_gems(dynamics, raw) if goal_key(g) not in skipped]
    if not gems:
        engine.spawn_plan = None; return None
    pos = np.asarray(state['position'], dtype=float); vel = np.asarray(state['velocity'], dtype=float)
    keys = frozenset(goal_key(g) for g in gems)
    plan = getattr(engine, 'spawn_plan', None)
    if plan is None or plan.keys != keys or float(np.linalg.norm(plan.origin[:2] - pos[:2])) > SV_REPLAN_U:
        items = {}
        for it in dynamics.get('items', []):
            if it.get('gem'): items[goal_key(it['position'])] = it
        plan = engine.spawn_plan = SpawnPlan(terrain, gems, items, pos, vel, previous=plan if plan is not None and plan.keys == keys else None)
        engine.spawn_plans_built = getattr(engine, 'spawn_plans_built', 0) + 1
    return plan


def candidate_value(engine, dynamics, rollout, meter_now, fired):
    """(cost, detail) of a candidate under the spawn plan, or None when there is no plan."""
    plan = getattr(engine, 'spawn_plan', None)
    if plan is None or not plan.order: return None
    from nav.marble_dynamics.gem_demo import possible_pickup
    states = rollout['states']; native = rollout.get('native_states')
    j = min(len(states) - 1, SV_EVAL_REPLIES) if SV_EVAL_REPLIES else len(states) - 1
    picks = {}
    for k in plan.keys_order:
        it = plan.items.get(k)
        if it is None: continue
        r = possible_pickup(states, it, native)
        if r is not None and r <= j: picks[k] = int(r)
    if picks and SV_AT_PICKUP:
        # sv4: a collecting plan is valued at its LAST pickup (the planner re-searches there; what the open-loop remainder does
        # after the gem never runs). KOTM audit 09:40: a fast plan picking at reply 14 lost to a jump picking at reply 30 because
        # its remainder braked and ended farther from the next gem.
        j = min(len(states) - 1, max(picks.values()))
    s = states[j]; t_end = j * .064
    m = 0. if fired else float(meter_now)
    m = min(1., m + t_end / _phys()[2])
    rest = plan.rest_time(s['position'], s['velocity'], m, 0., picks)
    cost = SV_W_U_PER_S * (t_end + rest) - SV_PICK_CREDIT * len(picks)
    return float(cost), dict(t_end=round(t_end, 3), rest_s=round(rest, 2), picks=len(picks), meter=round(m, 2))
