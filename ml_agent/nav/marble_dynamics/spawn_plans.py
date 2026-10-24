"""Spawn plans (2026-10-21, operator: "collecting a cluster needs PLANNING for how to collect the entire spawn ...
plan ahead MULTIPLE gems of the spawn so you finish the 2-3 second plan in a good spot to continue the rest").

A spawn = the gems near the target (SPAWN_RADIUS_U). A chain plan drives through an ORDER of up to CHAIN_MAX of them
without stopping, at a steady cluster speed, jumping for a gem on a raised pad (pad_hops) either so that it comes down
onto the gem ('descend', the human's single-gem hop) or just before it so the arc carries on over the gems behind it
('arc': a row of 3-4 gems in one jump, the operator's row technique). Orders offered: nearest-first from the target,
heading-aligned, a sweep along the direction of travel, TOP-DOWN (tallest pads first, then roll down through the lower
gems: fewer jumps than bottom-up), and detected ROWS (3+ collinear gems, approached from the near end).

The compiled/physics forecast runs each chain for the full horizon; chain_score counts EVERY gem the forecast collects
(the game's own per-tick pickup test), when, and how far the end state is from the gems still left, so the plan that
finishes the spawn fastest and ends pointed at the rest wins (candidate_cost scores a plan by its one target gem).
Nothing map-specific: gems and pads come from the game state and the mesh, the technique from the forecast.
"""
import math
import numpy as np
from nav.marble_dynamics import gem_demo as _gem_demo
from nav.marble_dynamics.gem_demo import Proposal, possible_pickup
from nav.marble_dynamics.pad_hops import pad_under, launch_distance, hop_time, JUMP_IMPULSE, GRAVITY, PAD_ABOVE_U

SPAWN_RADIUS_U = 9.       # gems this close (horizontally) to the target belong to its spawn
CHAIN_MAX = 4             # gems per chain plan (a 48-reply forecast covers 3 s: 3-4 gems at cluster speed)
CHAIN_SPEEDS = (5.5, 8.5)  # v4: steady speeds through the spawn (4.5 / 7 slowed the approaches: travel 8.3 vs 10.3 u/s in spawn3)
ARC_LEAD_S = .15          # 'arc' mode: jump this long before reaching the first gem (rising through it, flying on)
PICKUP_CREDIT = 100.      # per collected gem, the same credit candidate_cost gives its one gem
PICKUP_TIME_W = 4.        # u of cost per second until each pickup (gem_demo.PICKUP_TIME_WEIGHT)
ROW_LATERAL_U = .7        # a gem within this of the line through two others is on their row ...
ROW_SPACING_U = 3.5       # ... when consecutive row gems are at most this far apart
HOP_BELOW_U = .25         # a pad gem needs a hop when the marble centre is this far below the gem's own height minus its float
GEM_FLOAT_U = 0.          # a gem sits AT the height of the floor it is on (TreasureBox probe 22:40: pad top 5.0, gem z 5.0); was .3, which hid every 0.5 u pad from the hop test
HOP_SPEED_TOL_U_S = .4    # v2: jump when the speed is within this of the one that fits the distance ...
HOP_DIST_TOL_U = .3       # ... and the distance within this of speed x descending time
HOP_MIN_SPEED_U_S = 2.5   # v2: never fit the speed below this (a crawl)
WALK_DETOUR_RATIO = 1.6   # v6: a spawn gem must be walkable within this ratio of its straight distance ...
WALK_DETOUR_SLACK_U = 2.  # ... plus this (same level, no wall or drop between)
MAX_CHAINS = 8            # cap on chain candidates a decision (12 still failed a pool round's wall limit; Python forecasts) (probe: 30 specs a search, 95 percent collecting nothing; live round 15:37: holds 19 s at 18)
CHAIN_NEAR_U = 10.        # chains only when the marble is within this of the nearest spawn gem (travel is the tour's job)
CHAIN_INSIDE_U = 4.5      # v4: ... and INSIDE the spawn (within this of a spawn gem) or already slow (below CHAIN_SLOW_U_S): never on the approach
CHAIN_SLOW_U_S = 9.


def spawn_items(dynamics, target):
    t = np.asarray(target[:3], dtype=float)
    out = []
    for it in dynamics.get('items', []) or []:
        if not it.get('gem'):
            continue
        p = np.asarray(it['position'][:3], dtype=float)
        if float(np.linalg.norm(p[:2] - t[:2])) <= SPAWN_RADIUS_U:
            out.append(it)
    return out


def _pos(it):
    return np.asarray(it['position'][:3], dtype=float)


def _nn_chain(start_pos, items, first=None, k=CHAIN_MAX, turn_weight=0.):
    """Greedy nearest-neighbour order from start_pos; `first` forces the first gem; turn_weight adds u per radian of
    turn between legs (straight lines preferred)."""
    left = list(items); order = []
    here = np.asarray(start_pos[:2], dtype=float); heading = None
    if first is not None:
        order.append(first); left = [i for i in left if i['id'] != first['id']]
        heading = _pos(first)[:2] - here; here = _pos(first)[:2]
    while left and len(order) < k:
        def cost(i):
            d = _pos(i)[:2] - here; L = float(np.linalg.norm(d))
            if heading is None or turn_weight == 0. or L < 1e-6 or float(np.linalg.norm(heading)) < 1e-6:
                return L
            c = float(np.clip(np.dot(d, heading) / (L * float(np.linalg.norm(heading))), -1., 1.))
            return L + turn_weight * math.acos(c)
        nxt = min(left, key=cost)
        order.append(nxt); left = [i for i in left if i['id'] != nxt['id']]
        heading = _pos(nxt)[:2] - here; here = _pos(nxt)[:2]
    return order


def rows(items):
    """Rows of 3+ nearly collinear gems with consecutive spacing <= ROW_SPACING_U, as lists ordered along the row."""
    found = {}
    pts = [(_pos(i), i) for i in items]
    for a in range(len(pts)):
        for b in range(a + 1, len(pts)):
            pa, ia = pts[a]; pb, ib = pts[b]
            d = pb[:2] - pa[:2]; L = float(np.linalg.norm(d))
            if L < 1e-6 or L > ROW_SPACING_U:
                continue
            u = d / L
            members = []
            for p, it in pts:
                rel = p[:2] - pa[:2]; along = float(rel @ u); lateral = abs(float(rel[0] * u[1] - rel[1] * u[0]))
                if lateral <= ROW_LATERAL_U:
                    members.append((along, it))
            members.sort(key=lambda m: m[0])
            # keep the longest run around a-b with spacing <= ROW_SPACING_U
            runs = []; cur = [members[0]]
            for prev, nxt in zip(members, members[1:]):
                if nxt[0] - prev[0] <= ROW_SPACING_U:
                    cur.append(nxt)
                else:
                    runs.append(cur); cur = [nxt]
            runs.append(cur)
            for run in runs:
                ids = frozenset(m[1]['id'] for m in run)
                if len(run) >= 3 and ia['id'] in ids and ib['id'] in ids and ids not in found:
                    found[ids] = [m[1] for m in run]
    return list(found.values())


def orders(position, velocity, target_item, items, k=CHAIN_MAX):
    """Candidate gem orders for the spawn: [(kind, [items...]), ...] (deduplicated)."""
    p = np.asarray(position[:3], dtype=float); v = np.asarray(velocity[:2], dtype=float); sp = float(np.linalg.norm(v))
    out = []
    def add(kind, seq):
        seq = [i for i in seq if i is not None][:k]
        if not seq:
            return
        key = tuple(i['id'] for i in seq)
        if key in {tuple(i['id'] for i in s) for _, s in out}:
            return
        out.append((kind, seq))
    others = [i for i in items if target_item is None or i['id'] != target_item['id']]
    first = target_item if target_item is not None and any(i['id'] == target_item['id'] for i in items) else None
    add('nn', _nn_chain(p, items, first=first, k=k))
    add('straight', _nn_chain(p, items, first=first, k=k, turn_weight=1.5))
    if sp >= 2.:
        u = v / sp
        aligned = max(items, key=lambda i: float(((_pos(i)[:2] - p[:2]) / max(float(np.linalg.norm(_pos(i)[:2] - p[:2])), 1e-6)) @ u) - .05 * float(np.linalg.norm(_pos(i)[:2] - p[:2])))
        add('aligned', _nn_chain(p, items, first=aligned, k=k, turn_weight=1.5))
        sweep = sorted(items, key=lambda i: float((_pos(i)[:2] - p[:2]) @ u))
        add('sweep', sweep)
    # top-down: the tallest gems first (one climb, then roll down through the lower ones)
    top = sorted(items, key=lambda i: (-round(float(_pos(i)[2]), 1), float(np.linalg.norm(_pos(i)[:2] - p[:2]))))
    if len({round(float(_pos(i)[2]), 1) for i in items}) > 1:
        add('topdown', _nn_chain(_pos(top[0]), items, first=top[0], k=k, turn_weight=1.0))
    # rows, approached from the end nearest the marble, then the rest nearest-first
    for row in rows(items):
        if float(np.linalg.norm(_pos(row[-1])[:2] - p[:2])) < float(np.linalg.norm(_pos(row[0])[:2] - p[:2])):
            row = row[::-1]
        rest = [i for i in items if i['id'] not in {r['id'] for r in row}]
        tail = _nn_chain(_pos(row[-1]), rest, k=max(0, k - len(row))) if rest and len(row) < k else []
        add('row%d' % len(row), row[:k] + tail)
    return out


class ChainProposal:
    """Drive through `items` in order at a steady speed, hopping pad gems; after the last, aim at `after`.

    Picklable (module-level, no engine reference: pads are resolved at construction) so the real-time search workers
    can run it. The forecast calls observe_ticks after each reply's physics with the tick states; a collected gem
    advances the chain (any gem of the chain collected out of order is dropped from it too)."""
    def __init__(self, items, pads, steps, speed, after, mode='descend'):
        self.items = list(items); self.pads = list(pads); self.steps = int(steps); self.speed = float(speed)
        self.after = None if after is None else np.asarray(after[:3], dtype=float); self.mode = mode
        self.k = 0; self.jumped = set(); self.collected = set(); self.use = -1; self.jump = -1; self.blast = -1
        self.gem = _pos(self.items[0]) if self.items else self.after   # the search reads .gem on proposals
        self.min_speed = speed

    def _advance(self):
        while self.k < len(self.items) and self.items[self.k]['id'] in self.collected:
            self.k += 1

    def observe_ticks(self, states):
        for j in range(self.k, len(self.items)):
            it = self.items[j]
            if it['id'] not in self.collected and _gem_demo.tick_pickup(states, it) is not None:
                self.collected.add(it['id'])
        self._advance()

    def __call__(self, state, index):
        if index == 0:
            self.k = 0; self.jumped = set(); self.collected = set()
        p = np.asarray(state['position'], dtype=float); v = np.asarray(state['velocity'], dtype=float)
        # centre test as a fallback for the tick test (the forecast without tick states)
        while self.k < len(self.items) and float(np.linalg.norm(p - _pos(self.items[self.k]))) < .5:
            self.collected.add(self.items[self.k]['id']); self._advance()
        if self.k >= len(self.items):
            goal = self.after if self.after is not None else (_pos(self.items[-1]) if self.items else p)
            action = Proposal(goal, 0., speed_distance=None, min_speed=_gem_demo.PICKUP_MIN_SPEED if hasattr(_gem_demo, 'PICKUP_MIN_SPEED') else 6.)(state, 0)
            return action
        it = self.items[self.k]; g = _pos(it); pad = self.pads[self.k]
        d = float(np.linalg.norm(g[:2] - p[:2])); sp = float(np.hypot(v[0], v[1]))
        needs_hop = pad is not None and float(p[2]) < float(g[2]) + .05   # the marble centre on the pad would be g.z + 0.2
        target = self.speed; jump = False
        if needs_hop and it['id'] not in self.jumped:
            grounded = abs(float(v[2])) < 1.5
            if self.mode == 'arc':
                jump = grounded and d <= max(max(sp, 1.) * ARC_LEAD_S + .4, pad[1] + .3)
            else:
                # v2 (spawn1 audit: pad gems 1.29 jumps each, 23 needing 3+): the descending hop needs the launch
                # distance to MATCH the speed. Too fast for the distance (the usual case right after a pickup) ->
                # brake to the speed that fits, then jump; never jump early and fly past the gem.
                rise_here = max(float(g[2]) - (float(p[2]) - .2), .3)   # the pad top above the floor the marble is on now
                t_star = (hop_time(rise_here + PAD_ABOVE_U) or (JUMP_IMPULSE / GRAVITY)) * getattr(self, 'flight_factor', 1.)
                v_req = d / t_star
                if d <= pad[1] + .3:
                    jump = grounded                      # at the pad wall: the rising hop is all that is left
                elif sp <= v_req + HOP_SPEED_TOL_U_S and d <= sp * t_star + HOP_DIST_TOL_U:
                    jump = grounded                      # at the launch point with a matching speed
                elif d < self.speed * t_star:
                    target = max(HOP_MIN_SPEED_U_S, min(self.speed, v_req))   # too close for the plan speed: fit the speed to the distance
        action = Proposal(g, 0., speed_distance=1e3, min_speed=target, top=target)(state, 0)
        if jump:
            action = list(action); action[4] = 1; self.jumped.add(it['id'])
        return action


def chain_specs(engine, route, dynamics, gem, gem_item, steps, through):
    """Chain candidates for the target's spawn: [(label, ChainProposal), ...]."""
    items = spawn_items(dynamics, gem)
    if not items:
        return []
    st = dynamics['native']['state']
    p0 = np.asarray(st['position'][:2], dtype=float)
    if min(float(np.linalg.norm(_pos(it)[:2] - p0)) for it in items) > CHAIN_NEAR_U:
        return []
    nearest = min(float(np.linalg.norm(_pos(it)[:2] - p0)) for it in items)
    if nearest > CHAIN_INSIDE_U and float(np.hypot(*st['velocity'][:2])) >= CHAIN_SLOW_U_S:
        return []   # v4: approaching fast from outside: the travel families bring the marble in; chains take over inside
    # v6 (spawn4 Concentric 87 / 93 vs 131 / 112, 95-106 safe-hold decisions a round): straight-line neighbours can sit
    # on another ring behind a wall or a drop; keep only gems the marble can WALK to about as directly as it is near
    from nav.marble_dynamics.search import cached_route
    kept = []
    for it in items:
        gp = _pos(it); straight = float(np.linalg.norm(gp[:2] - p0))
        try:
            walk = float(cached_route(route.terrain, gp).distance(np.asarray(st['position'], dtype=float)))
        except Exception:
            walk = float('inf')
        if math.isfinite(walk) and walk <= WALK_DETOUR_RATIO * max(straight, 1.) + WALK_DETOUR_SLACK_U:
            kept.append(it)
    items = kept
    if not items:
        return []
    pads = {it['id']: pad_under(engine, _pos(it)) for it in items}
    after = through[1] if through is not None else None
    out = []
    target_item = gem_item if gem_item is not None and gem_item.get('gem') else None
    for kind, seq in orders(st['position'], st['velocity'], target_item, items):
        seq_pads = [pads[i['id']] for i in seq]
        any_pad = any(p is not None for p in seq_pads)
        rest = [i for i in items if i['id'] not in {s['id'] for s in seq}]
        tail = after
        if tail is None and rest:
            tail = min(rest, key=lambda i: float(np.linalg.norm(_pos(i)[:2] - _pos(seq[-1])[:2])))['position']
        for v in CHAIN_SPEEDS:
            for mode in ((('descend', 'arc') if kind.startswith('row') else ('descend',)) if any_pad else ('roll',)):
                cp = ChainProposal(seq, seq_pads, steps, v, tail, mode); cp.target_gem = np.asarray(gem[:3], dtype=float)
                out.append(('chain_%s_%d_v%g_%s' % (kind, len(seq), v, mode), cp))
                if len(out) >= MAX_CHAINS:
                    return out
    return out


def chain_score(proposal, label, info, rollout, dynamics):
    """Rescore a ChainProposal in place: assess's own cost (the target gem, its through credit, the end speed: the same
    scale as every other candidate) plus PICKUP_CREDIT (less the pickup-time charge) for every OTHER chain gem the
    forecast collects. v3 (2026-10-21 17:40): the first version replaced the cost wholesale and left out the through
    credit, so a chain lost every single-gem tie to the old jump plans (spawn2: chain hops pressed 16 times in two rounds)."""
    if not isinstance(proposal, ChainProposal) or info.get('rejected') is not None:
        return
    states = rollout['states']; native = rollout.get('native_states')
    target = getattr(proposal, 'target_gem', None)
    picks = []; extra = []
    for it in proposal.items:
        r = possible_pickup(states, it, native)
        if r is None:
            continue
        picks.append(int(r))
        if target is None or float(np.linalg.norm(_pos(it) - target)) > .1:
            extra.append(int(r))
    info['chain_items'] = len(proposal.items); info['chain_pickups'] = len(picks)
    info['chain_order'] = [[float(x) for x in _pos(it)] for it in proposal.items]   # v5: the controller follows this order
    if not picks:
        return
    if not math.isfinite(float(info.get('cost', math.inf))):
        return
    bonus = sum(-PICKUP_CREDIT + PICKUP_TIME_W * .064 * r for r in extra)
    info['cost'] = float(info['cost'] + bonus); info['chain_bonus'] = float(bonus)
    first = min(picks)
    if info.get('possible_pickup_interval') is None or first < info['possible_pickup_interval']:
        info['possible_pickup_interval'] = first
