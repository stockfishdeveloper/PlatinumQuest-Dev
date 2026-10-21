"""Pad hops (2026-10-21, operator: cluster collection on raised platforms).

A gem that sits on a small raised pad (TreasureBox: 122 of 153 gems, pad top 0.5 u above the floor, pad radius
0.5-1.25 u) can never be rolled onto: a 0.5 u step is taller than the marble's radius. It is always a jump, and the
only question is where the jump starts. The operator's demo (demo_20261021_151105.npz, 207 points): approach at
~3.8 u/s, jump with the gem 2.2 u ahead (1.5-2.9), come down ONTO the gem 0.62 s later. That is the descending half
of the jump arc (jumpImpulse 7.5 u/s, gravity 20): the marble is back down at pad height at 0.60-0.68 s, so the launch
point is speed x that time, and at the pad edge the marble is already ~1 u up, clear of the pad wall (the bump and
the spin-reset wait the planner showed: 39 of 162 pad pickups launched under 1.5 u out, 77 over 3 u, 18 pickups a
round needing three or more jumps against the human's 8).

Nothing map-specific: the pad is read from the collision mesh under and around the target gem, the timing from the
engine's jump constants; the compiled forecast verifies the pickup by tick as for every plan.
"""
import math
import numpy as np
from nav.marble_dynamics.search import SequenceProposal as _SequenceProposal

JUMP_IMPULSE = 7.5       # u/s, the marble datablock's jumpImpulse (edge_hits.JUMP_IMPULSE)
GRAVITY = 20.
PAD_MIN_U = .3           # a floor under the gem at least this far above the ring around it is a pad ...
PAD_MAX_U = 1.4          # ... and at most this (the jump's apex is 1.41 u; higher needs a Blast or an edge hit)
PAD_RING_U = (1.25, 1.75, 2.25)   # ring radii probed around the gem for the surrounding floor
PAD_EDGE_STEP_U = .25    # the pad radius is found by walking outward until the floor drops
PAD_EDGE_MAX_U = 2.5
PAD_LOWER_FRACTION = .3  # share of ring probes that must read lower (pads in a cluster have neighbours inside the ring: 14 of 153 demo gems)
PAD_ABOVE_U = .3         # aim the descending arc at pad top + this (the pickup box reaches ~0.4 u above the marble)
HOP_SPEEDS = (3.5, 5., 6.5)   # approach speeds offered (the human: 3.4-4.1 u/s at pad gems)
HOP_MAX_U = 12.          # pad hops are offered when the gem is within this horizontal distance
HOP_MIN_BELOW_U = .15    # ... and the marble centre is at least this far below the gem's floor (else a plain roll)
BLAST_GATE_RISE_U = 1.5  # no Blast candidates for a pad gem less than this above the marble ...
BLAST_GATE_NEAR_U = 8.   # ... within this horizontal distance (a jump reaches it; 32 of 52 Blasts gained under 1 u)


def hop_time(height):
    """Time after a jump at which the rising-then-falling marble is back DOWN at `height` above its launch centre
    (the descending branch). None when the jump cannot reach that height."""
    disc = JUMP_IMPULSE * JUMP_IMPULSE - 2. * GRAVITY * height
    if disc < 0.:
        return None
    return (JUMP_IMPULSE + math.sqrt(disc)) / GRAVITY


def launch_distance(speed, pad_height):
    """Horizontal distance from the gem at which to jump so the marble comes down onto the gem (pad top + PAD_ABOVE_U)."""
    t = hop_time(pad_height + PAD_ABOVE_U)
    if t is None:
        t = JUMP_IMPULSE / GRAVITY   # the apex: as high as it goes
    return float(speed) * t


def _floor_z(mesh, x, y, z_from, depth=4.):
    hit = mesh.raycast(np.array([x, y, z_from], dtype=float), np.array([x, y, z_from - depth], dtype=float), cull_backfaces=True)
    if hit is None or float(hit['normal'][2]) < .7:
        return None
    return float(hit['point'][2])


def pad_under(engine, gem):
    """(pad_height, pad_radius) when the gem sits on a raised pad, else None. Cached per gem position on the engine."""
    cache = getattr(engine, '_pad_cache', None)
    if cache is None:
        cache = engine._pad_cache = {}
    key = tuple(np.round(np.asarray(gem[:3], dtype=float), 2))
    if key in cache:
        return cache[key]
    out = _pad_under(engine.mesh, np.asarray(gem[:3], dtype=float))
    cache[key] = out
    return out


def _pad_under(mesh, g):
    top = _floor_z(mesh, g[0], g[1], g[2] + .5, depth=3.)
    if top is None:
        return None
    ring = []
    for r in PAD_RING_U:
        for a in np.linspace(0., 2. * math.pi, 8, endpoint=False):
            z = _floor_z(mesh, g[0] + r * math.cos(a), g[1] + r * math.sin(a), top + .5, depth=top + .5 - (top - 3.))
            if z is not None:
                ring.append(z)
    if len(ring) < 8:
        return None
    lower = [z for z in ring if z < top - PAD_MIN_U]
    if len(lower) < len(ring) * PAD_LOWER_FRACTION:
        return None   # the gem's floor is the surrounding floor (or a ramp): not a pad
    height = top - float(np.median(lower))
    if not (PAD_MIN_U <= height <= PAD_MAX_U):
        return None
    radius = PAD_EDGE_MAX_U
    for r in np.arange(PAD_EDGE_STEP_U, PAD_EDGE_MAX_U + 1e-9, PAD_EDGE_STEP_U):
        zs = [_floor_z(mesh, g[0] + r * math.cos(a), g[1] + r * math.sin(a), top + .5, depth=3.5) for a in np.linspace(0., 2. * math.pi, 8, endpoint=False)]
        zs = [z for z in zs if z is not None]
        if zs and min(zs) < top - PAD_MIN_U:
            radius = float(r)
            break
    return float(height), radius


def hop_specs(engine, route, gem, steps, through, state):
    """Pad-hop candidates for a pad gem ahead: [(label, proposal), ...] or []."""
    pad = pad_under(engine, gem)
    if pad is None:
        return []
    p = np.asarray(state['position'], dtype=float); g = np.asarray(gem[:3], dtype=float)
    if float(np.linalg.norm(g[:2] - p[:2])) > HOP_MAX_U:
        return []
    height, radius = pad
    if float(g[2] - p[2]) < HOP_MIN_BELOW_U:
        return []   # already up there (or level): no hop
    out = []
    for v in HOP_SPEEDS:
        out.append(('pad_hop_%g' % v, PadHopProposal(route, g, steps, through, v, height, radius)))
    return out


def blast_gated(engine, gem, state):
    """True when Blast candidates make no sense: the target is a low pad gem close by (a jump reaches it)."""
    p = np.asarray(state['position'], dtype=float); g = np.asarray(gem[:3], dtype=float)
    if float(g[2] - p[2]) >= BLAST_GATE_RISE_U or float(np.linalg.norm(g[:2] - p[:2])) > BLAST_GATE_NEAR_U:
        return False
    return pad_under(engine, gem) is not None


class PadHopProposal(_SequenceProposal):
    """Approach the pad gem at a steady slow speed, jump when the gem is launch_distance(speed, pad) ahead, and carry
    on toward it (the through context turns to the next gem after the pickup, as for every other plan)."""
    def __init__(self, route, gem, steps, through, speed, pad_height, pad_radius):
        super().__init__(route, np.asarray(gem[:3], dtype=float), steps, jump=-1, direct=True, through=through, fast=None)
        self.hop_speed = float(speed); self.pad_height = float(pad_height); self.pad_radius = float(pad_radius)
        self.trigger_u = launch_distance(self.hop_speed, self.pad_height); self.jumped = False; self.use = -1
        self.min_speed = self.hop_speed   # SequenceProposal: floor on the target speed (no braking into the gem)

    def __call__(self, state, index):
        if index == 0:
            self.jumped = False
        p = np.asarray(state['position'], dtype=float); v = np.asarray(state['velocity'], dtype=float)
        action = super().__call__(state, index)
        if not self.jumped and not getattr(self, 'passed', False):
            d = float(np.linalg.norm(self.gem[:2] - p[:2]))
            # jump when the gem is one hop ahead at the speed actually carried (never from the air: vz ~ 0 on the floor)
            trig = launch_distance(max(float(np.hypot(v[0], v[1])), 1.), self.pad_height)
            if d <= max(trig, self.pad_radius + .3) and abs(float(v[2])) < 1.5 and float(p[2]) < float(self.gem[2]) - HOP_MIN_BELOW_U:
                action = list(action); action[4] = 1; self.jumped = True
        if not self.jumped:
            # steady approach: cap the throttle-driven target speed at the hop speed (SequenceProposal's Proposal
            # already applied min_speed = hop speed; the top is applied here by scaling the stick when faster)
            sp = float(np.hypot(v[0], v[1]))
            if sp > self.hop_speed + .5:
                action = list(action); action[0] = 0.; action[1] = 0.; action[2] = 0.; action[3] = 0.   # coast down
        return action
