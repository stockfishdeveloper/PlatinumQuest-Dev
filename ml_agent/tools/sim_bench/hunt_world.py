"""Hunt rules for the offline simulator: gem group spawning and out-of-bounds respawns (2026-10-29, phase 1 of
docs/IMPROVEMENT_LOOP_PLAN_2026-10-29.md).

A port of the SOLO branch of platinum/server/scripts/huntGems.cs (spawnHuntGemGroup -> spawnHuntGemsInGroup ->
getCenterGems + spawnGemGroupSet) and of the quick-respawn rule (game.cs respawnPlayer, checkpoint.cs getCheckpointPos,
mp/gameConnection.cs getFurthestSpawnTrigger). The ten benchmark missions have no GemGroups, run game "Ultra" or
"Platinum" (so testSpawn's PlatinumQuest spawn chance never applies) and set only maxGemsPerSpawn / radiusFromGem; the
other parameters are the $Hunt:: defaults. Not ported: the multiplayer branch (block around the leader, spawn ramp, double
spawn groups, competitive left-behinds). The sampler draws from its own numpy Generator: the game's LCG is not reproduced,
only its rules, so a sampled future is a legal future, never the recorded one (lobby rule: the policy may average over
legal futures and never sees the hidden RNG).

Gem values: 1 + huntExtraValue (gems.cs): red 1, yellow 2, blue 5, platinum 10 (green 4, orange 3, purple 6, turquoise 7,
black -1, pink 1).

Respawn (measured on recorded rounds, 10-29): the decision after the one that observed the fall shows the marble at rest at
the trigger's position plus its `add` field (default 0 0 3), camera yaw reset, held powerup cleared and scheduled to come
back after $powerupDelay (600 ms), the Blast meter untouched, the pending bridge action cleared. The trigger is the one
farthest from the position of the nearest highest-value visible gem (46 of 51 times) or from the marble's last contact
position (5 of 51), among a random prefix of the trigger list, never the trigger used within the last 4 s.
"""
import io
import json
import math
import os
import re
import sys

import numpy as np

ML = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ML not in sys.path:
    sys.path.insert(0, ML)

GEM_VALUE = {'red': 1, 'yellow': 2, 'blue': 5, 'platinum': 10, 'green': 4, 'orange': 3, 'purple': 6, 'turquoise': 7,
             'black': -1, 'pink': 1}
GEM_BOUNDS = np.asarray([-0.3, -0.25, -0.05, 0.3, 0.25, 0.45])   # item world box about the gem position (all ten maps, journals)
HUNT_DEFAULTS = dict(radius=15., max_gems=7, min_points=5, min_gems=3, max_loops=15)   # huntGems.cs $Hunt::*
POWERUP_DELAY_MS = 600          # game.cs $powerupDelay: the held powerup returns this long after a respawn
RESPAWN_ADD_DEFAULT = (0., 0., 3.)
CAMERA_PITCH_DEFAULT = 0.45
POW_RESPAWN_MS = 7000           # item.cs / powerups.cs


def _mission_path(map_name):
    from generate_terrain_map import resolve_mission
    return resolve_mission(map_name)


def gem_value(datablock):
    name = datablock.lower()
    for colour, value in GEM_VALUE.items():
        if name.startswith('gemitem' + colour):
            return value
    return 1   # spawnGem: a gem without a hunt value becomes a red gem


class MissionHunt:
    """The static hunt facts of one mission: gem spawn items, spawn triggers, spawn parameters, round length."""

    def __init__(self, map_name):
        from nav.collision.mission import mission_objects
        self.name = map_name
        self.path = _mission_path(map_name)
        text = io.open(self.path, encoding='utf-8', errors='ignore').read()
        objects = mission_objects(text)
        info = next(o['fields'] for o in objects if o['class_name'] == 'ScriptObject' and 'gamemode' in o['fields'])
        if any(o['class_name'] == 'SimGroup' and o['name'] == 'GemGroups' for o in objects):
            raise ValueError(map_name + ': GemGroups missions are not ported')
        if info.get('game', '') == 'PlatinumQuest':
            raise ValueError(map_name + ': PlatinumQuest spawn chances are not ported')
        self.round_ms = int(float(info.get('time', '300000') or 300000))
        self.radius = float(info.get('radiusfromgem') or HUNT_DEFAULTS['radius'])
        self.max_gems = int(info.get('maxgemsperspawn') or HUNT_DEFAULTS['max_gems'])
        self.min_points = int(info.get('minpointsperspawn') or HUNT_DEFAULTS['min_points'])
        self.min_gems = int(info.get('mingemsperspawn') or HUNT_DEFAULTS['min_gems'])
        self.max_loops = HUNT_DEFAULTS['max_loops']
        self.spawn_block = float(info.get('spawnblock') or self.radius * 2)
        gems = []
        for o in objects:
            if o['class_name'] != 'Item':
                continue
            db = o['fields'].get('datablock', '')
            if not db.lower().startswith('gemitem'):
                continue
            pos = np.asarray([float(v) for v in o['fields']['position'].split()[:3]])
            gems.append((pos, gem_value(db), db))
        if not gems:
            raise ValueError(map_name + ': no gem items')
        self.gem_positions = np.asarray([g[0] for g in gems])
        self.gem_values = np.asarray([g[1] for g in gems], dtype=float)
        self.gem_datablocks = [g[2] for g in gems]
        self.extra_values = self.gem_values - 1
        self.triggers = []
        for o in objects:
            if o['fields'].get('datablock') == 'SpawnTrigger':
                pos = np.asarray([float(v) for v in o['fields']['position'].split()[:3]])
                add = o['fields'].get('add')
                add = np.asarray([float(v) for v in add.split()[:3]]) if add else np.asarray(RESPAWN_ADD_DEFAULT)
                self.triggers.append(pos + add)
        if not self.triggers:
            raise ValueError(map_name + ': no SpawnTrigger')
        self.triggers = np.asarray(self.triggers)

    def item(self, index, item_id):
        """A gem item dict in the journal's format (what the planner reads from dynamics['items'])."""
        p = self.gem_positions[index]
        return dict(id=int(item_id), gem=True, powerup=0, position=[float(v) for v in p],
                    bounds=[float(v) for v in np.r_[p, p] + GEM_BOUNDS], value=float(self.gem_values[index]))


class HuntSpawner:
    """huntGems.cs spawn selection with its own random generator. State: the last center (LastGemSpawner) and the
    per-gem spawn weights (kept for fidelity: the script's weight update is dead code, so they stay 0)."""

    def __init__(self, mission, seed):
        self.m = mission
        self.rng = np.random.default_rng(seed)
        self.last_center = None
        self.spawn_weight = np.zeros(len(mission.gem_positions))
        self.next_id = 10 ** 6   # synthetic item ids for spawned gems (unique within a round)
        self.spawns = 0

    def _randint(self, a, b):
        """TorqueScript getRandom(a, b): an integer in [a, b]."""
        return int(self.rng.integers(a, b + 1))

    def center(self, exclude, up):
        """getCenterGems for one center: ten random gems; those at least spawn_block (plus their weight) from the last
        center are valid; the farthest of the too-close ones found before any valid one is kept as a fallback."""
        m = self.m; n = len(m.gem_positions)
        valid = []; furthest = None; furthest_dist = 0.
        if self.last_center is not None:
            block = m.gem_positions[self.last_center]
        for _ in range(10):
            g = self._randint(0, n - 1)
            if g == exclude or g in up:
                continue
            if self.last_center is not None:
                dist = float(np.linalg.norm(m.gem_positions[g] - block)) + self.spawn_weight[g]
                if dist < m.spawn_block:
                    if not valid and dist > furthest_dist:
                        furthest_dist = dist - self.spawn_weight[g]; furthest = g
                    continue
                valid.append(g)
            else:
                valid.append(g)
        if furthest is not None:
            valid.append(furthest)
        if not valid:
            return None
        # five draws, the last one wins (totalDistance 0 >= bestDistance 0 for a single center)
        c = None
        for _ in range(5):
            c = valid[self._randint(0, len(valid) - 1)]
        return c

    def group(self, center, exclude, up):
        """spawnGemGroupSet: spawnables within the radius (doubled up to twice while short of min_points / min_gems),
        weighted radius - dist - |dz| + randint(0, extra + 3), the best max_gems - 1 of them spawn with the center."""
        m = self.m
        radius = m.radius; loops = 0; points = 1 + m.extra_values[center]
        spawnables = []; weights = {}
        d = m.gem_positions - m.gem_positions[center]; dist = np.linalg.norm(d, axis=1)
        while (points < m.min_points or len(spawnables) < m.min_gems) and loops < 2:
            for g in range(len(m.gem_positions)):
                if g == exclude or g == center or g in weights:
                    continue
                if dist[g] < radius:
                    spawnables.append(g)
                    weights[g] = radius - dist[g] - abs(d[g][2]) + self._randint(0, int(m.extra_values[g]) + 3)
                    points += 1 + m.extra_values[g]
            radius *= 2; loops += 1
        order = sorted(spawnables, key=lambda g: -weights[g])   # SortSpawnWeight: descending
        chosen = [center]; points = 1 + m.extra_values[center]; loops = 0
        while (points < m.min_points or len(chosen) < m.min_gems) and loops < m.max_loops:
            count = min(len(order), m.max_gems - 1)
            for g in order[:count]:
                if g in up or g in chosen:
                    continue
                chosen.append(g); points += 1 + m.extra_values[g]
            loops += 1
        return chosen

    def spawn(self, exclude=None, up=()):
        """The next group as gem indices. `exclude` = the gem whose pickup emptied the field (it cannot be the center),
        `up` = gems still standing (never re-spawned)."""
        up = set(up)
        c = self.center(exclude, up)
        if c is None:
            c = self.center(None, up)
        if c is None:
            return []
        chosen = self.group(c, exclude, up)
        self.last_center = c; self.spawns += 1
        return chosen

    def spawn_items(self, exclude=None, up=()):
        out = []
        for g in self.spawn(exclude, up):
            out.append((g, self.m.item(g, self.next_id))); self.next_id += 1
        return out


def respawn_point(mission, rng, marble_position, visible_gems, last_touch=None, last_trigger=None, since_last_s=math.inf):
    """getFurthestSpawnTrigger: the trigger farthest from the nearest highest-value visible gem (46/51) or the marble's
    last contact position (5/51), among a random prefix of the trigger list, skipping the trigger used within 4 s.
    Returns (index, position)."""
    T = mission.triggers; n = len(T)
    ref = np.asarray(marble_position, dtype=float)
    if visible_gems is not None and len(visible_gems) and int(rng.integers(0, 51)) > 5:
        # getNearestGem(highestValue=true): the highest value, ties by distance
        gems = sorted(visible_gems, key=lambda g: (-float(g[3]), float(np.linalg.norm(np.asarray(g[:3]) - ref))))
        ref = np.asarray(gems[0][:3], dtype=float)
    elif last_touch is not None:
        ref = np.asarray(last_touch, dtype=float)
    count = int(rng.integers(n // 2, 2 * n + 1)) + 3
    best = None; best_d = -1.
    for i in range(min(count, n)):
        if last_trigger is not None and i == last_trigger and since_last_s <= 4.:
            continue
        d = float(np.linalg.norm(T[i] - ref))
        if d > best_d:
            best = i; best_d = d
    if best is None:
        best = int(rng.integers(0, n))
    return best, T[best].copy()


def respawned_dynamics(dynamics, position, clock_ms):
    """The bridge dynamics dict of the marble just respawned: at rest at `position`, camera yaw 0, powerup cleared and
    restored after POWERUP_DELAY_MS, Blast meter kept, pending action cleared (as the pools' OOBCLICK leaves it)."""
    import copy
    d = copy.deepcopy(dynamics)
    st = d['native']['state']
    st['position'] = [float(v) for v in position]; st['velocity'] = [0., 0., 0.]; st['spin'] = [0., 0., 0.]
    st['camera_offset'] = [0., 1., -0.4349655210971832]; st['best_contact_normal'] = [0., 0., 1.]
    st['next_yaw'] = 0.; st['yaw'] = 0.; st['next_pitch'] = CAMERA_PITCH_DEFAULT; st['pitch'] = CAMERA_PITCH_DEFAULT
    st['oob'] = False; st['mode'] = 0; st['ground_ms'] = 0; st['smooth_camera'] = False
    st['effects'] = [[False, 0] for _ in range(6)]
    st['clock_ms'] = int(clock_ms); d['native']['frame_clock_ms'] = int(clock_ms)
    held = int(st.get('held', 0))
    inv = d.get('inventory') or {}
    held_type = int(inv.get('held_type', held) or held)
    st['held'] = 0
    if inv:
        inv['held_type'] = 0; inv['mega_active'] = False; inv['mega_left_ms'] = 0; inv['use_latched'] = False
        if held_type:
            inv['restore_type'] = held_type; inv['restore_left_ms'] = POWERUP_DELAY_MS - 64
        else:
            inv['restore_type'] = 0; inv['restore_left_ms'] = -1
        for k in [k for k in inv if k.startswith('_')]:
            del inv[k]
    if 'inventory_context' in st and inv:
        st['inventory_context'] = copy.deepcopy(inv)
    zero = {'x': 0, 'y': 0, 'z': 0, 'yaw': 0, 'pitch': 0, 'jump': False, 'use': False, 'trigger1': False}
    d['native']['last_move'] = dict(zero); d['native']['pending_move'] = dict(zero)
    d['bridge_action'] = [0.] * len(d['bridge_action']); d['pow_yaw_hold'] = 0
    if d.get('game_context'):
        d['game_context']['respawns'] = int(d['game_context'].get('respawns', 0)) + 1
    return d


# ----------------------------------------------------------------------------------------------- distribution check
def recorded_spawns(journal):
    """(decision, size, points, centroid, values) per recorded spawn of one journal (gem set growth)."""
    import gzip
    rows = []
    for line in gzip.open(journal, 'rt', encoding='utf-8'):
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    rows.sort(key=lambda d: d['decision'])
    prev = None; out = []
    for d in rows:
        gems = {i['id']: i for i in d['before']['items'] if i.get('gem')}
        g = frozenset(gems)
        if prev is not None and len(g - prev) >= 1:
            new = [gems[i] for i in g - prev]
            out.append(dict(decision=d['decision'], size=len(new), points=sum(float(x.get('value') or 1) for x in new),
                            centroid=np.mean([x['position'] for x in new], axis=0),
                            values=sorted(float(x.get('value') or 1) for x in new)))
        prev = g
    return out


def _stats(spawns):
    sizes = [s['size'] for s in spawns]; pts = [s['points'] for s in spawns]
    dd = [float(np.linalg.norm(a['centroid'] - b['centroid'])) for a, b in zip(spawns[1:], spawns)]
    vals = {}
    for s in spawns:
        for v in s['values']:
            vals[v] = vals.get(v, 0) + 1
    tot = sum(vals.values()) or 1
    return dict(n=len(spawns), size=float(np.mean(sizes)) if sizes else 0., points=float(np.mean(pts)) if pts else 0.,
                dist=float(np.mean(dd)) if dd else 0., dist_p10=float(np.percentile(dd, 10)) if dd else 0.,
                dist_min=float(min(dd)) if dd else 0., values={k: round(v / tot, 3) for k, v in sorted(vals.items())})


def sampled_spawns(mission, seed, n, rng_pick=None):
    """n consecutive sampled spawns; every group is collected in full and the last gem picked (random) is excluded."""
    sp = HuntSpawner(mission, seed); pick = np.random.default_rng(seed + 1)
    out = []; exclude = None
    for _ in range(n):
        idx = sp.spawn(exclude)
        if not idx:
            break
        out.append(dict(size=len(idx), points=float(sum(mission.gem_values[idx])),
                        centroid=np.mean(mission.gem_positions[idx], axis=0),
                        values=sorted(float(v) for v in mission.gem_values[idx])))
        exclude = idx[int(pick.integers(0, len(idx)))]
    return out


def validate(maps, pools, n_samples=400):
    import glob
    print('%-22s %-9s %4s %5s %5s %6s %6s %6s  %s' % ('map', 'source', 'n', 'size', 'pts', 'dist', 'd_p10', 'd_min', 'value shares'))
    for name in maps:
        rec = []
        for pool in pools:
            for f in glob.glob(os.path.join(ML, 'dynamics_reports', pool, name, '*', 'physics_plan', '1', 'decisions.jsonl.gz')):
                rec.append(recorded_spawns(f))
        R = _stats([s for r in rec for s in r]) if rec else None
        # recorded consecutive distances must stay within one round
        if rec:
            dd = [float(np.linalg.norm(a['centroid'] - b['centroid'])) for r in rec for a, b in zip(r[1:], r)]
            R['dist'] = float(np.mean(dd)); R['dist_p10'] = float(np.percentile(dd, 10)); R['dist_min'] = float(min(dd))
        m = MissionHunt(name)
        S = _stats(sampled_spawns(m, 12345, n_samples))
        for label, st in (('recorded', R), ('sampled', S)):
            if st is None:
                continue
            print('%-22s %-9s %4d %5.2f %5.1f %6.1f %6.1f %6.1f  %s' % (name[:22], label, st['n'], st['size'], st['points'], st['dist'], st['dist_p10'], st['dist_min'], st['values']))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--pools', nargs='*', default=['overnight_20261026/qr3_explore_baseline', 'overnight_20261026/qr3_explore_candidate',
                                                   'overnight_20261026/hold3_explore_baseline', 'overnight_20261026/hold3_explore_candidate'])
    ap.add_argument('--maps', nargs='*', default=['KingOfTheMarble_Hunt', 'Sprawl_Hunt', 'CubeIsle_Hunt', 'Bowl_Hunt', 'Duplex_Hunt',
                                                  'FracturedIslands_Hunt', 'Cragmire_Hunt', 'VortexEffect_Hunt', 'TreasureBox_Hunt', 'Concentric_Hunt'])
    o = ap.parse_args()
    if o.validate:
        validate(o.maps, o.pools)
