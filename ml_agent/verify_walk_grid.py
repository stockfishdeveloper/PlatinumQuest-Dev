"""Ground-truth the walk grid against the actual game, by teleporting the marble into it.

WHY THIS EXISTS. On King of the Marble the generated walk grid had PHANTOM HOLES: solid ground
marked non-walkable. Training then sampled goals only on the "real" cells, the policy learned to
avoid perfectly good floor, and every metric validated itself against the error. Fixing it was
worth a 5x jump in real-round score (2026-09-19). The standing rule is that any NEW map must be
verified this way before it is trained on.

WHAT IT DOES. Teleports the marble onto sampled cells with zero velocity, lets it settle, and
records whether it is still there. Two failure modes, and they need opposite fixes:

  PHANTOM HOLE   grid says NOT walkable, marble actually stands      -> usable ground being thrown
                                                                        away; training never goes there
  PHANTOM FLOOR  grid says walkable, marble actually falls           -> goals sampled in the void;
                                                                        the policy chases nothing

Samples three populations: cells the grid calls walkable, cells it calls non-walkable that sit
NEXT to walkable ground (the suspicious ones, interior to the map rather than off the edge), and
every gem spawn point, which matters most because training goals are drawn from exactly those.

    NAV_PORT=8920 NAV_MAP=Sprawl_Hunt python verify_walk_grid.py
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from nav.env import HuntEnv                       # noqa: E402
from nav.protocol import NOOP_ACTION              # noqa: E402
from nav.terrain import TerrainGrid, TerrainMap   # noqa: E402

PORT = int(os.environ.get('NAV_PORT', '8920'))
MAP = os.environ.get('NAV_MAP', 'Sprawl_Hunt')
N_EACH = int(os.environ.get('NAV_SAMPLES', '60'))     # cells per population
SETTLE = 45                                            # ticks to let the marble settle after a teleport
DROP_TOL = 1.2                                         # u below the target height = it fell


def probe(env, x, y, z, settle):
    """Teleport onto (x, y, z), let it settle, and report where it ended up.

    Returns (outcome, dz, drift). Outcome is one of:
      'stands'  still on that cell, did not drift
      'slides'  still at roughly that height but drifted away: a bevel or ramp it rolls off
      'falls'   ended up well below: no usable ground here
    """
    env.teleport(x, y, z + 0.5, 0.0, 0.0, 0.0, settle_ticks=3)
    for _ in range(settle):
        env.step(NOOP_ACTION, repeat=1)
    o = env.msg.obs
    px, py, pz = float(o[0]), float(o[1]), float(o[2])
    drift = math.hypot(px - x, py - y)
    dz = pz - z
    if dz < -DROP_TOL:
        return 'falls', dz, drift
    if drift > 1.5:
        return 'slides', dz, drift
    return 'stands', dz, drift


def main():
    t = TerrainGrid(TerrainMap.resolve(MAP))
    rng = np.random.default_rng(0)
    has_floor = np.isfinite(t.walk_top)
    nb = np.zeros_like(t.walkable)
    for dj in (-1, 0, 1):
        for di in (-1, 0, 1):
            nb |= np.roll(np.roll(t.walkable, dj, 0), di, 1)

    # THREE DISTINCT QUESTIONS. The first version of this probe lumped the last two together and
    # reported "100 % phantom holes", which was meaningless: every cell it sampled had ground and
    # was excluded for STEEPNESS, and a marble does briefly rest on a steep bevel.
    pops = [
        ('walkable (control)',              np.argwhere(t.walkable),                              'stands'),
        ('NO floor at all (true void)',     np.argwhere((~t.walkable) & (~has_floor) & nb),       'falls'),
        ('has floor, excluded as too steep', np.argwhere((~t.walkable) & has_floor & nb),         '?'),
    ]

    env = HuntEnv(PORT, speed=1)
    print(f'[verify] listening on {PORT}; launch the game with -autotrain {MAP} -aiport {PORT}')
    env.connect()
    env.control('LOCKSTEP 1')
    env.control('RENDEREVERY 1')
    print('[verify] connected on %s, walk grid %dx%d, %d walkable'
          % (env.info.get('mission', '?'), t.wH, t.wW, int(t.walkable.sum())))
    print()
    print('  %-36s %6s %8s %8s %8s' % ('population', 'n', 'stands', 'slides', 'falls'))
    results = {}
    for name, cells, expect in pops:
        if len(cells) == 0:
            continue
        idx = rng.choice(len(cells), size=min(N_EACH, len(cells)), replace=False)
        tally = {'stands': 0, 'slides': 0, 'falls': 0}
        ex = []
        for (j, i) in cells[idx]:
            x, y = float(t.wxs[i]), float(t.wys[j])
            z = float(t.walk_top[j, i]) if np.isfinite(t.walk_top[j, i]) else float(t.z_floor_min)
            out, dz, drift = probe(env, x, y, z, SETTLE)
            tally[out] += 1
            if out != expect and expect != '?' and len(ex) < 8:
                ex.append((round(x, 1), round(y, 1), round(z, 1), out, round(dz, 2), round(drift, 2)))
        n = sum(tally.values())
        print('  %-36s %6d %7d%% %7d%% %7d%%' % (name, n, 100 * tally['stands'] // n,
                                                 100 * tally['slides'] // n, 100 * tally['falls'] // n))
        if ex:
            print('       unexpected: %s' % ex)
        results[name] = {'n': n, **tally, 'examples': ex}

    spawns = [(j, i) for (_, _, _, j, i) in t.gem_spawns]
    tally = {'stands': 0, 'slides': 0, 'falls': 0}
    bad = []
    for (j, i) in spawns:
        x, y = float(t.wxs[i]), float(t.wys[j])
        z = float(t.walk_top[j, i]) if np.isfinite(t.walk_top[j, i]) else float(t.z_floor_min)
        out, dz, drift = probe(env, x, y, z, SETTLE)
        tally[out] += 1
        if out != 'stands' and len(bad) < 10:
            bad.append((round(x, 1), round(y, 1), round(z, 1), out, round(dz, 2)))
    n = sum(tally.values())
    print('  %-36s %6d %7d%% %7d%% %7d%%' % ('GEM SPAWN cells (goals go here)', n,
                                             100 * tally['stands'] // n, 100 * tally['slides'] // n, 100 * tally['falls'] // n))
    if bad:
        print('       not standing: %s' % bad)
    results['gem_spawns'] = {'n': n, **tally, 'examples': bad}

    out = os.path.join(HERE, 'logs', 'nav', 'verify_walk_grid_%s.json' % MAP)
    json.dump(results, open(out, 'w'), indent=1)
    print()
    v = results.get('NO floor at all (true void)', {})
    w = results.get('walkable (control)', {})
    st = results.get('has floor, excluded as too steep', {})
    print('VERDICT')
    print('  control: walkable cells that did NOT hold the marble: %d of %d' % (w.get('n',0)-w.get('stands',0), w.get('n',0)))
    print('  PHANTOM FLOOR: void cells the grid called walkable:    (see control above)')
    print('  true void cells that did NOT drop the marble:          %d of %d' % (v.get('stands',0)+v.get('slides',0), v.get('n',0)))
    if st.get('n'):
        usable = st.get('stands', 0)
        print('  slope-excluded ground that HOLDS the marble: %d of %d (%.0f%%) -> %s'
              % (usable, st['n'], 100.0*usable/st['n'],
                 'usable ground being discarded' if usable > 0.5*st['n'] else 'correctly excluded'))
    print('  gem spawns not standing: %d of %d' % (results['gem_spawns']['n'] - results['gem_spawns']['stands'], results['gem_spawns']['n']))
    print('  wrote', out)


if __name__ == '__main__':
    main()
