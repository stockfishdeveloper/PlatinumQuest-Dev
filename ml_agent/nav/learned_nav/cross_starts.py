"""Stage 6b: crossing-drill starts from the navigator's REAL pickup states on KOTM.

    python -m nav.learned_nav.cross_starts

Reads the hybrid round logs that carry a 'pickups' list (hybrid.py, 2026-09-29 evening: the state right after every
pickup) and writes datasets/learned_nav/cross/starts_<set>.json: for every pickup at a gem A from which a straight
line to another gem spawn B crosses one void run a jump can clear (route.Route.jump_gap) and B is at most MAX_U away,
one start per (state, B) with the angle between the marble's heading and the line. Sets: the rounds of TAGS_DEV go to
'dev' (iterate here), those of TAGS_TEST to 'test' (judge there, never tune). Nothing names a map: gem spawns come from
the logs, the void from the geometry export.
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
from nav.learned_nav.geometry import Geometry                                     # noqa: E402
from nav.learned_nav import route as RT                                          # noqa: E402

MAP = 'KingOfTheMarble_Hunt'
TAGS_DEV = ('s6b_st0',)
TAGS_TEST = ('s6b_st1',)
MAX_U = 17.5                 # straight-line distance to B at most (KOTM: the 14.1 u diagonals and the 17.0 u sides)
MIN_VOID = 3.0               # the crossing must be a real hole (the 1.3 u slots between the centre gems are not)
MAX_ANGLE = 110.0            # degrees between the heading at the pickup and the line to B (the human: median 30, up to ~120)
OUT_DIR = os.path.join(HERE, 'datasets', 'learned_nav', 'cross')


def rounds(tags):
    for t in tags:
        f = os.path.join(HERE, 'logs', 'learned_nav', 'rounds', f'{t}_{MAP}.jsonl')
        if os.path.exists(f):
            for line in open(f):
                yield t, json.loads(line)


def main():
    g = Geometry(MAP)
    rt = RT.Route(g)
    spawns = set()
    for tags in (TAGS_DEV, TAGS_TEST):
        for _, r in rounds(tags):
            for pk in r.get('pickups', []):
                spawns.add(tuple(pk['gem']))
                for q in pk['gems']:
                    spawns.add(tuple(q))
    spawns = sorted(spawns)
    pairs = {}
    for a in spawns:
        for b in spawns:
            st = math.hypot(b[0] - a[0], b[1] - a[1])
            if a == b or st > MAX_U:
                continue
            gap = rt.jump_gap(a[0], a[1], b)
            if gap is not None and gap >= MIN_VOID:
                pairs.setdefault(a, []).append((b, round(st, 2), round(gap, 2), round(rt.walk(a[0], a[1], b), 2)))
    print('gem spawns', len(spawns), 'launch gems', len(pairs), 'pairs', sum(len(v) for v in pairs.values()))
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, tags in (('dev', TAGS_DEV), ('test', TAGS_TEST)):
        starts = []
        for tag, r in rounds(tags):
            for pk in r.get('pickups', []):
                a = tuple(pk['gem'])
                if a not in pairs or pk['owner'] != 'nav':
                    continue
                v = pk['v']; sp = math.hypot(v[0], v[1])
                hv = math.atan2(v[1], v[0]) if sp > 0.5 else None
                for b, st, gap, walk in pairs[a]:
                    line = math.atan2(b[1] - pk['p'][1], b[0] - pk['p'][0])
                    ang = 0.0 if hv is None else math.degrees((hv - line + math.pi) % (2 * math.pi) - math.pi)
                    if abs(ang) > MAX_ANGLE:
                        continue
                    starts.append({'id': len(starts), 'tag': tag, 'i': pk['i'], 'p': pk['p'], 'v': pk['v'], 'w': pk['w'],
                                   'gem_a': list(a), 'gem_b': list(b), 'straight': st, 'void': gap, 'walk': walk,
                                   'speed': round(sp, 2), 'angle': round(ang, 1)})
        out = os.path.join(OUT_DIR, f'starts_{name}.json')
        json.dump(starts, open(out, 'w'), indent=0)
        sp = np.array([s['speed'] for s in starts]); an = np.abs([s['angle'] for s in starts])
        print(f'{name}: {len(starts)} starts from {len(set((s["tag"], s["i"]) for s in starts))} pickups -> {out}')
        if len(starts):
            print('  speed median %.1f (p25 %.1f p75 %.1f); |angle| median %.0f, within 30 deg %.2f' % (
                np.median(sp), *np.percentile(sp, [25, 75]), np.median(an), (an <= 30).mean()))


if __name__ == '__main__':
    main()
