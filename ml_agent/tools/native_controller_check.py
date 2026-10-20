"""Compiled routing / controllers / bridge (nav/native/controllers.inc) vs the Python originals, on random inputs.

usage (repo root, PYTHONPATH=ml_agent): python ml_agent/tools/native_controller_check.py [--n 3000] MAP [MAP ...]
"""
import argparse
import ctypes
import math
from pathlib import Path

import numpy as np

from nav.terrain import TerrainMap
from nav.routing import RoutePlanner, goal_key, line_clear, path_waypoint
from nav.marble_dynamics.static_cache import cached_terrain, cached_walking, load_queries
from nav.collision.experimental_sweep import CachedCompiledSweepMap
from nav.marble_dynamics.record import ML
from nav.marble_dynamics.physics import GeneralPhysics
from nav.marble_dynamics.bridge import translated
import nav.marble_dynamics.search as S
import nav.marble_dynamics.gem_demo as GD
from nav.native.marble_sim import NativeSim, _ptr

CACHE = Path(r'C:\Users\doug\AppData\Local\Temp\pq_checks\fast_nav_static_cache_20261013')
DLL = Path(r'C:\Users\doug\AppData\Local\Temp\pq_checks\compiled_sweep_20261012\sweep_kernel.dll')


def same(a, b):
    return a == b or (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b))


def main():
    p = argparse.ArgumentParser(); p.add_argument('maps', nargs='+'); p.add_argument('--n', type=int, default=3000); p.add_argument('--through-pool',type=Path)
    p.add_argument('--fast-pool',type=Path,help='Check actual fast forecasts against Python on recorded native states')
    a = p.parse_args()
    rng = np.random.default_rng(7)
    sim0 = None
    # scalar helpers (map independent)
    bad = 0
    for _ in range(a.n * 4):
        x = float(rng.standard_normal() * 10 ** rng.uniform(-8, 4))
        lib = None
    for name in a.maps:
        terrain = cached_terrain(TerrainMap.resolve(name), CACHE)
        mesh = CachedCompiledSweepMap(ML / 'collision_maps' / (name + '.npz'), DLL); load_queries(mesh, DLL, CACHE)
        walk = cached_walking(terrain, mesh, DLL, CACHE)
        engine = GeneralPhysics(mesh); sim = NativeSim(engine); L = sim.lib; th = sim.terrain(walk)
        res = {}
        def tally(key, ok):
            c = res.setdefault(key, [0, 0]); c[0] += 1; c[1] += bool(ok)
        # 1. scalar helpers
        for _ in range(a.n):
            x = float(rng.standard_normal() * 10 ** rng.uniform(-8, 3))
            tally('round6', same(L.ms_py_round(x, 6), round(x, 6)))
            tally('round3', same(L.ms_py_round(x, 3), round(x, 3)))
            tally('g6', same(L.ms_g6(x), float(format(x, '.6g'))))
            y = float(rng.standard_normal() * 10 ** rng.uniform(-3, 3))
            tally('hypot', same(L.ms_hypot(x, y), math.hypot(x, y)))
        # 2. grid lookups (random points, half of them on cell boundaries)
        nodes = walk._nodes
        x0, y0 = float(walk.wxs[0]), float(walk.wys[0])
        ji = (ctypes.c_int * 2)()
        for _ in range(a.n):
            if rng.random() < .5:
                x = x0 + (rng.integers(0, walk.wW) + .5) * walk.walk_res + float(rng.choice([-1, 0, 1])) * 1e-6
                y = y0 + (rng.integers(0, walk.wH) + .5) * walk.walk_res + float(rng.choice([-1, 0, 1])) * 1e-6
            else:
                x = x0 + rng.uniform(-2, walk.wW + 2) * walk.walk_res; y = y0 + rng.uniform(-2, walk.wH + 2) * walk.walk_res
            L.ms_cell_of(th, x, y, 1 if isinstance(x, np.floating) else 0, ji)
            j, i = walk.cell_of(x, y)
            tally('cell_of', (ji[0], ji[1]) == (j, i))
            z = float(rng.uniform(-20, 40)) if rng.random() < .8 else None
            k = walk.level_at(j, i, z)
            tally('level_at', L.ms_level_at(th, j, i, 0 if z is None else 1, 0. if z is None else z) == k)
        # 3. line of sight, waypoints, distances toward real walkable goals
        fields = {}
        idx = rng.integers(0, len(nodes), size=(a.n // 6, 2))
        wp = np.zeros(3)
        for s_, g_ in idx:
            ks, js, is_ = nodes[s_]; kg, jg, ig = nodes[g_]
            sx = float(walk.wxs[is_]) + rng.uniform(-.5, .5); sy = float(walk.wys[js]) + rng.uniform(-.5, .5)
            sz = float(walk.walk_z[ks, js, is_]) + .19
            gem = np.asarray([float(walk.wxs[ig]), float(walk.wys[jg]), float(walk.walk_z[kg, jg, ig]) + .3])
            if rng.random() < .5:   # a closer goal for a clearer mix
                gem = np.asarray([sx + rng.uniform(-8, 8), sy + rng.uniform(-8, 8), sz + rng.uniform(-1, 2)])
            tally('line_clear', bool(L.ms_line_clear(th, sx, sy, gem[0], gem[1], 1, sz, 1, gem[2], 1, 1)) == bool(line_clear(walk, sx, sy, gem[0], gem[1], az=sz, bz=gem[2])))
            key = goal_key(gem)
            if key not in fields:
                if len(fields) > 40:
                    fields.pop(next(iter(fields)))
                fields[key] = np.ascontiguousarray(walk.goal_field(gem[0], gem[1], z=gem[2]), dtype=np.float32)
            field = fields[key]
            r = L.ms_path_waypoint(th, _ptr(field), sx, sy, _ptr(np.ascontiguousarray(gem)), 1, sz, _ptr(wp))
            pw, routed = path_waypoint(walk, sx, sy, field, gem, mz=sz)
            tally('path_waypoint', bool(r) == bool(routed) and tuple(wp) == tuple(float(v) for v in pw))
            d = L.ms_dist_at(th, _ptr(field), sx, sy, key[0], key[1], 1, sz, 0)
            tally('dist_at', same(d, float(walk.dist_at(field, sx, sy, key[:2], z=sz))))
            # 4. a whole route-following decision (SequenceProposal + Proposal + joystick)
            route = RoutePlanner(walk); route.navigate((sx, sy, sz), gem)
            spec = S.SequenceProposal(route, gem, 24, offset=float(rng.choice([0., .5, -1.])), scale=float(rng.choice([1., .7, .4])),
                                      jump=int(rng.choice([-1, 0, 2])), use=int(rng.choice([-1, 0, 3])), blast=int(rng.choice([-1, 0])),
                                      direct=bool(rng.random() < .3), brake_after=int(rng.choice([12, 0])) if rng.random() < .3 else None)
            spec.min_speed = 8. if rng.random() < .3 else None   # ABI 8 per-plan pickup minimum
            spec.fast = (None, 16., 20.)[int(rng.integers(0, 3))]
            state = dict(position=[sx, sy, sz], velocity=[float(v) for v in rng.uniform(-12, 12, 3)], radius=.18975)
            for index in range(3):
                from nav.native.forecast import seq_array
                arr = seq_array(spec)
                want = spec(state, index)
                got = np.zeros(9); pv = np.asarray(state['position'] + state['velocity'] + [state['radius']], dtype=np.float64)
                L.ms_seq_action(th, _ptr(arr), _ptr(np.ascontiguousarray(spec.route.field, dtype=np.float32)), None,
                                _ptr(np.asarray(consts_now(), dtype=np.float64)), _ptr(pv), index, _ptr(got))
                ok = all(same(float(x), float(y)) for x, y in zip(got, want))
                tally('sequence_action', ok)
                if not ok and res['sequence_action'][0] - res['sequence_action'][1] <= 3:
                    print('   seq diff', name, index, list(np.round(got, 6)), [round(float(v), 6) for v in want])
                state = dict(state, position=[state['position'][0] + .3, state['position'][1] - .2, state['position'][2]])
        # 5. bridge translation of recorded-like actions
        for _ in range(a.n):
            act = [float(rng.choice([0., rng.uniform(0, 1.5)])) for _ in range(4)] + [int(rng.integers(0, 2)), float(rng.uniform(-7, 7)),
                   int(rng.random() < .1), float(rng.uniform(-4, 4)), int(rng.random() < .05)]
            hold = int(rng.integers(0, 13)) if rng.random() < .3 else 0; aim = float(rng.uniform(-4, 4))
            mv, cam, h2, a2 = translated(act, hold, aim)
            hh = ctypes.c_int(hold); aa = ctypes.c_double(aim); out = np.zeros(6); cy = ctypes.c_double(0)
            L.ms_translated(_ptr(np.asarray(act, dtype=np.float64)), ctypes.byref(hh), ctypes.byref(aa), _ptr(out), ctypes.byref(cy))
            ok = (out[0] == mv['x'] and out[1] == mv['y'] and bool(out[3]) == mv['jump'] and bool(out[4]) == mv['use']
                  and cy.value == cam[0]['value'] and hh.value == h2 and aa.value == a2)
            tally('translated', ok)
        if a.through_pool:
            check_through(a.through_pool,name,walk,engine,sim,tally)
        if a.fast_pool:
            check_fast(a.fast_pool,name,walk,engine,sim,tally)
        print(name, ' '.join('%s %d/%d' % (k, v[1], v[0]) for k, v in res.items()), flush=True)
        if any(n!=ok for n,ok in res.values()):raise SystemExit('Native/Python controller mismatch')



def check_through(pool,name,terrain,engine,sim,tally):
    import gzip,json,copy
    from nav.native.forecast import predict_native,_predict_per_reply
    journals=sorted((pool/name).rglob('decisions.jsonl.gz'))
    if not journals:raise ValueError('No recorded states for '+name)
    with gzip.open(journals[0],'rt',encoding='utf-8') as f:rows=[json.loads(line) for line in f]
    checked=0
    for row in rows[::83]:
        dynamics=row['before'];state=dynamics['native']['state']
        gems=[i for i in dynamics['items'] if i['gem']]
        if state['mode']!=0 or state['oob'] or len(gems)<2:continue
        gems.sort(key=lambda item:np.linalg.norm(np.asarray(item['position'])-state['position']))
        item,nitem=gems[:2];gem=np.asarray(item['position'],dtype=float);ngem=np.asarray(nitem['position'],dtype=float)
        def proposal():
            route=RoutePlanner(terrain);route.navigate(state['position'],gem)
            nr=RoutePlanner(terrain);nr.navigate(gem,ngem)
            spec=S.SequenceProposal(route,gem,48,through=(nr,ngem,float(nr.distance(gem))))
            spec.box=list(item['bounds']);return spec
        engine.bridge_physics_ms=16;engine.native_sim=sim
        a=predict_native(sim,engine,dynamics,proposal(),False)
        b=_predict_per_reply(sim,engine,dynamics,proposal(),False)
        ok=(a['proposed']==b['proposed'] and a['failure']==b['failure']
            and all(x[k]==y[k] for x,y in zip(a['states'],b['states']) for k in ('position','velocity','spin','radius')))
        tally('through_tick_forecast',ok)
        if not ok:print('through difference',name,row['decision'],flush=True)
        checked+=1
        if checked>=16:break
    if checked<4:raise ValueError('Not enough playable through-gem cases')


def check_fast(pool,name,terrain,engine,sim,tally):
    import gzip,json
    from nav.native.forecast import predict_native,_predict_per_reply
    journals=sorted((pool/name).rglob('decisions.jsonl.gz'))
    if not journals:raise ValueError('No recorded states for '+name)
    with gzip.open(journals[0],'rt',encoding='utf-8') as f:rows=[json.loads(line) for line in f]
    checked=0;different=0
    for row in rows[::173]:
        dynamics=row['before'];state=dynamics['native']['state']
        gems=[i for i in dynamics['items'] if i['gem']]
        if state['mode']!=0 or state['oob'] or not gems:continue
        goal=row.get('choice',{}).get('goal_context',{}).get('destination') or gems[0]['position']
        gem=np.asarray(goal,dtype=float)
        def proposal(fast,direct):
            route=RoutePlanner(terrain);route.navigate(state['position'],gem)
            return S.SequenceProposal(route,gem,48,direct=direct,fast=fast)
        engine.bridge_physics_ms=16;engine.native_sim=sim
        for direct in (False,True):
            normal=predict_native(sim,engine,dynamics,proposal(None,direct),False)
            for top in (16.,20.):
                a=predict_native(sim,engine,dynamics,proposal(top,direct),False)
                b=_predict_per_reply(sim,engine,dynamics,proposal(top,direct),False)
                ok=(a['proposed']==b['proposed'] and a['failure']==b['failure']
                    and len(a['states'])==len(b['states'])
                    and all(x[k]==y[k] for x,y in zip(a['states'],b['states']) for k in ('position','velocity','spin','radius')))
                tally('fast_forecast',ok)
                if not ok:print('fast difference',name,row['decision'],top,direct,flush=True)
                different+=int(a['proposed']!=normal['proposed'])
        checked+=1
        if checked>=8:break
    if checked<4:raise ValueError('Not enough playable fast cases')
    tally('fast_changes_controls',different>0)


def consts_now():
    return [GD.TARGET_SPEED_CAP, GD.BRAKE_DECEL, GD.THROTTLE_GAIN, GD.THROTTLE_CAP, S.PICKUP_MIN_SPEED, S.THROUGH_SWITCH_U, 1. if S.ROUTE_SPEED_LAW else 0.,
            float(S.OPEN_TOP or 0.), float(S.OPEN_NEAR_U), float(S.OPEN_FAR_U), float(S.TRAVEL_TOP or 0.), float(S.TRAVEL_NEAR_U), float(S.TRAVEL_FAR_U)]


if __name__ == '__main__':
    main()
