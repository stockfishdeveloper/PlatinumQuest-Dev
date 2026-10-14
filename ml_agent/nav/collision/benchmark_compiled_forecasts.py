"""Full-control/decision checks before enabling the compiled sweep in live play."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import time
import unittest

import numpy as np

from nav.collision.experimental_sweep import CompiledSweepMap
from nav.collision.mesh import CollisionMap,atomic_json
from nav.marble_dynamics.bridge import predict_replies
from nav.marble_dynamics.geometry_route import walking_terrain
from nav.marble_dynamics.ground_cache import MemoGroundClearance
from nav.marble_dynamics.landing import choose_landing
from nav.marble_dynamics.physics import GeneralPhysics
from nav.marble_dynamics.record import ML,sha
from nav.routing import RoutePlanner
from nav.terrain import TerrainGrid,TerrainMap


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dll',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__),Path(__file__).with_name('experimental_sweep.py'),Path(__file__).with_name('sweep_kernel.cpp'),
             Path(__file__).with_name('mesh.py'),a.dll,*Path(__file__).parents[1].joinpath('marble_dynamics').glob('*.py')]
    atomic_json(a.output/'prepared.json',dict(source_hashes={str(path):sha(path) for path in sources},
        native_suite='Existing immutable native/control/landing/recharge regressions with compiled geometry engine',
        landing_case='Fresh Concentric151113 failed terminal, all74 proposals, seed997/samples40',
        training=False,live_integrated=False))
    (a.output/a.dll.name).write_bytes(a.dll.read_bytes())
    suite=unittest.TestSuite()
    for name in ('test_native','test_bridge','test_landing','test_recharge','test_search','test_inventory_goals','test_cache_integration'):
        module=importlib.import_module('nav.marble_dynamics.'+name)
        # Isolated validation process only. Product imports/methods are untouched.
        if hasattr(module,'CollisionMap'):module.CollisionMap=lambda path:CompiledSweepMap(path,a.dll)
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
    native=unittest.TextTestRunner(verbosity=1).run(suite)
    atomic_json(a.output/'native_suite.json',dict(tests=native.testsRun,failures=len(native.failures),errors=len(native.errors),passed=native.wasSuccessful()))
    assert native.wasSuccessful(),'Compiled native fixture suite failed'
    cases=[]
    names=['KingOfTheMarble_Hunt','Sprawl_Hunt','Bowl_Hunt','Concentric_Hunt','CubeIsle_Hunt','Duplex_Hunt',
           'Cragmire_Hunt','FracturedIslands_Hunt','TreasureBox_Hunt','VortexEffect_Hunt']
    for name in names:
        base='reliability_geometry_validation_20261012' if name in names[:3] else 'reliability_geometry_pool_20261012'
        seed=101107 if name in names[:3] else 111108
        source=ML/'dynamics_reports'/base/name/str(seed)/'physics_plan/decisions.json'
        logs=json.loads(source.read_text());engines=[GeneralPhysics(CollisionMap(ML/'collision_maps'/(name+'.npz'))),
            GeneralPhysics(CompiledSweepMap(ML/'collision_maps'/(name+'.npz'),a.dll))]
        timings=[[],[]];mismatches=[];count=0
        for index in sorted(set(np.linspace(0,len(logs)-1,4,dtype=int))):
            snapshot=logs[index]['dynamics'];held=snapshot['native']['state']['held']
            actions=[[.3,.1,.5,0.,int(i==5),.7,int(i==7 and 1<=held<=5),.4,0] for i in range(40)]
            hashes=[]
            for which in (0,1):
                started=time.perf_counter();forecast=predict_replies(engines[which],snapshot,actions)
                timings[which].append(time.perf_counter()-started);hashes.append(digest(forecast))
            count+=1
            if hashes[0]!=hashes[1]:mismatches.append(dict(index=index,hashes=hashes))
        report=dict(map=name,forecasts=count,exact=not mismatches,mismatches=mismatches,
            median_s=[float(np.median(t)) for t in timings],source_sha256=sha(source))
        cases.append(report);atomic_json(a.output/'forecasts.json',dict(cases=cases))
        print(json.dumps(report),flush=True)
    source=ML/'dynamics_reports/reliability_landing_validation_20261012/Concentric_Hunt/151113/physics_plan/terminal_search.json'
    terminal=json.loads(source.read_text());terrain=TerrainGrid(TerrainMap.resolve('Concentric_Hunt'));searches=[]
    for compiled in (False,True):
        mesh=CompiledSweepMap(ML/'collision_maps/Concentric_Hunt.npz',a.dll) if compiled else CollisionMap(ML/'collision_maps/Concentric_Hunt.npz')
        engine=GeneralPhysics(mesh);geometry=walking_terrain(terrain,mesh)
        geometry.ground_clearance=MemoGroundClearance(geometry.ground_clearance);route=RoutePlanner(geometry)
        started=time.perf_counter();chosen,pairs=choose_landing(engine,route,terminal['dynamics'],terminal['target'],
            1000,80,40,997,True);elapsed=time.perf_counter()-started
        hashes=[digest([info,rollout]) for info,rollout in pairs]
        searches.append(dict(compiled=compiled,candidates=len(pairs),candidate_hashes=hashes,
            selected=chosen[0] if chosen else None,elapsed_s=elapsed))
    landing_exact=searches[0]['candidate_hashes']==searches[1]['candidate_hashes'] and searches[0]['selected']==searches[1]['selected']
    atomic_json(a.output/'landing_search.json',dict(source_sha256=sha(source),cases=searches,exact=landing_exact))
    passed=all(case['exact'] for case in cases) and landing_exact
    atomic_json(a.output/'summary.json',dict(native_tests=native.testsRun,native_passed=True,forecasts=sum(c['forecasts'] for c in cases),
        forecast_exact=all(case['exact'] for case in cases),landing_search_exact=landing_exact,
        landing_search_s=[case['elapsed_s'] for case in searches],passed=passed,training=False,live_integrated=False))
    print(json.dumps(dict(passed=passed,landing_search_s=[case['elapsed_s'] for case in searches])),flush=True)
    assert passed,'Compiled full-control forecasts/candidate decisions differ'


if __name__=='__main__':main()
