"""Paired output equivalence and timing against immutable before-change sources."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

from nav.collision.mesh import CollisionMap,atomic_json
from nav.routing import line_clear
from nav.terrain import TerrainGrid,TerrainMap
from nav.marble_dynamics.physics import GeneralPhysics
from nav.marble_dynamics.evaluate import load_rows,errors
from nav.marble_dynamics.rollout import predict
from nav.marble_dynamics.record import ML,sha


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def timed(fn,repeats=3):
    elapsed=[];result=None
    for _ in range(repeats):
        t=time.perf_counter();result=fn();elapsed.append(time.perf_counter()-t)
    return result,float(np.median(elapsed))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,default=ML/'dynamics_reports/search_speed_20261012')
    p.add_argument('--data',type=Path,default=ML/'dynamics_reports/reliability_validation_final_20261012')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    old_mesh=load(a.reference/'mesh.py','reference_mesh')
    old_route=load(a.reference/'routing.py','reference_route')
    old_physics=load(a.reference/'physics.py','reference_physics')
    rng=np.random.RandomState(6122026);query_cases=[];route_cases=[];rollouts=[]
    for path in sorted((ML/'collision_maps').glob('*.npz')):
        mesh=CollisionMap(path);terrain=TerrainGrid(TerrainMap.resolve(path.stem))
        reference=old_mesh.CollisionMap(path)
        lows=rng.uniform(mesh.bounds[0]-1,mesh.bounds[1]+1,(500,3))
        highs=lows+rng.uniform(0,4,(500,3))
        # Include inclusive face/corner bounds and whole-map/outside queries.
        boxes=list(zip(lows,highs))+[(mesh.lo[i],mesh.hi[i]) for i in range(0,len(mesh.lo),max(1,len(mesh.lo)//25))]
        queries=lambda obj:[obj.query_aabb(lo,hi) for lo,hi in boxes]
        baseline,bt=timed(lambda:queries(reference));current,ct=timed(lambda:queries(mesh))
        assert all(np.array_equal(x,y) for x,y in zip(baseline,current)),path.stem
        query_cases.append(dict(map=path.stem,cases=len(boxes),identical=True,before_s=bt,after_s=ct,speedup=bt/ct))
        walk=np.argwhere(np.isfinite(terrain.walk_z));pairs=[]
        for index in range(300):
            k,j,i=walk[rng.randint(len(walk))];kk,jj,ii=walk[rng.randint(len(walk))]
            values=[terrain.wxs[i],terrain.wys[j],terrain.wxs[ii],terrain.wys[jj]]
            cast=np.float32 if index%2 else float
            pairs.append((tuple(cast(v) for v in values),dict(az=float(terrain.walk_z[k,j,i])+.19,
                bz=float(terrain.walk_z[kk,jj,ii]) if index%3 else None)))
        routes=lambda fn:[fn(terrain,*args,**kwargs) for args,kwargs in pairs]
        baseline,bt=timed(lambda:routes(old_route.line_clear));current,ct=timed(lambda:routes(line_clear))
        assert baseline==current,path.stem
        route_cases.append(dict(map=path.stem,cases=len(pairs),identical=True,clear=sum(current),before_s=bt,after_s=ct,speedup=bt/ct))
        print(json.dumps(dict(map=path.stem,query_speedup=query_cases[-1]['speedup'],route_speedup=route_cases[-1]['speedup'])),flush=True)
    for name in ('KingOfTheMarble_Hunt','Sprawl_Hunt','Bowl_Hunt'):
        source=a.data/name/'51013/frozen_navigator/native.jsonl.gz';rows=load_rows(source)
        mesh=CollisionMap(ML/'collision_maps'/(name+'.npz'))
        old_engine=old_physics.GeneralPhysics(old_mesh.CollisionMap(mesh.path));engine=GeneralPhysics(mesh)
        for start in (0,32,64,124):
            window=rows[start:start+24]
            controls=[dict(move=r['move'],events=r['control_events']) for r in window]
            baseline,bt=timed(lambda:predict(old_engine,window[0]['before'],controls,64))
            current,ct=timed(lambda:predict(engine,window[0]['before'],controls,64))
            assert baseline['failure']==current['failure'] and len(baseline['states'])==len(current['states'])
            maximum=np.max([errors(x,y) for x,y in zip(baseline['states'],current['states'])],axis=0).tolist()
            assert maximum==[0.,0.,0.],(name,start,maximum)
            rollouts.append(dict(map=name,start=start,replies=len(controls),before_s=bt,after_s=ct,speedup=bt/ct,
                exact_position_velocity_spin=True,added_ambiguity_intervals=len(current['ambiguous_intervals'])-len(baseline['ambiguous_intervals']),
                source_sha256=sha(source)))
        print(json.dumps(dict(map=name,rollout_speedups=[r['speedup'] for r in rollouts if r['map']==name])),flush=True)
    atomic_json(a.output,dict(queries=query_cases,routes=route_cases,rollouts=rollouts,
        source_hashes={str(path):sha(path) for path in (Path(__file__),ML/'nav/routing.py',ML/'nav/collision/mesh.py',Path(__file__).with_name('physics.py'))},
        contract='Three-repeat median paired workstation timings; query/route boolean and rollout p/v/spin equivalence, not a real-time/full-search claim.',policy_training_started=False))


if __name__=='__main__':main()
