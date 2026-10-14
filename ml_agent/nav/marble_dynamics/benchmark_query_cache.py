"""Compare corrected cached queries and full forecasts with the original scan."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from nav.collision.mesh import CollisionMap,atomic_json
from nav.marble_dynamics.query_cache import CachedCollisionMap
from nav.marble_dynamics.physics import GeneralPhysics
from nav.marble_dynamics.bridge import predict_replies
from nav.marble_dynamics.record import ML,sha


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();results=[]
    root=ML/'dynamics_reports/reliability_geometry_pool_20261012'
    for name in ('Sprawl_Hunt','CubeIsle_Hunt','Duplex_Hunt'):
        source=(ML/'dynamics_reports/reliability_geometry_validation_20261012'/name/'101107/physics_plan/decisions.json') if name=='Sprawl_Hunt' else root/name/'111108/physics_plan/decisions.json'
        logs=json.loads(source.read_text());plain=CollisionMap(ML/'collision_maps'/(name+'.npz'));cached=CachedCollisionMap(ML/'collision_maps'/(name+'.npz'))
        rng=np.random.RandomState(79);queries=0
        for i in range(1800):
            state=logs[i%len(logs)]['dynamics']['native']['state'];position=np.asarray(state['position'])
            radius=float(rng.choice([.2,.5,1.5,8.]));lo=position-radius;hi=position+radius
            np.testing.assert_array_equal(cached.query_aabb(lo,hi),plain.query_aabb(lo,hi));queries+=1
        engines=[GeneralPhysics(plain),GeneralPhysics(cached)];times=[[],[]];forecasts=0
        for index in sorted(set(np.linspace(0,len(logs)-1,5,dtype=int))):
            dynamics=logs[index]['dynamics'];held=dynamics['native']['state']['held']
            actions=[[.3,.1,.5,0.,int(i==5),.7,int(i==7 and 1<=held<=5),.4,0] for i in range(40)]
            original=None
            for repeat in range(3):
                # Alternate timing order; no assumed speed win.
                for which in ((0,1) if repeat%2==0 else (1,0)):
                    started=time.perf_counter();forecast=predict_replies(engines[which],dynamics,actions);elapsed=time.perf_counter()-started
                    times[which].append(elapsed);current=digest(forecast)
                    if original is None:original=current
                    assert original==current,(name,index,repeat,which,'forecast differs')
                forecasts+=1
        medians=[float(np.median(t)) for t in times]
        result=dict(map=name,queries=queries,query_exact=True,rollouts=forecasts,rollout_exact=True,
            median_s=medians,median_speedup=medians[0]/medians[1],cache_hits=cached.cache_hits,cache_misses=cached.cache_misses,
            source=str(source),source_sha256=sha(source),geometry_sha256=plain.info['mesh_sha256'])
        results.append(result);print(json.dumps(result),flush=True)
    atomic_json(a.output,dict(cases=results,training=False,integrated=False,cache_source_sha256=sha(Path(__file__).with_name('query_cache.py'))))


if __name__=='__main__':main()
