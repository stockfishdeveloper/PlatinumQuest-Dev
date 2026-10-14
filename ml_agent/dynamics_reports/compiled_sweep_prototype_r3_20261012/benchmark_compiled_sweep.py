"""Compare an isolated compiled analytical sweep with the reviewed Python mesh.

Exact equality is reported separately from numerical distance. No live planner
integration or native executable replacement; all discrepancies are preserved.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from nav.collision.experimental_sweep import CompiledSweepMap
from nav.collision.mesh import CollisionMap,atomic_json
from nav.marble_dynamics.record import ML,sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dll',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--queries',type=int,default=900)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);rng=np.random.RandomState(817)
    names=['KingOfTheMarble_Hunt','Sprawl_Hunt','Bowl_Hunt','Concentric_Hunt','CubeIsle_Hunt',
           'Duplex_Hunt','Cragmire_Hunt','FracturedIslands_Hunt','TreasureBox_Hunt','VortexEffect_Hunt']
    sources=[Path(__file__),Path(__file__).with_name('mesh.py'),Path(__file__).with_name('experimental_sweep.py'),
             Path(__file__).with_name('sweep_kernel.cpp'),Path(__file__).with_name('build_sweep_kernel.ps1'),a.dll]
    atomic_json(a.output/'prepared.json',dict(maps=names,queries_per_map=a.queries,seed=817,
        source_hashes={str(path):sha(path) for path in sources},training=False,live_integrated=False))
    for path in sources:
        if path.suffix.lower() in ('.py','.cpp','.ps1'):(a.output/path.name).write_bytes(path.read_bytes())
    results=[]
    for name in names:
        plain=CollisionMap(ML/'collision_maps'/(name+'.npz'));fast=CompiledSweepMap(plain.path,a.dll)
        base='reliability_geometry_validation_20261012' if name in names[:3] else 'reliability_geometry_pool_20261012'
        seed=101107 if name in names[:3] else 111108
        source=ML/'dynamics_reports'/base/name/str(seed)/'physics_plan/decisions.json'
        logs=json.loads(source.read_text());queries=[]
        for i in range(a.queries):
            state=logs[i%len(logs)]['dynamics']['native']['state'];radius=float(rng.choice([state['radius'],.6666]))
            delta=float(rng.choice([.008,.016,.064]));dt=None if i%3==0 else delta
            if i%3==1:
                idx=rng.randint(len(plain.triangles));normal=plain.normals[idx]
                center=plain.triangles[idx].mean(axis=0)
                start=center+normal*(radius+rng.uniform(.02,2.))
                end=center-normal*(radius+rng.uniform(.02,1.))
            else:
                start=np.asarray(state['position'])+rng.normal(size=3)*radius*.3
                velocity=np.asarray(state['velocity'])+rng.normal(size=3)*20
                end=start+velocity*delta
            queries.append((start,end,radius,dt))
        timings=[];outputs=[]
        for engine in (plain,fast):
            started=time.perf_counter();outputs.append([engine.sweep_sphere(*query) for query in queries]);timings.append(time.perf_counter()-started)
        hits=sum(value is not None for value in outputs[0]);exact=0;identity=[];numeric=[];maximum=0.
        for index,(left,right) in enumerate(zip(*outputs)):
            if left==right:exact+=1;continue
            details=dict(index=index,query=[x.tolist() if isinstance(x,np.ndarray) else x for x in queries[index]],original=left,compiled=right)
            if left is None or right is None or left['triangle_id']!=right['triangle_id']:identity.append(details);continue
            error=max(abs(left['t']-right['t']),*[float(np.max(np.abs(np.asarray(left[field])-right[field]))) for field in ('center','point','normal')])
            maximum=max(maximum,error);numeric.append({**details,'maximum_component_error':error})
        report=dict(map=name,queries=len(queries),hits=hits,exact_queries=exact,contact_identity_mismatches=len(identity),
            numeric_mismatches=len(numeric),maximum_numeric_component_error=maximum,timing_s=timings,
            speedup=timings[0]/timings[1],geometry_sha256=plain.info['mesh_sha256'],source=str(source),source_sha256=sha(source))
        atomic_json(a.output/(name+'.json'),dict(**report,identity_mismatches=identity,numeric_mismatches_detail=numeric))
        results.append(report);atomic_json(a.output/'summary.json',dict(cases=results,training=False,live_integrated=False))
        print(json.dumps(report),flush=True)


if __name__=='__main__':main()
