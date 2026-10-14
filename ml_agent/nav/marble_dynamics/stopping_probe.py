"""Diagnose terminal braking viability without changing the live controller.

An endpoint can have four supported states yet already be unable to stop before
a ledge. Replay the complete proposed commands plus ordinary feedback braking;
the bridge's pending controls remain part of the forecast. No future native
observations are supplied to prediction and no state is modified.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

from nav.collision.experimental_sweep import CachedCompiledSweepMap
from nav.collision.mesh import atomic_json
from nav.marble_dynamics.bridge import predict_replies
from nav.marble_dynamics.geometry_route import walking_terrain
from nav.marble_dynamics.ground_cache import MemoGroundClearance
from nav.marble_dynamics.navigator import FrozenNavigator
from nav.marble_dynamics.physics import GeneralPhysics
from nav.marble_dynamics.record import ML,sha
from nav.marble_dynamics.search import BrakeContinuation,assess,search,useful_candidate
from nav.routing import RoutePlanner
from nav.terrain import TerrainGrid,TerrainMap


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dll',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--braking',type=int,default=48)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    checkpoint=ML/'models/nav/real_maps_20261010_A2/nav_latest.pth'
    fingerprint=sha(checkpoint)
    source=ML/'dynamics_reports/reliability_landing_validation_20261012/CubeIsle_Hunt/151113/physics_plan/decisions.json'
    logs=json.loads(source.read_text());initial=logs[503]
    name='CubeIsle_Hunt';dynamics=initial['dynamics'];gem=initial['gem']
    mesh=CachedCompiledSweepMap(ML/'collision_maps'/(name+'.npz'),a.dll)
    engine=GeneralPhysics(mesh);terrain=TerrainGrid(TerrainMap.resolve(name))
    geometric=walking_terrain(terrain,mesh)
    geometric.ground_clearance=MemoGroundClearance(geometric.ground_clearance)
    route=RoutePlanner(geometric);nav=FrozenNavigator(terrain,checkpoint)
    atomic_json(a.output/'prepared.json',dict(map=name,decision=503,source=str(source),
        source_sha256=sha(source),checkpoint_sha256=fingerprint,dll_sha256=sha(a.dll),
        diagnostic_source_sha256=sha(__file__),horizon=40,braking=a.braking,training=False,
        limitation='Archived development state; neural proposal history reset. Geometry proposals unchanged.'))
    started=time.perf_counter()
    _,pairs=search(engine,route,dynamics,np.asarray(initial['raw_observation'],np.float32),gem[:3],nav,40,brake_replies=48)
    route.navigate(dynamics['native']['state']['position'],gem[:3])
    checked=[];details=[]
    for info,rollout in pairs:
        if info['rejected'] or rollout is None:continue
        continuation=predict_replies(engine,dynamics,BrakeContinuation(rollout['proposed'],gem[:3],a.braking),stop_on_ambiguity=True)
        stop=assess(engine,route,dynamics,gem[:3],continuation,info['label']+'_stop')
        clone=dict(info)
        if stop['rejected']:clone['rejected']='no_stopping_continuation:'+stop['rejected']
        clone['stopping_check']=stop
        checked.append((clone,rollout));details.append(clone)
        if info['label']=='blast_0_jump_-1':
            atomic_json(a.output/'original_blast_forecast.json',rollout)
            atomic_json(a.output/'original_blast_braking_forecast.json',continuation)
    selected=useful_candidate(checked,route,route.distance(dynamics['native']['state']['position']),True)
    atomic_json(a.output/'report.json',dict(map=name,decision=503,elapsed_s=time.perf_counter()-started,
        original_selected_label=initial['chosen']['label'],candidates=details,
        rejections=dict(Counter(str(i['rejected']) for i,_ in checked)),
        selected=selected[0] if selected else None,checkpoint_unchanged=sha(checkpoint)==fingerprint,training=False))
    if selected:atomic_json(a.output/'selected_forecast.json',selected[1])
    print(json.dumps(dict(selected=selected[0] if selected else None,rejections=dict(Counter(str(i['rejected']) for i,_ in checked)))),flush=True)


if __name__=='__main__':main()
