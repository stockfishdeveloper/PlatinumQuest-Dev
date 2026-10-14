"""Check shorter receding-horizon alternatives at archived failed states.

Keep every physics, ambiguity, support and useful-progress check. A shorter
prediction is a smaller explicitly checked plan, not approval of an uncertain
long future. No execution, future observations or fitting.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

from nav.collision.experimental_sweep import CachedCompiledSweepMap
from nav.collision.mesh import atomic_json
from nav.marble_dynamics.geometry_route import walking_terrain
from nav.marble_dynamics.ground_cache import MemoGroundClearance
from nav.marble_dynamics.navigator import FrozenNavigator
from nav.marble_dynamics.physics import GeneralPhysics
from nav.marble_dynamics.record import ML,sha
from nav.marble_dynamics.search import search,useful_candidate
from nav.routing import RoutePlanner
from nav.terrain import TerrainGrid,TerrainMap


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dll',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    sources=[('reliability_global_treasure_20261012','TreasureBox_Hunt',111108),
             ('reliability_landing_validation_20261012','Concentric_Hunt',151113),
             ('reliability_landing_validation_20261012','CubeIsle_Hunt',151113),
             ('reliability_geometry_pool_20261012','FracturedIslands_Hunt',111108)]
    checkpoint=ML/'models/nav/real_maps_20261010_A2/nav_latest.pth';fingerprint=sha(checkpoint)
    atomic_json(a.output/'prepared.json',dict(cases=sources,horizons=[4,8,12,20,28],braking_replies=48,
        checkpoint_sha256=fingerprint,dll_sha256=sha(a.dll),source_sha256=sha(__file__),training=False,
        limitation='Archived development states. Frozen navigator proposal recurrent history reset; geometric proposals independent of that history.'))
    results=[]
    for directory,name,seed in sources:
        path=ML/'dynamics_reports'/directory/name/str(seed)/'physics_plan/terminal_search.json'
        terminal=json.loads(path.read_text());dynamics=terminal['dynamics'];raw=np.asarray(terminal['raw_observation'],dtype=np.float32)
        mesh=CachedCompiledSweepMap(ML/'collision_maps'/(name+'.npz'),a.dll);engine=GeneralPhysics(mesh)
        terrain=TerrainGrid(TerrainMap.resolve(name));geometric=walking_terrain(terrain,mesh)
        geometric.ground_clearance=MemoGroundClearance(geometric.ground_clearance)
        route=RoutePlanner(geometric);goal=terminal['target'];route.navigate(dynamics['native']['state']['position'],goal[:3])
        nav=FrozenNavigator(terrain,checkpoint);cases=[]
        for horizon in [4,8,12,20,28]:
            started=time.perf_counter();_,pairs=search(engine,route,dynamics,raw,goal[:3],nav,horizon,brake_replies=48)
            start=route.distance(dynamics['native']['state']['position']);chosen=useful_candidate(pairs,route,start,True)
            events=Counter();ambiguous=[]
            for info,rollout in pairs:
                if rollout and rollout['ambiguous_intervals']:
                    interval=rollout['ambiguous_intervals'][0]
                    for event in rollout['events'][interval]:
                        if 'ambiguous' in event.get('kind',''):
                            events[event['kind']]+=1
                            if len(ambiguous)<5:ambiguous.append(dict(label=info['label'],interval=interval,event=event))
            case=dict(horizon=horizon,selected=chosen[0] if chosen else None,
                route_progress_u=start-route.distance(chosen[1]['states'][-1]['position']) if chosen else None,
                rejections=dict(Counter(str(info.get('rejected')) for info,_ in pairs)),
                ambiguity_kinds=dict(events),example_ambiguities=ambiguous,elapsed_s=time.perf_counter()-started)
            cases.append(case)
            if chosen:atomic_json(a.output/(name+'.h'+str(horizon)+'.forecast.json'),chosen[1])
        result=dict(map=name,source=str(path),source_sha256=sha(path),cases=cases)
        results.append(result);atomic_json(a.output/(name+'.json'),result)
        atomic_json(a.output/'summary.json',dict(cases=results,training=False,checkpoint_unchanged=sha(checkpoint)==fingerprint))
        print(json.dumps(dict(map=name,selected=[dict(horizon=c['horizon'],label=c['selected']['label'],progress=c['route_progress_u']) for c in cases if c['selected']],
            ambiguity_kinds=cases[-1]['ambiguity_kinds'])),flush=True)
    assert sha(checkpoint)==fingerprint


if __name__=='__main__':main()
