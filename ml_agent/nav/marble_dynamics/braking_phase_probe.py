"""Test early ordinary braking as a general maneuver phase, not a ledge rule."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

from nav.collision.experimental_sweep import CachedCompiledSweepMap
from nav.collision.mesh import atomic_json
from nav.marble_dynamics.blast import available
from nav.marble_dynamics.bridge import predict_replies
from nav.marble_dynamics.geometry_route import walking_terrain
from nav.marble_dynamics.global_goals import observed_gems
from nav.marble_dynamics.ground_cache import MemoGroundClearance
from nav.marble_dynamics.physics import GeneralPhysics
from nav.marble_dynamics.record import ML,sha
from nav.marble_dynamics.search import SequenceProposal,assess,useful_candidate
from nav.routing import RoutePlanner
from nav.terrain import TerrainGrid,TerrainMap


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dll',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    source=ML/'dynamics_reports/reliability_landing_validation_20261012/CubeIsle_Hunt/151113/physics_plan/decisions.json'
    initial=json.loads(source.read_text())[503];dynamics=initial['dynamics'];raw=np.asarray(initial['raw_observation'],np.float32)
    mesh=CachedCompiledSweepMap(ML/'collision_maps/CubeIsle_Hunt.npz',a.dll);engine=GeneralPhysics(mesh)
    engine.terminal_stop_replies=48;engine.terminal_stop_scope='pickup'
    terrain=TerrainGrid(TerrainMap.resolve('CubeIsle_Hunt'));geometric=walking_terrain(terrain,mesh)
    geometric.ground_clearance=MemoGroundClearance(geometric.ground_clearance);route=RoutePlanner(geometric)
    goals=observed_gems(dynamics,raw);blasts=[-1,0,7] if available(dynamics.get('blast')) else [-1]
    sources=list(Path(__file__).parent.glob('*.py'))+[a.dll]
    atomic_json(a.output/'prepared.json',dict(map='CubeIsle_Hunt',decision=503,source=str(source),source_sha256=sha(source),
        source_hashes={str(p):sha(p) for p in sources},horizon=80,braking=48,scope='pickup',
        brake_after=[16,24,32,40,48],jump=[-1,0,5],blast=blasts,scale=[.4,.7,1.],
        direct=[False,True],goals=goals,training=False,
        limitation='Fixed general phase grid at an archived development state; no native collection claim.'))
    results=[];started=time.perf_counter();selected=None
    for gem in goals:
        route.navigate(dynamics['native']['state']['position'],gem[:3]);pairs=[]
        for brake in [16,24,32,40,48]:
            for jump in [-1,0,5]:
                for blast in blasts:
                    for scale in [.4,.7,1.]:
                        for direct in [False,True]:
                            spec=dict(brake_after=brake,jump=jump,blast=blast,scale=scale,direct=direct)
                            label='early_brake_'+str(len(pairs))
                            rollout=predict_replies(engine,dynamics,SequenceProposal(route,gem[:3],80,**spec),stop_on_ambiguity=True)
                            info=assess(engine,route,dynamics,gem[:3],rollout,label);info['specification']=spec
                            pairs.append((info,rollout))
        # This diagnosis specifically seeks collection plus its verified backup,
        # keeping mere progress distinct from a completed pickup forecast.
        eligible=[pair for pair in pairs if pair[0]['possible_pickup_interval'] is not None]
        choice=useful_candidate(eligible,route,route.distance(dynamics['native']['state']['position']),True)
        entry=dict(goal=gem,selected=choice[0] if choice else None,candidates=[i for i,_ in pairs],
            rejections=dict(Counter(str(info['rejected']) for info,_ in pairs)))
        results.append(entry);atomic_json(a.output/'progress.json',dict(attempts=results,training=False))
        print(json.dumps({k:entry[k] for k in ['goal','selected','rejections']}),flush=True)
        if choice:
            selected=dict(goal=gem,info=choice[0]);atomic_json(a.output/'selected_forecast.json',choice[1]);break
    atomic_json(a.output/'report.json',dict(attempts=results,selected=selected,elapsed_s=time.perf_counter()-started,training=False))


if __name__=='__main__':main()
