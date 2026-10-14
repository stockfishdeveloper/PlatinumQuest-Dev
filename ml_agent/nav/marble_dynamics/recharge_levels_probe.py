"""Check useful partial-charge waits with actual meter dynamics and full physics.

Charge levels propose WAIT DURATION only; neither meter nor pose is injected.
Every wait/action/landing is one continuous forecast from an archived real state.
This diagnostic optimizes legal controls, never weights, and claims no native gem.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import math
import time

import numpy as np
from nav.collision.mesh import atomic_json
from nav.collision.experimental_sweep import CachedCompiledSweepMap
from nav.marble_dynamics.record import ML,sha
from nav.marble_dynamics.physics import GeneralPhysics
from nav.marble_dynamics.geometry_route import walking_terrain
from nav.marble_dynamics.ground_cache import MemoGroundClearance
from nav.marble_dynamics.recharge import RechargeProposal
from nav.marble_dynamics.bridge import predict_replies
from nav.marble_dynamics.search import assess
from nav.routing import RoutePlanner
from nav.terrain import TerrainGrid,TerrainMap


def search_levels(engine,route,dynamics,gem,remaining_replies,round_ms):
    context=dynamics.get('blast');pairs=[]
    if not context or not context['enabled'] or not context['charging'] or context['frozen']:
        return None,pairs
    if any(context[k] for k in ('special','triple','fireball','cannon')):return None,pairs
    seen=set();gem=np.asarray(gem[:3]);route.navigate(dynamics['native']['state']['position'],gem)
    for level in [context['required'],.35,.5,.75,1.]:
        threshold=max(context['required'],level)
        if threshold<=context['value']:continue
        wait=max(1,int(math.ceil((threshold-context['value'])*context['charge_ms']/64.))+2)
        horizon=min(80,remaining_replies-wait)
        if wait>400 or horizon<16 or (wait+horizon)*64>=round_ms or wait in seen:continue
        seen.add(wait)
        for blast,jump in ((0,-1),(3,-1),(3,0),(7,3),(11,5)):
            for direct in (False,True):
                label=('direct_' if direct else '')+'charge_'+str(threshold)+'_wait_'+str(wait)+'_blast_'+str(blast)+'_jump_'+str(jump)
                proposal=RechargeProposal(route,gem,wait,horizon,blast,jump,direct)
                forecast=predict_replies(engine,dynamics,proposal,stop_on_ambiguity=True)
                info=assess(engine,route,dynamics,gem,forecast,label)
                info.update(charge_threshold=threshold,waiting_replies=wait,horizon=horizon,
                    requested_replies=proposal.steps,evaluated_replies=len(forecast['controls']))
                pairs.append((info,forecast))
    valid=[p for p in pairs if p[0]['rejected'] is None and p[0]['possible_pickup_interval'] is not None]
    return min(valid,key=lambda p:p[0]['cost']) if valid else None,pairs


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--map',required=True);p.add_argument('--remaining-replies',type=int,required=True)
    p.add_argument('--physics-frame-ms',type=int,choices=[16,64],required=True)
    p.add_argument('--dll',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);x=json.loads(a.source.read_text())
    sources=list(Path(__file__).parent.glob('*.py'))+[a.dll,ML/'nav/collision/experimental_sweep.py']
    atomic_json(a.output/'prepared.json',dict(source=str(a.source),source_sha256=sha(a.source),map=a.map,
        remaining_replies=a.remaining_replies,physics_frame_ms=a.physics_frame_ms,
        charge_levels=['observed required',.35,.5,.75,1.],source_hashes={str(path):sha(path) for path in sources},
        training=False,scored_navigation=False,contract='Only wait duration/control proposals. Actual observed meter/pose, continuous physics, all existing support/ambiguity gates.'))
    engine=GeneralPhysics(CachedCompiledSweepMap(ML/'collision_maps'/(a.map+'.npz'),a.dll));engine.bridge_physics_ms=a.physics_frame_ms
    terrain=walking_terrain(TerrainGrid(TerrainMap.resolve(a.map)),engine.mesh);terrain.ground_clearance=MemoGroundClearance(terrain.ground_clearance)
    route=RoutePlanner(terrain);start=time.perf_counter()
    selected,pairs=search_levels(engine,route,x['dynamics'],x['target'],a.remaining_replies,float(x['raw_observation'][31]))
    result=dict(selected=selected[0] if selected else None,candidates=[i for i,r in pairs],
        rejections=dict(Counter(str(i['rejected']) for i,r in pairs)),elapsed_s=time.perf_counter()-start,
        native_success=False,training=False)
    atomic_json(a.output/'report.json',result)
    if selected:atomic_json(a.output/'selected_forecast.json',selected[1])
    print(json.dumps({k:v for k,v in result.items() if k!='candidates'}),flush=True)


if __name__=='__main__':main()
