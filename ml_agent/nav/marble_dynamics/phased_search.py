"""Bounded general action exploration after the small ordinary proposal set fails.

Sample steering direction, phase length, throttle and discrete event times. The
full collision/physics model checks each future; no map/trick names or future
measurements determine the sampled controls. This optimizes actions, not weights.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from nav.marble_dynamics.bridge import predict_replies
from nav.marble_dynamics.blast import available as blast_available
from nav.marble_dynamics.search import SequenceProposal,BrakeContinuation,assess,useful_candidate


class AimProposal(SequenceProposal):
    def __init__(self,*args,aim_offset=0.,**kwargs):
        super().__init__(*args,**kwargs);self.aim_offset=aim_offset

    def __call__(self,state,index):
        action=super().__call__(state,index)
        if action[6]:action[7]-=self.aim_offset
        return action


def explore(engine,route,dynamics,gem,horizon=80,samples=40,seed=997,brake_replies=48,preserve_resources=True):
    gem=np.asarray(gem[:3]);state=dynamics['native']['state'];held=int(state['held'])
    route.navigate(state['position'],gem);start=route.distance(state['position'])
    rng=np.random.RandomState(seed);pairs=[]
    for i in range(samples):
        # Broad steering includes away from the current goal. A jump/use time
        # is independent of map landmarks and can precede or follow a turn.
        offset=float(rng.uniform(-math.pi,math.pi));turn=int(rng.randint(4,33))
        jump=int(rng.choice([-1,0,4,8,12,16,20,24,32]));use=-1;blast=-1
        if 1<=held<=5 and rng.rand()<.65:use=int(rng.randint(0,33))
        if blast_available(dynamics.get('blast')) and rng.rand()<.75:blast=int(rng.randint(0,37))
        steps=max(horizon,96) if state['effects'][5][0] or (held==5 and use>=0) else horizon
        specification=dict(offset=offset,turn_after=turn,scale=float(rng.choice([.4,.7,1.])),
            jump=jump,use=use,blast=blast,direct=bool(rng.randint(2)),aim_offset=float(rng.uniform(-math.pi,math.pi)))
        proposal=AimProposal(route,gem,steps,**specification);label='sampled_phase_'+str(seed)+'_'+str(i)
        rollout=predict_replies(engine,dynamics,proposal,stop_on_ambiguity=True)
        info=assess(engine,route,dynamics,gem,rollout,label)
        info.update(specification=specification,proposal_seed=seed,evaluated_replies=len(rollout['controls']),
            requested_replies=rollout['requested_replies'],stopped_on_ambiguity=rollout['stopped_on_ambiguity'])
        pairs.append((info,rollout))
        if brake_replies and info['rejected']=='no_supported_continuation':
            continued=predict_replies(engine,dynamics,BrakeContinuation(rollout['proposed'],gem,brake_replies),stop_on_ambiguity=True)
            checked=assess(engine,route,dynamics,gem,continued,label+'_brake_tail')
            checked.update(specification=specification,proposal_seed=seed,evaluated_replies=len(continued['controls']),
                requested_replies=continued['requested_replies'],stopped_on_ambiguity=continued['stopped_on_ambiguity'])
            pairs.append((checked,continued))
    return useful_candidate(pairs,route,start,preserve_resources),pairs


def main():
    from nav.collision.mesh import CollisionMap,atomic_json
    from nav.marble_dynamics.record import ML
    from nav.marble_dynamics.physics import GeneralPhysics
    from nav.marble_dynamics.geometry_route import walking_terrain
    from nav.routing import RoutePlanner
    from nav.terrain import TerrainGrid,TerrainMap
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--samples',type=int,default=40)
    p.add_argument('--seed',type=int,default=997);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    x=json.loads(a.source.read_text());name=a.source.parents[2].name
    engine=GeneralPhysics(CollisionMap(ML/'collision_maps'/(name+'.npz')))
    route=RoutePlanner(walking_terrain(TerrainGrid(TerrainMap.resolve(name)),engine.mesh))
    chosen,pairs=explore(engine,route,x['dynamics'],x['target'],samples=a.samples,seed=a.seed)
    result=dict(source=str(a.source),proposal_seed=a.seed,samples=a.samples,chosen=chosen[0] if chosen else None,
        candidates=[i for i,r in pairs],training=False)
    atomic_json(a.output/'report.json',result)
    if chosen:atomic_json(a.output/'selected_forecast.json',chosen[1])
    print(json.dumps(dict(chosen=result['chosen'],samples=a.samples,rejections={str(k):sum(i['rejected']==k for i,r in pairs) for k in {i['rejected'] for i,r in pairs}})),flush=True)


if __name__=='__main__':main()
