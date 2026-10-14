"""Predict deceleration before choosing a maneuver, without a state reset.

The previous search brakes only after an approach. In an airborne/high-momentum
state, approaching immediately can eliminate every stable continuation. Test an
ordinary braking prefix followed by the same general goal/event controls.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from nav.marble_dynamics.bridge import predict_replies
from nav.marble_dynamics.blast import available as blast_available
from nav.marble_dynamics.gem_demo import Proposal
from nav.marble_dynamics.search import SequenceProposal,assess,useful_candidate


class SettleProposal:
    def __init__(self,route,gem,prefix,horizon,jump=-1,use=-1,blast=-1,direct=False):
        self.gem=gem;self.prefix=prefix;self.steps=prefix+horizon
        self.after=SequenceProposal(route,gem,horizon,jump=jump,use=use,blast=blast,direct=direct)

    def __call__(self,state,index):
        if index<self.prefix:return Proposal(self.gem,brake=True)(state,0)
        return self.after(state,index-self.prefix)


def choose_settled(engine,route,dynamics,gem,horizon=80,preserve_resources=True):
    gem=np.asarray(gem[:3]);route.navigate(dynamics['native']['state']['position'],gem)
    start=route.distance(dynamics['native']['state']['position']);held=int(dynamics['native']['state']['held'])
    steps=max(96,horizon) if held==5 or dynamics['native']['state']['effects'][5][0] else horizon
    events=[(-1,-1,-1),(0,-1,-1)]
    if 1<=held<=5:events.append((0,3,-1))
    if blast_available(dynamics.get('blast')):events.append((0,-1,3))
    pairs=[]
    for prefix in (8,16,24,32):
        for jump,use,blast in events:
            for direct in (False,True):
                label=('direct_' if direct else '')+'settle_'+str(prefix)+'_jump_'+str(jump)+'_use_'+str(use)+'_blast_'+str(blast)
                proposal=SettleProposal(route,gem,prefix,steps,jump,use,blast,direct)
                rollout=predict_replies(engine,dynamics,proposal,stop_on_ambiguity=True)
                info=assess(engine,route,dynamics,gem,rollout,label)
                info.update(braking_prefix_replies=prefix,evaluated_replies=len(rollout['controls']),
                    requested_replies=proposal.steps,stopped_on_ambiguity=rollout['stopped_on_ambiguity'])
                pairs.append((info,rollout))
    return useful_candidate(pairs,route,start,preserve_resources),pairs


def main():
    from nav.collision.mesh import CollisionMap,atomic_json
    from nav.marble_dynamics.record import ML
    from nav.marble_dynamics.physics import GeneralPhysics
    from nav.marble_dynamics.geometry_route import walking_terrain
    from nav.routing import RoutePlanner
    from nav.terrain import TerrainGrid,TerrainMap
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    x=json.loads(a.source.read_text());name=a.source.parents[2].name
    engine=GeneralPhysics(CollisionMap(ML/'collision_maps'/(name+'.npz')))
    route=RoutePlanner(walking_terrain(TerrainGrid(TerrainMap.resolve(name)),engine.mesh))
    chosen,pairs=choose_settled(engine,route,x['dynamics'],x['target'])
    result=dict(source=str(a.source),chosen=chosen[0] if chosen else None,candidates=[i for i,r in pairs],training=False)
    atomic_json(a.output/'report.json',result)
    if chosen:atomic_json(a.output/'selected_forecast.json',chosen[1])
    print(json.dumps(dict(chosen=result['chosen'],rejections={str(k):sum(i['rejected']==k for i,r in pairs) for k in {i['rejected'] for i,r in pairs}})),flush=True)


if __name__=='__main__':main()
