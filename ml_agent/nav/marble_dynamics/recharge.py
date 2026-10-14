"""Physically checked waiting for an existing regular Blast meter to recharge.

Only a fallback when ordinary gem/resource planning has no useful future. The
entire brake/wait/action/landing sequence is predicted from the current real
snapshot; no full meter or favorable pose is injected into a hypothesis.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from nav.marble_dynamics.bridge import predict_replies
from nav.marble_dynamics.gem_demo import Proposal
from nav.marble_dynamics.search import SequenceProposal,assess


class RechargeProposal:
    def __init__(self,route,gem,wait_replies,horizon=80,blast=3,jump=0,direct=False):
        self.gem=gem;self.wait_replies=wait_replies;self.steps=wait_replies+horizon
        self.after=SequenceProposal(route,gem,horizon,blast=blast,jump=jump,direct=direct)

    def __call__(self,state,index):
        if index<self.wait_replies:return Proposal(self.gem,brake=True)(state,0)
        return self.after(state,index-self.wait_replies)


def choose_recharge(engine,route,dynamics,gem,remaining_replies,remaining_round_ms,horizon=80):
    context=dynamics.get('blast');pairs=[]
    if not context or not context['enabled'] or not context['charging'] or context['frozen']:
        return None,pairs
    if any(context[k] for k in ('special','triple','fireball','cannon')) or context['value']>=1.:
        return None,pairs
    wait=max(1,int(math.ceil((1.-context['value'])*context['charge_ms']/64.))+2)
    if wait>400 or wait+horizon>remaining_replies or (wait+horizon)*64>=remaining_round_ms:
        return None,pairs
    gem=np.asarray(gem[:3]);route.navigate(dynamics['native']['state']['position'],gem)
    for blast,jump in ((3,0),(7,3),(11,5)):
        for direct in (False,True):
            label=('direct_' if direct else '')+'recharge_'+str(wait)+'_blast_'+str(blast)+'_jump_'+str(jump)
            proposal=RechargeProposal(route,gem,wait,horizon,blast,jump,direct)
            rollout=predict_replies(engine,dynamics,proposal,stop_on_ambiguity=True)
            info=assess(engine,route,dynamics,gem,rollout,label)
            info.update(waiting_replies=wait,requested_replies=proposal.steps,evaluated_replies=len(rollout['controls']),
                stopped_on_ambiguity=rollout['stopped_on_ambiguity'],
                contract='Actual initial meter/pose; brake/wait/charge/action/landing predicted continuously.')
            pairs.append((info,rollout))
    valid=[p for p in pairs if p[0]['rejected'] is None and p[0]['possible_pickup_interval'] is not None]
    return min(valid,key=lambda p:p[0]['cost']) if valid else None,pairs


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
    chosen,pairs=choose_recharge(engine,route,x['dynamics'],x['target'],1000,float(x['raw_observation'][31]))
    result=dict(source=str(a.source),chosen=chosen[0] if chosen else None,candidates=[i for i,r in pairs],training=False)
    atomic_json(a.output/'report.json',result)
    if chosen:atomic_json(a.output/'selected_forecast.json',chosen[1])
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
