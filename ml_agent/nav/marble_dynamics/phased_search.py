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



def perturbations(actions, samples=12):
    """Deterministic local sequence search: steering phases and event times.

    Every proposal uses legal inputs. No map labels, recorded future states or
    special-case maneuvers influence the perturbations. Physics judges outcomes.
    """
    base=np.asarray(actions,dtype=float)
    if len(base)<4:return
    variants=[]
    for start,angle in ((0,-.25),(0,.25),(len(base)//3,-.5),(len(base)//3,.5),
                        (2*len(base)//3,-.5),(2*len(base)//3,.5)):
        trial=base.copy();end=min(len(base),start+max(4,len(base)//3))
        xy=np.column_stack((trial[start:end,3]-trial[start:end,2],trial[start:end,0]-trial[start:end,1]))
        co,si=math.cos(angle),math.sin(angle)
        rotated=np.column_stack((xy[:,0]*co-xy[:,1]*si,xy[:,0]*si+xy[:,1]*co))
        rotated/=np.maximum(1.,np.max(np.abs(rotated),axis=1))[:,None]
        trial[start:end,0]=np.maximum(rotated[:,1],0);trial[start:end,1]=np.maximum(-rotated[:,1],0)
        trial[start:end,2]=np.maximum(-rotated[:,0],0);trial[start:end,3]=np.maximum(rotated[:,0],0)
        variants.append(('turn_%d_%g'%(start,angle),trial))
    for column,name in ((4,'jump'),(6,'use'),(8,'blast')):
        pressed=np.flatnonzero(base[:,column])
        for shift in (-2,2):
            trial=base.copy();trial[:,column]=0
            if len(pressed):trial[np.clip(pressed+shift,0,len(base)-1),column]=1
            elif column==4:trial[max(0,len(base)//3+shift),column]=1
            else:continue
            variants.append((name+'_'+str(shift),trial))
    for label,trial in variants[:samples]:
        yield label,[[int(x) if j in (4,6,8) else round(float(x),6) for j,x in enumerate(row)] for row in trial]


def refine(engine,route,dynamics,gem,pairs,target_item=None,samples=12,failure_only=True):
    """One bounded local round around the best complete seed; keep normal gates."""
    from nav.marble_dynamics.planning_budget import PlanningBudgetExceeded,check
    if failure_only and useful_candidate(pairs,route,route.distance(dynamics['native']['state']['position'])) is not None:
        return []
    candidates=[p for p in pairs if p[1] is not None and not p[1]['failure']
                and len(p[1]['controls'])==len(p[1]['proposed']) and math.isfinite(p[0]['cost'])]
    if not candidates:return []
    info,seed=min(candidates,key=lambda p:(p[0]['rejected'] is not None,p[0]['cost']))
    result=[]
    for label,actions in perturbations(seed['proposed'],samples):
        try:
            check(engine)
            roll=predict_replies(engine,dynamics,actions,stop_on_ambiguity=True)
            detail=assess(engine,route,dynamics,gem,roll,'refine_'+label,target_item)
            detail['refinement_parent']=info['label']
            detail.update(evaluated_replies=len(roll['controls']),requested_replies=roll['requested_replies'],
                          stopped_on_ambiguity=roll['stopped_on_ambiguity'])
            result.append((detail,roll))
        except PlanningBudgetExceeded:break
        except ValueError:continue
    return result


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
