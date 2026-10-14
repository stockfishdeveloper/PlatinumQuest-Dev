"""General phased control search and persistent discrete-action commitment."""
import copy
import math

import numpy as np

from nav.marble_dynamics.bridge import predict_replies
from nav.marble_dynamics.gem_demo import Proposal, candidate_cost
from nav.marble_dynamics.navigator import NavigatorProposal
from nav.gems import visible_gems
from nav.routing import snap_to_walkable


class SequenceProposal:
    def __init__(self,route,gem,steps,offset=0.,scale=1.,jump=-1,use=-1,turn_after=6,brake_after=None,direct=False):
        self.route=copy.copy(route);self.gem=np.asarray(gem);self.steps=steps
        self.offset=offset;self.scale=scale;self.jump=jump;self.use=use
        self.turn_after=turn_after;self.brake_after=brake_after;self.direct=direct

    def __call__(self,state,index):
        goal=self.gem if self.direct else self.route.navigate(state['position'],self.gem)[0]
        # Phases vary steering, throttle, airborne controls and braking. They
        # describe legal controls, never a named trick or a map-specific launch.
        proposal=Proposal(goal,self.offset if index<self.turn_after else 0.,
                          jump_at=self.jump-index,use_at=self.use-index,
                          brake=self.brake_after is not None and index>=self.brake_after)
        action=proposal(state,0)
        action[:4]=[round(v*self.scale,6) for v in action[:4]]
        return action


class BrakeContinuation:
    """Keep a predicted control prefix, then predict ordinary feedback braking."""
    def __init__(self,prefix,gem,extra_replies):
        self.prefix=prefix;self.gem=gem;self.steps=len(prefix)+extra_replies

    def __call__(self,state,index):
        if index<len(self.prefix):return self.prefix[index]
        return Proposal(self.gem,brake=True)(state,0)


def terminal_support(mesh,tail):
    """Measure settled motion relative to the contacted surface, including slopes.

    World vertical velocity is nonzero during ordinary downhill rolling. Keep
    four supported states, the same separation/speed limits and upward normals;
    test bounce/separation along each actual surface normal instead.
    """
    speeds=[]
    for state in tail:
        p=np.asarray(state['position']);r=state['radius']
        hit=mesh.raycast(p+[0,0,.05],p-[0,0,r+.10],cull_backfaces=True)
        if hit is None or hit['normal'][2]<=.4:return False,False,None
        n=np.asarray(hit['normal']);n=n/np.linalg.norm(n)
        speeds.append(abs(float(np.asarray(state['velocity'])@n)))
    supported=len(tail)>=4
    stable=supported and all(s<2. for s in speeds) and all(np.linalg.norm(s['velocity'])<18. for s in tail)
    return supported,stable,max(speeds,default=None)


def assess(engine,route,dynamics,gem,rollout,label):
    states=rollout['states'];end=states[-1]
    item=next((i for i in dynamics.get('items',[]) if i['gem'] and np.linalg.norm(np.asarray(i['position'])-gem)<.1),None)
    routed=True;cost,close,pickup=candidate_cost(route,states,gem,routed,item)
    tail=states[-4:]
    supported,stable,normal_speed=terminal_support(engine.mesh,tail)
    reason=None
    if rollout['failure']:reason='prediction_failure'
    elif rollout['ambiguous_intervals']:reason='contact_ambiguity'
    elif not stable:reason='no_supported_continuation'
    if not math.isfinite(cost):reason='nonfinite_goal_cost'
    return dict(label=label,cost=cost,rejected=reason,minimum_gem_distance_u=close,
                end_position=end['position'],terminal_support=supported,stable_continuation=stable,
                maximum_tail_normal_speed_u_per_s=normal_speed,
                continuation_s=min(3,len(states)-1)*.064,possible_pickup_interval=pickup,
                ambiguous_intervals=rollout['ambiguous_intervals'],failure=rollout['failure'])


def search(engine,route,dynamics,raw,gem,navigator,horizon=40,warm=None,brake_replies=0):
    gem=np.asarray(gem[:3]);route.navigate(dynamics['native']['state']['position'],gem)
    held=int(dynamics['native']['state']['held'])
    steps=max(horizon,96) if held==5 or dynamics['native']['state']['effects'][5][0] else horizon
    specs=[('route',SequenceProposal(route,gem,steps)),('direct',SequenceProposal(route,gem,steps,direct=True))]
    for scale in (.4,.7):specs.append(('throttle_'+str(scale),SequenceProposal(route,gem,steps,scale=scale)))
    for jump in (0,2,5,9):
        specs.append(('jump_'+str(jump),SequenceProposal(route,gem,steps,jump=jump)))
        specs.append(('direct_jump_'+str(jump),SequenceProposal(route,gem,steps,jump=jump,direct=True)))
    for offset in (-1.,-.5,.5,1.):
        specs.append(('steering_'+str(offset),SequenceProposal(route,gem,steps,offset=offset,turn_after=12)))
    specs.append(('accelerate_then_brake',SequenceProposal(route,gem,steps,brake_after=12)))
    if 1<=held<=5:
        for use,jump in ((0,-1),(3,-1),(7,-1),(3,2),(7,5)):
            specs.append(('use_'+str(use)+'_jump_'+str(jump),SequenceProposal(route,gem,steps,use=use,jump=jump)))
            specs.append(('direct_use_'+str(use)+'_jump_'+str(jump),SequenceProposal(route,gem,steps,use=use,jump=jump,direct=True)))
    other_gems=[g for g in visible_gems(raw) if np.linalg.norm(np.asarray(g[:3])-gem)>.1]
    nxt=min(other_gems,key=lambda g:np.linalg.norm(np.asarray(g[:3])-gem)) if other_gems else None
    next_goal=snap_to_walkable(route.terrain,*nxt[:3])[0] if nxt else None
    specs.append(('frozen_navigator',NavigatorProposal(navigator,raw,route,gem,next_goal,steps)))
    pairs=[]
    for label,proposal in specs:
        try:rollout=predict_replies(engine,dynamics,proposal)
        except ValueError as exc:
            pairs.append((dict(label=label,rejected='unsupported_proposal',failure=str(exc),cost=math.inf),None));continue
        info=assess(engine,route,dynamics,gem,rollout,label)
        pairs.append((info,rollout))
        if brake_replies and info['rejected']=='no_supported_continuation':
            extended=predict_replies(engine,dynamics,BrakeContinuation(rollout['proposed'],gem,brake_replies))
            checked=assess(engine,route,dynamics,gem,extended,label+'_brake_tail')
            checked.update(proposal_replies=len(rollout['proposed']),braking_replies=brake_replies,
                           forecast_s=(len(extended['states'])-1)*.064)
            pairs.append((checked,extended))
    if warm:
        rollout=predict_replies(engine,dynamics,warm)
        pairs.append((assess(engine,route,dynamics,gem,rollout,'remaining_previous_plan'),rollout))
    valid=[p for p in pairs if p[0]['rejected'] is None]
    return min(valid,key=lambda p:p[0]['cost']) if valid else None,pairs


def retained_candidate(engine,route,dynamics,gem,memory,minimum_replies=8):
    """Revalidate a useful remaining plan before spending a full search budget.

    The same visible target must remain selected. Recompute from the NEW observed
    state; changed inventory/geometry events clear memory in the execution loop.
    A short, unsafe, ambiguous or no-progress remainder goes back to full search.
    Keep the absolute action schedule rather than installing the remainder again.
    """
    if (len(memory.remaining)<minimum_replies or memory.gem is None
            or np.linalg.norm(np.asarray(memory.gem)-gem[:3])>=.1):return None
    position=dynamics['native']['state']['position'];route.navigate(position,gem[:3])
    start=route.distance(position);forecast=predict_replies(engine,dynamics,memory.remaining)
    info=assess(engine,route,dynamics,np.asarray(gem[:3]),forecast,memory.label)
    if info['rejected']:return None
    progress=start-route.distance(forecast['states'][-1]['position'])
    if info['possible_pickup_interval'] is None and progress<=.3:return None
    info['revalidated_remainder']=True;info['predicted_progress_u']=progress
    return info,forecast


class PlanMemory:
    """Maintain an absolute schedule until its first jump/use reaches physics.

    A pickup, changed world or prediction error can always cancel it. Ordinary
    four-reply replanning cannot shift an already chosen event into the future.
    """
    def __init__(self):self.clear()

    def clear(self):self.actions=[];self.executed=0;self.lock_until=0;self.label=None;self.gem=None

    def accept(self,rollout,label,gem):
        self.actions=copy.deepcopy(rollout['proposed']);self.executed=0;self.label=label;self.gem=list(gem)
        events=[i for i,a in enumerate(self.actions[:13]) if a[4] or a[6]]
        self.lock_until=min(len(self.actions),events[0]+3) if events else 0

    @property
    def locked(self):return self.executed<self.lock_until and bool(self.remaining)

    @property
    def remaining(self):return self.actions[self.executed:]

    def advance(self):self.executed+=1
