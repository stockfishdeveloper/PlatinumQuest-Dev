"""General phased control search and persistent discrete-action commitment."""
import copy
import math

import numpy as np

from nav.marble_dynamics.bridge import predict_replies
from nav.marble_dynamics.gem_demo import Proposal, candidate_cost
from nav.marble_dynamics.navigator import NavigatorProposal
from nav.gems import visible_gems
from nav.routing import snap_to_walkable
from nav.marble_dynamics.blast import available as blast_available
from nav.marble_dynamics.planning_budget import PlanningBudgetExceeded


class SequenceProposal:
    def __init__(self,route,gem,steps,offset=0.,scale=1.,jump=-1,use=-1,turn_after=6,brake_after=None,direct=False,blast=-1):
        self.route=copy.copy(route);self.gem=np.asarray(gem);self.steps=steps
        self.offset=offset;self.scale=scale;self.jump=jump;self.use=use
        self.turn_after=turn_after;self.brake_after=brake_after;self.direct=direct
        self.blast=blast

    def __call__(self,state,index):
        goal=self.gem if self.direct else self.route.navigate(state['position'],self.gem)[0]
        # Phases vary steering, throttle, airborne controls and braking. They
        # describe legal controls, never a named trick or a map-specific launch.
        proposal=Proposal(goal,self.offset if index<self.turn_after else 0.,
                          jump_at=self.jump-index,use_at=self.use-index,
                          brake=self.brake_after is not None and index>=self.brake_after)
        action=proposal(state,0)
        action[:4]=[round(v*self.scale,6) for v in action[:4]]
        action[8]=int(index==self.blast)
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


def assess(engine,route,dynamics,gem,rollout,label,target_item=None,routed=True,_stop_checked=False):
    states=rollout['states'];end=states[-1]
    gem=np.asarray(gem[:3])
    item=None
    if routed:
        item=target_item or next((i for i in dynamics.get('items',[]) if i['gem'] and np.linalg.norm(np.asarray(i['position'])-gem)<.1),None)
    cost,close,pickup=candidate_cost(route,states,gem,routed,item)
    tail=states[-4:]
    supported,stable,normal_speed=terminal_support(engine.mesh,tail)
    reason=None
    if rollout['failure']:reason='prediction_failure'
    elif rollout['ambiguous_intervals']:reason='contact_ambiguity'
    elif not stable:reason='no_supported_continuation'
    if not math.isfinite(cost):reason='nonfinite_goal_cost'
    if reason is None and _stop_checked and getattr(engine,'terminal_stop_replies',0) and np.linalg.norm(end['velocity'])>=1.:
        reason='stopping_speed_above_limit'
    resources=sum(int(a['held']>0 and b['held']==0) for a,b in zip(states,states[1:]))
    resources+=sum(len(c.get('script_events',[])) for c in rollout.get('controls',[]))
    info=dict(label=label,cost=cost,rejected=reason,minimum_gem_distance_u=close,resources_consumed=resources,
                goal_item_id=item.get('id') if item else None,
                end_position=end['position'],terminal_support=supported,stable_continuation=stable,
                maximum_tail_normal_speed_u_per_s=normal_speed,
                continuation_s=min(3,len(states)-1)*.064,possible_pickup_interval=pickup,
                ambiguous_intervals=rollout['ambiguous_intervals'],failure=rollout['failure'])
    stopping_replies=getattr(engine,'terminal_stop_replies',0)
    stopping_scope=getattr(engine,'terminal_stop_scope','all')
    needs_stop=stopping_scope=='all' or pickup is not None or not routed
    if reason is None and stopping_replies and needs_stop and not _stop_checked:
        from nav.marble_dynamics.viability import braking_viability
        check,continued=braking_viability(engine,dynamics,rollout,gem,stopping_replies)
        if not check['valid']:
            info['rejected']='no_stopping_continuation:'+check['rejected']
            info['terminal_stop_check']=check
            return info
        # Keep the verified backup controls in the actual proposed schedule,
        # including the bridge's two delayed commands. A native gem pickup can
        # preserve this stopping destination via the existing landing contract.
        rollout.clear();rollout.update(continued)
        info=assess(engine,route,dynamics,gem,rollout,label,target_item,routed,True)
        info['terminal_stop_check']=check
        info['forecast_s']=(len(rollout['states'])-1)*.064
        if routed and item and item.get('gem') and info['possible_pickup_interval'] is not None:
            info['landing_goal']=list(rollout['states'][-1]['position'])
            info['landing_after_reply']=info['possible_pickup_interval']
            p=np.asarray(info['landing_goal']);r=rollout['states'][-1]['radius']
            hit=engine.mesh.raycast(p+[0,0,.05],p-[0,0,r+.10],cull_backfaces=True)
            info['landing_triangle_id']=hit['triangle_id'] if hit else None
    return info


def useful_candidate(pairs,route,start_distance,preserve_resources=False):
    """Choose among checked pickup/progress futures, with optional resource cost.

    Preserve mode prefers a predicted collection, then fewer actual modeled
    resource consumptions, then the existing route/continuation cost. A useful
    walking future therefore keeps scarce inventory unless a pickup or an
    otherwise inaccessible useful future justifies using it. No reward changes.
    """
    useful=[]
    for info,rollout in pairs:
        if info['rejected'] is not None or rollout is None:continue
        progress=start_distance-route.distance(rollout['states'][-1]['position'])
        if info['possible_pickup_interval'] is not None or progress>.3:
            useful.append((info,rollout))
    if not useful:return None
    def rank(pair):
        info=pair[0]
        return (int(info['possible_pickup_interval'] is None),info.get('resources_consumed',0),info['cost']) if preserve_resources else (info['cost'],)
    return min(useful,key=rank)


def search(engine,route,dynamics,raw,gem,navigator,horizon=40,warm=None,brake_replies=0,target_item=None):
    gem=np.asarray(gem[:3]);route.navigate(dynamics['native']['state']['position'],gem)
    held=int(dynamics['native']['state']['held'])
    # Holding an unused Helicopter must not lengthen every rolling proposal or
    # defeat explicitly short fallbacks. Actual activation/active flight still
    # receives the longer landing horizon and all existing physical checks.
    steps=max(horizon,96) if dynamics['native']['state']['effects'][5][0] else horizon
    use_steps=max(steps,96) if held==5 else steps
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
            specs.append(('use_'+str(use)+'_jump_'+str(jump),SequenceProposal(route,gem,use_steps,use=use,jump=jump)))
            specs.append(('direct_use_'+str(use)+'_jump_'+str(jump),SequenceProposal(route,gem,use_steps,use=use,jump=jump,direct=True)))
    if blast_available(dynamics.get('blast')):
        for blast,jump in ((0,-1),(3,0),(7,0),(7,3),(11,5)):
            for direct in (False,True):
                label=('direct_' if direct else '')+'blast_'+str(blast)+'_jump_'+str(jump)
                specs.append((label,SequenceProposal(route,gem,steps,blast=blast,jump=jump,direct=direct)))
    other_gems=[g for g in visible_gems(raw) if np.linalg.norm(np.asarray(g[:3])-gem)>.1]
    nxt=min(other_gems,key=lambda g:np.linalg.norm(np.asarray(g[:3])-gem)) if other_gems else None
    next_goal=snap_to_walkable(route.terrain,*nxt[:3])[0] if nxt else None
    navigator_spec=('frozen_navigator',NavigatorProposal(navigator,raw,route,gem,next_goal,use_steps))
    # Under a short wall budget, checking this last can starve the competent
    # learned proposal entirely. Ordering changes no physical acceptance gate.
    if getattr(engine,'navigator_first',False):specs.insert(0,navigator_spec)
    else:specs.append(navigator_spec)
    pairs=[]
    exhausted=False
    for label,proposal in specs:
        try:
            try:rollout=predict_replies(engine,dynamics,proposal,stop_on_ambiguity=True)
            except ValueError as exc:
                pairs.append((dict(label=label,rejected='unsupported_proposal',failure=str(exc),cost=math.inf),None));continue
            info=assess(engine,route,dynamics,gem,rollout,label,target_item)
            info.update(evaluated_replies=len(rollout['controls']),requested_replies=rollout['requested_replies'],
                        stopped_on_ambiguity=rollout['stopped_on_ambiguity'])
            pairs.append((info,rollout))
            if brake_replies and info['rejected']=='no_supported_continuation':
                extended=predict_replies(engine,dynamics,BrakeContinuation(rollout['proposed'],gem,brake_replies),stop_on_ambiguity=True)
                checked=assess(engine,route,dynamics,gem,extended,label+'_brake_tail',target_item)
                checked.update(proposal_replies=len(rollout['proposed']),braking_replies=brake_replies,
                               evaluated_replies=len(extended['controls']),requested_replies=extended['requested_replies'],
                               stopped_on_ambiguity=extended['stopped_on_ambiguity'],
                               forecast_s=(len(extended['states'])-1)*.064)
                pairs.append((checked,extended))
        except PlanningBudgetExceeded:
            if not getattr(engine,'use_completed_candidates_on_timeout',False):raise
            # A fully forecast and assessed candidate is not a partial future.
            # Keep only those already appended; the unfinished candidate is lost.
            exhausted=True
            for info,_ in pairs:info['search_stopped_at_deadline']=True
            break
    if warm and not exhausted:
        rollout=predict_replies(engine,dynamics,warm,stop_on_ambiguity=True)
        pairs.append((assess(engine,route,dynamics,gem,rollout,'remaining_previous_plan',target_item),rollout))
    valid=[p for p in pairs if p[0]['rejected'] is None]
    return min(valid,key=lambda p:p[0]['cost']) if valid else None,pairs


def retained_candidate(engine,route,dynamics,gem,memory,minimum_replies=8,target_item=None):
    """Revalidate a useful remaining plan before spending a full search budget.

    The same visible target must remain selected. Recompute from the NEW observed
    state; changed inventory/geometry events clear memory in the execution loop.
    A short, unsafe, ambiguous or no-progress remainder goes back to full search.
    Keep the absolute action schedule rather than installing the remainder again.
    """
    if (len(memory.remaining)<minimum_replies or memory.gem is None
            or np.linalg.norm(np.asarray(memory.gem)-gem[:3])>=.1):return None
    position=dynamics['native']['state']['position'];route.navigate(position,gem[:3])
    start=route.distance(position);forecast=predict_replies(engine,dynamics,memory.remaining,stop_on_ambiguity=True)
    info=assess(engine,route,dynamics,np.asarray(gem[:3]),forecast,memory.label,target_item,
                _stop_checked=memory.stopping_verified)
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

    def clear(self):
        self.actions=[];self.executed=0;self.lock_until=0;self.label=None;self.gem=None
        self.landing_goal=None;self.landing_after_reply=0;self.goal_item_id=None;self.landing_mode=False;self.landing_triangle_id=None
        self.observed_tail=[]
        self.stopping_verified=False
        self.physical_recovery=False
        self.last_landing_rejection=None

    def accept(self,rollout,label,gem,info=None):
        self.clear()
        self.actions=copy.deepcopy(rollout['proposed']);self.executed=0;self.label=label;self.gem=list(gem)
        events=[i for i,a in enumerate(self.actions[:13]) if a[4] or a[6] or a[8]]
        self.lock_until=min(len(self.actions),events[0]+3) if events else 0
        self.stopping_verified=bool(info and info.get('terminal_stop_check',{}).get('valid'))
        if info and info.get('landing_goal') is not None:
            self.landing_goal=list(info['landing_goal']);self.landing_after_reply=info['landing_after_reply']
            self.goal_item_id=info.get('goal_item_id');self.landing_triangle_id=info.get('landing_triangle_id')

    def observe_pickup(self,score_delta,items):
        """Keep a landing only after native score and selected-item disappearance."""
        if (score_delta>0 and self.landing_goal is not None and self.goal_item_id is not None
                and not any(i['id']==self.goal_item_id for i in items)):
            self.landing_mode=True;return True
        return False

    @property
    def locked(self):return (self.executed<self.lock_until or self.landing_goal is not None) and bool(self.remaining)

    @property
    def remaining(self):return self.actions[self.executed:]

    def advance(self,observed_state=None):
        self.executed+=1
        if observed_state is not None:
            self.observed_tail.append(copy.deepcopy(observed_state));self.observed_tail=self.observed_tail[-4:]
