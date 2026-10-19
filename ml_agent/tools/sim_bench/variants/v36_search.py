"""General phased control search and persistent discrete-action commitment."""
import copy
import math
import time
from collections import OrderedDict

import numpy as np

from nav.marble_dynamics.bridge import predict_replies
from nav.marble_dynamics.gem_demo import Proposal, candidate_cost, support
from nav.marble_dynamics.navigator import NavigatorProposal
from nav.gems import visible_gems
from nav.routing import snap_to_walkable,RoutePlanner,goal_key
from nav.marble_dynamics.blast import available as blast_available
from nav.marble_dynamics.planning_budget import PlanningBudgetExceeded


ROUTE_SPEED_LAW=True     # v16 (2026-10-15): stopping law on the remaining route length, not the next waypoint
PICKUP_MIN_SPEED=6.      # u/s floor on the target speed toward a gem (0 = brake to a stop at the gem, pre-v16)
THROUGH_GEM=True         # v22 (2026-10-15): score past a pickup toward the tour's next gem; 'through' proposals
THROUGH_SWITCH_U=.6      # u: a through proposal turns to the next gem once its path passes this close to the gem
THROUGH_REACH_U=18.      # u: through proposals are added only when the gem is this close along the route
BRAKING_BACKUP=True      # v22 (2026-10-15): the chosen plan must leave a braking backup that stays on the floor
SAFE_HOLD=True           # v22: with no useful plan, execute the best gate-checked plan instead of the unchecked actor
BACKUP_REPLIES=20        # replies of full braking after the plan's end (1.28 s)
BACKUP_CHECKS=4          # at most this many of the best useful candidates are checked per search
BACKUP_PENALTY_U=150.    # cost added to a useful plan without a braking backup (above the -100 pickup credit)
FALL_FLOOR_U=15.         # u: airborne after the braking backup with no floor this far below = a fall
OPEN_TOP=13.5            # v36: top target speed far from drops (0 = off); 11 (gem_demo cap) within OPEN_NEAR_U of a drop
OPEN_NEAR_U=4.
OPEN_FAR_U=8.
STOP_SPEED=2.            # u/s: when no useful plan exists, 'stop_N' (full braking from reply N) is tried above this speed


_FIELDS=OrderedDict()


def cached_route(terrain,goal):
    """A RoutePlanner on `goal` whose walk-distance field comes from a small cache (one Dijkstra per new goal)."""
    key=(id(terrain),goal_key(goal))
    field=_FIELDS.get(key)
    if field is None:
        field=terrain.goal_field(goal[0],goal[1],z=goal[2]);_FIELDS[key]=field
        while len(_FIELDS)>48:_FIELDS.popitem(last=False)
    else:_FIELDS.move_to_end(key)
    route=RoutePlanner(terrain);route.field=field;route.key=goal_key(goal);route.waypoint=None
    return route


def through_context(engine,route,gem,item):
    """(route to the next gem, its position, walk distance gem -> next) when the tour names a next gem for `gem`.

    Only for a real gem target outside a crossing phase; None keeps the pre-v22 scoring.
    """
    if not THROUGH_GEM or getattr(engine,'crossing_phase',False):return None
    if item is None or not item.get('gem'):return None
    nxt=(getattr(engine,'next_for',None) or {}).get(goal_key(gem))
    if nxt is None:return None
    ngem=np.asarray(nxt[:3],dtype=float);here=np.asarray(gem[:3],dtype=float)
    if np.linalg.norm(ngem-here)<.1:return None
    key=(goal_key(here),goal_key(ngem),id(route.terrain))
    cache=getattr(engine,'_through_cache',None)
    if cache is None or cache[0]!=key:
        nroute=cached_route(route.terrain,ngem);leg=float(nroute.distance(here))
        engine._through_cache=cache=(key,(nroute,ngem,leg) if math.isfinite(leg) else None)
    return cache[1]


def open_top(terrain,state):
    """v36: top target speed from the walk grid's distance to the nearest drop at the marble's cell."""
    if not OPEN_TOP:return None
    edge=getattr(terrain,'edge_dist',None)
    if edge is None:return None
    p=state['position'];j,i=terrain.cell_of(p[0],p[1]);k=terrain.level_at(j,i,float(p[2])-float(state.get('radius',.19)))
    if k<0:return None
    d=float(edge[k,j,i]);base=11.
    if d<=OPEN_NEAR_U:return None
    return base+(OPEN_TOP-base)*min(1.,(d-OPEN_NEAR_U)/(OPEN_FAR_U-OPEN_NEAR_U))


class SequenceProposal:
    def __init__(self,route,gem,steps,offset=0.,scale=1.,jump=-1,use=-1,turn_after=6,brake_after=None,direct=False,blast=-1,through=None):
        self.route=copy.copy(route);self.gem=np.asarray(gem);self.steps=steps
        self.offset=offset;self.scale=scale;self.jump=jump;self.use=use
        self.turn_after=turn_after;self.brake_after=brake_after;self.direct=direct
        self.blast=blast
        # through = through_context(...): drive through the gem without braking
        # for it, then follow the route to the next gem (v22).
        self.through=through;self.next_route=copy.copy(through[0]) if through is not None else None
        self.passed=False;self.previous=None

    def __call__(self,state,index):
        if self.through is not None:
            p=np.asarray(state['position'],dtype=float)
            if index==0:self.passed=False;self.previous=None
            if not self.passed and self.previous is not None:
                # closest approach of the last step's path to the gem
                a=self.previous;ab=p-a;t=float(np.clip(((self.gem[:3]-a)@ab)/max(float(ab@ab),1e-12),0.,1.))
                self.passed=float(np.linalg.norm(a+t*ab-self.gem[:3]))<THROUGH_SWITCH_U
            elif not self.passed:self.passed=float(np.linalg.norm(p-self.gem[:3]))<THROUGH_SWITCH_U
            self.previous=p
            if self.passed:
                ngem=self.through[1];goal=self.next_route.navigate(p,ngem)[0]
                remaining=float(self.next_route.distance(p))
                distance=remaining if math.isfinite(remaining) else float(np.linalg.norm(ngem[:2]-p[:2]))
                action=Proposal(goal,0.,speed_distance=distance,min_speed=PICKUP_MIN_SPEED,top=open_top(self.route.terrain,state))(state,0)
                action[:4]=[round(v*self.scale,6) for v in action[:4]]
                action[8]=0
                return action
        goal=self.gem if self.direct else self.route.navigate(state['position'],self.gem)[0]
        # Phases vary steering, throttle, airborne controls and braking. They
        # describe legal controls, never a named trick or a map-specific launch.
        speed_distance=None
        if ROUTE_SPEED_LAW and not self.direct:
            # The aim point is the next waypoint, a few units ahead: braking
            # for it capped every route-following candidate at 5-6 u/s (v12b
            # journals: mean 4-7 u/s, pickups at 4-6 u/s vs the human's 11.4).
            remaining=float(self.route.distance(state['position']))
            speed_distance=remaining if math.isfinite(remaining) else float(np.linalg.norm(self.gem[:2]-np.asarray(state['position'])[:2]))
        if self.through is not None:
            # the gem needs no stop: brake only for what follows it
            base=speed_distance if speed_distance is not None else float(np.linalg.norm(self.gem[:2]-np.asarray(state['position'])[:2]))
            speed_distance=base+self.through[2]
        proposal=Proposal(goal,self.offset if index<self.turn_after else 0.,
                          jump_at=self.jump-index,use_at=self.use-index,
                          brake=self.brake_after is not None and index>=self.brake_after,
                          speed_distance=speed_distance,min_speed=PICKUP_MIN_SPEED if not (self.brake_after is not None and index>=self.brake_after) else 0.,
                          top=open_top(self.route.terrain,state))
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


class LandingContinuation:
    """Keep a predicted control prefix, then follow the ordinary route until landing.

    An airborne candidate (a jump or blast over an obstacle) used to be rejected
    at the horizon because it had not landed yet. Continue it with the same
    general route-following controls and let the usual checks score the landing.
    """
    def __init__(self,prefix,route,gem,extra_replies):
        self.prefix=prefix;self.follow=SequenceProposal(route,gem,len(prefix)+extra_replies)
        self.steps=len(prefix)+extra_replies

    def __call__(self,state,index):
        if index<len(self.prefix):return self.prefix[index]
        return self.follow(state,index)


def interleave_specs(specs):
    """Round-robin the candidate families so a deadline never starves one of them.

    Families are drive, jump, blast, powerup use, steering and the learned
    proposal; within a family the original order is kept. No gate changes.
    """
    def family(label):
        base=label[7:] if label.startswith('direct_') else label
        if base.startswith('blast'):return 'blast'
        if base.startswith('use'):return 'use'
        if base.startswith('jump'):return 'jump'
        if base.startswith('steering'):return 'steer'
        if base.startswith('frozen_navigator'):return 'nav'
        return 'drive'
    groups={}
    for item in specs:groups.setdefault(family(item[0]),[]).append(item)
    order=['drive','jump','blast','use','steer','nav'];result=[]
    while any(groups.get(k) for k in order):
        for k in order:
            if groups.get(k):result.append(groups[k].pop(0))
    return result


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
    cost,close,pickup=candidate_cost(route,states,gem,routed,item,through_context(engine,route,gem,item) if routed else None)
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
    # A powerup's effect outlasts a short horizon: judged over 1.5 s a Super
    # Speed is still too fast to count as settled and loses to rolling. With
    # use_horizon set, use candidates are forecast over at least that many replies.
    use_floor=int(getattr(engine,'use_horizon',0) or 0)
    if use_floor and 1<=held<=4:use_steps=max(use_steps,use_floor)
    specs=[('route',SequenceProposal(route,gem,steps)),('direct',SequenceProposal(route,gem,steps,direct=True))]
    gem_item=target_item or next((i for i in dynamics.get('items',[]) if i['gem'] and np.linalg.norm(np.asarray(i['position'])-gem)<.1),None)
    through=through_context(engine,route,gem,gem_item)
    if through is not None:
        here=dynamics['native']['state']['position'];reach=float(route.distance(here))
        if not math.isfinite(reach):reach=float(np.linalg.norm(gem[:3]-np.asarray(here)))
        if reach<=THROUGH_REACH_U:
            specs+=[('through',SequenceProposal(route,gem,steps,through=through)),
                    ('direct_through',SequenceProposal(route,gem,steps,direct=True,through=through)),
                    ('through_0.7',SequenceProposal(route,gem,steps,scale=.7,through=through))]
    for scale in (.4,.7):specs.append(('throttle_'+str(scale),SequenceProposal(route,gem,steps,scale=scale)))
    # Climb crossings (opt-in) try more jump times and Blast-first families; gap
    # crossings keep the v7 families.
    climbing=bool(getattr(engine,'crossing_phase',False) and (getattr(engine,'crossing',None) or {}).get('climb'))
    jump_times=(0,2,4,6,8,10,12) if climbing else (0,2,5,9)
    for jump in jump_times:
        specs.append(('jump_'+str(jump),SequenceProposal(route,gem,steps,jump=jump)))
        specs.append(('direct_jump_'+str(jump),SequenceProposal(route,gem,steps,jump=jump,direct=True)))
    for offset in (-1.,-.5,.5,1.):
        specs.append(('steering_'+str(offset),SequenceProposal(route,gem,steps,offset=offset,turn_after=12)))
    specs.append(('accelerate_then_brake',SequenceProposal(route,gem,steps,brake_after=12)))
    if 1<=held<=5:
        for use,jump in ((0,-1),(3,-1),(7,-1),(3,2),(7,5)):
            specs.append(('use_'+str(use)+'_jump_'+str(jump),SequenceProposal(route,gem,use_steps,use=use,jump=jump)))
            specs.append(('direct_use_'+str(use)+'_jump_'+str(jump),SequenceProposal(route,gem,use_steps,use=use,jump=jump,direct=True)))
    if blast_available(dynamics.get('blast')) and not getattr(engine,'blast_reserved',False):
        if climbing:
            # Crossing: the launch may be several replies away; a Blast two replies
            # after the jump (while still rising) gives the highest flight. These
            # go first so the search deadline cannot starve them, direct aim only,
            # with a half-throttle variant for short ledges.
            blast_specs=[]
            for jump in jump_times:
                blast_specs.append(('direct_blast_'+str(jump+2)+'_jump_'+str(jump),SequenceProposal(route,gem,steps,blast=jump+2,jump=jump,direct=True)))
                blast_specs.append(('direct_half_blast_'+str(jump+2)+'_jump_'+str(jump),SequenceProposal(route,gem,steps,blast=jump+2,jump=jump,direct=True,scale=.5)))
            blast_specs.append(('direct_blast_0_jump_-1',SequenceProposal(route,gem,steps,blast=0,jump=-1,direct=True)))
            specs=blast_specs+specs
        else:
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
    if getattr(engine,'use_first',False) and 1<=held<=5:
        # Holding a powerup: judge its use candidates before the deadline can
        # cut them off; the ordinary families still follow (at most ~10 moved).
        uses=[s for s in specs if 'use' in s[0]];others=[s for s in specs if 'use' not in s[0]]
        specs=others[:2]+uses+others[2:]
    detour_first=None
    if held in (1,5) and float(gem[2])-float(dynamics['native']['state']['position'][2])>=float(getattr(engine,'use_first_rise_u',1.5) or 1.5):
        # v21 (2026-10-15): holding a Super Jump or Helicopter below a raised gem,
        # the use plans are the ones that can collect it; judge them first so the
        # search deadline cannot starve them (it did: decision 2237 of the v16
        # Sprawl s1 round chose 'steering' with the pickup plan never reached).
        uses=[s for s in specs if 'use' in s[0]];others=[s for s in specs if 'use' not in s[0]]
        specs=uses+others
        # and project their landings right after the family (the end-of-search
        # projection is skipped once the deadline has hit)
        detour_first=uses[-1][0] if uses else None
    if getattr(engine,'crossing_phase',False):
        # Crossing a gap: only candidates that can leave the ground are worth the
        # budget; rolling candidates would drive over the edge and are rejected anyway.
        specs=[s for s in specs if 'jump' in s[0]]
    elif detour_ratio(route,dynamics['native']['state']['position'],gem)>=float(getattr(engine,'detour_blast_ratio',1.5) or 1.5)             and blast_meter_value(dynamics)>=float(getattr(engine,'detour_blast_meter',0.5) or 0.5) and not getattr(engine,'blast_reserved',False):
        # The walking route is a long way round compared with the straight line
        # (a wall or a step between here and the gem) and the Blast meter is
        # charged: forecast the Blast+jump families first so the search deadline
        # cannot starve the one family that can go over instead of around. The
        # physics forecast still decides whether any of them is accepted.
        blasts=[s for s in specs if 'blast' in s[0]];others=[s for s in specs if 'blast' not in s[0]]
        specs=blasts+others
        if detour_first is None:detour_first=blasts[-1][0] if blasts else None
    if getattr(engine,'interleave_candidates',False):specs=interleave_specs(specs)
    if getattr(engine,'navigator_first',False):specs.insert(0,navigator_spec)
    else:specs.append(navigator_spec)
    landing_replies=int(getattr(engine,'landing_replies',0) or 0)
    if getattr(engine,'crossing_phase',False):
        # Crossing a gap to another walking component: airborne candidates must
        # be allowed to land on the far side before they are judged.
        landing_replies=max(landing_replies,int(getattr(engine,'crossing_landing_replies',40) or 40))
    pairs=[]
    exhausted=False
    ballistic=bool(getattr(engine,'ballistic_landing',False));airborne=[]
    reserve=float(getattr(engine,'ballistic_reserve_s',0.) or 0.);deadline=getattr(engine,'planning_deadline',None)
    for label,proposal in specs:
        if ballistic and reserve and deadline and time.perf_counter()>deadline-reserve:
            # Keep part of the budget for the landing projections of the
            # candidates already forecast; later families would otherwise
            # consume it and the long flights would never be judged.
            for info,_ in pairs:info['search_stopped_at_reserve']=True
            break
        try:
            try:rollout=predict_replies(engine,dynamics,proposal,stop_on_ambiguity=True)
            except ValueError as exc:
                pairs.append((dict(label=label,rejected='unsupported_proposal',failure=str(exc),cost=math.inf),None));continue
            info=assess(engine,route,dynamics,gem,rollout,label,target_item)
            info.update(evaluated_replies=len(rollout['controls']),requested_replies=rollout['requested_replies'],
                        stopped_on_ambiguity=rollout['stopped_on_ambiguity'])
            if 'use' in label and rollout['requested_replies']>steps and info['rejected'] is None and info['possible_pickup_interval'] is None:
                # A powerup candidate forecast over a longer horizon covers more
                # ground by itself; charge the extra time at a nominal rolling
                # speed so it wins only when the powerup is genuinely faster.
                penalty=(rollout['requested_replies']-steps)*.064*float(getattr(engine,'ballistic_time_rate',8.) or 8.)
                info['cost']+=penalty;info['time_penalty_u']=penalty
            pairs.append((info,rollout))
            if ballistic and info['rejected']=='no_supported_continuation':airborne.append((label,info,rollout))
            pickup_landing=int(getattr(engine,'pickup_landing_replies',0) or 0)
            collects=info.get('possible_pickup_interval') is not None
            extend=(landing_replies or (pickup_landing and collects)) and info['rejected']=='no_supported_continuation'
            if extend:
                # Still airborne (or not yet settled) at the horizon: follow the
                # route onward and score the landing with the same gates. With
                # only pickup_landing_replies set, this applies solely to a
                # candidate whose forecast already collects the gem (a jump or
                # blast up to a gem must not lose to standing underneath it).
                replies=landing_replies or pickup_landing
                landed=predict_replies(engine,dynamics,LandingContinuation(rollout['proposed'],route,gem,replies),stop_on_ambiguity=True)
                checked=assess(engine,route,dynamics,gem,landed,label+'_landing',target_item)
                checked.update(proposal_replies=len(rollout['proposed']),landing_replies=replies,pickup_only_landing=not landing_replies,
                               evaluated_replies=len(landed['controls']),requested_replies=landed['requested_replies'],
                               stopped_on_ambiguity=landed['stopped_on_ambiguity'],
                               forecast_s=(len(landed['states'])-1)*.064)
                pairs.append((checked,landed))
                if ballistic and collects and checked['rejected']=='no_supported_continuation':
                    # v21 (2026-10-15): a plan that COLLECTS the gem and is still in
                    # the air after the continuation (a Super Jump up to a platform)
                    # gets the no-input landing projection instead of a discard.
                    airborne.append((label+'_landing',checked,landed))
            if detour_first is not None and label==detour_first and ballistic and airborne:
                # Detour mode: the Blast families were forecast first; project their
                # landings now, before the rolling families spend the budget (the
                # end-of-search projection would be skipped at the deadline).
                project_airborne(engine,route,dynamics,gem,target_item,airborne,pairs);airborne=[]
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
    if ballistic and airborne and not exhausted:
        try:project_airborne(engine,route,dynamics,gem,target_item,airborne,pairs)
        except PlanningBudgetExceeded:
            if not getattr(engine,'use_completed_candidates_on_timeout',False):raise
            exhausted=True
            for info,_ in pairs:info['search_stopped_at_deadline']=True
    if warm and not exhausted:
        rollout=predict_replies(engine,dynamics,warm,stop_on_ambiguity=True)
        pairs.append((assess(engine,route,dynamics,gem,rollout,'remaining_previous_plan',target_item),rollout))
    apply_crossing_landing(pairs,engine)
    apply_blast_value(pairs,blast_meter_value(dynamics),float(getattr(engine,'blast_value_u',10.) if getattr(engine,'blast_value_u',None) is not None else 10.),float(getattr(engine,'blast_floor_u',0.) if getattr(engine,'blast_floor_u',None) is not None else 0.))
    apply_use_margin(pairs,float(getattr(engine,'use_margin_u',0.) or 0.))
    if BRAKING_BACKUP:check_backups(engine,route,dynamics,gem,pairs,steps,target_item)
    valid=[p for p in pairs if p[0]['rejected'] is None]
    return min(valid,key=lambda p:p[0]['cost']) if valid else None,pairs


def braking_backup(engine,dynamics,rollout,gem,replies=None):
    """Brake from the plan's end; (False, detail) only when the marble clearly leaves the floor.

    The terminal gate checks that a plan ENDS supported, not that the marble can stop
    afterwards: v21 Cube s2 fell three times right after the horizon of a plan that
    collected an edge gem at 8 u/s. Contact ambiguity or an unbuildable continuation is
    unknown, not a fall, and passes.
    """
    from nav.marble_dynamics.viability import resume_forecast
    replies=int(replies or BACKUP_REPLIES)
    try:start=resume_forecast(dynamics,rollout)
    except ValueError:return True,dict(skipped='unresumable')
    saved=getattr(engine,'planning_deadline',None);engine.planning_deadline=None
    try:cont=predict_replies(engine,start,Proposal(np.asarray(gem[:3],dtype=float),steps=replies,brake=True),stop_on_ambiguity=True)
    except ValueError as exc:return True,dict(skipped=str(exc)[:60])
    finally:engine.planning_deadline=saved
    end=cont['states'][-1];z0=float(rollout['states'][-1]['position'][2]);z1=float(end['position'][2])
    supported=support(engine.mesh,end);fell=False
    if not supported:
        # Still in the air after braking: a drop onto a lower floor is not a fall,
        # nothing below within FALL_FLOOR_U is.
        p=np.asarray(end['position'],dtype=float)
        hit=engine.mesh.raycast(p+[0,0,.05],p-[0,0,FALL_FLOOR_U],cull_backfaces=True)
        fell=hit is None or hit['normal'][2]<=.4
    return not fell,dict(supported=supported,drop_u=z0-z1,replies=len(cont['controls']),ambiguous=bool(cont['ambiguous_intervals']))


def check_backups(engine,route,dynamics,gem,pairs,steps=24,target_item=None):
    """Prefer plans with a braking backup: check the best useful candidates in cost order.

    The first one that passes wins (every cheaper one has failed); in ordinary
    states that is the first one checked. A useful plan without a backup (failed,
    or unchecked behind BACKUP_CHECKS failures) is charged BACKUP_PENALTY_U, not
    rejected: v22 rejected them and on a Cube slope near an edge gem, where no plan
    has a backup, every search then failed and the unchecked actor drove off the
    edge 11 times in 60 s (v21 took the best plan there: 81 points, 1 fall). When
    the best ones all fail their backups,
    full braking from now / in 4 / in 8 replies ('stop_N') is forecast and checked
    too; a stop with a backup then beats every penalised plan (the v21 Cube s2 fall
    at decision 1768 becomes stop_4). Stops are not ordinary candidates: as such
    they won 9 of 60 ordinary states on end speed alone (offline audit 2026-10-15).
    """
    position=dynamics['native']['state']['position'];start=route.distance(position)
    useful=[p for p in pairs if p[0]['rejected'] is None and p[1] is not None and (p[0].get('possible_pickup_interval') is not None
            or start-route.distance(p[1]['states'][-1]['position'])>.3)]
    useful.sort(key=lambda p:p[0]['cost'])
    for info,rollout in useful[:BACKUP_CHECKS]:
        ok,detail=braking_backup(engine,dynamics,rollout,gem)
        info['braking_backup']=detail
        if ok:return
        info['cost']=float(info['cost'])+BACKUP_PENALTY_U;info['no_braking_backup']=True
    for info,_ in useful[BACKUP_CHECKS:]:
        info['cost']=float(info['cost'])+BACKUP_PENALTY_U;info['braking_backup_unchecked']=True
    # v22d: stops only after useful plans FAILED their backups. With nothing useful
    # at all (often just the search deadline on a curved surface) v22c braked in open
    # terrain (Bowl s1: stop_N chosen 102 times while driving at 8-17 u/s); the
    # controller's actor/safe-hold rule handles that case.
    if not useful:return
    if getattr(engine,'crossing_phase',False):return
    saved=getattr(engine,'planning_deadline',None);engine.planning_deadline=None
    try:
        for n in (0,4,8):
            try:rollout=predict_replies(engine,dynamics,SequenceProposal(route,gem,steps,brake_after=n),stop_on_ambiguity=True)
            except ValueError:continue
            info=assess(engine,route,dynamics,gem,rollout,'stop_'+str(n),target_item)
            if info['rejected'] is None:
                ok,detail=braking_backup(engine,dynamics,rollout,gem);info['braking_backup']=detail
                if not ok:info['cost']=float(info['cost'])+BACKUP_PENALTY_U;info['no_braking_backup']=True
            pairs.append((info,rollout))
    finally:engine.planning_deadline=saved


def safe_hold_choice(engine,dynamics,pairs,gem,limit=4):
    """Best gate-checked candidate (by cost) with a braking backup; else the best gate-checked one; else None.

    Without any backup the plan still passed every physics gate for 1.5 s, which
    beats the unchecked actor (the v22 Cube loop fell off the edge under it).
    """
    valid=sorted([p for p in pairs if p[0]['rejected'] is None and p[1] is not None],key=lambda p:p[0]['cost'])
    for info,rollout in valid[:limit]:
        if info.get('braking_backup') is not None and not info.get('no_braking_backup'):return info,rollout
        if info.get('no_braking_backup'):continue
        ok,detail=braking_backup(engine,dynamics,rollout,gem) if BRAKING_BACKUP else (True,None)
        info['braking_backup']=detail
        if ok:return info,rollout
    if valid:
        valid[0][0]['safe_hold_without_backup']=True
        return valid[0]
    return None


def actor_forecast_unsafe(engine,dynamics,pairs,gem):
    """True when the search forecast the learned actor's proposal and none of its variants is
    gate-valid with a braking backup (it would leave the floor); False when one is, or when the
    deadline cut the search before the actor's proposal was forecast (unknown: v21 behaviour).
    Narrower definitions lost: late forecasts of a cut proposal (v24, v25: parking on Cragmire) and
    'unsafe only on a fall into the void' (v29: 94.8 / 102.7 vs 109.5 / 101.2, falls 37 vs 20)."""
    nav=[p for p in pairs if p[0]['label'].startswith('frozen_navigator') and p[1] is not None]
    if not nav:return False
    for info,rollout in nav:
        if info['rejected'] is not None or info.get('no_braking_backup'):continue
        if info.get('braking_backup') is not None:return False
        ok,detail=braking_backup(engine,dynamics,rollout,gem) if BRAKING_BACKUP else (True,None)
        info['braking_backup']=detail
        if ok:return False
    return True


def project_airborne(engine,route,dynamics,gem,target_item,airborne,pairs):
    """Operator-approved ballistic landing projection: finish airborne candidates
    with NO input. Coarse parabola filter for all of them, exact engine projection
    only for the few with the best coarse landing. Appends the landed candidates
    to `pairs`; raises PlanningBudgetExceeded at the deadline."""
    from nav.marble_dynamics.ballistic import coarse_landing,project_landing,MAX_FLIGHT_REPLIES,HELICOPTER_FLIGHT_REPLIES
    bounds=getattr(engine,'mission_bounds',None);ranked=[];collecting=[]
    for label,info,rollout in airborne:
        coarse=coarse_landing(engine.mesh,rollout['states'][-1],bounds);info['ballistic_coarse']=coarse
        if not (coarse['hit'] and coarse['floor_like']):continue
        # v21: a candidate whose forecast already collects the gem is projected
        # whatever its landing distance; the others compete for the few exact slots.
        if info.get('possible_pickup_interval') is not None:collecting.append((route.distance(coarse['point']),label,info,rollout))
        else:ranked.append((route.distance(coarse['point']),label,info,rollout))
    ranked.sort(key=lambda item:item[0]);collecting.sort(key=lambda item:item[0])
    for _,label,info,rollout in collecting+ranked[:int(getattr(engine,'ballistic_exact_limit',4) or 4)]:
        cap=HELICOPTER_FLIGHT_REPLIES if rollout['states'][-1]['effects'][5][0] else MAX_FLIGHT_REPLIES
        try:landed,reason=project_landing(engine,dynamics,rollout,cap)
        except ValueError as exc:info['ballistic']='unsupported:'+str(exc)[:40];continue
        if reason is not None:info['ballistic']=reason;continue
        checked=assess(engine,route,dynamics,gem,landed,label+'_ballistic',target_item)
        # A longer forecast covers more ground by itself. Charge the extra
        # flight time at a nominal rolling speed so a jump only wins when
        # it is genuinely faster than rolling for the same time.
        extra=max(0,len(landed['states'])-len(rollout['states']))
        penalty=extra*.064*float(getattr(engine,'ballistic_time_rate',8.) or 8.)
        if checked['possible_pickup_interval'] is None:checked['cost']+=penalty
        checked['time_penalty_u']=penalty
        checked.update(proposal_replies=len(rollout['proposed']),ballistic_replies=len(landed['proposed'])-len(rollout['proposed']),
                       evaluated_replies=len(landed['controls']),requested_replies=landed['requested_replies'],
                       stopped_on_ambiguity=landed['stopped_on_ambiguity'],forecast_s=(len(landed['states'])-1)*.064)
        pairs.append((checked,landed))


def detour_ratio(route,position,gem):
    """Walking-route length over straight-line distance to the gem (inf when no walk route)."""
    straight=float(np.linalg.norm(np.asarray(gem[:3])[:2]-np.asarray(position[:3])[:2]))
    if straight<4.:return 1.
    route.navigate(position,gem[:3]);d=float(route.distance(position))
    return d/straight if math.isfinite(d) else math.inf


def blast_meter_value(dynamics):
    blast=dynamics.get('blast') if isinstance(dynamics,dict) else None
    return float(blast['value']) if blast and blast.get('enabled',True) else 0.


def apply_blast_value(pairs,meter,value_u,floor_u=0.):
    """Charge a candidate that fires the Blast for the meter it spends.

    A Blast empties the meter (its power scales with the charge) and the meter
    takes 25 s to refill, so a half-meter Blast for one unit of progress costs
    the full-meter hop over the next wall. The charge is meter * value_u in
    route units, so a Blast is chosen only when it saves more walking than
    that; a predicted pickup keeps its -100 credit.
    """
    charge=meter*value_u+(floor_u if meter>0 else 0.)
    if charge<=0:return
    for info,rollout in pairs:
        if rollout is None or 'blast' not in info['label']:continue
        fired=any(c.get('script_events') for c in rollout.get('controls',[]))
        if not fired:continue
        # floor_u: a fixed charge per fire (v14, 2026-10-15). In the v12 journals
        # low-meter fires (< 0.5) gained a median 2.5 u in the 3 s after the fire
        # against 13 u before it; the per-meter charge alone is too small there.
        info['blast_charge_u']=charge;info['cost']=float(info['cost'])+charge


def apply_crossing_landing(pairs,engine):
    """In a crossing, a Blast or powerup candidate must end on the far side.

    Spending the meter to land short of a ledge, or back on the launch side,
    counts as nothing (rejected 'resource_without_landing'). Ordinary jumps keep
    their progress scoring so the marble can still creep to the edge and restage
    with a run-up, unless they end under the ledge.
    """
    crossing=getattr(engine,'crossing',None) if getattr(engine,'crossing_phase',False) else None
    crossings=getattr(engine,'crossings',None)
    if crossing is None or crossings is None:return
    land=np.asarray(crossing['land'],dtype=float)
    for info,rollout in pairs:
        # The staged goal is the landing cell, so a 'pickup' here means the flight
        # passed through that cell, not that a gem was collected: no exemption.
        if rollout is None or info['rejected'] is not None:continue
        end=np.asarray(rollout['states'][-1]['position'],dtype=float)
        landed=crossings.component_at(end)==crossing['target'] and abs(end[2]-land[2])<=1.5
        if landed:continue
        label=info['label']
        if 'blast' in label or 'use' in label:info['rejected']='resource_without_landing'
        elif not crossing.get('climb'):continue
        elif info.get('possible_pickup_interval') is not None:
            # Flying through the landing cell and coming down elsewhere is not a
            # pickup of anything.
            info['rejected']='phantom_landing_pickup'
        elif np.linalg.norm(end[:2]-land[:2])<3. and end[2]<land[2]-1.5:
            # Standing under the ledge is not progress toward a goal on top of it,
            # whatever the plan-view distance says.
            info['rejected']='under_the_ledge'


def apply_use_margin(pairs,margin):
    """A powerup-use candidate must beat the best non-use candidate by `margin` units.

    Marks the use candidates that fail as rejected ('use_margin'). A use that
    predicts a pickup when no non-use candidate does is kept whatever its cost:
    the powerup is then the only way to the gem in this horizon. Audit of v6
    (10-14): Super Speed and Helicopter uses that won on a small cost edge
    rarely led to a gem; Super Jump uses mostly did.
    """
    if margin<=0:return
    valid=[(info,r) for info,r in pairs if info['rejected'] is None and r is not None]
    non_use=[info for info,_ in valid if 'use' not in info['label']]
    if not non_use:return
    best=min(info['cost'] for info in non_use)
    non_use_pickup=any(info.get('possible_pickup_interval') is not None for info in non_use)
    for info,_ in valid:
        if 'use' not in info['label']:continue
        if info.get('possible_pickup_interval') is not None and not non_use_pickup:continue
        if info['cost']>best-margin:
            info['rejected']='use_margin';info['use_margin_shortfall_u']=info['cost']-(best-margin)


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
    # a safe hold (v22) makes no progress by definition; it stays while it stays safe
    if info['possible_pickup_interval'] is None and progress<=.3 and memory.label!='safe_hold':return None
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
