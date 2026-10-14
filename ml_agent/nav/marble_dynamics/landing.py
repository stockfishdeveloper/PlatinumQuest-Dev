"""Use complete-map support geometry to continue safely after a forecast pickup.

Sample actual collision surfaces, then replay each approach/landing control
sequence from the same initial observed state. Surface points are proposals;
the full mechanics/support/ambiguity checks still decide acceptability.
"""
import numpy as np

from nav.marble_dynamics.bridge import predict_replies
from nav.marble_dynamics.gem_demo import Proposal
from nav.marble_dynamics.search import assess


def landing_points(mesh,state,limit=8):
    p=np.asarray(state['position']);v=np.asarray(state['velocity']);r=state['radius']
    lead=p+np.asarray([v[0]*.5,v[1]*.5,0.])
    ids=np.flatnonzero(mesh.normals[:,2]>.4)
    points=mesh.closest_points(lead,ids)
    centers=mesh.triangles[ids].mean(axis=1);toward=centers-points
    distance=np.linalg.norm(toward,axis=1)
    # Move into the same convex triangle when the closest point is on a rim;
    # small faces use their centroid. No triangle disappears from the map.
    amount=np.minimum(1.,np.divide(r*2.5,distance,out=np.ones(len(ids)),where=distance>0))
    inset=points+toward*amount[:,None]
    body=inset+mesh.normals[ids]*(r+.01)
    cost=np.linalg.norm(body[:,:2]-lead[:2],axis=1)+.2*np.abs(body[:,2]-p[2])
    selected=[]
    for index in np.argsort(cost,kind='stable'):
        goal=body[index]
        if any(np.linalg.norm(goal-c['position'])<.75 for c in selected):continue
        if len(mesh.sphere_contacts(goal,r-.00001)['triangle_ids']):continue
        hit=mesh.raycast(goal+[0,0,.05],goal-[0,0,r+.10],cull_backfaces=True)
        if hit is None or hit['normal'][2]<=.4:continue
        selected.append(dict(position=goal.tolist(),triangle_id=int(ids[index]),cost=float(cost[index])))
        if len(selected)>=limit:break
    return selected


class LandingContinuation:
    def __init__(self,prefix,goal,replies):
        self.prefix=prefix;self.goal=goal;self.steps=len(prefix)+replies

    def __call__(self,state,index):
        if index<len(self.prefix):return self.prefix[index]
        return Proposal(self.goal)(state,0)


def alternatives(engine,route,dynamics,gem,pairs,landing_replies=80,limit=8):
    gem=np.asarray(gem[:3],dtype=float)
    extra=[];seen=set()
    for original,rollout in pairs:
        if rollout is None or rollout['failure'] or original['rejected'] is None:continue
        pickup=original.get('possible_pickup_interval')
        if pickup is None or any(i<pickup for i in rollout['ambiguous_intervals']):continue
        count=min(int(pickup),len(rollout['proposed']))
        if not count:continue
        prefix=rollout['proposed'][:count];key=tuple(tuple(a) for a in prefix)
        if key in seen:continue
        seen.add(key)
        for index,point in enumerate(landing_points(engine.mesh,rollout['states'][count],limit)):
            changed=predict_replies(engine,dynamics,LandingContinuation(prefix,point['position'],landing_replies),stop_on_ambiguity=True)
            info=assess(engine,route,dynamics,gem,changed,original['label']+'_landing_'+str(index))
            # The steering target need not coincide with the checked settled
            # endpoint. Commit completion to the endpoint that assess accepted;
            # otherwise an accurately landed plan can exhaust its controls while
            # waiting to arrive at a different, unverified point. Preserve any
            # existing stopping-check destination and the original proposal.
            destination=info.get('landing_goal')
            triangle=info.get('landing_triangle_id')
            if destination is None and info['rejected'] is None:
                destination=list(changed['states'][-1]['position'])
                state=changed['states'][-1];p=np.asarray(destination);r=state['radius']
                hit=engine.mesh.raycast(p+[0,0,.05],p-[0,0,r+.10],cull_backfaces=True)
                triangle=hit['triangle_id'] if hit else None
            info.update(landing_after_reply=count,
                landing_goal=destination if destination is not None else point['position'],
                landing_triangle_id=triangle if triangle is not None else point['triangle_id'],
                proposed_landing_goal=point['position'],proposed_landing_triangle_id=point['triangle_id'],
                original_label=original['label'],requested_replies=changed['requested_replies'],
                evaluated_replies=len(changed['controls']),stopped_on_ambiguity=changed['stopped_on_ambiguity'])
            extra.append((info,changed))
    return extra


def choose_landing(engine,route,dynamics,gem,remaining_replies,horizon=80,samples=40,seed=997,preserve_resources=True):
    """Bounded general fallback, requiring pickup and a checked support target."""
    from nav.marble_dynamics.phased_search import explore
    from nav.marble_dynamics.settle import choose_settled
    from nav.marble_dynamics.search import useful_candidate
    _,sampled=explore(engine,route,dynamics,gem,horizon=horizon,samples=samples,seed=seed)
    _,settled=choose_settled(engine,route,dynamics,gem)
    pairs=alternatives(engine,route,dynamics,gem,sampled+settled)
    for info,rollout in pairs:
        if len(rollout['proposed'])>remaining_replies and info['rejected'] is None:
            info['rejected']='beyond_execution_budget'
    start=route.distance(dynamics['native']['state']['position'])
    chosen=useful_candidate(pairs,route,start,preserve_resources)
    if chosen and chosen[0]['possible_pickup_interval'] is None:chosen=None
    return chosen,sampled+settled+pairs


def retained_landing(engine,route,dynamics,memory):
    """Revalidate the landing controls from the actual post-pickup native state.

    The geometric support point is not a gem or a fictitious pickup. Walking
    graph reachability is not a veto on an independently checked airborne leg.
    """
    if not memory.landing_mode or not memory.remaining:return None
    rollout=predict_replies(engine,dynamics,memory.remaining,stop_on_ambiguity=True)
    info=assess(engine,route,dynamics,memory.landing_goal,rollout,memory.label,routed=False,
                _stop_checked=memory.stopping_verified)
    info['possible_pickup_interval']=None
    info.update(goal_kind='landing',landing_goal=memory.landing_goal,landing_triangle_id=memory.landing_triangle_id,
                revalidated_remainder=True)
    if getattr(memory,'physical_recovery',False) and np.linalg.norm(rollout['states'][-1]['velocity'])>=1.:
        info['rejected']='recovery_stopping_speed_above_limit'
    info['physical_recovery']=getattr(memory,'physical_recovery',False)
    memory.last_landing_rejection=info if info['rejected'] else None
    return None if info['rejected'] else (info,rollout)


def landing_complete(engine,state,memory):
    from nav.marble_dynamics.search import terminal_support
    if not memory.landing_mode or memory.executed<memory.landing_after_reply:return False
    supported,stable,_=terminal_support(engine.mesh,memory.observed_tail)
    return bool(supported and stable and np.linalg.norm(state['velocity'])<1.
                and np.linalg.norm(np.asarray(state['position'])-memory.landing_goal)<.5)
