"""Checked connected pickup maneuvers and isolated experimental comparisons.

MODE='retain' leaves the existing two-level ranking intact. Its selected second
leg becomes executable only after a new complete-root forecast and a checked
braking continuation. No future observation or unknown spawn enters a plan.
MODE='score' compares validated composites under a common spawn-time objective;
'refine' additionally improves feasible sequences under the same work budget.
"""
from collections import Counter
import copy
import math

import numpy as np

MODE = 'off'
RESEARCH_HANDOFF = False
SCORED_MODES = ('score','refine','order')
SCORED_OFFERS = 4
REFINE_TRIALS = 20
MAX_GOALS = 4


def configure(args):
    global MODE, RESEARCH_HANDOFF
    if getattr(args,'connected_mode',None) is not None:MODE=args.connected_mode
    if getattr(args,'connected_research_handoff',None) is not None:
        RESEARCH_HANDOFF=bool(args.connected_research_handoff)


def spawn_cost(value, target_pickup):
    """One time-to-clear objective, without a bonus for a longer prediction.

    The existing spawn value includes -100 per forecast gem. Remove that term,
    retaining a single feasibility preference for a proven target pickup. All
    candidates still pay their ordinary contact, jump and resource charges.
    """
    if MODE not in SCORED_MODES:return value[0]
    from .spawn_value import SV_PICK_CREDIT
    return value[0]+SV_PICK_CREDIT*(value[1]['picks']-int(target_pickup is not None))


def order_value(info, legacy_replies, margin_replies):
    """Use the candidate objective across first targets as well as within them."""
    if MODE=='order' and info.get('spawn_value') is not None:
        from .spawn_value import SV_W_U_PER_S
        return float(info['cost']),margin_replies*.064*SV_W_U_PER_S
    return legacy_replies,margin_replies


def stats(engine):
    if not hasattr(engine, 'connected_stats'):engine.connected_stats = Counter()
    return engine.connected_stats


def offer(rollout, first, second, prefix_replies, followup=None):
    """Keep controls outside info (which is serialized in decision journals)."""
    if MODE == 'off':return
    actions = list(rollout['proposed']) if followup is None else list(rollout['proposed'][:prefix_replies]) + list(followup['proposed'])
    rollout['_connected_offer'] = dict(actions=actions, goals=[copy.deepcopy(first), copy.deepcopy(second)])


def materialize(engine, route, dynamics, gem, selected, deadline=None):
    if MODE == 'off' or selected is None:return selected
    info, old = selected;offer = old.get('_connected_offer')
    if info.get('connected'):return selected
    if not offer or info.get('landing_goal') is not None:return selected
    from .bridge import predict_replies
    from .gem_demo import possible_pickup, Proposal
    from .search import assess, terminal_support, BACKUP_REPLIES
    from .viability import resume_forecast
    from .planning_budget import check, PlanningBudgetExceeded
    counter = stats(engine);counter['offers'] += 1
    saved = getattr(engine, 'planning_deadline', None)
    # Use the already declared two-level allowance; do not grant a fresh budget.
    engine.planning_deadline = deadline if deadline is not None else saved
    try:
        check(engine)
        roll = predict_replies(engine, dynamics, offer['actions'], stop_on_ambiguity=True)
        checked = assess(engine, route, dynamics, np.asarray(gem[:3]), roll, info['label'])
        if checked['rejected']:
            counter['reject:' + checked['rejected']] += 1;return selected
        pickups = [possible_pickup(roll['states'], g, roll.get('native_states')) for g in offer['goals']]
        if any(k is None for k in pickups) or any(b<a for a,b in zip(pickups,pickups[1:])):
            counter['reject:pickup_order'] += 1;return selected
        start = resume_forecast(dynamics, roll)
        backup = predict_replies(engine, start, Proposal(np.asarray(offer['goals'][-1]['position']), steps=BACKUP_REPLIES, brake=True), stop_on_ambiguity=True)
        supported, stable, _ = terminal_support(engine.mesh, backup['states'][-4:])
        if backup['failure'] or backup['ambiguous_intervals'] or not supported or not stable:
            counter['reject:backup'] += 1;return selected
        goals=offer['goals']
        if MODE in SCORED_MODES:
            actual=[]
            for g in dynamics.get('items',[]):
                if not g.get('gem'):continue
                k=possible_pickup(roll['states'],g,roll.get('native_states'))
                if k is not None:actual.append((k,g))
            actual.sort(key=lambda x:x[0]);goals=[g for _,g in actual];pickups=[k for k,_ in actual]
        meta = dict(goals=[dict(id=g['id'], position=list(g['position'])) for g in goals],
                    pickups=list(map(int, pickups)), replies=len(roll['proposed']), mode=MODE)
        # Retention alone must preserve the selector's score and first-leg identity.
        checked['connected']=meta
        if MODE=='retain':checked['cost']=info['cost']
        else:checked['cost']+=float(info.get('blast_charge_u',0.) or 0.)
        for key in ('two_level_replies', 'two_level_next', 'blast_charge_u'):
            if key in info:checked[key] = info[key]
        roll['connected'] = meta;counter['materialized'] += 1
        return checked, roll
    except PlanningBudgetExceeded:
        counter['reject:budget'] += 1;return selected
    except ValueError:
        counter['reject:context'] += 1;return selected
    finally:engine.planning_deadline = saved


def expand(engine,route,dynamics,gem,pairs):
    """Offer validated composites to the SAME selector as ordinary candidates."""
    if MODE not in SCORED_MODES:return
    seeds=sorted((p for p in pairs if p[0]['rejected'] is None and p[1] is not None
                  and p[1].get('_connected_offer')),key=lambda p:p[0]['cost'])[:SCORED_OFFERS]
    for seed in seeds:
        checked=materialize(engine,route,dynamics,gem,seed,getattr(engine,'connected_deadline',None))
        if checked is not seed:pairs.append(checked)


def finalize(engine,route,dynamics,gem,selected):
    if MODE=='retain':
        return materialize(engine,route,dynamics,gem,selected,getattr(engine,'connected_deadline',None))
    if MODE=='refine' and selected and selected[0].get('connected'):
        return refine(engine,route,dynamics,gem,selected)
    return selected


def mutations(roll, pickups):
    """Local control phases around actual contacts; event timing stays explicit."""
    from nav.joystick import action_to_joystick
    from itertools import zip_longest
    base=np.asarray(roll['proposed'],dtype=float);n=len(base)
    cuts=sorted(set([0]+[max(0,int(k)-8) for k in pickups[:3]]))
    brakes=[];events=[];steers=[];powers=[]
    # Braking can begin BEFORE a cluster entry. These are controls, not penalties
    # for arriving quickly; the whole connected trajectory decides their value.
    for k in cuts:
        for width in (4,8):
            trial=base.copy()
            for j in range(k,min(n,k+width)):
                v=roll['states'][j]['velocity']
                trial[j,:6]=action_to_joystick(0.,0.,1.,jump=int(base[j,4]),brake=1.,vx=v[0],vy=v[1])
            brakes.append(('brake_%d_%d'%(k,width),trial.tolist()))
    for col,name in ((6,'use'),(8,'blast'),(4,'jump')):
        pressed=np.flatnonzero(base[:,col])
        if not len(pressed):continue
        for shift in (-2,2):
            trial=base.copy();trial[:,col]=0
            trial[np.clip(pressed+shift,0,n-1),col]=1
            # Move the powerup's world aim with its event, independently of camera yaw.
            if col==6:
                for p in pressed:trial[int(np.clip(p+shift,0,n-1)),7]=base[p,7]
            events.append((name+'_'+str(shift),trial.tolist()))
    for k in cuts:
        for angle,scale in ((-.15,1.),(.15,1.),(0.,.8),(0.,1.1)):
            trial=base.copy()
            for j in range(k,min(n,k+8)):
                x,y=base[j,3]-base[j,2],base[j,0]-base[j,1]
                mag=math.hypot(x,y);c,s=math.cos(angle),math.sin(angle)
                trial[j,:6]=action_to_joystick(c*x-s*y,s*x+c*y,min(1.,scale*mag/math.sqrt(2)),jump=int(base[j,4]))
            steers.append(('steer_%d_%g_%g'%(k,angle,scale),trial.tolist()))
    if int(roll['states'][0].get('held',0))==2 and not np.any(base[:,6]) and pickups:
        goals=(roll.get('connected') or {}).get('goals',[])
        if goals:
            for use_at in (0,min(6,n-1)):
                for width in (4,8):
                    trial=base.copy();trial[use_at,6]=1
                    delta=np.asarray(goals[0]['position'])-np.asarray(roll['states'][use_at]['position'])
                    trial[use_at,7]=math.atan2(delta[0],delta[1])
                    brake_at=max(use_at+4,int(pickups[0])-8)
                    for j in range(brake_at,min(n,brake_at+width)):
                        v=roll['states'][j]['velocity']
                        trial[j,:6]=action_to_joystick(0.,0.,1.,jump=int(base[j,4]),brake=1.,vx=v[0],vy=v[1])
                    powers.append(('use_brake_%d_%d'%(use_at,width),trial.tolist()))
    # A small budget must sample each parameter family, not exhaust itself on
    # whichever family happened to be listed first.
    for group in zip_longest(brakes,events,steers,powers):
        for variant in group:
            if variant is not None:yield variant


def refine(engine,route,dynamics,gem,selected):
    """A bounded feasible-seed search: order/extension, then joint control phases.

    Uses only visible items and the existing two-level time allowance. Every
    accepted update is forecast again from the real root including input queues.
    """
    from .planning_budget import check,PlanningBudgetExceeded
    from .search import follow_time,cached_route
    from .viability import resume_forecast
    from .gem_demo import possible_pickup
    counter=stats(engine);saved=getattr(engine,'planning_deadline',None)
    engine.planning_deadline=getattr(engine,'connected_deadline',None) or saved
    best=selected;initial=float(best[0]['cost']);trials=0
    items={g['id']:g for g in dynamics.get('items',[]) if g.get('gem')}

    def evaluate(actions,goals,label):
        nonlocal best,trials
        check(engine)
        if trials>=REFINE_TRIALS:return
        trials+=1;counter['refine_trials']+=1
        info=dict(best[0]);info.pop('connected',None);info['label']='connected_'+label
        seed=(info,dict(_connected_offer=dict(actions=actions,goals=goals)))
        candidate=materialize(engine,route,dynamics,gem,seed,engine.planning_deadline)
        extra_use=candidate is not seed and candidate[0].get('inventory_consumed',0)>best[0].get('inventory_consumed',0)
        margin=max(.01,float(getattr(engine,'use_margin_u',0.) or 0.)) if extra_use else .01
        if candidate is not seed and candidate[0]['cost']<best[0]['cost']-margin:
            candidate[0]['refinement_parent']=selected[0]['label'];best=candidate
            counter['refine_improvements']+=1

    try:
        # Branch at a real pickup: an alternate next target, then up to four gems.
        # Keep only actual collected items removed from the continuation world.
        for depth in range(1,MAX_GOALS):
            info,roll=best;meta=info['connected']
            if depth>len(meta['goals']):break
            k=meta['pickups'][depth-1]
            prefix=dict(roll,states=roll['states'][:k+1],controls=roll['controls'][:k],proposed=roll['proposed'][:k])
            start=resume_forecast(dynamics,prefix)
            got={gid for gid,g in items.items() if (possible_pickup(roll['states'],g,roll.get('native_states')) or 10**9)<=k}
            start['items']=[g for g in start.get('items',[]) if g['id'] not in got]
            p=np.asarray(start['native']['state']['position'])
            left=sorted((g for gid,g in items.items() if gid not in got),key=lambda g:np.linalg.norm(np.asarray(g['position'])-p))
            for g in left[:2]:
                if np.linalg.norm(np.asarray(g['position'])-p)>12.:continue
                if depth<len(meta['goals']) and meta['goals'][depth]['id']==g['id']:continue
                check(engine);target=np.asarray(g['position']);nr=cached_route(route.terrain,target)
                second,tail=follow_time(engine,nr,start,target,g,return_rollout=True)
                if tail is None:continue
                goals=[items[x['id']] for x in meta['goals'][:depth]]+[g]
                evaluate(list(roll['proposed'][:k])+list(tail['proposed']),goals,'order_'+str(depth))
        # Coordinate updates are local to a feasible seed, not blind discovery.
        for iteration in range(2):
            roll=best[1];meta=best[0]['connected'];goals=[items[g['id']] for g in meta['goals']]
            limit=trials+max(1,(REFINE_TRIALS-trials)//(2-iteration))
            for label,actions in mutations(roll,meta['pickups']):
                if trials>=limit:break
                evaluate(actions,goals,label)
    except PlanningBudgetExceeded:counter['refine_budget']+=1
    except ValueError:counter['refine_context']+=1
    finally:engine.planning_deadline=saved
    if best is not selected:
        best[0]['refinement_gain_u']=initial-float(best[0]['cost']);counter['refined_selected']+=1
    return best


def pickup(memory, score_delta, items):
    """Advance only on an observed own pickup and the expected item disappearance."""
    if MODE == 'off' or score_delta <= 0 or not memory.connected:return False
    # Item IDs can reappear in a new spawn. Never interpret that reuse as the
    # unexecuted second leg of a maneuver whose last predicted pickup is past.
    if memory.executed>=max(memory.connected['pickups']):return False
    live = {i['id'] for i in items};goals = memory.connected['goals']
    if goals[0]['id'] in live:return False
    while goals and goals[0]['id'] not in live:goals.pop(0)
    if not goals or not memory.remaining:return False
    memory.gem = list(goals[0]['position']);memory.connected_handoff = True
    memory.lock_from = memory.lock_until = 0
    return True


def retained(controller, dynamics):
    memory = controller.memory
    if MODE == 'off' or not memory.connected_handoff or not memory.connected:return None
    from .search import retained_candidate
    target = memory.connected['goals'][0]
    item = next((i for i in dynamics.get('items', []) if i['id'] == target['id']), None)
    if item is None:memory.clear();return None
    counter = stats(controller.engine)
    result = retained_candidate(controller.engine, controller.route, dynamics, np.asarray(target['position']), memory, target_item=item)
    if result is None:
        counter['handoff_rejected'] += 1;memory.clear();return None
    result[0]['connected'] = copy.deepcopy(memory.connected)
    counter['handoff_replies'] += 1
    return result, list(target['position']) + [float(item.get('value') or 1.), item['id']]
