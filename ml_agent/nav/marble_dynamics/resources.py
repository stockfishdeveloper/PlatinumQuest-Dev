"""Conditional inventory usefulness followed by actual resource acquisition.

Only invoked when ordinary gem search finds no useful checked future. Conditional
held-item hypotheses are explicitly hypothetical, never sent to the game. The
real acquisition phase sends ordinary controls toward an existing visible item;
gem planning resumes after observing the actual held inventory change.
"""
import copy

import numpy as np

from nav.protocol import RAW_POW_HELD
from nav.marble_dynamics.search import search,useful_candidate


def item_goal(item,position):
    p=list(item['position'])
    return p+[0.,float(np.linalg.norm(np.asarray(p)-position))]


def choose_resource(engine,route,dynamics,raw,gem,navigator,horizon=80,brake_replies=48,preserve_resources=False):
    position=np.asarray(dynamics['native']['state']['position'])
    held=int(dynamics['native']['state']['held'])
    items=[i for i in dynamics.get('items',[]) if not i['gem'] and 1<=i['powerup']<=5 and i['powerup']!=held]
    types=sorted({i['powerup'] for i in items});hypotheses=[];helpful=[]
    for powerup in types:
        hypothetical=copy.deepcopy(dynamics);hypothetical['native']['state']['held']=powerup
        conditional_raw=raw.copy();conditional_raw[RAW_POW_HELD]=powerup
        chosen,pairs=search(engine,route,hypothetical,conditional_raw,gem[:3],navigator,horizon,brake_replies=brake_replies)
        pickups=[info for info,rollout in pairs if info['rejected'] is None and info['possible_pickup_interval'] is not None]
        hypotheses.append(dict(hypothetical_held=powerup,conditional_pickup_candidates=pickups,
            contract='Conditional inventory at current pose, not a predicted acquisition or an action sent to the game.'))
        if pickups:helpful.append(powerup)
    candidates=[]
    # Existing visible native items only. Prefer the shortest checked walking
    # acquisition; full mechanics still verifies its actual movement controls.
    ordered=[]
    for item in items:
        if item['powerup'] not in helpful:continue
        route.navigate(position,item['position']);ordered.append((route.distance(position),item))
    ordered.sort(key=lambda pair:pair[0])
    for distance,item in ordered:
        goal=item_goal(item,position);route.navigate(position,goal[:3]);start=route.distance(position)
        chosen,pairs=search(engine,route,dynamics,raw,goal[:3],navigator,40,brake_replies=brake_replies,target_item=item)
        chosen=useful_candidate(pairs,route,start,preserve_resources)
        candidates.extend([{**info,'goal_kind':'resource','item_id':item['id'],'powerup':item['powerup']} for info,rollout in pairs])
        if chosen:
            info,rollout=chosen;progress=start-route.distance(rollout['states'][-1]['position'])
            if info['possible_pickup_interval'] is not None or progress>.3:
                info.update(goal_kind='resource',item_id=item['id'],powerup=item['powerup'],
                    conditional_gem_goal=list(gem[:3]),predicted_progress_u=progress)
                return chosen,item,goal,hypotheses,candidates
    return None,None,None,hypotheses,candidates
