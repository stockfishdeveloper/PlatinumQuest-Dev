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
from nav.marble_dynamics.planning_budget import check as check_budget

RESOURCE_BY_TIME=True    # 2026-10-25 night: choose_resource ranks helpful items by fetch time + the type's quickest conditional pickup
RESOURCE_FETCH_U_S=10.   # ... the walk to the item at this speed


def item_goal(item,position):
    p=list(item['position'])
    return p+[0.,float(np.linalg.norm(np.asarray(p)-position))]


def walk_distances(terrain,position,goals):
    """[route.navigate(position,g) then route.distance(position), for g in goals] from at most two forward Dijkstras.

    2026-10-16: one backward goal field per item costs 30 ms on Bowl (60 Super Speeds = 1.9 s, never inside the 2 s
    budget). The snapping is goal_field's (the item: nearest walkable node within 3 cells) and dist_at's (the marble:
    its own node, else the nearest within 2 cells + walk_res, else straight line + 5 u); a path's length is summed
    from the marble's end instead of the item's, the same up to float64 rounding before the field's float32.
    """
    import math
    from scipy.sparse.csgraph import dijkstra
    from nav.routing import goal_key
    t=terrain
    if t._graph is None:t._build_graph()
    x,y,z=position[0],position[1],position[2]
    j,i=t.cell_of(x,y);k=t.level_at(j,i,z)
    own=(k,j,i) if k>=0 else None;near=t._nearest_walkable(j,i,radius=2,z=z)
    starts=[int(t._node_idx[n]) if n is not None else -1 for n in (own,near)]
    valid=[s for s in starts if s>=0]
    rows=dijkstra(t._graph,directed=True,indices=sorted(set(valid))) if valid and t._graph is not None else None
    row={s:rows[n] for n,s in enumerate(sorted(set(valid)))} if rows is not None else {}
    out=[]
    for g in goals:
        gj,gi=t.cell_of(g[0],g[1]);src=t._nearest_walkable(gj,gi,radius=3,z=g[2])
        target=int(t._node_idx[src]) if src is not None else -1
        def reach(start):
            if target<0 or start<0:return math.inf
            return float(np.float32(row[start][target]))
        d=reach(starts[0])
        if not math.isfinite(d):
            d=reach(starts[1])
            d=d+t.walk_res if math.isfinite(d) else math.inf
        if not math.isfinite(d):
            key=goal_key(g);d=float(math.hypot(x-key[0],y-key[1]))+5.0
        out.append(d)
    return out


def choose_resource(engine,route,dynamics,raw,gem,navigator,horizon=80,brake_replies=48,preserve_resources=False,max_distance=None):
    """`max_distance` (2026-10-22): items farther than this (walking distance) are not fetched. The ledger over 80 rounds:
    powerup goals took 5-35 s a round; 114 fetches were farther than the nearest gem (Bowl: a Helicopter 161 u away
    right after a pickup, 10 s out and 10 s back, in every Bowl round)."""
    position=np.asarray(dynamics['native']['state']['position'])
    held=int(dynamics['native']['state']['held'])
    items=[i for i in dynamics.get('items',[]) if not i['gem'] and 1<=i['powerup']<=5 and i['powerup']!=held]
    types=sorted({i['powerup'] for i in items});hypotheses=[];helpful=[];pickup_s={}
    for powerup in types:
        check_budget(engine)
        hypothetical=copy.deepcopy(dynamics);hypothetical['native']['state']['held']=powerup
        conditional_raw=raw.copy();conditional_raw[RAW_POW_HELD]=powerup
        chosen,pairs=search(engine,route,hypothetical,conditional_raw,gem[:3],navigator,horizon,brake_replies=brake_replies)
        pickups=[info for info,rollout in pairs if info['rejected'] is None and info['possible_pickup_interval'] is not None]
        hypotheses.append(dict(hypothetical_held=powerup,conditional_pickup_candidates=pickups,
            contract='Conditional inventory at current pose, not a predicted acquisition or an action sent to the game.'))
        if pickups:
            helpful.append(powerup);pickup_s[powerup]=min(float(p['possible_pickup_interval']) for p in pickups)*.064
    candidates=[]
    # Existing visible native items only. Prefer the shortest checked walking
    # acquisition; full mechanics still verifies its actual movement controls.
    # 2026-10-16: route work between forecasts is Python; with the compiled forecast the deadline was otherwise
    # first checked many items later (v41 Bowl: 11-16 s decisions, one goal field per item)
    check_budget(engine)
    useful=[item for item in items if item['powerup'] in helpful]
    terrain=getattr(route,'terrain',None)
    if hasattr(terrain,'_nearest_walkable'):
        distances=walk_distances(terrain,position,[item['position'] for item in useful])
    else:
        distances=[]
        for item in useful:
            route.navigate(position,item['position']);distances.append(route.distance(position))
    ordered=list(zip(distances,useful))
    if RESOURCE_BY_TIME:
        # 2026-10-25 night (operator approved): the quickest way to the gem, not the nearest helpful item: the walk to the item at
        # RESOURCE_FETCH_U_S plus its type's quickest conditional pickup (the hypothesis search above)
        ordered.sort(key=lambda pair:pair[0]/RESOURCE_FETCH_U_S+pickup_s.get(pair[1]['powerup'],0.))
    else:ordered.sort(key=lambda pair:pair[0])
    for distance,item in ordered:
        if max_distance is not None and not (distance<=max_distance):continue
        check_budget(engine)
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
