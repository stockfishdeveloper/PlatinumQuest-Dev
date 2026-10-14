"""Bounded matched fresh-seed validation with a protected frozen navigator.

    python -m nav.marble_dynamics.reliability_eval --maps KingOfTheMarble_Hunt Sprawl_Hunt Bowl_Hunt

No training, checkpoint writes, rescue, artificial gem placement or maneuver rules.
Both branches stop at first fall or fixed decision budget. Validation seeds/maps
are familiar geometry; they are not a claim of unseen-map generalization.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import sys
from pathlib import Path
import time
from collections import Counter

import numpy as np

from nav.collision.mesh import CollisionMap,atomic_json
from nav.gems import visible_gems,plan_tour
from nav.routing import RoutePlanner,goal_key,snap_to_walkable
from nav.terrain import TerrainMap,TerrainGrid
from nav.protocol import NOOP_ACTION,RAW_SCORE,RAW_POW_ITEMS,RAW_POW_HELD
from nav.recovery import ProgressMonitor,simulation_time
from nav.marble_dynamics.record import ML,RUNTIME,game,command,archive,sha
from nav.marble_dynamics.commit_demo import restore
from nav.marble_dynamics.physics import GeneralPhysics
from nav.marble_dynamics.bridge import predict_replies,reply
from nav.marble_dynamics.evaluate import errors
from nav.marble_dynamics.events import external_changes
from nav.marble_dynamics.navigator import FrozenNavigator
from nav.marble_dynamics.search import search,assess,PlanMemory,retained_candidate,useful_candidate
from nav.marble_dynamics.resources import choose_resource,item_goal
from nav.marble_dynamics.geometry_route import walking_terrain
from nav.marble_dynamics.recharge import choose_recharge
from nav.marble_dynamics.blast import context_changes as blast_context_changes
from nav.marble_dynamics.landing import choose_landing,retained_landing,landing_complete
from nav.marble_dynamics.query_cache import CachedCollisionMap
from nav.marble_dynamics.ground_cache import MemoGroundClearance
from nav.marble_dynamics.inventory import context_changes as inventory_context_changes
from nav.marble_dynamics.global_goals import observed_gems
from nav.collision.experimental_sweep import CompiledSweepMap,CachedCompiledSweepMap
from nav.marble_dynamics.planning_budget import PlanningBudgetExceeded
from nav.marble_dynamics.viability import ordinary_recovery
from nav.source_archive import snapshot_sources


def tour(terrain,fields,raw,current):
    gems=visible_gems(raw)
    def distance(a,g):
        key=goal_key(g)
        if key not in fields:fields[key]=terrain.goal_field(g[0],g[1],jumps=False,z=g[2])
        return terrain.dist_at(fields[key],a[0],a[1],g[:2],z=a[2])
    gem,nxt=plan_tour(gems,current,raw[:3],raw[3:5],dist=distance)
    return gem,nxt,gems


def pending_reply_matches(observed,sent):
    # Script serialization uses six significant digits. Jump/use/Blast are
    # discrete; powerup aim is only meaningful on a use command.
    for i in (0,1,2,3,5):
        if abs(float(observed[i])-float(sent[i]))>2e-5:return False
    return all(bool(observed[i])==bool(sent[i]) for i in (4,6,8))


def pick(engine,route,dynamics,raw,gems,current,nav,horizon,memory,brake_replies=0,recovery_horizon=0,target_item=None,preserve_resources=False):
    position=dynamics['native']['state']['position'];all_pairs=[];options=[]
    for index,gem in enumerate([current]+[g for g in gems if np.linalg.norm(np.asarray(g[:3])-current[:3])>.1]):
        route.navigate(position,gem[:3]);start=route.distance(position)
        warm=memory.remaining if memory.gem and np.linalg.norm(np.asarray(memory.gem)-gem[:3])<.1 else None
        chosen,pairs=search(engine,route,dynamics,raw,gem[:3],nav,horizon,warm=warm,brake_replies=brake_replies,target_item=target_item)
        chosen=useful_candidate(pairs,route,start,preserve_resources)
        all_pairs.extend([{**p[0],'gem':list(gem[:3]),'search_horizon':horizon} for p in pairs])
        if chosen:
            info,rollout=chosen;progress=start-route.distance(rollout['states'][-1]['position'])
            pickup=info.get('possible_pickup_interval')
            if pickup is not None or progress>.3:
                rank=(int(pickup is not None),float(gem[3])/max((pickup or horizon)*.064,.064),progress)
                options.append((rank,chosen,gem))
                if index==0:break
    if not options:
        if recovery_horizon>horizon:
            chosen,extra,current=pick(engine,route,dynamics,raw,gems,current,nav,recovery_horizon,memory,brake_replies,target_item=target_item,preserve_resources=preserve_resources)
            return chosen,all_pairs+extra,current
        return None,all_pairs,current
    _,chosen,current=max(options,key=lambda p:p[0]);route.navigate(position,current[:3])
    return chosen,all_pairs,current


def branch(env,proc,world,name,seed,kind,args,terrain,engine,checkpoint_hash):
    folder=args.output/name/str(seed)/kind;folder.mkdir(parents=True,exist_ok=False)
    match=restore(env,world)
    if match['position_error_u']>1e-4 or match['velocity_error_u_per_s']>1e-4 or not match['score_equal']:
        raise RuntimeError('Fixture mismatch '+str(match))
    reference=world.get('native_reference')
    if reference is not None:
        current_native=env.msg.dynamics['native']['state']
        changed=external_changes(reference,current_native)
        if changed or any(reference[k]!=current_native[k] for k in ('mode','oob','smooth_camera')) or [e[0] for e in reference['effects']]!=[e[0] for e in current_native['effects']]:
            raise RuntimeError('Restored physical parameters/effects differ: '+str(changed))
        match['physical_parameters_effects_inventory_match']=True
    nav=FrozenNavigator(terrain,args.checkpoint)
    proposal_terrain=args.geometry_terrain if kind=='physics_plan' and args.geometry_route else terrain
    route=RoutePlanner(proposal_terrain);fields={};memory=PlanMemory()
    raw_file=RUNTIME/(args.output.name+'_'+name+'_'+str(seed)+'_'+kind+'.jsonl')
    command(env,'DYNAMICSRECORD '+raw_file.as_posix(),'dynamics_record')
    match.update(start_native_sim_clock_ms=env.msg.dynamics['native']['state']['clock_ms'],
        start_native_frame_clock_ms=env.msg.dynamics['native']['frame_clock_ms'],
        start_sim_clock_mod32=env.msg.dynamics['native']['state']['clock_ms']%32,
        start_frame_clock_mod32=env.msg.dynamics['native']['frame_clock_ms']%32,
        start_raw_observation_sha256=hashlib.sha256(np.asarray(env.msg.obs,dtype='<f4').tobytes()).hexdigest(),
        start_powerup_respawn_seconds=env.msg.obs[RAW_POW_ITEMS].reshape(3,5)[:,4].tolist())
    initial=float(env.msg.obs[RAW_SCORE]);logs=[];pickups=[];searches=[];current=None;stop='decision_budget'
    started=time.perf_counter();lock_steps=0;reuse_steps=0;full_search_steps=0;cancelled=[];budget_stop_elapsed=None
    resource=None;resource_decisions=[];acquisitions=[];recharge_decisions=[];landing_decisions=[];landing_events=[];global_goal_decisions=[];short_decisions=[]
    journal=(RUNTIME/(args.output.name+'_'+name+'_'+str(seed)+'_'+kind+'.decisions.jsonl')).open('w',encoding='utf-8',buffering=1)
    while len(logs)<args.decisions:
        if (args.output/'STOP').exists():stop='operator_stop';break
        snapshot=copy.deepcopy(env.msg.dynamics);state=snapshot['native']['state'];raw=env.msg.obs.copy()
        if state['mode']!=0 or state['oob']:stop='unsupported_state';break
        if kind=='physics_plan' and landing_complete(engine,state,memory):
            landing_events.append(dict(decision=len(logs),event='supported_landing_complete',position=state['position']))
            memory.clear()
        target,nxt,gems=tour(terrain,fields,raw,current)
        if target is None:
            if memory.landing_mode:target=memory.landing_goal+[0.,-1.]
            else:stop='no_live_gem';break
        gem_target=target
        if kind=='physics_plan' and args.global_goals and resource is None and memory.gem is not None and not memory.landing_mode:
            held_goal=next((g for g in observed_gems(snapshot,raw) if np.linalg.norm(np.asarray(g[:3])-memory.gem)<.1),None)
            if held_goal is not None and not any(np.linalg.norm(np.asarray(g[:3])-memory.gem)<.1 for g in gems):
                target=held_goal;gems=gems+[held_goal];gem_target=target
        if kind=='physics_plan' and resource is not None:
            present=next((item for item in snapshot['items'] if item['id']==resource['id']),None)
            if state['held']==resource['powerup'] or present is None:
                acquisitions.append(dict(decision=len(logs),item_id=resource['id'],wanted=resource['powerup'],
                    actual_held=state['held'],inventory_confirmed=state['held']==resource['powerup']))
                resource=None;memory.clear()
            else:
                target=item_goal(present,np.asarray(state['position']));gems=[target];resource=present
        t0=time.perf_counter();pairs=[];reused=False
        budget=getattr(args,'search_budget_s',0)
        engine.planning_deadline=t0+budget if budget else None
        try:
            if kind=='physics_plan':
                if memory.landing_mode:
                    retained=retained_landing(engine,route,snapshot,memory)
                    if retained is None:
                        atomic_json(folder/'landing_revalidation_failure.json',dict(decision=len(logs),
                            dynamics=snapshot,raw_observation=raw.tolist(),landing_goal=memory.landing_goal,
                            remaining_actions=memory.remaining,stopping_verified=memory.stopping_verified,
                            rejected_assessment=getattr(memory,'last_landing_rejection',None)))
                        cancelled.append(dict(decision=len(logs),reason='landing_revalidation_failed'))
                        memory.clear();stop='landing_revalidation_failed';break
                    info,forecast=retained;current=memory.landing_goal+[0.,-1.];reused=True;reuse_steps+=1
                elif memory.locked and any(np.linalg.norm(np.asarray(g[:3])-memory.gem)<.1 for g in gems):
                    current=next(g for g in gems if np.linalg.norm(np.asarray(g[:3])-memory.gem)<.1)
                    route.navigate(state['position'],current[:3])
                    forecast=predict_replies(engine,snapshot,memory.remaining)
                    info=assess(engine,route,snapshot,np.asarray(current[:3]),forecast,memory.label,resource,
                                _stop_checked=memory.stopping_verified)
                    if info['rejected']:
                        cancelled.append(dict(decision=len(logs),reason=info['rejected']));memory.clear()
                    else:lock_steps+=1
                else:
                    if memory.landing_goal is not None:
                        cancelled.append(dict(decision=len(logs),reason='selected_goal_disappeared_without_confirmed_pickup'))
                        memory.clear()
                    else:memory.lock_until=0
                if not reused and not memory.locked and args.reuse_plan:
                    retained=retained_candidate(engine,route,snapshot,np.asarray(target[:3]),memory,target_item=resource)
                    if retained:
                        info,forecast=retained;current=target;reused=True;reuse_steps+=1
                if not memory.locked and not reused:
                    full_search_steps+=1
                    chosen,pairs,current=pick(engine,route,snapshot,raw,gems,target,nav,args.horizon,memory,args.brake_continuation,args.recovery_horizon,resource,args.preserve_resources)
                    if not chosen and args.global_goals and resource is None:
                        all_gems=observed_gems(snapshot,raw)
                        extra=[g for g in all_gems if not any(np.linalg.norm(np.asarray(g[:3])-np.asarray(n[:3]))<.1 for n in gems)]
                        if extra:
                            chosen,extra_pairs,current=pick(engine,route,snapshot,raw,extra,extra[0],nav,args.horizon,memory,
                                args.brake_continuation,args.recovery_horizon,preserve_resources=args.preserve_resources)
                            pairs.extend(extra_pairs)
                            global_goal_decisions.append(dict(decision=len(logs),observed_gems=len(all_gems),nearby_gems=len(gems),
                                extra_goals=[list(g) for g in extra],selected=chosen[0] if chosen else None,selected_goal=list(current) if chosen else None))
                            atomic_json(folder/'global_goal_decisions.json',global_goal_decisions)
                    if not chosen and args.resource_acquisition and resource is None:
                        chosen,resource,resource_target,hypotheses,resource_pairs=choose_resource(
                            engine,route,snapshot,raw,target,nav,max(args.horizon,args.recovery_horizon),args.brake_continuation,args.preserve_resources)
                        resource_decisions.append(dict(decision=len(logs),actual_state=state,gem_goal=list(target),
                            conditional_hypotheses=hypotheses,selected_item=resource,candidates=resource_pairs))
                        atomic_json(folder/'resource_decisions.json',resource_decisions)
                        pairs.extend(resource_pairs)
                        if chosen:current=resource_target
                    if not chosen and args.recharge_continuation and resource is None:
                        chosen,recharge_pairs=choose_recharge(engine,route,snapshot,target,
                            args.decisions-len(logs),float(raw[31]),max(args.horizon,args.recovery_horizon,80))
                        recharge_decisions.append(dict(decision=len(logs),goal=list(target),
                            candidates=[info for info,rollout in recharge_pairs],selected=chosen[0] if chosen else None))
                        atomic_json(folder/'recharge_decisions.json',recharge_decisions)
                        pairs.extend([{**info,'gem':list(target[:3])} for info,rollout in recharge_pairs])
                        if chosen:current=target
                    if not chosen and args.landing_search and resource is None:
                        chosen,landing_pairs=choose_landing(engine,route,snapshot,target,
                            min(args.decisions-len(logs),max(0,int(float(raw[31])/64.))),
                            max(args.horizon,args.recovery_horizon,80),args.proposal_samples,args.proposal_seed,args.preserve_resources)
                        landing_decisions.append(dict(decision=len(logs),goal=list(target),
                            candidates=[info for info,rollout in landing_pairs],selected=chosen[0] if chosen else None))
                        atomic_json(folder/'landing_decisions.json',landing_decisions)
                        pairs.extend([{**info,'gem':list(target[:3])} for info,rollout in landing_pairs])
                        if chosen:current=target
                    if not chosen and getattr(args,'short_horizons',[]):
                        attempts=[]
                        short_gems=observed_gems(snapshot,raw) if args.global_goals and resource is None else gems
                        for short_horizon in sorted(set(args.short_horizons),reverse=True):
                            chosen,short_pairs,current=pick(engine,route,snapshot,raw,short_gems,target,nav,short_horizon,memory,
                                args.brake_continuation,0,resource,args.preserve_resources)
                            pairs.extend(short_pairs)
                            attempts.append(dict(horizon=short_horizon,candidates=short_pairs,selected=chosen[0] if chosen else None))
                            if chosen:break
                        short_decisions.append(dict(decision=len(logs),attempts=attempts,selected=chosen[0] if chosen else None,
                            selected_goal=list(current) if chosen else None))
                        atomic_json(folder/'short_horizon_decisions.json',short_decisions)
                    if not chosen:
                        atomic_json(folder/'terminal_search.json',dict(decision=len(logs),dynamics=snapshot,
                            raw_observation=raw.tolist(),target=list(target),candidates=pairs,resource_goal=resource))
                        stop='no_candidate_passed_checks';break
                    info,forecast=chosen;memory.accept(forecast,info['label'],current[:3],info)
                action=forecast['proposed'][0]
                commit=min(args.commit,len(forecast['proposed']),args.decisions-len(logs))
            else:
                current=target;waypoint,_=route.navigate(state['position'],current[:3])
                ngoal=snap_to_walkable(terrain,*nxt[:3])[0] if nxt else None
                action=nav.act(raw,waypoint,current[:3],ngoal)
                info=dict(label='frozen_navigator',cost=None,rejected=None)
                # Unmodelled native Blast/Mega actions are still legal for the
                # comparator. Log them explicitly instead of altering the policy.
                try:forecast=predict_replies(engine,snapshot,[action])
                except ValueError as exc:forecast=dict(failure=str(exc),states=[],proposed=[action])
                commit=1
        except PlanningBudgetExceeded as exc:
            stop='planning_time_budget'
            budget_stop_elapsed=time.perf_counter()-t0
            atomic_json(folder/'planning_budget_stop.json',dict(decision=len(logs),reason=str(exc),
                budget_s=budget,elapsed_s=budget_stop_elapsed,dynamics=snapshot,
                raw_observation=raw.tolist(),target=list(target),training=False,
                contract='No partial forecast or fabricated action executed; played-interval outcome only, not a certified stopped marble.'))
            break
        elapsed=time.perf_counter()-t0;searches.append(elapsed)
        for i in range(commit):
            snapshot=copy.deepcopy(env.msg.dynamics);before=env.msg.obs.copy();tick=env.msg.tick
            if logs and not pending_reply_matches(snapshot['bridge_action'],logs[-1]['action']):
                atomic_json(folder/'timing_violation.json',dict(decision=len(logs),
                    observed_bridge_action=snapshot['bridge_action'],expected_reply=logs[-1]['action'],
                    dynamics=snapshot,planning_s=elapsed,valid_comparison=False))
                stop='bridge_timing_contract_violation';break
            action=forecast['proposed'][i]
            waypoint,_=route.navigate(before[:3],current[:3])
            if kind=='physics_plan':
                others=[g for g in visible_gems(before) if np.linalg.norm(np.asarray(g[:3])-current[:3])>.1]
                near=min(others,key=lambda g:np.linalg.norm(np.asarray(g[:3])-current[:3])) if others else None
                next_goal=snap_to_walkable(terrain,*near[:3])[0] if near else None
                nav.act(before,waypoint,current[:3],next_goal)
            nav.sent_action(action,before)
            msg,outcome=env.step(action[:6],repeat=1,use_pow=int(action[6]),pow_yaw=action[7] if action[6] else None,use_blast=int(action[8]))
            actual=msg.dynamics['native']['state'];nav.outcome(before,msg.obs)
            predicted=forecast['states'][i+1] if len(forecast['states'])>i+1 else None
            error=errors(predicted,actual) if predicted else None
            events=external_changes(predicted,actual) if predicted else ['unmodelled_prediction']
            if predicted:events+=blast_context_changes(predicted,msg.dynamics)
            if predicted:events+=inventory_context_changes(predicted,msg.dynamics)
            delta=float(msg.obs[RAW_SCORE])-float(before[RAW_SCORE])
            if delta>0:pickups.append(dict(decision=len(logs),points=delta,position=actual['position']))
            logs.append(dict(decision=len(logs),tick=tick,dynamics=snapshot,raw_observation=before.tolist(),action=action,chosen=info,candidates=pairs if i==0 else [],
                             planning_s=elapsed if i==0 else 0.,predicted_next=predicted,actual_next=actual,errors=error,
                             external_events=events,score_delta=delta,outcome=outcome,gem=list(current),waypoint=list(waypoint),
                             goal_kind='landing' if memory.landing_mode else ('resource' if resource else 'gem'),
                             goal_item_id=resource['id'] if resource else info.get('goal_item_id'),
                             landing_triangle_id=memory.landing_triangle_id,
                             primary_gem_goal=list(gem_target),
                             lock_remaining=max(0,memory.lock_until-memory.executed)))
            journal.write(json.dumps(logs[-1])+'\n')
            memory.advance(actual)
            if outcome['fell'] or outcome['round_ended'] or outcome['reconnected']:stop='game_or_fall_event';break
            if kind=='physics_plan' and error and error[0]>.05 and not events:stop='prediction_error_gate';break
            if delta>0 and not events and memory.observe_pickup(delta,msg.dynamics['items']):
                landing_events.append(dict(decision=len(logs)-1,event='native_pickup_landing_commit',
                    item_id=memory.goal_item_id,points=delta,landing_goal=memory.landing_goal))
                break
            if delta>0 and not events and memory.landing_mode:break
            if delta>0 or events:
                if memory.locked:cancelled.append(dict(decision=len(logs)-1,reason='pickup_or_external_event'))
                memory.clear();break
        atomic_json(folder/'progress.json',dict(decisions=len(logs),points=float(env.msg.obs[RAW_SCORE])-initial,stop=stop))
        if len(searches)%6==0:print(json.dumps(dict(map=name,seed=seed,branch=kind,decisions=len(logs),points=float(env.msg.obs[RAW_SCORE])-initial,choice=info['label'])),flush=True)
        if stop!='decision_budget':break
    journal.close()
    # Archive the actual final reply boundary before closing recording. Budget
    # completion is not a landing guarantee; downstream native replay needs the
    # complete observed pending-input/resource context, not a reconstructed one.
    atomic_json(folder/'final_context.json',dict(dynamics=env.msg.dynamics,
        raw_observation=env.msg.obs.tolist(),target=list(current) if current is not None else None,
        played_decisions=len(logs),stop_reason=stop,source='actual_native_observation'))
    last_scored=float(env.msg.obs[RAW_SCORE]);command(env,'DYNAMICSRECORD off','dynamics_record');archive(raw_file,folder/'native.jsonl.gz')
    final=float(env.msg.obs[RAW_SCORE]);closing=final-last_scored
    if closing:pickups.append(dict(decision=len(logs),points=closing,phase='recording_close'))
    report=dict(map=name,seed=seed,branch=kind,actual_points=last_scored-initial,initial_score=initial,final_score=last_scored,
                status='invalid' if stop=='bridge_timing_contract_violation' else 'valid',
                valid_comparison=stop!='bridge_timing_contract_violation',
                recording_close_score_delta=closing,actual_pickup_events=pickups,decisions=len(logs),simulated_s=len(logs)*.064,
                falls=sum(int(l['outcome']['fell']) for l in logs),stop_reason=stop,initial_match=match,checkpoint_sha256=checkpoint_hash,
                checkpoint_update=nav.update if hasattr(nav,'update') else None,geometry_sha256=engine.mesh.info['mesh_sha256'],
                median_planning_s=float(np.median(searches)) if searches else None,planning_s=searches,elapsed_wall_s=time.perf_counter()-started,
                future_event_lock_revalidations=lock_steps,commitment_cancellations=cancelled,
                reuse_remaining_plan=args.reuse_plan,remaining_plan_revalidations=reuse_steps,full_searches=full_search_steps,
                braking_continuation_replies=args.brake_continuation,
                recovery_search_horizon=args.recovery_horizon,
                geometry_routing=proposal_terrain.geometry_route_report if args.geometry_route and kind=='physics_plan' else None,
                resource_acquisition_enabled=args.resource_acquisition,resource_goal_attempts=len(resource_decisions),
                preserve_resources=args.preserve_resources,
                recharge_continuation_enabled=args.recharge_continuation,recharge_goal_attempts=len(recharge_decisions),
                recharge_continuations_selected=sum(int(r['selected'] is not None) for r in recharge_decisions),
                landing_search_enabled=args.landing_search,landing_search_attempts=len(landing_decisions),
                landing_continuations_selected=sum(int(r['selected'] is not None) for r in landing_decisions),
                landing_events=landing_events,
                query_cache_enabled=args.query_cache,clearance_cache_enabled=args.clearance_cache,
                compiled_sweep_enabled=hasattr(engine.mesh,'kernel'),
                query_cache_hits=getattr(engine.mesh,'cache_hits',None),query_cache_misses=getattr(engine.mesh,'cache_misses',None),
                global_goals_enabled=args.global_goals,global_goal_attempts=len(global_goal_decisions),
                global_goals_selected=sum(int(d['selected'] is not None) for d in global_goal_decisions),
                short_horizons=getattr(args,'short_horizons',[]),short_horizon_attempts=len(short_decisions),
                short_continuations_selected=sum(int(d['selected'] is not None) for d in short_decisions),
                terminal_stop_replies=getattr(args,'terminal_stop_replies',0),
                terminal_stop_scope=getattr(args,'terminal_stop_scope','all'),
                physics_frame_ms=args.physics_frame_ms,decision_ms=64,
                search_budget_s=getattr(args,'search_budget_s',0),
                stopping_search_elapsed_s=budget_stop_elapsed,
                resource_acquisition_events=acquisitions,
                max_live_errors=np.max([l['errors'] for l in logs if l['errors']],axis=0).tolist() if any(l['errors'] for l in logs) else None,
                native_raw_sha256=sha(raw_file),owned_game_pid=proc.pid,policy_training_started=False,
                protocol='Local '+str(args.physics_frame_ms)+'ms physics/64ms decisions, DELAY 0, SSYAWFIX off; competitive off, no rescue; first fall or decision budget ends both branches',
                limitations='Familiar maps; deterministic A2 neural controller with walk-tour/routing, no rescue/stuck detour. Regular/item-charged Blast predicted with observed rules/context; triple Blast, Fireball, cannons and Mega actuation remain unsupported. Future item pickup/respawn changes require observed revalidation. Archived starts are development regressions. No full-round benchmark or calibrated landing guarantee.')
    atomic_json(folder/'decisions.json',logs);atomic_json(folder/'report.json',report)
    print(json.dumps({k:report[k] for k in ('map','seed','branch','actual_points','decisions','falls','stop_reason','median_planning_s','future_event_lock_revalidations')}),flush=True)
    return report


class ContinuousController:
    """Use the existing checked search throughout a native round, with counted fallback.

    The actor's recurrent state advances once per real observation. Hypothetical
    proposals use clones. Every reset cancels queued plans and inventory goals.
    No fixture restoration, grants or teleports occur during scored play.
    """
    def __init__(self,terrain,engine,checkpoint,args,kind):
        self.terrain=terrain;self.engine=engine;self.args=args;self.kind=kind
        if kind=='fast_navigator':
            from nav.marble_dynamics.fast_nav import FastNavigator
            self.nav=FastNavigator(engine.mesh,args.fast_checkpoint,terrain=terrain,base_checkpoint=checkpoint,cache_dir=getattr(args,'cache_dir',None))
        else:self.nav=FrozenNavigator(terrain,checkpoint)
        self.route=RoutePlanner(args.geometry_terrain if kind in ('physics_plan','physics_plan_reference','geometry_navigator','fast_navigator') and args.geometry_route else terrain)
        self.memory=PlanMemory();self.fields={};self.current=None;self.resource=None
        self.task_goal=None
        # Progress queries must not change the route used to produce actions.
        self.progress_route=RoutePlanner(self.route.terrain)
        self.actor_task_route=(RoutePlanner(self.route.terrain) if kind=='physics_plan'
            and getattr(args,'actor_task_during_recovery',False) else None)
        self.commit_left=0;self.cooldown=0;self.last_prediction=None;self.last_action=None
        self.counts=Counter();self.fallback_reasons=Counter();self.planning_times=[]
        self.observed_respawns=None
        engine.use_completed_candidates_on_timeout=True
        engine.navigator_first=getattr(args,'navigator_first',False)

    @property
    def uses_physics(self):
        return self.kind in ('physics_plan','physics_plan_reference')

    def invalidate(self,reason,reset_actor=False):
        self.memory.clear();self.current=None;self.resource=None;self.commit_left=0
        self.task_goal=None
        self.last_prediction=None;self.last_action=None;self.cooldown=0;self.route.reset()
        if getattr(self,'actor_task_route',None) is not None:self.actor_task_route.reset()
        self.observed_respawns=None
        self.counts['cancel:'+reason]+=1
        if reset_actor:
            if getattr(self,'kind',None)=='fast_navigator':
                # Stateless policy; keep the static map cache across respawns.
                self.nav.reset()
            else:self.nav=FrozenNavigator(self.terrain,model=self.nav.model)

    def observe(self,before,msg,outcome):
        if outcome['round_ended'] or outcome['reconnected'] or outcome['fell']:
            self.invalidate('round_or_fall',reset_actor=True);return
        game_context=msg.dynamics.get('game_context',{})
        if (game_context.get('available') and self.observed_respawns is not None
                and game_context['respawns']!=self.observed_respawns):
            self.invalidate('observed_native_respawn',reset_actor=True);return
        actual=msg.dynamics['native']['state'];self.nav.outcome(before,msg.obs)
        predicted=self.last_prediction
        events=external_changes(predicted,actual) if predicted else []
        if predicted:
            events+=blast_context_changes(predicted,msg.dynamics)
            events+=inventory_context_changes(predicted,msg.dynamics)
            if errors(predicted,actual)[0]>.05 and not events:events.append('position_prediction_gate')
        self.memory.advance(actual);self.commit_left=max(0,self.commit_left-1)
        picked=max(0.,float(outcome['gem_delta']))
        # Gem disappearance with a real pickup can retain a checked landing.
        if picked and not events and self.memory.observe_pickup(picked,msg.dynamics['items']):
            self.counts['post_pickup_landing']+=1;self.commit_left=0
        elif events or (picked and not self.memory.landing_mode):
            self.invalidate('observed_context_or_pickup')

    def _search(self,snapshot,raw,target,gems,remaining,skip_new_search=False):
        a=self.args;engine=self.engine;route=self.route;memory=self.memory
        state=snapshot['native']['state']
        if landing_complete(engine,state,memory):
            self.counts['landings_completed']+=1
            if memory.physical_recovery:
                self.counts['recoveries_completed']+=1
                next_target,_,_=tour(self.terrain,self.fields,raw,None)
                if next_target is not None:target=next_target
                elif gems:target=gems[0]
            memory.clear()
        if memory.landing_mode:
            retained=retained_landing(engine,route,snapshot,memory)
            if retained:
                if memory.physical_recovery:self.counts['recovery_revalidations']+=1
                return retained,memory.landing_goal+[0.,-1.]
            task=getattr(self,'task_goal',None)
            self.invalidate('landing_revalidation_failed');return None,task if task is not None else target
        if memory.locked:
            held=next((g for g in gems if np.linalg.norm(np.asarray(g[:3])-memory.gem)<.1),None)
            if held is not None:
                target=held;route.navigate(state['position'],target[:3])
                forecast=predict_replies(engine,snapshot,memory.remaining,stop_on_ambiguity=True)
                info=assess(engine,route,snapshot,np.asarray(target[:3]),forecast,memory.label,self.resource,
                            _stop_checked=memory.stopping_verified)
                if not info['rejected']:
                    self.counts['locked_revalidations']+=1;return (info,forecast),target
            memory.clear();self.commit_left=0
        if self.resource is not None:
            present=next((i for i in snapshot['items'] if i['id']==self.resource['id']),None)
            if present is None or state['held']==self.resource['powerup']:
                self.counts['resource_goal_observed_end']+=1;self.resource=None;memory.clear()
            else:
                self.resource=present;target=item_goal(present,np.asarray(state['position']));gems=[target]
        if a.reuse_plan:
            retained=retained_candidate(engine,route,snapshot,np.asarray(target[:3]),memory,target_item=self.resource)
            if retained:
                self.counts['remaining_plan_revalidations']+=1;return retained,target
        if skip_new_search:return None,target
        self.counts['full_searches']+=1
        chosen,pairs,target=pick(engine,route,snapshot,raw,gems,target,self.nav,a.horizon,memory,
            a.brake_continuation,a.recovery_horizon,self.resource,a.preserve_resources)
        if not chosen and a.resource_acquisition and self.resource is None:
            chosen,item,resource_target,_,_=choose_resource(engine,route,snapshot,raw,target,self.nav,
                max(a.horizon,a.recovery_horizon),a.brake_continuation,a.preserve_resources)
            if chosen:self.resource=item;target=resource_target;self.counts['resource_goals_selected']+=1
        if not chosen and a.recharge_continuation and self.resource is None:
            chosen,_=choose_recharge(engine,route,snapshot,target,remaining,float(raw[31]),
                                    max(a.horizon,a.recovery_horizon,80))
            if chosen:self.counts['recharge_plans_selected']+=1
        if not chosen and a.landing_search and self.resource is None:
            chosen,_=choose_landing(engine,route,snapshot,target,remaining,
                max(a.horizon,a.recovery_horizon,80),a.proposal_samples,a.proposal_seed,a.preserve_resources)
            if chosen:self.counts['landing_plans_selected']+=1
        if chosen:
            info,forecast=chosen;memory.accept(forecast,info['label'],target[:3],info)
        return chosen,target

    def action(self,msg,remaining):
        raw=msg.obs;dynamics=msg.dynamics;state=dynamics['native']['state']
        began_physical_recovery=self.memory.physical_recovery
        preserve_actor_task=(self.kind=='physics_plan' and getattr(self.args,'actor_task_during_recovery',False))
        context=dynamics.get('game_context',{})
        self.observed_respawns=context['respawns'] if context.get('available') else None
        target,nxt,gems=tour(self.terrain,self.fields,raw,self.current)
        if self.kind in ('physics_plan','physics_plan_reference','geometry_navigator','fast_navigator') and self.args.global_goals:
            gems=observed_gems(dynamics,raw)
            if target is None and gems:target=gems[0]
        if target is not None and (not self.memory.landing_mode or
                not destination_context(dynamics,getattr(self,'task_goal',None))['candidate_item_ids']):
            self.task_goal=list(target)
        if self.memory.landing_mode:target=self.memory.landing_goal+[0.,-1.]
        playable=state['mode']==0 and not state['oob']
        if not playable or target is None:
            self.last_prediction=None;self.last_action=None
            self.counts['neutral_nonplayable_or_no_goal']+=1
            return reply(NOOP_ACTION),dict(label='neutral',reason='nonplayable_or_no_goal')
        chosen=None;reason=None;t0=time.perf_counter()
        preserve_cooldown=(self.kind=='physics_plan' and getattr(self.args,'checked_recovery',False)
            and getattr(self.args,'recovery_preserve_search_cooldown',False))
        cooldown_before=self.cooldown
        if self.uses_physics:
            if self.cooldown and not (preserve_cooldown and self.memory.landing_mode):
                self.cooldown-=1;reason='search_cooldown'
            else:
                if preserve_cooldown and cooldown_before:self.cooldown-=1
                self.engine.planning_deadline=t0+self.args.search_budget_s if self.args.search_budget_s else None
                try:
                    if preserve_cooldown:
                        chosen,target=self._search(dynamics,raw,target,gems,remaining,
                            skip_new_search=bool(cooldown_before))
                    else:chosen,target=self._search(dynamics,raw,target,gems,remaining)
                except PlanningBudgetExceeded:reason='planning_time_budget'
                except ValueError as exc:reason='unsupported_context:'+str(exc)
                finally:self.engine.planning_deadline=None
                if not chosen and reason is None:
                    reason='search_cooldown' if preserve_cooldown and cooldown_before else 'no_checked_candidate'
                if not chosen and not (preserve_cooldown and cooldown_before):self.cooldown=self.args.fallback_cooldown
            if preserve_cooldown and cooldown_before:self.counts['recovery_search_cooldown_decisions']+=1
            self.planning_times.append(time.perf_counter()-t0)
        # A completed recovery can clear memory before search raises. Its local
        # gem target was never returned, so discard the stale outer endpoint.
        if preserve_actor_task and began_physical_recovery and not self.memory.landing_mode:
            task=getattr(self,'task_goal',None);task_context=destination_context(dynamics,task)
            if (task_context['kind'] in ('gem','powerup') and task_context['candidate_item_ids']
                    and not destination_context(dynamics,target)['candidate_item_ids']):
                target=list(task)
        if not self.memory.landing_mode and destination_context(dynamics,target)['candidate_item_ids']:
            self.task_goal=list(target)
        self.current=target;waypoint,_=self.route.navigate(state['position'],target[:3])
        next_goal=snap_to_walkable(self.terrain,*nxt[:3])[0] if nxt else None
        actor_target=target;actor_waypoint=waypoint;actor_task_preserved=False
        if preserve_actor_task and (self.memory.physical_recovery or began_physical_recovery):
            task=getattr(self,'task_goal',None);task_context=destination_context(dynamics,task)
            if task_context['kind'] in ('gem','powerup') and task_context['candidate_item_ids']:
                actor_target=task;actor_waypoint,_=self.actor_task_route.navigate(state['position'],task[:3])
                actor_task_preserved=True
        if self.kind=='fast_navigator':self.nav.set_observation(dynamics)
        actor_action=self.nav.act(raw,actor_waypoint,actor_target[:3],next_goal)
        actor_goal_context=dict(destination=list(map(float,actor_target[:3])),
            waypoint=list(map(float,actor_waypoint)),preserved_task_during_recovery=actor_task_preserved)
        landing_destination=self.memory.landing_mode
        recovery_audit=None
        if not chosen and self.kind=='physics_plan' and getattr(self.args,'checked_recovery',False):
            self.counts['recovery_attempts']+=1
            chosen,recovery_audit=ordinary_recovery(self.engine,dynamics,actor_action,
                replies=self.args.checked_recovery_replies,budget_s=self.args.checked_recovery_budget_s)
            if chosen:
                info,forecast=chosen
                self.memory.accept(forecast,info['label'],info['landing_goal'],info)
                self.memory.landing_mode=True;self.memory.physical_recovery=True
                self.memory.stopping_verified=True;self.resource=None
                if not preserve_cooldown:self.cooldown=0
                target=info['landing_goal']+[0.,-1.];self.current=target
                waypoint,_=self.route.navigate(state['position'],target[:3])
                landing_destination=True;self.counts['recoveries_selected']+=1
            else:self.counts['recovery_without_checked_backup']+=1
        if chosen:
            info,forecast=chosen;action=forecast['proposed'][0]
            if info.get('search_stopped_at_deadline'):self.counts['completed_candidate_after_deadline']+=1
            self.last_prediction=forecast['states'][1];self.counts['planner_decisions']+=1
        else:
            action=actor_action;self.last_prediction=None
            info=dict(label='A2_fallback' if self.uses_physics else self.kind,reason=reason)
            if self.uses_physics:
                self.memory.clear();self.resource=None;self.counts['fallback_decisions']+=1
                self.fallback_reasons[reason]+=1
        # Capture the real before-action destination and waypoint for all
        # branches, including fallback. Never infer these from later pickups.
        goal_context=destination_context(dynamics,target,
            ('physical_recovery' if self.memory.physical_recovery else 'geometric_landing') if landing_destination else None)
        info={**info,'goal_context':dict(**goal_context,
            waypoint=list(map(float,waypoint)),next_goal=list(map(float,next_goal)) if next_goal is not None else None,
            observed_gem_count=len(gems)),'actor_action':list(actor_action),'actor_goal_context':actor_goal_context,
            'task_goal_context':destination_context(dynamics,getattr(self,'task_goal',None))}
        if self.kind=='fast_navigator':
            info['base_actor_action']=self.nav.last_base_action
            info['unmasked_actor_action']=self.nav.last_unmasked_action
        if preserve_cooldown:
            info['primary_search_cooldown']=dict(before=cooldown_before,after=self.cooldown,
                retained_recovery_revalidation=bool(cooldown_before and chosen and info.get('revalidated_remainder')))
        if recovery_audit is not None:
            info['recovery_audit']=recovery_audit;info['primary_fallback_reason']=reason
        self.last_action=list(action);self.nav.sent_action(action,raw)
        return action,info


def destination_context(dynamics,target,kind=None):
    """Before-action item identity, preserving co-located gem ambiguity."""
    matching=[] if target is None else [item for item in dynamics.get('items',[])
        if np.linalg.norm(np.asarray(item['position'])-target[:3])<.1]
    inferred=('gem' if matching and all(i['gem'] for i in matching) else
        'powerup' if matching and all(not i['gem'] for i in matching) else 'unresolved_destination')
    return dict(destination=list(map(float,target[:3])) if target is not None else None,
        kind=kind or inferred,item_id=matching[0]['id'] if len(matching)==1 and kind is None else None,
        candidate_item_ids=[i['id'] for i in matching] if kind is None else [])


def continuous_branches(args,map_index,seed):
    if getattr(args,'fast_checkpoint',None):
        if getattr(args,'fast_only',False):return ('fast_navigator',)
        branches=('fast_navigator','geometry_navigator')
        return branches if (seed+map_index)%2==0 else branches[::-1]
    """Routing component comparison; balanced order declared before outcomes."""
    if getattr(args,'preflight_only',False):return ('physics_plan',)
    if getattr(args,'recovery_reference',False):
        branches=('physics_plan','physics_plan_reference')
        return branches if (seed+map_index)%2==0 else branches[::-1]
    if not getattr(args,'geometry_reference',False):return ('physics_plan','frozen_navigator')
    branches=('physics_plan','geometry_navigator','frozen_navigator')
    offset=(seed+map_index)%len(branches)
    return branches[offset:]+branches[:offset]


def await_full_round_context(env,max_wait=32):
    """Wait for requested live telemetry while the natural round is still fresh.

    A round restart can briefly emit observations before the new marble's native
    snapshot is available. Never use an older snapshot or silently start late.
    These neutral setup replies are counted separately from scored decisions.
    """
    duration=float(env.info.get('round_ms',0.))/1000.
    for waited in range(max_wait+1):
        if env.round_ended or env.reconnected:
            raise RuntimeError('Native round changed while awaiting live starting context')
        if (duration<=0 or float(env.msg.obs[RAW_SCORE])!=0.
                or not duration-.256<=env.time_left_s()<=duration+.01):
            raise RuntimeError('Natural full-round start is no longer fresh during telemetry setup')
        context=env.msg.dynamics
        world=getattr(env,'replay_world',None)
        if (isinstance(context,dict) and isinstance(context.get('native',{}).get('state'),dict)
                and context.get('game_context',{}).get('available')
                and isinstance(world,dict) and world.get('schema')==2
                and isinstance(world.get('header'),list) and 'random_opening' in world):
            return waited
        if waited<max_wait:
            _,outcome=env.step(NOOP_ACTION,repeat=1)
            if outcome['round_ended'] or outcome['reconnected']:
                raise RuntimeError('Native round changed while awaiting live starting context')
    raise RuntimeError('No audited live starting context after bounded neutral setup')


def attempt_stuck_rescue(env,controller,monitor,idle):
    """Avoid refused handshakes; only a real respawn resets actor history."""
    # Millisecond native clocks can lose a few ulps when converted to seconds.
    if idle<monitor.window_s-1e-9:return None
    game_context=env.msg.dynamics.get('game_context',{})
    if game_context.get('available') and game_context.get('can_quick_respawn') is False:
        controller.counts['rescue_blocked_by_game']+=1
        return None
    monitor.rescue_attempts+=1
    accepted=env.quick_rescue()
    controller.invalidate('legal_stuck_rescue' if accepted else 'legal_stuck_rescue_refused',
                          reset_actor=bool(accepted))
    if accepted:monitor.rescued(simulation_time(env),float('inf'))
    else:monitor.last_progress=simulation_time(env)
    return bool(accepted)


def start_layout(context,world):
    """Record actual visible gems and RNG; object IDs never define a layout."""
    gems=sorted([(item['position'],item['value']) for item in context['items'] if item['gem']])
    return dict(gems=gems,gem_sha256=hashlib.sha256(json.dumps(gems,separators=(',',':')).encode()).hexdigest(),
        position=context['native']['state']['position'],rng_state=world['header'][0] if world else None,
        applied_seed=world.get('eval_seed') if world else None,engine_seed=world.get('eval_engine_seed') if world else None,
        eval_round=world.get('eval_round') if world else None,
        random_opening=world.get('random_opening') if world else None)


def audit_start_layouts(reports,prior_pools=()):
    """Reject mismatched pairs or repeated confirmation openings, not outcomes."""
    groups={};failures=[];prior={}
    for folder in prior_pools:
        for report in json.loads((Path(folder)/'summary.json').read_text())['branches']:
            layout=report.get('start_layout')
            if layout:prior.setdefault(report['map'],set()).add(layout['gem_sha256'])
    for report in reports:
        layout=report.get('start_layout');key=(report['map'],report['seed'],report['round'])
        if not layout or not layout['gems'] or layout.get('random_opening') is not True:
            failures.append(dict(round=key,reason='Missing verified randomized opening'));continue
        expected=report.get('native_seed',report['seed'])+(layout.get('eval_round') or 0)-1
        if layout.get('applied_seed')!=expected or layout.get('engine_seed')!=expected:
            failures.append(dict(round=key,reason='Requested decimal seed did not reach native RNG exactly'))
        groups.setdefault(key,[]).append(layout)
    current={}
    for key,layouts in groups.items():
        signatures={r['gem_sha256'] for r in layouts}
        if len(signatures)!=1:failures.append(dict(round=key,reason='Controller opening layouts differ'))
        if len({(r.get('engine_seed'),r.get('rng_state')) for r in layouts})!=1:
            failures.append(dict(round=key,reason='Controller starting RNG states differ'))
        signature=layouts[0]['gem_sha256'];name=key[0]
        if signature in prior.get(name,set()) or signature in current.setdefault(name,set()):
            failures.append(dict(round=key,reason='Repeated confirmation opening'))
        current[name].add(signature)
    return dict(passed=not failures,failures=failures,round_groups=len(groups),
        contract='Actual visible gem layouts match across controllers and differ across declared confirmation starts. Full future RNG/world identity is not certified.')


def full_round_branch(env,proc,name,seed,kind,args,terrain,engine,checkpoint_hash,round_index):
    """Score original-duration native rounds; partial runs remain explicitly invalid."""
    folder=args.output/name/str(seed)/kind/str(round_index);folder.mkdir(parents=True,exist_ok=False)
    env.control('FIXEDSTEP '+str(args.physics_frame_ms)+' 64')
    command(env,'DYNAMICSLIVE 1','dynamics_live')
    env.replay_world=None
    env.control('REPLAYCAPTURE 1')
    startup_neutral_decisions=await_full_round_context(env)
    env.last_round_score=None
    controller=ContinuousController(terrain,engine,args.checkpoint,args,kind)
    monitor=ProgressMonitor(8.)
    initial=float(env.msg.obs[RAW_SCORE]);duration=float(env.info['round_ms'])/1000.
    start_left=env.time_left_s();context=copy.deepcopy(env.msg.dynamics)
    opening=start_layout(context,env.replay_world)
    native_seed=args.start_seeds[name]['seed'] if args.start_seed_manifest else seed
    if args.start_seed_manifest and opening['gem_sha256']!=args.start_seeds[name]['gem_sha256']:
        raise ValueError('Actual opening differs from predeclared native start manifest')
    if not context.get('game_context',{}).get('available'):
        raise RuntimeError('Full-round input audit requires live game-context counters')
    atomic_json(folder/'initial_context.json',dict(dynamics=context,raw=env.msg.obs.tolist(),
        start_time_left_s=start_left,initial_score=initial,round_seconds=duration,
        startup_neutral_decisions=startup_neutral_decisions))
    native=RUNTIME/(args.output.name+'_'+name+'_'+str(seed)+'_'+kind+'_'+str(round_index)+'.jsonl')
    journal=native.with_suffix('.decisions.jsonl');started=time.perf_counter()
    command(env,'DYNAMICSRECORD '+native.as_posix(),'dynamics_record')
    decisions=falls=0;points=0.;played=0.;uses=Counter();controls=[];latencies=[]
    stop='unknown';completed=False;in_fall=False
    with journal.open('w',encoding='utf-8',buffering=1) as stream:
        while decisions<int((duration+15.)/.064):
            if (args.output/'STOP').exists():stop='operator_stop';break
            if time.perf_counter()-started>=args.round_wall_limit_s:stop='round_wall_limit';break
            before=env.msg.obs.copy();dynamics=copy.deepcopy(env.msg.dynamics)
            before_s=simulation_time(env);countdown=env.time_left_s()>=duration
            remaining=max(1,int(env.time_left_s()/.064))
            t0=time.perf_counter()
            if countdown:
                action=reply(NOOP_ACTION);choice=dict(label='countdown_neutral')
                controller.last_prediction=None;controller.last_action=None
            else:action,choice=controller.action(env.msg,remaining)
            elapsed=time.perf_counter()-t0;latencies.append(elapsed)
            if controller.last_action is not None and decisions and controls and not pending_reply_matches(dynamics['bridge_action'],controls[-1]):
                stop='bridge_timing_contract_violation';break
            if action[6] or action[8]:uses[int(before[RAW_POW_HELD])]+=1
            msg,outcome=env.step(action[:6],repeat=1,use_pow=int(action[6]),
                pow_yaw=action[7] if action[6] else None,use_blast=int(action[8]))
            points+=max(0.,float(outcome['gem_delta']));decisions+=1
            stream.write(json.dumps(dict(decision=decisions,tick=msg.tick,before=dynamics,
                raw_observation=before.tolist(),action=action,choice=choice,latency_s=elapsed,
                predicted=controller.last_prediction,outcome=outcome))+'\n')
            controller.observe(before,msg,outcome);controls.append(list(action))
            if outcome['round_ended'] or outcome['reconnected']:
                completed=bool(outcome['round_ended'] and not outcome['reconnected'])
                stop='native_round_end' if completed else 'reconnected';break
            now=simulation_time(env)
            if outcome['fell']:
                if not in_fall:falls+=1
                in_fall=True
                env.control('OOBCLICK');controller.invalidate('legal_oob_click',reset_actor=True);controls.clear()
                points+=max(0.,float(env.control_info['gem_delta']))
                if env.control_info['round_ended'] or env.control_info['reconnected']:
                    completed=bool(env.control_info['round_ended'] and not env.control_info['reconnected'])
                    stop='native_round_end' if completed else 'reconnected';break
            else:in_fall=False
            raw=env.msg.obs;target=controller.task_goal;distance=float('inf')
            if target is not None:
                controller.progress_route.navigate(raw[:3],target[:3]);distance=controller.progress_route.distance(raw[:3])
            idle=monitor.observe(simulation_time(env),goal_key(target) if target is not None else None,distance,
                picked=outcome['gem_delta'],fell=outcome['fell'],gems=[goal_key(g) for g in visible_gems(raw)],
                groups_completed=(env.replay_world or {}).get('group_completions'))
            accepted=(attempt_stuck_rescue(env,controller,monitor,idle)
                      if not countdown and visible_gems(raw) and not outcome['fell'] else None)
            if accepted is not None:
                points+=max(0.,float(env.control_info['gem_delta']))
                controls.clear()
                if env.control_info['round_ended'] or env.control_info['reconnected']:
                    completed=bool(env.control_info['round_ended'] and not env.control_info['reconnected'])
                    stop='native_round_end' if completed else 'reconnected';break
            if not countdown:played+=max(0.,simulation_time(env)-before_s)
            if decisions%200==0:
                progress=dict(map=name,seed=seed,branch=kind,round=round_index,decisions=decisions,
                    points=points,time_left_s=env.time_left_s(),counts=dict(controller.counts),training=False)
                atomic_json(folder/'progress.json',progress);print(json.dumps(progress),flush=True)
        else:stop='decision_guard'
    score=env.last_round_score if completed else None
    atomic_json(folder/'final_context.json',dict(dynamics=env.msg.dynamics,raw=env.msg.obs.tolist(),completed=completed))
    command(env,'DYNAMICSRECORD off','dynamics_record')
    archive(native,folder/'native.jsonl.gz');archive(journal,folder/'decisions.jsonl.gz')
    valid=bool(completed and initial==0 and duration>0 and duration-.256<=start_left<=duration+.01
               and score is not None and math.isfinite(score) and score>=0 and abs(points-score)<.01)
    final_game=env.msg.dynamics.get('game_context',{})
    implicit_keys=(final_game['automatic_oob_key_requests']-context['game_context']['automatic_oob_key_requests']) if final_game.get('available') else None
    inputs_valid=implicit_keys==0
    report=dict(map=name,seed=seed,branch=kind,round=round_index,completed=completed,score_valid=valid,
        actual_points=score,accumulated_points=points,initial_score=initial,round_seconds=duration,
        automatic_oob_key_requests=implicit_keys,declared_inputs_valid=inputs_valid,
        start_time_left_s=start_left,start_context=context,start_layout=opening,native_seed=native_seed,played_seconds=played,decisions=decisions,
        startup_neutral_decisions=startup_neutral_decisions,
        falls=falls,stop_reason=stop,counts=dict(controller.counts),fallback_reasons=dict(controller.fallback_reasons),
        pow_use_commands=dict(uses),recovery=monitor.stats(),physics_frame_ms=args.physics_frame_ms,decision_ms=64,
        checkpoint_sha256=checkpoint_hash,protected_checkpoint_unchanged=sha(args.checkpoint)==checkpoint_hash,
        fast_checkpoint_sha256=sha(args.fast_checkpoint) if kind=='fast_navigator' else None,
        fast_axis_mode=controller.nav.model.axis_mode if kind=='fast_navigator' else None,
        fast_decoder=controller.nav.decoder if kind=='fast_navigator' else None,
        compiled_sweep_enabled=hasattr(engine.mesh,'kernel'),query_cache_enabled=args.query_cache,
        clearance_cache_enabled=args.clearance_cache,
        geometry_routing=args.geometry_terrain.geometry_route_report if args.geometry_route else None,
        median_decision_s=float(np.median(latencies)) if latencies else None,
        p95_decision_s=float(np.percentile(latencies,95)) if latencies else None,
        max_decision_s=max(latencies,default=None),elapsed_wall_s=time.perf_counter()-started,
        native_raw_sha256=sha(native),owned_game_pid=proc.pid,policy_training=False,
        protocol='Complete original-duration Hunt; competitive off; legal OOB click and 8s no-progress rescue; no restores/grants/teleports during scored play.',
        comparison_contract='Experimental physics cadence with matching A2 branch/rules. Separate actual native starting worlds; supplied seeds do not certify exact pairing. This hybrid is not the legacy real_run controller or the archived first-fall benchmark.')
    atomic_json(folder/'report.json',report)
    if not valid:raise RuntimeError('Invalid full-round result: '+stop+' '+str(report['actual_points']))
    if not inputs_valid:raise RuntimeError('Implicit OOB key input outside declared controller: '+str(implicit_keys))
    print(json.dumps({k:report[k] for k in ('map','branch','actual_points','falls','counts','elapsed_wall_s')}),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--maps',nargs='+',default=['KingOfTheMarble_Hunt','Sprawl_Hunt','Bowl_Hunt'])
    p.add_argument('--seeds',nargs='+',type=int,default=[15407,24462]);p.add_argument('--decisions',type=int,default=160)
    p.add_argument('--random-starts',action='store_true',help='Opt-in ordinary random Hunt opening instead of highest-value first spawn; audit actual layouts')
    p.add_argument('--distinct-from-pools',nargs='+',type=Path,default=[],help='Reject opening layouts repeated from these already completed randomized confirmation pools')
    p.add_argument('--start-seed-manifest',type=Path,help='Predeclared per-map native seeds and opening hashes selected by layout-only preflight')
    p.add_argument('--horizon',type=int,default=40);p.add_argument('--commit',type=int,default=4);p.add_argument('--port',type=int,default=9054)
    p.add_argument('--physics-frame-ms',type=int,choices=[16,64],default=64,help='Native-verified physics cadence with unchanged64ms policy decisions;16ms experimental,64ms existing default')
    p.add_argument('--search-budget-s',type=float,default=0,help='Optional bound per planning iteration; stop before sending any partial candidate on expiry;0 leaves existing search unbounded')
    p.add_argument('--checked-recovery',action='store_true',help='Full rounds only: complete physical stopping backups before unchecked actor fallback')
    p.add_argument('--checked-recovery-replies',type=int,default=40)
    p.add_argument('--checked-recovery-budget-s',type=float,default=.2)
    p.add_argument('--actor-task-during-recovery',action='store_true',help='Opt-in: actor retains observed task and independent route during physical stopping backup')
    p.add_argument('--recovery-preserve-search-cooldown',action='store_true',help='Opt-in: retained recovery revalidates independently of the primary failed-search cooldown')
    p.add_argument('--recovery-reference',action='store_true',help='Full rounds: compare same combined controller with and without checked recovery')
    p.add_argument('--preflight-only',action='store_true',help='Engineering native preflight: full checked-recovery controller only, no comparison mean')
    p.add_argument('--reuse-plan',action='store_true',help='Revalidate safe useful remaining commands before a fresh full search')
    p.add_argument('--brake-continuation',type=int,default=0,help='Extend unambiguous unsupported futures by this many feedback braking replies')
    p.add_argument('--recovery-horizon',type=int,default=0,help='Only after no useful checked candidate, search once with this longer horizon')
    p.add_argument('--geometry-route',action='store_true',help='Planner only: wall/ceiling-filtered walking proposals; comparator retains original routing')
    p.add_argument('--resource-acquisition',action='store_true',help='After no useful gem plan, use explicit conditional inventory hypotheses to choose an actual visible powerup acquisition')
    p.add_argument('--preserve-resources',action='store_true',help='Among useful checked futures, prefer pickup then fewer modeled resource consumptions before route cost')
    p.add_argument('--recharge-continuation',action='store_true',help='After gem/resource search fails, predict a bounded brake/wait/recharge/Blast pickup and landing from real state')
    p.add_argument('--landing-search',action='store_true',help='After other checks fail, sample general controls and whole-map support points; preserve/revalidate a selected landing after actual gem collection')
    p.add_argument('--proposal-seed',type=int,default=997);p.add_argument('--proposal-samples',type=int,default=40)
    p.add_argument('--query-cache',action='store_true',help='Bounded exact static AABB cache; every result retains original six comparisons')
    p.add_argument('--clearance-cache',action='store_true',help='Memoize exact repeated static walking-clearance endpoints/direction/radius')
    p.add_argument('--global-goals',action='store_true',help='After nearest-five goals fail, check every other currently visible native gem with the same physical search')
    p.add_argument('--sweep-dll',type=Path,help='Explicit independently validated strict-math analytical sweep DLL; otherwise original Python geometry')
    p.add_argument('--short-horizons',nargs='+',type=int,default=[],help='After all longer continuations fail, check smaller useful supported plans with every original physics/ambiguity gate')
    p.add_argument('--terminal-stop-replies',type=int,default=0,help='Require and retain a verified ordinary braking backup after each nominally supported candidate; default disabled')
    p.add_argument('--terminal-stop-scope',choices=['pickup','all'],default='pickup',help='Check target-pickup futures and post-pickup landings, or every progress future (more conservative)')
    p.add_argument('--checkpoint',type=Path,default=ML/'models/nav/real_maps_20261010_A2/nav_latest.pth')
    p.add_argument('--fast-checkpoint',type=Path,help='Opt-in full-map learned navigator; compare with geometry-routed A2, no expensive action search')
    p.add_argument('--fast-only',action='store_true',help='Engineering single-controller smoke test, not comparative evidence')
    p.add_argument('--fixture-data',type=Path,help='Reuse archived map/seed/fixture.json starting worlds for exact development regressions; never label these fresh validation')
    p.add_argument('--full-round',action='store_true',help='Separate continuous original-duration assisted hybrid/A2 evaluation; never first-fall scoring')
    p.add_argument('--navigator-first',action='store_true',help='Continuous search: physically check the frozen learned proposal before hand-built controls; acceptance gates unchanged')
    p.add_argument('--geometry-reference',action='store_true',help='Full rounds: add frozen A2 with geometry routing and no physical search; cyclic branch order across maps')
    p.add_argument('--rounds',type=int,default=1)
    p.add_argument('--round-wall-limit-s',type=float,default=1200.)
    p.add_argument('--fallback-cooldown',type=int,default=32,help='Real decisions between failed full searches; fallback is explicitly counted')
    p.add_argument('--output',type=Path,default=ML/'dynamics_reports/reliability_20261012')
    p.add_argument('--workers',type=int,default=1,help='Parallel native branches; maximum8 total games')
    p.add_argument('--cache-dir',type=Path,help='Fingerprint-bound persistent static graph/query/encoding cache')
    p.add_argument('--only-branch',choices=['fast_navigator','geometry_navigator','physics_plan','frozen_navigator','physics_plan_reference'])
    p.add_argument('--map-index-offset',type=int,default=0)
    p.add_argument('--prepared-parent',type=Path,help=argparse.SUPPRESS)
    a=p.parse_args()
    if not 1<=a.workers<=8:p.error('Workers must be1?8')
    if a.workers>1 and not a.full_round:p.error('Parallel workers require full rounds')
    if a.cache_dir and not (a.sweep_dll and a.query_cache and a.geometry_route):p.error('Persistent caches require compiled/query/geometry routing')
    if a.only_branch and not a.full_round:p.error('Branch restriction requires full rounds')
    if a.checked_recovery and not a.full_round:raise ValueError('Checked recovery requires natural full-round evaluation')
    if (a.actor_task_during_recovery or a.recovery_preserve_search_cooldown) and not a.checked_recovery:
        raise ValueError('Recovery task/cooldown variants require checked recovery')
    if a.preflight_only and (not a.full_round or not a.checked_recovery or a.recovery_reference or a.geometry_reference):raise ValueError('Preflight-only requires full checked-recovery rounds without comparison flags')
    if a.recovery_reference and (not a.checked_recovery or not a.full_round or a.geometry_reference):raise ValueError('Recovery reference requires checked recovery/full rounds, without geometry reference')
    if a.checked_recovery_replies<4 or a.checked_recovery_budget_s<=0:raise ValueError('Positive complete recovery limits required')
    if a.brake_continuation<0 or a.recovery_horizon<0 or a.proposal_samples<1:p.error('Continuation/recovery lengths must be nonnegative; proposal count must be positive')
    if a.clearance_cache and not a.geometry_route:p.error('--clearance-cache requires --geometry-route')
    if any(h<4 or h>=a.horizon for h in a.short_horizons):p.error('Short horizons must be at least4 and below the primary horizon')
    if a.terminal_stop_replies and a.terminal_stop_replies<4:p.error('Terminal stopping check must be disabled or at least4 replies')
    if a.search_budget_s<0 or not math.isfinite(a.search_budget_s):p.error('Search budget must be finite and nonnegative')
    if a.full_round and a.fixture_data:p.error('Full rounds use natural native starts, never archived restores')
    if a.random_starts and not a.full_round:p.error('Randomized opening requires full rounds')
    if a.distinct_from_pools and not a.random_starts:p.error('Distinct-opening checks require --random-starts')
    if a.start_seed_manifest and not a.random_starts:p.error('Opening seed manifest requires --random-starts')
    a.start_seeds=json.loads(a.start_seed_manifest.read_text())['maps'] if a.start_seed_manifest else {}
    if a.start_seed_manifest and any(name not in a.start_seeds for name in a.maps):p.error('Opening seed manifest does not cover every map')
    if a.navigator_first and not a.full_round:p.error('--navigator-first applies only to continuous rounds')
    if a.geometry_reference and not (a.full_round and a.geometry_route):p.error('--geometry-reference requires --full-round --geometry-route')
    if a.fast_checkpoint and not (a.full_round and a.geometry_route):p.error('--fast-checkpoint requires full rounds and geometry routing')
    if a.fast_only and not a.fast_checkpoint:p.error('--fast-only requires a fast checkpoint')
    if a.rounds<1 or a.round_wall_limit_s<=0 or a.fallback_cooldown<0:p.error('Invalid full-round bounds')
    a.output.mkdir(parents=True,exist_ok=False);fingerprint=sha(a.checkpoint);results=[]
    sources={str(p):sha(p) for p in Path(__file__).parent.glob('*.py')}
    if a.start_seed_manifest:sources[str(a.start_seed_manifest)]=sha(a.start_seed_manifest)
    if a.fast_checkpoint:
        a.fast_checkpoint=a.fast_checkpoint.resolve()
        sources[str(a.fast_checkpoint)]=sha(a.fast_checkpoint)
    for name in a.maps:
        for path in (TerrainMap.resolve(name),ML/'collision_maps'/(name+'.npz'),ML/'collision_maps'/(name+'.json')):
            sources[str(path)]=sha(path)
    # Capture imported actor/bridge/recovery modules as well as dynamics code.
    sources.update({str(p):sha(p) for p in (ML/'nav').glob('*.py')})
    for path in (ML.parent/'Marble Blast Platinum/platinum/client/scripts/ai/mlAgent.cs',
                 ML.parent/'Marble Blast Platinum/platinum/client/scripts/ai/trainingReplay.cs',
                 ML.parent/'Marble Blast Platinum/platinum/server/scripts/modes/hunt.cs',
                 ML.parent/'Marble Blast Platinum/platinum/server/scripts/huntGems.cs',
                 ML.parent/'Marble Blast Platinum/marbleblast_geometry_probe.exe',Path(__file__).with_name('engine_dynamics.inc'),
                 ML/'nav/routing.py',ML/'nav/terrain.py',ML/'nav/collision/mesh.py'):
        sources[str(path)]=sha(path)
    sources[str(ML/'nav/source_archive.py')]=sha(ML/'nav/source_archive.py')
    if a.sweep_dll:
        for path in (a.sweep_dll,ML/'nav/collision/experimental_sweep.py',ML/'nav/collision/sweep_kernel.cpp',ML/'nav/collision/build_sweep_kernel.ps1'):
            sources[str(path)]=sha(path)
        dest=a.output/'external_artifacts'/a.sweep_dll.name;dest.parent.mkdir();dest.write_bytes(a.sweep_dll.read_bytes())
    # Preserve executable Python/script source alongside hashes so later edits
    # cannot erase the exact implementation that produced this batch.
    if a.prepared_parent:
        parent=json.loads((a.prepared_parent/'prepared.json').read_text())
        if any(parent['source_hashes'].get(source)!=digest for source,digest in sources.items()):
            raise ValueError('Parallel child inputs differ from parent declaration')
        os.link(str(a.prepared_parent/'source_snapshot.zip'),str(a.output/'source_snapshot.zip'))
    else:
        snapshot_sources(a.output,[Path(source) for source in sources
            if Path(source).suffix.lower() in ('.py','.cs','.inc','.cpp','.ps1')])
    atomic_json(a.output/'prepared.json',dict(maps=a.maps,seeds=a.seeds,random_starts=a.random_starts,start_seed_manifest=str(a.start_seed_manifest) if a.start_seed_manifest else None,distinct_from_pools=list(map(str,a.distinct_from_pools)),parallel_workers=a.workers,cache_dir=str(a.cache_dir) if a.cache_dir else None,only_branch=a.only_branch,decisions=a.decisions,horizon=a.horizon,commit=a.commit,reuse_plan=a.reuse_plan,brake_continuation=a.brake_continuation,recovery_horizon=a.recovery_horizon,geometry_route=a.geometry_route,resource_acquisition=a.resource_acquisition,preserve_resources=a.preserve_resources,recharge_continuation=a.recharge_continuation,landing_search=a.landing_search,proposal_seed=a.proposal_seed,proposal_samples=a.proposal_samples,regular_blast_prediction=True,regular_blast_proposals=True,
                checkpoint=str(a.checkpoint),checkpoint_sha256=fingerprint,source_hashes=sources,fixture_data=str(a.fixture_data) if a.fixture_data else None,special_blast_prediction=True,
                query_cache=a.query_cache,clearance_cache=a.clearance_cache,global_goals=a.global_goals,
                sweep_dll=str(a.sweep_dll) if a.sweep_dll else None,
                short_horizons=a.short_horizons,
                terminal_stop_replies=a.terminal_stop_replies,
                terminal_stop_scope=a.terminal_stop_scope,
                physics_frame_ms=a.physics_frame_ms,decision_ms=64,
                search_budget_s=a.search_budget_s,
                checked_recovery=a.checked_recovery,checked_recovery_replies=a.checked_recovery_replies,
                checked_recovery_budget_s=a.checked_recovery_budget_s,recovery_reference=a.recovery_reference,
                actor_task_during_recovery=a.actor_task_during_recovery,
                recovery_preserve_search_cooldown=a.recovery_preserve_search_cooldown,
                preflight_only=a.preflight_only,
                full_round=a.full_round,rounds=a.rounds,round_wall_limit_s=a.round_wall_limit_s,
                completed_candidates_on_deadline=a.full_round,
                navigator_first=a.navigator_first,
                geometry_reference=a.geometry_reference,
                fast_checkpoint=str(a.fast_checkpoint) if a.fast_checkpoint else None,
                fast_only=a.fast_only,
                continuous_branch_order='fast_navigator only (engineering smoke)' if a.fast_only else 'Alternating fast_navigator/geometry_navigator, offset(seed+map_index)%2' if a.fast_checkpoint else 'physics_plan only (engineering preflight)' if a.preflight_only else 'Alternating physics_plan/physics_plan_reference, offset(seed+map_index)%2' if a.recovery_reference else 'Cyclic physics_plan/geometry_navigator/frozen_navigator, offset(seed+map_index)%3' if a.geometry_reference else 'physics_plan/frozen_navigator',
                rescue_progress_contract='real gem/resource destination; per-target credited-distance anchors; recovery/landing subgoals earn no credit',
                rescue_refusal_contract='Skip explicitly unavailable quick respawn; reset actor only after acceptance; 1ns tolerance at the 8s native-clock threshold',
                fallback_cooldown=a.fallback_cooldown,
                selection='Reused archived starting worlds: development regression' if a.fixture_data else 'Predeclared fresh-seed validation on familiar geometry; no outcome-based fixture selection',training=False))
    if a.workers>1:
        from nav.marble_dynamics.scaling_loop import parallel_pool
        parallel_pool(a,sys.argv[1:])
        if a.random_starts:
            starts=audit_start_layouts(json.loads((a.output/'summary.json').read_text())['branches'],a.distinct_from_pools)
            atomic_json(a.output/'start_layout_audit.json',starts)
            if not starts['passed']:raise ValueError('Randomized opening audit failed: '+str(starts['failures']))
        changed=[source for source,digest in sources.items() if sha(source)!=digest]
        atomic_json(a.output/'input_integrity.json',dict(passed=not changed,changed=changed,source_count=len(sources)))
        if changed:raise ValueError('Parallel pool input mutation')
        return
    # Warm/load the immutable policy before connecting an owned game.
    for map_index,name in enumerate(a.maps):
        if a.cache_dir:
            from nav.marble_dynamics.static_cache import cached_terrain,cached_walking,load_queries,save_queries,save_clearance
            terrain=cached_terrain(TerrainMap.resolve(name),a.cache_dir)
        else:terrain=TerrainGrid(TerrainMap.resolve(name))
        path=ML/'collision_maps'/(name+'.npz')
        mesh=(CachedCompiledSweepMap if a.query_cache else CompiledSweepMap)(path,a.sweep_dll) if a.sweep_dll else (CachedCollisionMap if a.query_cache else CollisionMap)(path)
        if a.cache_dir:load_queries(mesh,a.sweep_dll,a.cache_dir)
        engine=GeneralPhysics(mesh)
        engine.bridge_physics_ms=a.physics_frame_ms
        engine.terminal_stop_replies=a.terminal_stop_replies
        engine.terminal_stop_scope=a.terminal_stop_scope
        if a.geometry_route:
            a.geometry_terrain=cached_walking(terrain,mesh,a.sweep_dll,a.cache_dir,memo=a.clearance_cache) if a.cache_dir else walking_terrain(terrain,engine.mesh)
            if a.clearance_cache and not a.cache_dir:a.geometry_terrain.ground_clearance=MemoGroundClearance(a.geometry_terrain.ground_clearance)
            atomic_json(a.output/(name+'.geometry_route.json'),a.geometry_terrain.geometry_route_report)
        FrozenNavigator(terrain,a.checkpoint)
        for seed in a.seeds:
            # New native games load scripts from disk. Refuse a mixed-version
            # batch if source changed since the archived declaration.
            if any(sha(Path(source))!=fingerprint for source,fingerprint in sources.items()):
                raise RuntimeError('Batch source changed after preparation; use a new output for the new version')
            folder=a.output/name/str(seed);folder.mkdir(parents=True)
            if a.full_round:
                kinds=continuous_branches(a,map_index+a.map_index_offset,seed)
                if a.only_branch:
                    if a.only_branch not in kinds:raise ValueError('Branch not in declared comparison')
                    kinds=(a.only_branch,)
                for kind in kinds:
                    if any(sha(Path(source))!=digest for source,digest in sources.items()):
                        raise RuntimeError('Full-round source changed during declared batch')
                    native_seed=a.start_seeds[name]['seed'] if a.start_seed_manifest else seed
                    with game(name,a.port,obs_ms=64,seed=native_seed,process_cap=8,random_start=a.random_starts) as (env,proc):
                        env.set_game_rules(False)
                        # The mechanics game factory warms up midround. Wait for
                        # the NEXT actual native round; do not pretend that warm
                        # prefix or a REPLAYRESTORE is a complete natural start.
                        while not env.round_ended:
                            _,outcome=env.step(NOOP_ACTION,repeat=1)
                            if outcome['reconnected']:raise RuntimeError('Reconnect during pre-round warmup')
                        env.wait_new_round()
                        for ri in range(1,a.rounds+1):
                            try:
                                result=full_round_branch(env,proc,name,seed,kind,a,terrain,engine,fingerprint,ri)
                            except Exception as exc:
                                dest=folder/kind/str(ri);dest.mkdir(parents=True,exist_ok=True)
                                result=dict(map=name,seed=seed,branch=kind,round=ri,status='invalid',
                                    score_valid=False,reason=repr(exc),policy_training=False)
                                atomic_json(dest/'failure_context.json',dict(
                                    dynamics=env.msg.dynamics,raw=env.msg.obs.tolist(),
                                    time_left_s=env.time_left_s(),round_ended=env.round_ended,
                                    reconnected=env.reconnected,recent_lines=env.recent_lines))
                                stem=a.output.name+'_'+name+'_'+str(seed)+'_'+kind+'_'+str(ri)
                                raw=RUNTIME/(stem+'.jsonl');journal=raw.with_suffix('.decisions.jsonl')
                                for path,label in ((raw,'native'),(journal,'decisions')):
                                    if path.exists():archive(path,dest/(label+'.failed.jsonl.gz'))
                                atomic_json(dest/'failed.json',result)
                                results.append(result);atomic_json(a.output/'summary.json',dict(branches=results,
                                    full_round=True,policy_training_started=False))
                                raise
                            results.append(result)
                            if sha(a.checkpoint)!=fingerprint:raise RuntimeError('Protected checkpoint changed')
                            atomic_json(a.output/'summary.json',dict(branches=results,checkpoint_unchanged=True,
                                policy_training_started=False,full_round=True))
                            if ri<a.rounds:env.wait_new_round()
                    if any(sha(Path(source))!=digest for source,digest in sources.items()):
                        raise RuntimeError('Full-round source changed during declared batch')
                if a.cache_dir:
                    save_queries(mesh);save_clearance(a.geometry_terrain)
                continue
            with game(name,a.port,obs_ms=64,seed=seed,process_cap=8) as (env,proc):
                if a.physics_frame_ms==16:env.control('FIXEDSTEP 16 64')
                command(env,'DYNAMICSLIVE 1','dynamics_live');env.control('REPLAYCAPTURE 1')
                for _ in range(8):env.step(NOOP_ACTION,repeat=1)
                world=copy.deepcopy(env.replay_world);world['native_reference']=copy.deepcopy(env.msg.dynamics['native']['state'])
                fixture_source=None
                if a.fixture_data:
                    fixture_path=a.fixture_data/name/str(seed)/'fixture.json'
                    original=json.loads(fixture_path.read_text());world=copy.deepcopy(original['world'])
                    restore(env,world)
                    fixture_source=dict(path=str(fixture_path),sha256=sha(fixture_path),development_regression=True)
                atomic_json(folder/'fixture.json',dict(world=world,catalog=env.replay_catalog,dynamics=env.msg.dynamics,source=fixture_source))
                branches=('frozen_navigator','physics_plan') if seed%2 else ('physics_plan','frozen_navigator')
                for kind in branches:
                    try:
                        result=branch(env,proc,world,name,seed,kind,a,terrain,engine,fingerprint)
                    except Exception as exc:
                        dest=folder/kind;dest.mkdir(exist_ok=True)
                        try:command(env,'DYNAMICSRECORD off','dynamics_record')
                        except Exception:pass
                        stem=a.output.name+'_'+name+'_'+str(seed)+'_'+kind
                        raw=RUNTIME/(stem+'.jsonl');journal=RUNTIME/(stem+'.decisions.jsonl')
                        if raw.exists():archive(raw,dest/'native.failed.jsonl.gz')
                        if journal.exists():
                            archived=[json.loads(l) for l in journal.read_text().splitlines()]
                            atomic_json(dest/'decisions.failed.json',archived)
                        result=dict(map=name,seed=seed,branch=kind,status='invalid',reason=repr(exc),policy_training_started=False)
                        atomic_json(dest/'failed.json',result)
                        results.append(result);atomic_json(a.output/'summary.json',dict(branches=results,checkpoint_unchanged=sha(a.checkpoint)==fingerprint,policy_training_started=False))
                        raise
                    results.append(result)
                    if sha(a.checkpoint)!=fingerprint:raise RuntimeError('Protected checkpoint changed')
                    atomic_json(a.output/'summary.json',dict(branches=results,checkpoint_unchanged=True,policy_training_started=False))
                    if (a.output/'STOP').exists():return


if __name__=='__main__':main()
