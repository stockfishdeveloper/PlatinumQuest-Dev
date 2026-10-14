"""Bounded matched fresh-seed validation with a protected frozen navigator.

    python -m nav.marble_dynamics.reliability_eval --maps KingOfTheMarble_Hunt Sprawl_Hunt Bowl_Hunt

No training, checkpoint writes, rescue, artificial gem placement or maneuver rules.
Both branches stop at first fall or fixed decision budget. Validation seeds/maps
are familiar geometry; they are not a claim of unseen-map generalization.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import time

import numpy as np

from nav.collision.mesh import CollisionMap,atomic_json
from nav.gems import visible_gems,plan_tour
from nav.routing import RoutePlanner,goal_key,snap_to_walkable
from nav.terrain import TerrainMap,TerrainGrid
from nav.protocol import NOOP_ACTION,RAW_SCORE
from nav.marble_dynamics.record import ML,RUNTIME,game,command,archive,sha
from nav.marble_dynamics.commit_demo import restore
from nav.marble_dynamics.physics import GeneralPhysics
from nav.marble_dynamics.bridge import predict_replies
from nav.marble_dynamics.evaluate import errors
from nav.marble_dynamics.events import external_changes
from nav.marble_dynamics.navigator import FrozenNavigator
from nav.marble_dynamics.search import search,assess,PlanMemory,retained_candidate


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


def pick(engine,route,dynamics,raw,gems,current,nav,horizon,memory):
    position=dynamics['native']['state']['position'];all_pairs=[];options=[]
    for index,gem in enumerate([current]+[g for g in gems if np.linalg.norm(np.asarray(g[:3])-current[:3])>.1]):
        route.navigate(position,gem[:3]);start=route.distance(position)
        warm=memory.remaining if memory.gem and np.linalg.norm(np.asarray(memory.gem)-gem[:3])<.1 else None
        chosen,pairs=search(engine,route,dynamics,raw,gem[:3],nav,horizon,warm=warm)
        all_pairs.extend([{**p[0],'gem':list(gem[:3])} for p in pairs])
        if chosen:
            info,rollout=chosen;progress=start-route.distance(rollout['states'][-1]['position'])
            pickup=info.get('possible_pickup_interval')
            if pickup is not None or progress>.3:
                rank=(int(pickup is not None),float(gem[3])/max((pickup or horizon)*.064,.064),progress)
                options.append((rank,chosen,gem))
                if index==0:break
    if not options:return None,all_pairs,current
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
    nav=FrozenNavigator(terrain,args.checkpoint);route=RoutePlanner(terrain);fields={};memory=PlanMemory()
    raw_file=RUNTIME/(args.output.name+'_'+name+'_'+str(seed)+'_'+kind+'.jsonl')
    command(env,'DYNAMICSRECORD '+raw_file.as_posix(),'dynamics_record')
    initial=float(env.msg.obs[RAW_SCORE]);logs=[];pickups=[];searches=[];current=None;stop='decision_budget'
    started=time.perf_counter();lock_steps=0;reuse_steps=0;full_search_steps=0;cancelled=[]
    journal=(RUNTIME/(args.output.name+'_'+name+'_'+str(seed)+'_'+kind+'.decisions.jsonl')).open('w',encoding='utf-8',buffering=1)
    while len(logs)<args.decisions:
        if (args.output/'STOP').exists():stop='operator_stop';break
        snapshot=copy.deepcopy(env.msg.dynamics);state=snapshot['native']['state'];raw=env.msg.obs.copy()
        if state['mode']!=0 or state['oob']:stop='unsupported_state';break
        target,nxt,gems=tour(terrain,fields,raw,current)
        if target is None:stop='no_live_gem';break
        t0=time.perf_counter();pairs=[];reused=False
        if kind=='physics_plan':
            if memory.locked and any(np.linalg.norm(np.asarray(g[:3])-memory.gem)<.1 for g in gems):
                current=next(g for g in gems if np.linalg.norm(np.asarray(g[:3])-memory.gem)<.1)
                route.navigate(state['position'],current[:3])
                forecast=predict_replies(engine,snapshot,memory.remaining)
                info=assess(engine,route,snapshot,np.asarray(current[:3]),forecast,memory.label)
                if info['rejected']:
                    cancelled.append(dict(decision=len(logs),reason=info['rejected']));memory.clear()
                else:lock_steps+=1
            else:memory.lock_until=0
            if not memory.locked and args.reuse_plan:
                retained=retained_candidate(engine,route,snapshot,np.asarray(target[:3]),memory)
                if retained:
                    info,forecast=retained;current=target;reused=True;reuse_steps+=1
            if not memory.locked and not reused:
                full_search_steps+=1
                chosen,pairs,current=pick(engine,route,snapshot,raw,gems,target,nav,args.horizon,memory)
                if not chosen:
                    atomic_json(folder/'terminal_search.json',dict(decision=len(logs),dynamics=snapshot,
                        raw_observation=raw.tolist(),target=list(target),candidates=pairs))
                    stop='no_candidate_passed_checks';break
                info,forecast=chosen;memory.accept(forecast,info['label'],current[:3])
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
        elapsed=time.perf_counter()-t0;searches.append(elapsed)
        for i in range(commit):
            snapshot=copy.deepcopy(env.msg.dynamics);before=env.msg.obs.copy();tick=env.msg.tick
            if logs and not pending_reply_matches(snapshot['bridge_action'],logs[-1]['action']):
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
            delta=float(msg.obs[RAW_SCORE])-float(before[RAW_SCORE])
            if delta>0:pickups.append(dict(decision=len(logs),points=delta,position=actual['position']))
            logs.append(dict(decision=len(logs),tick=tick,dynamics=snapshot,raw_observation=before.tolist(),action=action,chosen=info,candidates=pairs if i==0 else [],
                             planning_s=elapsed if i==0 else 0.,predicted_next=predicted,actual_next=actual,errors=error,
                             external_events=events,score_delta=delta,outcome=outcome,gem=list(current),waypoint=list(waypoint),
                             lock_remaining=max(0,memory.lock_until-memory.executed)))
            journal.write(json.dumps(logs[-1])+'\n')
            memory.advance()
            if outcome['fell'] or outcome['round_ended'] or outcome['reconnected']:stop='game_or_fall_event';break
            if kind=='physics_plan' and error and error[0]>.05 and not events:stop='prediction_error_gate';break
            if delta>0 or events:
                if memory.locked:cancelled.append(dict(decision=len(logs)-1,reason='pickup_or_external_event'))
                memory.clear();break
        atomic_json(folder/'progress.json',dict(decisions=len(logs),points=float(env.msg.obs[RAW_SCORE])-initial,stop=stop))
        if len(searches)%6==0:print(json.dumps(dict(map=name,seed=seed,branch=kind,decisions=len(logs),points=float(env.msg.obs[RAW_SCORE])-initial,choice=info['label'])),flush=True)
        if stop!='decision_budget':break
    journal.close()
    last_scored=float(env.msg.obs[RAW_SCORE]);command(env,'DYNAMICSRECORD off','dynamics_record');archive(raw_file,folder/'native.jsonl.gz')
    final=float(env.msg.obs[RAW_SCORE]);closing=final-last_scored
    if closing:pickups.append(dict(decision=len(logs),points=closing,phase='recording_close'))
    report=dict(map=name,seed=seed,branch=kind,actual_points=last_scored-initial,initial_score=initial,final_score=last_scored,
                recording_close_score_delta=closing,actual_pickup_events=pickups,decisions=len(logs),simulated_s=len(logs)*.064,
                falls=sum(int(l['outcome']['fell']) for l in logs),stop_reason=stop,initial_match=match,checkpoint_sha256=checkpoint_hash,
                checkpoint_update=nav.update if hasattr(nav,'update') else None,geometry_sha256=engine.mesh.info['mesh_sha256'],
                median_planning_s=float(np.median(searches)) if searches else None,planning_s=searches,elapsed_wall_s=time.perf_counter()-started,
                future_event_lock_revalidations=lock_steps,commitment_cancellations=cancelled,
                reuse_remaining_plan=args.reuse_plan,remaining_plan_revalidations=reuse_steps,full_searches=full_search_steps,
                max_live_errors=np.max([l['errors'] for l in logs if l['errors']],axis=0).tolist() if any(l['errors'] for l in logs) else None,
                native_raw_sha256=sha(raw_file),owned_game_pid=proc.pid,policy_training_started=False,
                protocol='Local 64ms, DELAY 0, SSYAWFIX off; competitive off, no rescue; first fall or decision budget ends both branches',
                limitations='Fresh seeds on familiar maps; deterministic A2 neural controller with walk-tour/routing, no rescue/stuck detour. Planner rejects unmodelled Blast/Mega futures; comparator may execute them. No full-round benchmark or calibrated landing guarantee.')
    atomic_json(folder/'decisions.json',logs);atomic_json(folder/'report.json',report)
    print(json.dumps({k:report[k] for k in ('map','seed','branch','actual_points','decisions','falls','stop_reason','median_planning_s','future_event_lock_revalidations')}),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--maps',nargs='+',default=['KingOfTheMarble_Hunt','Sprawl_Hunt','Bowl_Hunt'])
    p.add_argument('--seeds',nargs='+',type=int,default=[15407,24462]);p.add_argument('--decisions',type=int,default=160)
    p.add_argument('--horizon',type=int,default=40);p.add_argument('--commit',type=int,default=4);p.add_argument('--port',type=int,default=9054)
    p.add_argument('--reuse-plan',action='store_true',help='Revalidate safe useful remaining commands before a fresh full search')
    p.add_argument('--checkpoint',type=Path,default=ML/'models/nav/real_maps_20261010_A2/nav_latest.pth')
    p.add_argument('--output',type=Path,default=ML/'dynamics_reports/reliability_20261012')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);fingerprint=sha(a.checkpoint);results=[]
    sources={str(p):sha(p) for p in Path(__file__).parent.glob('*.py')}
    for path in (ML.parent/'Marble Blast Platinum/platinum/client/scripts/ai/mlAgent.cs',
                 ML.parent/'Marble Blast Platinum/marbleblast_geometry_probe.exe',Path(__file__).with_name('engine_dynamics.inc'),
                 ML/'nav/routing.py',ML/'nav/terrain.py',ML/'nav/collision/mesh.py'):
        sources[str(path)]=sha(path)
    atomic_json(a.output/'prepared.json',dict(maps=a.maps,seeds=a.seeds,decisions=a.decisions,horizon=a.horizon,commit=a.commit,reuse_plan=a.reuse_plan,
                checkpoint=str(a.checkpoint),checkpoint_sha256=fingerprint,source_hashes=sources,selection='Predeclared fresh-seed validation on familiar geometry; no outcome-based fixture selection',training=False))
    # Warm/load the immutable policy before connecting an owned game.
    for name in a.maps:
        terrain=TerrainGrid(TerrainMap.resolve(name));engine=GeneralPhysics(CollisionMap(ML/'collision_maps'/(name+'.npz')))
        FrozenNavigator(terrain,a.checkpoint)
        for seed in a.seeds:
            folder=a.output/name/str(seed);folder.mkdir(parents=True)
            with game(name,a.port,obs_ms=64,seed=seed) as (env,proc):
                command(env,'DYNAMICSLIVE 1','dynamics_live');env.control('REPLAYCAPTURE 1')
                for _ in range(8):env.step(NOOP_ACTION,repeat=1)
                world=copy.deepcopy(env.replay_world);world['native_reference']=copy.deepcopy(env.msg.dynamics['native']['state'])
                atomic_json(folder/'fixture.json',dict(world=world,catalog=env.replay_catalog,dynamics=env.msg.dynamics))
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
