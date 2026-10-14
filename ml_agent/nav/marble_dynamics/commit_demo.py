"""Execute short selected plans and compare a simple controller on matched worlds.

Replay restore is controlled experimental setup, not a rescue during execution.
The comparator is a simple routed proportional controller, not the trained policy.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import time

import numpy as np

from nav.collision.mesh import atomic_json,CollisionMap
from nav.marble_dynamics.record import ML,RUNTIME,game,command,archive,sha
from nav.marble_dynamics.physics import GeneralPhysics
from nav.marble_dynamics.bridge import predict_replies
from nav.marble_dynamics.gem_demo import Proposal,plan,support,candidate_cost
from nav.marble_dynamics.events import external_changes
from nav.marble_dynamics.evaluate import errors
from nav.routing import RoutePlanner
from nav.terrain import TerrainGrid,TerrainMap
from nav.gems import visible_gems
from nav.protocol import NOOP_ACTION,RAW_SCORE


class SlowProposal(Proposal):
    def __init__(self,goal,scale,steps):super().__init__(goal,steps=steps);self.scale=scale
    def __call__(self,state,index):
        action=super().__call__(state,index)
        action[:4]=[round(v*self.scale,6) for v in action[:4]]
        return action


def choose_plan(engine,route,dynamics,gem,horizon):
    chosen,pairs,waypoint=plan(engine,route,dynamics,gem,horizon)
    item=next((t for t in dynamics.get('items',[]) if t['gem'] and np.linalg.norm(np.asarray(t['position'])-gem)<.1),None)
    for scale in (.35,.6):
        result=predict_replies(engine,dynamics,SlowProposal(waypoint,scale,horizon))
        end=result['states'][-1];rejected=None
        if result['failure']:rejected='prediction_failure'
        elif result['ambiguous_intervals']:rejected='contact_ambiguity'
        elif not support(engine.mesh,end):rejected='no_terminal_support'
        elif np.linalg.norm(end['velocity'])>22:rejected='terminal_speed_above_pilot_limit'
        routed=np.linalg.norm(np.asarray(waypoint)-gem)>.01
        cost,close,pickup=candidate_cost(route,result['states'],gem,routed,item)
        info=dict(label='slow_'+str(scale),cost=float(cost),rejected=rejected,minimum_gem_distance_u=close,
                  terminal_support=support(engine.mesh,end),end_position=end['position'],
                  ambiguous_intervals=result['ambiguous_intervals'],failure=result['failure'],possible_pickup_interval=pickup)
        pairs.append((info,result))
    accepted=[pair for pair in pairs if not pair[0]['rejected']]
    return min(accepted,key=lambda p:p[0]['cost']) if accepted else None,pairs,waypoint


def choose_gem_plan(engine,route,dynamics,gems,current,horizon):
    """Feed physically feasible progress back to gem selection; no map rules."""
    position=dynamics['native']['state']['position']
    chosen,pairs,waypoint=choose_plan(engine,route,dynamics,current[:3],horizon)
    start_distance=route.distance(position)
    if chosen:
        info,result=chosen
        progress=start_distance-route.distance(result['states'][-1]['position'])
        if info.get('possible_pickup_interval') is not None or progress>.3:
            return chosen,pairs,waypoint,current
    options=[]
    for gem in gems:
        gem=np.asarray(gem,float)
        if np.linalg.norm(gem[:3]-current[:3])<.1:continue
        candidate,alternatives,aim=choose_plan(engine,route,dynamics,gem[:3],horizon)
        if not candidate:continue
        info,result=candidate
        progress=route.distance(position)-route.distance(result['states'][-1]['position'])
        pickup=info.get('possible_pickup_interval')
        if pickup is None and progress<=.3:continue
        # Prefer an actual candidate pickup, then larger feasible progress.
        # Unknown reachability is not a permanent blacklist of this gem.
        rank=(1 if pickup is not None else 0,gem[3]/max((pickup or horizon)*.064,.064),progress)
        options.append((rank,candidate,alternatives,aim,gem))
    if options:
        _,chosen,pairs,waypoint,current=max(options,key=lambda o:o[0])
        route.navigate(position,current[:3])
        return chosen,pairs,waypoint,current
    return None,pairs,waypoint,current


def restore(env,world):
    if not world['supported']:raise RuntimeError('Native replay world is unsupported')
    words=lambda values:' '.join(format(float(v),'.17g') for v in values)
    payload='|'.join([words(world['header']),words(world['pose']),world['held'],
                      ';'.join(words(v) for v in world['items']),words(world['groups'])])
    env.debug=None;env.control('REPLAYRESTORE '+payload)
    for _ in range(16):
        if env.debug and env.debug[0]=='replay_restored':break
        if env.debug and env.debug[0]=='replay_error':raise RuntimeError(str(env.debug))
        env.step(NOOP_ACTION,repeat=1)
    else:raise RuntimeError('No fixture restore acknowledgement')
    for _ in range(4):env.step(NOOP_ACTION,repeat=1)
    current=env.replay_world
    if current['items']!=world['items'] or current['groups']!=world['groups']:
        # Respawn countdowns legitimately advance; compare hidden states,
        # weights and counts, and retain exact clocks as separate evidence.
        expected=[(r[0],r[2],r[3]) for r in world['items']]
        actual=[(r[0],r[2],r[3]) for r in current['items']]
        if expected!=actual or current['groups']!=world['groups']:raise RuntimeError('Replay item/group mismatch')
    return dict(position_error_u=float(np.linalg.norm(np.asarray(env.msg.dynamics['native']['state']['position'])-world['pose'][:3])),
                velocity_error_u_per_s=float(np.linalg.norm(np.asarray(env.msg.dynamics['native']['state']['velocity'])-world['pose'][3:6])),
                score_equal=float(env.msg.obs[RAW_SCORE])==float(world['header'][3]),item_states_match=True)


def run(name,args):
    folder=args.output/name
    if (folder/'comparison.json').exists():raise RuntimeError('Use a new output directory')
    folder.mkdir(parents=True,exist_ok=True)
    mesh=CollisionMap(ML/'collision_maps'/(name+'.npz'));engine=GeneralPhysics(mesh)
    terrain=TerrainGrid(TerrainMap.resolve(name));results=[]
    with game(name,args.port,obs_ms=64) as (env,proc):
        command(env,'DYNAMICSLIVE 1','dynamics_live');env.control('REPLAYCAPTURE 1')
        for _ in range(8):env.step(NOOP_ACTION,repeat=1)
        world=copy.deepcopy(env.replay_world);atomic_json(folder/'fixture.json',dict(world=world,catalog=env.replay_catalog,dynamics=env.msg.dynamics))
        for branch in ('routed_controller','physics_plan'):
            matched=restore(env,world)
            if matched['position_error_u']>.0001 or matched['velocity_error_u_per_s']>.0001 or not matched['score_equal']:
                raise RuntimeError('Initial fixture mismatch: '+str(matched))
            dest=folder/branch;dest.mkdir(exist_ok=True);route=RoutePlanner(terrain)
            raw=RUNTIME/(name+'.committed_'+branch+'.jsonl');command(env,'DYNAMICSRECORD '+raw.as_posix(),'dynamics_record')
            initial=float(env.msg.obs[RAW_SCORE]);log=[];pickups=[];reason='decision_budget';target=None;search_times=[]
            for search in range(args.decisions):
                if (args.output/'STOP').exists():reason='operator_stop';break
                if len(log)>=args.decisions:break
                dynamics=env.msg.dynamics;state=dynamics['native']['state'];gems=visible_gems(env.msg.obs)
                if not gems or state['mode']!=0 or state['oob']:reason='unsupported_state';break
                if target is None or not any(np.linalg.norm(np.asarray(g[:3])-target[:3])<.1 for g in gems):
                    target=np.asarray(min(gems,key=lambda g:np.linalg.norm(np.asarray(g[:3])-state['position'])),float)
                t0=time.perf_counter()
                if branch=='physics_plan':
                    chosen,pairs,waypoint,target=choose_gem_plan(engine,route,dynamics,gems,target,args.horizon)
                    if not chosen:reason='no_candidate_passed_checks';break
                    info,forecast=chosen
                else:
                    waypoint,_=route.navigate(state['position'],target[:3])
                    forecast=predict_replies(engine,dynamics,Proposal(waypoint,steps=args.horizon))
                    info=dict(label='routed_proportional_control',cost=None,rejected=None);pairs=[]
                elapsed=time.perf_counter()-t0;search_times.append(elapsed)
                if forecast['failure']:reason='prediction_failure';break
                commit=min(args.commit,args.decisions-len(log))
                interrupted=False
                for index in range(commit):
                    snapshot=env.msg.dynamics;tick=env.msg.tick;action=forecast['proposed'][index]
                    prior=float(env.msg.obs[RAW_SCORE]);msg,outcome=env.step(action[:6],repeat=1,use_pow=int(action[6]),pow_yaw=action[7] if action[6] else None)
                    actual=msg.dynamics['native']['state'];predicted=forecast['states'][index+1]
                    error=errors(predicted,actual);events=external_changes(predicted,actual);delta=float(msg.obs[RAW_SCORE])-prior
                    log.append(dict(decision=len(log),tick=tick,dynamics=snapshot,action=action,chosen=info,
                           candidates=[c for c,r in pairs] if index==0 else [],planning_s=elapsed if index==0 else 0.,
                           predicted_next=predicted,actual_next=actual,errors=error,external_events=events,
                           score_delta=delta,outcome=outcome,gem=target.tolist(),waypoint=list(waypoint),
                           committed_prediction_step=index+1,search=search))
                    if delta>0:
                        pickups.append(dict(decision=len(log)-1,points=delta,gem=target.tolist(),position=actual['position']))
                        print(json.dumps(dict(map=name,branch=branch,pickup_points=delta,events=len(pickups),decision=len(log)-1)),flush=True)
                    if outcome['fell'] or outcome['round_ended'] or outcome['reconnected']:reason='game_or_fall_event';interrupted=True;break
                    if error[0]>.05 and not events:reason='prediction_error_gate';interrupted=True;break
                    if len(pickups)>=args.gems:reason='collection_goal';interrupted=True;break
                    if events or delta>0:break
                atomic_json(dest/'progress.json',dict(decisions=len(log),events=len(pickups),reason='running'))
                if search%8==0:print(json.dumps(dict(map=name,branch=branch,decisions=len(log),events=len(pickups),choice=info['label'],planning_s=elapsed)),flush=True)
                if interrupted:break
            command(env,'DYNAMICSRECORD off','dynamics_record');archive(raw,dest/'native.jsonl.gz')
            final=float(env.msg.obs[RAW_SCORE]);closing=final-initial-sum(p['points'] for p in pickups)
            if closing>0:pickups.append(dict(decision=len(log),points=closing,gem=target.tolist(),position=env.msg.dynamics['native']['state']['position'],phase='recording_close'))
            report=dict(map=name,branch=branch,geometry_sha256=mesh.info['mesh_sha256'],setup='Matched replay restore of settled natural spawn; no rescue or teleport during execution',
                        initial_match=matched,initial_score=initial,final_score=final,actual_points=final-initial,
                        actual_pickup_events=pickups,decisions=len(log),stop_reason=reason,planning_s=search_times,
                        commit_decisions=args.commit,owned_game_pid=proc.pid,
                        max_live_errors=np.max([l['errors'] for l in log],axis=0).tolist() if log else None,
                        raw_sha256=sha(raw),source_hashes={p.name:sha(p) for p in (Path(__file__),Path(__file__).with_name('gem_demo.py'),Path(__file__).with_name('bridge.py'),Path(__file__).with_name('events.py'))},
                        policy_training_started=False,global_planner_approved=False,
                        limitations='Simple routed-controller comparison on one restored development world; not a trained-policy benchmark. Support/ambiguity checks are not calibrated uncertainty.')
            atomic_json(dest/'decisions.json',log);atomic_json(dest/'report.json',report);results.append(report)
        atomic_json(folder/'comparison.json',dict(map=name,branches=results,fixture_setup_only=True))
    return results


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--maps',nargs='+',default=['KingOfTheMarble_Hunt'])
    p.add_argument('--port',type=int,default=9054);p.add_argument('--decisions',type=int,default=160)
    p.add_argument('--gems',type=int,default=2);p.add_argument('--horizon',type=int,default=40);p.add_argument('--commit',type=int,default=4)
    p.add_argument('--output',type=Path,default=ML/'dynamics_reports/commit_demo_20261012')
    a=p.parse_args();summary=[]
    for name in a.maps:summary.extend(run(name,a));atomic_json(a.output/'summary.json',dict(branches=summary,policy_training_started=False))
