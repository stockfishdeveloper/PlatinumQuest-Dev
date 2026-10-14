"""Predicted bridge continuations and a conservative terminal stopping check.

These are predicted states, not native fixture resets. Reconstruct queued reply
translation exactly; continuation consumes no future observations. No game calls.
"""
import copy
import time

import numpy as np

from nav.marble_dynamics.bridge import predict_replies,translated
from nav.marble_dynamics.gem_demo import Proposal
from nav.marble_dynamics.navigator import NavigatorProposal
from nav.marble_dynamics.search import terminal_support,BrakeContinuation
from nav.marble_dynamics.planning_budget import check as check_budget,PlanningBudgetExceeded


def resume_forecast(dynamics,rollout):
    if rollout['failure'] or rollout['ambiguous_intervals']:
        raise ValueError('Cannot continue a failed or contact-ambiguous forecast')
    count=len(rollout['controls'])
    if count!=len(rollout['proposed']) or len(rollout['states'])!=count+1:
        raise ValueError('Incomplete forecast bridge state')
    result=copy.deepcopy(dynamics)
    pending=list(dynamics['bridge_action']);move=copy.deepcopy(dynamics['native']['pending_move'])
    hold=int(dynamics['pow_yaw_hold']);aim=float(dynamics['pow_yaw'])
    for action in rollout['proposed']:
        move,_,hold,aim=translated(pending,hold,aim)
        pending=list(action)
    state=copy.deepcopy(rollout['states'][-1])
    result['native']['state']=state
    result['native']['frame_clock_ms']=int(dynamics['native']['frame_clock_ms'])+64*count
    result['native']['pending_move']=move
    result['bridge_action']=pending;result['pow_yaw_hold']=hold;result['pow_yaw']=aim
    for source,key in [('blast_context','blast'),('inventory_context','inventory')]:
        if source in state:result[key]=copy.deepcopy(state[source])
    return result


def concatenate(prefix,continuation):
    if prefix.get('physics_frame_ms',64)!=continuation.get('physics_frame_ms',64):
        raise ValueError('Cannot join different physics frame cadences')
    result=dict(prefix)
    result['states']=prefix['states']+continuation['states'][1:]
    for key in ['controls','events','proposed']:result[key]=prefix[key]+continuation[key]
    if 'native_states' in prefix:
        result['native_states']=prefix['native_states']+continuation['native_states'][1:]
        result['native_controls']=prefix['native_controls']+continuation['native_controls']
    offset=len(prefix['controls'])
    result['failure']=dict(continuation['failure'],step=offset+continuation['failure']['step']) if continuation['failure'] else None
    result['ambiguous_intervals']=prefix['ambiguous_intervals']+[offset+i for i in continuation['ambiguous_intervals']]
    result['time_s']=[i*.064 for i in range(len(result['states']))]
    result['requested_replies']=prefix['requested_replies']+continuation['requested_replies']
    result['stopped_on_ambiguity']=continuation['stopped_on_ambiguity']
    return result


def braking_viability(engine,dynamics,rollout,goal,replies=48):
    """Require an unambiguous, supported low-speed ordinary braking backup.

    A finite braking test is a conservative heuristic, not a guarantee for every
    possible future action, disturbance or script event.
    """
    continuation=predict_replies(engine,resume_forecast(dynamics,rollout),
        Proposal(goal,steps=replies,brake=True),stop_on_ambiguity=True)
    supported,stable,normal_speed=terminal_support(engine.mesh,continuation['states'][-4:])
    speed=float(np.linalg.norm(continuation['states'][-1]['velocity']))
    reason=None
    if continuation['failure']:reason='prediction_failure'
    elif continuation['ambiguous_intervals']:reason='contact_ambiguity'
    elif not stable:reason='no_supported_stopping_continuation'
    elif speed>=1.:reason='stopping_speed_above_limit'
    info=dict(replies=replies,valid=reason is None,rejected=reason,supported=supported,
        stable=stable,end_speed_u_per_s=speed,maximum_tail_normal_speed_u_per_s=normal_speed,
        end_position=continuation['states'][-1]['position'],ambiguities=continuation['ambiguous_intervals'],
        failure=continuation['failure'])
    return info,concatenate(rollout,continuation)


class ActorBrakeContinuation:
    """Reuse the actual actor reply, then cloned actor futures and braking."""
    def __init__(self,navigator,raw,route,task_goal,actor_action,replies=40,prefix_replies=8,next_goal=None):
        if not 2<=prefix_replies<=replies-4:raise ValueError('Actor prefix needs a complete braking tail')
        self.steps=replies;self.prefix_replies=prefix_replies;self.task_goal=list(task_goal[:3])
        self.actor=NavigatorProposal(navigator,raw,route,self.task_goal,next_goal,prefix_replies,
            first_action=actor_action)

    def __call__(self,state,index):
        if index<self.prefix_replies:return self.actor(state,index)
        return Proposal(self.task_goal,brake=True)(state,0)


def ordinary_recovery(engine,dynamics,actor_action,replies=40,budget_s=.2,landing_limit=8,
        actor_continuation=None,task_route=None,task_goal=None):
    """Complete physical stopping backups; no gem-progress requirement.

    Checked stopping is opt-in in the development controller. The optional
    longer actor continuation is an offline experimental proposal.
    Existing queued inputs remain in every forecast. A stationary command alone
    is never considered proof of support, and script trigger volumes are not
    modeled by this physical-support test.
    """
    from nav.marble_dynamics.landing import landing_points,LandingContinuation
    from nav.marble_dynamics.gem_demo import possible_pickup
    if replies<4 or budget_s<=0 or landing_limit<0:raise ValueError('Invalid bounded recovery limits')
    if actor_continuation is not None and (task_route is None or task_goal is None
            or actor_continuation.steps!=replies):raise ValueError('Actor continuation needs task context and matching horizon')
    started=time.perf_counter();previous=getattr(engine,'planning_deadline',None)
    deadline=started+budget_s
    engine.planning_deadline=min(previous,deadline) if previous is not None else deadline
    candidates=[];chosen=None;stop='completed'
    position=list(dynamics['native']['state']['position'])

    def proposals():
        if actor_continuation is not None:
            yield 'actor'+str(actor_continuation.prefix_replies)+'_then_brake',actor_continuation,True
        yield 'actor_then_brake',BrakeContinuation([list(actor_action)],position,replies-1),False
        yield 'brake_now',Proposal(position,steps=replies,brake=True),False
        check_budget(engine)
        if landing_limit:
            for i,point in enumerate(landing_points(engine.mesh,dynamics['native']['state'],landing_limit)):
                yield 'support_target_'+str(i),LandingContinuation([],point['position'],replies),False

    try:
        for label,proposal,extended in proposals():
            check_budget(engine)
            try:rollout=predict_replies(engine,dynamics,proposal,stop_on_ambiguity=True)
            except ValueError as exc:
                candidates.append(dict(label=label,rejected='unsupported_context:'+str(exc)));continue
            end=rollout['states'][-1]
            supported,stable,normal_speed=terminal_support(engine.mesh,rollout['states'][-4:])
            speed=float(np.linalg.norm(end['velocity']))
            reason=None
            if rollout['failure']:reason='prediction_failure'
            elif rollout['ambiguous_intervals']:reason='contact_ambiguity'
            elif len(rollout['controls'])!=replies or len(rollout['states'])!=replies+1 or len(rollout['proposed'])!=replies:reason='incomplete_forecast'
            elif not stable:reason='no_supported_continuation'
            elif speed>=1.:reason='stopping_speed_above_limit'
            task_progress=None;task_pickup=None
            prefix=proposal.prefix_replies if extended else 1 if label=='actor_then_brake' else 0
            if extended and reason is None:
                measure=copy.copy(task_route);measure.navigate(position,task_goal[:3])
                start=float(measure.distance(position));finish=float(measure.distance(end['position']))
                if np.isfinite(start) and np.isfinite(finish):task_progress=start-finish
                item=next((i for i in dynamics.get('items',[])
                    if np.linalg.norm(np.asarray(i['position'])-task_goal[:3])<.1),None)
                if item is not None:task_pickup=possible_pickup(rollout['states'],item)
                if task_pickup is None and (task_progress is None or task_progress<=.3):reason='no_useful_task_progress'
                check_budget(engine)
            info=dict(label='recovery_'+label,rejected=reason,terminal_support=supported,
                stable_continuation=stable,end_speed_u_per_s=speed,end_position=end['position'],
                maximum_tail_normal_speed_u_per_s=normal_speed,requested_replies=replies,
                evaluated_replies=len(rollout['controls']),game_rule_volumes_checked=False,
                possible_pickup_interval=None,goal_kind='physical_recovery',
                landing_goal=list(end['position']),landing_after_reply=prefix+2 if extended else 0,landing_triangle_id=None,
                actor_prefix_replies=prefix,task_progress_u=task_progress,task_possible_pickup_interval=task_pickup,
                stopping_verified=reason is None,ambiguities=rollout['ambiguous_intervals'],failure=rollout['failure'])
            candidates.append(info)
            if reason is None:
                chosen=(info,rollout);break
    except PlanningBudgetExceeded:
        stop='deadline'
    finally:
        engine.planning_deadline=previous
    return chosen,dict(candidates=candidates,stop=stop,elapsed_wall_s=time.perf_counter()-started,
        budget_s=budget_s,physical_support_only=True,
        caveat='Complete queued-control physical forecasts, not native outcomes or certification against unmodeled script bounds/triggers.')


def main():
    """All first recorded recovery selections per native life; no gameplay."""
    import argparse
    import gzip
    import json
    from pathlib import Path
    from nav.collision.mesh import atomic_json
    from nav.collision.experimental_sweep import CachedCompiledSweepMap
    from nav.marble_dynamics.physics import GeneralPhysics
    from nav.marble_dynamics.geometry_route import walking_terrain
    from nav.marble_dynamics.reliability_audit import replay_actor_observations
    from nav.marble_dynamics.record import ML,sha
    from nav.terrain import TerrainMap,TerrainGrid
    from nav.routing import RoutePlanner
    from nav.source_archive import snapshot_sources
    p=argparse.ArgumentParser(description=main.__doc__)
    p.add_argument('--archived-actor-probe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prefix-replies',type=int,default=8)
    p.add_argument('--replies',type=int,default=40)
    p.add_argument('--budget-s',type=float,default=.2)
    p.add_argument('--checkpoint',type=Path,default=ML/'models/nav/real_maps_20261010_A2/nav_latest.pth')
    a=p.parse_args()
    if not 2<=a.prefix_replies<=a.replies-4 or a.budget_s<=0:raise ValueError('Invalid bounded prefix limits')
    source=a.archived_actor_probe;prepared=json.loads((source/'prepared.json').read_text(encoding='utf-8'))
    summary=json.loads((source/'summary.json').read_text(encoding='utf-8'))
    if not prepared.get('full_round'):raise ValueError('Requires observed full-round recovery records')
    if sha(a.checkpoint)!=prepared['checkpoint_sha256']:raise ValueError('Archive/checkpoint mismatch')
    a.output.mkdir(parents=True,exist_ok=False)
    paths=list(Path(__file__).parent.glob('*.py'))+list((ML/'nav').glob('*.py'))
    paths += [ML/'nav/collision/mesh.py',ML/'nav/collision/experimental_sweep.py']
    sources={str(path):sha(path) for path in paths}
    snapshot_sources(a.output,paths)
    atomic_json(a.output/'prepared.json',dict(source=str(source),source_summary_sha256=sha(source/'summary.json'),
        checkpoint_sha256=sha(a.checkpoint),source_hashes=sources,replies=a.replies,
        actor_prefix_replies=a.prefix_replies,budget_s=a.budget_s,
        selection='Every first successful recorded recovery selection per native object/respawn life, including refused reconstructions/contexts. Alternating baseline/variant order by sample index. No future native reanchoring.',
        setup='Load geometry and warm observed-task route before timed candidates; both branches use the same exact compiled query cache.',
        actor_reconstruction='Compare every actual actor output; mismatch invalidates history until observed reset. Only clones generate hypothetical later replies.',
        policy_training=False,native_gameplay=False))
    results=[]
    for report in summary['branches']:
        if report['branch']!='physics_plan' or not report.get('score_valid'):continue
        name=report['map'];folder=source/name/str(report['seed'])/report['branch']/str(report['round'])
        with gzip.open(folder/'decisions.jsonl.gz','rt',encoding='utf-8') as stream:logs=[json.loads(line) for line in stream]
        terrain=TerrainGrid(TerrainMap.resolve(name))
        dll=source/'external_artifacts'/Path(prepared['sweep_dll']).name
        mesh=CachedCompiledSweepMap(ML/'collision_maps'/(name+'.npz'),dll)
        engine=GeneralPhysics(mesh);engine.bridge_physics_ms=16
        route=RoutePlanner(walking_terrain(terrain,mesh));seen=set();selected=set()
        for index,row in enumerate(logs):
            if 'recovery_audit' not in row['choice'] or not row['choice']['label'].startswith('recovery_'):continue
            native=row['before']['native'];context=row['before'].get('game_context',{})
            life=(native['object_id'],context.get('respawns'))
            if life not in seen:seen.add(life);selected.add(index)
        replayed=set()
        for index,navigator,reconstruction in replay_actor_observations(logs,terrain,a.checkpoint):
            if index not in selected:continue
            replayed.add(index);row=logs[index];choice=row['choice'];goal=choice.get('task_goal_context',{}).get('destination')
            record=dict(map=name,seed=report['seed'],log_index=index,reconstruction=reconstruction)
            if not reconstruction['trusted']:record['refused']='untrusted_actor_history'
            elif goal is None or not any(np.linalg.norm(np.asarray(item['position'])-goal)<.1 for item in row['before']['items']):record['refused']='unobserved_task'
            elif np.linalg.norm(np.asarray(choice['actor_goal_context']['destination'])-goal)>.1:record['refused']='recorded_actor_task_disagreement'
            else:
                route.navigate(row['before']['native']['state']['position'],goal)
                actor=ActorBrakeContinuation(navigator,np.asarray(row['raw_observation'],dtype=np.float32),route,
                    goal,choice['actor_action'],a.replies,a.prefix_replies,choice['goal_context'].get('next_goal'))
                order=('baseline','actor_prefix') if len(results)%2==0 else ('actor_prefix','baseline')
                record['order']=order;record['branches']={}
                for kind in order:
                    chosen,audit=ordinary_recovery(engine,row['before'],choice['actor_action'],
                        replies=a.replies,budget_s=a.budget_s,
                        actor_continuation=actor if kind=='actor_prefix' else None,
                        task_route=route if kind=='actor_prefix' else None,task_goal=goal if kind=='actor_prefix' else None)
                    record['branches'][kind]=dict(selected=chosen[0] if chosen else None,audit=audit)
            results.append(record);atomic_json(a.output/'result.json',dict(cases=results,policy_training=False,native_gameplay=False))
            print(json.dumps(dict(map=name,log_index=index,refused=record.get('refused'),
                selected={k:v['selected']['label'] if v['selected'] else None for k,v in record.get('branches',{}).items()})),flush=True)
        for index in sorted(selected-replayed):
            results.append(dict(map=name,seed=report['seed'],log_index=index,refused='missing_actor_input_context'))
    if any(sha(Path(path))!=digest for path,digest in sources.items()):raise ValueError('Probe source changed during execution')
    atomic_json(a.output/'result.json',dict(cases=results,complete=True,policy_training=False,native_gameplay=False,
        source_unchanged=True,checkpoint_unchanged=sha(a.checkpoint)==prepared['checkpoint_sha256'],
        caveat='Archived development counterfactuals, not native pickups, complete-round comparison or calibrated landing guarantees. Timed failures and all selected lives are retained.'))


if __name__=='__main__':main()
