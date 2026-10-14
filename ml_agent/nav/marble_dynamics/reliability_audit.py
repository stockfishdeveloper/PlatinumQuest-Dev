"""Independent score, pending-control and native five-step replay evidence."""
import argparse
from bisect import bisect_right
from collections import Counter
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from nav.collision.mesh import CollisionMap,atomic_json
from nav.marble_dynamics.record import ML,sha
from nav.marble_dynamics.evaluate import load_rows,errors
from nav.marble_dynamics.physics import GeneralPhysics
from nav.marble_dynamics.bridge import predict_replies
from nav.marble_dynamics.rollout import predict
from nav.marble_dynamics.events import external_changes
from nav.marble_dynamics.reliability_eval import pending_reply_matches,continuous_branches
from nav.marble_dynamics.blast import event as blast_event
from nav.protocol import NOOP_ACTION
from nav.marble_dynamics.navigator import FrozenNavigator


def queued_power_command(logs, clock_ms, physics_ms=16):
    """Resolve the command actually reaching the native Move, not the nearest.

    The first subframe holds the older collected Move. Later subframes execute
    the pending previous macro reply. Unknown intervals never infer provenance.
    """
    if physics_ms != 16:
        return None, 'unsupported_physics_cadence'
    clocks=[row['before']['native']['frame_clock_ms'] for row in logs]
    index=bisect_right(clocks,clock_ms)-1
    if index < 0 or clock_ms-clocks[index] not in (0,16,32,48):
        return None, 'outside_recorded_macro'
    source=index-(2 if clock_ms==clocks[index] else 1)
    if source < 0:
        return None, 'unobserved_initial_queue'
    for j in range(source+1,index+1):
        a,b=logs[j-1],logs[j]
        if clocks[j]-clocks[j-1]!=64 or any(a['outcome'].get(k) for k in ('fell','round_ended','reconnected')):
            return None, 'discontinuous_control_interval'
        before,after=a['before'],b['before']
        ac,bc=before.get('game_context',{}),after.get('game_context',{})
        if before['native'].get('object_id')!=after['native'].get('object_id'):
            return None, 'body_changed'
        if ac.get('available') and bc.get('available') and ac.get('respawns')!=bc.get('respawns'):
            return None, 'native_respawn'
        if not pending_reply_matches(after['bridge_action'],a['action']):
            return None, 'pending_reply_mismatch'
    if not logs[source]['action'][6]:
        return None, 'queued_command_did_not_request_use'
    return source, None


def full_round_activation_audit(logs, native, physics_ms=16):
    commands=[];consumptions=[];unresolved=[]
    for i,row in enumerate(logs):
        action=row['action']
        if not(action[6] or action[8]):continue
        state=row['before']['native']['state']
        commands.append(dict(decision=row.get('decision',i),log_index=i,frame_clock_ms=row['before']['native']['frame_clock_ms'],
            held=state['held'],script_held=row['before'].get('inventory',{}).get('held_type'),
            use_power=bool(action[6]),use_blast=bool(action[8]),checked=row.get('predicted') is not None,
            label=row['choice']['label']))
    by_index={c['log_index']:c for c in commands}
    for row in native:
        if row['kind']!='transition' or not(row['before']['held']>0 and row['after']['held']==0):continue
        a,b=row['before'],row['after']
        event=dict(sim_ms=row['sim_ms'],held=a['held'],new_effects=b['effects'],
            mode_before=a['mode'],mode_after=b['mode'],native_move_use=bool(row['move']['use']))
        # Resets/warps can also empty inventory. Only native playable-use evidence
        # qualifies as a consumption; command provenance remains a separate test.
        if a['mode']!=0 or b['mode']!=0 or a['oob'] or b['oob'] or not row['move']['use']:
            event.update(reason='held_reset_without_playable_native_use',matched_command=None)
            unresolved.append(event);continue
        index,reason=queued_power_command(logs,row['sim_ms'],physics_ms)
        event.update(matched_command=by_index.get(index),attribution_error=reason,
            interpretation='Native playable held-to-empty use. Command attribution follows the measured delayed input queue; this does not establish a useful landing or gem pickup.')
        consumptions.append(event)
    return dict(commands=commands,confirmed_native_inventory_consumptions=consumptions,
        unclassified_inventory_resets=unresolved,
        attribution_contract='16ms physics/64ms replies: first subframe uses reply two macros earlier, later subframes use previous reply. Gaps/respawns/pending mismatches remain unknown; nearest timestamp is never provenance.')


def audit_full_round(root, report, physics_ms=16):
    path=root/report['map']/str(report['seed'])/report['branch']/str(report['round'])
    with gzip.open(path/'decisions.jsonl.gz','rt',encoding='utf-8') as source:
        logs=[json.loads(line) for line in source]
    with gzip.open(path/'native.jsonl.gz','rt',encoding='utf-8') as source:
        activations=full_round_activation_audit(logs,(json.loads(line) for line in source),physics_ms)
    atomic_json(path/'activation_audit.json',activations)
    pending=[];actor_mismatches=[];missing_goals=[];checked_goals=Counter()
    for i,row in enumerate(logs):
        before=row['before'];state=before['native']['state']
        if i and before['native']['frame_clock_ms']-logs[i-1]['before']['native']['frame_clock_ms']==64:
            if not pending_reply_matches(before['bridge_action'],logs[i-1]['action']):pending.append(i)
        if state['mode']!=0 or state['oob']:continue
        context=row['choice'].get('goal_context')
        if not context:missing_goals.append(i)
        if report['branch'] not in ('physics_plan','physics_plan_reference') and 'actor_action' in row['choice'] and row['action']!=row['choice']['actor_action']:
            actor_mismatches.append(i)
        if row.get('predicted') is not None:
            checked_goals[context['kind'] if context else 'unknown']+=1
    points=sum(row['outcome']['gem_delta'] for row in logs)
    if abs(points-report['actual_points'])>.01:raise ValueError('Logged pickups disagree with native end score')
    latency=[row['latency_s'] for row in logs]
    return dict(map=report['map'],seed=report['seed'],branch=report['branch'],round=report['round'],
        score_valid=report['score_valid'],declared_inputs_valid=report['declared_inputs_valid'],
        actual_points=points,decisions=len(logs),pending_reply_mismatches=pending,
        actor_output_mismatches=actor_mismatches,missing_playable_goal_contexts=missing_goals,
        checked_goal_kinds=dict(checked_goals),checked_decisions=sum(row.get('predicted') is not None for row in logs),
        decision_latency_s={str(q):float(np.percentile(latency,q)) for q in (50,95,99,100)},
        decision_log_sha256=sha(path/'decisions.jsonl.gz'),native_archive_sha256=sha(path/'native.jsonl.gz'),
        activation_audit=activations)


def fast_reply_from_logged_inputs(row,model,encoded,geo,decoder):
    """Replay learned outputs using recorded current base action, not future motion."""
    import torch
    from nav.marble_dynamics.fast_nav import features,decode_action,legal_action
    context=row['choice'].get('actor_goal_context')
    if context is None:return None
    base=row['choice'].get('base_actor_action')
    if model.residual and base is None:raise ValueError('Missing current base action')
    x,q=features(row['before'],row['raw_observation'],context['destination'],context['waypoint'],
        geo['center'],geo['scale'],context.get('kind','gem'),base)
    with torch.inference_mode():
        output=model.forward_cached(torch.from_numpy(x)[None],torch.from_numpy(q)[None],encoded)
    action=decode_action(output[0].numpy(),base,model.axis_mode)
    return legal_action(action,row['before'],row['raw_observation']) if decoder.get('legal_actions') else action


def audit_fast_outputs(root,report,checkpoint):
    import torch
    from nav.marble_dynamics.fast_nav import FastMapPolicy,SCHEMA
    from nav.marble_dynamics.student_data import geometry
    torch.set_num_threads(1)
    if sha(checkpoint)!=report['fast_checkpoint_sha256']:
        raise ValueError('Native learned checkpoint changed before output replay')
    saved=torch.load(str(checkpoint),map_location='cpu',weights_only=False)
    if saved['schema']!=SCHEMA:raise ValueError('Unknown fast checkpoint schema')
    model=FastMapPolicy(**saved['config']);model.load_state_dict(saved['model']);model.eval()
    geo=geometry(CollisionMap(ML/'collision_maps'/(report['map']+'.npz')))
    with torch.inference_mode():encoded=model.encode_map(*(torch.from_numpy(geo[k]) for k in ('features','bbox_lo','bbox_hi')))
    path=root/report['map']/str(report['seed'])/report['branch']/str(report['round'])/'decisions.jsonl.gz'
    checked=skipped=mismatches=0;examples=[]
    with gzip.open(path,'rt',encoding='utf-8') as source:
        for line in source:
            row=json.loads(line);action=fast_reply_from_logged_inputs(row,model,encoded,geo,saved.get('decoder',{}))
            if action is None:skipped+=1;continue
            checked+=1
            if not pending_reply_matches(action,row['action']):
                mismatches+=1
                if len(examples)<20:examples.append(dict(decision=row['decision'],replayed=action,recorded=row['action']))
    return dict(passed=bool(checked and not mismatches),checked=checked,skipped_without_actor_inputs=skipped,
        mismatch_count=mismatches,mismatch_examples=examples,checkpoint_sha256=sha(checkpoint),journal_sha256=sha(path),
        contract='Every recorded learned reply recomputed from current native observation, explicit actor destination/waypoint and recorded current A2 action. This does not independently reconstruct the recurrent A2 history or certify future physics.')


def replay_actor_observations(logs,terrain,checkpoint):
    """Yield post-observation/pre-send actor state, refusing unmatched histories.

    Logged waypoints/destinations are causal inputs from the same observation.
    Gaps, native respawns and falls reset the actor. A mismatch invalidates the
    reconstructed history until an observed reset; no latent-state repair.
    """
    navigator=FrozenNavigator(terrain,checkpoint);previous=None;trusted=True
    for index,row in enumerate(logs):
        raw=np.asarray(row['raw_observation'],dtype=np.float32)
        reset=False
        if previous is not None:
            before,prior=row['before'],previous['before']
            a,b=before.get('game_context',{}),prior.get('game_context',{})
            reset=(before['native']['frame_clock_ms']-prior['native']['frame_clock_ms']!=64
                or any(previous['outcome'].get(k) for k in ('fell','round_ended','reconnected'))
                or before['native']['object_id']!=prior['native']['object_id']
                or a.get('available') and b.get('available') and a['respawns']!=b['respawns'])
            if reset:
                navigator=FrozenNavigator(terrain,model=navigator.model);trusted=True
            else:navigator.outcome(np.asarray(previous['raw_observation'],dtype=np.float32),raw)
        context=row['choice'].get('actor_goal_context');recorded=row['choice'].get('actor_action')
        if context is not None and recorded is not None:
            action=navigator.act(raw,context['waypoint'],context['destination'],
                row['choice'].get('goal_context',{}).get('next_goal'))
            matched=pending_reply_matches(action,recorded);trusted=trusted and matched
            yield index,navigator,dict(trusted=trusted,action_matched=matched,observed_reset=reset)
            navigator.sent_action(row['action'],raw)
        previous=row


def audit_branch(root,report,engine):
    path=root/report['map']/str(report['seed'])/report['branch'];logs=json.loads((path/'decisions.json').read_text())
    native=load_rows(path/'native.jsonl.gz');by={r['sim_ms']:r for r in native}
    stale=[];pose=[];inputs=[];script_mismatches=[];script_excluded=[];blast_activations=[]
    for i,l in enumerate(logs):
        dynamics=l['dynamics'];row=by[dynamics['native']['frame_clock_ms']]
        if i and not pending_reply_matches(dynamics['bridge_action'],logs[i-1]['action']):stale.append(i)
        if errors(dynamics['native']['state'],row['before'])[0]>1e-7:pose.append(i)
        if dynamics['native']['pending_move']!=row['move']:inputs.append(i)
        try:
            impulse=blast_event(dynamics.get('blast'),dynamics['native']['state'],bool(dynamics['bridge_action'][8]))
        except ValueError as exc:
            script_excluded.append(dict(decision=i,reason=str(exc)));continue
        expected=np.asarray(dynamics['native']['state']['velocity'])
        if impulse is not None:
            expected=expected+impulse['delta_velocity'];blast_activations.append(dict(decision=i,event=impulse))
        velocity_error=float(np.linalg.norm(expected-row['before']['velocity']))
        if velocity_error>1e-5:script_mismatches.append(dict(decision=i,velocity_error=velocity_error))
    # A pickup can exclude an entire five-step window. Independently check each
    # remaining transition so a later contact mismatch cannot hide in that gap.
    one_step=[];one_excluded=[]
    for i,l in enumerate(logs):
        row=by[l['dynamics']['native']['frame_clock_ms']]
        if row['before']['mode']!=0 or row['before']['oob'] or l['outcome']['fell']:
            one_excluded.append(dict(decision=i,reason='nonplayable_or_fall'));continue
        control=dict(move=row['move'],events=row['control_events'])
        try:result=predict(engine,row['before'],[control],64)
        except ValueError as exc:one_excluded.append(dict(decision=i,reason=str(exc)));continue
        if result['failure']:
            one_step.append(dict(decision=i,failure=result['failure'],errors=None));continue
        changed=external_changes(result['states'][-1],row['after'])
        if changed:
            one_excluded.append(dict(decision=i,reason='external_event',changes=changed));continue
        one_step.append(dict(decision=i,failure=None,errors=errors(result['states'][-1],row['after']),
                             ambiguous_intervals=result['ambiguous_intervals']))
    score=sum(l['score_delta'] for l in logs)
    assert abs(score-report['actual_points'])<.01
    starts={0,max(0,len(logs)-5)}
    starts.update(range(0,max(0,len(logs)-4),5))
    for l in logs:
        if l['score_delta']>0:starts.update(range(max(0,l['decision']-4),l['decision']+1))
    trials=[];excluded=[];movement=[];camera=[]
    for start in sorted(starts):
        window=logs[start:start+5]
        if len(window)<5:continue
        if any(l['predicted_next'] is None or external_changes(l['predicted_next'],l['actual_next']) or l['outcome']['fell'] for l in window):
            excluded.append(dict(start=start,reason='external_event_or_fall'));continue
        try:pred=predict_replies(engine,window[0]['dynamics'],[l['action'] for l in window])
        except ValueError as exc:excluded.append(dict(start=start,reason=str(exc)));continue
        clock=window[0]['dynamics']['native']['frame_clock_ms']
        for i,c in enumerate(pred['controls']):
            actual=by[clock+i*64]
            if c['move']!=actual['move']:movement.append(dict(start=start,offset=i))
            if len(c['events'])!=len(actual['control_events']) or any(a['kind']!=b['kind'] or abs(a['value']-b['value'])>1e-5 for a,b in zip(c['events'],actual['control_events'])):
                camera.append(dict(start=start,offset=i))
        trials.append(dict(start=start,failure=pred['failure'],endpoint_errors=errors(pred['states'][-1],window[len(pred['states'])-2]['actual_next'])))
    native_use=[dict(clock=r['sim_ms'],type=r['before']['held']) for r in native if r['move']['use'] and r['before']['held'] and not r['after']['held']]
    return dict(map=report['map'],seed=report['seed'],branch=report['branch'],actual_points=score,
                native_transitions=len(native),snapshot_position_mismatches=pose,snapshot_move_mismatches=inputs,
                stale_pending_replies=stale,five_step_windows=len(trials),movement_mismatches=movement,camera_mismatches=camera,
                script_actuation_mismatches=script_mismatches,script_actuation_exclusions=script_excluded,
                regular_blast_activations=[b for b in blast_activations if not b['event'].get('special')],
                special_blast_activations=[b for b in blast_activations if b['event'].get('special')],
                excluded_windows=excluded,prediction_failures=sum(t['failure'] is not None for t in trials),
                maximum_five_step_errors=np.max([t['endpoint_errors'] for t in trials],axis=0).tolist() if trials else None,
                one_step_native_control_trials=one_step,one_step_exclusions=one_excluded,
                maximum_one_step_native_control_errors=np.max([t['errors'] for t in one_step if t['errors']],axis=0).tolist() if any(t['errors'] for t in one_step) else None,
                contact_discrepancies=[t for t in one_step if t['errors'] and (t['errors'][0]>1e-5 or t['errors'][2]>.01)],
                native_powerup_activations=native_use,jump_with_support_intervals=sum(int(r['move']['jump'] and r['telemetry']['support_steps']>0) for r in native),
                decision_log_sha256=sha(path/'decisions.json'),native_archive_sha256=sha(path/'native.jsonl.gz'),
                caveat='Actual recorded replies are exogenous audit inputs; prediction uses only one initial state. Supported jump intervals are a proxy, not an impulse counter.')


def full_round_completion(prepared,checks):
    """Check declared identities, rather than the currently written prefix."""
    expected={(name,int(seed),kind,ri) for mi,name in enumerate(prepared['maps'])
        for seed in prepared['seeds'] for kind in continuous_branches(SimpleNamespace(**prepared),mi,int(seed))
        if not prepared.get('only_branch') or kind==prepared['only_branch']
        for ri in range(1,int(prepared['rounds'])+1)}
    actual=[(r['map'],int(r['seed']),r['branch'],int(r['round'])) for r in checks]
    complete=len(actual)==len(expected) and set(actual)==expected
    return dict(expected_rounds=len(expected),complete_declared_rounds=complete,
        complete_pool=complete and len(set(prepared['maps']))==10 and not prepared.get('preflight_only',False))


def prefix_window_reason(logs,index):
    """Two continuous macros include the newly chosen reply after its delay."""
    if index+2>=len(logs):return 'recording_end'
    rows=logs[index:index+3];first=rows[0]['before']['native']
    for offset,row in enumerate(rows):
        native=row['before']['native'];state=native['state']
        if native['frame_clock_ms']!=first['frame_clock_ms']+64*offset:return 'control_or_clock_gap'
        if native['object_id']!=first['object_id']:return 'body_object_changed'
        if state['mode']!=0 or state['oob']:return 'nonplayable'
        context=row['before'].get('game_context',{})
        initial=rows[0]['before'].get('game_context',{})
        if context.get('available') and initial.get('available') and context['respawns']!=initial['respawns']:
            return 'native_respawn'
        if offset<2 and any(row['outcome'][k] for k in ('fell','round_ended','reconnected')):
            return 'fall_end_or_reconnect'
        if offset and not pending_reply_matches(row['before']['bridge_action'],rows[offset-1]['action']):
            return 'pending_reply_mismatch'
    return None


def chosen_reply_prefix_audit(logs,native,engine):
    """Predeclared selected recovery/use requests plus every128th checked reply.

    Only the initial native snapshot and the selected action enter prediction.
    The second reply is neutral; it cannot affect this128ms endpoint. Later
    actual states/controls are targets, never reanchoring inputs.
    """
    if getattr(engine,'bridge_physics_ms',None)!=16:raise ValueError('Requires16/64ms bridge')
    by={r['sim_ms']:r for r in native};cases=[];excluded=[];checked=0
    for index,row in enumerate(logs):
        if row.get('predicted') is None:continue
        checked+=1;choice=row['choice'];selected=[]
        if 'recovery_audit' in choice and choice['label'].startswith('recovery_'):selected.append('new_recovery')
        if row['action'][6] or row['action'][8]:selected.append('checked_power_request')
        if checked%128==0:selected.append('every128th_checked')
        if not selected:continue
        record=dict(decision=row['decision'],log_index=index,selection=selected,label=choice['label'])
        reason=prefix_window_reason(logs,index)
        if reason:
            excluded.append(dict(**record,reason=reason));continue
        clock=row['before']['native']['state']['clock_ms']
        frames=[by.get(clock+j*16) for j in range(8)]
        if any(frame is None for frame in frames):
            excluded.append(dict(**record,reason='missing_native_frame'));continue
        try:forecast=predict_replies(engine,row['before'],[row['action'],list(NOOP_ACTION)])
        except ValueError as exc:
            cases.append(dict(**record,status='prediction_refused',reason=str(exc)));continue
        if forecast['failure'] or len(forecast['native_controls'])!=8:
            cases.append(dict(**record,status='prediction_incomplete',failure=forecast['failure']));continue
        actual=logs[index+2]['before']['native']['state']
        changes=external_changes(forecast['states'][-1],actual)
        if changes:
            excluded.append(dict(**record,reason='external_state_change',fields=changes));continue
        movement=[];camera=[]
        for j,(control,frame) in enumerate(zip(forecast['native_controls'],frames)):
            if any(abs(float(control['move'][k])-float(frame['move'][k]))>1e-6 for k in control['move']):
                movement.append(j)
            expected=[(e['kind'],e['value']) for e in control.get('events',[])]
            got=[(e['kind'],e['value']) for e in frame['control_events']]
            if len(expected)!=len(got) or any(k!=q or abs(v-w)>1e-6 for (k,v),(q,w) in zip(expected,got)):
                camera.append(j)
        cases.append(dict(**record,status='compared',endpoint_errors=errors(forecast['states'][-1],actual),
            movement_mismatch_frames=movement,camera_mismatch_frames=camera,
            ambiguity_intervals=forecast['ambiguous_intervals']))
    compared=[c for c in cases if c['status']=='compared']
    return dict(schema='chosen_reply_128ms_native_prefix_audit_v1',cases=cases,excluded=excluded,
        compared=len(compared),prediction_refusals=sum(c['status']!='compared' for c in cases),
        movement_mismatches=sum(len(c['movement_mismatch_frames']) for c in compared),
        camera_mismatches=sum(len(c['camera_mismatch_frames']) for c in compared),
        maximum_endpoint_errors=np.max([c['endpoint_errors'] for c in compared],axis=0).tolist() if compared else None,
        caveat='128ms from one BEFORE state includes the newly selected reply beginning at80ms. Actual later state/control is comparison only. This is not a full40reply native recovery or calibrated landing/game-rule guarantee.',
        policy_training_started=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('data',type=Path)
    p.add_argument('--recompute-fast',action='store_true',help='Independently recompute every logged fast reply from current inputs and recorded base controls')
    a=p.parse_args()
    summary=json.loads((a.data/'summary.json').read_text());engines={};checks=[]
    prepared=json.loads((a.data/'prepared.json').read_text())
    if prepared.get('full_round'):
        for report in summary['branches']:
            if not report.get('score_valid') or not report.get('declared_inputs_valid'):continue
            check=audit_full_round(a.data,report,prepared['physics_frame_ms']);checks.append(check)
            if a.recompute_fast and report['branch']=='fast_navigator':
                check['fast_model_output_replay']=audit_fast_outputs(a.data,report,Path(prepared['fast_checkpoint']))
                if not check['fast_model_output_replay']['passed']:raise ValueError('Learned output replay failed')
            brief={k:check[k] for k in ('map','branch','actual_points','checked_decisions')}
            brief.update({k:len(check[k]) for k in ('pending_reply_mismatches','actor_output_mismatches','missing_playable_goal_contexts')})
            print(json.dumps(brief),flush=True)
            atomic_json(a.data/'full_round_audit.json',dict(rounds=checks,**full_round_completion(prepared,checks),
                audit_source_sha256=sha(Path(__file__)),policy_training_started=False))
        return
    audit=audit_branch
    if prepared.get('physics_frame_ms',64)==16:
        from nav.marble_dynamics.small_step_episode_audit import audit_branch as audit
    sources={str(p):sha(p) for p in Path(__file__).parent.glob('*.py')}
    for report in summary['branches']:
        if report.get('status')=='invalid':continue
        name=report['map']
        if name not in engines:engines[name]=GeneralPhysics(CollisionMap(ML/'collision_maps'/(name+'.npz')))
        check=audit(a.data,report,engines[name]);checks.append(check)
        print(json.dumps({k:check[k] for k in ('map','seed','branch','five_step_windows','stale_pending_replies','movement_mismatches','camera_mismatches','maximum_five_step_errors','maximum_one_step_native_control_errors','contact_discrepancies')}),flush=True)
        atomic_json(a.data/'audit.json',dict(branches=checks,source_hashes=sources,policy_training_started=False))


if __name__=='__main__':main()
