"""Independent per-native-frame and five-decision audits for16/64ms episodes.

Recorded native controls are exogenous inputs ONLY for individual-frame audits.
Five-decision forecasts use one initial snapshot plus recorded PROPOSED replies;
all later measured states are comparison targets, never predictor inputs.
"""
import copy
import numpy as np

from nav.marble_dynamics.evaluate import load_rows,errors
from nav.marble_dynamics.bridge import predict_replies
from nav.marble_dynamics.rollout import predict
from nav.marble_dynamics.events import external_changes
from nav.marble_dynamics.blast import event as blast_event
from nav.marble_dynamics.record import sha
from nav.marble_dynamics.reliability_eval import pending_reply_matches


def audit_branch(root,report,engine):
    import json
    folder=root/report['map']/str(report['seed'])/report['branch']
    logs=json.loads((folder/'decisions.json').read_text());native=load_rows(folder/'native.jsonl.gz')
    by={r['sim_ms']:r for r in native};stale=[];pose=[];inputs=[];scripts=[];excluded_scripts=[];blasts=[]
    one_step=[];one_excluded=[];windows=[];excluded=[];moves=[];cameras=[]
    engine.bridge_physics_ms=16
    for i,log in enumerate(logs):
        dynamics=log['dynamics'];row=by[dynamics['native']['frame_clock_ms']]
        if i and not pending_reply_matches(dynamics['bridge_action'],logs[i-1]['action']):stale.append(i)
        if errors(dynamics['native']['state'],row['before'])[0]>1e-7:pose.append(i)
        if dynamics['native']['pending_move']!=row['move']:inputs.append(i)
        try:impulse=blast_event(dynamics.get('blast'),dynamics['native']['state'],bool(dynamics['bridge_action'][8]))
        except ValueError as exc:excluded_scripts.append(dict(decision=i,reason=str(exc)));continue
        expected=np.asarray(dynamics['native']['state']['velocity'])
        if impulse is not None:
            expected=expected+impulse['delta_velocity'];blasts.append(dict(decision=i,event=impulse))
        discrepancy=float(np.linalg.norm(expected-row['before']['velocity']))
        if discrepancy>1e-5:scripts.append(dict(decision=i,velocity_error=discrepancy))
    begin=logs[0]['dynamics']['native']['frame_clock_ms'] if logs else 0;end=begin+64*len(logs)
    rows=[r for r in native if begin<=r['sim_ms']<end]
    for offset,row in enumerate(rows):
        macro=offset//4
        if row['before']['mode']!=0 or row['before']['oob'] or logs[macro]['outcome']['fell']:
            one_excluded.append(dict(decision=macro,native_frame=offset,reason='nonplayable_or_fall'));continue
        if row['dt_ms']!=16:raise ValueError('Unexpected native physics frame cadence')
        control=dict(move=row['move'],events=row['control_events'])
        result=predict(engine,row['before'],[control],16)
        if result['failure']:
            one_step.append(dict(decision=macro,native_frame=offset,failure=result['failure'],errors=None));continue
        changed=external_changes(result['states'][-1],row['after'])
        if changed:
            one_excluded.append(dict(decision=macro,native_frame=offset,reason='external_event',changes=changed));continue
        one_step.append(dict(decision=macro,native_frame=offset,failure=None,
            errors=errors(result['states'][-1],row['after']),ambiguous_intervals=result['ambiguous_intervals']))
    starts=set(range(0,max(0,len(logs)-4),5));starts.update([0,max(0,len(logs)-5)])
    for log in logs:
        if log['score_delta']>0:starts.update(range(max(0,log['decision']-4),log['decision']+1))
    for start in sorted(starts):
        window=logs[start:start+5]
        if len(window)<5:continue
        if any(l['predicted_next'] is None or external_changes(l['predicted_next'],l['actual_next']) or l['outcome']['fell'] for l in window):
            excluded.append(dict(start=start,reason='external_event_or_fall'));continue
        forecast=predict_replies(engine,window[0]['dynamics'],[l['action'] for l in window])
        clock=window[0]['dynamics']['native']['frame_clock_ms']
        for j,control in enumerate(forecast['native_controls']):
            actual=by[clock+16*j]
            if control['move']!=actual['move']:moves.append(dict(start=start,native_offset=j))
            expected=control['events'];got=actual['control_events']
            if len(expected)!=len(got) or any(a['kind']!=b['kind'] or abs(a['value']-b['value'])>1e-5 for a,b in zip(expected,got)):
                cameras.append(dict(start=start,native_offset=j))
        windows.append(dict(start=start,failure=forecast['failure'],
            endpoint_errors=errors(forecast['states'][-1],window[len(forecast['states'])-2]['actual_next'])))
    score=sum(l['score_delta'] for l in logs)
    if abs(score-report['actual_points'])>.01:raise ValueError('Native score deltas do not match report')
    checked=[v['errors'] for v in one_step if v['errors']]
    return dict(map=report['map'],seed=report['seed'],branch=report['branch'],actual_points=score,
        physics_frame_ms=16,decision_ms=64,native_transitions=len(rows),
        snapshot_position_mismatches=pose,snapshot_move_mismatches=inputs,stale_pending_replies=stale,
        five_step_windows=len(windows),movement_mismatches=moves,camera_mismatches=cameras,
        script_actuation_mismatches=scripts,script_actuation_exclusions=excluded_scripts,
        regular_blast_activations=[b for b in blasts if not b['event']['special']],
        special_blast_activations=[b for b in blasts if b['event']['special']],
        excluded_windows=excluded,prediction_failures=sum(w['failure'] is not None for w in windows),
        maximum_five_step_errors=np.max([w['endpoint_errors'] for w in windows],axis=0).tolist() if windows else None,
        one_step_native_control_trials=one_step,one_step_exclusions=one_excluded,
        maximum_one_step_native_control_errors=np.max(checked,axis=0).tolist() if checked else None,
        contact_discrepancies=[v for v in one_step if v['errors'] and (v['errors'][0]>1e-5 or v['errors'][2]>.01)],
        native_powerup_activations=[dict(clock=r['sim_ms'],type=r['before']['held']) for r in rows if r['move']['use'] and r['before']['held'] and not r['after']['held']],
        jump_with_support_intervals=sum(int(r['move']['jump'] and r['telemetry']['support_steps']>0) for r in rows),
        native_frame_count_expected=4*len(logs),decision_log_sha256=sha(folder/'decisions.json'),native_archive_sha256=sha(folder/'native.jsonl.gz'),
        caveat='Per-native-frame actual controls are audit only. Five-decision forecasts use one initial snapshot and proposed replies, without future state corrections. Unsupported/external transitions are explicit exclusions. Supported jump frames are a proxy, not impulse count.')
