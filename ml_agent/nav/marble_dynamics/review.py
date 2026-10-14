"""Audit saved native evidence and produce a consistent-version review report.

No game, policy, optimizer or checkpoint is started. Includes a candidate API check
using only the initial physical state and applied controls, not future camera data.
"""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from nav.collision.mesh import CollisionMap,atomic_json
from nav.collision.viewer import write_atomic
from nav.marble_dynamics.evaluate import continuous_prefix,errors,stats
from nav.marble_dynamics.physics import GeneralPhysics
from nav.marble_dynamics.representation import SurfaceHierarchy
from nav.marble_dynamics.rollout import predict
from nav.marble_dynamics.outcomes import compare as compare_outcomes
from nav.marble_dynamics.record import ML

DATASETS=('pilot_20261011','effects_20261012','timers_20261012','interventions_20261012','controls_20261012','wall_jumps_20261012')
EVALUATIONS=('pilot_20261011','timers_20261012','interventions_20261012','controls_20261012','wall_jumps_20261012')


def read(path):return json.loads(path.read_text())


def audit(name):
    folder=ML/'datasets/marble_dynamics'/name
    result=dict(dataset=name,maps=0,valid_trials=0,attempted=0,native_transitions=0,
                audited_archives=0,principal_archive_hash_checks=0,replays=0,replay_length_differences=0,pair_checks=[])
    for path in sorted(folder.glob('*/manifest.json')):
        manifest=read(path);mesh=CollisionMap(ML/'collision_maps'/(manifest['map']+'.npz'))
        if manifest['geometry_sha256']!=mesh.info['mesh_sha256']:raise ValueError('Dataset geometry changed')
        result['maps']+=1;result['attempted']+=len(manifest['trials'])
        for trial in manifest['trials']:
            if not trial['valid']:continue
            raw=gzip.decompress((path.parent/trial['file']).read_bytes())
            if hashlib.sha256(raw).hexdigest()!=trial['raw_sha256']:raise ValueError('Native evidence hash changed')
            records=[json.loads(s) for s in raw.decode().splitlines()]
            count=sum(r['kind']=='transition' for r in records)
            if records[0]['kind']!='header' or records[-1]['kind']!='complete':raise ValueError('Incomplete recording')
            if count!=trial['samples'] or count!=records[-1]['samples']:raise ValueError('Native transition count mismatch')
            result['valid_trials']+=1;result['native_transitions']+=count;result['audited_archives']+=1
            result['principal_archive_hash_checks']+=1
        for replay in manifest['replay_checks']:
            if not replay.get('valid'):raise ValueError('Invalid replay: retain and investigate')
            records=[json.loads(s) for s in gzip.decompress((path.parent/replay['file']).read_bytes()).decode().splitlines()]
            native=[r for r in records if r['kind']=='transition']
            if records[-1]['kind']!='complete' or records[-1]['samples']!=len(native):raise ValueError('Incomplete repeat')
            original=next(t for t in manifest['trials'] if t['id']==replay['id'])
            original_records=[json.loads(s) for s in gzip.decompress((path.parent/original['file']).read_bytes()).decode().splitlines()]
            original_rows=[r for r in original_records if r['kind']=='transition']
            maximum={k:max(float(np.linalg.norm(np.asarray(a['after'][k])-b['after'][k])) for a,b in zip(original_rows,native))
                     for k in ('position','velocity','spin')}
            if maximum!=replay['max_errors'] or any(v!=0 for v in maximum.values()):raise ValueError('Repeat physical mismatch')
            if min(len(native),len(original_rows))!=replay['matched_samples']:raise ValueError('Repeat shared-prefix length mismatch')
            result['audited_archives']+=1;result['replays']+=1
            result['replay_length_differences']+=not replay['same_length']
        result['pair_checks'].extend(dict(map=manifest['map'],**c) for c in manifest.get('pair_checks',[]))
    for pair in result['pair_checks']:
        if pair['maximum_initial_position_difference_u'] or pair['maximum_initial_velocity_difference_u_per_s']:
            raise ValueError('Intervention initial states differ')
    return result


def candidate_checks():
    output=[];groups=[];viewer_cases={}
    paths=[path for name in ('controls_20261012','wall_jumps_20261012')
           for path in sorted((ML/'datasets/marble_dynamics'/name).glob('*/manifest.json'))]
    for path in paths:
        manifest=read(path);name=manifest['map'];mesh=CollisionMap(ML/'collision_maps'/(name+'.npz'))
        engine=GeneralPhysics(mesh);hierarchy=SurfaceHierarchy.load(ML/'collision_maps/hierarchies'/(name+'.npz'),mesh)
        for trial in manifest['trials']:
            if not trial['valid']:continue
            records=[json.loads(s) for s in gzip.decompress((path.parent/trial['file']).read_bytes()).decode().splitlines()]
            rows,removed=continuous_prefix(trial,[r for r in records if r['kind']=='transition'])
            if not rows:continue
            clock=rows[0]['sim_ms'];dt=rows[0]['dt_ms']
            if any(r['sim_ms']!=clock+i*dt or r['dt_ms']!=dt for i,r in enumerate(rows)):
                raise ValueError('Nonuniform applied-control timing')
            if not all('control_events' in r for r in rows):raise ValueError('Missing complete applied-control contract')
            result=predict(engine,rows[0]['before'],[dict(move=r['move'],events=r['control_events']) for r in rows],dt,hierarchy=hierarchy)
            count=len(result['states'])-1;actual=[rows[0]['before']]+[r['after'] for r in rows[:count]]
            errs=[errors(p,a) for p,a in zip(result['states'],actual)]
            item=dict(dataset=path.parent.parent.name,map=name,trial=trial['id'],pair_group=trial['pair_group'],intervention=trial['intervention'],
                      samples=count,final_errors=errs[-1],max_position_error_u=max(e[0] for e in errs),
                      failure=result['failure'],ambiguous_intervals=result['ambiguous_intervals'],
                      corridor_triangles=len(result['corridor_triangle_ids']),truncated_external_samples=removed,
                      future_observations_supplied=False)
            output.append(item)
            times=result['time_s']
            case=dict(id=trial['id'],family=trial['family'],powerup=trial['spec']['powerup'],split=manifest['split'],
                      source_dataset=path.parent.parent.name,pair_group=trial['pair_group'],intervention=trial['intervention'],
                      time_s=times,position_error=[e[0] for e in errs],max_position_error_u=item['max_position_error_u'],
                      final_errors=item['final_errors'],truncated_external_samples=removed,rollout_failure=result['failure'],
                      ambiguous_contact_intervals=result['ambiguous_intervals'],
                      control_basis='Initial state + recorded applied Move and explicit camera-command events; no future observations',
                      native_use_events=[dict(sample=i,time_s=times[i],held_id=r['before']['held']) for i,r in enumerate(rows[:count])
                                         if r['move']['use'] and r['before']['held']],
                      outcomes=compare_outcomes(rows,actual,result['states'],result['events'],np.asarray(times)))
            for basis,states in [('actual',actual),('predicted',result['states'])]:
                for key in ('position','velocity','spin'):case[basis+'_'+key]=[s[key] for s in states]
            viewer_cases.setdefault(name,dict(map=name,geometry_sha256=mesh.info['mesh_sha256'],trajectories=[]))['trajectories'].append(case)
            z=[s['position'][2] for s in actual]
            groups.append(dict(dataset=path.parent.parent.name,map=name,trial=trial['id'],pair_group=trial['pair_group'],intervention=trial['intervention'],
                               initial_spin=actual[0]['spin'],peak_height_u=max(z),
                               final_position=actual[-1]['position'],final_velocity=actual[-1]['velocity'],
                               final_spin=actual[-1]['spin'],evaluated_duration_s=count*dt*.001,
                               native_jump_command_intervals=sum(bool(r['move']['jump']) for r in rows[:count]),
                               native_jump_commands_with_support=sum(bool(r['move']['jump']) and r['telemetry']['support_steps']>0 for r in rows[:count]),
                               native_use_events=[dict(sample=i,held_id=r['before']['held']) for i,r in enumerate(rows[:count])
                                                  if r['move']['use'] and r['before']['held']],
                               truncated_external_samples=removed))
    for name,payload in viewer_cases.items():
        payload['rollout_source_sha256']=hashlib.sha256(Path(__file__).with_name('rollout.py').read_bytes()).hexdigest()
        payload['review_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        atomic_json(ML/'dynamics_reports/candidate_replay_20261012'/(name+'.trajectories.json'),payload)
    return dict(cases=output,final_errors=stats([c['final_errors'] for c in output if not c['failure']]),
                by_dataset={name:stats([c['final_errors'] for c in output if c['dataset']==name and not c['failure']])
                            for name in ('controls_20261012','wall_jumps_20261012')},
                no_future_camera_or_state=True,planner_approved=False),groups


def main():
    inventory=[audit(name) for name in DATASETS]
    evaluations={}
    versions=set()
    for name in EVALUATIONS:
        summary=read(ML/'dynamics_reports'/name/'summary.json')
        if summary['stale_reports']:raise ValueError('Mixed-version evaluation reports')
        evaluations[name]=summary['maps']
        for report in summary['maps']:
            versions.add(report['model_version_sha256'])
            for file,sha in report['source_hashes'].items():
                if hashlib.sha256((ML/file).read_bytes()).hexdigest()!=sha:raise ValueError('Evaluation code changed since report')
    if len(versions)!=1:raise ValueError('Evaluation sets use different predictor versions')
    candidate,interventions=candidate_checks()
    hierarchy=read(ML/'collision_maps/hierarchies/checks.json')
    sensitivity=read(ML/'dynamics_reports/contact_sensitivity_20261012/CubeIsle_Hunt.sensitivity.json')
    result=dict(schema='marble_dynamics_review_v1',model_version_sha256=next(iter(versions)),datasets=inventory,
                candidate_rollout_source_sha256=hashlib.sha256(Path(__file__).with_name('rollout.py').read_bytes()).hexdigest(),
                review_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                totals={k:sum(d[k] for d in inventory) for k in ('valid_trials','native_transitions','audited_archives','principal_archive_hash_checks','replays','replay_length_differences')},
                evaluations=evaluations,candidate_api=candidate,matched_native_outcomes=interventions,
                hierarchy=hierarchy,sensitivity=sensitivity,policy_training_started=False,learned_dynamics_fit=False,
                learned_encoder_trained=False,planner_approved=False,uncertainty_calibrated=False)
    atomic_json(ML/'dynamics_reports/review_20261012.json',result)
    lines=['# General marble dynamics review (2026-10-12)','',
           'Native recording, complete-map access, an independent mechanics baseline and review overlays are implemented. '
           'The policy has not been connected or trained. Long-horizon contact outliers and external events keep planning gated.','',
           '## Evidence','',
           '| Dataset | Maps | Accepted fixtures | Native transitions | Repeats |',
           '|---|---:|---:|---:|---:|']
    for d in inventory:lines.append('| %s | %d | %d | %d | %d |'%(d['dataset'],d['maps'],d['valid_trials'],d['native_transitions'],d['replays']))
    lines.extend(['',f"All {result['totals']['audited_archives']} archives passed structure checks; {result['totals']['principal_archive_hash_checks']} principal recordings match their saved raw hashes. Repeats independently match on shared physical prefixes; "
                  f"{result['totals']['replay_length_differences']} differ in episode termination length. Repeat data is not counted as independent fixtures.",
                  'Map/trajectory roles were assigned before collection; historical policies have seen these maps. '
                  'These fixtures do not prove exhaustive mechanics coverage.','',
                  '## Broad ten-map validation','',
                  'Errors below are game units. Endpoints use variable-length continuous prefixes (external item/respawn/size changes stop a rollout). '
                  'Failures are retained separately; endpoint statistics exclude failed rollouts. Each legacy map has one refused Super Speed '
                  'one-step sample because the old schema lacks its prior contact normal.','',
                  '| Map | 16 ms p95 | 1 s p95 | 2 s p95 | Endpoint p95 | Endpoint max | Complete / accepted fixtures |',
                  '|---|---:|---:|---:|---:|---:|---:|'])
    for r in evaluations['pilot_20261011']:
        h=r['rollout_horizons'];f=r['rollout_final']
        lines.append('| %s | %.3g | %.4g | %.4g | %.4g | %.4g | %d / %d |'%(
            r['map'].replace('_Hunt',''),r['one_step']['p95'][0],h['1.0']['p95'][0],h['2.0']['p95'][0],
            f['p95'][0],f['max'][0],f['count'],r['accepted_native_fixtures']))
    f=candidate['final_errors']
    supported=sum(c['native_jump_commands_with_support'] for c in interventions if c['dataset']=='wall_jumps_20261012')
    requested=sum(c['native_jump_command_intervals'] for c in interventions if c['dataset']=='wall_jumps_20261012')
    lines.extend(['','## Matched mechanics and map access','',
                  f"The {len(candidate['cases'])} spin/jump/powerup candidates also run from their initial state and applied Moves/absolute camera commands with "
                  f"no future camera/state observations. Endpoint p95 = {f['p95'][0]:.6g}u; max = {f['max'][0]:.6g}u on these selected fixtures. "
                  'Physical outcomes, initial spin, actual used item IDs and event truncations are in review_20261012.json. '
                  'Matched landing differences illustrate mechanics, not learned policy skill.',
                  f"The additional wall-approach set has {supported}/{requested} actual jump-command intervals with native support, "
                  'versus 4/16 in the original wall-timing set. Support is a necessary actuation proxy, not an exact native impulse count. '
                  'No-jump controls and longer approaches expose timing-dependent contact outcomes.',
                  'An additional control-contract check found that Move-only replay of old tapes can miss by ~57u: the bridge sets '
                  'absolute camera yaw outside Move. The new controls dataset records those actual command events explicitly. '
                  'Proposed camera commands now drive the candidate interface; future observed camera state is not copied.',
                  'All 47,932 static contact triangles are retained in the hierarchy; 2,000 reloaded queries match the exact reference. '
                  '64–512 whole-map leaf tokens are accessible. The encoder is randomly initialized; exact distant detail remains queryable. '
                  'The Python hierarchy is slower than the vectorized reference on these map sizes.',
                  '', '## Remaining gates','',
                  '- Rare multi-contact/contact-set errors can lead to multi-unit rollout drift. Native local hull order and F32 handling reduce '
                  'errors but do not eliminate them. Tangent sqrt/NaN and slide/bounce branches are explicitly diagnosed.',
                  '- Microscopic perturbations produce alternative rebound outcomes. The selected 24-member demonstration is not calibrated '
                  'coverage; uncertainty must be measured on independent contact trajectories before relying on landing clearance.',
                  '- Mega/scripted size changes, Blast, pickups, respawns, arbitrary gravity rotation and moving actors need a discrete-event '
                  'or dynamic-object layer. Applied Move replay does not validate delayed bridge action execution.',
                  '- No learned dynamics model or learned map encoder has been fit, and no online planner or policy integration is approved.',
                  '', 'Open collision_maps/viewer/index.html, choose a recording, then focus/scrub the actual and predicted paths. '
                  'The primary change log and exact commands are in docs/HANDOFF_MARBLE_DYNAMICS_2026-10-11.md.',
                  '',f"Predictor/evaluation version: `{result['model_version_sha256']}`."])
    write_atomic(ML/'docs/REVIEW_MARBLE_DYNAMICS_2026-10-12.md','\n'.join(lines)+'\n')
    print(json.dumps(dict(totals=result['totals'],candidate_final=f,model_version=next(iter(versions)),planner_approved=False)),flush=True)


if __name__=='__main__':main()
