"""Matched native physics interventions: spin, jump timing and powerup modifiers.

No reward or policy is involved. Labels are physical outcomes. The same fixture
is varied systematically, retaining falls and discrete external events.
"""
from nav.source_archive import snapshot_sources,write_archive
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from nav.collision.mesh import CollisionMap,atomic_json
from nav.marble_dynamics.evaluate import load_rows
from nav.marble_dynamics.record import ML,RUNTIME,DEV,TEST,POWERUPS,Sampler,game,run_trial,archive,sha


def specifications(sampler,group,jump_contact=False):
    result=[]
    if jump_contact:
        # Start far enough from a wall to survive reset/command pipeline latency,
        # grounded with rolling spin. Timing variants then act before/near impact.
        # This is a general floor/wall mechanics fixture, not a route/trick rule.
        for _ in range(100):
            base=sampler.sample('edge_timing',group)
            if not base:continue
            n=np.asarray(base['target_normal']);direction=-n.copy();direction[2]=0
            direction/=np.linalg.norm(direction)
            position=np.asarray(base['position']);target=np.asarray(base['target_point'])
            offset=float((position-target)@n)
            position+=n*(2.5+base['radius']-offset)
            floor=sampler.mesh.raycast(position+[0,0,2],position-[0,0,30],cull_backfaces=True)
            if not floor or floor['normal'][2]<.99:continue
            position[2]=floor['point'][2]+base['radius']+.005
            if len(sampler.mesh.sphere_contacts(position,base['radius']+.002)['triangle_ids']):continue
            base['position']=position.tolist();base['velocity']=(direction*8.).tolist()
            base['spin']=(np.cross([0,0,1],direction*8.)/base['radius']).tolist()
            base['frames']=200;base['powerup']='none'
            for frame in (-1,0,8,12,16,20):
                spec=copy.deepcopy(base);spec['jump_frame']=frame
                result.append(('wall_jump_timing_'+str(group),dict(requested_jump_frame=frame if frame>=0 else None),spec))
            return result
        raise RuntimeError('No clear supported wall approach for matched jump timing')
    base=sampler.sample('effect_expiry',0)
    if base:
        base['family']='drop_spin';base['powerup']='none';base['frames']=200
        direction=np.asarray([np.cos(base['heading']),np.sin(base['heading']),0.])
        base['velocity']=(direction*6+np.asarray([0.,0.,-8.])).tolist()
        for spin in (-100.,0.,100.):
            spec=copy.deepcopy(base);spec['spin']=(np.cross([0,0,1],direction)*spin).tolist()
            result.append(('spin_'+str(group),dict(spin_rad_per_s=spin),spec))
    base=sampler.sample('edge_timing',0)
    if base:
        for frame in (0,4,8,12):
            spec=copy.deepcopy(base);spec['jump_frame']=frame
            result.append(('jump_timing_'+str(group),dict(requested_jump_frame=frame),spec))
    base=sampler.sample('effect_expiry',0)
    if base:
        base['family']='drop_spin';base['frames']=375
        for power in ('none',)+POWERUPS[:5]:
            spec=copy.deepcopy(base);spec['powerup']=power
            result.append(('power_modifier_'+str(group),dict(requested_powerup=power),spec))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--maps',nargs='+',default=['Concentric_Hunt','Sprawl_Hunt','CubeIsle_Hunt','Bowl_Hunt'])
    p.add_argument('--groups',type=int,default=1);p.add_argument('--port',type=int,default=9053)
    p.add_argument('--seed',type=int,default=None,help='Pin both native game and geometry sampling; omitted preserves old sampler/native defaults')
    p.add_argument('--jump-contact',action='store_true',help='Supported wall approaches with no-jump and varied jump timing')
    p.add_argument('--fixture-roundtrip-probe',action='store_true',help='Bounded atomic native copy/restore check in a separate executable; no intervention dataset')
    p.add_argument('--fixture-prefix-probe',action='store_true',help='Three bounded repeated physical prefixes from an in-process capture; no full-world restoration claim')
    p.add_argument('--fixture-natural-flight',action='store_true',help='Predeclared881015 prefix retry reaches free flight by native jump rather than teleport')
    p.add_argument('--fixture-response-probe',action='store_true',help='Four predeclared881016 paired physical responses; actual applied controls retained')
    p.add_argument('--fixture-pose-probe',action='store_true',help='Predeclared881017 atomic synthetic placement and repeated-prefix check')
    p.add_argument('--fixture-pool-probe',action='store_true',help='Declared six-training-map transfer preflight of four physical responses per map')
    p.add_argument('--fixture-collection',action='store_true',help='Declared72-request varied ordinary native physical-pair collection; no fit')
    p.add_argument('--reader-index-from',type=Path,help='Write a compatible metadata index referencing immutable physical-pair archives; no record copies')
    p.add_argument('--prepare-paired-supervision',action='store_true',help='Prepare first-changed-native-Move action differences from verified train-only physical pairs')
    p.add_argument('--fixture-script-probe',action='store_true',help='Six declared independent Cube power/script-state observability cases; no fit')
    p.add_argument('--fixture-state-coverage-probe',action='store_true',help='Six declared Cube held/released-use and extended native context checks; no fit')
    p.add_argument('--world-transition-audit-from',type=Path,help='Prepare observed next-state schema prototype from completed six-power observability records; no fit')
    p.add_argument('--prepare-observed-context-from',type=Path,help='Prepare semantic current-observation/known-reply features and next-state targets; engineering only')
    p.add_argument('--paired-analytical-audit',action='store_true',help='Compare analytical and velocity-drift predictions on90verified train-only action pairs; no fit')
    p.add_argument('--paired-open-loop-audit',action='store_true',help='Compare complete recorded8/16-frame known-native-control forecasts on90train pairs; no fit')
    p.add_argument('--compiled-sweep-dll',type=Path,help='Explicit validated ABI3 sweep DLL for the paired analytical audit')
    p.add_argument('--fixture-scene-probe',action='store_true',help='Declared ten-map frozen force/player scene inventory, no motion labels or fit')
    p.add_argument('--output',type=Path,default=ML/'datasets/marble_dynamics/interventions_20261012')
    args=p.parse_args();summary=[]
    if args.fixture_scene_probe:return scene_context_probe(args)
    if args.paired_analytical_audit or args.paired_open_loop_audit:
        return audit_paired_analytical_baseline(args.output,args.compiled_sweep_dll,args.paired_open_loop_audit)
    if args.prepare_observed_context_from:return prepare_observed_context(args.prepare_observed_context_from,args.output)
    if args.world_transition_audit_from:return prepare_observed_world_transitions(args.world_transition_audit_from,args.output)
    if args.fixture_script_probe or args.fixture_state_coverage_probe:return script_context_probe(args)
    if args.prepare_paired_supervision:return prepare_action_difference_data(args.output)
    if args.reader_index_from:return prepare_physical_reader_index(args.reader_index_from,args.output)
    if sum((args.fixture_roundtrip_probe,args.fixture_prefix_probe,args.fixture_response_probe,args.fixture_pose_probe,args.fixture_pool_probe,args.fixture_collection))>1:raise ValueError('Choose one fixture protocol')
    if args.fixture_natural_flight and not args.fixture_prefix_probe:raise ValueError('Natural-flight switch requires the declared prefix protocol')
    if args.fixture_pool_probe:return fixture_pool_probe(args)
    if args.fixture_collection:return collect_native_physical_pairs(args)
    if args.fixture_roundtrip_probe or args.fixture_prefix_probe or args.fixture_response_probe or args.fixture_pose_probe:
        return fixture_roundtrip_probe(args)
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'prepared.json').exists():raise RuntimeError('Preserve existing collection; use a new output directory')
    sources=list(Path(__file__).parent.glob('*.py'))+[ML/'nav/collision/mesh.py',
        ML.parent/'Marble Blast Platinum/platinum/client/scripts/ai/mlAgent.cs',
        ML.parent/'Marble Blast Platinum/marbleblast_geometry_probe.exe']
    fingerprints={str(path):sha(path) for path in sources}
    atomic_json(args.output/'prepared.json',dict(maps=args.maps,groups=args.groups,jump_contact=args.jump_contact,
        seed=args.seed,source_hashes=fingerprints,training=False,scored_navigation=False,
        contract='Controlled native mechanics fixtures; setup-only respawn/grant/teleport. Complete initial native states and actual exogenous controls retained. No policy fit or game-score optimization.'))
    snapshot_sources(args.output,[path for path in sources if path.suffix.lower() in ('.py','.cs')])
    for mi,name in enumerate(args.maps):
        if any(sha(Path(path))!=fingerprint for path,fingerprint in fingerprints.items()):
            raise RuntimeError('Collection source changed after preparation; use a new output')
        mesh=CollisionMap(ML/'collision_maps'/(name+'.npz'));sampler=Sampler(mesh,(args.seed if args.seed is not None else 20261012)+mi)
        dest=args.output/name;dest.mkdir(parents=True,exist_ok=True)
        if (dest/'manifest.json').exists():raise RuntimeError('Preserve existing map collection: '+str(dest))
        manifest=dict(schema='marble_trajectories_v1',map=name,geometry_sha256=mesh.info['mesh_sha256'],
                      split='dev' if name in DEV else 'test' if name in TEST else 'train',trials=[],replay_checks=[],
                      probe_executable_sha256=sha(ML.parent/'Marble Blast Platinum/marbleblast_geometry_probe.exe'),
                      native_instrumentation_sha256=sha(Path(__file__).with_name('engine_dynamics.inc')),
                      contract='Matched starting fixture with one intervention; actual initial p/v/spin and applied controls retained. No policy/reward.')
        with game(name,args.port,seed=args.seed+mi if args.seed is not None else None) as (env,proc):
            manifest['owned_game_pid']=proc.pid
            for group in range(args.groups):
                for pair_group,intervention,spec in specifications(sampler,group,args.jump_contact):
                    ident=len(manifest['trials']);raw=RUNTIME/(name+'.intervention_%04d.jsonl'%ident)
                    result=run_trial(env,spec,raw)
                    row=dict(id=ident,family=spec['family'],spec=spec,pair_group=pair_group,intervention=intervention,**result)
                    if result['valid']:
                        saved=dest/('trial_%04d.jsonl.gz'%ident);archive(raw,saved);row['file']=saved.name;row['raw_sha256']=sha(raw)
                        row['actual_initial_state']=load_rows(saved)[0]['before']
                    manifest['trials'].append(row);atomic_json(dest/'manifest.json',manifest)
                    print(json.dumps(dict(map=name,trial=ident,group=pair_group,valid=row['valid'],samples=row.get('samples'))),flush=True)
        groups={}
        for row in manifest['trials']:
            if row['valid']:groups.setdefault(row['pair_group'],[]).append(row)
        matched=[]
        for key,trials in groups.items():
            first=trials[0]['actual_initial_state']
            matched.append(dict(group=key,trials=len(trials),
                                maximum_initial_position_difference_u=max(float(np.linalg.norm(np.asarray(t['actual_initial_state']['position'])-first['position'])) for t in trials),
                                maximum_initial_velocity_difference_u_per_s=max(float(np.linalg.norm(np.asarray(t['actual_initial_state']['velocity'])-first['velocity'])) for t in trials)))
        manifest['pair_checks']=matched;atomic_json(dest/'manifest.json',manifest)
        summary.append(dict(map=name,trials=len(manifest['trials']),valid=sum(t['valid'] for t in manifest['trials']),
                            native_transitions=sum(t.get('samples',0) for t in manifest['trials']),pair_checks=matched))
        atomic_json(args.output/'collection.json',dict(maps=summary,policy_training_started=False))


def frozen_fixture_command(env,word,timeout_s=5,ack_name='fixture_roundtrip'):
    """Receive only DEBUG while preserving the pending frozen observation.

    HuntEnv.control intentionally waits for another observation; using it here
    consumes the ack and deadlocks. This helper refuses unsolicited observations
    and disconnects rather than silently reconnecting/restarting the fixture.
    """
    import select,time
    from nav.protocol import parse_message
    if timeout_s<=0:raise TimeoutError('No frozen fixture acknowledgement: deadline already expired')
    deadline=time.perf_counter()+timeout_s
    env.conn.sendall((word+'\n').encode('utf-8'))
    while True:
        index=env.buf.find(b'\n')
        if index>=0:
            line,env.buf=env.buf[:index+1],env.buf[index+1:]
            ack=parse_message(line.decode('utf-8','replace'))
            if ack.kind!='debug' or not ack.fields or ack.fields[0]!=ack_name:
                raise ValueError('Unexpected frozen acknowledgement: '+ack.raw)
            return ack
        remaining=deadline-time.perf_counter()
        if remaining<=0 or not select.select([env.conn],[],[],remaining)[0]:
            raise TimeoutError('No frozen fixture acknowledgement')
        block=env.conn.recv(65536)
        if not block:raise ConnectionError('Frozen fixture disconnected')
        env.buf+=block


def fixture_prefix_context(env,output,context,intervention=None,replies=8):
    """Compare all measured frames; a refused restore remains a visible result."""
    from nav.protocol import NOOP_ACTION
    from nav.marble_dynamics.evaluate import errors
    capture=(output/(context+'.capture.json')).resolve()
    ack=frozen_fixture_command(env,'FIXTURECAPTURE '+capture.as_posix(),ack_name='fixture_capture')
    initial=json.loads(capture.read_text(encoding='utf-8'))
    result=dict(context=context,capture=initial,branches=[],passed=False,refused=initial.get('refused'),decisions=0)
    if ack.fields[1]!='1' or not initial['passed']:return result
    for branch in range(2):
        if branch:
            restore=(output/(context+'.restore.json')).resolve()
            ack=frozen_fixture_command(env,'FIXTURERESTORE '+restore.as_posix(),ack_name='fixture_restore')
            restored=json.loads(restore.read_text(encoding='utf-8'));result['restore']=restored
            if ack.fields[1]!='1' or not restored['passed']:
                result['refused']=restored.get('refused') or 'native_restore_failed';return result
            result['restored_field_equality']=all(initial[k]==restored[k] for k in ('client_fields','server_fields'))
        raw=(output/(context+'.branch'+str(branch)+'.native.jsonl')).resolve()
        ack=frozen_fixture_command(env,'FIXTURERECORD '+raw.as_posix(),ack_name='fixture_record')
        if ack.fields[1]!='1':raise ValueError('Frozen record start refused')
        outcomes=[]
        for index in range(replies):
            reply=intervention if branch and index==0 and intervention else {}
            _,outcome=env.step(reply.get('action',NOOP_ACTION),repeat=1,use_pow=reply.get('use',0),pow_yaw=reply.get('pow_yaw'))
            outcomes.append(outcome);result['decisions']+=1
            if any(outcome[k] for k in ('fell','round_ended','reconnected')):break
        ack=frozen_fixture_command(env,'FIXTURERECORD off',ack_name='fixture_record')
        if ack.fields[1]!='1':raise ValueError('Frozen record close refused')
        packed=raw.with_suffix(raw.suffix+'.gz');archive(raw,packed);rows=load_rows(packed)
        # archive leaves the verified original: remove this closed duplicate only
        # after exact gzip byte comparison, with its removal receipt in the result.
        import gzip
        digest=sha(raw)
        if gzip.decompress(packed.read_bytes())!=raw.read_bytes():raise ValueError('Compressed fixture record disagrees')
        raw.unlink()
        result['branches'].append(dict(rows=len(rows),native_sha256=digest,compressed_sha256=sha(packed),
            raw_duplicate_removed=True,outcomes=outcomes,first_before=rows[0]['before'] if rows else None,
            first_move=rows[0]['move'] if rows else None,path=packed.name))
    a,b=[load_rows(output/v['path']) for v in result['branches']]
    if len(a)!=replies or len(b)!=replies:result['refused']='incomplete_native_prefix';return result
    keys=('position','velocity','spin','camera_offset','gravity_direction','best_contact_normal','ground_ms',
        'radius','datablock','parameters','yaw','pitch','next_yaw','next_pitch','smooth_camera','held','mode','oob','effects')
    result['initial_mismatch_fields']=[k for k in keys if a[0]['before'][k]!=b[0]['before'][k]]
    result['initial_tick_phase_equal']=a[0]['sim_ms']%32==b[0]['sim_ms']%32
    result['move_mismatches']=[i for i,(x,y) in enumerate(zip(a,b)) if x['move']!=y['move']]
    result['camera_event_mismatches']=[i for i,(x,y) in enumerate(zip(a,b)) if
        [(v['kind'],v['value']) for v in x['control_events']]!=[(v['kind'],v['value']) for v in y['control_events']]]
    result['maximum_motion_errors']=np.max([errors(x['after'],y['after']) for x,y in zip(a,b)],axis=0).tolist()
    result['game_events_observed']=any(v['gem_delta'] or any(v[k] for k in ('fell','round_ended','reconnected'))
        for branch in result['branches'] for v in branch['outcomes'])
    if intervention:
        changed=result['move_mismatches'];first=changed[0] if changed else None
        result['intervention']=intervention;result['first_changed_native_move']=first
        result['first_changed_move_before_mismatch_fields']=[k for k in keys if a[first]['before'][k]!=b[first]['before'][k]] if first is not None else None
        result['unchanged_prefix_motion_errors']=np.max([errors(x['after'],y['after']) for x,y in zip(a[:first],b[:first])],axis=0).tolist() if first else None
        result['native_consumptions']=[dict(branch=j,frame=i,held=r['before']['held'],move=r['move'])
            for j,rows in enumerate((a,b)) for i,r in enumerate(rows)
            if r['before']['held']>0 and r['after']['held']==0 and r['move']['use']]
        result['initial_states_verified']=bool(result.get('restored_field_equality')) and not result['initial_mismatch_fields'] and \
            result['initial_tick_phase_equal'] and not result['game_events_observed']
        result['passed']=result['initial_states_verified'] and first is not None and not result['first_changed_move_before_mismatch_fields'] and \
            (result['unchanged_prefix_motion_errors'] is None or all(v<=1e-9 for v in result['unchanged_prefix_motion_errors']))
        return result
    result['passed']=bool(result.get('restored_field_equality')) and not result['initial_mismatch_fields'] and \
        result['initial_tick_phase_equal'] and not result['move_mismatches'] and not result['camera_event_mismatches'] and \
        not result['game_events_observed'] and all(v<=1e-9 for v in result['maximum_motion_errors'])
    return result


def paired_motion_example(pair,folder,geo):
    """Future motion is a target only; later divergent states are never paired."""
    from nav.marble_dynamics.student_data import state_features,motion,MOTION_SCALES
    if not pair.get('passed') or pair.get('refused') or pair.get('game_events_observed'):
        raise ValueError('Unverified or event-contaminated physical pair')
    if not pair.get('restored_field_equality') or not pair.get('initial_tick_phase_equal'):
        raise ValueError('Copied physical fields or tick phase not verified')
    if pair.get('initial_mismatch_fields') or pair.get('first_changed_move_before_mismatch_fields'):
        raise ValueError('Physical BEFORE mismatch')
    branches=pair['branches']
    if len(branches)!=2:raise ValueError('Exactly two physical branches required')
    rows=[]
    import gzip,hashlib
    for branch in branches:
        path=folder/branch['path']
        if sha(path)!=branch['compressed_sha256'] or hashlib.sha256(gzip.decompress(path.read_bytes())).hexdigest()!=branch['native_sha256']:
            raise ValueError('Physical branch archive changed')
        rows.append(load_rows(path))
    changed=[i for i,(a,b) in enumerate(zip(*rows)) if a['move']!=b['move']]
    index=pair.get('first_changed_native_move')
    if not changed or index!=changed[0]:raise ValueError('First changed native Move is not aligned')
    for i in range(index):
        if rows[0][i]['after']!=rows[1][i]['after']:
            # Clock advances between branches; all recorded physical values must agree.
            if any(rows[0][i]['after'][k]!=rows[1][i]['after'][k] for k in rows[0][i]['after'] if k!='clock_ms'):
                raise ValueError('Earlier native physical prefix differs')
    a,b=[r[index] for r in rows]
    if a['sim_ms']%32!=b['sim_ms']%32:raise ValueError('Native tick phase differs')
    if any(a['before'][k]!=b['before'][k] for k in a['before'] if k!='clock_ms'):
        raise ValueError('Actual first-changed-Move physical BEFORE differs')
    xs=[];ys=[]
    for row in (a,b):
        before,after=row['before'],row['after']
        if row['dt_ms']!=16 or before['mode']!=0 or after['mode']!=0 or before['oob'] or after['oob']:
            raise ValueError('Not a playable native16ms motion transition')
        if any(before[k]!=after[k] for k in ('radius','datablock','parameters')):
            raise ValueError('Within-step body transition requires a different prediction task')
        x,state=state_features(before,dict(move=row['move'],events=row['control_events']),16,geo['center'],geo['scale'])
        y=((motion(after)-motion(state))/MOTION_SCALES).astype(np.float32)
        if not np.all(np.isfinite(y)):raise ValueError('Nonfinite native motion label')
        xs.append(x);ys.append(y)
    return np.asarray(xs),np.asarray(ys),dict(native_frame=index,clock_phase=a['sim_ms']%32,
        moves=[a['move'],b['move']],target_difference=(ys[1]-ys[0]).tolist(),
        audit_support_steps=[r['telemetry']['support_steps'] for r in (a,b)])


def prepare_action_difference_data(output):
    """Metadata/tensors only, referencing canonical immutable native recordings."""
    from nav.marble_dynamics.student_data import geometry,STATE_FEATURES,MOTION_SCALES
    names=('Sprawl_Hunt','KingOfTheMarble_Hunt','Duplex_Hunt','Cragmire_Hunt','TreasureBox_Hunt','FracturedIslands_Hunt')
    roots=(ML/'datasets/marble_dynamics/native_inprocess_interventions_20261013',
        ML/'dynamics_reports/native_fixture_transfer_pool_20261013')
    output.mkdir(parents=True,exist_ok=False)
    snapshot_sources(output,[Path(__file__),Path(__file__).with_name('student_data.py'),
        Path(__file__).with_name('evaluate.py'),ML/'nav/collision/mesh.py',ML/'nav/source_archive.py'])
    records=[];sources={};excluded=[];audits=[];hashes={}
    for name in names:
        mesh=CollisionMap(ML/'collision_maps'/(name+'.npz'));geo=geometry(mesh);xs=[];ys=[]
        for root in roots:
            folder=root/name;manifest=folder/('manifest.json' if root==roots[0] else 'result.json')
            sources[str(manifest)]=sha(manifest);data=json.loads(manifest.read_text(encoding='utf-8'))
            for pair in data.get('pairs',data.get('checks',[])):
                context=pair.get('stratum',pair.get('context'))
                try:x,y,audit=paired_motion_example(pair,folder,geo)
                except ValueError as error:
                    excluded.append(dict(map=name,manifest=str(manifest),context=context,reason=str(error)));continue
                xs.append(x);ys.append(y);audits.append(dict(map=name,manifest=str(manifest),context=context,**audit))
                for branch in pair['branches']:
                    path=folder/branch['path'];sources[str(path)]=sha(path)
        if not xs:raise ValueError('No verified action pairs on '+name)
        file=output/(name+'.pairs.npz');np.savez_compressed(file,x=np.asarray(xs),y=np.asarray(ys));hashes[file.name]=sha(file)
        records.append(dict(map=name,role='train',pairs=len(xs),geometry_file_sha256=sha(ML/'collision_maps'/(name+'.npz'))))
    atomic_json(output/'audit.json',dict(pairs=audits,excluded=excluded,within_step_support_is_audit_only=True))
    atomic_json(output/'manifest.json',dict(schema='native_action_difference_v1',state_features=STATE_FEATURES,
        motion_scales=MOTION_SCALES.tolist(),map_records=records,pairs=sum(r['pairs'] for r in records),
        sources=sources,output_hashes=hashes,preparation_complete=True,policy_training=False,
        contract='First actual changed Move only; same verified physical BEFORE and earlier prefix; BEFORE+actual control inputs, AFTER motion targets. No future support label/map identity/family/reward input. Training maps only.'))
    print(json.dumps(dict(output=str(output),pairs=sum(r['pairs'] for r in records),excluded=len(excluded))),flush=True)


def fixture_roundtrip_probe(args):
    """Explicit engineering mode, isolated from historical collection/readers."""
    import subprocess,time
    from nav.env import HuntEnv
    from nav.protocol import NOOP_ACTION,RAW_POS,format_teleport
    from nav.marble_dynamics.record import command
    response=args.fixture_response_probe;pose=args.fixture_pose_probe;prefix=args.fixture_prefix_probe or response or pose
    expected_seed=881017 if pose else 881016 if response else 881015 if args.fixture_natural_flight else 881014 if prefix else 881013
    pool_protocol=getattr(args,'pool_protocol',None)
    if pool_protocol:
        declared=json.loads(pool_protocol.read_text(encoding='utf-8'))
        permitted=any(args.maps==[v['map']] and args.seed==v['seed'] for v in declared['maps'])
    else:permitted=args.maps==['CubeIsle_Hunt'] and args.seed==expected_seed
    if not permitted or args.groups!=1:
        raise ValueError('Fixture protocol is declared only for Cube/'+str(expected_seed)+'/one group')
    args.output.mkdir(parents=True,exist_ok=False)
    root=ML.parent;exe=root/'Marble Blast Platinum/marbleblast_fixture_probe.exe'
    protected={root/'Marble Blast Platinum/marbleblast_geometry_probe.exe':'084ebc5cce1444f032788282c66a35f95a728cd99c59ef4f75daa40175807926',
        root/'Marble Blast Platinum/marbleblast_mbx.exe':'f5d765ec6bb6c39fcf1d59f93a80458b9ece9d4df475a0f7b3aec473400ba3d7',
        root/'Marble Blast Platinum/platinum/client/scripts/ai/mlAgent.cs':'1e191747a6bba8bd7c419b70488cb7f968ac2d58d4639e5d7237508730b8a1f1',
        ML/'models/nav/real_maps_20261010_A2/nav_latest.pth':'ae88f2e70a376fda06be6aed59e8bd17fd6b7b0a876bdde9c40838eadddbfb04'}
    if any(sha(p)!=h for p,h in protected.items()):raise ValueError('Protected artifact changed')
    sources=list(Path(__file__).parent.glob('*.py'))+[Path(__file__).with_name(x) for x in ('engine_dynamics.inc','engine_fixture.inc','build_probe.ps1')]
    sources += [ML/'nav/collision/build_probe.ps1',root/'Marble Blast Platinum/platinum/client/scripts/ai/mlAgent.cs',
        root/'Marble Blast Platinum/platinum/client/scripts/ai/socketBridge.cs']
    fingerprints={str(p):sha(p) for p in sources};snapshot_sources(args.output,sources)
    native=Path('C:/Users/doug/src/OpenPQ-TGEMIT-mbx');native_files=[native/'engine/game/marble'/x for x in
        ('marble.cc','marble.h','aiGeometryExport.inc','aiMarbleDynamics.inc','aiMarbleFixture.inc')]
    write_archive(args.output/'native_source_snapshot.zip',native_files,native)
    native_hashes={str(p):sha(p) for p in native_files}
    atomic_json(args.output/'prepared.json',dict(map=args.maps[0],seed=args.seed,executable=str(exe),executable_sha256=sha(exe),
        protected_sha256={str(p):h for p,h in protected.items()},source_hashes=fingerprints,native_source_hashes=native_hashes,
        maximum_checks=4 if response else 3 if prefix else 9,maximum_post_startup_decisions=240,wall_limit_s=120,training=False,
        fixture_prefix_probe=args.fixture_prefix_probe,fixture_response_probe=response,fixture_pose_probe=pose,fixture_natural_flight=args.fixture_natural_flight,
        contract='Four paired eight-frame physical responses with actual delayed native controls, no world/clock rewind or model fit.' if response else
            'Repeated eight-frame native prefixes after in-process physical capture/restore; no world/clock rewind or causal action-only claim.' if prefix else
            'Atomic physical copy/perturb/restore at a frozen observation. No physical step/world restore/paired-action causal claim. Every refusal retained.'))
    env=HuntEnv(args.port,speed=3,log=lambda s:print(s,flush=True));env.obs_ms=16;env.srv.settimeout(70)
    startup=subprocess.STARTUPINFO();startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
    proc=None;checks=[];start=time.perf_counter();post_startup=0;startup_ticks=None;failure=None
    log=(args.output/'game.log').open('wb')
    try:
        proc=subprocess.Popen([str(exe),'-autotrain',args.maps[0],'-aiport',str(args.port),'-offline','-aicompetitive','0','-aiseed',str(args.seed)],
            cwd=str(exe.parent),startupinfo=startup,stdout=log,stderr=log)
        env.connect();env.conn.settimeout(15)
        # A restarted game cannot continue an owned physical fixture. Production
        # HuntEnv reconnects intentionally; this isolated check must instead fail.
        def refuse_reconnect():raise ConnectionError('Owned fixture connection lost; refusing another game')
        env._accept=refuse_reconnect
        if env.info.get('mission')!=args.maps[0]:raise ValueError('Unexpected fixture map')
        env.control('SPINWAIT 4');command(env,'DYNAMICSWAIT 600000','dynamics_wait')
        env.control('FIXEDSTEP 16');env.control('CONTACT 1')
        for _ in range(400):
            env.step(NOOP_ACTION,repeat=1)
            if time.perf_counter()-start>120:raise TimeoutError('Fixture startup wall limit')
        startup_ticks=env.ticks
        for context in (('initial','ordinary_controls','jump_or_flight','held_speed') if response else ('initial','ordinary_controls','jump_or_flight')):
            if context=='ordinary_controls':
                for _ in range(16):env.step([1,0,0,0,0,0],repeat=1);post_startup+=1
            elif context=='jump_or_flight':
                if pose:
                    p=env.msg.obs[RAW_POS].astype(float).tolist();p[2]+=4
                    requested=dict(position=p,velocity=[4,0,-2],spin=[0,60,30])
                    path=(args.output/'synthetic_pose.json').resolve()
                    values=[v for k in ('position','velocity','spin') for v in requested[k]]
                    ack=frozen_fixture_command(env,'FIXTUREPOSE '+path.as_posix()+' '+' '.join(format(v,'.17g') for v in values),ack_name='fixture_pose')
                    actual=json.loads(path.read_text(encoding='utf-8'))
                    atomic_json(args.output/'synthetic_pose_request.json',dict(requested=requested,native=actual,ack=ack.fields))
                    if ack.fields[1]!='1' or not actual['passed']:raise ValueError('Native synthetic placement refused: '+str(actual['refused']))
                elif args.fixture_natural_flight or response:
                    for _ in range(4):env.step([0,0,0,0,1,0],repeat=1);post_startup+=1
                else:
                    p=env.msg.obs[RAW_POS].astype(float).tolist();p[2]+=4
                    env.control(format_teleport(*p,4,0,-2,spin=[0,60,30]))
                    for _ in range(2):env.step(NOOP_ACTION,repeat=1);post_startup+=1
                    env.step([0,0,0,0,1,0],repeat=1);post_startup+=1
            elif context=='held_speed':
                env.control('GIVEPOW SuperSpeedItem')
                for _ in range(4):env.step(NOOP_ACTION,repeat=1);post_startup+=1
            if prefix:
                # Drain commands to neutral before capture; do not assert the
                # requested pose/spin is the actual native action-time fixture.
                for _ in range(4):env.step(NOOP_ACTION,repeat=1);post_startup+=1
                interventions={'initial':dict(action=[0,0,0,0,1,0]),'ordinary_controls':dict(action=[0,1,0,0,0,0]),
                    'jump_or_flight':dict(action=[0,0,0,0,1,0]),'held_speed':dict(use=1,pow_yaw=0)}
                check=fixture_prefix_context(env,args.output,context,interventions[context] if response else None)
                checks.append(check);post_startup+=check['decisions']
                atomic_json(args.output/'result.json',dict(checks=checks,complete=False,training=False))
                print(json.dumps({k:v for k,v in check.items() if k in ('context','passed','refused','initial_mismatch_fields','move_mismatches','maximum_motion_errors')}),flush=True)
                continue
            for _ in range(3):
                if time.perf_counter()-start>120 or post_startup>240:raise TimeoutError('Fixture declared bound')
                path=(args.output/('check_%02d.json'%len(checks))).resolve()
                # Do not call env.step while waiting: the native extension handles
                # this word without Parent::onLine/unfreezing the current step.
                ack=frozen_fixture_command(env,'FIXTURECHECK '+path.as_posix(),min(5,120-(time.perf_counter()-start)))
                if not path.exists():raise ValueError('Native fixture report missing')
                r=json.loads(path.read_text(encoding='utf-8'));r.update(context=context,ack=ack.fields)
                r['measured_equal']=r.get('before')==r.get('after') and r.get('fields_before')==r.get('fields_after')
                if r.get('passed') and not(r['measured_equal'] and r['clock_before']==r['clock_after'] and ack.fields[1]=='1'):
                    raise ValueError('Native success disagrees with independent equality check')
                checks.append(r);atomic_json(args.output/'result.json',dict(checks=checks,complete=False,training=False))
                print(json.dumps(dict(context=context,passed=r['passed'],refused=r['refused'],clock=r.get('clock_before'))),flush=True)
            # Verify normal bridge physics resumes after the atomic extension.
            _,outcome=env.step(NOOP_ACTION,repeat=1);post_startup+=1
            if outcome['round_ended'] or outcome['reconnected']:raise ValueError('Round/connection changed')
    except Exception as exc:
        failure=repr(exc)
    finally:
        env.close()
        if proc:
            proc.terminate()
            try:proc.wait(timeout=15)
            except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=10)
        log.close()
    unchanged=all(sha(p)==h for p,h in protected.items())
    sources_unchanged=all(sha(Path(p))==h for p,h in fingerprints.items()) and all(sha(Path(p))==h for p,h in native_hashes.items())
    observed_post_startup=env.ticks-startup_ticks if startup_ticks is not None else None
    if failure is None and (time.perf_counter()-start>120 or observed_post_startup>240):failure='Declared fixture bound exceeded'
    result=dict(checks=checks,complete=failure is None,passed=sum(bool(r.get('passed')) for r in checks),
        refusals=sum(bool(r.get('refused')) for r in checks),failure=failure,wall_s=time.perf_counter()-start,
        post_startup_decisions=observed_post_startup,manually_counted_reply_decisions=post_startup,
        protected_unchanged=unchanged,source_unchanged=sources_unchanged,training=False,
        caveat='Bounded physical-response probe with measured initial states/applied Moves; no full-world restoration, model dataset/fit or general causal guarantee.' if response else
            'Short native repeated physical prefixes only; no whole-world restoration, changed-action intervention or general causal-pair claim.' if prefix else
            'Copied physical fields only. Zero physics inside each check; no whole-world restore, future prefix validation or general causal-pair claim.')
    atomic_json(args.output/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k!='checks'}),flush=True)
    if failure or not unchanged or not sources_unchanged:raise RuntimeError('Fixture probe failed: '+str(result['failure']))
    return result


def fixture_pool_probe(args):
    """Finite engineering transfer check, excluding the neural holdout maps."""
    import copy,time
    names=['Sprawl_Hunt','KingOfTheMarble_Hunt','Duplex_Hunt','Cragmire_Hunt','TreasureBox_Hunt','FracturedIslands_Hunt']
    if args.maps!=names or args.seed!=981013 or args.groups!=1:raise ValueError('Use the declared six-map pool/981013/one group')
    args.output.mkdir(parents=True,exist_ok=False);protocol=args.output/'protocol.json'
    atomic_json(protocol,dict(schema='general_mechanics_six_map_transfer_preflight_v1',
        maps=[dict(map=name,seed=args.seed+i) for i,name in enumerate(names)],
        physical_pairs_per_map=4,reply_ms=16,frames_per_branch=8,maximum_post_startup_decisions_per_map=240,
        wall_limit_per_map_s=120,training=False,scored_navigation=False,holdout_labels_opened=False,
        contract='Existing matched jump/reverse/air-jump/held-Speed responses; observed physical-state and native controls audited. All refusals/failures retained. No whole-world/policy-skill claim.'))
    results=[];started=time.perf_counter()
    for i,name in enumerate(names):
        child=copy.copy(args);child.maps=[name];child.seed=args.seed+i;child.output=args.output/name
        child.fixture_pool_probe=False;child.fixture_response_probe=True;child.pool_protocol=protocol
        try:r=fixture_roundtrip_probe(child)
        except Exception as exc:
            r=json.loads((child.output/'result.json').read_text(encoding='utf-8')) if (child.output/'result.json').exists() else {}
            r.update(failure=repr(exc),complete=False)
        results.append(dict(map=name,seed=child.seed,**r));atomic_json(args.output/'result.json',dict(maps=results,
            **fixture_pool_completion(results,names,args.seed),training=False,scored_navigation=False,wall_s=time.perf_counter()-started))
        print(json.dumps(dict(map=name,complete=r['complete'],passed=r.get('passed'),refusals=r.get('refusals'),failure=r.get('failure'))),flush=True)


def fixture_pool_completion(results,names,seed):
    """Attempted maps alone do not establish a complete physical contract."""
    from collections import Counter
    expected=Counter((name,seed+i) for i,name in enumerate(names))
    identities=Counter((r.get('map'),r.get('seed')) for r in results)
    attempted=identities==expected
    valid=all(r.get('complete') and not r.get('failure') and r.get('passed')==4 and r.get('refusals')==0 and
        r.get('protected_unchanged') and r.get('source_unchanged') and len(r.get('checks',[]))==4 and
        all(c.get('passed') for c in r['checks']) for r in results)
    return dict(all_declared_maps_attempted=attempted,complete=attempted and valid,
        passed_physical_pairs=sum(r.get('passed',0) for r in results))


def collect_native_physical_pairs(args):
    """Small versioned ordinary coverage; actual Native Moves are the labels."""
    import subprocess,time
    from nav.env import HuntEnv
    from nav.protocol import NOOP_ACTION
    from nav.marble_dynamics.record import command
    names=['Sprawl_Hunt','KingOfTheMarble_Hunt','Duplex_Hunt','Cragmire_Hunt','TreasureBox_Hunt','FracturedIslands_Hunt']
    if args.maps!=names or args.seed!=1081013 or args.groups!=1:raise ValueError('Use declared six-map/1081013/one-group collection')
    args.output.mkdir(parents=True,exist_ok=False);root=ML.parent;exe=root/'Marble Blast Platinum/marbleblast_fixture_probe.exe'
    protected_paths=[root/'Marble Blast Platinum'/x for x in ('marbleblast_geometry_probe.exe','marbleblast_mbx.exe')]
    protected_paths += [root/'Marble Blast Platinum/platinum/client/scripts/ai/mlAgent.cs',ML/'models/nav/real_maps_20261010_A2/nav_latest.pth']
    protected={str(p):sha(p) for p in protected_paths}
    sources=list(Path(__file__).parent.glob('*.py'))+[Path(__file__).with_name(x) for x in ('engine_dynamics.inc','engine_fixture.inc','build_probe.ps1')]
    sources += [ML/'nav/collision/build_probe.ps1',ML/'nav/collision/mesh.py',root/'Marble Blast Platinum/platinum/client/scripts/ai/mlAgent.cs',
        root/'Marble Blast Platinum/platinum/client/scripts/ai/socketBridge.cs']
    fingerprints={str(p):sha(p) for p in sources};snapshot_sources(args.output,sources)
    native=Path('C:/Users/doug/src/OpenPQ-TGEMIT-mbx');native_files=[native/'engine/game/marble'/x for x in
        ('marble.cc','marble.h','aiGeometryExport.inc','aiMarbleDynamics.inc','aiMarbleFixture.inc')]
    write_archive(args.output/'native_source_snapshot.zip',native_files,native);native_hashes={str(p):sha(p) for p in native_files}
    meshes={name:CollisionMap(ML/'collision_maps'/(name+'.npz')) for name in names}
    geometry_hashes={str(ML/'collision_maps'/(name+suffix)):sha(ML/'collision_maps'/(name+suffix)) for name in names for suffix in ('.npz','.json')}
    strata=('roll','brake','jump','flight_up8','flight_up16','flight_down8','flight_down20',
        'wall_impact','edge_timing','ceiling_impact','spin_low','spin_high')
    requests={}
    for mi,name in enumerate(names):
        sampler=Sampler(meshes[name],args.seed+mi);requests[name]=[]
        for index,label in enumerate(strata):
            family='drop_spin' if label.startswith(('flight_','spin_')) else label
            spec=sampler.sample(family,index)
            if spec:
                if label.startswith('flight_'):spec['velocity'][2]=float(label.rsplit('up',1)[1]) if '_up' in label else -float(label.rsplit('down',1)[1])
                if label=='spin_low':spec['spin']=[0.,0.,0.]
                if label=='spin_high':spec['spin']=sampler.rng.uniform(-180,180,3).tolist()
                if len(meshes[name].sphere_contacts(spec['position'],spec['radius']+.002)['triangle_ids']):spec=None
            reply=dict(action=[0,1,0,0,0,0] if label=='brake' else [1,0,0,0,0,0] if label=='roll' else [0,0,0,0,1,0])
            requests[name].append(dict(stratum=label,spec=spec,intervention=reply))
    atomic_json(args.output/'prepared.json',dict(schema='native_inprocess_physical_pairs_v1',maps=names,seeds=[args.seed+i for i in range(6)],
        requests=requests,requested_pairs=72,reply_ms=16,frames_per_branch=16,maximum_recorded_transitions=2304,
        maximum_post_startup_observations_per_map=1600,wall_limit_per_map_s=120,source_hashes=fingerprints,
        native_source_hashes=native_hashes,geometry_hashes=geometry_hashes,protected_sha256=protected,
        executable_sha256=sha(exe),training=False,scored_navigation=False,holdout_labels_opened=False,
        contract='Synthetic setup/native contact establishment; same-process paired physical starts, actual delayed Move/camera labels. Preserve all refusals/falls/mismatches. No world rewind, power use or model fit.'))
    summaries=[];batch_start=time.perf_counter()
    for mi,name in enumerate(names):
        folder=args.output/name;folder.mkdir();pairs=[];failure=None;start=time.perf_counter();startup_ticks=None
        env=HuntEnv(args.port,speed=3,log=lambda s:print(s,flush=True));env.obs_ms=16;env.srv.settimeout(70)
        startup=subprocess.STARTUPINFO();startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
        proc=None;log=(folder/'game.log').open('wb')
        manifest=dict(schema='marble_native_physical_pairs_v1',map=name,split='train',seed=args.seed+mi,
            geometry_sha256=meshes[name].info['mesh_sha256'],trials=[],pairs=pairs,collection_complete=False)
        try:
            proc=subprocess.Popen([str(exe),'-autotrain',name,'-aiport',str(args.port),'-offline','-aicompetitive','0','-aiseed',str(args.seed+mi)],
                cwd=str(exe.parent),startupinfo=startup,stdout=log,stderr=log)
            env.connect();env.conn.settimeout(15)
            def refuse_reconnect():raise ConnectionError('Owned collection disconnected; refusing another game')
            env._accept=refuse_reconnect
            if env.info.get('mission')!=name:raise ValueError('Unexpected collection map')
            env.control('SPINWAIT 4');command(env,'DYNAMICSWAIT 600000','dynamics_wait');env.control('FIXEDSTEP 16');env.control('CONTACT 1')
            for _ in range(400):
                env.step(NOOP_ACTION,repeat=1)
                if time.perf_counter()-start>120:raise TimeoutError('Collection startup bound')
            startup_ticks=env.ticks
            for index,request in enumerate(requests[name]):
                if time.perf_counter()-start>120 or env.ticks-startup_ticks>1600:raise TimeoutError('Declared collection bound')
                key='%02d_%s'%(index,request['stratum']);spec=request['spec'];record=dict(id=index,**request)
                if spec is None:record.update(passed=False,refused='no_clear_sampled_start',branches=[])
                else:
                    env.control('RESPAWN')
                    for _ in range(32):env.step(NOOP_ACTION,repeat=1)
                    pose=(folder/(key+'.pose.json')).resolve();values=[v for k in ('position','velocity','spin') for v in spec[k]]
                    ack=frozen_fixture_command(env,'FIXTUREPOSE '+pose.as_posix()+' '+' '.join(format(v,'.17g') for v in values),
                        timeout_s=min(5,120-(time.perf_counter()-start)),ack_name='fixture_pose')
                    placed=json.loads(pose.read_text(encoding='utf-8'));record['native_placement']=placed
                    if ack.fields[1]!='1' or not placed['passed']:record.update(passed=False,refused=placed.get('refused') or 'placement_refused',branches=[])
                    else:
                        for _ in range(4):env.step(NOOP_ACTION,repeat=1)
                        record.update(fixture_prefix_context(env,folder,key,request['intervention'],replies=16))
                        for branch in record['branches']:
                            rows=load_rows(folder/branch['path']);valid=bool(rows) and all(r['before']['mode']==0 and not r['before']['oob'] for r in rows)
                            manifest['trials'].append(dict(id=len(manifest['trials']),file=branch['path'],valid=valid,
                                pair_group=index,stratum=request['stratum'],samples=len(rows),raw_sha256=branch['native_sha256'],
                                paired_start_verified=record['passed'],actual_initial_state=branch['first_before'],
                                fell=any(v['fell'] for v in branch['outcomes']),
                                native_contact_steps=sum(r['telemetry']['contact_steps'] for r in rows),
                                native_collisions=sum(r['telemetry']['collisions'] for r in rows)))
                pairs.append(record);atomic_json(folder/'manifest.json',manifest)
                print(json.dumps(dict(map=name,stratum=request['stratum'],paired_start_verified=record['passed'],refused=record.get('refused'),branches=len(record['branches']))),flush=True)
        except Exception as exc:failure=repr(exc)
        finally:
            env.close()
            if proc:
                proc.terminate()
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=10)
            log.close()
        observed=env.ticks-startup_ticks if startup_ticks is not None else None
        if failure is None and (time.perf_counter()-start>120 or observed>1600):failure='Declared collection bound exceeded'
        manifest.update(collection_complete=failure is None and len(pairs)==12,failure=failure,
            wall_s=time.perf_counter()-start,post_startup_observations=observed);atomic_json(folder/'manifest.json',manifest)
        summaries.append(dict(map=name,complete=manifest['collection_complete'],requests=len(pairs),paired_starts_verified=sum(bool(p.get('passed')) for p in pairs),
            refusals=sum(bool(p.get('refused')) for p in pairs),trials=len(manifest['trials']),
            native_transitions=sum(t['samples'] for t in manifest['trials']),failure=failure,
            wall_s=manifest['wall_s'],post_startup_observations=observed))
        source_unchanged=all(sha(Path(p))==h for p,h in {**fingerprints,**native_hashes,**geometry_hashes}.items())
        protected_unchanged=all(sha(Path(p))==h for p,h in protected.items())
        atomic_json(args.output/'collection.json',dict(maps=summaries,all_maps_attempted=len(summaries)==len(names),
            complete=len(summaries)==len(names) and all(s['complete'] for s in summaries),source_unchanged=source_unchanged,
            protected_unchanged=protected_unchanged,training=False,scored_navigation=False,wall_s=time.perf_counter()-batch_start))
        if not source_unchanged or not protected_unchanged:raise RuntimeError('Collection source/protected artifacts changed')


def prepare_physical_reader_index(source,output):
    """Versioned metadata adapter; sealed source manifests/readers stay intact."""
    import os
    prepared=json.loads((source/'prepared.json').read_text(encoding='utf-8'))
    collection=json.loads((source/'collection.json').read_text(encoding='utf-8'))
    if prepared['schema']!='native_inprocess_physical_pairs_v1' or not collection['complete'] or not collection['source_unchanged'] or not collection['protected_unchanged']:
        raise ValueError('Requires complete source/integrity-audited native collection')
    output.mkdir(parents=True,exist_ok=False);indices=[]
    for name in prepared['maps']:
        origin=source/name/'manifest.json';manifest=json.loads(origin.read_text(encoding='utf-8'));folder=output/name;folder.mkdir()
        trials=[]
        for trial in manifest['trials']:
            original=source/name/trial['file'];rows=load_rows(original)
            import gzip,hashlib
            if hashlib.sha256(gzip.decompress(original.read_bytes())).hexdigest()!=trial['raw_sha256']:raise ValueError('Source archive mismatch')
            item=dict(trial);item.update(file=Path(os.path.relpath(original,folder)).as_posix(),family=trial['stratum'])
            # Individual motion validity is separate from full paired return.
            # The sealed reader excludes nonplayable/body-transition rows itself.
            item['valid']=bool(rows);trials.append(item)
        indexed=dict(schema='marble_trajectories_v1',map=name,split=manifest['split'],seed=manifest['seed'],
            geometry_sha256=manifest['geometry_sha256'],trials=trials,replay_checks=[],collection_complete=True,
            source_manifest=str(origin.resolve()),source_manifest_sha256=sha(origin),
            contract='Metadata-only index of unchanged measured physical archives. Pair failure does not invalidate every native one-step label; reader filters each row. No raw record duplication or source-manifest edits.')
        atomic_json(folder/'manifest.json',indexed);indices.append(dict(map=name,trials=len(trials),source_manifest_sha256=sha(origin)))
    atomic_json(output/'prepared.json',dict(source=str(source.resolve()),source_prepared_sha256=sha(source/'prepared.json'),
        source_collection_sha256=sha(source/'collection.json'),indices=indices,complete=True,metadata_only=True,
        adapter_source_sha256=sha(Path(__file__)),training=False,raw_archives_copied=0))


def script_context_probe(args):
    """Observe native and script transitions; this never restores a game world."""
    import subprocess,time,gzip
    from nav.env import HuntEnv
    from nav.protocol import NOOP_ACTION,RAW_POS
    from nav.marble_dynamics.record import command
    coverage=getattr(args,'fixture_state_coverage_probe',False)
    seed=1381013 if coverage else 1181013;use_steps=(1,2,3,5) if coverage else (1,)
    if args.maps!=['CubeIsle_Hunt'] or args.seed!=seed or args.groups!=1:
        raise ValueError('Script-state preflight requires declared Cube/seed/one six-case batch')
    args.output.mkdir(parents=True,exist_ok=False);root=ML.parent;exe=root/'Marble Blast Platinum/marbleblast_fixture_probe.exe'
    paths=[root/'Marble Blast Platinum'/n for n in ('marbleblast_geometry_probe.exe','marbleblast_mbx.exe')]
    paths += [root/'Marble Blast Platinum/platinum/client/scripts/ai/mlAgent.cs',ML/'models/nav/real_maps_20261010_A2/nav_latest.pth']
    protected={str(p):sha(p) for p in paths}
    sources=list(Path(__file__).parent.glob('*.py'))+[Path(__file__).with_name(n) for n in ('engine_dynamics.inc','engine_fixture.inc','build_probe.ps1')]
    sources += [root/'Marble Blast Platinum/platinum'/p for p in ('server/scripts/marble.cs','server/scripts/powerups.cs','client/scripts/mp/marble.cs','client/scripts/ai/mlAgent.cs')]
    fingerprints={str(p):sha(p) for p in sources};snapshot_sources(args.output,sources)
    native=Path('C:/Users/doug/src/OpenPQ-TGEMIT-mbx');native_files=[native/'engine/game/marble'/n for n in ('marble.cc','marble.h','aiGeometryExport.inc','aiMarbleDynamics.inc','aiMarbleFixture.inc')]
    native_hashes={str(p):sha(p) for p in native_files};write_archive(args.output/'native_source_snapshot.zip',native_files,native)
    protocol=dict(map='CubeIsle_Hunt',powers=list(POWERUPS),seeds=list(range(seed,seed+6)),use_steps=list(use_steps),
        frames_per_case=64,maximum_post_startup_observations=240,wall_limit_s=120,
        source_hashes=fingerprints,native_source_hashes=native_hashes,protected_sha256=protected,
        executable_sha256=sha(exe),training=False,world_restored=False,
        extended_state_coverage=coverage,
        contract='Fresh owned process/type; setup-only grant/descending spinning pose; declared use/release sequence. Native/script observability and boundary audit, not paired causality or skill.')
    atomic_json(args.output/'prepared.json',protocol);results=[];batch_start=time.perf_counter()
    for index,power in enumerate(POWERUPS):
        if (args.output/'STOP').exists():break
        folder=args.output/power;folder.mkdir();start=time.perf_counter();startup_ticks=None;proc=None
        env=HuntEnv(args.port,speed=3,log=lambda s:print(s,flush=True));env.obs_ms=16;env.srv.settimeout(70)
        startup=subprocess.STARTUPINFO();startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
        log=(folder/'game.log').open('wb');result=dict(power=power,seed=args.seed+index,passed=False,observations=[])
        try:
            proc=subprocess.Popen([str(exe),'-autotrain','CubeIsle_Hunt','-aiport',str(args.port),'-offline','-aicompetitive','0','-aiseed',str(args.seed+index)],
                cwd=str(exe.parent),startupinfo=startup,stdout=log,stderr=log)
            env.connect();env.conn.settimeout(15)
            def refuse_reconnect():raise ConnectionError('Script-state preflight disconnected; no replacement game')
            env._accept=refuse_reconnect
            if env.info.get('mission')!='CubeIsle_Hunt':raise ValueError('Unexpected mission')
            env.control('SPINWAIT 4');command(env,'DYNAMICSWAIT 600000','dynamics_wait');env.control('FIXEDSTEP 16')
            command(env,'DYNAMICSLIVE 1','dynamics_live');env.control('CONTACT 1')
            for _ in range(400):
                env.step(NOOP_ACTION,repeat=1)
                if time.perf_counter()-start>120:raise TimeoutError('Script preflight startup bound')
            startup_ticks=env.ticks;env.control('GIVEPOW '+power)
            for _ in range(4):env.step(NOOP_ACTION,repeat=1)
            position=env.msg.obs[RAW_POS].astype(float).tolist();position[2]+=3
            request=dict(position=position,velocity=[3.,0.,-8.],spin=[0.,40.,25.])
            values=[v for k in ('position','velocity','spin') for v in request[k]];pose=(folder/'pose.json').resolve()
            ack=frozen_fixture_command(env,'FIXTUREPOSE '+pose.as_posix()+' '+' '.join(format(v,'.17g') for v in values),ack_name='fixture_pose')
            actual=json.loads(pose.read_text());result['requested_pose']=request;result['native_placement']=actual
            if ack.fields[1]!='1' or not actual['passed']:raise ValueError('Native setup refused: '+str(actual.get('refused')))
            for _ in range(4):env.step(NOOP_ACTION,repeat=1)
            raw=(folder/'native.jsonl').resolve();ack=frozen_fixture_command(env,'FIXTURERECORD '+raw.as_posix(),ack_name='fixture_record')
            if ack.fields[1]!='1':raise ValueError('Record start refused')
            for step in range(64):
                if (args.output/'STOP').exists() or time.perf_counter()-start>120 or env.ticks-startup_ticks>=240:
                    raise TimeoutError('Script preflight stopped or bounded')
                _,outcome=env.step(NOOP_ACTION,repeat=1,use_pow=int(step in use_steps),pow_yaw=0 if step in use_steps else None)
                result['observations'].append(dict(step=step,outcome=outcome,dynamics=env.msg.dynamics))
                if any(outcome[k] for k in ('fell','round_ended','reconnected')):break
            ack=frozen_fixture_command(env,'FIXTURERECORD off',ack_name='fixture_record')
            if ack.fields[1]!='1':raise ValueError('Record close refused')
            packed=raw.with_suffix('.jsonl.gz');archive(raw,packed)
            if gzip.decompress(packed.read_bytes())!=raw.read_bytes():raise ValueError('Native archive differs')
            result.update(native_sha256=sha(raw),compressed_sha256=sha(packed),raw_duplicate_removed=True);raw.unlink()
            rows=load_rows(packed);result['native_frames']=len(rows)
            if len(rows)!=len(result['observations']):raise ValueError('Observation/native transition alignment differs')
            first=rows[0]['before'];context=first['script_context'];result['first_native_before']=first
            if context['target']['held_item']!=power or not context['authority'] or context['authority']['held_item']!=power:
                raise ValueError('Script inventory export disagrees with granted item')
            if context['target']['held_item_power_id']!=index+1 or context['authority']['held_item_power_id']!=index+1:
                raise ValueError('Script item type export differs')
            if first['held']!=(index+1 if index<5 else 0):raise ValueError('Native held ID export differs')
            result['native_use_frames']=[i for i,r in enumerate(rows) if r['move']['use']]
            result['native_consumptions']=[i for i,r in enumerate(rows) if r['move']['use'] and r['before']['held'] and not r['after']['held']]
            result['body_boundaries']=[dict(frame=i,within_step=r['before']['radius']!=r['after']['radius'],
                before_radius=r['before']['radius'],after_radius=r['after']['radius']) for i,r in enumerate(rows) if
                r['before']['radius']!=r['after']['radius'] or i and rows[i-1]['after']['radius']!=r['before']['radius']]
            result['ever_native_effect_active']=[any(r[s]['effects'][i][0] for r in rows for s in ('before','after')) for i in range(6)]
            result['ever_authority_mega_scheduled']=any(r[s]['script_context']['authority']['mega_schedule']['pending'] for r in rows for s in ('before','after'))
            result['script_inventory_consumed']=any(r[s]['script_context']['authority']['held_item']!=power for r in rows for s in ('before','after'))
            result['passed']=bool(result['native_use_frames']) and result['script_inventory_consumed']
            if coverage:
                for row in rows:
                    for phase in ('before','after'):
                        state=row[phase];native=state['native_context'];bridge=state['script_context']['bridge_control']
                        if native['schema']!='observed_native_context_v1' or bridge['schema']!='observed_bridge_control_v1':
                            raise ValueError('Extended observed context schema missing')
                        if native['runtime_mass_observed'] or native['last_contact_decoded']:
                            raise ValueError('Unverified hidden fields must remain explicitly unavailable')
                result['extended_context_present']=True
        except Exception as error:result['failure']=repr(error)
        finally:
            result['post_startup_observations']=env.ticks-startup_ticks if startup_ticks is not None else None
            result['wall_s']=time.perf_counter()-start
            if proc is not None and proc.poll() is None:proc.terminate();proc.wait(timeout=10)
            env.close();log.close();atomic_json(folder/'result.json',result)
        results.append({k:v for k,v in result.items() if k not in ('observations','first_native_before')})
        atomic_json(args.output/'result.json',dict(cases=results,complete=False,training=False))
        print(json.dumps({k:v for k,v in results[-1].items() if k in ('power','seed','passed','native_frames','native_use_frames',
            'native_consumptions','body_boundaries','ever_authority_mega_scheduled','failure','post_startup_observations','wall_s')}),flush=True)
    intact=all(sha(p)==v for p,v in fingerprints.items()) and all(sha(p)==v for p,v in native_hashes.items())
    protected_intact=all(sha(p)==v for p,v in protected.items()) and sha(exe)==protocol['executable_sha256']
    complete=len(results)==6 and all(r['passed'] and not r.get('failure') and r['seed']==seed+i and r['power']==POWERUPS[i] and
        r['post_startup_observations']<=240 and r['wall_s']<=120 for i,r in enumerate(results)) and intact and protected_intact
    atomic_json(args.output/'result.json',dict(cases=results,complete=complete,all_declared_cases_attempted=len(results)==6,
        source_hashes_unchanged=intact,protected_artifacts_unchanged=protected_intact,wall_s=time.perf_counter()-batch_start,
        training=False,paired_causality=False,world_restored=False))
    if not complete:raise RuntimeError('Script observability preflight incomplete/failed; preserve every case')


def observed_world_example(current,future,new_reply):
    """Raw reviewed task prototype, not finalized normalized neural features."""
    a=current['native'];b=future['native']
    if b['frame_clock_ms']-a['frame_clock_ms']!=16:raise ValueError('Expected adjacent16ms live observations')
    for index,observation in enumerate((current,future)):
        state=observation['native']['state']
        if index==0 and (state['mode']!=0 or state['oob']):raise ValueError('Nonplayable current observed world state')
        if 'script_context' not in state or not state['script_context']['authority']:
            raise ValueError('Missing observed script/local authority context')
        if len(observation['bridge_action'])!=9:raise ValueError('Missing queued bridge reply')
    if len(new_reply)!=9 or not np.all(np.isfinite(new_reply)):raise ValueError('Known new reply must have nine finite controls')
    before,after=a['state'],b['state']
    changed=[k for k in ('radius','datablock','parameters','held','effects') if before[k]!=after[k]]
    return dict(input=dict(current_observed_dynamics=copy.deepcopy(current),new_reply=list(new_reply),dt_ms=16),
        target=dict(next_observed_dynamics=copy.deepcopy(future)),audit_only=dict(changed_state_fields=changed,
            body_transition=any(k in changed for k in ('radius','datablock','parameters')),
            next_oob=bool(after['oob']),next_mode=int(after['mode'])))


def prepare_observed_world_transitions(source,output):
    """Retain every declared adjacent observation; never derive input from labels."""
    import gzip
    from nav.source_archive import verify_archive
    protocol=json.loads((source/'prepared.json').read_text());result=json.loads((source/'result.json').read_text())
    if not result['complete'] or protocol['map']!='CubeIsle_Hunt' or protocol['powers']!=list(POWERUPS):
        raise ValueError('Requires the completed declared six-power Cube observability preflight')
    if not json.loads((source/'independent_audit.json').read_text())['all_six_contracts_pass']:
        raise ValueError('Independent source audit missing')
    output.mkdir(parents=True,exist_ok=False);sources={str(source/f):sha(source/f) for f in
        ('prepared.json','result.json','independent_audit.json','source_snapshot.zip','native_source_snapshot.zip')}
    for f in ('source_snapshot.zip','native_source_snapshot.zip'):verify_archive(source/f)
    snapshot_sources(output,[Path(__file__),ML/'nav/source_archive.py'])
    examples=[];counts=[];boundary=[]
    for index,power in enumerate(POWERUPS):
        folder=source/power;path=folder/'result.json';sources[str(path)]=sha(path)
        recorded=json.loads(path.read_text());observations=recorded['observations'];native=folder/'native.jsonl.gz';sources[str(native)]=sha(native)
        if recorded['seed']!=protocol['seeds'][index] or len(observations)!=64 or sha(native)!=recorded['compressed_sha256']:
            raise ValueError('Declared source identity/count/hash differs')
        # The source protocol predeclares step1 as the only use request. This is
        # a known command, verified against metadata, never inferred from motion.
        commands=[[0.,0.,0.,0.,0.,0.,int(i==1),0.,0.] for i in range(64)]
        for i,obs in enumerate(observations):
            if obs['step']!=i or obs['dynamics']['bridge_action']!=commands[i]:raise ValueError('Declared reply/observation alignment differs')
        for i in range(63):
            example=observed_world_example(observations[i]['dynamics'],observations[i+1]['dynamics'],commands[i+1])
            example['provenance']=dict(power=power,source=str(path),source_step=i,seed=recorded['seed'])
            examples.append(example)
            if example['audit_only']['body_transition']:boundary.append(dict(power=power,source_step=i,
                changed_fields=example['audit_only']['changed_state_fields']))
        counts.append(dict(power=power,examples=63))
    path=output/'observed_transitions.jsonl.gz';path.write_bytes(gzip.compress(('\n'.join(json.dumps(x,separators=(',',':')) for x in examples)+'\n').encode(),mtime=0))
    reread=[json.loads(v) for v in gzip.decompress(path.read_bytes()).decode().splitlines()]
    if reread!=examples:raise ValueError('Compressed observed-state task differs')
    geometry=ML/'collision_maps/CubeIsle_Hunt.npz'
    atomic_json(output/'manifest.json',dict(schema='observed_world_transition_prototype_v1',role='dev_engineering',map='CubeIsle_Hunt',
        examples=len(examples),case_counts=counts,body_boundaries=boundary,sources=sources,data_file=path.name,data_sha256=sha(path),
        geometry_file=str(geometry),geometry_sha256=sha(geometry),preparation_complete=True,training=False,
        complete_hidden_state=False,normalized_neural_features=False,identity_fields_are_provenance=True,
        contract='CURRENT synchronized observation+known new reply are inputs; queued reply remains separate. NEXT observation is target only. Fixed16ms/owned local context; full source geometry retained. No fit, promotion or whole-world restore.'))
    print(json.dumps(dict(output=str(output),examples=len(examples),body_boundaries=boundary,training=False)),flush=True)


def audit_paired_analytical_baseline(output,dll=None,open_loop=False):
    """Static-scene physical BEFORE/Move predictions; future motion is scored only."""
    import time
    from nav.marble_dynamics.physics import GeneralPhysics
    from nav.marble_dynamics.student_data import geometry,state_features,motion
    from nav.collision.experimental_sweep import CompiledSweepMap
    from nav.source_archive import verify_archive
    from nav.marble_dynamics.rollout import predict
    source=ML/'datasets/marble_dynamics/native_action_difference_20261013'
    manifest=json.loads((source/'manifest.json').read_text());reference=json.loads((source/'audit.json').read_text())
    verify_archive(source/'source_snapshot.zip')
    if manifest['schema']!='native_action_difference_v1' or len(reference['pairs'])!=90:raise ValueError('Unexpected declared pair source')
    output.mkdir(parents=True,exist_ok=False)
    sources=list(Path(__file__).parent.glob('*.py'))+[ML/'nav/collision'/name for name in ('mesh.py','experimental_sweep.py')]
    fingerprints={str(p):sha(p) for p in sources};snapshot_sources(output,sources)
    inputs={str(source/'manifest.json'):sha(source/'manifest.json'),str(source/'audit.json'):sha(source/'audit.json')}
    for key,value in manifest['sources'].items():
        if sha(Path(key))!=value:raise ValueError('Pair source hash changed')
        inputs[key]=value
    for key,value in manifest['output_hashes'].items():
        if sha(source/key)!=value:raise ValueError('Pair tensor hash changed')
        inputs[str(source/key)]=value
    protected=[ML/'models/nav/real_maps_20261010_A2/nav_latest.pth',ML.parent/'Marble Blast Platinum/platinum/client/scripts/ai/mlAgent.cs']
    protected += [ML.parent/'Marble Blast Platinum'/n for n in ('marbleblast_geometry_probe.exe','marbleblast_mbx.exe')]
    protected_hashes={str(p):sha(p) for p in protected};geometry_hashes={}
    atomic_json(output/'prepared.json',dict(pair_manifest=str(source/'manifest.json'),pair_manifest_sha256=sha(source/'manifest.json'),
        maximum_branch_evaluations=180,maximum_forecast_steps=2880 if open_loop else 180,maximum_wall_s=300,
        frames_per_forecast=[8,16] if open_loop else [1],open_loop=open_loop,role='train_diagnostic',training=False,test_motion_labels_opened=False,
        source_hashes=fingerprints,input_hashes=inputs,protected_sha256=protected_hashes,
        compiled_sweep_dll=str(dll) if dll else None,compiled_sweep_sha256=sha(dll) if dll else None,
        contract='Exact recorded physical BEFORE/actual native Move+camera events/full mesh. Native AFTER scores only; no future contacts/state supplied.'))
    start=time.perf_counter();records=[];cached={};engines={};failure=None
    try:
        for item in reference['pairs']:
            if time.perf_counter()-start>300:raise TimeoutError('Declared analytical audit wall bound')
            name=item['map'];path=Path(item['manifest'])
            if name in ('Bowl_Hunt','VortexEffect_Hunt','CubeIsle_Hunt','Concentric_Hunt'):raise ValueError('Only original six training maps allowed')
            if name not in engines:
                mesh_path=ML/'collision_maps'/(name+'.npz');mesh=CompiledSweepMap(mesh_path,dll) if dll else CollisionMap(mesh_path)
                engines[name]=(GeneralPhysics(mesh),geometry(mesh));geometry_hashes[str(mesh_path)]=sha(mesh_path)
            engine,geo=engines[name];data=cached.setdefault(str(path),json.loads(path.read_text()));pairs=data.get('pairs',data.get('checks',[]))
            pair=next(p for p in pairs if p.get('stratum',p.get('context'))==item['context'])
            paired_motion_example(pair,path.parent,geo)
            forecasts=[];drifts=[];targets=[];branches=[]
            for branch in pair['branches']:
                native_rows=load_rows(path.parent/branch['path']);row=native_rows[item['native_frame']]
                _,initial=state_features(row['before'],dict(move=row['move'],events=row['control_events']),16,geo['center'],geo['scale'])
                prefix={}
                if open_loop:
                    frames=len(native_rows)
                    if frames not in (8,16) or any(r['dt_ms']!=16 for r in native_rows):raise ValueError('Expected complete recorded8/16-frame native prefix')
                    for i,r in enumerate(native_rows):
                        if any(r['before'][k]!=r['after'][k] for k in ('radius','datablock','parameters')):raise ValueError('Within-step external body event')
                        if i and any(native_rows[i-1]['after'][k]!=r['before'][k] for k in ('position','velocity','spin','radius','datablock','parameters')):
                            raise ValueError('External between-frame physical event')
                    controls=[dict(move=r['move'],events=r['control_events']) for r in native_rows]
                    rollout=predict(engine,native_rows[0]['before'],controls,dt_ms=16,max_steps=frames)
                    if rollout['failure'] or len(rollout['states'])!=frames+1:raise ValueError('Open-loop forecast failed: '+str(rollout['failure']))
                    measured=np.asarray([motion(r['after']) for r in native_rows]);predicted=np.asarray([motion(s) for s in rollout['states'][1:]])
                    initial_motion=motion(native_rows[0]['before']);cv=np.repeat(initial_motion[None,:],frames,axis=0)
                    cv[:,:3]+=np.arange(1,frames+1)[:,None]*.016*initial_motion[3:6]
                    errors=lambda values:np.column_stack([np.linalg.norm((values-measured)[:,i:i+3],axis=1) for i in (0,3,6)])
                    prefix=dict(frames=frames,predicted_motion=predicted.tolist(),actual_motion=measured.tolist(),
                        analytical_error=errors(predicted).tolist(),velocity_drift_error=errors(cv).tolist(),
                        maximum_analytical_error=np.max(errors(predicted),axis=0).tolist(),controls=controls)
                    forecast=predicted[-1];actual=measured[-1];drift=cv[-1];events=rollout['events']
                else:
                    actual=motion(row['after']);drift=motion(initial);drift[:3]+=drift[3:6]*.016
                    predicted,events=engine.step(initial,row['move'],16);forecast=motion(predicted)
                if not np.all(np.isfinite(forecast)):raise ValueError('Nonfinite analytical forecast')
                error=lambda value:[float(np.linalg.norm((value-actual)[i:i+3])) for i in (0,3,6)]
                forecasts.append(forecast);drifts.append(drift);targets.append(actual)
                branches.append(dict(actual_move=row['move'],analytical_error=error(forecast),velocity_drift_error=error(drift),
                    predicted_motion=forecast.tolist(),actual_motion=actual.tolist(),support_steps_audit_only=row['telemetry']['support_steps'],events=events,
                    open_loop_prefix=prefix))
            effect=targets[1]-targets[0]
            effect_error=lambda values:[float(np.linalg.norm(((values[1]-values[0])-effect)[i:i+3])) for i in (0,3,6)]
            records.append(dict(map=name,context=item['context'],manifest=str(path),native_frame=item['native_frame'],branches=branches,
                analytical_action_effect_error=effect_error(forecasts),velocity_drift_action_effect_error=effect_error(drifts),actual_action_effect=effect.tolist()))
            atomic_json(output/'result.json',dict(complete=False,pairs=records,training=False))
    except Exception as error:failure=repr(error)
    intact=all(sha(Path(k))==v for k,v in {**fingerprints,**inputs,**protected_hashes,**geometry_hashes}.items())
    if dll:intact=intact and sha(dll)==json.loads((output/'prepared.json').read_text())['compiled_sweep_sha256']
    summary=[]
    for name in engines:
        selected=[p for p in records if p['map']==name]
        if not selected:continue
        b=[branch for p in selected for branch in p['branches']]
        summary.append(dict(map=name,pairs=len(selected),
            prefix_pairs_by_frames={str(n):sum(p['branches'][0]['open_loop_prefix'].get('frames')==n for p in selected) for n in (8,16)} if open_loop else {},
            mean_analytical_error=np.mean([x['analytical_error'] for x in b],axis=0).tolist(),
            mean_velocity_drift_error=np.mean([x['velocity_drift_error'] for x in b],axis=0).tolist(),
            mean_analytical_action_effect_error=np.mean([x['analytical_action_effect_error'] for x in selected],axis=0).tolist(),
            mean_velocity_drift_action_effect_error=np.mean([x['velocity_drift_action_effect_error'] for x in selected],axis=0).tolist()))
    result=dict(complete=len(records)==90 and failure is None and intact,pairs=records,per_map=summary,failure=failure,
        source_input_geometry_protected_unchanged=intact,geometry_hashes=geometry_hashes,wall_s=time.perf_counter()-start,
        error_order=['position_u','velocity_u_per_s','spin_rad_per_s'],forecast_horizon_ms=[128,256] if open_loop else [16],
        error_scope='final forecast state',training=False,test_motion_labels_opened=False)
    atomic_json(output/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('pairs','geometry_hashes')}),flush=True)
    if not result['complete']:raise RuntimeError('Analytical audit incomplete/failed; evidence retained')


def observable_state_tokens(observation,center,scale):
    """Semantic LIVE state only; IDs/absolute clocks are provenance, not features.

    Variable contact tokens retain every observed contact. This is an engineering
    interface, not a claim that the observed state closes all world dependencies.
    """
    from nav.marble_dynamics.student_data import PARAMETERS,PARAMETER_SCALES
    if not np.isfinite(scale) or scale<=0:raise ValueError('Invalid map normalization')
    def vector(value,length,divisor=1.):
        a=np.asarray(value,dtype=float)
        if a.shape!=(length,) or not np.all(np.isfinite(a)):raise ValueError('Missing/nonfinite observed vector')
        return (a/divisor).tolist()
    def one_hot(value,count):
        if isinstance(value,bool) or int(value)!=value or not 0<=int(value)<count:raise ValueError('Unknown observed inventory type')
        return [float(i==int(value)) for i in range(count)]
    def schedule(value):return [float(value['pending']),float(value['remaining_ms'])/10000. if value['pending'] else 0.]
    def move(value):return vector([value[k] for k in ('x','y','z','yaw','pitch','jump','use','trigger1')],8)
    native=observation['native'];state=native['state'];script=state.get('script_context');context=state.get('native_context')
    if not script or not context or not script.get('bridge_control') or not script.get('target') or not script.get('authority'):
        raise ValueError('Missing current observed control/native context')
    if context['schema']!='observed_native_context_v1' or script['bridge_control']['schema']!='observed_bridge_control_v1':
        raise ValueError('Unsupported observed context schema')
    center=np.asarray(vector(center,3));position=np.asarray(vector(state['position'],3));radius=float(state['radius'])
    if not np.isfinite(radius) or radius<=0:raise ValueError('Invalid observed radius')
    bridge=script['bridge_control'];pending=context['pending_update'];bits=int(pending['types'])
    if bits<0 or bits & ~63:raise ValueError('Unknown pending native update type')
    def pending_vector(key,bit,length,divisor):
        if not bits & (1<<bit):return [0.]*length
        if pending[key] is None:raise ValueError('Missing active pending update')
        return vector(pending[key],length,divisor)
    def contact(c):
        return dict(relative_position=((np.asarray(vector(c['position'],3))-position)/radius).tolist(),
            normal=vector(c['normal'],3),surface_velocity=vector(c['surface_velocity'],3,25.),
            distance=float(c['distance'])/radius,normal_force=float(c['normal_force'])/20.,
            friction=float(c['friction']),restitution=float(c['restitution']),force=float(c['force'])/100.)
    body=dict(position=((position-center)/scale).tolist(),velocity=vector(state['velocity'],3,25.),spin=vector(state['spin'],3,100.),
        radius=radius/.18975,gravity_direction=vector(state['gravity_direction'],3),gravity_up=vector(context['gravity_up'],3),
        best_normal=vector(state['best_contact_normal'],3),camera_offset=vector(state['camera_offset'],3),
        angles=[v for key in ('yaw','next_yaw','pitch','next_pitch') for v in (float(np.sin(state[key])),float(np.cos(state[key])))],
        ground_seconds=float(state['ground_ms'])/1000.,mode=int(state['mode']),oob=float(state['oob']),smooth_camera=float(state['smooth_camera']),
        native_held=one_hot(state['held'],6),parameters=vector([state['parameters'][k] for k in PARAMETERS],len(PARAMETERS)),
        native_effects=[],effect_duration_seconds=vector(state['parameters']['effect_duration_ms'],6,1000.))
    body['parameters']=(np.asarray(body['parameters'])/PARAMETER_SCALES).tolist()
    if len(state['effects'])!=6 or len(state['effect_end_clock_ms'])!=6:raise ValueError('Missing native effect state')
    for i,(active,left) in enumerate(state['effects']):
        # Inactive endTime is uninitialized in the native constructor. Never
        # interpret it, even if its raw provenance value happens to be finite.
        signed=(int(state['effect_end_clock_ms'][i])-int(state['clock_ms'])) if active else 0
        body['native_effects'].append([float(active),float(left)/10000. if active else 0.,signed/10000.])
    inventories={}
    for side in ('target','authority'):
        s=script[side]
        if len(s['active_flags'])!=7 or len(s['effect_schedules'])!=7:raise ValueError('Missing script effect state')
        inventories[side]=dict(held=one_hot(s['held_item_power_id'],7),native_held=one_hot(s['native_held'],6),
            active=vector(s['active_flags'],7),effect_schedules=[schedule(v) for v in s['effect_schedules']],mega_schedule=schedule(s['mega_schedule']))
    interpolation=context['gravity_interpolation'] if context['gravity_interpolation_active'] else None
    if context['gravity_interpolation_active'] and not interpolation:raise ValueError('Missing active gravity interpolation')
    gravity=dict(previous_direction=vector(context['previous_gravity_direction'],3),
        global_matrix=vector(context['global_gravity_matrix'],16),previous_matrix=vector(context['previous_gravity_matrix'],16),
        default_transform=vector(context['default_gravity_transform'],16),interpolation_active=float(bool(interpolation)),
        interpolation_time=[float(interpolation[k])/1000. for k in ('elapsed_ms','duration_ms')] if interpolation else [0.,0.],
        start_quaternion=vector(interpolation['start_quaternion'],4) if interpolation else [0.]*4,
        end_quaternion=vector(interpolation['end_quaternion'],4) if interpolation else [0.]*4)
    transform=pending_vector('transform',2,16,1.)
    if bits & 4:
        for i in range(3):transform[3+4*i]=(transform[3+4*i]-center[i])/scale
    g=pending['gravity'] if bits & 16 else None
    if bits & 16 and not g:raise ValueError('Missing active pending gravity')
    tokens=dict(body=body,inventory=inventories,gravity=gravity,
        controls=dict(bridge_pending_reply=vector(observation['bridge_action'],9),input_move=vector(observation['input_move'],4),
            native_last_move=move(native['last_move']),native_pending_move=move(native['pending_move']),
            power_use_latched=float(bridge['power_use_latched']),power_yaw_hold_seconds=float(bridge['power_yaw_hold_ticks'])*.016,
            power_yaw=[float(np.sin(bridge['power_yaw'])),float(np.cos(bridge['power_yaw']))],
            configured_yaw_hold_seconds=float(bridge['configured_yaw_hold_ticks'])*.016,ss_yaw_fix=float(bridge['ss_yaw_fix']),
            fast_powerups=float(bridge['fast_powerups']),live=float(bridge['live'])),
        pending_update=dict(active=[float(bool(bits & (1<<i))) for i in range(6)],
            velocity=pending_vector('velocity',0,3,25.),spin=pending_vector('spin',1,3,100.),transform=transform,
            camera=pending_vector('camera',3,3,1.),gravity=[x for key in ('right','back','down') for x in vector(g[key],3)]+[float(g['instant'])] if g else [0.]*10,
            radius=vector([pending['radius']],1,.18975) if bits & 32 else [0.]),
        contacts=[contact(c) for c in context['current_contacts']],best_contact=contact(context['best_contact']) if context['best_contact'] else None,
        timing=dict(native_sim_phase=(int(state['clock_ms'])%32)/32.,observed_frame_phase=(int(native['frame_clock_ms'])%32)/32.,
            clock_offset=(int(state['clock_ms'])-int(native['frame_clock_ms']))/32.,dt_seconds=.016),
        coverage=dict(runtime_mass_observed=float(context['runtime_mass_observed']),historical_last_contact_observed=float(context['last_contact_decoded']),
            external_scene_observed=0.,complete_hidden_state=0.),max_force_radius=float(context['max_force_radius'])/scale)
    def finite(value):
        if isinstance(value,dict):return all(finite(v) for v in value.values())
        if isinstance(value,list):return all(finite(v) for v in value)
        return value is None or isinstance(value,(int,float)) and np.isfinite(value)
    if not finite(tokens):raise ValueError('Nonfinite semantic observed state')
    return tokens


def observable_context_example(current,future,new_reply,center,scale):
    raw=observed_world_example(current,future,new_reply)
    return dict(input=dict(current=observable_state_tokens(raw['input']['current_observed_dynamics'],center,scale),
        new_reply=raw['input']['new_reply'],dt_seconds=.016),
        target=dict(next=observable_state_tokens(raw['target']['next_observed_dynamics'],center,scale)),audit_only=raw['audit_only'])


def prepare_observed_context(source,output):
    """Distinct engineering data; sealed physical readers are never overwritten."""
    import gzip
    from nav.marble_dynamics.student_data import geometry,SURFACE_FEATURES
    from nav.source_archive import verify_archive
    protocol=json.loads((source/'prepared.json').read_text());result=json.loads((source/'result.json').read_text())
    if not result['complete'] or not protocol.get('extended_state_coverage') or not json.loads((source/'independent_audit.json').read_text())['all_six_contracts_pass']:
        raise ValueError('Requires independently verified extended state coverage')
    verify_archive(source/'source_snapshot.zip');verify_archive(source/'native_source_snapshot.zip')
    output.mkdir(parents=True,exist_ok=False);mesh_path=ML/'collision_maps/CubeIsle_Hunt.npz';mesh=CollisionMap(mesh_path);geo=geometry(mesh)
    np.savez_compressed(output/'CubeIsle_Hunt.geometry.npz',**geo)
    snapshot_sources(output,[Path(__file__),Path(__file__).with_name('student_data.py'),ML/'nav/source_archive.py'])
    records=[];sources={str(source/'prepared.json'):sha(source/'prepared.json'),str(source/'result.json'):sha(source/'result.json')}
    for index,power in enumerate(protocol['powers']):
        path=source/power/'result.json';record=json.loads(path.read_text());sources[str(path)]=sha(path)
        tape=source/power/'native.jsonl.gz';sources[str(tape)]=sha(tape)
        if sha(tape)!=record['compressed_sha256'] or record['seed']!=protocol['seeds'][index] or len(record['observations'])!=64:raise ValueError('Recorded source differs')
        commands=[[0.,0.,0.,0.,0.,0.,int(i in protocol['use_steps']),0.,0.] for i in range(64)]
        for i,observation in enumerate(record['observations']):
            if observation['step']!=i or observation['dynamics']['bridge_action']!=commands[i]:raise ValueError('Known reply alignment differs')
        for i in range(63):
            example=observable_context_example(record['observations'][i]['dynamics'],record['observations'][i+1]['dynamics'],commands[i+1],geo['center'],geo['scale'])
            example['provenance']=dict(power=power,seed=record['seed'],source_step=i,source=str(path));records.append(example)
    archive=output/'examples.jsonl.gz'
    with gzip.open(archive,'wt',encoding='utf-8') as stream:
        for record in records:stream.write(json.dumps(record,allow_nan=False)+'\n')
    atomic_json(output/'manifest.json',dict(schema='observable_context_tokens_v1',role='dev_engineering',examples=len(records),normalized_state_features=True,
        training=False,complete_hidden_state=False,object_ids_are_features=False,geometry_source=str(mesh_path),geometry_source_sha256=sha(mesh_path),
        geometry_faces=len(geo['features']),surface_feature_names=SURFACE_FEATURES,geometry_center=geo['center'].tolist(),geometry_scale=geo['scale'],
        outputs={p.name:sha(p) for p in (archive,output/'CubeIsle_Hunt.geometry.npz')},source_sha256=sources,
        scales=dict(velocity=25.,spin=100.,radius=.18975,timer_ms=10000.,force=100.,normal_force=20.),
        omitted_state=['protected_runtime_mass','historical_last_contact','external_dynamic_scene','arbitrary_script_custom_state'],
        source_command_schedule=protocol['use_steps']))
    print(json.dumps(dict(examples=len(records),geometry_faces=len(geo['features']),output=str(output),training=False)),flush=True)


def scene_context_probe(args):
    import subprocess,time
    from nav.env import HuntEnv
    from nav.protocol import NOOP_ACTION
    from nav.marble_dynamics.record import command
    names=json.loads((ML/'configs/real_maps_20261009.json').read_text())['maps']
    if args.seed!=1281013 or args.groups!=1 or args.maps!=names:raise ValueError('Scene preflight requires declared ten-map order/1281013/one group')
    args.output.mkdir(parents=True,exist_ok=False);root=ML.parent;exe=root/'Marble Blast Platinum/marbleblast_fixture_probe.exe'
    protected_paths=[root/'Marble Blast Platinum'/n for n in ('marbleblast_geometry_probe.exe','marbleblast_mbx.exe')]
    protected_paths += [root/'Marble Blast Platinum/platinum/client/scripts/ai/mlAgent.cs',ML/'models/nav/real_maps_20261010_A2/nav_latest.pth']
    protected={str(p):sha(p) for p in protected_paths};sources=list(Path(__file__).parent.glob('*.py'))
    sources += [Path(__file__).with_name(n) for n in ('engine_dynamics.inc','engine_fixture.inc','build_probe.ps1')]
    sources += [ML/'nav/collision/mesh.py',root/'Marble Blast Platinum/platinum/client/scripts/ai/mlAgent.cs']
    fingerprints={str(p):sha(p) for p in sources};snapshot_sources(args.output,sources)
    native=Path('C:/Users/doug/src/OpenPQ-TGEMIT-mbx');native_files=[native/'engine/game/marble'/n for n in ('marble.cc','marble.h','aiGeometryExport.inc','aiMarbleDynamics.inc','aiMarbleFixture.inc')]
    native_hashes={str(p):sha(p) for p in native_files};write_archive(args.output/'native_source_snapshot.zip',native_files,native)
    geometry={str(ML/'collision_maps'/(name+ext)):sha(ML/'collision_maps'/(name+ext)) for name in names for ext in ('.npz','.json')}
    protocol=dict(maps=[dict(map=n,seed=args.seed+i) for i,n in enumerate(names)],maximum_post_startup_observations=240,
        wall_limit_s=120,source_hashes=fingerprints,native_source_hashes=native_hashes,geometry_hashes=geometry,
        protected_sha256=protected,executable_sha256=sha(exe),training=False,test_motion_labels_opened=False,
        contract='Frozen client/server whole-container force/player inventory; no force evaluation, actuation, physics labels or state restore. One neutral reply after snapshot checks bridge resumes.')
    atomic_json(args.output/'prepared.json',protocol);results=[];batch_start=time.perf_counter()
    for index,name in enumerate(names):
        if (args.output/'STOP').exists():break
        folder=args.output/name;folder.mkdir();start=time.perf_counter();proc=None;startup_ticks=None
        env=HuntEnv(args.port,speed=3,log=lambda s:print(s,flush=True));env.obs_ms=16;env.srv.settimeout(70)
        startup=subprocess.STARTUPINFO();startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
        log=(folder/'game.log').open('wb');result=dict(map=name,seed=args.seed+index,passed=False)
        try:
            proc=subprocess.Popen([str(exe),'-autotrain',name,'-aiport',str(args.port),'-offline','-aicompetitive','0','-aiseed',str(args.seed+index)],
                cwd=str(exe.parent),startupinfo=startup,stdout=log,stderr=log)
            env.connect();env.conn.settimeout(15)
            def refuse_reconnect():raise ConnectionError('Owned scene audit disconnected; no replacement process')
            env._accept=refuse_reconnect
            if env.info.get('mission')!=name:raise ValueError('Unexpected mission')
            env.control('SPINWAIT 4');command(env,'DYNAMICSWAIT 600000','dynamics_wait');env.control('FIXEDSTEP 16')
            for _ in range(400):
                env.step(NOOP_ACTION,repeat=1)
                if time.perf_counter()-start>120:raise TimeoutError('Scene startup bound')
            startup_ticks=env.ticks;pending=env.msg;path=(folder/'scene.json').resolve()
            ack=frozen_fixture_command(env,'FIXTURESCENE '+path.as_posix(),timeout_s=min(5,120-(time.perf_counter()-start)),ack_name='fixture_scene')
            if not path.exists():raise ValueError('Native scene file missing')
            scene=json.loads(path.read_text());result['scene_sha256']=sha(path)
            if ack.fields[1]!='1' or not scene['physical_fields_unchanged']:raise ValueError('Passive scene/native clock contract failed')
            if env.msg is not pending:raise ValueError('Frozen scene command consumed pending observation')
            for side in ('client','server'):
                s=scene[side]
                if s['force_count']!=len(s['force_objects']) or s['player_count']!=len(s['player_bodies']):raise ValueError('Scene inventory counts differ')
                if sum(bool(p['owned_body']) for p in s['player_bodies'])!=1:raise ValueError('Owned player body not uniquely identified')
            result['counts']={side:dict(forces=scene[side]['force_count'],players=scene[side]['player_count'],
                unsupported_force_classes=[p['class'] for p in scene[side]['force_objects'] if not p['force_definitions_supported']]) for side in ('client','server')}
            _,outcome=env.step(NOOP_ACTION,repeat=1);result['resumed_outcome']=outcome
            result['passed']=not any(outcome[k] for k in ('reconnected','round_ended'))
        except Exception as error:result['failure']=repr(error)
        finally:
            result['post_startup_observations']=env.ticks-startup_ticks if startup_ticks is not None else None;result['wall_s']=time.perf_counter()-start
            if proc is not None and proc.poll() is None:proc.terminate();proc.wait(timeout=10)
            env.close();log.close();atomic_json(folder/'result.json',result)
        results.append(result);atomic_json(args.output/'result.json',dict(cases=results,complete=False,training=False));print(json.dumps(result),flush=True)
    intact=all(sha(p)==v for p,v in fingerprints.items()) and all(sha(p)==v for p,v in native_hashes.items())
    geometry_intact=all(sha(p)==v for p,v in geometry.items());protected_intact=all(sha(p)==v for p,v in protected.items()) and sha(exe)==protocol['executable_sha256']
    complete=len(results)==len(names) and all(r['passed'] and not r.get('failure') and r['map']==names[i] and r['seed']==1281013+i and
        r['post_startup_observations']<=240 and r['wall_s']<=120 for i,r in enumerate(results)) and intact and geometry_intact and protected_intact
    atomic_json(args.output/'result.json',dict(cases=results,complete=complete,source_hashes_unchanged=intact,
        geometry_hashes_unchanged=geometry_intact,protected_artifacts_unchanged=protected_intact,wall_s=time.perf_counter()-batch_start,
        training=False,test_motion_labels_opened=False,complete_force_forecast=False))
    if not complete:raise RuntimeError('Scene coverage audit incomplete/failed; every result retained')


if __name__=='__main__':main()
