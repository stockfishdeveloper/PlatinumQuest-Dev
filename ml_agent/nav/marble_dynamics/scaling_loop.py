"""Fixed-recipe navigator data scaling, parallel pools and balanced corrections.

One maintained phase driver; workers write separate evidence and failures. It
does not grant powers, restore scored worlds, alter losses or promote a default
policy. A selected parent requires two additional ten-map seeds.
"""
import argparse
import ast
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import socket
import subprocess
import sys
import time

import numpy as np

from nav.collision.mesh import atomic_json
from nav.marble_dynamics.record import ML,ROOT,sha
from nav.marble_dynamics.static_cache import CollectionProfile,cached_terrain,cached_walking,load_queries,save_queries,save_clearance,marble_processes


def check_recipe(protocol):
    source=Path(protocol['recipe_file']);tree=ast.parse(source.read_text())
    nodes={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    changed=[n for n,v in protocol['recipe_ast_sha256'].items()
        if n not in nodes or hashlib.sha256(ast.dump(nodes[n],include_attributes=False).encode()).hexdigest()!=v]
    if changed:raise ValueError('Frozen navigator recipe changed: '+str(changed))
    return True


def strip_options(arguments,options):
    result=[];i=0
    while i<len(arguments):
        if arguments[i] in options:
            i+=1
            while i<len(arguments) and not arguments[i].startswith('--'):i+=1
        else:result.append(arguments[i]);i+=1
    return result


def checked_move(source,destination,root):
    source=Path(source).resolve();destination=Path(destination).resolve();root=Path(root).resolve()
    if not source.is_relative_to(root) or not destination.is_relative_to(root):
        raise ValueError('Evidence move escaped declared pool root')
    if destination.exists():raise ValueError('Refuse overwrite of evidence: '+str(destination))
    destination.parent.mkdir(parents=True,exist_ok=True);source.rename(destination)


def parallel_pool(args,arguments):
    """Run up to four native map pairs at once; every branch has its own port."""
    from nav.marble_dynamics.reliability_eval import continuous_branches
    root=args.output.resolve();started=time.perf_counter()
    ports=queue.Queue()
    for port in range(args.port,args.port+args.workers):
        with socket.socket() as probe:probe.bind(('0.0.0.0',port))
        ports.put(port)
    common=strip_options(arguments,{'--maps','--seeds','--output','--port','--workers','--only-branch','--prepared-parent','--map-index-offset'})
    jobs=[(index,name,seed,kind) for index,name in enumerate(args.maps)
        for seed in args.seeds for kind in continuous_branches(args,index,seed)
        if not args.only_branch or kind==args.only_branch]
    results=[];failures=[]
    def run(job):
        index,name,seed,kind=job;port=ports.get()
        folder=root/'_jobs'/(name+'.'+str(seed)+'.'+kind);log=folder.with_suffix(folder.suffix+'.log')
        folder.parent.mkdir(exist_ok=True);start=time.perf_counter()
        command=[sys.executable,'-u','-m','nav.marble_dynamics.reliability_eval',*common,
            '--maps',name,'--seeds',str(seed),'--output',str(folder),'--port',str(port),
            '--workers','1','--only-branch',kind,'--map-index-offset',str(index),
            '--prepared-parent',str(root)]
        try:
            with log.open('wb') as stream:
                proc=subprocess.Popen(command,cwd=str(ROOT),stdout=stream,stderr=stream,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                while proc.poll() is None:
                    if (root/'STOP').exists() and folder.exists():(folder/'STOP').touch()
                    time.sleep(.5)
                code=proc.returncode
            reports=json.loads((folder/'summary.json').read_text())['branches'] if (folder/'summary.json').exists() else []
            if code or len(reports)!=args.rounds or any(not r.get('score_valid') or not r.get('declared_inputs_valid') for r in reports):
                return dict(job=job,status='invalid',returncode=code,reports=reports,log=str(log),elapsed_s=time.perf_counter()-start)
            for report in reports:
                relative=Path(name)/str(seed)/kind/str(report['round'])
                checked_move(folder/relative,root/relative,root)
            return dict(job=job,status='valid',reports=reports,elapsed_s=time.perf_counter()-start)
        finally:ports.put(port)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_jobs={executor.submit(run,job):job for job in jobs}
        for future in as_completed(future_jobs):
            try:result=future.result()
            except Exception as exc:result=dict(job=future_jobs[future],status='invalid',reason=repr(exc),reports=[])
            results.extend(result['reports'])
            if result['status']!='valid':failures.append(result)
            atomic_json(root/'summary.json',dict(branches=results,failures=failures,
                full_round=True,policy_training_started=False,checkpoint_unchanged=True,
                elapsed_wall_s=time.perf_counter()-started,parallel_workers=args.workers,
                maximum_native_processes=8,complete=len(results)==len(jobs)*args.rounds and not failures))
            print(json.dumps(dict(pool_progress=len(results),expected=len(jobs)*args.rounds,
                job=result['job'],status=result['status'],elapsed_wall_s=time.perf_counter()-started)),flush=True)
    if failures:raise RuntimeError('Parallel pool preserved invalid jobs: '+str(len(failures)))
    return results


def correction_worker(job):
    from nav.marble_dynamics.fast_nav import collect_advice,filter_corrections
    from nav.collision.experimental_sweep import CachedCompiledSweepMap
    from nav.terrain import TerrainMap
    name=job['map'];root=Path(job['output']);root.mkdir(parents=True,exist_ok=False)
    rollouts=Path(job['rollouts']);dataset=Path(job['dataset']);dll=Path(job['dll']);cache=Path(job['cache'])
    exclusions=root/'excluded_states.json';atomic_json(exclusions,{name:job.get('exclude_keys',[])})
    warm=time.perf_counter();mesh=CachedCompiledSweepMap(ML/'collision_maps'/(name+'.npz'),dll)
    terrain=cached_terrain(TerrainMap.resolve(name),cache);load_queries(mesh,dll,cache)
    walking=cached_walking(terrain,mesh,dll,cache);save_queries(mesh);save_clearance(walking)
    cold_setup=time.perf_counter()-warm
    with CollectionProfile() as profile:
        raw=collect_advice(rollouts,dataset,root/'raw',dll,job['attempts'],job['query_s'],
            maps=[name],wall_s=14400,cache_dir=cache,candidate_limit=job.get('candidate_limit'),exclude_states=exclusions)
        filtered=filter_corrections(root/'raw',dataset,root/'corrections',dll,job['query_s'],
            maps=[name],wall_s=14400,cache_dir=cache,defer_replay=True)
    report=dict(map=name,raw=raw['maps'][name],corrections=filtered['maps'][name],
        profile=profile.report(),cold_setup_s=cold_setup,worker_pid=os.getpid(),
        corrections_path=str(root/'corrections'))
    atomic_json(root/'worker_report.json',report);return report


def case_paths(folder,manifest,name=None):
    if manifest.get('case_shards'):
        return [Path(folder)/s['path'] for s in manifest['case_shards'] if name is None or s['map']==name]
    return [Path(folder)/manifest.get('cases_path','cases.json')]


def read_cases(path):
    if Path(path).suffix=='.gz':
        with gzip.open(path,'rt',encoding='utf-8') as stream:yield from map(json.loads,stream)
    else:yield from json.loads(Path(path).read_text())


def accepted_state_keys(folders,names):
    from nav.marble_dynamics.fast_nav import advice_state_key
    keys={n:set() for n in names}
    for folder in map(Path,folders):
        manifest=json.loads((folder/'manifest.json').read_text())
        if not manifest.get('complete'):raise ValueError('Incomplete correction evidence')
        for name in names:
            if name not in manifest['maps']:continue
            with np.load(folder/(name+'.npz'),allow_pickle=False) as z:
                keys[name].update(advice_state_key(x,q) for x,q in zip(z['x'],z['q']))
    return keys


def distinct_correction_counts(folders,names,prior=()):
    current=accepted_state_keys(folders,names);known=accepted_state_keys(prior,names)
    return {name:len(current[name]-known[name]) for name in names}


def queried_state_keys(folders,name,dataset,receipt_root):
    """Recover old sampled-input identities once, without replaying forecasts."""
    from nav.marble_dynamics.fast_nav import advice_state_key,features
    keys=accepted_state_keys(folders,[name])[name]
    with np.load(Path(dataset)/(name+'.npz'),allow_pickle=False) as sealed:
        center=sealed['center'].copy();scale=sealed['scale'].copy()
    for folder in map(Path,folders):
        manifest=json.loads((folder/'manifest.json').read_text());parent=manifest.get('parent_advice')
        if not parent:continue
        raw=Path(parent);meta=json.loads((raw/'manifest.json').read_text())
        if meta.get('queried_states_path'):
            path=raw/meta['queried_states_path']
            if sha(path)!=meta['queried_states_sha256']:raise ValueError('Sampled input identities changed')
            keys.update(json.loads(path.read_text()).get(name,[]));continue
        receipt=Path(receipt_root)/name/(sha(raw/'manifest.json')+'.json')
        if receipt.exists():
            keys.update(json.loads(receipt.read_text())['state_keys']);continue
        protocol=json.loads((raw/'protocol.json').read_text());rollouts=Path(protocol['rollouts'])
        pool=json.loads((rollouts/'summary.json').read_text())
        reports=[r for r in pool['branches'] if r['map']==name and r['branch']=='fast_navigator']
        if not reports:continue
        if len(reports)!=1:raise ValueError('Ambiguous original sampled-state journal')
        report=reports[0];journal=rollouts/name/str(report['seed'])/'fast_navigator'/str(report['round'])/'decisions.jsonl.gz'
        with gzip.open(journal,'rt',encoding='utf-8') as stream:
            rows={r['decision']:r for r in map(json.loads,stream)}
        sampled=set()
        for path in case_paths(raw,meta,name):
            if path.suffix=='.gz':
                # Headers precede large forecasts; do not deserialize/replay them.
                with gzip.open(path,'rt',encoding='utf-8') as stream:
                    for line in stream:
                        header=line[:1024]
                        map_match=re.search(r'"map"\s*:\s*"([^"\\]+)"',header)
                        match=re.search(r'"decision"\s*:\s*(\d+)',header)
                        if not map_match or not match:raise ValueError('Invalid original query header')
                        if map_match.group(1)==name:sampled.add(int(match.group(1)))
            else:sampled.update(c['decision'] for c in read_cases(path) if c['map']==name)
        recovered=set()
        for decision in sampled:
            row=rows[decision];goal=row['choice']['actor_goal_context']
            x,q=features(row['before'],row['raw_observation'],goal['destination'],goal['waypoint'],center,scale,base_action=row['choice']['base_actor_action'])
            recovered.add(advice_state_key(x,q))
        atomic_json(receipt,dict(raw_manifest_sha256=sha(raw/'manifest.json'),state_keys=sorted(recovered),
            scope='Recorded sampled inputs only; no independent forecast replay or archive audit.'))
        keys.update(recovered)
    return sorted(keys)


def merge_corrections(folders,dataset,output,min_total=0,min_per_map=0,prior=()):
    """Count distinct exact inputs across journals; retain original evidence."""
    from nav.marble_dynamics.fast_nav import protected_hashes
    from nav.marble_dynamics.fast_nav import advice_state_key
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    original=json.loads((Path(dataset)/'manifest.json').read_text());maps=original['maps']
    grouped={name:[] for name in maps};sources={};shards=[];counts={};label_counts={}
    protected=protected_hashes();historical_code={};journal_lookup={};cache_dirs=set()
    prior_keys=accepted_state_keys(prior,maps);duplicates={};raw_counts={}
    for folder in map(Path,prior):
        prior_manifest=json.loads((folder/'manifest.json').read_text())
        sources[str(folder/'manifest.json')]=sha(folder/'manifest.json')
        for name,record in prior_manifest['maps'].items():sources[str(folder/(name+'.npz'))]=record['tensor_sha256']
    for folder in map(Path,folders):
        manifest=json.loads((folder/'manifest.json').read_text())
        if not manifest.get('complete') or not manifest.get('first_reply_causal_corrections'):
            raise ValueError('Incomplete/unfiltered correction worker')
        historical_code[str(folder)]={p:h for p,h in manifest['sources'].items() if Path(p).suffix in ('.py','.cpp')}
        sources.update({p:h for p,h in manifest['sources'].items() if Path(p).suffix not in ('.py','.cpp')})
        journal_lookup.update({h:p for p,h in manifest['sources'].items() if p.endswith('decisions.jsonl.gz')})
        worker_protocol=json.loads((folder/'protocol.json').read_text())
        if worker_protocol.get('cache_dir'):cache_dirs.add(worker_protocol['cache_dir'])
        sources[str(folder/'manifest.json')]=sha(folder/'manifest.json')
        for name,record in manifest['maps'].items():
            grouped[name].append((folder,manifest));sources[str(folder/(name+'.npz'))]=record['tensor_sha256']
            for path in case_paths(folder,manifest,name):
                sources[str(path)]=sha(path)
    for name in maps:
        values={k:[] for k in ('x','q','y','decision')};seen=set();states=set(prior_keys[name]);buttons=np.zeros(3,dtype=int)
        duplicates[name]=0;raw_counts[name]=0;merged_cases=output/(name+'.cases.jsonl.gz')
        with gzip.open(merged_cases,'wt',encoding='utf-8') as stream:
            for folder,manifest in grouped[name]:
                with np.load(folder/(name+'.npz'),allow_pickle=False) as z:
                    ix=0
                    for path in case_paths(folder,manifest,name):
                        for case in read_cases(path):
                            if case['map']!=name:continue
                            saved=case
                            if case['accepted']:
                                if ix>=len(z['y']):raise ValueError('Correction tensor/case count mismatch')
                                provenance=(case['source_sha256'],case['decision'])
                                if provenance in seen:raise ValueError('Duplicate correction provenance: '+name)
                                seen.add(provenance);raw_counts[name]+=1
                                key=advice_state_key(z['x'][ix],z['q'][ix])
                                if key in states:
                                    duplicates[name]+=1
                                    saved=dict(map=name,decision=case['decision'],source_sha256=case['source_sha256'],
                                        accepted=False,accepted_in_worker=True,merge_exclusion='duplicate_model_input',
                                        worker_cases=str(path),source_journal=case.get('source_journal'))
                                else:
                                    states.add(key)
                                    for k in values:values[k].append(z[k][ix].copy())
                                    buttons+=(z['y'][ix,6:]>0.5)
                                ix+=1
                            stream.write(json.dumps(saved)+'\n')
                    if ix!=len(z['y']):raise ValueError('Correction tensor/case count mismatch')
        shards.append(dict(map=name,path=merged_cases.name,sha256=sha(merged_cases)));sources[str(merged_cases)]=sha(merged_cases)
        shapes={'x':(0,original['state_size']),'q':(0,4,3),'y':(0,9),'decision':(0,)}
        merged={k:np.asarray(v,dtype=np.int32 if k=='decision' else np.float32).reshape((-1,)+shapes[k][1:]) if v else np.empty(shapes[k],dtype=np.int32 if k=='decision' else np.float32) for k,v in values.items()}
        np.savez_compressed(output/(name+'.npz'),**merged)
        counts[name]=dict(accepted=len(merged['y']),tensor_sha256=sha(output/(name+'.npz')))
        label_counts[name]=dict(jump=int(buttons[0]),use=int(buttons[1]),Blast=int(buttons[2]))
    from nav.source_archive import snapshot_sources
    code=list(Path(__file__).parent.glob('*.py'))+list((ML/'nav').glob('*.py'))+list((ML/'nav/collision').glob('*.py'))
    sources.update({str(p):sha(p) for p in code});snapshot_sources(output,code)
    integrity=all(sha(p)==v for p,v in sources.items()) and protected_hashes()==protected
    quota=sum(c['accepted'] for c in counts.values())>=min_total and all(c['accepted']>=min_per_map for c in counts.values())
    result=dict(schema='analytical_navigation_advice_v1',complete=integrity,maps=counts,sources=sources,
        protected=protected,native_outcomes=False,first_reply_causal_corrections=True,
        historical_code_hashes=historical_code,case_shards=shards,cases_sha256=hashlib.sha256(json.dumps(shards,sort_keys=True).encode()).hexdigest(),
        accepted=sum(c['accepted'] for c in counts.values()),label_counts=label_counts,
        raw_accepted_by_map=raw_counts,duplicate_inputs_excluded_by_map=duplicates,
        quota_met=quota,minimum_total=min_total,minimum_per_map=min_per_map,independent_replay_pending=True)
    atomic_json(output/'manifest.json',result)
    atomic_json(output/'protocol.json',dict(dataset=str(dataset),worker_corrections=list(map(str,folders)),
        scope='Distinct exact x/q student states across journals; signed zero normalized. Original worker cases retained. Identical causal filter; replay deferred until selection.',
        rollouts=None,journal_lookup=journal_lookup,cache_dir=next(iter(cache_dirs)) if len(cache_dirs)==1 else None))
    atomic_json(output/'collection_integrity.json',dict(passed=integrity,manifest_sha256=sha(output/'manifest.json'),
        independent_replay=False,replay_pending=True,quota_met=quota))
    if not integrity:raise ValueError('Merged correction integrity failed')
    return result


def collect_parallel(rollouts,dataset,output,dll,cache,workers=8,attempts=1200,query_s=1.,maps=None,candidate_limit=None,map_costs=None,exclude_folders=()):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    names=maps or json.loads((Path(dataset)/'manifest.json').read_text())['maps']
    jobs=[dict(map=n,rollouts=str(rollouts),dataset=str(dataset),output=str(output/'workers'/n),
        dll=str(dll),cache=str(cache),attempts=attempts,query_s=query_s,candidate_limit=candidate_limit.get(n) if isinstance(candidate_limit,dict) else candidate_limit,
        exclude_keys=queried_state_keys(exclude_folders,n,dataset,output.parent/'query_state_receipts')) for n in names]
    jobs=order_collection_jobs(jobs,map_costs)
    reports=[]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures={executor.submit(correction_worker,job):job for job in jobs}
        for future in as_completed(futures):
            try:report=future.result();reports.append(report)
            except Exception as exc:
                atomic_json(output/'failure.json',dict(job=futures[future],reason=repr(exc)));raise
            atomic_json(output/'progress.json',dict(maps=reports,elapsed_wall_s=time.perf_counter()-started,complete=False))
            print(json.dumps(dict(correction_progress=report['map'],accepted=report['corrections']['accepted'],
                elapsed_wall_s=time.perf_counter()-started)),flush=True)
    result=dict(complete=True,workers=workers,maps=reports,elapsed_wall_s=time.perf_counter()-started,
        corrections=[r['corrections_path'] for r in reports])
    atomic_json(output/'result.json',result);atomic_json(output/'progress.json',result);return result


def pool_table(pool,teacher):
    prepared=json.loads((Path(pool)/'prepared.json').read_text());data=json.loads((Path(pool)/'summary.json').read_text())
    teacher_data=json.loads((Path(teacher)/'summary.json').read_text());table=[]
    for name in prepared['maps']:
        def records(branch):return [r for r in data['branches'] if r['map']==name and r['branch']==branch]
        student=records('fast_navigator');base=records('geometry_navigator')
        if len(student)!=len(prepared['seeds'])*prepared['rounds'] or len(base)!=len(student) or any(not r.get('score_valid') or not r.get('declared_inputs_valid') for r in student+base):
            raise ValueError('Incomplete or invalid map pair: '+name)
        teachers=[r for r in teacher_data['branches'] if r['map']==name and r['branch']=='physics_plan']
        s=float(np.mean([r['actual_points'] for r in student]));b=float(np.mean([r['actual_points'] for r in base]));t=float(np.mean([r['actual_points'] for r in teachers]))
        table.append(dict(map=name,student_points=s,geometry_A2_points=b,planner_points=t,
            student_to_planner=s/t if t else None,
            student_falls=float(np.mean([r['falls'] for r in student])),A2_falls=float(np.mean([r['falls'] for r in base])),
            student_accepted_rescues=float(np.mean([r['recovery']['rescues'] for r in student])),
            A2_accepted_rescues=float(np.mean([r['recovery']['rescues'] for r in base]))))
    means={k:float(np.mean([r[k] for r in table])) for k in ('student_points','geometry_A2_points','planner_points')}
    result=dict(per_map=table,equal_map=means,student_to_planner=means['student_points']/means['planner_points'],
        student_seeds=prepared['seeds'],teacher_seeds=sorted({r['seed'] for r in teacher_data['branches']}),
        teacher_source=str(teacher),teacher_comparison='Same-seed/current-protocol teacher in separate native worlds; exact starting-world equivalence is not certified.' if sorted(prepared['seeds'])==sorted({r['seed'] for r in teacher_data['branches']}) else 'Archived familiar-map teacher; different seed/source version, not a paired teacher round.',
        winner_screen=means['student_points']>means['geometry_A2_points'],
        pool_wall_s=data.get('elapsed_wall_s'))
    atomic_json(Path(pool)/'iteration_table.json',result)
    lines=['| Map | Student | Geometry A2 | Planner (archived) | Student/planner | Falls student/A2 | Accepted rescues student/A2 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in table:
        ratio='n/a' if r['student_to_planner'] is None else f"{r['student_to_planner']:.1%}"
        lines.append(f"| {r['map']} | {r['student_points']:g} | {r['geometry_A2_points']:g} | {r['planner_points']:g} | {ratio} | {r['student_falls']:g}/{r['A2_falls']:g} | {r['student_accepted_rescues']:g}/{r['A2_accepted_rescues']:g} |")
    lines.extend(['',result['teacher_comparison'],f"Student seeds: {prepared['seeds']}; archived teacher seeds: {result['teacher_seeds']}.",
        f"Equal-map points: student{means['student_points']:.2f}, A2{means['geometry_A2_points']:.2f}, planner{means['planner_points']:.2f}; ratio{result['student_to_planner']:.1%}."])
    (Path(pool)/'iteration_table.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return result


def phase_note(root,stage,**details):
    status=dict(utc=datetime.now(timezone.utc).isoformat(),stage=stage,**details)
    atomic_json(Path(root)/'status.json',status)
    print(json.dumps(status),flush=True)
    text=status['utc']+' — fixed-recipe scaling: '+stage+'; '+json.dumps(details)+'\n'
    for name in ('HANDOFF_FAST_NAVIGATOR_2026-10-13.md','HANDOFF_NAV_TRAINING_LOG.md'):
        with (ML/'docs'/name).open('a',encoding='utf-8') as stream:stream.write('\n'+text)
    return status


def execute_phase(root,label,command):
    root=Path(root)
    if (root/'STOP').exists():raise InterruptedError('User stop marker')
    log=root/(label+'.log');started=time.perf_counter()
    with log.open('wb') as stream:
        process=subprocess.Popen(command,cwd=str(ROOT),stdout=stream,stderr=stream,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        phase_note(root,'running '+label,owned_worker_pid=process.pid,log=str(log))
        while process.poll() is None:
            if (root/'STOP').exists() and '--output' in command:
                folder=Path(command[command.index('--output')+1])
                if folder.exists():(folder/'STOP').touch()
            time.sleep(1.)
    elapsed=time.perf_counter()-started
    if process.returncode:raise RuntimeError(label+' failed; preserved log '+str(log))
    return elapsed


def pool_command(args,checkpoint,output,seed,maps=None,student_only=False,teacher_only=False,random_start=False,distinct_from=(),start_manifest=None):
    names=maps or json.loads((args.dataset/'manifest.json').read_text())['maps']
    command=[sys.executable,'-u','-m','nav.marble_dynamics.reliability_eval',
        '--maps',*names,'--seeds',str(seed),'--full-round','--geometry-route',
        '--query-cache','--clearance-cache','--physics-frame-ms','16',
        '--sweep-dll',str(args.sweep_dll),'--cache-dir',str(args.cache),
        '--workers',str(args.workers),'--port',str(args.port),
        '--round-wall-limit-s','600','--output',str(output)]
    if teacher_only:
        command+=['--horizon','24','--reuse-plan','--resource-acquisition','--preserve-resources',
            '--recharge-continuation','--landing-search','--proposal-samples','20',
            '--search-budget-s','1','--fallback-cooldown','64','--only-branch','physics_plan']
    else:
        command+=['--fast-checkpoint',str(checkpoint)]
        if student_only:command+=['--fast-only']
    if random_start:command+=['--random-starts']
    if distinct_from:command+=['--distinct-from-pools',*map(str,distinct_from)]
    if start_manifest:command+=['--start-seed-manifest',str(start_manifest)]
    return command


def audit_selected(args,pools,advice=None,skip_existing=True):
    """One audit bundle per selected candidate, after its pools; cache receipts."""
    from nav.source_archive import verify_archive
    seen_archives={}
    for folder in pools:
        folder=Path(folder)
        if skip_existing and (folder/'full_round_audit.json').exists() and (folder/'source_integrity.json').exists():continue
        prepared=json.loads((folder/'prepared.json').read_text())
        # Code history is archived. Models, geometry, bridge and kernel inputs
        # must still match, and the current frozen recipe is checked separately.
        changed=[p for p,h in prepared['source_hashes'].items() if sha(p)!=h]
        archive=folder/'source_snapshot.zip';digest=sha(archive)
        if digest not in seen_archives:seen_archives[digest]=verify_archive(archive)
        atomic_json(folder/'source_integrity.json',dict(source_hashes=len(prepared['source_hashes']),
            changed=changed,archived_source_files=len(seen_archives[digest]['files'])))
        if changed:raise ValueError('Selected pool sources changed')
        execute_phase(args.output,'audit_'+folder.name,[sys.executable,'-u','-m',
            'nav.marble_dynamics.reliability_audit',str(folder),'--recompute-fast'])
    if advice:
        execute_phase(args.output,'audit_advice_'+Path(advice).name,[sys.executable,'-u','-m',
            'nav.marble_dynamics.fast_nav','audit-advice','--folder',str(advice),
            '--dataset',str(args.dataset),'--sweep-dll',str(args.sweep_dll)])
    execute_phase(args.output,'tests_'+Path(pools[-1]).name,[sys.executable,'-m','unittest',
        'discover','-s','ml_agent/nav/marble_dynamics','-p','test_*.py'])


def confirm(args,checkpoint,screen,index):
    tables=[];pools=[]
    for offset in (1,2):
        seed=args.seed+index*10+offset
        starts=getattr(args,'confirmation_start_manifests',None)
        manifest=starts[offset-1] if starts else None
        if manifest:seed=json.loads(Path(manifest).read_text())['master_seed']
        pool=args.output/('confirmation_'+str(index)+'_'+str(seed))
        elapsed=execute_phase(args.output,'pool_'+pool.name,pool_command(args,checkpoint,pool,seed,
            random_start=True,distinct_from=pools,start_manifest=manifest))
        table=pool_table(pool,args.teacher);table['pool_wall_s']=elapsed;tables.append(table);pools.append(pool)
    screen_score=screen['equal_map']['student_points']
    # Each seed must beat its same-protocol A2. Preserve per-map trades in table.
    passed=all(t['winner_screen'] for t in tables)
    return dict(passed=passed,pools=pools,tables=tables,screen_score=screen_score)



def gain_summary(protocol,initial,iterations):
    names=set(protocol['development_reference']['maps'])
    development=[float(np.mean([r['student_points'] for r in it['table']['per_map']
        if r['map'] in names])) for it in iterations]
    full=[it['table']['equal_map']['student_points'] for it in iterations]
    return dict(initial_ten_map=initial['equal_map'],three_map_reference=protocol['development_reference'],
        qualifying_scaled_loops=sum(it.get('qualifying_scaled_loop',True) for it in iterations),
        scaled_three_map_means=development,scaled_ten_map_means=full,
        gain_over_three_map_reference=any(v>protocol['development_reference']['student_mean'] for v in development),
        gain_over_initial_ten_map=any(v>initial['equal_map']['student_points'] for v in full))


def no_gain_stop(protocol,initial,iterations):
    qualifying=sum(it.get('qualifying_scaled_loop',True) for it in iterations)
    return qualifying>=protocol['minimum_scaled_loops'] and not gain_summary(protocol,initial,iterations)['gain_over_three_map_reference']


def completed_collection(folder):
    """Reuse a closed collection with its final integrity receipt; no archive replay."""
    folder=Path(folder);result=json.loads((folder/'result.json').read_text())
    if not result.get('complete') or not result.get('maps'):
        raise ValueError('Refuse incomplete correction collection: '+str(folder))
    for report in result['maps']:
        data=Path(report['corrections_path'])
        if not data.resolve().is_relative_to(folder.resolve()):
            raise ValueError('Collection evidence escaped its declared root')
        manifest=json.loads((data/'manifest.json').read_text())
        receipt=json.loads((data/'collection_integrity.json').read_text())
        if not manifest.get('complete') or not manifest.get('first_reply_causal_corrections') or not receipt.get('passed') or receipt.get('manifest_sha256')!=sha(data/'manifest.json'):
            raise ValueError('Closed collection lacks matching final integrity evidence')
        if manifest['maps'][report['map']]['accepted']!=report['corrections']['accepted']:
            raise ValueError('Collection report/correction count mismatch')
    return result


def recover_remaining_collections(root,iteration):
    """Recover every closed remaining-state batch before deciding map deficits."""
    prefix='iteration_'+str(iteration)+'_remaining_states_';suffix='_corrections';found=[]
    for folder in Path(root).glob(prefix+'*'+suffix):
        number=folder.name[len(prefix):-len(suffix)]
        if not number.isdigit():raise ValueError('Unexpected remaining-state folder')
        found.append((int(number),folder))
    found.sort()
    return [(number,folder,completed_collection(folder)) for number,folder in found]


def reuse_student_pool(pool,checkpoint,seed,needed):
    """An interrupted supervisor may leave a complete native child pool."""
    prepared=json.loads((pool/'prepared.json').read_text());summary=json.loads((pool/'summary.json').read_text())
    if (not summary.get('complete') or prepared.get('fast_checkpoint')!=str(checkpoint)
            or prepared.get('seeds')!=[seed] or not set(needed).issubset(prepared['maps'])
            or prepared['source_hashes'].get(str(checkpoint))!=sha(checkpoint)):
        raise ValueError('Existing supplemental pool is incomplete or belongs to another student/protocol')
    if any(r['branch']!='fast_navigator' or r.get('round',1)!=1 for r in summary['branches']):
        raise ValueError('Supplemental state evidence is not the declared student round')
    return float(summary['elapsed_wall_s'])


def initial_confirmation_receipts(args,pools):
    """Consume existing selected-parent receipts without re-auditing unchanged data."""
    expected=sha(args.initial_checkpoint)
    for folder in pools:
        source=json.loads((folder/'source_integrity.json').read_text())
        audit=json.loads((folder/'full_round_audit.json').read_text())
        if source.get('changed') or source.get('source_hashes',0)<1 or not audit.get('complete_pool') or not audit.get('complete_declared_rounds'):
            raise ValueError('Resume requires completed initial confirmation audits')
        learned=[r.get('fast_model_output_replay',{}) for r in audit['rounds'] if r['branch']=='fast_navigator']
        if not learned or any(not r.get('passed') or r.get('checkpoint_sha256')!=expected for r in learned):
            raise ValueError('Resume requires matching passed learned-output replays')
    test_log=args.output/('tests_'+Path(pools[-1]).name+'.log')
    if not test_log.read_text().rstrip().endswith('OK'):
        raise ValueError('Resume requires the completed selected-parent test receipt')


def write_loop_state(args,protocol,parent,latest_pool,selected,folders,iterations,best,candidate_limit,next_iteration):
    state=dict(schema='fixed_navigator_loop_state_v1',parent=str(parent),parent_sha256=sha(parent),
        latest_pool=str(latest_pool),selected=selected,all_correction_folders=folders,
        iterations=iterations,best_screen_points=best,candidate_limit=candidate_limit,
        next_iteration=next_iteration,recipe_ast_sha256=protocol['recipe_ast_sha256'])
    atomic_json(args.output/'loop_state.json',state)


def resume_loop_state(args,protocol):
    state=json.loads((args.output/'loop_state.json').read_text())
    if state.get('schema')!='fixed_navigator_loop_state_v1' or state.get('recipe_ast_sha256')!=protocol['recipe_ast_sha256']:
        raise ValueError('Resume recipe/state mismatch')
    if sha(state['parent'])!=state['parent_sha256']:
        raise ValueError('Resume parent changed')
    if state['next_iteration']!=len(state['iterations'])+1:
        raise ValueError('Resume is only supported at a recorded iteration boundary')
    return state


def order_collection_jobs(jobs,costs):
    """Longest measured query work first; sampling and per-map jobs unchanged."""
    def cost(job):
        value=float((costs or {}).get(job['map'],0) or 0)
        return value*job['attempts'] if np.isfinite(value) and value>0 else 0
    return sorted(jobs,key=cost,reverse=True)


def drive(args):
    root=args.output;root.mkdir(parents=True,exist_ok=True)
    if (root/'status.json').exists() and not args.resume:raise ValueError('Refuse overwrite of an existing scaling phase; inspect/resume explicitly')
    if (root/'STOP').exists():raise InterruptedError('Stop marker remains; inspect before resuming')
    protocol=json.loads((root/'phase_protocol.json').read_text());check_recipe(protocol)
    initial=pool_table(args.initial_pool,args.teacher);parent=args.initial_checkpoint
    latest_pool=args.initial_pool;selected=[];all_correction_folders=[str(args.initial_corrections)] if args.initial_corrections else [];iterations=[]
    best_screen_points=initial['equal_map']['student_points'];candidate_limit={};map_costs={}
    if args.collection_profile:
        observed=json.loads(args.collection_profile.read_text())['maps']
        candidate_limit={r['map']:5 for r in observed if (r['raw'].get('query_median_s') or 0)>.5}
        map_costs={r['map']:r['raw'].get('query_median_s') or 0 for r in observed}
    phase_note(root,'initial ten-map screen',table=str(args.initial_pool/'iteration_table.md'),
        points=initial['equal_map'],frozen_recipe=True,game_cap=8,dashboard_pid=33732)
    next_iteration=1
    resumed=(root/'loop_state.json').exists() and args.resume
    if resumed:
        saved=resume_loop_state(args,protocol)
        parent=Path(saved['parent']);latest_pool=Path(saved['latest_pool']);selected=saved['selected']
        all_correction_folders=saved['all_correction_folders'];iterations=saved['iterations']
        best_screen_points=saved['best_screen_points'];candidate_limit=saved['candidate_limit'];next_iteration=saved['next_iteration']
        for it in iterations:
            for r in it.get('collection_profiles',[]):map_costs[r['map']]=r['raw'].get('query_median_s') or 0
        phase_note(root,'resumed recorded iteration boundary',next_iteration=next_iteration,checkpoint=str(parent))
    elif initial['winner_screen']:
        if args.resume:
            pools=[root/('confirmation_0_'+str(args.seed+offset)) for offset in (1,2)]
            tables=[json.loads((p/'iteration_table.json').read_text()) for p in pools]
            initial_confirmation_receipts(args,pools)
            confirmation=dict(passed=all(t['winner_screen'] for t in tables),pools=pools,tables=tables)
        else:confirmation=confirm(args,parent,initial,0)
        latest_pool=confirmation['pools'][-1]
        if confirmation['passed']:
            if not args.resume:audit_selected(args,[args.initial_pool,*confirmation['pools']])
            selected.append(dict(checkpoint=str(parent),pools=list(map(str,confirmation['pools']))))
            phase_note(root,'initial candidate confirmed',checkpoint=str(parent),
                confirmation_points=[t['equal_map'] for t in confirmation['tables']])
        else:phase_note(root,'initial candidate remains provisional',confirmation_points=[t['equal_map'] for t in confirmation['tables']])
    # The original57/44 model remains an explicit experimental starting point
    # if confirmation fails. It is not called a selected parent or a default.
    write_loop_state(args,protocol,parent,latest_pool,selected,all_correction_folders,iterations,best_screen_points,candidate_limit,next_iteration)
    iteration=next_iteration
    while True:
        if args.stop_after_iteration is not None and iteration>args.stop_after_iteration:
            phase_note(root,'requested iteration already completed; no further iteration started',iteration=iteration-1)
            return dict(status='requested_iteration_complete',iterations=iterations,selected=selected)
        if (root/'STOP').exists():raise InterruptedError('User stop marker')
        check_recipe(protocol);collection_started=time.perf_counter();folders=[];reports=[]
        collection=root/('iteration_'+str(iteration)+'_collection')
        phase_note(root,'collecting corrections',iteration=iteration,student_pool=str(latest_pool),
            workers=args.workers,target_new=protocol['minimum_new_accepted'])
        reused_collection=collection.exists()
        result=completed_collection(collection) if reused_collection else collect_parallel(
            latest_pool,args.dataset,collection,args.sweep_dll,args.cache,args.workers,args.attempts,
            protocol['query_wall_s'],candidate_limit=candidate_limit,map_costs=map_costs,exclude_folders=all_correction_folders)
        folders.extend(result['corrections']);reports.extend(result['maps'])
        names=json.loads((args.dataset/'manifest.json').read_text())['maps']
        recovered=recover_remaining_collections(root,iteration)
        for number,remaining,batch in recovered:
            folders.extend(batch['corrections']);reports.extend(batch['maps'])
            phase_note(root,'recovered closed remaining-state collection',iteration=iteration,
                batch=number,path=str(remaining))
        counts=distinct_correction_counts(folders,names,all_correction_folders)
        # Additional rounds collect new states from the same student checkpoint;
        # keep the causal filter and horizons unchanged. Never fit below quota.
        supplemental=0;coverage_pass=max((n for n,_,_ in recovered),default=0);state_rollout_s=0.
        carried_extra_s=sum(batch['elapsed_wall_s'] for _,_,batch in recovered);fresh_round=False;coverage_pool=latest_pool
        while sum(counts.values())<protocol['minimum_new_accepted'] or any(v<protocol['minimum_per_map'] for v in counts.values()):
            if (root/'STOP').exists():raise InterruptedError('Maintenance or user stop marker; closed collection retained')
            needy=[n for n,v in counts.items() if v<protocol['minimum_per_map']]
            if not needy:needy=list(counts)
            pool=root/('iteration_'+str(iteration)+'_extra_states_'+str(supplemental+1))
            closed_extra=root/(pool.name+'_corrections')
            if closed_extra.exists() or fresh_round:
                supplemental+=1
                if supplemental>8:
                    phase_note(root,'blocked: insufficient distinct accepted states',iteration=iteration,counts=counts,
                        reason='Eight additional student state pools exhausted; fixed filter and distinct quotas unchanged.')
                    return dict(status='blocked',iterations=iterations,counts=counts,selected=selected)
                if closed_extra.exists():
                    extra=completed_collection(closed_extra)
                    native=json.loads((pool/'summary.json').read_text())
                    if not native.get('complete'):raise ValueError('Incomplete supplemental student pool')
                    old_native_s=native['elapsed_wall_s'];state_rollout_s+=old_native_s
                    carried_extra_s+=extra['elapsed_wall_s']+old_native_s
                    phase_note(root,'reused closed supplemental collection',iteration=iteration,distinct_counts=counts)
                else:
                    seed=args.seed+1000+iteration*100+supplemental
                    checkpoint=Path(json.loads((latest_pool/'prepared.json').read_text())['fast_checkpoint'])
                    if pool.exists():
                        old_native_s=reuse_student_pool(pool,checkpoint,seed,needy)
                        state_rollout_s+=old_native_s;carried_extra_s+=old_native_s
                        phase_note(root,'reused completed student native pool',iteration=iteration,path=str(pool),maps=needy)
                    else:
                        state_rollout_s+=execute_phase(root,'extra_'+pool.name,pool_command(args,checkpoint,pool,seed,needy,student_only=True))
                    phase_note(root,'collecting supplemental corrections',iteration=iteration,workers=min(args.workers,len(needy)),maps=needy)
                    extra=collect_parallel(pool,args.dataset,closed_extra,args.sweep_dll,args.cache,
                        args.workers,args.attempts,protocol['query_wall_s'],maps=needy,candidate_limit=candidate_limit,map_costs=map_costs,
                        exclude_folders=all_correction_folders+folders)
                # Partial historic supplemental pools need not contain all needy maps.
                coverage_pool=pool if set(needy).issubset(json.loads((pool/'prepared.json').read_text())['maps']) else latest_pool
            else:
                coverage_pass+=1
                remaining=root/('iteration_'+str(iteration)+'_remaining_states_'+str(coverage_pass)+'_corrections')
                phase_note(root,'collecting previously unqueried student states',iteration=iteration,distinct_counts=counts,
                    workers=min(args.workers,len(needy)),maps=needy,student_pool=str(coverage_pool),progress_directory=str(remaining/'workers'))
                reused_remaining=remaining.exists()
                extra=completed_collection(remaining) if reused_remaining else collect_parallel(
                    coverage_pool,args.dataset,remaining,args.sweep_dll,args.cache,args.workers,args.attempts,
                    protocol['query_wall_s'],maps=needy,candidate_limit=candidate_limit,map_costs=map_costs,
                    exclude_folders=all_correction_folders+folders)
                if reused_remaining:carried_extra_s+=extra['elapsed_wall_s']
            folders.extend(extra['corrections']);reports.extend(extra['maps'])
            counts=distinct_correction_counts(folders,names,all_correction_folders)
            fresh_round=any(r['raw']['attempts']<args.attempts and counts[r['map']]<protocol['minimum_per_map'] for r in extra['maps'])
            phase_note(root,'additional distinct correction coverage',iteration=iteration,counts=counts)
        if (root/'STOP').exists():raise InterruptedError('Maintenance or user stop marker; closed collection retained')
        new_data=root/('iteration_'+str(iteration)+'_new_corrections')
        new=merge_corrections(folders,args.dataset,new_data,protocol['minimum_new_accepted'],protocol['minimum_per_map'],prior=all_correction_folders)
        if not new['quota_met']:raise ValueError('Correction quota failed after merge')
        carried_collection_s=result['elapsed_wall_s'] if reused_collection else 0.
        collection_s=time.perf_counter()-collection_started+carried_collection_s+carried_extra_s
        all_correction_folders.extend(folders)
        training_data=root/('iteration_'+str(iteration)+'_training_corrections')
        accumulated=merge_corrections(all_correction_folders,args.dataset,training_data)
        fit=root/('iteration_'+str(iteration)+'_fit')
        phase_note(root,'fitting fixed recipe',iteration=iteration,new_accepted=new['accepted'],
            cumulative_accepted=accumulated['accepted'],label_counts=new['label_counts'],collection_wall_s=collection_s)
        fit_s=execute_phase(root,'fit_'+str(iteration),[sys.executable,'-u','-m','nav.marble_dynamics.fast_nav',
            'fit','--dataset',str(args.dataset),'--advice',str(training_data),'--initial',str(parent),
            '--output',str(fit),'--updates',str(protocol['fit_updates']),'--wall-s','900',
            '--preserve-base','--linear-axes','--advice-actions','--defer-advice-replay'])
        checkpoint=fit/'final.pth';check_recipe(protocol)
        pool=root/('iteration_'+str(iteration)+'_screen');seed=args.seed+iteration*10
        pool_s=execute_phase(root,'pool_'+str(iteration),pool_command(args,checkpoint,pool,seed))
        latest_pool=pool;table=pool_table(pool,args.teacher)
        record=dict(iteration=iteration,checkpoint=str(checkpoint),new_accepted=new['accepted'],
            cumulative_accepted=accumulated['accepted'],label_counts=new['label_counts'],table=table,
            collection_wall_s=collection_s,fit_wall_s=fit_s,pool_wall_s=pool_s,
            carried_collection_wall_s=carried_collection_s,supplemental_student_pool_wall_s=state_rollout_s,
            carried_supplemental_wall_s=carried_extra_s,qualifying_scaled_loop=True,
            collection_profiles=reports,selected=False)
        if table['winner_screen']:
            confirmation=confirm(args,checkpoint,table,iteration)
            latest_pool=confirmation['pools'][-1]
            record['confirmation']={k:v for k,v in confirmation.items() if k!='pools'}
            if confirmation['passed']:
                parent=checkpoint;record['selected']=True
                best_screen_points=table['equal_map']['student_points']
                selected.append(dict(checkpoint=str(parent),pools=list(map(str,confirmation['pools']))))
                teacher_pools=[]
                if all(t['student_to_planner']>=protocol['stop_teacher_ratio'] for t in confirmation['tables']):
                    matched=[]
                    for confirmation_pool in confirmation['pools']:
                        start_protocol=json.loads((confirmation_pool/'prepared.json').read_text())
                        actual_seed=start_protocol['seeds'][0]
                        teacher_pool=root/('iteration_'+str(iteration)+'_teacher_'+str(actual_seed))
                        teacher_manifest=Path(start_protocol['start_seed_manifest']) if start_protocol.get('start_seed_manifest') else None
                        execute_phase(root,'teacher_'+str(actual_seed),pool_command(args,None,teacher_pool,actual_seed,teacher_only=True,
                            random_start=start_protocol.get('random_starts',False),start_manifest=teacher_manifest))
                        teacher_pools.append(teacher_pool);matched.append(pool_table(confirmation_pool,teacher_pool))
                    record['matched_teacher_confirmation']=matched
                    teacher_gate=all(t['student_to_planner']>=protocol['stop_teacher_ratio'] for t in matched)
                else:teacher_gate=False
                audit_selected(args,[pool,*confirmation['pools'],*teacher_pools],training_data)
                if teacher_gate:
                    iterations.append(record);atomic_json(root/'iterations.json',iterations)
                    phase_note(root,'80 percent teacher gate reached; stopped for RL handoff',iteration=iteration,
                        checkpoint=str(parent),ratios=[t['student_to_planner'] for t in matched])
                    return dict(status='RL_handoff_ready',iterations=iterations,selected=selected)
        for r in reports:map_costs[r['map']]=r['raw'].get('query_median_s') or 0
        medians=[r['raw'].get('query_median_s') for r in reports if r['raw'].get('query_median_s') is not None]
        slow={r['map'] for r in reports if (r['raw'].get('query_median_s') or 0)>.5}
        for name in slow:candidate_limit[name]=max(3,candidate_limit.get(name,6)-1)
        if slow:
            record['next_collection_candidate_limit']=candidate_limit.copy()
            record['candidate_reduction_reason']='Measured map median search exceeds500ms; fewer candidates, identical horizons and causal filter.'
        iterations.append(record);atomic_json(root/'iterations.json',iterations)
        write_loop_state(args,protocol,parent,latest_pool,selected,all_correction_folders,iterations,best_screen_points,candidate_limit,iteration+1)
        phase_note(root,'iteration complete',iteration=iteration,table=str(pool/'iteration_table.md'),
            points=table['equal_map'],selected=record['selected'],label_counts=new['label_counts'],
            wall_s=dict(collection=collection_s,fit=fit_s,pool=pool_s))
        gains=gain_summary(protocol,initial,iterations);atomic_json(root/'gain_summary.json',gains)
        if args.stop_after_iteration is not None and iteration>=args.stop_after_iteration:
            phase_note(root,'requested iteration completed; fixed-recipe loop stopped',iteration=iteration,
                selected=selected,recipe_changed=False,next_iteration_started=False)
            return dict(status='requested_iteration_complete',iterations=iterations,selected=selected)
        if no_gain_stop(protocol,initial,iterations):
            phase_note(root,'four scaled iterations without development gain; stopped for phase report',selected=selected,
                recipe_changed=False,original_three_map_reference=protocol['development_reference'])
            return dict(status='no_gain_after_scaled_iterations',iterations=iterations,selected=selected)
        if gains['qualifying_scaled_loops']>=protocol['minimum_scaled_loops']:
            phase_note(root,'data scaling shows gain; continue frozen recipe',completed_iterations=iteration,gain_summary=gains)
        iteration+=1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--initial-pool',type=Path,required=True);parser.add_argument('--initial-checkpoint',type=Path,required=True)
    parser.add_argument('--initial-corrections',type=Path)
    parser.add_argument('--teacher',type=Path,required=True);parser.add_argument('--sweep-dll',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True);parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--attempts',type=int,default=1200);parser.add_argument('--port',type=int,default=9054)
    parser.add_argument('--seed',type=int,default=1681124)
    parser.add_argument('--collection-profile',type=Path,help='Prior measured query timings for candidate-count reduction')
    parser.add_argument('--collect-only',action='store_true',help='Reusable collection diagnostic; no fit or native evaluation')
    parser.add_argument('--resume',action='store_true',help='Resume a recorded boundary or the completed initial collection; never overwrite partial evidence')
    parser.add_argument('--stop-after-iteration',type=int,help='Complete this iteration, including its pool, then stop before collecting the next')
    parser.add_argument('--confirmation-start-manifests',nargs=2,type=Path,help='Two layout-preflight manifests for genuinely different native confirmation starts')
    args=parser.parse_args()
    if not 1<=args.workers<=8 or args.attempts<(1 if args.collect_only else 300):parser.error('Workers1–8 and at least300 attempted states/map for the scaled phase')
    for key in ('output','dataset','initial_pool','initial_checkpoint','teacher','sweep_dll','cache'):
        setattr(args,key,getattr(args,key).resolve())
    if args.initial_corrections:args.initial_corrections=args.initial_corrections.resolve()
    try:
        result=collect_parallel(args.initial_pool,args.dataset,args.output,args.sweep_dll,args.cache,args.workers,args.attempts) if args.collect_only else drive(args)
        atomic_json(args.output/'phase_result.json',result)
    except Exception as exc:
        phase_note(args.output,'stopped with preserved failure',reason=repr(exc))
        atomic_json(args.output/'phase_failure.json',dict(reason=repr(exc)));raise


if __name__=='__main__':main()
