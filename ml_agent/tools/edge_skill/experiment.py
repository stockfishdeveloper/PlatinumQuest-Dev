"""Single entry point for the overnight edge-skill campaign and its evidence."""
import argparse
import copy
import gzip
import json
import time
from collections import Counter

import numpy as np

from nav.marble_dynamics.record import game,command
from nav.native.forecast import predict_many
from .common import ART,NIGHT,RUNTIME,engine_for,load_json,write_json,sha
from .maps import diverse_maps,transform,real_fixtures,sloped_maps
from .native_lab import export_map,prepare,place
from .execution import Policy,initial,observed,legacy_case,rotation,make_case,solve,forecast,assess,checked_forecast,certify,resume_static,local,perturb,make_executor
from .audit import audit_native_contract,audit_forecast_contract


def dump(path,row):
    with path.open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')


def rows(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def make_suites():
    manifest=load_json(NIGHT/'maps.json');result={s:[] for s in ('train','dev','test')}
    for entry in manifest['maps']:
        for cell in entry['cells']:
            for j in range(18):
                kind='roll' if cell['family']=='low' else ('air_rising' if j%6==4 else 'air_descending' if j%6==5 else 'ground')
                c=make_case(cell,j,839121+int(entry['name'][-2:])*100000+cell['index']*100+j,start_kind=kind)
                result[cell['split']].append(c)
    for split,cases in result.items():
        path=NIGHT/(split+'_cases.json')
        if path.exists():
            assert load_json(path)==cases,'Frozen suite differs'
        else:write_json(path,cases)
    write_json(NIGHT/'suite_manifest.json',dict(seed=839121,counts={k:len(v) for k,v in result.items()},
               selection='All geometry cells; 18 varied starts per cell. Test groups 6/7 unopened until freeze.',
               primary='Ground/airborne jump contacts on high lips. Low lip rolling reported separately.',
               success='Audited lip impulse, no wall assist, supported goal/exit for 0.512s, supported braking continuation for 0.384s.',
               files={k:sha(NIGHT/(k+'_cases.json')) for k in result} ))


def collect(a):
    cases=load_json(NIGHT/a.input) if a.input else load_json(NIGHT/(a.split+'_cases.json'))
    # Stride geometry rather than spending the first probe on one geometry's neighboring starts.
    if not a.input:cases=sorted(cases,key=lambda c:((int(c['id'].split('_')[-1])-c['cell']['index'])%18,c['cell']['index'],c['cell']['map']))
    if a.offset:cases=cases[a.offset:]
    if a.limit:cases=cases[:a.limit]
    out=NIGHT/a.out;done={r['case']['id'] for r in rows(out)} if out.exists() else set()
    engines={};count=0;begin=time.perf_counter();warm={}
    for path in a.warm:
        for r in rows(NIGHT/path):warm.setdefault(r['case']['id'],[]).append(r['params'])
    for case in cases:
        if case['id'] in done:continue
        if a.stable:case['stable_inputs']=True
        name=case['cell']['map']
        if name not in engines:engines[name]=engine_for(name)
        best=solve(engines[name],case,budget=a.budget,seed=case['seed'],initial_params=case.get('teacher_params',[])+warm.get(case['id'],[]),robust_samples=a.robust)
        row=dict(case=case,params=best['params'],alternatives=best['alternatives'],result=best['result'],stats=best['stats'])
        if 'robustness' in best:row['robustness']=best['robustness']
        dump(out,row);count+=1
        print(case['id'],best['result']['success'],round(best['result']['score'],3),round(best['stats']['wall_s'],3),flush=True)
        write_json(out.with_suffix('.progress.json'),dict(complete=False,new_cases=count,wall_s=time.perf_counter()-begin,last=case['id']))
    rr=rows(out);write_json(out.with_suffix('.progress.json'),dict(complete=True,cases=len(rr),success=sum(r['result']['success'] for r in rr),wall_s=time.perf_counter()-begin))


def anchors(a):
    """Pre-contact states on known successful trajectories, not arbitrary airborne guesses."""
    source=rows(NIGHT/a.input);cases=[];engines={};checks=[];labels=[]
    for row in source:
        if not row['result']['success'] or row['case']['cell']['family']=='low':continue
        case=row['case'];name=case['cell']['map']
        if name not in engines:engines[name]=engine_for(name)
        d=initial(case);r=checked_forecast(engines[name],case,row['params'],d)
        if not r['result']['success']:continue
        hit=r['result']['lip_tick'];roll=r['rollout'];eligible=[]
        # Callable policies retain the trajectory but not each intermediate
        # bridge table. Compile their recorded actions once to recover exact
        # queued moves for resumable initial states.
        replay=predict_many(engines[name],d,[roll['proposed']])[0]
        assert np.max(np.abs(replay['native_states'].rows[:,:9]-roll['native_states'].rows[:,:9]))<1e-8
        roll=replay
        for k in range(1,(hit-2)//4):
            s=roll['states'][k];p=local(case['cell'],s['position'])
            if p[0]<-.35 and p[2]>.27:eligible.append(k)
        if not eligible:continue
        ids=sorted(set(eligible[i] for i in np.linspace(0,len(eligible)-1,min(5,len(eligible)),dtype=int)))
        for k in ids:
            dd=resume_static(d,roll,k);c=observed(case,dd);c['initial']=dd
            c['airborne']=True;c['jump_committed']=True;c['start_kind']='air_rising' if c['velocity'][2]>.6 else 'air_descending'
            c['id']=case['id']+'_anchor_%03d'%k;c['teacher_params']=[row['params']];c['parent_id']=case['id'];c['seed']+=k*37
            proof=checked_forecast(engines[name],c,row['params'],dd)
            c['continuation_proof']=bool(proof['result']['success'])
            cases.append(c);checks.append(dict(id=c['id'],kind=c['start_kind'],continuation_success=c['continuation_proof']))
            labels.append(dict(case=c,params=row['params'],alternatives=[row['params']],result=proof['result'],stats=dict(source='verified_continuation')))
    write_json(NIGHT/a.out,cases);write_json((NIGHT/a.out).with_suffix('.proof.json'),dict(cases=checks,counts=dict(Counter((c['start_kind'] for c in cases)))))
    target=(NIGHT/a.out).with_suffix('.jsonl')
    if target.exists():raise FileExistsError(target)
    for label in labels:dump(target,label)
    print('anchors',len(cases),'proven continuations',sum(c['continuation_proof'] for c in cases),flush=True)


def stress(a):
    source=rows(NIGHT/a.input);out=NIGHT/a.out;engines={}
    done={r['id'] for r in rows(out)} if out.exists() else set()
    for row in source:
        c=row['case']
        if not row['result']['success'] or c['id'] in done:continue
        name=c['cell']['map']
        if name not in engines:engines[name]=engine_for(name)
        tests=[certify(engines[name],c,row['params'],samples=a.samples,size=size,seed=c['seed']+93601) for size in a.sizes]
        dump(out,dict(id=c['id'],start_kind=c['start_kind'],family=c['cell']['family'],tests=tests))
        print(c['id'],[(t['size'],t['successes'],t['samples']) for t in tests],flush=True)


def transfer(a):
    """Separate unchanged-parameter tolerance from planning at the observed start."""
    source=rows(NIGHT/a.input);out=NIGHT/a.out;engines={}
    done={r['case']['id'] for r in rows(out)} if out.exists() else set()
    for row in source:
        case=row['case']
        if not row['result']['success'] or case['cell']['family']=='low':continue
        name=case['cell']['map']
        if name not in engines:engines[name]=engine_for(name)
        engine=engines[name];d=initial(case)
        for j in range(a.samples):
            ident=case['id']+'_perturb_%03d'%j
            if ident in done:continue
            c,dd=perturb(case,d,case['seed']+126773+j*7919,a.size)
            c.update(id=ident,parent_id=case['id'],initial=dd,seed=case['seed']+126773+j*7919)
            fixed=checked_forecast(engine,c,row['params'],dd)
            fresh=solve(engine,c,dd,budget=a.budget,seed=c['seed'],initial_params=[row['params']])
            dump(out,dict(case=c,params=fresh['params'],alternatives=fresh['alternatives'],result=fresh['result'],stats=fresh['stats'],
                          fixed=fixed['result'],perturbation_size=a.size))
            print(ident,'fixed',fixed['result']['success'],'fresh',fresh['result']['success'],flush=True)


def evaluate(a):
    from .learn import Proposer
    model=Proposer(NIGHT/a.model);cases=load_json(NIGHT/a.input) if a.input else load_json(NIGHT/(a.split+'_cases.json'))
    if a.offset:cases=cases[a.offset:]
    if a.limit:cases=cases[:a.limit]
    out=NIGHT/a.out;done={r['case']['id'] for r in rows(out)} if out.exists() else set();engines={}
    for c in cases:
        if c['id'] in done:continue
        if a.stable or model.stable_inputs:c['stable_inputs']=True
        name=c['cell']['map']
        if name not in engines:engines[name]=engine_for(name)
        begin=time.perf_counter()
        if a.method=='analytic':best=solve(engines[name],c,budget=a.budget,seed=c['seed'])
        else:best=model.propose(engines[name],c,budget=a.budget,seed=c['seed'],neural=a.method!='retrieval',retrieval=a.method!='neural')
        elapsed=time.perf_counter()-begin
        dump(out,dict(case=c,params=best['params'],alternatives=best['alternatives'],result=best['result'],stats=best['stats'],
                      method=a.method,model=a.model,wall_s=elapsed))
        print(c['id'],best['result']['success'],round(elapsed*1000,1),'ms',flush=True)
    rr=rows(out);write_json(out.with_suffix('.summary.json'),dict(complete=True,cases=len(rr),success=sum(r['result']['success'] for r in rr),
                        time_ms=np.percentile([r['wall_s']*1000 for r in rr],[50,95,99]).tolist()))


def native(a):
    source=rows(NIGHT/a.input)
    if a.limit:source=source[:a.limit]
    out=NIGHT/a.out;out.mkdir(exist_ok=True);results=load_json(out/'summary.json')['cases'] if (out/'summary.json').exists() else []
    completed={r['id'] for r in results}
    proposer=None
    if a.model:
        from .learn import Proposer
        proposer=Proposer(NIGHT/a.model)
    batches=[]
    for name in sorted({r['case']['cell']['map'] for r in source}):
        remaining=[r for r in source if r['case']['cell']['map']==name and r['case']['id'] not in completed]
        size=12 if name.endswith('_Hunt') else max(1,len(remaining))
        batches.extend((name,remaining[j:j+size]) for j in range(0,len(remaining),size))
    for name,remaining in batches:
        engine=engine_for(name)
        with game(name,a.port,obs_ms=16,seed=116031,process_cap=2,watch=False) as (env,proc):
            prepare(env)
            for row in remaining:
                original=row['case'];cell=original['cell'];rot=rotation(cell)
                command(env,'DYNAMICSDETAIL 1','dynamics_detail')
                raw=RUNTIME/('overnight_'+original['id']+'.jsonl')
                command(env,'DYNAMICSRECORD '+raw.as_posix(),'dynamics_record')
                try:d=place(env,transform(cell,original['position']),(rot@original['velocity']).tolist(),(rot@original['spin']).tolist())
                except RuntimeError as exc:
                    command(env,'DYNAMICSRECORD off','dynamics_record');command(env,'DYNAMICSDETAIL 0','dynamics_detail')
                    setup=dict(id=original['id'],start_kind=original['start_kind'],setup_error=str(exc),admission=None,
                               result=dict(success=False,reason='invalid_setup'),execution=a.execution,model=a.model)
                    results.append(setup);write_json(out/(original['id']+'.setup.json'),dict(case=original,comparison=setup))
                    write_json(out/'summary.json',dict(complete=False,cases=results));print(original['id'],'setup failed',str(exc),flush=True)
                    continue
                case=observed(original,d);t0=d['native']['frame_clock_ms']
                if proposer is not None:case['stable_inputs']=proposer.stable_inputs
                prefix=[];prefix_plan=None
                if a.handoff is not None:
                    if proposer is None:raise ValueError('Native handoff requires a student checkpoint')
                    prefix_plan=solve(engine,case,d,budget=a.teacher_budget,seed=case['seed'],initial_params=[row['params']])
                    if not prefix_plan['result']['success']:
                        command(env,'DYNAMICSRECORD off','dynamics_record');command(env,'DYNAMICSDETAIL 0','dynamics_detail')
                        setup=dict(id=original['id'],start_kind=original['start_kind'],admission=None,result=dict(success=False,reason='prefix_declined'))
                        results.append(setup);write_json(out/'summary.json',dict(complete=False,cases=results));continue
                    eligible=[]
                    for k in range(1,max(1,(prefix_plan['result']['lip_tick']-2)//4)):
                        p=local(cell,prefix_plan['rollout']['states'][k]['position'])
                        if p[0]<-.35 and p[2]>case.get('start_floor',0.)+.29:eligible.append(k)
                    count=eligible[min(len(eligible)-1,int(a.handoff*len(eligible)))] if eligible else 0
                    prefix_policy=make_executor(prefix_plan,engine,d)
                    for k in range(count):
                        actual=copy.deepcopy(env.msg.dynamics);action=prefix_policy(actual['native']['state'],k)
                        predicted=predict_many(engine,actual,[[action]])[0]
                        _,outcome=env.step(action[:6],repeat=1,use_pow=0,use_blast=0)
                        if any(outcome[x] for x in ('fell','round_ended','reconnected')):raise RuntimeError('Native handoff prefix terminated')
                        prefix_policy.observe_ticks(predicted['native_states']);prefix.append(action)
                    d=copy.deepcopy(env.msg.dynamics);case=observed(case,d);t0=d['native']['frame_clock_ms']
                    case['airborne']=case['position'][2]>case.get('start_floor',0.)+.25
                    case['jump_committed']=bool(case.get('jump_committed') or any(x[4] for x in prefix))
                    if case['airborne']:case['start_kind']='air_rising' if case['velocity'][2]>.6 else 'air_descending'
                if proposer is not None:candidate=proposer.propose(engine,case,d,budget=a.budget,seed=case['seed'])
                else:candidate=solve(engine,case,d,budget=a.budget,seed=case['seed'],initial_params=[row['params']]) if a.resolve else checked_forecast(engine,case,row['params'],d)
                root_candidate=candidate;root_plan=candidate['params'];plan=root_plan;prediction=candidate['rollout']
                policy=make_executor(candidate,engine,d) if candidate['result']['success'] else Policy(case,plan,engine=engine,d=d)
                decisions=[];deviations=[];finished_at=None;refreshes=[];active=prediction;active_case=case;active_index=0
                retained=a.execution=='retained' and candidate['result']['success']
                for k in range(policy.steps):
                    actual=copy.deepcopy(env.msg.dynamics)
                    if retained:
                        expected=active['states'][active_index];state=actual['native']['state']
                        pe=float(np.linalg.norm(np.asarray(state['position'])-expected['position']))
                        ve=float(np.linalg.norm(np.asarray(state['velocity'])-expected['velocity']))
                        # Do not replace an independently checked command merely
                        # because feedback differs at floating-point noise scale.
                        # Before contact, a material drift triggers a new checked
                        # plan from the observed state, including queued inputs.
                        contact_tick=candidate['result']['lip_tick']
                        if (pe>.003 or ve>.03) and active_index*4<contact_tick:
                            fresh_case=observed(active_case,actual)
                            fresh_case['airborne']=fresh_case['position'][2]>fresh_case.get('start_floor',0.)+.25
                            fresh_case['jump_committed']=bool(fresh_case.get('jump_committed') or any(x[4] for x in active['proposed'][:active_index]))
                            fresh=solve(engine,fresh_case,actual,budget=96,seed=case['seed']+k,initial_params=[plan])
                            refreshes.append(dict(reply=k,position_error=pe,velocity_error=ve,accepted=fresh['result']['success'],params=fresh['params'],stats=fresh['stats']))
                            if fresh['result']['success']:
                                candidate=fresh;active=fresh['rollout'];active_case=fresh_case;active_index=0;plan=fresh['params']
                        action=active['proposed'][active_index]
                    else:action=policy(actual['native']['state'],k)
                    next_prediction=predict_many(engine,actual,[[action]])[0]
                    _,outcome=env.step(action[:6],repeat=1,use_pow=0,use_blast=0)
                    after=copy.deepcopy(env.msg.dynamics['native']['state'])
                    error=float(np.linalg.norm(np.asarray(after['position'])-next_prediction['states'][-1]['position']))
                    deviations.append(error)
                    if not retained and error<.025:policy.observe_ticks(next_prediction['native_states'])
                    decisions.append(dict(before=actual,action=action,after=after,outcome=outcome))
                    if retained:
                        active_index+=1
                        if active_index>=len(active['proposed']):break
                    if policy.finished and finished_at is None:finished_at=k
                    if (finished_at is not None and k>=finished_at+10) or any(outcome[x] for x in ('fell','round_ended','reconnected')):break
                t1=env.msg.dynamics['native']['frame_clock_ms']
                command(env,'DYNAMICSRECORD off','dynamics_record');command(env,'DYNAMICSDETAIL 0','dynamics_detail')
                records=[json.loads(l) for l in raw.read_text().splitlines() if l.strip()]
                transitions=[r for r in records if r.get('kind')=='transition' and t0<=r['sim_ms']<t1]
                states=[transitions[0]['before']]+[r['after'] for r in transitions]
                nr=np.asarray([[*s['position'],*s['velocity'],*s['spin'],s['radius']] for s in states])
                screen=assess(case,nr);audit=audit_native_contract(case,transitions,(screen['finish_tick']+24) if screen['finish_tick'] is not None else None)
                result=assess(case,nr,contact=audit);result['contact_audit']=audit
                result['resources_valid']=bool(result['resources_valid'] and all(not any(e[0] for e in s['effects']) for s in states)
                                               and all(not v['action'][6] and not v['action'][8] for v in decisions))
                result['success']=bool(result['success'] and result['resources_valid'])
                replay=predict_many(engine,d,[[v['action'] for v in decisions]])[0]
                er=np.linalg.norm(nr[:,:3]-replay['native_states'].rows[:len(nr),:3],axis=1)
                pred_result=assess(case,prediction['native_states'].rows)
                comparison=dict(id=case['id'],start_kind=case.get('start_kind','ground'),result=result,predicted=pred_result,
                                admission=bool(root_candidate['result']['success']),max_one_reply_error=max(deviations),max_replay_error=float(er.max()),
                                wall_stats=root_candidate.get('stats'),fell=any(v['outcome']['fell'] for v in decisions),frames=len(nr)-1,
                                execution=a.execution,refreshes=refreshes,model=a.model,prefix_replies=len(prefix),handoff=a.handoff)
                results.append(comparison)
                payload=dict(case=case,original=original,params=root_plan,initial=d,decisions=decisions,native_ticks=states,comparison=comparison,
                             prefix_actions=prefix,prefix_params=prefix_plan['params'] if prefix_plan is not None else None)
                (out/(case['id']+'.json.gz')).write_bytes(gzip.compress(json.dumps(payload).encode(),mtime=0))
                (out/(case['id']+'.native.jsonl.gz')).write_bytes(gzip.compress(raw.read_bytes(),mtime=0))
                write_json(out/'summary.json',dict(complete=False,cases=results))
                print(case['id'],'native',result['success'],'pred',pred_result['success'],'error',float(er.max()),flush=True)
    write_json(out/'summary.json',dict(complete=True,cases=results))


def main():
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='command',required=True)
    g=sub.add_parser('geometry');g.add_argument('--export',action='store_true');g.add_argument('--port',type=int,default=9580)
    sub.add_parser('real-geometry')
    sg=sub.add_parser('sloped-geometry');sg.add_argument('--export',action='store_true');sg.add_argument('--port',type=int,default=9580)
    s=sub.add_parser('collect');s.add_argument('--split',choices=('train','dev','test'),default='dev');s.add_argument('--budget',type=int,default=192)
    s.add_argument('--out',required=True);s.add_argument('--limit',type=int,default=0);s.add_argument('--offset',type=int,default=0)
    s.add_argument('--input');s.add_argument('--warm',nargs='*',default=[]);s.add_argument('--robust',type=int,default=0);s.add_argument('--stable',action='store_true')
    h=sub.add_parser('anchors');h.add_argument('--input',required=True);h.add_argument('--out',required=True)
    t=sub.add_parser('stress');t.add_argument('--input',required=True);t.add_argument('--out',required=True)
    t.add_argument('--samples',type=int,default=16);t.add_argument('--sizes',type=float,nargs='+',default=[1.,2.,4.])
    f=sub.add_parser('transfer');f.add_argument('--input',required=True);f.add_argument('--out',required=True)
    f.add_argument('--samples',type=int,default=12);f.add_argument('--size',type=float,default=1.);f.add_argument('--budget',type=int,default=96)
    e=sub.add_parser('evaluate');e.add_argument('--model',required=True);e.add_argument('--method',choices=('combined','neural','retrieval','analytic'),default='combined')
    e.add_argument('--split',choices=('train','dev','test'),default='dev');e.add_argument('--input');e.add_argument('--out',required=True)
    e.add_argument('--budget',type=int,default=64);e.add_argument('--limit',type=int,default=0);e.add_argument('--offset',type=int,default=0)
    e.add_argument('--stable',action='store_true')
    n=sub.add_parser('native');n.add_argument('--input',required=True);n.add_argument('--out',required=True);n.add_argument('--resolve',action='store_true')
    n.add_argument('--budget',type=int,default=192);n.add_argument('--limit',type=int,default=0);n.add_argument('--port',type=int,default=9580)
    n.add_argument('--execution',choices=('policy','retained'),default='policy');n.add_argument('--model')
    n.add_argument('--handoff',type=float,help='Transfer a real mid-jump prefix to the student at this fraction of its pre-contact air states.')
    n.add_argument('--teacher-budget',type=int,default=384)
    a=ap.parse_args()
    if a.command=='geometry':
        m=diverse_maps()
        if a.export:
            for entry in m['maps']:
                if not (NIGHT/(entry['name']+'.initial.json')).exists():export_map(entry,a.port)
        make_suites()
    elif a.command=='collect':collect(a)
    elif a.command=='native':native(a)
    elif a.command=='anchors':anchors(a)
    elif a.command=='stress':stress(a)
    elif a.command=='transfer':transfer(a)
    elif a.command=='evaluate':evaluate(a)
    elif a.command=='real-geometry':real_fixtures()
    elif a.command=='sloped-geometry':
        from scipy.spatial import cKDTree
        manifest=sloped_maps();cases={s:[] for s in ('train','dev','test')};checks=[]
        for group,entry in enumerate(manifest['maps']):
            if a.export:
                if not (NIGHT/(entry['name']+'.initial.json')).exists():export_map(entry,a.port)
                engine=engine_for(entry['name']);points=np.asarray([p for c in entry['cells'] for p in c['geometry_corners']])
                error=cKDTree(engine.mesh.triangles.reshape(-1,3)).query(points)[0]
                checks.append(dict(map=entry['name'],max_corner_error=float(error.max())))
                assert error.max()<.002,'Tilted native geometry differs from its declared frame'
            for c in entry['cells']:
                for j in range(18):
                    kind='air_rising' if j%6==4 else 'air_descending' if j%6==5 else 'ground'
                    case=make_case(c,j,6110382+group*100000+c['index']*100+j,kind);case['stable_inputs']=True;cases[c['split']].append(case)
        for split,cc in cases.items():
            path=NIGHT/('sloped_'+split+'_cases.json')
            if path.exists():assert load_json(path)==cc
            else:write_json(path,cc)
        if checks:write_json(NIGHT/'sloped_geometry_conformance.json',checks)


if __name__=='__main__':main()
