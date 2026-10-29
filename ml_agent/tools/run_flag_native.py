"""Matched ten-map native scoring of planner command-line flags (the overnight harness, generalised; 2026-10-27).

Both arms run tools/native_worker_bench.py with the same workload as the other native campaigns (two games, eight candidate
workers each, fixed .172 s allowance, order gate .35); the candidate appends --candidate-flag values, the baseline appends
--baseline-flag values. Exploratory seed first; the fresh seed runs only when the exploratory equal-map mean improves.
Audits per map: layout, initial motion and the actual GO observation must match; cross-arm canonical source hashes and
the workload must match. Waits for --after status files to read stage 'complete' (one campaign at a time).
usage (repo root, PYTHONPATH=ml_agent):
  python ml_agent/tools/run_flag_native.py --label NAME --seeds S1 S2 --candidate-flag=--x
      [--baseline-flag=--y] [--after other_native_status.json ...] [--require-global module.ATTR=value ...]
"""
import argparse,concurrent.futures,gzip,json,os
from pathlib import Path
import subprocess,sys,time
from tools.traversal_results import native_pair,start_motion
from tools.run_traversal_screen import MAPS

ROOT=Path('ml_agent/dynamics_reports/overnight_20261026')   # campaign results, plans and status files (gitignored)


def save(name,value):
    from nav.collision.mesh import atomic_json
    atomic_json(ROOT/name,value)


def first_go(path):
    with gzip.open(path,'rt') as f:
        for line in f:
            context=json.loads(line)['before']
            if context['native']['state']['mode']==0:return start_motion(context)
    raise RuntimeError('Missing actual GO observation')


def lead_budget(lead):return min(2.,max(.05,lead*.064-.02))   # tools/native_worker_bench.py's search budget for --fixed-lead


def run_pair(a,label,seed,index):
    from nav.marble_dynamics import realtime as rt
    dirs=[ROOT/(label+'_'+arm) for arm in ('baseline','candidate')]
    if any(d.exists() for d in dirs):raise FileExistsError('Existing scoring evidence: '+label)
    def run(i):
        port=a.port_base+4*(i if index==0 else 1-i)   # the second seed swaps the arms' ports
        cmd=[sys.executable,'-u',a.bench,
             '--candidate-workers',str(a.workers_per_game),'--fixed-lead',str(a.fixed_lead),'--order-min-left-s',str(.35),
             '--maps',*a.maps,'--seeds',str(seed),'--random-starts',*rt.FLAGS[2:],
             '--workers','1','--round-wall-limit-s','1200','--port',str(port),'--output',str(dirs[i])]
        cmd+=list(a.candidate_flag if i else a.baseline_flag)
        with (ROOT/(label+'_'+str(i)+'.log')).open('w') as out,(ROOT/(label+'_'+str(i)+'.err')).open('w') as err:
            return subprocess.run(cmd,env=dict(os.environ,PYTHONPATH='ml_agent'),stdout=out,stderr=err).returncode
    checked={}
    def audit_available():
        for name in a.maps:
            if name in checked:continue
            folders=[d/name/str(seed)/'physics_plan/1' for d in dirs]
            if not all((f/'report.json').exists() for f in folders):continue
            reports=[json.loads((f/'report.json').read_text()) for f in folders]
            checks=dict(layout=reports[0]['start_layout']==reports[1]['start_layout'],
                        initial=start_motion(reports[0]['start_context'])==start_motion(reports[1]['start_context']),
                        actual_go=first_go(folders[0]/'decisions.jsonl.gz')==first_go(folders[1]/'decisions.jsonl.gz'))
            checked[name]=dict(passed=all(checks.values()),checks=checks,points=[r['actual_points'] for r in reports])
            save(label+'_motion_audit.json',checked);print(name,json.dumps(checked[name]),flush=True)
            if not checked[name]['passed']:
                for d in dirs:(d/'STOP').write_text('Failed matched initial/GO/layout audit; retain evidence.')
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        futures=[pool.submit(run,i) for i in (0,1)]
        while not all(f.done() for f in futures):
            audit_available()
            if any(f.done() and f.result()!=0 for f in futures):
                for d in dirs:
                    if d.exists():(d/'STOP').write_text('Other arm failed; retain invalid partials.')
            time.sleep(5)
        exits=[f.result() for f in futures]
    if any(exits):raise RuntimeError('Native batch failed: '+str(exits))
    audit_available()
    if len(checked)!=len(a.maps) or not all(x['passed'] for x in checked.values()):raise RuntimeError('Incomplete motion audit')
    prepared=[json.loads((d/'prepared.json').read_text()) for d in dirs]
    if prepared[0]['source_hashes']!=prepared[1]['source_hashes']:raise RuntimeError('Cross-arm canonical source mismatch')
    worker=[json.loads((d/'wrapper_protocol.json').read_text()) for d in dirs]
    if any(p['candidate_workers']!=a.workers_per_game or abs(p['budget_s']-lead_budget(a.fixed_lead))>1e-9 or p['order_min_left_s']!=.35 for p in worker):
        raise RuntimeError('Workload mismatch')
    for want in a.require_global:
        # module.ATTR=value: the candidate's recorded runtime global must equal value (the baseline's must differ)
        key,value=want.split('=',1);module,attr=key.rsplit('.',1)
        for i,directory in enumerate(dirs):
            rounds=json.loads((directory/'wrapper_runtime.json').read_text())['rounds']
            seen={json.dumps(row['globals'][module][attr]) for row in rounds}
            if i and seen!={json.dumps(json.loads(value))}:raise RuntimeError('Candidate global %s not %s: %s'%(key,value,seen))
            if not i and json.dumps(json.loads(value)) in seen:raise RuntimeError('Baseline global %s equals the candidate: %s'%(key,seen))
    result=native_pair(*dirs);result['actual_go_audits_passed']=True
    save(label+'_comparison.json',result)
    if not result['complete'] or result['valid_pairs']!=len(a.maps):raise RuntimeError('Incomplete comparison')
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--label',required=True);p.add_argument('--seeds',nargs=2,type=int,required=True)
    p.add_argument('--candidate-flag',action='append',default=[]);p.add_argument('--baseline-flag',action='append',default=[])
    p.add_argument('--after',action='append',default=[]);p.add_argument('--require-global',action='append',default=[])
    p.add_argument('--note',default='')
    p.add_argument('--workers-per-game',type=int,default=8,help='forecast workers per arm (8 = the other campaigns; 6 when the commit limit is tight)')
    p.add_argument('--port-base',type=int,default=9440,help='arm ports port_base and port_base+4 (9440 = the other drivers; use 9460 to run beside one)')
    p.add_argument('--always-confirm',action='store_true',help='run the fresh seed whatever the exploratory mean (20 pairs; 2026-10-28: one seed has an SD of ~6 a round)')
    p.add_argument('--bench',default='ml_agent/tools/native_worker_bench.py',help='the per-arm tool, the same for both arms (2026-10-29: the other agent\'s wrappers, e.g. ml_agent/tools/follow_backup_bench.py, take their own flag and pass the rest to native_worker_bench)')
    p.add_argument('--fixed-lead',type=int,default=3,help='both arms (3 = the .172 s allowance; 2026-10-29: 16 = a ~1 s budget that almost never binds, so the planner is deterministic and the arms split only where the change acts)')
    p.add_argument('--maps',nargs='+',default=None,help='a subset of the ten maps (default all; for short checks)')
    a=p.parse_args()
    a.maps=list(a.maps or MAPS)
    if not set(a.maps)<=set(MAPS) or len(set(a.maps))!=len(a.maps):raise SystemExit('unknown or repeated map in --maps')
    if not a.candidate_flag:raise SystemExit('a candidate flag is required')
    status_name=a.label+'_native_status.json'
    if (ROOT/(a.label+'_native_plan.json')).exists():raise FileExistsError('Refusing to overwrite an existing campaign')
    plan=dict(maps=list(a.maps),exploratory_seed=a.seeds[0],confirmation_seed=a.seeds[1],fixed_budget_s=lead_budget(a.fixed_lead),fixed_lead=a.fixed_lead,
              candidate_workers_per_game=a.workers_per_game,simultaneous_games=2,baseline_flags=a.baseline_flag,candidate_flags=a.candidate_flag,
              required_globals=a.require_global,after=a.after,note=a.note,port_base=a.port_base,
              always_confirm=a.always_confirm,bench=a.bench,
              selection=('Run every map on both seeds (arm ports swapped on the fresh seed).' if a.always_confirm else
                         'Run every map. If the exploratory equal-map mean improves, repeat all ten on the fresh seed (arm ports swapped).')
                        +' Retain per-map regressions and all failed audits.')
    save(a.label+'_native_plan.json',plan);save(status_name,dict(stage='waiting',after=a.after))
    try:
        for other in a.after:
            while True:
                status=json.loads((ROOT/other).read_text())
                if status['stage']=='complete':break
                if status['stage']=='failed':raise RuntimeError('Earlier campaign failed: '+other)
                time.sleep(5)
        save(status_name,dict(stage='exploratory'))
        explore=run_pair(a,a.label+'_explore',a.seeds[0],0)
        if explore['delta']>0 or a.always_confirm:
            save(status_name,dict(stage='confirmation',exploratory_delta=explore['delta']))
            run_pair(a,a.label+'_confirm',a.seeds[1],1)
            combined=native_pair(ROOT/(a.label+'_explore_baseline'),ROOT/(a.label+'_explore_candidate'),
                                 ROOT/(a.label+'_confirm_baseline'),ROOT/(a.label+'_confirm_candidate'))
            save(a.label+'_combined_comparison.json',combined)
            save(status_name,dict(stage='complete',exploratory_delta=explore['delta'],combined_delta=combined['delta']))
        else:
            save(status_name,dict(stage='complete',exploratory_delta=explore['delta']))
    except BaseException as exc:
        save(status_name,dict(stage='failed',error=str(exc)));raise


if __name__=='__main__':main()
