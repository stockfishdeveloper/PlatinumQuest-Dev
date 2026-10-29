"""Full native rounds with production candidate workers and a fixed lead budget.

Remaining arguments go to reliability_eval. Requires --workers 1 --full-round
--only-branch physics_plan. This is direct-control native evaluation, NOT the
asynchronous live queue. Both arms must use this same wrapper and workload.
The wrapper, runtime globals, ordinary reports and all audits are preserved.
"""
import argparse
from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
import traceback


def prepare_countdown(env):
    """Use charge phase when available; otherwise observe neutral GO at 16 ms.

    Non-charging maps have no meter clock. Their paired rounds start unspun,
    just after natural GO; no scored state is restored or resource granted.
    """
    state=env.msg.dynamics['native']['state']
    if state['mode']!=2:raise RuntimeError('Native preparation requires Start mode')
    blast=env.msg.dynamics.get('blast',{})
    if blast.get('enabled') and blast.get('charging'):
        from tools.realtime_bench import align_countdown
        return dict(method='charge_phase',**align_countdown(env))
    from nav.protocol import NOOP_ACTION
    env.control('FIXEDSTEP 16 16');env.obs_ms=16
    for _ in range(256):
        state=env.msg.dynamics['native']['state']
        if state['mode']==0 and not state.get('oob'):break
        if state['mode']!=2:raise RuntimeError('Unexpected mode while awaiting natural GO')
        env.step(NOOP_ACTION,repeat=1)
    else:raise RuntimeError('Natural GO not observed within countdown bound')
    go_clock=float(state['clock_ms'])
    env.control('FIXEDSTEP 16 64');env.obs_ms=64
    after=env.msg.dynamics['native']['state'];elapsed=float(after['clock_ms'])-go_clock
    if after['mode']!=0 or after.get('oob') or not 0<elapsed<=64.:
        raise RuntimeError('Unexpected cadence restoration after neutral GO')
    if env.time_left_s()<float(env.info['round_ms'])/1000.-.128:
        raise RuntimeError('Neutral GO preparation consumed too much scored time')
    return dict(method='neutral_go_noncharging',go_to_snapshot_ms=elapsed,
        physics_frame_ms=16,decision_ms=64,prespin=False)


@contextmanager
def candidate_workers(module,factory,initializer,workers,on_round=None,prepare_round=None):
    """Only add evaluation workers; preserve the controller's planning methods."""
    original_controller=module.ContinuousController
    original_round=module.full_round_branch
    current={};owned=[]

    def close_owned():
        while owned:
            engine,pool=owned.pop()
            try:pool.close()
            finally:engine.parallel=None

    class WorkerController(original_controller):
        def __init__(self,terrain,engine,checkpoint,args,kind):
            super().__init__(terrain,engine,checkpoint,args,kind)
            if not self.uses_physics:return
            if current.get('engine') is not engine or getattr(engine,'parallel',None) is not None:
                raise RuntimeError('Unexpected candidate-pool ownership')
            pool=factory(workers,initializer(args,current['name']))
            engine.parallel=pool;owned.append((engine,pool))
            pool.bind(dict(terrain=terrain,geometry=args.geometry_terrain,
                mesh=engine.mesh,sim=engine.native_sim,model=self.nav.model,
                crossings=engine.crossings,contact_order=getattr(engine,'contact_order',None)))

    def round_branch(env,proc,name,seed,kind,args,terrain,engine,checkpoint_hash,round_index):
        if current:raise RuntimeError('Native rounds must run sequentially within each arm')
        current.update(name=name,engine=engine)
        try:
            if on_round is not None:on_round(name,seed,round_index)
            kwargs=dict(prepare_round=prepare_round) if prepare_round is not None else {}
            return original_round(env,proc,name,seed,kind,args,terrain,engine,checkpoint_hash,round_index,**kwargs)
        finally:
            try:close_owned()
            finally:current.clear()

    module.ContinuousController=WorkerController;module.full_round_branch=round_branch
    try:yield
    finally:
        module.ContinuousController=original_controller;module.full_round_branch=original_round
        close_owned()


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-workers',type=int,default=8)
    parser.add_argument('--fixed-lead',type=int,default=3)
    parser.add_argument('--order-min-left-s',type=float,default=.35)
    own,remaining=parser.parse_known_args()
    scan=argparse.ArgumentParser(add_help=False)
    scan.add_argument('--output',type=Path,required=True)
    scan.add_argument('--workers',type=int,default=1)
    scan.add_argument('--only-branch')
    scan.add_argument('--full-round',action='store_true')
    scan.add_argument('--geometry-route',action='store_true')
    scan.add_argument('--sweep-dll',type=Path)
    args,_=scan.parse_known_args(remaining)
    if (not 1<=own.candidate_workers<=16 or not 1<=own.fixed_lead<=16 or not 0<=own.order_min_left_s<=.35 or
            args.workers!=1 or args.only_branch!='physics_plan' or not args.full_round or
            not args.geometry_route or args.sweep_dll is None):
        parser.error('Requires 1..16 candidate workers/lead, one native worker, full physics_plan, compiled geometry')
    output=args.output
    if output.exists():raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).resolve();fingerprint=sha(source)
    wrapper_bytes=source.read_bytes()
    from nav.marble_dynamics import reliability_eval as re,search as search
    from nav.marble_dynamics.parallel_search import ParallelEvaluator,worker_init,module_globals
    from nav.collision.mesh import atomic_json
    from tools.realtime_bench import align_countdown
    preparation_source=Path(inspect.getfile(align_countdown)).resolve()
    preparation_bytes=preparation_source.read_bytes();preparation_fingerprint=sha(preparation_source)
    if 'prepare_round' not in inspect.signature(re.full_round_branch).parameters:
        raise RuntimeError('Native round setup callback is not installed; do not run an unaligned comparison')
    budget=min(2.,max(.05,own.fixed_lead*.064-.02))
    re.RESEARCH_EVERY=1 if own.fixed_lead>=2 else 10**6
    re.ORDER_CANDIDATES=3 if own.fixed_lead>=2 else 1
    re.ORDER_EXTRA_S=0.;search.CLUSTER_EXTRA_S=0.;search.EXTRA_JUMP_S=.05
    re.ORDER_MIN_LEFT_S=own.order_min_left_s
    forwarded=remaining+['--search-budget-s',str(budget),'--stall-budget-s',str(budget)]
    declaration=dict(schema='native_candidate_workers_v1',wrapper=str(source),wrapper_sha256=fingerprint,
        candidate_workers=own.candidate_workers,fixed_lead=own.fixed_lead,budget_s=budget,
        order_min_left_s=own.order_min_left_s,
        preparation_source=str(preparation_source),preparation_sha256=preparation_fingerprint,
        noncharging_start='Natural neutral GO, then cadence restore; unspun and paired, not exact live deployment',
        countdown_charge_phase_ms=1984,
        arguments=forwarded,scope='Ordinary complete native rounds; direct observations, no live queue latency. '
            'Fixed runtime-like search allowance; not exact live deployment or a serial-pool comparison.')
    atomic_json(output.parent/(output.name+'_wrapper_protocol.json'),declaration)
    observed=[];started=time.perf_counter();error=None

    def observe_round(name,seed,round_index):
        if sha(source)!=fingerprint:raise RuntimeError('Wrapper changed during evaluation')
        observed.append(dict(map=name,seed=seed,round=round_index,globals=module_globals()))
        atomic_json(output/'wrapper_runtime.json',dict(rounds=observed))

    old_argv=sys.argv
    try:
        sys.argv=['reliability_eval',*forwarded]
        with candidate_workers(re,ParallelEvaluator,worker_init,own.candidate_workers,observe_round,prepare_countdown):re.main()
    except BaseException:
        error=traceback.format_exc()
        output.mkdir(parents=True,exist_ok=True)
        atomic_json(output/'wrapper_failure.json',dict(traceback=error))
    finally:
        sys.argv=old_argv
        output.mkdir(parents=True,exist_ok=True)
        (output/'wrapper_source.py').write_bytes(wrapper_bytes)
        (output/'preparation_source.py').write_bytes(preparation_bytes)
        atomic_json(output/'wrapper_protocol.json',declaration)
    summary_path=output/'summary.json'
    summary=json.loads(summary_path.read_text()) if summary_path.exists() else dict(branches=[])
    prepared_path=output/'prepared.json'
    prepared=json.loads(prepared_path.read_text()) if prepared_path.exists() else {}
    rows=summary['branches']
    expected=len(prepared.get('maps',[]))*len(prepared.get('seeds',[]))*prepared.get('rounds',1)
    unchanged=sha(source)==fingerprint and sha(preparation_source)==preparation_fingerprint and bool(prepared.get('source_hashes')) and all(
        sha(p)==h for p,h in prepared.get('source_hashes',{}).items())
    atomic_json(output/'input_integrity.json',dict(passed=unchanged,wrapper_unchanged=sha(source)==fingerprint))
    audit=re.audit_start_layouts(rows) if rows else dict(passed=False,reason='No rounds')
    atomic_json(output/'start_layout_audit.json',audit)
    valid=expected>0 and len(rows)==expected and all(r.get('score_valid') and
        r.get('completed') and r.get('declared_inputs_valid') for r in rows)
    summary.update(complete=not error and unchanged and valid and audit['passed'],
        candidate_search_workers=own.candidate_workers,fixed_lead=own.fixed_lead,
        wrapper_protocol='wrapper_protocol.json',elapsed_wall_s=time.perf_counter()-started)
    atomic_json(summary_path,summary)
    if not summary['complete']:raise RuntimeError('Native worker benchmark incomplete; inspect preserved audits/reports')


if __name__=='__main__':main()
