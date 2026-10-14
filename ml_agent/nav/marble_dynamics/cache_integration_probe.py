"""Matched native development comparison of exact cache integration.

Both variants use current source and the same archived starting world. This
isolates caches, rather than comparing against a previous software revision.
No training, scored-play restores, grants or rescues.
"""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from nav.collision.mesh import CollisionMap,atomic_json
from nav.marble_dynamics.commit_demo import restore
from nav.marble_dynamics.geometry_route import walking_terrain
from nav.marble_dynamics.ground_cache import MemoGroundClearance
from nav.marble_dynamics.navigator import FrozenNavigator
from nav.marble_dynamics.physics import GeneralPhysics
from nav.marble_dynamics.query_cache import CachedCollisionMap
from nav.marble_dynamics.record import ML,RUNTIME,archive,command,game,sha
from nav.marble_dynamics.reliability_eval import branch
from nav.protocol import NOOP_ACTION
from nav.terrain import TerrainGrid,TerrainMap
from nav.collision.experimental_sweep import CachedCompiledSweepMap


def compare_logs(logs,reports):
    """Compare physical decisions while retaining restored-instance provenance.

    Native restore recreates item IDs and does not rewind the engine clock.
    Map IDs to the observed item's identity, and clocks to each current frame.
    Inactive effect deadlines are stale native bookkeeping; active deadlines are
    compared relative to the current frame. Keep all physical state/controls.
    """
    catalogs=[]
    for recording in logs:
        catalogs.append({item['id']:(bool(item['gem']),item['powerup'],tuple(item['position']),item['value'])
            for entry in recording for item in entry['dynamics']['items']})
    def choice(value,catalog):
        value=copy.deepcopy(value)
        if value.get('goal_item_id') is not None:value['goal_item_id']=catalog[value['goal_item_id']]
        return value
    def canonical_state(entry,key):
        state=copy.deepcopy(entry[key])
        if state is None:return None
        clock=entry['dynamics']['native']['frame_clock_ms']
        if 'clock_ms' in state:state['clock_ms']-=clock
        if 'effect_end_clock_ms' in state:state['effect_end_clock_ms']=[t-clock if state['effects'][i][0] else None
            for i,t in enumerate(state['effect_end_clock_ms'])]
        return state
    mismatches=[];maximum=0.
    for index,(left,right) in enumerate(zip(*logs)):
        different=[key for key in ('action','gem','goal_kind','score_delta','external_events') if left[key]!=right[key]]
        if choice(left['chosen'],catalogs[0])!=choice(right['chosen'],catalogs[1]):different.append('chosen')
        if canonical_state(left,'predicted_next')!=canonical_state(right,'predicted_next'):different.append('predicted_next')
        if canonical_state(left,'actual_next')!=canonical_state(right,'actual_next'):different.append('native_state')
        for field in ('position','velocity','spin'):
            error=float(np.linalg.norm(np.asarray(left['actual_next'][field])-right['actual_next'][field]));maximum=max(maximum,error)
            if error>1e-7:different.append('native_'+field)
        if different:mismatches.append(dict(decision=index,fields=different))
    equal=len(logs[0])==len(logs[1]) and not mismatches and all(reports[0][k]==reports[1][k] for k in ('actual_points','falls','stop_reason'))
    return dict(variants=reports,controls_choices_scores_native_equal=equal,mismatches=mismatches,
        compared_decisions=min(map(len,logs)),maximum_native_state_difference=maximum,
        identity_normalization='Observed item geometry/type/value; state clocks and active effect deadlines relative to current frame; inactive stale deadlines omitted. Original logs unchanged.',
        training=False,caveat='One exact archived start and bounded decisions; cache equivalence development check, not performance/generalization benchmark')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--port',type=int,default=9054)
    p.add_argument('--decisions',type=int,default=320)
    p.add_argument('--sweep-dll',type=Path,help='Compare cached Python versus cached compiled sweep; otherwise compare caches only')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    name='Concentric_Hunt';seed=141112
    # The archived successful live regression names its original fixture explicitly.
    previous=ML/'dynamics_reports/reliability_landing_regression_r2_20261012'/name/str(seed)/'fixture.json'
    metadata=json.loads(previous.read_text());world=copy.deepcopy(metadata['world'])
    source=Path(metadata['source']['path'])
    checkpoint=ML/'models/nav/real_maps_20261010_A2/nav_latest.pth';fingerprint=sha(checkpoint)
    sources=list(Path(__file__).parent.glob('*.py'))+[ML/'nav/routing.py',ML/'nav/terrain.py',ML/'nav/collision/mesh.py',
        ML.parent/'Marble Blast Platinum/platinum/client/scripts/ai/mlAgent.cs',ML.parent/'Marble Blast Platinum/marbleblast_geometry_probe.exe']
    if a.sweep_dll:
        sources.extend([a.sweep_dll,ML/'nav/collision/experimental_sweep.py',ML/'nav/collision/sweep_kernel.cpp',ML/'nav/collision/build_sweep_kernel.ps1'])
        saved=a.output/'external_artifacts'/a.sweep_dll.name;saved.parent.mkdir();saved.write_bytes(a.sweep_dll.read_bytes())
    prepared=dict(maps=[name],seeds=[seed],decisions=a.decisions,horizon=40,commit=4,reuse_plan=True,
        brake_continuation=48,recovery_horizon=80,geometry_route=True,resource_acquisition=True,
        preserve_resources=True,recharge_continuation=True,landing_search=True,proposal_seed=997,proposal_samples=40,
        global_goals=False,fixture_data=str(source.parent.parent.parent),checkpoint=str(checkpoint),checkpoint_sha256=fingerprint,
        source_hashes={str(path):sha(path) for path in sources},training=False,
        selection='Two current-source planner variants; same archived start; only compiled sweep differs' if a.sweep_dll else 'Two current-source planner variants; same archived start, only exact caches differ')
    atomic_json(a.output/'prepared.json',prepared)
    for path in sources:
        if path.suffix.lower() not in ('.py','.cs','.cpp','.ps1'):continue
        saved=a.output/'source_snapshot'/path.resolve().relative_to(ML.parent.resolve())
        saved.parent.mkdir(parents=True,exist_ok=True);saved.write_bytes(path.read_bytes())
    terrain=TerrainGrid(TerrainMap.resolve(name));engines={};arguments={}
    variants=[('reference_cached',True,None),('compiled_cached',True,a.sweep_dll)] if a.sweep_dll else [('uncached',False,None),('cached',True,None)]
    for label,cached,dll in variants:
        root=a.output/(a.output.name+'_'+label);root.mkdir()
        options={**prepared,'output':root,'checkpoint':checkpoint,'query_cache':cached,'clearance_cache':cached,'sweep_dll':dll}
        arguments[label]=SimpleNamespace(**options)
        path=ML/'collision_maps'/(name+'.npz')
        engine=GeneralPhysics(CachedCompiledSweepMap(path,dll) if dll else (CachedCollisionMap if cached else CollisionMap)(path))
        geometry=walking_terrain(terrain,engine.mesh)
        if cached:geometry.ground_clearance=MemoGroundClearance(geometry.ground_clearance)
        arguments[label].geometry_terrain=geometry;engines[label]=engine
        atomic_json(root/'prepared.json',{**prepared,'query_cache':cached,'clearance_cache':cached,'sweep_dll':str(dll) if dll else None})
    FrozenNavigator(terrain,checkpoint)
    reports=[]
    with game(name,a.port,obs_ms=64,seed=seed) as (env,proc):
        command(env,'DYNAMICSLIVE 1','dynamics_live');env.control('REPLAYCAPTURE 1')
        for _ in range(8):env.step(NOOP_ACTION,repeat=1)
        for label,cached,dll in variants:
            if any(sha(Path(path))!=fingerprint for path,fingerprint in prepared['source_hashes'].items()):
                raise RuntimeError('Integration source changed after declaration')
            args=arguments[label];folder=args.output/name/str(seed);folder.mkdir(parents=True)
            restore(env,world)
            atomic_json(folder/'fixture.json',dict(world=world,catalog=env.replay_catalog,dynamics=env.msg.dynamics,
                source=dict(path=str(source),sha256=sha(source),development_regression=True)))
            try:report=branch(env,proc,world,name,seed,'physics_plan',args,terrain,engines[label],fingerprint)
            except Exception as exc:
                dest=folder/'physics_plan';dest.mkdir(exist_ok=True)
                try:command(env,'DYNAMICSRECORD off','dynamics_record')
                except Exception:pass
                stem=args.output.name+'_'+name+'_'+str(seed)+'_physics_plan'
                raw=RUNTIME/(stem+'.jsonl');journal=RUNTIME/(stem+'.decisions.jsonl')
                if raw.exists():archive(raw,dest/'native.failed.jsonl.gz')
                if journal.exists():atomic_json(dest/'decisions.failed.json',[json.loads(l) for l in journal.read_text().splitlines()])
                atomic_json(dest/'failed.json',dict(reason=repr(exc),training=False));raise
            reports.append(report)
            atomic_json(args.output/'summary.json',dict(branches=[report],checkpoint_unchanged=sha(checkpoint)==fingerprint,policy_training_started=False))
    logs=[json.loads((arguments[label].output/name/str(seed)/'physics_plan/decisions.json').read_text()) for label,cached,dll in variants]
    result=compare_logs(logs,reports);result['checkpoint_unchanged']=sha(checkpoint)==fingerprint
    atomic_json(a.output/'comparison.json',result)
    print(json.dumps({k:result[k] for k in ('controls_choices_scores_native_equal','compared_decisions','maximum_native_state_difference','mismatches','checkpoint_unchanged')}),flush=True)
    assert result['controls_choices_scores_native_equal'],'Exact cache integration changed behavior; inspect preserved variants'
    assert sha(checkpoint)==fingerprint


if __name__=='__main__':main()
