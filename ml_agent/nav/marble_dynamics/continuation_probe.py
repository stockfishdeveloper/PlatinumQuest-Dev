"""Replay archived ordinary inputs, then execute a newly checked continuation.

Development regression only: the archived start world is restored before play,
then every former reply is replayed and checked against its recorded outcome.
No teleport, rescue or inventory grant occurs in either execution phase.
"""
from nav.source_archive import snapshot_sources
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

from nav.collision.mesh import CollisionMap,atomic_json
from nav.marble_dynamics.record import ML,RUNTIME,game,command,archive,sha
from nav.marble_dynamics.commit_demo import restore
from nav.marble_dynamics.physics import GeneralPhysics
from nav.marble_dynamics.evaluate import errors
from nav.marble_dynamics.navigator import FrozenNavigator
from nav.marble_dynamics.search import search
from nav.marble_dynamics.geometry_route import walking_terrain
from nav.routing import RoutePlanner
from nav.terrain import TerrainGrid,TerrainMap
from nav.protocol import NOOP_ACTION,RAW_SCORE


def run():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path);p.add_argument('--port',type=int,default=9053)
    p.add_argument('--recharge',action='store_true',help='Use the physically checked wait/charge continuation rather than immediate search')
    p.add_argument('--landing-search',action='store_true',help='Explore general control phases and full-map support points for a checked pickup/landing continuation')
    p.add_argument('--global-goals',action='store_true',help='Isolate extra currently observed gem goals after exact old-prefix replay; progress plans need not predict pickup')
    p.add_argument('--proposal-seed',type=int,default=997);p.add_argument('--proposal-samples',type=int,default=40)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    old=json.loads((a.source/'decisions.json').read_text());terminal=json.loads((a.source/'terminal_search.json').read_text())
    fixture=json.loads((a.source.parent/'fixture.json').read_text());report=json.loads((a.source/'report.json').read_text())
    name=report['map'];seed=report['seed'];checkpoint=ML/'models/nav/real_maps_20261010_A2/nav_latest.pth'
    checkpoint_hash=sha(checkpoint);engine=GeneralPhysics(CollisionMap(ML/'collision_maps'/(name+'.npz')))
    terrain=TerrainGrid(TerrainMap.resolve(name));route=RoutePlanner(walking_terrain(terrain,engine.mesh))
    nav=FrozenNavigator(terrain,checkpoint);raw_path=RUNTIME/(a.output.name+'.native.jsonl')
    atomic_json(a.output/'prepared.json',dict(source=str(a.source),seed=seed,map=name,
        original_decisions_sha256=sha(a.source/'decisions.json'),prefix_replies=len(old),horizon=80,brake_replies=48,
        checkpoint_sha256=checkpoint_hash,recharge=a.recharge,landing_search=a.landing_search,global_goals=a.global_goals,
        proposal_seed=a.proposal_seed,proposal_samples=a.proposal_samples,training=False,development_regression=True))
    sources=list(Path(__file__).parent.glob('*.py'))+[ML/'nav/routing.py',ML/'nav/terrain.py',ML/'nav/collision/mesh.py',
        ML.parent/'Marble Blast Platinum/platinum/client/scripts/ai/mlAgent.cs',ML.parent/'Marble Blast Platinum/marbleblast_geometry_probe.exe']
    prepared=json.loads((a.output/'prepared.json').read_text());prepared['source_hashes']={str(p):sha(p) for p in sources}
    atomic_json(a.output/'prepared.json',prepared)
    snapshot_sources(a.output,[path for path in sources if path.suffix.lower() in ('.py','.cs')])
    prefix_errors=[];logs=[];selected=None
    journal=(RUNTIME/(a.output.name+'.decisions.jsonl')).open('w',encoding='utf-8',buffering=1)
    with game(name,a.port,obs_ms=64,seed=seed) as (env,proc):
        command(env,'DYNAMICSLIVE 1','dynamics_live');env.control('REPLAYCAPTURE 1')
        for _ in range(8):env.step(NOOP_ACTION,repeat=1)
        matched=restore(env,fixture['world'])
        if matched['position_error_u']>1e-5 or not matched['score_equal']:raise RuntimeError('Initial archived fixture mismatch')
        command(env,'DYNAMICSRECORD '+raw_path.as_posix(),'dynamics_record')
        for i,record in enumerate(old):
            action=record['action'];before=copy.deepcopy(env.msg.dynamics)
            _,outcome=env.step(action[:6],repeat=1,use_pow=int(action[6]),pow_yaw=action[7] if action[6] else None,use_blast=int(action[8]))
            actual=copy.deepcopy(env.msg.dynamics['native']['state']);error=errors(actual,record['actual_next'])
            prefix_errors.append(error)
            if error[0]>1e-4 or error[1]>1e-3:raise RuntimeError('Archived input replay drift '+str((i,error)))
            logs.append(dict(phase='prefix',index=i,dynamics=before,action=action,actual_next=actual,errors=error))
            journal.write(json.dumps(logs[-1])+'\n')
        snapshot=copy.deepcopy(env.msg.dynamics);raw=env.msg.obs.copy()
        endpoint_error=errors(snapshot['native']['state'],terminal['dynamics']['native']['state'])
        expected_items=sorted((i['gem'],i['powerup'],tuple(i['position'])) for i in terminal['dynamics']['items'])
        actual_items=sorted((i['gem'],i['powerup'],tuple(i['position'])) for i in snapshot['items'])
        if expected_items!=actual_items:raise RuntimeError('Replayed visible item state differs')
        if float(raw[RAW_SCORE])!=report['final_score']:raise RuntimeError('Replayed score differs')
        goal=terminal['target'];nav=FrozenNavigator(terrain,checkpoint)
        if a.global_goals:
            from nav.marble_dynamics.global_goals import observed_gems
            from nav.marble_dynamics.reliability_eval import pick
            from nav.marble_dynamics.search import PlanMemory
            from nav.gems import visible_gems
            nearby=visible_gems(raw)
            extra=[g for g in observed_gems(snapshot,raw) if not any(np.linalg.norm(np.asarray(g[:3])-np.asarray(n[:3]))<.1 for n in nearby)]
            if not extra:raise RuntimeError('No extra observed gem goal at replayed endpoint')
            chosen,infos,goal=pick(engine,route,snapshot,raw,extra,extra[0],nav,40,PlanMemory(),48,80,preserve_resources=True)
            pairs=[(info,None) for info in infos]
        elif a.landing_search:
            from nav.marble_dynamics.phased_search import explore
            from nav.marble_dynamics.settle import choose_settled
            from nav.marble_dynamics.landing import alternatives
            from nav.marble_dynamics.search import useful_candidate
            _,sampled=explore(engine,route,snapshot,goal,seed=a.proposal_seed,samples=a.proposal_samples)
            _,settled=choose_settled(engine,route,snapshot,goal)
            pairs=alternatives(engine,route,snapshot,goal,sampled+settled)
            chosen=useful_candidate(pairs,route,route.distance(snapshot['native']['state']['position']),True)
        elif a.recharge:
            from nav.marble_dynamics.recharge import choose_recharge
            chosen,pairs=choose_recharge(engine,route,snapshot,goal,1000,float(raw[31]))
        else:chosen,pairs=search(engine,route,snapshot,raw,goal[:3],nav,80,brake_replies=48)
        atomic_json(a.output/'continuation_search.json',dict(dynamics=snapshot,goal=goal,candidates=[i for i,r in pairs]))
        if chosen is None or (not a.global_goals and chosen[0]['possible_pickup_interval'] is None):raise RuntimeError('No checked continuation for requested mode')
        selected,forecast=chosen;atomic_json(a.output/'selected_forecast.json',forecast)
        initial_route_distance=route.distance(snapshot['native']['state']['position'])
        score=float(raw[RAW_SCORE]);pickups=[]
        for i,action in enumerate(forecast['proposed']):
            before=copy.deepcopy(env.msg.dynamics)
            _,outcome=env.step(action[:6],repeat=1,use_pow=int(action[6]),pow_yaw=action[7] if action[6] else None,use_blast=int(action[8]))
            actual=copy.deepcopy(env.msg.dynamics['native']['state']);error=errors(forecast['states'][i+1],actual)
            delta=float(env.msg.obs[RAW_SCORE])-score;score+=delta
            if delta:pickups.append(dict(interval=i,points=delta))
            logs.append(dict(phase='continuation',index=i,dynamics=before,action=action,actual_next=actual,
                predicted_next=forecast['states'][i+1],score_delta=delta,errors=error,outcome=outcome))
            journal.write(json.dumps(logs[-1])+'\n')
            if outcome['fell'] or outcome['round_ended'] or outcome['reconnected']:break
        final_route_distance=route.distance(env.msg.dynamics['native']['state']['position'])
        command(env,'DYNAMICSRECORD off','dynamics_record')
    journal.close();archive(raw_path,a.output/'native.jsonl.gz');atomic_json(a.output/'decisions.json',logs)
    result=dict(map=name,seed=seed,selected=selected,prefix_replies=len(old),
        max_prefix_errors=np.max(prefix_errors,axis=0).tolist(),endpoint_replay_errors=endpoint_error,
        actual_continuation_points=score-report['final_score'],pickups=pickups,
        goal=goal,initial_route_distance_u=initial_route_distance,final_route_distance_u=final_route_distance,
        actual_route_progress_u=initial_route_distance-final_route_distance,
        max_continuation_errors=np.max([r['errors'] for r in logs if r['phase']=='continuation'],axis=0).tolist(),
        falls=sum(int(r['outcome']['fell']) for r in logs if r['phase']=='continuation'),
        checkpoint_unchanged=sha(checkpoint)==checkpoint_hash,training=False,
        caveat='Reused archived development trajectory, not a fresh-start/full-round comparison.')
    atomic_json(a.output/'report.json',result);print(json.dumps(result),flush=True)


def main():
    try:run()
    except Exception as exc:
        # Preserve failed evidence after the owned-game context has closed.
        if '--output' in sys.argv:
            output=Path(sys.argv[sys.argv.index('--output')+1])
            if output.exists() and not (output/'report.json').exists() and not (output/'failed.json').exists():
                raw=RUNTIME/(output.name+'.native.jsonl');journal=RUNTIME/(output.name+'.decisions.jsonl')
                if raw.exists():archive(raw,output/'native.failed.jsonl.gz')
                if journal.exists():atomic_json(output/'decisions.failed.json',[json.loads(l) for l in journal.read_text().splitlines()])
                atomic_json(output/'failed.json',dict(reason=repr(exc),training=False))
        raise


if __name__=='__main__':main()
