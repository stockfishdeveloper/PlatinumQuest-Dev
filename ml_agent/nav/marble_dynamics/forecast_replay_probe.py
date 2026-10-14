"""Execute archived-prefix physics fixtures then explicitly saved ordinary inputs.

No grants, rescues or resets during execution; initial restore is labeled setup.
Detailed future native contacts are audit outputs, never predictor inputs.
"""
from nav.source_archive import snapshot_sources
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from nav.collision.mesh import CollisionMap,atomic_json
from nav.marble_dynamics.bridge import predict_replies
from nav.marble_dynamics.commit_demo import restore
from nav.marble_dynamics.evaluate import errors,load_rows
from nav.marble_dynamics.physics import GeneralPhysics
from nav.marble_dynamics.record import ML,RUNTIME,archive,command,game,sha
from nav.protocol import NOOP_ACTION,RAW_SCORE


def execute(source,actions,output,port):
    output.mkdir(parents=True,exist_ok=False)
    report=json.loads((source/'report.json').read_text());prefix=json.loads((source/'decisions.json').read_text())
    fixture=json.loads((source.parent/'fixture.json').read_text());name=report['map'];seed=report['seed']
    terminal=json.loads((source/'terminal_search.json').read_text())
    raw=RUNTIME/(output.name+'.native.jsonl');journal=RUNTIME/(output.name+'.decisions.jsonl');logs=[];prefix_errors=[]
    atomic_json(output/'prepared.json',dict(source=str(source),source_decisions_sha256=sha(source/'decisions.json'),
        prefix_replies=len(prefix),actions=actions,training=False,scored_navigation=False,
        declaration='Archived-state mechanics fixture; detail contacts only for independent outcome audit'))
    engine=GeneralPhysics(CollisionMap(ML/'collision_maps'/(name+'.npz')))
    route=None
    if actions=='route24':
        from nav.marble_dynamics.geometry_route import walking_terrain
        from nav.routing import RoutePlanner
        from nav.terrain import TerrainGrid,TerrainMap
        route=RoutePlanner(walking_terrain(TerrainGrid(TerrainMap.resolve(name)),engine.mesh))
    try:
        with game(name,port,obs_ms=64,seed=seed) as (env,proc):
            command(env,'DYNAMICSLIVE 1','dynamics_live');env.control('REPLAYCAPTURE 1')
            for _ in range(8):env.step(NOOP_ACTION,repeat=1)
            # Control acknowledgements may execute NOOP frames. Set recording
            # up BEFORE the restore/prefix, never at a moving terminal state.
            command(env,'DYNAMICSDETAIL 1','dynamics_detail');command(env,'DYNAMICSRECORD '+raw.as_posix(),'dynamics_record')
            match=restore(env,fixture['world'])
            if match['position_error_u']>1e-5 or not match['score_equal']:raise RuntimeError('Archived fixture mismatch')
            for index,record in enumerate(prefix):
                action=record['action'];_,outcome=env.step(action[:6],repeat=1,use_pow=int(action[6]),pow_yaw=action[7] if action[6] else None,use_blast=int(action[8]))
                error=errors(env.msg.dynamics['native']['state'],record['actual_next']);prefix_errors.append(error)
                if error[0]>1e-4 or error[1]>1e-3:raise RuntimeError('Prefix replay drift '+str((index,error)))
            snapshot=copy.deepcopy(env.msg.dynamics)
            endpoint_error=errors(snapshot['native']['state'],terminal['dynamics']['native']['state'])
            if endpoint_error[0]>1e-4 or endpoint_error[1]>1e-3 or endpoint_error[2]>1e-3:
                raise RuntimeError('Terminal fixture drift '+str(endpoint_error))
            if snapshot['native']['pending_move']!=terminal['dynamics']['native']['pending_move']:
                raise RuntimeError('Terminal pending native Move differs')
            if route is not None:
                from nav.marble_dynamics.search import SequenceProposal
                forecast=predict_replies(engine,snapshot,SequenceProposal(route,terminal['target'][:3],24))
                actions=forecast['proposed']
            else:forecast=predict_replies(engine,snapshot,actions)
            atomic_json(output/'initial.json',dict(dynamics=snapshot,raw_observation=env.msg.obs.tolist()))
            atomic_json(output/'forecast.json',forecast)
            if forecast['failure']:raise RuntimeError('Forecast failed '+str(forecast['failure']))
            score=float(env.msg.obs[RAW_SCORE])
            with journal.open('w',encoding='utf-8',buffering=1) as stream:
                for index,action in enumerate(actions):
                    before=copy.deepcopy(env.msg.dynamics)
                    _,outcome=env.step(action[:6],repeat=1,use_pow=int(action[6]),pow_yaw=action[7] if action[6] else None,use_blast=int(action[8]))
                    actual=copy.deepcopy(env.msg.dynamics['native']['state'])
                    entry=dict(index=index,dynamics=before,action=action,actual_next=actual,
                        predicted_next=forecast['states'][index+1],errors=errors(forecast['states'][index+1],actual),
                        score_delta=float(env.msg.obs[RAW_SCORE])-score,outcome=outcome)
                    score+=entry['score_delta'];logs.append(entry);stream.write(json.dumps(entry)+'\n')
                    if outcome['fell'] or outcome['round_ended'] or outcome['reconnected']:break
            command(env,'DYNAMICSRECORD off','dynamics_record');command(env,'DYNAMICSDETAIL 0','dynamics_detail')
        archive(raw,output/'native.jsonl.gz');atomic_json(output/'decisions.json',logs)
        grazing=[]
        for row in load_rows(output/'native.jsonl.gz'):
            clock=snapshot['native']['frame_clock_ms']
            if not clock<=row['sim_ms']<clock+len(logs)*64:continue
            for step in row.get('substep_trace',[]):
                velocity=np.asarray(step['contact_entry_velocity']);normals=[np.asarray(c['normal']) for c in step['contacts']]
                dots=[float(velocity@normal) for normal in normals]
                if len(dots)>1 and any(value==0 for value in dots[:-1]) and any(value<-.001 for value in dots[1:]):
                    grazing.append(dict(clock=row['sim_ms'],velocity=velocity.tolist(),normals=[n.tolist() for n in normals],normal_velocities=dots))
        result=dict(map=name,seed=seed,prefix_replies=len(prefix),max_prefix_errors=np.max(prefix_errors,axis=0).tolist(),
            endpoint_replay_errors=endpoint_error,
            replies=len(logs),forecast_ambiguity_intervals=forecast['ambiguous_intervals'],
            maximum_errors=np.max([l['errors'] for l in logs],axis=0).tolist(),actual_points=sum(l['score_delta'] for l in logs),
            falls=sum(int(l['outcome']['fell']) for l in logs),native_grazing_examples=grazing,training=False,scored_navigation=False)
        atomic_json(output/'report.json',result)
        print(json.dumps({k:v for k,v in result.items() if k!='native_grazing_examples'}),flush=True)
    except Exception as exc:
        if raw.exists():archive(raw,output/'native.failed.jsonl.gz')
        if journal.exists():atomic_json(output/'decisions.failed.json',[json.loads(line) for line in journal.read_text().splitlines()])
        atomic_json(output/'failed.json',dict(reason=repr(exc),training=False));raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--port',type=int,default=9053);p.add_argument('--grazing-route',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    cases=[('treasure_grazing','reliability_global_treasure_20261012','TreasureBox_Hunt',111108,None),
           ('concentric_short8','reliability_landing_validation_20261012','Concentric_Hunt',151113,8),
           ('fractured_short28','reliability_geometry_pool_20261012','FracturedIslands_Hunt',111108,28)]
    if a.grazing_route:cases=[('treasure_route24','reliability_global_treasure_20261012','TreasureBox_Hunt',111108,None)]
    sources=list(Path(__file__).parent.glob('*.py'))+[ML/'nav/collision/mesh.py',
        ML.parent/'Marble Blast Platinum/platinum/client/scripts/ai/mlAgent.cs',ML.parent/'Marble Blast Platinum/marbleblast_geometry_probe.exe']
    atomic_json(a.output/'prepared.json',dict(cases=cases,grazing_route=a.grazing_route,source_hashes={str(path):sha(path) for path in sources},training=False,scored_navigation=False))
    snapshot_sources(a.output,[path for path in sources if path.suffix.lower() in ('.py','.cs')])
    for label,directory,name,seed,horizon in cases:
        source=ML/'dynamics_reports'/directory/name/str(seed)/'physics_plan'
        actions=json.loads((ML/'dynamics_reports/short_horizon_diagnostics_20261012'/(name+'.h'+str(horizon)+'.forecast.json')).read_text())['proposed'] if horizon else [[0.]*9 for _ in range(24)]
        if a.grazing_route:actions='route24'
        execute(source,actions,a.output/label,a.port)


if __name__=='__main__':main()
