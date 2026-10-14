"""Native small-frame powerup/Blast actuation fixtures after existing-item pickup.

Teleport is labeled setup only. Powers present in the hunt pool use existing
items; Bounce/Shock (absent from all ten maps) use explicit fixture-only grants
BEFORE recording. No reset/rescue/grant during recording, checkpoint or fitting.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from nav.collision.mesh import atomic_json
from nav.marble_dynamics.record import ML,RUNTIME,archive,command,game,sha
from nav.protocol import NOOP_ACTION,RAW_SCORE,format_teleport


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--port',type=int,default=9055)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);results=[]
    cases=[(1,'Bowl_Hunt'),(2,'Concentric_Hunt'),(5,'Bowl_Hunt'),
        ('regular_blast','Concentric_Hunt'),('special_blast','Concentric_Hunt'),
        (3,'Bowl_Hunt'),(4,'Bowl_Hunt')]
    sources=list(Path(__file__).parent.glob('*.py'))+[ML/'nav/collision/mesh.py',
        ML.parent/'Marble Blast Platinum/platinum/client/scripts/ai/mlAgent.cs',
        ML.parent/'Marble Blast Platinum/platinum/client/scripts/blast.cs',
        ML.parent/'Marble Blast Platinum/platinum/server/scripts/powerups.cs',
        ML.parent/'Marble Blast Platinum/marbleblast_geometry_probe.exe']
    hashes={str(path):sha(path) for path in sources}
    coverage={}
    for path in (ML/'collision_maps').glob('*_Hunt.json'):
        info=json.loads(path.read_text());coverage[path.stem]=sorted({i['datablock'] for i in info['mission_annotations'] if i['class_name']=='Item' and 'Gem' not in i['datablock']})
    atomic_json(a.output/'prepared.json',dict(cases=cases,physics_ms=16,decision_ms=64,replies=100,
        source_hashes=hashes,training=False,scored_navigation=False,
        mission_resource_coverage=coverage,fixture_only_grants={3:'SuperBounceItem',4:'ShockAbsorberItem'},
        declaration='Existing items for Super Jump, Speed, Helicopter, item-charged Blast; teleport and actual pickup BEFORE recording. Regular Blast natural meter. Bounce/Shock absent from hunt pool: explicit fixture-only GIVEPOW BEFORE recording. No grants/in-play resets/rescue during recorded mechanics; no scored navigation claim.'))
    for path in sources:
        saved=a.output/'source_snapshot'/path.resolve().relative_to(ML.parent.resolve())
        saved.parent.mkdir(parents=True,exist_ok=True);saved.write_bytes(path.read_bytes())
    for index,(power,map_name) in enumerate(cases):
        if any(sha(Path(p))!=fingerprint for p,fingerprint in hashes.items()):raise RuntimeError('Batch source changed')
        folder=a.output/str(power);folder.mkdir();raw=RUNTIME/(a.output.name+'_'+str(power)+'.native.jsonl');logs=[]
        try:
            with game(map_name,a.port,obs_ms=16,seed=221120+index) as (env,proc):
                command(env,'DYNAMICSLIVE 1','dynamics_live');env.obs_ms=64;env.control('FIXEDSTEP 16 64')
                for _ in range(8):env.step(NOOP_ACTION,repeat=1)
                item=None
                grant={3:'SuperBounceItem',4:'ShockAbsorberItem'}.get(power)
                atomic_json(folder/'native_item_catalog.json',env.msg.dynamics['items'])
                if grant:
                    env.control('GIVEPOW '+grant)
                    for _ in range(16):
                        env.step(NOOP_ACTION,repeat=1)
                        if env.msg.dynamics['native']['state']['held']==power:break
                    else:raise RuntimeError('Fixture-only native inventory not confirmed')
                elif isinstance(power,int) or power=='special_blast':
                    item_type=7 if power=='special_blast' else power
                    items=[i for i in env.msg.dynamics['items'] if i['powerup']==item_type and not i['gem']]
                    if not items:raise RuntimeError('No existing visible powerup '+str(power))
                    position=np.asarray(env.msg.dynamics['native']['state']['position'])
                    item=min(items,key=lambda i:np.linalg.norm(np.asarray(i['position'])-position))
                    env.control(format_teleport(*item['position'],spin=(0,0,0)))
                    for _ in range(16):
                        env.step(NOOP_ACTION,repeat=1)
                        if (env.msg.dynamics['blast']['special'] if power=='special_blast' else env.msg.dynamics['native']['state']['held']==power):break
                    else:raise RuntimeError('Existing item pickup not confirmed')
                    if any(i['id']==item['id'] for i in env.msg.dynamics['items']):raise RuntimeError('Item still visible')
                elif env.msg.dynamics['blast']['special']:
                    raise RuntimeError('Natural Blast unexpectedly item-charged')
                command(env,'DYNAMICSDETAIL 1','dynamics_detail')
                command(env,'DYNAMICSRECORD '+raw.as_posix(),'dynamics_record')
                atomic_json(folder/'fixture.json',dict(existing_item=item,fixture_only_grant=grant,dynamics=env.msg.dynamics,setup_only=True))
                score=float(env.msg.obs[RAW_SCORE])
                for i in range(100):
                    action=[0.,0.,0.,0.,int(i==45),.3 if i<32 else -1.1,
                        int(isinstance(power,int) and i==0),.7,
                        int(power in ('regular_blast','special_blast') and i in [0,12,90])]
                    before=copy.deepcopy(env.msg.dynamics)
                    _,outcome=env.step(action[:6],repeat=1,use_pow=int(action[6]),pow_yaw=action[7] if action[6] else None,use_blast=int(action[8]))
                    logs.append(dict(index=i,dynamics=before,action=action,actual_next=copy.deepcopy(env.msg.dynamics),
                        raw_observation=env.msg.obs.tolist(),score_delta=float(env.msg.obs[RAW_SCORE])-score,outcome=outcome))
                    score=float(env.msg.obs[RAW_SCORE])
                    if any(outcome[k] for k in ['fell','round_ended','reconnected']):break
                command(env,'DYNAMICSRECORD off','dynamics_record');command(env,'DYNAMICSDETAIL 0','dynamics_detail')
            archive(raw,folder/'native.jsonl.gz');atomic_json(folder/'observations.json',logs)
            report=dict(map=map_name,power=power,fixture_only_grant=grant,replies=len(logs),actual_points=sum(l['score_delta'] for l in logs),
                falls=sum(int(l['outcome']['fell']) for l in logs),native_sha256=sha(folder/'native.jsonl.gz'),
                training=False,scored_navigation=False)
            atomic_json(folder/'report.json',report);results.append(report)
            atomic_json(a.output/'summary.json',dict(cases=results,training=False,scored_navigation=False))
            print(json.dumps(report),flush=True)
        except Exception as exc:
            if raw.exists():archive(raw,folder/'native.failed.jsonl.gz')
            atomic_json(folder/'observations.failed.json',logs)
            atomic_json(folder/'failed.json',dict(reason=repr(exc),training=False));raise


if __name__=='__main__':main()
