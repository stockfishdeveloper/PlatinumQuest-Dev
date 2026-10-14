"""Controlled mechanics fixture: teleport onto an existing native Blast item.

The teleport is setup only, not a scored navigation result or inventory grant.
After native pickup, record ordinary controls and the real special Blast impulse.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from nav.collision.mesh import atomic_json
from nav.marble_dynamics.record import ML,RUNTIME,game,command,archive,sha
from nav.protocol import NOOP_ACTION,format_teleport


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--map',default='Concentric_Hunt');p.add_argument('--port',type=int,default=9053)
    p.add_argument('--seed',type=int,default=161114);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__),Path(__file__).with_name('blast.py'),ML.parent/'Marble Blast Platinum/platinum/client/scripts/ai/mlAgent.cs',
        ML.parent/'Marble Blast Platinum/platinum/client/scripts/blast.cs',ML.parent/'Marble Blast Platinum/platinum/client/scripts/default.bind.cs',
        ML.parent/'Marble Blast Platinum/platinum/server/scripts/powerups.cs',ML.parent/'Marble Blast Platinum/platinum/server/scripts/mp/blast.cs']
    atomic_json(a.output/'prepared.json',dict(map=a.map,seed=a.seed,training=False,
        contract='Controlled teleport onto existing visible native Blast item before recording; no score/navigation claim or inventory grant.',
        blast_reply_indices=[0,12,100],jump_reply_indices=[99],decisions=170,source_hashes={str(s):sha(s) for s in sources}))
    for s in sources:
        dest=a.output/'source_snapshot'/s.resolve().relative_to(ML.parent.resolve())
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(s.read_bytes())
    observations=[];raw=RUNTIME/(a.output.name+'.native.jsonl')
    try:
        with game(a.map,a.port,obs_ms=64,seed=a.seed) as (env,proc):
            command(env,'DYNAMICSLIVE 1','dynamics_live')
            for _ in range(8):env.step(NOOP_ACTION,repeat=1)
            candidates=[i for i in env.msg.dynamics['items'] if i['powerup']==7 and not i['gem']]
            if not candidates:raise RuntimeError('No existing Blast item on selected map')
            position=np.asarray(env.msg.dynamics['native']['state']['position'])
            item=min(candidates,key=lambda i:np.linalg.norm(np.asarray(i['position'])-position))
            env.control(format_teleport(*item['position'],spin=(0,0,0)))
            for _ in range(16):
                env.step(NOOP_ACTION,repeat=1)
                if env.msg.dynamics['blast']['special']:break
            else:raise RuntimeError('Existing item did not establish observed special Blast')
            if any(i['id']==item['id'] for i in env.msg.dynamics['items']):raise RuntimeError('Selected item still visible after pickup')
            atomic_json(a.output/'fixture.json',dict(item=item,observed_after_setup=env.msg.dynamics,training=False))
            command(env,'DYNAMICSRECORD '+raw.as_posix(),'dynamics_record')
            for i in range(170):
                action=[0.,0.,0.,0.,int(i==99),.3,0,.3,int(i in (0,12,100))]
                before=copy.deepcopy(env.msg.dynamics)
                _,outcome=env.step(action[:6],repeat=1,use_pow=action[6],pow_yaw=action[7],use_blast=action[8])
                observations.append(dict(index=i,dynamics=before,raw=env.msg.obs.tolist(),action=action,
                    actual_next=copy.deepcopy(env.msg.dynamics),outcome=outcome))
                if outcome['round_ended'] or outcome['reconnected']:raise RuntimeError('Interrupted mechanics fixture')
                if outcome['fell']:break
            command(env,'DYNAMICSRECORD off','dynamics_record')
        archive(raw,a.output/'native.jsonl.gz');atomic_json(a.output/'observations.json',observations)
        atomic_json(a.output/'report.json',dict(samples=len(observations),first_special=observations[0]['dynamics']['blast']['special'],
            falls=sum(int(o['outcome']['fell']) for o in observations),training=False,scored_navigation=False))
        print(json.dumps(dict(samples=len(observations),recorded=True,scored_navigation=False)),flush=True)
    except Exception as exc:
        if raw.exists():archive(raw,a.output/'native.failed.jsonl.gz')
        atomic_json(a.output/'observations.failed.json',observations)
        atomic_json(a.output/'failed.json',dict(reason=repr(exc),training=False));raise


if __name__=='__main__':main()
