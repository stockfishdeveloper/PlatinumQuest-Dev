"""Record ordinary Blast inputs from a settled natural spawn; no inventory grant."""
import argparse
import copy
import json
from pathlib import Path

from nav.collision.mesh import atomic_json
from nav.marble_dynamics.record import ML,RUNTIME,game,command,archive,sha
from nav.protocol import NOOP_ACTION


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--map',default='Concentric_Hunt');p.add_argument('--port',type=int,default=9053)
    p.add_argument('--seed',type=int,default=131110);p.add_argument('--decisions',type=int,default=170)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__),ML.parent/'Marble Blast Platinum/platinum/client/scripts/ai/mlAgent.cs',
             ML.parent/'Marble Blast Platinum/platinum/client/scripts/blast.cs',
             ML.parent/'Marble Blast Platinum/platinum/client/scripts/default.bind.cs']
    atomic_json(a.output/'prepared.json',dict(map=a.map,seed=a.seed,decisions=a.decisions,
        blast_reply_indices=[0,12,100],jump_reply_indices=[99],source_hashes={str(s):sha(s) for s in sources},
        training=False,contract='Actual ordinary inputs at natural spawn, no grants or teleports; not a score comparison.'))
    for s in sources:
        dest=a.output/'source_snapshot'/s.resolve().relative_to(ML.parent.resolve())
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(s.read_bytes())
    raw=RUNTIME/(a.output.name+'.native.jsonl');observations=[]
    with game(a.map,a.port,obs_ms=64,seed=a.seed) as (env,proc):
        command(env,'DYNAMICSLIVE 1','dynamics_live')
        for _ in range(8):env.step(NOOP_ACTION,repeat=1)
        command(env,'DYNAMICSRECORD '+raw.as_posix(),'dynamics_record')
        for i in range(a.decisions):
            action=[0.,0.,0.,0.,int(i==99),.3,0,.3,int(i in (0,12,100))]
            observations.append(dict(index=i,dynamics=copy.deepcopy(env.msg.dynamics),
                raw=env.msg.obs.tolist(),action=action))
            _,info=env.step(action[:6],repeat=1,use_pow=action[6],pow_yaw=action[7],use_blast=action[8])
            if info['round_ended'] or info['reconnected']:raise RuntimeError('Interrupted Blast probe')
        command(env,'DYNAMICSRECORD off','dynamics_record')
    archive(raw,a.output/'native.jsonl.gz')
    atomic_json(a.output/'observations.json',observations)
    print(json.dumps(dict(samples=len(observations),native_sha256=sha(a.output/'native.jsonl.gz'),training=False)),flush=True)


if __name__=='__main__':main()
