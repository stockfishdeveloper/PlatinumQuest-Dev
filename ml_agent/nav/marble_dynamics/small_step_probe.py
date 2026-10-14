"""Collect native 16ms physics frames inside ordinary 64ms decisions.

Separate timing/physics fixture. No checkpoint, model fitting, reset during play,
inventory grant or future-native input to prediction. Existing default bridge
supports 64ms physics only; this measures the smaller-frame contract first.
"""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path

from nav.collision.mesh import atomic_json
from nav.marble_dynamics.bridge import reply
from nav.marble_dynamics.evaluate import load_rows
from nav.marble_dynamics.record import ML,RUNTIME,archive,command,game,sha
from nav.protocol import NOOP_ACTION,RAW_SCORE


def controls():
    result=[]
    for i in range(32):
        if i<10:action=reply([.6,0,0,.2,int(i==3),.3])
        elif i<20:action=reply([0,.4,.3,0,int(i==14),-1.1])
        elif i<26:action=reply([.2,0,0,.5,0,2.2])
        else:action=reply(NOOP_ACTION)
        result.append(action)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--port',type=int,default=9055);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    maps=['Concentric_Hunt','KingOfTheMarble_Hunt','Bowl_Hunt'];actions=controls();results=[]
    sources=list(Path(__file__).parent.glob('*.py'))+[ML/'nav/collision/mesh.py',
        ML.parent/'Marble Blast Platinum/platinum/client/scripts/ai/mlAgent.cs',
        ML.parent/'Marble Blast Platinum/marbleblast_geometry_probe.exe']
    hashes={str(path):sha(path) for path in sources}
    atomic_json(a.output/'prepared.json',dict(maps=maps,physics_frame_ms=16,decision_ms=64,
        proposal_inputs=actions,source_hashes=hashes,training=False,scored_navigation=False,
        declaration='Bounded native timing/control fixtures on natural spawns; no future native input to prediction.'))
    for path in sources:
        saved=a.output/'source_snapshot'/path.resolve().relative_to(ML.parent.resolve())
        saved.parent.mkdir(parents=True,exist_ok=True);saved.write_bytes(path.read_bytes())
    for index,name in enumerate(maps):
        if any(sha(Path(path))!=fingerprint for path,fingerprint in hashes.items()):
            raise RuntimeError('Source changed during timing collection')
        folder=a.output/name;folder.mkdir();raw=RUNTIME/(a.output.name+'_'+name+'.native.jsonl')
        logs=[]
        try:
            with game(name,a.port,obs_ms=16,seed=211119+index) as (env,proc):
                command(env,'DYNAMICSLIVE 1','dynamics_live')
                env.obs_ms=64;env.control('FIXEDSTEP 16 64')
                for _ in range(8):env.step(NOOP_ACTION,repeat=1)
                command(env,'DYNAMICSDETAIL 1','dynamics_detail')
                command(env,'DYNAMICSRECORD '+raw.as_posix(),'dynamics_record')
                initial=copy.deepcopy(env.msg.dynamics);score=float(env.msg.obs[RAW_SCORE])
                atomic_json(folder/'initial.json',dict(dynamics=initial,raw_observation=env.msg.obs.tolist()))
                for i,action in enumerate(actions):
                    before=copy.deepcopy(env.msg.dynamics);raw_obs=env.msg.obs.tolist()
                    _,outcome=env.step(action[:6],repeat=1,use_pow=int(action[6]),pow_yaw=action[7] if action[6] else None,use_blast=int(action[8]))
                    logs.append(dict(index=i,dynamics=before,raw_observation=raw_obs,action=action,
                        actual_next=copy.deepcopy(env.msg.dynamics['native']['state']),
                        next_frame_clock_ms=env.msg.dynamics['native']['frame_clock_ms'],outcome=outcome,
                        score_delta=float(env.msg.obs[RAW_SCORE])-score))
                    score=float(env.msg.obs[RAW_SCORE])
                    if any(outcome[k] for k in ['fell','round_ended','reconnected']):break
                command(env,'DYNAMICSRECORD off','dynamics_record')
                command(env,'DYNAMICSDETAIL 0','dynamics_detail')
            archive(raw,folder/'native.jsonl.gz');atomic_json(folder/'decisions.json',logs)
            first=initial['native']['frame_clock_ms'];last=logs[-1]['next_frame_clock_ms']
            native=[row for row in load_rows(folder/'native.jsonl.gz') if first<=row['sim_ms']<last]
            report=dict(map=name,replies=len(logs),actual_points=sum(l['score_delta'] for l in logs),
                falls=sum(int(l['outcome']['fell']) for l in logs),native_transitions=len(native),
                native_dt_ms=dict(Counter(row['dt_ms'] for row in native)),
                observation_intervals_ms=[l['next_frame_clock_ms']-l['dynamics']['native']['frame_clock_ms'] for l in logs],
                native_sha256=sha(folder/'native.jsonl.gz'),training=False,scored_navigation=False)
            atomic_json(folder/'report.json',report);results.append(report)
            atomic_json(a.output/'summary.json',dict(cases=results,training=False,scored_navigation=False))
            print(json.dumps(report),flush=True)
        except Exception as exc:
            if raw.exists():archive(raw,folder/'native.failed.jsonl.gz')
            atomic_json(folder/'decisions.failed.json',logs)
            atomic_json(folder/'failed.json',dict(reason=repr(exc),training=False));raise


if __name__=='__main__':main()
