"""Native replay of a16ms archived prefix, then general feedback braking.

Setup restoration only; no grant, reset, rescue or future native prediction input.
This can expose a real fall that an earlier bounded episode did not play through.
"""
from nav.source_archive import snapshot_sources
import argparse
import copy
import json
from pathlib import Path
import numpy as np

from nav.collision.mesh import CollisionMap,atomic_json
from nav.marble_dynamics.record import ML,RUNTIME,game,command,archive,sha
from nav.marble_dynamics.commit_demo import restore
from nav.marble_dynamics.physics import GeneralPhysics
from nav.marble_dynamics.bridge import predict_replies
from nav.marble_dynamics.evaluate import errors
from nav.marble_dynamics.events import external_changes
from nav.marble_dynamics.gem_demo import Proposal
from nav.marble_dynamics.reliability_eval import pending_reply_matches
from nav.protocol import NOOP_ACTION,RAW_SCORE


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--port',type=int,default=9055)
    p.add_argument('--replies',type=int,default=96);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    report=json.loads((a.source/'report.json').read_text());prefix=json.loads((a.source/'decisions.json').read_text())
    terminal_path=next((a.source/name for name in ('terminal_search.json','final_context.json') if (a.source/name).exists()),None)
    terminal=json.loads(terminal_path.read_text()) if terminal_path else None
    fixture=json.loads((a.source.parent/'fixture.json').read_text())
    name=report['map'];seed=report['seed'];logs=[];prefix_errors=[];raw=RUNTIME/(a.output.name+'.native.jsonl')
    sources=list(Path(__file__).parent.glob('*.py'))+[ML/'nav/collision/mesh.py',ML.parent/'Marble Blast Platinum/platinum/client/scripts/ai/mlAgent.cs',ML.parent/'Marble Blast Platinum/marbleblast_geometry_probe.exe']
    atomic_json(a.output/'prepared.json',dict(source=str(a.source),source_decisions_sha256=sha(a.source/'decisions.json'),
        source_hashes={str(path):sha(path) for path in sources},replies=a.replies,physics_frame_ms=16,decision_ms=64,
        terminal_context_source=str(terminal_path) if terminal_path else 'independently observed after exact prefix replay',
        training=False,scored_navigation=False,contract='Restore archived world BEFORE exact prefix; record general predicted-feedback brake continuation. No in-play grants/reset/rescue; future native states are assertions only.'))
    snapshot_sources(a.output,[path for path in sources if path.suffix.lower() in ('.py','.cs')])
    engine=GeneralPhysics(CollisionMap(ML/'collision_maps'/(name+'.npz')));engine.bridge_physics_ms=16
    try:
        with game(name,a.port,obs_ms=64,seed=seed) as (env,proc):
            env.control('FIXEDSTEP 16 64');command(env,'DYNAMICSLIVE 1','dynamics_live');env.control('REPLAYCAPTURE 1')
            for _ in range(8):env.step(NOOP_ACTION,repeat=1)
            command(env,'DYNAMICSDETAIL 1','dynamics_detail');command(env,'DYNAMICSRECORD '+raw.as_posix(),'dynamics_record')
            match=restore(env,fixture['world'])
            if match['position_error_u']>1e-5 or not match['score_equal']:raise RuntimeError('Initial world restore mismatch')
            for i,record in enumerate(prefix):
                action=record['action'];_,outcome=env.step(action[:6],repeat=1,use_pow=int(action[6]),pow_yaw=action[7] if action[6] else None,use_blast=int(action[8]))
                difference=errors(env.msg.dynamics['native']['state'],record['actual_next']);prefix_errors.append(difference)
                if difference[0]>1e-4 or difference[1]>1e-3 or outcome['fell']:raise RuntimeError('Prefix replay drift '+str((i,difference)))
            snapshot=copy.deepcopy(env.msg.dynamics)
            reference=terminal['dynamics']['native']['state'] if terminal else prefix[-1]['actual_next']
            difference=errors(snapshot['native']['state'],reference)
            if any(x>limit for x,limit in zip(difference,[1e-4,1e-3,1e-3])):raise RuntimeError('Terminal pose/spin drift '+str(difference))
            if external_changes(reference,snapshot['native']['state']):raise RuntimeError('Terminal physical/resource context differs')
            if terminal and snapshot['native']['pending_move']!=terminal['dynamics']['native']['pending_move']:raise RuntimeError('Pending native Move differs')
            if not pending_reply_matches(snapshot['bridge_action'],prefix[-1]['action']):raise RuntimeError('Terminal reply differs from replayed prefix')
            target=(terminal.get('target') if terminal else None) or prefix[-1]['gem']
            forecast=predict_replies(engine,snapshot,Proposal(target[:3],steps=a.replies,brake=True))
            atomic_json(a.output/'initial.json',dict(dynamics=snapshot,raw_observation=env.msg.obs.tolist()))
            atomic_json(a.output/'forecast.json',forecast)
            if forecast['failure']:raise RuntimeError('Forecast failed '+str(forecast['failure']))
            score=float(env.msg.obs[RAW_SCORE])
            for i,action in enumerate(forecast['proposed']):
                before=copy.deepcopy(env.msg.dynamics);_,outcome=env.step(action[:6],repeat=1,use_pow=int(action[6]),pow_yaw=action[7] if action[6] else None,use_blast=int(action[8]))
                actual=copy.deepcopy(env.msg.dynamics['native']['state']);predicted=forecast['states'][i+1]
                entry=dict(index=i,dynamics=before,action=action,actual_next=actual,predicted_next=predicted,
                    errors=errors(predicted,actual),score_delta=float(env.msg.obs[RAW_SCORE])-score,outcome=outcome)
                logs.append(entry);score+=entry['score_delta'];atomic_json(a.output/'decisions.partial.json',logs)
                if any(outcome[k] for k in ['fell','round_ended','reconnected']):break
            command(env,'DYNAMICSRECORD off','dynamics_record');command(env,'DYNAMICSDETAIL 0','dynamics_detail')
        archive(raw,a.output/'native.jsonl.gz');atomic_json(a.output/'decisions.json',logs)
        result=dict(map=name,seed=seed,prefix_replies=len(prefix),maximum_prefix_errors=np.max(prefix_errors,axis=0).tolist(),
            endpoint_replay_errors=difference,replies=len(logs),actual_points=sum(l['score_delta'] for l in logs),
            falls=sum(int(l['outcome']['fell']) for l in logs),maximum_errors=np.max([l['errors'] for l in logs],axis=0).tolist(),
            forecast_ambiguity_intervals=forecast['ambiguous_intervals'],training=False,scored_navigation=False)
        atomic_json(a.output/'report.json',result);print(json.dumps(result),flush=True)
    except Exception as exc:
        if raw.exists():archive(raw,a.output/'native.failed.jsonl.gz')
        atomic_json(a.output/'decisions.failed.json',logs);atomic_json(a.output/'failed.json',dict(reason=repr(exc),training=False));raise


if __name__=='__main__':main()
