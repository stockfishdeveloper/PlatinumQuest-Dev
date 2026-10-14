"""Record real Mega activation/expiry after pickup of an existing native item.

Setup teleport is explicitly a mechanics fixture, never a scored navigation
attempt. No inventory grant, artificial item, fitted model or policy update.
"""
from nav.source_archive import snapshot_sources
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from nav.collision.mesh import atomic_json
from nav.marble_dynamics.record import ML,RUNTIME,game,command,archive,sha
from nav.protocol import NOOP_ACTION,format_teleport,RAW_POW_HELD


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--map',default='Concentric_Hunt')
    p.add_argument('--port',type=int,default=9053);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--physics-frame-ms',type=int,choices=[16,64],default=64)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__),ML.parent/'Marble Blast Platinum/platinum/client/scripts/ai/mlAgent.cs',
        ML.parent/'Marble Blast Platinum/platinum/server/scripts/powerups.cs',
        ML.parent/'Marble Blast Platinum/platinum/server/scripts/mp/gameConnection.cs',
        ML.parent/'Marble Blast Platinum/platinum/server/scripts/mp/ghost.cs']
    atomic_json(a.output/'prepared.json',dict(map=a.map,seed=171115,training=False,decisions=200,
        physics_frame_ms=a.physics_frame_ms,decision_ms=64,
        use_reply_indices=[0],jump_reply_indices=[98],scored_navigation=False,
        setup='Teleport onto existing native item, confirm held6; no grant. Record ordinary use and later shrink.',source_hashes={str(s):sha(s) for s in sources}))
    snapshot_sources(a.output,[path for path in sources if path.suffix.lower() in ('.py','.cs')])
    raw=RUNTIME/(a.output.name+'.native.jsonl');observations=[]
    try:
        with game(a.map,a.port,obs_ms=64,seed=171115) as (env,proc):
            if a.physics_frame_ms==16:env.control('FIXEDSTEP 16 64')
            command(env,'DYNAMICSLIVE 1','dynamics_live')
            for _ in range(8):env.step(NOOP_ACTION,repeat=1)
            items=[i for i in env.msg.dynamics['items'] if i['powerup']==6 and not i['gem']]
            if not items:raise RuntimeError('No existing native Mega item')
            position=np.asarray(env.msg.dynamics['native']['state']['position'])
            item=min(items,key=lambda i:np.linalg.norm(np.asarray(i['position'])-position))
            env.control(format_teleport(*item['position'],spin=(0,0,0)))
            for _ in range(16):
                env.step(NOOP_ACTION,repeat=1)
                if int(env.msg.obs[RAW_POW_HELD])==6:break
            else:raise RuntimeError('Native Mega pickup did not establish held6')
            atomic_json(a.output/'fixture.json',dict(item=item,dynamics=env.msg.dynamics,raw_held_type=int(env.msg.obs[RAW_POW_HELD])))
            command(env,'DYNAMICSRECORD '+raw.as_posix(),'dynamics_record')
            for i in range(200):
                action=[0.,0.,0.,0.,int(i==98),.3,int(i==0),.3,0]
                before=copy.deepcopy(env.msg.dynamics);raw_held=int(env.msg.obs[RAW_POW_HELD]);raw_before=env.msg.obs.tolist()
                _,outcome=env.step(action[:6],repeat=1,use_pow=action[6],pow_yaw=action[7],use_blast=0)
                observations.append(dict(index=i,dynamics=before,action=action,raw_held_before=raw_held,
                    raw_held_after=int(env.msg.obs[RAW_POW_HELD]),raw_before=raw_before,
                    raw_after=env.msg.obs.tolist(),actual_next=copy.deepcopy(env.msg.dynamics),outcome=outcome))
                if outcome['round_ended'] or outcome['reconnected']:raise RuntimeError('Interrupted native Mega fixture')
                if outcome['fell']:break
            command(env,'DYNAMICSRECORD off','dynamics_record')
        archive(raw,a.output/'native.jsonl.gz');atomic_json(a.output/'observations.json',observations)
        changes=[dict(index=i,before=x['dynamics']['native']['state']['radius'],after=x['actual_next']['native']['state']['radius'],
            held_before=x['dynamics']['native']['state']['held'],held_after=x['actual_next']['native']['state']['held'])
            for i,x in enumerate(observations) if x['dynamics']['native']['state']['radius']!=x['actual_next']['native']['state']['radius']]
        report=dict(samples=len(observations),radius_changes=changes,falls=sum(int(o['outcome']['fell']) for o in observations),training=False,scored_navigation=False)
        atomic_json(a.output/'report.json',report);print(json.dumps(report),flush=True)
    except Exception as exc:
        if raw.exists():archive(raw,a.output/'native.failed.jsonl.gz')
        atomic_json(a.output/'observations.failed.json',observations);atomic_json(a.output/'failed.json',dict(reason=repr(exc),training=False));raise


if __name__=='__main__':main()
