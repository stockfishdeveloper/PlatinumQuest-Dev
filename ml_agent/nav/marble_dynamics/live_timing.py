"""Bounded native command-timing evidence at the intended 64 ms decision interval."""
import argparse
import json
import time
from pathlib import Path

from nav.collision.mesh import atomic_json
from nav.marble_dynamics.record import ML, RUNTIME, game, command, archive, sha
from nav.marble_dynamics.evaluate import load_rows
from nav.protocol import NOOP_ACTION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', default='Concentric_Hunt')
    parser.add_argument('--port', type=int, default=9053)
    parser.add_argument('--powerup', default='none')
    parser.add_argument('--pause-reply-s',type=float,default=0.,help='Pause one reply to verify long-compute lockstep timing')
    parser.add_argument('--output', type=Path, default=ML/'datasets/marble_dynamics/live_timing_20261012')
    args = parser.parse_args()
    dest = args.output/args.map
    if (dest/'manifest.json').exists():
        raise RuntimeError('Use a new output directory to preserve timing evidence')
    dest.mkdir(parents=True, exist_ok=True)
    raw = RUNTIME/(args.map+'.live_timing.jsonl')
    with game(args.map,args.port,obs_ms=64) as (env,proc):
        if args.powerup!='none':
            env.control('GIVEPOW '+args.powerup)
            for _ in range(5):env.step(NOOP_ACTION,repeat=1)
        command(env,'DYNAMICSLIVE 1','dynamics_live')
        for _ in range(6):env.step(NOOP_ACTION,repeat=1)
        command(env,'DYNAMICSRECORD '+raw.as_posix(),'dynamics_record')
        observations=[]
        for i in range(48):
            if i==8 and args.pause_reply_s:time.sleep(args.pause_reply_s)
            # Deliberately distinct world inputs/yaws, including jumps and a
            # partial analog axis. No fitted timing constants are assumed here.
            phases=[(0.,0.,0.,.4,0,0.),(.35,0.,0.,0.,0,.6),
                    (0.,0.,.2,0.,1,-.4),(0.,.3,0.,0.,0,1.2),NOOP_ACTION]
            action=phases[(i//6)%len(phases)]
            use=int(args.powerup!='none' and i==8)
            observations.append(dict(index=i,tick=env.msg.tick,dynamics=env.msg.dynamics,
                                     raw_observation=env.msg.obs.tolist(),joystick=list(action),
                                     use=use,pow_yaw=1.0 if use else None))
            _,info=env.step(action,repeat=1,use_pow=use,pow_yaw=1.0 if use else None)
            observations[-1]['outcome']=info
            if info['fell'] or info['round_ended'] or info['reconnected']:break
        observations.append(dict(index=len(observations),tick=env.msg.tick,dynamics=env.msg.dynamics,
                                 raw_observation=env.msg.obs.tolist(),joystick=None))
        command(env,'DYNAMICSRECORD off','dynamics_record')
        archive(raw,dest/'native.jsonl.gz')
        rows=load_rows(dest/'native.jsonl.gz')
        manifest=dict(map=args.map,obs_ms=64,owned_game_pid=proc.pid,
                      native_transitions=len(rows),observations=observations,
                      raw_sha256=sha(raw),probe_executable_sha256=sha(ML.parent/'Marble Blast Platinum/marbleblast_geometry_probe.exe'),
                      instrumentation_sha256=sha(Path(__file__).with_name('engine_dynamics.inc')),
                      policy_training_started=False)
        atomic_json(dest/'manifest.json',manifest)
        print(json.dumps(dict(map=args.map,powerup=args.powerup,native_transitions=len(rows),captured_observations=len(observations))),flush=True)
        for obs in observations[:3]:
            dyn=obs['dynamics']
            if not dyn:raise RuntimeError('No synchronized native state')
            clock=dyn['native'].get('frame_clock_ms',dyn['native']['state']['clock_ms'])
            matched=[r for r in rows if r['sim_ms']==clock]
            print(json.dumps(dict(index=obs['index'],clock=clock,sent=obs['joystick'],
                 bridge=dyn['bridge_action'],inputs=dyn['input_move'],last=dyn['native']['last_move'],
                 upcoming=[dict(move=r['move'],events=r['control_events']) for r in matched])),flush=True)


if __name__=='__main__':main()
