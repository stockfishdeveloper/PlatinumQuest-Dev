"""Owned general-physics trials; actual inputs/states recorded by the native probe.

python -m nav.marble_dynamics.record --maps Sprawl_Hunt Concentric_Hunt --trials 40
No policy is loaded. Runtime files are outside OneDrive; evidence is atomically archived.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from nav.collision.mesh import CollisionMap, atomic_json
from nav.env import HuntEnv
from nav.protocol import NOOP_ACTION, RAW_POS, RAW_VEL, format_teleport
from nav.joystick import action_to_joystick

ROOT = Path(__file__).resolve().parents[3]
ML = ROOT/'ml_agent'
DEFAULT_OUTPUT = ML/'datasets/marble_dynamics/pilot_20261011'
RUNTIME = Path(os.environ.get('TEMP', '.'))/'pq_checks/marble_dynamics_20261011'
FAMILIES = ('roll', 'switch', 'brake', 'jump', 'drop_spin', 'wall_impact', 'edge_timing', 'ceiling_impact', 'powerup')
OPTIONAL_FAMILIES = ('effect_expiry',)
POWERUPS = ('SuperJumpItem','SuperSpeedItem','SuperBounceItem','ShockAbsorberItem','HelicopterItem','MegaMarbleItem')
DEV = {'Concentric_Hunt', 'CubeIsle_Hunt'}
TEST = {'Bowl_Hunt', 'VortexEffect_Hunt'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def archive(path, output):
    tmp=output.with_suffix(output.suffix+'.tmp')
    tmp.write_bytes(gzip.compress(path.read_bytes(),mtime=0))
    os.replace(str(tmp),str(output))


def command(env, text, ack=None):
    env.debug=None
    env.control(text)
    if ack:
        for _ in range(32):
            if env.debug and env.debug[0]==ack:
                if len(env.debug)<2 or env.debug[1]!='1':
                    raise RuntimeError('Probe command refused: '+text+' '+str(env.debug))
                return
            _, inf=env.step(NOOP_ACTION,repeat=1)
            if inf['round_ended'] or inf['reconnected']:
                raise RuntimeError('Round ended during command: '+text)
        raise RuntimeError('No acknowledgement: '+text)


@contextmanager
def game(name, port, obs_ms=16, seed=None, process_cap=None, random_start=False, watch=False):
    RUNTIME.mkdir(parents=True,exist_ok=True)
    exe=ROOT/'Marble Blast Platinum/marbleblast_geometry_probe.exe'
    env=HuntEnv(port,speed=3,log=lambda s: print(s,flush=True))
    env.obs_ms=obs_ms
    env.srv.settimeout(70)
    # Operator viewing (watch): visible window with the free camera (-aifreecam,
    # every 1x viewing launch); otherwise owned probes stay hidden.
    startup=None if watch else subprocess.STARTUPINFO()
    if not watch:
        startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow=0
    # Parallel owned probes may play the same map. Keep their native console
    # evidence separate rather than truncating/interleaving one shared file.
    log=(RUNTIME/(name+'.port'+str(port)+'.seed'+str(seed)+'.game.log')).open('wb')
    proc=None
    try:
        if seed is not None and not 0<int(seed)<=2147383647:raise ValueError('Seed must be a positive native signed integer with round headroom')
        arguments=[str(exe),'-autotrain',name,'-aiport',str(port),'-offline','-aicompetitive','0']
        if seed is not None:arguments+=['-aiseed',str(seed)]
        if random_start:
            if seed is None:raise ValueError('Randomized Hunt opening requires an explicit seed')
            arguments+=['-airandomstart']
        if watch:arguments+=['-aifreecam']
        if process_cap is None:
            proc=subprocess.Popen(arguments,cwd=str(exe.parent),startupinfo=startup,stdout=log,stderr=log)
        else:
            from nav.marble_dynamics.static_cache import guarded_launch
            proc=guarded_launch(arguments,process_cap,cwd=str(exe.parent),startupinfo=startup,stdout=log,stderr=log)
        env.connect()
        env.conn.settimeout(35)
        if env.info.get('mission')!=name:
            raise RuntimeError('Unexpected mission: '+str(env.info))
        env.control('SPINWAIT 4')
        # Explicit owned physics probes can spend minutes in a conservative
        # cold search. Keep production's default wait unchanged and prevent
        # silent free-running while this fixture is still computing a reply.
        command(env,'DYNAMICSWAIT 600000','dynamics_wait')
        env.control('FIXEDSTEP '+str(obs_ms))
        env.control('CONTACT 1')
        for _ in range(400):
            env.step(NOOP_ACTION,repeat=1)
        yield env,proc
    finally:
        env.close()
        if proc:
            proc.terminate()
            try: proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill();proc.wait(timeout=10)
        log.close()


class Sampler:
    def __init__(self, mesh, seed):
        self.mesh=mesh;self.rng=np.random.RandomState(seed)
        self.normals=mesh.normals
        tri=mesh.triangles
        self.centers=tri.mean(axis=1)
        self.area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)*.5
        items=np.asarray([a['position'] for a in mesh.info['mission_annotations'] if a['class_name']=='Item'])
        self.play_lo=items.min(axis=0)-8 if len(items) else mesh.bounds[0]
        self.play_hi=items.max(axis=0)+8 if len(items) else mesh.bounds[1]
        self.regional=np.all((self.centers[:,:2]>=self.play_lo[:2])&(self.centers[:,:2]<=self.play_hi[:2]),axis=1)
        self.regional &= self.centers[:,2]<=self.play_hi[2]

    def sample(self, family, number):
        r=self.rng;m=self.mesh
        if family in ('wall_impact','edge_timing'):
            mask=np.abs(self.normals[:,2])<.15
        elif family=='ceiling_impact':
            mask=self.normals[:,2]<-.6
        elif family=='effect_expiry':
            mask=self.normals[:,2]>.999
        else:
            mask=self.normals[:,2]>.45
        ids=np.flatnonzero(mask&self.regional&(self.area>.2))
        if not len(ids):return None
        weights=np.sqrt(self.area[ids]);weights/=weights.sum()
        for _ in range(180):
            idx=int(r.choice(ids,p=weights));n=self.normals[idx].astype(float)
            weights2=r.uniform(.2,1,3);weights2/=weights2.sum()
            point=weights2 @ m.triangles[idx]
            heading=r.uniform(-math.pi,math.pi)
            direction=np.array([math.cos(heading),math.sin(heading),0.])
            speed=float(r.choice([0.,3.,8.,14.,22.,32.]))
            radius=.18975
            pos=point+n*(radius+.015)
            vel=direction*speed
            spin=np.cross(n,vel)/radius
            if family in ('roll','switch','brake','jump'):
                vel-=n*np.dot(n,vel)
                spin=np.cross(n,vel)/radius
                spin=spin*r.choice([0.,.5,1.,1.5]) + r.normal(0,8,3)
            elif family=='drop_spin':
                pos[2]+=r.uniform(.5,4.)
                vel[2]=-r.uniform(0,12)
                spin=r.normal(0,r.choice([0.,30.,90.,180.]),3)
            elif family in ('wall_impact','edge_timing','ceiling_impact'):
                pos=point+n*(radius+r.uniform(.15,1.5))
                vel=-n*r.uniform(6,28)+direction*r.uniform(0,6)
                spin=r.normal(0,r.choice([0.,30.,100.,200.]),3)
                if family=='edge_timing':
                    floor=m.raycast(pos+np.array([0,0,2.]),pos-np.array([0,0,30.]),cull_backfaces=True)
                    if not floor or floor['normal'][2]<.5:continue
                    pos[2]=floor['point'][2]+radius+.015
                    vel[2]=0
                direction=-n.copy();direction[2]=0
                if np.linalg.norm(direction[:2])>0:direction/=np.linalg.norm(direction[:2])
            elif family in ('powerup','effect_expiry'):
                vel-=n*np.dot(n,vel)
                spin=np.cross(n,vel)/radius
                if family=='effect_expiry':
                    pos[2]+=2.;vel=np.asarray([0.,0.,-3.]);spin=np.asarray([0.,0.,60.])
            if len(m.sphere_contacts(pos,radius+.002)['triangle_ids']):continue
            return dict(position=pos.tolist(),velocity=vel.tolist(),spin=spin.tolist(),radius=radius,
                        target_point=point.tolist(),target_normal=n.tolist(),triangle_id=idx,
                        heading=math.atan2(direction[1],direction[0]),family=family,
                        jump_frame=int(r.randint(0,14)),turn_frame=int(r.randint(10,35)),
                        powerup=POWERUPS[number%len(POWERUPS)] if family in ('powerup','effect_expiry') else 'none',
                    use_frame=12,frames=(750 if number%6==5 else 375) if family in ('powerup','effect_expiry') and number%6>=2 else 200)
        return None


def reset(env, spec):
    command(env,'DYNAMICSRECORD off','dynamics_record')
    env.step(NOOP_ACTION,repeat=3)
    env.control('RESPAWN')
    env.step(NOOP_ACTION,repeat=12)
    env.control('GIVEPOW none')
    env.step(NOOP_ACTION,repeat=4)
    if spec['powerup']!='none':
        env.control('GIVEPOW '+spec['powerup']);env.step(NOOP_ACTION,repeat=4)
    env.control(format_teleport(*spec['position'],*spec['velocity'],spin=spec['spin']))
    # Drain the two-observation bridge pipeline while holding a neutral action.
    env.step(NOOP_ACTION,repeat=2)
    return env.msg.obs


def run_trial(env, spec, path):
    start=reset(env,spec)
    if np.linalg.norm(start[RAW_POS]-spec['position'])>.06 or np.linalg.norm(start[RAW_VEL]-spec['velocity'])>.15:
        return dict(valid=False,reason='fixture_state_mismatch',observed_position=start[RAW_POS].tolist(),observed_velocity=start[RAW_VEL].tolist())
    command(env,'DYNAMICSRECORD '+path.as_posix(),'dynamics_record')
    sent=[];fell=False
    for frame in range(spec['frames']):
        heading=spec['heading'];thr=0.;jump=0
        if spec['family'] in ('roll','switch','jump','edge_timing','powerup'):
            thr=1.
        if spec['family']=='switch' and frame>=spec['turn_frame']:heading+=math.pi*.5
        if spec['family']=='brake':
            v=env.msg.obs[RAW_VEL]
            heading=math.atan2(-v[1],-v[0]);thr=1.
        if spec['family'] in ('jump','edge_timing') and frame==spec['jump_frame']:jump=1
        dx,dy=math.cos(heading),math.sin(heading)
        v=env.msg.obs[RAW_VEL]
        joystick=action_to_joystick(dx,dy,thr,jump,0,v[0],v[1]) if thr or jump else NOOP_ACTION
        use=int(spec['powerup']!='none' and frame==spec['use_frame'])
        msg,inf=env.step(joystick,repeat=1,use_pow=use,pow_yaw=math.atan2(dx,dy) if use else None)
        sent.append(dict(frame=frame,joystick=list(joystick),use=use,
                         pow_yaw=math.atan2(dx,dy) if use else None,
                         camera_yaw=joystick[5] if len(joystick)>5 else 0.,
                         observation=msg.obs[:6].tolist(),oob=bool(inf['fell'])))
        if inf['round_ended'] or inf['reconnected']:
            raise RuntimeError('Round changed inside an owned bounded trial')
        if inf['fell']:
            fell=True;break
    command(env,'DYNAMICSRECORD off','dynamics_record')
    records=[json.loads(line) for line in path.read_text().splitlines()]
    rows=[r for r in records if r['kind']=='transition']
    if not records or records[0]['kind']!='header' or records[-1]['kind']!='complete' or records[-1]['samples']!=len(rows):
        raise RuntimeError('Incomplete native recording')
    if not rows:return dict(valid=False,reason='no_native_samples')
    if any(value is None for value in rows[0]['before']['parameters'].values()):
        raise RuntimeError('Missing native physical parameter')
    external=[]
    for i,(a,b) in enumerate(zip(rows,rows[1:])):
        for key in ('position','velocity','spin','radius','gravity_direction'):
            delta=np.linalg.norm(np.asarray(a['after'][key])-b['before'][key])
            if delta>1e-5:external.append(dict(index=i+1,field=key,error=float(delta)))
        for key in ('datablock','parameters'):
            if a['after'][key]!=b['before'][key]:external.append(dict(index=i+1,field=key,error=1.))
    return dict(valid=True,samples=len(rows),fell=fell,external_state_changes=external,
                native_dt_ms=sorted(set(r['dt_ms'] for r in rows)),sent_controls=sent)


def collect_map(name, args, seed):
    mesh=CollisionMap(ML/'collision_maps'/(name+'.npz'))
    sampler=Sampler(mesh,seed)
    dest=args.output/name;dest.mkdir(parents=True,exist_ok=True)
    manifest=dict(schema='marble_trajectories_v1',map=name,geometry_sha256=mesh.info['mesh_sha256'],
                  split='dev' if name in DEV else 'test' if name in TEST else 'train',
                  seed=seed,probe_executable_sha256=sha(ROOT/'Marble Blast Platinum/marbleblast_geometry_probe.exe'),
                  native_instrumentation_sha256=sha(Path(__file__).with_name('engine_dynamics.inc')),
                  script_sha256=sha(ROOT/'Marble Blast Platinum/platinum/client/scripts/ai/mlAgent.cs'),
                  trials=[],replay_checks=[])
    work=RUNTIME/name;work.mkdir(parents=True,exist_ok=True)
    with game(name,args.port) as (env,proc):
        manifest['owned_game_pid']=proc.pid
        # Ensure original mode/table parameters match the representation's intended game settings.
        for number in range(args.trials):
            family=args.families[number%len(args.families)]
            spec=sampler.sample(family,number//len(args.families))
            if spec is None:
                manifest['trials'].append(dict(id=number,family=family,valid=False,reason='no_clear_sampled_start'))
                continue
            raw=work/(f'trial_{number:04d}.jsonl')
            result=run_trial(env,spec,raw)
            row=dict(id=number,family=family,spec=spec,**result)
            if result['valid']:
                archive(raw,dest/(raw.name+'.gz'))
                row['file']=raw.name+'.gz';row['raw_sha256']=sha(raw)
                # Repeat varied ordinary/contact trials, preserving both runs for comparison.
                if number<args.replays:
                    repeat=work/(f'trial_{number:04d}.repeat.jsonl')
                    repeated=run_trial(env,spec,repeat)
                    if not repeated['valid']:
                        manifest['replay_checks'].append(dict(id=number,valid=False,reason=repeated.get('reason')))
                        manifest['trials'].append(row)
                        atomic_json(dest/'manifest.json',manifest)
                        continue
                    archive(repeat,dest/(repeat.name+'.gz'))
                    a=[r for r in map(json.loads,raw.read_text().splitlines()) if r['kind']=='transition']
                    b=[r for r in map(json.loads,repeat.read_text().splitlines()) if r['kind']=='transition']
                    length=min(len(a),len(b))
                    errors={key:max((float(np.linalg.norm(np.asarray(x['after'][key])-y['after'][key])) for x,y in zip(a[:length],b[:length])),default=None) for key in ('position','velocity','spin')}
                    manifest['replay_checks'].append(dict(id=number,valid=repeated['valid'],matched_samples=length,
                                                        same_length=len(a)==len(b),max_errors=errors,file=repeat.name+'.gz'))
            manifest['trials'].append(row)
            atomic_json(dest/'manifest.json',manifest)
            print(json.dumps(dict(map=name,trial=number,family=family,valid=result['valid'],samples=result.get('samples'),
                                  external_changes=len(result.get('external_state_changes',[])))),flush=True)
    valid=[r for r in manifest['trials'] if r['valid']]
    manifest['summary']=dict(valid_trials=len(valid),attempted=len(manifest['trials']),
                             native_transitions=sum(r['samples'] for r in valid),families=dict(Counter(r['family'] for r in valid)))
    atomic_json(dest/'manifest.json',manifest)
    return manifest['summary']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--maps',nargs='+',default=['Sprawl_Hunt','Concentric_Hunt'])
    p.add_argument('--trials',type=int,default=36)
    p.add_argument('--families',nargs='+',choices=FAMILIES+OPTIONAL_FAMILIES,default=list(FAMILIES))
    p.add_argument('--replays',type=int,default=8)
    p.add_argument('--port',type=int,default=9053)
    p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    # These roles are assigned before sampling; repeats stay in their source trajectory's role.
    atomic_json(args.output/'split_contract.json',dict(schema='marble_split_v1',dev=sorted(DEV),test=sorted(TEST),
                train='all remaining configured maps',scope='held out from this new dynamics experiment; prior policies may have seen them'))
    results={}
    for i,name in enumerate(args.maps):
        results[name]=collect_map(name,args,20261011+i)
        atomic_json(args.output/'collection.json',dict(maps=results,policy_training_started=False))
    print(json.dumps(results),flush=True)


if __name__=='__main__':main()
