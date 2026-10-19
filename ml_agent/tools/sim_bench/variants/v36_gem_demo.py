"""Bounded whole-map, physics-scored live gem collection; no learned policy.

Existing ground routes propose aims. Candidate replies are predicted from the
current synchronized native state and pending controls. Selected ordinary game
inputs execute in lockstep; only actual score deltas count as collections.
"""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from nav.collision.mesh import CollisionMap,atomic_json
from nav.marble_dynamics.physics import GeneralPhysics
from nav.marble_dynamics.record import ML,RUNTIME,game,command,archive,sha
from nav.marble_dynamics.bridge import predict_replies,reply
from nav.marble_dynamics.evaluate import errors
from nav.marble_dynamics.events import external_changes
from nav.gems import visible_gems
from nav.joystick import action_to_joystick
from nav.protocol import NOOP_ACTION,RAW_SCORE
from nav.routing import RoutePlanner
from nav.terrain import TerrainGrid,TerrainMap


TARGET_SPEED_CAP=11.     # u/s top target speed. v26 (13.5) tied v23 on 30 rounds (103.9 vs 103.8) with falls 51 vs 32: +20 Sprawl,
                         # +13 Fractured, +9 Bowl but -19 Cube, -17 Concentric (edge maps); kept at 11 (2026-10-15)
BRAKE_DECEL=7.           # u/s^2 assumed for the stopping-distance law
THROTTLE_GAIN=4.         # v23 (2026-10-15): u/s^2 of commanded acceleration per unit of stick (throttle = |accel| / gain);
THROTTLE_CAP=1.          # pre-v23 12 / 0.85 faded the stick from 5 u/s below the target speed, so every proposal crept
                         # toward it (8-9 u/s peaks on open ground with an 11 u/s target); offline (72 v22d s2 states) gain 4 /
                         # cap 1 raised the chosen plans' peak speed on all 6 maps and progress per horizon by 3-19%


class Proposal:
    def __init__(self,goal,offset=0.,jump_at=-1,steps=24,coast=False,brake=False,use_at=-1,speed_distance=None,min_speed=0.,top=None):
        self.goal=np.asarray(goal,float);self.offset=offset;self.jump_at=jump_at;self.steps=steps;self.coast=coast;self.brake=brake;self.use_at=use_at
        # speed_distance: distance the stopping law should use instead of the
        # distance to the aim point (the remaining route length when the aim
        # point is only the next waypoint); min_speed: floor on the target speed
        # (a gem needs no stop: the human collects at full speed).
        self.speed_distance=speed_distance;self.min_speed=min_speed
        self.top=top   # top target speed for this call (None = TARGET_SPEED_CAP); v36 open-area speed

    def __call__(self,state,index):
        p=np.asarray(state['position']);v=np.asarray(state['velocity'])
        if self.brake and np.linalg.norm(v[:2])<=.1:
            return reply(NOOP_ACTION)
        delta=self.goal[:2]-p[:2];distance=np.linalg.norm(delta)
        stop_distance=distance if self.speed_distance is None else max(float(self.speed_distance),float(distance))
        target_speed=min(TARGET_SPEED_CAP if self.top is None else float(self.top),max(float(self.min_speed),math.sqrt(max(0.,2*BRAKE_DECEL*(stop_distance-.25)))))
        desired=delta/max(distance,1e-9)*target_speed
        accel=(desired-v[:2])*2.
        if index<6 and self.offset:
            c,s=math.cos(self.offset),math.sin(self.offset);accel=np.array([c*accel[0]-s*accel[1],s*accel[0]+c*accel[1]])
        magnitude=np.linalg.norm(accel)
        throttle=min(THROTTLE_CAP,magnitude/THROTTLE_GAIN) if magnitude>.05 and not self.coast else 0.
        joystick=action_to_joystick(float(accel[0]),float(accel[1]),throttle,index==self.jump_at,
                                    int(self.brake),float(v[0]),float(v[1]))
        return reply(joystick,int(index==self.use_at),math.atan2(delta[0],delta[1]))


def support(mesh,state):
    p=np.asarray(state['position']);r=state['radius']
    hit=mesh.raycast(p+[0,0,.05],p-[0,0,r+.10],cull_backfaces=True)
    return bool(hit and hit['normal'][2]>.4)


def possible_pickup(states,item):
    bounds=np.asarray(item['bounds']);lo,hi=bounds[:3],bounds[3:]
    for i,(a,b) in enumerate(zip(states,states[1:])):
        # Native item query uses a swept AABB expanded by radius + 0.2u.
        # Server eligibility/timing still requires actual pickup confirmation.
        p,q=np.asarray(a['position']),np.asarray(b['position']);r=b['radius']+.2
        if np.all(np.maximum(p,q)+r>=lo) and np.all(np.minimum(p,q)-r<=hi):return i+1
    return None


THROUGH_EVAL_REPLIES=24   # v22: post-pickup progress toward the next gem is read at this reply (the planner horizon)
END_VELOCITY_WEIGHT=.12   # u per u/s of end speed (penalty). v17 tried -0.3 (credit) on 2026-10-15: KOTM/Cube lost and falls rose; kept at the pre-v17 value. Note: speed carried past the horizon is future
                          # progress (credit), not a hazard; the physics gates (support, stability, falls) keep the
                          # safety. Pre-v17 the term was +0.12 (a penalty) and the planner preferred plans that end
                          # slow: 'accelerate_then_brake' won 423 of 2877 KOTM decisions on v16.


def candidate_cost(route,states,gem,routed,item,after=None):
    end=states[-1]
    distances=[float(np.linalg.norm(np.asarray(s['position'])-gem[:3])) for s in states]
    if after is not None and item is not None:
        # v22 (2026-10-15) through-gem: after a predicted pickup the rest of the
        # forecast is scored by its progress toward the NEXT gem of the tour
        # (after = (route to it, its position, walk distance gem -> it)). The old
        # terms measured the distance back to the collected gem, which charged
        # ~1.2 u for every unit driven past it, so every plan stopped at the gem
        # (v21 journals: 5.6 u/s at the pickup, 3.1 u/s a second later).
        # The progress is read at a common time (THROUGH_EVAL_REPLIES) so that a
        # longer forecast (landing continuation, ballistic projection, 48-reply
        # powerup forecast) earns nothing for its extra flight time.
        pickup=possible_pickup(states,item)
        if pickup is not None:
            nroute,ngem,leg=after;j=min(len(states)-1,THROUGH_EVAL_REPLIES)
            here=np.asarray(gem[:3],dtype=float);post=0.;speed=float(np.linalg.norm(end['velocity']))
            if pickup<=j:
                s=states[j];p=np.asarray(s['position'],dtype=float);d=float(nroute.distance(p))
                if math.isfinite(d):post=(d+.20*float(np.linalg.norm(p-ngem)))-(leg+.20*float(np.linalg.norm(here-ngem)))
                else:post=1.2*(float(np.linalg.norm(p-ngem))-float(np.linalg.norm(here-ngem)))
                speed=float(np.linalg.norm(s['velocity']))
            cost=post+END_VELOCITY_WEIGHT*speed-100.+pickup*.064*4.
            return float(cost),min(distances),pickup
    cost=(route.distance(end['position']) if routed else distances[-1])+END_VELOCITY_WEIGHT*np.linalg.norm(end['velocity'])
    if routed:cost+=.20*distances[-1]
    pickup=possible_pickup(states,item) if item else next((i for i,d in enumerate(distances) if d<.55),None)
    if pickup is not None:cost-=100.;cost+=pickup*.064*4.
    return float(cost),min(distances),pickup


def plan(engine,route,dynamics,gem,horizon=24):
    start=dynamics['native']['state'];waypoint,routed=route.navigate(start['position'],gem)
    specs=[('route',Proposal(waypoint,steps=horizon))]
    for angle in (-.6,.6):specs.append(('steer_'+str(angle),Proposal(waypoint,angle,steps=horizon)))
    specs += [('coast',Proposal(waypoint,steps=horizon,coast=True)),
              ('brake',Proposal(waypoint,steps=horizon,brake=True)),
              ('jump_now',Proposal(waypoint,jump_at=0,steps=horizon)),
              ('jump_later',Proposal(waypoint,jump_at=5,steps=horizon))]
    if routed:specs += [('direct',Proposal(gem,steps=horizon)),('direct_jump',Proposal(gem,jump_at=0,steps=horizon))]
    if 1<=start['held']<=5:
        specs += [('use_now',Proposal(waypoint,steps=horizon,use_at=0)),
                  ('use_later',Proposal(waypoint,steps=horizon,use_at=5))]
    candidates=[]
    for label,proposal in specs:
        rollout=predict_replies(engine,dynamics,proposal)
        end=rollout['states'][-1];reason=None
        if rollout['failure']:reason='prediction_failure'
        elif rollout['ambiguous_intervals']:reason='contact_ambiguity'
        elif not support(engine.mesh,end):reason='no_terminal_support'
        elif np.linalg.norm(end['velocity'])>22:reason='terminal_speed_above_pilot_limit'
        # A collection proxy prioritizes a 3D approach, never creates game points.
        # Route distance is only a heuristic for unfinished, supported candidates.
        item=next((t for t in dynamics.get('items',[]) if t['gem'] and np.linalg.norm(np.asarray(t['position'])-gem[:3])<.1),None)
        cost,close,pickup=candidate_cost(route,rollout['states'],gem,routed,item)
        candidate=dict(label=label,cost=float(cost),rejected=reason,minimum_gem_distance_u=close,
                       end_position=end['position'],terminal_support=support(engine.mesh,end),
                       ambiguous_intervals=rollout['ambiguous_intervals'],failure=rollout['failure'],
                       possible_pickup_interval=pickup,pickup_basis='native item bounds' if item else 'legacy proximity proxy')
        candidates.append((candidate,rollout))
    valid=[pair for pair in candidates if pair[0]['rejected'] is None]
    if not valid:return None,candidates,waypoint
    return min(valid,key=lambda pair:pair[0]['cost']),candidates,waypoint


def run_map(name,args):
    dest=args.output/name
    if (dest/'report.json').exists():raise RuntimeError('Preserve existing report: use a new output directory')
    dest.mkdir(parents=True,exist_ok=True)
    mesh=CollisionMap(ML/'collision_maps'/(name+'.npz'));engine=GeneralPhysics(mesh)
    route=RoutePlanner(TerrainGrid(TerrainMap.resolve(name)))
    raw=RUNTIME/(name+'.gem_demo.jsonl');log=[];pickups=[];failures=[];timings=[]
    started=time.perf_counter();target=None;score=0.;reason='decision_budget'
    with game(name,args.port,obs_ms=64) as (env,proc):
        command(env,'DYNAMICSLIVE 1','dynamics_live')
        for _ in range(6):env.step(NOOP_ACTION,repeat=1)
        command(env,'DYNAMICSRECORD '+raw.as_posix(),'dynamics_record')
        spawn=env.msg.dynamics['native']['state'];initial_score=float(env.msg.obs[RAW_SCORE])
        for dec in range(args.decisions):
            dynamics=env.msg.dynamics;state=dynamics['native']['state'];gems=visible_gems(env.msg.obs)
            if state['oob'] or state['mode']!=0:reason='unsupported_marble_state';break
            if target is None or not any(np.linalg.norm(np.asarray(g[:3])-target[:3])<.1 for g in gems):
                if not gems:reason='no_live_gem';break
                target=min(gems,key=lambda g:np.linalg.norm(np.asarray(g[:3])-state['position']))
            tick=env.msg.tick;t0=time.perf_counter()
            chosen,candidates,waypoint=plan(engine,route,dynamics,np.asarray(target[:3]),args.horizon)
            elapsed=time.perf_counter()-t0;timings.append(elapsed)
            if chosen is None:
                reason='no_candidate_passed_pilot_checks';failures.append(dict(decision=dec,candidates=[c for c,r in candidates]));break
            chosen_info,rollout=chosen;action=rollout['proposed'][0]
            before_score=float(env.msg.obs[RAW_SCORE])
            msg,info=env.step(action[:6],repeat=1,use_pow=int(action[6]),pow_yaw=action[7] if action[6] else None)
            actual=msg.dynamics['native']['state'];error=errors(rollout['states'][1],actual)
            physical_event=external_changes(rollout['states'][1],actual)
            delta=float(msg.obs[RAW_SCORE])-before_score
            entry=dict(decision=dec,tick=tick,gem=list(target),waypoint=list(waypoint),
                       dynamics=dynamics,action=action,chosen=chosen_info,candidates=[c for c,r in candidates],
                       planning_s=elapsed,predicted_next=rollout['states'][1],actual_next=actual,
                       errors=error,external_events=physical_event,score_delta=delta,outcome=info,
                       predicted_horizon_position=[s['position'] for s in rollout['states']])
            log.append(entry)
            if delta>0:
                score+=delta;pickups.append(dict(decision=dec,points=delta,gem=list(target),position=actual['position'],bridge_delta=info['gem_delta']))
                print(json.dumps(dict(map=name,decision=dec,pickup_points=delta,total_points=score,errors=error)),flush=True)
            if dec%10==0:print(json.dumps(dict(map=name,decision=dec,target_distance_u=chosen_info['minimum_gem_distance_u'],choice=chosen_info['label'],planning_s=elapsed,points=score)),flush=True)
            atomic_json(dest/'progress.json',dict(map=name,decisions=len(log),points=score,reason='running',last_error=error,
                                                  position=actual['position'],gem=list(target)))
            if info['round_ended'] or info['reconnected']:reason='round_or_connection_changed';break
            if info['fell']:reason='fell';break
            if error[0]>.05 and not physical_event:
                reason='live_position_error_above_pilot_limit';failures.append(dict(decision=dec,errors=error));break
            if len(pickups)>=args.gems:reason='collection_goal';break
        command(env,'DYNAMICSRECORD off','dynamics_record');archive(raw,dest/'native.jsonl.gz')
        final_score=float(env.msg.obs[RAW_SCORE])
        report=dict(map=name,geometry_sha256=mesh.info['mesh_sha256'],spawn_state=spawn,
                    setup='natural game spawn; no teleport, fixture restore, GIVEPOW or gem replacement',
                    owned_game_pid=proc.pid,initial_score=initial_score,final_score=final_score,
                    actual_points=score,actual_pickup_events=pickups,decisions=len(log),stop_reason=reason,
                    failures=failures,planning_s=timings,elapsed_wall_s=time.perf_counter()-started,
                    max_live_errors=np.max([r['errors'] for r in log],axis=0).tolist() if log else None,
                    native_raw_sha256=sha(raw),probe_executable_sha256=sha(ML.parent/'Marble Blast Platinum/marbleblast_geometry_probe.exe'),
                    source_hashes={p.name:sha(p) for p in (Path(__file__),Path(__file__).with_name('bridge.py'),
                                  Path(__file__).with_name('physics.py'),ML.parent/'Marble Blast Platinum/platinum/client/scripts/ai/mlAgent.cs')},
                    policy_training_started=False,global_planner_approved=False,
                    limitations='Bounded pilot; terminal support/speed and contact-ambiguity checks are not calibrated landing uncertainty. Gem proximity is a proposal score; game points are authoritative.')
        atomic_json(dest/'decisions.json',log);atomic_json(dest/'report.json',report)
        print(json.dumps(report),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--maps',nargs='+',default=['Concentric_Hunt'])
    p.add_argument('--decisions',type=int,default=180);p.add_argument('--gems',type=int,default=3)
    p.add_argument('--horizon',type=int,default=24);p.add_argument('--port',type=int,default=9053)
    p.add_argument('--output',type=Path,default=ML/'dynamics_reports/gem_demo_20261012')
    args=p.parse_args();summary=[]
    for name in args.maps:
        summary.append(run_map(name,args));atomic_json(args.output/'summary.json',dict(maps=summary,policy_training_started=False))


if __name__=='__main__':main()
