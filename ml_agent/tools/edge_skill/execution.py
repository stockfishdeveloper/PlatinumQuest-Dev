"""Contact-conditioned edge execution, isolated from the production planner.

An opportunity supplies an exact lip segment, landing rectangle, and exit state.
Search chooses approach/contact parameters. The same feedback law is forecast and
executed; the contract includes supported exit and a braking continuation.
"""
import copy
import math
import os
import time

import numpy as np

from nav.joystick import action_to_joystick
from nav.marble_dynamics.bridge import reply
from nav.native.forecast import predict_many
from .common import ART,NIGHT,load_json,local,rotation
from .maps import transform

RADIUS=.189750001
DT=.016
REPLY=.064
HORIZON=128


def initial(case):
    if 'initial' in case:return copy.deepcopy(case['initial'])
    path=NIGHT/(case['cell']['map']+'.initial.json')
    if not path.exists():path=ART/(case['cell']['map']+'.initial.json')
    if not path.exists():path=ART/'edge_skill_runup_v2.initial.json'
    d=copy.deepcopy(load_json(path)['dynamics']);s=d['native']['state'];r=rotation(case['cell'])
    for key in ('velocity','spin'):s[key]=(r@np.asarray(case[key])).tolist()
    s['position']=transform(case['cell'],case['position']);s['ground_ms']=0 if case.get('airborne') else 128
    s['best_contact_normal']=[0.,0.,1.];s['held']=0;s['oob']=False;s['mode']=0
    s['effects']=[[False,0] for _ in range(6)];s['effect_end_clock_ms']=[0]*6
    d['items']=[]
    return d


def platform(cell,points):
    p=np.asarray(points);theta=cell.get('landing_yaw',0.)
    if 'landing_inverse' in cell:return (p[...,:2]-np.asarray(cell['landing_center']))@np.asarray(cell['landing_inverse']).T
    r=np.array([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]])
    return (p[...,:2]-np.asarray(cell['landing_center']))@r


def inside(cell,points,margin=.29):
    q=platform(cell,points)
    return np.all(np.abs(q)<=np.asarray(cell['landing_size'])/2-margin,axis=-1)


def legacy_case(case):
    """Give the old straight drills an explicit landing rectangle and exit vector."""
    c=copy.deepcopy(case);cell=c['cell'];end=cell.get('landing_end',42.);start=cell['landing_x']
    cell.setdefault('landing_center',[(start+end)/2,0.]);cell.setdefault('landing_size',[end-start,cell['width']])
    cell.setdefault('landing_yaw',0.)
    c.setdefault('exit_velocity',[c.get('exit_speed',8.),0.]);c.setdefault('airborne',False)
    c.setdefault('goal_radius',.9);c.setdefault('required_contact_mode',None)
    return c


def observed(case,d):
    c=copy.deepcopy(case);s=d['native']['state'];r=rotation(c['cell'])
    c.update(position=local(c['cell'],s['position']).tolist(),velocity=(np.asarray(s['velocity'])@r).tolist(),
             spin=(np.asarray(s['spin'])@r).tolist())
    c['jump_committed']=bool(c.get('jump_committed') or d['bridge_action'][4] or d['native']['pending_move'].get('jump'))
    c.pop('initial',None)
    return c


def lip_screen(cell,rows):
    rows=np.asarray(rows);p=local(cell,rows[:,:3]);v=rows[:,3:6]@rotation(cell)
    slope=cell.get('lip_slope_y',0.);projection=p[:,[0,2]].copy()
    projection[:,1]=(p[:,2]-cell['lip_z']-slope*p[:,1])/math.sqrt(1+slope*slope)
    a=projection[:-1];delta=projection[1:]-projection[:-1]
    f=np.clip(-np.einsum('ij,ij->i',a,delta)/(np.einsum('ij,ij->i',delta,delta)+1e-15),0,1)
    q=a+delta*f[:,None];dist=np.linalg.norm(q,axis=1)
    kick=v[1:,2]-v[:-1,2]+20*DT
    hits=np.flatnonzero((dist<RADIUS+.10)&(q[:,1]>.01)&(q[:,1]<RADIUS+.02)&(q[:,0]<.015)
                       &(abs(p[:-1,1])<cell['width']/2-.1)&(kick>.6))
    return (int(hits[0]+1) if len(hits) else None),float(dist.min()) if len(dist) else 99.


def assess(case,rows,require_lip=True,contact=None):
    rows=np.asarray(rows);cell=case['cell'];p=local(cell,rows[:,:3]);v=rows[:,3:6]@rotation(cell)
    hit,edge_error=lip_screen(cell,rows)
    mode=('rise' if v[hit-1,2]>.6 else 'descend' if v[hit-1,2]<-.6 else 'roll') if hit else None
    if case.get('required_contact_mode') is not None and mode!=case['required_contact_mode']:hit=None
    if contact is not None:hit=contact.get('tick') if contact.get('valid') else None
    goal=np.asarray(case['goal']);exit_v=np.asarray(case['exit_velocity'])
    horizontal=np.linalg.norm((p-goal)[:,:2],axis=1)
    slope=np.asarray(cell.get('landing_slope',[0.,0.]));height=goal[2]+(p[:,:2]-goal[:2])@slope
    supported=(abs(p[:,2]-height)<.04)&(abs(v[:,2]-v[:,:2]@slope)<.65)&inside(cell,p)
    velocity_error=np.linalg.norm(v[:,:2]-exit_v,axis=1)
    permitted=np.arange(len(p))> (hit if hit is not None else (len(p) if require_lip else -1))
    entries=np.flatnonzero(supported&(horizontal<case.get('goal_radius',.9))&(velocity_error<2.5)&permitted)
    finish=None;reach=None;window=int(case.get('exit_window_ticks',32))   # 2026-11-03 planner use: a shorter supported exit, no braking backup (the planner's own continuation and backup checks judge what follows)
    for i in entries:
        end=i+window
        if end>=len(p):continue
        if supported[i:end+1].all() and (velocity_error[i:end+1]<3.).all():
            reach=int(i);finish=int(end);break
    # Completion is a supported exit state. Later controls are the checked braking backup.
    backup=bool(finish is not None and finish+24<len(p) and supported[finish:finish+25].all())
    if not case.get('require_backup',True):backup=bool(finish is not None)
    resources_valid=bool(np.all(abs(rows[:,9]-RADIUS)<1e-5) and (rows.shape[1]<25 or not np.any(rows[:,19:25])))
    success=bool((hit is not None or not require_lip) and finish is not None and backup and resources_valid)
    valid_support=supported&permitted
    landing_error=float(horizontal[valid_support].min()) if valid_support.any() else 15.
    exit_error=float((horizontal[valid_support]+.7*velocity_error[valid_support]).min()) if valid_support.any() else 20.
    # Position alone rewards flying through the goal too fast to use the platform.
    # Score position and velocity at the SAME supported instant.
    min_goal=float(np.linalg.norm((p-goal)*[1,1,2],axis=1).min())
    score=25.*(hit is None and require_lip)+3*max(0,edge_error-RADIUS)+min_goal+exit_error
    if hit is not None:
        score+=float(max(0.,-v[hit,0]))
    if success:score=-100+finish*DT+.1*float(velocity_error[reach])
    return dict(success=success,lip_tick=hit,reach_tick=reach,finish_tick=finish,backup=backup,resources_valid=resources_valid,
                goal_error=min_goal,landing_error=landing_error,exit_error=exit_error,edge_error=edge_error,score=float(score),
                target_time=reach*DT if reach is not None else None,
                exit_velocity=v[finish,:2].tolist() if finish is not None else None,
                contact_mode=mode)


class Policy:
    """One feedback policy, identical during forecasting and actual execution.

    Parameters: approach speed, contact y, contact-normal z, jump-distance
    correction, descending branch, landing offset, airborne position gain.
    """
    def __init__(self,case,params,steps=HORIZON,engine=None,d=None):
        self.case=case;self.cell=case['cell'];self.params=list(map(float,params));self.steps=steps
        self.params.extend([0.]*(10-len(self.params)))
        self.rot=rotation(self.cell);self.origin=np.asarray(self.cell['origin']);self.goal=np.asarray(case['goal'])
        self.exit=np.asarray(case['exit_velocity']);self.exit_speed=float(np.linalg.norm(self.exit))
        self.u=self.exit/max(self.exit_speed,1e-6)
        self.landing_slope=np.asarray(self.cell.get('landing_slope',[0.,0.]));self.lip_slope=self.cell.get('lip_slope_y',0.)
        self.jumped=bool(case.get('airborne') or case.get('jump_committed'));self.phase='approach';self.contact_seen=False
        self.entered=False;self.exit_count=0;self.finished=False;self.done_at=None
        self.engine=engine;self.d=d

    def observe_ticks(self,states):
        rows=states.rows if hasattr(states,'rows') else np.asarray([[*s['position'],*s['velocity'],*s['spin'],s['radius']] for s in states])
        hit,_=lip_screen(self.cell,rows)
        if hit is not None:self.contact_seen=True;self.phase='landing'

    def __call__(self,state,index):
        p=(np.asarray(state['position'])-self.origin)@self.rot;v=np.asarray(state['velocity'])@self.rot
        speed,cy,nz,jump_offset,descending,land_offset,gain,velocity_gain,side_offset,intercept=self.params
        height=self.goal[2]+float((p[:2]-self.goal[:2])@self.landing_slope)
        supported=abs(p[2]-height)<.045 and abs(v[2]-float(v[:2]@self.landing_slope))<.7 and bool(inside(self.cell,p))
        if p[0]>.22:self.phase='landing'
        if supported and self.phase=='landing':
            if np.linalg.norm(p[:2]-self.goal[:2])<self.case.get('goal_radius',.9) and np.linalg.norm(v[:2]-self.exit)<2.5:self.entered=True
            if self.entered and np.linalg.norm(v[:2]-self.exit)<3.:self.exit_count+=1
            else:self.exit_count=0
            if self.exit_count>=9:
                self.finished=True
                if self.done_at is None:self.done_at=index
        elif not self.finished:self.exit_count=0
        control_p=p;control_v=v;control_landing=self.phase=='landing';next_d=None
        if intercept>1.5:
            if self.engine is None or self.d is None:raise ValueError('Queued-state guidance requires the initial bridge state and engine')
            actual=copy.deepcopy(self.d);actual['native']['state']=copy.deepcopy(state)
            future=predict_many(self.engine,actual,[[[0.,0.,0.,0.,0,0.,0,0.,0.]]])[0]
            next_d=resume_static(actual,future,1);fs=next_d['native']['state']
            control_p=(np.asarray(fs['position'])-self.origin)@self.rot;control_v=np.asarray(fs['velocity'])@self.rot
            future_hit,_=lip_screen(self.cell,future['native_states'].rows)
            control_landing=control_landing or future_hit is not None or control_p[0]>.22
        cp,cv=control_p,control_v
        control_height=self.goal[2]+float((cp[:2]-self.goal[:2])@self.landing_slope)
        control_vz=cv[2]-float(cv[:2]@self.landing_slope)
        control_supported=abs(cp[2]-control_height)<.045 and abs(control_vz)<.7 and bool(inside(self.cell,cp))
        jump=0
        if self.finished:
            # This backup is used only if the caller has not handed off at completion.
            acc=-v[:2]*5.
        elif not control_landing:
            center_y=cy-RADIUS*self.lip_slope*nz
            desired=np.array([speed,np.clip((center_y-p[1])*2.,-speed,speed)])
            acc=(desired-v[:2])*2.5
            # For an airborne approach the vertical arc fixes when contact can
            # occur. Guide horizontal motion to that intercept instead of merely
            # chasing a speed. Search retains the ordinary approach as a second
            # controller family because saturated intercepts are not always best.
            floor=self.case.get('start_floor',0.)
            on_floor=abs(p[2]-floor-RADIUS)<.05 and abs(v[2])<.7
            future_on_floor=abs(cp[2]-floor-RADIUS)<.05 and abs(cv[2])<.7
            if intercept>.5 and not future_on_floor:
                height=self.cell['lip_z']+self.lip_slope*cy+RADIUS*nz
                disc=cv[2]**2-40*(height-cp[2])
                if disc>=0:
                    times=[(cv[2]-math.sqrt(disc))/20,(cv[2]+math.sqrt(disc))/20]
                    positive=[t for t in times if t>.035]
                    if positive:
                        t=max(positive) if descending>.5 else min(positive)
                        target=np.array([-RADIUS*math.sqrt(max(0.,1-nz*nz*(1+self.lip_slope**2))),center_y])
                        acc=2.*(target-cp[:2]-cv[:2]*t)/max(t*t,.025)
            if not self.jumped and self.case.get('start_kind','ground')!='roll':
                height=self.cell['lip_z']+self.lip_slope*cy+RADIUS*nz-floor-RADIUS
                disc=7.5**2-40*height
                if disc>=0:
                    flight=(7.5+(1 if descending>.5 else -1)*math.sqrt(disc))/20
                    trigger=max(v[0],1.)*(flight+.080)+jump_offset
                    if -p[0]<=trigger and on_floor:
                        jump=1;self.jumped=True
        elif control_supported:
            error=self.goal[:2]-cp[:2];along=float(error@self.u);lateral=error-self.u*along
            desired=self.exit+self.u*max(0.,min(2.,along*.35))+lateral*2.
            acc=(desired-cv[:2])*3.
        else:
            disc=control_vz**2+40*(cp[2]-control_height)
            t=(control_vz+math.sqrt(max(0.,disc)))/20
            if t>.10:
                aim=self.goal[:2]-self.exit*.30+self.u*land_offset+np.array([-self.u[1],self.u[0]])*side_offset
                t=max(t,.18)
                acc=gain*(aim-cp[:2]-cv[:2]*t)/(t*t)+velocity_gain*(self.exit-cv[:2])/t
            else:acc=(self.exit-cv[:2])*2.5+(self.goal[:2]-cp[:2])*.5
        length=float(np.linalg.norm(acc));ground=supported or (abs(v[2])<.7 and abs(p[2]-RADIUS-self.case.get('start_floor',0.))<.05)
        maximum=8.05 if ground else 5.*math.sqrt(2.)
        throttle=min(1.,length/maximum)
        if self.case.get('stable_inputs'):
            # The bridge quantizes positive camera axes in 1/16 increments.
            # Exact diagonal/full-throttle commands sit on a bin boundary;
            # microscopic state/formatting changes can otherwise drop one axis
            # by 1/16. Encode the intended bin away from its boundary. The game
            # still clamps the applied axes to their ordinary maximum of one.
            throttle=min(1.0002,(math.floor(throttle*16)+.5)/16)
        world=self.rot@np.array([acc[0],acc[1],0.])
        action=reply(action_to_joystick(float(world[0]),float(world[1]),throttle,jump))
        if next_d is not None:next_d['bridge_action']=action;self.d=next_d
        return action


def forecast(engine,case,params,d=None):
    d=initial(case) if d is None else d;steps=int(case.get('horizon_replies',HORIZON));policies=[Policy(case,p,steps=steps,engine=engine,d=d) for p in params]
    if os.environ.get('EDGE_SKILL_PYTHON')=='1':rolls=predict_many(engine,d,policies)
    else:
        from .compiled import predict
        rolls,memory=predict(engine,case,d,params,steps,early_stop=True)
        for policy,m in zip(policies,memory):
            policy.jumped=bool(m[0]);policy.phase='landing' if m[1] else 'approach';policy.contact_seen=bool(m[2])
            policy.entered=bool(m[3]);policy.exit_count=int(m[4]);policy.finished=bool(m[5]);policy.done_at=int(m[6]) if m[6]>=0 else None
    out=[]
    for par,policy,r in zip(params,policies,rolls):
        if isinstance(r,Exception) or r['failure']:result=dict(success=False,score=1e6,failure=str(r if isinstance(r,Exception) else r['failure']))
        else:result=assess(case,r['native_states'].rows)
        out.append(dict(params=list(map(float,par)),result=result,rollout=r,_forecast_policy=policy))
    return out


def seeds(case):
    cell=case['cell'];p=np.asarray(case['position']);v=np.asarray(case['velocity']);goal=np.asarray(case['goal'])
    width=cell['width']/2-.4
    cy=float(np.clip((p[1]+goal[1])*.5,-width,width))
    y=[float(np.clip(y,-width,width)) for y in (p[1],cy,goal[1]*.65)]
    speeds=sorted(set(round(float(np.clip(x,3.,22.)),2) for x in (v[0],7.,11.,15.,19.)))
    return [[s,yy,nz,offset,desc,0.,gain,vg,0.,intercept] for desc in (0,1) for s in speeds for yy in y
            for nz in (.45,.75,.9) for offset in (-.5,.5) for gain,vg,intercept in ((2.,0.,0.),(6.,-2.,0.),(2.,0.,1.),(6.,-2.,1.),(2.,0.,2.),(6.,-2.,2.))]


def clip_params(case,p):
    q=np.asarray(list(p)+[0.]*(10-len(p))).copy();width=case['cell']['width']/2-.35
    q[0]=np.clip(q[0],2,25);q[1]=np.clip(q[1],-width,width);q[2]=np.clip(q[2],.12,.98)
    q[3]=np.clip(q[3],-3.5,3.5);q[4]=float(q[4]>.5);q[5]=np.clip(q[5],-2,2);q[6]=np.clip(q[6],1,6)
    q[7]=np.clip(q[7],-3,1);q[8]=np.clip(q[8],-3,3)
    q[9]=float(np.clip(round(q[9]),0,2))
    return q.tolist()


def solve(engine,case,d=None,budget=192,seed=0,initial_params=None,robust_samples=0,max_alternatives=3,accept_early=False):
    from .audit import audit_forecast_contract
    d=initial(case) if d is None else d;case=observed(case,d)
    begin=time.perf_counter();rng=np.random.default_rng(seed)
    warm=[clip_params(case,p) for p in (initial_params or [])][:budget//2];candidates=seeds(case)
    # Cover both jump branches before local refinement, rather than truncating one branch.
    if len(candidates)>budget//2:
        ids=np.linspace(0,len(candidates)-1,max(0,budget//2-len(warm)),dtype=int);candidates=[candidates[i] for i in ids]
    candidates=warm+candidates
    results=forecast(engine,case,candidates,d)
    if accept_early and not robust_samples:
        # Execution needs a checked solution. Refining an already valid warm
        # proposal for a slightly faster finish can wait for planner integration.
        for candidate in sorted(results,key=lambda r:r['result']['score'])[:8]:
            if not candidate['result']['success']:continue
            audit=audit_forecast_contract(engine,case,d,candidate['rollout'],candidate['result']['finish_tick']+24)
            checked=assess(case,candidate['rollout']['native_states'].rows,contact=audit)
            checked['contact_audit']=audit;candidate['result']=checked
            if checked['success']:
                candidate['alternatives']=[candidate['params']]
                candidate['stats']=dict(candidates=len(results),wall_s=time.perf_counter()-begin,
                                        screen_successes=sum(r['result']['success'] for r in results),early_accept=True)
                candidate['case']=case;candidate['planned_frame_clock_ms']=int(d['native']['frame_clock_ms'])
                return candidate
    while len(results)<budget:
        elite=sorted(results,key=lambda r:r['result']['score'])[:8];batch=[]
        scale=np.array([1.5,.45,.12,.45,.0,.4,.5,.4,.4,0.])*max(.22,.8**(len(results)//96))
        for j in range(min(32,budget-len(results))):
            p=np.asarray(elite[j%len(elite)]['params'])+rng.normal(size=10)*scale
            if rng.random()<.08:p[4]=1-p[4]
            if rng.random()<.08:p[9]=rng.integers(0,3)
            batch.append(clip_params(case,p))
        results+=forecast(engine,case,batch,d)
    ranked=sorted(results,key=lambda r:r['result']['score']);verified=[]
    shortlist=[]
    for candidate in ranked:
        if not candidate['result']['success']:break
        if not robust_samples or all(np.linalg.norm((np.asarray(candidate['params'])-np.asarray(x['params']))/[4,1.5,.25,.8,1,1.5,2,1,1,1])>.25 for x in shortlist):
            shortlist.append(candidate)
        if len(shortlist)>=(32 if robust_samples else 8):break
    for candidate in shortlist:
        if not candidate['result']['success']:break
        audit=audit_forecast_contract(engine,case,d,candidate['rollout'],candidate['result']['finish_tick']+24)
        result=assess(case,candidate['rollout']['native_states'].rows,contact=audit)
        result['contact_audit']=audit;candidate['result']=result
        if result['success']:verified.append(candidate)
        if len(verified)>=(16 if robust_samples else max_alternatives):break
    best=verified[0] if verified else min(ranked[:8],key=lambda r:r['result']['score'])
    if not verified:best['result']['success']=False
    best['alternatives']=[x['params'] for x in verified]
    if robust_samples and verified:
        pool=[];uncertainty_seed=seed+831721
        def robust_rank(candidate):
            cert=certify(engine,case,candidate['params'],d,samples=min(robust_samples,8),seed=uncertainty_seed)
            candidate['robustness']=cert
            # Failure fraction dominates speed. Within it, favor trajectories
            # that remain close to the joint position/velocity objective.
            cost=100*(1-cert['successes']/cert['samples'])+float(np.mean([np.clip(r['score']+100,0,150) for r in cert['results']]))*.02
            return cost
        for candidate in verified:pool.append((robust_rank(candidate),candidate))
        for iteration in range(3):
            elite=sorted(pool,key=lambda x:x[0])[:3];pp=[]
            for j in range(24):
                delta=rng.normal(size=10)*np.array([.6,.18,.05,.20,0,.20,.25,.16,.16,0])*(.7**iteration)
                pp.append(clip_params(case,np.asarray(elite[j%len(elite)][1]['params'])+delta))
            for candidate in forecast(engine,case,pp,d):
                if not candidate['result']['success']:continue
                audit=audit_forecast_contract(engine,case,d,candidate['rollout'],candidate['result']['finish_tick']+24)
                candidate['result']=assess(case,candidate['rollout']['native_states'].rows,contact=audit);candidate['result']['contact_audit']=audit
                if candidate['result']['success']:pool.append((robust_rank(candidate),candidate))
        best=min(pool,key=lambda x:x[0])[1]
        best['robustness']=certify(engine,case,best['params'],d,samples=robust_samples,seed=seed+928641)
        best['alternatives']=[x[1]['params'] for x in sorted(pool,key=lambda x:x[0])[:8]]
    best['stats']=dict(candidates=len(results),wall_s=time.perf_counter()-begin,screen_successes=sum(r['result']['success'] for r in results),early_accept=False)
    best['case']=observed(case,d);best['planned_frame_clock_ms']=int(d['native']['frame_clock_ms'])
    return best


def make_executor(candidate,engine,d):
    """Bind the checked controller and its encoding to the exact observed root."""
    if not candidate['result']['success']:raise ValueError('The skill declined this opportunity')
    if int(d['native']['frame_clock_ms'])!=candidate['planned_frame_clock_ms']:
        raise ValueError('The game advanced after planning; replan from the current observed state')
    return Policy(candidate['case'],candidate['params'],engine=engine,d=d)


def make_case(cell,index,seed,start_kind='ground'):
    rng=np.random.default_rng(seed);speed=float(rng.uniform(2,24));angle=float(rng.uniform(-.85,.85))
    if index%7==0:speed=float(rng.uniform(0,2))
    distance=float(rng.uniform(1.2,29.));y=float(rng.uniform(-min(6,cell['width']/2+1),min(6,cell['width']/2+1)))
    vx,vy=speed*math.cos(angle),speed*math.sin(angle);p=[-distance,y,RADIUS];v=[vx,vy,0.]
    airborne=start_kind in ('air_rising','air_descending')
    if airborne:
        flight=float(rng.uniform(.06,.29) if start_kind=='air_rising' else rng.uniform(.42,.68))
        p[2]=RADIUS+7.5*flight-10*flight*flight;v[2]=7.5-20*flight
        p[0]=-float(rng.uniform(.5,10.))
    spin=(np.asarray([-vy/RADIUS,vx/RADIUS,0.])*rng.uniform(.5,1.5)+rng.normal(0,3,3)).tolist()
    theta=cell['landing_yaw'];exit_v=[cell['exit_speed']*math.cos(theta),cell['exit_speed']*math.sin(theta)]
    return dict(id=cell['id']+'_'+start_kind+'_%04d'%index,cell=cell,position=p,velocity=v,spin=spin,
                goal=cell['goal_local'],exit_velocity=exit_v,goal_radius=.9,airborne=airborne,
                start_kind=start_kind,start_floor=0.,seed=seed,split=cell['split'],required_contact_mode=None)


def resume_static(d,r,k):
    """Resume this unpowered static laboratory, including the queued bridge move.

    This deliberately does not claim to clone a live Hunt world. Native replay
    conformance is measured separately for the laboratory's contact regimes.
    """
    from nav.native.forecast import resume_at,_move_dict
    try:return resume_at(d,r,k)
    except ValueError:
        if r['failure'] or any(a[6] or a[8] for a in r['proposed']):raise
        if d['native']['state']['held'] or any(e[0] for e in d['native']['state']['effects']):raise
        out=copy.deepcopy(d);out['items']=[];out['native']['state']=copy.deepcopy(r['states'][k])
        row=r['reply_table'][k-1]
        out['native']['frame_clock_ms']=int(d['native']['frame_clock_ms'])+64*k
        out['native']['pending_move']=_move_dict(row[15:21]);out['bridge_action']=list(r['proposed'][k-1])
        out['pow_yaw_hold']=int(row[38]);out['pow_yaw']=float(row[39])
        for source,key in [('blast_context','blast'),('inventory_context','inventory')]:
            if source in out['native']['state']:out[key]=copy.deepcopy(out['native']['state'][source])
        return out


def perturb(case,d,seed,size=1.):
    """Local state uncertainty, larger than the measured native replay error."""
    rng=np.random.default_rng(seed);out=copy.deepcopy(d);s=out['native']['state'];rot=rotation(case['cell'])
    p=local(case['cell'],s['position']);v=np.asarray(s['velocity'])@rot;w=np.asarray(s['spin'])@rot
    p[:2]+=rng.uniform(-.03,.03,2)*size
    if p[2]>case.get('start_floor',0.)+RADIUS+.06:p[2]+=rng.uniform(-.02,.02)*size
    v[:2]+=rng.uniform(-.25,.25,2)*size
    if case.get('airborne'):v[2]+=rng.uniform(-.15,.15)*size
    w+=rng.uniform(-1.,1.,3)*size
    s['position']=transform(case['cell'],p);s['velocity']=(rot@v).tolist();s['spin']=(rot@w).tolist()
    return observed(case,out),out


def checked_forecast(engine,case,params,d=None):
    from .audit import audit_forecast_contract
    d=initial(case) if d is None else d;r=forecast(engine,case,[params],d)[0]
    if r['result']['success']:
        a=audit_forecast_contract(engine,case,d,r['rollout'],r['result']['finish_tick']+24)
        r['result']=assess(case,r['rollout']['native_states'].rows,contact=a);r['result']['contact_audit']=a
    r['case']=observed(case,d);r['planned_frame_clock_ms']=int(d['native']['frame_clock_ms'])
    return r


def certify(engine,case,params,d=None,samples=16,size=1.,seed=0):
    d=initial(case) if d is None else d;rr=[]
    nominal=checked_forecast(engine,case,params,d)
    for i in range(samples):
        c,dd=perturb(case,d,seed+i*7919,size)
        r=checked_forecast(engine,c,params,dd);rr.append(r['result'])
    return dict(nominal=nominal['result'],samples=samples,size=size,seed=seed,
                successes=sum(r['success'] for r in rr),all_pass=bool(nominal['result']['success'] and all(r['success'] for r in rr)),results=rr)
