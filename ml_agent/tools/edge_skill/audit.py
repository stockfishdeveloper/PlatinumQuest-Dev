"""Independent contact evidence, separate from the fast trajectory screen."""
import copy

import numpy as np

from nav.marble_dynamics.rollout import apply_control
from .common import local,rotation


def contact_kind(case, point, normal):
    c = case['cell']; p = local(c, point); n = np.asarray(normal) @ rotation(c)
    lip_z=c['lip_z']+c.get('lip_slope_y',0.)*p[1]
    on_edge = abs(p[0]) < .025 and abs(p[2]-lip_z) < .025 and abs(p[1]) < c['width']/2-.02
    # A lower-half lip contact has an upward and backward-facing normal.
    if on_edge and .06 < n[2] < .995 and n[0] < -.04:
        return 'lip'
    if abs(p[0]) < .025 and p[2] < lip_z-.025 and n[0] < -.8 and abs(n[2]) < .08:
        return 'wall'
    return 'other'


def contract_contacts(case,contacts,until=None):
    """Contact contract for v3: genuine lip impulse, no wall-assisted approach/landing."""
    events=[r for r in contacts if until is None or r['tick']<=until]
    lips=[r for r in events if r['kind']=='lip' and r['kick']>.5]
    hit=lips[0] if lips else None
    walls=[r for r in events if r['kind']=='wall' and r['approach']>.5 and r['kick_norm']>.5]
    mode_ok=bool(hit and (case.get('required_contact_mode') is None or hit['mode']==case['required_contact_mode']))
    return dict(valid=bool(hit and mode_ok and not walls),tick=hit['tick'] if hit else None,first_lip=hit,
                walls=walls,contacts=events,reason='no_lip' if hit is None else 'wall_assist' if walls else 'mode' if not mode_ok else 'ok')


def audit_native_contract(case,transitions,until=None):
    events=[];rot=rotation(case['cell'])
    for i,r in enumerate(transitions):
        if until is not None and i+1>until:break
        for sub in r.get('substep_trace',[]):
            before=np.asarray(sub['contact_entry_velocity']);after=np.asarray(sub['pre_integration']['velocity'])
            for contact in sub.get('contacts',[]):
                n=np.asarray(contact['normal']);kind=contact_kind(case,contact['position'],n)
                if abs(n[2])<.08:kind='wall'
                if kind=='other':continue
                lv=before@rot;av=after@rot
                events.append(dict(kind=kind,tick=i+1,point=contact['position'],normal=contact['normal'],
                                   kick=float(after[2]-before[2]),kick_norm=float(np.linalg.norm(after-before)),
                                   approach=float(-before@n),mode='rise' if before[2]>.6 else 'descend' if before[2]<-.6 else 'roll',
                                   entry_velocity=before.tolist(),exit_velocity=after.tolist(),
                                   forward_retention=float(av[0]/max(lv[0],1e-6))))
    return contract_contacts(case,events,until)


def forecast_events(engine,dynamics,rollout,until=None,compiled=True):
    if compiled:
        from .compiled import contact_events
        packed,_=contact_events(engine,dynamics,rollout,until)
        for row in packed:
            e=dict(kind='support' if row[0]==2 else 'collision',triangle_id=int(row[1]),a=row[3:6].tolist(),b=row[6:9].tolist())
            yield int(row[2]),row[12:15],row[9:12],row[15:18],[e]
        return
    state=copy.deepcopy(dynamics['native']['state']);clock=dynamics['native']['frame_clock_ms']
    for i,control in enumerate(rollout['native_controls']):
        if until is not None and i+1>until:break
        state['clock_ms']=clock+i*16;before=np.asarray(state['velocity']).copy();before_p=np.asarray(state['position']).copy()
        move=apply_control(state,control);state,events=engine.native_sim.step(state,move,16)
        yield i+1,before,before_p,np.asarray(state['velocity']),events


def audit_forecast_contract(engine,case,dynamics,rollout,until=None,compiled=True):
    """Use collision AND support-contact events; a tick-overlap impact need not emit a sweep collision."""
    events=[]
    rot=rotation(case['cell']);cell=case['cell']
    for tick,before,before_p,after,step_events in forecast_events(engine,dynamics,rollout,until,compiled):
        dv=after-before+np.array([0.,0.,.32])
        for e in step_events:
            if e['kind'] not in ('collision','support'):continue
            normal=np.asarray(e['a']);kind='other';point=e['b']
            if e['kind']=='collision':kind=contact_kind(case,point,normal)
            else:
                tid=e['triangle_id'];n=normal@rot
                slope=cell.get('lip_slope_y',0.)
                if tid>=0 and .06<n[2]<.995 and n[0]<-.04 and abs(n[1]+slope*n[2])<.08:
                    tri=local(cell,engine.mesh.triangles[tid]);on=(abs(tri[:,0])<.025)&(abs(tri[:,2]-cell['lip_z']-slope*tri[:,1])<.025)
                    lp=local(cell,before_p)
                    if int(on.sum())>=2 and abs(lp[1])<cell['width']/2-.2:
                        kind='lip';point=transform_contact(cell,lp[1]+slope*(lp[2]-cell['lip_z']-slope*lp[1])/(1+slope*slope))
            if abs(normal[2])<.08:kind='wall'
            if kind=='other':continue
            lv=before@rot;av=after@rot
            events.append(dict(kind=kind,tick=tick,point=point,normal=e['a'],evidence=e['kind'],
                               kick=float(dv[2]),kick_norm=float(np.linalg.norm(dv)),approach=float(-before@normal),
                               mode='rise' if before[2]>.6 else 'descend' if before[2]<-.6 else 'roll',
                               entry_velocity=before.tolist(),exit_velocity=after.tolist(),
                               forward_retention=float(av[0]/max(lv[0],1e-6))))
    return contract_contacts(case,events,until)


def transform_contact(cell,y):
    from .maps import transform
    return transform(cell,[0.,float(y),cell['lip_z']+cell.get('lip_slope_y',0.)*y])
