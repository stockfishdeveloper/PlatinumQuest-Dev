"""Predict the existing local 64 ms lockstep bridge, including queued controls.

The observation occurs before its pending reply is executed. Native movement
has already been collected. Therefore the proposed reply changes camera one
frame later and movement/jump/use two frames later. No bridge behavior changes.
"""
import copy
import math

import numpy as np

from nav.marble_dynamics.rollout import apply_control
from nav.marble_dynamics.blast import event as blast_event,advance as advance_blast
from nav.marble_dynamics.inventory import check as check_inventory
from nav.marble_dynamics.planning_budget import check as check_budget


def quantize(value):
    value=np.float32(value)
    return float((int(np.float32(np.float32(value+np.float32(1))*np.float32(16)))-16)/16) if -1<=value<=1 else float(np.clip(value,-1,1))


def reply(joystick,use=0,pow_yaw=None):
    js=list(joystick)+[0.]*(6-len(joystick))
    return js[:6]+[int(use),js[5] if pow_yaw is None else pow_yaw,0.]


def translated(action,hold=0,pow_yaw=0.):
    fwd,back,left,right,jump,yaw,use,aim,blast=action
    if use:hold=12;pow_yaw=float(np.float32(round(aim,6)))
    if hold:yaw=pow_yaw;hold-=1
    f32=lambda value:float(np.float32(value))
    g6=lambda value:float(format(value,'.6g'))
    # Reply text has six decimal places. Torque dAtof and variable assignments
    # are F32; script function arguments use %g (six significant digits).
    # Native numeric callbacks feed the expression stack directly: mSin/mCos
    # results are F32, but do NOT first pass through the argument formatter.
    # Preserve these conversions even near a Move's 1/16 boundary.
    yaw=f32(g6(f32(round(yaw,6))))
    script_yaw=f32(g6(yaw))
    c=f32(math.cos(script_yaw));s=f32(math.sin(script_yaw))
    wx=f32(f32(round(right,6))-f32(round(left,6)))
    wy=f32(f32(round(fwd,6))-f32(round(back,6)))
    cx=g6(f32(wx*c-wy*s));cy=g6(f32(wx*s+wy*c))
    move=dict(x=quantize(cx),y=quantize(cy),z=0.,yaw=0.,pitch=0.,
              jump=bool(jump),use=bool(use),trigger1=False)
    return move,[dict(kind='camera_yaw',value=yaw)],hold,pow_yaw


def observe_proposal_ticks(proposal, states):
    """Deliver completed native sweeps to an item-aware feedback proposal."""
    for _ in range(4):
        hook = getattr(proposal, 'observe_ticks', None)
        if hook is not None:
            hook(states)
            return
        proposal = getattr(proposal, 'after', getattr(proposal, 'follow', None))
        if proposal is None:
            return


def _predict_replies_strict(engine,dynamics,actions,dt_ms=64,stop_on_ambiguity=False):
    if dt_ms!=64:raise ValueError('Only verified local 64 ms lockstep is supported')
    physics_ms=getattr(engine,'bridge_physics_ms',64)
    if physics_ms==16:
        sim=getattr(engine,'native_sim',None)
        if sim is not None:
            # 2026-10-16: compiled physics (nav/native), the same results as predict_small_step
            from nav.native.forecast import predict_native
            return predict_native(sim,engine,dynamics,actions,stop_on_ambiguity)
        from nav.marble_dynamics.bridge_small_step import predict_small_step
        return predict_small_step(engine,dynamics,actions,stop_on_ambiguity)
    if physics_ms!=64:raise ValueError('Only native-verified16ms or64ms physics frames are supported')
    native=dynamics['native']
    state=copy.deepcopy(native['state'])
    if dynamics.get('blast') is not None:state['blast_context']=copy.deepcopy(dynamics['blast'])
    if dynamics.get('inventory') is not None:state['inventory_context']=copy.deepcopy(dynamics['inventory'])
    if state['oob'] or state['mode']!=0:raise ValueError('Playable non-OOB state required')
    clock=int(native['frame_clock_ms'])
    move=copy.deepcopy(native['pending_move']);pending=list(dynamics['bridge_action'])
    hold=int(dynamics['pow_yaw_hold']);aim=float(dynamics['pow_yaw'])
    states=[copy.deepcopy(state)];events=[];controls=[];failure=None
    count=len(actions) if not callable(actions) else int(actions.steps)
    proposed=[]
    for i in range(count):
        check_budget(engine)
        action=actions(state,i) if callable(actions) else actions[i]
        proposed.append(list(action))
        next_move,camera,hold,aim=translated(pending,hold,aim)
        state['clock_ms']=clock+i*dt_ms
        control=dict(move=move,events=camera)
        try:
            check_inventory(state.get('inventory_context'),state,pending)
            impulse=blast_event(state.get('blast_context'),state,bool(pending[8]))
            if impulse is not None:control['script_events']=[impulse]
            move=apply_control(state,control)
            state,ev=engine.step(state,move,dt_ms)
            advance_blast(state.get('blast_context'),impulse is not None,dt_ms)
        except ValueError as exc:
            failure=dict(step=i,reason=str(exc));break
        states.append(copy.deepcopy(state));controls.append(control);events.append(ev)
        move=next_move;pending=list(action)
        if stop_on_ambiguity and any(e['kind']=='ambiguous_contact' for e in ev):break
    return dict(states=states,controls=controls,events=events,proposed=proposed,failure=failure,
                time_s=[i*.064 for i in range(len(states))],
                ambiguous_intervals=[i for i,ev in enumerate(events) if any(e['kind']=='ambiguous_contact' for e in ev)],
                requested_replies=count,stopped_on_ambiguity=bool(stop_on_ambiguity and events and any(e['kind']=='ambiguous_contact' for e in events[-1])),
                assumptions='Local 64 ms lockstep, DELAY 0, SSYAWFIX off; static geometry; observed regular/item-charged Blast context; no triple Blast, Mega, new pickup or respawn prediction')


def validate(folder,output):
    import json
    from nav.collision.mesh import CollisionMap,atomic_json
    from nav.marble_dynamics.record import ML
    from nav.marble_dynamics.physics import GeneralPhysics
    from nav.marble_dynamics.evaluate import load_rows,errors
    folder=folder.resolve();manifest=json.loads((folder/'manifest.json').read_text())
    rows=load_rows(folder/'native.jsonl.gz');engine=GeneralPhysics(CollisionMap(ML/'collision_maps'/(manifest['map']+'.npz')))
    by_clock={r['sim_ms']:r for r in rows};records=[]
    observations=manifest['observations']
    for start,obs in enumerate(observations[:-5]):
        actions=[reply(o['joystick'],o.get('use',0),o.get('pow_yaw')) for o in observations[start:start+5]]
        pred=predict_replies(engine,obs['dynamics'],actions)
        native_rows=[by_clock[obs['dynamics']['native']['frame_clock_ms']+i*64] for i in range(len(pred['controls']))]
        mismatch=[]
        for i,(ctrl,row) in enumerate(zip(pred['controls'],native_rows)):
            if ctrl['move']!=row['move']:mismatch.append(dict(step=i,predicted=ctrl['move'],native=row['move']))
            got=[(e['kind'],e['value']) for e in row['control_events']]
            expected=[(e['kind'],e['value']) for e in ctrl['events']]
            if len(got)!=len(expected) or any(k!=q or abs(v-w)>1e-6 for (k,v),(q,w) in zip(got,expected)):
                mismatch.append(dict(step=i,predicted_camera=expected,native_camera=got))
        records.append(dict(start=start,control_mismatches=mismatch,failure=pred['failure'],
                            errors=errors(pred['states'][-1],native_rows[-1]['after'])))
    report=dict(map=manifest['map'],windows=len(records),control_mismatches=sum(len(r['control_mismatches']) for r in records),
                failures=sum(r['failure'] is not None for r in records),cases=records,
                maximum_errors=np.max([r['errors'] for r in records],axis=0).tolist(),policy_training_started=False)
    atomic_json(output,report)
    print(json.dumps({k:v for k,v in report.items() if k!='cases'}),flush=True)


if __name__=='__main__':
    import argparse
    from pathlib import Path
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();validate(a.folder,a.output)


ACCEPT_AMBIGUITY=True     # 2026-10-18 test: forecasts run through ambiguous contacts and are judged like any other forecast


def mega_relevant(engine,dynamics,actions=None):
    """2026-10-20 Mega proposals (flag --mega-proposals -> engine.mega_enabled): an active Mega, or a held one that
    these actions fire (a proposal with use>=0, or a reply list with a use). Other plans keep the exact compiled path."""
    if not getattr(engine,'mega_enabled',False):return False
    inv=dynamics.get('inventory') or {}
    if inv.get('permanent_mega'):return False
    if inv.get('mega_active'):return True
    if inv.get('held_type')!=6 or actions is None:return False
    if callable(actions):return int(getattr(actions,'use',-1))>=0
    return any(bool(row[6]) for row in actions if len(row)>6)


def predict_replies(engine,dynamics,actions,dt_ms=64,stop_on_ambiguity=False):
    if dt_ms==64 and mega_relevant(engine,dynamics,actions):
        # The compiled forecast refuses Mega transitions (inventory_check); the Python per-tick forecast models the
        # measured script events (growth +16 ms, impulse +32 ms, expiry, shrink) on the same compiled physics step.
        from nav.marble_dynamics.script_forecast import predict_events
        return predict_events(engine,dynamics,actions,model_mega=True)
    if not ACCEPT_AMBIGUITY:return _predict_replies_strict(engine,dynamics,actions,dt_ms,stop_on_ambiguity)
    r=_predict_replies_strict(engine,dynamics,actions,dt_ms,False)
    if r.get('ambiguous_intervals'):r['ambiguous_accepted']=r['ambiguous_intervals'];r['ambiguous_intervals']=[]
    return r
