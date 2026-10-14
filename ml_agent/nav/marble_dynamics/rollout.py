"""Bounded candidate prediction without a future observation tape.

Controls contain already-applied native Moves plus absolute camera-command events,
not delayed bridge ACTION commands. A plain Move assumes no absolute camera setter.
Start with complete recorded/live state, then advance the known simulation clock.
No future measured pose, spin, contact normal, effect state or camera is supplied.
External pickups, respawns, size/gravity changes and moving actors are not modeled.
"""
import copy
import numpy as np


def apply_control(state,control):
    """Apply recorded/proposed exogenous commands; never copy observed camera state."""
    if 'move' not in control:return control
    for event in control.get('script_events',[]):
        if event['kind']!='blast_impulse':raise ValueError('Unsupported script actuation: '+event['kind'])
        state['velocity']=(np.asarray(state['velocity'])+event['delta_velocity']).tolist()
    for event in control.get('events',[]):
        value=float(np.float32(event['value']))
        if event['kind']=='camera_yaw':state['yaw']=value;state['next_yaw']=value
        elif event['kind']=='camera_pitch':state['pitch']=value
        else:raise ValueError('Unsupported external command: '+event['kind'])
    return control['move']


def predict(engine,initial_state,moves,dt_ms=16,max_steps=1024,hierarchy=None):
    if not isinstance(dt_ms,int) or dt_ms<=0:raise ValueError('Positive integer dt_ms required')
    if max_steps<1:raise ValueError('Positive max_steps required')
    if 'clock_ms' not in initial_state:raise ValueError('Complete state with simulation clock required')
    if hierarchy is not None and hierarchy.mesh.info['mesh_sha256']!=engine.mesh.info['mesh_sha256']:
        raise ValueError('Candidate hierarchy uses different geometry')
    state=copy.deepcopy(initial_state);states=[state];events=[];controls=[];failure=None
    clock=int(state['clock_ms'])
    for i,move in enumerate(moves):
        if i>=max_steps:raise ValueError('Candidate exceeds bounded prediction length')
        before=copy.deepcopy(states[-1]);before['clock_ms']=clock+i*dt_ms
        control=copy.deepcopy(move)
        try:
            move=apply_control(before,control)
            state,step_events=engine.step(before,move,dt_ms)
        except ValueError as exc:
            failure=dict(sample=i,reason=str(exc));break
        states.append(state);events.append(step_events);controls.append(control)
    result=dict(states=states,events=events,controls=controls,
                time_s=[i*dt_ms*.001 for i in range(len(states))],
                next_clock_ms=clock+len(events)*dt_ms,failure=failure,
                ambiguous_intervals=[i for i,ev in enumerate(events) if any(e['kind']=='ambiguous_contact' for e in ev)],
                uncertainty_calibrated=False,planner_approved=False,
                assumptions='Known applied Move timing; static geometry; no external discrete events')
    if hierarchy is not None:
        result['corridor_triangle_ids']=hierarchy.candidate_corridor(
            [s['position'] for s in states],max(s['radius'] for s in states)).tolist()
    return result
