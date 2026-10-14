"""Experimental measured 16ms-physics/64ms-decision bridge.

Camera consumes the previous reply at the first native frame. Movement changes
at the second16ms frame, then remains held for the other two. The new reply waits
until the next macro observation. No native future controls/states are inputs.
Native fixture timing: previous reply Blast before the first frame, powerup Move
at the second frame. Blast meter resets after the first frame's charge, then
charges in the three remaining frames. Main evaluator integration is separate.
"""
import copy
from nav.marble_dynamics.state_copy import clone as clone_state

from nav.marble_dynamics.bridge import translated
from nav.marble_dynamics.inventory import check as check_inventory
from nav.marble_dynamics.rollout import apply_control
from nav.marble_dynamics.blast import event as blast_event,advance as advance_blast
from nav.marble_dynamics.planning_budget import check as check_budget


def predict_small_step(engine,dynamics,actions,stop_on_ambiguity=False):
    native=dynamics['native'];state=clone_state(native['state'])
    if state['oob'] or state['mode']!=0:raise ValueError('Playable non-OOB state required')
    if dynamics.get('blast') is not None:state['blast_context']=copy.deepcopy(dynamics['blast'])
    if dynamics.get('inventory') is not None:state['inventory_context']=copy.deepcopy(dynamics['inventory'])
    clock=int(native['frame_clock_ms']);move=copy.deepcopy(native['pending_move'])
    pending=list(dynamics['bridge_action']);hold=int(dynamics['pow_yaw_hold']);aim=float(dynamics['pow_yaw'])
    count=len(actions) if not callable(actions) else int(actions.steps)
    states=[clone_state(state)];native_states=[clone_state(state)]
    controls=[];native_controls=[];events=[];proposed=[];failure=None;stopped=False
    for i in range(count):
        check_budget(engine)
        action=actions(state,i) if callable(actions) else actions[i]
        proposed.append(list(action));next_move,camera,hold,aim=translated(pending,hold,aim)
        frames=[];macro_events=[]
        try:
            check_inventory(state.get('inventory_context'),state,pending)
            for subframe in range(4):
                state['clock_ms']=clock+i*64+subframe*16
                control=dict(move=copy.deepcopy(move if subframe==0 else next_move),events=camera if subframe==0 else [])
                impulse=blast_event(state.get('blast_context'),state,bool(pending[8])) if subframe==0 else None
                if impulse is not None:control['script_events']=[impulse]
                apply_control(state,control);state,ev=engine.step(state,control['move'],16)
                advance_blast(state.get('blast_context'),impulse is not None,16)
                frames.append(control);native_controls.append(control)
                native_states.append(clone_state(state));macro_events.extend(ev)
                if stop_on_ambiguity and any(e['kind']=='ambiguous_contact' for e in ev):
                    stopped=True;break
        except ValueError as exc:
            failure=dict(step=i,reason=str(exc));break
        macro=dict(move=copy.deepcopy(frames[0]['move']),events=copy.deepcopy(frames[0]['events']),subframes=frames)
        script_events=[ev for frame in frames for ev in frame.get('script_events',[])]
        if script_events:macro['script_events']=script_events
        states.append(clone_state(state));controls.append(macro);events.append(macro_events)
        move=next_move;pending=list(action)
        if stopped:break
    return dict(states=states,controls=controls,events=events,proposed=proposed,failure=failure,
        native_states=native_states,native_controls=native_controls,
        time_s=[i*.064 for i in range(len(states))],
        ambiguous_intervals=[i for i,ev in enumerate(events) if any(e['kind']=='ambiguous_contact' for e in ev)],
        requested_replies=count,stopped_on_ambiguity=stopped,physics_frame_ms=16,decision_ms=64,
        assumptions='Measured local16ms physics/64ms decisions, static geometry, ordinary movement/jump and native powers1-5, observed regular/item-charged Blast. Timed Mega/triple Blast/Fireball/cannon and external pickup/respawn transitions refused/unmodeled. No future native inputs.')
