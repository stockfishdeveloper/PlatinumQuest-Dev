"""Regular and item-charged Blast script actuation for the owned local bridge.

The context is observed game state, not a granted resource. Triple Blast,
Fireball and cannons remain explicit unsupported branches. Script/vector F32 and
six-significant-digit boundaries are part of the impulse, including mass division.
"""
import math
import numpy as np


def f32(value):return float(np.float32(value))
def script_arg(value):return f32(format(f32(value),'.6g'))


def available(context):
    return bool(context and context['enabled'] and not context['frozen']
        and not any(context[k] for k in ('triple','fireball','cannon'))
        and context['value']>=context['required'])


def regular_available(context):
    return available(context) and not context['special']


def context_changes(predicted,observed_dynamics):
    context=predicted.get('blast_context');observed=observed_dynamics.get('blast')
    if context is None or observed is None:return []
    return ['blast_context:'+key for key,value in context.items()
        if (abs(value-observed[key])>2e-6 if key=='value' else value!=observed[key])]


def event(context,state,requested):
    if not requested:return None
    if context is None:raise ValueError('Blast context unavailable in this snapshot')
    if any(context[k] for k in ('triple','fireball','cannon')):
        raise ValueError('Triple Blast, Fireball or cannon actuation is not verified')
    if not available(context):return None
    # mSqrt takes a script argument; multiplication is F32, SPC formats the
    # vector, VectorMult formats it again, and applyImpulse receives Point3F.
    amount=f32(context['special_power']) if context['special'] else f32(math.sqrt(max(0.,script_arg(context['value']))))
    scalar=script_arg(f32(amount*-f32(context['power'])))
    gravity=[script_arg(x) for x in state['gravity_direction']]
    push=[script_arg(f32(scalar*g)) for g in gravity]
    mass=f32(state['parameters']['mass'])
    if mass<=0:raise ValueError('Positive native mass required for Blast')
    return dict(kind='blast_impulse',impulse=push,
        delta_velocity=[f32(f32(x)/mass) for x in push],meter_before=context['value'],special=bool(context['special']))


def advance(context,fired,dt_ms):
    if context is None:return
    if fired:
        # Frame advancement charged BEFORE the script observation/action; the
        # reset is therefore visible as zero at the next observation.
        context['value']=0.
        context['special']=False
    elif context['charging']:
        if context['charge_ms']<=0:raise ValueError('Positive Blast charge time required')
        context['value']=min(1.,max(0.,f32(f32(context['value'])+f32(dt_ms/context['charge_ms']))))
