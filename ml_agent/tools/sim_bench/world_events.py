"""World-side pickup and Mega transitions, independent of the planner's proposal switches."""
from nav.marble_dynamics.script_forecast import forecast_items, predict_events


def advance_world(engine, dynamics, action):
    """Advance real world events even when the planner cannot propose their use.

    In particular, different powerups replace inventory, and a touched Mega is
    script-held (native held=0). The proposal flag must be restored on failure too.
    """
    enabled = getattr(engine, 'mega_enabled', False)
    native = getattr(engine, 'native_sim', None)
    had_native = hasattr(engine, 'native_sim')
    engine.mega_enabled = not (dynamics.get('inventory') or {}).get('permanent_mega', False)
    try:
        if native is None:
            class PythonStep:
                def step(self, state, move, dt_ms):
                    return engine.step(state, move, dt_ms)
            engine.native_sim = PythonStep()
            result = predict_events(engine, dynamics, [action], model_mega=engine.mega_enabled)
            result['assumptions'] = '16 ms Python physics with tick-ordered item and Mega world events'
            return result
        return forecast_items(engine, dynamics, [action])
    finally:
        engine.mega_enabled = enabled
        if had_native:engine.native_sim = native
        elif hasattr(engine, 'native_sim'):del engine.native_sim


def consumed_powerups(rollout):
    return {event['item']['id'] for event in rollout.get('item_events', [])
            if not event['item'].get('gem')}
