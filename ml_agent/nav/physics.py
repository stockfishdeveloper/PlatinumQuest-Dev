"""Measured marble physics shared by the terrain graph, the observation and the model (2026-09-23).

Everything here comes from physics/jump_envelope.json, written by `python -m nav.measure_jump`
(the game as the oracle: teleport onto a flat strip at speed v, jump, record apex / flight / range).
It describes the MARBLE, not a map, so it holds on every map with the same marble datablock.

KOTM measurement 2026-09-23 (step 16 ms, jump held 2 ticks, forward held through the flight):
    apex 1.33-1.35 u and flight 0.75-0.77 s at every speed (a jump adds ~7.3 u/s vertically),
    range(v) = 0.75 v + 2.0 u  (v = 5.05 -> 5.95, 9.52 -> 8.85, 13.29 -> 11.71, 17.16 -> 14.67).
Two identical trials per condition: the simulation is deterministic under the fixed step.
"""
import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENVELOPE_PATH = os.path.join(_HERE, 'physics', 'jump_envelope.json')

# fallbacks (the KOTM measurement above) if the json is missing
_DEFAULT_SPEED = np.array([0.0, 5.05, 5.61, 7.72, 9.52, 11.39, 13.29, 15.22, 17.16])
_DEFAULT_RANGE = np.array([1.81, 5.95, 6.39, 7.28, 8.85, 10.26, 11.71, 13.18, 14.67])
JUMP_APEX = 1.33            # u: the most a jump can rise (landing must be lower than this above the lip)
JUMP_FLIGHT_S = 0.76        # s: same-height flight time
LANDING_MARGIN = 0.5        # u: the marble centre must clear the far lip by this much to count as crossable
CRUISE_SPEED = 8.0          # u/s: the speed the terrain graph assumes when admitting a jump edge
                            # (the navigator's measured pace between pickups is 7-9 u/s)


def _load():
    try:
        d = json.load(open(ENVELOPE_PATH))
        rows = sorted(d.get('hold', []), key=lambda r: r['speed'])
        rows = [r for r in rows if r['range'] > 0]
        if len(rows) >= 3:
            sp = np.array([r['speed'] for r in rows]); rg = np.array([r['range'] for r in rows])
            apex = float(np.median([r['apex'] for r in rows])); fl = float(np.median([r['flight_s'] for r in rows]))
            return sp, rg, apex, fl
    except (OSError, ValueError, KeyError):
        pass
    return _DEFAULT_SPEED, _DEFAULT_RANGE, JUMP_APEX, JUMP_FLIGHT_S


_SPEED, _RANGE, JUMP_APEX, JUMP_FLIGHT_S = _load()
_SLOPE = float((_RANGE[-1] - _RANGE[-3]) / (_SPEED[-1] - _SPEED[-3]))    # extrapolation beyond the table


def jump_range(speed):
    """Horizontal distance (u) a jump covers from takeoff speed `speed` (u/s), forward held, landing
    at the takeoff height. Scalar or array."""
    s = np.asarray(speed, dtype=np.float64)
    r = np.interp(s, _SPEED, _RANGE)
    over = s > _SPEED[-1]
    if np.any(over):
        r = np.where(over, _RANGE[-1] + (s - _SPEED[-1]) * _SLOPE, r)
    return float(r) if np.ndim(speed) == 0 else r


def crossable(gap, speed, margin=LANDING_MARGIN):
    """True if a gap of `gap` u (lip to far floor) is within a jump from takeoff speed `speed`."""
    return np.asarray(gap) + margin <= jump_range(speed)


def speed_for_gap(gap, margin=LANDING_MARGIN):
    """Takeoff speed (u/s) needed to clear `gap` u with the landing margin."""
    need = float(gap) + margin
    return float(np.interp(need, _RANGE, _SPEED)) if need <= _RANGE[-1] else float(_SPEED[-1] + (need - _RANGE[-1]) / _SLOPE)


MAX_JUMP_GAP = float(jump_range(CRUISE_SPEED) - LANDING_MARGIN)   # the widest gap the graph admits (~7.5 u)
