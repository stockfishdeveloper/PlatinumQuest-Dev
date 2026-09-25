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
LANDING_MARGIN = 1.0        # 0.5 -> 1.0 on 2026-09-24 (HANDOFF 28.34): approved jumps succeeded only 66-71 % at 0.5
                            # (forced-jump eval 117 pts / 56 falls). With 1.0 the marble centre must clear the far lip
                            # by a full unit: the 7 u holes drop out at cruise speed, corner cuts and bays stay in.
CRUISE_SPEED = 11.0         # 8 -> 11 on 2026-09-24 21:40 (HANDOFF 28.36): the speed the terrain graph assumes when
                            # admitting a jump edge. The human demo of 21:28 made 17 gap jumps (all landed) at a mean
                            # takeoff speed of 10.7 u/s crossing 9-15 u; at 8 u/s the graph admitted <= 6.5 u and held
                            # an edge for 2 of the 17. The field must value the cut at the speed a marble CAN carry;
                            # the crossable flag (speed-conditioned) still decides when it actually may jump.
MIN_JUMP_GAP = 3.0          # u: gaps narrower than this are rolled around, never jumped (the 2 u centre hole made the
                            # policy 'jump for the sake of jumping', operator 2026-09-24). KOTM-ONLY HACK, per the
                            # operator: NOT a long-term lower bound. Remove (0.0) or replace with a detour-based test
                            # before training other maps, where short gaps may be the jumps that matter.


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
