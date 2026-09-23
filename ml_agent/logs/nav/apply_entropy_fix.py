"""Apply the entropy-controller fix to nav/ppo_recurrent.py (HANDOFF 28.12). Run ONLY between runs:
the trainer reads this module at start, so apply it after the align2 eval (update 16,175) and restart.

Replaces the bang-bang controller (raise below 0.20, cut above 0.30, coefficient floor -0.03) with a
proportional rule toward one target, clamped to [0, ENT_COEF_MAX_P] so it can never become a penalty.
Measured on the 14,255 -> 15,285 lineage: entropy cycled 0.13 -> 0.43 with a ~165-update period, the
coefficient swung -0.03 -> +0.07 and was NEGATIVE 45 % of the time.
"""
import os, re, sys

p = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'nav', 'ppo_recurrent.py')
s = open(p, encoding='utf-8').read()
old_fn = s[s.index('def _adapt_entropy(ent):'):s.index('VALUE_COEF = 0.5')]
new_fn = '''def _adapt_entropy(ent):
    """Proportional controller (2026-09-22, HANDOFF 28.12): move the entropy coefficient toward the
    value that holds the smoothed entropy at ENT_TARGET, clamped to [0, ENT_COEF_MAX_P].

    Replaces the bang-bang rule (raise below ENT_BAND_LO, cut above ENT_CUT_AT, floor -0.03). That rule
    was a limit cycle: entropy 0.13 <-> 0.43 with a ~165-update period, the coefficient -0.03 <-> +0.07
    and NEGATIVE on 45 % of updates, i.e. the policy was alternately pushed to sharpen and to spread on
    a 45-minute cycle, and every A/B eval landed on a different phase of it. The coefficient can no
    longer go negative: penalising entropy drives a hesitant policy toward determinism, the opposite
    of what exploring faster lines near edges needs. Returns (coefficient, smoothed entropy).
    """
    global _ent_coef, _ent_ema
    _ent_ema = ent if _ent_ema is None else 0.9 * _ent_ema + 0.1 * ent
    _ent_coef = _ent_coef + ENT_GAIN * (ENT_TARGET - _ent_ema)
    _ent_coef = min(ENT_COEF_MAX_P, max(0.0, _ent_coef))
    return _ent_coef, _ent_ema


'''
assert 'ENT_BAND_LO' in old_fn, 'controller not found as expected'
s = s.replace(old_fn, new_fn, 1)
consts = '''ENT_TARGET = 0.30      # differential entropy to hold (direction std ~0.22 rad, 12-13 deg): the level the
                       # old controller was effectively pinning at the bottom of its 0.2-0.8 band.
ENT_GAIN = 0.002       # coefficient change per update per unit of entropy error: an error of 0.1
                       # moves the coefficient 0.0002 per update, ~25 updates per 0.005, damped on purpose
ENT_COEF_MAX_P = 0.02  # ceiling for the proportional controller; the floor is 0 (never a penalty)
'''
s = s.replace('_ent_coef = ENTROPY_COEF      # live value, adapted by _adapt_entropy()',
              consts + '_ent_coef = min(ENT_COEF_MAX_P, max(0.0, ENTROPY_COEF))      # live value, adapted by _adapt_entropy()', 1)
s = s.replace('ENTROPY_COEF = 0.01    # RAISED 0.002 -> 0.01 on 2026-09-20 08:15,',
              'ENTROPY_COEF = 0.005   # 2026-09-22: starting value for the proportional controller (28.12).\n                       # (superseded) RAISED 0.002 -> 0.01 on 2026-09-20 08:15,', 1)
open(p, 'w', encoding='utf-8').write(s)
import ast; ast.parse(s)
print('entropy fix applied to', p)
