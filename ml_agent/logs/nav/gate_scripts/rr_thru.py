# real_run with the through-gem helper's settings overridden for this run only (2026-10-05; evaluation / watching).
# THRU_ACT_U, THRU_MAX_TURN_DEG, THRU_GAIN_S, THRU_PICK_MARGIN_U override nav.through_gem; NAV_THROUGH_GEM=1 must be set.
import os, sys, runpy
sys.path.insert(0, os.getcwd())
import nav.through_gem as T
for name, key in (('ACT_U', 'THRU_ACT_U'), ('MAX_TURN_DEG', 'THRU_MAX_TURN_DEG'), ('GAIN_S', 'THRU_GAIN_S'),
                  ('PICK_MARGIN_U', 'THRU_PICK_MARGIN_U')):
    if os.environ.get(key):
        setattr(T, name, float(os.environ[key]))
print('through_gem', {n: getattr(T, n) for n in ('ACT_U', 'MAX_TURN_DEG', 'GAIN_S', 'PICK_MARGIN_U')}, flush=True)
sys.argv = ['nav.real_run']
runpy.run_module('nav.real_run', run_name='__main__')
