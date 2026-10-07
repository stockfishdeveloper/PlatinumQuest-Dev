# real_run with the kick-vs-no-kick approval switched on for this evaluator only (2026-10-05; evaluation, not training).
# PATCH_COMPARE=1: nav.ss_two_gem.COMPARE_NO_KICK = True (the no-kick drive must arrive on floor; next-gem cases only).
import os, sys, runpy
sys.path.insert(0, os.getcwd())
import nav.ss_two_gem as T
if os.environ.get('PATCH_COMPARE', '0') == '1':
    T.COMPARE_NO_KICK = True
print('COMPARE_NO_KICK', T.COMPARE_NO_KICK, 'horizon', T.COMPARE_HORIZON_S if T.COMPARE_NO_KICK else T.FAST_HORIZON_S, flush=True)
sys.argv = ['nav.real_run']
runpy.run_module('nav.real_run', run_name='__main__')
