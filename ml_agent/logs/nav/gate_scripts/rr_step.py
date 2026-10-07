# real_run with the game physics in NAV_PHYS_STEP ms frames while the model still decides every OBS_MS (64) ms
# (2026-10-06, evaluation only; operator: test the best model on the real game's physics step). Needs mlAgent.cs with
# the two-word FIXEDSTEP ("FIXEDSTEP 16 64"). Training is unchanged.
import os, sys, runpy
sys.path.insert(0, os.getcwd())
import nav.env as E
step = int(os.environ.get('NAV_PHYS_STEP', '0'))
if step:
    _set_speed = E.HuntEnv.set_speed

    def set_speed(self, n):
        _set_speed(self, n)
        if E.TRAINING_MODE:
            self.control(f'FIXEDSTEP {step} {E.OBS_MS}')   # after the standard FIXEDSTEP 64 of every (re)connect / round

    E.HuntEnv.set_speed = set_speed
print(f'physics step {step or E.OBS_MS} ms, a decision every {E.OBS_MS} ms', flush=True)
sys.argv = ['nav.real_run']
runpy.run_module('nav.real_run', run_name='__main__')
