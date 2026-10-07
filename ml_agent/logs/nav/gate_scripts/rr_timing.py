# real_run with the LOBBY / smooth-viewer timing reproduced in fast lockstep games (2026-10-06, evaluation only):
# the game physics in 16 ms frames with an observation every 16 ms (FIXEDSTEP 16), a decision every 4 observations
# (set NAV_VIEW_SUBSTEPS=4: real_run repeats the decision over the remaining slices), and an optional fixed action
# delay in 16 ms ticks (NAV_ACTION_DELAY, the bridge's DELAY word; 4 = a decision takes effect 64 ms after the
# observation it was made on, as in training). Training is unchanged.
import os, sys, runpy
sys.path.insert(0, os.getcwd())
import nav.env as E
step = int(os.environ.get('NAV_PHYS_STEP', '16'))
delay = int(os.environ.get('NAV_ACTION_DELAY', '0'))
_set_speed = E.HuntEnv.set_speed


def set_speed(self, n):
    _set_speed(self, n)
    if E.TRAINING_MODE:
        self.control(f'FIXEDSTEP {step}')          # observation interval follows the step (16 ms, like the lobby)
        self.control(f'DELAY {delay}')


E.HuntEnv.set_speed = set_speed
print(f'lobby timing: physics and observations every {step} ms, {os.environ.get("NAV_VIEW_SUBSTEPS", "1")} observations '
      f'per decision, action delay {delay} ticks', flush=True)
sys.argv = ['nav.real_run']
runpy.run_module('nav.real_run', run_name='__main__')
