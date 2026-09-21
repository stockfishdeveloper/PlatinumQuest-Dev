"""Inference-time test: does strengthening the dir_head's goal-bearing path actually buy speed?

WHAT THIS TESTS. The direction head reads [hidden state, observation vector]. Measured on
nav_eval_flatgem_1208.pth (HANDOFF 21a): it is 65x more sensitive to the hidden state than to the
goal bearing wired straight into it, because training grew the hidden columns to 2.10x their
initialisation and shrank the goal columns to 0.83x. The commanded direction therefore points at
the gem ON AVERAGE but scatters +-60-70 deg around it, and useful thrust is the cosine of that
scatter: 0.57 for the agent against 0.91 for the human.

Replaying real chases through modified copies of the head showed the steadiness is recoverable by
re-weighting alone: aim concentration 0.452 as trained, 0.709 at x10, 0.893 at x30 (human 0.91).

This script applies that scaling at INFERENCE TIME ONLY and plays real rounds with it. Nothing is
trained and no checkpoint is written, so it is a pure causal test of the chain

    stronger goal-bearing path -> steadier aim -> more useful thrust -> more speed -> more gems

If speed and gems do not move while aim concentration does, the chain is broken somewhere and no
amount of training on this idea is worth spending.

    NAV_DIRSKIP=10 python -m nav.real_run        (via this module's patching; see run_dirskip.ps1)
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SCALE = float(os.environ.get('NAV_DIRSKIP', '1'))


def patch(model):
    """Scale the dir_head's goal-bearing columns (vec[0:2]) by SCALE, in place."""
    from nav.model import HIDDEN
    if SCALE == 1.0:
        print('[dirskip] NAV_DIRSKIP=1, model left unmodified')
        return model
    with torch.no_grad():
        w = model.dir_head[0].weight
        before = w[:, HIDDEN:HIDDEN + 2].norm().item()
        w[:, HIDDEN:HIDDEN + 2] *= SCALE
        after = w[:, HIDDEN:HIDDEN + 2].norm().item()
    print('[dirskip] goal-bearing columns scaled x%g: weight norm %.3f -> %.3f' % (SCALE, before, after))
    return model


# Patch NavActorCritic.load_state_dict so real_run picks the change up without being edited.
def install():
    from nav.model import NavActorCritic
    orig = NavActorCritic.load_state_dict

    def wrapped(self, *a, **k):
        out = orig(self, *a, **k)
        patch(self)
        return out
    NavActorCritic.load_state_dict = wrapped


install()

if __name__ == '__main__':
    from nav import real_run
    real_run.main()
