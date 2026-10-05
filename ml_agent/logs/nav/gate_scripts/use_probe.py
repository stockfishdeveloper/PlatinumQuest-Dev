# Probe the learned use preference in approved states: wraps nav.ss_drill_eval, records p_learn = (sigmoid(use) - EPS) / (1 - EPS)
# at every decision the approval mask allows (up = 1). Run from ml_agent:
#   python use_probe.py --ckpt <pth> --stage 2 --no-use --n 60 --port 9972 --tag probe
import runpy, sys, os, json
import torch
sys.path.insert(0, os.getcwd())
import nav.model as M
vals = []
orig = M.NavActorCritic.heads
def heads(self, h, vec, crop=None):
    out = orig(self, h, vec, crop)
    use = out[4]
    up = self.use_prior(vec)
    p = torch.sigmoid(use)
    pl = (p - M.USE_EPS) / (1.0 - M.USE_EPS)
    for i in range(vec.shape[0]):
        if float(up[i]) > 0.5:
            vals.append(round(float(pl[i]), 4))
    return out
M.NavActorCritic.heads = heads
sys.argv = ['nav.ss_drill_eval'] + sys.argv[1:]
try:
    runpy.run_module('nav.ss_drill_eval', run_name='__main__')
finally:
    import numpy as np
    a = np.array(vals) if vals else np.zeros(1)
    print('USE PROBE: approved decisions %d | p_learn mean %.3f median %.3f p90 %.3f max %.3f | share > 0.5: %.1f%%, > 0.375: %.1f%%' % (
        len(vals), a.mean(), np.median(a), np.percentile(a, 90), a.max(), 100 * (a > 0.5).mean(), 100 * (a > 0.375).mean()), flush=True)
    json.dump(vals, open(os.path.join('logs', 'nav', 'use_probe_vals.json'), 'w'))
