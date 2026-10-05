# real_run with inference-time patches (40.33/40.34 diagnosis).
# PATCH_RES=off: no residual; post: residual only while a kick is pending or within SS_RES_RECENT_S of one; keep: as is.
# PATCH_LEVEL=<dz>: a Super Speed kick also needs LEVEL braking room: floor within dz of the kick's floor height along
# the resulting heading for the stop (or to the next gem + margin); a bank or rim the marble would climb does not count.
import os, sys, runpy, math
import numpy as np
sys.path.insert(0, os.getcwd())
import nav.model as M
import nav.obs as O
mode = os.environ.get('PATCH_RES', 'keep')
if mode == 'off':
    M.SS_RES_SCALE = 0.0
elif mode == 'post':
    def g(vec):
        pending = vec[:, M.VEC_USE + 2] > 0.5
        recent = vec[:, M.VEC_USE + 5] < (M.SS_RES_RECENT_S / M.USE_T_SCALE)
        return (pending | recent).float()
    M.NavActorCritic.ss_gate = staticmethod(g)
LAM = os.environ.get('PATCH_LAM')
if LAM:
    O.SS_LAM_MAX = float(LAM)
LEVEL_DZ = float(os.environ.get('PATCH_LEVEL', '0'))
if LEVEL_DZ > 0:
    orig_ok = O.ss_kick_ok
    def level_run(terrain, x, y, z, hx, hy, max_u, step=0.5):
        z0 = terrain.floor_height(x, y, z)
        ds = (np.arange(1, int(max_u / step) + 1) * step).astype(np.float32)
        h = terrain.heights_at(x + hx * ds, y + hy * ds)
        for s in range(len(ds)):
            col = h[:, s]; fin = np.isfinite(col)
            if not fin.any() or float(np.min(np.abs(col[fin] - z0))) > LEVEL_DZ:
                return float(ds[s])
        return math.inf
    def ok(terrain, x, y, z, rvx, rvy, sp, d_target, ux, uy, nxt_rel=None):
        if not orig_ok(terrain, x, y, z, rvx, rvy, sp, d_target, ux, uy, nxt_rel):
            return False
        if sp < 1e-3:
            return True
        n = math.hypot(rvx, rvy); hx, hy = rvx / n, rvy / n
        run = level_run(terrain, x, y, z, hx, hy, O.SS_SCAN_U)
        along = max(0.0, d_target * (hx * ux + hy * uy))
        if run > max(sp * sp / (2.0 * O.SS_DECEL), along) + O.SS_RUN_MARGIN:
            return True
        if nxt_rel is not None:
            ndx, ndy = nxt_rel; nd = math.hypot(ndx, ndy)
            if nd > along and (ndx * hx + ndy * hy) / max(nd, 1e-6) >= O.SS_NEXT_COS and run > nd + O.SS_NEXT_MARGIN:
                return True
        return False
    O.ss_kick_ok = ok
print('PATCH_RES', mode, 'PATCH_LEVEL', LEVEL_DZ, 'SS_LAM_MAX', O.SS_LAM_MAX, flush=True)
sys.argv = ['nav.real_run']
runpy.run_module('nav.real_run', run_name='__main__')
