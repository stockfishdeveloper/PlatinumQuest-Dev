"""Fall guard for the hybrid (stage 5b): the navigator drives; near a void edge the step model checks that its next
action still leaves a way to stay on the floor, and replaces it with the safest action when it does not.

The navigator falls about 1.4 times a KOTM round (2026-09-29: 13 rounds), mostly rolling into a hole at walking speed.
A safety filter needs no model of the navigator: after its proposed action, a few fixed recovery programs (brake; steer
toward its target, along the velocity, or away from the nearest edge, each ending in a brake) are rolled out with the
step model. If at least one stays up (no fall, no flight off a lip), the navigator's action is sent. If none does, the
recovery program that stays up is sent instead (the one ending nearest the target); if every one falls anyway, the
navigator's action is kept. Geometry, the model and the navigator's target only: nothing map-specific.
"""
import math

import numpy as np

from nav.learned_nav import planner as PL
from nav.learned_nav import dynamics3 as D3

REACH_S = 1.0                # check only when a void edge is within speed x REACH_S + REACH_U of the marble
REACH_U = 1.5
CLEAR_U = 0.4                # a recovery counts only if it stays this far from every void edge
TURNS = (0.0, math.radians(35), -math.radians(35), math.radians(70), -math.radians(70))


class Guard:
    def __init__(self, planner):
        self.pl = planner            # its geometry, edge index and step ensemble
        self.g = planner.g

    def _edge(self, p):
        """Nearest void or drop edge within the edge index's reach (D3.EDGE_REACH): (distance, outward normal) or None."""
        E = self.pl.eidx.nearest(np.array([p[0]]), np.array([p[1]]), np.array([p[2] - PL.R]), np.ones(1), np.zeros(1))
        E = E.reshape(D3.N_EDGE, 7)
        risky = (E[:, 3] + E[:, 4]) > 0
        if not risky.any():
            return None
        k = int(np.argmin(np.where(risky, E[:, 0], np.inf)))
        return float(E[k, 0]), (float(E[k, 1]), float(E[k, 2]))

    def near_void(self, p, v):
        """A void or drop edge within D3.EDGE_REACH, or floor missing along the velocity within speed x REACH_S +
        REACH_U (sampled every 0.5 u at the marble's floor level)."""
        if self._edge(p) is not None:
            return True
        sp = math.hypot(v[0], v[1])
        if sp < 0.5:
            return False
        reach = sp * REACH_S + REACH_U
        d = np.arange(0.5, reach + 0.01, 0.5)
        ux, uy = v[0] / sp, v[1] / sp
        lv0, _ = D3.level_below(self.g, np.array([p[0]]), np.array([p[1]]), np.array([p[2] + 0.3]))
        if not np.isfinite(lv0[0]):
            return True
        lv, _ = D3.level_below(self.g, p[0] + ux * d, p[1] + uy * d, np.full(len(d), p[2] + 0.3))
        return bool((~np.isfinite(lv) | (lv < lv0[0] - 0.5)).any())

    def _recoveries(self, p, v, target, first=None):
        """Recovery programs; with `first` (mode, angle, throttle, jump), each starts with that decision."""
        sp = math.hypot(v[0], v[1])
        hv = math.atan2(v[1], v[0]) if sp > 0.5 else 0.0
        tg = math.atan2(target[1] - p[1], target[0] - p[0]) if target is not None else hv
        # the nearest void edge's outward normal: steer away from it
        e = self._edge(p)
        away = math.atan2(-e[1][1], -e[1][0]) if e is not None else hv + math.pi
        heads = [tg + t for t in TURNS] + [hv + t for t in TURNS[1:]] + [away]
        n = len(heads) + 1
        pr = PL.Programs(n)
        for k, h in enumerate(heads):
            pr.ang[k] = h
        pr.mode[n - 1] = PL.MODE_BRAKE                              # brake throughout
        pr.mode[:, 24:] = PL.MODE_BRAKE                             # every recovery ends braking (can it stop?)
        if first is not None:
            m, a, th, j = first
            pr.mode[:, 0] = m; pr.ang[:, 0] = a; pr.thr[:, 0] = th; pr.jump[:, 0] = j
        pr.after[:] = PL.MODE_BRAKE
        return pr

    def _roll(self, p, v, w, Uc, Jc, Up, Jp, pr, floor_ref, target):
        n = len(pr)
        sim = PL.simulate(self.pl.steps, self.g, self.pl.eidx, self.pl.dev, np.tile(p, (n, 1)), np.tile(v, (n, 1)),
                          np.tile(w, (n, 1)), np.tile(Uc, (n, 1)), np.full(n, float(Jc)), np.tile(Up, (n, 1)),
                          np.full(n, float(Jp)), pr, False, floor_ref)
        gem = np.asarray(target[:3], float) if target is not None else p
        jd = PL.judge(sim, gem)
        # a margin for model error: the recovery must also keep CLEAR_U from every void edge on the floor (the first
        # version, "does not fall", allowed the navigator's actions until the marble was on the lip)
        clear = np.where(np.isfinite(sim['clear_t']), sim['clear_t'], np.inf).min(1)
        ok = ~jd['fell'] & ~sim['unplanned'] & (clear >= CLEAR_U)
        end = np.linalg.norm(sim['path'][:, -1, :2] - gem[:2], axis=1)
        return ok, end

    def check(self, p, v, w, Uc, Jc, Up, Jp, js_nav, target, floor_ref):
        """(joystick to send, overridden?)"""
        U, J = PL.reply_vector(js_nav)
        mag = float(np.hypot(U[0], U[1]))
        first = (PL.MODE_DIR, math.atan2(U[1], U[0]) if mag > 1e-6 else 0.0, min(1.0, mag / PL.SQ2), int(J > 0))
        if mag <= 1e-6:
            first = (PL.MODE_NONE, 0.0, 0.0, int(J > 0))
        ok, _ = self._roll(p, v, w, Uc, Jc, Up, Jp, self._recoveries(p, v, target, first), floor_ref, target)
        if ok.any():
            return js_nav, False
        pr = self._recoveries(p, v, target)
        ok, end = self._roll(p, v, w, Uc, Jc, Up, Jp, pr, floor_ref, target)
        if not ok.any():
            return js_nav, False
        k = int(np.argmin(np.where(ok, end, np.inf)))
        return self.pl.reply(pr, k, v), True
