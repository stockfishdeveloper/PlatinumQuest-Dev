"""Super-Speed-aware gem order (2026-10-05 20:30; evaluation opt-in NAV_SS_TOUR in nav/real_run.py, training unchanged).

While a Super Speed is held, credit the gem orders whose first pickup sets up a safe post-pickup kick: after taking gem A,
a kick through B and then C (the one-push rule of nav.ss_two_gem, from the state the marble arrives at A in) that beats
the same drive without a kick. The credit (time saved x CREDIT_SPEED, in the tour's distance units) is subtracted from
those orders in nav.gems.plan_tour, so the route prefers arriving where a kick pays, which is how the operator's demo gets
its post-pickup turnarounds. Physics, terrain heights and visible gems only; nothing map specific.
"""
import math
import numpy as np
from nav import ss_two_gem as S
from nav.protocol import RAW_POW_HELD

ARRIVE_SPEED = 8.0          # u/s at a pickup (the navigator brakes into gems: ~7.7 measured)
CREDIT_SPEED = 9.0          # u/s: converts seconds saved into the tour's distance units (typical travel speed)
MAX_FIRST = 3               # candidate first gems (nearest)
MAX_SECOND = 2              # candidate second gems per first (nearest to it, at least SS_MIN_U away)
SS_MIN_U = 8.0              # u: the use prior fires only with the waypoint at least this far (model.USE_RUN_U)
MARBLE_R = 0.19


def _xy(g):
    return np.array([float(g[0]), float(g[1])])


def kick_credits(terrain, raw, gems):
    """{(A, B, C): credit in u}, keyed by rounded gem positions (key_of); {} unless a Super Speed is held. The caller
    maps the keys onto the current gem tuples (their distance field changes every decision): order_credits()."""
    if len(raw) <= RAW_POW_HELD or int(raw[RAW_POW_HELD]) != 2 or len(gems) < 3:
        return {}
    pos = np.array([float(raw[0]), float(raw[1])])
    firsts = sorted(gems, key=lambda g: float(np.linalg.norm(_xy(g) - pos)))[:MAX_FIRST]
    out = {}
    for A in firsts:
        a = _xy(A)
        d = a - pos
        dn = float(np.linalg.norm(d))
        if dn < 1e-6:
            continue
        v_arr = ARRIVE_SPEED * d / dn
        w_arr = np.array([-v_arr[1] / MARBLE_R, v_arr[0] / MARBLE_R, 0.0])     # rolling: r (w_y, -w_x) = v
        za = float(terrain.floor_z(a[0], a[1], float(A[2]))) + MARBLE_R
        rest = [g for g in gems if g is not A]
        seconds = [g for g in sorted(rest, key=lambda g: float(np.linalg.norm(_xy(g) - a)))
                   if float(np.linalg.norm(_xy(g) - a)) >= SS_MIN_U][:MAX_SECOND]
        for B in seconds:
            b = _xy(B)
            others = [g for g in rest if g is not B]
            if not others:
                continue
            C = min(others, key=lambda g: float(np.linalg.norm(_xy(g) - b)))
            c = _xy(C)
            ok, aim, _ = S.plan_fast(terrain, a, za, v_arr, w_arr, b, c)
            if not ok:
                continue
            aims = np.array([aim, aim]); kicks = np.array([1.0, 0.0])
            t1, t2, off, v2, p2, z2 = S._fast_batch(terrain, a, za, v_arr, w_arr, aims, b, c,
                                                    horizon_s=S.COMPARE_HORIZON_S, kicks=kicks)
            t_kick = float(t2[0]); t_no = float(t2[1]) if np.isfinite(t2[1]) else S.COMPARE_HORIZON_S
            saving = t_no - t_kick
            if np.isfinite(t_kick) and saving > 0:
                out[(key_of(A), key_of(B), key_of(C))] = saving * CREDIT_SPEED
    return out


def key_of(g):
    return (round(float(g[0]), 1), round(float(g[1]), 1))


def order_credits(cache, gems):
    """The cached credits re-keyed onto the current gem tuples, for nav.gems.plan_tour(order_credits=...)."""
    if not cache:
        return None
    by = {key_of(g): g for g in gems}
    out = {(by[a], by[b], by[c]): v for (a, b, c), v in cache.items() if a in by and b in by and c in by}
    return out or None
