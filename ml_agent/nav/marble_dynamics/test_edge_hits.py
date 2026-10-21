import unittest
from types import SimpleNamespace

import numpy as np

from nav.marble_dynamics.edge_hits import lips, candidates, jump_replies, rise_time, climb_needed
from nav.marble_dynamics.search import edge_climb


def square(x0, y0, z, size=4.):
    a = [x0, y0, z]; b = [x0 + size, y0, z]; c = [x0 + size, y0 + size, z]; d = [x0, y0 + size, z]
    # two triangles; polygon boundary edges flagged as mesh.py does (the diagonal is not a boundary)
    return [[a, b, c], [a, c, d]], [[0, 0, 1], [0, 0, 1]], [[True, True, False], [False, True, True]]


class Lips(unittest.TestCase):
    def test_exposed_edges_only(self):
        t1, n1, b1 = square(0, 0, 0); t2, n2, b2 = square(4, 0, 0)   # two flat squares sharing the edge x=4
        segs, ns = lips(t1 + t2, n1 + n2, b1 + b2)
        self.assertEqual(len(segs), 6)   # 8 outer edges; the shared seam is dropped
        vertical = [s for s in segs if abs(s[0, 0] - s[1, 0]) < 1e-6]
        self.assertNotIn(4.0, [round(float(s[0, 0]), 1) for s in vertical])
        self.assertTrue(all(n[2] > .75 for n in ns))

    def test_wall_edge_is_a_lip_but_a_slope_seam_is_not(self):
        t, n, b = square(0, 0, 0)
        wall = [[[4, 0, 0], [4, 4, 0], [4, 4, -3]]]; wn = [[1, 0, 0]]; wb = [[True, True, True]]
        segs, _ = lips(t + wall, n + wn, b + wb)
        self.assertEqual(len(segs), 4)   # the wall shares the x=4 edge but is not a floor: the lip stays exposed
        slope = [[[4, 0, 0], [4, 4, 0], [8, 4, -3]]]; sn = [[0.8, 0, 0.6]]; sb = [[True, True, True]]
        segs, _ = lips(t + slope, n + sn, b + sb)
        self.assertEqual(len(segs), 3)   # a steep slope (not a floor, not a wall) continues the surface: not a lip


class Candidates(unittest.TestCase):
    def engine(self, z=1.5):
        t, n, b = square(0, -2, z)   # a ledge x in [0,4], y in [-2,2], top at height z
        mesh = SimpleNamespace(triangles=np.asarray(t, float), normals=np.asarray(n, float),
                               boundary_edges=np.asarray(b, bool))
        return SimpleNamespace(mesh=mesh)

    def test_climb_needed(self):
        self.assertTrue(climb_needed([0, 0, 0], [10, 0, 2]))
        self.assertFalse(climb_needed([0, 0, 0], [10, 0, .5]))         # not higher enough
        self.assertFalse(climb_needed([0, 0, 0], [85, 0, 1.6]))        # far away and barely higher (Sprawl case)
        self.assertTrue(climb_needed([0, 0, 0], [60, 0, 4]))           # far but a real grade

    def test_lip_of_the_ledge_above(self):
        pts = candidates(self.engine(1.5), [-5, 0, 0], [10, 0, 3])
        self.assertEqual([round(p['point'][0], 2) for p in pts], [0.0, 4.0])   # both lips of the ledge top, nearest first
        self.assertAlmostEqual(pts[0]['rise'], 1.5)
        self.assertAlmostEqual(pts[0]['ahead'], 5.0)

    def test_lips_below_or_too_high_are_skipped(self):
        self.assertEqual(candidates(self.engine(-1.), [-5, 0, 0], [10, 0, 3]), [])    # below the marble
        self.assertEqual(candidates(self.engine(5.), [-5, 0, 0], [10, 0, 7]), [])     # beyond a wall-hit climb

    def test_lips_along_the_approach_are_skipped(self):
        pts = candidates(self.engine(1.5), [2, -8, 0], [2, 10, 3])   # approaching along y: x=0 / x=4 lips run along it
        self.assertTrue(all(abs(abs(p['point'][1]) - 2) < 1e-6 for p in pts))


class JumpTiming(unittest.TestCase):
    def test_rise_time(self):
        self.assertAlmostEqual(rise_time(0.), 0.)
        self.assertIsNone(rise_time(2.))                     # apex is 7.5^2/40 = 1.41 u
        self.assertAlmostEqual(rise_time(1.0), (7.5 - (56.25 - 40.) ** .5) / 20., places=6)

    def test_jump_replies_track_distance_and_speed(self):
        near = jump_replies(3., 10., 1.)        # 0.3 s to the lip, ~0.14 s to rise 0.9 u: jump at reply ~2-3
        far = jump_replies(9., 10., 1.)         # 0.9 s to the lip: jump about 0.6 s in, reply ~11-12
        self.assertLess(max(near), min(far))
        self.assertEqual(len(far), 3)   # within a jump's reach: the lip-height timing and its neighbours only
        self.assertTrue(all(j >= 0 for j in near + far))
        high = jump_replies(5., 10., 2.5)   # above the jump's reach: apex timing plus early-contact timing
        self.assertIn(int(round((0.5 - 0.10) / 0.064)), high)
        self.assertIn(int(round((0.5 - 0.375) / 0.064)), high)
        slow = jump_replies(5., 1., 1.)         # slow marble: timed with the assumed approach speed
        self.assertEqual(slow, jump_replies(5., 10., 1.))


class ClimbGate(unittest.TestCase):
    def test_edge_plan_must_reach_the_lip(self):
        spec = SimpleNamespace(_edge_lip_z=2.0)
        low = dict(states=[dict(position=[0, 0, 0.2]), dict(position=[1, 0, 1.0]), dict(position=[2, 0, 0.3])])
        info = dict(rejected=None); edge_climb(spec, 'edge_hit_0_jump_3', info, low)
        self.assertEqual(info['rejected'], 'edge_no_climb')
        high = dict(states=[dict(position=[0, 0, 0.2]), dict(position=[1, 0, 1.8]), dict(position=[2, 0, 2.1])])
        info = dict(rejected=None); edge_climb(spec, 'edge_hit_0_jump_3', info, high)
        self.assertIsNone(info['rejected'])
        info = dict(rejected=None); edge_climb(SimpleNamespace(), 'direct', info, low)   # ordinary plans untouched
        self.assertIsNone(info['rejected'])


if __name__ == '__main__':
    unittest.main()
