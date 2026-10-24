"""Pad hops: timing from the jump constants and pad detection on a synthetic mesh (2026-10-21)."""
import math
import unittest
import numpy as np
from nav.marble_dynamics import pad_hops as PH


class FakeMesh:
    """A flat floor at z=0 with a round pad of radius 0.75 and top z=0.5 centred at (10, 10)."""
    def __init__(self, pad_h=.5, pad_r=.75):
        self.pad_h = pad_h; self.pad_r = pad_r

    def raycast(self, a, b, cull_backfaces=True):
        x, y = float(a[0]), float(a[1])
        z = self.pad_h if math.hypot(x - 10., y - 10.) <= self.pad_r else 0.
        if a[2] >= z >= b[2]:
            return {'point': np.array([x, y, z]), 'normal': np.array([0., 0., 1.]), 't': float(a[2] - z)}
        return None


class Engine:
    def __init__(self, mesh):
        self.mesh = mesh


class HopTiming(unittest.TestCase):
    def test_descending_time_matches_the_demo(self):
        # pad top 0.5 u, aim 0.3 above it: back down at 0.62 s (the human's median jump -> pickup time)
        self.assertAlmostEqual(PH.hop_time(.8), .621, places=2)
        self.assertIsNone(PH.hop_time(2.))              # above the apex

    def test_launch_distance_at_the_demo_speed(self):
        d = PH.launch_distance(3.8, .5)
        self.assertTrue(2.2 <= d <= 2.5, d)             # the human: 2.2 u median (1.5-2.9)
        self.assertGreater(PH.launch_distance(6.5, .5), d)


class PadDetection(unittest.TestCase):
    def test_pad_gem(self):
        pad = PH.pad_under(Engine(FakeMesh()), [10., 10., .9])
        self.assertIsNotNone(pad)
        h, r = pad
        self.assertAlmostEqual(h, .5, places=2)
        self.assertTrue(.5 <= r <= 1., r)

    def test_flat_gem(self):
        self.assertIsNone(PH.pad_under(Engine(FakeMesh()), [3., 3., .4]))

    def test_tall_block_is_not_a_pad(self):
        self.assertIsNone(PH.pad_under(Engine(FakeMesh(pad_h=2.5)), [10., 10., 2.9]))

    def test_blast_gate(self):
        e = Engine(FakeMesh())
        self.assertTrue(PH.blast_gated(e, [10., 10., .9], {'position': [6., 10., .2]}))
        self.assertFalse(PH.blast_gated(e, [3., 3., .4], {'position': [0., 3., .2]}))      # flat gem: Blasts stay
        self.assertFalse(PH.blast_gated(e, [10., 10., .9], {'position': [-5., 10., .2]}))  # too far for the gate


if __name__ == '__main__':
    unittest.main()


class PadArrival(unittest.TestCase):
    def test_arrival_speed_before_a_pad_gem(self):
        e = Engine(FakeMesh())
        gem = [7.5, 10., .4]                     # a flat gem 2.5 u before the pad gem at (10, 10)
        through = (None, [10., 10., .9], 2.5)
        v = PH.hop_arrival_speed(e, gem, through)
        self.assertIsNotNone(v)
        self.assertTrue(3.5 <= v <= 4.5, v)      # 2.5 u / 0.62 s
        self.assertIsNone(PH.hop_arrival_speed(e, gem, (None, [3., 3., .4], 5.)))   # next gem flat: no rule
        self.assertIsNone(PH.hop_arrival_speed(e, [-2., 10., .4], (None, [10., 10., .9], 12.)))   # too far
