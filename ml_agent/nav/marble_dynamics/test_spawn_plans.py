"""Spawn plans: orders, rows, the chain controller and the whole-spawn score (2026-10-21)."""
import pickle
import unittest
import numpy as np
from nav.marble_dynamics import spawn_plans as SP


def gem(i, x, y, z):
    return dict(id=i, gem=True, position=[x, y, z], bounds=[x - .3, y - .3, z - .3, x + .3, y + .3, z + .3])


ROW = [gem(0, 0., 0., 2.5), gem(1, 2., 0., 3.), gem(2, 4., 0., 2.5), gem(3, 6., 0., 3.)]
OFF = gem(9, 2., 3., 3.5)


class Orders(unittest.TestCase):
    def test_rows_found(self):
        rows = SP.rows(ROW + [OFF])
        self.assertEqual([[g['id'] for g in r] for r in rows], [[0, 1, 2, 3]])

    def test_orders_include_row_topdown_and_aligned(self):
        got = SP.orders([-3., 0., 2.4], [5., 0., 0.], ROW[0], ROW + [OFF]); kinds = dict(got)
        seqs = [[g['id'] for g in s] for _, s in got]
        self.assertIn('nn', kinds); self.assertIn('topdown', kinds)
        self.assertIn([0, 1, 2, 3], seqs)                      # the row (deduplicated against an equal nearest-first order)
        self.assertEqual(kinds['topdown'][0]['id'], 9)          # the tallest gem first
        self.assertTrue(all(len(s) <= SP.CHAIN_MAX for s in kinds.values()))

    def test_row_approached_from_the_near_end(self):
        seqs = [[g['id'] for g in s] for _, s in SP.orders([9., 0., 2.4], [-5., 0., 0.], ROW[3], ROW)]
        self.assertIn([3, 2, 1, 0], seqs)
        self.assertNotIn([0, 1, 2, 3], seqs)


class Chain(unittest.TestCase):
    def test_picklable_and_advances_on_pickup(self):
        cp = SP.ChainProposal(ROW[:3], [None, (.5, .5), None], 48, 6.5, [8., 0., 2.5], 'descend')
        cp = pickle.loads(pickle.dumps(cp))
        a = cp({'position': [-3., 0., 2.7], 'velocity': [5., 0., 0.], 'radius': .2}, 0)
        self.assertEqual(len(a), 9); self.assertEqual(a[4], 0)
        cp.collected.add(0); cp._advance(); self.assertEqual(cp.k, 1)
        # below the pad gem and one hop out at 4 u/s: jump
        a = cp({'position': [2. - SP.launch_distance(4., .5) + .05, 0., 2.7], 'velocity': [4., 0., 0.], 'radius': .2}, 1)
        self.assertEqual(a[4], 1)
        a = cp({'position': [1.5, 0., 2.9], 'velocity': [4., 0., 3.], 'radius': .2}, 2)
        self.assertEqual(a[4], 0)                              # one jump per gem

    def test_arc_mode_jumps_later(self):
        d = SP.ChainProposal(ROW[1:2], [(.5, .5)], 48, 6.5, None, 'descend')
        a = SP.ChainProposal(ROW[1:2], [(.5, .5)], 48, 6.5, None, 'arc')
        # gems sit at floor height: gem z 3.0 on a 0.5 pad, the marble centre at 2.7 (floor 2.5) -> rise 0.5
        fast = {'position': [2. - 3.5, 0., 2.7], 'velocity': [6.5, 0., 0.], 'radius': .2}
        self.assertEqual(d(fast, 0)[4], 0)                    # too fast for 3.5 u: brake first (v2)
        self.assertEqual(a(fast, 0)[4], 0)
        v = 3.5 / SP.hop_time(.8)
        fit = {'position': [2. - 3.5, 0., 2.7], 'velocity': [v, 0., 0.], 'radius': .2}
        self.assertEqual(d(fit, 1)[4], 1); self.assertEqual(a(fit, 1)[4], 0)   # arc waits until just before the gem


class Score(unittest.TestCase):
    def test_two_pickups_beat_one(self):
        def states_through(points):
            return [dict(position=list(p), velocity=[5., 0., 0.], radius=.2) for p in points]
        two = states_through([[-1., 0., 2.5], [0., 0., 2.5], [1., 0., 2.5], [2., 0., 3.], [3., 0., 2.6]])
        one = states_through([[-1., 0., 2.5], [0., 0., 2.5], [.5, 0., 2.5], [.8, 0., 2.5], [1., 0., 2.5]])   # the game's box test reaches 0.4 u
        cp = SP.ChainProposal(ROW[:2], [None, (.5, .5)], 48, 6.5, [4., 0., 2.5], 'descend'); cp.target_gem = np.array(ROW[0]['position'])
        i2 = dict(cost=-100., rejected=None, possible_pickup_interval=1); i1 = dict(cost=-100., rejected=None, possible_pickup_interval=1)
        SP.chain_score(cp, 'chain_nn_2_v6.5_descend', i2, dict(states=two), {})
        SP.chain_score(cp, 'chain_nn_2_v6.5_descend', i1, dict(states=one), {})
        self.assertEqual(i2['chain_pickups'], 2); self.assertEqual(i1['chain_pickups'], 1)
        self.assertLess(i2['cost'], i1['cost'] - 80.)
        self.assertEqual(i2['possible_pickup_interval'], 1)


if __name__ == '__main__':
    unittest.main()
