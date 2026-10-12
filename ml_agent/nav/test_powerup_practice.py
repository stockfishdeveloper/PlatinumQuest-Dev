"""Handling resets must consume existing inventory without creating actor transitions."""
import unittest
from types import SimpleNamespace
import numpy as np

from nav.vec_worker import InstanceWorker
from nav.protocol import RAW_POW_HELD, RAW_POW_BLAST


class HandlingTests(unittest.TestCase):
    def worker(self, held=1, charge=.8, consumes=True):
        w = InstanceWorker.__new__(InstanceWorker)
        raw = np.zeros(61); raw[RAW_POW_HELD] = held; raw[RAW_POW_BLAST] = charge
        w.env = SimpleNamespace(msg=SimpleNamespace(obs=raw), replay_world={'clock_ms': 1000})
        calls = []
        def step(js, **kwargs):
            calls.append(kwargs)
            w.env.replay_world['clock_ms'] += 64
            if consumes and kwargs.get('use_pow'):
                raw[RAW_POW_HELD] = 0
            if consumes and kwargs.get('use_blast'):
                raw[RAW_POW_BLAST] = 0
            return w.env.msg, dict(fell=False, round_ended=False, reconnected=False, gem_delta=2.)
        w.env.step = step; w.env.pos = lambda: raw[:3]
        w.warm = []; w.crop = np.zeros((6,32,32)); w.vec = np.zeros(100)
        w._build = lambda: (w.crop, w.vec, False)
        w.use_sent = [False,False]; w.goal = (1,2,3)
        w.segs = SimpleNamespace(begin_replay=lambda *a: None, next_goal=lambda: None)
        w.steps = 50; w.ep_reward = 0.; w.drill_cur = {'fire': None, 'points': 0.}
        return w, calls

    def test_sj_is_reset_only_and_points_excluded(self):
        w, calls = self.worker()
        w._activate_handling_reset()
        self.assertEqual(calls, [dict(use_pow=1, use_blast=0, repeat=1), dict(repeat=1)])
        self.assertEqual(w.steps, 50)
        self.assertEqual(w.ep_reward, 0.)
        self.assertIsNone(w.drill_cur['fire'])
        self.assertEqual(w.drill_cur['points'], 0.)
        self.assertEqual(w.drill_cur['reset_points'], 4.)
        self.assertEqual(w.drill_cur['reset_activation_ms'], 128)
        self.assertEqual(w.replay_start_ms, 1128)
        self.assertEqual(len(w.warm), 2)

    def test_blast_uses_existing_charge(self):
        w, calls = self.worker(held=0)
        w._activate_handling_reset()
        self.assertEqual(calls[0]['use_blast'], 1)
        self.assertEqual(calls[0]['use_pow'], 0)

    def test_speed_and_helicopter_consume_recorded_inventory(self):
        for held in (2, 5):
            with self.subTest(held=held):
                w, calls = self.worker(held=held)
                w._activate_handling_reset()
                self.assertEqual(calls[0]['use_pow'], 1)
                self.assertEqual(calls[0]['use_blast'], 0)
                self.assertEqual(w.steps, 50)
                self.assertEqual(w.ep_reward, 0.)
                self.assertEqual(w.env.msg.obs[RAW_POW_HELD], 0)

    def test_missing_or_unconfirmed_resource_fails(self):
        for held, charge, consumes in [(3,.8,True), (0,0.,True), (1,.8,False)]:
            w, _ = self.worker(held, charge, consumes)
            with self.assertRaises((ValueError, RuntimeError)):
                w._activate_handling_reset()


if __name__ == '__main__':
    unittest.main()
