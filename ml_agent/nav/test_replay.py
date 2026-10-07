"""CPU regression checks for replay, scoring and recurrent PPO contracts.

python -m unittest nav.test_replay -v   (does not connect to a game or write checkpoints)
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from nav.replay import validate_start, restore_command, load_starts, split_for, paired_summary, REWARD_VERSION
from nav.model import NavActorCritic, HIDDEN
from nav.obs import VEC_DIM, VEC_POW, VEC_USE, VEC_NEXT
from nav.terrain import CROP_SHAPE
from nav.ppo_recurrent import Rollout, GAMMA, ppo_update
from nav.train_nav import prepare_points_critic, save_ckpt
from nav.waypoints import SegmentManager

torch.set_num_threads(1)


def fixture():
    return {'schema': 2, 'id': 'session:round:10', 'split_group': 'session:round', 'mission': 'test',
            'catalog': [[1, 'GemItem', [0, 0, 0]]], 'goal': [0, 0, 0], 'next': [1, 0, 0],
            'raw': [0.] * 61, 'hist': [],
            'world': {'schema': 2, 'supported': True, 'clock_ms': 10, 'pose': [0.] * 9,
                      'header': [12, 150000, 30000, 5, 0, 0, .5, 0, 0], 'held': 'SuperSpeedItem_MBU',
                      'items': [[0, -1, 0, 0]], 'groups': []}}


class ReplayTests(unittest.TestCase):
    def test_final_score_comes_from_engine_end_message(self):
        from nav.env import HuntEnv
        env = HuntEnv.__new__(HuntEnv)
        env.ticks = 0; env.recent_lines = []; env.round_ended = False; env.last_round_score = None
        lines = iter(['[]|-176|0|1', json.dumps([0] * 61) + '|0|0|0|2'])
        env._readline = lambda: next(lines)
        env._send = lambda word: None
        env._recv_obs()
        self.assertTrue(env.round_ended)
        self.assertEqual(env.last_round_score, 176)

    def test_legacy_cannot_be_used_as_real_game(self):
        with self.assertRaisesRegex(ValueError, 'legacy'):
            validate_start({'pre_target': [0, 0], 'chain': [[22, 0]]})

    def test_target_preserved_and_duplicate_ids_rejected(self):
        s = fixture()
        before = copy.deepcopy(s)
        restore_command(s)
        self.assertEqual(s, before)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'starts.json'
            path.write_text(json.dumps([s, s]))
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                load_starts(path)

    def test_split_keeps_overlapping_round_together(self):
        a, b = fixture(), fixture()
        b['id'] += ':another-fire'
        self.assertEqual(split_for(a), split_for(b))

    def test_paired_uncertainty_counts_rounds_not_correlated_starts(self):
        rows = []
        for group, delta in [('round_a', -1), ('round_b', 1)]:
            for start in range(2):
                for branch, points in [('no_use', 5), ('learned', 5 + delta)]:
                    rows.append({'id': f'{group}:{start}', 'split_group': group,
                                 'branch': branch, 'status': 'ok', 'points': points, 'falls': 0})
        summary = paired_summary(rows)
        grouped = summary['round_grouped']
        self.assertEqual(summary['paired_n'], 4)
        self.assertEqual(grouped['n'], 2)
        self.assertEqual(grouped['comparisons']['learned-no_use']['points']['mean'], 0)
        self.assertAlmostEqual(grouped['comparisons']['learned-no_use']['points']['se'], 1)
        self.assertLess(summary['comparisons']['learned-no_use']['points']['se'], 1)
        rows[-1]['split_group'] = 'wrong_round'
        with self.assertRaisesRegex(ValueError, 'inconsistent source round'):
            paired_summary(rows)

    def test_failed_restoration_is_visible_and_not_substituted(self):
        rows = [{'id': 'a', 'branch': b, 'status': 'ok', 'points': p, 'falls': 0}
                for b, p in [('no_use', 5), ('force', 7)]]
        rows += [{'id': 'b', 'branch': 'no_use', 'status': 'ok', 'points': 1, 'falls': 0},
                 {'id': 'b', 'branch': 'force', 'status': 'failed', 'reason': 'bad pose'}]
        report = paired_summary(rows)
        self.assertEqual(report['paired_ids'], ['a'])
        self.assertEqual(len(report['failed']), 1)
        self.assertEqual(report['comparisons']['force-no_use']['points']['mean'], 2)
        with self.assertRaises(ValueError):
            paired_summary(rows + rows[:1])

    def test_invalid_restore_data_rejected(self):
        s = fixture(); s['world']['pose'][0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            restore_command(s)
        s = fixture(); s['world']['held'] = 'x;quit();'
        with self.assertRaises(ValueError):
            restore_command(s)


class FlatTerrain:
    z_floor_min = -1

    def goal_field(self, *args): return None
    def dist_at(self, field, x, y, goal): return float(np.hypot(x - goal[0], y - goal[1]))
    def walkable_at(self, *args): return True
    def floor_z(self, *args): return 0.


class LearningTests(unittest.TestCase):
    def test_checkpoint_output_does_not_overwrite_resume_source(self):
        m = NavActorCritic(); prepare_points_critic(m, None)
        opt = torch.optim.Adam(m.parameters())
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'protected.pth'
            source.write_bytes(b'protected input')
            with patch.dict('os.environ', {'NAV_CKPT': str(source)}), patch('nav.train_nav.CKPT_DIR', tmp):
                output = save_ckpt(m, opt, 0, 0, {}, 'test', numbered=False)
            self.assertEqual(source.read_bytes(), b'protected input')
            ck = torch.load(output, map_location='cpu', weights_only=False)
            self.assertEqual(ck['reward_version'], REWARD_VERSION)
            self.assertEqual(ck['critic_warmup_remaining'], m.critic_warmup_remaining)

    def test_game_points_only_and_no_six_pickup_terminal(self):
        s = SegmentManager(FlatTerrain(), np.random.default_rng(0))
        s.real_mode = True; s.drill_mode = True; s.drill_window = True
        s.begin_replay((0, 0, .2), (0, 0, .2), (1, 0, .2))
        s._on_map = lambda p: True
        with patch('nav.waypoints.edge_time_cost', return_value=0):
            # Geometrically on the target, but no game pickup: no reward.
            reward, done, _ = s.step((0, 0, .2), False, False, False, picked=0)
            self.assertEqual(reward, 0); self.assertFalse(done)
            for points in [1, 5, 2, 1, 1, 5, 1, 2]:
                reward, done, _ = s.step((0, 0, .2), False, False, False, picked=points)
                self.assertEqual(reward, points); self.assertFalse(done)
            reward, done, _ = s.step((0, 0, .2), True, False, False, picked=0)
            self.assertEqual(reward, 0); self.assertFalse(done)

    def test_elapsed_time_discount_and_true_terminal(self):
        r = Rollout(2, 'cpu')
        r.n = 2; r.value[:2] = [1, 2]; r.reward[:2] = [0, 3]
        r.duration[:2] = [3, 1]; r.done[:2] = [0, 1]
        adv, ret = r.gae(9999)
        self.assertAlmostEqual(ret[1], 3)
        from nav.ppo_recurrent import LAMBDA
        self.assertAlmostEqual(ret[0], GAMMA ** 3 * (2 + LAMBDA ** 3), places=5)

    def test_approach_can_prepare_before_fire_is_approved(self):
        import nav.model as M
        v = torch.zeros(1, VEC_DIM); v[0, VEC_USE + 5] = 1
        v[0, 0] = 1; v[0, 2] = .2; v[0, 8] = 1; v[0, VEC_POW + 1] = 1
        v[0, VEC_NEXT + 1] = 1; v[0, VEC_NEXT + 2] = .3; v[0, VEC_NEXT + 4] = 1
        self.assertEqual(NavActorCritic.ss_gate(v).item(), 0)    # 40.51: preparation is off by default
        old = M.SS_RES_APPROACH
        M.SS_RES_APPROACH = True
        try:
            self.assertEqual(NavActorCritic.ss_gate(v).item(), 1)
            self.assertEqual(NavActorCritic.use_prior(v).item(), 0)
            v[0, VEC_POW + 1] = 0
            self.assertEqual(NavActorCritic.ss_gate(v).item(), 0)
        finally:
            M.SS_RES_APPROACH = old

    def test_warm_reset_scores_the_actual_action(self):
        torch.manual_seed(5)
        m = NavActorCritic()
        c = torch.zeros(1, *CROP_SHAPE); v = torch.zeros(1, VEC_DIM)
        v[:, VEC_USE + 5] = 1
        h = torch.randn(1, HIDDEN)
        o = m.act(c, v, h)
        lp, _, _, _ = m.evaluate_seq(c[None], v[None], h, o['action_buf'][None], torch.ones(1, 1), h_reset=h[None])
        torch.testing.assert_close(lp[0], o['logp'])

    def test_reward_migration_and_critic_update_preserve_actor(self):
        torch.manual_seed(7)
        m = NavActorCritic(); m.freeze_base()
        before = {k: t.clone() for k, t in m.state_dict().items() if not k.startswith('value')}
        self.assertTrue(prepare_points_critic(m, {'reward_version': 'old_shaped'}))
        r = Rollout(4, 'cpu')
        h = m.initial_state(1, 'cpu'); c = torch.zeros(1, *CROP_SHAPE); v = torch.zeros(1, VEC_DIM)
        v[:, VEC_USE + 5] = 1
        for i in range(4):
            o = m.act(c, v, h)
            r.add(c[0].numpy(), v[0].numpy(), o['action_buf'][0].numpy(), o['logp'].item(),
                  o['value'].item(), float(i == 3), float(i == 3), float(i == 0), h[0].numpy())
            h = o['h_next']
        opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=.001)
        with patch('nav.ppo_recurrent.SEQ_LEN', 4), patch('nav.ppo_recurrent.EPOCHS', 1):
            ppo_update(m, opt, [r], [0], critic_only=True)
        for k, value in before.items():
            torch.testing.assert_close(m.state_dict()[k], value, rtol=0, atol=0)
        self.assertGreater(m.value_head[-1].weight.abs().sum().item(), 0)
        self.assertFalse(prepare_points_critic(m, {'reward_version': REWARD_VERSION, 'critic_warmup_remaining': 4}))
        self.assertEqual(m.critic_warmup_remaining, 4)


if __name__ == '__main__':
    unittest.main()
