"""Behavioral regression checks; no game processes or optimizer runs."""
import unittest
import tempfile
import copy
import json
import os
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch, Mock
import numpy as np
import torch

from nav.routing import RoutePlanner, line_clear, path_waypoint, goal_key
from nav.recovery import ProgressMonitor
from nav.terrain import TerrainGrid
from nav.model import NavActorCritic
from nav.obs import VEC_DIM, VEC_POW, ObsBuilder
from nav.protocol import RAW_DIM, RAW_POW_HELD, parse_message
from nav.env import HuntEnv

torch.set_num_threads(1)


class RoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.terrain = TerrainGrid(str(Path(__file__).resolve().parents[1] / 'terrain_maps/terrain_Cragmire_Hunt.npz'))

    def test_under_deck_is_a_distinct_destination(self):
        t = self.terrain
        route = RoutePlanner(t)
        route.navigate((1., -2., 13.), (1., -2., 13.))
        self.assertEqual(route.distance((1., -2., 13.)), 0.)
        route.navigate((1., -2., 13.), (1., -2., 21.))
        self.assertGreater(route.distance((1., -2., 13.)), 30.)
        self.assertEqual(route.key, goal_key((1., -2., 21.)))

    def test_ground_line_is_not_a_shortcut_to_the_deck(self):
        t = self.terrain
        self.assertFalse(line_clear(t, 1, -2, 1, -2, az=13, bz=21))
        f = t.goal_field(1, -2, z=21)
        wp, routed = path_waypoint(t, 1, -2, f, (1, -2, 21), mz=13)
        self.assertTrue(routed)
        self.assertNotEqual(wp, (1, -2, 21))
        self.assertTrue(line_clear(t, 1, -2, wp[0], wp[1], az=13, bz=wp[2]))

    def test_next_gem_is_not_replaced_by_a_route_corner(self):
        raw = np.zeros(RAW_DIM, dtype=np.float32)
        raw[:3] = (1, -2, 13.19)
        ob = ObsBuilder(self.terrain)
        _, vec, _ = ob.build(raw, (6, -2, 13), (10, -2, 21), gem_goal=(1, -2, 21))
        from nav.obs import VEC_NEXT
        self.assertAlmostEqual(float(vec[0]), 1.)
        self.assertGreater(float(vec[VEC_NEXT + 3]), .7)

    def test_route_corner_is_not_a_super_speed_pickup(self):
        raw = np.zeros(RAW_DIM, dtype=np.float32)
        raw[:3] = (1, -2, 13.19); raw[3] = 2; raw[RAW_POW_HELD] = 2
        _, vec, _ = ObsBuilder(self.terrain).build(raw, (6, -2, 13), (10, -2, 21), gem_goal=(1, -2, 21))
        self.assertEqual(float(vec[VEC_POW + 26]), 0.)
        self.assertEqual(float(vec[VEC_POW + 27]), 0.)


class EscapeTests(unittest.TestCase):
    def vec(self):
        v = torch.zeros(1, VEC_DIM)
        v[0, 0] = 1.; v[0, 8] = 1.
        v[0, 10:42:2] = 1.
        v[0, 10] = .025; v[0, 11] = .1
        return v

    # 2026-10-09 20:45 (operator): every held powerup is eligible at any time; the blast whenever charged; nothing
    # situational. The old escape / lip / kick approvals are kept in use_prior_physics_reference and no longer gate.
    def test_any_held_powerup_is_eligible_anywhere(self):
        for held in range(7):
            v = self.vec(); v[0, VEC_POW + held] = 1.; v[0, 11] = 0.; v[0, 8] = 0.   # airborne, open floor, no rise
            self.assertEqual(float(NavActorCritic.use_prior(v)[0]), 1., held)

    def test_blast_needs_charge(self):
        v = self.vec()
        self.assertEqual(float(NavActorCritic.use_prior(v)[0]), 0.)
        v[0, VEC_POW + 7] = .5
        self.assertEqual(float(NavActorCritic.use_prior(v)[0]), 1.)
        v[0, VEC_POW + 7] = 0.; v[0, VEC_POW + 8] = 1.      # a special (armed) blast counts as charged
        self.assertEqual(float(NavActorCritic.use_prior(v)[0]), 1.)

    def test_nothing_held_and_no_charge_cannot_use(self):
        v = self.vec(); v[0, 11] = 0.
        self.assertEqual(float(NavActorCritic.use_prior(v)[0]), 0.)


class RecoveryTests(unittest.TestCase):
    def test_round_start_requires_known_metadata_and_finished_countdown(self):
        env = HuntEnv.__new__(HuntEnv)
        env.time_left_s = Mock(return_value=180.)
        env.info = {}
        self.assertFalse(env.round_started())
        env.info = {'round_ms': 180000.}
        self.assertFalse(env.round_started())
        env.time_left_s.return_value = 179.936
        self.assertTrue(env.round_started())

    def test_recorder_rejects_missing_or_new_connection_state_but_keeps_valid_starts(self):
        from nav.vec_worker import InstanceWorker
        from nav.test_replay import fixture
        start = fixture()
        worker = InstanceWorker.__new__(InstanceWorker)
        row = dict(world=start['world'], raw=start['raw'], goal=start['goal'], next=None)
        worker.hist = [row, row]; worker.session_id = 'test'; worker.round_id = 1
        worker.mission = 'test'; worker.live_rejected = 0; worker._write_live = Mock()
        worker.env = SimpleNamespace(replay_world=None, replay_catalog=start['catalog'], reconnected=False)
        worker._record_approach_at(0)
        worker.env.replay_world = dict(start['world'], clock_ms=500)
        worker.env.replay_catalog = None
        worker._record_approach_at(0)
        worker.env.replay_catalog = start['catalog']; worker.env.reconnected = True
        worker._record_approach_at(0)
        worker.env.reconnected = False; worker.env.replay_world['clock_ms'] = 1
        worker._record_approach_at(0)
        worker._write_live.assert_not_called()
        worker.env.replay_world['clock_ms'] = 500
        worker._record_approach_at(0)
        self.assertEqual(worker._write_live.call_args[0][0]['approach_ms'], 490)

    def test_reconnect_exits_step_before_inventory_recording_or_old_map_logic(self):
        from nav.vec_worker import InstanceWorker
        raw = np.zeros(RAW_DIM); raw[RAW_POW_HELD] = 2
        worker = InstanceWorker.__new__(InstanceWorker)
        worker.real = True; worker.drill = False; worker.drill_stage = 0
        worker.goal = (1., 2., 3.); worker.vec = np.zeros(VEC_DIM)
        worker.prof = {'game': 0.}; worker.recovery_pending = False
        worker.round_points = 7.; worker._record_approach = Mock()
        worker.finish_round_transition = Mock(return_value='boundary')
        env = SimpleNamespace(replay_world={'clock_ms': 1000}, info={'round_ms': 180000},
                              reconnected=False, msg=SimpleNamespace(obs=raw),
                              time_left_s=lambda: 170.)
        def reconnect(*args, **kwargs):
            env.replay_world = None; env.info = {}; env.reconnected = True
            env.msg = SimpleNamespace(obs=np.zeros(RAW_DIM))
            return env.msg, dict(fell=False, gem_delta=2., round_ended=False, reconnected=True)
        env.step = reconnect; worker.env = env
        with patch('nav.vec_worker.POTENTIAL_K', 0.), patch('nav.vec_worker.ACTION_SMOOTH', 1.):
            self.assertEqual(worker.step([0.] * 8), 'boundary')
        worker._record_approach.assert_not_called()
        self.assertEqual(worker.round_points, 9.)
        self.assertEqual(worker.finish_round_transition.call_args.kwargs['duration_steps'], 1.)

    def test_duration_falls_back_when_replay_telemetry_disappears(self):
        from nav.vec_worker import InstanceWorker
        worker = InstanceWorker.__new__(InstanceWorker)
        worker.env = SimpleNamespace(replay_world=None, reconnected=False)
        with patch('nav.vec_worker.simulation_time', return_value=100.192):
            self.assertAlmostEqual(worker._transition_duration(5000, 100.), 3.)

    def test_early_terminal_preserves_points_and_settles_shaping_once(self):
        from nav.vec_worker import InstanceWorker
        raw = np.zeros(RAW_DIM)
        worker = InstanceWorker.__new__(InstanceWorker)
        worker.real = True; worker.drill = False; worker.drill_stage = 0
        worker.goal = (1., 2., 3.); worker.vec = np.zeros(VEC_DIM)
        worker.prof = {'game': 0.}; worker.recovery_pending = False
        worker.round_points = 7.; worker.ep_reward = 8.
        worker._shaping = Mock(return_value=1.)
        worker.segs = Mock(); worker._round_over = Mock()
        worker.sync_map = Mock(); worker.start_segment = Mock(); worker.log = Mock()
        worker._reply = lambda **kw: kw
        env = SimpleNamespace(replay_world={'clock_ms': 1000}, info={'round_ms': 180000},
                              reconnected=False, msg=SimpleNamespace(obs=raw),
                              connections=1, wait_new_round=Mock(), rtf=lambda: 1.)
        def finish(*args, **kwargs):
            env.replay_world['clock_ms'] = 1192
            return env.msg, dict(fell=False, gem_delta=3., round_ended=True, reconnected=False)
        env.step = finish; worker.env = env
        with patch('nav.vec_worker.POTENTIAL_K', 0.), patch('nav.vec_worker.ACTION_SMOOTH', 1.):
            reply = worker.step([0.] * 8)
        self.assertEqual(reply['r'], 4.)
        self.assertEqual(reply['ep_reward'], 12.)
        self.assertEqual(reply['duration_steps'], 3.)
        self.assertTrue(reply['done']); self.assertFalse(reply['skip'])
        worker._shaping.assert_called_once()
        self.assertEqual(worker.round_points, 10.)

    def test_discounted_worker_accounts_for_retarget_without_resetting_potential(self):
        from nav.vec_worker import InstanceWorker
        from nav.ppo_recurrent import GAMMA
        worker = InstanceWorker.__new__(InstanceWorker)
        worker.real = True; worker.drill = False; worker.goal = (5., 0., 0.)
        worker.phi_prev = -30.; worker.phi_key = (0., 0., 0.)
        worker._phi = Mock(return_value=-10.); worker.shape_frozen = 0; worker.round_shaping = 0.
        with patch('nav.vec_worker.SHAPING_MODE', 'discounted'), \
             patch('nav.vec_worker.SHAPING_GAMMA', GAMMA), patch('nav.vec_worker.POTENTIAL_K', .05):
            reward = worker._shaping((1., 0., 0.), 3., False, (0., 0., 0.))
        self.assertAlmostEqual(reward, .05 * (GAMMA ** 3 * -10 + 30))
        self.assertEqual(worker.phi_prev, -10.)
        self.assertEqual(worker.phi_key, worker.goal)

    def test_critic_warms_for_changed_shaping_but_preserves_actor(self):
        from nav.train_nav import prepare_points_critic
        from nav.replay import REWARD_VERSION
        from nav.recovery import shaping_configuration
        model = NavActorCritic()
        before = model.use_head[-1].weight.detach().clone()
        with patch.dict('os.environ', NAV_SHAPING_MODE='discounted', NAV_POTENTIAL_K='.05'):
            config = shaping_configuration()
            same = dict(reward_version=REWARD_VERSION, reward_configuration=config, critic_warmup_remaining=3)
            self.assertFalse(prepare_points_critic(model, same))
            self.assertEqual(model.critic_warmup_remaining, 3)
            self.assertTrue(prepare_points_critic(model, dict(reward_version=REWARD_VERSION)))
        self.assertGreater(model.critic_warmup_remaining, 0)
        torch.testing.assert_close(model.use_head[-1].weight, before)

    def test_wait_is_real_seconds_and_route_progress_restarts_it(self):
        m = ProgressMonitor()
        self.assertEqual(m.observe(100, 'gem', 20), 0)
        self.assertAlmostEqual(m.observe(107.9, 'gem', 20), 7.9)
        self.assertEqual(m.observe(108, 'gem', 18), 0)
        self.assertEqual(m.observe(116, 'gem', 18), 8)

    def test_pickups_are_progress_and_target_oscillation_is_not(self):
        m = ProgressMonitor(); m.observe(0, 'a', 20)
        self.assertEqual(m.observe(4, 'b', 10), 4)
        self.assertEqual(m.observe(8, 'a', 20), 8)
        self.assertEqual(m.observe(9, 'a', 20, picked=1), 0)

    def test_rescue_does_not_look_like_a_powerup_escape(self):
        m = ProgressMonitor(); m.observe(0, 'a', 20); m.escape_use(2)
        m.rescued(8.064, 30)
        self.assertEqual(m.rescues, 1)
        self.assertEqual(m.failed_escape_uses, 1)
        self.assertEqual(m.powerup_escapes, 0)
        self.assertAlmostEqual(m.observe(9, 'a', 30), .936)

    def test_groups_use_engine_telemetry_instead_of_nearest_visible_gems(self):
        m = ProgressMonitor()
        m.observe(0, 'a', 10, gems=['a'], groups_completed=0)
        m.observe(1, 'b', 10, picked=1, gems=['b'], groups_completed=0)
        self.assertEqual(m.group_completions, 0)
        m.observe(2, 'c', 10, groups_completed=1)
        self.assertEqual(m.group_completions, 1)

    def test_bridge_acknowledgments_parse(self):
        self.assertEqual(parse_message('RESCUE|0').kind, 'rescue')
        self.assertEqual(parse_message('RULES|1').fields, ['1'])

    def test_delayed_rescue_ack_keeps_every_pickup_during_the_control(self):
        env = HuntEnv.__new__(HuntEnv)
        env.game_rules = {'competitive': False}
        def control(word):
            self.assertEqual(word, 'QUICKRESPAWN')
            env.control_info = dict(gem_delta=2., fell=False, round_ended=False, reconnected=False)
        def step(action, repeat):
            env.rescue_accepted = True
            return None, dict(gem_delta=1., fell=False, round_ended=False, reconnected=False)
        env.control = control; env.step = step
        self.assertTrue(env.quick_rescue())
        self.assertEqual(env.control_info['gem_delta'], 3.)

    def test_competitive_rescue_does_not_send_a_command(self):
        env = HuntEnv.__new__(HuntEnv); env.game_rules = {'competitive': True}
        env.control = Mock()
        self.assertFalse(env.quick_rescue())
        env.control.assert_not_called()

    def test_rules_ack_can_arrive_after_the_first_observation(self):
        env = HuntEnv.__new__(HuntEnv)
        env.control = lambda word: None
        def step(action, repeat):
            env.game_rules = {'competitive': True}
        env.step = step
        env.set_game_rules(True)
        self.assertTrue(env.game_rules['competitive'])

    def test_dashboard_keeps_long_rounds_and_recovery_fields(self):
        from nav import dashboard_nav
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'train_nav_test.log').write_text(
                '[12:34:56] [0] GAME map=Sprawl_Hunt points=350 gems=220 falls=4 rtf=3 pow=f1:2 '
                'rescues=2 rescue_attempts=3 stalled_s=12.5 powerup_escapes=1 group_completions=4\n')
            with patch.object(dashboard_nav, 'LOG_DIR', directory):
                games = dashboard_nav.parse_logs()[-1]
        self.assertEqual(len(games), 1)
        self.assertEqual(games[0][3], 350)
        self.assertEqual(games[0][8]['rescues'], 2)
        self.assertEqual(games[0][8]['group_completions'], 4)

    def test_rescue_crossing_round_end_preserves_terminal_reward_and_duration(self):
        from nav.vec_worker import InstanceWorker
        worker = InstanceWorker.__new__(InstanceWorker)
        worker.drill_stage = 0; worker.drill = False; worker.real = True
        worker.ep_reward = 8.; worker.segs = Mock(); worker.env = Mock()
        worker.env.rtf.return_value = 1.
        worker._round_over = Mock(); worker.sync_map = Mock(); worker.start_segment = Mock()
        worker.log = Mock(); worker._reply = lambda **kw: kw
        reply = worker.finish_round_transition(dict(round_ended=True, reconnected=False),
                                               reward=3., ep=8., duration_steps=3.)
        self.assertTrue(reply['done']); self.assertFalse(reply['skip'])
        self.assertEqual(reply['r'], 3.); self.assertEqual(reply['ep_reward'], 8.)
        self.assertEqual(reply['duration_steps'], 3.)
        worker.env.wait_new_round.assert_called_once()
        worker._round_over.assert_called_once()

    def test_escape_practice_keeps_paired_seeds_together_and_excludes_holdouts(self):
        from nav.escape_practice import assemble
        from nav.test_replay import fixture
        rows = []
        for label in ('base', 'clusters'):
            start = copy.deepcopy(fixture())
            start['id'] = label
            start['split_group'] = f'phase_r_{label}_assisted:test:1'
            rows.append(start)
        heldout = copy.deepcopy(rows[0]); heldout['id'] = 'holdout'; heldout['mission'] = 'Horizon_Hunt'
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'captured.jsonl').write_text('\n'.join(json.dumps(s) for s in rows + [heldout]))
            counts = assemble(directory, ['test'])
            dev = json.loads(Path(directory, 'starts_dev.json').read_text())
            evaluation = json.loads(Path(directory, 'starts_eval.json').read_text())
        self.assertEqual(counts['total'], 2)
        self.assertTrue(len(dev) == 2 or len(evaluation) == 2)
        self.assertEqual({s['split_group'] for s in dev + evaluation}, {'phase_r_paired:test:1'})

    def test_normal_special_blast_is_recorded_without_regular_charge(self):
        from nav.escape_practice import EscapeRecorder
        from nav.test_replay import fixture
        from nav.protocol import RAW_POW_SPECIAL, RAW_SCORE, RAW_TIME_LEFT
        start = fixture(); start['world']['held'] = 'none'; start['world']['header'][7] = 1
        raw = np.array(start['raw']); raw[RAW_POW_SPECIAL] = 1
        raw[RAW_SCORE] = 5; raw[RAW_TIME_LEFT] = 150000
        env = SimpleNamespace(msg=SimpleNamespace(obs=raw), replay_world=start['world'], replay_catalog=start['catalog'])
        vec = np.zeros(VEC_DIM); vec[8] = 1.; vec[3] = .1
        with tempfile.TemporaryDirectory() as directory:
            recorder = EscapeRecorder(directory, 'normal_game', 'test')
            recorder.observe(env, (0, 0, 1), (0, 0, 1), None, vec, [False, False, False, None, 9.], 1)
            self.assertEqual(recorder.count, 1)
            row = json.loads(Path(directory, 'normal_game_test.jsonl').read_text())
        self.assertEqual(row['resource'], 'blast')
        self.assertEqual(row['world']['header'][7], 1)

    def test_dashboard_finds_checkpoint_in_isolated_run_directory(self):
        from nav import dashboard_nav
        with tempfile.TemporaryDirectory() as directory:
            old = Path(directory, 'nav_old.pth'); old.write_bytes(b'test placeholder')
            new = Path(directory, 'run/nav_latest.pth'); new.parent.mkdir(); new.write_bytes(b'test placeholder')
            os.utime(old, (1, 1)); os.utime(new, (2, 2))
            with patch.object(dashboard_nav, 'CKPT_DIR', directory), \
                 patch.object(dashboard_nav, 'parse_logs', return_value=([], [], {}, [], [])), \
                 patch.object(dashboard_nav, 'eval_series', return_value={}):
                state = dashboard_nav.build_state()
        self.assertEqual(state['ckpt'].replace('\\', '/'), 'run/nav_latest.pth')


if __name__ == '__main__':
    unittest.main()
