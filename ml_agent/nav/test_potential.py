"""Discounted trajectory accounting, including goal switches and true terminals."""
import unittest
from unittest.mock import patch
from nav.recovery import potential_term as F, legacy_progress_term, shaping_configuration
from nav.ppo_recurrent import GAMMA

K = .05


def shaped_return(path, durations, terminal=True, legacy=False):
    weight, total, previous = 1., 0., path[0]
    for i, (after, dt) in enumerate(zip(path[1:], durations)):
        ended = terminal and i == len(durations) - 1
        reward = (legacy_progress_term(previous, after, K, terminal=ended) if legacy
                  else F(previous, after, dt, GAMMA, K, terminal=ended))
        total += weight * reward
        weight *= GAMMA ** dt
        previous = previous if after is None else after
    return total


class PotentialTermTests(unittest.TestCase):
    def test_progress_and_regress_use_learner_discount(self):
        self.assertGreater(F(-30., -20., 1., GAMMA, K), 0.)
        self.assertLess(F(-20., -30., 1., GAMMA, K), 0.)
        self.assertAlmostEqual(F(-20., -20., 3., GAMMA, K), K * (GAMMA ** 3 * -20 + 20))

    def test_idle_and_loop_have_same_complete_discounted_return(self):
        loop = [-40., -10.] + [-10.] * 155 + [-40., 0.]
        idle = [-40.] * (len(loop) - 1) + [0.]
        durations = [1.] * (len(loop) - 1)
        self.assertAlmostEqual(shaped_return(loop, durations), 2.)
        self.assertAlmostEqual(shaped_return(idle, durations), 2.)
        self.assertGreater(shaped_return(loop, durations, legacy=True),
                           shaped_return(idle, durations, legacy=True) + 1.)

    def test_variable_duration_terminal_cancels_flight_and_respawn_path(self):
        self.assertAlmostEqual(shaped_return([-40., -12., -55., 0.], [2., 156., 3.]), 2.)
        self.assertAlmostEqual(shaped_return([-40., -40., -40., 0.], [2., 156., 3.]), 2.)

    def test_target_changes_remain_in_the_ledger(self):
        self.assertAlmostEqual(F(-20., -5., 1., GAMMA, K, goal_changed=True), .7525)
        self.assertAlmostEqual(shaped_return([-20., -5., -60., -2., 0.], [1., 3., 1., 2.]), 1.)

    def test_true_terminal_overrides_target_change(self):
        self.assertAlmostEqual(F(-20., -5., 3., GAMMA, K, terminal=True, goal_changed=True), 1.)

    def test_truncated_window_retains_discounted_final_potential(self):
        result = shaped_return([-30., -10., -5.], [2., 3.], terminal=False)
        self.assertAlmostEqual(result, K * (GAMMA ** 5 * -5 + 30))

    def test_undefined_floor_keeps_history_and_still_telescopes(self):
        self.assertAlmostEqual(shaped_return([-40., None, None, -8., 0.], [1., 80., 2., 1.]), 2.)

    def test_uninitialized_or_disabled_term_is_silent(self):
        self.assertEqual(F(None, -5., 1., GAMMA, K), 0.)
        self.assertEqual(F(-5., -3., 1., GAMMA, 0.), 0.)

    def test_legacy_mode_is_explicitly_the_overnight_heuristic(self):
        self.assertEqual(legacy_progress_term(-20., -5., K, goal_changed=True), 0.)
        self.assertEqual(legacy_progress_term(-20., -20., K), 0.)

    def test_disabled_metadata_is_identical_across_mode_choices(self):
        with patch.dict('os.environ', NAV_POTENTIAL_K='0', NAV_SHAPING_MODE='legacy'):
            legacy = shaping_configuration(GAMMA)
        with patch.dict('os.environ', NAV_POTENTIAL_K='0', NAV_SHAPING_MODE='discounted'):
            self.assertEqual(legacy, shaping_configuration(GAMMA))


if __name__ == '__main__':
    unittest.main()
