"""Route experiment contracts. CPU only, no checkpoints or game connection."""
import unittest

from nav.gems import plan_tour
from nav.protocol import RAW_POW_HELD
from nav.ss_routes import first_turn_costs
from nav.ss_prepare import prepare_action
from nav.ss_follow import follow_action
from nav.ss_command_aim import predict_aim
from nav.obs import VEC_DIM, VEC_POW, VEC_NEXT


class Floor:
    def __init__(self, room=float('inf')):
        self.room = room

    def floor_z(self, x, y, z):
        return 0.0

    def gap_along(self, *args, **kwargs):
        return self.room, 0, 0


class RoutesTests(unittest.TestCase):
    def raw(self, held=2):
        raw = [0.] * (RAW_POW_HELD + 1)
        raw[3] = 10.
        raw[RAW_POW_HELD] = held
        return raw

    def test_only_survivable_first_turn_receives_cost(self):
        behind = (-12., 0., 0., 1., 12.)
        forward = (12., 0., 0., 1., 12.)
        short = (-4., 0., 0., 1., 4.)
        raised = (-12., 0., 2., 1., 12.)
        costs = first_turn_costs(Floor(), self.raw(), [behind, forward, short, raised])
        self.assertEqual(set(costs), {behind})
        self.assertAlmostEqual(costs[behind], 1.28)
        self.assertEqual(first_turn_costs(Floor(5), self.raw(), [behind]), {})

    def test_empty_slot_and_airborne_never_discount(self):
        gems = [(-12., 0., 0., 1., 12.)]
        self.assertEqual(first_turn_costs(Floor(), self.raw(0), gems), {})
        raw = self.raw(); raw[5] = 2.
        self.assertEqual(first_turn_costs(Floor(), raw, gems), {})

    def test_no_discount_preserves_original_route(self):
        gems = [(-12., 0., 0., 1., 12.), (12., 0., 0., 1., 12.), (8., 8., 0., 1., 11.)]
        for current in [None] + gems:
            a = plan_tour(gems, current, (0, 0), (10, 0))
            b = plan_tour(gems, current, (0, 0), (10, 0), first_turn_costs={})
            self.assertEqual(a, b)

    def test_held_charge_can_change_initial_turn(self):
        behind = (-12., 0., 0., 1., 12.)
        forward = (15., 0., 0., 1., 15.)
        gems = [behind, forward]
        self.assertEqual(plan_tour(gems, None, (0, 0), (10, 0))[0], forward)
        self.assertEqual(plan_tour(gems, None, (0, 0), (10, 0),
                                  first_turn_costs=first_turn_costs(Floor(), self.raw(), gems))[0], behind)

    def test_preparation_requires_approved_next_turn_and_clear_floor(self):
        vec = [0.] * VEC_DIM
        vec[0] = 1.; vec[2] = .2; vec[8] = 1.
        vec[VEC_POW + 1] = 1.; vec[VEC_POW + 27] = 1.
        vec[VEC_NEXT] = -1.; vec[VEC_NEXT + 2] = .04; vec[VEC_NEXT + 4] = 1.
        action = [-1., 0., .9, 0., 1., 0.]
        changed, active = prepare_action(action, Floor(), self.raw(), vec)
        self.assertTrue(active)
        self.assertEqual(changed, [1., 0., 1., 0., 0., 0.])
        self.assertEqual(action, [-1., 0., .9, 0., 1., 0.])
        self.assertFalse(prepare_action(action, Floor(11), self.raw(), vec)[1])
        firing = list(action); firing[5] = 1.
        self.assertFalse(prepare_action(firing, Floor(), self.raw(), vec)[1])
        vec[VEC_POW + 27] = 0.
        self.assertFalse(prepare_action(action, Floor(), self.raw(), vec)[1])

    def test_follow_brakes_and_corrects_sideways_velocity(self):
        vec = [0.] * VEC_DIM
        vec[0] = 1.; vec[2] = .2; vec[8] = 1.
        raw = self.raw(); raw[3:5] = [20., 2.]
        original = [1., 0., .9, 0., 0., 0.]
        result, active = follow_action(original, raw, vec)
        self.assertTrue(active)
        self.assertLess(result[0], 0.)
        self.assertLess(result[1], 0.)
        self.assertEqual(result[5], original[5])
        raw[3:5] = [5., 0.]
        self.assertFalse(follow_action(original, raw, vec)[1])

    def test_command_aim_uses_pending_position_and_velocity(self):
        raw = self.raw(); previous = self.raw(); previous[3] = 9.5
        prediction = predict_aim(Floor(), raw, previous, (0., 12., 0.))
        self.assertIsNotNone(prediction)
        self.assertAlmostEqual(prediction['position'][0], .656)
        self.assertAlmostEqual(prediction['velocity'][0], 10.5)
        self.assertAlmostEqual(prediction['effective_dv'], 24.104)
        vx, vy = prediction['result_velocity']
        dx, dy = -prediction['position'][0], 12. - prediction['position'][1]
        self.assertAlmostEqual(vx * dy - vy * dx, 0., places=6)
        self.assertIsNone(predict_aim(Floor(4), raw, previous, (0., 12., 0.)))
        previous[3] = -10.
        self.assertIsNone(predict_aim(Floor(), raw, previous, (0., 12., 0.)))


if __name__ == '__main__':
    unittest.main()
