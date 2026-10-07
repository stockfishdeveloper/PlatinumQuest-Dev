"""Checks for the restricted floor predictor, independent of game performance."""
import unittest
import numpy as np
from nav.ss_floor_model import step, step_batch


class FloorModelTests(unittest.TestCase):
    def test_batch_matches_scalar_across_slip_and_camera_saturation(self):
        rng = np.random.RandomState(42)
        p = rng.randn(32, 2)
        v = rng.randn(32, 2) * 15
        w = rng.randn(32, 3) * 80
        u = rng.randn(32, 2) * 1.4
        u[:4] = 0
        w[4:8, :2] = np.stack((-v[4:8, 1], v[4:8, 0]), axis=-1) / .18975
        yaw = rng.uniform(-3., 3., 32)
        batched = step_batch(p, v, w, u, yaw)
        scalar = [step(p[i], v[i], w[i], u[i], yaw[i]) for i in range(32)]
        for k in range(3):
            np.testing.assert_allclose(batched[k], np.stack([row[k] for row in scalar]), atol=1e-10)

    def test_sliding_impulse_exchanges_linear_and_rotational_momentum(self):
        # No input, enough slip to avoid the rolling/braking regime for 64 ms.
        p, v, w = step([0., 0.], [25., 0.], [0., 0., 0.], [0., 0.], 0.)
        roll = .18975 * np.array([w[1], -w[0]])
        np.testing.assert_allclose(v, [25 - 14 * .064, 0], atol=1e-10)
        np.testing.assert_allclose(v + .4 * roll, [25., 0.], atol=1e-10)
        self.assertGreater(roll[0], 0)
        self.assertGreater(p[0], 0)


if __name__ == '__main__':
    unittest.main()
