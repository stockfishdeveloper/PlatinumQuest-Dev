"""Native unused Helicopter must not defeat short ordinary-control proposals."""
import copy
import json
from pathlib import Path
import unittest
import numpy as np

from nav.collision.mesh import CollisionMap
from nav.marble_dynamics.record import ML
from nav.marble_dynamics.physics import GeneralPhysics
from nav.marble_dynamics.navigator import FrozenNavigator
from nav.marble_dynamics.search import search
from nav.marble_dynamics.phased_search import explore
from nav.terrain import TerrainGrid,TerrainMap
from nav.routing import RoutePlanner


class HelicopterHorizonRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture=json.loads((Path(__file__).with_name('fixtures')/'held_helicopter_horizon.json').read_text())
        cls.engine=GeneralPhysics(CollisionMap(ML/'collision_maps/CubeIsle_Hunt.npz'))
        cls.engine.bridge_physics_ms=16
        cls.terrain=TerrainGrid(TerrainMap.resolve('CubeIsle_Hunt'))

    def test_short_rolling_retains_requested_horizon_but_activation_retains_landing_time(self):
        c=self.fixture;nav=FrozenNavigator(self.terrain,ML/'models/nav/real_maps_20261010_A2/nav_latest.pth')
        _,pairs=search(self.engine,RoutePlanner(self.terrain),c['dynamics'],np.asarray(c['raw'],np.float32),c['gem'][:3],nav,horizon=8)
        info={i['label']:i for i,r in pairs}
        for name in ['route','direct','throttle_0.4','jump_0','steering_0.5','accelerate_then_brake']:
            self.assertEqual(info[name]['requested_replies'],8)
        for name in ['use_0_jump_-1','direct_use_3_jump_2','frozen_navigator']:
            self.assertEqual(info[name]['requested_replies'],96)

    def test_random_phases_only_extend_for_actual_activation(self):
        c=self.fixture;_,pairs=explore(self.engine,RoutePlanner(self.terrain),c['dynamics'],c['gem'][:3],horizon=8,samples=16,seed=997,brake_replies=0)
        unused=[i for i,r in pairs if i['specification']['use']<0];used=[i for i,r in pairs if i['specification']['use']>=0]
        self.assertTrue(unused);self.assertTrue(used)
        self.assertTrue(all(i['requested_replies']==8 for i in unused))
        self.assertTrue(all(i['requested_replies']==96 for i in used))


if __name__=='__main__':unittest.main()
