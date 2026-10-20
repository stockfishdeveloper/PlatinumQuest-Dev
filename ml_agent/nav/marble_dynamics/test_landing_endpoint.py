"""A native landed plan must complete at its checked endpoint, not its aim point."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from nav.collision.mesh import CollisionMap
from nav.marble_dynamics.bridge import predict_replies
from nav.marble_dynamics.landing import alternatives, landing_complete
from nav.marble_dynamics.physics import GeneralPhysics
from nav.marble_dynamics.phased_search import AimProposal
from nav.marble_dynamics.record import ML
from nav.marble_dynamics.search import assess, PlanMemory


class NativeLandingEndpoint(unittest.TestCase):
    # This native fixture predates the faster v23 throttle and through-gem
    # speed law. Recreate its measured inputs, not today's proposal defaults.
    @patch.multiple('nav.marble_dynamics.gem_demo', TARGET_SPEED_CAP=11.,
        BRAKE_DECEL=7., THROTTLE_GAIN=12., THROTTLE_CAP=.85)
    @patch.multiple('nav.marble_dynamics.search', PICKUP_MIN_SPEED=0.,
        ROUTE_SPEED_LAW=False, OPEN_TOP=0.)
    def test_completed_native_landing_uses_checked_destination(self):
        path=Path(__file__).with_name('fixtures')/'settled_landing_endpoint.json'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
            '1d25021c204948f112e888fa60e1f8207913f6c322faa6fc79148f9e2c7b1327')
        c=json.loads(path.read_text())
        original_path=path.with_name('settled_landing_original_proposal.json')
        self.assertEqual(hashlib.sha256(original_path.read_bytes()).hexdigest(),
            '08a217cf63cf851fa5059bb6645f6c62caafa346254b3214de78325aaaa699b9')
        recorded=json.loads(original_path.read_text())['original']
        engine=GeneralPhysics(CollisionMap(ML/'collision_maps'/(c['map']+'.npz')))
        engine.bridge_physics_ms=16
        goal=c['gem'][:3]
        route=SimpleNamespace(terrain=None,distance=lambda p:float(np.linalg.norm(np.asarray(p)-goal)))
        count=c['chosen']['possible_pickup_interval']
        prefix=predict_replies(engine,c['dynamics'],AimProposal(route,goal,
            recorded['requested_replies'],**recorded['specification']))
        original=assess(engine,route,c['dynamics'],goal,prefix,'native_pickup_prefix')
        self.assertIsNotNone(original['rejected'])
        self.assertIsNotNone(original['possible_pickup_interval'])
        candidates=alternatives(engine,route,c['dynamics'],goal,[(original,prefix)],80,8)
        matching=[(info,rollout) for info,rollout in candidates if info['rejected'] is None
            and np.linalg.norm(np.asarray(info['proposed_landing_goal'])-
                               c['chosen']['proposed_landing_goal'])<1e-4]
        self.assertTrue(matching)
        info,rollout=matching[0]
        self.assertEqual(info['landing_goal'],rollout['states'][-1]['position'])
        self.assertEqual(rollout['proposed'][:count],c['actions'][:count])
        actual=c['actual_tail'][-1]
        self.assertLess(np.linalg.norm(np.asarray(info['landing_goal'])-actual['position']),1e-4)
        self.assertGreater(np.linalg.norm(np.asarray(info['proposed_landing_goal'])-
                                         actual['position']),.5)
        memory=PlanMemory()
        memory.accept(rollout,info['label'],goal,info)
        memory.executed=len(c['actions'])
        self.assertTrue(memory.observe_pickup(5,c['final_context']['dynamics']['items']))
        memory.observed_tail=copy.deepcopy(c['actual_tail'])
        self.assertTrue(landing_complete(engine,actual,memory))
        # This is the original native failure: it did land, but its aim point
        # was outside the completion tolerance. Keep that negative control.
        memory.landing_goal=list(info['proposed_landing_goal'])
        self.assertFalse(landing_complete(engine,actual,memory))
        memory.landing_goal=list(info['landing_goal'])
        moving=copy.deepcopy(actual);moving['velocity']=[2.,0.,0.]
        self.assertFalse(landing_complete(engine,moving,memory))


if __name__=='__main__':
    unittest.main()
