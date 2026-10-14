"""Native pickup/landing and post-pickup commitment regressions."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from nav.collision.mesh import CollisionMap
from nav.marble_dynamics.bridge import predict_replies
from nav.marble_dynamics.evaluate import errors
from nav.marble_dynamics.landing import landing_complete,landing_points,retained_landing
from nav.marble_dynamics.physics import GeneralPhysics
from nav.marble_dynamics.record import ML
from nav.marble_dynamics.search import PlanMemory


class LandingRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case=json.loads(Path(__file__).with_name('fixtures').joinpath('landing_bridge.json').read_text())
        cls.engine=GeneralPhysics(CollisionMap(ML/'collision_maps'/(cls.case['map']+'.npz')))

    def memory(self):
        info=copy.deepcopy(self.case['selected'])
        goal=next(i for i in self.case['dynamics']['items'] if i['gem'] and i['value']==5)
        info['goal_item_id']=goal['id']
        memory=PlanMemory();memory.accept(dict(proposed=self.case['actions']),info['label'],goal['position'],info)
        return memory,goal

    def test_full_native_pickup_and_landing_forecast(self):
        result=predict_replies(self.engine,self.case['dynamics'],self.case['actions'])
        self.assertIsNone(result['failure']);self.assertFalse(result['ambiguous_intervals'])
        maxima=np.max([errors(a,b) for a,b in zip(result['states'][1:],self.case['actual'])],axis=0)
        self.assertLess(maxima[0],1e-4);self.assertLess(maxima[1],1e-3);self.assertLess(maxima[2],1e-3)
        self.assertEqual(self.case['actual_points'],5);self.assertEqual(self.case['falls'],0)

    def test_actual_selected_pickup_preserves_absolute_landing_schedule(self):
        memory,goal=self.memory()
        for state in self.case['actual'][:38]:memory.advance(state)
        self.assertTrue(memory.locked)
        self.assertFalse(memory.observe_pickup(5,[goal]))
        self.assertFalse(memory.observe_pickup(0,[]))
        self.assertTrue(memory.observe_pickup(5,self.case['pickup_snapshot']['items']))
        self.assertTrue(memory.landing_mode);self.assertTrue(memory.locked)
        self.assertEqual(memory.remaining,self.case['actions'][38:])
        retained=retained_landing(self.engine,SimpleNamespace(distance=lambda p:np.inf),self.case['pickup_snapshot'],memory)
        self.assertIsNotNone(retained);self.assertEqual(retained[0]['goal_kind'],'landing')
        self.assertIsNone(retained[0]['possible_pickup_interval'])
        self.assertEqual(retained[1]['proposed'],self.case['actions'][38:])
        self.assertFalse(landing_complete(self.engine,self.case['actual'][37],memory))
        for state in self.case['actual'][38:]:memory.advance(state)
        self.assertTrue(landing_complete(self.engine,self.case['actual'][-1],memory))
        memory.clear();self.assertFalse(memory.landing_mode);self.assertIsNone(memory.landing_goal)

    def test_full_map_surface_proposals_keep_distant_support(self):
        from nav.collision.test_queries import fixture
        mesh=fixture([[[100,-2,0],[104,-2,0],[102,2,0]],
                      [[0,-2,0],[4,-2,0],[2,2,0]]],[[0,0,1],[0,0,1]])
        state=dict(position=[2,0,3],velocity=[0,0,0],radius=.2)
        points=landing_points(mesh,state,limit=8)
        self.assertTrue(any(p['position'][0]>100 for p in points))
        self.assertTrue(all(p['triangle_id'] in (0,1) for p in points))


if __name__=='__main__':unittest.main()
