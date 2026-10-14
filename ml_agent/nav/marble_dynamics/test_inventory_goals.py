"""Script-only inventory boundaries and goals beyond the nearest-five block."""
import copy
import json
from pathlib import Path
import unittest

import numpy as np

from nav.collision.mesh import cross_rows,CollisionMap
from nav.marble_dynamics.bridge import predict_replies,reply
from nav.marble_dynamics.global_goals import observed_gems
from nav.marble_dynamics.inventory import context_changes
from nav.marble_dynamics.navigator import project_observation
from nav.marble_dynamics.physics import GeneralPhysics
from nav.marble_dynamics.record import ML
from nav.protocol import RAW_DIM,RAW_GEMS,RAW_POW_HELD


class InventoryGoalRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case=json.loads(Path(__file__).with_name('fixtures').joinpath('live_bridge.json').read_text())['cases'][0]
        cls.engine=GeneralPhysics(CollisionMap(ML/'collision_maps/Concentric_Hunt.npz'))

    def test_script_only_held_type_is_preserved_and_use_is_refused(self):
        d=copy.deepcopy(self.case['dynamics']);d['native']['state']['held']=0
        d['inventory']=dict(held_type=6,mega_active=False,permanent_mega=False,normal_radius=0.,mega_timer_known=False,mega_left_ms=0)
        actions=[reply([0,0,0,0,0,0],use=int(i==0)) for i in range(4)]
        result=predict_replies(self.engine,d,actions)
        self.assertIsNotNone(result['failure']);self.assertIn('Script-held Mega',result['failure']['reason'])
        raw=np.zeros(RAW_DIM,dtype=np.float32)
        projected=project_observation(raw,raw[:3],result['states'][0],0)
        self.assertEqual(projected[RAW_POW_HELD],6);self.assertEqual(raw[RAW_POW_HELD],0)
        self.assertEqual(context_changes(result['states'][0],dict(inventory={**d['inventory'],'held_type':0})),['script_inventory:held_type'])

    def test_timed_active_or_shrinking_mega_cannot_silently_use_constant_radius(self):
        d=copy.deepcopy(self.case['dynamics'])
        d['inventory']=dict(held_type=0,mega_active=True,permanent_mega=False,normal_radius=.18975,mega_timer_known=True,mega_left_ms=20)
        result=predict_replies(self.engine,d,[reply([0,0,0,0,0,0])])
        self.assertIn('activation/expiry',result['failure']['reason'])
        d['inventory']['mega_active']=False;d['native']['state']['radius']=.6666
        result=predict_replies(self.engine,d,[reply([0,0,0,0,0,0])])
        self.assertIn('radius transition',result['failure']['reason'])

    def test_all_observed_gems_preserve_nearest_coordinates_and_exclude_other_items(self):
        raw=np.full(RAW_DIM,-999.,dtype=np.float32);raw[:3]=[2,3,4]
        raw[RAW_GEMS.start:RAW_GEMS.start+5]=[1.000001,2,3,2,4]
        nearby=tuple(map(float,[3.000001,5,7]))
        d=dict(native=dict(state=dict(position=[2,3,4])),items=[
            dict(gem=True,value=2,position=nearby),
            *[dict(gem=True,value=1,position=[30+i,0,0]) for i in range(7)],
            dict(gem=False,value=100,position=[2,3,4]),dict(gem=True,value=0,position=[2,3,4])])
        before=copy.deepcopy(d);goals=observed_gems(d,raw)
        self.assertEqual(len(goals),8);self.assertEqual(goals[0][:3],(float(raw[0])+float(raw[6]),5.,7.))
        self.assertTrue(any(g[0]==36 for g in goals));self.assertEqual(d,before)

    def test_row_cross_preserves_float_bits_at_zero_and_wide_scales(self):
        rng=np.random.RandomState(181);a=rng.normal(size=(5000,3))*np.exp2(rng.randint(-40,40,(5000,1)))
        b=rng.normal(size=(5000,3))*np.exp2(rng.randint(-40,40,(5000,1)))
        a[:2]=[[0.,-0.,0.],[-0.,0.,-0.]];b[:2]=[[1.,0.,-1.],[0.,-1.,0.]]
        np.testing.assert_array_equal(cross_rows(a,b).view(np.uint64),np.cross(a,b).view(np.uint64))


if __name__=='__main__':unittest.main()
