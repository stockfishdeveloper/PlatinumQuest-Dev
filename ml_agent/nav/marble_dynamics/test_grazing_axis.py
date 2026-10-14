"""Exact-axis floor/wall fixture and genuinely fragile grazing signs."""
import json
from pathlib import Path
import unittest

import numpy as np

from nav.collision.mesh import CollisionMap
from nav.marble_dynamics.bridge import predict_replies
from nav.marble_dynamics.evaluate import errors
from nav.marble_dynamics.physics import GeneralPhysics,grazing_contacts
from nav.marble_dynamics.record import ML


class GrazingAxisRegressions(unittest.TestCase):
    def contacts(self):
        return [dict(normal=np.array([0.,0.,1.]),plane_normal=np.array([0.,0.,1.])),
                dict(normal=np.array([1.,0.,0.]),plane_normal=np.array([1.,0.,0.]))]

    def test_exact_axis_zero_is_not_rounding_sensitive(self):
        dots,fragile=grazing_contacts(np.array([-2.,1.,0.]),self.contacts(),.5)
        self.assertEqual(dots,[0.,-2.]);self.assertEqual(fragile,[])
        contacts=self.contacts();contacts[0]['plane_normal']*=float(np.float32(.99999994))
        self.assertEqual(grazing_contacts(np.array([-2.,1.,0.]),contacts,.5)[1],[])

    def test_near_zero_and_oblique_cancellation_remain_fragile(self):
        self.assertEqual(grazing_contacts(np.array([-2.,1.,1e-14]),self.contacts(),.5)[1],[0])
        contacts=[dict(normal=np.array([1.,1.,0.])/np.sqrt(2),plane_normal=np.array([1.,1.,0.])/np.sqrt(2)),
                  dict(normal=np.array([1.,0.,0.]))]
        self.assertEqual(grazing_contacts(np.array([-2.,2.,0.]),contacts,.5)[1],[0])
        contacts=self.contacts();contacts[0]['plane_normal']=np.array([1.,0.,0.])
        self.assertEqual(grazing_contacts(np.array([-2.,1.,0.]),contacts,.5)[1],[0])

    def test_native_floor_wall_and_corrected_tangent_edge(self):
        case=json.loads(Path(__file__).with_name('fixtures').joinpath('grazing_axis_bridge.json').read_text())
        engine=GeneralPhysics(CollisionMap(ML/'collision_maps'/(case['map']+'.npz')))
        result=predict_replies(engine,case['dynamics'],case['actions'])
        self.assertIsNone(result['failure'])
        maxima=np.max([errors(a,b) for a,b in zip(result['states'][1:],case['actual'])],axis=0)
        self.assertLess(maxima[0],1e-6);self.assertLess(maxima[1],1e-5);self.assertLess(maxima[2],1e-4)
        self.assertEqual(result['ambiguous_intervals'],[])
        # Native float32 radius multiplication now resolves this edge directly.
        # Compare all original native states, rather than requiring the old flag.
        edge=case['separate_geometry_case'];result=predict_replies(engine,edge['dynamics'],edge['actions'])
        self.assertIsNone(result['failure']);self.assertEqual(result['ambiguous_intervals'],[])
        self.assertEqual(len(result['states'])-1,len(edge['actual']))
        maxima=np.max([errors(a,b) for a,b in zip(result['states'][1:],edge['actual'])],axis=0)
        self.assertLess(maxima[0],1e-6);self.assertLess(maxima[1],1e-5);self.assertLess(maxima[2],1e-4)


if __name__=='__main__':unittest.main()
