"""Analytic contact cases, including boundaries that floor rasters cannot retain."""
import unittest
import hashlib
import tempfile
from pathlib import Path

import numpy as np

from nav.collision.mesh import CollisionMap, atomic_json
from nav.collision.mission import mission_objects


def fixture(triangles, normals, boundary_edges=None):
    m=CollisionMap.__new__(CollisionMap)
    m.triangles=np.asarray(triangles,dtype=float)
    m.normals=np.asarray(normals,dtype=float)
    m.planes=np.column_stack([m.normals,-np.einsum('ij,ij->i',m.normals,m.triangles[:,0])])
    m.lo,m.hi=m.triangles.min(axis=1),m.triangles.max(axis=1)
    m.material_ids=np.zeros(len(m.triangles),dtype=np.int32)
    m.object_ids=np.ones(len(m.triangles),dtype=np.int32)
    m.surface_keys=np.zeros(len(m.triangles),dtype=np.uint32)
    m.boundary_edges=np.ones((len(m.triangles),3),dtype=bool) if boundary_edges is None else np.asarray(boundary_edges)
    m.info=dict(materials=[dict(name='test',friction=1.,restitution=.5,force=0.,resolved=True)])
    return m


class Queries(unittest.TestCase):
    def test_layered_ceiling_and_floor(self):
        m=fixture([[[-2,-2,0],[2,-2,0],[0,2,0]],
                   [[-2,-2,2],[0,2,2],[2,-2,2]]],[[0,0,1],[0,0,-1]])
        self.assertAlmostEqual(m.raycast([0,0,1],[0,0,-1])['point'][2],0.)
        self.assertAlmostEqual(m.raycast([0,0,1],[0,0,3])['point'][2],2.)
        self.assertIsNone(m.raycast([0,0,.5],[0,0,1.5]))
        self.assertEqual(len(m.query_aabb([-3,-3,-1],[3,3,3])),2)
        self.assertEqual(m.features([1])['normals'][0,2],-1.)
        f=m.features([1],origin=[0,0,10])
        np.testing.assert_allclose(f['vertices'][0] @ f['planes'][0,:3]+f['planes'][0,3],0.)

    def test_finite_one_sided_ray(self):
        m=fixture([[[0,0,0],[2,0,0],[0,2,0]]],[[0,0,1]])
        self.assertIsNone(m.raycast([.5,.5,-1],[.5,.5,1]))
        self.assertIsNotNone(m.raycast([.5,.5,-1],[.5,.5,1],cull_backfaces=False))
        self.assertIsNone(m.raycast([.5,.5,2],[.5,.5,1]))
        # The AABB touches the plane, but the actual edge contact is beyond end.
        self.assertIsNone(m.sweep_sphere([1,-.12,1],[1,-.12,.18],.2))

    def test_sphere_face_edge_corner(self):
        m=fixture([[[0,0,0],[2,0,0],[0,2,0]]],[[0,0,1]])
        hit=m.sweep_sphere([.5,.5,1],[.5,.5,-1],.2)
        self.assertAlmostEqual(hit['t'],.4)
        np.testing.assert_allclose(hit['point'],[.5,.5,0],atol=1e-10)
        # Edge contact occurs at z=sqrt(r^2-offset^2), not at z=r.
        hit=m.sweep_sphere([1,-.12,1],[1,-.12,-1],.2)
        self.assertAlmostEqual(hit['center'][2],.16)
        hit=m.sweep_sphere([-.12,-.12,1],[-.12,-.12,-1],.2)
        self.assertAlmostEqual(hit['center'][2],np.sqrt(.2**2-2*.12**2))

    def test_zero_area_contact_primitives(self):
        m=fixture([[[0,0,0],[1,0,0],[2,0,0]]],[[0,0,1]])
        with np.errstate(all='raise'):
            self.assertEqual(len(m.sphere_contacts([.5,.1,0],.2)['triangle_ids']),1)
            self.assertIsNone(m.raycast([.5,0,1],[.5,0,-1]))
            hit=m.sweep_sphere([.5,0,1],[.5,0,-1],.2)
            self.assertAlmostEqual(hit['center'][2],.2)

    def test_preserve_physical_polygon_boundaries(self):
        m=fixture([[[0,0,0],[2,0,0],[2,2,0]],[[0,0,0],[2,2,0],[0,2,0]]],
                   [[0,0,1],[0,0,1]],[[True,True,False],[False,True,True]])
        self.assertEqual(int(m.boundary_edges.sum()),4)
        hit=m.sweep_sphere([1,1,1],[1,1,-1],.2)
        self.assertAlmostEqual(hit['t'],.4)

    def test_comment_and_child_safe_mission_annotations(self):
        objects=mission_objects('''// new TSStatic(bogus) {position="1 2 3";};
        new SimGroup(g) {new TSStatic(a) {position="4 5 6";shapeName="x.dts";};};''')
        self.assertEqual(len(objects),2)
        self.assertEqual(objects[0]['fields'],{})
        self.assertEqual(objects[1]['fields']['position'],'4 5 6')

    def test_native_plane_avoids_false_thin_fan_surface(self):
        with tempfile.TemporaryDirectory(prefix='pq_collision_test_') as tmp:
            path=Path(tmp)/'thin.npz'
            # Rounding put the middle point just off the authoritative floor
            # plane. This is a line contact primitive, not a far-reaching face.
            source=np.asarray([[[0,0,0],[2,0,0],[1,0,.000002]]],dtype=np.float32)
            np.savez_compressed(path,triangles=source,normals=[[0,0,1]],planes=[[0,0,1,0]],
                                material_ids=[0],object_ids=[1],surface_keys=[0],boundary_edges=[[True,True,True]])
            atomic_json(path.with_suffix('.json'),dict(schema='collision_map_v1',
                        mesh_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),objects=[dict(id=1)],
                        materials=[dict(name='test',friction=1.,restitution=.5,force=0.,resolved=True)]))
            m=CollisionMap(path)
            self.assertGreater(m.source_triangles[0,2,2],0.)
            self.assertEqual(m.triangles[0,2,2],0.)
            self.assertIsNone(m.sweep_sphere([1,6,1],[1,-2,-1],.2))
            self.assertIsNone(m.raycast([1,6,1],[1,-2,-1]))


if __name__=='__main__':
    unittest.main()
