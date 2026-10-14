"""Isolated compiled sweep prototype; not enabled in the live evaluator.

The original full mesh and broad phase remain authoritative. No approximate
geometry, motion, learned fit or call into future game physics. Caller must
provide an independently built validation DLL explicitly.
"""
import ctypes

import numpy as np

from nav.collision.mesh import CollisionMap


class CompiledSweepMap(CollisionMap):
    def __init__(self,path,dll):
        super().__init__(path)
        self.library=ctypes.CDLL(str(dll));self.kernel=self.library.sweep_candidates
        self.kernel.restype=None
        self.kernel.argtypes=[ctypes.c_int64,*([ctypes.c_void_p]*11),*([ctypes.c_double]*3),ctypes.c_void_p,ctypes.c_void_p]
        self._face_geometry(np.asarray([],dtype=np.int64))
        self.calls=0

    def sweep_sphere(self,start,end,radius,native_dt_s=None):
        start=np.ascontiguousarray(start,dtype=np.float64);end=np.asarray(end,dtype=np.float64);velocity=np.ascontiguousarray(end-start)
        if radius<=0:raise ValueError('Radius must be positive')
        speed2=float(velocity@velocity)
        if speed2<1e-18:return None
        ids=np.ascontiguousarray(self.query_aabb(np.minimum(start,end)-radius,np.maximum(start,end)+radius),dtype=np.int64)
        if not len(ids):return None
        times=np.empty(len(ids));points=np.empty((len(ids),3))
        arrays=[ids,self.triangles,self.source_triangles,self.planes,self.boundary_edges,self._face_ab,
                self._face_ac,self._face_cross,self._face_nn,start,velocity]
        self.kernel(len(ids),*[a.ctypes.data for a in arrays],speed2,radius,native_dt_s or 0.,
                    times.ctypes.data,points.ctypes.data)
        self.calls+=1
        if not np.isfinite(times).any():return None
        i=int(np.argmin(times));idx=int(ids[i]);center=start+times[i]*velocity;normal=center-points[i]
        normal/=np.linalg.norm(normal)
        return dict(triangle_id=idx,object_id=int(self.object_ids[idx]),t=float(times[i]),center=center.tolist(),
                    point=points[i].tolist(),normal=normal.tolist(),material=self.info['materials'][int(self.material_ids[idx])])
