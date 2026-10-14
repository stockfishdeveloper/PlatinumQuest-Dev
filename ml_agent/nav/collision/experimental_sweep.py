"""Compiled sweep prototype, enabled only by an explicit evaluator DLL option.

The original full mesh and broad phase remain authoritative. No approximate
geometry, motion, learned fit or call into future game physics. Caller must
provide an independently built validation DLL explicitly.
"""
import ctypes

import numpy as np

from nav.collision.mesh import CollisionMap
from nav.marble_dynamics.query_cache import CachedCollisionMap


class CompiledSweepMap(CollisionMap):
    def __init__(self,path,dll):
        super().__init__(path)
        self.library=ctypes.CDLL(str(dll))
        version=self.library.sweep_kernel_abi;version.argtypes=[];version.restype=ctypes.c_int
        if version()!=3:raise ValueError('Incompatible analytical sweep ABI')
        self.kernel=self.library.sweep_candidates
        self.kernel.restype=None
        self.kernel.argtypes=[ctypes.c_int64,*([ctypes.c_void_p]*15),*([ctypes.c_double]*3),ctypes.c_void_p,ctypes.c_void_p]
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
        # NumPy's matrix-vector reductions have a distinct rounding order from
        # ordinary scalar dot products. Preserve those exact operations here;
        # the compiled kernel replaces the remaining temporary-array arithmetic.
        planes=self.planes[ids];normals=planes[:,:3];tri=self.source_triangles[ids]
        approaches=normals@velocity;distances=normals@start+planes[:,3]
        edge_dots=np.stack([(tri[:,(j+1)%3]-tri[:,j])@velocity for j in range(3)])
        vertex_dots=np.stack([(start-tri[:,j])@velocity for j in range(3)])
        arrays=[ids,self.triangles,self.source_triangles,self.planes,self.boundary_edges,self._face_ab,
                self._face_ac,self._face_cross,self._face_nn,start,velocity,approaches,distances,edge_dots,vertex_dots]
        self.kernel(len(ids),*[a.ctypes.data for a in arrays],speed2,radius,native_dt_s or 0.,
                    times.ctypes.data,points.ctypes.data)
        self.calls+=1
        if not np.isfinite(times).any():return None
        i=int(np.argmin(times));idx=int(ids[i]);center=start+times[i]*velocity;normal=center-points[i]
        normal/=np.linalg.norm(normal)
        return dict(triangle_id=idx,object_id=int(self.object_ids[idx]),t=float(times[i]),center=center.tolist(),
                    point=points[i].tolist(),normal=normal.tolist(),material=self.info['materials'][int(self.material_ids[idx])])


class CachedCompiledSweepMap(CompiledSweepMap,CachedCollisionMap):
    """Keep the exact broad-phase cache and compiled narrow phase independent."""
    pass
