"""Exact static AABB queries using bounded cached enclosing regions.

The coarse box is only a superset index: every returned triangle still passes
the original six inclusive comparisons against the requested box. Large or
non-finite queries retain the ordinary full scan. No collision detail is removed.
"""
from collections import OrderedDict
import math

import numpy as np

from nav.collision.mesh import CollisionMap


class CachedCollisionMap(CollisionMap):
    def __init__(self,path,cell_size=1.,cache_entries=2048):
        super().__init__(path)
        if not math.isfinite(cell_size) or cell_size<=0 or cache_entries<1:raise ValueError('Finite positive cache cell/capacity required')
        self.cache_cell=float(cell_size);self.cache_entries=int(cache_entries)
        self.query_regions=OrderedDict();self.cache_hits=0;self.cache_misses=0

    def query_aabb(self,lo,hi):
        # Same exact semantics as before (2026-10-15 speed-up: scalar maths
        # instead of array/tuple conversions on this hot path).
        try:
            l0=float(lo[0]);l1=float(lo[1]);l2=float(lo[2]);h0=float(hi[0]);h1=float(hi[1]);h2=float(hi[2])
        except (TypeError,IndexError,ValueError):
            return super().query_aabb(lo,hi)
        cell=self.cache_cell;span=4*cell
        if not (math.isfinite(l0) and math.isfinite(l1) and math.isfinite(l2) and math.isfinite(h0) and math.isfinite(h1) and math.isfinite(h2)
                and 0<=h0-l0<=span and 0<=h1-l1<=span and 0<=h2-l2<=span):
            return super().query_aabb(lo,hi)
        try:
            key=(math.floor(l0/cell),math.floor(l1/cell),math.floor(l2/cell),math.ceil(h0/cell),math.ceil(h1/cell),math.ceil(h2/cell))
        except OverflowError:
            return super().query_aabb(lo,hi)
        # Check every lookup: two rounded division results can share a key
        # even when one request lies outside the cached enclosing region.
        if key[0]*cell>l0 or key[1]*cell>l1 or key[2]*cell>l2 or key[3]*cell<h0 or key[4]*cell<h1 or key[5]*cell<h2:
            return super().query_aabb(lo,hi)
        regions=self.query_regions;ids=regions.get(key)
        if ids is None:
            ids=super().query_aabb((key[0]*cell,key[1]*cell,key[2]*cell),(key[3]*cell,key[4]*cell,key[5]*cell));regions[key]=ids;self.cache_misses+=1
            if len(regions)>self.cache_entries:regions.popitem(last=False)
        else:
            self.cache_hits+=1;regions.move_to_end(key)
        low=self.lo[ids];high=self.hi[ids]
        keep=(high[:,0]>=l0)&(low[:,0]<=h0)&(high[:,1]>=l1)&(low[:,1]<=h1)&(high[:,2]>=l2)&(low[:,2]<=h2)
        return ids[keep]
