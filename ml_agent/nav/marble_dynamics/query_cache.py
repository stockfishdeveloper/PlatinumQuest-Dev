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
        lo=np.asarray(lo);hi=np.asarray(hi)
        lower_values=tuple(map(float,lo));upper_values=tuple(map(float,hi));span=4*self.cache_cell
        if not all(math.isfinite(a) and math.isfinite(b) and 0<=b-a<=span for a,b in zip(lower_values,upper_values)):
            return super().query_aabb(lo,hi)
        try:
            key=tuple(math.floor(v/self.cache_cell) for v in lower_values)+tuple(math.ceil(v/self.cache_cell) for v in upper_values)
        except OverflowError:
            return super().query_aabb(lo,hi)
        lower=tuple(v*self.cache_cell for v in key[:3]);upper=tuple(v*self.cache_cell for v in key[3:])
        # Check every lookup: two rounded division results can share a key
        # even when one request lies outside the cached enclosing region.
        if any(a>b for a,b in zip(lower,lower_values)) or any(a<b for a,b in zip(upper,upper_values)):
            return super().query_aabb(lo,hi)
        ids=self.query_regions.get(key)
        if ids is None:
            ids=super().query_aabb(lower,upper);self.query_regions[key]=ids;self.cache_misses+=1
            if len(self.query_regions)>self.cache_entries:self.query_regions.popitem(last=False)
        else:
            self.cache_hits+=1;self.query_regions.move_to_end(key)
        low=self.lo[ids];high=self.hi[ids]
        keep=(high[:,0]>=lo[0])&(low[:,0]<=hi[0])&(high[:,1]>=lo[1])&(low[:,1]<=hi[1])&(high[:,2]>=lo[2])&(low[:,2]<=hi[2])
        return ids[keep]
