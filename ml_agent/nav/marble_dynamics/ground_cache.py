"""Bounded memoization of exact repeated static walking-clearance segments.

Keys preserve every coordinate and direction; no rounding, spatial coarsening or
triangle removal. First evaluations use the original geometry predicate.
"""
from collections import OrderedDict
import math

import numpy as np


class MemoGroundClearance:
    def __init__(self,original,entries=16384):
        if entries<1:raise ValueError('Positive clearance cache capacity required')
        self.original=original;self.entries=int(entries);self.regions=OrderedDict()
        self.hits=0;self.misses=0;self.checked=0;self.blocked=0

    @property
    def radius(self):return self.original.radius

    def point_clear(self,point):return self.original.point_clear(point)

    def segment(self,start,end):
        self.checked+=1;key=(self.radius,*map(float,start),*map(float,end))
        if not all(math.isfinite(v) for v in key):return self.original.segment(start,end)
        if key in self.regions:
            clear=self.regions[key];self.hits+=1;self.regions.move_to_end(key)
        else:
            clear=self.original.segment(start,end);self.regions[key]=clear;self.misses+=1
            if len(self.regions)>self.entries:self.regions.popitem(last=False)
        self.blocked+=int(not clear)
        return clear

    def __call__(self,floor_points):
        centers=[np.asarray(p)+[0,0,self.radius+.001] for p in floor_points]
        return all(self.segment(a,b) for a,b in zip(centers,centers[1:]))
