"""Exact query regressions, including a demonstrated cached float-boundary miss."""
from collections import OrderedDict
import unittest

import numpy as np

from nav.collision.test_queries import fixture
from nav.marble_dynamics.query_cache import CachedCollisionMap


def cached(mesh,cell=1.):
    result=CachedCollisionMap.__new__(CachedCollisionMap);result.__dict__.update(mesh.__dict__)
    result.cache_cell=cell;result.cache_entries=8;result.query_regions=OrderedDict()
    result.cache_hits=0;result.cache_misses=0
    return result


class QueryCacheRegressions(unittest.TestCase):
    def test_hit_rechecks_float_enclosure(self):
        edge=-1000.;below=float(np.nextafter(edge,-np.inf))
        mesh=fixture([[[below,0,0],[below,.05,0],[below,0,.05]]],[[1,0,0]])
        query=cached(mesh,.1);upper=[edge,.1,.1]
        self.assertEqual(query.query_aabb([edge,0,0],upper).tolist(),[])
        self.assertEqual(query.cache_misses,1)
        self.assertEqual(query.query_aabb([below,0,0],upper).tolist(),[0])
        self.assertEqual(query.query_aabb([below,0,0],upper).tolist(),mesh.query_aabb([below,0,0],upper).tolist())

    def test_small_large_nonfinite_and_reversed_queries_match_original(self):
        mesh=fixture([[[-2,0,0],[2,0,0],[0,2,0]],[[0,0,2],[2,0,2],[0,2,2]]],[[0,0,1],[0,0,1]])
        rng=np.random.RandomState(77)
        for cell in (.1,.3,1.,1.1):
            query=cached(mesh,cell)
            boxes=[([-np.inf]*3,[np.inf]*3),([1,1,1],[-1,-1,-1]),([np.nan,0,0],[1,1,1])]
            for _ in range(100):
                lower=rng.uniform(-4,4,3);upper=lower+rng.uniform(0,5,3);boxes.append((lower,upper))
            for lower,upper in boxes:
                expected=mesh.query_aabb(lower,upper)
                np.testing.assert_array_equal(query.query_aabb(lower,upper),expected)
                np.testing.assert_array_equal(query.query_aabb(lower,upper),expected)


if __name__=='__main__':unittest.main()
