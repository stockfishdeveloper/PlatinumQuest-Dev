"""Exact native-state tree copy with independent mutable forecast histories."""
import copy
import gzip
import json
from pathlib import Path
import unittest

import numpy as np
from nav.marble_dynamics.state_copy import clone


class NativeStateCopyRegressions(unittest.TestCase):
    def test_every_fixture_state_matches_deepcopy(self):
        fixture=json.loads(gzip.decompress((Path(__file__).with_name('fixtures')/'small_step_bridge.json.gz').read_bytes()))
        for case in fixture['cases']:
            for row in case['native_rows']:
                self.assertEqual(clone(row['before']),copy.deepcopy(row['before']))
                self.assertEqual(clone(row['after']),copy.deepcopy(row['after']))

    def test_nested_effects_parameters_and_contexts_do_not_share_mutable_state(self):
        original=dict(position=[1.,2.,3.],effects=[[False,0],[True,400]],
            parameters=dict(effect_duration_ms=[500,1000]),blast_context=dict(value=.2),
            nested=dict(extra=[dict(value=[8.])]))
        expected=copy.deepcopy(original);a=clone(original);b=clone(a)
        b['position'][0]=77.;b['effects'][1][1]=0;b['parameters']['effect_duration_ms'][0]=0
        b['blast_context']['value']=1.;b['nested']['extra'][0]['value'][0]=0
        self.assertEqual(original,expected);self.assertEqual(a,expected)

    def test_unusual_non_json_extension_uses_independent_deepcopy(self):
        original=dict(extension=np.asarray([1.,2.,3.]),tuple_value=([4.],))
        cloned=clone(original);cloned['extension'][0]=0;cloned['tuple_value'][0][0]=0
        np.testing.assert_array_equal(original['extension'],[1.,2.,3.])
        self.assertEqual(original['tuple_value'],([4.],))


if __name__=='__main__':unittest.main()
