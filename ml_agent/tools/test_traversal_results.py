import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from tools.traversal_results import screen


class EqualRewardTiming(unittest.TestCase):
    def test_different_rewards_and_failures_remain_in_totals_without_time_claim(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            rows={
                'base':[dict(decision=1,t_s=10.,points=1.,why='trip_done',t_spawn=5.,t_trip=5.),
                        dict(decision=2,t_s=12.,points=2.,why='trip_done',t_spawn=6.,t_trip=6.)],
                'candidate':[dict(decision=1,t_s=20.,points=5.,why='trip_done',t_spawn=4.,t_trip=16.),
                             dict(decision=2,t_s=8.,points=0.,why='fell',t_spawn=None,t_trip=None)]}
            for variant,items in rows.items():
                for row in items:row.update(map='map',pool='pool',variant=variant)
                (root/(variant+'.map.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in items))
            with contextlib.redirect_stdout(io.StringIO()):result=screen(root,'base')['candidate']
            self.assertEqual(result['pairs'],2)
            self.assertIsNone(result['time_ratio'])
            self.assertEqual(result['point_mismatch_completions'],1)
            self.assertEqual((result['base_points'],result['points']),(3.,5.))
            self.assertEqual((result['base_failures'],result['failures']),(0,1))
            self.assertAlmostEqual(result['group_ratio'],.8)


if __name__=='__main__':unittest.main()
