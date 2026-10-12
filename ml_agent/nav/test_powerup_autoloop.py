import copy
import json
from pathlib import Path
import tempfile
import unittest

from nav.powerup_autoloop import Loop, accept, practice_choice
from nav.prepare_powerup_curriculum import select


class LoopDecisionTests(unittest.TestCase):
    def test_operator_stop_signals_only_active_run(self):
        with tempfile.TemporaryDirectory() as d:
            loop = Loop(Path(d)/'loop')
            loop.active_stop = Path(d)/'run_STOP'
            loop.stopfile.touch()
            with self.assertRaises(InterruptedError): loop.check_stop()
            self.assertTrue(loop.active_stop.exists())

    def test_resume_preserves_best_and_advances_iteration(self):
        with tempfile.TemporaryDirectory() as d:
            previous = Path(d)/'previous.json'
            previous.write_text(json.dumps(dict(iteration=1,best={'points':40.4},source='frozen',child_pid=12)))
            loop = Loop(Path(d)/'new_loop',previous)
            self.assertEqual(loop.state['iteration'],1)
            self.assertEqual(loop.state['source'],'frozen')
            self.assertNotIn('child_pid',loop.state)

    def test_score_gain_cannot_hide_kotm_or_fall_regression(self):
        base = dict(points=33.1, kotm=124.5, falls=21.15)
        self.assertTrue(accept(dict(points=40.4,kotm=125.5,falls=17.5),base,base))
        self.assertFalse(accept(dict(points=40.4,kotm=90,falls=17.5),base,base))
        self.assertFalse(accept(dict(points=40.4,kotm=125.5,falls=40),base,base))
        self.assertFalse(accept(dict(points=33.2,kotm=125.5,falls=17.5),base,base))

    def example(self):
        candidates, rows = [], []
        for i in range(6):
            s = dict(id=str(i),mission='test',split_group='round'+str(i),raw=[0]*61)
            s['raw'][38] = 2; candidates.append(s)
            rows.extend([dict(id=str(i),branch='no_use',status='ok',points=2,falls=0),
                         dict(id=str(i),branch='handling',status='ok',points=4 if i<3 else 1,falls=0,reset_points=0)])
        holdout = [dict(mission='test',split_group='held'+str(i),raw=candidates[0]['raw']) for i in range(2)]
        return candidates,dict(rows=rows),holdout

    def test_requires_use_save_examples_and_independent_rounds(self):
        c,s,h = self.example()
        self.assertEqual(len(practice_choice(c,s,h)['dev']),6)
        self.assertIsNone(practice_choice(c,s,h[:1]))
        clustered = copy.deepcopy(c)
        for row in clustered[:3]: row['split_group']='same'
        self.assertIsNone(practice_choice(clustered,s,h))

    def test_candidate_cap_can_supply_both_sides_of_judgment_pool(self):
        c,s,h = self.example()
        for row in c:
            row.update(world={'pose':[0,0,0]},goal=[10,0,0],hist=[{}]*32)
        selected,_ = select(c,per_resource=6)
        self.assertIsNotNone(practice_choice(selected,s,h))
        selected,_ = select(c,per_resource=2)
        self.assertIsNone(practice_choice(selected,s,h))

    def test_failed_or_zero_signal_activations_do_not_qualify(self):
        c,s,h = self.example()
        s['rows'][1]['status']='failed'
        self.assertIsNone(practice_choice(c,s,h))
        for r in s['rows']:
            if r['branch']=='handling': r.update(status='ok',points=0,reset_points=5)
        self.assertIsNone(practice_choice(c,s,h))


if __name__ == '__main__':
    unittest.main()
