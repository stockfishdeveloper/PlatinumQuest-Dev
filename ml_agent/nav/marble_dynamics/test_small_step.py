"""Independent native 16ms mechanics with 64ms proposed-input queue timing."""
import copy
import gzip
import json
from pathlib import Path
import unittest

import numpy as np

from nav.collision.mesh import CollisionMap
from nav.marble_dynamics.record import ML,sha
from nav.marble_dynamics.physics import GeneralPhysics
from nav.marble_dynamics.bridge_small_step import predict_small_step
from nav.marble_dynamics.evaluate import errors


class SmallStepRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path(__file__).with_name('fixtures')/'small_step_bridge.json.gz'
        manifest=json.loads(path.with_suffix('.manifest.json').read_text())
        if sha(path)!=manifest['sha256']:raise ValueError('Native fixture changed')
        cls.cases=json.loads(gzip.decompress(path.read_bytes()))['cases']
        cls.engines={name:GeneralPhysics(CollisionMap(ML/'collision_maps'/(name+'.npz')))
                     for name in {c['map'] for c in cls.cases}}

    def test_full_proposed_input_trajectories_match_native_without_reanchoring(self):
        for case in self.cases:
            with self.subTest(map=case['map'],power=case['power']):
                # Native rows are comparison targets only, not predictor inputs.
                before=copy.deepcopy(case['dynamics'])
                forecast=predict_small_step(self.engines[case['map']],before,case['actions'])
                self.assertEqual(before,case['dynamics'])
                self.assertIsNone(forecast['failure'])
                self.assertEqual(len(forecast['native_controls']),len(case['native_rows']))
                self.assertEqual(len(forecast['states']),len(case['actions'])+1)
                self.assertEqual(forecast['ambiguous_intervals'],[])
                measures=[]
                for control,predicted,row in zip(forecast['native_controls'],forecast['native_states'][1:],case['native_rows']):
                    self.assertEqual(control['move'],row['move'])
                    camera=[(e['kind'],e['value']) for e in control['events']]
                    actual=[(e['kind'],e['value']) for e in row['control_events']]
                    self.assertEqual(len(camera),len(actual))
                    for (kind,value),(actual_kind,actual_value) in zip(camera,actual):
                        self.assertEqual(kind,actual_kind);self.assertLess(abs(value-actual_value),1e-7)
                    measures.append(errors(predicted,row['after']))
                bounds=np.max(measures,axis=0)
                self.assertLess(bounds[0],3e-6)
                self.assertLess(bounds[1],6e-5)
                self.assertLess(bounds[2],4e-5)
                if case['expected_blast']:
                    for state,actual in zip(forecast['states'][1:],case['expected_blast']):
                        self.assertEqual(state['blast_context']['special'],actual['special'])
                        self.assertLess(abs(state['blast_context']['value']-actual['value']),1e-6)

    def test_two_reply_prefix_includes_first_action_but_cannot_include_second(self):
        case=next(c for c in self.cases if c['power']==2);engine=self.engines[case['map']]
        first=case['actions'][0];neutral=[0.]*9
        a=predict_small_step(engine,case['dynamics'],[first,neutral])
        b=predict_small_step(engine,case['dynamics'],[first,[1.,0.,0.,0.,1.,2.,1.,2.,1.]])
        c=predict_small_step(engine,case['dynamics'],[neutral,neutral])
        self.assertEqual(a['states'],b['states']);self.assertEqual(a['native_controls'],b['native_controls'])
        self.assertEqual(a['states'][1],c['states'][1])
        self.assertNotEqual(a['states'][2]['held'],c['states'][2]['held'])
        self.assertLess(errors(a['states'][2],case['native_rows'][7]['after'])[0],1e-6)

    def test_native_speed_cutoff_keeps_position_but_advances_spin(self):
        case=next(c for c in self.cases if c['power']==1)
        row=case['native_rows'][0]
        state,events=self.engines[case['map']].step(row['before'],row['move'],16)
        self.assertEqual(state['position'],row['before']['position'])
        self.assertEqual(state['position'],row['after']['position'])
        self.assertNotEqual(state['velocity'],row['before']['velocity'])
        self.assertNotEqual(state['spin'],row['before']['spin'])
        self.assertLess(errors(state,row['after'])[2],1e-9)

    def test_powerup_use_waits_for_second_frame_of_next_reply(self):
        for case in self.cases:
            if not isinstance(case['power'],int):continue
            forecast=predict_small_step(self.engines[case['map']],case['dynamics'],case['actions'][:3])
            self.assertEqual([s['held'] for s in forecast['native_states'][:6]],[case['power']]*6)
            self.assertEqual(forecast['native_states'][6]['held'],0)
            self.assertEqual(forecast['native_controls'][4]['events'][0]['value'],float(np.float32(.7)))
            self.assertFalse(forecast['native_controls'][4]['move']['use'])
            self.assertTrue(forecast['native_controls'][5]['move']['use'])

    def test_blast_reset_leaves_three_frames_of_charge(self):
        for case in self.cases:
            if case['power'] not in ('regular_blast','special_blast'):continue
            forecast=predict_small_step(self.engines[case['map']],case['dynamics'],case['actions'][:3])
            self.assertNotIn('script_events',forecast['native_controls'][3])
            event=forecast['native_controls'][4]['script_events'][0]
            self.assertEqual(event['special'],case['power']=='special_blast')
            self.assertEqual(forecast['native_states'][5]['blast_context']['value'],0.)
            self.assertAlmostEqual(forecast['states'][2]['blast_context']['value'],.00192,places=8)

    def test_unverified_mega_and_blast_modes_still_refused(self):
        case=self.cases[0];dynamics=copy.deepcopy(case['dynamics'])
        dynamics['inventory']['held_type']=6;dynamics['bridge_action'][6]=1
        forecast=predict_small_step(self.engines[case['map']],dynamics,[case['actions'][0]])
        self.assertIn('Mega',forecast['failure']['reason'])
        dynamics=copy.deepcopy(case['dynamics']);dynamics['blast']['triple']=True;dynamics['bridge_action'][8]=1
        forecast=predict_small_step(self.engines[case['map']],dynamics,[case['actions'][0]])
        self.assertIn('Triple Blast',forecast['failure']['reason'])

    def test_dispatch_and_split_continuations_preserve_every_frame(self):
        from nav.marble_dynamics.bridge import predict_replies
        from nav.marble_dynamics.viability import resume_forecast,concatenate
        # Include ordinary controls, use/aim hold, Blast reset/charge and effects.
        for case in self.cases:
            with self.subTest(map=case['map'],power=case['power']):
                engine=self.engines[case['map']];engine.bridge_physics_ms=16
                try:
                    actions=case['actions'][:20];split=3
                    whole=predict_replies(engine,case['dynamics'],actions)
                    prefix=predict_replies(engine,case['dynamics'],actions[:split])
                    rest=predict_replies(engine,resume_forecast(case['dynamics'],prefix),actions[split:])
                    joined=concatenate(prefix,rest)
                    for key in ['states','native_states','controls','native_controls','events','proposed']:
                        self.assertEqual(joined[key],whole[key])
                    self.assertEqual(whole,predict_small_step(engine,case['dynamics'],actions))
                finally:del engine.bridge_physics_ms

    def test_independent_episode_audit_checks_all_native_frames(self):
        import tempfile
        from nav.collision.mesh import atomic_json
        from nav.marble_dynamics.small_step_episode_audit import audit_branch
        case=next(c for c in self.cases if c['power']=='regular_blast')
        source=ML/case['source']
        recorded=json.loads((source/'observations.json').read_text())[:20]
        engine=self.engines[case['map']]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/case['map']/'0'/'fixture';folder.mkdir(parents=True)
            logs=[dict(decision=l['index'],dynamics=l['dynamics'],action=l['action'],
                       predicted_next=l['actual_next']['native']['state'],actual_next=l['actual_next']['native']['state'],
                       score_delta=l['score_delta'],outcome=l['outcome']) for l in recorded]
            atomic_json(folder/'decisions.json',logs)
            (folder/'native.jsonl.gz').write_bytes(gzip.compress(('\n'.join(json.dumps(r) for r in case['native_rows'][:80])+'\n').encode(),mtime=0))
            try:
                report=audit_branch(root,dict(map=case['map'],seed=0,branch='fixture',actual_points=0.),engine)
                self.assertEqual(report['native_transitions'],80)
                self.assertEqual(report['five_step_windows'],4)
                self.assertEqual(report['movement_mismatches'],[])
                self.assertEqual(report['camera_mismatches'],[])
                self.assertEqual(report['script_actuation_mismatches'],[])
                self.assertEqual(report['contact_discrepancies'],[])
                self.assertEqual(len(report['regular_blast_activations']),1)
            finally:del engine.bridge_physics_ms


if __name__=='__main__':unittest.main()
