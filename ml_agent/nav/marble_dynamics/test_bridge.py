"""Control/timing regressions from archived game execution, including jumps."""
import copy
import json
from pathlib import Path
import unittest

from nav.collision.mesh import CollisionMap
from nav.marble_dynamics.record import ML
from nav.marble_dynamics.bridge import predict_replies,reply
from nav.marble_dynamics.physics import GeneralPhysics
from nav.marble_dynamics.evaluate import errors
from nav.protocol import parse_message,RAW_DIM
from nav.marble_dynamics.events import external_changes


class LiveBridgeRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture=json.loads(Path(__file__).with_name('fixtures').joinpath('live_bridge.json').read_text())
        cls.engine=GeneralPhysics(CollisionMap(ML/'collision_maps/Concentric_Hunt.npz'))

    def test_pending_inputs_and_absolute_camera_match_native(self):
        for case in self.fixture['cases']:
            result=predict_replies(self.engine,case['dynamics'],[reply(j) for j in case['joysticks']])
            self.assertIsNone(result['failure'])
            for control,row,state in zip(result['controls'],case['rows'],result['states'][1:]):
                self.assertEqual(control['move'],row['move'])
                self.assertAlmostEqual(control['events'][0]['value'],row['control_events'][0]['value'],places=6)
                self.assertLess(errors(state,row['after'])[0],1e-6)

    def test_incorrect_pending_move_changes_native_outcome(self):
        case=self.fixture['cases'][1];wrong=copy.deepcopy(case['dynamics'])
        wrong['native']['pending_move']['x']=0.;wrong['native']['pending_move']['y']=0.
        result=predict_replies(self.engine,wrong,[reply(j) for j in case['joysticks']])
        self.assertGreater(errors(result['states'][-1],case['rows'][-1]['after'])[0],.001)

    def test_synchronized_payload_preserves_observation_and_tick(self):
        dyn=self.fixture['cases'][0]['dynamics'];raw=[0.]*RAW_DIM
        msg=parse_message(json.dumps(raw)+'|2|0|0|DYNAMICS:'+json.dumps(dyn)+'|42')
        self.assertEqual(msg.tick,'42');self.assertEqual(msg.gem_delta,2.)
        self.assertEqual(msg.dynamics,dyn);self.assertEqual(len(msg.obs),RAW_DIM)
        self.assertIsNone(parse_message(json.dumps(raw)+'|0|0|0|43').dynamics)

    def test_parameter_round_trip_is_not_an_external_event(self):
        case=self.fixture['cases'][0];row=case['rows'][0]
        state,_=self.engine.step(row['before'],row['move'],64)
        self.assertEqual(external_changes(state,row['after']),[])
        changed=copy.deepcopy(row['after']);changed['parameters']['gravity']=12.
        self.assertEqual(external_changes(state,changed),['parameters'])
        changed=copy.deepcopy(row['after']);changed['held']=5
        self.assertEqual(external_changes(state,changed),['held'])

    def test_stationary_braking_does_not_create_a_launch(self):
        from nav.marble_dynamics.gem_demo import Proposal
        state=copy.deepcopy(self.fixture['cases'][0]['dynamics']['native']['state'])
        state['velocity']=[0.,0.,0.]
        goal=[state['position'][0]+30,state['position'][1]+20,state['position'][2]]
        action=Proposal(goal,brake=True)(state,0)
        self.assertEqual(action[:5],[0.,0.,0.,0.,0])

    def test_script_saturation_rounding_matches_native_policy_commands(self):
        case=json.loads(Path(__file__).with_name('fixtures').joinpath('bridge_rounding.json').read_text())
        engine=GeneralPhysics(CollisionMap(ML/'collision_maps/KingOfTheMarble_Hunt.npz'))
        result=predict_replies(engine,case['dynamics'],case['actions'])
        self.assertIsNone(result['failure'])
        for control,row in zip(result['controls'],case['native_rows']):
            self.assertEqual(control['move'],row['move'])
        self.assertLess(errors(result['states'][-1],case['native_rows'][-1]['after'])[0],.00001)

    def test_float_assignments_callbacks_and_camera_rounding_match_native(self):
        fixture=json.loads(Path(__file__).with_name('fixtures').joinpath('bridge_float_stack.json').read_text())
        engine=GeneralPhysics(CollisionMap(ML/'collision_maps/KingOfTheMarble_Hunt.npz'))
        for case in fixture['cases']:
            result=predict_replies(engine,case['dynamics'],case['actions'])
            self.assertIsNone(result['failure'])
            for control,row in zip(result['controls'],case['native_rows']):
                self.assertEqual(control['move'],row['move'])
                self.assertAlmostEqual(control['events'][0]['value'],row['control_events'][0]['value'],places=7)
            self.assertLess(errors(result['states'][-1],case['native_rows'][-1]['after'])[0],.000001)

    def test_regular_blast_full_partial_and_refused_match_native(self):
        fixture=json.loads(Path(__file__).with_name('fixtures').joinpath('blast_bridge.json').read_text())
        for case in fixture['cases']:
            before=copy.deepcopy(case['dynamics'])
            result=predict_replies(self.engine,case['dynamics'],case['actions'])
            self.assertIsNone(result['failure']);self.assertEqual(case['dynamics'],before)
            for control,state,row in zip(result['controls'],result['states'][1:],case['rows']):
                self.assertEqual(control['move'],row['move'])
                position,velocity,spin=errors(state,row['after'])
                self.assertLess(position,2e-6);self.assertLess(velocity,5e-5);self.assertLess(spin,1e-8)
            fired=[i for i,c in enumerate(result['controls']) if c.get('script_events')]
            self.assertEqual(fired,[1] if case['start']==0 else [6] if case['start']==95 else [])

    def test_blast_missing_or_unverified_rules_refuse_prediction(self):
        case=json.loads(Path(__file__).with_name('fixtures').joinpath('blast_bridge.json').read_text())['cases'][0]
        dynamics=copy.deepcopy(case['dynamics']);dynamics['bridge_action'][8]=1
        for flag in ('triple','fireball','cannon'):
            modified=copy.deepcopy(dynamics);modified['blast'][flag]=True
            result=predict_replies(self.engine,modified,[reply([0,0,0,0,0,0])])
            self.assertIsNotNone(result['failure']);self.assertEqual(len(result['controls']),0)
        dynamics.pop('blast')
        result=predict_replies(self.engine,dynamics,[reply([0,0,0,0,0,0])])
        self.assertIn('context unavailable',result['failure']['reason'])

    def test_native_item_charged_blast_impulse_and_consumption(self):
        fixture=json.loads(Path(__file__).with_name('fixtures').joinpath('special_blast_bridge.json').read_text())
        for case in fixture['cases']:
            result=predict_replies(self.engine,case['dynamics'],case['actions'])
            self.assertIsNone(result['failure'])
            for control,state,row,context in zip(result['controls'],result['states'][1:],case['rows'],case['observed_context']):
                self.assertEqual(control['move'],row['move'])
                self.assertLess(errors(state,row['after'])[0],2e-6)
                self.assertLess(errors(state,row['after'])[1],5e-5)
                self.assertEqual(state['blast_context']['special'],context['special'])
                self.assertLess(abs(state['blast_context']['value']-context['value']),2e-6)
        result=predict_replies(self.engine,fixture['cases'][0]['dynamics'],fixture['cases'][0]['actions'])
        self.assertTrue(result['controls'][1]['script_events'][0]['special'])
        self.assertFalse(result['states'][2]['blast_context']['special'])


if __name__=='__main__':unittest.main()
