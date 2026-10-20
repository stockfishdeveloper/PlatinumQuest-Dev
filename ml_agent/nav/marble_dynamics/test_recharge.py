"""Recharge planning retains native initial conditions and legal delayed inputs."""
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

from nav.collision.mesh import CollisionMap
from nav.marble_dynamics.record import ML
from nav.marble_dynamics.physics import GeneralPhysics
from nav.marble_dynamics.bridge import predict_replies
from nav.marble_dynamics.evaluate import errors
from nav.marble_dynamics.recharge import RechargeProposal,choose_recharge


class RechargeRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case=json.loads(Path(__file__).with_name('fixtures').joinpath('blast_bridge.json').read_text())['cases'][1]
        cls.engine=GeneralPhysics(CollisionMap(ML/'collision_maps/Concentric_Hunt.npz'))

    def test_wait_brakes_and_charges_real_meter_before_delayed_blast(self):
        case=self.case;dynamics=copy.deepcopy(case['dynamics']);before=copy.deepcopy(dynamics)
        context=dynamics['blast'];wait=math.ceil((1-context['value'])*context['charge_ms']/64)+2
        position=dynamics['native']['state']['position']
        route=SimpleNamespace(terrain=None,navigate=lambda p,g:(g,False),
            distance=lambda p:math.dist(p,position))
        forecast=predict_replies(self.engine,dynamics,RechargeProposal(route,position,wait,40,3,0))
        self.assertEqual(dynamics,before);self.assertIsNone(forecast['failure'])
        # First six waiting frames match independent native outcomes before
        # the recorded partial-charge blast/jump diverges from this proposal.
        for predicted,row in zip(forecast['states'][1:7],case['rows'][:6]):
            self.assertLess(errors(predicted,row['after'])[0],1e-7)
        impulses=[i for i,c in enumerate(forecast['controls']) if c.get('script_events')]
        self.assertEqual(impulses,[wait+4])
        self.assertEqual(forecast['states'][wait]['blast_context']['value'],1.)
        self.assertEqual(forecast['controls'][wait+4]['script_events'][0]['delta_velocity'][2],10.)
        self.assertTrue(all(not a[4] and not a[6] and not a[8] for a in forecast['proposed'][:wait]))

    def test_wait_requires_available_time_and_supported_regular_rules(self):
        dynamics=copy.deepcopy(self.case['dynamics']);goal=[0,0,0]
        for budget,time_ms in ((10,240000),(1000,1000)):
            self.assertEqual(choose_recharge(None,None,dynamics,goal,budget,time_ms),(None,[]))
        dynamics['blast']['charging']=False
        self.assertEqual(choose_recharge(None,None,dynamics,goal,1000,240000),(None,[]))
        dynamics['blast']['charging']=True;dynamics['blast']['special']=True
        self.assertEqual(choose_recharge(None,None,dynamics,goal,1000,240000),(None,[]))

    def test_measured_long_wait_action_and_landing_without_future_state_inputs(self):
        from nav.marble_dynamics.blast import context_changes
        case=json.loads(Path(__file__).with_name('fixtures').joinpath('recharge_bridge.json').read_text())
        prediction=predict_replies(self.engine,case['dynamics'],case['actions'])
        self.assertIsNone(prediction['failure'])
        for check in case['checks']:
            state=prediction['states'][check['index']+1]
            position,velocity,spin=errors(state,check['after'])
            self.assertLess(position,5e-6);self.assertLess(velocity,5e-5);self.assertLess(spin,1e-4)
            if check['next_context'] is not None:self.assertEqual(context_changes(state,dict(blast=check['next_context'])),[])

    def test_unmodeled_blast_pickup_or_mode_change_cancels_forecast(self):
        from nav.marble_dynamics.blast import context_changes
        context=copy.deepcopy(self.case['dynamics']['blast']);predicted=dict(blast_context=context)
        changed=copy.deepcopy(context);changed['special']=True
        self.assertEqual(context_changes(predicted,dict(blast=changed)),['blast_context:special'])
        changed=copy.deepcopy(context);changed['value']+=.05
        self.assertEqual(context_changes(predicted,dict(blast=changed)),['blast_context:value'])


if __name__=='__main__':unittest.main()
