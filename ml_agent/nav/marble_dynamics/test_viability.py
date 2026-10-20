"""Native stopping exposure and exact split forecasts across queued controls."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch,Mock
from types import SimpleNamespace

import numpy as np

from nav.collision.mesh import CollisionMap
from nav.marble_dynamics.bridge import predict_replies,reply
from nav.marble_dynamics.evaluate import errors
from nav.marble_dynamics.physics import GeneralPhysics
from nav.marble_dynamics.record import ML
from nav.marble_dynamics.search import terminal_support
from nav.marble_dynamics.viability import resume_forecast,concatenate,braking_viability,ordinary_recovery,ActorBrakeContinuation


def fixture(name):
    return json.loads(Path(__file__).with_name('fixtures').joinpath(name).read_text())


class RecoveryContracts(unittest.TestCase):
    def test_longer_actor_prefix_reuses_real_reply_without_double_recurrence(self):
        from nav.protocol import RAW_DIM
        c=fixture('landing_bridge.json');state=c['dynamics']['native']['state']
        actor=Mock();clone=Mock();actor.clone.return_value=clone
        clone.act.return_value=[.3,0,0,0,0,0,0,0,0]
        route=SimpleNamespace(navigate=lambda p,g:(list(g),False))
        raw=np.zeros(RAW_DIM,dtype=np.float32);raw[:3]=state['position']
        first=[.8,0,0,0,0,0,0,0,0]
        proposal=ActorBrakeContinuation(actor,raw,route,[10,20,30],first,replies=12,prefix_replies=8)
        self.assertEqual(proposal(state,0),first)
        clone.act.assert_not_called();actor.act.assert_not_called();actor.sent_action.assert_not_called()
        clone.sent_action.assert_called_once()
        proposal(state,1);clone.act.assert_called_once();clone.outcome.assert_called_once()
        self.assertEqual(clone.sent_action.call_count,2)
        proposal(state,8);self.assertEqual(clone.act.call_count,1)
        self.assertEqual(clone.sent_action.call_count,2)

    def test_longer_actor_prefix_requires_progress_and_preserves_old_backups(self):
        class Route:
            def navigate(self,p,g):self.goal=list(g)
            def distance(self,p):return 10.-p[0]
        route=Route();route.goal=[99,99,99]
        dynamics={'native':{'state':{'position':[0,0,0]}},'items':[]}
        engine=SimpleNamespace(mesh=None,planning_deadline=None)
        proposal=SimpleNamespace(steps=12,prefix_replies=8)
        for progress in (0.,1.):
            rollout=self.rollout(12);rollout['states'][-1]['position'][0]=progress
            with self.subTest(progress=progress),patch('nav.marble_dynamics.viability.predict_replies',return_value=rollout),patch('nav.marble_dynamics.viability.terminal_support',return_value=(True,True,0)):
                chosen,audit=ordinary_recovery(engine,dynamics,[0.]*9,replies=12,budget_s=1,
                    landing_limit=0,actor_continuation=proposal,task_route=route,task_goal=[10,0,0])
            self.assertEqual(route.goal,[99,99,99])
            if progress:
                self.assertEqual(chosen[0]['actor_prefix_replies'],8)
                self.assertEqual(chosen[0]['landing_after_reply'],10)
                self.assertEqual(chosen[0]['task_progress_u'],1.)
            else:
                self.assertEqual(audit['candidates'][0]['rejected'],'no_useful_task_progress')
                self.assertEqual(chosen[0]['label'],'recovery_actor_then_brake')

    def test_longer_actor_prefix_cannot_complete_before_queued_actions_finish(self):
        from nav.marble_dynamics.landing import landing_complete
        from nav.marble_dynamics.search import PlanMemory
        memory=PlanMemory();memory.landing_mode=True;memory.physical_recovery=True
        memory.landing_goal=[0,0,0];memory.landing_after_reply=10
        state={'position':[0,0,0],'velocity':[0,0,0]};memory.observed_tail=[state]*4
        with patch('nav.marble_dynamics.search.terminal_support',return_value=(True,True,0)):
            memory.executed=9;self.assertFalse(landing_complete(SimpleNamespace(mesh=None),state,memory))
            memory.executed=10;self.assertTrue(landing_complete(SimpleNamespace(mesh=None),state,memory))

    def rollout(self,replies=4,speed=0):
        states=[{'position':[0,0,0],'velocity':[speed,0,0]} for _ in range(replies+1)]
        return dict(states=states,controls=[{}]*replies,proposed=[[0.]*9]*replies,failure=None,ambiguous_intervals=[])

    def test_incomplete_or_ambiguous_forecast_is_never_accepted_as_recovery(self):
        dynamics={'native':{'state':{'position':[0,0,0]}}};engine=SimpleNamespace(mesh=None,planning_deadline=None)
        for change in ('partial','ambiguity','speed'):
            rollout=self.rollout(3 if change=='partial' else 4,1 if change=='speed' else 0)
            if change=='ambiguity':rollout['ambiguous_intervals']=[0]
            with self.subTest(change=change),patch('nav.marble_dynamics.viability.predict_replies',return_value=rollout),patch('nav.marble_dynamics.viability.terminal_support',return_value=(True,True,0)):
                chosen,audit=ordinary_recovery(engine,dynamics,[0.]*9,replies=4,budget_s=1,landing_limit=0)
                self.assertIsNone(chosen);self.assertTrue(audit['candidates'])
                self.assertTrue(all(not c['stopping_verified'] for c in audit['candidates']))
                self.assertIsNone(engine.planning_deadline)

    def test_expired_parent_budget_restored_and_no_partial_plan_accepted(self):
        engine=SimpleNamespace(mesh=None,planning_deadline=0.)
        with patch('nav.marble_dynamics.viability.predict_replies') as predict:
            chosen,audit=ordinary_recovery(engine,{'native':{'state':{'position':[0,0,0]}}},[0.]*9,replies=4)
        predict.assert_not_called();self.assertIsNone(chosen)
        self.assertEqual(audit['stop'],'deadline');self.assertEqual(engine.planning_deadline,0.)

    def test_native_grounded_fixture_recovers_without_claiming_game_rule_safety(self):
        c=fixture('landing_bridge.json');mesh=CollisionMap(ML/'collision_maps'/(c['map']+'.npz'))
        engine=GeneralPhysics(mesh);landing=predict_replies(engine,c['dynamics'],c['actions'])
        before=resume_forecast(c['dynamics'],landing);original=copy.deepcopy(before)
        chosen,audit=ordinary_recovery(engine,before,[0.]*9,replies=40,budget_s=10,landing_limit=0)
        self.assertIsNotNone(chosen)
        self.assertTrue(chosen[0]['stopping_verified']);self.assertLess(chosen[0]['end_speed_u_per_s'],1)
        self.assertFalse(chosen[0]['game_rule_volumes_checked']);self.assertEqual(before,original)
        self.assertEqual(len(chosen[1]['proposed']),40)


class ContinuationRegressions(unittest.TestCase):
    def test_pickup_truncation_does_not_reuse_longer_forecast_bridge(self):
        c=fixture('landing_bridge.json')
        engine=GeneralPhysics(CollisionMap(ML/'collision_maps'/(c['map']+'.npz')))
        prefix=predict_replies(engine,c['dynamics'],c['actions'][:3])
        expected=resume_forecast(c['dynamics'],prefix)
        # two-level gem scoring slices states/controls/proposed but carries
        # other metadata from the original longer forecast.
        prefix['bridge_resume']=dict(reply_count=48,pending_move={'invalid':True},
            bridge_action=[99]*9,pow_yaw_hold=999,pow_yaw=999.)
        self.assertEqual(resume_forecast(c['dynamics'],prefix),expected)

    def test_enabled_assessment_refuses_the_supported_but_unstoppable_lip(self):
        from types import SimpleNamespace
        from nav.marble_dynamics.search import assess
        c=fixture('terminal_stopping_bridge.json')
        engine=GeneralPhysics(CollisionMap(ML/'collision_maps'/(c['map']+'.npz')))
        route=SimpleNamespace(distance=lambda p:float(np.linalg.norm(np.asarray(p)-c['goal'])))
        original=predict_replies(engine,c['dynamics'],c['actions'][:40])
        self.assertIsNone(assess(engine,route,c['dynamics'],c['goal'],copy.deepcopy(original),'lip')['rejected'])
        engine.terminal_stop_replies=48
        info=assess(engine,route,c['dynamics'],c['goal'],copy.deepcopy(original),'lip')
        self.assertEqual(info['rejected'],'no_stopping_continuation:no_supported_stopping_continuation')
        engine.terminal_stop_scope='pickup'
        scoped=assess(engine,route,c['dynamics'],c['goal'],copy.deepcopy(original),'lip')
        self.assertEqual(scoped['rejected'],info['rejected'])

    def test_pickup_scope_keeps_ordinary_progress_assessment_unchanged(self):
        from types import SimpleNamespace
        from nav.marble_dynamics.search import assess
        c=fixture('landing_bridge.json');engine=GeneralPhysics(CollisionMap(ML/'collision_maps'/(c['map']+'.npz')))
        landing=predict_replies(engine,c['dynamics'],c['actions'])
        dynamics=resume_forecast(c['dynamics'],landing)
        controls=[[0.]*9 for _ in range(16)]
        rollout=predict_replies(engine,dynamics,controls)
        goal=(np.asarray(rollout['states'][-1]['position'])+[100.,0.,0.]).tolist()
        route=SimpleNamespace(distance=lambda p:float(np.linalg.norm(np.asarray(p)-goal)))
        baseline=assess(engine,route,dynamics,goal,copy.deepcopy(rollout),'progress')
        self.assertIsNone(baseline['rejected']);self.assertIsNone(baseline['possible_pickup_interval'])
        engine.terminal_stop_replies=48;engine.terminal_stop_scope='pickup'
        checked=copy.deepcopy(rollout)
        self.assertEqual(assess(engine,route,dynamics,goal,checked,'progress'),baseline)
        self.assertEqual(checked,rollout)

    def test_enabled_assessment_retains_backup_and_native_pickup_commitment(self):
        from types import SimpleNamespace
        from nav.marble_dynamics.search import assess,PlanMemory
        c=fixture('landing_bridge.json');engine=GeneralPhysics(CollisionMap(ML/'collision_maps'/(c['map']+'.npz')))
        engine.terminal_stop_replies=48
        item=next(i for i in c['dynamics']['items'] if i['gem'] and i['value']==5)
        route=SimpleNamespace(distance=lambda p:float(np.linalg.norm(np.asarray(p)-item['position'])))
        rollout=predict_replies(engine,c['dynamics'],c['actions']);initial=copy.deepcopy(c['dynamics'])
        info=assess(engine,route,c['dynamics'],item['position'],rollout,'landing')
        self.assertIsNone(info['rejected']);self.assertTrue(info['terminal_stop_check']['valid'])
        self.assertEqual(len(rollout['proposed']),len(c['actions'])+48)
        self.assertEqual(c['dynamics'],initial)
        memory=PlanMemory();memory.accept(rollout,info['label'],item['position'],info)
        self.assertFalse(memory.observe_pickup(0,[]))
        self.assertFalse(memory.observe_pickup(5,[item]))
        self.assertTrue(memory.observe_pickup(5,[]));self.assertTrue(memory.landing_mode)
        self.assertEqual(memory.landing_goal,rollout['states'][-1]['position'])

    def test_split_forecasts_preserve_complete_queued_bridge_and_blast(self):
        cases=[(c['dynamics'],[reply(j) for j in c['joysticks']],'Concentric_Hunt')
            for c in fixture('live_bridge.json')['cases']]
        for name in ['blast_bridge.json','special_blast_bridge.json']:
            cases.extend((c['dynamics'],c['actions'],'Concentric_Hunt') for c in fixture(name)['cases'])
        c=fixture('landing_bridge.json');cases.append((c['dynamics'],c['actions'],c['map']))
        engines={}
        for dynamics,actions,name in cases:
            if name not in engines:engines[name]=GeneralPhysics(CollisionMap(ML/'collision_maps'/(name+'.npz')))
            engine=engines[name];original=copy.deepcopy(dynamics)
            full=predict_replies(engine,dynamics,actions)
            self.assertIsNone(full['failure']);self.assertEqual(full['ambiguous_intervals'],[])
            for split in sorted(set(i for i in [0,1,2,3,8,len(actions)//2] if i<len(actions))):
                with self.subTest(map=name,split=split,length=len(actions)):
                    prefix=predict_replies(engine,dynamics,actions[:split])
                    tail=predict_replies(engine,resume_forecast(dynamics,prefix),actions[split:])
                    self.assertEqual(full,concatenate(prefix,tail))
            self.assertEqual(dynamics,original)

    def test_native_supported_lip_is_not_a_viable_stopping_state(self):
        c=fixture('terminal_stopping_bridge.json')
        engine=GeneralPhysics(CollisionMap(ML/'collision_maps'/(c['map']+'.npz')))
        full=predict_replies(engine,c['dynamics'],c['actions'])
        maxima=np.max([errors(a,b) for a,b in zip(full['states'][1:],c['actual'])],axis=0)
        self.assertLess(maxima[0],1e-5);self.assertLess(maxima[1],1e-4);self.assertLess(maxima[2],1e-3)
        prefix=predict_replies(engine,c['dynamics'],c['actions'][:40])
        self.assertTrue(terminal_support(engine.mesh,prefix['states'][-4:])[1])
        check,joined=braking_viability(engine,c['dynamics'],prefix,c['goal'],12)
        self.assertFalse(check['valid'])
        self.assertEqual(check['rejected'],'no_supported_stopping_continuation')
        self.assertEqual(full,joined)

    def test_verified_landing_has_a_checked_stopping_backup(self):
        c=fixture('landing_bridge.json')
        engine=GeneralPhysics(CollisionMap(ML/'collision_maps'/(c['map']+'.npz')))
        result=predict_replies(engine,c['dynamics'],c['actions'])
        check,_=braking_viability(engine,c['dynamics'],result,c['selected']['landing_goal'],48)
        self.assertTrue(check['valid'],check)

    def test_native_pickup_revalidation_keeps_one_verified_backup(self):
        from types import SimpleNamespace
        from nav.marble_dynamics.search import assess,PlanMemory
        from nav.marble_dynamics.landing import retained_landing
        c=fixture('landing_bridge.json');engine=GeneralPhysics(CollisionMap(ML/'collision_maps'/(c['map']+'.npz')))
        engine.terminal_stop_replies=48;engine.terminal_stop_scope='pickup'
        item=next(i for i in c['dynamics']['items'] if i['gem'] and i['value']==5)
        route=SimpleNamespace(distance=lambda p:float(np.linalg.norm(np.asarray(p)-item['position'])))
        rollout=predict_replies(engine,c['dynamics'],c['actions'])
        info=assess(engine,route,c['dynamics'],item['position'],rollout,'native_landing')
        self.assertIsNone(info['rejected']);memory=PlanMemory()
        memory.accept(rollout,info['label'],item['position'],info)
        for state in c['actual'][:38]:memory.advance(state)
        self.assertTrue(memory.observe_pickup(5,c['pickup_snapshot']['items']))
        planned=len(memory.remaining);goal=list(memory.landing_goal)
        retained=retained_landing(engine,route,c['pickup_snapshot'],memory)
        self.assertIsNotNone(retained,memory.last_landing_rejection)
        self.assertEqual(len(retained[1]['proposed']),planned)
        self.assertEqual(len(memory.remaining),planned);self.assertEqual(memory.landing_goal,goal)

    def test_landing_alternatives_keep_verified_stop_instead_of_proposed_point(self):
        from types import SimpleNamespace
        from nav.marble_dynamics.search import assess
        from nav.marble_dynamics.landing import alternatives
        c=fixture('landing_bridge.json');engine=GeneralPhysics(CollisionMap(ML/'collision_maps'/(c['map']+'.npz')))
        engine.terminal_stop_replies=48;engine.terminal_stop_scope='pickup'
        item=next(i for i in c['dynamics']['items'] if i['gem'] and i['value']==5)
        route=SimpleNamespace(distance=lambda p:float(np.linalg.norm(np.asarray(p)-item['position'])))
        prefix=predict_replies(engine,c['dynamics'],c['actions'][:40])
        info=assess(engine,route,c['dynamics'],item['position'],prefix,'airborne_pickup')
        self.assertIsNotNone(info['rejected']);self.assertIsNotNone(info['possible_pickup_interval'])
        candidates=alternatives(engine,route,c['dynamics'],item['position'],[(info,prefix)],80,8)
        valid=[(i,r) for i,r in candidates if i['rejected'] is None]
        self.assertTrue(valid)
        for info,rollout in valid:
            self.assertEqual(info['landing_goal'],rollout['states'][-1]['position'])
            self.assertIn('proposed_landing_goal',info)

    def test_resume_refuses_uncertain_or_incomplete_forecasts(self):
        c=fixture('grazing_axis_bridge.json');e=c['separate_geometry_case']
        engine=GeneralPhysics(CollisionMap(ML/'collision_maps'/(c['map']+'.npz')))
        uncertain=predict_replies(engine,e['dynamics'],e['actions'])
        # The old native tangent fixture is now deterministic after the F32
        # correction. Fault-inject declared uncertainty to retain the refusal
        # contract independently of a particular physics discrepancy.
        uncertain['ambiguous_intervals']=[5]
        with self.assertRaisesRegex(ValueError,'contact-ambiguous'):resume_forecast(e['dynamics'],uncertain)
        complete=predict_replies(engine,c['dynamics'],c['actions'][:3]);complete['proposed'].pop()
        with self.assertRaisesRegex(ValueError,'Incomplete'):resume_forecast(c['dynamics'],complete)


if __name__=='__main__':unittest.main()
