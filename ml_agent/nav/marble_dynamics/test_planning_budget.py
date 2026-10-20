"""A search deadline aborts before proposing/executing an unfinished action."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
from unittest.mock import Mock,patch

from nav.marble_dynamics.bridge import predict_replies
from nav.marble_dynamics.bridge_small_step import predict_small_step
from nav.marble_dynamics.planning_budget import PlanningBudgetExceeded
from nav.marble_dynamics.search import search


class PlanningBudgetRegressions(unittest.TestCase):
    def setUp(self):
        # These tests isolate the primary search's deadline. Optional secondary
        # searches and braking backups have separate contracts and budgets.
        for name,value in [('EXTRA_JUMP_TIMES',()),('BRAKING_BACKUP',False)]:
            patched=patch('nav.marble_dynamics.search.'+name,value)
            patched.start();self.addCleanup(patched.stop)

    def route(self):
        route=Mock();route.distance.return_value=10.
        return route

    def test_learned_first_order_preserves_checks_and_legacy_order(self):
        dynamics=json.loads((Path(__file__).with_name('fixtures')/'live_bridge.json').read_text())['cases'][0]['dynamics']
        future=dict(states=[{},{}],controls=[{}],proposed=[[0]*9],requested_replies=1,
                    stopped_on_ambiguity=False)
        for first,rejected in [(False,None),(True,None),(True,'contact_ambiguity')]:
            engine=SimpleNamespace(use_completed_candidates_on_timeout=True,navigator_first=first)
            label='frozen_navigator' if first else 'route'
            with self.subTest(first=first,rejected=rejected), \
                 patch('nav.marble_dynamics.search.visible_gems',return_value=[]), \
                 patch('nav.marble_dynamics.search.blast_available',return_value=False), \
                 patch('nav.marble_dynamics.search.predict_replies',side_effect=[future,PlanningBudgetExceeded('deadline')]), \
                 patch('nav.marble_dynamics.search.assess',return_value=dict(label=label,cost=1.,rejected=rejected)) as assess:
                chosen,pairs=search(engine,self.route(),dynamics,np.zeros(61,np.float32),[1,2,3],Mock())
            self.assertEqual(assess.call_args.args[5],label)
            self.assertEqual(len(pairs),1)
            self.assertIsNone(chosen) if rejected else self.assertEqual(chosen[0]['label'],label)

    def test_continuous_search_keeps_only_completed_checked_candidates(self):
        dynamics=json.loads((Path(__file__).with_name('fixtures')/'live_bridge.json').read_text())['cases'][0]['dynamics']
        for enabled,rejected in [(False,None),(True,None),(True,'contact_ambiguity')]:
            future=dict(states=[{},{}],controls=[{}],proposed=[[0]*9],requested_replies=1,
                        stopped_on_ambiguity=False)
            assessment=dict(label='route',cost=1.,rejected=rejected)
            engine=SimpleNamespace(use_completed_candidates_on_timeout=enabled)
            navigator=Mock();route=self.route()
            with self.subTest(enabled=enabled,rejected=rejected), \
                 patch('nav.marble_dynamics.search.visible_gems',return_value=[]), \
                 patch('nav.marble_dynamics.search.blast_available',return_value=False), \
                 patch('nav.marble_dynamics.search.predict_replies',side_effect=[future,PlanningBudgetExceeded('deadline')]) as predict, \
                 patch('nav.marble_dynamics.search.assess',return_value=assessment) as assess:
                if not enabled:
                    with self.assertRaises(PlanningBudgetExceeded):
                        search(engine,route,dynamics,np.zeros(61,np.float32),[1,2,3],navigator)
                else:
                    chosen,pairs=search(engine,route,dynamics,np.zeros(61,np.float32),[1,2,3],navigator,warm=[[1]*9])
                    self.assertEqual(len(pairs),1)
                    self.assertIs(pairs[0][1],future)
                    self.assertEqual(future['proposed'],[[0]*9])
                    self.assertIsNone(chosen) if rejected else self.assertIs(chosen[1],future)
                    self.assertTrue(pairs[0][0]['search_stopped_at_deadline'])
                self.assertEqual(predict.call_count,2)
                self.assertEqual(assess.call_count,1)

    def test_deadline_before_any_completed_candidate_cannot_create_action(self):
        dynamics=json.loads((Path(__file__).with_name('fixtures')/'live_bridge.json').read_text())['cases'][0]['dynamics']
        engine=SimpleNamespace(use_completed_candidates_on_timeout=True)
        with patch('nav.marble_dynamics.search.visible_gems',return_value=[]), \
             patch('nav.marble_dynamics.search.blast_available',return_value=False), \
             patch('nav.marble_dynamics.search.predict_replies',side_effect=PlanningBudgetExceeded('deadline')):
            chosen,pairs=search(engine,self.route(),dynamics,np.zeros(61,np.float32),[1,2,3],Mock())
        self.assertIsNone(chosen);self.assertEqual(pairs,[])

    def test_expired_deadline_never_steps_physics_or_consumes_action_callback(self):
        dynamics=json.loads((Path(__file__).with_name('fixtures')/'live_bridge.json').read_text())['cases'][0]['dynamics']
        expected=copy.deepcopy(dynamics)
        for cadence in [64,16]:
            engine=SimpleNamespace(bridge_physics_ms=cadence,planning_deadline=-1.,step=Mock())
            actions=Mock(steps=1)
            with self.assertRaises(PlanningBudgetExceeded):predict_replies(engine,dynamics,actions)
            self.assertEqual(engine.step.call_count,0);self.assertEqual(actions.call_count,0)
            self.assertEqual(dynamics,expected)
        engine=SimpleNamespace(planning_deadline=-1.,step=Mock());actions=Mock(steps=1)
        with self.assertRaises(PlanningBudgetExceeded):predict_small_step(engine,dynamics,actions)
        self.assertEqual(engine.step.call_count,0);self.assertEqual(actions.call_count,0)


if __name__=='__main__':unittest.main()
