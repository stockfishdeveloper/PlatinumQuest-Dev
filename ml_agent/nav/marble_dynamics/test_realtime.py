"""Runtime ownership, checked-continuation and script pickup regressions."""
import copy
from collections import Counter
import pickle
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from nav.marble_dynamics.realtime import AheadPlanner, WorldManager
from nav.marble_dynamics.script_forecast import _collect
from nav.marble_dynamics.viability import JoinedSequence


class RuntimeContracts(unittest.TestCase):
    def test_benchmark_rescue_counter_preserves_protocol_acknowledgment(self):
        from tools.realtime_bench import BenchEnv
        from nav.env import HuntEnv
        env = object.__new__(BenchEnv)
        env.rescue_attempts = env.accepted_rescue_count = 0
        for accepted in (True, False, True, True):
            def acknowledge():
                env.rescue_accepted = accepted
                return accepted
            with patch.object(HuntEnv, 'quick_rescue', side_effect=acknowledge):
                self.assertIs(env.quick_rescue(), accepted)
            self.assertIs(env.rescue_accepted, accepted)
        self.assertEqual((env.rescue_attempts, env.accepted_rescue_count), (4, 3))

    def test_map_switch_closes_old_workers_before_starting_new(self):
        events = []
        def factory(name):
            events.append(('open', name))
            if name == 'broken':
                raise RuntimeError('load failed')
            return SimpleNamespace(name=name, close=lambda: events.append(('close', name)))
        manager = WorldManager(factory)
        first = manager.get('a')
        self.assertIs(manager.get('a'), first)
        manager.get('b')
        with self.assertRaises(RuntimeError):
            manager.get('broken')
        self.assertIsNone(manager.world)
        manager.close()
        self.assertEqual(events, [('open','a'), ('close','a'), ('open','b'), ('close','b'), ('open','broken')])

    def planner(self):
        state = dict(position=[0,0,0], velocity=[0,0,0], spin=[0,0,0], radius=.19,
            yaw=0., gravity_direction=[0,0,-1], held=0, effects=[], parameters={'mass':1}, mode=0, oob=False)
        p = object.__new__(AheadPlanner)
        bridge = dict(pending_move={'x':0.,'y':0.,'jump':False}, pow_yaw_hold=0, pow_yaw=0.)
        p.cv = threading.Condition(); p.checked = {7: ([1]*9, copy.deepcopy(state), [0]*9, copy.deepcopy(bridge))}
        p.latest = (7, SimpleNamespace(dynamics=dict(native={'state':state,'pending_move':bridge['pending_move']},
            bridge_action=[0]*9, pow_yaw_hold=0, pow_yaw=0.)), {})
        p.stats = Counter(); p.committed = {}; p.generation = 2; p.next = 7; p.reset = False
        return p, state

    def test_checked_continuation_invalidates_inflight_result(self):
        p, _ = self.planner()
        with p.cv:
            self.assertTrue(p._checked_reply(7))
        self.assertEqual(p.committed[7], [1]*9)
        self.assertEqual((p.next, p.generation, p.reset), (8, 3, 'checked_continuation'))

    def test_changed_physics_or_state_refuses_checked_continuation(self):
        changes = dict(position=[.01,0,0], radius=.66, held=2, mode=2, oob=True,
            yaw=.1, gravity_direction=[0,-1,0], parameters={'mass':2})
        for key, value in changes.items():
            with self.subTest(key=key):
                p, state = self.planner(); state[key] = value
                with p.cv:
                    self.assertFalse(p._checked_reply(7))
                self.assertFalse(p.committed)
                self.assertFalse(p.checked)

    def test_serialized_float_precision_does_not_cancel_checked_action(self):
        p,state=self.planner()
        state['parameters']['mass']=1.10000002
        p.checked[7][1]['parameters']['mass']=1.100000023841858
        p.latest[1].dynamics['bridge_action'][0]=.801234
        p.checked[7][2][0]=.8012344
        with p.cv:self.assertTrue(p._checked_reply(7))

    def test_nonfinite_or_missing_context_refuses_checked_action(self):
        p,state=self.planner();state['position'][0]=float('nan')
        with p.cv:self.assertFalse(p._checked_reply(7))
        p,state=self.planner();p.checked[7][1]['blast_context']={'value':.5,'special':False}
        with p.cv:self.assertFalse(p._checked_reply(7))
        p,state=self.planner();p.checked[7][1]['blast_context']={'value':.5,'special':False}
        p.latest[1].dynamics['blast']={'value':float('nan'),'special':False}
        with p.cv:self.assertFalse(p._checked_reply(7))
        p,state=self.planner();p.latest[1].dynamics['bridge_action'][0]=float('nan')
        with p.cv:self.assertFalse(p._checked_reply(7))

    def test_changed_blast_rules_refuse_checked_action(self):
        p,state=self.planner()
        p.checked[7][1]['blast_context']={'value':.5,'special':False,'power':10.}
        p.latest[1].dynamics['blast']={'value':.5,'special':False,'power':20.}
        with p.cv:self.assertFalse(p._checked_reply(7))

    def test_pending_native_move_or_aim_latch_change_refuses_continuation(self):
        p,state=self.planner();p.latest[1].dynamics['native']['pending_move']['jump']=True
        with p.cv:self.assertFalse(p._checked_reply(7))
        p,state=self.planner();p.latest[1].dynamics['pow_yaw_hold']=1
        with p.cv:self.assertFalse(p._checked_reply(7))

    def test_different_preceding_action_refuses_checked_continuation(self):
        p, _ = self.planner(); p.latest[1].dynamics['bridge_action'][4] = 1
        with p.cv:
            self.assertFalse(p._checked_reply(7))

    def test_lazy_join_preserves_slices_empty_parts_and_pickle(self):
        x = JoinedSequence([], [1,2]); x = JoinedSequence(x, [99,3,4], 1)
        y = pickle.loads(pickle.dumps(x))
        self.assertEqual(list(y), [1,2,3,4])
        self.assertEqual(y[::-1], [4,3,2,1]); self.assertEqual(y[-1], 4)
        with self.assertRaises(IndexError):
            y[4]


class PickupContracts(unittest.TestCase):
    def state(self):
        return dict(position=[0,0,0], radius=.19, held=1,
            inventory_context=dict(held_type=1), blast_context=dict(enabled=True,value=.2,special=False))

    def item(self, kind):
        return dict(gem=False, powerup=kind, bounds=[-.1,-.1,-.1,.1,.1,.1])

    def test_different_inventory_replaces_existing_powerup(self):
        state = self.state(); item = self.item(2)
        remaining, events = _collect(state, copy.deepcopy(state), [item], 3)
        self.assertEqual(state['held'], 2); self.assertEqual(state['inventory_context']['held_type'], 2)
        self.assertFalse(remaining); self.assertEqual(events[0]['interval'], 0)

    def test_same_inventory_pickup_remains_available(self):
        state = self.state(); item = self.item(1)
        remaining, events = _collect(state, copy.deepcopy(state), [item], 4)
        self.assertEqual(remaining, [item]); self.assertFalse(events)

    def test_blast_pickup_charges_without_replacing_inventory(self):
        state = self.state()
        remaining, _ = _collect(state, copy.deepcopy(state), [self.item(7)], 5)
        self.assertFalse(remaining); self.assertEqual(state['held'], 1)
        self.assertEqual(state['blast_context'], dict(enabled=True,value=1.,special=True))

    def test_held_type_can_become_eligible_after_other_pickup(self):
        state = self.state()
        with self.assertRaisesRegex(ValueError, 'query order'):
            _collect(state, copy.deepcopy(state), [self.item(1), self.item(2)], 1)

    def test_duplicate_powerup_identity_order_is_not_invented(self):
        state = self.state()
        with self.assertRaisesRegex(ValueError, 'query order'):
            _collect(state, copy.deepcopy(state), [self.item(2), self.item(2)], 1)
        remaining, events = _collect(state, copy.deepcopy(state), [self.item(1), self.item(1)], 1)
        self.assertEqual(len(remaining), 2)
        self.assertFalse(events)

    def test_ambiguous_replacement_order_is_not_invented(self):
        state = self.state()
        with self.assertRaisesRegex(ValueError, 'query order'):
            _collect(state, copy.deepcopy(state), [self.item(2), self.item(3)], 1)


class SequenceSearchContracts(unittest.TestCase):
    def test_useful_checked_plan_is_not_perturbed_in_failure_only_mode(self):
        from unittest.mock import patch
        from nav.marble_dynamics.phased_search import refine
        dynamics={'native':{'state':{'position':[0,0,0]}}}
        route=SimpleNamespace(distance=lambda p:10.-p[0])
        pairs=[({'rejected':None,'possible_pickup_interval':None,'cost':2.},
                {'states':[{'position':[0,0,0]},{'position':[2,0,0]}]})]
        with patch('nav.marble_dynamics.phased_search.predict_replies') as predict:
            self.assertEqual(refine(SimpleNamespace(),route,dynamics,[3,0,0],pairs),[])
        predict.assert_not_called()

    def test_perturbations_are_deterministic_legal_and_do_not_change_seed(self):
        from nav.marble_dynamics.phased_search import perturbations
        actions=[[1.,0.,0.,0.,int(i==4),0.,int(i==7),0.,int(i==9)] for i in range(20)]
        before=copy.deepcopy(actions);one=list(perturbations(actions));two=list(perturbations(actions))
        self.assertEqual(one,two);self.assertEqual(actions,before);self.assertEqual(len(one),12)
        self.assertTrue(any(label.startswith('jump') for label,_ in one))
        self.assertTrue(any(label.startswith('use') for label,_ in one))
        self.assertTrue(any(label.startswith('blast') for label,_ in one))
        for label,rows in one:
            self.assertEqual(len(rows),len(actions))
            for row in rows:
                self.assertTrue(all(0.<=value<=1. for value in row[:4]))
                self.assertTrue(all(row[k] in (0,1) for k in (4,6,8)))


if __name__ == '__main__':
    unittest.main()
