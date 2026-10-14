"""Regressions against recorded native outcomes, not copied model formulas."""
import json
from pathlib import Path
import unittest

import hxDif
import numpy as np

from nav.collision.mesh import CollisionMap
from nav.marble_dynamics.physics import GeneralPhysics
from nav.marble_dynamics.evaluate import errors,continuous_prefix
from nav.marble_dynamics.record import ML
from nav.marble_dynamics.rollout import predict


class NativeRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture=json.loads(Path(__file__).with_name('fixtures').joinpath('native_regressions.json').read_text())
        cls.fixtures=fixture['samples'];cls.control_case=fixture['control_case'];cls.f32_contact_case=fixture['native_f32_contact_case']
        cls.engine=GeneralPhysics(CollisionMap(ML/'collision_maps/Concentric_Hunt.npz'))

    def test_overspeed_preserves_fractional_spin_target(self):
        row=self.fixtures[0]['row'];pred,_=self.engine.step(row['before'],row['move'],row['dt_ms'])
        self.assertLess(errors(pred,row['after'])[2],.0001)

    def test_local_order_at_floor_wall_corner(self):
        case=self.fixtures[1]['order_case']
        ambiguities=[]
        contacts=self.engine.contacts(np.asarray(case['position']),float(np.float32(case['radius'])),ambiguities)
        # Native F32 radius multiplication produces NaN separation at this
        # tangent. The formerly omitted contact must now match native directly.
        native=[c['normal'] for c in case['contacts']]
        alternate=[a['candidate_normal'] for a in ambiguities if 'candidate_normal' in a]
        self.assertEqual(len(contacts),len(native));self.assertEqual(alternate,[])
        np.testing.assert_allclose([c['normal'] for c in contacts],native,atol=.0001)
        row=self.fixtures[1]['row'];pred,_=self.engine.step(row['before'],row['move'],row['dt_ms'])
        p,v,w=errors(pred,row['after'])
        # Measured native comparison bounds prevent recurrence of the earlier
        # 0.05u / 5.4u/s / 16.7rad/s ordering failure. They do not certify landing.
        self.assertLess(p,.005);self.assertLess(v,.5);self.assertLess(w,2.)

    def test_native_f32_contact_radius_preserves_full_rebound_prefix(self):
        case=self.f32_contact_case;engine=GeneralPhysics(CollisionMap(ML/'collision_maps'/(case['map']+'.npz')))
        forecast=predict(engine,case['initial_state'],case['controls'],dt_ms=16,max_steps=16)
        self.assertIsNone(forecast['failure']);self.assertEqual(len(forecast['states']),17)
        for predicted,expected in zip(forecast['states'][1:],case['native_motion']):
            measured=np.asarray(expected);motion=np.concatenate([predicted[k] for k in ('position','velocity','spin')])
            for offset,bound in [(0,1e-6),(3,1e-4),(6,1e-4)]:
                self.assertLess(np.linalg.norm((motion-measured)[offset:offset+3]),bound)

    def test_native_effect_clock_before_and_after_expiry(self):
        for sample in self.fixtures[2:4]:
            row=sample['row'];pred,_=self.engine.step(row['before'],row['move'],row['dt_ms'])
            self.assertEqual(pred['effects'][5],row['after']['effects'][5])
        self.assertTrue(self.fixtures[2]['row']['after']['effects'][5][0])
        self.assertFalse(self.fixtures[3]['row']['after']['effects'][5][0])

    def test_missing_speed_contact_state_is_refused(self):
        row=self.fixtures[4]['row']
        with self.assertRaisesRegex(ValueError,'previous best-contact'):
            self.engine.step(row['before'],row['move'],row['dt_ms'])

    def test_candidate_clock_advances_without_future_state(self):
        rows=[sample['row'] for sample in self.fixtures[2:4]]
        result=predict(self.engine,rows[0]['before'],[r['move'] for r in rows])
        self.assertIsNone(result['failure'])
        for state,row in zip(result['states'][1:],rows):
            self.assertEqual(state['clock_ms'],row['sim_ms'])
            self.assertEqual(state['effects'][5],row['after']['effects'][5])
            self.assertLess(errors(state,row['after'])[0],.0001)
        self.assertEqual(result['next_clock_ms'],rows[-1]['sim_ms']+16)

    def test_absolute_camera_command_is_part_of_control(self):
        rows=self.control_case['rows']
        controls=[dict(move=r['move'],events=r['control_events']) for r in rows]
        predicted=predict(self.engine,rows[0]['before'],controls)
        missing=predict(self.engine,rows[0]['before'],[r['move'] for r in rows])
        self.assertIsNone(predicted['failure'])
        self.assertLess(errors(predicted['states'][-1],rows[-1]['after'])[0],.0001)
        self.assertGreater(errors(missing['states'][-1],rows[-1]['after'])[0],.001)

    def test_inventory_pickup_delimits_continuous_rollout(self):
        rows=[sample['row'] for sample in self.fixtures[5:7]]
        kept,removed=continuous_prefix({},rows)
        self.assertEqual(len(kept),1);self.assertEqual(removed,1)
        self.assertEqual(rows[0]['after']['held'],3);self.assertEqual(rows[1]['before']['held'],5)

    def test_native_collision_iteration_limit_discards_slice_remainder(self):
        case=json.loads(Path(__file__).with_name('fixtures').joinpath('kotm_iteration_limit.json').read_text())
        engine=GeneralPhysics(CollisionMap(ML/'collision_maps'/(case['map']+'.npz')))
        row=case['native_row'];pred,events=engine.step(row['before'],row['move'],row['dt_ms'])
        self.assertLess(errors(pred,row['after'])[0],.000001)
        self.assertLess(errors(pred,row['after'])[1],.00001)
        lost=[e['discarded_time_s'] for e in events if e['kind']=='native_solver_limit']
        self.assertEqual(len(lost),1)
        self.assertGreater(lost[0],.006)

    def test_distinct_ramp_support_tie_exposes_native_branch(self):
        case=json.loads(Path(__file__).with_name('fixtures').joinpath('kotm_support_tie.json').read_text())
        engine=GeneralPhysics(CollisionMap(ML/'collision_maps'/(case['map']+'.npz')))
        row=case['row']
        result=predict(engine,row['before'],[dict(move=row['move'],events=row['control_events'])],row['dt_ms'])
        tied=[e for e in result['events'][0] if e.get('support_force_tie')]
        self.assertTrue(tied)
        self.assertEqual(result['ambiguous_intervals'],[0])
        actual=np.asarray(row['after']['best_contact_normal'])
        self.assertTrue(any(np.linalg.norm(actual-n)<1e-7 for e in tied for n in e['candidate_normals']))
        # Distinct surfaces must be retained as alternatives; neither a fixed
        # native winner nor a looser spin-error tolerance masks this uncertainty.
        self.assertTrue(any(len(set(e['candidate_triangle_ids']))>1 for e in tied))

    def test_native_hull_bounds_enclose_source_points(self):
        sources={}
        for path in (ML/'collision_maps').glob('*.json'):
            for source in json.loads(path.read_text()).get('source_inventory',[]):
                if source['path'].lower().endswith('.dif'):sources[source['path']]=source
        self.assertEqual(len(sources),21)
        count=0
        for path in sources:
            interior=hxDif.Dif.Load(str(ML.parent/path)).interiors[0]
            for hull in interior.convexHulls:
                lo=np.asarray([hull.minX,hull.minY,hull.minZ]);hi=np.asarray([hull.maxX,hull.maxY,hull.maxZ])
                self.assertTrue(np.all(lo<=hi))
                points=np.asarray([[interior.points[i].x,interior.points[i].y,interior.points[i].z]
                                   for i in interior.hullIndices[hull.hullStart:hull.hullStart+hull.hullCount]])
                self.assertTrue(np.all(points>=lo)&np.all(points<=hi));count+=1
        self.assertEqual(count,5168)


class FrozenFixtureTransport(unittest.TestCase):
    def test_expired_deadline_does_not_send_a_fixture_command(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from nav.marble_dynamics.pairs import frozen_fixture_command
        conn=Mock();env=SimpleNamespace(conn=conn,buf=b'DEBUG|fixture_roundtrip|1\n')
        with self.assertRaises(TimeoutError):frozen_fixture_command(env,'FIXTURECHECK report.json',timeout_s=0)
        conn.sendall.assert_not_called()

    def test_pool_completion_requires_actual_contracts_and_declared_identities(self):
        import copy
        from nav.marble_dynamics.pairs import fixture_pool_completion
        good=dict(map='A',seed=7,complete=True,failure=None,passed=4,refusals=0,
            protected_unchanged=True,source_unchanged=True,checks=[dict(passed=True) for _ in range(4)])
        self.assertTrue(fixture_pool_completion([good],['A'],7)['complete'])
        for field,value in [('failure','source changed'),('source_unchanged',False),('refusals',1),('seed',8),('checks',[])]:
            bad=copy.deepcopy(good);bad[field]=value
            self.assertFalse(fixture_pool_completion([bad],['A'],7)['complete'],field)
        self.assertFalse(fixture_pool_completion([good,good],['A'],7)['complete'])

    def test_ack_preserves_pending_observation_and_does_not_call_control(self):
        import socket
        from types import SimpleNamespace
        from unittest.mock import Mock
        from nav.marble_dynamics.pairs import frozen_fixture_command
        left,right=socket.socketpair();pending=object()
        try:
            env=SimpleNamespace(conn=left,buf=b'DEBUG|fixture_roundtrip|1\n',msg=pending,
                control=Mock(side_effect=AssertionError('control must not advance physics')))
            ack=frozen_fixture_command(env,'FIXTURECHECK report.json')
            self.assertEqual(right.recv(100),b'FIXTURECHECK report.json\n')
            self.assertEqual(ack.fields,['fixture_roundtrip','1'])
            self.assertIs(env.msg,pending);env.control.assert_not_called();self.assertEqual(env.buf,b'')
        finally:left.close();right.close()

    def test_unexpected_observation_is_refused_without_sending_an_action(self):
        import socket
        from types import SimpleNamespace
        from nav.marble_dynamics.pairs import frozen_fixture_command
        left,right=socket.socketpair()
        try:
            env=SimpleNamespace(conn=left,buf=b'[]|0|0|1\n')
            with self.assertRaisesRegex(ValueError,'Unexpected frozen acknowledgement'):
                frozen_fixture_command(env,'FIXTURECHECK report.json')
            self.assertEqual(right.recv(100),b'FIXTURECHECK report.json\n')
            right.settimeout(.01)
            with self.assertRaises(socket.timeout):right.recv(100)
        finally:left.close();right.close()

    def test_missing_ack_times_out_instead_of_waiting_for_a_new_game(self):
        import socket
        from types import SimpleNamespace
        from nav.marble_dynamics.pairs import frozen_fixture_command
        left,right=socket.socketpair()
        try:
            env=SimpleNamespace(conn=left,buf=b'')
            with self.assertRaisesRegex(TimeoutError,'No frozen fixture acknowledgement'):
                frozen_fixture_command(env,'FIXTURECHECK report.json',timeout_s=.01)
        finally:left.close();right.close()


if __name__=='__main__':unittest.main()
