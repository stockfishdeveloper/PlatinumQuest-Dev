"""Regressions for delayed maneuvers and isolation of hypothetical policy state."""
import copy
import json
from pathlib import Path
import unittest

import numpy as np

from nav.marble_dynamics.bridge import predict_replies,reply
from nav.marble_dynamics.record import ML
from nav.marble_dynamics.physics import GeneralPhysics
from nav.collision.mesh import CollisionMap
from nav.marble_dynamics.search import PlanMemory,SequenceProposal,retained_candidate,terminal_support,BrakeContinuation,assess
from nav.marble_dynamics.navigator import FrozenNavigator,NavigatorProposal,project_observation
from nav.terrain import TerrainMap,TerrainGrid
from nav.routing import RoutePlanner
from nav.protocol import RAW_DIM,RAW_GEMS


class HelicopterGain(unittest.TestCase):
    """HELI_MIN_GAIN_S (2026-10-29): a held Helicopter is used only when it arrives at the gem 3 s before rolling would."""
    def plan(self,label,pickup,end,used,replies=48,speed=11.):
        info=dict(label=label,rejected=None,possible_pickup_interval=pickup,inventory_consumed=1 if used else 0)
        return info,dict(states=[dict(position=[0.,0.,0.],velocity=[0.,0.,0.])]*replies+[dict(position=end,velocity=[speed,0.,0.])])

    def apply(self,pairs,gem_x=100.,routed=True):
        from types import SimpleNamespace
        from nav.marble_dynamics.search import apply_heli_gain
        # a straight walking route along x to the gem (or none at all)
        route=SimpleNamespace(navigate=lambda a,b:None,distance=lambda p:gem_x-p[0] if routed else float('inf'))
        apply_heli_gain(pairs,route,dict(native=dict(state=dict(position=[0.,0.,0.]))),np.asarray([gem_x,0.,0.]),3.)

    def test_recorded_cube_isle_use_that_saves_nothing_is_rejected(self):
        # real-time decision 462: the Helicopter plan collects at reply 83 (5.3 s); rolling ends 27 u short after 48 replies
        heli=self.plan('use_7_jump_-1',83,[100.,0.,0.],True,96);roll=self.plan('route',None,[73.,0.,0.],False,speed=14.)
        self.apply([heli,roll])
        self.assertEqual(heli[0]['rejected'],'heli_no_gain');self.assertIsNone(roll[0]['rejected'])
        self.assertAlmostEqual(heli[0]['heli_rolling_s'],round(48*.064+27./14.,2))

    def test_a_use_that_saves_three_seconds_stays(self):
        heli=self.plan('use_0_jump_-1',30,[100.,0.,0.],True,96);roll=self.plan('route',None,[40.,0.,0.],False)
        self.apply([heli,roll])           # rolling: 3.07 s + 60 u at 11 u/s = 8.5 s; the Helicopter 1.9 s
        self.assertIsNone(heli[0]['rejected'])

    def test_far_gem_reached_after_the_forecast_still_counts(self):
        # the operator's big map: the Helicopter forecast ends 10 u short (90 % of the way) at 20 u/s, rolling 150 u short
        heli=self.plan('use_0_jump_-1',None,[190.,0.,0.],True,96,speed=20.);roll=self.plan('route',None,[50.,0.,0.],False,speed=12.)
        self.apply([heli,roll],gem_x=200.)
        self.assertIsNone(heli[0]['rejected'])
        self.assertAlmostEqual(heli[0]['heli_arrival_s'],round(96*.064+10./20.,2))
        self.assertGreater(heli[0]['heli_gain_s'],8.)

    def test_no_rolling_route_leaves_the_use_allowed(self):
        heli=self.plan('use_0_jump_-1',None,[90.,0.,0.],True,96);roll=self.plan('route',None,[3.,0.,0.],False)
        self.apply([heli,roll],routed=False)   # the Helicopter is judged on the straight line
        self.assertIsNone(heli[0]['rejected'])

    def test_a_rolling_pickup_is_the_reference(self):
        heli=self.plan('use_0_jump_-1',20,[100.,0.,0.],True,96);roll=self.plan('route',40,[100.,0.,0.],False)
        self.apply([heli,roll])           # 1.3 s against rolling's 2.6 s: 1.3 s saved, under 3
        self.assertEqual(heli[0]['rejected'],'heli_no_gain')

    def test_a_slower_unfinished_use_is_rejected(self):
        heli=self.plan('use_3_jump_-1',None,[80.,0.,0.],True,96);roll=self.plan('route',None,[73.,0.,0.],False,speed=14.)
        self.apply([heli,roll])           # 6.1 s + 20 u at 11 u/s = 8.0 s against rolling's 5.0 s
        self.assertEqual(heli[0]['rejected'],'heli_no_gain')


class RollLaunchOptions(unittest.TestCase):
    def test_short_landings_preserve_the_ordinary_rolling_launch(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from nav.marble_dynamics import search as S
        route=SimpleNamespace();gem=np.asarray([40.,0.,0.])
        with patch.multiple(S,POW_LAUNCH_AT=(0,9),LAND_SHORT_SHARES=(.4,1.),LAND_SHORT=False):
            ordinary=dict(S.roll_launch_specs(route,gem,48))
            with patch.object(S,'LAND_SHORT',True):augmented=dict(S.roll_launch_specs(route,gem,48))
        self.assertEqual(len(ordinary),2)
        self.assertEqual(len(augmented),6)
        for label,plan in ordinary.items():
            kept=augmented[label]
            self.assertEqual((kept.use,kept.jump,kept.land_short),(plan.use,-1,0.))
            self.assertEqual(kept.direct_from,plan.direct_from)
            self.assertEqual(kept.lead_k,plan.lead_k)
        self.assertEqual({p.land_short for p in augmented.values()},{0.,.4,1.})


class PowerupTimeGain(unittest.TestCase):
    def pair(self,label,pickup,cost,consumed=0,rejected=None,next_pickup=None):
        return (dict(label=label,possible_pickup_interval=pickup,cost=cost,
                     inventory_consumed=consumed,rejected=rejected),
                dict(states=dict(next_pickup=next_pickup)))

    def apply(self,pairs,next_gem=False,margin=1.):
        from types import SimpleNamespace
        from unittest.mock import patch
        from nav.marble_dynamics.search import apply_use_gain,goal_key
        gem=np.asarray([0.,0.,0.]);nxt=[7.,0.,0.]
        engine=SimpleNamespace(next_for={goal_key(gem):nxt} if next_gem else {})
        dynamics=dict(items=[dict(id=2,gem=True,position=nxt)] if next_gem else [])
        with patch('nav.marble_dynamics.search._gem_demo.possible_pickup',
                   side_effect=lambda states,*a:states['next_pickup']):
            apply_use_gain(pairs,engine,dynamics,gem,margin)

    def test_recorded_fractured_use_loses_despite_better_contact_cost(self):
        use=self.pair('direct_use_0_jump_-1_ballistic',34,-70.677426,1)
        ordinary=self.pair('direct_jump_9_ballistic',18,-.603838)
        navigator=self.pair('frozen_navigator_landing',47,-37.945459)
        self.apply([use,ordinary,navigator])
        self.assertEqual(use[0]['rejected'],'use_no_time_gain')
        self.assertAlmostEqual(use[0]['use_gain_s'],-1.024)
        self.assertIsNone(ordinary[0]['rejected'])

    def test_one_reply_saving_does_not_spend_inventory(self):
        use=self.pair('use',19,-1000.,1);plain=self.pair('drive',20,0.)
        self.apply([use,plain])
        self.assertEqual(use[0]['rejected'],'use_no_time_gain')

    def test_real_time_saving_can_justify_spending(self):
        use=self.pair('use',14,100.,1);plain=self.pair('drive',30,-100.)
        self.apply([use,plain])
        self.assertIsNone(use[0]['rejected'])
        self.assertAlmostEqual(use[0]['use_gain_s'],1.024)

    def test_no_ordinary_pickup_does_not_block_reachability_or_travel(self):
        for pickup in (None,20):
            use=self.pair('use',pickup,-100.,1);plain=self.pair('drive',None,0.)
            self.apply([use,plain])
            self.assertIsNone(use[0]['rejected'])

    def test_rejected_ordinary_pickup_is_not_a_counterfactual(self):
        use=self.pair('use',20,0.,1);unsafe=self.pair('drive',1,-100.,rejected='backup_fall')
        self.apply([use,unsafe])
        self.assertIsNone(use[0]['rejected'])

    def test_inventory_consumption_is_detected_without_a_use_label(self):
        use=self.pair('remaining_previous_plan',20,-100.,1)
        self.apply([use,self.pair('drive',10,0.)])
        self.assertEqual(use[0]['rejected'],'use_no_time_gain')

    def test_unused_inventory_and_blast_are_not_mistaken_for_consumption(self):
        blast=self.pair('blast',20,-100.);blast[0]['resources_consumed']=1
        self.apply([blast,self.pair('use_attempt_without_consumption',30,0.)])
        self.assertIsNone(blast[0]['rejected'])

    def test_first_gem_saving_cannot_hide_slower_pair_completion(self):
        use=self.pair('use',1,-100.,1,next_pickup=60)
        plain=self.pair('drive',20,0.,next_pickup=40)
        self.apply([use,plain],next_gem=True)
        self.assertEqual(use[0]['rejected'],'use_no_time_gain')
        self.assertEqual(use[0]['use_gain_scope'],'target_and_next')

    def test_faster_pair_can_justify_slower_first_pickup(self):
        use=self.pair('use',20,-100.,1,next_pickup=30)
        plain=self.pair('drive',10,0.,next_pickup=50)
        self.apply([use,plain],next_gem=True)
        self.assertIsNone(use[0]['rejected'])
        self.assertAlmostEqual(use[0]['use_gain_s'],1.28)

    def test_missing_pair_completion_cannot_beat_a_proven_ordinary_pair(self):
        use=self.pair('use',1,-100.,1)
        plain=self.pair('drive',20,0.,next_pickup=40)
        self.apply([use,plain],next_gem=True)
        self.assertEqual(use[0]['rejected'],'use_completion_unproven')

    def test_no_pickup_use_does_not_displace_a_proven_ordinary_pickup(self):
        use=self.pair('use',None,-100.,1)
        self.apply([use,self.pair('drive',20,0.)])
        self.assertEqual(use[0]['rejected'],'use_completion_unproven')

    def test_zero_margin_explicitly_disables_the_gate(self):
        use=self.pair('use',30,-100.,1)
        self.apply([use,self.pair('drive',1,0.)],margin=0.)
        self.assertIsNone(use[0]['rejected'])


class CheckedPowerupDetours(unittest.TestCase):
    def test_distance_fallback_does_not_prove_connectivity_to_a_pickup_floor(self):
        from types import SimpleNamespace
        from nav.marble_dynamics.reliability_eval import detour_route_proven
        terrain=SimpleNamespace(cell_of=lambda x,y:(0,0),
            _route_node=lambda *a:(0,0,0),_nearest_walkable=lambda *a,**kw:(0,0,0),
            walk_z=np.asarray([[[0.]]]))
        route=SimpleNamespace(terrain=terrain,field=np.asarray([[[np.inf]]]),distance=lambda p:5.)
        item=dict(position=[0.,0.,.2],bounds=[-.2,-.2,0.,.2,.2,1.])
        dynamics=dict(native=dict(state=dict(position=[0.,0.,.2],radius=.2)))
        self.assertFalse(detour_route_proven(route,item,dynamics))
        route.field[0,0,0]=32.
        self.assertTrue(detour_route_proven(route,item,dynamics))
        terrain.walk_z[0,0,0]=14.
        self.assertFalse(detour_route_proven(route,item,dynamics))

    def test_unreachable_cheapest_item_does_not_hide_a_reachable_alternative(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from nav.marble_dynamics import reliability_eval as RE
        position=[0.,0.,10.];gem=np.asarray([10.,0.,10.,1.,0.])
        items=[dict(id=1,gem=False,powerup=1,position=[2.,0.,-5.]),
               dict(id=2,gem=False,powerup=2,position=[3.,0.,10.])]
        dynamics=dict(native=dict(state=dict(position=position,held=0)),items=items)
        route=SimpleNamespace()
        route.navigate=lambda p,g:setattr(route,'goal',np.asarray(g))
        route.distance=lambda p:float(np.linalg.norm(np.asarray(p)[:2]-route.goal[:2]))
        engine=SimpleNamespace(opportunistic_powerups=4.,checked_powerup_detours=True,
                               next_for={},crossings=None)
        calls=[]
        def fake_search(e,r,d,raw,g,nav,horizon,**kwargs):
            item=kwargs['target_item'];calls.append(item['id'])
            info=dict(label='item_'+str(item['id']),rejected=None,cost=-100.,
                      possible_pickup_interval=20 if item['id']==2 else None)
            roll=dict(states=[dict(position=position),dict(position=[2.,0.,10.])])
            return (info,roll),[(info,roll)]
        with patch.object(RE,'search',side_effect=fake_search):
            chosen,_,_=RE.pick(engine,route,dynamics,None,[gem],gem,None,48,PlanMemory())
        self.assertEqual(calls,[1,2])
        self.assertEqual(chosen[0]['item_id'],2)

    def test_optional_detour_needs_a_reachable_pickup_not_horizontal_progress(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from nav.marble_dynamics import reliability_eval as RE

        position=[0.,0.,10.]
        item=dict(id=12,gem=False,powerup=1,position=[2.,0.,-5.])
        gem=np.asarray([10.,0.,10.,1.,0.])
        dynamics=dict(native=dict(state=dict(position=position,held=0)),items=[item])

        for checked,pickup,expected in [(False,None,'item'),(True,None,'gem'),(True,20,'item')]:
            with self.subTest(checked=checked,pickup=pickup):
                route=SimpleNamespace()
                route.navigate=lambda p,g:setattr(route,'goal',np.asarray(g))
                # Deliberately reproduce a route estimate blind to the floor.
                route.distance=lambda p:float(np.linalg.norm(np.asarray(p)[:2]-route.goal[:2]))
                engine=SimpleNamespace(opportunistic_powerups=4.,checked_powerup_detours=checked,
                                       next_for={},crossings=None)
                def fake_search(e,r,d,raw,g,nav,horizon,**kwargs):
                    is_item=kwargs.get('target_item') is not None
                    info=dict(label='item' if is_item else 'gem',rejected=None,cost=-100.,
                              possible_pickup_interval=pickup if is_item else 10)
                    roll=dict(states=[dict(position=position),dict(position=[2.,0.,10.])])
                    return (info,roll),[(info,roll)]
                with patch.object(RE,'search',side_effect=fake_search):
                    chosen,_,_=RE.pick(engine,route,dynamics,None,[gem],gem,None,48,PlanMemory())
                self.assertEqual(chosen[0]['label'],expected)


class RetainedBackupRisk(unittest.TestCase):
    def test_failed_backup_remains_penalized_and_unknown_cannot_clear_it(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        for verdict in [(False,dict(supported=False)),(True,dict(skipped='unresumable')),
                        (True,dict(ambiguous=True))]:
            memory=PlanMemory()
            memory.accept(dict(proposed=[[0.]*9 for _ in range(12)]),'route',[10.,0.,0.],
                          dict(no_braking_backup=True))
            route=SimpleNamespace(navigate=lambda p,g:None,distance=lambda p:10.-p[0])
            dyn=dict(native=dict(state=dict(position=[0.,0.,0.])))
            forecast=dict(states=[dict(position=[8.,0.,0.])])
            with self.subTest(verdict=verdict), \
                    patch('nav.marble_dynamics.search.predict_replies',return_value=forecast), \
                    patch('nav.marble_dynamics.search.assess',return_value=dict(cost=-100.,rejected=None,possible_pickup_interval=5)), \
                    patch('nav.marble_dynamics.search.braking_backup',return_value=verdict):
                info,_=retained_candidate(SimpleNamespace(recheck_risky_plans=True),route,dyn,
                                           np.asarray([10.,0.,0.]),memory)
            self.assertEqual(info['cost'],50.)
            self.assertTrue(memory.backup_risk)
            self.assertTrue(info['retained_backup_rechecked'])
            self.assertEqual(memory.executed,0)

    def test_safe_recheck_clears_risk_and_ordinary_plans_skip_extra_work(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from nav.marble_dynamics.search import recheck_retained_backup
        memory=PlanMemory();memory.backup_risk=True;info=dict(cost=-10.)
        with patch('nav.marble_dynamics.search.braking_backup',return_value=(True,dict(supported=True))) as check:
            recheck_retained_backup(SimpleNamespace(recheck_risky_plans=True),{},[0,0,0],memory,info,{})
            self.assertFalse(memory.backup_risk)
            self.assertEqual(info['cost'],-10.)
            recheck_retained_backup(SimpleNamespace(recheck_risky_plans=True),{},[0,0,0],memory,{}, {})
            check.assert_called_once()
        memory.backup_risk=True
        with patch('nav.marble_dynamics.search.braking_backup') as check:
            recheck_retained_backup(SimpleNamespace(recheck_risky_plans=False),{},[0,0,0],memory,{}, {})
            check.assert_not_called()
        memory.clear();self.assertFalse(memory.backup_risk)


class FastControllerSerialization(unittest.TestCase):
    def test_fast_speed_and_floor_survive_native_serialization(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from nav.native.forecast import seq_array,Q_SIZE
        route=SimpleNamespace(key=(10.,0.,0.),waypoint=None)
        with patch('nav.marble_dynamics.search.FAST_MIN_SPEED',7.5):
            fast=seq_array(SequenceProposal(route,np.asarray([10.,0.,0.]),48,fast=20.))
            ordinary=seq_array(SequenceProposal(route,np.asarray([10.,0.,0.]),48))
        self.assertEqual(len(fast),Q_SIZE)
        self.assertEqual(tuple(fast[49:51]),(20.,7.5))
        self.assertEqual(ordinary[49],0.)
        np.testing.assert_array_equal(fast[:49],ordinary[:49])


class TickPickupRegressions(unittest.TestCase):
    def proposal(self):
        from types import SimpleNamespace
        terrain=SimpleNamespace()
        route=SimpleNamespace(terrain=terrain,navigate=lambda p,g:(g,False),distance=lambda p:2.)
        result=SequenceProposal(route,np.asarray([0.,0.,0.]),8,through=(route,np.asarray([2.,0.,0.]),2.))
        result.box=[-.05,-.05,-.05,.05,.05,.05]
        return result

    def test_macro_chord_must_not_count_a_pickup_between_native_ticks(self):
        p=self.proposal()
        ticks=[dict(position=point,radius=.1) for point in ([-1,0,0],[-1,1,0],[0,1,0],[1,1,0],[1,0,0])]
        p.observe_ticks(ticks)
        self.assertFalse(p.passed)
        p.observe_ticks([dict(position=[-1,0,0],radius=.1),dict(position=[1,0,0],radius=.1)])
        self.assertTrue(p.passed)

    def test_initial_overlap_waits_for_an_actual_native_tick(self):
        from unittest.mock import patch
        p=self.proposal();state=dict(position=[0.,0.,0.],velocity=[0.,0.,0.],radius=.1)
        with patch('nav.marble_dynamics.search.leg_top',return_value=None):
            p(state,0)
        self.assertFalse(p.passed)
        p.observe_ticks([state,state]);self.assertTrue(p.passed)

    def test_larger_radius_reaches_items_the_normal_marble_misses(self):
        p=self.proposal()
        ticks=[dict(position=[.7,0,0],radius=.19)]*2
        p.observe_ticks(ticks);self.assertFalse(p.passed)
        p.observe_ticks([dict(position=[.7,0,0],radius=.6666)]*2)
        self.assertTrue(p.passed)


class SearchRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case=json.loads(Path(__file__).with_name('fixtures').joinpath('live_bridge.json').read_text())['cases'][0]
        cls.engine=GeneralPhysics(CollisionMap(ML/'collision_maps/Concentric_Hunt.npz'))

    def test_replanning_keeps_delayed_jump_at_original_native_frame(self):
        actions=[reply([0.,0.,0.,0.,int(i==5),0.]) for i in range(12)]
        memory=PlanMemory();memory.accept({'proposed':actions},'jump_5',[0,0,0])
        emitted=[]
        for _ in range(4):emitted.append(memory.remaining[0]);memory.advance()
        self.assertTrue(memory.locked)
        for _ in range(4):emitted.append(memory.remaining[0]);memory.advance()
        self.assertFalse(memory.locked)
        self.assertEqual([i for i,a in enumerate(emitted) if a[4]],[5])
        forecast=predict_replies(self.engine,self.case['dynamics'],emitted)
        self.assertIsNone(forecast['failure'])
        self.assertTrue(forecast['controls'][7]['move']['jump'])
        memory.clear();self.assertFalse(memory.remaining)

    def test_forecast_item_positions_are_world_fixed_without_invented_pickups(self):
        raw=np.full(RAW_DIM,-999.,dtype=np.float32);raw[:3]=[1,2,3]
        raw[RAW_GEMS.start:RAW_GEMS.start+5]=[10,20,30,2,37]
        state=copy.deepcopy(self.case['dynamics']['native']['state']);state['position']=[4,6,8]
        projected=project_observation(raw,raw[:3],state,1)
        np.testing.assert_array_equal(raw[RAW_GEMS.start:RAW_GEMS.start+3],[10,20,30])
        np.testing.assert_array_equal(projected[RAW_GEMS.start:RAW_GEMS.start+3],[7,16,25])
        self.assertEqual(projected[RAW_GEMS.start+3],2)

    def test_stale_bridge_reply_is_detected_instead_of_blamed_on_physics(self):
        from nav.marble_dynamics.reliability_eval import pending_reply_matches
        action=reply([.184371,0.,1.187858,0.,0,-2.20221])
        observed=[.184371,0.,1.18786,0.,0,-2.20221,0.,0.,0.]
        self.assertTrue(pending_reply_matches(observed,action))
        observed[5]=-1.55782
        self.assertFalse(pending_reply_matches(observed,action))

    def test_reuse_does_not_follow_changed_or_exhausted_target(self):
        memory=PlanMemory();actions=[reply([0,0,0,0,0,0]) for _ in range(12)]
        memory.accept(dict(proposed=actions),'coast',[0,0,0])
        self.assertIsNone(retained_candidate(None,None,None,np.asarray([4,0,0]),memory))
        for _ in range(5):memory.advance()
        self.assertIsNone(retained_candidate(None,None,None,np.asarray([0,0,0]),memory))
        self.assertEqual(memory.executed,5)

    def test_sloped_supported_continuation_distinguishes_rolling_and_rebound(self):
        from nav.collision.test_queries import fixture
        normal=np.asarray([1.,0.,2.])/np.sqrt(5.)
        mesh=fixture([[[-20,-20,10],[20,-20,-10],[0,20,0]]],[normal])
        tail=[dict(position=[i*.1,0,.2/normal[2]-i*.05],radius=.2,velocity=[6.,-4.,-3.]) for i in range(4)]
        supported,stable,speed=terminal_support(mesh,tail)
        self.assertTrue(supported);self.assertTrue(stable);self.assertLess(speed,1e-12)
        rebound=copy.deepcopy(tail)
        for state in rebound:state['velocity']=(np.asarray(state['velocity'])+normal*3.).tolist()
        self.assertEqual(terminal_support(mesh,rebound)[:2],(True,False))
        fast=copy.deepcopy(tail)
        for state in fast:state['velocity']=[40.,0.,-20.]
        self.assertEqual(terminal_support(mesh,fast)[:2],(True,False))
        detached=copy.deepcopy(tail)
        for state in detached:state['position'][2]+=.5
        self.assertEqual(terminal_support(mesh,detached)[:2],(False,False))
        self.assertFalse(terminal_support(mesh,tail[:2])[1])

    def test_braking_extension_keeps_prefix_and_does_not_repeat_jump(self):
        actions=[reply([.5,0,0,0,int(i==5),0]) for i in range(12)]
        before=copy.deepcopy(actions)
        base=predict_replies(self.engine,self.case['dynamics'],actions)
        extended=predict_replies(self.engine,self.case['dynamics'],BrakeContinuation(actions,[0,0,0],20))
        self.assertEqual(actions,before)
        self.assertIsNone(extended['failure'])
        self.assertEqual(extended['proposed'][:12],actions)
        self.assertEqual(extended['controls'][:12],base['controls'])
        self.assertEqual(extended['states'][:13],base['states'])
        self.assertEqual(len(extended['proposed']),32)
        self.assertTrue(all(not a[4] and not a[6] and not a[8] for a in extended['proposed'][12:]))

    def test_ground_proposals_detect_walls_and_low_ceilings_above_a_known_floor(self):
        from nav.collision.test_queries import fixture
        from nav.marble_dynamics.geometry_route import GroundClearance
        from nav.routing import line_clear
        from types import SimpleNamespace
        wall=fixture([[[0,-10,0],[0,10,0],[0,0,4]],
                      [[-10,-10,0],[10,-10,0],[0,10,0]]],[[-1,0,0],[0,0,1]])
        check=GroundClearance(wall)
        self.assertFalse(check.segment([-1,0,.2],[1,0,.2]))
        self.assertFalse(check.segment([1,0,.2],[-1,0,.2]))
        self.assertTrue(check.segment([-1,-2,.2],[-1,2,.2]))
        terrain=SimpleNamespace(cell_of=lambda x,y:(0,int((x+10)*2)),level_at=lambda j,i,z:0,
            walk_z=np.zeros((1,1,50)))
        self.assertTrue(line_clear(terrain,-1,0,1,0,az=0,bz=0))
        terrain.ground_clearance=check
        self.assertFalse(line_clear(terrain,-1,0,1,0,az=0,bz=0))
        ceiling=fixture([[[-10,-10,.3],[0,10,.3],[10,-10,.3]]],[[0,0,-1]])
        self.assertFalse(GroundClearance(ceiling).segment([-1,0,.2],[1,0,.2]))

    def test_early_rejection_preserves_native_contact_prefix(self):
        case=json.loads(Path(__file__).with_name('fixtures').joinpath('kotm_support_tie.json').read_text())
        row=case['row'];yaw=next(e['value'] for e in row['control_events'] if e['kind']=='camera_yaw')
        dynamics=dict(native=dict(state=row['before'],frame_clock_ms=row['sim_ms'],pending_move=row['move']),
                      bridge_action=reply([0,0,0,0,0,yaw]),pow_yaw_hold=0,pow_yaw=0.)
        engine=GeneralPhysics(CollisionMap(ML/'collision_maps/KingOfTheMarble_Hunt.npz'))
        actions=[reply([0,0,0,0,0,yaw]) for _ in range(8)]
        from unittest.mock import patch
        with patch('nav.marble_dynamics.bridge.ACCEPT_AMBIGUITY',False):
            complete=predict_replies(engine,dynamics,actions)
            early=predict_replies(engine,dynamics,actions,stop_on_ambiguity=True)
        self.assertTrue(early['stopped_on_ambiguity'])
        self.assertEqual(early['requested_replies'],8)
        self.assertEqual(len(early['controls']),complete['ambiguous_intervals'][0]+1)
        n=len(early['controls'])
        self.assertEqual(early['controls'],complete['controls'][:n])
        self.assertEqual(early['states'],complete['states'][:n+1])
        self.assertEqual(early['events'],complete['events'][:n])

    def test_resource_selection_is_conditional_without_changing_real_inventory(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from nav.protocol import RAW_POW_HELD
        from nav.marble_dynamics.resources import choose_resource
        dyn=copy.deepcopy(self.case['dynamics']);dyn['native']['state']['held']=2
        p=dyn['native']['state']['position'];goal=np.asarray(p)+[0,0,5]
        items=[dict(id=11,gem=False,powerup=1,position=(np.asarray(p)+[3,0,0]).tolist()),
               dict(id=15,gem=False,powerup=5,position=(np.asarray(p)+[5,0,0]).tolist()),
               dict(id=16,gem=False,powerup=6,position=(np.asarray(p)+[1,0,0]).tolist())]
        dyn['items']=items;raw=np.zeros(RAW_DIM,dtype=np.float32);raw[RAW_POW_HELD]=2
        before=copy.deepcopy(dyn);raw_before=raw.copy();calls=[]
        route=SimpleNamespace(goal=None)
        def navigate(position,target):route.goal=np.asarray(target);return target,True
        route.navigate=navigate;route.distance=lambda position:float(np.linalg.norm(np.asarray(position)-route.goal))
        def fake_search(engine,r,snapshot,observation,target,nav,horizon,**kwargs):
            calls.append((snapshot['native']['state']['held'],kwargs.get('target_item')))
            if 'target_item' not in kwargs:
                info=dict(label='conditional',rejected=None,possible_pickup_interval=6 if snapshot['native']['state']['held']==1 else None)
                return None,[(info,None)]
            info=dict(label='actual_acquisition',rejected=None,possible_pickup_interval=None,cost=2.)
            end=(np.asarray(p)+[2,0,0]).tolist()
            rollout=dict(states=[dict(position=p),dict(position=end)])
            return (info,rollout),[(info,rollout)]
        with patch('nav.marble_dynamics.resources.search',fake_search):
            chosen,item,target,hypotheses,candidates=choose_resource(None,route,dyn,raw,goal,None)
        self.assertEqual(item['id'],11);self.assertEqual(chosen[0]['goal_kind'],'resource')
        self.assertEqual([h['hypothetical_held'] for h in hypotheses],[1,5])
        self.assertEqual(calls[-1][0],2);self.assertEqual(calls[-1][1]['id'],11)
        self.assertEqual(dyn,before);np.testing.assert_array_equal(raw,raw_before)

    def test_resource_pickup_proxy_uses_native_bounds(self):
        from nav.collision.test_queries import fixture
        from types import SimpleNamespace
        mesh=fixture([[[-10,-10,0],[10,-10,0],[0,10,0]]],[[0,0,1]])
        states=[dict(position=[.8,0,.2],radius=.2,velocity=[0,0,0],held=0) for _ in range(5)]
        rollout=dict(states=states,ambiguous_intervals=[],failure=None)
        route=SimpleNamespace(distance=lambda position:1.)
        item=dict(gem=False,position=[1.,0,.2],bounds=[1.55,-1,0,1.8,1,1])
        info=assess(SimpleNamespace(mesh=mesh),route,{},np.asarray(item['position']),rollout,'resource',item)
        self.assertIsNone(info['possible_pickup_interval'])
        self.assertTrue(info['stable_continuation'])

    def test_blast_commitment_reaches_its_script_phase(self):
        actions=[reply([0,0,0,0,0,0]) for _ in range(20)];actions[5][8]=1
        memory=PlanMemory();memory.accept(dict(proposed=actions),'timed_blast',[0,0,0])
        for _ in range(5):memory.advance()
        self.assertTrue(memory.locked);self.assertEqual(memory.remaining[0][8],1)
        memory.advance();self.assertTrue(memory.locked)
        for _ in range(2):memory.advance()
        self.assertFalse(memory.locked)

    def test_predicted_blast_meter_updates_proposal_without_mutating_real_obs(self):
        from nav.protocol import RAW_POW_BLAST
        raw=np.zeros(RAW_DIM,dtype=np.float32);raw[RAW_POW_BLAST]=1.
        state=copy.deepcopy(self.case['dynamics']['native']['state'])
        state['blast_context']=dict(value=.125,special=False)
        projected=project_observation(raw,raw[:3],state,4)
        self.assertEqual(raw[RAW_POW_BLAST],1.);self.assertEqual(projected[RAW_POW_BLAST],.125)

    def test_resource_selection_preserves_inventory_until_pickup_or_needed_progress(self):
        from nav.marble_dynamics.search import useful_candidate
        from types import SimpleNamespace
        route=SimpleNamespace(distance=lambda p:p[0])
        def pair(label,end,cost,resources=0,pickup=None,rejected=None):
            return (dict(label=label,cost=cost,rejected=rejected,resources_consumed=resources,possible_pickup_interval=pickup),
                dict(states=[dict(position=[end,0,0])]))
        idle=pair('idle',10,0);walking=pair('walking',8,8);shortcut=pair('blast_shortcut',5,5,1)
        pickup=pair('blast_pickup',9,-90,1,8);rejected=pair('unsafe',0,-100,0,0,'contact_ambiguity')
        self.assertEqual(useful_candidate([idle,walking,shortcut,rejected],route,10,True)[0]['label'],'walking')
        self.assertEqual(useful_candidate([idle,shortcut],route,10,True)[0]['label'],'blast_shortcut')
        self.assertEqual(useful_candidate([walking,shortcut,pickup],route,10,True)[0]['label'],'blast_pickup')
        self.assertEqual(useful_candidate([idle,walking,shortcut],route,10,False)[0]['label'],'blast_shortcut')
        self.assertIsNone(useful_candidate([idle,rejected],route,10,True))

    def test_frozen_proposal_does_not_advance_live_recurrence_or_use_history(self):
        terrain=TerrainGrid(TerrainMap.resolve('Concentric_Hunt'))
        nav=FrozenNavigator(terrain,ML/'models/nav/real_maps_20261010_A2/nav_latest.pth')
        raw=np.zeros(RAW_DIM,dtype=np.float32);state=self.case['dynamics']['native']['state']
        raw[:3]=state['position'];raw[3:6]=state['velocity'];raw[35:38]=state['spin'];raw[RAW_GEMS]=-999;raw[46:]=-999
        gem=np.asarray(state['position'])+[3.,2.,0.]
        proposal=NavigatorProposal(nav,raw,RoutePlanner(terrain),gem,None,4)
        h=nav.h.clone();sent=list(nav.sent);airborne=nav.builder.airborne
        try:predict_replies(self.engine,self.case['dynamics'],proposal)
        except ValueError:pass  # Unsupported Blast request is an explicit rejection.
        self.assertTrue(nav.h.equal(h));self.assertEqual(nav.sent,sent)
        self.assertEqual(nav.builder.airborne,airborne);self.assertEqual(nav.decision,0)
        self.assertTrue(all(not p.requires_grad for p in nav.model.parameters()))


if __name__=='__main__':unittest.main()


def test_apply_blast_value_charges_only_fired_blasts():
    from nav.marble_dynamics.search import apply_blast_value
    fired=dict(label='direct_blast_7_jump_3',cost=50.,rejected=None)
    idle=dict(label='blast_3_jump_0',cost=50.,rejected=None)
    roll=dict(label='route',cost=40.,rejected=None)
    pairs=[(fired,dict(controls=[{},{'script_events':[1]}])),(idle,dict(controls=[{},{}])),(roll,dict(controls=[{}]))]
    apply_blast_value(pairs,0.6,10.)
    assert abs(fired['cost']-56.)<1e-9 and fired['blast_charge_u']==6.
    assert idle['cost']==50. and roll['cost']==40.
    apply_blast_value(pairs,0.,10.);assert abs(fired['cost']-56.)<1e-9


def test_apply_blast_value_floor():
    from nav.marble_dynamics.search import apply_blast_value
    fired=dict(label='blast_3_jump_0',cost=50.,rejected=None)
    pairs=[(fired,dict(controls=[{'script_events':[1]}]))]
    apply_blast_value(pairs,0.2,10.,5.)
    assert abs(fired['cost']-57.)<1e-9 and abs(fired['blast_charge_u']-7.)<1e-9


class MegaProposals(unittest.TestCase):
    def test_forecast_routes_through_script_events_only_for_a_relevant_mega(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from nav.marble_dynamics import bridge
        calls=[]
        with patch('nav.marble_dynamics.script_forecast.predict_events',side_effect=lambda *a,**k:calls.append(k) or 'events'),                 patch('nav.marble_dynamics.bridge._predict_replies_strict',return_value=dict(ambiguous_intervals=[])) as strict:
            on=SimpleNamespace(mega_enabled=True);off=SimpleNamespace(mega_enabled=False)
            held=dict(inventory=dict(held_type=6,mega_active=False,permanent_mega=False))
            class _P:
                steps=24
                def __init__(self,use):self.use=use
                def __call__(self,state,i):return [0]*9
            use=_P(3);nouse=_P(-1)
            self.assertEqual(bridge.predict_replies(on,held,use),'events')
            self.assertEqual(bridge.predict_replies(on,held,[[0]*9,[0,0,0,0,0,0,1,0,0]]),'events')
            self.assertIsInstance(bridge.predict_replies(on,held,nouse),dict)   # a held Mega, not fired: compiled path
            self.assertIsInstance(bridge.predict_replies(on,held,[[0]*9]),dict)
            self.assertTrue(calls[-1]['model_mega'])
            self.assertEqual(bridge.predict_replies(on,dict(inventory=dict(held_type=0,mega_active=True,permanent_mega=False)),[]),'events')
            for dyn in (dict(inventory=dict(held_type=2,mega_active=False,permanent_mega=False)),
                        dict(inventory=dict(held_type=6,mega_active=False,permanent_mega=True)),{}):
                self.assertIsInstance(bridge.predict_replies(on,dyn,[]),dict)
            self.assertIsInstance(bridge.predict_replies(off,held,use),dict)
            self.assertEqual(len(calls),3);self.assertEqual(strict.call_count,6)
