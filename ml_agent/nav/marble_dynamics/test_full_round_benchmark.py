"""Guard against promoting truncated, restarted or differently timed play scores."""
import copy
import unittest
from collections import Counter
from types import SimpleNamespace
from unittest.mock import Mock,patch

from nav.marble_dynamics.full_round_benchmark import EXPECTED_SETTINGS, checked_rounds
from nav.marble_dynamics.reliability_eval import ContinuousController,await_full_round_context,continuous_branches,destination_context,attempt_stuck_rescue
from nav.protocol import RAW_DIM,RAW_SCORE,NOOP_ACTION
import numpy as np
from nav.marble_dynamics.search import PlanMemory
from nav.marble_dynamics.reliability_viewer import round_trace_evidence


class FullRoundScores(unittest.TestCase):
    def test_opening_layout_ignores_native_ids_and_item_order_but_not_positions(self):
        from nav.marble_dynamics.reliability_eval import start_layout
        context=dict(items=[dict(id=1,gem=True,position=[1.,2.,3.],value=2),
            dict(id=2,gem=True,position=[4.,5.,6.],value=5)],native=dict(state=dict(position=[0.,0.,1.])))
        world=dict(header=[12],eval_seed=123,eval_round=2,random_opening=True)
        first=start_layout(context,world)
        changed=copy.deepcopy(context);changed['items'].reverse();changed['items'][0]['id']=999
        self.assertEqual(first['gem_sha256'],start_layout(changed,world)['gem_sha256'])
        changed['items'][0]['position'][0]+=.1
        self.assertNotEqual(first['gem_sha256'],start_layout(changed,world)['gem_sha256'])

    def test_confirmation_rejects_mismatched_and_repeated_actual_openings(self):
        from nav.marble_dynamics.reliability_eval import audit_start_layouts
        def row(seed,signature):
            return dict(map='map',seed=seed,round=1,start_layout=dict(gems=[[0,0,0]],
                gem_sha256=signature,random_opening=True,applied_seed=seed,engine_seed=seed,eval_round=1))
        self.assertTrue(audit_start_layouts([row(1,'a'),row(1,'a'),row(2,'b'),row(2,'b')])['passed'])
        self.assertFalse(audit_start_layouts([row(1,'a'),row(1,'b')])['passed'])
        self.assertFalse(audit_start_layouts([row(1,'a'),row(1,'a'),row(2,'a'),row(2,'a')])['passed'])

    def rescue_case(self,available=True,allowed=True,accepted=True):
        env=SimpleNamespace(msg=SimpleNamespace(dynamics=dict(game_context=dict(available=available,can_quick_respawn=allowed))),
            replay_world=dict(clock_ms=18000),quick_rescue=Mock(return_value=accepted))
        controller=SimpleNamespace(counts=Counter(),invalidate=Mock())
        monitor=SimpleNamespace(window_s=8.,rescue_attempts=0,last_progress=0.,rescued=Mock())
        return env,controller,monitor

    def test_unavailable_rescue_does_not_insert_neutral_controls_or_reset_history(self):
        env,controller,monitor=self.rescue_case(allowed=False)
        self.assertIsNone(attempt_stuck_rescue(env,controller,monitor,12.))
        env.quick_rescue.assert_not_called();controller.invalidate.assert_not_called()
        self.assertEqual(monitor.rescue_attempts,0);self.assertEqual(monitor.last_progress,0.)

    def test_rescue_clock_boundary_tolerates_only_roundoff_and_resets_after_acceptance(self):
        env,controller,monitor=self.rescue_case()
        self.assertIsNone(attempt_stuck_rescue(env,controller,monitor,7.999))
        env.quick_rescue.assert_not_called()
        self.assertTrue(attempt_stuck_rescue(env,controller,monitor,8.-1e-12))
        controller.invalidate.assert_called_once_with('legal_stuck_rescue',reset_actor=True)
        monitor.rescued.assert_called_once_with(18.,float('inf'))

    def test_unknown_availability_keeps_fallback_but_refusal_preserves_actor(self):
        env,controller,monitor=self.rescue_case(available=False,allowed=False,accepted=False)
        self.assertFalse(attempt_stuck_rescue(env,controller,monitor,8.))
        controller.invalidate.assert_called_once_with('legal_stuck_rescue_refused',reset_actor=False)
        monitor.rescued.assert_not_called();self.assertEqual(monitor.rescue_attempts,1)
        self.assertEqual(monitor.last_progress,18.)

    def test_executed_prefix_refuses_control_gaps_resets_and_stale_replies(self):
        from nav.marble_dynamics.reliability_audit import prefix_window_reason
        action=[0.]*9
        rows=[dict(before=dict(native=dict(frame_clock_ms=i*64,object_id=3,state=dict(mode=0,oob=False)),
            game_context=dict(available=True,respawns=1),bridge_action=list(action)),action=list(action),
            outcome=dict(fell=False,round_ended=False,reconnected=False)) for i in range(3)]
        self.assertIsNone(prefix_window_reason(rows,0))
        self.assertEqual(prefix_window_reason(rows[:2],0),'recording_end')
        for field,value,reason in [('frame_clock_ms',129,'control_or_clock_gap'),('object_id',4,'body_object_changed')]:
            changed=copy.deepcopy(rows);changed[2]['before']['native'][field]=value
            self.assertEqual(prefix_window_reason(changed,0),reason)
        changed=copy.deepcopy(rows);changed[1]['before']['game_context']['respawns']=2
        self.assertEqual(prefix_window_reason(changed,0),'native_respawn')
        changed=copy.deepcopy(rows);changed[0]['outcome']['fell']=True
        self.assertEqual(prefix_window_reason(changed,0),'fall_end_or_reconnect')
        changed=copy.deepcopy(rows);changed[2]['before']['bridge_action'][4]=1
        self.assertEqual(prefix_window_reason(changed,0),'pending_reply_mismatch')

    def test_audit_prefix_is_not_a_completed_pool_and_preflight_is_not_a_pool(self):
        from nav.marble_dynamics.reliability_audit import full_round_completion
        maps=['map'+str(i) for i in range(10)]
        prepared=dict(maps=maps,seeds=[1],rounds=1)
        checks=[dict(map=m,seed=1,branch=b,round=1) for m in maps for b in ('physics_plan','frozen_navigator')]
        self.assertFalse(full_round_completion(prepared,checks[:2])['complete_pool'])
        self.assertTrue(full_round_completion(prepared,checks)['complete_pool'])
        self.assertFalse(full_round_completion(prepared,checks[:-1]+[checks[0]])['complete_pool'])
        prepared.update(maps=maps[:2],preflight_only=True)
        checks=[dict(map=m,seed=1,branch='physics_plan',round=1) for m in maps[:2]]
        result=full_round_completion(prepared,checks)
        self.assertTrue(result['complete_declared_rounds']);self.assertFalse(result['complete_pool'])

    def data(self):
        return dict(evaluation_settings=copy.deepcopy(EXPECTED_SETTINGS), rounds=[
            dict(completed=True, engine_points=128.0, points=128.0,
                 initial_score=0.0, round_seconds=300.0, start_time_left_s=300.0,
                 played_seconds=299.968, falls=5, rescues=2)])

    def test_complete_round_keeps_falls_and_rescues(self):
        row = checked_rounds(self.data(), 1)[0]
        self.assertEqual((row['engine_points'], row['falls'], row['rescues']), (128, 5, 2))

    def test_replay_simulation_clock_need_not_equal_hunt_countdown(self):
        data = self.data()
        data['rounds'][0].update(round_seconds=180.0, start_time_left_s=180.0,
            played_seconds=184.321)
        self.assertEqual(checked_rounds(data, 1)[0]['played_seconds'], 184.321)

    def test_partial_restart_or_late_start_cannot_be_a_score(self):
        for field, value in [('completed', False), ('played_seconds', 19.2),
                             ('start_time_left_s', 250.0), ('initial_score', 3.0)]:
            data = self.data()
            data['rounds'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                checked_rounds(data, 1)

    def test_native_score_missing_disagreeing_or_nonfinite_is_rejected(self):
        for value in (None, float('nan'), 127.0):
            data = self.data()
            data['rounds'][0]['engine_points'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                checked_rounds(data, 1)

    def test_missing_round_or_changed_cadence_rescue_is_rejected(self):
        with self.assertRaises(ValueError):
            checked_rounds(self.data(), 2)
        for key, value in [('decision_ms', 16), ('rescue_s', 0.0), ('oob_click', False)]:
            data = self.data()
            data['evaluation_settings'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                checked_rounds(data, 1)


class FullRoundStartup(unittest.TestCase):
    def env(self,contexts):
        obs=np.zeros(RAW_DIM,np.float32)
        env=SimpleNamespace(info={'round_ms':300000},round_ended=False,reconnected=False,
            msg=SimpleNamespace(obs=obs,dynamics=contexts[0]),time_left_s=Mock(return_value=300.),
            replay_world=dict(schema=2,header=[1],random_opening=False))
        pending=iter(contexts[1:])
        def advance(action,repeat):
            self.assertEqual(action,NOOP_ACTION);self.assertEqual(repeat,1)
            env.msg=SimpleNamespace(obs=obs,dynamics=next(pending))
            return env.msg,dict(round_ended=False,reconnected=False)
        env.step=Mock(side_effect=advance)
        return env

    def ready(self):
        return {'native':{'state':{}},'game_context':{'available':True}}

    def test_missing_startup_snapshots_wait_neutrally_until_audited_payload(self):
        env=self.env([None,{},dict(native={'state':{}},game_context={'available':False}),self.ready()])
        self.assertEqual(await_full_round_context(env),3)
        self.assertEqual(env.step.call_count,3)
        self.assertIsNotNone(env.msg.dynamics)

    def test_ready_context_starts_without_extra_game_time(self):
        env=self.env([self.ready()])
        self.assertEqual(await_full_round_context(env),0)
        env.step.assert_not_called()

    def test_live_context_does_not_start_before_replay_world_arrives(self):
        env=self.env([self.ready()]);env.replay_world=None
        def advance(action,repeat):
            self.assertEqual(action,NOOP_ACTION)
            env.replay_world=dict(schema=2,header=[1],random_opening=True)
            return env.msg,dict(round_ended=False,reconnected=False)
        env.step.side_effect=advance
        self.assertEqual(await_full_round_context(env),1)
        self.assertEqual(env.step.call_count,1)

    def test_missing_context_cannot_become_an_unbounded_wait(self):
        env=self.env([None]*4)
        with self.assertRaisesRegex(RuntimeError,'bounded neutral setup'):
            await_full_round_context(env,max_wait=3)
        self.assertEqual(env.step.call_count,3)

    def test_late_nonzero_or_changed_round_refuses_scored_start(self):
        for kind in ('late','scored','ended','reconnected'):
            env=self.env([self.ready()])
            if kind=='late':env.time_left_s.return_value=299.
            elif kind=='scored':env.msg.obs[RAW_SCORE]=1.
            else:setattr(env,'round_ended' if kind=='ended' else 'reconnected',True)
            with self.subTest(kind=kind),self.assertRaises(RuntimeError):
                await_full_round_context(env)
            env.step.assert_not_called()

    def test_round_end_during_setup_is_rejected(self):
        env=self.env([None,self.ready()])
        env.step.side_effect=None
        env.step.return_value=(env.msg,dict(round_ended=True,reconnected=False))
        with self.assertRaisesRegex(RuntimeError,'Native round changed'):
            await_full_round_context(env)
        self.assertEqual(env.step.call_count,1)


class ContinuousEvents(unittest.TestCase):
    def test_risky_research_deadline_keeps_the_revalidated_prefix(self):
        from nav.marble_dynamics.planning_budget import PlanningBudgetExceeded
        controller=self.controller();controller.resource=None
        controller.engine.recheck_risky_plans=True
        controller.args=SimpleNamespace(reuse_plan=True,horizon=48,brake_continuation=0,
            recovery_horizon=0,preserve_resources=False)
        target=[1.,2.,3.,1.,-1.]
        retained=(dict(label='risky',possible_pickup_interval=5,no_braking_backup=True),
                  dict(proposed=[[0.]*9]*8))
        with patch('nav.marble_dynamics.reliability_eval.landing_complete',return_value=False), \
                patch('nav.marble_dynamics.reliability_eval.retained_candidate',return_value=retained), \
                patch('nav.marble_dynamics.reliability_eval.pick',side_effect=PlanningBudgetExceeded):
            chosen,goal=controller._search(dict(native=dict(state=dict(position=[0,0,0]))),
                                           None,target,[target],100)
        self.assertIs(chosen,retained)
        self.assertEqual(goal,target)
        self.assertEqual(controller.counts['risky_plan_research_deadline'],1)

    def test_risky_retained_pickup_is_replaced_only_by_a_backed_pickup(self):
        # 2026-10-20: safe progress never replaces a held pickup plan (first version did; ten-map screen lost points)
        target=[1.,2.,3.,1.,-1.]
        forecast=dict(proposed=[[0.]*9]*8)
        cases=[(False,None,'old_pickup',0),(True,None,'old_pickup',0),(True,5,'backed_pickup',1)]
        for risky,replacement_pickup,expected,replaced in cases:
            controller=self.controller();controller.resource=None
            controller.engine.recheck_risky_plans=True
            controller.args=SimpleNamespace(reuse_plan=True,horizon=48,brake_continuation=0,
                recovery_horizon=0,preserve_resources=False,resource_acquisition=False,
                recharge_continuation=False,landing_search=False)
            retained=(dict(label='old_pickup',possible_pickup_interval=5,no_braking_backup=risky),forecast)
            label='backed_pickup' if replacement_pickup else 'safe_progress'
            replacement=(dict(label=label,possible_pickup_interval=replacement_pickup),forecast)
            with patch('nav.marble_dynamics.reliability_eval.landing_complete',return_value=False),                     patch('nav.marble_dynamics.reliability_eval.retained_candidate',return_value=retained),                     patch('nav.marble_dynamics.reliability_eval.pick',return_value=(replacement,[],target)) as search:
                chosen,_=controller._search(dict(native=dict(state=dict(position=[0,0,0]))),
                                            None,target,[target],100)
            with self.subTest(risky=risky,replacement=label):
                self.assertEqual(chosen[0]['label'],expected)
                self.assertEqual(search.call_count,int(risky))
                self.assertEqual(controller.counts['risky_plan_replaced'],replaced)
                self.assertEqual(controller.counts['research_found_pickup'],replaced)

    def test_risky_progress_plan_keeps_ordinary_research_cadence(self):
        target=[1.,2.,3.,1.,-1.]
        forecast=dict(proposed=[[0.]*9]*8)
        controller=self.controller();controller.resource=None
        controller.engine.recheck_risky_plans=True
        controller.args=SimpleNamespace(reuse_plan=True,horizon=48,brake_continuation=0,
            recovery_horizon=0,preserve_resources=False,resource_acquisition=False,
            recharge_continuation=False,landing_search=False)
        retained=(dict(label='risky_progress',possible_pickup_interval=None,no_braking_backup=True),forecast)
        replacement=(dict(label='safe_progress',possible_pickup_interval=None),forecast)
        with patch('nav.marble_dynamics.reliability_eval.landing_complete',return_value=False),                 patch('nav.marble_dynamics.reliability_eval.retained_candidate',return_value=retained),                 patch('nav.marble_dynamics.reliability_eval.pick',return_value=(replacement,[],target)):
            chosen,_=controller._search(dict(native=dict(state=dict(position=[0,0,0]))),
                                        None,target,[target],100)
        self.assertEqual(chosen[0]['label'],'risky_progress')
        self.assertEqual(controller.counts['risky_plan_researches'],0)
        self.assertEqual(controller.counts['risky_plan_replaced'],0)

    def controller(self):
        controller=ContinuousController.__new__(ContinuousController)
        controller.memory=PlanMemory()
        controller.memory.actions=[[0]*9]*8;controller.memory.gem=[1,2,3]
        controller.current=[1,2,3];controller.resource={'id':42}
        controller.commit_left=4;controller.cooldown=32
        controller.last_prediction={'position':[1,2,3]};controller.last_action=[1]*9
        controller.route=Mock();controller.terrain=object()
        controller.kind='frozen_navigator';controller.engine=SimpleNamespace(mesh=None)
        controller.args=SimpleNamespace();controller.fields={};controller.decision_index=0
        controller.task_goal=None;controller.order_commit=None
        controller.stall=None;controller.stall_pending=False;controller.skipped={}
        controller.nav=Mock();controller.nav.model=object()
        from collections import Counter
        controller.counts=Counter()
        return controller

    def test_checked_rebase_preserves_stall_history_and_recovery_goal(self):
        controller=self.controller()
        controller.task_goal=[1,2,3];controller.stall={'goal':(1,2,3),'since':100}
        controller.stall_pending=True;controller.observed_respawns=7
        for _ in range(10):
            controller.invalidate('realtime_checked_rebase',preserve_task=True)
        self.assertEqual(controller.current,[1,2,3])
        self.assertEqual(controller.task_goal,[1,2,3])
        self.assertEqual(controller.resource,{'id':42})
        self.assertEqual(controller.stall,{'goal':(1,2,3),'since':100})
        self.assertTrue(controller.stall_pending)
        self.assertEqual(controller.observed_respawns,7)
        self.assertEqual(controller.cooldown,32)
        self.assertFalse(controller.memory.actions)
        self.assertIsNone(controller.last_prediction)
        controller.invalidate('real_respawn')
        self.assertIsNone(controller.task_goal)
        self.assertIsNone(controller.stall)
        self.assertIsNone(controller.resource)

    def test_no_live_gem_returns_complete_neutral_bridge_reply(self):
        controller=self.controller();controller.kind='frozen_navigator';controller.fields={}
        msg=SimpleNamespace(obs=None,dynamics={'native':{'state':{'mode':0,'oob':False}}})
        with patch('nav.marble_dynamics.reliability_eval.tour',return_value=(None,None,[])):
            action,info=controller.action(msg,100)
        self.assertEqual(len(action),9)
        self.assertEqual((action[4],action[6],action[8]),(0,0,0))
        self.assertEqual(info['label'],'neutral')

    def test_fall_or_end_cancels_old_commands_without_reading_next_round_state(self):
        for event in ('fell','round_ended','reconnected'):
            controller=self.controller()
            outcome=dict(fell=False,round_ended=False,reconnected=False,gem_delta=0)
            outcome[event]=True
            # End/reconnect observations need not carry the previous native state.
            with self.subTest(event=event), patch('nav.marble_dynamics.reliability_eval.FrozenNavigator') as reset:
                controller.observe(None,SimpleNamespace(dynamics=None),outcome)
                self.assertFalse(controller.memory.remaining)
                self.assertIsNone(controller.current)
                self.assertIsNone(controller.resource)
                self.assertIsNone(controller.last_action)
                self.assertIsNone(controller.last_prediction)
                self.assertEqual(controller.cooldown,0)
                reset.assert_called_once()

    def test_real_pickup_retains_landing_even_after_selected_gem_disappears(self):
        controller=self.controller();controller.last_prediction=None
        controller.memory.landing_goal=[4,5,6];controller.memory.goal_item_id=42
        controller.memory.landing_after_reply=2
        msg=SimpleNamespace(obs=None,dynamics={'native':{'state':{}},'items':[{'id':99}]})
        controller.observe(None,msg,dict(fell=False,round_ended=False,reconnected=False,gem_delta=5))
        self.assertTrue(controller.memory.landing_mode)
        self.assertEqual(controller.memory.landing_goal,[4,5,6])
        self.assertEqual(controller.memory.executed,1)
        self.assertEqual(controller.counts['post_pickup_landing'],1)

    def test_native_respawn_cancels_actor_and_plan_when_oob_flag_was_missed(self):
        controller=self.controller();controller.observed_respawns=10
        msg=SimpleNamespace(dynamics={'game_context':{'available':True,'respawns':11}})
        with patch('nav.marble_dynamics.reliability_eval.FrozenNavigator') as reset:
            controller.observe(None,msg,dict(fell=False,round_ended=False,reconnected=False,gem_delta=0))
        self.assertFalse(controller.memory.remaining)
        self.assertIsNone(controller.observed_respawns)
        self.assertIsNone(controller.last_action)
        self.assertEqual(controller.counts['cancel:observed_native_respawn'],1)
        reset.assert_called_once()


class GeometryComponent(unittest.TestCase):
    def test_declared_order_contains_all_branches_and_balances_first(self):
        args=SimpleNamespace(geometry_reference=True)
        orders=[continuous_branches(args,i,681013) for i in range(3)]
        expected={'physics_plan','geometry_navigator','frozen_navigator'}
        self.assertEqual({order[0] for order in orders},expected)
        for order in orders:self.assertEqual(set(order),expected)
        self.assertEqual(continuous_branches(SimpleNamespace(),0,681013),('physics_plan','frozen_navigator'))

    def test_geometry_route_changes_routing_without_changing_actor_terrain(self):
        terrain=object();geometry=object();engine=SimpleNamespace()
        args=SimpleNamespace(geometry_route=True,geometry_terrain=geometry,navigator_first=True)
        with patch('nav.marble_dynamics.reliability_eval.FrozenNavigator') as actor,patch('nav.marble_dynamics.reliability_eval.RoutePlanner') as route:
            route.return_value.terrain=geometry
            ContinuousController(terrain,engine,'protected.pth',args,'geometry_navigator')
            actor.assert_called_once_with(terrain,'protected.pth')
            self.assertEqual(route.call_count,2)
            self.assertTrue(all(c.args==(geometry,) for c in route.call_args_list))

    def test_geometry_actor_never_searches_and_logs_before_destination(self):
        controller=ContinuousEvents().controller();controller.kind='geometry_navigator'
        controller.args=SimpleNamespace(global_goals=True);controller.fields={}
        controller._search=Mock(side_effect=AssertionError('Routing-only branch invoked physical search'))
        goal=(7.,8.,9.,2.,12.);waypoint=[3.,4.,5.]
        controller.route.navigate.return_value=(waypoint,False)
        action=[.2,0,0,0,0,0,0,0,0];controller.nav.act.return_value=action
        msg=SimpleNamespace(obs=np.zeros(RAW_DIM),dynamics={'native':{'state':{'mode':0,'oob':False,'position':[0,0,0]}},
            'items':[{'id':41,'gem':True,'position':[7.,8.,9.]}]})
        msg.obs[31]=100000.  # ordinary play; this test does not exercise endgame target selection
        with patch('nav.marble_dynamics.reliability_eval.tour',return_value=(goal,None,[goal])),patch('nav.marble_dynamics.reliability_eval.observed_gems',return_value=[goal]):
            sent,info=controller.action(msg,100)
        controller._search.assert_not_called()
        self.assertEqual(sent,action);self.assertIsNone(controller.last_prediction)
        self.assertEqual(info['label'],'geometry_navigator')
        self.assertEqual(info['goal_context']['destination'],[7.,8.,9.])
        self.assertEqual(info['goal_context']['waypoint'],waypoint)
        self.assertEqual(info['goal_context']['item_id'],41)
        self.assertEqual(info['actor_action'],action)
        self.assertEqual(controller.counts['planner_decisions'],0)


class CheckedRecoveryController(unittest.TestCase):
    def controller_and_message(self,kind):
        controller=ContinuousEvents().controller();controller.kind=kind;controller.fields={}
        controller.args=SimpleNamespace(global_goals=False,checked_recovery=True,checked_recovery_replies=40,
            checked_recovery_budget_s=.2,search_budget_s=1.,fallback_cooldown=64)
        controller.cooldown=0;controller.fallback_reasons={};controller.planning_times=[]
        controller.engine=SimpleNamespace(planning_deadline=None,mesh=None)
        target=(7.,8.,9.,2.,12.);controller._search=Mock(return_value=(None,target))
        controller.route.navigate.return_value=([3.,4.,5.],False)
        controller.nav.act.return_value=[.2,0,0,0,0,0,0,0,0]
        msg=SimpleNamespace(obs=np.zeros(RAW_DIM),dynamics={'native':{'state':{'mode':0,'oob':False,'position':[0,0,0]}},
            'items':[{'id':41,'gem':True,'position':[7.,8.,9.]}]})
        msg.obs[31]=100000.
        from collections import Counter
        controller.fallback_reasons=Counter()
        return controller,msg,target

    def test_recovery_retained_and_actor_destination_remains_distinct(self):
        controller,msg,target=self.controller_and_message('physics_plan')
        brake=[0.]*9;info=dict(label='recovery_brake_now',landing_goal=[1.,2.,3.],landing_after_reply=0,
            landing_triangle_id=None,stopping_verified=True)
        forecast=dict(proposed=[brake]*40,states=[{'position':[0,0,0]}]*41)
        with patch('nav.marble_dynamics.reliability_eval.tour',return_value=(target,None,[target])),patch('nav.marble_dynamics.reliability_eval.ordinary_recovery',return_value=((info,forecast),{'stop':'completed'})):
            action,choice=controller.action(msg,100)
        self.assertEqual(action,brake);controller.nav.act.assert_called_once()
        self.assertTrue(controller.memory.physical_recovery);self.assertTrue(controller.memory.landing_mode)
        self.assertTrue(controller.memory.stopping_verified);self.assertEqual(controller.cooldown,0)
        self.assertEqual(choice['goal_context']['destination'],[1.,2.,3.])
        self.assertEqual(choice['goal_context']['kind'],'physical_recovery')
        self.assertEqual(choice['actor_goal_context']['destination'],[7.,8.,9.])
        self.assertEqual(choice['task_goal_context']['destination'],[7.,8.,9.])
        self.assertEqual(choice['task_goal_context']['kind'],'gem')
        self.assertEqual(choice['primary_fallback_reason'],'no_checked_candidate')
        controller.invalidate('test_reset');self.assertFalse(controller.memory.physical_recovery)
        self.assertIsNone(controller.task_goal)

    def test_retained_stop_keeps_real_task_and_duplicate_gems_keep_their_kind(self):
        controller,msg,target=self.controller_and_message('physics_plan')
        controller.task_goal=list(target);controller.memory.landing_mode=True
        controller.memory.landing_goal=[1.,2.,3.];controller.memory.physical_recovery=True
        brake=[0.]*9;info=dict(label='retained_recovery')
        forecast=dict(proposed=[brake],states=[{},{}])
        controller._search.return_value=((info,forecast),[1.,2.,3.,0.,-1.])
        msg.dynamics['items'].append({'id':42,'gem':True,'position':[7.,8.,9.]})
        with patch('nav.marble_dynamics.reliability_eval.tour',return_value=(target,None,[target])):
            action,choice=controller.action(msg,100)
        self.assertEqual(choice['goal_context']['kind'],'physical_recovery')
        self.assertEqual(choice['task_goal_context']['kind'],'gem')
        self.assertIsNone(choice['task_goal_context']['item_id'])
        self.assertEqual(choice['task_goal_context']['candidate_item_ids'],[41,42])
        self.assertEqual(choice['task_goal_context']['destination'],[7.,8.,9.])

    def test_reference_searches_physics_but_never_adds_recovery(self):
        controller,msg,target=self.controller_and_message('physics_plan_reference')
        with patch('nav.marble_dynamics.reliability_eval.tour',return_value=(target,None,[target])),patch('nav.marble_dynamics.reliability_eval.ordinary_recovery') as recovery:
            action,choice=controller.action(msg,100)
        controller._search.assert_called_once();recovery.assert_not_called()
        self.assertEqual(action,controller.nav.act.return_value);self.assertEqual(choice['label'],'A2_fallback')

    def test_task_variant_uses_independent_route_and_one_actor_update(self):
        controller,msg,target=self.controller_and_message('physics_plan')
        controller.args.actor_task_during_recovery=True
        controller.actor_task_route=Mock();controller.actor_task_route.navigate.return_value=([6.,7.,8.],False)
        controller.task_goal=list(target);controller.memory.landing_mode=True
        controller.memory.physical_recovery=True;controller.memory.landing_goal=[1.,2.,3.]
        brake=[0.]*9;forecast=dict(proposed=[brake],states=[{},{}])
        controller._search.return_value=((dict(label='retained_stop'),forecast),[1.,2.,3.,0.,-1.])
        with patch('nav.marble_dynamics.reliability_eval.tour',return_value=(target,None,[target])):
            action,choice=controller.action(msg,100)
        self.assertEqual(action,brake)
        controller.route.navigate.assert_called_once_with([0,0,0],[1.,2.,3.])
        controller.actor_task_route.navigate.assert_called_once_with([0,0,0],[7.,8.,9.])
        controller.nav.act.assert_called_once_with(msg.obs,[6.,7.,8.],[7.,8.,9.],None)
        controller.nav.sent_action.assert_called_once_with(brake,msg.obs)
        self.assertTrue(choice['actor_goal_context']['preserved_task_during_recovery'])
        self.assertEqual(choice['goal_context']['destination'],[1.,2.,3.])
        self.assertEqual(choice['actor_goal_context']['destination'],[7.,8.,9.])
        controller.invalidate('test_reset');controller.actor_task_route.reset.assert_called_once()

    def test_task_variant_replaces_consumed_resource_with_observed_gem(self):
        controller,msg,target=self.controller_and_message('physics_plan')
        controller.args.actor_task_during_recovery=True
        controller.actor_task_route=Mock();controller.actor_task_route.navigate.return_value=([6.,7.,8.],False)
        # This resource no longer appears in the actual item list.
        controller.task_goal=[4.,5.,6.,0.,-1.]
        controller.memory.landing_mode=True;controller.memory.physical_recovery=True
        controller.memory.landing_goal=[1.,2.,3.]
        forecast=dict(proposed=[[0.]*9],states=[{},{}])
        controller._search.return_value=((dict(label='retained_stop'),forecast),[1.,2.,3.,0.,-1.])
        with patch('nav.marble_dynamics.reliability_eval.tour',return_value=(target,None,[target])):
            _,choice=controller.action(msg,100)
        self.assertEqual(choice['task_goal_context']['candidate_item_ids'],[41])
        self.assertEqual(choice['actor_goal_context']['destination'],[7.,8.,9.])
        controller.actor_task_route.navigate.assert_called_once_with([0,0,0],[7.,8.,9.])

    def test_cooldown_variant_revalidates_recovery_without_resetting_primary_delay(self):
        controller,msg,target=self.controller_and_message('physics_plan')
        controller.args.recovery_preserve_search_cooldown=True
        brake=[0.]*9;info=dict(label='recovery_brake_now',landing_goal=[1.,2.,3.],landing_after_reply=0,
            landing_triangle_id=None,stopping_verified=True)
        forecast=dict(proposed=[brake]*40,states=[{}]*41)
        with patch('nav.marble_dynamics.reliability_eval.tour',return_value=(target,None,[target])),patch('nav.marble_dynamics.reliability_eval.ordinary_recovery',return_value=((info,forecast),{})):
            controller.action(msg,100)
            self.assertEqual(controller.cooldown,64)
            controller._search.reset_mock()
            controller._search.return_value=((dict(label='retained_stop'),forecast),[1.,2.,3.,0.,-1.])
            controller.action(msg,100)
            self.assertEqual(controller.cooldown,63)
            controller._search.assert_called_once_with(msg.dynamics,msg.obs,[1.,2.,3.,0.,-1.],
                [target],100,skip_new_search=True)
            # A completed backup with no new broad search must not restart64.
            controller._search.return_value=(None,target)
            controller.action(msg,100)
            self.assertEqual(controller.cooldown,62)
        self.assertEqual(controller.counts['recovery_search_cooldown_decisions'],2)

    def test_completed_recovery_obeys_cooldown_before_new_broad_search(self):
        controller,msg,target=self.controller_and_message('physics_plan')
        del controller._search
        controller.args.reuse_plan=False;controller.memory.landing_mode=True
        controller.memory.physical_recovery=True;controller.memory.landing_goal=[1.,2.,3.]
        with patch('nav.marble_dynamics.reliability_eval.landing_complete',return_value=True),patch('nav.marble_dynamics.reliability_eval.tour',return_value=(target,None,[target])),patch('nav.marble_dynamics.reliability_eval.pick') as broad:
            chosen,returned=controller._search(msg.dynamics,msg.obs,[1.,2.,3.,0.,-1.],[target],100,
                skip_new_search=True)
        broad.assert_not_called();self.assertIsNone(chosen);self.assertEqual(returned,target)
        self.assertFalse(controller.memory.landing_mode)
        self.assertEqual(controller.counts['recoveries_completed'],1)

    def test_search_timeout_after_completed_recovery_cannot_restore_stale_actor_goal(self):
        from nav.marble_dynamics.planning_budget import PlanningBudgetExceeded
        for checked in (True,False):
            controller,msg,target=self.controller_and_message('physics_plan')
            controller.args.actor_task_during_recovery=True
            controller.actor_task_route=Mock();controller.actor_task_route.navigate.return_value=([6.,7.,8.],False)
            controller.task_goal=list(target);controller.memory.landing_mode=True
            controller.memory.physical_recovery=True;controller.memory.landing_goal=[1.,2.,3.]
            def timed_out(*args,**kwargs):
                controller.memory.clear()
                raise PlanningBudgetExceeded('Search after completed recovery')
            controller._search.side_effect=timed_out
            info=dict(label='recovery_brake_now',landing_goal=[1.,2.,3.],landing_after_reply=0,
                landing_triangle_id=None,stopping_verified=True)
            forecast=dict(proposed=[[0.]*9]*40,states=[{}]*41)
            recovery=((info,forecast),{}) if checked else (None,{})
            with self.subTest(checked=checked),patch('nav.marble_dynamics.reliability_eval.tour',return_value=(target,None,[target])),patch('nav.marble_dynamics.reliability_eval.ordinary_recovery',return_value=recovery):
                action,choice=controller.action(msg,100)
            controller.nav.act.assert_called_once_with(msg.obs,[6.,7.,8.],[7.,8.,9.],None)
            controller.nav.sent_action.assert_called_once_with(action,msg.obs)
            self.assertEqual(choice['actor_goal_context']['destination'],[7.,8.,9.])
            self.assertTrue(choice['actor_goal_context']['preserved_task_during_recovery'])
            self.assertEqual(choice['task_goal_context']['candidate_item_ids'],[41])
            if not checked:
                self.assertEqual(choice['label'],'A2_fallback')
                self.assertEqual(choice['goal_context']['destination'],[7.,8.,9.])

    def test_reference_order_balances_without_changing_default(self):
        args=SimpleNamespace(recovery_reference=True)
        self.assertEqual(continuous_branches(args,0,681014),('physics_plan','physics_plan_reference'))
        self.assertEqual(continuous_branches(args,1,681014),('physics_plan_reference','physics_plan'))
        self.assertEqual(continuous_branches(SimpleNamespace(preflight_only=True),0,781014),('physics_plan',))


class NativeConsumptionAttribution(unittest.TestCase):
    def rows(self):
        rows=[];previous=[0.]*9
        for i in range(4):
            action=[0.]*9;action[6]=int(i in (1,2))
            rows.append(dict(before={'native':{'frame_clock_ms':100+i*64,'object_id':1,'state':{'held':1}},
                'bridge_action':previous.copy(),'game_context':{'available':True,'respawns':0}},
                action=action,choice={'label':'checked' if i==1 else 'fallback'},predicted={} if i==1 else None,
                outcome={'fell':False,'round_ended':False,'reconnected':False}))
            previous=action
        return rows

    def test_consecutive_requests_attribute_consumption_to_queued_older_command(self):
        from nav.marble_dynamics.reliability_audit import queued_power_command
        # Newer decision2 is closer (16ms), but decision1 reaches this Move.
        self.assertEqual(queued_power_command(self.rows(),244),(1,None))
        # First subframe still holds the Move from two replies earlier.
        self.assertEqual(queued_power_command(self.rows(),292),(1,None))

    def test_control_gap_or_native_respawn_does_not_guess_command_identity(self):
        from nav.marble_dynamics.reliability_audit import queued_power_command
        rows=self.rows();rows[1]['outcome']['fell']=True
        self.assertEqual(queued_power_command(rows,244),(None,'discontinuous_control_interval'))
        rows=self.rows();rows[2]['before']['game_context']['respawns']=1
        self.assertEqual(queued_power_command(rows,244),(None,'native_respawn'))
        rows=self.rows();rows[2]['before']['bridge_action'][6]=0
        self.assertEqual(queued_power_command(rows,244),(None,'pending_reply_mismatch'))

    def test_native_reset_is_separate_from_playable_inventory_consumption(self):
        from nav.marble_dynamics.reliability_audit import full_round_activation_audit
        event=dict(kind='transition',sim_ms=244,move={'use':True},
            before={'held':1,'mode':0,'oob':False},after={'held':0,'mode':0,'oob':False,'effects':[]})
        reset=copy.deepcopy(event);reset['sim_ms']=308;reset['move']['use']=False
        rows=self.rows()
        for i,row in enumerate(rows):row['decision']=i+1
        audit=full_round_activation_audit(rows,[event,reset])
        self.assertEqual(len(audit['confirmed_native_inventory_consumptions']),1)
        use=audit['confirmed_native_inventory_consumptions'][0]
        self.assertEqual(use['matched_command']['decision'],2)
        self.assertEqual(use['matched_command']['log_index'],1);self.assertTrue(use['matched_command']['checked'])
        self.assertEqual(len(audit['unclassified_inventory_resets']),1)


class FullRoundReview(unittest.TestCase):
    def row(self,clock,respawns=0,position=None):
        return dict(before={'native':{'frame_clock_ms':clock,'object_id':1,'state':{'position':position or [0,0,0]}},
            'game_context':{'available':True,'respawns':respawns},
            'items':[{'id':42,'gem':True,'position':[10,20,30]}]},
            choice={'label':'A2_fallback'},predicted=None,
            outcome={'fell':False,'round_ended':False,'reconnected':False})

    def test_respawn_and_control_gaps_break_line_without_deleting_measured_point(self):
        logs=[self.row(100),self.row(164,1,[50,0,0]),self.row(292,1,[51,0,0])]
        original=copy.deepcopy(logs)
        gaps,reasons,_,_=round_trace_evidence(logs)
        self.assertEqual(gaps,[False,True,True])
        self.assertIn('native_respawn',reasons[1]);self.assertIn('unobserved_control_interval',reasons[2])
        self.assertEqual(logs,original)

    def test_checked_item_comes_from_same_before_and_fallback_goal_stays_unknown(self):
        a=self.row(100);a['choice']['goal_item_id']=42;a['predicted']={}
        b=self.row(164);b['choice']['goal_item_id']=42
        c=self.row(228);c['choice']['goal_item_id']=99;c['predicted']={}
        c['before']['items']=[{'id':42,'gem':True,'position':[999,999,999]}]
        _,_,goals,kinds=round_trace_evidence([a,b,c])
        self.assertEqual(goals,[[10,20,30],[None,None,None],[None,None,None]])
        self.assertEqual(kinds,['checked_observed_gem','unknown','unknown'])

    def test_explicit_destination_is_preserved_instead_of_item_or_waypoint(self):
        row=self.row(100);row['choice']['goal_context']=dict(destination=[10.001,20,30],waypoint=[3,4,5],kind='gem')
        _,_,goals,kinds=round_trace_evidence([row])
        self.assertEqual(goals,[[10.001,20,30]]);self.assertEqual(kinds,['gem'])


if __name__ == '__main__':
    unittest.main()
