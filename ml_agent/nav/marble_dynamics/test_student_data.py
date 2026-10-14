"""Native-input and full collision-information contracts for an unfitted student."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import hashlib
import gzip

import numpy as np
import torch
from nav.collision.mesh import CollisionMap
from nav.marble_dynamics.record import ML
from nav.marble_dynamics.student_data import geometry,state_features,STATE_FEATURES,SURFACE_FEATURES
from nav.marble_dynamics.fit_motion_student import (KinematicResidualStudent,RelativeGeometryResidualStudent,
    ContactContextResidualStudent,closest_triangle_context,contact_geometry_features,ordinary_motion_eligible)
from nav.marble_dynamics.student_cache import CompleteMapMotionCache
from nav.marble_dynamics.fit_motion_student import native_motion_loss
from nav.marble_dynamics.pairs import paired_motion_example,observed_world_example,observable_state_tokens,observable_context_example
from nav.marble_dynamics.fast_nav import FastMapPolicy,features as navigation_features,action_target,decode_action,button_eligibility,objective,legal_action,causal_activation_advantage,advice_indices
from nav.marble_dynamics.fast_nav import configure_activation_only,activation_objective,assert_movement_preserved
from nav.marble_dynamics.fast_nav import base_preservation_objective,preservation_fit_objective,causal_action_advantage


class FastNavigationContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path(__file__).with_name('fixtures')/'fast_navigation_current.json'
        cls.row=json.loads(path.read_text())['row']
        cls.goal=cls.row['choice']['goal_context']
        cls.mesh=CollisionMap(ML/'collision_maps/Sprawl_Hunt.npz');cls.geo=geometry(cls.mesh)

    def build(self,dynamics=None,raw=None):
        return navigation_features(dynamics or self.row['before'],self.row['raw_observation'] if raw is None else raw,
            self.goal['destination'],self.goal['waypoint'],self.geo['center'],self.geo['scale'])

    def test_scores_ids_future_motion_and_inactive_timers_are_not_inputs(self):
        d=copy.deepcopy(self.row['before']);raw=list(self.row['raw_observation'])
        d['native']['object_id']=92836;d['native']['state']['datablock']=42
        d['future_position']=[900,800,700];raw[33]=100000;raw[34]=0
        d['native']['state']['effects']=[(a,t if a else float('nan')) for a,t in d['native']['state']['effects']]
        for old,new in zip(self.build(),self.build(d,raw)):np.testing.assert_array_equal(old,new)

    def test_spin_inventory_and_pending_reply_distinguish_current_states(self):
        original,_=self.build()
        for key in ('spin','bridge','inventory'):
            d=copy.deepcopy(self.row['before']);raw=list(self.row['raw_observation'])
            if key=='spin':d['native']['state']['spin'][2]+=40
            elif key=='bridge':d['bridge_action'][4]=1-d['bridge_action'][4]
            else:raw[38]=6
            changed,_=self.build(d,raw);self.assertFalse(np.array_equal(original,changed))

    def test_all_faces_are_encoded_and_cached_forward_matches_training(self):
        x,q=self.build();torch.manual_seed(12)
        model=FastMapPolicy(len(x),self.geo['features'].shape[1]).eval()
        surfaces,lo,hi=[torch.from_numpy(self.geo[k]) for k in ('features','bbox_lo','bbox_hi')]
        with torch.inference_mode():
            encoded=model.encode_map(surfaces,lo,hi)
            self.assertEqual(len(encoded[0]),len(self.mesh.triangles))
            full=model(torch.from_numpy(x)[None],torch.from_numpy(q)[None],surfaces,lo,hi)
            cached=model.forward_cached(torch.from_numpy(x)[None],torch.from_numpy(q)[None],encoded)
        torch.testing.assert_close(full,cached,rtol=0,atol=0)

    def test_distant_destination_changes_spatial_queries(self):
        x,q=self.build();goal=np.asarray(self.goal['destination'])+[200,0,40]
        nx,nq=navigation_features(self.row['before'],self.row['raw_observation'],goal,
            self.goal['waypoint'],self.geo['center'],self.geo['scale'])
        self.assertFalse(np.array_equal(x,nx));self.assertFalse(np.array_equal(q[1],nq[1]))
        np.testing.assert_array_equal(q[0],nq[0])

    def test_decode_recovers_world_reply_with_independent_buttons(self):
        action=[.4,0,.8,0,1,-2.1,0,.2,1]
        target=action_target(action);prediction=target.copy()
        prediction[:2]=np.arctanh(target[:2]);prediction[6:]=np.where(target[6:]>.5,4,-4)
        decoded=decode_action(prediction)
        np.testing.assert_allclose(decoded,action,atol=1e-6)

    def test_nonfinite_and_unknown_power_inputs_refuse(self):
        d=copy.deepcopy(self.row['before']);d['native']['state']['spin'][0]=float('nan')
        with self.assertRaises(ValueError):self.build(d)
        raw=list(self.row['raw_observation']);raw[38]=7
        with self.assertRaises(ValueError):self.build(raw=raw)

    def test_zero_residual_preserves_base_reply_and_noncanonical_angles(self):
        base=[1.414214,0,0,1.414214,1,3.737003,0,-4.12,1]
        x,q=navigation_features(self.row['before'],self.row['raw_observation'],self.goal['destination'],
            self.goal['waypoint'],self.geo['center'],self.geo['scale'],base_action=base)
        model=FastMapPolicy(len(x),self.geo['features'].shape[1],residual=True).eval()
        with torch.inference_mode():
            output=model(torch.from_numpy(x)[None],torch.from_numpy(q)[None],
                *(torch.from_numpy(self.geo[k]) for k in ('features','bbox_lo','bbox_hi')))[0].numpy()
        self.assertEqual(decode_action(output,base),base)

    def test_supervision_mask_tracks_actual_inventory_and_blast_charge(self):
        d=copy.deepcopy(self.row['before']);raw=list(self.row['raw_observation'])
        raw[38:41]=[0,0,0];x,_=self.build(d,raw)
        mask=button_eligibility(torch.from_numpy(x)[None],torch.zeros(1,9))
        torch.testing.assert_close(mask,torch.tensor([[1.,0.,0.]]))
        raw[38:41]=[5,.1,1];x,_=self.build(d,raw)
        mask=button_eligibility(torch.from_numpy(x)[None],torch.zeros(1,9))
        torch.testing.assert_close(mask,torch.ones(1,3))

    def test_valid_positive_target_is_never_silenced_and_masks_affect_gradients(self):
        x=torch.zeros(1,109);target=torch.zeros(1,9);target[0,7]=1
        eligible=button_eligibility(x,target)
        torch.testing.assert_close(eligible,torch.tensor([[1.,1.,0.]]))
        pred=torch.zeros(1,9,requires_grad=True)
        objective(pred,target,torch.ones(1),torch.tensor([1.,8.,8.]),eligible).backward()
        self.assertLess(float(pred.grad[0,7]),0.);self.assertEqual(float(pred.grad[0,8]),0.)
        self.assertGreater(abs(float(pred.grad[0,7])),abs(float(pred.grad[0,6])))

    def test_unknown_layout_cannot_silently_mask_supervision(self):
        with self.assertRaises(ValueError):button_eligibility(torch.zeros(1,108),torch.zeros(1,9))

    def test_linear_residual_preserves_base_and_can_learn_braking_at_full_throttle(self):
        base=[1.414214,0,0,1.414214,1,3.737003,0,-4.12,1]
        x,q=navigation_features(self.row['before'],self.row['raw_observation'],self.goal['destination'],
            self.goal['waypoint'],self.geo['center'],self.geo['scale'],base_action=base)
        model=FastMapPolicy(len(x),self.geo['features'].shape[1],residual=True,axis_mode='linear')
        pred=model(torch.from_numpy(x)[None],torch.from_numpy(q)[None],
            *(torch.from_numpy(self.geo[k]) for k in ('features','bbox_lo','bbox_hi')))
        self.assertEqual(decode_action(pred[0].detach().numpy(),base,'linear'),base)
        target=torch.from_numpy(action_target(base))[None];target[:,:2]=0
        objective(pred,target,torch.ones(1),axis_mode='linear').backward()
        self.assertGreater(float(model.head[-1].bias.grad[0]),.4)
        self.assertGreater(float(model.head[-1].bias.grad[1]),.4)

    def test_linear_output_serializes_bounded_controls_and_refuses_unknown_mode(self):
        pred=np.asarray([20.,-20.,0.,1.,0.,1.,-1.,-1.,-1.])
        action=decode_action(pred,axis_mode='linear')
        self.assertEqual(action[:4],[0.,1.414214,0.,1.414214])
        with self.assertRaises(ValueError):decode_action(pred,axis_mode='unknown')

    def test_unavailable_buttons_cannot_reset_aim_latch_or_fire_empty_blast(self):
        d=copy.deepcopy(self.row['before']);raw=list(self.row['raw_observation']);raw[38]=0
        d['blast'].update(enabled=True,frozen=False,value=.1,required=.2,special=True)
        action=[.4,0,0,.2,1,-2.,1,1.2,1]
        result=legal_action(action,d,raw)
        self.assertEqual(result[6],0);self.assertEqual(result[8],0)
        self.assertEqual([result[i] for i in (0,1,2,3,4,5,7)],[action[i] for i in (0,1,2,3,4,5,7)])
        self.assertEqual(action[6],1)  # No mutation of logged unmasked reply.

    def test_held_powers_are_allowed_in_air_without_geometry_prior(self):
        d=copy.deepcopy(self.row['before']);d['native']['state']['ground_ms']=0
        d['blast'].update(enabled=True,frozen=False,value=.2,required=.2)
        raw=list(self.row['raw_observation']);action=[0,0,0,0,1,0,1,0,1]
        for held in range(1,7):
            raw[38]=held;self.assertEqual(legal_action(action,d,raw),action)

    def test_disabled_frozen_and_triple_meter_rules_remain_distinct(self):
        d=copy.deepcopy(self.row['before']);raw=list(self.row['raw_observation']);action=[0]*9;action[8]=1
        d['blast'].update(enabled=True,frozen=False,value=.32,required=.2,triple=True)
        self.assertEqual(legal_action(action,d,raw)[8],0)
        d['blast']['value']=.33;self.assertEqual(legal_action(action,d,raw)[8],1)
        d['blast']['frozen']=True;self.assertEqual(legal_action(action,d,raw)[8],0)
        d['blast'].update(frozen=False,enabled=False);self.assertEqual(legal_action(action,d,raw)[8],0)

    def test_activation_needs_matched_horizon_and_real_advantage(self):
        forecast=dict(requested_replies=48,controls=[{}]*48,failure=None,ambiguous_intervals=[])
        info=dict(rejected=None,stable_continuation=True,possible_pickup_interval=None,cost=10.,resources_consumed=1)
        control=copy.deepcopy(forecast);old=copy.deepcopy(info)
        old['resources_consumed']=0
        self.assertFalse(causal_activation_advantage(info,forecast,old,control))
        old['cost']=10.5;self.assertTrue(causal_activation_advantage(info,forecast,old,control))
        control['requested_replies']=24;control['controls']=control['controls'][:24]
        self.assertFalse(causal_activation_advantage(info,forecast,old,control))

    def test_incomplete_or_ambiguous_activation_control_cannot_supply_a_label(self):
        p=dict(requested_replies=4,controls=[{}]*4,failure=None,ambiguous_intervals=[])
        i=dict(rejected=None,stable_continuation=True,possible_pickup_interval=2,cost=-990.,resources_consumed=1)
        old=dict(rejected=None,stable_continuation=True,possible_pickup_interval=None,cost=10.)
        self.assertTrue(causal_activation_advantage(i,p,old,p))
        for change in [dict(controls=[{}]),dict(ambiguous_intervals=[2]),dict(failure='unknown')]:
            bad=dict(p,**change);self.assertFalse(causal_activation_advantage(i,p,old,bad))
        same=dict(old,possible_pickup_interval=2);self.assertFalse(causal_activation_advantage(i,p,same,p))
        queued=dict(old,resources_consumed=1);self.assertFalse(causal_activation_advantage(i,p,queued,p))

    def test_action_sampling_reserves_existing_validated_labels_without_inventing_them(self):
        y=torch.zeros(8,9);y[0,7]=1;y[1,8]=1
        ix=advice_indices(dict(y=y),np.random.RandomState(7),12,True)
        self.assertEqual(len(ix),12);self.assertTrue(((ix>=0)&(ix<8)).all())
        self.assertGreaterEqual(int(y[ix,7].sum()),2);self.assertGreaterEqual(int(y[ix,8].sum()),2)
        absent=advice_indices(dict(y=torch.zeros(8,9)),np.random.RandomState(7),12,True)
        self.assertTrue(((absent>=0)&(absent<8)).all())

    def test_first_reply_correction_requires_same_tail_and_advantage(self):
        forecast=dict(requested_replies=4,controls=[{}]*4,proposed=[[0.]*9 for _ in range(4)],failure=None,ambiguous_intervals=[])
        baseline=copy.deepcopy(forecast);baseline['proposed'][0][4]=1
        info=dict(rejected=None,stable_continuation=True,possible_pickup_interval=None,cost=10.)
        old=dict(info,cost=10.5)
        self.assertTrue(causal_action_advantage(info,forecast,old,baseline))
        self.assertFalse(causal_action_advantage(info,forecast,info,baseline))
        tail=copy.deepcopy(baseline);tail['proposed'][2][4]=1
        self.assertFalse(causal_action_advantage(info,forecast,old,tail))
        for changed in (dict(controls=[{}]),dict(failure='unsupported'),dict(ambiguous_intervals=[2])):
            self.assertFalse(causal_action_advantage(info,forecast,old,dict(baseline,**changed)))
        self.assertFalse(causal_action_advantage(info,forecast,old,None))
        bad=dict(old,rejected='nonfinite_goal_cost',stable_continuation=False)
        self.assertFalse(causal_action_advantage(info,forecast,bad,baseline))

    def test_current_base_anchor_has_zero_loss_and_does_not_escalate_logits(self):
        base=torch.tensor([[.4,-.8,.6,.8,0.,1.,1.,0.,0.]])
        pred=base.clone();pred[:,6:]=torch.tensor([2.,-2.,-2.]);pred.requires_grad_()
        loss=base_preservation_objective(pred,base)
        self.assertEqual(float(loss),0.);loss.backward()
        self.assertTrue(torch.equal(pred.grad,torch.zeros_like(pred)))
        changed=pred.detach().clone();changed[0,6]=4.
        self.assertGreater(float(base_preservation_objective(changed,base)),0.)

    def test_only_train_anchor_rows_ignore_teacher_labels_and_advice_still_learns(self):
        x=torch.zeros(2,109);base=x[:,-9:];base[:,3]=1.;base[:,5]=1.
        pred=base.clone();pred[:,6:]=-2.;pred.requires_grad_()
        y=base.clone();y[0,:2]=1.;y[0,6:]=1.;y[1,7]=1.;original=y.clone()
        loss=preservation_fit_objective(pred,x,y,torch.tensor([10.,1.]),1,'linear')
        loss.backward()
        self.assertTrue(torch.equal(pred.grad[0],torch.zeros_like(pred.grad[0])))
        self.assertLess(float(pred.grad[1,7]),0.)
        self.assertTrue(torch.equal(y,original))

    def test_activation_learning_cannot_erase_pretrained_movement_or_jump(self):
        torch.manual_seed(201013)
        model=FastMapPolicy(109,4,residual=True,axis_mode='linear')
        with torch.no_grad():
            model.head[-1].weight.normal_(0,.05);model.head[-1].bias.normal_(0,.05)
        original={k:v.clone() for k,v in model.state_dict().items()}
        x=torch.randn(8,109)*.1;x[:,41]=1;x[:,47]=.5;x[:,-3:]=0
        q=torch.randn(8,4,3)*.1;surfaces=torch.randn(12,4)
        lo=torch.zeros(12,3);hi=lo+.1
        before=model(x,q,surfaces,lo,hi).detach()
        configure_activation_only(model)
        optimizer=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=.03)
        target=torch.zeros(8,9);target[:,7:]=1
        for _ in range(40):
            pred=model(x,q,surfaces,lo,hi)
            loss=activation_objective(pred,target,x,torch.ones(8))
            optimizer.zero_grad();loss.backward();optimizer.step()
        after=model(x,q,surfaces,lo,hi).detach()
        self.assertTrue(torch.equal(before[:,:7],after[:,:7]))
        self.assertTrue((after[:,7:]>before[:,7:]).all())
        self.assertTrue(assert_movement_preserved(model,original))
        with torch.no_grad():model.state[0].weight[0,0]+=1
        with self.assertRaises(ValueError):assert_movement_preserved(model,original)

    def test_activation_loss_has_no_gradient_from_unavailable_negative_buttons(self):
        x=torch.zeros(1,109);target=torch.zeros(1,9)
        pred=torch.zeros(1,9,requires_grad=True)
        activation_objective(pred,target,x,torch.ones(1)).backward()
        self.assertTrue(torch.equal(pred.grad,torch.zeros_like(pred.grad)))

    def test_independent_output_replay_uses_current_logged_base_and_refuses_missing_it(self):
        from nav.marble_dynamics.reliability_audit import fast_reply_from_logged_inputs
        row=copy.deepcopy(self.row);base=row['action']
        row['choice']['base_actor_action']=base
        row['choice']['actor_goal_context']=dict(destination=self.goal['destination'],waypoint=self.goal['waypoint'])
        model=FastMapPolicy(109,self.geo['features'].shape[1],residual=True,axis_mode='linear')
        with torch.inference_mode():encoded=model.encode_map(*(torch.from_numpy(self.geo[k]) for k in ('features','bbox_lo','bbox_hi')))
        self.assertEqual(fast_reply_from_logged_inputs(row,model,encoded,self.geo,{}),base)
        del row['choice']['base_actor_action']
        with self.assertRaises(ValueError):fast_reply_from_logged_inputs(row,model,encoded,self.geo,{})


class MotionPreparationContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.row=json.loads((Path(__file__).with_name('fixtures')/'live_bridge.json').read_text())['cases'][0]['rows'][0]

    def test_every_collision_face_boundary_and_native_plane_is_available(self):
        names=json.loads((ML/'configs/real_maps_20261009.json').read_text())['maps']
        for name in names:
            mesh=CollisionMap(ML/'collision_maps'/(name+'.npz'));g=geometry(mesh);features=g['features']
            self.assertEqual(len(features),len(mesh.triangles))
            np.testing.assert_array_equal(g['triangle_ids'],np.arange(len(mesh.triangles)))
            np.testing.assert_array_equal(features[:,-3:],mesh.boundary_edges.astype(np.float32))
            # Raw native normal can differ from normalized display normal.
            ix=SURFACE_FEATURES.index('native_plane_x')
            np.testing.assert_array_equal(features[:,ix:ix+3],mesh.planes[:,:3].astype(np.float32))
            expected=((mesh.planes[:,3]+mesh.planes[:,:3]@g['center'])/g['scale']).astype(np.float32)
            np.testing.assert_array_equal(features[:,ix+3],expected)
            np.testing.assert_array_equal(g['bbox_lo'],((mesh.lo-g['center'])/g['scale']).astype(np.float32))
            np.testing.assert_array_equal(g['bbox_hi'],((mesh.hi-g['center'])/g['scale']).astype(np.float32))

    def test_camera_commands_keep_order_and_do_not_mutate_native_start(self):
        before=copy.deepcopy(self.row['before']);untouched=copy.deepcopy(before)
        # Two native setter callbacks can share a step; the last wins in F32.
        control=dict(move=self.row['move'],events=[dict(kind='camera_yaw',value=1.123456789),
            dict(kind='camera_yaw',value=-2.234567891),dict(kind='camera_pitch',value=.123456789)])
        features,canonical=state_features(before,control,16,np.zeros(3),100.)
        self.assertEqual(before,untouched)
        yaw=float(np.float32(-2.234567891));pitch=float(np.float32(.123456789))
        self.assertEqual(canonical['yaw'],yaw);self.assertEqual(canonical['next_yaw'],yaw)
        self.assertEqual(canonical['pitch'],pitch)
        self.assertEqual(features[STATE_FEATURES.index('yaw_sin')],np.float32(np.sin(yaw)))

    def test_expiry_boundary_is_not_aliased_by_clamped_native_ttl(self):
        before=copy.deepcopy(self.row['before']);other=copy.deepcopy(before)
        for s in (before,other):s['effects'][5]=[True,0]
        before['effect_end_clock_ms'][5]=before['clock_ms']
        other['effect_end_clock_ms'][5]=other['clock_ms']-1
        control=dict(move=self.row['move'],events=[])
        a,_=state_features(before,control,16,np.zeros(3),100.)
        b,_=state_features(other,control,16,np.zeros(3),100.)
        ix=STATE_FEATURES.index('effect_signed_clock_left_5')
        self.assertNotEqual(a[ix],b[ix])
        # Expired/uninitialized INACTIVE timer clocks must not leak random junk.
        before['effects'][5]=[False,0];other['effects'][5]=[False,0]
        other['effect_end_clock_ms'][5]=4000000000
        a,_=state_features(before,control,16,np.zeros(3),100.)
        b,_=state_features(other,control,16,np.zeros(3),100.)
        np.testing.assert_array_equal(a,b)

    def test_scores_or_future_annotations_do_not_enter_physical_input(self):
        before=copy.deepcopy(self.row['before']);other=copy.deepcopy(before)
        other.update(game_score=500,map_id=999,future_position=[999,999,999],future_reward=1e12)
        control=dict(move=self.row['move'],events=[])
        a,_=state_features(before,control,16,np.zeros(3),100.)
        b,_=state_features(other,control,16,np.zeros(3),100.)
        np.testing.assert_array_equal(a,b)
        other['held']=6
        with self.assertRaises(ValueError):state_features(other,control,16,np.zeros(3),100.)

    def test_zero_residual_matches_known_drift_for_each_input_duration(self):
        model=KinematicResidualStudent().eval()
        x=torch.zeros((2,len(STATE_FEATURES)))
        x[:,3:6]=torch.tensor([[.4,-.8,.2],[-.1,.2,-.3]])
        x[:,STATE_FEATURES.index('dt_s')]=torch.tensor([.016,.008])
        x[:,STATE_FEATURES.index('map_scale')]=1.
        surface=torch.zeros((1,len(SURFACE_FEATURES)))
        with torch.inference_mode():prediction=model(x,surface,torch.zeros((1,3)),torch.ones((1,3)))
        expected=torch.zeros((2,9))
        expected[:,:3]=x[:,3:6]*25.*x[:,STATE_FEATURES.index('dt_s'):STATE_FEATURES.index('dt_s')+1]
        torch.testing.assert_close(prediction,expected,rtol=0,atol=0)
        self.assertTrue(bool(torch.isfinite(prediction).all()))

    def test_original_forward_cache_rejects_residual_variant(self):
        model=KinematicResidualStudent().eval().requires_grad_(False)
        surface=torch.zeros((1,len(SURFACE_FEATURES)))
        with self.assertRaisesRegex(ValueError,'original MotionStudent forward'):
            CompleteMapMotionCache(model,surface,torch.zeros((1,3)),torch.ones((1,3)))

    def test_relative_geometry_keeps_every_face_and_coordinate_origin_consistency(self):
        torch.manual_seed(123)
        model=RelativeGeometryResidualStudent().eval()
        # Nonzero readout exercises the geometry, not just the v*dt baseline.
        torch.nn.init.normal_(model.motion[-1].weight,std=.03)
        x=torch.zeros((2,len(STATE_FEATURES)));x[:,:3]=torch.tensor([[.01,.02,.003],[.05,-.02,.002]])
        x[:,3:6]=.1;x[:,STATE_FEATURES.index('map_scale')]=1.
        x[:,STATE_FEATURES.index('radius_map')]=.0018975;x[:,STATE_FEATURES.index('dt_s')]=.016
        surface=torch.rand((3,len(SURFACE_FEATURES)))*.1
        lo=torch.full((3,3),-.1);hi=torch.full((3,3),.1)
        original=[a.clone() for a in (x,surface,lo,hi)];encoded=[]
        hook=model.surface.register_forward_pre_hook(lambda module,args:encoded.append(args[0].shape))
        shift=torch.tensor([.125,-.25,.0625]);moved_x=x.clone();moved_x[:,:3]+=shift
        moved_surface=surface.clone()
        for start in (0,3,6,9,12,15,SURFACE_FEATURES.index('center_x')):
            moved_surface[:,start:start+3]+=shift
        ix=SURFACE_FEATURES.index('native_plane_d')
        moved_surface[:,ix]-=torch.sum(surface[:,ix-3:ix]*shift,dim=-1)
        with torch.inference_mode():
            expected=model(x,surface,lo,hi)
            actual=model(moved_x,moved_surface,lo+shift,hi+shift)
        hook.remove()
        self.assertEqual(encoded,[torch.Size([2,3,len(SURFACE_FEATURES)])]*2)
        torch.testing.assert_close(actual,expected,rtol=0,atol=1e-6)
        for before,after in zip(original,(x,surface,lo,hi)):torch.testing.assert_close(before,after,rtol=0,atol=0)

    def test_relative_geometry_zero_head_and_cache_guard(self):
        model=RelativeGeometryResidualStudent().eval().requires_grad_(False)
        x=torch.zeros((1,len(STATE_FEATURES)));x[:,3:6]=torch.tensor([[.4,-.8,.2]])
        x[:,STATE_FEATURES.index('map_scale')]=1.;x[:,STATE_FEATURES.index('radius_map')]=.0018975
        x[:,STATE_FEATURES.index('dt_s')]=.016;surface=torch.zeros((2,len(SURFACE_FEATURES)))
        with torch.inference_mode():pred=model(x,surface,torch.zeros((2,3)),torch.ones((2,3)))
        expected=torch.zeros((1,9));expected[:,:3]=x[:,3:6]*25.*.016
        torch.testing.assert_close(pred,expected,rtol=0,atol=0)
        with self.assertRaisesRegex(ValueError,'original MotionStudent forward'):
            CompleteMapMotionCache(model,surface,torch.zeros((2,3)),torch.ones((2,3)))


    def test_exact_triangle_context_face_edge_corner_and_degenerate(self):
        triangle=torch.tensor([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]],dtype=torch.float64)
        points=torch.tensor([[.2,.2,1.],[.6,-.4,.2],[2.,0.,0.]],dtype=torch.float64)
        closest,_=closest_triangle_context(points,triangle)
        torch.testing.assert_close(closest[:,0],torch.tensor([[.2,.2,0.],[.6,0.,0.],[1.,0.,0.]],dtype=torch.float64),rtol=0,atol=1e-12)
        degenerate=torch.tensor([[[0.,0.,0.],[1.,0.,0.],[1.,0.,0.]],[[0.,0.,0.]]*3],dtype=torch.float64)
        result,_=closest_triangle_context(points,degenerate)
        self.assertTrue(bool(torch.isfinite(result).all()))
        torch.testing.assert_close(result[0,0],torch.tensor([.2,0.,0.],dtype=torch.float64),rtol=0,atol=1e-12)
        torch.testing.assert_close(result[:,1],torch.zeros((3,3),dtype=torch.float64),rtol=0,atol=0)

    def test_source_boundary_feature_does_not_invent_fan_diagonal_or_missing_edge(self):
        x=torch.zeros((1,len(STATE_FEATURES)));x[:,:3]=torch.tensor([[.2,0.,.1]])
        x[:,STATE_FEATURES.index('radius_map')]=.1
        surface=torch.zeros((2,len(SURFACE_FEATURES)))
        triangle=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]).flatten()
        surface[:,:9]=triangle;surface[:,9:18]=triangle
        surface[0,-3:]=torch.tensor([0.,1.,0.]) # Only BC is a real boundary.
        features,distance=contact_geometry_features(x,surface)
        self.assertEqual(tuple(features.shape),(1,2,9));self.assertEqual(float(features[0,0,8]),1.)
        self.assertEqual(float(features[0,1,8]),0.)
        torch.testing.assert_close(features[0,1,4:8],torch.zeros(4),rtol=0,atol=0)
        self.assertGreater(float(features[0,0,7]),float(features[0,0,3]))
        torch.testing.assert_close(distance,torch.tensor([[.1,.1]]),rtol=0,atol=1e-7)

    def test_exact_context_agrees_with_reader_on_stored_geometry(self):
        mesh=CollisionMap(ML/'collision_maps/Concentric_Hunt.npz');g=geometry(mesh)
        indexes=np.unique(np.concatenate([np.arange(0,len(mesh.triangles),37),np.flatnonzero(np.linalg.norm(np.cross(mesh.triangles[:,1]-mesh.triangles[:,0],mesh.triangles[:,2]-mesh.triangles[:,0]),axis=1)<1e-12)]))
        triangles=torch.as_tensor(g['features'][indexes,:9].reshape(-1,3,3),dtype=torch.float64)
        point=(mesh.bounds.mean(axis=0)-g['center'])/g['scale']
        actual,_=closest_triangle_context(torch.as_tensor(point[None],dtype=torch.float64),triangles)
        expected=mesh.closest_points(point*g['scale']+g['center'],indexes)
        np.testing.assert_allclose(actual[0].numpy()*g['scale']+g['center'],expected,rtol=0,atol=2e-5)

    def test_contact_context_keeps_all_faces_and_origin_invariance(self):
        torch.manual_seed(123);model=ContactContextResidualStudent().eval()
        torch.nn.init.normal_(model.motion[-1].weight,std=.03)
        x=torch.zeros((2,len(STATE_FEATURES)));x[:,:3]=torch.tensor([[.01,.02,.003],[.05,-.02,.002]])
        x[:,3:6]=.1;x[:,STATE_FEATURES.index('map_scale')]=1.
        x[:,STATE_FEATURES.index('radius_map')]=.0018975;x[:,STATE_FEATURES.index('dt_s')]=.016
        surface=torch.rand((3,len(SURFACE_FEATURES)))*.1;surface[:,-3:]=torch.tensor([[1.,0.,1.],[0.,0.,0.],[0.,1.,0.]])
        lo=torch.full((3,3),-.1);hi=torch.full((3,3),.1)
        originals=[t.clone() for t in (x,surface,lo,hi)];encoded=[]
        hook=model.surface.register_forward_pre_hook(lambda module,args:encoded.append(args[0].shape))
        shift=torch.tensor([.125,-.25,.0625]);moved_x=x.clone();moved_x[:,:3]+=shift;moved_surface=surface.clone()
        for start in (0,3,6,9,12,15,SURFACE_FEATURES.index('center_x')):moved_surface[:,start:start+3]+=shift
        ix=SURFACE_FEATURES.index('native_plane_d');moved_surface[:,ix]-=torch.sum(surface[:,ix-3:ix]*shift,dim=-1)
        with torch.inference_mode():expected=model(x,surface,lo,hi);actual=model(moved_x,moved_surface,lo+shift,hi+shift)
        hook.remove();self.assertEqual(encoded,[torch.Size([2,3,len(SURFACE_FEATURES)+9])]*2)
        torch.testing.assert_close(actual,expected,rtol=0,atol=1e-6)
        for before,after in zip(originals,(x,surface,lo,hi)):torch.testing.assert_close(before,after,rtol=0,atol=0)

    def test_contact_zero_head_matches_actual_duration_and_cache_refuses_variant(self):
        model=ContactContextResidualStudent().eval().requires_grad_(False)
        x=torch.zeros((2,len(STATE_FEATURES)));x[:,3:6]=torch.tensor([[.4,-.8,.2],[-.1,.2,-.3]])
        x[:,STATE_FEATURES.index('map_scale')]=1.;x[:,STATE_FEATURES.index('radius_map')]=.0018975
        x[:,STATE_FEATURES.index('dt_s')]=torch.tensor([.016,.008]);surface=torch.zeros((2,len(SURFACE_FEATURES)))
        with torch.inference_mode():pred=model(x,surface,torch.zeros((2,3)),torch.ones((2,3)))
        expected=torch.zeros((2,9));expected[:,:3]=x[:,3:6]*25.*x[:,STATE_FEATURES.index('dt_s'):STATE_FEATURES.index('dt_s')+1]
        torch.testing.assert_close(pred,expected,rtol=0,atol=0)
        with self.assertRaisesRegex(ValueError,'original MotionStudent forward'):CompleteMapMotionCache(model,surface,torch.zeros((2,3)),torch.ones((2,3)))

    def test_ordinary_gate_checks_every_map_and_motion_component(self):
        record=dict(by_stratum={'0':dict(mean_error=[1.,2.,3.],constant_velocity_mean_error=[1.,2.,3.])})
        dev=dict(maps=[copy.deepcopy(record),copy.deepcopy(record)])
        self.assertTrue(ordinary_motion_eligible(dev))
        for component in range(3):
            worse=copy.deepcopy(dev);worse['maps'][1]['by_stratum']['0']['mean_error'][component]*=1.06
            self.assertFalse(ordinary_motion_eligible(worse))
        self.assertFalse(ordinary_motion_eligible({'maps':[]}))


class PairedNativeSupervisionContracts(unittest.TestCase):
    def test_default_loss_is_exactly_the_original_marginal_mse(self):
        prediction=torch.arange(18,dtype=torch.float32).reshape(2,9)
        target=prediction*.7
        self.assertTrue(torch.equal(native_motion_loss(prediction,target),torch.mean((prediction-target)**2)))

    def test_action_difference_loss_cancels_shared_motion_bias(self):
        target=torch.arange(36,dtype=torch.float32).reshape(2,2,9)/20
        prediction=target+3
        ordinary=torch.zeros(1,9)
        self.assertAlmostEqual(float(native_motion_loss(ordinary,ordinary,prediction,target)),0.,places=10)
        prediction=prediction.clone();prediction[:,1,2]+=1
        self.assertAlmostEqual(float(native_motion_loss(ordinary,ordinary,prediction,target)),1/9,places=6)

    def test_incomplete_or_malformed_pair_loss_refuses(self):
        ordinary=torch.zeros(1,9)
        with self.assertRaises(ValueError):native_motion_loss(ordinary,ordinary,ordinary,None)
        with self.assertRaises(ValueError):native_motion_loss(ordinary,ordinary,ordinary,ordinary)

    def example(self):
        samples=json.loads((Path(__file__).with_name('fixtures')/'native_regressions.json').read_text())['samples']
        row=next(s['row'] for s in samples if s['row']['dt_ms']==16 and s['row']['before']['mode']==0 and
            s['row']['after']['mode']==0 and not s['row']['before']['oob'] and not s['row']['after']['oob'])
        row.setdefault('control_events',[])  # Older native fixture has no camera callbacks.
        other=copy.deepcopy(row);other['move']['jump']=not row['move']['jump']
        pair=dict(passed=True,restored_field_equality=True,initial_tick_phase_equal=True,
            first_changed_native_move=0,branches=[dict(path='a.gz',compressed_sha256='archive',native_sha256=hashlib.sha256(b'raw').hexdigest()),
                dict(path='b.gz',compressed_sha256='archive',native_sha256=hashlib.sha256(b'raw').hexdigest())])
        return pair,[row],[other],dict(center=np.zeros(3),scale=100.)

    def run_example(self,pair,a,b,geo):
        with patch('nav.marble_dynamics.pairs.sha',return_value='archive'),patch.object(Path,'read_bytes',return_value=b'packed'),\
            patch('gzip.decompress',return_value=b'raw'),patch('nav.marble_dynamics.pairs.load_rows',side_effect=[a,b]):
            return paired_motion_example(pair,Path('.'),geo)

    def test_future_motion_changes_labels_but_never_inputs(self):
        pair,a,b,geo=self.example();x,y,_=self.run_example(pair,a,b,geo)
        other=copy.deepcopy(b);other[0]['after']['velocity'][0]+=5
        x2,y2,_=self.run_example(pair,a,other,geo)
        np.testing.assert_array_equal(x,x2);self.assertFalse(np.array_equal(y,y2))

    def test_actual_before_mismatch_or_delayed_move_misalignment_refuses(self):
        pair,a,b,geo=self.example();bad=copy.deepcopy(b);bad[0]['before']['spin'][0]+=1
        with self.assertRaisesRegex(ValueError,'physical BEFORE'):self.run_example(pair,a,bad,geo)
        pair['first_changed_native_move']=1
        with self.assertRaisesRegex(ValueError,'not aligned'):self.run_example(pair,a,b,geo)

    def test_failed_world_event_or_restore_cannot_supply_pair_labels(self):
        pair,a,b,geo=self.example()
        for key,value in [('passed',False),('game_events_observed',True),('restored_field_equality',False)]:
            bad=dict(pair);bad[key]=value
            with self.assertRaises(ValueError):self.run_example(bad,a,b,geo)


class ObservedWorldTaskContracts(unittest.TestCase):
    def observations(self):
        row=json.loads((Path(__file__).with_name('fixtures')/'native_regressions.json').read_text())['samples'][0]['row']
        state=copy.deepcopy(row['before']);state['script_context']=dict(authority=dict(held_item_power_id=6))
        current=dict(native=dict(frame_clock_ms=100,state=state),bridge_action=[0,0,0,0,0,0,1,0,0])
        future=copy.deepcopy(current);future['native']['frame_clock_ms']=116;future['native']['state']['radius']*=3
        future['bridge_action']=[0]*9
        return current,future

    def test_future_body_motion_never_change_current_inputs(self):
        current,future=self.observations();a=observed_world_example(current,future,[0]*9)
        changed=copy.deepcopy(future);changed['native']['state']['velocity'][0]+=20
        b=observed_world_example(current,changed,[0]*9)
        self.assertEqual(a['input'],b['input']);self.assertNotEqual(a['target'],b['target'])
        self.assertTrue(a['audit_only']['body_transition'])
        current['native']['state']['radius']+=1
        self.assertNotEqual(a['input']['current_observed_dynamics'],current)

    def test_pending_use_and_new_neutral_reply_remain_distinct(self):
        current,future=self.observations();a=observed_world_example(current,future,[0]*9)
        self.assertEqual(a['input']['current_observed_dynamics']['bridge_action'][6],1)
        self.assertEqual(a['input']['new_reply'][6],0)
        self.assertEqual(a['input']['current_observed_dynamics']['native']['state']['script_context']['authority']['held_item_power_id'],6)

    def test_phase_missing_context_and_nonfinite_reply_refuse(self):
        current,future=self.observations();bad=copy.deepcopy(future);bad['native']['frame_clock_ms']+=16
        with self.assertRaises(ValueError):observed_world_example(current,bad,[0]*9)
        bad=copy.deepcopy(current);del bad['native']['state']['script_context']
        with self.assertRaises(ValueError):observed_world_example(bad,future,[0]*9)
        with self.assertRaises(ValueError):observed_world_example(current,future,[float('nan')]+[0]*8)

    def test_future_fall_is_a_target_and_not_silently_excluded(self):
        current,future=self.observations();future['native']['state']['oob']=True
        example=observed_world_example(current,future,[0]*9)
        self.assertTrue(example['target']['next_observed_dynamics']['native']['state']['oob'])
        self.assertTrue(example['audit_only']['next_oob'])
        self.assertFalse(example['input']['current_observed_dynamics']['native']['state']['oob'])
        with self.assertRaises(ValueError):observed_world_example(future,future,[0]*9)


class ObservableTokenContracts(unittest.TestCase):
    def observation(self):
        row=json.loads((Path(__file__).with_name('fixtures')/'native_regressions.json').read_text())['samples'][0]['row']
        state=copy.deepcopy(row['before']);state['clock_ms']=100;state['effects']=[[False,0] for _ in range(6)]
        body=dict(object_id=99,native_held=0,held_item='MegaMarbleItem',held_item_power_id=6,raw_held_datablock='987',
            active_flags=[False]*7,effect_schedules=[dict(id=900+i,pending=False,remaining_ms=0) for i in range(7)],
            mega_schedule=dict(id=989,pending=False,remaining_ms=0))
        state['held']=0;state['script_context']=dict(schema='observed_local_script_v1',target=copy.deepcopy(body),authority=copy.deepcopy(body),
            bridge_control=dict(schema='observed_bridge_control_v1',power_use_latched=False,power_yaw_hold_ticks=0,power_yaw=0.,
                configured_yaw_hold_ticks=12,ss_yaw_fix=False,fast_powerups=False,live=False))
        identity=np.eye(4).reshape(-1).tolist()
        state['native_context']=dict(schema='observed_native_context_v1',gravity_up=[0,0,1],previous_gravity_direction=[0,0,-1],
            global_gravity_matrix=identity,previous_gravity_matrix=identity,default_gravity_transform=identity,
            gravity_interpolation_active=False,gravity_interpolation=None,
            pending_update=dict(types=0,velocity=None,spin=None,transform=None,camera=None,gravity=None,radius=None),
            current_contacts=[],best_contact=None,max_force_radius=50,runtime_mass_observed=False,last_contact_decoded=False)
        move={k:0 for k in ('x','y','z','yaw','pitch','jump','use','trigger1')}
        return dict(native=dict(frame_clock_ms=100,state=state,last_move=copy.deepcopy(move),pending_move=copy.deepcopy(move)),
            bridge_action=[0]*9,input_move=[0]*4)

    def tokens(self,observation):return observable_state_tokens(observation,[0.,0.,0.],100.)

    def test_object_event_ids_and_inactive_garbage_are_not_features(self):
        a=self.observation();b=copy.deepcopy(a);b['native']['object_id']=123456
        b['native']['state']['effect_end_clock_ms']=[float('nan')]*6
        for side in ('target','authority'):
            s=b['native']['state']['script_context'][side];s['object_id']=555;s['raw_held_datablock']='different';s['held_item']='differentName'
            s['mega_schedule'].update(id=789,remaining_ms=float('nan'))
            for schedule in s['effect_schedules']:schedule.update(id=789,remaining_ms=float('nan'))
        self.assertEqual(self.tokens(a),self.tokens(b))

    def test_script_mega_is_distinct_from_empty_native_inventory(self):
        a=self.observation();b=copy.deepcopy(a)
        for side in ('target','authority'):b['native']['state']['script_context'][side]['held_item_power_id']=0
        ta,tb=self.tokens(a),self.tokens(b)
        self.assertEqual(ta['body'],tb['body']);self.assertNotEqual(ta['inventory'],tb['inventory'])

    def test_latch_distinguishes_identical_physical_states(self):
        a=self.observation();b=copy.deepcopy(a);b['native']['state']['script_context']['bridge_control']['power_use_latched']=True
        ta,tb=self.tokens(a),self.tokens(b);self.assertEqual(ta['body'],tb['body']);self.assertNotEqual(ta['controls'],tb['controls'])

    def test_future_motion_and_falls_are_targets_only(self):
        a=self.observation();b=copy.deepcopy(a);b['native']['frame_clock_ms']+=16;b['native']['state']['clock_ms']+=16
        first=observable_context_example(a,b,[0]*9,[0]*3,100.)
        b['native']['state']['velocity'][0]+=25;b['native']['state']['oob']=True
        second=observable_context_example(a,b,[0]*9,[0]*3,100.)
        self.assertEqual(first['input'],second['input']);self.assertNotEqual(first['target'],second['target'])
        self.assertEqual(second['target']['next']['body']['oob'],1.)

    def test_active_deadline_uses_relative_time_with_expiry_sign(self):
        a=self.observation();s=a['native']['state'];s['effects'][5]=[True,0];s['effect_end_clock_ms'][5]=99
        self.assertLess(self.tokens(a)['body']['native_effects'][5][2],0.)
        b=copy.deepcopy(a);b['native']['frame_clock_ms']+=32000;b['native']['state']['clock_ms']+=32000
        b['native']['state']['effect_end_clock_ms'][5]+=32000
        self.assertEqual(self.tokens(a),self.tokens(b))

    def test_active_pending_update_is_distinct_and_inactive_values_are_ignored(self):
        a=self.observation();b=copy.deepcopy(a);u=b['native']['state']['native_context']['pending_update'];u['radius']=float('nan')
        self.assertEqual(self.tokens(a),self.tokens(b));u.update(types=32,radius=.6666)
        self.assertNotEqual(self.tokens(a),self.tokens(b))
        self.assertAlmostEqual(self.tokens(b)['pending_update']['radius'][0],.6666/.18975)

    def test_contact_material_and_spin_are_retained(self):
        a=self.observation();b=copy.deepcopy(a);s=b['native']['state'];s['spin'][0]+=100
        c=dict(position=list(s['position']),normal=[0,0,1],surface_velocity=[0,0,0],distance=s['radius'],normal_force=20,friction=1,restitution=1,force=0)
        s['native_context']['current_contacts']=[c,copy.deepcopy(c)]
        t=self.tokens(b);self.assertEqual(len(t['contacts']),2);self.assertNotEqual(self.tokens(a)['body']['spin'],t['body']['spin'])
        old=t['contacts'];s['native_context']['current_contacts'][0]['friction']=3
        self.assertNotEqual(old,self.tokens(b)['contacts'])

    def test_missing_unknown_or_nonfinite_observed_state_refuses(self):
        a=self.observation();b=copy.deepcopy(a);del b['native']['state']['native_context']
        with self.assertRaises(ValueError):self.tokens(b)
        b=copy.deepcopy(a);b['native']['state']['script_context']['authority']['held_item_power_id']=9
        with self.assertRaises(ValueError):self.tokens(b)
        b=copy.deepcopy(a);b['native']['state']['native_context']['pending_update']['types']=64
        with self.assertRaises(ValueError):self.tokens(b)
        b=copy.deepcopy(a);b['native']['state']['spin'][0]=float('nan')
        with self.assertRaises(ValueError):self.tokens(b)
        b=copy.deepcopy(a);b['native']['state']['native_context']['pending_update'].update(types=32,radius=None)
        with self.assertRaises(ValueError):self.tokens(b)


if __name__=='__main__':unittest.main()
