"""Scaling infrastructure must preserve exact geometry, provenance and quotas."""
import copy
from collections import OrderedDict
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch,Mock

import numpy as np
from scipy.sparse import csr_matrix

from nav.collision.mesh import atomic_json
from nav.collision.test_queries import fixture
from nav.marble_dynamics.fast_nav import load_advice,collection_integrity,ensure_case_journal
from nav.marble_dynamics.record import sha
from nav.marble_dynamics.scaling_loop import check_recipe,strip_options,checked_move
from nav.marble_dynamics.static_cache import write_cache,read_cache,cached_walking,guarded_launch
from nav.marble_dynamics.test_query_cache import cached
from nav.marble_dynamics.reliability_audit import full_round_completion


class ScalingContracts(unittest.TestCase):
    def test_input_identity_is_independent_of_journal_and_signed_zero(self):
        from nav.marble_dynamics.fast_nav import advice_state_key
        q=np.zeros((4,3),np.float32);a=np.asarray([0.,1.],np.float32);b=a.copy();b[0]=-0.
        self.assertEqual(advice_state_key(a,q),advice_state_key(b,q))
        changed=q.copy();changed[0,0]=np.nextafter(np.float32(0),np.float32(1))
        self.assertNotEqual(advice_state_key(a,q),advice_state_key(a,changed))
        with self.assertRaises(ValueError):advice_state_key(np.asarray([np.nan]),q)

    def correction_fixture(self,root,tag,inputs):
        import gzip
        folder=root/tag;folder.mkdir();xs=np.asarray(inputs,np.float32).reshape(-1,2)
        np.savez_compressed(folder/'map.npz',x=xs,q=np.zeros((len(xs),4,3),np.float32),
            y=np.zeros((len(xs),9),np.float32),decision=np.arange(len(xs),dtype=np.int32))
        with gzip.open(folder/'cases.jsonl.gz','wt') as stream:
            for i in range(len(xs)):
                stream.write(json.dumps(dict(map='map',decision=i,source_sha256=tag,accepted=True))+'\n')
        manifest=dict(complete=True,first_reply_causal_corrections=True,sources={},cases_path='cases.jsonl.gz',
            maps={'map':dict(accepted=len(xs),tensor_sha256=sha(folder/'map.npz'))})
        atomic_json(folder/'manifest.json',manifest);atomic_json(folder/'protocol.json',{})
        return folder

    def test_cross_journal_duplicates_cannot_satisfy_new_or_map_quota(self):
        from nav.marble_dynamics.scaling_loop import distinct_correction_counts,merge_corrections,read_cases
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);dataset=root/'dataset';dataset.mkdir()
            atomic_json(dataset/'manifest.json',dict(maps=['map'],state_size=2))
            old=self.correction_fixture(root,'old',[[0,0]])
            first=self.correction_fixture(root,'first',[[0,0],[1,1]])
            another=self.correction_fixture(root,'another',[[1,1],[2,2]])
            self.assertEqual(distinct_correction_counts([first,another],['map'],[old]),{'map':2})
            with patch('nav.marble_dynamics.fast_nav.protected_hashes',return_value={}),patch('nav.source_archive.snapshot_sources'):
                result=merge_corrections([first,another],dataset,root/'merged',3,3,prior=[old])
            self.assertFalse(result['quota_met']);self.assertEqual(result['accepted'],2)
            self.assertEqual(result['duplicate_inputs_excluded_by_map'],{'map':2})
            with np.load(root/'merged/map.npz') as z:np.testing.assert_array_equal(z['x'],[[1,1],[2,2]])
            cases=list(read_cases(root/'merged/map.cases.jsonl.gz'))
            self.assertEqual(sum(c['accepted'] for c in cases),2)
            self.assertEqual(sum(c.get('accepted_in_worker',False) for c in cases),2)
            self.assertTrue((another/'cases.jsonl.gz').exists())

    def test_duplicate_inflated_old_run_does_not_count_toward_four_scaled_loops(self):
        from nav.marble_dynamics.scaling_loop import no_gain_stop
        protocol=dict(minimum_scaled_loops=4,development_reference=dict(maps=['map'],student_mean=57,a2_mean=44))
        initial=dict(equal_map=dict(student_points=41.8))
        row=dict(table=dict(per_map=[dict(map='map',student_points=56)],equal_map=dict(student_points=40)))
        invalid=dict(row,qualifying_scaled_loop=False)
        self.assertFalse(no_gain_stop(protocol,initial,[invalid,row,row,row]))
        self.assertTrue(no_gain_stop(protocol,initial,[invalid,row,row,row,row]))

    def test_sampled_identity_recovery_includes_rejected_queries_and_is_cached(self):
        import gzip
        from nav.marble_dynamics.fast_nav import advice_state_key
        from nav.marble_dynamics.scaling_loop import queried_state_keys
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);dataset=root/'dataset';dataset.mkdir()
            np.savez_compressed(dataset/'map.npz',center=np.zeros(3),scale=np.asarray(1.))
            data=self.correction_fixture(root,'corrections',[]);raw=root/'raw';raw.mkdir()
            native=root/'pool/map/1/fast_navigator/1';native.mkdir(parents=True)
            rows=[dict(decision=i,before={},raw_observation=[i],choice=dict(base_actor_action=[0]*9,
                actor_goal_context=dict(destination=[0]*3,waypoint=[0]*3))) for i in (1,2)]
            with gzip.open(native/'decisions.jsonl.gz','wt') as stream:
                for row in rows:stream.write(json.dumps(row)+'\n')
            atomic_json(root/'pool/summary.json',dict(branches=[dict(map='map',seed=1,branch='fast_navigator',round=1)]))
            with gzip.open(raw/'cases.jsonl.gz','wt') as stream:
                for row in rows:stream.write(json.dumps(dict(map='map',decision=row['decision'],accepted=row['decision']==1))+'\n')
            atomic_json(raw/'manifest.json',dict(cases_path='cases.jsonl.gz'))
            atomic_json(raw/'protocol.json',dict(rollouts=str(root/'pool')))
            manifest=json.loads((data/'manifest.json').read_text());manifest['parent_advice']=str(raw);atomic_json(data/'manifest.json',manifest)
            def inputs(before,obs,*args,**kwargs):return np.asarray([obs[0]],np.float32),np.zeros((4,3),np.float32)
            with patch('nav.marble_dynamics.fast_nav.features',side_effect=inputs):
                actual=queried_state_keys([data],'map',dataset,root/'receipts')
            expected=sorted(advice_state_key(np.asarray([i]),np.zeros((4,3))) for i in (1,2))
            self.assertEqual(actual,expected)
            with patch('nav.marble_dynamics.fast_nav.features',side_effect=AssertionError('unchanged recovery must be reused')):
                self.assertEqual(queried_state_keys([data],'map',dataset,root/'receipts'),expected)

    def test_zero_accepted_forecasts_have_a_readable_empty_journal(self):
        from nav.marble_dynamics.scaling_loop import read_cases
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);ensure_case_journal(folder)
            self.assertEqual(list(read_cases(folder/'cases.jsonl.gz')),[])
            digest=sha(folder/'cases.jsonl.gz');ensure_case_journal(folder)
            self.assertEqual(sha(folder/'cases.jsonl.gz'),digest)

    def test_cache_preserves_float_bits_sparse_order_and_exact_keys(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'cache.bin'
            value=dict(array=np.asarray([-0.,np.nextafter(1.,2.),np.inf,np.nan]),
                graph=csr_matrix(([.3,.7],[1,0],[0,1,2]),shape=(2,2)),
                keys=OrderedDict([((.189750001,0.,0.,0.,1.,2.,3.),True)]))
            write_cache(path,'input_A',value);actual=read_cache(path,'input_A')
            self.assertEqual(actual['array'].tobytes(),value['array'].tobytes())
            for key in ('data','indices','indptr'):
                self.assertEqual(getattr(actual['graph'],key).tobytes(),getattr(value['graph'],key).tobytes())
            self.assertEqual(actual['keys'],value['keys']);self.assertIsNone(read_cache(path,'input_B'))
            raw=bytearray(path.read_bytes());raw[-1]^=1;path.write_bytes(raw)
            with self.assertRaises(ValueError):read_cache(path,'input_A')

    def test_loaded_broadphase_cache_still_rechecks_requested_box(self):
        below=float(np.nextafter(-1000.,-np.inf))
        mesh=fixture([[[below,0,0],[below,.05,0],[below,0,.05]]],[[1,0,0]])
        query=cached(mesh,.1);query.query_aabb([-1000.,0,0],[-1000.,.1,.1])
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'query.bin';write_cache(path,'same_geometry',query.query_regions)
            restored=cached(mesh,.1);restored.query_regions=read_cache(path,'same_geometry')
            np.testing.assert_array_equal(restored.query_aabb([below,0,0],[-1000.,.1,.1]),mesh.query_aabb([below,0,0],[-1000.,.1,.1]))

    def test_walking_cache_loads_filtered_graph_without_rebuilding(self):
        base=SimpleNamespace(static_cache_key='terrain_A')
        graph=csr_matrix(([1.25,2.5],[1,0],[0,1,2]),shape=(2,2))
        class Clearance:
            def __init__(self,mesh):self.radius=.189750001;self.point_cache={}
        result=copy.copy(base);result._graph=result._graph_nojump=graph
        result.component=np.asarray([[[0,0]]],dtype=np.int64)
        result.ground_clearance=Clearance(None);result.ground_clearance.point_cache={(1.,2.,3.):True}
        result.geometry_route_report=dict(construction_s=110.)
        builder=Mock(return_value=result)
        with tempfile.TemporaryDirectory() as temp,patch('nav.marble_dynamics.static_cache.fingerprint',return_value='graph_A'),patch('nav.marble_dynamics.static_cache.mesh_key',return_value='mesh_A'),patch('nav.marble_dynamics.geometry_route.walking_terrain',builder),patch('nav.marble_dynamics.geometry_route.GroundClearance',Clearance):
            first=cached_walking(base,None,None,temp);second=cached_walking(base,None,None,temp)
            self.assertEqual(builder.call_count,1)
            np.testing.assert_array_equal(first._graph.data,second._graph.data)
            np.testing.assert_array_equal(first._graph.indices,second._graph.indices)
            np.testing.assert_array_equal(first.component,second.component)
            self.assertTrue(second.geometry_route_report['persistent_graph_cache_hit'])
            self.assertEqual(second.ground_clearance.original.point_cache,{(1.,2.,3.):True})

    def test_native_guard_counts_non_owned_games_and_never_terminates_them(self):
        eight=[dict(pid=n,name='marbleblast.exe') for n in range(8)]
        process=Mock()
        with patch('nav.marble_dynamics.static_cache.file_lock',return_value=nullcontext()),patch('nav.marble_dynamics.static_cache.marble_processes',side_effect=[eight,eight[:7]]),patch('nav.marble_dynamics.static_cache.time.sleep'),patch('subprocess.Popen',return_value=process) as launch:
            self.assertIs(guarded_launch(['game'],8,cwd='workspace'),process)
            launch.assert_called_once_with(['game'],cwd='workspace');process.terminate.assert_not_called()
        with self.assertRaises(ValueError):guarded_launch(['game'],9)

    def test_correction_replay_is_pending_until_explicitly_authorized(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            np.savez_compressed(folder/'map.npz',x=np.zeros((1,109),np.float32),q=np.zeros((1,4,3),np.float32),y=np.zeros((1,9),np.float32))
            manifest=dict(schema='analytical_navigation_advice_v1',complete=True,cases_sha256='recorded',maps={'map':{'tensor_sha256':sha(folder/'map.npz')}})
            atomic_json(folder/'manifest.json',manifest);collection_integrity(folder,manifest)
            interface=dict(maps=['map'],state_size=109)
            with self.assertRaises(FileNotFoundError):load_advice(folder,interface)
            self.assertEqual(len(load_advice(folder,interface,defer_replay=True)['map']['y']),1)
            manifest['complete']=False;atomic_json(folder/'manifest.json',manifest)
            with self.assertRaises(ValueError):load_advice(folder,interface,defer_replay=True)

    def test_frozen_recipe_detects_changed_loss_or_architecture(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'recipe.py';path.write_text('def loss(x):\n    return x * .4\n')
            import ast
            digest=hashlib.sha256(ast.dump(ast.parse(path.read_text()).body[0],include_attributes=False).encode()).hexdigest()
            protocol={'recipe_file':str(path),'recipe_ast_sha256':{'loss':digest}}
            self.assertTrue(check_recipe(protocol));path.write_text('def loss(x):\n    return x * .3\n')
            with self.assertRaises(ValueError):check_recipe(protocol)

    def test_worker_arguments_do_not_duplicate_multi_value_options(self):
        original=['--maps','one','two','--seeds','1','2','--full-round','--workers','8','--output','old','--horizon','24']
        self.assertEqual(strip_options(original,{'--maps','--seeds','--workers','--output'}),['--full-round','--horizon','24'])

    def test_evidence_move_refuses_path_escape_and_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'pool';root.mkdir();source=root/'source';source.mkdir()
            with self.assertRaises(ValueError):checked_move(source,Path(temp)/'outside',root)
            target=root/'target';target.mkdir()
            with self.assertRaises(ValueError):checked_move(source,target,root)
            target.rmdir();checked_move(source,target,root);self.assertTrue(target.exists())

    def test_teacher_only_pool_requires_all_declared_rounds(self):
        prepared=dict(maps=['map'],seeds=[1],rounds=1,only_branch='physics_plan')
        checks=[dict(map='map',seed=1,round=1,branch='physics_plan')]
        self.assertTrue(full_round_completion(prepared,checks)['complete_declared_rounds'])
        self.assertFalse(full_round_completion(prepared,[])['complete_declared_rounds'])



    def test_four_loop_stop_requires_no_development_gain(self):
        from nav.marble_dynamics.scaling_loop import no_gain_stop,gain_summary
        protocol=dict(minimum_scaled_loops=4,development_reference=dict(maps=['one'],student_mean=57,a2_mean=44))
        initial=dict(equal_map=dict(student_points=41.8))
        def record(score):return dict(table=dict(per_map=[dict(map='one',student_points=score)],equal_map=dict(student_points=60)))
        rows=[record(56)]*4
        self.assertFalse(no_gain_stop(protocol,initial,rows[:3]))
        self.assertTrue(no_gain_stop(protocol,initial,rows))
        self.assertTrue(gain_summary(protocol,initial,rows)['gain_over_initial_ten_map'])
        self.assertFalse(no_gain_stop(protocol,initial,[record(58),*rows[1:]]))

    def test_closed_collection_reuses_only_matching_final_integrity(self):
        from nav.marble_dynamics.scaling_loop import completed_collection
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);data=root/'workers/map/corrections';data.mkdir(parents=True)
            manifest=dict(complete=True,first_reply_causal_corrections=True,maps={'map':dict(accepted=2)})
            atomic_json(data/'manifest.json',manifest)
            atomic_json(data/'collection_integrity.json',dict(passed=True,manifest_sha256=sha(data/'manifest.json')))
            result=dict(complete=True,maps=[dict(map='map',corrections_path=str(data),corrections=dict(accepted=2))])
            atomic_json(root/'result.json',result)
            self.assertEqual(completed_collection(root),result)
            manifest['maps']['map']['accepted']=3;atomic_json(data/'manifest.json',manifest)
            with self.assertRaises(ValueError):completed_collection(root)

    def test_resume_refuses_incomplete_collection_without_overwrite(self):
        from nav.marble_dynamics.scaling_loop import completed_collection
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);atomic_json(root/'result.json',dict(complete=False,maps=[]))
            before=(root/'result.json').read_bytes()
            with self.assertRaises(ValueError):completed_collection(root)
            self.assertEqual((root/'result.json').read_bytes(),before)

    def test_iteration_state_preserves_parent_and_rejects_changed_recipe(self):
        from nav.marble_dynamics.scaling_loop import write_loop_state,resume_loop_state
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);parent=root/'parent.pth';parent.write_bytes(b'fixed-parent')
            args=SimpleNamespace(output=root);protocol=dict(recipe_ast_sha256={'loss':'unchanged'})
            write_loop_state(args,protocol,parent,root/'latest',['confirmed'],['old','new'],[{}]*4,42,{'Bowl':5},5)
            state=resume_loop_state(args,protocol)
            self.assertEqual(state['next_iteration'],5);self.assertEqual(state['all_correction_folders'],['old','new'])
            with self.assertRaises(ValueError):resume_loop_state(args,dict(recipe_ast_sha256={'loss':'changed'}))
            parent.write_bytes(b'changed-parent')
            with self.assertRaises(ValueError):resume_loop_state(args,protocol)

    def test_initial_resume_requires_model_audit_and_suite_receipts(self):
        from nav.marble_dynamics.scaling_loop import initial_confirmation_receipts
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);checkpoint=root/'parent.pth';checkpoint.write_bytes(b'parent')
            pool=root/'confirmation';pool.mkdir();args=SimpleNamespace(output=root,initial_checkpoint=checkpoint)
            atomic_json(pool/'source_integrity.json',dict(changed=[],source_hashes=10))
            audit=dict(complete_pool=True,complete_declared_rounds=True,rounds=[dict(branch='fast_navigator',
                fast_model_output_replay=dict(passed=True,checkpoint_sha256=sha(checkpoint)))])
            atomic_json(pool/'full_round_audit.json',audit)
            (root/'tests_confirmation.log').write_text('suite incomplete')
            with self.assertRaises(ValueError):initial_confirmation_receipts(args,[pool])
            (root/'tests_confirmation.log').write_text('Ran 196 tests\n\nOK\n')
            initial_confirmation_receipts(args,[pool])
            audit['rounds'][0]['fast_model_output_replay']['passed']=False;atomic_json(pool/'full_round_audit.json',audit)
            with self.assertRaises(ValueError):initial_confirmation_receipts(args,[pool])


    def test_collection_schedules_measured_slow_jobs_first_without_changing_jobs(self):
        from nav.marble_dynamics.scaling_loop import order_collection_jobs
        jobs=[dict(map=n,attempts=1200) for n in ('fast','slow','medium','unknown')]
        actual=order_collection_jobs(jobs,{'fast':.2,'slow':.7,'medium':.4,'unknown':float('nan')})
        self.assertEqual([j['map'] for j in actual],['slow','medium','fast','unknown'])
        self.assertEqual(sorted(jobs,key=lambda j:j['map']),sorted(actual,key=lambda j:j['map']))


class FailureAblationContracts(unittest.TestCase):
    def test_safety_resume_replays_exact_original_sampler_cursor(self):
        import torch
        from nav.marble_dynamics.fast_nav import ablation_batch_indices,advice_indices
        names=['large','tiny'];data={n:dict(train=np.arange(57+i)) for i,n in enumerate(names)}
        extra={n:dict(y=torch.zeros(size,9)) for n,size in zip(names,[25,2])}
        extra['large']['y'][1,7]=1;extra['large']['y'][2,8]=1
        def initialized():
            rng=np.random.RandomState(101313)
            for n in names:rng.choice(data[n]['train'],min(512,len(data[n]['train'])),replace=False)
            return rng
        original=initialized();replayed=initialized()
        for update in range(37):
            n=names[update%len(names)];ix=original.choice(data[n]['train'],36,replace=False)
            ax=advice_indices(extra[n],original,12,True)
            actual=ablation_batch_indices(data,extra,names,update,replayed)
            self.assertEqual(actual[0],n);np.testing.assert_array_equal(actual[1],ix)
            np.testing.assert_array_equal(actual[2],ax)
        np.testing.assert_array_equal(original.get_state()[1],replayed.get_state()[1])
        self.assertEqual(original.get_state()[2:],replayed.get_state()[2:])

    def test_resume_recovers_all_closed_remaining_batches_before_needy_maps(self):
        from nav.marble_dynamics.scaling_loop import recover_remaining_collections
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for n in (10,1,2):(root/('iteration_3_remaining_states_'+str(n)+'_corrections')).mkdir()
            (root/'iteration_2_remaining_states_1_corrections').mkdir()
            with patch('nav.marble_dynamics.scaling_loop.completed_collection',side_effect=lambda p:dict(path=str(p))):
                batches=recover_remaining_collections(root,3)
            self.assertEqual([n for n,_,_ in batches],[1,2,10])
            self.assertEqual(len({str(p) for _,p,_ in batches}),3)

    def test_loss_ratio_does_not_depend_on_anchor_batch_occupancy(self):
        import torch
        from nav.marble_dynamics.fast_nav import explicit_preservation_loss
        x=torch.zeros(3,109);y=torch.zeros(3,9);prediction=torch.zeros(3,9)
        expected=explicit_preservation_loss(prediction,x,y,2)
        def repeat_anchors(t):return torch.cat([t[:2].repeat(8,1),t[2:]])
        self.assertTrue(torch.isclose(expected,explicit_preservation_loss(
            repeat_anchors(prediction),repeat_anchors(x),repeat_anchors(y),16)))

    def test_monitor_matches_positive_action_sampling_without_live_floor(self):
        import torch
        from nav.marble_dynamics.fast_nav import advice_monitor_weights
        y=torch.zeros(100,9);y[1,7]=1;y[2,8]=1
        weight=advice_monitor_weights(y)
        self.assertAlmostEqual(float(weight.sum()),1.,places=6)
        self.assertAlmostEqual(float(weight[1]-weight[0]),2/12,places=6)
        self.assertTrue(torch.allclose(advice_monitor_weights(torch.zeros(100,9)),torch.ones(100)/100))

    def test_plateau_requires_repeated_checks_and_resets_on_improvement(self):
        from nav.marble_dynamics.fast_nav import plateau_step
        best,stale=.5,0
        for i in range(5):
            best,stale=plateau_step(best,stale,.501)
            self.assertEqual(stale,i+1)
        self.assertEqual(plateau_step(best,stale,.49),(.49,0))
        with self.assertRaises(ValueError):plateau_step(best,stale,float('nan'))

    def test_native_bounds_keep_lower_decks_and_refuse_unmodeled_rotation(self):
        from nav.marble_dynamics.fast_nav import mission_inbounds,fully_outside_inbounds
        volume=dict(datablock='InBoundsTrigger',position=[-10,10,-5],rotation='1 0 0 0',
            scale='20 20 100',polyhedron='0 0 0 1 0 0 0 -1 0 0 0 1')
        bounds=mission_inbounds(dict(mission_annotations=[volume]))
        self.assertFalse(fully_outside_inbounds(dict(position=[0,0,-4],radius=.2),bounds))
        self.assertFalse(fully_outside_inbounds(dict(position=[0,0,-5.1],radius=.2),bounds))
        self.assertTrue(fully_outside_inbounds(dict(position=[0,0,-5.3],radius=.2),bounds))
        volume['rotation']='0 0 1 45'
        with self.assertRaises(ValueError):mission_inbounds(dict(mission_annotations=[volume]))

    def test_ambiguity_cannot_become_a_fall_label(self):
        from nav.marble_dynamics.fast_nav import classify_student_failure
        start=dict(position=[0,0,1],velocity=[0,0,0],radius=.2,gravity_direction=[0,0,-1])
        end=dict(start,position=[0,0,-10],velocity=[0,0,-20])
        prediction=dict(states=[start,end],native_states=[start,end],failure=None,ambiguous_intervals=[0])
        window=dict(actions=[[0]*9],gem_delta=0,stop='native_fall',complete=False)
        verdict=classify_student_failure(prediction,Mock(),Mock(),[0,0,1],window,
            (np.array([-10,-10,-5]),np.array([10,10,100])))
        self.assertFalse(verdict['accepted']);self.assertEqual(verdict['reason'],'unknown_prediction')

    def test_control_window_uses_actual_student_and_excludes_world_goal_changes(self):
        from nav.marble_dynamics.fast_nav import student_control_window
        action=[0]*9;different=[1]*9
        state=dict(oob=False,mode=0,radius=.2,gravity_direction=[0,0,-1],held=0)
        row=dict(before=dict(native=dict(state=state,object_id=1,frame_clock_ms=64)),action=action,
            choice=dict(actor_action=action,base_actor_action=different,actor_goal_context=dict(destination=[1,2,3])),outcome={})
        next_row=copy.deepcopy(row);next_row['before']['native']['frame_clock_ms']=128
        next_row['choice']['actor_goal_context']['destination']=[5,6,7]
        window=student_control_window([row,next_row],0,2)
        self.assertEqual(window['actions'],[action]);self.assertEqual(window['stop'],'student_goal_changed')
        self.assertFalse(window['complete'])


if __name__=='__main__':unittest.main()
