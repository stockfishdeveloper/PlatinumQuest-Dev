from types import SimpleNamespace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools.native_worker_bench import candidate_workers,prepare_countdown


class NonchargingCountdown(unittest.TestCase):
    def env(self,clock):
        state=dict(mode=2,clock_ms=clock,oob=False)
        env=SimpleNamespace(msg=SimpleNamespace(dynamics=dict(native=dict(state=state),
            blast=dict(enabled=True,charging=False,value=0))),obs_ms=64,
            info=dict(round_ms=240000),commands=[])
        def advance():
            state['clock_ms']+=16
            if state['clock_ms']>=3500:state['mode']=0
        def control(word):env.commands.append(word);advance()
        env.control=control;env.step=lambda *a,**k:advance()
        env.time_left_s=lambda:240.-max(0.,state['clock_ms']-3500)/1000.
        return env

    def test_setup_jitter_reaches_identical_natural_go_without_charging(self):
        endings=[]
        for clock in (1700,1764):
            env=self.env(clock);result=prepare_countdown(env)
            self.assertEqual(result['method'],'neutral_go_noncharging')
            self.assertEqual(result['go_to_snapshot_ms'],16)
            self.assertEqual(env.commands,['FIXEDSTEP 16 16','FIXEDSTEP 16 64'])
            self.assertFalse(env.msg.dynamics['blast']['charging'])
            self.assertEqual(env.msg.dynamics['blast']['value'],0)
            endings.append(env.msg.dynamics['native']['state'])
        self.assertEqual(*endings)

    def test_late_setup_is_rejected(self):
        env=self.env(3516);env.msg.dynamics['native']['state']['mode']=0
        with self.assertRaises(RuntimeError):prepare_countdown(env)
        self.assertEqual(env.commands,[])


class WorkerOwnership(unittest.TestCase):
    def test_main_leaves_output_creation_to_the_native_benchmark(self):
        from nav.marble_dynamics import reliability_eval as re,search
        from tools import native_worker_bench as bench
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'rounds'
            def native_main():
                self.assertFalse(output.exists())
                output.mkdir()
                (output/'prepared.json').write_text(json.dumps(dict(maps=['map'],seeds=[1],rounds=1,
                    source_hashes={str(Path(bench.__file__).resolve()):bench.sha(bench.__file__)})))
                (output/'summary.json').write_text(json.dumps(dict(branches=[dict(
                    score_valid=True,completed=True,declared_inputs_valid=True)])))
            argv=['native_worker_bench','--output',str(output),'--workers','1',
                '--only-branch','physics_plan','--full-round','--geometry-route',
                '--sweep-dll','unused.dll','--order-min-left-s','.015']
            with patch.object(sys,'argv',argv),patch.object(re,'main',side_effect=native_main), \
                    patch.object(re,'audit_start_layouts',return_value=dict(passed=True)), \
                    patch.multiple(re,RESEARCH_EVERY=re.RESEARCH_EVERY,ORDER_CANDIDATES=re.ORDER_CANDIDATES,
                        ORDER_EXTRA_S=re.ORDER_EXTRA_S,ORDER_MIN_LEFT_S=re.ORDER_MIN_LEFT_S), \
                    patch.multiple(search,CLUSTER_EXTRA_S=search.CLUSTER_EXTRA_S,EXTRA_JUMP_S=search.EXTRA_JUMP_S):
                bench.main()
            self.assertTrue(json.loads((output/'summary.json').read_text())['complete'])
            self.assertEqual(json.loads((output/'wrapper_protocol.json').read_text())['order_min_left_s'],.015)
            self.assertEqual((output/'wrapper_source.py').read_bytes(),Path(bench.__file__).read_bytes())

    def test_main_refuses_existing_results_without_overwriting_them(self):
        from tools import native_worker_bench as bench
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'rounds';output.mkdir()
            marker=output/'original';marker.write_text('preserved')
            argv=['native_worker_bench','--output',str(output),'--workers','1',
                '--only-branch','physics_plan','--full-round','--geometry-route','--sweep-dll','unused.dll']
            with patch.object(sys,'argv',argv),self.assertRaises(FileExistsError):bench.main()
            self.assertEqual(marker.read_text(),'preserved')

    def test_pool_is_bound_and_closed_on_success_and_round_failure(self):
        for fail in (False,True):
            with self.subTest(fail=fail):
                events=[]
                class Controller:
                    uses_physics=True
                    def __init__(self,*args):self.nav=SimpleNamespace(model='model')
                class Pool:
                    def __init__(self,n,setup):events.append(('create',n,setup))
                    def bind(self,refs):events.append(('bind',refs['model']))
                    def close(self):events.append(('close',))
                engine=SimpleNamespace(parallel=None,mesh='mesh',native_sim='sim',crossings=None)
                args=SimpleNamespace(geometry_terrain='geometry')
                def run(env,proc,name,seed,kind,a,terrain,e,checkpoint,ri):
                    module.ContinuousController(terrain,e,'checkpoint',a,kind)
                    if fail:raise ValueError('round failed')
                    return 42
                module=SimpleNamespace(ContinuousController=Controller,full_round_branch=run)
                with candidate_workers(module,Pool,lambda a,n:n,8):
                    if fail:
                        with self.assertRaisesRegex(ValueError,'round failed'):
                            module.full_round_branch(None,None,'map',1,'physics_plan',args,'terrain',engine,'sha',1)
                    else:
                        self.assertEqual(module.full_round_branch(None,None,'map',1,'physics_plan',args,'terrain',engine,'sha',1),42)
                    self.assertIsNone(engine.parallel)
                self.assertIs(module.ContinuousController,Controller)
                self.assertIs(module.full_round_branch,run)
                self.assertEqual(events,[('create',8,'map'),('bind','model'),('close',)])


if __name__=='__main__':unittest.main()
