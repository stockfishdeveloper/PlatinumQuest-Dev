"""Countdown setup must not give one scored arm extra prespin or Blast."""
from types import SimpleNamespace
import unittest

from tools.realtime_bench import align_countdown


class CountdownAlignment(unittest.TestCase):
    def env(self,elapsed_ms,restore_step=16):
        env=SimpleNamespace(msg=SimpleNamespace(dynamics=dict(
            native=dict(state=dict(mode=2)),
            blast=dict(value=elapsed_ms/25000.,charge_ms=25000.,enabled=True,charging=True))),
            obs_ms=64,commands=[],restore_step=restore_step)
        def advance(ms):
            env.msg.dynamics['blast']['value']+=ms/25000.
        def control(word):
            env.commands.append(word)
            advance(16 if word=='FIXEDSTEP 16 16' else env.restore_step)
        env.control=control
        env.step=lambda *a,**k:advance(16)
        return env

    def test_one_reply_of_setup_jitter_reaches_the_same_prespin_phase(self):
        for restore_step in (16,64):
            results=[]
            for initial in (1712.,1776.):
                env=self.env(initial,restore_step)
                result=align_countdown(env)
                results.append(env.msg.dynamics['blast']['value'])
                self.assertAlmostEqual(result['prespin_start_charge_ms'],1984.+restore_step)
                self.assertEqual(env.obs_ms,64)
                self.assertEqual(env.commands,['FIXEDSTEP 16 16','FIXEDSTEP 16 64'])
            self.assertAlmostEqual(*results)

    def test_late_or_non_start_setup_fails_instead_of_repairing_state(self):
        for late,mode in ((2000.,2),(1776.,0)):
            env=self.env(late);env.msg.dynamics['native']['state']['mode']=mode
            with self.assertRaises(RuntimeError):align_countdown(env)
            self.assertEqual(env.commands,[])


if __name__=='__main__':unittest.main()
