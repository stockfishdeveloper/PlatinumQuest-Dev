"""A native divergent contact must remain flagged, despite earlier accurate steps."""
import hashlib
import json
from pathlib import Path
import unittest

from nav.collision.mesh import CollisionMap
from nav.marble_dynamics.physics import GeneralPhysics
from nav.marble_dynamics.rollout import apply_control
from nav.marble_dynamics.state_copy import clone
from nav.marble_dynamics.evaluate import errors
from nav.marble_dynamics.record import ML


class RichNativeContactFork(unittest.TestCase):
    def test_actual_sprawl_step_exposes_a_future_branch(self):
        path=Path(__file__).with_name('fixtures')/'rich_sprawl_contact_fork.json'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
            'a826ae9d0ec5fc1897dc974e53644cd5e66e4961f17a0e3959da8509a656aaf8')
        case=json.loads(path.read_text());row=case['native_row']
        self.assertEqual(row['index'],case['whole_forecast_first_ambiguity'])
        self.assertEqual(case['whole_forecast_first_error_over_001u_state'],row['index']+1)
        engine=GeneralPhysics(CollisionMap(ML/'collision_maps'/(case['map']+'.npz')))
        before=clone(row['before']);move=apply_control(before,dict(move=row['move'],events=row['control_events']))
        predicted,events=engine.step(before,move,row['dt_ms'])
        # Native-vs-port divergence is not hidden with a generous accuracy bound.
        # It must remain visibly uncertain before any planner accepts the future.
        self.assertGreater(errors(predicted,row['after'])[0],.01)
        self.assertTrue(any(e['kind']=='ambiguous_contact' for e in events))


if __name__=='__main__':unittest.main()
