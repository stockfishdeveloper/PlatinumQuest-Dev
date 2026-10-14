"""Restored-instance comparison must normalize metadata, not physical changes."""
import copy
import unittest

from nav.marble_dynamics.cache_integration_probe import compare_logs


class CacheComparisonRegressions(unittest.TestCase):
    def pair(self):
        item=dict(id=17,gem=True,powerup=0,position=[1,2,3],value=5)
        state=dict(position=[0,0,0],velocity=[1,2,3],spin=[4,5,6],held=0,radius=.2,
            clock_ms=1000,effects=[[True,50],[False,0]],effect_end_clock_ms=[1050,7])
        entry=dict(dynamics=dict(native=dict(frame_clock_ms=1000),items=[item]),action=[0]*9,
            chosen=dict(label='route',goal_item_id=17),gem=[1,2,3,5],goal_kind='gem',score_delta=0,
            external_events=[],predicted_next=state,actual_next=copy.deepcopy(state))
        other=copy.deepcopy(entry);other['dynamics']['native']['frame_clock_ms']+=2000
        other['dynamics']['items'][0]['id']=81;other['chosen']['goal_item_id']=81
        for key in ('predicted_next','actual_next'):
            other[key]['clock_ms']+=2000;other[key]['effect_end_clock_ms']=[3050,777]
        return entry,other

    def compare(self,left,right):
        return compare_logs([[left],[right]],[dict(actual_points=0,falls=0,stop_reason='budget')]*2)

    def test_recreated_item_and_absolute_clock_are_not_behavior_changes(self):
        left,right=self.pair();before=copy.deepcopy(right)
        self.assertTrue(self.compare(left,right)['controls_choices_scores_native_equal'])
        self.assertEqual(right,before)

    def test_active_expiry_difference_is_not_hidden(self):
        left,right=self.pair();right['predicted_next']['effect_end_clock_ms'][0]+=1
        result=self.compare(left,right)
        self.assertFalse(result['controls_choices_scores_native_equal'])
        self.assertEqual(result['mismatches'][0]['fields'],['predicted_next'])

    def test_physical_or_goal_identity_difference_is_not_hidden(self):
        left,right=self.pair();right['actual_next']['velocity'][0]+=.01
        self.assertFalse(self.compare(left,right)['controls_choices_scores_native_equal'])
        left,right=self.pair();right['dynamics']['items'][0]['value']=2
        result=self.compare(left,right)
        self.assertIn('chosen',result['mismatches'][0]['fields'])


if __name__=='__main__':unittest.main()
