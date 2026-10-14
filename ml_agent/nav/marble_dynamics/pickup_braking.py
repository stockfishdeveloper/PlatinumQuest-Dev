"""Re-predict braking after the forecast collection, instead of chasing a spent goal.

Only proposed commands precede braking; the complete changed future is replayed
from the original observed state and must pass every existing physical check.
An old ambiguity after the pickup is not waived: the changed trajectory itself
must be unambiguous. No synthetic pickup/score or future observed pose is supplied.
"""
from nav.marble_dynamics.bridge import predict_replies
from nav.marble_dynamics.search import BrakeContinuation,assess


def alternatives(engine,route,dynamics,gem,pairs,brake_replies=64):
    results=[]
    for original,rollout in pairs:
        if rollout is None:continue
        pickup=original.get('possible_pickup_interval')
        if pickup is None or original['rejected'] is None or rollout['failure']:continue
        if any(i<pickup for i in rollout['ambiguous_intervals']):continue
        count=min(int(pickup),len(rollout['proposed']))
        if not count:continue
        changed=predict_replies(engine,dynamics,BrakeContinuation(rollout['proposed'][:count],gem,brake_replies),stop_on_ambiguity=True)
        info=assess(engine,route,dynamics,gem,changed,original['label']+'_pickup_brake')
        info.update(pickup_braking_after_reply=count,original_label=original['label'],
            requested_replies=changed['requested_replies'],evaluated_replies=len(changed['controls']),
            stopped_on_ambiguity=changed['stopped_on_ambiguity'])
        results.append((info,changed))
    return results
