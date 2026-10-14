"""Detect actual external state changes without confusing F32 JSON formatting."""
import numpy as np


def external_changes(predicted,actual):
    changed=[]
    for key in ('datablock','held'):
        if predicted[key]!=actual[key]:changed.append(key)
    if np.float32(predicted['radius'])!=np.float32(actual['radius']):changed.append('radius')
    if not np.array_equal(np.asarray(predicted['gravity_direction'],np.float32),np.asarray(actual['gravity_direction'],np.float32)):
        changed.append('gravity_direction')
    a,b=predicted['parameters'],actual['parameters']
    if a.keys()!=b.keys() or any(not np.array_equal(np.asarray(a[k],np.float32),np.asarray(b[k],np.float32)) for k in a.keys()&b.keys()):
        changed.append('parameters')
    return changed
