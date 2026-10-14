"""Expose every currently observed native gem to fallback physical goal search.

Preserve the legacy nearest-five coordinates for matching items. Extra goals
come from current visible native telemetry, never mission definitions/hidden
spawn fixtures or future observations. No factorial tour over the larger pool.
"""
import numpy as np

from nav.gems import visible_gems


def observed_gems(dynamics,raw):
    nearby=visible_gems(raw);position=np.asarray(dynamics['native']['state']['position'])
    goals=[]
    for item in dynamics.get('items',[]):
        if not item['gem'] or item['value']<=0:continue
        point=np.asarray(item['position']);matches=[g for g in nearby if np.linalg.norm(np.asarray(g[:3])-point)<.1]
        goal=matches[0] if matches else (*map(float,point),float(item['value']),float(np.linalg.norm(point-position)))
        goals.append(goal)
    return sorted(goals,key=lambda g:g[4])
