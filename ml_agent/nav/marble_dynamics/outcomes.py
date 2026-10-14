"""Outcome checks aligned to native 16 ms intervals, without future-state inputs."""
import numpy as np


def first_landing(support):
    airborne=0
    for i,touching in enumerate(support):
        if touching:
            if airborne>=2:return i
            airborne=0
        else:airborne+=1
    return None


def compare(rows,actual,predicted,events,times):
    count=len(predicted)-1
    native_support=[r['telemetry']['support_steps']>0 for r in rows[:count]]
    model_support=[any(e['kind']=='support' for e in step) for step in events[:count]]
    native_collisions=[i for i,r in enumerate(rows[:count]) if r['telemetry']['collisions']>0]
    model_collisions=[i for i,step in enumerate(events[:count]) if any(e['kind']=='collision' for e in step)]
    unmatched=set(native_collisions);matched=0
    for i in model_collisions:
        candidates=[j for j in unmatched if abs(j-i)<=1]
        if candidates:
            unmatched.remove(min(candidates,key=lambda j:abs(j-i)));matched+=1
    actual_landing=first_landing(native_support);predicted_landing=first_landing(model_support)
    landing=dict(actual_interval=actual_landing,predicted_interval=predicted_landing,
                 both_detected=actual_landing is not None and predicted_landing is not None)
    if landing['both_detected']:
        a=actual[actual_landing+1];p=predicted[predicted_landing+1]
        landing.update(time_error_s=float(times[predicted_landing+1]-times[actual_landing+1]),
                       position_error_u=float(np.linalg.norm(np.asarray(a['position'])-p['position'])),
                       velocity_error_u_per_s=float(np.linalg.norm(np.asarray(a['velocity'])-p['velocity'])),
                       spin_error_rad_per_s=float(np.linalg.norm(np.asarray(a['spin'])-p['spin'])))
        index=int(np.searchsorted(times,times[actual_landing+1]+1.))
        if index<len(predicted):
            landing['one_second_continuation_position_error_u']=float(np.linalg.norm(
                np.asarray(actual[index]['position'])-predicted[index]['position']))
    return dict(interval_resolution_s=float(max((r['dt_ms'] for r in rows[:count]),default=0)/1000),
                peak_height_error_u=float(abs(max(a['position'][2] for a in actual[:count+1])-
                                             max(p['position'][2] for p in predicted))),
                collision_intervals=dict(native=len(native_collisions),predicted=len(model_collisions),
                                         matched_within_one_interval=matched,
                                         precision=matched/len(model_collisions) if model_collisions else None,
                                         recall=matched/len(native_collisions) if native_collisions else None),
                support_interval_disagreements=sum(a!=b for a,b in zip(native_support,model_support)),
                landing=landing,
                caveat='Native support is a solver contact flag; first support after >=2 airborne intervals. Not a certified stable landing.')
