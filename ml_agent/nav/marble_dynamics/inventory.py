"""Observed script inventory boundaries; refuse unverified Mega transitions."""


def check(context,state,pending):
    if context is None:return
    if context['mega_active'] and not context['permanent_mega']:
        raise ValueError('Timed Mega activation/expiry requires a verified script-event model')
    normal=context.get('normal_radius',0.)
    if normal>0 and not context['mega_active'] and abs(state['radius']-normal)>1e-5:
        raise ValueError('Script/native radius transition is not verified')
    if pending[6] and context['held_type']==6:
        raise ValueError('Script-held Mega activation is not verified')


def context_changes(predicted,observed_dynamics):
    a=predicted.get('inventory_context');b=observed_dynamics.get('inventory')
    if a is None or b is None:return []
    changed=[]
    if (a['held_type']==6 or b['held_type']==6) and a['held_type']!=b['held_type']:
        changed.append('script_inventory:held_type')
    for key in ('mega_active','permanent_mega'):
        if a[key]!=b[key]:changed.append('script_inventory:'+key)
    return changed
