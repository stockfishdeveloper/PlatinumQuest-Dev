"""Exact independent cloning for native JSON state trees, without copy memo work.

Native snapshots are acyclic JSON dictionaries/lists/scalars. Unusual extension
values fall back to deepcopy. No quantization, state sharing or model change.
"""
from copy import deepcopy


_ATOMS=(str,int,float,bool,type(None))


def clone(value):
    kind=type(value)
    if kind is dict:
        return {key:(item if type(item) in _ATOMS else clone(item)) for key,item in value.items()}
    if kind is list:
        return [item if type(item) in _ATOMS else clone(item) for item in value]
    if kind in _ATOMS:return value
    return deepcopy(value)
