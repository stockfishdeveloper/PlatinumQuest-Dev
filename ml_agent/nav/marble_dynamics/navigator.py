"""Read-only deterministic navigator; hypothetical state is independently cloned."""
import copy
import math

import numpy as np
import torch

from nav.model import NavActorCritic
from nav.obs import ObsBuilder, NAV_OBS_VERSION, fill_use_state, ss_aim
from nav.joystick import action_to_joystick
from nav.protocol import RAW_GEMS, RAW_POW_ITEMS, RAW_SPIN, RAW_POW_HELD,RAW_POW_BLAST,RAW_POW_SPECIAL
from nav.marble_dynamics.bridge import reply
from nav.marble_dynamics.fast_policy import deterministic_act
from nav.native.obs_terrain import native_queries
from nav.native.route_native import navigate as native_navigate

FAST_ACT = True     # 2026-10-19 (search speed-up toward real time): fast_policy.deterministic_act (bit-identical to model.act)
FAST_OBS = True     # the observation's terrain queries in C++ (nav/native/obs_terrain.py, bit-identical; NumPy fallback)


def project_observation(raw, initial_position, state, index):
    """Current item locations remain fixed; only predicted physical state evolves.

    No future pickups/score/spawn events are invented. This is a proposal input,
    not an observation sent to the real navigator or a model of the item system.
    """
    result=raw.copy();delta=np.asarray(initial_position)-state['position']
    gems=result[RAW_GEMS].reshape(5,5)
    for g in gems:
        if g[3]>-500:
            g[:3]+=delta;g[4]=np.linalg.norm(g[:3])
    for item in result[RAW_POW_ITEMS].reshape(3,5):
        if item[0]>0:item[1:4]+=delta
    result[:3]=state['position'];result[3:6]=state['velocity'];result[RAW_SPIN]=state['spin']
    result[RAW_POW_HELD]=state['held']
    if state.get('inventory_context',{}).get('held_type')==6:result[RAW_POW_HELD]=6
    if state.get('blast_context') is not None:
        result[RAW_POW_BLAST]=state['blast_context']['value']
        result[RAW_POW_SPECIAL]=int(state['blast_context']['special'])
    result[43:46]=[state['effects'][i][1]/1000. if state['effects'][i][0] else 0. for i in (3,4,5)]
    result[31]-=index*64;result[32]+=index*64
    return result


class FrozenNavigator:
    def __init__(self,terrain,checkpoint=None,model=None):
        self.terrain=terrain;self.builder=ObsBuilder(native_queries(terrain) if FAST_OBS else terrain)
        if model is None:
            torch.set_num_threads(1)
            ck=torch.load(str(checkpoint),map_location='cpu')
            if ck.get('obs_version')!=NAV_OBS_VERSION:raise ValueError('Checkpoint observation version mismatch')
            model=NavActorCritic();model.load_state_dict(ck['model']);model.eval()
            model.requires_grad_(False)
            self.update=ck.get('update')
        self.model=model;self.h=model.initial_state(1,torch.device('cpu'))
        self.sent=[False,False];self.pending=None;self.aim=None;self.fire=-10**9;self.decision=0

    def clone(self):
        other=copy.copy(self);other.h=self.h.clone();other.sent=list(self.sent)
        other.builder=ObsBuilder(self.builder.terrain);other.builder.airborne=self.builder.airborne
        return other

    def act(self,raw,waypoint,gem,next_goal=None):
        crop,vec,_=self.builder.build(raw,waypoint,next_goal,gem_goal=gem)
        fill_use_state(vec,*self.sent,self.pending is not None,self.aim,(self.decision-self.fire)*.064)
        with torch.inference_mode():
            if FAST_ACT:
                a,self.h=deterministic_act(self.model,crop,vec,self.h)
            else:
                out=self.model.act(torch.from_numpy(crop).unsqueeze(0),torch.from_numpy(vec).unsqueeze(0),self.h,deterministic=True)
                self.h=out['h_next'];a=out['action_game'][0].tolist()
        v=raw[3:6];js=action_to_joystick(*a[:5],float(v[0]),float(v[1]))
        held=int(raw[RAW_POW_HELD]);use=a[5]>.5
        aim=ss_aim(vec) if held==2 else None
        action=reply(js,int(use and held>0),math.atan2(*aim) if aim is not None else None)
        action[8]=int(use and held==0)
        return action

    def sent_action(self,action,before):
        sent=bool(action[6] or action[8])
        if action[6] and int(before[RAW_POW_HELD])==2:
            self.aim=(math.sin(action[7]),math.cos(action[7]))
            if self.pending is None:self.pending=self.decision
        self.sent=[sent,self.sent[0]];self.decision+=1

    def outcome(self,before,after):
        held=int(before[RAW_POW_HELD]);now=int(after[RAW_POW_HELD])
        if held==2 and now==0:self.fire=self.decision-1;self.pending=None;self.aim=None
        if self.pending is not None and (self.decision-self.pending>4 or now!=2):self.pending=None;self.aim=None


class NavigatorProposal:
    def __init__(self,navigator,raw,route,gem,next_goal,steps,first_action=None):
        self.navigator=navigator.clone();self.raw=raw.copy();self.initial_position=raw[:3].copy()
        self.route=copy.copy(route);self.gem=gem;self.next_goal=next_goal;self.steps=steps;self.previous=None
        # With a supplied reply, the real actor has already advanced at this
        # observation. Reuse that reply, then advance only this clone in futures.
        self.first_action=list(first_action) if first_action is not None else None
        self.native=None   # the engine's compiled simulator: the route is followed with its routing (route_native, exact)

    def __call__(self,state,index):
        raw=project_observation(self.raw,self.initial_position,state,index)
        if self.previous is not None:self.navigator.outcome(self.previous,raw)
        waypoint,_=(native_navigate(self.native,self.route,state['position'],self.gem) if self.native is not None
                    else self.route.navigate(state['position'],self.gem))
        action=(list(self.first_action) if index==0 and self.first_action is not None
            else self.navigator.act(raw,waypoint,self.gem,self.next_goal))
        self.previous=raw
        if action[6] and int(raw[RAW_POW_HELD])>5:
            raise ValueError('Frozen proposal requested an unsupported Mega event')
        self.navigator.sent_action(action,raw)
        return action
