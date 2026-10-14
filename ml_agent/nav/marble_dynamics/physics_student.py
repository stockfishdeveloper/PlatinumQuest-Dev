"""Unfitted motion surrogate with complete-map attention, separate from A2.

This first benchmark predicts one native16ms motion delta. It is not yet an
autoregressive world model, a safety gate or an executable gameplay controller.
"""
import math
import torch
from torch import nn
from nav.marble_dynamics.student_data import STATE_FEATURES,SURFACE_FEATURES


class MotionStudent(nn.Module):
    def __init__(self,surface_features=len(SURFACE_FEATURES),width=64):
        super().__init__();self.width=width
        self.state=nn.Sequential(nn.Linear(len(STATE_FEATURES),128),nn.SiLU(),nn.Linear(128,128),nn.SiLU())
        self.surface=nn.Sequential(nn.Linear(surface_features,128),nn.SiLU(),nn.Linear(128,width),nn.SiLU())
        self.query=nn.Linear(128,width);self.key=nn.Linear(width,width);self.value=nn.Linear(width,width)
        self.motion=nn.Sequential(nn.Linear(128+width,128),nn.SiLU(),nn.Linear(128,9))

    def forward(self,states,surfaces,bbox_lo,bbox_hi):
        if len(surfaces)!=len(bbox_lo) or len(surfaces)!=len(bbox_hi):
            raise ValueError('Complete surface/bounds alignment required')
        if not len(surfaces):raise ValueError('A complete nonempty map is required')
        state=self.state(states);encoded=self.surface(surfaces)
        logits=self.query(state)@self.key(encoded).T/math.sqrt(self.width)
        # A continuous locality prior, never a top-k/ray-range/surface cutoff.
        # Every triangle is encoded and appears in the attention distribution.
        p=states[:,:3];scale=states[:,STATE_FEATURES.index('map_scale')]*100.
        reach=states[:,STATE_FEATURES.index('radius_map')]+(
            torch.linalg.vector_norm(states[:,3:6],dim=1)*25.+45.)*.016/scale+.05/scale
        distance=torch.maximum(torch.maximum(bbox_lo[None]-p[:,None],p[:,None]-bbox_hi[None]),
            torch.zeros((),device=states.device,dtype=states.dtype))
        prior=-torch.clamp(torch.sum(distance**2,dim=-1)/reach[:,None]**2,max=40.)
        attention=torch.softmax(logits+prior,dim=-1)
        context=attention@self.value(encoded)
        return self.motion(torch.cat([state,context],dim=1))
