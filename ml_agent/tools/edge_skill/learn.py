"""Train a geometry-local multi-proposal model and retain retrieval as an ablation."""
import argparse
import math
import time

import numpy as np
import torch
from torch import nn

from .common import NIGHT,write_json,sha
from .execution import initial,observed,local,rotation,clip_params,solve

SCALE=np.asarray([12.,5.,1.,2.,1.,2.,4.,2.,2.,1.],np.float32)
HEADS=8


def features(case,d=None):
    d=initial(case) if d is None else d;s=d['native']['state'];c=case['cell'];rot=rotation(c)
    p=local(c,s['position']);v=np.asarray(s['velocity'])@rot;w=np.asarray(s['spin'])@rot
    pending=d['bridge_action'];move=d['native']['pending_move'];yaw=s['yaw'];mx,my=move['x'],move['y']
    world=np.asarray([pending[3]-pending[2],pending[0]-pending[1],0.])@rot
    collected=np.asarray([mx*math.cos(yaw)+my*math.sin(yaw),my*math.cos(yaw)-mx*math.sin(yaw),0.])@rot
    return np.asarray([*list(p/[15,6,3]),*list(v/15),*list(w/100),c['lip_z']/1.5,c['width']/10,c.get('lip_depth',1.5)/3,
                       *list(np.asarray(case['goal'])/[10,6,3]),*list(np.asarray(c['landing_center'])/12),
                       *list(np.asarray(c['landing_size'])/15),math.sin(c['landing_yaw']),math.cos(c['landing_yaw']),
                       *list(np.asarray(case['exit_velocity'])/10),s['radius'],min(s['ground_ms'],128)/128,
                       *list(np.asarray(s['best_contact_normal'])@rot),*world[:2],float(pending[4]),
                       *collected[:2],float(move['jump']),float(case.get('airborne',False)),float(case.get('jump_committed',False)),
                       c.get('lip_slope_y',0.),*c.get('landing_slope',[0.,0.])],np.float32)


class Network(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(n,192),nn.SiLU(),nn.Linear(192,192),nn.SiLU(),nn.Linear(192,HEADS*10))
    def forward(self,x):return self.net(x).reshape(-1,HEADS,10)


def train(inputs,output,epochs=800):
    from .experiment import rows
    torch.set_num_threads(1);torch.manual_seed(33017);rng=np.random.default_rng(33017)
    examples=[]
    for path in inputs:
        for r in rows(path):
            if not r['result']['success']:continue
            if r['case']['split']!='train':raise ValueError('Training data must come only from training geometry.')
            x=features(r['case'])
            for p in r.get('alternatives') or [r['params']]:examples.append((x,clip_params(r['case'],p),r['case']))
    x=np.asarray([r[0] for r in examples],np.float32);y=np.asarray([r[1] for r in examples],np.float32)/SCALE
    encodings={bool(r[2].get('stable_inputs',False)) for r in examples}
    if len(encodings)!=1:raise ValueError('Do not mix labels produced by different input encodings.')
    stable_inputs=encodings.pop()
    mean=x.mean(0);std=np.maximum(x.std(0),.08)
    net=Network(x.shape[1]);xt=torch.tensor((x-mean)/std);yt=torch.tensor(y)
    # Distinct initial heads preserve incompatible rising/descending and guidance solutions.
    centers=y[rng.choice(len(y),HEADS,replace=False)].copy()
    for _ in range(30):
        assignment=((y[:,None,:]-centers[None,:,:])**2).mean(2).argmin(1)
        for k in range(HEADS):
            if np.any(assignment==k):centers[k]=y[assignment==k].mean(0)
    with torch.no_grad():
        net.net[-1].weight.mul_(.03);net.net[-1].bias.copy_(torch.tensor(centers.reshape(-1)))
    opt=torch.optim.AdamW(net.parameters(),lr=.0015,weight_decay=.0002)
    begin=time.perf_counter();history=[]
    for epoch in range(epochs):
        permutation=rng.permutation(len(x));losses=[]
        for j in range(0,len(x),256):
            ix=permutation[j:j+256];pred=net(xt[ix])
            loss=((pred-yt[ix,None,:])**2).mean(2).min(1).values.mean()
            opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
        if epoch%100==0 or epoch==epochs-1:
            history.append(dict(epoch=epoch,loss=float(np.mean(losses))));print(history[-1],flush=True)
    if output.exists():raise FileExistsError(output)
    torch.save(dict(schema='edge_skill_proposer_v3',state_dict=net.state_dict(),feature_count=x.shape[1],mean=mean,std=std,
                    x=x,y=y*SCALE,stable_inputs=stable_inputs,source=[dict(file=str(p),sha256=sha(p)) for p in inputs]),output)
    write_json(output.with_suffix('.json'),dict(examples=len(x),unique_cases=len(set(r[2]['id'] for r in examples)),
               geometry_count=len(set(r[2]['cell']['id'] for r in examples)),parameters=sum(p.numel() for p in net.parameters()),
               wall_s=time.perf_counter()-begin,history=history,sha256=sha(output),stable_inputs=stable_inputs))


class Proposer:
    def __init__(self,path):
        torch.set_num_threads(1);ck=torch.load(path,map_location='cpu',weights_only=False)
        if ck['schema']!='edge_skill_proposer_v3':raise ValueError('Wrong checkpoint schema')
        self.net=Network(ck['feature_count']);self.net.load_state_dict(ck['state_dict']);self.net.eval()
        self.mean=ck['mean'];self.std=ck['std'];self.x=(ck['x']-self.mean)/self.std;self.y=ck['y']
        self.stable_inputs=bool(ck.get('stable_inputs',False))
    def params(self,case,d=None,neural=True,retrieval=True):
        x=(features(case,d)[:len(self.mean)]-self.mean)/self.std;out=[]
        if neural:
            with torch.inference_mode():out.extend((self.net(torch.tensor(x[None]))[0].numpy()*SCALE).tolist())
        if retrieval:
            dist=((self.x-x)**2).mean(1);chosen=[]
            # Neighbor labels from one teacher root are highly correlated. Keep
            # distinct parameters so retrieval supplies actual alternatives.
            for i in np.argsort(dist)[:256]:
                p=self.y[i]
                if all(np.linalg.norm((p-q)/SCALE)>.035 for q in chosen):chosen.append(p)
                if len(chosen)>=16:break
            out.extend(p.tolist() for p in chosen)
        return [clip_params(case,p) for p in out]
    def propose(self,engine,case,d=None,budget=64,seed=0,neural=True,retrieval=True,robust_samples=0,accept_early=True):
        if 'stable_inputs' not in case:case=dict(case,stable_inputs=self.stable_inputs)
        d=initial(case) if d is None else d
        case=observed(case,d)
        result=solve(engine,case,d,budget=budget,seed=seed,initial_params=self.params(case,d,neural,retrieval),robust_samples=robust_samples,max_alternatives=1,accept_early=accept_early)
        result['controller_contract']=dict(stable_inputs=case['stable_inputs'])
        return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',nargs='+',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--epochs',type=int,default=800);a=ap.parse_args()
    train([NIGHT/p for p in a.input],NIGHT/a.out,a.epochs)


if __name__=='__main__':main()
