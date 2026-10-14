"""Complete-map surface hierarchy and an untrained neural access prototype.

All triangles belong to a leaf. Coarse tokens summarize every leaf; exact detail
remains queryable anywhere, without a nearest-K or local-distance cutoff. Pooling
is deliberately not claimed to preserve small details in a latent vector.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from nav.collision.mesh import CollisionMap,atomic_json
from nav.marble_dynamics.record import ML


class SurfaceHierarchy:
    def __init__(self,mesh,leaf_size=24):
        if leaf_size<1:raise ValueError('Leaf size must be positive')
        self.mesh=mesh;self.leaf_size=leaf_size
        centers=(mesh.lo+mesh.hi)*.5
        order=[];nodes=[];leaves=[]

        def build(ids):
            lo=mesh.lo[ids].min(axis=0);hi=mesh.hi[ids].max(axis=0)
            index=len(nodes);nodes.append(dict(lo=lo,hi=hi,left=-1,right=-1,start=-1,count=len(ids)))
            if len(ids)<=leaf_size:
                nodes[index]['start']=len(order);order.extend(ids.tolist());leaves.append(index)
            else:
                axis=int(np.argmax(np.ptp(centers[ids],axis=0)))
                sorted_ids=ids[np.argsort(centers[ids,axis],kind='stable')];half=len(ids)//2
                nodes[index]['left']=build(sorted_ids[:half]);nodes[index]['right']=build(sorted_ids[half:])
            return index
        build(np.arange(len(mesh.triangles)))
        self.order=np.asarray(order,np.int32)
        self.bounds=np.asarray([[n['lo'],n['hi']] for n in nodes])
        self.children=np.asarray([[n['left'],n['right']] for n in nodes],np.int32)
        self.starts=np.asarray([n['start'] for n in nodes],np.int32)
        self.counts=np.asarray([n['count'] for n in nodes],np.int32)
        self.leaves=np.asarray(leaves,np.int32)
        self.triangle_leaf=np.empty(len(order),np.int32)
        for leaf,node in enumerate(self.leaves):
            ids=self.order[self.starts[node]:self.starts[node]+self.counts[node]]
            self.triangle_leaf[ids]=leaf
        if not np.array_equal(np.sort(self.order),np.arange(len(mesh.triangles))):
            raise RuntimeError('Hierarchy must include every triangle exactly once')

    def query_aabb(self,lo,hi):
        lo=np.asarray(lo,float);hi=np.asarray(hi,float)
        if np.any(hi<lo):raise ValueError('Query bounds are reversed')
        stack=[0];chunks=[]
        while stack:
            node=stack.pop();bounds=self.bounds[node]
            if np.any(bounds[1]<lo) or np.any(bounds[0]>hi):continue
            if self.starts[node]>=0:
                ids=self.order[self.starts[node]:self.starts[node]+self.counts[node]]
                keep=np.all(self.mesh.hi[ids]>=lo,axis=1)&np.all(self.mesh.lo[ids]<=hi,axis=1)
                chunks.append(ids[keep])
            else:stack.extend(self.children[node].tolist())
        return np.sort(np.concatenate(chunks)) if chunks else np.empty(0,np.int32)

    def detail(self,lo,hi,origin=(0,0,0)):
        return self.mesh.features(self.query_aabb(lo,hi),origin)

    def candidate_corridor(self,positions,radius,margin=0.):
        """Conservative swept-sphere AABB union, including walls and undersides."""
        p=np.asarray(positions,float)
        if p.ndim!=2 or p.shape[1]!=3 or not len(p):raise ValueError('Expected nonempty [N,3] path')
        if radius<=0 or margin<0:raise ValueError('Invalid radius/margin')
        pad=radius+margin
        segments=zip(p[:-1],p[1:]) if len(p)>1 else [(p[0],p[0])]
        chunks=[self.query_aabb(np.minimum(a,b)-pad,np.maximum(a,b)+pad) for a,b in segments]
        return np.unique(np.concatenate(chunks))

    def model_inputs(self,origin=(0,0,0)):
        """Static intrinsic features + all leaf bounds relative to query origin.

        No object/triangle IDs are supplied as learned features. The returned ID
        association is for exact retrieval/provenance, not map memorization.
        """
        mesh=self.mesh;leaf_bounds=self.bounds[self.leaves]
        leaf_centers=leaf_bounds.mean(axis=1);center=leaf_centers[self.triangle_leaf]
        props=np.asarray([[m['friction'],m['restitution'],m['force']] for m in mesh.info['materials']])
        planes=mesh.planes.copy();planes[:,3]+=np.einsum('ij,ij->i',planes[:,:3],center)
        area=np.linalg.norm(np.cross(mesh.triangles[:,1]-mesh.triangles[:,0],mesh.triangles[:,2]-mesh.triangles[:,0]),axis=1)*.5
        features=np.concatenate([(mesh.triangles-center[:,None,:]).reshape(-1,9),mesh.normals,planes,
                                 props[mesh.material_ids],mesh.boundary_edges.astype(float),area[:,None]],axis=1)
        return dict(triangle_features=features.astype(np.float32),triangle_leaf=self.triangle_leaf,
                    leaf_bounds=(leaf_bounds-np.asarray(origin)).astype(np.float32),
                    exact_triangle_ids=np.arange(len(mesh.triangles)))

    def save(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp.npz')
        np.savez_compressed(tmp,triangle_order=self.order,bounds=self.bounds,children=self.children,
                            starts=self.starts,counts=self.counts,leaves=self.leaves,triangle_leaf=self.triangle_leaf)
        os.replace(str(tmp),str(path))
        atomic_json(path.with_suffix('.json'),dict(schema='complete_surface_hierarchy_v1',
                    source_mesh_sha256=self.mesh.info['mesh_sha256'],hierarchy_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    triangles=len(self.order),leaves=len(self.leaves),nodes=len(self.bounds),leaf_size=self.leaf_size,
                    all_triangles_preserved=True,coarse_tokens_lossless=False,exact_queries_retained=True,
                    encoding_ready_for_policy=False))

    @classmethod
    def load(cls,path,mesh):
        path=Path(path);meta=json.loads(path.with_suffix('.json').read_text())
        if meta['schema']!='complete_surface_hierarchy_v1' or meta['source_mesh_sha256']!=mesh.info['mesh_sha256']:
            raise ValueError('Hierarchy geometry identity mismatch')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=meta['hierarchy_sha256']:raise ValueError('Hierarchy hash mismatch')
        result=cls.__new__(cls);result.mesh=mesh;result.leaf_size=meta['leaf_size']
        with np.load(path,allow_pickle=False) as data:
            for key,name in [('triangle_order','order'),('bounds','bounds'),('children','children'),('starts','starts'),
                             ('counts','counts'),('leaves','leaves'),('triangle_leaf','triangle_leaf')]:setattr(result,name,data[key])
        if not np.array_equal(np.sort(result.order),np.arange(len(mesh.triangles))):raise ValueError('Incomplete triangle membership')
        return result


def make_encoder(feature_count=23,width=64):
    """Randomly initialized prototype; no learned skill or fidelity claim."""
    import torch
    from torch import nn

    class CompleteMapEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.triangles=nn.Sequential(nn.Linear(feature_count,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
            self.leaves=nn.Sequential(nn.Linear(width*2+6,width),nn.SiLU(),nn.Linear(width,width))

        def forward(self,features,triangle_leaf,leaf_bounds):
            signed_log=lambda x:torch.sign(x)*torch.log1p(torch.abs(x))
            embedded=self.triangles(signed_log(features))
            n=len(leaf_bounds);index=triangle_leaf[:,None].expand(-1,width)
            sums=torch.zeros(n,width,device=embedded.device,dtype=embedded.dtype).scatter_add_(0,index,embedded)
            counts=torch.bincount(triangle_leaf,minlength=n).clamp_min(1)[:,None]
            maxima=torch.full_like(sums,-torch.inf).scatter_reduce_(0,index,embedded,reduce='amax',include_self=False)
            # Every triangle participates. Relative leaf location/extent makes
            # far-away platforms accessible, while exact detail remains external.
            return self.leaves(torch.cat([sums/counts,maxima,signed_log(leaf_bounds.reshape(n,6))],dim=1))
    return CompleteMapEncoder()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ML/'collision_maps/hierarchies')
    parser.add_argument('--leaf-size',type=int,default=24)
    parser.add_argument('--encoder-smoke',action='store_true')
    args=parser.parse_args();rows=[];rng=np.random.RandomState(611)
    names=json.loads((ML/'configs/real_maps_20261009.json').read_text())['maps']
    for name in names:
        mesh=CollisionMap(ML/'collision_maps'/(name+'.npz'));started=time.perf_counter()
        hierarchy=SurfaceHierarchy(mesh,args.leaf_size);hierarchy.save(args.output/(name+'.npz'))
        restored=SurfaceHierarchy.load(args.output/(name+'.npz'),mesh)
        timings=[];reference_times=[]
        # Tight contact boxes, medium lookahead and whole-map spans. No near-only test.
        for i in range(200):
            center=mesh.triangles[int(rng.randint(len(mesh.triangles)))].mean(axis=0)
            half=float(rng.choice([.19,1.,5.,15.,60.]))
            lo=center-half;hi=center+half
            if i==199:lo=mesh.bounds[0]-1;hi=mesh.bounds[1]+1
            t=time.perf_counter();expected=mesh.query_aabb(lo,hi);reference_times.append(time.perf_counter()-t)
            t=time.perf_counter();actual=restored.query_aabb(lo,hi);timings.append(time.perf_counter()-t)
            if not np.array_equal(actual,expected):raise RuntimeError(name+' hierarchy omitted/query-added triangles')
        encoded=None
        if args.encoder_smoke:
            import torch
            torch.manual_seed(611);torch.set_num_threads(2)
            inputs=hierarchy.model_inputs(mesh.bounds.mean(axis=0))
            model=make_encoder(inputs['triangle_features'].shape[1])
            tensors=[torch.from_numpy(inputs[k]) for k in ('triangle_features','triangle_leaf','leaf_bounds')]
            tensors[1]=tensors[1].long()
            with torch.no_grad():
                model(*tensors);t=time.perf_counter();tokens=model(*tensors);elapsed=time.perf_counter()-t
            if not torch.isfinite(tokens).all() or tokens.shape!=(len(hierarchy.leaves),64):raise RuntimeError('Invalid map encoder output')
            encoded=dict(shape=list(tokens.shape),cpu_forward_ms=elapsed*1000,learned=False,device='CPU',
                         all_triangles_input=len(inputs['triangle_features']),feature_count=inputs['triangle_features'].shape[1])
        row=dict(map=name,triangles=len(mesh.triangles),leaf_tokens=len(hierarchy.leaves),
                 exact_query_checks=200,all_triangles_preserved=True,
                 query_ms_median=float(np.median(timings)*1000),query_ms_p95=float(np.percentile(timings,95)*1000),
                 vectorized_reference_ms_median=float(np.median(reference_times)*1000),encoder_smoke=encoded,
                 elapsed_s=time.perf_counter()-started,policy_connected=False)
        rows.append(row);atomic_json(args.output/'checks.json',dict(maps=rows,learned_encoder_trained=False,planner_approved=False))
        print(json.dumps(row),flush=True)


if __name__=='__main__':main()
