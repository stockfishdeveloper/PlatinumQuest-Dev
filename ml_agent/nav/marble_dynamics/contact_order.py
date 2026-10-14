"""Native DIF hull/bin traversal order over the unchanged collision polygons.

Sequential impulses are sensitive to order. A whole-object export has a different
bin traversal than a small contact query. Geometry identity stays in the NPZ;
this layer reconstructs source hull order, rather than sorting by a guessed trick.
"""
from collections import defaultdict
import hashlib
from pathlib import Path

import numpy as np

import hxDif


def vector(v):return np.asarray([v.x,v.y,v.z],np.float32)


class NativeContactOrder:
    def __init__(self,mesh,polygons):
        self.objects={};self.mesh=mesh
        root=Path(__file__).resolve().parents[3]
        by_file={Path(r['path']).name.lower():r for r in mesh.info['source_inventory']}
        groups=defaultdict(list)
        for pid,p in enumerate(polygons):
            idx=p['first_triangle'];groups[int(mesh.object_ids[idx])].append(pid)
        self.mapping_checks=[];self.unranked_contacts=0
        for oid,obj in mesh.objects.items():
            if obj['class']!='InteriorInstance':continue
            source=by_file[Path(obj['interior_file'].replace('\\','/')).name.lower()];path=root/source['path']
            if hashlib.sha256(path.read_bytes()).hexdigest()!=source['sha256']:raise ValueError('DIF source changed')
            interior=hxDif.Dif.Load(str(path)).interiors[0]
            matrix=np.asarray(obj['transform'],np.float32).reshape(4,4)
            scale=np.asarray(obj['scale'],np.float32)
            inverse=np.linalg.inv(matrix).astype(np.float32)
            candidates=defaultdict(list)
            for pid in groups[oid]:
                p=polygons[pid];raw=p['native']
                candidates[(raw['surface_key'],raw['material_index'],len(p['vertices']))].append(pid)
            referenced=set(int(s) for s in interior.hullSurfaceIndices)
            surface_polygons={};maximum_error=0.
            for sid in referenced:
                null=bool(sid&0x80000000);surface=interior.nullSurfaces[sid&0x7fffffff] if null else interior.surfaces[sid]
                count=surface.windingCount
                order=list(range(count)) if null else [0]+list(range(1,count,2))+list(range((count-1)&~1,0,-2))
                if not null:order=[i for j,i in enumerate(order) if surface.fanMask&(1<<j)]
                winding=[int(interior.windings[surface.windingStart+i]) for i in order]
                points=np.asarray([vector(interior.points[i]) for i in winding],np.float32)
                world=((points*scale)@matrix[:3,:3].T+matrix[:3,3]).astype(np.float32).astype(float)
                plane=int(surface.planeIndex)
                if not null and surface.planeFlipped:plane|=0x8000
                material=0 if null else int(surface.textureIndex)
                possible=candidates[(plane,material,len(points))]
                if not possible:raise ValueError('No exported polygon for DIF surface '+str((oid,sid,plane,material,len(points))))
                distances=[np.linalg.norm(polygons[pid]['vertices'].mean(axis=0)-world.mean(axis=0)) for pid in possible]
                pid=possible[int(np.argmin(distances))]
                error=float(np.max(np.min(np.linalg.norm(world[:,None,:]-polygons[pid]['vertices'][None,:,:],axis=2),axis=1)))
                if error>.001:raise ValueError('Source surface/export vertex mismatch '+str((oid,sid,error)))
                maximum_error=max(maximum_error,error);surface_polygons[sid]=pid
            hlo=np.asarray([[h.minX,h.minY,h.minZ] for h in interior.convexHulls],np.float32)
            hhi=np.asarray([[h.maxX,h.maxY,h.maxZ] for h in interior.convexHulls],np.float32)
            if np.any(hlo>hhi):raise ValueError('Reversed DIF hull bounds')
            bbox=interior.boundingBox
            blo=np.asarray([bbox.minX,bbox.minY,bbox.minZ],np.float32)
            bhi=np.asarray([bbox.maxX,bbox.maxY,bbox.maxZ],np.float32)
            hull_polygons=[[surface_polygons[int(sid)] for sid in interior.hullSurfaceIndices[h.surfaceStart:h.surfaceStart+h.surfaceCount]]
                           for h in interior.convexHulls]
            self.objects[oid]=dict(inverse=inverse,scale=scale,hull_lo=hlo,hull_hi=hhi,bbox_lo=blo,bbox_hi=bhi,
                                   bins=interior.coordBins,indices=interior.coordBinIndices,hull_polygons=hull_polygons)
            self.mapping_checks.append(dict(object_id=oid,surfaces=len(surface_polygons),polygons=len(set(surface_polygons.values())),
                                            hulls=len(hull_polygons),maximum_vertex_match_error_u=maximum_error,
                                            source_sha256=source['sha256']))

    def ranks(self,oid,p,r):
        obj=self.objects[oid];inv=obj['inverse'];scale=obj['scale']
        lo=(p-r-.0001).astype(np.float32);hi=(p+r+.0001).astype(np.float32)
        center=((lo+hi)*np.float32(.5)@inv[:3,:3].T+inv[:3,3])/scale
        width=(hi-lo);broad=np.abs(inv[:3,:3])@width/scale
        query_lo=center-broad;query_hi=center+broad
        bbox_lo=obj['bbox_lo'][:2];bbox_hi=obj['bbox_hi'][:2];bin_size=(bbox_hi-bbox_lo)/np.float32(16.)
        starts=np.maximum(0,np.floor((np.maximum(bbox_lo,query_lo[:2])-bbox_lo)/bin_size).astype(int))
        ends=np.minimum(16,np.ceil((np.minimum(bbox_hi,query_hi[:2])-bbox_lo)/bin_size).astype(int))
        half=np.abs(inv[:3,:3])@(width*np.float32(.5))/scale
        # Conservative rotated-box cull. The engine also uses an oriented SAT
        # cull; rare rotated/nonuniform cases remain explicitly uncalibrated.
        actual_lo=center-half;actual_hi=center+half
        seen=set();ranks={}
        for x in range(starts[0],ends[0]):
            for y in range(starts[1],ends[1]):
                bin=obj['bins'][x*16+y]
                for hid in obj['indices'][bin.binStart:bin.binStart+bin.binCount]:
                    hid=int(hid)
                    if hid in seen:continue
                    seen.add(hid)
                    if np.any(obj['hull_hi'][hid]<query_lo) or np.any(obj['hull_lo'][hid]>query_hi):continue
                    if np.any(obj['hull_hi'][hid]<actual_lo) or np.any(obj['hull_lo'][hid]>actual_hi):continue
                    for pid in obj['hull_polygons'][hid]:
                        if pid not in ranks:ranks[pid]=len(ranks)
        return ranks

    def reorder(self,contacts,p,r):
        groups={}
        for c in contacts:
            oid=int(self.mesh.object_ids[c['triangle_id']]);groups.setdefault(oid,[]).append(c)
        result=[]
        for oid,group in groups.items():
            if oid in self.objects:
                ranks=self.ranks(oid,p,r)
                self.unranked_contacts+=sum(c['polygon_id'] not in ranks for c in group)
                group.sort(key=lambda c:ranks.get(c['polygon_id'],len(ranks)+c['polygon_id']))
            result.extend(group)
        return result
