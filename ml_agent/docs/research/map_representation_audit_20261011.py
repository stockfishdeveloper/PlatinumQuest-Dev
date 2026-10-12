import collections, hashlib, json, math, os, re, sys, time
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
ML = ROOT / 'ml_agent'
sys.path.insert(0, str(ML))
import hxDif
from generate_terrain_map import resolve_mission, parse_interiors, transform_surfaces, climbable, material_friction, friction_table
from generate_map_topdown import extract_surfaces
from nav.learned_nav.geometry import material_table
names = json.loads((ML / 'configs/real_maps_20261009.json').read_text())['maps']
materials = material_table()
cache = {}
report = {'schema': 'map_representation_research_v1', 'date': '2026-10-11', 'method': 'Read-only static extraction and artifact audit; no game probes, training, encoder changes, or cache rebuilds.', 'maps': []}
for name in names:
    begin = time.perf_counter()
    mission = Path(resolve_mission(name))
    text = mission.read_text(errors='replace')
    classes = collections.Counter(re.findall(r'\bnew\s+(\w+)\s*\(', text))
    object_blocks = [(cl, body) for cl, body in re.findall(r'\bnew\s+(\w+)\s*\([^)]*\)\s*\{([^{}]*)\}', text, re.S)]
    other_objects = []
    trigger_types = collections.Counter()
    for cl, body in object_blocks:
        db = re.search(r'dataBlock\s*=\s*"([^"]+)"', body)
        desc = {'class': cl, 'datablock': db.group(1) if db else None}
        if cl in ('StaticShape', 'TSStatic', 'PathedInterior', 'TerrainBlock'):
            other_objects.append(desc)
        if cl == 'Trigger':
            trigger_types[desc['datablock']] += 1
    triangles = 0
    orientation = collections.Counter()
    tex = collections.Counter()
    floorlike_count = 0
    points = []
    sources = []
    instances = parse_interiors(str(mission))
    for dif_path, pos, rot, scale in instances:
        if dif_path not in cache:
            d = hxDif.Dif.Load(dif_path)
            level = d.interiors[0]
            hull_surfaces = [int(x) for x in level.hullSurfaceIndices]
            source_info = {'path': str(Path(dif_path).relative_to(ROOT)), 'sha256': hashlib.sha256(Path(dif_path).read_bytes()).hexdigest(), 'detail_levels': len(d.interiors), 'render_surface_count_level0': len(level.surfaces), 'null_surfaces_level0': len(level.nullSurfaces), 'null_surfaces_referenced_by_hulls': len({x & 0x7fffffff for x in hull_surfaces if x & 0x80000000}), 'render_surfaces_referenced_by_hulls': len({x for x in hull_surfaces if not x & 0x80000000}), 'convex_hulls_level0': len(level.convexHulls), 'static_mesh_count_level0': getattr(level, 'numStaticMeshes', None)}
            # Mirrors current extract_surfaces: all DIF detail levels, so report each source's level count explicitly.
            cache[dif_path] = (extract_surfaces(d, with_material=True), source_info)
        surfaces, source_info = cache[dif_path]
        sources.append(dict(source_info, position=list(pos), rotation=list(rot), scale=list(scale)))
        for verts, normal, material in transform_surfaces(surfaces, pos, rot, scale):
            if len(verts) < 3:
                continue
            v = np.asarray(verts)
            points.append(v)
            cnt = sum(np.linalg.norm(np.cross(v[j] - v[0], v[j+1] - v[0])) >= 1e-9 for j in range(1, len(v)-1))
            triangles += int(cnt)
            nz = normal[2]
            orientation['up' if nz > 0.3 else 'down' if nz < -0.3 else 'side'] += int(cnt)
            if climbable(normal, material_friction(material)):
                floorlike_count += int(cnt)
            tex[str(material).split('/')[-1].lower()] += int(cnt)
    p = np.concatenate(points)
    mn, mx = p.min(axis=0), p.max(axis=0)
    path = ML / 'terrain_maps' / ('terrain_' + name + '.npz')
    with np.load(path, allow_pickle=False) as data:
        h = data['heights']
        occupied_levels = np.isfinite(h).sum(axis=0)
        terrain = {'keys': list(data.files), 'shape': list(h.shape), 'resolution': float(data['grid_res']), 'extent_xy_u': [round(float(data['xs'][-1]-data['xs'][0]), 3), round(float(data['ys'][-1]-data['ys'][0]), 3)], 'height_min_max_u': [float(np.nanmin(h)), float(np.nanmax(h))], 'maximum_stored_levels': int(occupied_levels.max()), 'multilevel_cells': int((occupied_levels > 1).sum()), 'uncompressed_bytes': int(sum(data[k].nbytes for k in data.files)), 'compressed_bytes': path.stat().st_size}
    est_mesh_bytes = triangles * (9*4 + 3*4 + 4)
    dense_dims = np.ceil((mx-mn)/0.25).astype(np.int64) + 1
    row = {'map': name, 'mission': str(mission.relative_to(ROOT)), 'mission_sha256': hashlib.sha256(mission.read_bytes()).hexdigest(), 'mission_classes': dict(classes), 'trigger_datablocks': dict(trigger_types), 'non_interior_geometry_objects': other_objects, 'interior_instances': len(instances), 'source_files': sources, 'render_mesh_candidate_triangles': triangles, 'triangle_orientations': dict(orientation), 'triangles_accepted_by_current_climbable_rule': floorlike_count, 'mesh_bounds_min': mn.tolist(), 'mesh_bounds_max': mx.tolist(), 'mesh_extents_u': (mx-mn).round(3).tolist(), 'estimated_float32_triangle_normal_material_bytes': est_mesh_bytes, 'dense_025u_volume_dimensions': dense_dims.tolist(), 'dense_025u_volume_bytes_per_channel': int(math.prod(int(x) for x in dense_dims) * 4), 'terrain_artifact': terrain, 'material_triangle_counts': dict(tex), 'known_material_properties': {m: list(materials[m]) for m in tex if m in materials}, 'unmapped_texture_names_defaulted': [m for m in tex if m not in materials], 'audit_seconds': round(time.perf_counter()-begin, 3)}
    current_materials = friction_table()
    row['material_friction_mismatches_old_exporter_vs_current'] = {
        k: {'old': materials.get(k, ('DefaultMaterial', 1., 1., 0.))[1], 'current': current_materials[k]}
        for k in tex if k in current_materials
        and abs(materials.get(k, ('DefaultMaterial', 1., 1., 0.))[1] - current_materials[k]) > 1e-8
    }
    report['maps'].append(row)
    print(json.dumps({'map': name, 'triangles': triangles, 'up_side_down': dict(orientation), 'extents': row['mesh_extents_u'], 'null_collision_surfaces': sum(s['null_surfaces_referenced_by_hulls'] for s in sources), 'other_geometry': other_objects, 'seconds': row['audit_seconds']}), flush=True)
report['notes'] = ['Triangle counts describe a candidate mesh from the existing render-surface extractor, not a verified full collision export.', 'Dense voxel estimates use full candidate mesh AABB at 0.25 u and float32; include decorative geometry and are upper-bound design comparisons, not measured network memory.', 'Material defaults and source hashes require runtime validation; static mission and internal replay state are not equivalent to information visible on a remote client.', 'Layer count reports stored terrain, not uncapped physical layer count.', 'No updated mesh artifacts were generated.']
report['totals'] = {'maps': len(report['maps']), 'candidate_triangles': sum(r['render_mesh_candidate_triangles'] for r in report['maps']), 'unique_dif_files_read': len(cache)}
report['material_table_check'] = {
    'older_exporter_mapping_entries': len(materials),
    'current_friction_mapping_entries': len(friction_table()),
    'mismatch_maps': sum(bool(m['material_friction_mismatches_old_exporter_vs_current']) for m in report['maps']),
}
report['reproduction_script'] = 'ml_agent/docs/research/map_representation_audit_20261011.py'
unique_sources = [item[1] for item in cache.values()]
report['collision_source_check'] = {
    'detail_level_counts': sorted({s['detail_levels'] for s in unique_sources}),
    'hull_referenced_null_surfaces': sum(s['null_surfaces_referenced_by_hulls'] for s in unique_sources),
}
report['notes'].append('Null-surface and detail-level observations are reported in collision_source_check; they do not prove full mesh coverage.')
output = ML / 'docs/MAP_REPRESENTATION_AUDIT_2026-10-11.json'
tmp = output.with_suffix('.tmp.json')
tmp.write_text(json.dumps(report, indent=2), encoding='utf-8')
os.replace(str(tmp), str(output))
print('saved ' + str(output), flush=True)
