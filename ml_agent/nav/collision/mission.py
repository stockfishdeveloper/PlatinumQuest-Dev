"""Static mission annotations and source coverage, independent of mesh conversion."""
import collections
import re

import numpy as np


TOKEN = re.compile(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|[A-Za-z_$][\w:$]*|[^\s]')


def mission_objects(text):
    """Read nested new Object blocks, excluding comments and children's fields."""
    tokens = [m.group() for m in TOKEN.finditer(text) if not m.group().startswith(('//','/*'))]
    objects, stack = [], []
    i = 0
    while i < len(tokens):
        if tokens[i] == 'new' and i+2 < len(tokens) and tokens[i+2] == '(':
            cl = tokens[i+1]
            j = i+3
            name = []
            while j < len(tokens) and tokens[j] != ')':
                name.append(tokens[j]); j += 1
            if j+1 < len(tokens) and tokens[j+1] == '{':
                obj = dict(class_name=cl, name=''.join(name), fields={})
                objects.append(obj)
                stack.append(obj)
                i = j+2
                continue
        if tokens[i] == '}' and stack:
            stack.pop()
        elif stack and i+1 < len(tokens) and tokens[i+1] == '=':
            key, j, value = tokens[i], i+2, []
            while j < len(tokens) and tokens[j] != ';':
                value.append(tokens[j]); j += 1
            v = ''.join(value)
            if len(value)==1 and v.startswith('"'):
                v = v[1:-1]
            stack[-1]['fields'][key.lower()] = v
            i = j
        i += 1
    return objects


def annotations_and_coverage(path, runtime):
    objects = mission_objects(path.read_text(errors='replace'))
    annotations, coverage = [], []
    for obj in objects:
        f, cl = obj['fields'], obj['class_name']
        pos = [float(v) for v in f.get('position','0 0 0').split()]
        if len(pos) != 3:
            continue
        db = f.get('datablock','')
        if cl in ('Item','Trigger') or ('spawn' in db.lower() and cl=='StaticShape'):
            annotations.append(dict(class_name=cl, datablock=db, position=pos,
                                    rotation=f.get('rotation','0 0 1 0'), scale=f.get('scale','1 1 1'),
                                    polyhedron=f.get('polyhedron'), source='mission_definition'))
        if cl not in ('InteriorInstance','TSStatic','StaticShape'):
            continue
        matches = [r for r in runtime if r['class']==cl and np.linalg.norm(np.asarray(r['position'])-pos)<.01]
        if db:
            matches = [r for r in matches if r['datablock'].lower()==db.lower()]
        if cl == 'InteriorInstance':
            matches = [r for r in matches if r['interior_file'].replace('\\','/').lower().endswith(
                f.get('interiorfile','').replace('\\','/').split('/')[-1].lower())]
        if cl == 'TSStatic' and any(r['shape_name'] for r in matches):
            matches = [r for r in matches if r['shape_name'].replace('\\','/').lower().endswith(
                f.get('shapename','').replace('\\','/').split('/')[-1].lower())]
        if cl == 'TSStatic' and len(matches)==1 and not matches[0]['shape_name']:
            # Some ghost fields cannot be obtained through getDataField. Keep
            # the provenance explicit instead of claiming a native filename.
            matches[0]['shape_name'] = f.get('shapename','')
            matches[0]['shape_name_provenance'] = 'mission_matched_class_and_position'
        # Missing disabled/no-collision shapes are reported, never silently called covered.
        coverage.append(dict(class_name=cl, datablock=db, position=pos,
                              source_file=f.get('interiorfile',f.get('shapename')),
                              runtime_matches=[r['id'] for r in matches],
                              collision_polygons=sum(r['polygons'] for r in matches)))
    return dict(mission_annotations=annotations, source_object_coverage=coverage,
                parsed_mission_classes=dict(collections.Counter(o['class_name'] for o in objects)))
