"""Paths, provenance, and engine setup shared by the standalone skill tools."""
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ML = ROOT / 'ml_agent'
ART = ML / 'dynamics_reports' / 'edge_skill_20261102'
NIGHT = ML / 'dynamics_reports' / 'edge_skill_20261103'
RUNTIME = Path(os.environ.get('TEMP', '.')) / 'pq_checks' / 'edge_skill_20261102'
MAP_DIR = ROOT / 'Marble Blast Platinum/platinum/data/multiplayer/hunt/custom'
ASSET = ROOT / 'Marble Blast Platinum/platinum/data/interiors_mbp/neutral2_square.dif'


def rotation(cell):
    a=math.radians(cell['yaw_deg'])
    return np.array([[math.cos(a),-math.sin(a),0.],[math.sin(a),math.cos(a),0.],[0.,0.,1.]])


def local(cell,points):
    return (np.asarray(points)-np.asarray(cell['origin']))@rotation(cell)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    from nav.collision.mesh import atomic_json
    atomic_json(Path(path), value)


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def engine_for(name):
    from nav.collision.mesh import CollisionMap
    from nav.marble_dynamics.physics import GeneralPhysics
    from nav.native import attach
    path=NIGHT/'geometry'/(name+'.npz')
    if not path.exists(): path=ART/'geometry'/(name+'.npz')
    if not path.exists(): path=ML/'collision_maps'/(name+'.npz')
    engine = GeneralPhysics(CollisionMap(path))
    attach(engine)
    engine.bridge_physics_ms = 16
    engine.planning_deadline = None
    return engine
