# real_run with a chosen terrain map file for this run only (2026-10-05; evaluation). NAV_MAP_FILE = path to the .npz.
import os, sys, runpy
sys.path.insert(0, os.getcwd())
import terrain_obs
path = os.environ.get('NAV_MAP_FILE')
if path:
    _orig = terrain_obs.TerrainMap.resolve
    terrain_obs.TerrainMap.resolve = staticmethod(lambda name_or_path: path)
print('terrain map file', path or '(default)', flush=True)
sys.argv = ['nav.real_run']
runpy.run_module('nav.real_run', run_name='__main__')
