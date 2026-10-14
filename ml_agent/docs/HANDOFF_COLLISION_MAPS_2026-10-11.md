# Full collision map export and review viewer — 2026-10-11

**Current continuation (2026-10-13):** the map exports below now feed the experimental full-map navigator.
See [the fast navigator handoff](HANDOFF_FAST_NAVIGATOR_2026-10-13.md) for the confirmed parent,
frozen data-scaling recipe, running supervisor, distinct-state collection repair and current results.
The remainder of this document records the October 11 collision-export milestone.

Operator authorized conversion of all ten selected Hunt maps, an interactive 3D review of the exact model-facing
files, and engine/teleport verification. Accuracy and inclusion of all contact orientations take priority over
compression. **Completed:** model-facing collision files and offline interactive viewers exist for all ten maps.
At that export milestone, no policy training had started and the existing policy used its previous observations.

Open [the ten-map gallery](../collision_maps/viewer/index.html) in a browser (double-click the HTML file in Explorer).
Use orbit/pan, scroll zoom, material/direction colors, polygon boundaries, object visibility, “Focus game area”,
“Fit all geometry”, top/below views and surface selection. NPZ, metadata and OBJ links are available in each map.
No server or internet connection is needed to use the gallery.

## Implementation log

- Added `nav/collision/engine_geometry.inc`: read-only export from the engine's actual `PLC_Collision`
  polygon builders, using the marble's static collision mask. World transforms, normals, material friction,
  restitution and force are retained. No floor/climbability filter or nearby-only crop is applied.
- Instrumented the local engine checkout `C:/Users/doug/src/OpenPQ-TGEMIT-mbx` by copying that include into
  `engine/game/marble/aiGeometryExport.inc` and including it at the end of `marble.cc`. Built the `TGEMit_Demo`
  target with its existing Release/Ninja configuration. Copied the result to the separate
  `Marble Blast Platinum/marbleblast_geometry_probe.exe`; the regular game executable is not replaced.
- Added explicit `GEOMETRYEXPORT` and `GEOMETRYRAYS` probe commands to `mlAgent.cs`, guarded by native-function
  availability. These do not change normal actions, observations, training rewards or physics.
- Concentric pilot succeeded: 26 enabled collision-container objects, including all 16 TSStatic glass instances,
  and 1,458 raw polygons. Conversion yielded 2,733 triangles. Export commands execute after pending bridge
  observations, so the collector must advance neutral ticks until the output/ack arrives; checking immediately
  after `control()` incorrectly appeared to show a failed export. This is handled in the collector.
- Added a hash-checked NPZ/JSON map format and `CollisionMap` API: full world mesh, arbitrary AABB queries,
  triangle ray intersections, closest points and sphere overlaps, plus relative vertices/normals/material
  features for future model inputs. Sphere overlap is a geometric query, not a replacement dynamics simulator.
- Independent sight-ray checks found differences between native BSP/LOS rays and contact polygons. Added explicit
  disposable-session `GEOMETRYSWEEPS` checks using the actual `Marble::testMove` swept-sphere routine, and a geometric
  continuous sphere sweep query covering triangle faces, edges and vertices. These probes restore the marble's
  position/contact cache, but native shape collision notifications may be queued; never run a sweep batch during
  a policy evaluation/training round. Sight-ray disagreement is retained rather than mislabeled a passing check.
- Fixed probe integration: this engine supports only one argument to `isFunction`, unlike newer Torque variants.
  Guard the method by the global probe function's presence. Set the test timestep explicitly to 16 ms rather than
  inheriting the production environment's current 64 ms observation interval.
- Fixed continuous sphere queries to retain original native collision planes and polygon-boundary edge flags.
  Treating fan triangulation diagonals as physical polygon edges generated four premature collisions in the first
  Concentric marble-size sweep batch. After this fix, all 3,433 native sweep hit/miss and center checks agree,
  worst center error 0.000031 units. Five material seam/tie winners still differ and are explicitly reported;
  all candidate surfaces/materials are retained. Also replaced dot-product barycentric determinants with cross
  products to avoid cancellation on very thin exported triangles. Concentric live probes: 32 accepted teleports,
  all produced contact, including floors, walls, undersides, slopes and each of the 16 glass colliders.
- Added mission annotation parsing and explicit per-source runtime collider accounting. Fixed coverage metadata for
  Concentric: TSStatic ghosts returned an empty persistent `shapeName` field despite exporting their full collision
  geometry. An unambiguous class/position match fills the mission source filename with explicit provenance; it is
  not mislabeled a native filename. No mesh feature was missing. The first all-map pass stopped on this metadata
  assertion; the collector always closed its own game process.
- Added reproducible offline Plotly viewers and OBJ exports from `CollisionMap`, with physical material/direction
  coloring, per-object visibility, native polygon boundaries, item/spawn annotations, surface inspection, pan,
  orbit and zoom. Full geometry stays present, including large enclosure/background collision surfaces.
- First nine-map sphere checks: all hit/miss and center errors within 0.002 units. Two Duplex marble-size edge
  normal differences remain explicit; maximum center error 0.000863 units. One Cragmire live underside teleport
  was immediately pulled to spawn, so it is an interrupted probe, not evidence of a missing face or a passing
  physical test. Subsequent reports distinguish interruptions from valid live contacts.
- Accuracy audit found 58 zero-area fan triangles in the initial ten-map conversion. Changed conversion to retain
  their edges/corners instead of treating zero area as proof of non-contact: the native marble sweep tests those
  primitives too. Queries explicitly handle zero-area faces without numerical warnings. Re-running the complete
  ten-map verification after this change. Added explicit per-trial test resets to cancel pending OOB schedules;
  still classify a mid-trial respawn as interrupted, and preserve it rather than reporting contact success.
- Partial re-collection now merges the collection index instead of overwriting earlier maps, and uses each map's
  index in the canonical ten-map list for stable verification seeds. Raw engine snapshots and ray/sweep evidence
  are saved as compressed local artifacts so later query changes can be checked without losing probe evidence.
- Fixed the root cause of empty native asset names: client ghosts disable `ModStaticFields`, so `getDataField`
  returned empty for both InteriorInstance and TSStatic. The exporter now reads registered typed fields without
  changing any object flags. This allows exact filename matching for interiors that share the same position
  (Bowl, Cragmire and Treasure Box), replacing ambiguous position-only source accounting. Rebuilt the probe and
  re-collected the ten maps with the stricter source identity checks.
- Added analytic query tests for layered floors/ceilings, finite and one-sided rays, continuous sphere contact
  with faces/edges/corners, zero-area contact primitives and comment-safe nested mission parsing. Six passed.
  Added independent raw-vertex/polygon-area accounting, source asset hashes and viewer/model/browser hash checks.
- Browser checking found and fixed a JavaScript escaping error in the viewer generator: Python consumed the
  source-identity text's newline escapes. The HTML template is now a raw string. Tested per-object visibility,
  material/direction colors, boundaries, view presets and offline WebGL rendering in an owned headless Chrome.
- Improved camera focus using mission item/spawn extents while retaining full geometry and a “Fit all geometry”
  control. Cragmire's 400-unit enclosure otherwise dwarfed its actual game area. Added explicit orbit/pan controls.
- Extended verification beyond face targets: 200 known-clear airborne sweeps and 200 long sweeps per map.
  This exposed two query bugs that the short face probes did not cover. Edge/corner sweeps could return a contact
  past the segment endpoint; now they require `t < 1`. Some float32 fan vertices lie slightly off their native
  collision plane, producing tiny triangles with misleading geometric planes. Queries and the viewer now use
  faces projected onto the stored native plane, while the exact original float32 vertex coordinates remain in
  the NPZ and are used for native edge/corner checks. The largest correction is 0.000205 game units. AABB bounds
  include both source and contact coordinates. No coarse terrain grid or feature simplification is introduced.
- Fixed finite sphere regression coverage and added independent native-contact-point coverage checking. Every
  contact from the expanded initial batch lies within 0.002 units of a represented surface, including the long
  sweeps that disagree about first impact. Remaining first-impact disagreements are concentrated in Duplex's
  rotated, nonuniformly scaled arch parts. Investigating native long-step broadphase selection with shorter
  swept segments rather than claiming those long-step comparisons pass.
- MAJOR accuracy fix: using an object's own world AABB in `buildPolyList` did not export every hull of some rotated,
  nonuniformly scaled Duplex interiors. A padded cube query recovered **140 omitted triangles**. After re-export,
  Duplex's 10,383 triangles pass all 11,483 expanded native sphere sweeps with no exceptions and no unexplained
  contact points; worst center error 0.000034 units. The apparent edge-normal differences also disappeared.
  Added an independent check against each DIF's total hull-referenced surface count, rather than assuming an
  object-wide query is sufficient coverage. Re-collecting the complete ten-map set with this fix.
- Native long sweeps now run in segments no longer than one unit, reflecting local collision selection during
  actual gameplay rather than one artificial hundred-unit physics step. The geometric query tests the whole
  path continuously. The probe confirms it uses the client collision container. This is test instrumentation,
  not a modification to native gameplay physics. Original expanded-batch discrepancies were not suppressed.

## Final validation

The ten files contain **47,932 triangles/contact primitives**. All 58 InteriorInstance objects match the source
missions and account for every hull-referenced collision surface in the audited DIF files. All 42 Duplex glass
StaticShape objects and 16 Concentric TSStatic glass objects are represented and received individual live contact
probes. No climbability, walking-route, nearby-only, normal-direction or accessibility pruning was applied.

**58,932 native swept-sphere checks** include a target on every triangle, 500 marble-size targets, 200 edge targets,
200 known-clear airborne segments and 200 long segments per map. There are no hit/miss, contact-position, normal,
unrepresented-contact or unexplained-material mismatches in the final batch. The largest center difference is
0.000041 game units. Four Concentric material-boundary tie winners differ: both candidate materials/surfaces are
present, and the native winner depends on contact ordering/numerical details. The single-material sweep result
should not be treated as an exact native tie-breaker. Exact cases are preserved in Concentric's validation JSON.

**216 of 216 valid live teleport trials produced contact.** Two additional trials were interrupted by server
respawn and were excluded, with their telemetry retained. Seven analytic query/parser regression tests passed.
Source vertex/area accounting, all source asset hashes and all viewer identities passed. A headless browser
loaded and exercised every viewer with no JavaScript errors; it hashed the actual plotted coordinates and
matched the model-decoded geometry byte for byte, with the full triangle count. Preview PNGs are saved for
KOTM, Duplex, Cragmire and Concentric.

Native sight rays are also preserved as diagnostic evidence. Their BSP solid / DTS LOS semantics differ from
marble contact polygons, so their disagreement counts are not used as the collision acceptance criterion.
Statistical tests do not prove every possible trajectory, but source hull accounting plus complete retained
vertex/plane/material data and independent contact tests provide stronger coverage than floor teleports alone.

| Map | Contact primitives | Native sweeps | Valid live contacts |
|---|---:|---:|---:|
| King of the Marble | 1,459 | 2,559 | 15/15 |
| Sprawl | 4,594 | 5,694 | 16/16 |
| Cube Isle | 1,950 | 3,050 | 16/16 |
| Bowl | 2,716 | 3,816 | 16/16 |
| Duplex | 10,383 | 11,483 | 58/58 |
| Fractured Islands | 4,944 | 6,044 | 15/15 |
| Cragmire | 8,133 | 9,233 | 16/16 |
| Vortex Effect | 6,777 | 7,877 | 16/16 |
| Treasure Box | 4,240 | 5,340 | 16/16 |
| Concentric | 2,736 | 3,836 | 32/32 |

## Files and model query contract

- `collision_maps/<map>.npz`: native world-space source fan vertices, normals, planes, material/object/native
  surface tags and true polygon-boundary flags. Zero-area fans are retained because native edges/corners can
  contact the marble. `CollisionMap` decodes contact faces on their stored native planes; the maximum correction
  is 0.000205 units. Original source coordinates stay available as `source_triangles` and are used for edge/corner
  collision. The viewer and model features use the same decoded `triangles`.
- `<map>.json`: source mission/asset hashes, native object provenance/transforms, actual material properties,
  original raw snapshot hash, source coverage, interior hull coverage and mission item/spawn/trigger definitions.
  `resolved=false` means the native material callback returned null and the marble uses its actual fallback
  (friction 1, restitution 1, force 0); it is not a texture-derived guess.
- `<map>.validation.json`: native ray/sphere comparisons, exceptions and live teleport/contact telemetry.
  `snapshots/` preserves compressed engine snapshots and ray/sphere inputs/results. `verification.json`
  summarizes independent accounting, hashes, memory and local query timings.
- `viewer/`: ten offline HTML viewers, shared local Plotly library, gallery index, OBJ exports, browser coordinate
  hash checks and four PNG previews. Geometry is not downsampled. Visibility/camera controls affect display only.
- `nav/collision/`: decoder/query API, collector, mission annotations, viewer builder, checks, analytic tests,
  canonical native include and reproducible `build_probe.ps1`. Native source checkout was instrumented outside
  this repository; the include and build helper here preserve that change for handoff.

`CollisionMap` exposes all geometry and arbitrary AABB queries, finite face rays, closest points, sphere overlaps
and continuous sphere sweeps across faces/edges/corners. `features(indices, origin)` returns relative vertices,
normals, native planes in that frame, physical material values and provenance/boundary tags. Whole-map queries
are allowed; there is no nearest-K truncation. IDs are lookup/provenance identifiers, not recommended neural
numeric features. Native surface keys are not globally unique surface IDs. Sweeps require a clear start; overlap
queries are geometric contacts, not a solid-inside/outside classifier or a substitute dynamics simulator.

Mission markers are definitions, not a live claim that each gem/powerup exists. Trigger definitions are retained
as context, not rendered or queried as solid walls. Moving objects, other marbles, current powerup availability,
gravity/trigger effects and future trajectories need separate live state. No learned map encoder, world model,
powerup controller or training-observation migration was introduced. Connecting these files to that architecture
is the next task, after the operator's visual review.

## Reproduction

Run from the repository root in PowerShell, using the existing Python installation:

```powershell
$py = 'C:/Users/doug/AppData/Local/Programs/Python/Python39/python.exe'
$env:PYTHONPATH = 'ml_agent'
# Only needed when rebuilding the separate executable; uses the existing configured engine checkout.
& ./ml_agent/nav/collision/build_probe.ps1
& $py -m nav.collision.collect
& $py -m nav.collision.viewer
& $py -m unittest nav.collision.test_queries
# Optional browser verification requires Playwright and a Chrome executable.
& $py -m nav.collision.browser_check
& $py -m nav.collision.verify
```

The collector binds port 9051 before launching one hidden, owned game at a time, uses neutral actions / explicit
test teleports at 16 ms, and closes only its own game PID in `finally`. It does not load a policy or optimizer.
Do not send native sweep batches during a policy round: the routine may queue shape collision notifications.
Raw runtime work is under `%TEMP%/pq_checks/collision_maps_20261011`; compressed evidence is copied into the
artifact directory. Browser checking used Playwright installed into a task-specific temporary directory and a
separate headless Chrome session, leaving the user's browsing session alone. The engine build emits preexisting
VS/audio/cast warnings but succeeds. No recursive deletion or OneDrive-setting changes were performed.

Generated artifacts remain local. NPZ/JSON and the probe executable are ignored under the existing rules;
HTML/OBJ/PNG/compressed evidence appear as untracked files. No ignore rules were changed and no files were staged
or committed. Keep the complete `collision_maps/` directory when handing off the review: the HTML files require
the adjacent Plotly library and model metadata. No checkpoint promotions were made.

For the browser check on this computer, the verification-only dependencies are already installed here:

```powershell
$env:PYTHONPATH = 'ml_agent;' + (Join-Path $env:TEMP 'pq_checks/collision_maps_20261011/browser_deps')
& $py -m nav.collision.browser_check
```

`verify` checks the saved browser report as well as map data. Regenerate the browser report after regenerating
the viewer if source IDs/hashes changed. The viewers themselves have no Python or Playwright dependency.

Added regressions that directly exercise the discovered failures: an edge hit beyond the finite segment whose
AABB still touches the floor, and an off-plane float32 fan that previously created a false distant floor hit.
The final analytic test suite has **seven passing tests**. Native planes and material tables also match every
raw snapshot exactly. OBJ output uses 17 significant digits to preserve decoded double coordinates on import.

Native source baseline: `OpenPQ-TGEMIT-mbx` commit `cb44e8a69f7ae79ce349c8519bd519647f429bb6`, plus the include hook
described above. The regular `marbleblast_mbx.exe` remains 7,510,016 bytes with its original September 30 mtime;
SHA-256 `f5d765ec6bb6c39fcf1d59f93a80458b9ece9d4df475a0f7b3aec473400ba3d7`. The separate probe's hash and native
instrumentation hash are recorded in each map manifest. A final process check found no game/probe processes.

## Visual review update: material ambiguity callouts (2026-10-11)

- Added orange **A1–A4** callouts to Concentric's offline HTML, derived directly from the four native contact
  points in its validation records. A contact dot marks the exact point; a stem connects it to a diamond label
  four game units above the point so it is visible from the default view. Hover shows the game/query friction
  and restitution values. Clicking a callout shows the full comparison in the selected-surface panel.
- Added a visibility toggle and four sidebar focus buttons with world coordinates. The gallery's Concentric
  exception entry links to the annotated viewer. The generator retains these additions on regeneration; the
  collision NPZ, material records and original validation results are unchanged.
- Fixed a camera bug discovered during browser review: Plotly restyles can mutate its full-layout axis ranges,
  making later focus buttons target positions many map widths away. Focus now uses the original world-to-camera
  frame and waits for visibility/camera updates in sequence. Browser checks exercise all four buttons and the
  hide/show/refocus sequence, preventing that failure from silently returning.
- Rechecked all ten viewers: no JavaScript errors; plotted geometry hashes still match the model files. The four
  rendered contact coordinates match their native validation points exactly. All focus buttons and callout
  visibility checks pass. Evidence: `viewer/browser_checks.json`, updated Concentric preview, and
  `viewer/Concentric_Hunt.ambiguity_focus.png`.
- Operator visually confirmed that all four locations lie on normal-surface / high-friction tarmac seams.
  This is accepted material-boundary uncertainty for the representation; no surface is omitted. Future dynamics
  predictions should allow either material at an exact seam and favor maneuvers tolerant of a small landing
  displacement. This update implements visualization only; no dynamics or training behavior was changed.
