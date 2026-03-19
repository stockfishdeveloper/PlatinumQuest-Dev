# Camera Angle Investigation — 2026-03-17

## The Problem

The ML agent (PPO, 2D continuous action space) was getting 70+ gems/game during training but performed inconsistently when the camera was rotated. At certain camera angles performance dropped ~15%, with the marble missing gems by ~10 degrees — just enough to overshoot, turn around, and waste time. The model never completely broke at any angle, which made this subtle and confusing.

## Why This Shouldn't Happen

All observations are rotated into camera space by observer.cs using `$cameraYaw`. All actions (fwd/back/left/right) are camera-relative by the Torque engine. If the rotation is correct, the model should see an identical situation regardless of camera angle. Camera direction should be invisible to the model.

## The Investigation

### 1. Camera Test Script (`camera_test.py`)

Built a script that plays 36 games at camera angles 0-350 degrees (10-degree increments), recording gems per game. The model sets the camera via a 5th field in the action protocol, parsed by `executeAction` in mlAgent.cs.

**Result:** Clear periodic dip pattern. Best angles got 82-86 gems, worst got 38-54 gems. Dips at roughly 0, 60, 120, 180, 230, 290, 350 degrees.

### 2. Reproducibility Check

Ran the test twice. Correlation between runs: **0.97**. This is a real, deterministic pattern, not noise.

### 3. First Hypothesis: `$cameraYaw` vs `setCameraYaw` Desync

Discovered that `$MP::MyMarble.setCameraYaw()` only updates the marble's internal camera, while `$cameraYaw` (the global variable observer.cs reads) is separate. Fixed mlAgent.cs to use `setMarbleCamYaw()` which updates both.

**Result:** No change. Same pattern.

### 4. Position Hypothesis

Absolute position in camera space is the only non-invariant observation (world position rotated by camera yaw gives different numbers at different angles). Created `camera_test_nopos.py` which zeroes out obs[0:3] (position) before feeding to the model.

**Result:** No change. Same pattern. Position is not the culprit.

### 5. Engine Force Direction Test (`camera_action_test.py`)

Tested whether `$mvForwardAction=1.0` actually pushes the marble in the camera direction. Set camera to each angle, pushed forward from rest, measured velocity direction.

**First version** (short settle time): Wild results — offsets up to 21 degrees, velocity magnitudes varying 3x. Looked like a massive engine bug.

**Second version** (`camera_action_test2.py`, 3-second settle): Offset was a consistent **-1.8 degrees at every angle**. The first test was contaminated by residual velocity from previous tests. The engine is fine.

### 6. `$mvYaw` Drift Hypothesis

Discovered that `AIAgent::clearInputs()` never zeroes `$mvYaw`, `$mvYawLeftSpeed`, or `$mvYawRightSpeed`. These are per-tick camera rotation deltas. Any residual from mouse input would cause continuous camera drift.

Fixed `clearInputs()` to zero all three variables.

**Result:** No change. Same pattern. `$mvYaw` was already zero during testing.

### 7. Real Observation Probe (`probe_real_obs.py`)

Hooked into the Actor's forward pass to capture pre-tanh and post-tanh values with real game observations. This disproved the earlier theory that tanh was saturated — pre-tanh values were tiny (+-0.03), well within the linear region. The model IS actively steering toward gems and tracking their angle precisely.

### 8. The Breakthrough: `camera_obs_compare.py`

Captured raw observations at camera=0 and camera=90 with the marble at the same world position, same gem visible.

**At 0 degrees:** Gem (3, 17) in world space appeared as **(3.0, 17.0)** in camera space. Correct.

**At 90 degrees:** Same gem should appear as **(-17.0, 3.0)** in camera space. Actual: **(-12.68, 11.71)**.

The rotation was **wrong**. Back-solving: the effective yaw was 57.3 degrees (1.0 radian) when we set 90 degrees (1.5708 radians). The `$cameraYaw` global and the marble's internal `getCameraYaw()` were returning different values.

## The Root Cause

Observer.cs read camera yaw from `$cameraYaw` (a TorqueScript global variable), but the engine applies movement forces relative to `$MP::MyMarble.getCameraYaw()` (the marble object's internal camera). These two values can diverge because:

- `playGui.cs` modifies `$cameraYaw` every frame by adding `$mvYawLeftSpeed` / `$mvYawRightSpeed`
- The engine modifies the marble's internal camera by processing the Move struct (`$mvYaw` delta)
- `getMarbleCamYaw()` reads the marble's internal value and writes it to `$cameraYaw`, but this only happens when called explicitly
- Various code paths set one but not the other

The result: observer.cs rotated observations by one angle, but the engine applied movement at a slightly different angle. The model learned to compensate for this mismatch at its training camera angle, but the compensation was wrong at other angles, creating the periodic performance dips.

## The Fix

**Changed observer.cs to read yaw from `$MP::MyMarble.getCameraYaw()` instead of `$cameraYaw`.**

Three locations in observer.cs were changed:

1. `collectSelfState()` (line 65) — position and velocity rotation
2. `collectGems()` (line 125) — gem relative position rotation
3. `collectOpponents()` (line 247) — opponent rotation

This ensures observations are rotated by the exact same yaw the engine uses for movement. The `$cameraYaw` global is no longer used for any AI observation.

**Verification:** After the fix, `camera_obs_compare.py` showed gem (3, 17) correctly appearing as (-17.0, 3.0) at camera=90 degrees.

## Consequence

The fix changes the observation values slightly (by however much `$cameraYaw` diverged from `getCameraYaw()` during training). The existing trained model was calibrated to the old (wrong) values, so it performs worse with the fix applied. A fresh training run is required.

Additionally, camera yaw randomization was added to the training script — random yaw at episode start and after each OOB — so the model learns to be camera-invariant from the start.

## Files in This Directory

| File | Purpose |
|------|---------|
| `camera_test.py` | Main test: 36 games at 10-degree increments, records gems/game |
| `camera_test_nopos.py` | Same test but zeroes position obs — ruled out position as cause |
| `camera_spin_test.py` | Slowly rotates camera 360 degrees to verify camera control works |
| `camera_action_test.py` | Tests if engine forward force aligns with camera (v1, flawed) |
| `camera_action_test2.py` | Same test with proper settle time — confirmed engine is fine |
| `camera_obs_compare.py` | Compares raw observations at two camera angles — found the bug |
| `probe_real_obs.py` | Hooks into Actor to log pre/post tanh values with real game data |
| `camera_test_results.json` | Results from camera_test.py (multiple runs, all identical pattern) |
| `camera_test_results.html` | Polar chart visualization of results |
| `camera_test_nopos_results.json` | Results with position zeroed |
| `camera_action_test_results.json` | Engine force direction test v1 results |
| `camera_action_test2_results.json` | Engine force direction test v2 results |
