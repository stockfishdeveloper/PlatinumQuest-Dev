"""Diagnostic flat-contact model; not used by training or policy inference.

Equations follow Marble::computeMoveForces/applyContactForces in the local
OpenPQ-TGEMIT-mbx engine/game/marble/marble.cc. Restricted to a stationary,
horizontal, uninterrupted surface, ordinary marble, no jump/active effects.
No collision/edge handling is provided. Validate against game transitions
before using this in any controller. Inputs are in world coordinates; yaw is
the camera yaw actually applied, including a powerup yaw lock if present.
"""
import math
import numpy as np


def step(position, velocity, spin, world_input, yaw, dt=.064, radius=.18975,
         surface_friction=1.0, gravity=20.0, max_roll=15.0,
         angular_acceleration=75.0, braking_acceleration=30.0):
    """One held command, in <=8 ms engine substeps; return p, v, omega.

    All vectors have two horizontal components except three-component spin.
    Camera axes are independently clamped, as in the default game bridge.
    """
    p = np.asarray(position, dtype=float).copy()
    v = np.asarray(velocity, dtype=float).copy()
    w = np.asarray(spin, dtype=float).copy()
    if p.shape != (2,) or v.shape != (2,) or w.shape != (3,):
        raise ValueError('expected xy position/velocity and xyz spin')
    if dt <= 0 or radius <= 0:
        raise ValueError('positive duration and radius required')
    c, s = math.cos(yaw), math.sin(yaw)
    to_camera = np.array([[c, -s], [s, c]])
    command = np.clip(to_camera @ np.asarray(world_input, dtype=float), -1., 1.)
    centered = not np.any(command)
    remaining = dt
    while remaining > 1e-10:
        h = min(.008, remaining)
        remaining -= h
        roll = radius * np.array([w[1], -w[0]])
        desired = command * max_roll
        if not centered:
            current = to_camera @ roll
            keep = ((desired > 0) & (current > desired)) | ((desired < 0) & (current < desired))
            desired = np.where(keep, current, desired)
            world_desired = to_camera.T @ desired
            target_w = np.array([-world_desired[1], world_desired[0], 0.]) / radius
            control = target_w - w
            control *= min(1., angular_acceleration / max(np.linalg.norm(control), 1e-12))
        else:
            control = np.zeros(3)
        slip = v - roll
        slip_speed = np.linalg.norm(slip)
        friction_accel = .7 * surface_friction * gravity
        # Translation and rotation together remove 3.5 times the linear delta.
        slipping = slip_speed >= 3.5 * friction_accel * h and slip_speed > 0
        amount = min(friction_accel, slip_speed / (3.5 * h))
        friction_v = -slip * (amount / max(slip_speed, 1e-12))
        friction_w = 2.5 / radius * np.array([friction_v[1], -friction_v[0], 0.])
        accel = friction_v.copy()
        if not slipping:
            if centered:
                control = -w
                control *= min(1., braking_acceleration / max(np.linalg.norm(control), 1e-12))
            controlled_accel = radius * np.array([control[1], -control[0]])
            magnitude = np.linalg.norm(controlled_accel)
            if magnitude > 1.1 * surface_friction * gravity:
                controlled_accel *= friction_accel / magnitude
            accel += controlled_accel
        v += accel * h
        w += (control + friction_w) * h
        p += v * h
    return p, v, w


def step_batch(position, velocity, spin, world_input, yaw, dt=.064, radius=.18975,
               surface_friction=1.0):
    """Vectorized equivalent for independent candidate commands on ordinary floor."""
    p, v, w = (np.asarray(x, dtype=float).copy() for x in (position, velocity, spin))
    u = np.asarray(world_input, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or v.shape != p.shape or w.shape != (len(p), 3) or u.shape != p.shape:
        raise ValueError('expected N x 2 position, velocity, input and N x 3 spin')
    if dt <= 0 or radius <= 0:
        raise ValueError('positive duration and radius required')
    c, s = np.cos(yaw), np.sin(yaw)
    cmd = np.clip(np.stack((c * u[:, 0] - s * u[:, 1],
                            s * u[:, 0] + c * u[:, 1]), axis=-1), -1., 1.)
    centered = ~np.any(cmd, axis=-1)
    friction_accel = .7 * surface_friction * 20.
    remaining = dt
    while remaining > 1e-10:
        h = min(.008, remaining); remaining -= h
        roll = radius * np.stack((w[:, 1], -w[:, 0]), axis=-1)
        current = np.stack((c * roll[:, 0] - s * roll[:, 1],
                            s * roll[:, 0] + c * roll[:, 1]), axis=-1)
        desired = cmd * 15.
        keep = ((desired > 0) & (current > desired)) | ((desired < 0) & (current < desired))
        desired = np.where(keep, current, desired)
        dx, dy = c * desired[:, 0] + s * desired[:, 1], -s * desired[:, 0] + c * desired[:, 1]
        target_w = np.stack((-dy, dx, np.zeros(len(p))), axis=-1) / radius
        control = target_w - w
        control *= np.minimum(1., 75. / np.maximum(np.linalg.norm(control, axis=-1), 1e-12))[:, None]
        control[centered] = 0.
        slip = v - roll
        mag = np.linalg.norm(slip, axis=-1)
        slipping = (mag >= 3.5 * friction_accel * h) & (mag > 0)
        amount = np.minimum(friction_accel, mag / (3.5 * h))
        av = -slip * (amount / np.maximum(mag, 1e-12))[:, None]
        aw = 2.5 / radius * np.stack((av[:, 1], -av[:, 0], np.zeros(len(p))), axis=-1)
        center_control = -w * np.minimum(1., 30. / np.maximum(np.linalg.norm(w, axis=-1), 1e-12))[:, None]
        control = np.where((centered & ~slipping)[:, None], center_control, control)
        controlled = radius * np.stack((control[:, 1], -control[:, 0]), axis=-1)
        cmag = np.linalg.norm(controlled, axis=-1)
        scale = np.where(cmag > 1.1 * surface_friction * 20., friction_accel / np.maximum(cmag, 1e-12), 1.)
        av += np.where(slipping[:, None], 0., controlled * scale[:, None])
        v += av * h; w += (aw + control) * h; p += v * h
    return p, v, w
