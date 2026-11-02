"""Compiled marble physics for the planner (2026-10-16): marble_sim.cpp (the 16 ms step and one 64 ms reply),
marble_sim.py (map loading, state packing), forecast.py (predict_native, the compiled predict_small_step)."""


def attach(engine, dll=None):
    """Give a GeneralPhysics engine the compiled forecast (bridge.predict_replies then uses it); returns the sim."""
    from nav.native.marble_sim import NativeSim
    engine.native_sim = NativeSim(engine, dll)
    from nav.native.route_native import register
    register(engine.native_sim)          # 2026-10-19: RoutePlanner.distance on its registered terrains runs in C++ (exact)
    engine.mesh._native_ray = engine.native_sim   # 2026-11-01: CollisionMap.raycast on this mesh runs in C++ (exact, verify mode PQ_VERIFY_PORT)
    return engine.native_sim
