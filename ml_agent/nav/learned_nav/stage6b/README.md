# Stage 6b scripts (2026-09-29 night / 09-30)

One-off scripts behind log section 34 (docs/HANDOFF_NAV_TRAINING_LOG.md), kept as they were run. They use absolute
paths to this checkout; run them from `ml_agent/`.

| script | what it does |
|---|---|
| s6_legs.py | stage 6 round logs: route legs (time from takeover, start speed), shortcut probes, detour legs |
| step_probe.py `<port>` | times bare bridge decisions under different render / sleep settings, then the game's wait as a function of the Python reply delay (start the script, then `marbleblast_mbx.exe -autotrain kotmjump_p0 -aiport <port>`) |
| step_probe_hold.py `<port>` | the same after holding one reply 5 s (measure the game's CPU meanwhile), with a blocking and a polling receive |
| hyb_sample.py `<port> <seconds>` | a sampling profiler around hybrid.play: where a decision's wall time goes |

The locked-session stall these were written for is described in log 34.2; the cause is still open.
The drill itself is a module: `nav/learned_nav/crossdrill.py` (starts from `cross_starts.py`, the fit in `cross_fit.py`),
launched with `nav/learned_nav/run_one.ps1`; the KOTM gate arms with `nav/learned_nav/run_many.ps1`.
