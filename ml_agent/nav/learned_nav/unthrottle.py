"""Opt processes out of Windows power throttling (EcoQoS) and background timer-resolution coarsening.

    python -m nav.learned_nav.unthrottle [pid ...]        (no pid: every marbleblast_mbx.exe, plus this process)

Under a locked or occluded session Windows 11 throttles a process whose window is not visible: its threads get a
fraction of a core and its 1 ms sleeps become ~200 ms. The lockstep bridge then paid ~200 ms for every reply that
missed the game's current main-loop iteration (2026-09-29 night: a KOTM round took 12 min instead of 40 s). The
operator's `powercfg /powerthrottling disable` (2026-09-16) covered marbleblast.exe only and needs admin; this uses
SetProcessInformation(ProcessPowerThrottling), which the process owner may call without admin.
"""
import ctypes
import ctypes.wintypes as wt
import subprocess
import sys

PROCESS_SET_INFORMATION = 0x0200
PROCESS_QUERY_INFORMATION = 0x0400
ProcessPowerThrottling = 4
PROCESS_POWER_THROTTLING_CURRENT_VERSION = 1
PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1
PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION = 0x4


class PROCESS_POWER_THROTTLING_STATE(ctypes.Structure):
    _fields_ = [('Version', wt.ULONG), ('ControlMask', wt.ULONG), ('StateMask', wt.ULONG)]


def unthrottle(pid):
    """True if the process is now exempt from execution-speed throttling and timer coarsening."""
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(PROCESS_SET_INFORMATION | PROCESS_QUERY_INFORMATION, False, int(pid))
    if not h:
        return False
    try:
        st = PROCESS_POWER_THROTTLING_STATE(PROCESS_POWER_THROTTLING_CURRENT_VERSION,
                                            PROCESS_POWER_THROTTLING_EXECUTION_SPEED | PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION,
                                            0)
        ok = k32.SetProcessInformation(h, ProcessPowerThrottling, ctypes.byref(st), ctypes.sizeof(st))
        if not ok:
            # older Windows: the timer-resolution bit is unknown; execution speed alone
            st.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED
            ok = k32.SetProcessInformation(h, ProcessPowerThrottling, ctypes.byref(st), ctypes.sizeof(st))
        return bool(ok)
    finally:
        k32.CloseHandle(h)


def game_pids(image='marbleblast_mbx.exe'):
    out = subprocess.run(['tasklist', '/FI', f'IMAGENAME eq {image}', '/FO', 'CSV', '/NH'], capture_output=True, text=True).stdout
    pids = []
    for line in out.splitlines():
        parts = [p.strip('"') for p in line.split('","')]
        if len(parts) > 1 and parts[0].lower() == image:
            try:
                pids.append(int(parts[1]))
            except ValueError:
                pass
    return pids


def unthrottle_games(log=None):
    """Every running game plus this process; returns the pids handled."""
    import os
    done = []
    for pid in game_pids() + [os.getpid()]:
        if unthrottle(pid):
            done.append(pid)
    if log is not None:
        log(f'unthrottle: {done}')
    return done


if __name__ == '__main__':
    if len(sys.argv) > 1:
        for a in sys.argv[1:]:
            print(a, unthrottle(int(a)))
    else:
        print(unthrottle_games())
