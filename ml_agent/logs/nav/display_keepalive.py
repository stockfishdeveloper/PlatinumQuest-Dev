"""Keep the display ON while training runs (2026-09-22). Windows 11 drops a process's 1 ms timer
resolution when its window is not visible; when the display sleeps every game window becomes invisible,
the engine's sleeps go from 1 ms to 15.6 ms, and training throughput halves (wall_s 16 -> 35 per update,
measured 03:36 when the display timed out 60 min after the last game launch). No admin needed; kill to undo."""
import ctypes, time
ES_CONTINUOUS, ES_DISPLAY_REQUIRED, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000002, 0x00000001
user32 = ctypes.WinDLL('user32'); kernel32 = ctypes.WinDLL('kernel32')
user32.mouse_event(0x0001, 1, 0, 0, 0); user32.mouse_event(0x0001, -1, 0, 0, 0)   # nudge: wakes the display
kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_DISPLAY_REQUIRED | ES_SYSTEM_REQUIRED)
print('display keepalive armed', flush=True)
while True:
    time.sleep(50)
    kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_DISPLAY_REQUIRED | ES_SYSTEM_REQUIRED)
