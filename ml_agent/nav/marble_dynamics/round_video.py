"""Record a real planner round to a video that plays at 1x (operator viewing, 2026-10-16).

A journal replay into a fresh game is not exact over a whole round: start-up timing and contacts differ between runs, a
gem is then collected a few ticks apart and the next gem groups spawn elsewhere (replay_watch: 60 points vs 117 recorded
on Cube Isle). The faithful way to watch a round at real speed is to film the REAL run: one frame of the game window after
every 64 ms decision, written to an MP4 at 1000/64 = 15.625 fps. The planner's search pauses (the game is held in
lockstep) leave no trace in the video.

Used by reliability_eval --record-video PATH (visible window, every frame drawn, no real-time pacing). The window must
stay on screen and uncovered while recording (the frames are taken from the screen).
"""
import ctypes
import ctypes.wintypes as wt

import numpy as np

FPS = 1000. / 64.


class WindowRecorder:
    def __init__(self, pid, path, fps=FPS):
        try:
            ctypes.windll.user32.SetProcessDPIAware()   # screen coordinates in physical pixels, as ImageGrab uses
        except Exception:
            pass
        self.pid = int(pid); self.path = str(path); self.fps = fps
        self.hwnd = None; self.writer = None; self.size = None; self.frames = 0; self.misses = 0

    def _find(self):
        user32 = ctypes.windll.user32; found = []

        @ctypes.WINFUNCTYPE(ctypes.c_bool, wt.HWND, wt.LPARAM)
        def visit(hwnd, lparam):
            owner = wt.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
            if owner.value == self.pid and user32.IsWindowVisible(hwnd):
                r = wt.RECT()
                user32.GetClientRect(hwnd, ctypes.byref(r))
                if r.right - r.left > 200 and r.bottom - r.top > 150:
                    found.append((int((r.right - r.left) * (r.bottom - r.top)), hwnd))
            return True

        user32.EnumWindows(visit, 0)
        return max(found)[1] if found else None

    def _bbox(self):
        user32 = ctypes.windll.user32; r = wt.RECT(); pt = wt.POINT(0, 0)
        user32.GetClientRect(self.hwnd, ctypes.byref(r))
        user32.ClientToScreen(self.hwnd, ctypes.byref(pt))
        return pt.x, pt.y, pt.x + r.right, pt.y + r.bottom

    def start(self):
        self.hwnd = self._find()
        if self.hwnd is None:
            raise RuntimeError('game window not found for pid ' + str(self.pid))
        ctypes.windll.user32.ShowWindow(self.hwnd, 9)          # SW_RESTORE
        ctypes.windll.user32.SetForegroundWindow(self.hwnd)

    def capture(self):
        """One frame of the game window's client area (call after every decision)."""
        from PIL import ImageGrab
        import cv2
        try:
            if self.hwnd is None:
                self.start()
            image = ImageGrab.grab(bbox=self._bbox(), all_screens=True)
        except Exception:
            self.misses += 1
            return
        frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        if self.writer is None:
            h, w = frame.shape[:2]
            self.size = (w - w % 2, h - h % 2)
            self.writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, self.size)
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size)
        self.writer.write(frame)
        self.frames += 1

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        return dict(path=self.path, frames=self.frames, seconds=self.frames / self.fps, misses=self.misses)
