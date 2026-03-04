"""
splash.py — The Local Titan: Loading Splash Screen
====================================================

Displays a professional splash screen using tkinter (stdlib — no extra
dependencies) while the ~5.5 GB Qwen model is loaded from disk to RAM.

Features:
  - Dark-themed window matching the app's zinc/indigo palette.
  - Animated indigo progress bar with pulsing gradient.
  - Status text that updates as loading progresses.
  - Prevents the user from clicking the .exe multiple times
    (checks for existing splash via a named mutex on Windows).
  - Runs on a separate thread so the main thread can proceed
    with server startup.

Usage:
    from splash import SplashScreen
    splash = SplashScreen()
    splash.show()          # Non-blocking (runs in a thread)
    splash.update_status("Loading tokenizer...")
    splash.close()         # Dismiss when ready
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from typing import Optional

log = logging.getLogger("splash")


class SplashScreen:
    """Tkinter-based splash screen for model loading."""

    # ── Design tokens (matches index.py dark theme) ──────────────────
    _BG = "#09090b"           # zinc-950
    _BG_CARD = "#18181b"      # zinc-900
    _BORDER = "#3f3f46"       # zinc-700
    _TEXT = "#fafafa"          # zinc-50
    _TEXT_DIM = "#a1a1aa"      # zinc-400
    _ACCENT = "#6366f1"        # indigo-500
    _ACCENT_GLOW = "#818cf8"   # indigo-400

    _WIDTH = 520
    _HEIGHT = 280

    def __init__(self):
        self._root = None
        self._thread: Optional[threading.Thread] = None
        self._status_var = None
        self._progress_bar = None
        self._canvas = None
        self._running = threading.Event()
        self._close_requested = threading.Event()

        # Windows: prevent multiple instances via a named mutex
        self._mutex = None
        if sys.platform == "win32":
            try:
                import ctypes
                self._mutex = ctypes.windll.kernel32.CreateMutexW(
                    None, True, "LocalTitan_SplashMutex"
                )
                last_error = ctypes.windll.kernel32.GetLastError()
                if last_error == 183:  # ERROR_ALREADY_EXISTS
                    log.info("Splash already running — skipping.")
                    self._mutex = None
                    return
            except Exception:
                pass

    def show(self):
        """Show the splash screen on a background thread (non-blocking)."""
        if self._mutex is None and sys.platform == "win32":
            return  # Another instance is showing the splash

        self._thread = threading.Thread(
            target=self._run_tk_loop,
            daemon=True,
            name="splash-screen",
        )
        self._thread.start()
        # Wait for tkinter to initialize
        self._running.wait(timeout=5.0)

    def _run_tk_loop(self):
        """Create and run the tkinter splash window."""
        try:
            import tkinter as tk
        except ImportError:
            log.warning("tkinter not available — skipping splash screen.")
            self._running.set()
            return

        root = tk.Tk()
        self._root = root

        # ── Window setup ─────────────────────────────────────────────
        root.overrideredirect(True)  # No title bar
        root.attributes("-topmost", True)
        root.configure(bg=self._BG)
        root.resizable(False, False)

        # Center on screen
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        x = (screen_w - self._WIDTH) // 2
        y = (screen_h - self._HEIGHT) // 2
        root.geometry(f"{self._WIDTH}x{self._HEIGHT}+{x}+{y}")

        # Rounded corner effect (Windows 11)
        if sys.platform == "win32":
            try:
                from ctypes import windll, c_int, byref, sizeof
                DWMWA_WINDOW_CORNER_PREFERENCE = 33
                DWM_CORNERS_ROUND = c_int(2)
                hwnd = windll.user32.GetParent(root.winfo_id())
                windll.dwmapi.DwmSetWindowAttribute(
                    hwnd,
                    DWMWA_WINDOW_CORNER_PREFERENCE,
                    byref(DWM_CORNERS_ROUND),
                    sizeof(DWM_CORNERS_ROUND),
                )
            except Exception:
                pass

        # ── Content ──────────────────────────────────────────────────
        # Main card frame
        card = tk.Frame(
            root, bg=self._BG_CARD,
            highlightbackground=self._BORDER,
            highlightthickness=1,
        )
        card.pack(expand=True, fill="both", padx=2, pady=2)

        # CPU icon (text-based)
        icon_label = tk.Label(
            card, text="⚡", font=("Segoe UI Emoji", 28),
            bg=self._BG_CARD, fg=self._ACCENT,
        )
        icon_label.pack(pady=(30, 5))

        # Title
        title = tk.Label(
            card, text="The Local Titan",
            font=("Segoe UI", 20, "bold"),
            bg=self._BG_CARD, fg=self._TEXT,
        )
        title.pack()

        # Subtitle
        subtitle = tk.Label(
            card, text="Loading Intel Optimization...",
            font=("Segoe UI", 10),
            bg=self._BG_CARD, fg=self._TEXT_DIM,
        )
        subtitle.pack(pady=(2, 15))

        # ── Animated progress bar ────────────────────────────────────
        bar_frame = tk.Frame(card, bg=self._BG, height=4)
        bar_frame.pack(fill="x", padx=40)
        bar_frame.pack_propagate(False)

        canvas = tk.Canvas(
            bar_frame, bg=self._BG, height=4,
            highlightthickness=0, bd=0,
        )
        canvas.pack(fill="both", expand=True)
        self._canvas = canvas

        # ── Status text ──────────────────────────────────────────────
        self._status_var = tk.StringVar(
            value="Initializing Qwen 3.5-VL INT4 pipeline..."
        )
        status_label = tk.Label(
            card, textvariable=self._status_var,
            font=("Segoe UI", 9),
            bg=self._BG_CARD, fg=self._TEXT_DIM,
        )
        status_label.pack(pady=(12, 5))

        # Version tag
        ver_label = tk.Label(
            card, text="OpenVINO 2026.0 · INT4 · 100% Offline",
            font=("Segoe UI", 8),
            bg=self._BG_CARD, fg=self._BORDER,
        )
        ver_label.pack(pady=(0, 15))

        # Signal that the window is ready
        self._running.set()

        # Start animation
        self._animate_progress(canvas)

        # Poll for close request
        self._poll_close(root)

        # Run tkinter main loop
        try:
            root.mainloop()
        except Exception:
            pass

    def _animate_progress(self, canvas):
        """Animate a sliding gradient bar on the canvas."""
        if self._close_requested.is_set():
            return

        try:
            w = canvas.winfo_width()
            if w < 10:
                w = self._WIDTH - 80

            canvas.delete("bar")

            # Calculate position (oscillate back and forth)
            period = 2.0  # seconds per cycle
            t = (time.time() % period) / period
            # Ease in-out
            if t < 0.5:
                pos = 4 * t * t * t
            else:
                pos = 1 - pow(-2 * t + 2, 3) / 2

            bar_width = int(w * 0.35)
            x_start = int(pos * (w - bar_width))
            x_end = x_start + bar_width

            canvas.create_rectangle(
                x_start, 0, x_end, 4,
                fill=self._ACCENT, outline="",
                tags="bar",
            )

            # Schedule next frame (~60fps)
            canvas.after(16, self._animate_progress, canvas)
        except Exception:
            pass

    def _poll_close(self, root):
        """Check if close was requested from another thread."""
        if self._close_requested.is_set():
            try:
                root.destroy()
            except Exception:
                pass
            return
        root.after(100, self._poll_close, root)

    def update_status(self, text: str):
        """Update the status text from any thread."""
        if self._status_var and self._root:
            try:
                self._root.after(0, self._status_var.set, text)
            except Exception:
                pass

    def close(self):
        """Request the splash screen to close (thread-safe)."""
        self._close_requested.set()

        # Release Windows mutex
        if self._mutex and sys.platform == "win32":
            try:
                import ctypes
                ctypes.windll.kernel32.ReleaseMutex(self._mutex)
                ctypes.windll.kernel32.CloseHandle(self._mutex)
            except Exception:
                pass

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

        log.info("Splash screen closed.")
