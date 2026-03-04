"""
main.py — The Local Titan: Stealth Launcher
=============================================

Professional entry point for both dev and frozen (.exe) contexts.

Launch sequence:
  1. Suppress subprocess console windows (frozen mode only).
  2. Show a tkinter splash screen while the model loads into RAM.
  3. Start the Reflex backend server in a background thread.
  4. Open the UI in a pywebview native window (preferred) or
     fall back to webbrowser.open_new_tab.
  5. Close the splash once the UI is ready.

Pre-flight checks (from the previous version) still run before any
UI work to catch missing model directories early.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("launcher")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
APP_NAME = "The Local Titan"
MODEL_DIR_NAME = "qwen_ov_int4"
MIN_MODEL_SIZE_GB = 1.5

REFLEX_HOST = "127.0.0.1"
REFLEX_PORT = 8000
IS_FROZEN = getattr(sys, "frozen", False)
# UI is on 3000 in dev mode, or same as backend in some prod configs
# For this launcher, we default to the Reflex frontend port
UI_URL = f"http://{REFLEX_HOST}:3000" if not IS_FROZEN else f"http://{REFLEX_HOST}:{REFLEX_PORT}"

_GB = 1024 ** 3
_MB = 1024 ** 2


# ═══════════════════════════════════════════════════════════════════════════
# CONSOLE SUPPRESSION (Windows frozen mode)
# ═══════════════════════════════════════════════════════════════════════════
def suppress_subprocess_console():
    """Prevent child processes from spawning visible console windows.

    When running as a PyInstaller --windowed .exe, Reflex (and any
    subprocess it spawns for Next.js) would flash a black console
    window.  We override subprocess.Popen's default to use CREATE_NO_WINDOW.
    """
    if sys.platform != "win32" or not IS_FROZEN:
        return

    # Patch subprocess.Popen to always suppress the console
    _original_popen = subprocess.Popen

    class SilentPopen(_original_popen):  # type: ignore[misc]
        def __init__(self, *args, **kwargs):
            if "creationflags" not in kwargs:
                CREATE_NO_WINDOW = 0x08000000
                kwargs["creationflags"] = CREATE_NO_WINDOW
            if kwargs.get("shell") is None:
                kwargs["shell"] = False
            super().__init__(*args, **kwargs)

    subprocess.Popen = SilentPopen  # type: ignore[misc]
    log.info("Console suppression active — no subprocess windows will appear.")


# ═══════════════════════════════════════════════════════════════════════════
# PATH RESOLUTION
# ═══════════════════════════════════════════════════════════════════════════
def _resolve_model_dir() -> Path:
    """Find the model directory in both frozen and dev contexts."""
    if IS_FROZEN:
        base = Path(sys._MEIPASS)  # type: ignore[attr-defined]
    else:
        base = Path.cwd()
    return base / MODEL_DIR_NAME


# ═══════════════════════════════════════════════════════════════════════════
# PRE-FLIGHT CHECKS
# ═══════════════════════════════════════════════════════════════════════════
def preflight_check() -> bool:
    """Validate that the model directory exists and has sufficient data."""
    log.info("╔══════════════════════════════════════════════════╗")
    log.info(f"║   {APP_NAME} — Pre-Flight Check               ║")
    log.info("╚══════════════════════════════════════════════════╝")

    model_dir = _resolve_model_dir()

    # Check 1: Directory exists
    if not model_dir.exists():
        log.error(
            f"Model directory NOT FOUND: {model_dir}\n\n"
            f"  Run: python model_convert.py\n"
        )
        return False
    if not model_dir.is_dir():
        log.error(f"'{model_dir}' exists but is not a directory.")
        return False
    log.info(f"  ✓ Model directory found: {model_dir}")

    # Check 2: IR files
    xml_files = list(model_dir.rglob("*.xml"))
    bin_files = list(model_dir.rglob("*.bin"))
    if not xml_files:
        log.error(f"No .xml IR files in: {model_dir}")
        return False
    log.info(f"  ✓ {len(xml_files)} .xml, {len(bin_files)} .bin file(s)")

    # Check 3: Minimum size
    total_bytes = sum(
        f.stat().st_size for f in model_dir.rglob("*") if f.is_file()
    )
    total_gb = total_bytes / _GB
    if total_gb < MIN_MODEL_SIZE_GB:
        log.error(
            f"Model too small: {total_bytes / _MB:,.1f} MB "
            f"(need ≥ {MIN_MODEL_SIZE_GB} GB)"
        )
        return False
    log.info(f"  ✓ Model size: {total_gb:.2f} GB")

    # Check 4: Tokenizer config
    tok_files = (
        list(model_dir.rglob("tokenizer_config.json"))
        + list(model_dir.rglob("tokenizer.json"))
    )
    if tok_files:
        log.info(f"  ✓ Tokenizer config found")
    else:
        log.warning("  ⚠ No tokenizer config — VLMPipeline may fail.")

    log.info("  All pre-flight checks PASSED.\n")
    return True


# ═══════════════════════════════════════════════════════════════════════════
# REFLEX SERVER (background thread)
# ═══════════════════════════════════════════════════════════════════════════
def _start_reflex_server():
    """Start the Reflex server. Use 'reflex run' in dev, or direct import in frozen."""
    import subprocess
    import os

    try:
        if not IS_FROZEN:
            log.info("Spawning 'reflex run'...")
            # For development, we must use the 'reflex run' command to get
            # the hot-reloading dev server and proper API/UI port mapping.
            # Using shell=True for Windows compatibility with npm-based commands.
            server_proc = subprocess.Popen(
                ["reflex", "run"],
                cwd=os.getcwd(),
                shell=True,
                env=os.environ.copy()
            )
            # Keep track of the process if needed, or just let it run
            # In index-locked environments, the main thread will exit if this fails.
        else:
            # In frozen mode, the app should be imported and handled differently
            # but for now, we'll assume the frozen exe handles its own entry.
            from local_titan import app  # noqa: F401
    except Exception as e:
        log.error(f"Reflex server failed: {e}", exc_info=True)


def _wait_for_server(timeout: float = 60.0) -> bool:
    """Poll the Reflex backend until it responds or timeout."""
    import urllib.request
    import urllib.error

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(UI_URL, timeout=2)
            return True
        except (urllib.error.URLError, OSError):
            time.sleep(0.5)
    return False


# ═══════════════════════════════════════════════════════════════════════════
# UI WINDOW (pywebview preferred, webbrowser fallback)
# ═══════════════════════════════════════════════════════════════════════════
def _open_native_window():
    """Open the UI in a native desktop window using pywebview.

    Falls back to the default browser if pywebview is not installed.
    pywebview gives us a chrome-less native window with no address bar,
    making the app feel like a desktop application.
    """
    try:
        import webview  # pywebview

        log.info("Opening native desktop window via pywebview...")
        webview.create_window(
            APP_NAME,
            url=UI_URL,
            width=1400,
            height=900,
            min_size=(1024, 700),
            text_select=True,
            confirm_close=True,
        )
        # webview.start() blocks until the window is closed
        webview.start(
            gui="edgechromium",  # Use Edge WebView2 on Windows
            debug=not IS_FROZEN,
        )
        log.info("Native window closed — shutting down.")
        os._exit(0)  # Clean exit after window close

    except ImportError:
        log.info(
            "pywebview not installed — falling back to browser.\n"
            "  Install for native window: pip install pywebview"
        )
        webbrowser.open_new_tab(UI_URL)

    except Exception as e:
        log.warning(f"pywebview failed ({e}) — falling back to browser.")
        webbrowser.open_new_tab(UI_URL)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    """Stealth launch sequence."""

    # Step 0: Suppress console windows in frozen mode
    suppress_subprocess_console()

    # Step 1: Pre-flight model validation
    if not preflight_check():
        log.error("═══════════════════════════════════════════════════")
        log.error("  PRE-FLIGHT FAILED — Application cannot start.")
        log.error("═══════════════════════════════════════════════════")
        if IS_FROZEN:
            # Show error in a message box for .exe users
            try:
                import ctypes
                ctypes.windll.user32.MessageBoxW(  # type: ignore
                    0,
                    "Model not found. Please ensure 'qwen_ov_int4/' "
                    "is in the same folder as the .exe.\n\n"
                    "Run model_convert.py to export the model.",
                    f"{APP_NAME} — Error",
                    0x10,  # MB_ICONERROR
                )
            except Exception:
                input("\nPress Enter to exit...")
        else:
            input("\nPress Enter to exit...")
        sys.exit(1)

    # Step 2: Show splash screen
    splash = None
    try:
        from splash import SplashScreen
        splash = SplashScreen()
        splash.show()
        log.info("Splash screen displayed.")
    except Exception as e:
        log.warning(f"Splash screen unavailable: {e}")

    # Step 3: Start Reflex server in background
    log.info("Starting Reflex backend server...")
    server_thread = threading.Thread(
        target=_start_reflex_server,
        daemon=True,
        name="reflex-server",
    )
    server_thread.start()

    # Step 4: Wait for server to be ready
    log.info(f"Waiting for server at {UI_URL}...")
    if _wait_for_server(timeout=90.0):
        log.info("Server is ready!")
    else:
        log.warning("Server did not respond within 90s — opening UI anyway.")

    # Step 5: Close splash, open UI
    if splash:
        try:
            splash.close()
        except Exception:
            pass

    _open_native_window()

    # Keep main thread alive if using browser fallback
    # (pywebview blocks and calls os._exit internally)
    try:
        server_thread.join()
    except KeyboardInterrupt:
        log.info("Interrupted — shutting down.")
        sys.exit(0)


if __name__ == "__main__":
    main()
