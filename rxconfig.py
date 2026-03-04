"""
rxconfig.py — The Local Titan: Reflex Configuration
====================================================

Handles two execution contexts:
  1. Development:  `reflex run` — standard Reflex dev server behavior.
  2. Production:   PyInstaller-frozen .exe — static frontend assets are
                   read from the bundled `_static/` dir inside sys._MEIPASS,
                   and the SQLite DB writes to AppData (writable location).
"""

import os
import sys
from pathlib import Path

import reflex as rx

# ---------------------------------------------------------------------------
# Detect frozen (PyInstaller) vs development mode
# ---------------------------------------------------------------------------
IS_FROZEN = getattr(sys, "frozen", False)

if IS_FROZEN:
    # Inside PyInstaller bundle: sys._MEIPASS is the temp extraction root
    _MEIPASS = Path(sys._MEIPASS)  # type: ignore[attr-defined]

    # The Reflex compiled frontend was bundled at .web/_static
    _STATIC_DIR = _MEIPASS / ".web" / "_static"

    # Writable location for the SQLite DB (bundle root is read-only)
    if sys.platform == "win32":
        _APP_DATA = Path(os.environ.get("LOCALAPPDATA", ".")) / "LocalTitan"
    else:
        _APP_DATA = Path.home() / ".local_titan"

    _APP_DATA.mkdir(parents=True, exist_ok=True)
    _DB_URL = f"sqlite:///{_APP_DATA / 'reflex.db'}"

    config = rx.Config(
        app_name="local_titan",
        db_url=_DB_URL,
        frontend_port=3000,
        backend_port=8000,
        # Point Reflex to the pre-built static assets inside the bundle
        frontend_path=str(_STATIC_DIR) if _STATIC_DIR.exists() else None,
        # Disable hot-reload in production
        loglevel="warning",
    )

else:
    # Standard development mode
    config = rx.Config(
        app_name="local_titan",
        db_url="sqlite:///reflex.db",
        frontend_port=3000,
        backend_port=8000,
        api_url="http://localhost:8000",
    )
