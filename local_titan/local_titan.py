"""
app.py — The Local Titan: Reflex Application Entry Point
=========================================================

This is the main entry point for both `reflex run` (development)
and the PyInstaller-bundled .exe (production).
"""

import reflex as rx

from frontend.pages.index import index


# ═══════════════════════════════════════════════════════════════════════════
# APP CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
app = rx.App(
    theme=rx.theme(
        appearance="dark",
        accent_color="iris",
        radius="medium",
        scaling="100%",
    ),
    style={
        "font_family": "'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif",
        "::selection": {
            "background_color": "rgba(99, 102, 241, 0.3)",
        },
    },
    head_components=[
        rx.el.link(
            rel="preconnect",
            href="https://fonts.googleapis.com",
        ),
        rx.el.link(
            rel="preconnect",
            href="https://fonts.gstatic.com",
            crossorigin="",
        ),
        rx.el.link(
            href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap",
            rel="stylesheet",
        ),
    ],
)

from frontend.state import AppState

app.add_page(
    index,
    route="/",
    title="The Local Titan | Document Intelligence",
    description=(
        "Intel-optimized local AI for document field extraction, "
        "grounding, and PII redaction. Powered by Qwen 3.5-VL + OpenVINO."
    ),
    on_load=AppState.initialize_engine,
)
