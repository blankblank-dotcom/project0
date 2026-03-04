"""
frontend/pages/index.py — The Local Titan: Main Dashboard Page
==============================================================

Dark-mode split-view layout:
  Left panel  → Uploaded document image with SVG bounding-box overlays
  Right panel → Scrollable data table with extracted key-value pairs

Built with Reflex + Shadcn-inspired component styling.
"""

from __future__ import annotations

import reflex as rx

from frontend.state import AppState


# ═══════════════════════════════════════════════════════════════════════════
# COLOR TOKENS (Shadcn Dark Mode Palette)
# ═══════════════════════════════════════════════════════════════════════════
_BG_ROOT = "#09090b"          # zinc-950
_BG_CARD = "#18181b"          # zinc-900
_BG_CARD_HOVER = "#27272a"    # zinc-800
_BORDER = "#3f3f46"           # zinc-700
_TEXT_PRIMARY = "#fafafa"      # zinc-50
_TEXT_SECONDARY = "#a1a1aa"    # zinc-400
_TEXT_MUTED = "#71717a"        # zinc-500
_ACCENT = "#6366f1"           # indigo-500
_ACCENT_HOVER = "#818cf8"     # indigo-400
_ACCENT_GLOW = "rgba(99, 102, 241, 0.15)"
_DESTRUCTIVE = "#ef4444"      # red-500
_SUCCESS = "#22c55e"          # green-500
_WARNING = "#f59e0b"          # amber-500
_OVERLAY_BOX = "rgba(239, 68, 68, 0.30)"   # semi-transparent red
_OVERLAY_BORDER = "rgba(239, 68, 68, 0.80)"


# ═══════════════════════════════════════════════════════════════════════════
# SVG BOUNDING BOX OVERLAY
# ═══════════════════════════════════════════════════════════════════════════
def _bounding_box_svg() -> rx.Component:
    """Render SVG overlay with semi-transparent red rectangles for each
    detected bounding box. Coordinates are normalized [0,1] → viewBox %.
    """
    return rx.el.svg(
        rx.cond(
            AppState.bounding_boxes.length() > 0,
            rx.foreach(
                AppState.bounding_boxes,
                _render_single_box,
            ),
            rx.fragment(),
        ),
        view_box="0 0 100 100",
        preserve_aspect_ratio="none",
        xmlns="http://www.w3.org/2000/svg",
        style={
            "position": "absolute",
            "top": "0",
            "left": "0",
            "width": "100%",
            "height": "100%",
            "pointer_events": "none",
            "z_index": "10",
        },
    )


def _render_single_box(box: dict) -> rx.Component:
    """Render a single bounding box as an SVG rect + label.

    The dict has pre-computed SVG coordinates (as strings):
        x, y, width, height, label_y, label
    """
    return rx.fragment(
        rx.el.rect(
            x=box["x"],
            y=box["y"],
            width=box["width"],
            height=box["height"],
            fill=_OVERLAY_BOX,
            stroke=_OVERLAY_BORDER,
            stroke_width="0.3",
            rx="0.3",
        ),
        rx.el.text(
            box["label"],
            x=box["x"],
            y=box["label_y"],
            fill=_DESTRUCTIVE,
            font_size="2",
            font_family="Inter, system-ui, sans-serif",
            font_weight="600",
        ),
    )


# ═══════════════════════════════════════════════════════════════════════════
# HEADER BAR + "MODEL THINKING..." PROGRESS BAR
# ═══════════════════════════════════════════════════════════════════════════
def _thinking_progress_bar() -> rx.Component:
    """Indeterminate progress bar shown while the model is generating.

    This provides a strong visual signal that the P-cores are busy
    and the inference lock is held. The bar pulses with an indigo
    gradient animation.
    """
    return rx.cond(
        AppState.is_processing,
        rx.box(
            rx.box(
                background=f"linear-gradient(90deg, transparent, {_ACCENT}, transparent)",
                height="100%",
                width="40%",
                border_radius="2px",
                style={
                    "animation": "thinking-slide 1.5s ease-in-out infinite",
                },
            ),
            width="100%",
            height="3px",
            background=f"rgba(99, 102, 241, 0.1)",
            overflow="hidden",
            position="relative",
            # Keyframes injected via rx.script in the page layout
        ),
        # Hidden spacer when not processing (prevents layout shift)
        rx.box(height="3px", width="100%"),
    )


def _toast_notification() -> rx.Component:
    """Shadcn-style toast notification overlay.
    
    Slide-in animation from the bottom-right for system alerts.
    """
    return rx.cond(
        AppState.show_toast,
        rx.box(
            rx.hstack(
                rx.icon(
                    rx.match(
                        AppState.toast_type,
                        ("success", "check-circle"),
                        ("warning", "triangle-alert"),
                        ("error", "circle-x"),
                        "info",
                    ),
                    size=18,
                    color=rx.match(
                        AppState.toast_type,
                        ("success", _SUCCESS),
                        ("warning", _WARNING),
                        ("error", _DESTRUCTIVE),
                        _ACCENT,
                    ),
                ),
                rx.text(
                    AppState.toast_message,
                    size="2",
                    weight="medium",
                    color=_TEXT_PRIMARY,
                ),
                spacing="3",
                align="center",
            ),
            position="fixed",
            bottom="24px",
            right="24px",
            background=_BG_CARD,
            border=f"1px solid {_BORDER}",
            border_radius="8px",
            padding="12px 20px",
            z_index="1000",
            style={
                "animation": "toast-slide-in 0.3s ease-out",
                "box_shadow": "0 10px 15px -3px rgba(0, 0, 0, 0.4)",
            },
        ),
        rx.fragment(),
    )



def _header() -> rx.Component:
    return rx.box(
        rx.hstack(
            # Logo + Title
            rx.hstack(
                rx.icon("cpu", size=22, color=_ACCENT),
                rx.heading(
                    "The Local Titan",
                    size="5",
                    weight="bold",
                    style={"color": _TEXT_PRIMARY},
                ),
                rx.badge(
                    "INT4 · Offline",
                    variant="outline",
                    color_scheme="iris",
                    size="1",
                ),
                rx.cond(
                    AppState.document_type != "Unknown",
                    rx.badge(
                        AppState.document_type,
                        color_scheme="blue",
                        variant="solid",
                        size="2",
                    ),
                    rx.fragment(),
                ),
                spacing="3",
                align="center",
            ),
            # Global Search Bar
            rx.hstack(
                rx.input(
                    rx.input.slot(rx.icon("search", size=14, color=_TEXT_MUTED)),
                    placeholder="Search documents...",
                    value=AppState.search_query,
                    on_change=AppState.set_search_query,
                    on_key_down=AppState.perform_semantic_search_on_enter,
                    on_blur=AppState.perform_semantic_search,
                    style={
                        "width": "300px",
                        "background": _BG_ROOT,
                        "border": f"1px solid {_BORDER}",
                        "color": _TEXT_PRIMARY,
                        "border_radius": "6px",
                    },
                ),
                # Search Results Popover/Dropdown (Simplified as a cond for this demo)
                rx.cond(
                    AppState.is_searching,
                    rx.spinner(size="1", color=_ACCENT),
                ),
                spacing="2",
                align="center",
            ),
            
            # Status indicator + Hardware badge
            rx.hstack(
                rx.cond(
                    AppState.is_engine_ready,
                    rx.badge("● Engine Ready", color_scheme="green", size="1"),
                    rx.badge("○ Loading...", color_scheme="yellow", size="1"),
                ),
                # Hardware device badge — shows which chip is doing inference
                rx.cond(
                    AppState.active_device != "",
                    rx.badge(
                        rx.hstack(
                            rx.icon(
                                AppState.active_device_icon,
                                size=12,
                            ),
                            rx.text(
                                AppState.active_device,
                                size="1",
                            ),
                            spacing="1",
                            align="center",
                        ),
                        color_scheme=AppState.active_device_color,
                        variant="surface",
                        size="1",
                    ),
                    rx.fragment(),
                ),
                rx.cond(
                    AppState.is_processing,
                    rx.hstack(
                        rx.spinner(size="1"),
                        rx.text(
                            AppState.processing_status,
                            size="1",
                            color=_TEXT_SECONDARY,
                        ),
                        spacing="2",
                        align="center",
                    ),
                    rx.fragment(),
                ),
                spacing="3",
                align="center",
            ),
            justify="between",
            align="center",
            width="100%",
        ),
        padding="16px 24px",
        border_bottom=f"1px solid {_BORDER}",
        background=_BG_CARD,
        backdrop_filter="blur(12px)",
    )


# ═══════════════════════════════════════════════════════════════════════════
# LEFT PANEL — DOCUMENT VIEWER + SVG OVERLAYS
# ═══════════════════════════════════════════════════════════════════════════
def _upload_zone() -> rx.Component:
    """Dropzone shown when no document is loaded."""
    return rx.upload(
        rx.center(
            rx.vstack(
                rx.icon("upload", size=48, color=_TEXT_MUTED),
                rx.text(
                    "Drop a document here",
                    size="4",
                    weight="medium",
                    color=_TEXT_SECONDARY,
                ),
                rx.text(
                    "PDF · PNG · JPG · TIFF",
                    size="2",
                    color=_TEXT_MUTED,
                ),
                spacing="2",
                align="center",
            ),
            height="100%",
        ),
        id="doc_upload",
        accept={
            "application/pdf": [".pdf"],
            "image/png": [".png"],
            "image/jpeg": [".jpg", ".jpeg"],
            "image/tiff": [".tiff"],
        },
        max_files=1,
        border=f"2px dashed {_BORDER}",
        border_radius="12px",
        padding="48px",
        width="100%",
        height="400px",
        cursor="pointer",
        on_drop=AppState.handle_upload(rx.upload_files(upload_id="doc_upload")),
        style={
            "_hover": {
                "border_color": _ACCENT,
                "background": _ACCENT_GLOW,
            },
            "transition": "all 0.2s ease",
        },
    )


def _document_viewer() -> rx.Component:
    """Image viewer with SVG bounding box overlay."""
    return rx.box(
        # The image
        rx.image(
            src=AppState.display_image,
            width="100%",
            height="auto",
            border_radius="8px",
            style={"display": "block"},
        ),
        # SVG overlay (positioned absolutely on top)
        _bounding_box_svg(),
        position="relative",
        width="100%",
        border_radius="8px",
        overflow="hidden",
    )


def _page_navigation() -> rx.Component:
    """PDF page navigation controls."""
    return rx.cond(
        AppState.page_count > 1,
        rx.hstack(
            rx.icon_button(
                rx.icon("chevron-left", size=16),
                variant="ghost",
                size="1",
                on_click=AppState.prev_page,
                disabled=AppState.current_page <= 1,
            ),
            rx.text(
                AppState.page_indicator,
                size="2",
                color=_TEXT_SECONDARY,
            ),
            rx.icon_button(
                rx.icon("chevron-right", size=16),
                variant="ghost",
                size="1",
                on_click=AppState.next_page,
                disabled=AppState.current_page >= AppState.page_count,
            ),
            spacing="2",
            align="center",
            justify="center",
        ),
        rx.fragment(),
    )


def _left_panel() -> rx.Component:
    """Left panel: upload zone or document viewer with overlays."""
    return rx.box(
        rx.vstack(
            # Panel header
            rx.hstack(
                rx.hstack(
                    rx.icon("file-scan", size=16, color=_ACCENT),
                    rx.text(
                        "Document Viewer",
                        size="2",
                        weight="bold",
                        color=_TEXT_PRIMARY,
                    ),
                    spacing="2",
                    align="center",
                ),
                rx.cond(
                    AppState.has_document,
                    rx.hstack(
                        rx.cond(
                            AppState.redacted_image,
                            rx.icon_button(
                                rx.icon(
                                    rx.cond(
                                        AppState.show_redacted,
                                        "eye-off",
                                        "eye",
                                    ),
                                    size=14,
                                ),
                                variant="ghost",
                                size="1",
                                on_click=AppState.toggle_redacted_view,
                                title="Toggle redacted view",
                            ),
                            rx.fragment(),
                        ),
                        rx.icon_button(
                            rx.icon("x", size=14),
                            variant="ghost",
                            size="1",
                            color_scheme="red",
                            on_click=AppState.clear_document,
                            title="Clear document",
                        ),
                        spacing="1",
                    ),
                    rx.fragment(),
                ),
                justify="between",
                align="center",
                width="100%",
            ),
            # Content
            rx.cond(
                AppState.has_document,
                rx.vstack(
                    _document_viewer(),
                    _page_navigation(),
                    spacing="3",
                    width="100%",
                ),
                _upload_zone(),
            ),
            spacing="4",
            width="100%",
        ),
        background=_BG_CARD,
        border=f"1px solid {_BORDER}",
        border_radius="12px",
        padding="20px",
        flex="1",
        min_width="0",
    )


# ═══════════════════════════════════════════════════════════════════════════
# RIGHT PANEL — DATA TABLE + ACTIONS
# ═══════════════════════════════════════════════════════════════════════════
def _action_buttons() -> rx.Component:
    """Action buttons for extraction, redaction, and download."""
    return rx.vstack(
        rx.hstack(
            rx.button(
                rx.hstack(
                    rx.icon("scan-search", size=14),
                    rx.text("Extract Fields"),
                    spacing="2",
                    align="center",
                ),
                on_click=AppState.run_extraction,
                disabled=~AppState.has_document | AppState.is_processing | ~AppState.is_engine_ready,
                variant="solid",
                color_scheme="iris",
                size="2",
                style={
                    "cursor": rx.cond(
                        AppState.has_document & ~AppState.is_processing,
                        "pointer",
                        "not-allowed",
                    ),
                },
            ),
            rx.button(
                rx.hstack(
                    rx.icon("shield-alert", size=14),
                    rx.text("Detect PII"),
                    spacing="2",
                    align="center",
                ),
                on_click=AppState.run_pii_redaction,
                disabled=~AppState.has_document | AppState.is_processing,
                variant="outline",
                color_scheme="red",
                size="2",
            ),
            rx.button(
                rx.hstack(
                    rx.icon("download", size=14),
                    rx.text("Export Excel"),
                    spacing="2",
                    align="center",
                ),
                on_click=AppState.download_results,
                disabled=~AppState.has_results | AppState.is_processing,
                variant="outline",
                color_scheme="green",
                size="2",
            ),
            spacing="2",
            width="100%",
            flex_wrap="wrap",
        ),
        # Batch progress ring (visible during batch processing)
        rx.cond(
            AppState.is_batch_active,
            _batch_progress_ring(),
            rx.fragment(),
        ),
        spacing="3",
        width="100%",
    )


def _batch_progress_ring() -> rx.Component:
    """Circular progress indicator for batch document processing."""
    return rx.hstack(
        # SVG progress ring
        rx.el.svg(
            # Background track
            rx.el.circle(
                cx="18",
                cy="18",
                r="15.9",
                fill="none",
                stroke=_BORDER,
                stroke_width="2",
            ),
            # Progress arc
            rx.el.circle(
                cx="18",
                cy="18",
                r="15.9",
                fill="none",
                stroke=_ACCENT,
                stroke_width="2.5",
                stroke_linecap="round",
                stroke_dasharray="100 100",
                stroke_dashoffset=rx.cond(
                    AppState.batch_progress > 0,
                    100 - (AppState.batch_progress * 100),
                    100,
                ),
                transform="rotate(-90 18 18)",
                style={"transition": "stroke-dashoffset 0.4s ease"},
            ),
            view_box="0 0 36 36",
            width="40",
            height="40",
        ),
        rx.vstack(
            rx.text(
                AppState.batch_progress_label,
                size="2",
                weight="bold",
                color=_TEXT_PRIMARY,
            ),
            rx.hstack(
                rx.text(
                    AppState.processing_status,
                    size="1",
                    color=_TEXT_SECONDARY,
                    max_width="450px",
                    overflow="hidden",
                    text_overflow="ellipsis",
                    white_space="nowrap",
                ),
                rx.cond(
                    AppState.time_remaining_str != "",
                    rx.badge(
                        AppState.time_remaining_str,
                        variant="surface",
                        color_scheme="blue",
                    ),
                    rx.fragment(),
                ),
                spacing="2",
                align="center",
            ),
            spacing="1",
        ),
        spacing="3",
        align="center",
        padding="12px",
        background=_BG_ROOT,
        border_radius="8px",
        border=f"1px solid {_BORDER}",
        width="100%",
    )


def _data_table() -> rx.Component:
    """Scrollable table of extracted key-value pairs."""
    return rx.cond(
        AppState.has_results,
        rx.box(
            rx.table.root(
                rx.table.header(
                    rx.table.row(
                        rx.table.column_header_cell(
                            "Field",
                            style={"color": _TEXT_SECONDARY, "font_size": "12px"},
                        ),
                        rx.table.column_header_cell(
                            "Value",
                            style={"color": _TEXT_SECONDARY, "font_size": "12px"},
                        ),
                    ),
                ),
                rx.table.body(
                    rx.foreach(
                        AppState.output_data,
                        lambda row: rx.table.row(
                            rx.table.cell(
                                rx.text(row["field"], size="2", color=_ACCENT),
                            ),
                            rx.table.cell(
                                rx.text(row["value"], size="2", color=_TEXT_PRIMARY),
                            ),
                            style={
                                "_hover": {"background": _BG_CARD_HOVER},
                                "transition": "background 0.15s ease",
                            },
                        ),
                    ),
                ),
                variant="ghost",
                size="1",
                width="100%",
            ),
            max_height="500px",
            overflow_y="auto",
            border_radius="8px",
            border=f"1px solid {_BORDER}",
            style={
                "&::-webkit-scrollbar": {"width": "6px"},
                "&::-webkit-scrollbar-track": {"background": _BG_ROOT},
                "&::-webkit-scrollbar-thumb": {
                    "background": _BORDER,
                    "border_radius": "3px",
                },
            },
        ),
        # Empty state
        rx.center(
            rx.vstack(
                rx.icon("table-2", size=36, color=_TEXT_MUTED),
                rx.text(
                    "No data extracted yet",
                    size="2",
                    color=_TEXT_MUTED,
                ),
                rx.text(
                    "Upload and click Extract Fields, or ask a question below.",
                    size="1",
                    color=_TEXT_MUTED,
                ),
                # CTA for Chat
                rx.button(
                    rx.hstack(
                        rx.icon("message-square-plus", size=14),
                        rx.text("Ask Knowledge Assistant"),
                        spacing="2",
                        align="center",
                    ),
                    on_click=AppState.toggle_chat,
                    variant="soft",
                    color_scheme="iris",
                    size="1",
                    margin_top="12px",
                ),
                spacing="2",
                align="center",
            ),
            height="200px",
            border=f"1px dashed {_BORDER}",
            border_radius="8px",
        ),
    )


def _raw_output_panel() -> rx.Component:
    """Collapsible raw model output viewer."""
    return rx.cond(
        AppState.raw_model_output != "",
        rx.box(
            rx.hstack(
                rx.icon("terminal", size=14, color=_TEXT_MUTED),
                rx.text("Raw Model Output", size="1", color=_TEXT_MUTED),
                spacing="2",
                align="center",
            ),
            rx.box(
                rx.code(
                    AppState.raw_model_output,
                    style={
                        "white_space": "pre-wrap",
                        "word_break": "break-all",
                        "font_size": "11px",
                        "color": _TEXT_SECONDARY,
                        "background": _BG_ROOT,
                        "padding": "12px",
                        "border_radius": "6px",
                        "max_height": "200px",
                        "overflow_y": "auto",
                        "display": "block",
                    },
                ),
                margin_top="8px",
            ),
            padding="12px",
            border=f"1px solid {_BORDER}",
            border_radius="8px",
            margin_top="4px",
        ),
        rx.fragment(),
    )


def _right_panel() -> rx.Component:
    """Right panel: actions + data table + raw output."""
    return rx.box(
        rx.vstack(
            # Panel header
            rx.hstack(
                rx.icon("table-2", size=16, color=_ACCENT),
                rx.text(
                    "Extracted Data",
                    size="2",
                    weight="bold",
                    color=_TEXT_PRIMARY,
                ),
                spacing="2",
                align="center",
            ),
            # Action buttons
            _action_buttons(),
            # Error display
            rx.cond(
                AppState.error_message != "",
                rx.callout(
                    AppState.error_message,
                    icon="triangle-alert",
                    color_scheme="red",
                    size="1",
                    width="100%",
                ),
                rx.fragment(),
            ),
            # Data table
            _data_table(),
            # Raw output
            _raw_output_panel(),
            spacing="4",
            width="100%",
        ),
        background=_BG_CARD,
        border=f"1px solid {_BORDER}",
        border_radius="12px",
        padding="20px",
        flex="1",
        min_width="0",
    )


# ═══════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════
def _footer() -> rx.Component:
    """Professional status bar with real-time metrics."""
    return rx.box(
        rx.hstack(
            # Left side: Mode info
            rx.hstack(
                rx.text(
                    "Titan Cluster Status:",
                    size="1",
                    weight="bold",
                    color=_TEXT_MUTED,
                ),
                rx.text(
                    rx.cond(
                        AppState.is_queue_paused,
                        "PAUSED (RAM Safety)",
                        "100% Offline · INT4 Optimized"
                    ),
                    size="1",
                    color=rx.cond(AppState.is_queue_paused, _DESTRUCTIVE, _TEXT_SECONDARY),
                ),
                rx.hstack(
                    rx.icon("heart", size=10, color=rx.cond(AppState.is_queue_paused, _WARNING, _ACCENT)),
                    rx.text(f"Pulse: {AppState.last_heartbeat}", size="1", color=_TEXT_MUTED),
                    spacing="1",
                    padding_left="12px",
                ),
                spacing="2",
            ),
            
            # Right side: Metrics
            rx.hstack(
                # Latency
                rx.hstack(
                    rx.icon("timer", size=12, color=_ACCENT),
                    rx.text(
                        rx.cond(
                            AppState.inference_latency_ms > 0,
                            f"{AppState.inference_latency_ms}ms / page",
                            "--- ms / page"
                        ),
                        size="1",
                    ),
                    spacing="1",
                    padding_right="12px",
                ),
                # RAM
                rx.hstack(
                    rx.icon("activity", size=12, color=rx.cond((AppState.ram_total_gb > 0) & (AppState.ram_usage_gb > AppState.ram_total_gb * 0.8), _DESTRUCTIVE, _SUCCESS)),
                    rx.text(
                        f"RAM: {AppState.ram_usage_gb}GB / {AppState.ram_total_gb}GB",
                        size="1",
                    ),
                    spacing="1",
                    padding_right="4px",
                ),
                # Privacy Shield (Encryption)
                rx.popover.root(
                    rx.popover.trigger(
                        rx.icon_button(
                            rx.icon(rx.cond(AppState.is_encrypted, "lock", "unlock"), size=12),
                            variant="ghost",
                            color_scheme=rx.cond(AppState.is_encrypted, "green", "gray"),
                            size="1",
                        ),
                    ),
                    rx.popover.content(
                        rx.vstack(
                            rx.hstack(
                                rx.icon("shield-check", color=_ACCENT, size=16),
                                rx.text("Privacy Shield", weight="bold", size="2"),
                                spacing="2",
                            ),
                            rx.text(
                                "Set a master password to encrypt your local results and RAG memory.",
                                size="1",
                                color=_TEXT_MUTED,
                            ),
                            rx.input(
                                placeholder="Master Password...",
                                type="password",
                                value=AppState.master_password,
                                on_change=AppState.set_master_password,
                                size="1",
                                variant="surface",
                            ),
                            rx.text(
                                rx.cond(
                                    AppState.is_encrypted,
                                    "✓ Value-level encryption active.",
                                    "⚠ Storage is currently plain text."
                                ),
                                size="1",
                                color=rx.cond(AppState.is_encrypted, _SUCCESS, _WARNING),
                            ),
                            spacing="3",
                            padding="4px",
                        ),
                        side="top",
                        align="end",
                        width="240px",
                    ),
                ),
                spacing="3",
                align="center",
            ),
            justify="between",
            width="100%",
        ),
        padding="8px 24px",
        background=_BG_ROOT,
        border_top=f"1px solid {_BORDER}",
        backdrop_filter="blur(8px)",
    )




# ═══════════════════════════════════════════════════════════════════════════
# KNOWLEDGE ASSISTANT (CHAT)
# ═══════════════════════════════════════════════════════════════════════════
def _chat_drawer() -> rx.Component:
    """A slide-out drawer for the Knowledge Assistant."""
    return rx.drawer.root(
        rx.drawer.trigger(
            rx.button(
                rx.icon("message-square-quote", size=16),
                "Knowledge Assistant",
                variant="soft",
                color_scheme="iris",
                position="fixed",
                bottom="80px",
                right="24px",
                z_index="100",
                style={"border_radius": "20px", "box_shadow": "0 4px 12px rgba(99, 102, 241, 0.3)"}
            )
        ),
        rx.drawer.overlay(background_color="rgba(0, 0, 0, 0.4)"),
        rx.drawer.portal(
            rx.drawer.content(
                rx.vstack(
                    rx.hstack(
                        rx.icon("brain-circuit", color=_ACCENT),
                        rx.heading("Knowledge Assistant", size="4"),
                        rx.spacer(),
                        rx.drawer.close(rx.icon("x", cursor="pointer")),
                        width="100%",
                        padding="20px",
                        border_bottom=f"1px solid {_BORDER}",
                    ),
                    # Chat Messages
                    rx.box(
                        rx.vstack(
                            rx.foreach(
                                AppState.chat_history,
                                lambda msg: rx.box(
                                    rx.text(
                                        msg["content"],
                                        size="2",
                                        padding="10px 14px",
                                        background=rx.cond(msg["role"] == "user", _ACCENT, _BG_CARD_HOVER),
                                        border_radius="12px",
                                        max_width="85%",
                                    ),
                                    display="flex",
                                    justify_content=rx.cond(msg["role"] == "user", "flex-end", "flex-start"),
                                    width="100%",
                                )
                            ),
                            spacing="3",
                            padding="20px",
                        ),
                        flex="1",
                        overflow_y="auto",
                        width="100%",
                    ),
                    # Input
                    rx.hstack(
                        rx.input(
                            placeholder="Ask about your document history...",
                            value=AppState.chat_input,
                            on_change=AppState.set_chat_input,
                            on_key_down=AppState.chat_on_enter,
                            flex="1",
                            variant="surface",
                        ),
                        rx.icon_button(
                            rx.icon("send", size=16),
                            on_click=AppState.chat_with_knowledge_base,
                            loading=AppState.is_chatting,
                            variant="solid",
                            color_scheme="iris",
                        ),
                        padding="20px",
                        border_top=f"1px solid {_BORDER}",
                        width="100%",
                        background=_BG_CARD,
                    ),
                    height="100%",
                    background=_BG_ROOT,
                    color=_TEXT_PRIMARY,
                ),
                top="0",
                bottom="0",
                right="0",
                width="400px",
                height="100%",
                background=_BG_ROOT,
                border_left=f"1px solid {_BORDER}",
            )
        ),
        open=AppState.is_chat_open,
        on_open_change=AppState.set_is_chat_open,
    )


def _search_results_overlay() -> rx.Component:
    """Floating search results list."""
    return rx.cond(
        AppState.search_results.length() > 0,
        rx.box(
            rx.vstack(
                rx.text("Library Results", size="1", weight="bold", color=_TEXT_MUTED, padding_x="12px", padding_y="8px"),
                rx.foreach(
                    AppState.search_results,
                    lambda res: rx.box(
                        rx.vstack(
                            rx.hstack(
                                rx.icon("file-text", size=12, color=_ACCENT),
                                rx.text(res["filename"], size="1", weight="bold", color=_TEXT_PRIMARY),
                                spacing="2",
                            ),
                            rx.text(res["text"], size="1", color=_TEXT_SECONDARY, line_clamp=2),
                            spacing="1",
                            align_items="start",
                        ),
                        padding="10px 12px",
                        cursor="pointer",
                        _hover={"background": _BG_CARD_HOVER},
                        border_radius="6px",
                    )
                ),
                spacing="0",
            ),
            position="absolute",
            top="70px",
            left="24px",
            width="320px",
            background=_BG_CARD,
            border=f"1px solid {_BORDER}",
            border_radius="8px",
            z_index="1000",
            box_shadow="0 10px 30px rgba(0,0,0,0.5)",
            padding="4px",
        ),
        rx.fragment()
    )


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PAGE
# ═══════════════════════════════════════════════════════════════════════════
def index() -> rx.Component:
    return rx.box(
        rx.script(
            """
            if (!document.getElementById('titan-styles')) {
                const style = document.createElement('style');
                style.id = 'titan-styles';
                style.textContent = `
                    @keyframes thinking-slide {
                        0% { transform: translateX(-150%); }
                        100% { transform: translateX(250%); }
                    }
                    @keyframes toast-slide-in {
                        0% { transform: translateY(100%) scale(0.9); opacity: 0; }
                        100% { transform: translateY(0) scale(1); opacity: 1; }
                    }
                `;
                document.head.appendChild(style);
            }
            """
        ),
        rx.vstack(
            _header(),
            _thinking_progress_bar(),
            rx.hstack(
                _left_panel(),
                _right_panel(),
                width="100%",
                spacing="6",
                padding="24px",
                align="start",
            ),
            _search_results_overlay(),
            _chat_drawer(),
            _toast_notification(),
            _footer(),
            width="100%",
            min_height="100vh",
            background=_BG_ROOT,
            spacing="0",
        ),
        id="app_root",
        style={
            "font-family": "Inter, system-ui, sans-serif",
            "color": _TEXT_PRIMARY,
        },
    )
