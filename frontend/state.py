"""
frontend/state.py — The Local Titan: Reflex Application State
=============================================================

Central state management for the entire application. All UI reactivity
flows through this single State class.

Key design decisions:
  - Async event handlers: inference runs in a background task so the UI
    thread never blocks (critical for perceived responsiveness).
  - Bounding boxes stored as normalized [ymin, xmin, ymax, xmax] dicts
    matching the engine's GroundedObject output — the UI maps these to
    SVG overlays without any coordinate transform.
  - Image data round-trips as base64 data URIs to avoid filesystem
    temp-file management in the bundled .exe.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import sys
import tempfile
import time
import psutil
from pathlib import Path

from typing import Optional, List, Dict, Any
from datetime import datetime

import reflex as rx
from PIL import Image
from pydantic import BaseModel, Field, ValidationError

# Local Backend
from backend.engine import InferenceEngine, EngineResult, discover_best_device
from backend import db

# ---------------------------------------------------------------------------
# Backend Management (Singleton for non-picklable objects)
# ---------------------------------------------------------------------------
class BackendManager:
    """Manages non-serializable backend objects outside the Reflex state.
    
    Reflex attempts to pickle the entire AppState for persistence/syncing.
    Objects like InferenceEngine (containing OpenVINO pipelines) cannot
    be pickled, leading to StateSerializationErrors.
    """
    engine: Optional[InferenceEngine] = None
    queue = None  # Will hold ProcessingQueue instance
    current_pil_image: Optional[Image.Image] = None
    pdf_pages: list[Any] = []

backend = BackendManager()

log = logging.getLogger("state")


# ---------------------------------------------------------------------------
# Writable App Data Directory (safe inside frozen .exe)
# ---------------------------------------------------------------------------
def get_app_data_dir() -> Path:
    """Return a writable directory for temp files and local data.

    Inside a PyInstaller-frozen .exe the working directory and the
    bundle root (sys._MEIPASS) are both READ-ONLY. We must use a
    user-writable location instead.

    On Windows: %LOCALAPPDATA%/LocalTitan   (e.g. C:/Users/X/AppData/Local/LocalTitan)
    Fallback:   system temp directory
    """
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir()))
    else:
        base = Path(tempfile.gettempdir())

    app_dir = base / "LocalTitan"
    app_dir.mkdir(parents=True, exist_ok=True)
    return app_dir


def get_temp_dir() -> Path:
    """Return a writable temp directory under the app data folder."""
    tmp = get_app_data_dir() / "temp"
    tmp.mkdir(parents=True, exist_ok=True)
    return tmp


def _obj_to_svg_dict(obj) -> dict:
    """Convert a GroundedObject to a flat dict with pre-computed SVG values.

    All values are strings so the state var can be typed as
    list[dict[str, str]], which Reflex can serialize for rx.foreach.
    """
    ymin, xmin, ymax, xmax = obj.box_2d
    return {
        "label": f"{obj.label}: {obj.value}" if obj.value else obj.label,
        "x": str(xmin * 100),
        "y": str(ymin * 100),
        "width": str((xmax - xmin) * 100),
        "height": str((ymax - ymin) * 100),
        "label_y": str(ymin * 100 - 0.5),
    }


class _ParsedField(BaseModel):
    label: str = Field(min_length=1)
    value: str = ""
    box_2d: list[float] = Field(default_factory=list)

    @property
    def has_box(self) -> bool:
        return len(self.box_2d) == 4 and any(v != 0.0 for v in self.box_2d)


def _parse_kv_lines(text: str) -> list[_ParsedField]:
    """Fallback parser for plain `Field: Value` text."""
    if not text:
        return []

    fields: list[_ParsedField] = []
    seen = set()
    for raw_line in text.splitlines():
        line = raw_line.strip().lstrip("-•").strip()
        if not line or ":" not in line:
            continue

        left, right = line.split(":", 1)
        label = left.strip()
        value = right.strip()
        if not label:
            continue
        # Filter low-signal values
        if value.lower() in {"n/a", "none", "not found"}:
            continue

        key = label.lower()
        if key in seen:
            continue
        seen.add(key)

        try:
            fields.append(_ParsedField(label=label, value=value))
        except ValidationError:
            continue

    return fields


def _parse_engine_result(result: Any) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Convert engine output to UI-safe (output_data, bounding_boxes).

    - Trust `result.objects` when present, but validate/normalize it.
    - Fallback to parsing `result.raw_text` as `Field: Value` lines.
    """
    raw_text = getattr(result, "raw_text", "") or ""
    objs = getattr(result, "objects", None)

    parsed: list[_ParsedField] = []

    if objs:
        for obj in objs:
            # Accept both dataclass objects and dicts.
            payload = {
                "label": getattr(obj, "label", None) if not isinstance(obj, dict) else obj.get("label"),
                "value": getattr(obj, "value", None) if not isinstance(obj, dict) else obj.get("value", ""),
                "box_2d": getattr(obj, "box_2d", None) if not isinstance(obj, dict) else obj.get("box_2d", []),
            }
            try:
                pf = _ParsedField.model_validate(payload)
            except ValidationError:
                continue
            parsed.append(pf)

    if not parsed:
        parsed = _parse_kv_lines(raw_text)

    # Build sidebar table
    output_data = [{"field": f.label, "value": (f.value or "Detected")} for f in parsed]

    # Build SVG boxes only for real boxes
    bounding_boxes: list[dict[str, str]] = []
    if objs:
        for obj in objs:
            try:
                box = getattr(obj, "box_2d", None) if not isinstance(obj, dict) else obj.get("box_2d")
                if isinstance(box, list) and len(box) == 4 and any(v != 0.0 for v in box):
                    bounding_boxes.append(_obj_to_svg_dict(obj))
            except Exception:
                continue

    return output_data, bounding_boxes


# ═══════════════════════════════════════════════════════════════════════════
# APPLICATION STATE
# ═══════════════════════════════════════════════════════════════════════════
class AppState(rx.State):
    """Reactive state for The Local Titan dashboard.

    State Variables:
        is_processing:    True while the model is running inference.
        is_engine_ready:  True once the VLMPipeline has been loaded.
        output_data:      List of extracted key-value pairs for the data table.
        bounding_boxes:   List of grounded regions for SVG overlay rendering.
        raw_model_output: Full raw text response from the model.
        uploaded_image:   Base64 data URI of the currently displayed image.
        image_filename:   Original filename of the uploaded document.
        error_message:    User-facing error string (empty = no error).
        processing_status: Progress message shown during inference.
        page_count:       Total pages in the loaded PDF.
        current_page:     1-indexed page currently being viewed/processed.
        redacted_image:   Base64 data URI of the auto-redacted version.
        show_redacted:    Toggle between original and redacted view.
    """

    # --- Processing State ---
    is_processing: bool = False
    is_engine_ready: bool = False
    processing_status: str = ""
    error_message: str = ""
    toast_message: str = ""
    toast_type: str = "info"  # success, warning, error, info
    document_type: str = "Unknown"

    # --- Hardware Device ---
    active_device: str = ""            # e.g. "Intel CPU", "Intel Iris Xe GPU"
    active_device_icon: str = "cpu"    # Lucide icon name
    active_device_color: str = "green" # Badge color scheme

    # --- Memory Hygiene ---
    last_activity_time: float = 0.0
    is_standby_mode: bool = False

    # --- Batch Processing ---
    batch_total: int = 0               # Total files in batch
    batch_current: int = 0             # Currently processing file index
    batch_filenames: list[str] = []    # Filenames in current batch

    # --- Document Data ---
    uploaded_image: str = ""
    image_filename: str = ""
    page_count: int = 0
    current_page: int = 1
    document_type: str = "Unknown"
    document_confidence: float = 0.0
    document_alternatives: list[dict[str, Any]] = []

    # --- Metrics ---
    inference_latency_ms: int = 0
    ram_usage_gb: float = 0.0
    ram_total_gb: float = 0.0


    # --- Model Output ---
    raw_model_output: str = ""
    output_data: list[dict[str, str]] = []
    bounding_boxes: list[dict[str, str]] = []

    # --- Redaction ---
    redacted_image: str = ""
    show_redacted: bool = False

    show_toast: bool = False
    success_message: str = ""

    # --- Benchmarking & ETA ---
    calibrated_ms_per_page: float = 2500.0  # Default baseline for i5
    time_remaining_str: str = ""           # e.g. "ETA: 45s"

    # --- System Health & Concurrency ---
    last_heartbeat: str = ""           # Timestamp for heartbeat UI
    is_queue_paused: bool = False      # True if RAM > 85%
    
    # --- Privacy & Security ---
    master_password: str = ""          # Optional password for local encryption
    is_encrypted: bool = False         # UI flag indicating encryption active

    # --- RAG & Global Search ---
    search_query: str = ""
    search_results: list[dict[str, str]] = []
    chat_input: str = ""
    chat_history: list[dict[str, str]] = []
    is_searching: bool = False
    is_chatting: bool = False
    is_chat_open: bool = False

    def toggle_chat(self):
        self.is_chat_open = not self.is_chat_open

    def set_search_query(self, query: str):
        """Explicit setter for search query."""
        self.search_query = query

    def set_chat_input(self, text: str):
        """Explicit setter for chat input."""
        self.chat_input = text


    # --- Spatial Grounding ---
    last_click_box: list[int] = []    # [y1, x1, y2, x2] normalized to 1000


    # --- Internal (not sent to frontend) ---
    # These are now managed via the 'backend' singleton to avoid 
    # StateSerializationErrors with non-picklable OpenVINO objects.

    # ===================================================================
    # ENGINE LIFECYCLE
    # ===================================================================
    @rx.event
    async def initialize_engine(self):
        """Load the VLMPipeline on app startup (runs once or on wake)."""
        print("\n[EVENT] initialize_engine triggered")
        import time
        import psutil
        self.last_activity_time = time.time()
        self.is_standby_mode = False
        self.last_heartbeat = datetime.now().strftime("%H:%M:%S")
        
        # Initialize the processing queue if needed
        from backend.processor import ProcessingQueue
        if backend.queue is None:
            backend.queue = ProcessingQueue(maxsize=5) # Buffer 5 pages max to save RAM

        # Get total RAM once
        self.ram_total_gb = round(psutil.virtual_memory().total / (1024**3), 1)


        if backend.engine is not None and getattr(backend.engine, "_pipe", None) is not None:
            self.is_engine_ready = True
            return

        self.processing_status = "Loading AI model..."
        self.is_processing = True
        yield  # Push status to UI immediately

        try:
            from backend.engine import InferenceEngine

            # Run the heavy model load in a background thread so the
            # Reflex UI event loop stays responsive during the ~10-30s load.
            if backend.engine is None:
                print("[EVENT] Starting InferenceEngine load (background thread)...")
                backend.engine = await asyncio.to_thread(InferenceEngine)
                print("[EVENT] InferenceEngine load complete.")
            else:
                # Wake from standby (re-initialize pipeline)
                await asyncio.to_thread(backend.engine._load_pipeline)

            self.is_engine_ready = True
            self.processing_status = "Model ready."
            self.error_message = ""

            # Expose hardware info to the UI
            self.active_device = backend.engine.device_name
            self.active_device_icon = backend.engine.device_icon
            self.active_device_color = backend.engine.device_color
            log.info(
                f"Engine initialized on {backend.engine.device_name} "
                f"({backend.engine.device_id})"
            )

        except Exception as e:
            self.error_message = f"Failed to load model: {str(e)}"
            self.is_engine_ready = False
            log.error(f"Engine init failed: {e}", exc_info=True)
        finally:
            self.is_processing = False
            
    async def set_master_password(self, password: str):
        """Update the master password for local encryption."""
        self.master_password = password
        self.is_encrypted = len(password) > 0
        
        # Synchronize with backend security layer
        # Derive key and set in DB (blocking crypto ops → run in thread)
        import asyncio
        from backend.db import set_master_password as set_db_pw
        await asyncio.to_thread(set_db_pw, password)
        
        if self.is_encrypted:
            self.success_message = "Encryption layer active: All results and RAG history will be encrypted at rest."
            self.show_toast = True
        else:
            self.success_message = "Encryption layer disabled: Local storage is now in plain text."
            self.show_toast = True
            
    async def _resource_monitor(self):
        """Monitor idle time and system resources (RAM)."""
        import time
        import psutil
        import gc
        while True:
            await asyncio.sleep(5)  # Increased frequency for health monitor
            
            async with self:
                # 1. Update RAM usage
                mem = psutil.virtual_memory()
                self.ram_usage_gb = round(mem.used / (1024**3), 1)
                self.last_heartbeat = datetime.now().strftime("%H:%M:%S")
                
                # 2. Batch Health Monitor (RAM Safety)
                ram_percent = mem.percent
                if ram_percent > 85:
                    if self._queue and not self._queue.is_paused:
                        log.warning(f"CRITICAL RAM: {ram_percent}% — Pausing processing queue.")
                        self._queue.pause()
                        self.is_queue_paused = True
                        self.trigger_toast("System memory low — pausing batch...", "warning")
                        gc.collect() # Force immediate cleanup
                elif ram_percent < 70:
                    if self._queue and self._queue.is_paused:
                        log.info(f"RAM Recovered: {ram_percent}% — Resuming queue.")
                        self._queue.resume()
                        self.is_queue_paused = False
                
                # 3. Idle Standby Logic
                if self.is_standby_mode or not self.is_engine_ready or backend.engine is None:
                    continue
                
                idle_time = time.time() - self.last_activity_time
                if idle_time > 600:  # 10 minutes
                    self.is_standby_mode = True
                    self.is_engine_ready = False
                    self.processing_status = "Model on Standby (Click to wake)"
                    log.info("Idle timeout reached. Unloading model to save memory.")
                    await asyncio.to_thread(backend.engine.unload_pipeline)

    @rx.event
    async def trigger_toast(self, message: str, type: str = "info"):
        """Trigger a toast notification."""
        self.toast_message = message
        self.toast_type = type
        self.show_toast = True
        yield
        await asyncio.sleep(4)
        self.show_toast = False


    # ===================================================================
    # FILE UPLOAD
    # ===================================================================
    @rx.event
    async def handle_upload(self, files: list[rx.UploadFile]):
        """Handle file upload from the UI dropzone (Image/PDF/PPTX/DOCX)."""
        if not files:
            return

        file = files[0]
        self.image_filename = file.filename or "document"
        self.error_message = ""
        self.output_data = []
        self.bounding_boxes = []
        self.raw_model_output = ""
        self.redacted_image = ""
        self.show_redacted = False
        self.processing_status = "Scanning document..."
        yield

        try:
            upload_data = await file.read()
            suffix = Path(self.image_filename).suffix.lower()

            # Step 1: Instant Modal Preview (Base64)
            # Use the new parser for fast preview generation
            from backend.parser import RouteToCanvas, get_preview_base64
            
            try:
                # Instant check if it's a multimodal document
                RouteToCanvas(self.image_filename)
                # If it didn't raise ValueError, try to get a base64 thumbnail
                # Write to temp for parser access
                tmp_dir = get_temp_dir()
                tmp_path = tmp_dir / f"preview_{id(upload_data)}{suffix}"
                tmp_path.write_bytes(upload_data)
                
                preview_b64 = await asyncio.to_thread(get_preview_base64, tmp_path)
                if preview_b64:
                    self.uploaded_image = preview_b64
                    self.processing_status = "Loading full document..."
                    yield
                
                # Step 2: Route to specific loaders
                if suffix == ".pdf":
                    await self._load_pdf(upload_data)
                else:
                    await self._load_multimodal_document(upload_data, suffix)
                
                tmp_path.unlink(missing_ok=True)
                
            except ValueError:
                # Not a multimodal doc, must be a standard image
                if suffix in (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"):
                    await self._load_image(upload_data)
                else:
                    self.error_message = f"Unsupported format: {suffix}"
                    return

        except Exception as e:
            self.error_message = f"Upload failed: {str(e)}"
            log.error(f"Upload error: {e}", exc_info=True)
            self.processing_status = ""


    async def _load_image(self, data: bytes):
        """Load an image file into state."""
        import time
        self.last_activity_time = time.time()
        from backend.processor import dynamic_resize

        img = Image.open(io.BytesIO(data)).convert("RGB")
        img = dynamic_resize(img)

        backend.current_pil_image = img
        
        # Save frontend UI memory by sending only a low-res JPEG thumbnail
        thumb = img.copy()
        thumb.thumbnail((800, 1000), Image.Resampling.LANCZOS)
        self.uploaded_image = _pil_to_data_uri(thumb, fmt="JPEG")
        
        self.page_count = 1
        self.current_page = 1
        
        if backend.engine:
            await asyncio.to_thread(backend.engine.clear_cache)

    async def _load_pdf(self, data: bytes):
        """Unified loader for PDF using the enhanced parser."""
        import time
        self.last_activity_time = time.time()
        
        # We reuse the multimodal logic for consistency
        await self._load_multimodal_document(data, ".pdf")

    async def _load_multimodal_document(self, data: bytes, suffix: str):
        """Convert PDF/PPTX/DOCX into an image sequence for state."""
        from backend.parser import convert_to_image_sequence
        from backend.processor import PageImage
        
        tmp_dir = get_temp_dir()
        tmp_path = tmp_dir / f"multimodal_{id(data)}{suffix}"
        tmp_path.write_bytes(data)
        
        try:
            # Render pages to BytesIO buffers
            images_buffers = await asyncio.to_thread(convert_to_image_sequence, tmp_path)
            
            # Convert buffers to PIL images for batch processing
            pil_images = [Image.open(buf).convert("RGB") for buf in images_buffers]
            
            # OPTIMIZATION: We could process the whole batch immediately here 
            # if the user requested extraction, but for now we follow 
            # the PageImage pattern for UI display.
            
            pages_list = []
            for idx, pil_img in enumerate(pil_images, start=1):
                pages_list.append(
                    PageImage(
                        image=pil_img,
                        page_number=idx,
                        original_size=pil_img.size,
                        source_file=self.image_filename
                    )
                )

            
            self.page_count = len(pages_list)
            if pages_list:
                first = pages_list[0]
                backend.current_pil_image = first.image
                
                # Update preview with high-res or just keep the thumbnail proxy
                # We'll regenerate a fresh low-res JPEG for consistent UI scaling
                thumb = first.image.copy()
                thumb.thumbnail((800, 1000), Image.Resampling.LANCZOS)
                self.uploaded_image = _pil_to_data_uri(thumb, fmt="JPEG")
                
                self.current_page = 1
                backend.pdf_pages = pages_list
                
                if backend.engine:
                    await asyncio.to_thread(backend.engine.clear_cache)
            else:
                self.error_message = f"Failed to extract content from {suffix}."
        finally:
            tmp_path.unlink(missing_ok=True)


    # ===================================================================
    # PAGE NAVIGATION
    # ===================================================================
    @rx.event
    async def next_page(self):
        """Navigate to the next PDF page."""
        import time
        self.last_activity_time = time.time()
        if backend.pdf_pages and self.current_page < self.page_count:
            self.current_page += 1
            page = backend.pdf_pages[self.current_page - 1]
            backend.current_pil_image = page.image
            
            thumb = page.image.copy()
            thumb.thumbnail((800, 1000), Image.Resampling.LANCZOS)
            self.uploaded_image = _pil_to_data_uri(thumb, fmt="JPEG")
            
            # Clear previous results for the new page
            self.output_data = []
            self.bounding_boxes = []
            self.raw_model_output = ""
            self.redacted_image = ""
            self.show_redacted = False
            
            if self._engine:
                await asyncio.to_thread(self._engine.clear_cache)

    @rx.event
    async def prev_page(self):
        """Navigate to the previous PDF page."""
        import time
        self.last_activity_time = time.time()
        if backend.pdf_pages and self.current_page > 1:
            self.current_page -= 1
            page = backend.pdf_pages[self.current_page - 1]
            backend.current_pil_image = page.image
            
            thumb = page.image.copy()
            thumb.thumbnail((800, 1000), Image.Resampling.LANCZOS)
            self.uploaded_image = _pil_to_data_uri(thumb, fmt="JPEG")
            
            self.output_data = []
            self.bounding_boxes = []
            self.raw_model_output = ""
            self.redacted_image = ""
            self.show_redacted = False
            
            if self._engine:
                await asyncio.to_thread(self._engine.clear_cache)

    # ===================================================================
    # SPATIAL GROUNDING Q&A
    # ===================================================================
    @rx.event
    async def handle_image_click(self, x: int, y: int, width: int, height: int):
        """Capture click coordinates on document and prepare spatial query.
        
        Maps UI (px) coordinates to normalized [0, 1000] range.
        """
        if not self.is_engine_ready:
            return

        # 1. Normalize coordinates
        # Qwen-VL expects [y1, x1, y2, x2] normalized to 1000
        norm_x = int((x / width) * 1000)
        norm_y = int((y / height) * 1000)
        
        # 2. Create a small focus box (1% of image size)
        box_size = 10
        y1 = max(0, norm_y - box_size)
        x1 = max(0, norm_x - box_size)
        y2 = min(1000, norm_y + box_size)
        x2 = min(1000, norm_x + box_size)
        
        self.last_click_box = [y1, x1, y2, x2]
        self.processing_status = f"📍 Click detected at {x}, {y}. Ready for query."
        
        # Trigger query (for UX, we can show a prompt dialog here)
        # For now, we'll auto-trigger with a generic "What is this?" 
        # but the UI should have a text input.
        yield

    @rx.event
    async def run_spatial_extraction(self, prompt: str = "What is this?"):
        """Run a coordinate-based Q&A query."""
        if not self.last_click_box:
            self.error_message = "Click the document first to select a region."
            return

        if backend.engine.is_locked:
            self.error_message = "Model is still thinking..."
            return

        self.is_processing = True
        self.processing_status = f"🔍 Analyzing region {self.last_click_box}..."
        self.error_message = ""
        yield

        try:
            start_time = time.perf_counter()
            result = await asyncio.to_thread(
                backend.engine.process_spatial_query,
                backend.current_pil_image,
                tuple(self.last_click_box),
                prompt
            )
            self.inference_latency_ms = int((time.perf_counter() - start_time) * 1000)


            if result.error:
                self.error_message = result.error
            else:
                self.raw_model_output = result.raw_text
                # If the model returns boxes for the answer, show them
                self.bounding_boxes = [_obj_to_svg_dict(obj) for obj in result.objects]
                
                # Update output data for the sidebar
                self.output_data = [
                    {"field": "🗺️ Location Answer", "value": result.raw_text}
                ]
                self.processing_status = "Analysis complete."

        except Exception as e:
            self.error_message = f"Spatial query failed: {e}"
            log.error(f"Spatial error: {e}", exc_info=True)
        finally:
            self.is_processing = False

    # ===================================================================
    # INFERENCE — EXTRACT

    # ===================================================================
    @rx.event
    async def run_extraction(self):
        """Run document field extraction with grounding on the current page.

        Guarded by the engine's inference lock — if the model is already
        generating, the request is rejected with a user-friendly message
        rather than queued (which would freeze perceived responsiveness).
        """
        if not self.is_engine_ready or backend.engine is None:
            self.error_message = "Model not loaded. Please wait for initialization."
            return

        if backend.current_pil_image is None:
            self.error_message = "No document loaded. Upload a file first."
            return

        # Lock guard: prevent double-click while model is thinking
        if backend.engine.is_locked:
            self.error_message = "Model is still thinking... please wait."
            return

        self.is_processing = True
        self.processing_status = "🧠 Model Thinking..."
        self.error_message = ""
        yield  # Push "Model Thinking..." + progress bar to UI immediately

        result: EngineResult | None = None
        try:
            # Single-pass: classify + extract in one inference call
            # (previously 3 separate passes; now ~3x faster)
            self.processing_status = "⚡ Classifying & Extracting (single pass)..."
            yield
            start_time = time.perf_counter()
            result = await asyncio.to_thread(
                backend.engine.classify_and_extract,
                backend.current_pil_image,
            )
            self.inference_latency_ms = int((time.perf_counter() - start_time) * 1000)

            # Populate classification state from the combined result
            self.document_type = result.document_type
            self.document_confidence = result.confidence
            self.document_alternatives = []

            if result.error:
                self.error_message = result.error
            else:
                # PII gate: never write sensitive text into Reflex state
                if getattr(result, "has_pii", False):
                    summary = getattr(result, "pii_summary", {}) or {}
                    self.error_message = (
                        "Sensitive data detected. Please use PII Redaction before viewing results."
                    )
                    self.output_data = [
                        {"field": "PII Detected", "value": json.dumps(summary) if summary else "True"}
                    ]
                    self.bounding_boxes = []
                    self.raw_model_output = ""
                    self.processing_status = "🔒 PII detected — redaction required."
                    return

                self.raw_model_output = result.raw_text

                output_data, bounding_boxes = _parse_engine_result(result)
                self.output_data = output_data
                self.bounding_boxes = bounding_boxes

                self.processing_status = f"Done. Found {len(self.output_data)} fields in {self.document_type}."

        except Exception as e:
            self.error_message = f"Extraction failed: {str(e)}"
            log.error(f"Extraction error: {e}", exc_info=True)

        finally:
            self.is_processing = False

        # --- Index for RAG ---
        if result and not result.error and result.raw_text and not getattr(result, "has_pii", False):
            try:
                # Extract embedding
                embedding = await asyncio.to_thread(
                    backend.engine.get_embeddings,
                    image=backend.current_pil_image,
                    text=result.raw_text
                )
                
                # Index in ChromaDB
                doc_id = f"{self.image_filename}_{int(time.time())}"
                await asyncio.to_thread(
                    db.index_document,
                    content=result.raw_text,
                    metadata={
                        "filename": self.image_filename,
                        "timestamp": str(datetime.now()),
                        "type": "extraction"
                    },
                    document_id=doc_id,
                    embedding=embedding
                )
            except Exception as e:
                log.error(f"RAG Indexing failed: {e}")


    # ===================================================================
    # INFERENCE — PII DETECT + AUTO-REDACT
    # ===================================================================
    @rx.event
    async def run_pii_redaction(self):
        """Detect PII and produce a redacted version of the current page."""
        if not self.is_engine_ready or backend.engine is None:
            self.error_message = "Model not loaded."
            return

        if backend.current_pil_image is None:
            self.error_message = "No document loaded."
            return

        # Lock guard
        if backend.engine.is_locked:
            self.error_message = "Model is still thinking... please wait."
            return

        self.is_processing = True
        self.processing_status = "🧠 Model Thinking — scanning for PII..."
        self.error_message = ""
        yield

        result = None
        try:
            from backend.processor import auto_redact_from_engine_result

            # Step 1: Detect PII (offloaded to worker thread)
            start_time = time.perf_counter()
            result = await asyncio.to_thread(
                backend.engine.detect_pii,
                backend.current_pil_image,
            )
            self.inference_latency_ms = int((time.perf_counter() - start_time) * 1000)


            if result.error:
                self.error_message = result.error
                return

            # Update bounding boxes (PII regions)
            self.bounding_boxes = [_obj_to_svg_dict(obj) for obj in result.objects]
            self.output_data = [
                {"field": f"🔒 {obj.label}", "value": "REDACTED"}
                for obj in result.objects
            ]

            # Step 2: Auto-redact (lightweight PIL op, also threaded)
            if result.objects:
                redacted_img = await asyncio.to_thread(
                    auto_redact_from_engine_result,
                    backend.current_pil_image,
                    result,
                )
                self.redacted_image = _pil_to_data_uri(redacted_img)
                self.show_redacted = True
                self.processing_status = (
                    f"Redacted {len(result.objects)} PII region(s)."
                )
            else:
                self.processing_status = "No PII detected."

        except Exception as e:
            self.error_message = f"PII scan failed: {str(e)}"
            log.error(f"PII redaction error: {e}", exc_info=True)

        finally:
            self.is_processing = False
        self.is_processing = False

    # ===================================================================
    # UI TOGGLES
    # ===================================================================
    def on_load(self):
        """Initial terminal log and background checks."""
        log.info("The Local Titan: Dashboard Loaded")
        
        # Load benchmark calibration if available
        import json
        import os
        if os.path.exists("benchmark_results.json"):
            try:
                with open("benchmark_results.json", "r") as f:
                    data = json.load(f)
                    self.calibrated_ms_per_page = data.get("avg_ms_per_page", 2500.0)
                    log.info(f"Loaded hardware calibration: {self.calibrated_ms_per_page:.0f}ms/page")
            except Exception as e:
                log.warning(f"Failed to load benchmark: {e}")

        return self.check_hardware_status()
    @rx.event
    def toggle_redacted_view(self):
        """Switch between original and redacted image view."""
        if self.redacted_image:
            self.show_redacted = not self.show_redacted

    @rx.event
    async def clear_document(self):
        """Reset all state for a fresh document."""
        import time
        self.last_activity_time = time.time()
        self.uploaded_image = ""
        self.image_filename = ""
        self.output_data = []
        self.bounding_boxes = []
        self.raw_model_output = ""
        self.redacted_image = ""
        self.show_redacted = False
        self.error_message = ""
        self.processing_status = ""
        self.page_count = 0
        self.current_page = 1
        backend.current_pil_image = None
        backend.pdf_pages = []
        self.batch_total = 0
        self.batch_current = 0
        self.batch_filenames = []
        
        if backend.engine:
            await asyncio.to_thread(backend.engine.clear_cache)

    # ===================================================================
    # DATA EXPORT
    # ===================================================================
    @rx.event
    async def download_results(self):
        """Export current extraction results to Excel and trigger download."""
        if not self.output_data:
            self.error_message = "No data to export. Run extraction first."
            return

        self.processing_status = "Generating Excel export..."
        yield

        try:
            from backend.processor import DataExporter

            exporter = DataExporter(
                output_data=self.output_data,
                source_filename=self.image_filename or "document",
                bounding_boxes=self.bounding_boxes,
            )

            excel_path = await asyncio.to_thread(exporter.to_excel)

            # rx.download(url=...) requires a URL starting with '/', not a
            # Windows filesystem path.  Read the file bytes and send as data.
            excel_bytes = excel_path.read_bytes()
            self.processing_status = ""
            yield rx.download(data=excel_bytes, filename=excel_path.name)

        except Exception as e:
            self.error_message = f"Export failed: {str(e)}"
            self.processing_status = ""
            log.error(f"Export error: {e}", exc_info=True)


    # ===================================================================
    # BATCH PROCESSING
    # ===================================================================
    @rx.event
    async def run_batch_extraction(self, files: list[rx.UploadFile]):
        """Process a batch of uploaded files with robust concurrency control.
        
        Uses a producer-consumer model:
          - Producer: Background task extracts pages from PDFs (E-cores).
          - Consumer: Main loop pulls from the queue and runs AI (P-cores).
        """
        if not self.is_engine_ready or backend.engine is None:
            self.error_message = "Engine not ready. Please wait."
            return

        total = len(files)
        self.batch_total = total
        self.batch_current = 0
        self.is_processing = True
        self.error_message = ""
        yield

        from backend.processor import INFERENCE_SEMAPHORE, extract_pages_from_pdf, PageImage
        from PIL import Image as PILImage
        
        # Helper to push pages to queue in background
        async def producer(upload_files):
            loop = asyncio.get_event_loop()
            for f in upload_files:
                temp_path = get_temp_dir() / f.filename
                with open(temp_path, "wb") as tmp:
                    tmp.write(await f.read())
                
                if f.filename.lower().endswith(".pdf"):
                    def run_extraction():
                        # Process PDF pages one by one (on E-cores)
                        for p in extract_pages_from_pdf(temp_path):
                            # Put into the asyncio queue (blocks worker thread if queue full)
                            asyncio.run_coroutine_threadsafe(backend.queue.put(p), loop).result()
                    
                    await asyncio.to_thread(run_extraction)
                else:
                    # Single image
                    img = PILImage.open(temp_path).convert("RGB")
                    await backend.queue.put(PageImage(image=img, page_number=1, original_size=img.size, source_file=str(temp_path)))

        # Start producer in background
        p_task = asyncio.create_task(producer(files))

        processed_pages = 0
        while not p_task.done() or backend.queue.qsize > 0:
            # Check for completion
            if backend.queue.qsize == 0 and p_task.done():
                break
                
            # Pop next page from queue (blocks if queue is empty or paused by monitor)
            try:
                page = await asyncio.wait_for(backend.queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            
            self.processing_status = f"🧠 AI Processing: {Path(page.source_file).name} (Page {page.page_number})"
            self.last_heartbeat = datetime.now().strftime("%H:%M:%S")
            
            # Calculate ETA
            remaining_pages = (total - processed_pages)
            if remaining_pages > 0:
                eta_seconds = (remaining_pages * self.calibrated_ms_per_page) / 1000
                if eta_seconds > 60:
                    self.time_remaining_str = f"ETA: {int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
                else:
                    self.time_remaining_str = f"ETA: {int(eta_seconds)}s"
            else:
                self.time_remaining_str = ""
                
            yield

            try:
                # 1. Acquire SEMAPHORE — ensure exactly one OpenVINO inference runs on P-cores
                async with INFERENCE_SEMAPHORE:
                    result = await asyncio.to_thread(
                        backend.engine.process_document,
                        page.image,
                        "Extract all fields and key-value pairs."
                    )
                
                # 2. Save results
                from backend.processor import save_batch_result
                page_data = [
                    {"field": obj["label"], "value": str(obj["box_2d"])}
                    for obj in (result.objects if hasattr(result, 'objects') else [])
                ]
                if not page_data and result.raw_text:
                    page_data = [{"field": "raw_text", "value": result.raw_text}]

                await asyncio.to_thread(
                    save_batch_result,
                    Path(page.source_file).name, page.page_number, page_data, result.raw_text or ""
                )

                # Update UI
                processed_pages += 1
                self.batch_current = min(processed_pages, total) # Close enough for progress bar
                
                # 3. Memory Cleanup during batch
                import gc
                gc.collect() 
                
            except Exception as e:
                log.error(f"Batch inference failed: {e}")
            
            backend.queue.task_done()
            yield

        self.is_processing = False
        self.processing_status = f"✓ Batch complete: {total} file(s) processed."
        self.time_remaining_str = ""
        yield



    # ===================================================================
    # RAG EVENTS
    # ===================================================================
    
    @rx.event
    async def perform_semantic_search(self):
        """Search the local knowledge base for relevant documents."""
        if not self.search_query.strip():
            self.search_results = []
            return
            
        self.is_searching = True
        yield
        
        try:
            results = await asyncio.to_thread(db.semantic_search, self.search_query)
            if results:
                # Format for UI
                formatted = []
                # ChromaDB returns nested lists for documents/metadatas
                for i in range(len(results['documents'][0])):
                    formatted.append({
                        "text": results['documents'][0][i][:200] + "...",
                        "filename": results['metadatas'][0][i].get('filename', 'Unknown'),
                        "score": str(results['distances'][0][i]) if 'distances' in results else "0"
                    })
                self.search_results = formatted
        except Exception as e:
            log.error(f"Search event failed: {e}")
        finally:
            self.is_searching = False

    async def perform_semantic_search_on_enter(self, key: str):
        """Trigger search only on Enter key."""
        if key == "Enter":
            return self.perform_semantic_search

    @rx.event
    async def chat_with_knowledge_base(self):
        """Context-aware chat using retrieved knowledge."""
        if not self.chat_input.strip() or not backend.engine:
            return
            
        user_msg = self.chat_input
        self.chat_history = self.chat_history + [{"role": "user", "content": user_msg}]
        self.chat_input = ""
        self.is_chatting = True
        yield
        
        try:
            # 1. Retrieve Context
            search_res = await asyncio.to_thread(db.semantic_search, user_msg, n_results=5)
            context_blocks = []
            if search_res and search_res['documents']:
                for doc in search_res['documents'][0]:
                    context_blocks.append(doc)
            
            context_str = "\n\n---\n\n".join(context_blocks)
            
            # 2. Build RAG Prompt
            rag_prompt = (
                "You are the 'Local Titan' Knowledge Assistant. "
                "Answer the user's question using ONLY the provided context snippets from past documents. "
                "If the answer isn't in the context, say you don't know.\n\n"
                f"RELEVANT KNOWLEDGE:\n{context_str}\n\n"
                f"QUESTION: {user_msg}\n\n"
                "ANSWER:"
            )
            
            # 3. Generate Answer
            # Use current image if available for multimodal context, else text-only
            result = await asyncio.to_thread(
                backend.engine.process_document,
                image=backend.current_pil_image if backend.current_pil_image else Image.new('RGB', (1,1)),
                prompt=rag_prompt,
                max_new_tokens=512,  # Limit response length
            )
            
            if result.error:
                self.chat_history = self.chat_history + [{"role": "assistant", "content": f"Error: {result.error}"}]
            else:
                self.chat_history = self.chat_history + [{"role": "assistant", "content": result.raw_text.strip()}]
                
        except Exception as e:
            log.error(f"Chat failed: {e}")
            self.chat_history = self.chat_history + [{"role": "assistant", "content": "I encountered an error while searching the knowledge base."}]
        finally:
            self.is_chatting = False

    async def chat_on_enter(self, key: str):
        """Trigger chat only on Enter key."""
        if key == "Enter":
            await self.chat_with_knowledge_base()
            return  # Explicit return None


    # ===================================================================

    # COMPUTED PROPERTIES
    # ===================================================================
    @rx.var
    def has_document(self) -> bool:
        return bool(self.uploaded_image)

    @rx.var
    def has_results(self) -> bool:
        return len(self.output_data) > 0

    @rx.var
    def display_image(self) -> str:
        """Return the appropriate image (original or redacted)."""
        if self.show_redacted and self.redacted_image:
            return self.redacted_image
        return self.uploaded_image

    @rx.var
    def page_indicator(self) -> str:
        if self.page_count <= 1:
            return ""
        return f"Page {self.current_page} of {self.page_count}"

    @rx.var
    def thinking_progress(self) -> int:
        if self.is_processing:
            return -1
        return 0

    @rx.var
    def batch_progress(self) -> float:
        """Batch progress as 0.0–1.0 fraction for the progress ring."""
        if self.batch_total <= 0:
            return 0.0
        return self.batch_current / self.batch_total

    @rx.var
    def batch_progress_label(self) -> str:
        """Human-readable batch progress label."""
        if self.batch_total <= 0:
            return ""
        return f"{self.batch_current}/{self.batch_total}"

    @rx.var
    def is_batch_active(self) -> bool:
        return self.batch_total > 0 and self.batch_current < self.batch_total


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def _pil_to_data_uri(image: Image.Image, fmt: str = "PNG") -> str:
    """Convert a PIL Image to a base64 data URI for embedding in HTML."""
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def _format_box(box: list[float]) -> str:
    """Format a bounding box as a readable string for the data table."""
    if len(box) == 4:
        return f"({box[0]:.3f}, {box[1]:.3f}, {box[2]:.3f}, {box[3]:.3f})"
    return str(box)
