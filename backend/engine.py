"""
backend/engine.py — The Local Titan: VLMPipeline Inference Engine
=================================================================

Provides the bridge between the Qwen 3.5-VL-4B INT4 model (on disk)
and the rest of the application. All inference goes through this module.

Key responsibilities:
  1. Initialize the OpenVINO VLMPipeline from the local INT4 IR folder.
  2. Accept a PIL.Image + prompt, return the model's text response.
  3. Parse Qwen's grounding output into structured JSON bounding boxes.

Qwen 3.5-VL Grounding Format (output by the model as text tokens):
  <|object_ref_start|>Label<|object_ref_end|><|box_start|>(y1,x1),(y2,x2)<|box_end|>

  Coordinates are normalized to [0, 1000] by default.
  We convert them to [0.0, 1.0] ratios for frontend SVG overlay mapping.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log = logging.getLogger("engine")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL_DIR = "./qwen_ov_int4"

# Device priority: NPU (zero-energy AI accelerator) > GPU (Iris Xe) > CPU
DEVICE_PRIORITY = ["NPU", "GPU", "CPU"]

# Human-readable labels + icons for the UI hardware badge
DEVICE_INFO = {
    "NPU":  {"label": "Intel NPU",        "icon": "zap",       "color": "amber"},
    "GPU":  {"label": "Intel Iris Xe GPU", "icon": "monitor",   "color": "cyan"},
    "CPU":  {"label": "Intel CPU",         "icon": "cpu",       "color": "green"},
}


# ---------------------------------------------------------------------------
# Dynamic Device Discovery
# ---------------------------------------------------------------------------
def discover_best_device() -> str:
    """Query OpenVINO for available hardware and select the best device."""
    try:
        import openvino as ov
        core = ov.Core()
        available = core.available_devices
        log.info(f"OpenVINO available devices: {available}")

        for preferred in DEVICE_PRIORITY:
            if preferred in available:
                log.info(f"Selected device: {preferred} (priority match)")
                return preferred
            for dev in available:
                if dev.startswith(preferred):
                    log.info(f"Selected device: {dev} (prefix match for {preferred})")
                    return dev

        log.warning("No preferred device found — falling back to CPU.")
        return "CPU"

    except ImportError:
        log.warning("openvino not installed — defaulting to CPU.")
        return "CPU"
    except Exception as e:
        log.warning(f"Device discovery failed ({e}) — defaulting to CPU.")
        return "CPU"


# ---------------------------------------------------------------------------
# Frozen Executable Path Resolution
# ---------------------------------------------------------------------------
def get_resource_path(relative_path: str) -> Path:
    """Resolve a path for both development and PyInstaller-frozen contexts."""
    if getattr(sys, 'frozen', False):
        base_path = Path(sys._MEIPASS)  # type: ignore[attr-defined]
    else:
        base_path = Path.cwd()

    clean = relative_path.lstrip("./").lstrip("\\.")
    resolved = base_path / clean
    log.debug(f"Resource path: '{relative_path}' → '{resolved}'")
    return resolved

# Qwen 3.5-VL grounding regex
_GROUNDING_PATTERN = re.compile(
    r"<\|object_ref_start\|>"
    r"(?P<label>[^<]+)"
    r"<\|object_ref_end\|>"
    r"<\|box_start\|>"
    r"\((?P<y1>\d+),(?P<x1>\d+)\)"
    r",\s*"
    r"\((?P<y2>\d+),(?P<x2>\d+)\)"
    r"<\|box_end\|>",
    re.DOTALL,
)

# Coordinate normalization base
_COORD_SCALE = 1000.0


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------
@dataclass
class GroundedObject:
    label: str
    value: str = ""
    box_2d: list[float] = field(default_factory=list)  # [ymin, xmin, ymax, xmax]

    def to_dict(self) -> dict:
        return asdict(self)

@dataclass
class EngineResult:
    raw_text: str = ""
    objects: list[GroundedObject] = field(default_factory=list)
    document_type: str = "Unknown"
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "raw_text": self.raw_text,
            "objects": [obj.to_dict() for obj in self.objects],
            "error": self.error,
        }

OV_CONFIG = {
    "CPU_BIND_THREAD": "YES",
    "CPU_THROUGHPUT_STREAMS": "1",
}


# ═══════════════════════════════════════════════════════════════════════════
# INFERENCE ENGINE
# ═══════════════════════════════════════════════════════════════════════════
class InferenceEngine:
    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR, device: str | None = None) -> None:
        self._model_dir = get_resource_path(model_dir)
        self._device = device or discover_best_device()
        self._device_info = DEVICE_INFO.get(self._device.split(".")[0], DEVICE_INFO["CPU"])
        self._pipe = None
        self._inference_lock = threading.Lock()
        log.info(f"InferenceEngine init: model_dir='{self._model_dir}', device='{self._device}'")
        self._validate_model_dir()
        self._load_pipeline()

    def _validate_model_dir(self) -> None:
        if not self._model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self._model_dir}")
        xml_files = list(self._model_dir.glob("*.xml")) or list(self._model_dir.rglob("*.xml"))
        if not xml_files:
            raise FileNotFoundError(f"No IR files found in: {self._model_dir}")

    def _load_pipeline(self) -> None:
        try:
            import openvino_genai as ov_genai
            import openvino as ov
        except ImportError:
            raise ImportError("openvino-genai not installed.")

        device_base = self._device.split(".")[0]
        if device_base == "CPU":
            self._pipe = ov_genai.VLMPipeline(str(self._model_dir), self._device, **{"config": OV_CONFIG})
        else:
            self._pipe = ov_genai.VLMPipeline(str(self._model_dir), self._device)
        log.info(f"✓ VLMPipeline loaded on {self._device_info['label']}")

    @property
    def is_locked(self) -> bool:
        return self._inference_lock.locked()

    def process_document(
        self,
        image: Image.Image,
        prompt: str = "Extract all text and key-value pairs.",
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> EngineResult:
        if self._pipe is None:
            return EngineResult(error="Pipeline not initialized.")
        
        if image.mode != "RGB":
            image = image.convert("RGB")

        with self._inference_lock:
            try:
                import openvino_genai as ov_genai
                import openvino as ov
                from backend.processor import pil_to_numpy_raw

                config = ov_genai.GenerationConfig()
                config.max_new_tokens = max_new_tokens
                config.temperature = temperature
                config.do_sample = temperature > 0
                config.repetition_penalty = 1.2

                image_tensor = ov.Tensor(pil_to_numpy_raw(image))
                raw_output = self._pipe.generate(prompt, image=image_tensor, generation_config=config)
                
                raw_text = raw_output.texts[0] if hasattr(raw_output, "texts") else str(raw_output)
                objects = self._parse_grounding(raw_text)

                return EngineResult(raw_text=raw_text, objects=objects)
            except Exception as e:
                log.error(f"Inference failed: {e}", exc_info=True)
                return EngineResult(error=str(e))

    def get_grounding_prompt(self, document_type: str) -> str:
        base = (
            "For each specific field, YOU MUST extract its EXACT text value and its precise location. "
            "STRICT FORMAT IS MANDATORY: <|object_ref_start|>Field: Value<|object_ref_end|><|box_start|>(y1,x1),(y2,x2)<|box_end|>\n"
            "If spatial tags fail, output 'Field: Value' on new lines.\n"
            "DO NOT EXPLAIN. DO NOT REPEAT. BE EXHAUSTIVE.\n"
        )
        t = document_type.lower()
        if "resume" in t:
            return base + "EXTRACT: Full Name, Email, Phone, LinkedIn, GitHub, Location, Job Title, Employer, Dates, Summary, Skills, Degree, University."
        if "invoice" in t:
            return base + "EXTRACT: Vendor Name, Vendor Address, Invoice Number, Invoice Date, Due Date, Items (Desc, Qty, Price, Total), Subtotal, Tax, Grand Total."
        return base + "Extract all identifiable data fields and structured information."

    def classify_document(self, image: Image.Image) -> str:
        res = self.process_document(image, "Classify this document (Resume, Invoice, ID, Other). Return only the type name.", max_new_tokens=32)
        return res.raw_text.strip().replace("'", "").replace('"', "")

    def process_document_with_grounding(self, image: Image.Image, prompt: Optional[str] = None, document_type: str = "Unknown") -> EngineResult:
        p = prompt or self.get_grounding_prompt(document_type)
        return self.process_document(image, p, max_new_tokens=4096, temperature=0.0)

    def detect_pii(self, image: Image.Image) -> EngineResult:
        """Detect PII with bounding boxes."""
        prompt = (
            "Detect and ground all PII: Full Name, Social Security Number, Date of Birth, Phone, Email, Address. "
            "Use format: <|object_ref_start|>Label: Value<|object_ref_end|><|box_start|>(y1,x1),(y2,x2)<|box_end|>"
        )
        return self.process_document_with_grounding(image, prompt=prompt)

    def process_spatial_query(self, image: Image.Image, box_coords: tuple[int, int, int, int], query: str) -> EngineResult:
        """Run a query about a specific region."""
        y1, x1, y2, x2 = box_coords
        spatial_prompt = (
            f"<|object_ref_start|>selected region<|object_ref_end|>"
            f"<|box_start|>({y1},{x1}),({y2},{x2})<|box_end|>\n"
            f"Question: {query}"
        )
        return self.process_document(image, spatial_prompt, max_new_tokens=512)

    def get_embeddings(self, image: Optional[Image.Image] = None, text: Optional[str] = None) -> list[float]:
        """Simulate or extract semantic embeddings."""
        seed = (text or "") + (str(image.size) if image else "")
        rng = np.random.RandomState(hash(seed) & 0xFFFFFFFF)
        return rng.standard_normal(1024).tolist()

    @staticmethod
    def _parse_grounding(raw_text: str) -> list[GroundedObject]:
        objects: list[GroundedObject] = []
        seen_labels = set()

        # 1. Tags (High Precision)
        for match in _GROUNDING_PATTERN.finditer(raw_text):
            raw_label = match.group("label").strip()
            l, v = map(str.strip, raw_label.split(":", 1)) if ":" in raw_label else (raw_label, "")
            try:
                y1, x1, y2, x2 = [float(match.group(k)) / _COORD_SCALE for k in ("y1", "x1", "y2", "x2")]
                objects.append(GroundedObject(label=l, value=v, box_2d=[min(y1, y2), min(x1, x2), max(y1, y2), max(x1, x2)]))
                seen_labels.add(l.lower())
            except: continue

        # 2. Naked Coords (Spatial Fallback)
        if not objects:
            coord_regex = re.compile(r"\((?P<y1>\d+),(?P<x1>\d+)\),\s*\((?P<y2>\d+),(?P<x2>\d+)\)")
            for match in coord_regex.finditer(raw_text):
                try:
                    y1, x1, y2, x2 = [float(match.group(k)) / _COORD_SCALE for k in ("y1", "x1", "y2", "x2")]
                    pos = match.start()
                    pre = raw_text[max(0, pos-60):pos].split("\n")[-1].strip().rstrip(":").strip()
                    label = pre if pre else f"Region {len(objects)+1}"
                    objects.append(GroundedObject(label=label, value="", box_2d=[min(y1, y2), min(x1, x2), max(y1, y2), max(x1, x2)]))
                    seen_labels.add(label.lower())
                except: continue

        # 3. Semantic (Plain Text Fallback - Enhanced for missing colons)
        if not objects or len(objects) < 2:
            # First try line-based "Label: Value"
            semantic_regex = re.compile(r"^(?P<label>[^:\n\r]{2,40}):\s*(?P<value>[^\n\r]+)$", re.MULTILINE)
            for match in semantic_regex.finditer(raw_text):
                l = match.group("label").strip()
                if l.lower() in ("education", "experience", "summary", "skills") or l.lower() in seen_labels: continue
                v = match.group("value").strip()
                if len(v) > 1:
                    objects.append(GroundedObject(label=l, value=v, box_2d=[0.0, 0.0, 0.0, 0.0]))
                    seen_labels.add(l.lower())

            # Secondary recovery for missing colons: "Emailjdoe@email.com"
            common_labels = ["Email", "LinkedIn", "Phone", "Name", "GitHub", "Address"]
            for line in raw_text.split("\n"):
                line = line.strip()
                if ":" in line or not line: continue
                for cl in common_labels:
                    if line.startswith(cl) and len(line) > len(cl) + 2:
                        val = line[len(cl):].strip()
                        if cl.lower() not in seen_labels:
                            objects.append(GroundedObject(label=cl, value=val, box_2d=[0.0, 0.0, 0.0, 0.0]))
                            seen_labels.add(cl.lower())

        return objects

    def is_ready(self) -> bool:
        return self._pipe is not None

    def clear_cache(self) -> None:
        import gc
        with self._inference_lock:
            gc.collect()

    def unload_pipeline(self) -> None:
        import gc
        with self._inference_lock:
            if self._pipe: del self._pipe
            self._pipe = None
            gc.collect()
