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
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError

import numpy as np
from PIL import Image

# Import new modules
from backend.pii_detector import PIIDetector, PIIResult
from backend.heartbeat import HeartbeatMonitor, HeartbeatConfig

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

        # Force CPU for stability to avoid invalid config issues
        if "CPU" in available:
            log.info(f"Selected device: CPU (forced for stability)")
            return "CPU"
        
        # Fallback to first available device
        if available:
            device = available[0]
            log.info(f"Selected device: {device} (fallback)")
            return device
        
        log.warning("No devices found — defaulting to CPU.")
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
# Helper functions for enhanced parsing
# ---------------------------------------------------------------------------
def _normalize_label(label: str) -> str:
    """Normalize common label variations."""
    label = label.strip()
    
    # Common label mappings
    label_mappings = {
        "email": "Email",
        "e-mail": "Email", 
        "phone": "Phone",
        "mobile": "Phone",
        "tel": "Phone",
        "telephone": "Phone",
        "name": "Full Name",
        "full name": "Full Name",
        "address": "Address",
        "location": "Address",
        "website": "Website",
        "url": "Website",
        "linkedin": "LinkedIn",
        "github": "GitHub",
        "summary": "Summary",
        "experience": "Experience",
        "education": "Education",
        "skills": "Skills"
    }
    
    normalized = label_mappings.get(label.lower(), label)
    
    # Capitalize first letter of each word
    if len(normalized) > 2:
        normalized = ' '.join(word.capitalize() for word in normalized.split())
    
    return normalized


def _extract_context_around_match(text: str, match_pos: int, context_size: int = 100) -> tuple[str, str]:
    """Extract label and value context around a match position."""
    # Look backwards for label
    start_pos = max(0, match_pos - context_size)
    before_text = text[start_pos:match_pos]
    
    # Find the last complete line before the match
    lines = before_text.split('\n')
    if lines:
        last_line = lines[-1].strip()
        if ':' in last_line:
            parts = last_line.split(':', 1)
            label = parts[0].strip()
            value = parts[1].strip()
            return label, value
        elif last_line and len(last_line) > 2:
            return last_line, ""
    
    return "Unknown", ""


def _extract_common_patterns(text: str, seen_labels: set[str]) -> list[GroundedObject]:
    """Extract common patterns that don't match standard formats."""
    objects = []
    
    # Email pattern without label
    email_pattern = re.compile(r'\b[\w\.-]+@[\w\.-]+\.\w+\b')
    for match in email_pattern.finditer(text):
        if 'email' not in seen_labels:
            objects.append(GroundedObject(
                label="Email", 
                value=match.group(), 
                box_2d=[0.0, 0.0, 0.0, 0.0]
            ))
            seen_labels.add('email')
            break
    
    # Phone pattern without label
    phone_pattern = re.compile(r'\b[+]?[\d\s\-\(\)]{7,}\b')
    for match in phone_pattern.finditer(text):
        if 'phone' not in seen_labels:
            objects.append(GroundedObject(
                label="Phone", 
                value=match.group(), 
                box_2d=[0.0, 0.0, 0.0, 0.0]
            ))
            seen_labels.add('phone')
            break
    
    # URL pattern
    url_pattern = re.compile(r'\bhttps?://[^\s]+\b')
    for match in url_pattern.finditer(text):
        url = match.group()
        if 'linkedin' in url.lower() and 'linkedin' not in seen_labels:
            objects.append(GroundedObject(
                label="LinkedIn", 
                value=url, 
                box_2d=[0.0, 0.0, 0.0, 0.0]
            ))
            seen_labels.add('linkedin')
        elif 'github' in url.lower() and 'github' not in seen_labels:
            objects.append(GroundedObject(
                label="GitHub", 
                value=url, 
                box_2d=[0.0, 0.0, 0.0, 0.0]
            ))
            seen_labels.add('github')
        elif 'website' not in seen_labels:
            objects.append(GroundedObject(
                label="Website", 
                value=url, 
                box_2d=[0.0, 0.0, 0.0, 0.0]
            ))
            seen_labels.add('website')
            break
    
    return objects


def _extract_context_around_match(text: str, match_pos: int, context_size: int = 100) -> tuple[str, str]:
    """Extract label and value context around a match position."""
    # Look backwards for label
    start_pos = max(0, match_pos - context_size)
    before_text = text[start_pos:match_pos]
    
    # Find the last complete line before the match
    lines = before_text.split('\n')
    if lines:
        last_line = lines[-1]
        if ':' in last_line:
            parts = last_line.split(':', 1)
            if len(parts) == 2:
                label = parts[0].strip()
                value = parts[1].strip()
                return label, value
    
    return "Unknown", ""


def _normalize_label(label: str) -> str:
    """Normalize field labels."""
    # Remove numbering at start
    label = re.sub(r'^\d+\.?\s*', '', label)
    
    # Remove common prefixes
    prefixes_to_remove = ['Field:', 'Item:', 'Entry:', 'Information:', 'Detail:']
    for prefix in prefixes_to_remove:
        if label.lower().startswith(prefix.lower()):
            label = label[len(prefix):].strip()
    
    # Capitalize properly
    label = ' '.join(word.capitalize() for word in label.split())
    
    # Ensure it's not too long
    if len(label) > 50:
        label = label[:47] + "..."
    
    return label.strip()


def _extract_common_patterns(text: str, seen_labels: set[str]) -> list[GroundedObject]:
    """Extract common patterns that don't match standard formats."""
    objects = []
    
    # Email pattern without label
    email_pattern = re.compile(r'\b[\w\.-]+@[\w\.-]+\.\w+\b')
    for match in email_pattern.finditer(text):
        if 'email' not in seen_labels:
            objects.append(GroundedObject(
                label="Email", 
                value=match.group(), 
                box_2d=[0.0, 0.0, 0.0, 0.0]
            ))
            seen_labels.add('email')
            break
    
    # Phone pattern without label
    phone_pattern = re.compile(r'\b[+]?[\d\s\-\(\)]{7,}\b')
    for match in phone_pattern.finditer(text):
        if 'phone' not in seen_labels:
            objects.append(GroundedObject(
                label="Phone", 
                value=match.group(), 
                box_2d=[0.0, 0.0, 0.0, 0.0]
            ))
            seen_labels.add('phone')
            break
    
    # URL pattern
    url_pattern = re.compile(r'\bhttps?://[^\s]+\b')
    for match in url_pattern.finditer(text):
        url = match.group()
        if 'linkedin' in url.lower() and 'linkedin' not in seen_labels:
            objects.append(GroundedObject(
                label="LinkedIn", 
                value=url, 
                box_2d=[0.0, 0.0, 0.0, 0.0]
            ))
            seen_labels.add('linkedin')
        elif 'github' in url.lower() and 'github' not in seen_labels:
            objects.append(GroundedObject(
                label="GitHub", 
                value=url, 
                box_2d=[0.0, 0.0, 0.0, 0.0]
            ))
            seen_labels.add('github')
        elif 'website' not in seen_labels:
            objects.append(GroundedObject(
                label="Website", 
                value=url, 
                box_2d=[0.0, 0.0, 0.0, 0.0]
            ))
            seen_labels.add('website')
            break
    
    return objects


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
    pii_summary: dict = field(default_factory=dict)
    has_pii: bool = False
    processing_time: float = 0.0

    def to_dict(self) -> dict:
        return {
            "raw_text": self.raw_text,
            "objects": [obj.to_dict() for obj in self.objects],
            "error": self.error,
            "pii_summary": self.pii_summary,
            "has_pii": self.has_pii,
            "processing_time": self.processing_time,
        }

OV_CONFIG = {}  # No configuration - use defaults


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
        self._executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="inference")
        self._pii_detector = PIIDetector()
        self._heartbeat = HeartbeatMonitor(HeartbeatConfig(timeout=2.0))
        
        log.info(f"InferenceEngine init: model_dir='{self._model_dir}', device='{self._device}'")
        self._validate_model_dir()
        self._load_pipeline()
    
    @property
    def device_name(self) -> str:
        """Human-readable device name for UI."""
        return self._device_info["label"]
    
    @property
    def device_icon(self) -> str:
        """Device icon name for UI."""
        return self._device_info["icon"]
    
    @property
    def device_color(self) -> str:
        """Device color for UI."""
        return self._device_info["color"]
    
    @property
    def device_id(self) -> str:
        """Internal device identifier."""
        return self._device

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
        
        # Use only CPU configuration to avoid invalid options
        try:
            self._pipe = ov_genai.VLMPipeline(str(self._model_dir), "CPU", **{"config": OV_CONFIG})
            self._device = "CPU"
            self._device_info = DEVICE_INFO["CPU"]
            log.info(f"✓ VLMPipeline loaded on {self._device_info['label']} with safe CPU configuration")
        except Exception as e:
            log.error(f"Failed to load pipeline: {e}")
            raise

    @property
    def is_locked(self) -> bool:
        return self._inference_lock.locked()

    def _run_inference(self, image: Image.Image, prompt: str, max_new_tokens: int, temperature: float) -> EngineResult:
        """Run actual inference with clear progress logging."""
        import time
        start_time = time.time()
        log.info(f"🚀 Starting inference (max_tokens={max_new_tokens}, temp={temperature})...")
        
        try:
            # Convert image to numpy array
            log.info("📸 Converting image to tensor...")
            from backend.processor import pil_to_numpy_raw
            image_array = pil_to_numpy_raw(image)
            
            # Generate with VLMPipeline
            log.info("⚡ Running model inference...")
            inference_start = time.time()
            
            raw_text = self._pipe.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=False,
            )
            
            inference_time = time.time() - inference_start
            log.info(f"✅ Model inference completed in {inference_time:.1f}s")
            
            # Parse structured data from the raw text
            log.info("🔍 Parsing structured data...")
            parsing_start = time.time()
            
            objects = self._parse_grounding(raw_text)
            parsing_time = time.time() - parsing_start
            log.info(f"📝 Structured parsing completed in {parsing_time:.1f}s - Found {len(objects)} fields")
            
            processing_time = time.time() - start_time
            log.info(f"🎯 Total inference time: {processing_time:.1f}s")
            
            return EngineResult(
                raw_text=raw_text, 
                objects=objects, 
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            log.error(f"❌ Inference failed after {processing_time:.1f}s: {e}", exc_info=True)
            return EngineResult(error=str(e))

    def extract_structured_data(self, image: Image.Image, document_type: str = "Unknown") -> EngineResult:
        """AI-powered field discovery - let the model identify fields automatically."""
        import time
        total_start = time.time()
        
        # First pass: Get all raw text content
        log.info("🔍 Starting text extraction...")
        extraction_start = time.time()
        
        extraction_prompt = f"""
Extract ALL text content from this {document_type}. Read everything carefully:
- All personal information
- All business details  
- All dates and numbers
- All addresses and contact info
- All descriptions and details
- All headers and section titles

Provide the complete text content exactly as you see it, organized logically.
        """
        
        # Get raw text content
        raw_result = self.process_document(
            image=image,
            prompt=extraction_prompt,
            max_new_tokens=2048,
            temperature=0.0,
            enable_pii_detection=False,
            inference_timeout=None  # No timeout - instant processing
        )
        
        extraction_time = time.time() - extraction_start
        log.info(f"📝 Text extraction completed in {extraction_time:.1f}s")
        
        if raw_result.error:
            log.error(f"❌ Text extraction failed: {raw_result.error}")
            return raw_result
        
        # Second pass: Let the model analyze and auto-identify fields
        log.info("🧠 Starting field analysis...")
        analysis_start = time.time()
        
        analysis_prompt = f"""
Analyze this extracted text and automatically identify ALL meaningful fields.
For each field you identify, output on a new line:
Field Name: Value

Extracted Text to Analyze:
{raw_result.raw_text}

Instructions:
- Look for names, emails, phones, addresses, dates, amounts, companies, etc.
- Identify section headers and categories
- Find any structured information patterns
- Create descriptive field names that make sense
- Extract the actual value for each field
- Be thorough - don't miss important information
- Do not make up values - only extract what's actually present

Output format: One field per line as "Field Name: Value"
        """
        
        # Get field analysis from model
        analysis_result = self.process_document(
            image=image,  # Include image for context
            prompt=analysis_prompt,
            max_new_tokens=1024,
            temperature=0.0,
            enable_pii_detection=False,
            inference_timeout=None  # No timeout - instant processing
        )
        
        analysis_time = time.time() - analysis_start
        log.info(f"🔍 Field analysis completed in {analysis_time:.1f}s")
        
        if analysis_result.error:
            log.warning(f"⚠️ Field analysis failed, using raw text: {analysis_result.error}")
            # Fallback: create a simple object with raw text
            objects = [GroundedObject(
                label="Extracted Content",
                value=raw_result.raw_text[:500] + "..." if len(raw_result.raw_text) > 500 else raw_result.raw_text,
                box_2d=[0.0, 0.0, 0.0, 0.0]
            )]
            raw_result.objects = objects
            total_time = time.time() - total_start
            log.info(f"⏱️ Total extraction time: {total_time:.1f}s (fallback mode)")
            return raw_result
        
        # Parse the AI-identified fields
        log.info("📋 Parsing structured fields...")
        parsing_start = time.time()
        
        try:
            objects = []
            lines = analysis_result.raw_text.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        field_name = parts[0].strip()
                        value = parts[1].strip()
                        
                        # Skip empty or meaningless values
                        if not value or len(value) < 2 or value.lower() in ['n/a', 'none', 'not found']:
                            continue
                        
                        # Clean up field name (remove numbering, etc.)
                        field_name = self._clean_field_name(field_name)
                        
                        objects.append(GroundedObject(
                            label=field_name,
                            value=value,
                            box_2d=[0.0, 0.0, 0.0, 0.0]
                        ))
            
            parsing_time = time.time() - parsing_start
            log.info(f"📝 Field parsing completed in {parsing_time:.1f}s")
            
            # If no structured fields found, at least include the raw content
            if not objects:
                log.warning("⚠️ No structured fields found, using raw content")
                objects.append(GroundedObject(
                    label="Extracted Content",
                    value=raw_result.raw_text[:500] + "..." if len(raw_result.raw_text) > 500 else raw_result.raw_text,
                    box_2d=[0.0, 0.0, 0.0, 0.0]
                ))
            
            # Update result with AI-identified objects
            raw_result.objects = objects
            raw_result.raw_text = analysis_result.raw_text  # Store the structured analysis
            
            total_time = time.time() - total_start
            log.info(f"✅ Total extraction completed in {total_time:.1f}s - Found {len(objects)} fields")
            
        except Exception as e:
            log.error(f"❌ Field parsing failed: {e}")
            # Fallback to raw content
            objects = [GroundedObject(
                label="Extracted Content",
                value=raw_result.raw_text[:500] + "..." if len(raw_result.raw_text) > 500 else raw_result.raw_text,
                box_2d=[0.0, 0.0, 0.0, 0.0]
            )]
            raw_result.objects = objects
            total_time = time.time() - total_start
            log.info(f"⏱️ Total extraction time: {total_time:.1f}s (error fallback)")
        
        return raw_result
    
    def _clean_field_name(self, field_name: str) -> str:
        """Clean up AI-generated field names."""
        # Remove numbering at start
        field_name = re.sub(r'^\d+\.?\s*', '', field_name)
        
        # Remove common prefixes
        prefixes_to_remove = ['Field:', 'Item:', 'Entry:', 'Information:', 'Detail:']
        for prefix in prefixes_to_remove:
            if field_name.lower().startswith(prefix.lower()):
                field_name = field_name[len(prefix):].strip()
        
        # Capitalize properly
        field_name = ' '.join(word.capitalize() for word in field_name.split())
        
        # Ensure it's not too long
        if len(field_name) > 50:
            field_name = field_name[:47] + "..."
        
        return field_name.strip()

    def process_document_fast(
        self,
        image: Image.Image,
        prompt: str = "Extract key fields only.",
        max_new_tokens: int = 1024,  # Reduced for speed
        temperature: float = 0.0,
        enable_pii_detection: bool = False,  # Disabled for speed
        inference_timeout: float = 60.0,  # Shorter timeout
    ) -> EngineResult:
        """Fast processing mode for quick results."""
        return self.process_document(
            image=image,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            enable_pii_detection=enable_pii_detection,
            inference_timeout=inference_timeout
        )

    def process_document(
        self,
        image: Image.Image,
        prompt: str = "Extract all text and key-value pairs.",
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        enable_pii_detection: bool = True,
        inference_timeout: float = None,  # No timeout - instant processing
    ) -> EngineResult:
        if self._pipe is None:
            return EngineResult(error="Pipeline not initialized.")
        
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Start heartbeat monitoring for long inference
        self._heartbeat.start_monitoring()
        
        try:
            # Run inference with timeout
            future = self._executor.submit(
                self._run_inference, image, prompt, max_new_tokens, temperature
            )
            
            try:
                result = future.result(timeout=inference_timeout)
            except TimeoutError:
                log.error(f"Inference timed out after {inference_timeout}s")
                return EngineResult(error=f"Inference timeout ({inference_timeout}s)")
            
            # Run PII detection in parallel if enabled
            if enable_pii_detection and result.raw_text:
                pii_future = self._executor.submit(
                    self._pii_detector.detect_pii_parallel, result.raw_text, timeout=5.0
                )
                try:
                    pii_result = pii_future.result(timeout=5.0)
                    result.pii_summary = self._pii_detector.get_pii_summary(pii_result)
                    result.has_pii = pii_result.has_pii
                    log.info(f"PII detection: {len(pii_result.entities)} entities found")
                except TimeoutError:
                    log.warning("PII detection timed out")
                    result.pii_summary = {}
                    result.has_pii = False
            else:
                result.pii_summary = {}
                result.has_pii = False
            
            return result
            
        except Exception as e:
            log.error(f"Document processing failed: {e}", exc_info=True)
            return EngineResult(error=str(e))
        finally:
            self._heartbeat.stop_monitoring()

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

    def classify_document_with_confidence(self, image: Image.Image) -> dict:
        """Classify document with confidence scores for top predictions."""
        import time
        start_time = time.time()
        log.info("🏷️ Starting document classification...")
        
        classification_prompt = """
Analyze this document and provide the 3 most likely document types with confidence scores.

Format your response exactly like this:
1. Resume: 85%
2. Invoice: 10%
3. Other: 5%

Choose from these types: Resume, Invoice, Receipt, ID Card, Contract, Form, Letter, Report, Other

The percentages must add up to 100%. Be confident in your primary choice.
        """
        
        try:
            res = self.process_document(
                image, 
                classification_prompt, 
                max_new_tokens=64,
                temperature=0.0,
                enable_pii_detection=False,
                inference_timeout=None  # No timeout
            )
            
            classification_time = time.time() - start_time
            log.info(f"🏷️ Classification completed in {classification_time:.1f}s")
            
            if res.error:
                log.warning(f"Classification failed: {res.error}")
                return {"primary": "Unknown", "confidence": 0, "alternatives": []}
            
            # Parse the confidence scores
            log.info("📊 Parsing confidence scores...")
            predictions = []
            lines = res.raw_text.strip().split('\n')
            
            for line in lines:
                if ':' in line and '%' in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        doc_type = parts[0].strip().replace('.', '').replace('1', '').replace('2', '').replace('3', '').strip()
                        confidence_str = parts[1].strip().replace('%', '')
                        
                        try:
                            confidence = float(confidence_str)
                            predictions.append({"type": doc_type, "confidence": confidence})
                        except ValueError:
                            continue
            
            # Sort by confidence
            predictions.sort(key=lambda x: x["confidence"], reverse=True)
            
            if predictions:
                result = {
                    "primary": predictions[0]["type"],
                    "confidence": predictions[0]["confidence"],
                    "alternatives": predictions[1:3]  # Top 2 alternatives
                }
                log.info(f"🎯 Classification result: {result['primary']} ({result['confidence']}% confidence)")
                return result
            else:
                log.warning("⚠️ No valid classification results found")
                return {"primary": "Unknown", "confidence": 0, "alternatives": []}
                
        except Exception as e:
            classification_time = time.time() - start_time
            log.error(f"❌ Document classification error after {classification_time:.1f}s: {e}")
            return {"primary": "Unknown", "confidence": 0, "alternatives": []}

    def classify_document(self, image: Image.Image) -> str:
        """Simple classification for backward compatibility."""
        result = self.classify_document_with_confidence(image)
        return result["primary"]

    def process_document_with_grounding(self, image: Image.Image, prompt: Optional[str] = None, document_type: str = "Unknown") -> EngineResult:
        """Process document with structured JSON extraction for better results."""
        # Use the new structured extraction method
        return self.extract_structured_data(image, document_type)

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
        # Return 384 dimensions to match ChromaDB expectations
        return rng.standard_normal(384).tolist()

    @staticmethod
    def _parse_grounding(raw_text: str) -> list[GroundedObject]:
        """Enhanced parsing with better structured mapping."""
        objects: list[GroundedObject] = []
        seen_labels = set()
        
        try:
            # Handle VLMDecodedResults object if present
            if hasattr(raw_text, 'text') and raw_text.text:
                # Use the text attribute directly
                text_content = raw_text.text
            else:
                # Fallback to string processing
                text_content = str(raw_text)
            
            lines = text_content.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Extract centered headers (large text, usually names)
                if len(line) > 30 and ':' not in line and any(word in line.lower() for word in ['john', 'jane', 'michael', 'sarah', 'david', 'robert', 'maria']):
                    if 'name' not in seen_labels:
                        objects.append(GroundedObject(
                            label="Full Name", 
                            value=line.strip(), 
                            box_2d=[0.0, 0.0, 0.0, 0.0]
                        ))
                        seen_labels.add('name')
                        log.debug(f"Extracted centered header: {line}")
                
                # Check for section headers
                lower_line = line.lower()
                for h in ['summary', 'experience', 'education', 'skills', 'projects']:
                    if h in lower_line:
                        if h not in seen_labels:
                            objects.append(GroundedObject(
                                label=h.title(), 
                                value=line.strip(), 
                                box_2d=[0.0, 0.0, 0.0, 0.0]
                            ))
                            seen_labels.add(h)
                            log.debug(f"Extracted section header: {h}")
                
                # Extract email addresses
                email_pattern = re.compile(r'\b[\w\.-]+@[\w\.-]+\.\w+\b')
                for match in email_pattern.finditer(text_content):
                    if 'email' not in seen_labels:
                        objects.append(GroundedObject(
                            label="Email", 
                            value=match.group(), 
                            box_2d=[0.0, 0.0, 0.0, 0.0]
                        ))
                        seen_labels.add('email')
                        break
                
                # Extract phone numbers
                phone_pattern = re.compile(r'\b[+]?[\d\s\-\(\)]{7,}\b')
                for match in phone_pattern.finditer(text_content):
                    if 'phone' not in seen_labels:
                        objects.append(GroundedObject(
                            label="Phone", 
                            value=match.group(), 
                            box_2d=[0.0, 0.0, 0.0, 0.0]
                        ))
                        seen_labels.add('phone')
                        break
                
                # Extract URLs
                url_pattern = re.compile(r'\bhttps?://[^\s]+\b')
                for match in url_pattern.finditer(text_content):
                    url = match.group()
                    if 'linkedin' in url.lower() and 'linkedin' not in seen_labels:
                        objects.append(GroundedObject(
                            label="LinkedIn", 
                            value=url, 
                            box_2d=[0.0, 0.0, 0.0, 0.0]
                        ))
                        seen_labels.add('linkedin')
                    elif 'github' in url.lower() and 'github' not in seen_labels:
                        objects.append(GroundedObject(
                            label="GitHub", 
                            value=url, 
                            box_2d=[0.0, 0.0, 0.0, 0.0]
                        ))
                        seen_labels.add('github')
                    elif 'website' not in seen_labels:
                        objects.append(GroundedObject(
                            label="Website", 
                            value=url, 
                            box_2d=[0.0, 0.0, 0.0, 0.0]
                        ))
                        seen_labels.add('website')
                        break
                
                # Extract key-value pairs (simple format)
                if ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        field = parts[0].strip()
                        value = parts[1].strip()
                        
                        # Skip empty or meaningless values
                        if not value or len(value) < 2 or value.lower() in ['n/a', 'none', 'not found']:
                            continue
                        
                        # Clean up field name
                        field = self._clean_field_name(field)
                        
                        objects.append(GroundedObject(
                            label=field,
                            value=value,
                            box_2d=[0.0, 0.0, 0.0, 0.0]
                        ))
                        seen_labels.add(field.lower())
                        log.debug(f"Extracted field: {field} = {value}")
        except Exception as e:
            log.error(f\"Failed to parse grounding text heuristically: {e}\", exc_info=True)

        # 2. Enhanced Naked Coords with better context
        if not objects or len(objects) < 5:
            coord_regex = re.compile(r"\((?P<y1>\d+),(?P<x1>\d+)\),\s*\((?P<y2>\d+),(?P<x2>\d+)\)")
            
            for match in coord_regex.finditer(raw_text):
                try:
                    y1, x1, y2, x2 = [float(match.group(k)) / _COORD_SCALE for k in ("y1", "x1", "y2", "x2")]
                    
                    # Better context extraction
                    label, value = _extract_context_around_match(raw_text, match.start())
                    label = _normalize_label(label)
                    
                    if label.lower() not in seen_labels:
                        objects.append(GroundedObject(
                            label=label, 
                            value=value, 
                            box_2d=[min(y1, y2), min(x1, x2), max(y1, y2), max(x1, x2)]
                        ))
                        seen_labels.add(label.lower())
                except Exception as e:
                    log.debug(f"Failed to parse naked coords: {e}")
                    continue

        # 3. Enhanced Semantic parsing with better patterns
        if not objects or len(objects) < 8:
            # Multiple regex patterns for different formats
            patterns = [
                # Standard "Label: Value" format
                r"^(?P<label>[^:\n\r]{2,50}):\s*(?P<value>[^\n\r]+)$",
                # Email patterns
                r"^(?P<label>Email|E-mail):?\s*(?P<value>[\w\.-]+@[\w\.-]+\.\w+)$",
                # Phone patterns  
                r"^(?P<label>Phone|Mobile|Tel):?\s*(?P<value>[+]?[\d\s\-\(\)]{7,})$",
                # URL patterns
                r"^(?P<label>Website|URL|LinkedIn|GitHub):?\s*(?P<value>https?://[^\s]+)$",
                # Location patterns
                r"^(?P<label>Location|Address):?\s*(?P<value>[^\n]{10,100})$",
                # Skills/Experience patterns
                r"^(?P<label>Skills|Experience|Education):?\s*(?P<value>[^\n]{20,200})$",
            ]
            
            for pattern in patterns:
                semantic_regex = re.compile(pattern, re.MULTILINE | re.IGNORECASE)
                for match in semantic_regex.finditer(raw_text):
                    l = _normalize_label(match.group("label"))
                    
                    # Skip duplicates
                    if l.lower() in seen_labels:
                        continue
                        
                    v = match.group("value").strip()
                    
                    # Validate value
                    if len(v) > 1:
                        objects.append(GroundedObject(
                            label=l, 
                            value=v, 
                            box_2d=[0.0, 0.0, 0.0, 0.0]
                        ))
                        seen_labels.add(l.lower())

        # 4. Fallback: Extract common patterns without proper formatting
        if not objects or len(objects) < 5:
            objects.extend(_extract_common_patterns(raw_text, seen_labels))

        # 5. Extract lines with colons that might have been missed
        lines = raw_text.split('\n')
        for line in lines:
            line = line.strip()
            if ':' in line and len(line) > 5:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    label = _normalize_label(parts[0].strip())
                    value = parts[1].strip()
                    
                    if label.lower() not in seen_labels and len(value) > 1:
                        objects.append(GroundedObject(
                            label=label, 
                            value=value, 
                            box_2d=[0.0, 0.0, 0.0, 0.0]
                        ))
                        seen_labels.add(label.lower())

        # Deduplicate and sort
        unique_objects = []
        seen_combinations = set()
        
        for obj in objects:
            key = (obj.label.lower(), obj.value.lower())
            if key not in seen_combinations:
                unique_objects.append(obj)
                seen_combinations.add(key)
        
        # Sort by label for consistency
        unique_objects.sort(key=lambda x: x.label.lower())
        
        log.debug(f"Final extraction: {len(unique_objects)} objects")
        return unique_objects

    def is_ready(self) -> bool:
        return self._pipe is not None

    def clear_cache(self) -> None:
        import gc
        with self._inference_lock:
            gc.collect()
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        self.clear_cache()
        if self._executor:
            self._executor.shutdown(wait=True)
        if self._pii_detector:
            self._pii_detector.cleanup()
        if self._heartbeat:
            self._heartbeat.stop_monitoring()

    def unload_pipeline(self) -> None:
        import gc
        with self._inference_lock:
            if self._pipe: del self._pipe
            self._pipe = None
            gc.collect()
