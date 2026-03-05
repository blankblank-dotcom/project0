"""
backend/pii_detector.py — PII Detection Layer using Presidio
===========================================================

Parallel PII detection that runs alongside VLM inference.
Provides structured PII identification with confidence scores.
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Dict, Optional
from PIL import Image

log = logging.getLogger("pii_detector")

@dataclass
class PIIEntity:
    entity_type: str
    text: str
    confidence: float
    start: int
    end: int

@dataclass 
class PIIResult:
    entities: List[PIIEntity]
    processing_time: float
    text_length: int
    has_pii: bool

class PIIDetector:
    def __init__(self):
        self._analyzer = None
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="pii")
        self._init_lock = threading.Lock()
        self._initialize_analyzer()
    
    def _initialize_analyzer(self):
        """Lazy initialization of Presidio analyzer."""
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine
            
            with self._init_lock:
                if self._analyzer is None:
                    self._analyzer = AnalyzerEngine()
                    self._anonymizer = AnonymizerEngine()
                    log.info("✓ Presidio PII analyzer initialized")
        except ImportError:
            log.warning("Presidio not available - PII detection disabled")
            self._analyzer = None
        except Exception as e:
            log.error(f"Failed to initialize Presidio: {e}")
            self._analyzer = None
    
    def detect_pii_from_text(self, text: str) -> PIIResult:
        """Detect PII in text using Presidio."""
        if not self._analyzer or not text:
            return PIIResult(entities=[], processing_time=0.0, text_length=len(text), has_pii=False)
        
        start_time = time.time()
        try:
            results = self._analyzer.analyze(
                text=text,
                language='en',
                entities=['PERSON', 'EMAIL_ADDRESS', 'PHONE_NUMBER', 'IBAN_CODE', 
                         'CREDIT_CARD', 'IP_ADDRESS', 'LOCATION', 'DATE_TIME',
                         'NRP', 'URL', 'US_SSN', 'US_DRIVER_LICENSE']
            )
            
            entities = []
            for result in results:
                entity = PIIEntity(
                    entity_type=result.entity_type,
                    text=text[result.start:result.end],
                    confidence=result.score,
                    start=result.start,
                    end=result.end
                )
                entities.append(entity)
            
            processing_time = time.time() - start_time
            has_pii = len(entities) > 0
            
            log.debug(f"PII detection: {len(entities)} entities in {processing_time:.3f}s")
            
            return PIIResult(
                entities=entities,
                processing_time=processing_time,
                text_length=len(text),
                has_pii=has_pii
            )
            
        except Exception as e:
            log.error(f"PII detection failed: {e}")
            return PIIResult(entities=[], processing_time=0.0, text_length=len(text), has_pii=False)
    
    def detect_pii_parallel(self, text: str, timeout: float = 5.0) -> PIIResult:
        """Run PII detection in parallel with timeout."""
        if not self._analyzer:
            return PIIResult(entities=[], processing_time=0.0, text_length=len(text), has_pii=False)
        
        try:
            future = self._executor.submit(self.detect_pii_from_text, text)
            return future.result(timeout=timeout)
        except Exception as e:
            log.warning(f"Parallel PII detection failed: {e}")
            return PIIResult(entities=[], processing_time=0.0, text_length=len(text), has_pii=False)
    
    def anonymize_text(self, text: str, entities: List[PIIEntity]) -> str:
        """Anonymize PII in text."""
        if not self._anonymizer or not entities:
            return text
        
        try:
            from presidio_anonymizer import OperatorConfig
            
            # Convert entities back to Presidio format
            presidio_entities = []
            for entity in entities:
                from presidio_analyzer import RecognizerResult
                presidio_entity = RecognizerResult(
                    entity_type=entity.entity_type,
                    start=entity.start,
                    end=entity.end,
                    score=entity.confidence
                )
                presidio_entities.append(presidio_entity)
            
            # Anonymize with default operators
            anonymized = self._anonymizer.anonymize(
                text=text,
                analyzer_results=presidio_entities,
                operators={"DEFAULT": OperatorConfig("replace", {"new_value": "[REDACTED]"})}
            )
            
            return anonymized.text
            
        except Exception as e:
            log.error(f"Anonymization failed: {e}")
            return text
    
    def get_pii_summary(self, pii_result: PIIResult) -> Dict[str, int]:
        """Get summary of PII types detected."""
        summary = {}
        for entity in pii_result.entities:
            summary[entity.entity_type] = summary.get(entity.entity_type, 0) + 1
        return summary
    
    def cleanup(self):
        """Cleanup resources."""
        if self._executor:
            self._executor.shutdown(wait=True)
