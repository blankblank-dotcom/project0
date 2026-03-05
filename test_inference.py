#!/usr/bin/env python3
"""
Test script to verify inference timeout fixes and NPU optimizations.
"""

import time
import logging
from PIL import Image
from backend.engine import InferenceEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("test")

def create_test_image():
    """Create a simple test image for inference."""
    from PIL import ImageDraw
    
    # Create a simple test image with text
    img = Image.new('RGB', (1024, 768), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw some test content
    draw.text((50, 50), "Test Document", fill='black')
    draw.text((50, 100), "Name: John Doe", fill='black')
    draw.text((50, 150), "Email: john.doe@example.com", fill='black')
    draw.text((50, 200), "Phone: (555) 123-4567", fill='black')
    
    return img

def test_engine_initialization():
    """Test engine initialization with device detection."""
    log.info("Testing engine initialization...")
    
    try:
        engine = InferenceEngine()
        log.info(f"✓ Engine initialized successfully")
        log.info(f"  Device: {engine.device_name}")
        log.info(f"  Device ID: {engine.device_id}")
        log.info(f"  Icon: {engine.device_icon}")
        log.info(f"  Color: {engine.device_color}")
        return engine
    except Exception as e:
        log.error(f"✗ Engine initialization failed: {e}")
        return None

def test_fast_inference(engine):
    """Test fast inference mode."""
    log.info("Testing fast inference mode...")
    
    try:
        test_img = create_test_image()
        
        start_time = time.time()
        result = engine.process_document_fast(
            image=test_img,
            prompt="Extract name and email only.",
            inference_timeout=30.0  # Should complete well before this
        )
        elapsed = time.time() - start_time
        
        if result.error:
            log.error(f"✗ Fast inference failed: {result.error}")
        else:
            log.info(f"✓ Fast inference completed in {elapsed:.2f}s")
            log.info(f"  Objects found: {len(result.objects)}")
            log.info(f"  Raw text length: {len(result.raw_text)}")
            log.info(f"  PII detected: {result.has_pii}")
        
        return result
        
    except Exception as e:
        log.error(f"✗ Fast inference exception: {e}")
        return None

def test_single_pass_inference(engine):
    """Test combined classify+extract single-pass mode."""
    log.info("Testing single-pass classify+extract mode...")

    try:
        test_img = create_test_image()

        start_time = time.time()
        result = engine.classify_and_extract(test_img)
        elapsed = time.time() - start_time

        if result.error:
            log.error(f"✗ Single-pass failed: {result.error}")
        else:
            log.info(f"✓ Single-pass completed in {elapsed:.2f}s")
            log.info(f"  Document type: {result.document_type} ({result.confidence}% confidence)")
            log.info(f"  Fields found: {len(result.objects)}")
            log.info(f"  Raw text length: {len(result.raw_text)}")
            for obj in result.objects[:5]:  # Show first 5 fields
                log.info(f"    {obj.label}: {obj.value}")

        return result

    except Exception as e:
        log.error(f"✗ Single-pass exception: {e}")
        return None


def test_full_inference(engine):
    """Test full inference mode with longer timeout."""
    log.info("Testing full inference mode...")
    
    try:
        test_img = create_test_image()
        
        start_time = time.time()
        result = engine.process_document(
            image=test_img,
            prompt="Extract all text and key-value pairs with locations.",
            inference_timeout=120.0  # 2 minute timeout
        )
        elapsed = time.time() - start_time
        
        if result.error:
            log.error(f"✗ Full inference failed: {result.error}")
        else:
            log.info(f"✓ Full inference completed in {elapsed:.2f}s")
            log.info(f"  Objects found: {len(result.objects)}")
            log.info(f"  Raw text length: {len(result.raw_text)}")
            log.info(f"  PII detected: {result.has_pii}")
            log.info(f"  PII summary: {result.pii_summary}")
        
        return result
        
    except Exception as e:
        log.error(f"✗ Full inference exception: {e}")
        return None

def main():
    """Run all tests."""
    log.info("=" * 60)
    log.info("THE LOCAL TITAN - INFERENCE TIMEOUT TEST")
    log.info("=" * 60)
    
    # Test 1: Engine initialization
    engine = test_engine_initialization()
    if not engine:
        log.error("Cannot proceed without engine initialization")
        return
    
    # Test 2: Fast inference
    fast_result = test_fast_inference(engine)

    # Test 3: Single-pass classify+extract (the optimized path)
    single_result = test_single_pass_inference(engine)

    # Test 4: Full inference (legacy two-pass, kept for comparison)
    full_result = test_full_inference(engine)

    # Cleanup
    try:
        engine.cleanup()
        log.info("✓ Engine cleanup completed")
    except Exception as e:
        log.error(f"✗ Engine cleanup failed: {e}")

    log.info("=" * 60)
    log.info("TEST SUMMARY")
    log.info("=" * 60)

    if fast_result and not fast_result.error:
        log.info("✓ Fast inference: PASSED")
    else:
        log.info("✗ Fast inference: FAILED")

    if single_result and not single_result.error:
        log.info("✓ Single-pass inference: PASSED")
    else:
        log.info("✗ Single-pass inference: FAILED")

    if full_result and not full_result.error:
        log.info("✓ Full inference: PASSED")
    else:
        log.info("✗ Full inference: FAILED")

    log.info("Test completed.")

if __name__ == "__main__":
    main()
