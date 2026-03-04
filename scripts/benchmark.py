import time
import sys
import os
from pathlib import Path
from PIL import Image
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    import openvino as ov
    from backend.engine import InferenceEngine, discover_best_device
except ImportError as e:
    print(f"Error: Required modules not found. {e}")
    sys.exit(1)

def run_benchmark(model_path="./qwen_ov_int4"):
    print("="*60)
    print(" THE LOCAL TITAN: INFERENCE BENCHMARK SUITE")
    print("="*60)
    
    device = discover_best_device()
    print(f"Hardware: {device}")
    
    if not os.path.exists(model_path):
        print(f"✗ Error: Model not found at {model_path}")
        return

    print("Loading pipeline...")
    start_load = time.perf_counter()
    try:
        pipeline = InferenceEngine(model_path, device=device)
    except Exception as e:
        print(f"✗ Failed to load pipeline: {e}")
        return
    load_time = time.perf_counter() - start_load
    print(f"Pipeline loaded in {load_time:.2f}s")

    # Create a dummy image
    img = Image.new("RGB", (640, 480), "white")
    prompt = "Describe this document."

    print("\nWarm-up run (1/1)...")
    pipeline.process_document(img, prompt)

    print("\nProfiling 5 iterative runs...")
    stats = {
        "encoding_ms": [],
        "ttft_ms": [],
        "tps": [],
        "total_inference_ms": []
    }

    for i in range(5):
        print(f"Run {i+1}/5...", end="", flush=True)
        
        # We wrap the generate call to measure timings
        # Since the original generate doesn't return internal timings, 
        # we estimate based on the full call if we can't patch it easily.
        # But for the benchmark, let's assume we want total page turnaround.
        
        start_run = time.perf_counter()
        # In a real VLM process:
        # 1. Resize/Normalize (Processor)
        # 2. Vision Encoding
        # 3. LLM Prefill
        # 4. LLM Generation
        
        # We'll measure the whole block for "Page Turnaround"
        res = pipeline.process_document(img, prompt)
        if res.error:
            print(f" ✗ Error: {res.error}")
            continue
        end_run = time.perf_counter()
        
        elapsed_ms = (end_run - start_run) * 1000
        stats["total_inference_ms"].append(elapsed_ms)
        
        # Estimate TPS (Qwen usually outputs ~20-50 tokens for 'describe')
        # This is a rough proxy for UI responsiveness
        token_count = len(res.raw_text.split()) # Very rough estimate
        tps = token_count / (end_run - start_run)
        stats["tps"].append(tps)
        
        print(f" {elapsed_ms:.0f}ms")

    avg_ms = sum(stats["total_inference_ms"]) / len(stats["total_inference_ms"])
    avg_tps = sum(stats["tps"]) / len(stats["tps"])

    print("\n" + "-"*60)
    print(" RESULTS SUMMARY")
    print("-"*60)
    print(f" Average Page Processing: {avg_ms:.0f} ms")
    print(f" Estimated Throughput:   {avg_tps:.1f} tokens/sec")
    print(f" Recommended UI ETA:     {avg_ms/1000:.2f} sec/page")
    print("-"*60)

    # Save to a local config so the UI can absorb it
    benchmark_data = {
        "device": device,
        "avg_ms_per_page": avg_ms,
        "avg_tps": avg_tps,
        "timestamp": time.time()
    }
    
    with open("benchmark_results.json", "w") as f:
        json.dump(benchmark_data, f, indent=4)
    print("✓ Results saved to benchmark_results.json")

if __name__ == "__main__":
    run_benchmark()
