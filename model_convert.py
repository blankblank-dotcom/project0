"""
model_convert.py — The Local Titan: Qwen 2-VL-2B → OpenVINO INT4 Export
==========================================================================

Converts the Qwen/Qwen2-VL-2B-Instruct model to OpenVINO IR format with
NNCF INT4 data-aware weight compression for efficient inference on Intel
CPUs (i5+) and Intel Arc iGPUs.

Pipeline:
  1. Pre-Flight Check   → Verify ≥12GB free RAM via psutil
  2. Model Download     → Pull from HuggingFace via optimum-cli
  3. INT4 Export        → NNCF data-aware weight compression → IR files
  4. Validation         → Confirm output artifacts exist

Output: ./qwen_ov_int4/  (OpenVINO IR: .xml + .bin files)

Usage:
  python model_convert.py
  python model_convert.py --output ./custom_output_dir
  python model_convert.py --skip-ram-check
"""

import argparse
import gc
import logging
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_ID = "./qwen2vl_2b_local"  # Using locally downloaded weights
DEFAULT_OUTPUT_DIR = "./qwen_ov_int4"
MIN_FREE_RAM_GB = 12  # Minimum free RAM to safely run quantization
WEIGHT_FORMAT = "int4"
COMPRESSION_RATIO = 0.8  # NNCF data-aware compression target ratio

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("model_convert")


# ═══════════════════════════════════════════════════════════════════════════
# 1. PRE-FLIGHT CHECK
# ═══════════════════════════════════════════════════════════════════════════
def preflight_ram_check(min_gb: float = MIN_FREE_RAM_GB) -> None:
    """Verify sufficient free RAM before starting the heavy quantization.

    Prevents the system from hanging or thrashing swap on a 16GB machine
    where the OS + background apps may already consume 4-6GB.
    """
    try:
        import psutil
    except ImportError:
        log.error(
            "psutil is not installed. Run: pip install psutil\n"
            "Or use --skip-ram-check to bypass this check (NOT recommended)."
        )
        sys.exit(1)

    mem = psutil.virtual_memory()
    free_gb = mem.available / (1024 ** 3)
    total_gb = mem.total / (1024 ** 3)

    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║          PRE-FLIGHT SYSTEM RAM CHECK            ║")
    log.info("╠══════════════════════════════════════════════════╣")
    log.info(f"║  Total RAM  : {total_gb:>6.1f} GB                        ║")
    log.info(f"║  Available  : {free_gb:>6.1f} GB                        ║")
    log.info(f"║  Required   : {min_gb:>6.1f} GB (minimum free)          ║")
    log.info("╚══════════════════════════════════════════════════╝")

    if free_gb < min_gb:
        log.error(
            f"INSUFFICIENT MEMORY: {free_gb:.1f} GB free, need {min_gb:.1f} GB.\n"
            f"Close other applications (browsers, IDEs) and retry.\n"
            f"Current memory pressure: {mem.percent:.0f}% used."
        )
        sys.exit(1)

    log.info(f"✓ RAM check passed — {free_gb:.1f} GB available, proceeding.\n")


# ═══════════════════════════════════════════════════════════════════════════
# 2. MODEL DOWNLOAD & INT4 EXPORT VIA OPTIMUM-CLI
# ═══════════════════════════════════════════════════════════════════════════
def export_model_to_openvino_int4(output_dir: str) -> None:
    """Download Qwen 3.5-VL-4B and export to OpenVINO IR with INT4
    data-aware weight compression using the optimum-cli tool.

    The `optimum-cli` tool from optimum-intel handles:
      - Downloading the model from HuggingFace Hub
      - Converting PyTorch weights → OpenVINO IR (.xml/.bin)
      - Applying NNCF INT4 data-aware weight compression in one pass

    This is the 2026 standard approach — a single CLI command replaces
    the older multi-step manual conversion workflow.
    """
    output_path = Path(output_dir)

    # Clean previous export if it exists
    if output_path.exists():
        log.warning(f"Output directory '{output_dir}' already exists.")
        log.warning("Removing previous export to ensure a clean conversion...")
        shutil.rmtree(output_path)

    output_path.mkdir(parents=True, exist_ok=True)

    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║     STARTING OPENVINO INT4 MODEL EXPORT         ║")
    log.info("╠══════════════════════════════════════════════════╣")
    log.info(f"║  Model    : {MODEL_ID:<37s} ║")
    log.info(f"║  Format   : {WEIGHT_FORMAT:<37s} ║")
    log.info(f"║  Method   : Data-Aware Weight Compression       ║")
    log.info(f"║  Output   : {output_dir:<37s} ║")
    log.info("╚══════════════════════════════════════════════════╝")

    # -----------------------------------------------------------------------
    # Build the optimum-cli export command
    #
    # Flags explained:
    #   --model             : HuggingFace model ID to download & convert
    #   --task              : Task hint for the export pipeline
    #   --weight-format     : Target quantization (int4)
    #   --data-aware        : Enable NNCF data-aware compression (calibrates
    #                         on a small dataset for better INT4 accuracy)
    #   --ratio             : Fraction of layers to compress (0.8 = 80%)
    #   --group-size        : Quantization group size (128 is standard)
    #   --sym               : Use symmetric quantization
    #   --output            : Destination directory for IR files
    # -----------------------------------------------------------------------
    cmd = [
        "optimum-cli", "export", "openvino",
        "--model", MODEL_ID,
        "--task", "image-text-to-text",
        "--weight-format", WEIGHT_FORMAT,
        "--ratio", str(COMPRESSION_RATIO),
        "--group-size", "128",
        "--sym",
        "--trust-remote-code",
        str(output_path),
    ]

    log.info(f"Running: {' '.join(cmd)}\n")

    try:
        # Use shell=True for optimum-cli if it's an entry point script
        process = subprocess.run(
            cmd,
            check=True,
            text=True,
            shell=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
    except subprocess.CalledProcessError as e:
        log.error(f"Model export FAILED with return code {e.returncode}.")
        log.error(
            "Common fixes:\n"
            "  1. Ensure you're logged into HuggingFace: huggingface-cli login\n"
            "  2. Accept the Qwen license at: https://huggingface.co/Qwen/Qwen3.5-VL-4B-Instruct\n"
            "  3. Verify optimum-intel is installed: pip install optimum-intel[nncf]>=1.22.0\n"
            "  4. Check you have enough disk space (~15GB for download + conversion)."
        )
        sys.exit(1)
    except FileNotFoundError:
        log.error(
            "optimum-cli not found. Install with:\n"
            "  pip install optimum-intel[nncf]>=1.22.0"
        )
        sys.exit(1)

    log.info("\n✓ Model export completed successfully.\n")


# ═══════════════════════════════════════════════════════════════════════════
# 3. VALIDATION — VERIFY OUTPUT ARTIFACTS
# ═══════════════════════════════════════════════════════════════════════════
def validate_output(output_dir: str) -> None:
    """Confirm the exported IR files exist and report their sizes."""
    output_path = Path(output_dir)

    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║          POST-EXPORT VALIDATION                 ║")
    log.info("╚══════════════════════════════════════════════════╝")

    if not output_path.exists():
        log.error(f"Output directory '{output_dir}' does not exist!")
        sys.exit(1)

    # List all generated files with sizes
    files = sorted(output_path.rglob("*"))
    total_size_mb = 0
    ir_found = False

    for f in files:
        if f.is_file():
            size_mb = f.stat().st_size / (1024 ** 2)
            total_size_mb += size_mb
            marker = ""
            if f.suffix in (".xml", ".bin"):
                ir_found = True
                marker = " ← IR"
            log.info(f"  {str(f.relative_to(output_path)):<45s}  {size_mb:>8.1f} MB{marker}")

    log.info(f"\n  Total export size: {total_size_mb:,.1f} MB")

    if not ir_found:
        log.error(
            "FAILED: No .xml/.bin IR files found. The export did not produce "
            "the required OpenVINO artifacts."
        )
        sys.exit(1)
    else:
        log.info("✓ OpenVINO IR files confirmed.\n")

    # Check if model fits within the memory budget
    if total_size_mb < 3500:
        log.info(
            f"✓ Model size ({total_size_mb:,.0f} MB) is within the "
            f"<3.5 GB VRAM/RAM inference budget."
        )
    else:
        log.warning(
            f"⚠ Model size ({total_size_mb:,.0f} MB) exceeds 3.5 GB target. "
            f"Consider increasing compression ratio or reducing group size."
        )


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(
        description="The Local Titan — Qwen 3.5-VL-4B → OpenVINO INT4 Export",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python model_convert.py\n"
            "  python model_convert.py --output ./my_model\n"
            "  python model_convert.py --skip-ram-check\n"
        ),
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for IR files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--skip-ram-check",
        action="store_true",
        help="Skip the pre-flight RAM check (NOT recommended on 16GB systems)",
    )

    args = parser.parse_args()

    log.info("=" * 54)
    log.info("  THE LOCAL TITAN — Model Conversion Pipeline")
    log.info(f"  Qwen 2-VL-2B-Instruct → OpenVINO IR ({WEIGHT_FORMAT})")
    log.info("=" * 54 + "\n")

    # Step 1: Pre-Flight
    if not args.skip_ram_check:
        preflight_ram_check()
    else:
        log.warning("RAM check SKIPPED by user flag. Proceed with caution.\n")

    # Step 2: Force garbage collection before heavy operation
    gc.collect()

    # Step 3: Export
    export_model_to_openvino_int4(args.output)

    # Step 4: Validate
    validate_output(args.output)

    log.info("\n🏁 Pipeline complete. Model is ready for VLMPipeline inference.")
    log.info(f"   Load with: ov_genai.VLMPipeline('{args.output}', 'CPU')\n")


if __name__ == "__main__":
    main()
