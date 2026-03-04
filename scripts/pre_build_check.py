import os
import sys
from pathlib import Path
import logging

# Configure minimal logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("pre-build")

def check_model_directory(model_dir="./qwen_ov_int4"):
    """Verify that the INT4 model files exist."""
    print(f"[1/3] Checking OpenVINO Model: {model_dir}")
    path = Path(model_dir)
    if not path.exists():
        print(f"  [X] MISSING: Directory {model_dir} not found.")
        return False
    
    xml_files = list(path.glob("*.xml"))
    bin_files = list(path.glob("*.bin"))
    
    if xml_files and bin_files:
        print(f"  [V] FOUND: {len(xml_files)} IR xml(s) and {len(bin_files)} bin(s).")
        return True
    else:
        print(f"  [X] INVALID: IR files (.xml/.bin) missing in {model_dir}")
        return False

def check_reflex_assets():
    """Check if reflex export has been run."""
    print("[2/3] Checking Reflex Static Assets: .web/_static")
    path = Path(".web/_static")
    if path.exists() and any(path.iterdir()):
        print("  [V] FOUND: Web assets detected. 'reflex export' has been run.")
        return True
    else:
        print("  [X] MISSING: .web/_static is empty or missing. Run 'reflex export' first.")
        return False

def check_openvino_hardware():
    """Verify OpenVINO hardware detection."""
    print("[3/3] Checking OpenVINO Hardware Discovery")
    try:
        import openvino as ov
        core = ov.Core()
        devices = core.available_devices
        print(f"  [V] ONLINE: OpenVINO Core initialized. Devices: {devices}")
        
        has_accel = any("NPU" in d or "GPU" in d for d in devices)
        if has_accel:
            print("  [V] ACCEL: Intel iGPU/NPU detected for high-performance inference.")
        else:
            print("  [!] WARNING: Only CPU detected. UI will run slower on this machine.")
        return True
    except Exception as e:
        print(f"  [X] FAILED: OpenVINO error: {e}")
        return False

def perform_audit():
    """Audit requirements vs installed packages."""
    print("\n--- Environment Audit ---")
    try:
        import pkg_resources
        requirements_file = Path("requirements.txt")
        if not requirements_file.exists():
            print("  [X] MISSING: requirements.txt not found.")
            return

        with open(requirements_file, "r") as f:
            lines = f.readlines()
            reqs = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith("#"):
                    # Handle comments on same line
                    line = line.split("#")[0].strip()
                    # Handle version specifiers and extras
                    name = line.split(">=")[0].split("==")[0].split("<=")[0].split("~=")[0].split("[")[0].strip().lower()
                    if name:
                        reqs.append(name)

        # Dev ignore list
        dev_only = ["pytest", "black", "flake8", "isort", "mypy", "twine", "wheel"]
        
        installed = {pkg.key for pkg in pkg_resources.working_set}
        found_dev = [d for d in dev_only if d in installed]

        if found_dev:
            print(f"  [!] BLOAT ALERT: Dev-only libs found: {found_dev}")
            print("  (Tip: Uninstall these before final build to reduce .exe size)")
        else:
            print("  [V] CLEAN: No obvious dev-only libraries detected.")
            
        # Check core reqs
        missing = [r for r in reqs if r not in installed]
        if missing:
            print(f"  [X] MISSING REQS: {missing}")
        else:
            print("  [V] SYNCED: All core requirements are installed.")

    except Exception as e:
        print(f"  [!] AUDIT FAILED: {e}")

def run_diagnostic():
    print("="*60)
    print(" THE LOCAL TITAN: PRE-BUILD INTEGRITY AUDIT")
    print("="*60)
    
    m = check_model_directory()
    r = check_reflex_assets()
    o = check_openvino_hardware()
    
    perform_audit()
    
    print("\n" + "="*60)
    if all([m, r, o]):
        print(" STATUS: READY TO BUILD  [V]")
        print(" All critical paths are verified for production packaging.")
    else:
        print(" STATUS: NOT READY  [X]")
        print(" Please address the missing components above before building.")
    print("="*60)

if __name__ == "__main__":
    run_diagnostic()
