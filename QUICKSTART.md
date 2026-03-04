# 🚀 The Local Titan: From-Scratch Quick Start Guide

Follow these steps to set up and launch your offline AI Document Intelligence system on a fresh Windows machine.

---

## 🏗 1. Environment Setup

### Prerequisites
- **Python**: 3.11+ (Recommended)
- **Node.js**: v20 or higher  
  *(Install via PowerShell: `winget install OpenJS.NodeJS.LTS`)*
- **Git**: [Download here](https://git-scm.com/downloads)
- **Hardware**: Intel Core i5/i7 (12th Gen+) for optimal performance.

### Installation
Open PowerShell or CMD and run:

```powershell
# 1. Clone the project
git clone https://github.com/local-titan/local-titan.git
cd local-titan

# 2. Create a virtual environment (Recommended)
python -m venv venv
.\venv\Scripts\activate

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Install OpenVINO optimized extensions
pip install "optimum-intel[nncf]" openvino-genai
```

---

## 🔍 2. System Integrity Audit

Before running the AI, verify that your environment and hardware drivers are correctly configured:

```powershell
python scripts/pre_build_check.py
```

> [!TIP]
> If OpenVINO fails to detect your **GPU** or **NPU**, ensure you have the latest [Intel Graphics Drivers](https://www.intel.com/content/www/us/en/support/detect.html) installed.

---

## 🧪 3. Hardware Calibration

Every Intel chip performs differently. Run the benchmark to calibrate the UI progress bars and ETA estimations:

```powershell
python scripts/benchmark.py
```
*This will create a `benchmark_results.json` file which the dashboard uses to provide accurate "Time Remaining" feedback.*

---

## 💻 4. Launching the Dashboard

Start the reactive Reflex interface:

```powershell
reflex run
```

Once initialized, open your browser to **`http://localhost:3000`**.

---

## 🛡 5. Using Privacy Features

1.  **Burned-in Redaction**: Toggle "Auto-Redact" and click "Detect PII". The system physically replaces pixels with black rectangles.
2.  **Storage Encryption**: 
    - Click the **Lock Icon** in the bottom-right footer.
    - Set a Master Password.
    - Your local SQLite and ChromaDB files are now encrypted using **AES-GCM**.
3.  **Metadata Scrubbing**: Simply click "Export". All exported Excel/CSV files are automatically stripped of EXIF and ICC metadata.

---

## 📦 6. Packaging as a Windows Executable

To distribute **The Local Titan** as a standalone `.exe` and a professional installer:

### 1. Install Packaging Tools
```powershell
pip install pyinstaller
```
*Note: [Inno Setup 6](https://jrsoftware.org/isdl.php) must be installed to generate the `.exe` installer.*

### 2. Export Frontend Assets
Produce the static production build of the Reflex UI:
```powershell
reflex export
```

### 3. Run the Build Script
This script handles PyInstaller configuration, OpenVINO DLL bundling, and Inno Setup compilation:
```powershell
python build/build_exe.py
```

### 4. Locate Your Build
- **Standalone Folder**: `dist/TheLocalTitan/`
- **Windows Installer**: `build/Output/LocalTitan_Setup.exe`

---

## 🆘 Troubleshooting
- **Node.js not detected?** Reflex requires Node.js v20+ to compile the frontend assets. Download and install it from [nodejs.org](https://nodejs.org/).
- **Out of Memory?** The system will automatically pause if RAM exceeds 85%. Close other apps (like Chrome) to resume processing.
- **Model Missing?** Ensure the `qwen_ov_int4` folder contains the `.xml` and `.bin` IR files.
