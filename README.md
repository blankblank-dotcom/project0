# 🪐 The Local Titan: Offline Document Intelligence

**The Local Titan** is a professional-grade, privacy-first Document AI dashboard designed for high-performance inference on consumer hardware. Optimized for Windows 11 and Intel hardware, it leverages the **Qwen 3.5-VL-4B** model quantized to **INT4** to provide state-of-the-art document understanding with zero data exfiltration.

![The Local Titan Header](https://source.unsplash.com/featured/?artificial-intelligence,document)

---

## ⚡ Core Pillars

### 1. Hardened Privacy Shield
- **Burned-in Redaction**: IR-level pixel replacement for PII (not just a visual box).
- **Metadata Scrubbing**: Automated stripping of EXIF, ICC profiles, and hidden document headers.
- **AES-GCM Encryption**: Optional value-level encryption for local SQLite and ChromaDB files.

### 2. Multi-Modal Intelligence
- **Universal Parser**: Native support for **PDF**, **DOCX**, and **PPTX** with automated image sequencing.
- **Spatial Grounding**: Interactive "Visual Q&A"—click any region of a document to ask targeted questions.
- **Local RAG**: Persistent semantic search across your entire document history.

### 3. Professional Performance
- **Hardware Acceleration**: Deeply integrated with **OpenVINO** for NPU, iGPU (Iris Xe), and CPU processing.
- **RAM Safety Monitor**: Intelligent auto-pause/resume logic to protect systems with 16GB RAM.
- **Calibrated ETA**: Real-world "Time Remaining" estimation based on on-device benchmarks.

---

## 🚀 Quick Start

### 1. Prerequisites
- **OS**: Windows 10/11 (64-bit)
- **Hardware**: Intel Core i5/i7 (12th Gen+) with Iris Xe iGPU or NPU 
- **Python**: 3.10+

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/local-titan/local-titan.git
cd local-titan

# Install core dependencies
pip install -r requirements.txt

# Sync hardware drivers
python scripts/pre_build_check.py
```

### 3. Launching the System
```bash
# Start the Reflex dashboard
reflex run
```

---

## 🛠 Project Architecture

| Component | Responsibility | Technology |
|:---|:---|:---|
| **Pipeline** | INT4 VLM Inference | OpenVINO, Qwen 3.5-VL |
| **Storage** | Semantic & Metadata DB | ChromaDB, SQLite |
| **Dashboard** | Reactive UI | Reflex (Next.js/FastAPI) |
| **Packaging** | Windows Deployment | PyInstaller, Inno Setup |

---

## 📊 Benchmarking & Verification

The Local Titan includes a built-in benchmarking suite to calibrate the UI for your specific hardware:
```bash
python scripts/benchmark.py
```
This tool measures **Image Encoding Latency**, **TTFT**, and **Tokens Per Second** to provide accurate progress bars in the dashboard.

---

## 🛡 Security Policy
As an offline-first tool, **The Local Titan** never connects to the internet during processing. Your documents never leave your local machine. Encryption is handled via the `cryptography` library using PBKDF2 for key derivation.

---

## 📜 License
Unlicensed — Private Enterprise Build.
