# 🎙️ VoiceAssistant: Real-time STT with IBM Granite Speech

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-orange?logo=apache&logoColor=white)](LICENSE)
[![Model](https://img.shields.io/badge/Model-Granite%20Speech%204.1%202B-green?logo=ibm&logoColor=white)](https://huggingface.co/ibm-granite/granite-speech-4.1-2b)
[![Framework](https://img.shields.io/badge/Framework-Transformers-yellow?logo=huggingface&logoColor=white)](https://github.com/huggingface/transformers)

A high-performance, real-time speech-to-text utility powered by **IBM Granite Speech 4.1 2B**. Featuring a beautiful terminal UI with live RMS monitoring and scrollable transcripts.

---

## ✨ Features

- **Real-time Transcription**: Low-latency ASR using a sliding window approach.
- **Multilingual Support**: English, French, German, Spanish, Portuguese, and Japanese.
- **Task Specificity**: Support for `transcribe`, `transcribe-punct` (with punctuation), and `translate`.
- **Professional TUI**: Built with `Rich`, featuring:
  - Live RMS level meter.
  - Model and device status dashboard.
  - History-aware, scrollable transcript.
- **High-speed Asset Acquisition**: Custom Rust-based downloader for the 4.6GB model.
- **Modern Stack**: Built with `uv`, `torch`, and `transformers`.

---

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.10 or higher.
- [uv](https://github.com/astral-sh/uv) (recommended for package management).
- CUDA-compatible GPU (strongly recommended for real-time performance).

### 2. Installation

Clone the repository and sync dependencies:

```bash
git clone https://github.com/rzafiamy/voiceassistant.git
cd voiceassistant
uv sync
```

### 3. Fast Model Download

The model is approximately 4.6GB. Use the optimized download script to saturate your bandwidth:

```bash
# Optional: Provide HF_TOKEN for faster downloads if needed
./download_model.sh [YOUR_HF_TOKEN]
```

### 4. Run the Assistant

Start the real-time transcription:

```bash
uv run voiceassistant
```

---

## 🛠️ Configuration & Usage

You can customize the behavior via CLI arguments:

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--task` | `transcribe-punct` | Task: `transcribe`, `transcribe-punct`, or `translate` |
| `--lang` | `English` | Target language for translation |
| `--device` | `auto` | Execution device: `auto`, `cuda`, or `cpu` |
| `--window` | `5.0` | Audio window size in seconds |
| `--step` | `2.0` | Inference step (overlap) in seconds |
| `--input-device` | `None` | Specific PyAudio device index |

**Example: Translate French to English with high-speed inference:**
```bash
uv run voiceassistant --task translate --lang English --device cuda
```

---

## 🏗️ Project Structure

```text
.
├── LICENSE             # Apache 2.0 License
├── README.md           # This file
├── pyproject.toml      # Project configuration and dependencies
├── download_model.sh   # High-speed model downloader
├── src/
│   └── voiceassistant/
│       ├── __init__.py
│       └── main.py     # Main application logic
└── models/             # Local model storage (git-ignored)
```

---

## 📜 License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **IBM Research** for the incredible Granite Speech 4.1 2B model.
- **Hugging Face** for the `transformers` library.
- **Astral** for the `uv` package manager.

---
<p align="center">Made with ❤️ for the AI community.</p>