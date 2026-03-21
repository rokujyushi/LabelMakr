# Manual installation 🧰

Current LabelMakr+ uses a 3-runtime layout.

- `gui/`: GUI only
- `runtime_a/`: Whisper / NeMo transcription
- `runtime_b/`: SOFA / pydomino alignment
- `shared/`: common assets, FFmpeg, corpus, setup scripts

For Windows, the recommended method is still running [shared/setup_CPU.bat](../shared/setup_CPU.bat) or [shared/setup_GPU.bat](../shared/setup_GPU.bat).

If you need to install manually, use the steps below.

## Requirements

- Python 3.12 recommended
- Windows is the primary target for the current portable workflow
- Separate environments or embedded runtimes for `gui`, `runtime_a`, and `runtime_b`

## 1. Prepare the workspace

Clone the repository and keep the folder structure intact.

```bash
git clone https://github.com/rokujyushi/LabelMakr.git
cd LabelMakr
```

## 2. Prepare Python for each runtime

Create one environment for each runtime, or place an embedded Python under each of these folders:

- `gui/python`
- `runtime_a/python`
- `runtime_b/python`

If you prefer venv instead, the runtime code also detects:

- `gui/.venv`
- `runtime_a/.venv`
- `runtime_b/.venv`

## 3. Install GUI dependencies

```bash
python -m pip install -r gui/requirements.txt
```

## 4. Install transcription runtime dependencies

CPU example:

```bash
python -m pip install torch torchvision torchaudio torchcodec
python -m pip install -r runtime_a/requirements.txt
```

GPU example:

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python -m pip install torchcodec
python -m pip install -r runtime_a/requirements.txt
```

## 5. Install alignment runtime dependencies

CPU example:

```bash
python -m pip install torch torchvision torchaudio
python -m pip install -r runtime_b/requirements.txt
```

GPU example:

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r runtime_b/requirements.txt
```

Install `pydomino` separately if you need that aligner.

## 6. Install shared assets

Run:

```bash
python shared/install_assets.py
```

This installs assets into the current runtime layout:

- shared FFmpeg -> `shared/ffmpeg.exe`, `shared/ffprobe.exe`
- shared models -> `shared/models/`
- Japanese g2p -> `runtime_a/g2p-jp/`
- SOFA -> `runtime_b/SOFA/`
- pydomino ONNX -> `runtime_b/onnx_model/`

## 7. Add your corpus

Place WAV files under:

```text
shared/corpus/
```

Nested speaker folders are fine.

## 8. Start the app

Run:

```bash
gui/run.bat
```

## Notes

- NeMo may download model weights on first use.
- `runtime_b` expects SOFA assets and pydomino ONNX to exist before alignment.
- Root `requirements.txt` is now a compatibility aggregate; prefer the runtime-specific files.
