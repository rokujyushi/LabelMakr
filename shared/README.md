# Shared Runtime Assets

This folder is the shared root for files used by the GUI, `runtime_a`, and `runtime_b`.

Primary contents:
- `assets/`
- `models/`
- `corpus/`
- `ffmpeg.exe`
- `ffprobe.exe`
- `get-pip.py`
- `install_assets.py`
- `setup_CPU.bat`
- `setup_GPU.bat`
- `set_env.bat`

Runtime-specific assets are intentionally kept outside `shared/`:
- `runtime_a/g2p-jp/`
- `runtime_b/SOFA/`
- `runtime_b/onnx_model/`

Typical flow:
- Run `shared/setup_CPU.bat` or `shared/setup_GPU.bat`
- Place input audio in `shared/corpus/`
- Start the app from `gui/run.bat`
