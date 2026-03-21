import argparse
import os
import sys
from pathlib import Path

RUNTIME_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUNTIME_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.nemo_metadata import DEFAULT_NEMO_MODEL


def configure_runtime_environment(working_dir: str):
    working_root = Path(working_dir).resolve()
    shared_root = (working_root / "shared").resolve()
    corpus_dir = (working_root / "corpus").resolve()
    g2p_exe = (working_root / "runtime_a" / "g2p-jp" / "japanese_g2p.exe").resolve()
    ffmpeg_exe = (shared_root / "ffmpeg.exe").resolve()

    os.environ["LABELMAKR_WORKSPACE_ROOT"] = str(working_root)
    os.environ["LABELMAKR_SHARED_ROOT"] = str(shared_root)
    os.environ["LABELMAKR_CORPUS_DIR"] = str(corpus_dir)
    os.environ["LABELMAKR_G2P_JP_EXE"] = str(g2p_exe)
    os.environ["LABELMAKR_FFMPEG_EXE"] = str(ffmpeg_exe)


def parse_args():
    parser = argparse.ArgumentParser(description="runtime_a worker for Whisper/NeMo transcription")
    parser.add_argument("--backend", choices=["whisper", "nemo"], required=True)
    parser.add_argument("--lang", required=True)
    parser.add_argument("--working-dir", default=str(PROJECT_ROOT))
    parser.add_argument("--whisper-model", default="medium")
    parser.add_argument("--nemo-model", default=DEFAULT_NEMO_MODEL)
    parser.add_argument("--force-cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    configure_runtime_environment(args.working_dir)
    os.chdir(args.working_dir)

    if args.backend == "nemo":
        import runtime_a.nemo_func as nemo_func

        transcriber = nemo_func.NeMoTranscriber(
            args.lang,
            model_id=args.nemo_model,
            force_cpu=args.force_cpu,
        )
        transcriber.run_transcription(args.lang)
        return

    import runtime_a.whisper_func as whisper_func

    transcriber = whisper_func.Transcriber(args.lang, args.whisper_model)
    transcriber.run_transcription(args.lang)


if __name__ == "__main__":
    main()
