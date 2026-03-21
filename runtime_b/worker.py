import argparse
import logging
import os
import sys
from pathlib import Path

RUNTIME_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUNTIME_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


logger = logging.getLogger(__name__)
logging.basicConfig(format="| %(levelname)s | %(message)s | %(asctime)s |", datefmt="%H:%M:%S")
logger.setLevel(logging.INFO)


def configure_runtime_environment(working_dir: str):
    working_root = Path(working_dir).resolve()
    shared_root = (working_root / "shared").resolve()
    corpus_dir = (working_root / "corpus").resolve()
    sofa_root = (working_root / "runtime_b" / "SOFA").resolve()
    ffmpeg_exe = (shared_root / "ffmpeg.exe").resolve()
    ffprobe_exe = (shared_root / "ffprobe.exe").resolve()

    os.environ["LABELMAKR_WORKSPACE_ROOT"] = str(working_root)
    os.environ["LABELMAKR_SHARED_ROOT"] = str(shared_root)
    os.environ["LABELMAKR_CORPUS_DIR"] = str(corpus_dir)
    os.environ["LABELMAKR_SOFA_ROOT"] = str(sofa_root)
    os.environ["LABELMAKR_FFMPEG_EXE"] = str(ffmpeg_exe)
    os.environ["LABELMAKR_FFPROBE_EXE"] = str(ffprobe_exe)


def parse_args():
    parser = argparse.ArgumentParser(description="runtime_b worker for SOFA/pydomino alignment and label fixing")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sofa_parser = subparsers.add_parser("sofa")
    sofa_parser.add_argument("--working-dir", default=str(PROJECT_ROOT))
    sofa_parser.add_argument("--ckpt", required=True)
    sofa_parser.add_argument("--dictionary", required=True)
    sofa_parser.add_argument("--op-format", default="htk")
    sofa_parser.add_argument("--lang", default="EN")
    sofa_parser.add_argument("--matmul", action="store_true")
    sofa_parser.add_argument("--g2p", action="store_true")
    sofa_parser.add_argument("--g2p-model")
    sofa_parser.add_argument("--g2p-cfg")

    pydomino_parser = subparsers.add_parser("pydomino")
    pydomino_parser.add_argument("--working-dir", default=str(PROJECT_ROOT))
    pydomino_parser.add_argument("--onnx-path", required=True)
    pydomino_parser.add_argument("--output-format", default="htk")
    pydomino_parser.add_argument("--min-aligned-timeframe", type=int, default=3)
    pydomino_parser.add_argument("--corpus-dir", default="corpus")

    labbu_parser = subparsers.add_parser("labbu")
    labbu_parser.add_argument("--working-dir", default=str(PROJECT_ROOT))
    labbu_parser.add_argument("--lang", default="EN")
    labbu_parser.add_argument("--corpus-dir", default="corpus")
    labbu_parser.add_argument("--dxer", action="store_true")
    labbu_parser.add_argument("--uhr-merge", action="store_true")
    labbu_parser.add_argument("--merge-h", action="store_true")
    labbu_parser.add_argument("--merge-dupes", action="store_true")

    return parser.parse_args()


def run_label_fixes(lang: str, corpus_dir: str, dxer: bool, uhr_merge: bool, merge_h: bool, merge_dupes: bool):
    import runtime_b.labbu_func as labbu_module

    corpus_root = Path(corpus_dir).resolve()
    if not corpus_root.exists():
        logger.warning(f'Corpus directory could not be found: {corpus_root}')
        return

    fixer = labbu_module.labbu_func(lang=lang)
    label_files = sorted(corpus_root.glob('*/labels/*.lab'))

    if not label_files:
        logger.warning(f'No label files found under: {corpus_root}')
        return

    for file_path in label_files:
        fixer.load(file_path)

        if dxer:
            fixer.dxer()
        if uhr_merge:
            fixer.fix_uh_r()
        if merge_h:
            fixer.merge_short_hh()
        if merge_dupes:
            fixer.merge_dupes()

        fixer.save(file_path)

    logger.info(f'Finished fixing labels for {len(label_files)} file(s).')


def main():
    args = parse_args()
    configure_runtime_environment(args.working_dir)
    os.chdir(args.working_dir)

    if args.command == "sofa":
        import runtime_b.sofa_func as sofa_func

        sofa_func.infer_sofa(
            ckpt=args.ckpt,
            dictionary=args.dictionary,
            op_format=args.op_format,
            matmul_bool=args.matmul,
            lang=args.lang,
            g2p_bool=args.g2p,
            g2p_model=args.g2p_model,
            g2p_cfg=args.g2p_cfg,
        )
        return

    if args.command == "labbu":
        run_label_fixes(
            lang=args.lang,
            corpus_dir=args.corpus_dir,
            dxer=args.dxer,
            uhr_merge=args.uhr_merge,
            merge_h=args.merge_h,
            merge_dupes=args.merge_dupes,
        )
        return

    import runtime_b.pydomino_func as pydomino_func

    pydomino_func.infer_pydomino(
        onnx_path=args.onnx_path,
        output_format=args.output_format,
        min_aligned_timeframe=args.min_aligned_timeframe,
        corpus_dir=args.corpus_dir,
    )


if __name__ == "__main__":
    main()
