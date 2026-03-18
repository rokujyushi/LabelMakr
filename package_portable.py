import argparse
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "dist"
DEFAULT_PACKAGE_NAME = "LabelMakr_v031"
DEFAULT_PYTHON_EMBED_URL = "https://www.python.org/ftp/python/3.14.3/python-3.14.3-embed-amd64.zip"

REQUIRED_DIRS = [
    "assets",
    "models",
    "strings",
]

REQUIRED_FILES = [
    "CHANGELOG.txt",
    "ffmpeg.exe",
    "get-pip.py",
    "install_assets.py",
    "labbu.py",
    "labbu_func.py",
    "labelmakr.py",
    "pydomino_func.py",
    "requirements.txt",
    "run.bat",
    "setup_CPU.bat",
    "setup_GPU.bat",
    "setup_guide.txt",
    "set_env.bat",
    "sofa_func.py",
    "whisper_func.py",
]

OPTIONAL_FILES = [
    "README.md",
    "readme.txt",
    "LICENSE",
]

EXCLUDED_NAMES = {
    ".git",
    ".github",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".venv",
    "dist",
    "build",
    "plan-labelMakrPortableSetup.prompt.md",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package the current LabelMakr environment into a distributable folder."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the packaged folder will be created.",
    )
    parser.add_argument(
        "--package-name",
        default=DEFAULT_PACKAGE_NAME,
        help="Name of the packaged folder.",
    )
    parser.add_argument(
        "--python-dir",
        type=Path,
        default=None,
        help="Existing portable Python directory to include as /python. If omitted, the embed package URL is used.",
    )
    parser.add_argument(
        "--python-embed-url",
        default=DEFAULT_PYTHON_EMBED_URL,
        help="Python embeddable zip URL to download into /python when --python-dir is omitted.",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=None,
        help="Optional corpus directory to include as /corpus.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output folder.",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Also create a zip archive next to the output folder.",
    )
    return parser.parse_args()


def ensure_exists(path: Path, kind: str):
    if not path.exists():
        raise FileNotFoundError(f"{kind} not found: {path}")


def copy_tree(source: Path, destination: Path):
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(*EXCLUDED_NAMES))


def copy_file(source: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def download_file(url: str, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, open(destination, "wb") as output_file:
        shutil.copyfileobj(response, output_file)


def populate_python_dir(package_root: Path, args: argparse.Namespace):
    source_dir = resolve_source(args.python_dir, "python")
    destination = package_root / "python"

    if source_dir is not None:
        ensure_exists(source_dir, "python source")
        if not source_dir.is_dir():
            raise NotADirectoryError(f"python source is not a directory: {source_dir}")
        copy_tree(source_dir, destination)
        return

    with tempfile.TemporaryDirectory(prefix="labelmakr_python_") as temp_dir:
        archive_path = Path(temp_dir) / "python_embed.zip"
        download_file(args.python_embed_url, archive_path)
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(destination)


def resolve_source(preferred: Optional[Path], fallback_name: str) -> Optional[Path]:
    if preferred is not None:
        return preferred.resolve()
    fallback = PROJECT_ROOT / fallback_name
    if fallback.exists():
        return fallback
    return None


def package_workspace(package_root: Path, args: argparse.Namespace):
    for directory_name in REQUIRED_DIRS:
        source_dir = PROJECT_ROOT / directory_name
        ensure_exists(source_dir, "Required directory")
        copy_tree(source_dir, package_root / directory_name)

    for file_name in REQUIRED_FILES:
        source_file = PROJECT_ROOT / file_name
        ensure_exists(source_file, "Required file")
        copy_file(source_file, package_root / file_name)

    for file_name in OPTIONAL_FILES:
        source_file = PROJECT_ROOT / file_name
        if source_file.exists():
            copy_file(source_file, package_root / file_name)

    populate_python_dir(package_root, args)

    optional_sources = {
        "corpus": resolve_source(args.corpus_dir, "corpus"),
    }

    for target_name, source_dir in optional_sources.items():
        if source_dir is None:
            if target_name == "corpus":
                (package_root / target_name).mkdir(parents=True, exist_ok=True)
            continue

        ensure_exists(source_dir, f"{target_name} source")
        if not source_dir.is_dir():
            raise NotADirectoryError(f"{target_name} source is not a directory: {source_dir}")
        copy_tree(source_dir, package_root / target_name)


def main():
    args = parse_args()

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    package_root = output_root / args.package_name

    if package_root.exists():
        if not args.force:
            raise FileExistsError(
                f"Output folder already exists: {package_root}. Re-run with --force to overwrite."
            )
        shutil.rmtree(package_root)

    package_root.mkdir(parents=True, exist_ok=True)
    package_workspace(package_root, args)

    if args.zip:
        archive_path = output_root / args.package_name
        if (output_root / f"{args.package_name}.zip").exists():
            (output_root / f"{args.package_name}.zip").unlink()
        shutil.make_archive(str(archive_path), "zip", output_root, args.package_name)
        print(f"Created zip archive: {archive_path}.zip")

    print(f"Packaged LabelMakr to: {package_root}")


if __name__ == "__main__":
    main()