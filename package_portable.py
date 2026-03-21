import argparse
import os
import subprocess
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "dist"
DEFAULT_PACKAGE_NAME = "LabelMakr+_v051"
DEFAULT_PYTHON_EMBED_URL = "https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip"
PYDOMINO_INSTALL_URL = "git+https://github.com/DwangoMediaVillage/pydomino"
DEFAULT_PYDOMINO_MODEL_URL = "https://raw.githubusercontent.com/DwangoMediaVillage/pydomino/main/onnx_model/phoneme_transition_model.onnx"
PYDOMINO_MODEL_FILE_NAME = "phoneme_transition_model.onnx"
PYDOMINO_STATUS_FILE = "PYDOMINO_INSTALL_STATUS.txt"
PYDOMINO_RUNTIME_PACKAGES = ["numpy"]
RUNTIME_BOOTSTRAP_PACKAGES = ["setuptools<81", "packaging<25", "wheel"]
BUILD_BOOTSTRAP_PACKAGES = ["setuptools<81", "packaging<25", "wheel"]
COMMON_VCVARS64_PATHS = [
    Path("C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Auxiliary/Build/vcvars64.bat"),
    Path("C:/Program Files/Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat"),
    Path("C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat"),
    Path("C:/Program Files (x86)/Microsoft Visual Studio/2019/Professional/VC/Auxiliary/Build/vcvars64.bat"),
    Path("C:/Program Files (x86)/Microsoft Visual Studio/2019/BuildTools/VC/Auxiliary/Build/vcvars64.bat"),
]

REQUIRED_DIRS = [
    "gui",
    "runtime_a",
    "runtime_b",
    "shared",
]

PYTHON_RUNTIME_DIRS = [
    "gui",
    "runtime_a",
    "runtime_b",
]

REQUIRED_FILES = [
    "CHANGELOG.txt",
    "setup_guide.txt",
]

FALLBACK_DIR_COPIES = {
    "onnx_model": Path("runtime_b") / "onnx_model",
}

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

SANITIZED_PYTHON_ENV_KEYS = [
    "PYTHONHOME",
    "PYTHONPATH",
    "PYTHONEXECUTABLE",
    "PYTHONWEXECUTABLE",
    "PYTHON_EXECUTABLE",
    "PYTHONW_EXECUTABLE",
    "PYTHON_BIN_PATH",
    "PYTHON_LIB_PATH",
    "TCL_LIBRARY",
    "TK_LIBRARY",
]


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
        "--build-python",
        type=Path,
        default=None,
        help="Full Python executable used to build a pydomino wheel outside the packaged runtimes. Its major.minor must match the packaged Python.",
    )
    parser.add_argument(
        "--build-vcvars",
        type=Path,
        default=None,
        help="Optional path to vcvars64.bat used when building pydomino on Windows.",
    )
    parser.add_argument(
        "--pydomino-wheel",
        type=Path,
        default=None,
        help="Prebuilt pydomino wheel to install into the packaged Python instead of building from source.",
    )
    parser.add_argument(
        "--pydomino-model",
        type=Path,
        default=None,
        help="Optional local pydomino ONNX model file or directory containing it. If omitted, a local onnx_model folder is used when available, otherwise the default model URL is downloaded.",
    )
    parser.add_argument(
        "--pydomino-model-url",
        default=DEFAULT_PYDOMINO_MODEL_URL,
        help="Fallback URL used to download the default pydomino ONNX model when no local model file is found.",
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
    parser.add_argument(
        "--skip-preinstall-pydomino",
        action="store_true",
        help="Do not preinstall pydomino into the packaged portable Python.",
    )
    parser.add_argument(
        "--require-pydomino",
        action="store_true",
        help="Fail packaging if pydomino cannot be preinstalled.",
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


def run_command(command, cwd: Optional[Path] = None):
    subprocess.run(command, cwd=cwd, check=True)


def run_command_capture(command, cwd: Optional[Path] = None) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def build_sanitized_env(extra_env: Optional[dict[str, str]] = None) -> dict[str, str]:
    env = os.environ.copy()
    for key in SANITIZED_PYTHON_ENV_KEYS:
        env.pop(key, None)
    if extra_env:
        env.update(extra_env)
    return env


def write_status_file(package_root: Path, message: str):
    (package_root / PYDOMINO_STATUS_FILE).write_text(f"{message}\n", encoding="utf-8")


def get_python_executable(path: Path) -> Path:
    if path.is_dir():
        return path / "python.exe"
    return path


def get_python_version(python_exe: Path) -> tuple[int, int, int]:
    ensure_exists(python_exe, "Python executable")
    output = run_command_capture(
        [
            str(python_exe),
            "-c",
            "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')",
        ]
    )
    major_str, minor_str, micro_str = output.split(".")
    return int(major_str), int(minor_str), int(micro_str)


def get_python_build_root(python_exe: Path) -> Path:
    ensure_exists(python_exe, "Python executable")
    output = run_command_capture(
        [
            str(python_exe),
            "-c",
            "import sys; print(sys.base_prefix)",
        ]
    )
    return Path(output).resolve()


def ensure_directory(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def get_archive_path(output_root: Path, package_name: str) -> Path:
    return output_root / f"{package_name}.zip"


def has_matching_abi(version_a: tuple[int, int, int], version_b: tuple[int, int, int]) -> bool:
    return version_a[:2] == version_b[:2]


def find_vcvars64(preferred: Optional[Path]) -> Optional[Path]:
    if preferred is not None:
        resolved = preferred.resolve()
        ensure_exists(resolved, "vcvars64.bat")
        return resolved

    env_path = os.environ.get("VCVARS64_PATH")
    if env_path:
        candidate = Path(env_path).resolve()
        if candidate.exists():
            return candidate

    for candidate in COMMON_VCVARS64_PATHS:
        if candidate.exists():
            return candidate
    return None


def get_scripts_dir(python_exe: Path) -> Path:
    python_parent = python_exe.resolve().parent
    if python_parent.name.lower() == "scripts":
        return python_parent
    return python_parent / "Scripts"


def run_windows_batch_command(command, vcvars_path: Path, cwd: Optional[Path] = None, extra_path: Optional[Path] = None):
    command_line = subprocess.list2cmdline([str(part) for part in command])
    with tempfile.NamedTemporaryFile("w", suffix=".cmd", delete=False, encoding="utf-8") as script_file:
        script_path = Path(script_file.name)
        script_file.write("@echo off\n")
        script_file.write(f'call "{vcvars_path}"\n')
        if extra_path is not None:
            script_file.write(f'set "PATH={extra_path};%PATH%"\n')
        script_file.write(f"{command_line}\n")

    try:
        subprocess.run(
            ["cmd.exe", "/d", "/c", str(script_path)],
            cwd=cwd,
            check=True,
            env=build_sanitized_env(),
        )
    finally:
        script_path.unlink(missing_ok=True)


def copy_if_exists(source: Path, destination: Path) -> bool:
    if not source.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def copy_optional_tree(source: Path, destination: Path) -> bool:
    if not source.exists():
        return False
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    return True


def resolve_optional_file(preferred: Optional[Path], fallback_dir_name: str, file_name: str) -> Optional[Path]:
    candidates = []
    if preferred is not None:
        preferred = preferred.resolve()
        if preferred.is_dir():
            candidates.append(preferred / file_name)
        else:
            candidates.append(preferred)

    candidates.extend([
        PROJECT_ROOT / fallback_dir_name / file_name,
        PROJECT_ROOT / file_name,
    ])

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def populate_pydomino_model(package_root: Path, args: argparse.Namespace):
    local_model = resolve_optional_file(args.pydomino_model, "onnx_model", PYDOMINO_MODEL_FILE_NAME)
    destinations = [
        package_root / "python" / "onnx_model" / PYDOMINO_MODEL_FILE_NAME,
        package_root / "onnx_model" / PYDOMINO_MODEL_FILE_NAME,
    ]

    if local_model is not None:
        for destination in destinations:
            copy_file(local_model, destination)
        return

    for destination in destinations:
        download_file(args.pydomino_model_url, destination)


def copy_tkinter_runtime(source_python_root: Path, destination_python_dir: Path):
    source_root = source_python_root.resolve()
    destination_root = destination_python_dir.resolve()

    copied_any = False

    copied_any = copy_optional_tree(source_root / "Lib" / "tkinter", destination_root / "Lib" / "tkinter") or copied_any
    copied_any = copy_optional_tree(source_root / "tcl", destination_root / "tcl") or copied_any

    for file_name in ("_tkinter.pyd", "tcl86t.dll", "tk86t.dll", "zlib1.dll"):
        copied_any = copy_if_exists(source_root / "DLLs" / file_name, destination_root / "DLLs" / file_name) or copied_any
        copied_any = copy_if_exists(source_root / file_name, destination_root / "DLLs" / file_name) or copied_any
        copied_any = copy_if_exists(source_root / "DLLs" / file_name, destination_root / file_name) or copied_any
        copied_any = copy_if_exists(source_root / file_name, destination_root / file_name) or copied_any

    if not copied_any:
        raise FileNotFoundError(
            f"Could not find tkinter runtime files in Python installation: {source_root}"
        )


def write_portable_sitecustomize(python_dir: Path):
    sitecustomize_path = python_dir / "Lib" / "site-packages" / "sitecustomize.py"
    sitecustomize_path.write_text(
        """import ctypes
import os
from pathlib import Path

PORTABLE_DLL_DIR_HANDLES = []


def preload_portable_dll(*candidates):
    if os.name != \"nt\" or not hasattr(ctypes, \"WinDLL\"):
        return

    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            ctypes.WinDLL(str(candidate))
            return
        except OSError:
            continue


def bootstrap_portable_runtime():
    if os.name != \"nt\" or not hasattr(os, \"add_dll_directory\"):
        return

    python_root = Path(__file__).resolve().parents[2]
    if not python_root.exists():
        return

    for candidate in (
        python_root,
        python_root / \"DLLs\",
        python_root / \"tcl\",
        python_root / \"tcl\" / \"tcl8.6\",
        python_root / \"tcl\" / \"tk8.6\",
    ):
        if candidate.exists():
            PORTABLE_DLL_DIR_HANDLES.append(os.add_dll_directory(str(candidate)))

    os.environ.setdefault(\"TCL_LIBRARY\", str(python_root / \"tcl\" / \"tcl8.6\"))
    os.environ.setdefault(\"TK_LIBRARY\", str(python_root / \"tcl\" / \"tk8.6\"))

    preload_portable_dll(python_root / \"DLLs\" / \"zlib1.dll\", python_root / \"zlib1.dll\")
    preload_portable_dll(python_root / \"DLLs\" / \"tcl86t.dll\", python_root / \"tcl86t.dll\")
    preload_portable_dll(python_root / \"DLLs\" / \"tk86t.dll\", python_root / \"tk86t.dll\")
    preload_portable_dll(python_root / \"DLLs\" / \"_tkinter.pyd\", python_root / \"_tkinter.pyd\")


bootstrap_portable_runtime()
""",
        encoding="utf-8",
    )


def resolve_tkinter_source(args: argparse.Namespace) -> Optional[Path]:
    candidates: list[Path] = []

    if args.python_dir is not None:
        python_dir = args.python_dir.resolve()
        candidates.append(get_python_build_root(get_python_executable(python_dir)))

    if args.build_python is not None:
        candidates.append(get_python_build_root(get_python_executable(args.build_python.resolve())))

    candidates.append(get_python_build_root(Path(sys.executable).resolve()))

    seen = set()
    for candidate in candidates:
        candidate_key = str(candidate)
        if candidate_key in seen:
            continue
        seen.add(candidate_key)

        if (candidate / "Lib" / "tkinter").exists() and (candidate / "tcl").exists():
            return candidate
    return None


def configure_embedded_python(python_dir: Path):
    (python_dir / "Lib" / "site-packages").mkdir(parents=True, exist_ok=True)
    (python_dir / "Scripts").mkdir(parents=True, exist_ok=True)

    pth_files = list(python_dir.glob("python*._pth"))
    for pth_file in pth_files:
        zip_name = None
        existing_lines = pth_file.read_text(encoding="utf-8").splitlines()
        for line in existing_lines:
            stripped = line.strip()
            if stripped.endswith(".zip"):
                zip_name = stripped
                break

        if zip_name is None:
            zip_name = "python314.zip"

        pth_file.write_text(
            "\n".join([
                zip_name,
                ".",
                "DLLs",
                "Lib",
                "Lib\\site-packages",
                "import site",
                "",
            ]),
            encoding="utf-8",
        )


def populate_python_dir(package_root: Path, args: argparse.Namespace):
    source_dir = resolve_source(args.python_dir, "python")
    destination = package_root / "python"

    if source_dir is not None:
        ensure_exists(source_dir, "python source")
        if not source_dir.is_dir():
            raise NotADirectoryError(f"python source is not a directory: {source_dir}")
        copy_tree(source_dir, destination)
        configure_embedded_python(destination)
    else:
        with tempfile.TemporaryDirectory(prefix="labelmakr_python_") as temp_dir:
            archive_path = Path(temp_dir) / "python_embed.zip"
            download_file(args.python_embed_url, archive_path)
            with zipfile.ZipFile(archive_path, "r") as archive:
                archive.extractall(destination)
        configure_embedded_python(destination)

    tkinter_source = resolve_tkinter_source(args)
    if tkinter_source is not None:
        copy_tkinter_runtime(tkinter_source, destination)

    write_portable_sitecustomize(destination)


def resolve_get_pip_script(package_root: Path) -> Path:
    candidates = [
        package_root / "get-pip.py",
        package_root.parent / "shared" / "get-pip.py",
        PROJECT_ROOT / "shared" / "get-pip.py",
        PROJECT_ROOT / "get-pip.py",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"get-pip.py could not be found for runtime: {package_root}")


def has_python_build_support(python_dir: Path) -> bool:
    include_dirs = [python_dir / "Include", python_dir / "include"]
    libs_dir = python_dir / "libs"
    has_headers = any(include_dir.is_dir() for include_dir in include_dirs)
    has_import_lib = libs_dir.is_dir() and any(libs_dir.glob("python*.lib"))
    return has_headers and has_import_lib


def bootstrap_pip_only(python_exe: Path, package_root: Path):
    run_command([str(python_exe), str(resolve_get_pip_script(package_root))], cwd=package_root)
    run_command(
        [
            str(python_exe),
            "-m",
            "pip",
            "install",
            *RUNTIME_BOOTSTRAP_PACKAGES,
        ],
        cwd=package_root,
    )


def build_pydomino_wheel(build_python_exe: Path, vcvars_path: Path) -> Path:
    build_python_exe = get_python_executable(build_python_exe).resolve()
    scripts_dir = get_scripts_dir(build_python_exe)
    build_python_root = get_python_build_root(build_python_exe)

    subprocess.run(
        [
            str(build_python_exe),
            "-m",
            "pip",
            "install",
            *BUILD_BOOTSTRAP_PACKAGES,
            "cmake",
        ],
        check=True,
        env=build_sanitized_env(
            {
                "PYTHON_EXECUTABLE": str(build_python_exe),
                "PYTHONHOME": str(build_python_root),
            }
        ),
    )

    with tempfile.TemporaryDirectory(prefix="labelmakr_pydomino_wheel_") as temp_dir:
        wheel_dir = Path(temp_dir) / "wheelhouse"
        wheel_dir.mkdir(parents=True, exist_ok=True)
        run_windows_batch_command(
            [
                str(build_python_exe),
                "-m",
                "pip",
                "wheel",
                "--no-build-isolation",
                "--no-deps",
                "--wheel-dir",
                str(wheel_dir),
                PYDOMINO_INSTALL_URL,
            ],
            vcvars_path=vcvars_path,
            extra_path=scripts_dir,
        )
        wheels = sorted(wheel_dir.glob("pydomino*.whl"))
        if not wheels:
            raise FileNotFoundError("pydomino wheel build finished without producing a wheel file.")

        persisted_wheel = PROJECT_ROOT / "dist" / "wheelhouse" / wheels[0].name
        persisted_wheel.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(wheels[0], persisted_wheel)
        return persisted_wheel


def install_wheel_into_python(python_exe: Path, wheel_path: Path):
    ensure_exists(wheel_path, "pydomino wheel")
    if PYDOMINO_RUNTIME_PACKAGES:
        run_command(
            [
                str(python_exe),
                "-m",
                "pip",
                "install",
                *PYDOMINO_RUNTIME_PACKAGES,
            ]
        )
    run_command(
        [
            str(python_exe),
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            str(wheel_path),
        ]
    )


def resolve_build_python(preferred: Optional[Path], target_version: tuple[int, int, int]) -> Optional[Path]:
    candidates = []
    if preferred is not None:
        candidates.append(get_python_executable(preferred).resolve())
    else:
        local_venv_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
        if local_venv_python.exists():
            candidates.append(local_venv_python.resolve())
        candidates.append(Path(sys.executable).resolve())

    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            candidate_version = get_python_version(candidate)
            candidate_root = get_python_build_root(candidate)
        except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
            continue

        if not has_matching_abi(candidate_version, target_version):
            continue
        if not has_python_build_support(candidate_root):
            continue
        return candidate
    return None


def preinstall_pydomino(runtime_b_root: Path, args: argparse.Namespace):
    python_dir = runtime_b_root / "python"
    python_exe = python_dir / "python.exe"
    if not python_exe.exists():
        raise FileNotFoundError(f"Portable python executable not found: {python_exe}")

    bootstrap_pip_only(python_exe, runtime_b_root)

    if args.pydomino_wheel is not None:
        install_wheel_into_python(python_exe, args.pydomino_wheel.resolve())
        return

    target_version = get_python_version(python_exe)
    build_python = resolve_build_python(args.build_python, target_version)
    if build_python is None:
        raise RuntimeError(
            "A compatible full Python environment for building the pydomino wheel could not be found. "
            "Use --build-python (or package_portable.bat with .venv) so the wheel is built outside runtime_b, "
            "or provide --pydomino-wheel explicitly."
        )

    vcvars_path = find_vcvars64(args.build_vcvars)
    if vcvars_path is None:
        raise RuntimeError(
            "Building pydomino on Windows requires vcvars64.bat, but none was found. "
            "Install Visual Studio Build Tools or pass --build-vcvars explicitly."
        )

    wheel_path = build_pydomino_wheel(build_python, vcvars_path)
    install_wheel_into_python(python_exe, wheel_path)


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

    for source_name, target_relative_path in FALLBACK_DIR_COPIES.items():
        source_dir = PROJECT_ROOT / source_name
        destination_dir = package_root / target_relative_path
        if source_dir.exists() and not destination_dir.exists():
            copy_tree(source_dir, destination_dir)

    for file_name in REQUIRED_FILES:
        source_file = PROJECT_ROOT / file_name
        ensure_exists(source_file, "Required file")
        copy_file(source_file, package_root / file_name)

    for file_name in OPTIONAL_FILES:
        source_file = PROJECT_ROOT / file_name
        if source_file.exists():
            copy_file(source_file, package_root / file_name)

    for runtime_name in PYTHON_RUNTIME_DIRS:
        populate_python_dir(package_root / runtime_name, args)

    optional_sources = {
        "corpus": resolve_source(args.corpus_dir, "corpus"),
    }

    for target_name, source_dir in optional_sources.items():
        target_root = package_root / "shared"
        if source_dir is None:
            if target_name == "corpus":
                (target_root / target_name).mkdir(parents=True, exist_ok=True)
            continue

        ensure_exists(source_dir, f"{target_name} source")
        if not source_dir.is_dir():
            raise NotADirectoryError(f"{target_name} source is not a directory: {source_dir}")
        copy_tree(source_dir, target_root / target_name)


def main():
    args = parse_args()

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    package_root = output_root / args.package_name
    archive_path = get_archive_path(output_root, args.package_name)

    if args.zip and archive_path.exists():
        archive_path.unlink()

    if package_root.exists():
        if not args.force:
            raise FileExistsError(
                f"Output folder already exists: {package_root}. Re-run with --force to overwrite."
            )
        shutil.rmtree(package_root)

    package_root.mkdir(parents=True, exist_ok=True)
    package_workspace(package_root, args)

    pydomino_status = None
    if not args.skip_preinstall_pydomino:
        try:
            runtime_b_root = package_root / "runtime_b"
            preinstall_pydomino(runtime_b_root, args)
            populate_pydomino_model(runtime_b_root, args)
            pydomino_status = "pydomino was preinstalled successfully."
        except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as exc:
            pydomino_status = (
                "pydomino was not preinstalled.\n"
                f"Reason: {exc}\n"
                "Portable packaging completed without pydomino support. "
                "To bundle it while keeping python-embed, rebuild with either a matching full Python via "
                "--build-python or a prebuilt wheel via --pydomino-wheel."
            )
            write_status_file(package_root, pydomino_status)
            print(f"Warning: {pydomino_status}")
            if args.require_pydomino:
                raise RuntimeError("pydomino preinstall failed and --require-pydomino was set.") from exc
    elif args.skip_preinstall_pydomino:
        pydomino_status = "pydomino preinstall was skipped by request."
        write_status_file(package_root, pydomino_status)

    if args.zip:
        shutil.make_archive(str(archive_path.with_suffix('')), "zip", output_root, args.package_name)
        print(f"Created zip archive: {archive_path}")

    print(f"Packaged LabelMakr to: {package_root}")
    print("Runtime requirements are not preinstalled. Run shared/setup_CPU.bat or shared/setup_GPU.bat inside the packaged folder.")
    if pydomino_status is not None:
        print(pydomino_status)


if __name__ == "__main__":
    main()