"""Build and upload morse.zip as a new version of a private Kaggle Dataset."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path


HANDLE_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+$")
PROJECT_FILES = (
    Path(".gitignore"),
    Path("AGENTS.md"),
    Path("README.md"),
    Path("curriculum-plan.json"),
    Path("generate_analysis.py"),
    Path("kaggle_train.py"),
    Path("pyproject.toml"),
    Path("reference.wav"),
    Path("upload_kaggle.py"),
)
PROJECT_SOURCE_PATTERNS = (
    "src/morse_timing/*.py",
    "tests/*.py",
)


def build_project_archive(
    project_directory: Path,
    archive_path: Path,
    upload_model_dir: Path | None = None,
) -> None:
    """Create a fresh source bundle at the requested archive path."""

    project_directory = project_directory.resolve()
    archive_path = archive_path.resolve()
    model_files: list[Path] = []
    if upload_model_dir is not None:
        upload_model_dir = upload_model_dir.resolve()
        if not upload_model_dir.is_dir():
            raise NotADirectoryError(
                f"Upload model directory not found: {upload_model_dir}"
            )
        model_files = sorted(
            path for path in upload_model_dir.iterdir() if path.is_file()
        )
        if not model_files:
            raise FileNotFoundError(
                f"Upload model directory contains no files: {upload_model_dir}"
            )
    selected_files: list[Path] = []
    for relative_path in PROJECT_FILES:
        source = project_directory / relative_path
        if not source.is_file():
            raise FileNotFoundError(f"Required bundle file not found: {source}")
        selected_files.append(source)
    for pattern in PROJECT_SOURCE_PATTERNS:
        matches = sorted(project_directory.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No bundle files match: {pattern}")
        selected_files.extend(path for path in matches if path.is_file())

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = tempfile.NamedTemporaryFile(
        prefix=f".{archive_path.name}.",
        suffix=".tmp",
        dir=archive_path.parent,
        delete=False,
    )
    temporary_path = Path(temporary_file.name)
    temporary_file.close()
    try:
        with zipfile.ZipFile(
            temporary_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            archive.writestr("morse/", "")
            for source in selected_files:
                relative_path = source.relative_to(project_directory)
                archive.write(source, Path("morse") / relative_path)
            if upload_model_dir is not None:
                archive.writestr("models/", "")
                for source in model_files:
                    archive.write(source, Path("models") / source.name)
        validate_archive(temporary_path)
        temporary_path.replace(archive_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def validate_archive(path: Path) -> None:
    """Ensure the bundle has the expected source root and artifact layout."""

    if not path.is_file():
        raise FileNotFoundError(f"Archive not found: {path}")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
    if "morse/" not in names:
        raise ValueError("Archive must contain a top-level morse/ directory")
    forbidden = tuple(
        name
        for name in names
        if name.startswith(("morse/audio/", "morse/models/"))
    )
    if forbidden:
        raise ValueError(
            "Archive unexpectedly contains artifacts inside morse/; "
            "models must use the top-level models/ directory"
        )


def upload_archive(
    handle: str,
    archive_path: Path,
    version_notes: str,
) -> None:
    """Upload only the source archive to a private Kaggle Dataset."""

    if not HANDLE_PATTERN.fullmatch(handle):
        raise ValueError("Dataset handle must have the form username/dataset-slug")
    validate_archive(archive_path)
    try:
        import kagglehub
    except ImportError as error:
        raise RuntimeError(
            "Install the pinned project upload dependency: kagglehub==0.4.1"
        ) from error
    credential_files = (
        Path.home() / ".kaggle" / "access_token",
        Path.home() / ".kaggle" / "kaggle.json",
    )
    if not os.environ.get("KAGGLE_API_TOKEN") and not any(
        path.is_file() for path in credential_files
    ):
        kagglehub.login()
    with tempfile.TemporaryDirectory(prefix="morse-kaggle-upload-") as directory:
        staged_archive = Path(directory) / "morse.zip"
        shutil.copy2(archive_path, staged_archive)
        kagglehub.dataset_upload(handle, directory, version_notes=version_notes)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the private dataset upload command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("handle", help="Kaggle handle: username/dataset-slug")
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("morse.zip"),
        help="Archive path to rebuild and upload",
    )
    parser.add_argument(
        "--notes",
        default="Updated Morse training source",
        help="Kaggle dataset version notes",
    )
    parser.add_argument(
        "--upload-model-dir",
        type=Path,
        help=(
            "Optional directory whose files are added under the archive's "
            "top-level models/ directory"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Build and upload the source bundle, then print the notebook cell."""

    args = build_argument_parser().parse_args(argv)
    archive_path = args.archive.resolve()
    project_directory = Path(__file__).resolve().parent
    build_project_archive(
        project_directory,
        archive_path,
        args.upload_model_dir,
    )
    print(f"built={archive_path}")
    upload_archive(args.handle, archive_path, args.notes)
    print(f"uploaded=https://www.kaggle.com/datasets/{args.handle}")


if __name__ == "__main__":
    main()
