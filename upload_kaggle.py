"""Upload morse.zip as a new version of a private Kaggle Dataset."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path


HANDLE_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+$")


def validate_archive(path: Path) -> None:
    """Ensure the bundle has the expected root and excludes large artifacts."""

    if not path.is_file():
        raise FileNotFoundError(f"Archive not found: {path}")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
    if "morse/" not in names:
        raise ValueError("Archive must contain a top-level morse/ directory")
    forbidden = tuple(
        name for name in names if name.startswith(("morse/audio/", "morse/models/"))
    )
    if forbidden:
        raise ValueError("Archive unexpectedly contains audio/ or models/")


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
        help="Source archive to upload",
    )
    parser.add_argument(
        "--notes",
        default="Updated Morse training source",
        help="Kaggle dataset version notes",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Validate and upload the source bundle, then print the notebook cell."""

    args = build_argument_parser().parse_args(argv)
    upload_archive(args.handle, args.archive.resolve(), args.notes)
    print(f"uploaded=https://www.kaggle.com/datasets/{args.handle}")
    print("\nKaggle notebook bootstrap cell:\n")
    print(
        "import kagglehub, os, pathlib, shutil, zipfile\n"
        f'dataset_dir = pathlib.Path(kagglehub.dataset_download("{args.handle}", '
        "force_download=True))\n"
        'project_dir = pathlib.Path("/kaggle/working/morse")\n'
        'os.chdir("/kaggle/working")\n'
        "if project_dir.exists():\n"
        "    shutil.rmtree(project_dir)\n"
        'source_dir = dataset_dir / "morse"\n'
        "if source_dir.is_dir():\n"
        "    shutil.copytree(source_dir, project_dir)\n"
        "else:\n"
        '    archives = list(dataset_dir.rglob("morse.zip"))\n'
        "    if not archives:\n"
        '        raise FileNotFoundError(f"No morse/ or morse.zip under {dataset_dir}")\n'
        "    with zipfile.ZipFile(archives[0]) as archive:\n"
        '        archive.extractall("/kaggle/working")\n'
        "print(project_dir)"
    )


if __name__ == "__main__":
    main()
