from pathlib import Path
import zipfile

import pytest

from upload_kaggle import (
    PROJECT_FILES,
    build_project_archive,
    validate_archive,
)


def test_build_project_archive_replaces_bundle_with_current_sources(
    tmp_path: Path,
) -> None:
    project_directory = tmp_path / "project"
    for relative_path in PROJECT_FILES:
        source = project_directory / relative_path
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(
            b"reference" if relative_path.name == "reference.wav" else b"current"
        )
    package_file = project_directory / "src/morse_timing/module.py"
    package_file.parent.mkdir(parents=True)
    package_file.write_text("PACKAGE = True\n", encoding="utf-8")
    test_file = project_directory / "tests/test_module.py"
    test_file.parent.mkdir()
    test_file.write_text("def test_module(): pass\n", encoding="utf-8")
    model_file = project_directory / "models/model.pt"
    model_file.parent.mkdir()
    model_file.write_bytes(b"model")
    archive_path = tmp_path / "output" / "morse.zip"
    archive_path.parent.mkdir()
    archive_path.write_bytes(b"old archive")

    build_project_archive(project_directory, archive_path)

    validate_archive(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        assert archive.read("morse/reference.wav") == b"reference"
    assert "morse/src/morse_timing/module.py" in names
    assert "morse/tests/test_module.py" in names
    assert not any(name.startswith("morse/models/") for name in names)


def test_validate_archive_accepts_morse_root_without_artifacts(tmp_path: Path) -> None:
    archive_path = tmp_path / "morse.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("morse/", "")
        archive.writestr("morse/pyproject.toml", "")

    validate_archive(archive_path)


def test_validate_archive_rejects_model_files(tmp_path: Path) -> None:
    archive_path = tmp_path / "morse.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("morse/", "")
        archive.writestr("morse/models/model.pt", "")

    with pytest.raises(ValueError, match="models"):
        validate_archive(archive_path)
