from pathlib import Path
import zipfile

import pytest

import upload_kaggle
from upload_kaggle import build_project_archive, validate_archive


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


def test_validate_archive_accepts_top_level_model_files(tmp_path: Path) -> None:
    archive_path = tmp_path / "morse.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("morse/", "")
        archive.writestr("morse/pyproject.toml", "")
        archive.writestr("models/", "")
        archive.writestr("models/checkpoint.pt", "checkpoint")

    validate_archive(archive_path)


def test_build_project_archive_adds_upload_model_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_directory = tmp_path / "project"
    project_directory.mkdir()
    (project_directory / "pyproject.toml").write_text("", encoding="utf-8")
    model_directory = tmp_path / "model-files"
    model_directory.mkdir()
    (model_directory / "checkpoint.pt").write_bytes(b"checkpoint")
    ignored_directory = model_directory / "old-run"
    ignored_directory.mkdir()
    (ignored_directory / "old.pt").write_bytes(b"old")
    archive_path = tmp_path / "morse.zip"
    monkeypatch.setattr(
        upload_kaggle,
        "PROJECT_FILES",
        (Path("pyproject.toml"),),
    )
    monkeypatch.setattr(upload_kaggle, "PROJECT_SOURCE_PATTERNS", ())

    build_project_archive(
        project_directory,
        archive_path,
        upload_model_dir=model_directory,
    )

    with zipfile.ZipFile(archive_path) as archive:
        assert archive.read("models/checkpoint.pt") == b"checkpoint"
        assert "models/old-run/old.pt" not in archive.namelist()


def test_build_project_archive_rejects_empty_upload_model_directory(
    tmp_path: Path,
) -> None:
    model_directory = tmp_path / "empty-models"
    model_directory.mkdir()

    with pytest.raises(FileNotFoundError, match="contains no files"):
        build_project_archive(
            tmp_path,
            tmp_path / "morse.zip",
            upload_model_dir=model_directory,
        )
