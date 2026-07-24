from pathlib import Path
import zipfile

import pytest

from upload_kaggle import validate_archive


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
