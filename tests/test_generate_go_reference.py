import json
from pathlib import Path

import numpy as np
import pytest
import torch

from morse_timing.audio_dataset import Stage1DatasetConfig
from morse_timing.audio_model import AudioModelConfig, MorseAudioCTCModel
from morse_timing.audio_train import OverfitMetrics, save_checkpoint
from morse_timing.generate_go_reference import generate_reference


def create_checkpoint(path: Path) -> None:
    model = MorseAudioCTCModel(
        AudioModelConfig(
            projection_size=8,
            hidden_size=7,
            dense_layers=2,
            num_lstm_layers=1,
            sequence_model="lstm",
        )
    )
    save_checkpoint(
        path,
        model,
        Stage1DatasetConfig(),
        [],
        1,
        OverfitMetrics(0.1, 0.0, 0.0, 1.0, 1.0, "E", "E"),
    )


def test_generate_reference_writes_repeatable_raw_tensors(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    create_checkpoint(checkpoint)
    first = tmp_path / "first"
    second = tmp_path / "second"

    generate_reference(checkpoint, first, 0.05)
    generate_reference(checkpoint, second, 0.05)

    metadata = json.loads((first / "metadata.json").read_text())
    waveform = np.fromfile(first / "waveform.f32", dtype="<f4")
    logits = np.fromfile(first / "expected_logits.f32", dtype="<f4")

    assert metadata["sample_rate"] == 8_000
    assert metadata["input"]["waveform"]["shape"] == [400]
    assert metadata["input"]["waveform"]["element_count"] == waveform.size
    assert metadata["expected_outputs"]["logits"]["shape"][1] == 5
    assert metadata["expected_outputs"]["logits"]["element_count"] == logits.size
    assert (first / "waveform.f32").read_bytes() == (
        second / "waveform.f32"
    ).read_bytes()
    assert (first / "expected_logits.f32").read_bytes() == (
        second / "expected_logits.f32"
    ).read_bytes()


def test_generate_reference_rejects_non_positive_duration(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Duration must be positive"):
        generate_reference(tmp_path / "unused.pt", tmp_path / "output", 0.0)


def test_reference_generator_uses_decoder_waveform_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "model.pt"
    create_checkpoint(checkpoint)
    calls: list[tuple[tuple[int, ...], int, int]] = []

    def predict(
        self: object,
        waveform: torch.Tensor,
        sample_rate: int,
        chunk_frames: int,
    ) -> torch.Tensor:
        calls.append((tuple(waveform.shape), sample_rate, chunk_frames))
        return torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)

    monkeypatch.setattr(
        "morse_timing.audio_inference.MorseAudioDecoder.predict_waveform_logits",
        predict,
    )

    generate_reference(checkpoint, tmp_path / "output", 0.05)

    assert calls == [((400,), 8_000, 25)]
    assert np.array_equal(
        np.fromfile(tmp_path / "output" / "expected_logits.f32", dtype="<f4"),
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
    )
