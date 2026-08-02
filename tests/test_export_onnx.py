from pathlib import Path

import torch

from morse_timing.audio_dataset import Stage1DatasetConfig
from morse_timing.audio_model import AudioModelConfig, MorseAudioCTCModel
from morse_timing.audio_train import OverfitMetrics, save_checkpoint
from morse_timing.export_onnx import (
    StreamingOnnxWrapper,
    load_wrapper,
    verify_wrapper_matches_model,
)


def test_streaming_onnx_wrapper_has_explicit_reusable_state() -> None:
    config = AudioModelConfig(
        projection_size=8,
        hidden_size=7,
        dense_layers=2,
        num_lstm_layers=1,
        sequence_model="lstm",
    )
    model = MorseAudioCTCModel(config).eval()
    wrapper = StreamingOnnxWrapper(model)
    features = torch.randn(1, 9, 65)
    hidden = torch.zeros(1, 1, 7)
    cell = torch.zeros_like(hidden)

    logits, next_hidden, next_cell = verify_wrapper_matches_model(
        wrapper, features, hidden, cell
    )

    assert logits.shape == (1, 9, 5)
    assert next_hidden.shape == hidden.shape
    assert next_cell.shape == cell.shape


def test_export_loader_restores_checkpoint_weights(tmp_path: Path) -> None:
    config = AudioModelConfig(
        projection_size=8,
        hidden_size=7,
        dense_layers=2,
        sequence_model="lstm",
    )
    model = MorseAudioCTCModel(config)
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(
        checkpoint,
        model,
        Stage1DatasetConfig(),
        [],
        1,
        OverfitMetrics(0.1, 0.0, 0.0, 1.0, 1.0, "E", "E"),
    )

    wrapper, loaded = load_wrapper(checkpoint)

    assert loaded["model_config"]["sequence_model"] == "lstm"
    for expected, restored in zip(
        model.parameters(), wrapper.model.parameters(), strict=True
    ):
        assert torch.equal(expected, restored)
