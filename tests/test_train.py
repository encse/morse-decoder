from dataclasses import asdict
from pathlib import Path

import torch
import json
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from morse_timing.audio_dataset import CleanAudioMorseDataset, Stage1DatasetConfig
from morse_timing.audio_dataset import collate_audio_sequences
from morse_timing.audio_model import AudioModelConfig, MorseAudioCTCModel
from morse_timing.audio_train import OverfitMetrics
from morse_timing.audio_train import evaluate_overfit_dataset, train_epoch
from morse_timing.train import (
    automatic_output_path,
    cache_dataset,
    initialize_model_from_checkpoint,
    load_training_checkpoint,
    save_training_checkpoint,
)


def test_cache_dataset_materializes_each_spectrogram_once() -> None:
    generated = CleanAudioMorseDataset(2, texts=["E", "T"])

    cached = cache_dataset(generated, "test", num_workers=0)

    assert len(cached) == 2
    assert [sample.text for sample in cached] == ["E", "T"]
    assert all(sample.spectrogram.ndim == 2 for sample in cached)


def test_general_training_checkpoint_restores_complete_state(tmp_path: Path) -> None:
    config = AudioModelConfig(
        first_conv_channels=2,
        second_conv_channels=4,
        projection_size=8,
        hidden_size=8,
        num_gru_layers=1,
    )
    model = MorseAudioCTCModel(config)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer)
    checkpoint = tmp_path / "stage1.pt"
    metrics = OverfitMetrics(0.5, 0.2, 0.25, 0.3, 0.4, "E", "T")
    save_training_checkpoint(
        checkpoint,
        model,
        optimizer,
        scheduler,
        Stage1DatasetConfig(),
        7,
        0.2,
        0.25,
        metrics,
        {"train_samples": 10, "validation_samples": 2, "seed": 42},
    )

    restored_model = MorseAudioCTCModel(config)
    restored_optimizer = AdamW(restored_model.parameters(), lr=9e-3)
    restored_scheduler = ReduceLROnPlateau(restored_optimizer)
    start_epoch, best_error, best_character_error = load_training_checkpoint(
        checkpoint,
        restored_model,
        restored_optimizer,
        restored_scheduler,
        torch.device("cpu"),
    )

    assert start_epoch == 8
    assert best_error == 0.2
    assert best_character_error == 0.25
    assert restored_optimizer.param_groups[0]["lr"] == 1e-3
    for expected, restored in zip(
        model.parameters(), restored_model.parameters(), strict=True
    ):
        assert torch.equal(expected, restored)
    metadata = json.loads(checkpoint.with_suffix(".json").read_text())
    assert metadata["dataset_config"]["wpm"] == 20.0
    assert metadata["model_config"]["sequence_model"] == "lstm"
    assert "model_state" not in metadata


def test_automatic_output_name_describes_effective_ranges() -> None:
    path = automatic_output_path(
        AudioModelConfig(sequence_model="tcn"),
        Stage1DatasetConfig(
            min_wpm=10.0,
            max_wpm=40.0,
            min_frequency_hz=400.0,
            max_frequency_hz=1_200.0,
        ),
    )

    assert path == Path(
        "models/audio-stage1-tcn-wpm10-40-freq400-1200hz-"
        "jitter0pct-noise-power0.pt"
    )


def test_init_from_restores_only_model_weights(tmp_path: Path) -> None:
    config = AudioModelConfig(
        first_conv_channels=2,
        second_conv_channels=4,
        projection_size=8,
        hidden_size=8,
        num_gru_layers=1,
    )
    source = MorseAudioCTCModel(config)
    checkpoint = tmp_path / "weights.pt"
    torch.save(
        {
            "model_config": asdict(config),
            "model_state": source.state_dict(),
        },
        checkpoint,
    )
    target = MorseAudioCTCModel(config)

    initialize_model_from_checkpoint(checkpoint, target, torch.device("cpu"))

    for expected, restored in zip(source.parameters(), target.parameters(), strict=True):
        assert torch.equal(expected, restored)


def test_init_from_old_checkpoint_ignores_removed_frequency_head(
    tmp_path: Path,
) -> None:
    config = AudioModelConfig(projection_size=8, hidden_size=8, dense_layers=2)
    source = MorseAudioCTCModel(config)
    old_state = dict(source.state_dict())
    old_state["frequency_head.weight"] = torch.zeros(1, config.hidden_size)
    old_state["frequency_head.bias"] = torch.zeros(1)
    checkpoint = tmp_path / "old-weights.pt"
    torch.save(
        {"model_config": asdict(config), "model_state": old_state}, checkpoint
    )
    target = MorseAudioCTCModel(config)

    initialize_model_from_checkpoint(checkpoint, target, torch.device("cpu"))

    assert not hasattr(target, "frequency_head")
    assert torch.equal(target.classifier.weight, source.classifier.weight)


def test_ctc_objective_trains_and_evaluates_token_sequences() -> None:
    samples = CleanAudioMorseDataset(2, texts=["E", "T"])
    loader = torch.utils.data.DataLoader(
        [samples[0], samples[1]],
        batch_size=2,
        collate_fn=collate_audio_sequences,
    )
    model = MorseAudioCTCModel(
        AudioModelConfig(
            projection_size=8,
            hidden_size=8,
            dense_layers=2,
        )
    )
    optimizer = AdamW(model.parameters(), lr=1e-3)

    loss = train_epoch(
        model,
        loader,
        optimizer,
        torch.device("cpu"),
        gradient_clip=5.0,
        log_interval=0,
    )
    metrics = evaluate_overfit_dataset(
        model,
        loader,
        torch.device("cpu"),
    )

    assert loss > 0.0
    assert torch.isfinite(torch.tensor(metrics.loss))


def test_resume_rejects_a_legacy_frame_objective(tmp_path: Path) -> None:
    config = AudioModelConfig(projection_size=8, hidden_size=8, dense_layers=2)
    model = MorseAudioCTCModel(config)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer)
    checkpoint = tmp_path / "frame.pt"
    save_training_checkpoint(
        checkpoint,
        model,
        optimizer,
        scheduler,
        Stage1DatasetConfig(),
        1,
        0.5,
        0.5,
        OverfitMetrics(0.5, 0.5, 0.5, 0.0, 0.0, "E", "T"),
        {},
    )
    values = torch.load(checkpoint, map_location="cpu", weights_only=True)
    values["training_objective"] = "frame_events"
    torch.save(values, checkpoint)

    with __import__("pytest").raises(ValueError, match="CTC"):
        load_training_checkpoint(
            checkpoint,
            model,
            optimizer,
            scheduler,
            torch.device("cpu"),
        )
