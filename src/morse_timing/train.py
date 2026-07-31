"""Train and validate the CTC Morse audio model."""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy  # Load the Conda OpenMP runtime before PyTorch on macOS.
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from morse_timing.audio_dataset import (
    AudioBatch,
    AudioSequenceSample,
    CleanAudioMorseDataset,
    Stage1DatasetConfig,
    collate_audio_sequences,
    restore_stage1_dataset_config,
)
from morse_timing.audio_model import (
    AudioModelConfig,
    MorseAudioCTCModel,
)
from morse_timing.audio_train import (
    OverfitMetrics,
    TONE_ACTIVITY_LOSS_WEIGHT,
    evaluate_overfit_dataset,
    select_device,
    set_seed,
    train_epoch,
)


def create_loader(
    dataset: list[AudioSequenceSample],
    batch_size: int,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
) -> DataLoader[AudioBatch]:
    """Create a deterministic DataLoader over precomputed features."""

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=pin_memory,
        collate_fn=collate_audio_sequences,
        generator=torch.Generator().manual_seed(seed),
    )


def cache_dataset(
    dataset: CleanAudioMorseDataset,
    label: str,
    num_workers: int,
) -> list[AudioSequenceSample]:
    """Generate every waveform and spectrogram once before training starts."""

    loader = DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=False,
    )
    cached: list[AudioSequenceSample] = []
    report_every = max(1, len(dataset) // 10)
    print(f"caching_{label}={len(dataset)}", flush=True)
    for index, sample in enumerate(loader, start=1):
        cached.append(sample)
        if index % report_every == 0 or index == len(dataset):
            print(f"  cached_{label}={index}/{len(dataset)}", flush=True)
    return cached


def save_training_checkpoint(
    path: Path,
    model: MorseAudioCTCModel,
    optimizer: AdamW,
    scheduler: ReduceLROnPlateau,
    dataset_config: Stage1DatasetConfig,
    epoch: int,
    best_token_error_rate: float,
    best_character_error_rate: float,
    metrics: OverfitMetrics,
    experiment: dict[str, int | float | bool],
) -> None:
    """Save model, optimizer, scheduler, preprocessing, and validation state."""

    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
            "format_version": 1,
            "purpose": "curriculum",
            "training_objective": "ctc",
            "model_config": asdict(model.config),
            "model_state": model.state_dict(),
            "dataset_config": asdict(dataset_config),
            "texts": [],
            "epoch": epoch,
            "metrics": asdict(metrics),
            "best_token_error_rate": best_token_error_rate,
            "best_character_error_rate": best_character_error_rate,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "experiment": experiment,
        }
    torch.save(checkpoint, path)
    metadata = {
        key: value
        for key, value in checkpoint.items()
        if key not in {"model_state", "optimizer_state", "scheduler_state"}
    }
    metadata["checkpoint"] = str(path)
    path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_experiment_configuration(
    path: Path | None,
) -> tuple[AudioModelConfig, Stage1DatasetConfig]:
    """Load inherited model and data configuration or return project defaults."""

    if path is None:
        return AudioModelConfig(), Stage1DatasetConfig()
    metadata_path = path.with_suffix(".json")
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return (
            AudioModelConfig(**metadata["model_config"]),
            restore_stage1_dataset_config(metadata["dataset_config"]),
        )
    checkpoint: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    return (
        AudioModelConfig(**checkpoint["model_config"]),
        restore_stage1_dataset_config(checkpoint["dataset_config"]),
    )


def automatic_output_path(
    model_config: AudioModelConfig,
    dataset_config: Stage1DatasetConfig,
) -> Path:
    """Build a readable filename from the effective experiment configuration."""

    def number(value: float) -> str:
        return f"{value:g}".replace(".", "p")

    if dataset_config.min_wpm is None:
        wpm = f"wpm{number(dataset_config.wpm)}"
    else:
        wpm = f"wpm{number(dataset_config.min_wpm)}-{number(dataset_config.max_wpm)}"
    if dataset_config.min_frequency_hz is None:
        frequency = f"freq{number(dataset_config.audio.frequency_hz)}hz"
    else:
        frequency = (
            f"freq{number(dataset_config.min_frequency_hz)}-"
            f"{number(dataset_config.max_frequency_hz)}hz"
        )
    jitter = f"jitter{number(dataset_config.timing_jitter * 100.0)}pct"
    noise = f"noise-power{number(dataset_config.noise_power)}"
    if dataset_config.noise_percent > 0.0:
        noise += f"-noise{number(dataset_config.noise_percent)}pct"
    return Path("models") / (
        f"audio-stage1-{model_config.sequence_model}-{wpm}-{frequency}-{jitter}-{noise}.pt"
    )


def load_training_checkpoint(
    path: Path,
    model: MorseAudioCTCModel,
    optimizer: AdamW,
    scheduler: ReduceLROnPlateau,
    device: torch.device,
) -> tuple[int, float, float]:
    """Restore a compatible curriculum training checkpoint."""

    checkpoint: dict[str, Any] = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )
    if checkpoint.get("purpose") != "curriculum":
        raise ValueError("Resume checkpoint is not a curriculum checkpoint")
    if checkpoint.get("training_objective") != "ctc":
        raise ValueError("Only CTC training checkpoints can be resumed")
    checkpoint_model_config = AudioModelConfig(**checkpoint["model_config"])
    if asdict(checkpoint_model_config) != asdict(model.config):
        raise ValueError("Resume checkpoint model configuration does not match CLI options")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    metrics = checkpoint.get("metrics", {})
    best_character_error_rate = checkpoint.get(
        "best_character_error_rate",
        metrics.get("character_error_rate", float("inf")),
    )
    return (
        int(checkpoint["epoch"]) + 1,
        float(checkpoint["best_token_error_rate"]),
        float(best_character_error_rate),
    )


def initialize_model_from_checkpoint(
    path: Path,
    model: MorseAudioCTCModel,
    device: torch.device,
) -> None:
    """Load compatible model weights without restoring training state."""

    checkpoint: dict[str, Any] = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )
    if "model_config" not in checkpoint or "model_state" not in checkpoint:
        raise ValueError("Initialization checkpoint does not contain a model")
    checkpoint_model_config = AudioModelConfig(**checkpoint["model_config"])
    if asdict(checkpoint_model_config) != asdict(model.config):
        raise ValueError(
            "Initialization checkpoint model configuration does not match CLI options"
        )
    model.load_state_dict(checkpoint["model_state"])


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the curriculum-stage training command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--train-samples", type=int, default=6_000)
    parser.add_argument("--validation-samples", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-5)
    parser.add_argument("--scheduler-patience", type=int, default=2)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--early-stopping", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--wpm", type=float)
    parser.add_argument("--min-wpm", type=float)
    parser.add_argument("--max-wpm", type=float)
    parser.add_argument("--min-frequency", type=float)
    parser.add_argument("--max-frequency", type=float)
    parser.add_argument("--frequency", type=float)
    parser.add_argument("--timing-jitter", type=float)
    parser.add_argument("--min-intra-character-gap-units", type=float)
    parser.add_argument("--max-intra-character-gap-units", type=float)
    parser.add_argument("--min-character-gap-units", type=float)
    parser.add_argument("--max-character-gap-units", type=float)
    parser.add_argument("--min-word-gap-units", type=float)
    parser.add_argument("--max-word-gap-units", type=float)
    parser.add_argument("--character-gap-extreme-probability", type=float)
    parser.add_argument("--character-gap-extreme-width-units", type=float)
    parser.add_argument("--noise-percent", type=float)
    parser.add_argument("--noise-power", type=float)
    parser.add_argument("--min-amplitude-percent", type=float)
    parser.add_argument("--max-amplitude-percent", type=float)
    parser.add_argument("--fade-depth-percent", type=float)
    parser.add_argument("--min-fade-frequency", type=float)
    parser.add_argument("--max-fade-frequency", type=float)
    parser.add_argument("--min-rise-fall-ms", type=float)
    parser.add_argument("--max-rise-fall-ms", type=float)
    parser.add_argument("--min-characters", type=int)
    parser.add_argument("--max-characters", type=int)
    parser.add_argument("--space-probability", type=float)
    parser.add_argument("--word-boundary-sample-probability", type=float)
    parser.add_argument(
        "--extended-space-probability",
        "--doubled-space-probability",
        dest="extended_space_probability",
        type=float,
    )
    parser.add_argument("--min-extended-space-multiplier", type=int)
    parser.add_argument("--max-extended-space-multiplier", type=int)
    parser.add_argument("--noise-only-probability", type=float)
    parser.add_argument(
        "--input-filter",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Apply a sampled receiver filter to every generated input",
    )
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--projection-size", type=int)
    parser.add_argument("--gru-layers", type=int)
    parser.add_argument("--lstm-layers", type=int)
    parser.add_argument("--dense-layers", type=int)
    parser.add_argument("--sequence-model", choices=("gru", "lstm", "tcn"))
    parser.add_argument("--tcn-layers", type=int)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--profile-batches", type=int, default=0)
    parser.add_argument(
        "--regenerate-every",
        type=int,
        default=0,
        help="Regenerate the cached training set every N epochs; zero disables it",
    )
    parser.add_argument(
        "--perfect-epochs",
        type=int,
        default=1,
        help="Consecutive perfect validation epochs required before stopping",
    )
    parser.add_argument(
        "--target-exact-text",
        type=float,
        help="Stop after reaching this exact-text accuracy",
    )
    parser.add_argument(
        "--target-epochs",
        type=int,
        default=2,
        help="Consecutive target-accuracy epochs required before stopping",
    )
    parser.add_argument("--leading-silence", type=float)
    parser.add_argument("--trailing-silence", type=float)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--last-output", type=Path)
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--resume", type=Path)
    checkpoint_group.add_argument("--init-from", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Train one curriculum range and select the best validation score."""

    args = build_argument_parser().parse_args(argv)
    positive_counts = (
        args.epochs,
        args.train_samples,
        args.validation_samples,
        args.batch_size,
    )
    if any(value <= 0 for value in positive_counts):
        raise ValueError("Epoch, sample, and batch counts must be positive")
    if (
        args.num_workers < 0
        or args.log_interval < 0
        or args.profile_batches < 0
        or args.regenerate_every < 0
    ):
        raise ValueError("Worker, log interval, and profile counts cannot be negative")
    if args.perfect_epochs <= 0:
        raise ValueError("Perfect epoch count must be positive")
    if args.target_epochs <= 0:
        raise ValueError("Target epoch count must be positive")
    if (
        args.target_exact_text is not None
        and not 0.0 <= args.target_exact_text <= 1.0
    ):
        raise ValueError("Target exact-text accuracy must be between zero and one")
    set_seed(args.seed)
    device = select_device(args.device)
    inherited_model, inherited_dataset = load_experiment_configuration(
        args.resume or args.init_from
    )
    model_overrides = {
        "hidden_size": args.hidden_size,
        "projection_size": args.projection_size,
        "num_gru_layers": args.gru_layers,
        "num_lstm_layers": args.lstm_layers,
        "dense_layers": args.dense_layers,
        "sequence_model": args.sequence_model,
        "tcn_layers": args.tcn_layers,
    }
    model_config = replace(
        inherited_model,
        **{key: value for key, value in model_overrides.items() if value is not None},
    )
    dataset_config = inherited_dataset
    if args.wpm is not None:
        dataset_config = replace(dataset_config, wpm=args.wpm, min_wpm=None, max_wpm=None)
    elif args.min_wpm is not None or args.max_wpm is not None:
        dataset_config = replace(
            dataset_config, min_wpm=args.min_wpm, max_wpm=args.max_wpm
        )
    if args.frequency is not None:
        dataset_config = replace(
            dataset_config,
            min_frequency_hz=None,
            max_frequency_hz=None,
            audio=replace(dataset_config.audio, frequency_hz=args.frequency),
        )
    elif args.min_frequency is not None or args.max_frequency is not None:
        dataset_config = replace(
            dataset_config,
            min_frequency_hz=args.min_frequency,
            max_frequency_hz=args.max_frequency,
        )
    dataset_config = replace(
        dataset_config,
        **{
            key: value
            for key, value in {
                "min_characters": args.min_characters,
                "max_characters": args.max_characters,
                "space_probability": args.space_probability,
                "word_boundary_sample_probability": (
                    args.word_boundary_sample_probability
                ),
                "extended_space_probability": args.extended_space_probability,
                "min_extended_space_multiplier": (
                    args.min_extended_space_multiplier
                ),
                "max_extended_space_multiplier": (
                    args.max_extended_space_multiplier
                ),
                "noise_only_probability": args.noise_only_probability,
                "apply_input_filter": args.input_filter,
            }.items()
            if value is not None
        },
    )
    if args.timing_jitter is not None:
        dataset_config = replace(dataset_config, timing_jitter=args.timing_jitter)
    gap_timing_overrides = {
        key: value
        for key, value in {
            "min_intra_character_units": args.min_intra_character_gap_units,
            "max_intra_character_units": args.max_intra_character_gap_units,
            "min_character_units": args.min_character_gap_units,
            "max_character_units": args.max_character_gap_units,
            "min_word_units": args.min_word_gap_units,
            "max_word_units": args.max_word_gap_units,
            "character_extreme_probability": (
                args.character_gap_extreme_probability
            ),
            "character_extreme_width_units": (
                args.character_gap_extreme_width_units
            ),
        }.items()
        if value is not None
    }
    if gap_timing_overrides:
        dataset_config = replace(
            dataset_config,
            gap_timing=replace(dataset_config.gap_timing, **gap_timing_overrides),
        )
    if args.noise_percent is not None:
        dataset_config = replace(
            dataset_config,
            noise_percent=args.noise_percent,
        )
    if args.noise_power is not None:
        dataset_config = replace(
            dataset_config,
            noise_power=args.noise_power,
        )
    if args.min_amplitude_percent is not None or args.max_amplitude_percent is not None:
        dataset_config = replace(
            dataset_config,
            min_amplitude_percent=(
                dataset_config.min_amplitude_percent
                if args.min_amplitude_percent is None
                else args.min_amplitude_percent
            ),
            max_amplitude_percent=(
                dataset_config.max_amplitude_percent
                if args.max_amplitude_percent is None
                else args.max_amplitude_percent
            ),
        )
    dataset_config = replace(
        dataset_config,
        **{
            key: value
            for key, value in {
                "fade_depth_percent": args.fade_depth_percent,
                "min_fade_frequency_hz": args.min_fade_frequency,
                "max_fade_frequency_hz": args.max_fade_frequency,
                "min_rise_fall_ms": args.min_rise_fall_ms,
                "max_rise_fall_ms": args.max_rise_fall_ms,
            }.items()
            if value is not None
        },
    )
    if args.leading_silence is not None or args.trailing_silence is not None:
        dataset_config = replace(
            dataset_config,
            **{
                key: value
                for key, value in {
                    "leading_silence_seconds": args.leading_silence,
                    "trailing_silence_seconds": args.trailing_silence,
                }.items()
                if value is not None
            },
        )
    output = args.output or automatic_output_path(model_config, dataset_config)
    generated_training_dataset = CleanAudioMorseDataset(
        args.train_samples,
        dataset_config,
        seed=args.seed,
    )
    generated_validation_dataset = CleanAudioMorseDataset(
        args.validation_samples,
        dataset_config,
        seed=args.seed + 1_000_000,
    )
    training_dataset = cache_dataset(
        generated_training_dataset,
        "train",
        args.num_workers,
    )
    validation_dataset = cache_dataset(
        generated_validation_dataset,
        "validation",
        args.num_workers,
    )
    pin_memory = device.type == "cuda"
    training_loader = create_loader(
        training_dataset,
        args.batch_size,
        True,
        args.seed,
        pin_memory,
    )
    validation_loader = create_loader(
        validation_dataset,
        args.batch_size,
        False,
        args.seed + 1,
        pin_memory,
    )
    model = MorseAudioCTCModel(model_config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.minimum_learning_rate,
    )
    start_epoch = 1
    best_token_error_rate = float("inf")
    best_character_error_rate = float("inf")
    if args.resume:
        (
            start_epoch,
            best_token_error_rate,
            best_character_error_rate,
        ) = load_training_checkpoint(
            args.resume,
            model,
            optimizer,
            scheduler,
            device,
        )
        print(f"resumed={args.resume} start_epoch={start_epoch}", flush=True)
    elif args.init_from:
        initialize_model_from_checkpoint(args.init_from, model, device)
        print(f"initialized_from={args.init_from} start_epoch=1", flush=True)
    last_output = args.last_output or output.with_name(
        f"{output.stem}.last{output.suffix}"
    )
    experiment: dict[str, int | float | bool] = {
        "train_samples": args.train_samples,
        "validation_samples": args.validation_samples,
        "seed": args.seed,
        "regenerate_every": args.regenerate_every,
        "perfect_epochs": args.perfect_epochs,
        "target_epochs": args.target_epochs,
        "tone_activity_loss_weight": TONE_ACTIVITY_LOSS_WEIGHT,
        "curriculum_target_reached": False,
    }
    if args.target_exact_text is not None:
        experiment["target_exact_text"] = args.target_exact_text
    print(
        f"device={device} objective=ctc "
        f"parameters={sum(parameter.numel() for parameter in model.parameters()):,} "
        f"train_batches={len(training_loader)} validation_batches={len(validation_loader)}",
        flush=True,
    )
    epochs_without_improvement = 0
    consecutive_perfect_epochs = 0
    consecutive_target_epochs = 0
    regeneration_count = 0
    for epoch in range(start_epoch, args.epochs + 1):
        if (
            args.regenerate_every > 0
            and epoch > start_epoch
            and (epoch - 1) % args.regenerate_every == 0
        ):
            generation = (epoch - 1) // args.regenerate_every
            regeneration_seed = args.seed + generation * 10_000_000
            regeneration_count += 1
            consecutive_perfect_epochs = 0
            consecutive_target_epochs = 0
            print(
                f"regenerating_train epoch={epoch} seed={regeneration_seed}",
                flush=True,
            )
            del training_loader
            del training_dataset
            gc.collect()
            training_dataset = cache_dataset(
                CleanAudioMorseDataset(
                    args.train_samples,
                    dataset_config,
                    seed=regeneration_seed,
                ),
                "train",
                args.num_workers,
            )
            training_loader = create_loader(
                training_dataset,
                args.batch_size,
                True,
                regeneration_seed,
                pin_memory,
            )
        print(f"epoch={epoch:03d} train", flush=True)
        training_loss = train_epoch(
            model,
            training_loader,
            optimizer,
            device,
            args.gradient_clip,
            args.log_interval,
            args.profile_batches if epoch == start_epoch else 0,
        )
        validation = evaluate_overfit_dataset(
            model,
            validation_loader,
            device,
        )
        scheduler.step(validation.ctc_loss)
        learning_rate = optimizer.param_groups[0]["lr"]
        print(
            f"epoch={epoch:03d} train_loss={training_loss:.5f} "
            f"validation_loss={validation.loss:.5f} "
            f"validation_ctc_loss={validation.ctc_loss:.5f} "
            f"token_error_rate={validation.token_error_rate:.4f} "
            f"character_error_rate={validation.character_error_rate:.4f} "
            f"exact_tokens={validation.exact_token_accuracy:.4f} "
            f"exact_text={validation.exact_text_accuracy:.4f} "
            f"lr={learning_rate:.6g} "
            f"example={validation.example_reference!r}->{validation.example_prediction!r}",
            flush=True,
        )
        improved = (
            validation.token_error_rate < best_token_error_rate
            or (
                validation.token_error_rate == best_token_error_rate
                and validation.character_error_rate < best_character_error_rate
            )
        )
        if improved:
            best_token_error_rate = validation.token_error_rate
            best_character_error_rate = validation.character_error_rate
            epochs_without_improvement = 0
            save_training_checkpoint(
                output,
                model,
                optimizer,
                scheduler,
                dataset_config,
                epoch,
                best_token_error_rate,
                best_character_error_rate,
                validation,
                experiment,
            )
            print(f"saved_best={output}", flush=True)
        else:
            epochs_without_improvement += 1
        save_training_checkpoint(
            last_output,
            model,
            optimizer,
            scheduler,
            dataset_config,
            epoch,
            best_token_error_rate,
            best_character_error_rate,
            validation,
            experiment,
        )
        consecutive_perfect_epochs = (
            consecutive_perfect_epochs + 1
            if validation.token_error_rate == 0.0
            and validation.character_error_rate == 0.0
            else 0
        )
        consecutive_target_epochs = (
            consecutive_target_epochs + 1
            if args.target_exact_text is not None
            and validation.exact_text_accuracy >= args.target_exact_text
            else 0
        )
        if consecutive_target_epochs >= args.target_epochs:
            completed_experiment = dict(experiment)
            completed_experiment["curriculum_target_reached"] = True
            save_training_checkpoint(
                last_output,
                model,
                optimizer,
                scheduler,
                dataset_config,
                epoch,
                best_token_error_rate,
                best_character_error_rate,
                validation,
                completed_experiment,
            )
            print(
                f"curriculum_target_reached epoch={epoch} "
                f"exact_text={validation.exact_text_accuracy:.4f} "
                f"consecutive={consecutive_target_epochs}",
                flush=True,
            )
            break
        if (
            consecutive_perfect_epochs >= args.perfect_epochs
            and (args.regenerate_every == 0 or regeneration_count > 0)
        ):
            print(
                f"training_target_reached epoch={epoch} "
                f"consecutive_perfect={consecutive_perfect_epochs}",
                flush=True,
            )
            break
        if args.early_stopping > 0 and epochs_without_improvement >= args.early_stopping:
            print(f"early_stopping epoch={epoch}", flush=True)
            break


if __name__ == "__main__":
    main()
