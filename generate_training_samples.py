"""Export reproducible trainer inputs as matching WAV, PNG, and JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from morse_timing.audio import save_wav
from morse_timing.audio_dataset import (
    CleanAudioMorseDataset,
    restore_stage1_dataset_config,
)
from morse_timing.spectrogram import save_spectrogram_image


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the training-sample preview command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_json",
        type=Path,
        help="Checkpoint metadata JSON containing dataset_config",
    )
    parser.add_argument(
        "count",
        type=int,
        help="Number of training samples to generate",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("analysis/training-samples"),
        help="Directory receiving matching WAV, PNG, and JSON files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Dataset seed; defaults to experiment.seed from the JSON, then 42",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First deterministic dataset index to export",
    )
    return parser


def load_dataset_values(path: Path) -> tuple[dict[str, Any], int]:
    """Load dataset configuration and its saved training seed."""

    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("Model JSON must contain an object")
    dataset_values = values.get("dataset_config", values)
    if not isinstance(dataset_values, dict):
        raise ValueError("dataset_config must contain an object")
    experiment = values.get("experiment", {})
    saved_seed = experiment.get("seed", 42) if isinstance(experiment, dict) else 42
    return dataset_values, int(saved_seed)


def main(argv: list[str] | None = None) -> None:
    """Generate exact trainer inputs and save their inspectable artifacts."""

    args = build_argument_parser().parse_args(argv)
    if args.count <= 0:
        raise ValueError("Sample count must be positive")
    if args.start_index < 0:
        raise ValueError("Start index cannot be negative")
    if not args.model_json.is_file():
        raise FileNotFoundError(f"Model JSON not found: {args.model_json}")

    dataset_values, saved_seed = load_dataset_values(args.model_json)
    config = restore_stage1_dataset_config(dataset_values)
    seed = saved_seed if args.seed is None else args.seed
    dataset_size = args.start_index + args.count
    dataset = CleanAudioMorseDataset(dataset_size, config, seed=seed)
    args.output_directory.mkdir(parents=True, exist_ok=True)

    for index in range(args.start_index, dataset_size):
        rendered = dataset.render(index)
        stem = f"sample-{index:05d}"
        wav_path = args.output_directory / f"{stem}.wav"
        png_path = args.output_directory / f"{stem}.png"
        json_path = args.output_directory / f"{stem}.json"
        save_wav(wav_path, rendered.waveform, config.audio.sample_rate)
        text_label = rendered.sequence.text or "(noise only)"
        gap_label = rendered.parameters.get("word_gap_multipliers", [])
        title = f"{stem}: {text_label}"
        if gap_label:
            title += f" | word gaps: {gap_label}"
        save_spectrogram_image(
            rendered.spectrogram,
            png_path,
            title=title,
            maximum_frequency_hz=min(2_000.0, config.audio.sample_rate / 2.0),
        )
        sample_metadata = {
            "source": str(args.model_json),
            "seed": seed,
            **rendered.parameters,
            "wav": wav_path.name,
            "png": png_path.name,
            "duration_seconds": rendered.spectrogram.duration_seconds,
            "spectrogram_shape": list(rendered.spectrogram.values.shape),
        }
        json_path.write_text(
            json.dumps(sample_metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(
            f"generated={index - args.start_index + 1}/{args.count} "
            f"index={index} text={text_label!r}",
            flush=True,
        )
    print(f"saved={args.output_directory}")


if __name__ == "__main__":
    main()
