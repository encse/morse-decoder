"""Run the resumable adaptive Morse augmentation curriculum."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy  # Load the Conda OpenMP runtime before PyTorch on macOS.


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the curriculum training command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    return parser


def next_random_ranges(
    current: dict[str, tuple[float, float]],
    dimensions: dict[str, dict[str, float]],
    selection_seed: int,
    completed_stages: int,
    excluded_dimensions: set[str] | frozenset[str] | None = None,
) -> dict[str, tuple[float, float]] | None:
    """Advance one randomly selected unfinished dimension by one step."""

    excluded = excluded_dimensions or set()
    candidates: list[tuple[str, tuple[float, float]]] = []
    for name, specification in dimensions.items():
        if name not in {
            "wpm", "frequency", "jitter", "noise", "fade_depth",
            "noise_power", "amplitude", "fade_frequency", "rise_fall",
        }:
            raise ValueError(f"Unsupported curriculum dimension: {name}")
        if name in excluded:
            continue
        lower = float(specification["lower_limit"])
        upper = float(specification["upper_limit"])
        step = float(specification["step"])
        if step <= 0.0 or lower > upper:
            raise ValueError(f"Invalid limits or step for curriculum dimension {name}")
        minimum, maximum = current[name]
        if not lower <= minimum <= maximum <= upper:
            raise ValueError(f"Current {name} range is outside its curriculum limits")
        if name in {"jitter", "noise", "noise_power", "fade_depth"}:
            next_minimum = lower
            next_maximum = min(upper, maximum + step)
        else:
            next_minimum = max(lower, minimum - step)
            next_maximum = min(upper, maximum + step)
        next_range = (next_minimum, next_maximum)
        if next_range != (minimum, maximum):
            candidates.append((name, next_range))
    if not candidates:
        return None
    candidates.sort()
    rng = numpy.random.default_rng(
        numpy.random.SeedSequence([selection_seed, completed_stages])
    )
    selected_name, selected_range = candidates[int(rng.integers(len(candidates)))]
    updated = dict(current)
    updated[selected_name] = selected_range
    return updated


def _range_options(ranges: dict[str, tuple[float, float]]) -> list[str]:
    options: list[str] = []
    if "wpm" in ranges:
        minimum, maximum = ranges["wpm"]
        options.extend(("--min-wpm", str(minimum), "--max-wpm", str(maximum)))
    if "frequency" in ranges:
        minimum, maximum = ranges["frequency"]
        options.extend(
            ("--min-frequency", str(minimum), "--max-frequency", str(maximum))
        )
    if "jitter" in ranges:
        options.extend(("--timing-jitter", str(ranges["jitter"][1])))
    if "noise" in ranges:
        options.extend(("--noise-percent", str(ranges["noise"][1])))
    if "noise_power" in ranges:
        options.extend(("--noise-power", str(ranges["noise_power"][1])))
    if "amplitude" in ranges:
        minimum, maximum = ranges["amplitude"]
        options.extend(
            (
                "--min-amplitude-percent",
                str(minimum),
                "--max-amplitude-percent",
                str(maximum),
            )
        )
    if "fade_depth" in ranges:
        options.extend(("--fade-depth-percent", str(ranges["fade_depth"][1])))
    if "fade_frequency" in ranges:
        minimum, maximum = ranges["fade_frequency"]
        options.extend(
            (
                "--min-fade-frequency",
                str(minimum),
                "--max-fade-frequency",
                str(maximum),
            )
        )
    if "rise_fall" in ranges:
        minimum, maximum = ranges["rise_fall"]
        options.extend(
            (
                "--min-rise-fall-ms",
                str(minimum),
                "--max-rise-fall-ms",
                str(maximum),
            )
        )
    return options


def _publish_named_checkpoint(source: Path, destination: Path) -> None:
    """Publish the latest successful stage under a stable plan-level name."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    source_metadata = source.with_suffix(".json")
    if source_metadata.exists():
        metadata = json.loads(source_metadata.read_text(encoding="utf-8"))
        metadata["checkpoint"] = str(destination)
        destination.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _training_command(
    training: dict[str, Any],
    checkpoint: Path | None,
    output: Path,
    ranges: dict[str, tuple[float, float]],
    device_override: str | None,
    target_exact_text: float | None = None,
    target_epochs: int = 2,
    resume: bool = False,
    apply_input_filter: bool | None = None,
) -> list[str]:
    """Build one isolated training command for a curriculum range."""

    command = [
        sys.executable,
        "-m",
        "morse_timing.train",
        "--epochs",
        str(training.get("epochs", 1_000_000)),
        "--early-stopping",
        str(training.get("early_stopping", 0)),
        "--train-samples",
        str(training.get("train_samples", 6_000)),
        "--validation-samples",
        str(training.get("validation_samples", 600)),
        "--batch-size",
        str(training.get("batch_size", 32)),
        "--num-workers",
        str(training.get("num_workers", 0)),
        "--regenerate-every",
        str(training.get("regenerate_every", 10)),
        "--perfect-epochs",
        str(training.get("perfect_epochs", 5)),
        "--device",
        device_override or str(training.get("device", "auto")),
    ]
    if checkpoint is None:
        if resume:
            raise ValueError("Cannot resume without a checkpoint")
    else:
        command.extend(("--resume" if resume else "--init-from", str(checkpoint)))
    selected_input_filter = (
        checkpoint is not None
        if apply_input_filter is None
        else apply_input_filter
    )
    command.append(
        "--input-filter" if selected_input_filter else "--no-input-filter"
    )
    command.extend(("--output", str(output)))
    command.extend(_range_options(ranges))
    if "learning_rate" in training:
        command.extend(("--learning-rate", str(training["learning_rate"])))
    if "minimum_learning_rate" in training:
        command.extend(
            ("--minimum-learning-rate", str(training["minimum_learning_rate"]))
        )
    if "noise_only_probability" in training:
        command.extend(
            (
                "--noise-only-probability",
                str(training["noise_only_probability"]),
            )
        )
    if target_exact_text is not None:
        command.extend(
            (
                "--target-exact-text",
                str(target_exact_text),
                "--target-epochs",
                str(target_epochs),
            )
        )
    return command


def _adaptive_start_ranges(
    dimensions: dict[str, dict[str, float]],
) -> dict[str, tuple[float, float]]:
    """Validate and return one exact initial value for every dimension."""

    ranges: dict[str, tuple[float, float]] = {}
    for name, specification in dimensions.items():
        lower = float(specification["lower_limit"])
        upper = float(specification["upper_limit"])
        start = float(specification["start"])
        step = float(specification["step"])
        if step <= 0.0 or not lower <= start <= upper:
            raise ValueError(f"Invalid adaptive curriculum range for {name}")
        ranges[name] = (start, start)
    return ranges


def _checkpoint_exact_text(path: Path) -> float:
    """Read exact-text validation accuracy from a saved checkpoint."""

    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    return float(metadata["metrics"]["exact_text_accuracy"])


def _checkpoint_reached_target(path: Path) -> bool:
    """Read the trainer's explicit curriculum success marker."""

    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    return bool(metadata["experiment"]["curriculum_target_reached"])


def _archive_completed_stage(
    checkpoint: Path,
    named_checkpoint: Path,
    stage: int,
    ranges: dict[str, tuple[float, float]],
) -> Path:
    """Keep an immutable checkpoint for one completed curriculum stage."""

    archive_directory = named_checkpoint.with_name(
        f"{named_checkpoint.stem}.stages"
    )
    archive_directory.mkdir(parents=True, exist_ok=True)
    destination = archive_directory / f"stage-{stage:03d}.pt"
    if not destination.exists():
        shutil.copy2(checkpoint, destination)
        source_metadata = checkpoint.with_suffix(".json")
        if source_metadata.exists():
            metadata = json.loads(source_metadata.read_text(encoding="utf-8"))
            metadata["checkpoint"] = str(destination)
            metadata["curriculum_stage"] = stage
            metadata["curriculum_ranges"] = {
                name: list(values) for name, values in ranges.items()
            }
            destination.with_suffix(".json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    return destination


def _print_stage_summary(
    stage: int,
    checkpoint: Path,
    exact_text: float,
    ranges: dict[str, tuple[float, float]],
) -> None:
    """Print a searchable summary of one completed curriculum checkpoint."""

    capabilities = json.dumps(
        {name: list(values) for name, values in ranges.items()},
        sort_keys=True,
    )
    print(
        f"CURRICULUM_CHECKPOINT_READY stage={stage} "
        f"exact_text={exact_text:.4f}",
        flush=True,
    )
    print(f"  checkpoint={checkpoint}", flush=True)
    print(f"  capabilities={capabilities}", flush=True)


def _decode_reference_wav(checkpoint: Path, wav_path: Path, device: str):
    """Decode one diagnostic WAV without coupling it to curriculum decisions."""

    from morse_timing.audio_inference import MorseAudioDecoder

    return MorseAudioDecoder.load(checkpoint, device).decode_wav(wav_path)


def _print_reference_wav_result(
    stage: int,
    checkpoint: Path,
    wav_path: Path,
    device: str,
) -> None:
    """Print a best-effort diagnostic decode for one completed stage."""

    print(
        f"REFERENCE_WAV_RESULT stage={stage} wav={wav_path} "
        f"checkpoint={checkpoint}",
        flush=True,
    )
    try:
        result = _decode_reference_wav(checkpoint, wav_path, device)
    except Exception as error:
        print(
            f"  reference_error={type(error).__name__}: {error}",
            flush=True,
        )
        return
    print(f"  predicted_morse={result.predicted_morse}", flush=True)
    print(f"  decoded_text={result.decoded_text!r}", flush=True)
    print(
        f"  valid={result.valid} duration={result.duration_seconds:.3f}s",
        flush=True,
    )
    if result.error:
        print(f"  error={result.error}", flush=True)


def _print_stage_separator() -> None:
    """Leave two empty lines between curriculum stages."""

    print(flush=True)
    print(flush=True)


def _write_adaptive_state(
    path: Path,
    *,
    threshold: float,
    target_epochs: int,
    max_epochs_per_dimension: int,
    selection_seed: int,
    dimensions: dict[str, dict[str, float]],
    completed_stages: int,
    successful_ranges: dict[str, tuple[float, float]],
    ranges: dict[str, tuple[float, float]],
    selected_dimension: str | None,
    failed_dimensions: set[str],
    attempt_number: int,
) -> None:
    """Persist enough state to resume or choose a new dimension safely."""

    path.write_text(
        json.dumps(
            {
                "exact_text_threshold": threshold,
                "required_epochs": target_epochs,
                "max_epochs_per_dimension": max_epochs_per_dimension,
                "selection_seed": selection_seed,
                "dimensions": dimensions,
                "completed_stages": completed_stages,
                "successful_ranges": {
                    name: list(values) for name, values in successful_ranges.items()
                },
                "ranges": {name: list(values) for name, values in ranges.items()},
                "selected_dimension": selected_dimension,
                "failed_dimensions": sorted(failed_dimensions),
                "attempt_number": attempt_number,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _run_adaptive_plan(
    plan: dict[str, Any],
    named_checkpoint: Path,
    training: dict[str, Any],
    dimensions: dict[str, dict[str, float]],
    device_override: str | None,
) -> None:
    """Expand one random range after validation reaches the configured threshold."""

    adaptive = dict(plan["adaptive"])
    threshold = float(adaptive.get("exact_text_threshold", 0.9))
    target_epochs = int(adaptive.get("required_epochs", 2))
    max_epochs_per_dimension = int(adaptive.get("max_epochs_per_dimension", 50))
    configured_selection_seed = int(
        adaptive.get("selection_seed", training.get("seed", 42))
    )
    reference_wav_value = plan.get("reference_wav")
    reference_wav = (
        None if reference_wav_value is None else Path(str(reference_wav_value))
    )
    reference_device = device_override or str(training.get("device", "auto"))
    if (
        not 0.0 < threshold <= 1.0
        or target_epochs <= 0
        or max_epochs_per_dimension <= 0
    ):
        raise ValueError("Invalid adaptive curriculum target")
    state_path = named_checkpoint.with_name(
        f"{named_checkpoint.stem}.curriculum.json"
    )
    state_path.parent.mkdir(parents=True, exist_ok=True)
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("dimensions") != dimensions:
            raise ValueError(
                "Existing curriculum state belongs to a different plan; "
                "use an empty output directory for a new curriculum"
            )
        completed_stages = int(state.get("completed_stages", 0))
        selection_seed = int(
            state.get("selection_seed", configured_selection_seed)
        )
        ranges = {
            name: (float(values[0]), float(values[1]))
            for name, values in state["ranges"].items()
        }
        failed_dimensions = set(state.get("failed_dimensions", ()))
        attempt_number = int(state.get("attempt_number", 1))
        selected_dimension = state.get("selected_dimension")
        successful_ranges = {
            name: (float(values[0]), float(values[1]))
            for name, values in state["successful_ranges"].items()
        }
    else:
        completed_stages = 0
        selection_seed = configured_selection_seed
        ranges = _adaptive_start_ranges(dimensions)
        successful_ranges = dict(ranges)
        selected_dimension = None
        failed_dimensions: set[str] = set()
        attempt_number = 1
    training_for_stage = dict(training)
    training_for_stage["epochs"] = max_epochs_per_dimension
    while True:
        stage_number = completed_stages + 1
        attempt_suffix = (
            ""
            if attempt_number == 1
            else f".attempt-{attempt_number:02d}-{selected_dimension or 'initial'}"
        )
        working_checkpoint = named_checkpoint.with_name(
            f"{named_checkpoint.stem}.working-stage-{stage_number:03d}"
            f"{attempt_suffix}"
            f"{named_checkpoint.suffix}"
        )
        working_last_checkpoint = working_checkpoint.with_name(
            f"{working_checkpoint.stem}.last{working_checkpoint.suffix}"
        )
        stable_checkpoint = named_checkpoint if named_checkpoint.exists() else None
        if completed_stages > 0 and stable_checkpoint is None:
            raise FileNotFoundError(
                f"Curriculum state exists without its model: {named_checkpoint}"
            )
        if working_last_checkpoint.exists():
            previous_model = working_last_checkpoint
            resume_training = True
        elif working_checkpoint.exists():
            previous_model = working_checkpoint
            resume_training = True
        else:
            previous_model = stable_checkpoint
            resume_training = False
        _write_adaptive_state(
            state_path,
            threshold=threshold,
            target_epochs=target_epochs,
            max_epochs_per_dimension=max_epochs_per_dimension,
            selection_seed=selection_seed,
            dimensions=dimensions,
            completed_stages=completed_stages,
            successful_ranges=successful_ranges,
            ranges=ranges,
            selected_dimension=selected_dimension,
            failed_dimensions=failed_dimensions,
            attempt_number=attempt_number,
        )
        source = (
            f"resume={previous_model}"
            if resume_training
            else (
                f"init_from={previous_model}"
                if previous_model is not None
                else "new_model"
            )
        )
        print(
            f"adaptive_training={ranges} selected_dimension={selected_dimension} "
            f"attempt={attempt_number} target_exact_text={threshold:g} "
            f"max_epochs={max_epochs_per_dimension} "
            f"{source} "
            f"output={working_checkpoint}",
            flush=True,
        )
        subprocess.run(
            _training_command(
                training_for_stage,
                previous_model,
                working_checkpoint,
                ranges,
                device_override,
                threshold,
                target_epochs,
                resume=resume_training,
                apply_input_filter=completed_stages > 0,
            ),
            check=True,
        )
        stage_checkpoint = (
            working_last_checkpoint
            if working_last_checkpoint.exists()
            else working_checkpoint
        )
        achieved = _checkpoint_exact_text(stage_checkpoint)
        target_reached = _checkpoint_reached_target(stage_checkpoint)
        if not target_reached:
            print(
                f"adaptive_target_not_reached exact_text={achieved:.4f} "
                f"after_epochs={max_epochs_per_dimension} "
                f"selected_dimension={selected_dimension}",
                flush=True,
            )
            if selected_dimension is None:
                print("adaptive_initial_stage_failed", flush=True)
                _print_stage_separator()
                return
            failed_dimensions.add(selected_dimension)
            next_ranges = next_random_ranges(
                successful_ranges,
                dimensions,
                selection_seed,
                completed_stages,
                excluded_dimensions=failed_dimensions,
            )
            if next_ranges is None:
                _write_adaptive_state(
                    state_path,
                    threshold=threshold,
                    target_epochs=target_epochs,
                    max_epochs_per_dimension=max_epochs_per_dimension,
                    selection_seed=selection_seed,
                    dimensions=dimensions,
                    completed_stages=completed_stages,
                    successful_ranges=successful_ranges,
                    ranges=successful_ranges,
                    selected_dimension=None,
                    failed_dimensions=failed_dimensions,
                    attempt_number=attempt_number,
                )
                print(
                    "adaptive_dimensions_exhausted "
                    f"failed_dimensions={sorted(failed_dimensions)}",
                    flush=True,
                )
                _print_stage_separator()
                return
            selected_dimension = next(
                name for name in dimensions
                if next_ranges[name] != successful_ranges[name]
            )
            ranges = next_ranges
            attempt_number += 1
            print(
                f"adaptive_retry_with_dimension={selected_dimension} "
                f"excluded_dimensions={sorted(failed_dimensions)} "
                f"next_ranges={ranges}",
                flush=True,
            )
            _print_stage_separator()
            continue
        _publish_named_checkpoint(stage_checkpoint, named_checkpoint)
        completed_stages += 1
        archived_checkpoint = _archive_completed_stage(
            named_checkpoint,
            named_checkpoint,
            completed_stages,
            ranges,
        )
        _print_stage_summary(
            completed_stages,
            archived_checkpoint,
            achieved,
            ranges,
        )
        if reference_wav is not None:
            _print_reference_wav_result(
                completed_stages,
                archived_checkpoint,
                reference_wav,
                reference_device,
            )
        successful_ranges = dict(ranges)
        next_ranges = next_random_ranges(
            successful_ranges,
            dimensions,
            selection_seed,
            completed_stages,
            excluded_dimensions=failed_dimensions,
        )
        if next_ranges is None:
            unrestricted_next = next_random_ranges(
                successful_ranges,
                dimensions,
                selection_seed,
                completed_stages,
            )
            message = (
                "adaptive_curriculum_complete"
                if unrestricted_next is None
                else "adaptive_dimensions_exhausted"
            )
            _write_adaptive_state(
                state_path,
                threshold=threshold,
                target_epochs=target_epochs,
                max_epochs_per_dimension=max_epochs_per_dimension,
                selection_seed=selection_seed,
                dimensions=dimensions,
                completed_stages=completed_stages,
                successful_ranges=successful_ranges,
                ranges=successful_ranges,
                selected_dimension=None,
                failed_dimensions=failed_dimensions,
                attempt_number=1,
            )
            print(
                f"{message} exact_text={achieved:.4f} "
                f"failed_dimensions={sorted(failed_dimensions)}",
                flush=True,
            )
            _print_stage_separator()
            return
        selected_dimension = next(
            name for name in dimensions
            if next_ranges[name] != successful_ranges[name]
        )
        print(
            f"adaptive_next_dimension={selected_dimension} "
            f"next_ranges={next_ranges}",
            flush=True,
        )
        _print_stage_separator()
        ranges = next_ranges
        attempt_number = 1


def run_joint_plan(plan_path: Path, device_override: str | None = None) -> None:
    """Run the adaptive curriculum plan."""

    plan: dict[str, Any] = json.loads(plan_path.read_text(encoding="utf-8"))
    name = str(plan["name"])
    output_directory = Path(plan.get("output_directory", "models"))
    named_checkpoint = output_directory / f"{name}.pt"
    training = dict(plan.get("training", {}))
    dimensions: dict[str, dict[str, float]] = dict(plan["dimensions"])
    _run_adaptive_plan(
        plan,
        named_checkpoint,
        training,
        dimensions,
        device_override,
    )


def main(argv: list[str] | None = None) -> None:
    """Run the JSON-configured curriculum plan."""

    args = build_argument_parser().parse_args(argv)
    run_joint_plan(args.plan, None if args.device == "auto" else args.device)


if __name__ == "__main__":
    main()
