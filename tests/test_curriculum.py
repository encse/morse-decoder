import json
from pathlib import Path

import numpy  # Load the Conda OpenMP runtime before PyTorch on macOS.

import morse_timing.curriculum as curriculum_module
from morse_timing.audio_train import TONE_ACTIVITY_LOSS_WEIGHT
from morse_timing.curriculum import (
    _adaptive_start_ranges,
    _archive_completed_stage,
    _print_stage_summary,
    _training_command,
    next_random_ranges,
)


def test_completed_curriculum_stage_is_archived_with_ranges(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint.with_suffix(".json").write_text(
        '{"checkpoint": "model.pt", "metrics": {}}',
        encoding="utf-8",
    )

    archived = _archive_completed_stage(
        checkpoint,
        checkpoint,
        3,
        {"wpm": (14.0, 26.0)},
    )

    assert archived.read_bytes() == b"checkpoint"
    metadata = json.loads(
        archived.with_suffix(".json").read_text(encoding="utf-8")
    )
    assert metadata["curriculum_stage"] == 3
    assert metadata["curriculum_ranges"] == {"wpm": [14.0, 26.0]}


def test_completed_stage_summary_is_searchable(capsys) -> None:
    _print_stage_summary(
        4,
        Path("models/stage-004.pt"),
        0.934,
        {"wpm": (12.0, 28.0), "noise_power": (0.0, 20.0)},
    )

    output = capsys.readouterr().out
    assert "CURRICULUM_CHECKPOINT_READY stage=4 exact_text=0.9340" in output
    assert "checkpoint=models/stage-004.pt" in output
    assert '"wpm": [12.0, 28.0]' in output


def test_adaptive_curriculum_starts_at_exact_configured_values() -> None:
    dimensions = {
        "wpm": {
            "lower_limit": 10,
            "upper_limit": 40,
            "start": 25,
            "step": 2,
        },
        "noise": {
            "lower_limit": 0,
            "upper_limit": 180,
            "start": 0,
            "step": 20,
        },
    }

    assert _adaptive_start_ranges(dimensions) == {
        "wpm": (25, 25),
        "noise": (0, 0),
    }


def test_adaptive_stage_restarts_learning_rate_and_sets_accuracy_target() -> None:
    command = _training_command(
        {
            "learning_rate": 0.0001,
            "minimum_learning_rate": 0.00001,
            "noise_only_probability": 0.2,
        },
        Path("source.pt"),
        Path("output.pt"),
        {"wpm": (18, 22)},
        "cpu",
        target_exact_text=0.9,
        target_epochs=2,
    )

    assert command[command.index("--learning-rate") + 1] == "0.0001"
    assert command[command.index("--target-exact-text") + 1] == "0.9"
    assert command[command.index("--target-epochs") + 1] == "2"
    assert command[command.index("--noise-only-probability") + 1] == "0.2"


def test_initial_stage_creates_a_new_model_with_the_same_loss() -> None:
    command = _training_command(
        {},
        None,
        Path("output.pt"),
        {"wpm": (25, 25)},
        "cpu",
        target_exact_text=0.9,
    )

    assert "--resume" not in command
    assert "--init-from" not in command
    assert "--tone-activity-loss-weight" not in command
    assert TONE_ACTIVITY_LOSS_WEIGHT == 0.3


def test_first_curriculum_stage_builds_the_model_from_scratch(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    named_checkpoint = tmp_path / "curriculum.pt"
    reference_wav = tmp_path / "reference.wav"
    dimensions = {
        "wpm": {
            "lower_limit": 25,
            "upper_limit": 25,
            "start": 25,
            "step": 2,
        }
    }
    plan = {
        "reference_wav": str(reference_wav),
        "adaptive": {
            "exact_text_threshold": 0.9,
            "required_epochs": 2,
            "max_epochs_per_dimension": 50,
            "selection_seed": 42,
        }
    }
    commands: list[list[str]] = []
    reference_calls: list[tuple[int, Path, Path, str]] = []

    def fake_training(command: list[str], check: bool) -> None:
        assert check
        commands.append(command)
        output = Path(command[command.index("--output") + 1])
        output.write_bytes(b"center-model")
        output.with_suffix(".json").write_text(
            json.dumps(
                {
                    "checkpoint": str(output),
                    "experiment": {"curriculum_target_reached": True},
                    "metrics": {"exact_text_accuracy": 0.95},
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(curriculum_module.subprocess, "run", fake_training)
    monkeypatch.setattr(
        curriculum_module,
        "_print_reference_wav_result",
        lambda stage, checkpoint, wav_path, device: reference_calls.append(
            (stage, checkpoint, wav_path, device)
        ),
    )

    curriculum_module._run_adaptive_plan(
        plan,
        named_checkpoint,
        {},
        dimensions,
        "cpu",
    )

    assert len(commands) == 1
    assert "--resume" not in commands[0]
    assert "--init-from" not in commands[0]
    assert commands[0][commands[0].index("--min-wpm") + 1] == "25.0"
    assert commands[0][commands[0].index("--max-wpm") + 1] == "25.0"
    assert named_checkpoint.read_bytes() == b"center-model"
    assert reference_calls == [
        (
            1,
            tmp_path / "curriculum.stages" / "stage-001.pt",
            reference_wav,
            "cpu",
        )
    ]
    assert capsys.readouterr().out.endswith("\n\n\n")


def test_reference_wav_result_prints_morse_and_text(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    class Result:
        predicted_morse = "... --- ..."
        decoded_text = "SOS"
        valid = True
        duration_seconds = 3.5
        error = None

    monkeypatch.setattr(
        curriculum_module,
        "_decode_reference_wav",
        lambda checkpoint, wav_path, device: Result(),
    )

    curriculum_module._print_reference_wav_result(
        7,
        tmp_path / "stage-007.pt",
        Path("reference.wav"),
        "cpu",
    )

    output = capsys.readouterr().out
    assert "REFERENCE_WAV_RESULT stage=7 wav=reference.wav" in output
    assert "predicted_morse=... --- ..." in output
    assert "decoded_text='SOS'" in output
    assert "valid=True duration=3.500s" in output


def test_adaptive_stage_can_resume_without_resetting_its_epoch_counter() -> None:
    command = _training_command(
        {},
        Path("working.last.pt"),
        Path("working.pt"),
        {"wpm": (18, 22)},
        "cpu",
        resume=True,
    )

    assert "--resume" in command
    assert "--init-from" not in command


def test_adaptive_curriculum_expands_exactly_one_random_dimension() -> None:
    dimensions = {
        "wpm": {"lower_limit": 10, "upper_limit": 40, "step": 2},
        "frequency": {"lower_limit": 100, "upper_limit": 2000, "step": 100},
        "jitter": {"lower_limit": 0, "upper_limit": 0.2, "step": 0.02},
        "noise": {"lower_limit": 0, "upper_limit": 50, "step": 5},
    }

    current = {
        "wpm": (20, 20),
        "frequency": (700, 700),
        "jitter": (0, 0),
        "noise": (0, 0),
    }
    updated = next_random_ranges(
        current,
        dimensions,
        selection_seed=42,
        completed_stages=1,
    )

    assert updated is not None
    assert sum(updated[name] != current[name] for name in current) == 1
    assert updated == next_random_ranges(
        current,
        dimensions,
        selection_seed=42,
        completed_stages=1,
    )


def test_adaptive_curriculum_never_selects_an_excluded_dimension() -> None:
    dimensions = {
        "wpm": {"lower_limit": 18, "upper_limit": 22, "step": 2},
        "frequency": {"lower_limit": 600, "upper_limit": 800, "step": 100},
    }
    current = {"wpm": (20, 20), "frequency": (700, 700)}

    updated = next_random_ranges(
        current,
        dimensions,
        selection_seed=42,
        completed_stages=1,
        excluded_dimensions={"wpm"},
    )

    assert updated == {"wpm": (20, 20), "frequency": (600, 800)}
    assert (
        next_random_ranges(
            current,
            dimensions,
            selection_seed=42,
            completed_stages=1,
            excluded_dimensions={"wpm", "frequency"},
        )
        is None
    )


def test_failed_dimension_is_abandoned_and_next_dimension_starts_from_stable_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    named_checkpoint = tmp_path / "curriculum.pt"
    named_checkpoint.write_bytes(b"stable")
    state_path = tmp_path / "curriculum.curriculum.json"
    state_path.write_text(
        json.dumps(
            {
                "completed_stages": 1,
                "selection_seed": 42,
                "dimensions": {
                    "wpm": {
                        "lower_limit": 18,
                        "upper_limit": 22,
                        "start": 20,
                        "step": 2,
                    },
                    "frequency": {
                        "lower_limit": 600,
                        "upper_limit": 800,
                        "start": 700,
                        "step": 100,
                    },
                },
                "successful_ranges": {
                    "wpm": [20, 20],
                    "frequency": [700, 700],
                },
                "ranges": {
                    "wpm": [18, 22],
                    "frequency": [700, 700],
                },
                "selected_dimension": "wpm",
                "failed_dimensions": [],
                "attempt_number": 1,
            }
        ),
        encoding="utf-8",
    )
    plan = {
        "adaptive": {
            "exact_text_threshold": 0.9,
            "required_epochs": 2,
            "max_epochs_per_dimension": 50,
            "selection_seed": 42,
        }
    }
    dimensions = {
        "wpm": {
            "lower_limit": 18,
            "upper_limit": 22,
            "start": 20,
            "step": 2,
        },
        "frequency": {
            "lower_limit": 600,
            "upper_limit": 800,
            "start": 700,
            "step": 100,
        },
    }
    commands: list[list[str]] = []

    def fake_training(command: list[str], check: bool) -> None:
        assert check
        commands.append(command)
        output = Path(command[command.index("--output") + 1])
        output.write_bytes(f"attempt-{len(commands)}".encode())
        exact_text = 0.8 if len(commands) == 1 else 0.95
        output.with_suffix(".json").write_text(
            json.dumps(
                {
                    "checkpoint": str(output),
                    "training_objective": "ctc",
                    "epoch": 50,
                    "experiment": {
                        "curriculum_target_reached": len(commands) == 2,
                    },
                    "metrics": {"exact_text_accuracy": exact_text},
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(curriculum_module.subprocess, "run", fake_training)

    curriculum_module._run_adaptive_plan(
        plan,
        named_checkpoint,
        {"epochs": 1_000_000},
        dimensions,
        "cpu",
    )

    assert len(commands) == 2
    assert commands[0][commands[0].index("--epochs") + 1] == "50"
    assert commands[1][commands[1].index("--epochs") + 1] == "50"
    assert commands[1][commands[1].index("--init-from") + 1] == str(
        named_checkpoint
    )
    assert "attempt-02-frequency" in commands[1][
        commands[1].index("--output") + 1
    ]
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["completed_stages"] == 2
    assert state["failed_dimensions"] == ["wpm"]
    assert state["successful_ranges"] == {
        "frequency": [600.0, 800.0],
        "wpm": [20.0, 20.0],
    }
    assert state["selected_dimension"] is None


def test_adaptive_curriculum_excludes_dimensions_at_their_limits() -> None:
    dimensions = {
        "wpm": {"lower_limit": 10, "upper_limit": 40, "step": 2},
        "noise": {"lower_limit": 0, "upper_limit": 50, "step": 5},
    }

    assert next_random_ranges(
        {"wpm": (10, 40), "noise": (0, 0)},
        dimensions,
        selection_seed=42,
        completed_stages=1,
    ) == {"wpm": (10, 40), "noise": (0, 5)}


def test_joint_curriculum_returns_none_at_all_limits() -> None:
    dimensions = {
        "wpm": {"lower_limit": 10, "upper_limit": 40, "step": 2},
        "noise": {"lower_limit": 0, "upper_limit": 50, "step": 5},
    }

    assert next_random_ranges(
        {"wpm": (10, 40), "noise": (0, 50)},
        dimensions,
        selection_seed=42,
        completed_stages=1,
    ) is None


def test_adaptive_curriculum_expands_noise_power_when_amplitude_is_complete() -> None:
    dimensions = {
        "noise_power": {"lower_limit": 0, "upper_limit": 200, "step": 20},
        "amplitude": {"lower_limit": 10, "upper_limit": 150, "step": 10},
    }

    assert next_random_ranges(
        {"noise_power": (0, 20), "amplitude": (10, 150)},
        dimensions,
        selection_seed=42,
        completed_stages=1,
    ) == {
        "noise_power": (0, 40),
        "amplitude": (10, 150),
    }
