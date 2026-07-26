"""Run resumable Morse LSTM training in a Kaggle notebook environment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the Kaggle training command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-directory",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing pyproject.toml and src",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("/kaggle/working/models"),
        help="Writable checkpoint directory",
    )
    parser.add_argument("--train-samples", type=int, default=12_000)
    parser.add_argument("--validation-samples", type=int, default=1_200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser


def run(command: list[str], project_directory: Path) -> None:
    """Run one training process with the local source tree importable."""

    environment = os.environ.copy()
    source_directory = str(project_directory / "src")
    existing_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{source_directory}{os.pathsep}{existing_path}"
        if existing_path
        else source_directory
    )
    print("command=" + " ".join(command), flush=True)
    subprocess.run(command, cwd=project_directory, env=environment, check=True)


def require_kaggle_cuda(project_directory: Path) -> None:
    """Fail early unless the existing environment provides CUDA PyTorch."""

    sys.path.insert(0, str(project_directory / "src"))
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is unavailable. This launcher intentionally installs no packages."
        ) from error
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. In Kaggle, enable a GPU accelerator before training."
        )
    print(
        f"torch={torch.__version__} cuda={torch.version.cuda} "
        f"gpu={torch.cuda.get_device_name(0)}",
        flush=True,
    )


def train_curriculum(args: argparse.Namespace) -> None:
    """Run the joint curriculum using a Kaggle-specific generated plan."""

    source_plan = args.project_directory / "curriculum-plan.json"
    plan = json.loads(source_plan.read_text(encoding="utf-8"))
    plan["output_directory"] = str(args.output_directory)
    training = plan.setdefault("training", {})
    training.update(
        {
            "device": "cuda",
            "train_samples": args.train_samples,
            "validation_samples": args.validation_samples,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
        }
    )
    generated_plan = Path("/kaggle/working/curriculum-plan.json")
    generated_plan.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    run(
        [
            sys.executable,
            "-m",
            "morse_timing.curriculum",
            "--plan",
            str(generated_plan),
            "--device",
            "cuda",
        ],
        args.project_directory,
    )


def main(argv: list[str] | None = None) -> None:
    """Validate Kaggle CUDA and run the resumable curriculum."""

    args = build_argument_parser().parse_args(argv)
    args.project_directory = args.project_directory.resolve()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    require_kaggle_cuda(args.project_directory)
    train_curriculum(args)


if __name__ == "__main__":
    main()
