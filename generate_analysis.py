"""Generate clean, low-pass, and band-pass HELLO WORLD analysis examples."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the example-generation command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/final.pt"),
        help="Checkpoint relative to the project root",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the shared noisy condition in the filtered examples",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Generate all three reports through the public audio inference CLI."""

    args = build_argument_parser().parse_args(argv)
    project_directory = Path(__file__).resolve().parent
    checkpoint = (
        args.checkpoint
        if args.checkpoint.is_absolute()
        else project_directory / args.checkpoint
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    output_directory = project_directory / "analysis"
    output_directory.mkdir(parents=True, exist_ok=True)
    examples = (
        ("clean", "clean", ()),
        ("lowpass-1500hz", "random", ("--lowpass-cutoff-hz", "1500")),
        (
            "bandpass-750hz",
            "random",
            ("--bandpass-bandwidth-hz", "750"),
        ),
    )
    for name, profile, filter_options in examples:
        command = [
            sys.executable,
            "-m",
            "morse_timing.audio_inference",
            str(checkpoint),
            "HELLO WORLD",
            "--profile",
            profile,
            "--repetitions",
            "1",
            "--gap-seconds",
            "15",
            *filter_options,
            "--output",
            str(output_directory / f"hello-world-{name}.png"),
        ]
        if profile == "random":
            command.extend(("--seed", str(args.seed)))
        print(f"Generating {name} analysis...", flush=True)
        subprocess.run(command, cwd=project_directory, check=True)


if __name__ == "__main__":
    main()
