"""Generate waveform and logits reference data for Go inference tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy  # Load the Conda OpenMP runtime before PyTorch on macOS.
import torch
from torch import Tensor

from morse_timing.audio_inference import MorseAudioDecoder


def save_float32(path: Path, tensor: Tensor) -> None:
    """Save a contiguous tensor as little-endian raw float32 values."""

    values = (
        tensor.detach()
        .cpu()
        .contiguous()
        .numpy()
        .astype("<f4", copy=False)
    )
    path.write_bytes(values.tobytes())


def tensor_metadata(path: Path, tensor: Tensor) -> dict[str, Any]:
    """Describe a raw tensor sufficiently for a Go test to load it."""

    return {
        "file": path.name,
        "shape": list(tensor.shape),
        "element_count": tensor.numel(),
        "dtype": "float32",
        "byte_order": "little-endian",
    }


def generate_reference(
    checkpoint: Path,
    output_directory: Path,
    duration_seconds: float,
    chunk_frames: int = 25,
) -> Path:
    """Generate deterministic waveform input and expected model logits."""

    if duration_seconds <= 0.0:
        raise ValueError("Duration must be positive")

    decoder = MorseAudioDecoder.load(checkpoint, "cpu")
    sample_rate = decoder.dataset_config.audio.sample_rate
    sample_count = round(duration_seconds * sample_rate)
    if sample_count <= 0:
        raise ValueError("Duration is too short to contain a sample")

    # A fixed seed and tones make the input repeatable but non-trivial.
    generator = torch.Generator(device="cpu").manual_seed(42)
    time = torch.arange(sample_count, dtype=torch.float32) / sample_rate
    waveform = (
        0.4 * torch.sin(2.0 * torch.pi * 700.0 * time)
        + 0.2 * torch.sin(2.0 * torch.pi * 950.0 * time)
        + 0.01
        * torch.randn(
            sample_count,
            generator=generator,
            dtype=torch.float32,
        )
    )

    logits = decoder.predict_waveform_logits(
        waveform,
        sample_rate,
        chunk_frames=chunk_frames,
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    waveform_path = output_directory / "waveform.f32"
    logits_path = output_directory / "expected_logits.f32"
    save_float32(waveform_path, waveform)
    save_float32(logits_path, logits)

    metadata = {
        "source_checkpoint": str(checkpoint),
        "sample_rate": sample_rate,
        "duration_seconds": waveform.numel() / sample_rate,
        "input": {
            "waveform": tensor_metadata(waveform_path, waveform),
        },
        "expected_outputs": {
            "logits": tensor_metadata(logits_path, logits),
        },
    }
    metadata_path = output_directory / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"waveform={waveform_path}")
    print(f"logits={logits_path}")
    print(f"metadata={metadata_path}")
    print(f"sample_rate={sample_rate}")
    print(f"sample_count={waveform.numel()}")
    print(f"logits_shape={list(logits.shape)}")
    return metadata_path


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the reference generation command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("go-reference"))
    parser.add_argument("--duration", type=float, default=0.5)
    parser.add_argument("--chunk-frames", type=int, default=25)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Generate reference files from command-line arguments."""

    args = build_argument_parser().parse_args(argv)
    generate_reference(
        checkpoint=args.checkpoint,
        output_directory=args.output,
        duration_seconds=args.duration,
        chunk_frames=args.chunk_frames,
    )


if __name__ == "__main__":
    main()
