"""Export a trained streaming Morse LSTM checkpoint to ONNX."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy  # Load the Conda OpenMP runtime before PyTorch on macOS.
import torch
from torch import Tensor, nn

from morse_timing.audio_dataset import restore_stage1_dataset_config
from morse_timing.audio_model import AudioModelConfig, MorseAudioCTCModel


class StreamingOnnxWrapper(nn.Module):
    """Expose explicit LSTM state tensors as ONNX inputs and outputs."""

    def __init__(self, model: MorseAudioCTCModel) -> None:
        super().__init__()
        if model.config.sequence_model != "lstm":
            raise ValueError("ONNX streaming export requires an LSTM checkpoint")
        self.model = model.eval()

    def forward(
        self,
        features: Tensor,
        hidden_state: Tensor,
        cell_state: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return logits and the state to pass to the next audio chunk."""

        projected = self.model.frame_projection(features)
        encoded, (next_hidden, next_cell) = self.model.sequence_encoder(
            projected,
            (hidden_state, cell_state),
        )
        logits = self.model.classify_frames(encoded)
        return logits, next_hidden, next_cell


def load_wrapper(
    checkpoint_path: Path,
) -> tuple[StreamingOnnxWrapper, dict[str, Any]]:
    """Load a training checkpoint and construct its streaming export wrapper."""

    checkpoint: dict[str, Any] = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    config = AudioModelConfig(**checkpoint["model_config"])
    model = MorseAudioCTCModel(config)
    model.load_state_dict(checkpoint["model_state"])
    return StreamingOnnxWrapper(model), checkpoint


def export_checkpoint(
    checkpoint_path: Path,
    output_path: Path,
    chunk_frames: int = 25,
    opset_version: int = 17,
) -> None:
    """Export one checkpoint with dynamic batch and chunk dimensions."""

    if chunk_frames <= 0:
        raise ValueError("Chunk frame count must be positive")
    wrapper, checkpoint = load_wrapper(checkpoint_path)
    config = wrapper.model.config
    features = torch.zeros(1, chunk_frames, config.frequency_bins)
    hidden_state = torch.zeros(config.num_lstm_layers, 1, config.hidden_size)
    cell_state = torch.zeros_like(hidden_state)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.onnx.export(
            wrapper,
            (features, hidden_state, cell_state),
            output_path,
            input_names=("features", "hidden_state", "cell_state"),
            output_names=("logits", "next_hidden_state", "next_cell_state"),
            dynamic_axes={
                "features": {0: "batch", 1: "frames"},
                "hidden_state": {1: "batch"},
                "cell_state": {1: "batch"},
                "logits": {0: "batch", 1: "frames"},
                "next_hidden_state": {1: "batch"},
                "next_cell_state": {1: "batch"},
            },
            opset_version=opset_version,
            dynamo=False,
        )
    except ModuleNotFoundError as error:
        if error.name == "onnx":
            raise RuntimeError(
                "The 'onnx' package is required for export but is not installed"
            ) from error
        raise
    dataset_config = restore_stage1_dataset_config(checkpoint["dataset_config"])
    metadata = {
        "source_checkpoint": str(checkpoint_path),
        "onnx_model": str(output_path),
        "opset_version": opset_version,
        "example_chunk_frames": chunk_frames,
        "inputs": {
            "features": ["batch", "frames", config.frequency_bins],
            "hidden_state": [config.num_lstm_layers, "batch", config.hidden_size],
            "cell_state": [config.num_lstm_layers, "batch", config.hidden_size],
        },
        "outputs": {
            "logits": ["batch", "frames", 5],
            "next_hidden_state": [
                config.num_lstm_layers,
                "batch",
                config.hidden_size,
            ],
            "next_cell_state": [
                config.num_lstm_layers,
                "batch",
                config.hidden_size,
            ],
        },
        "model_config": asdict(config),
        "dataset_config": asdict(dataset_config),
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the checkpoint export command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--chunk-frames", type=int, default=25)
    parser.add_argument("--opset", type=int, default=17)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Export a checkpoint and describe its streaming tensor interface."""

    args = build_argument_parser().parse_args(argv)
    output = args.output or args.checkpoint.with_suffix(".onnx")
    export_checkpoint(args.checkpoint, output, args.chunk_frames, args.opset)
    print(f"onnx={output}")
    print(f"metadata={output.with_suffix(output.suffix + '.json')}")


if __name__ == "__main__":
    main()
