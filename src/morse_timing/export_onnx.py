"""Export a trained streaming Morse LSTM checkpoint to ONNX."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy  # Load the Conda OpenMP runtime before PyTorch on macOS.
import torch
from torch import Tensor, nn

from morse_timing.audio_dataset import restore_stage1_dataset_config
from morse_timing.audio_model import AudioModelConfig, MorseAudioCTCModel
from morse_timing.generate_go_reference import generate_reference


class StreamingOnnxWrapper(nn.Module):
    """Export a fixed-batch streaming model using only 2D linear inputs."""

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
        """Return logits and reusable recurrent state."""

        # [1, frames, frequency_bins] -> [frames, frequency_bins]
        features_2d = features.squeeze(0)

        # Every Linear inside frame_projection now receives a 2D tensor.
        projected_2d = self.model.frame_projection(features_2d)

        # LSTM still expects [batch, frames, projection_size].
        projected = projected_2d.unsqueeze(0)

        encoded, (next_hidden, next_cell) = self.model.sequence_encoder(
            projected,
            (hidden_state, cell_state),
        )

        # [1, frames, hidden_size] -> [frames, hidden_size]
        encoded_2d = encoded.squeeze(0)

        # Every Linear inside classify_frames receives a 2D tensor.
        logits_2d = self.model.classify_frames(encoded_2d)

        # Restore [batch, frames, token_count].
        logits = logits_2d.unsqueeze(0)

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

    incompatible = model.load_state_dict(
        checkpoint["model_state"],
        strict=False,
    )
    allowed_missing_keys = {
        "frequency_head.weight",
        "frequency_head.bias",
    }

    if (
        incompatible.unexpected_keys
        or not set(incompatible.missing_keys).issubset(allowed_missing_keys)
    ):
        raise ValueError(
            "Checkpoint model weights are incompatible: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )

    return StreamingOnnxWrapper(model), checkpoint


def save_float32_tensor(path: Path, tensor: Tensor) -> None:
    """Save a contiguous tensor as little-endian raw float32 values."""

    values = (
        tensor.detach()
        .cpu()
        .contiguous()
        .numpy()
        .astype("<f4", copy=False)
    )
    path.write_bytes(values.tobytes())


def tensor_description(path: Path, tensor: Tensor) -> dict[str, Any]:
    """Return metadata needed to load one raw tensor."""

    return {
        "file": path.name,
        "shape": list(tensor.shape),
        "element_count": tensor.numel(),
        "dtype": "float32",
        "byte_order": "little-endian",
    }


def export_state_dict(
    output_path: Path,
    model: MorseAudioCTCModel,
) -> Path:
    """Export model tensors as raw little-endian float32 files."""

    weights_directory = output_path.with_suffix(".weights")
    weights_directory.mkdir(parents=True, exist_ok=True)

    tensors: dict[str, Any] = {}

    for name, tensor in model.state_dict().items():
        values = (
            tensor.detach()
            .cpu()
            .contiguous()
            .numpy()
            .astype("<f4", copy=False)
        )

        safe_name = name.replace(".", "__")
        filename = f"{safe_name}.f32"
        path = weights_directory / filename
        path.write_bytes(values.tobytes())

        tensors[name] = {
            "file": filename,
            "shape": list(values.shape),
            "element_count": int(values.size),
        }

    metadata = {
        "dtype": "float32",
        "byte_order": "little-endian",
        "tensors": tensors,
    }

    metadata_path = weights_directory / "weights.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return metadata_path


def verify_wrapper_matches_model(
    wrapper: StreamingOnnxWrapper,
    features: Tensor,
    hidden_state: Tensor,
    cell_state: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Verify that the Go-compatible wrapper preserves production semantics."""

    with torch.inference_mode():
        wrapped = wrapper(features, hidden_state, cell_state)
        production_logits, production_state = wrapper.model.forward_stream(
            features,
            (hidden_state, cell_state),
        )

    production = (
        production_logits,
        production_state[0],
        production_state[1],
    )
    names = ("logits", "hidden state", "cell state")
    for name, actual, expected in zip(names, wrapped, production, strict=True):
        try:
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        except AssertionError as error:
            raise RuntimeError(
                f"ONNX wrapper diverges from production model {name}"
            ) from error
    return wrapped

def export_checkpoint(
    checkpoint_path: Path,
    output_path: Path,
    chunk_frames: int = 25,
    opset_version: int = 17,
    reference_duration_seconds: float = 0.5,
) -> None:
    """Export one checkpoint and deterministic reference test tensors."""

    if chunk_frames <= 0:
        raise ValueError("Chunk frame count must be positive")
    if reference_duration_seconds <= 0.0:
        raise ValueError("Reference duration must be positive")

    wrapper, checkpoint = load_wrapper(checkpoint_path)
    config = wrapper.model.config

    generator = torch.Generator(device="cpu").manual_seed(42)

    features = torch.randn(
        1,
        chunk_frames,
        config.frequency_bins,
        generator=generator,
        dtype=torch.float32,
    )
    hidden_state = (
        torch.randn(
            config.num_lstm_layers,
            1,
            config.hidden_size,
            generator=generator,
            dtype=torch.float32,
        )
        * 0.1
    )
    cell_state = (
        torch.randn(
            config.num_lstm_layers,
            1,
            config.hidden_size,
            generator=generator,
            dtype=torch.float32,
        )
        * 0.1
    )

    (
        expected_logits,
        expected_hidden_state,
        expected_cell_state,
    ) = verify_wrapper_matches_model(
        wrapper,
        features,
        hidden_state,
        cell_state,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        torch.onnx.export(
            wrapper,
            (features, hidden_state, cell_state),
            output_path,
            input_names=(
                "features",
                "hidden_state",
                "cell_state",
            ),
            output_names=(
                "logits",
                "next_hidden_state",
                "next_cell_state",
            ),
            opset_version=opset_version,
            dynamo=False,
        )
    except ModuleNotFoundError as error:
        if error.name == "onnx":
            raise RuntimeError(
                "The 'onnx' package is required for export but is not installed"
            ) from error
        raise

    try:
        import onnx
    except ImportError as error:
        raise RuntimeError(
            "The 'onnx' package is required for model inspection"
        ) from error

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    operator_counts = Counter(
        node.op_type
        for node in onnx_model.graph.node
    )

    test_data_directory = output_path.with_suffix(
        output_path.suffix + ".testdata"
    )
    test_data_directory.mkdir(parents=True, exist_ok=True)

    tensor_files = {
        "features": test_data_directory / "features.f32",
        "hidden_state": test_data_directory / "hidden_state.f32",
        "cell_state": test_data_directory / "cell_state.f32",
        "expected_logits": test_data_directory / "expected_logits.f32",
        "expected_hidden_state": (
            test_data_directory / "expected_hidden_state.f32"
        ),
        "expected_cell_state": (
            test_data_directory / "expected_cell_state.f32"
        ),
    }

    tensors = {
        "features": features,
        "hidden_state": hidden_state,
        "cell_state": cell_state,
        "expected_logits": expected_logits,
        "expected_hidden_state": expected_hidden_state,
        "expected_cell_state": expected_cell_state,
    }

    for name, tensor in tensors.items():
        save_float32_tensor(tensor_files[name], tensor)

    dataset_config = restore_stage1_dataset_config(
        checkpoint["dataset_config"]
    )
    reference_directory = output_path.with_suffix(
        output_path.suffix + ".reference"
    )
    reference_metadata_path = generate_reference(
        checkpoint_path,
        reference_directory,
        reference_duration_seconds,
        chunk_frames,
    )

    metadata = {
        "source_checkpoint": str(checkpoint_path),
        "onnx_model": output_path.name,
        "opset_version": opset_version,
        "example_chunk_frames": chunk_frames,
        "waveform_reference": {
            "directory": reference_directory.name,
            "metadata": reference_metadata_path.name,
        },
        "operators": dict(sorted(operator_counts.items())),
        "inputs": {
            "features": {
                "onnx_name": "features",
                **tensor_description(
                    tensor_files["features"],
                    features,
                ),
            },
            "hidden_state": {
                "onnx_name": "hidden_state",
                **tensor_description(
                    tensor_files["hidden_state"],
                    hidden_state,
                ),
            },
            "cell_state": {
                "onnx_name": "cell_state",
                **tensor_description(
                    tensor_files["cell_state"],
                    cell_state,
                ),
            },
        },
        "expected_outputs": {
            "logits": {
                "onnx_name": "logits",
                **tensor_description(
                    tensor_files["expected_logits"],
                    expected_logits,
                ),
            },
            "next_hidden_state": {
                "onnx_name": "next_hidden_state",
                **tensor_description(
                    tensor_files["expected_hidden_state"],
                    expected_hidden_state,
                ),
            },
            "next_cell_state": {
                "onnx_name": "next_cell_state",
                **tensor_description(
                    tensor_files["expected_cell_state"],
                    expected_cell_state,
                ),
            },
        },
        "model_config": asdict(config),
        "dataset_config": asdict(dataset_config),
    }

    metadata_path = output_path.with_suffix(
        output_path.suffix + ".json"
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


    weights_metadata_path = export_state_dict(
        output_path,
        wrapper.model,
    )

    print(f"weights={weights_metadata_path}")
    print("model tensors:")

    for name, tensor in wrapper.model.state_dict().items():
        print(f"  {name}: {list(tensor.shape)}")
        
    print(f"onnx={output_path}")
    print(f"metadata={metadata_path}")
    print(f"test_data={test_data_directory}")
    print("operators:")
    for operator, count in sorted(operator_counts.items()):
        print(f"  {operator}: {count}")

    print("tensor_shapes:")
    for name, tensor in tensors.items():
        print(f"  {name}: {list(tensor.shape)}")


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the checkpoint export command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--chunk-frames", type=int, default=25)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--reference-duration", type=float, default=0.5)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Export a checkpoint and its deterministic reference tensors."""

    args = build_argument_parser().parse_args(argv)
    output = args.output or args.checkpoint.with_suffix(".onnx")

    export_checkpoint(
        args.checkpoint,
        output,
        args.chunk_frames,
        args.opset,
        args.reference_duration,
    )


if __name__ == "__main__":
    main()
