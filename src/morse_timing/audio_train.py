"""Shared loss, metrics, checkpoint, and epoch helpers for CTC training."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from morse_timing.audio_dataset import (
    AudioBatch,
    Stage1DatasetConfig,
)
from morse_timing.audio_model import (
    MorseAudioCTCModel,
    compute_ctc_loss,
)
from morse_timing.audio_tokens import (
    AudioToken,
    collapse_ctc_path,
    decode_audio_tokens,
    format_audio_tokens_as_morse,
    normalize_audio_tokens,
)


@dataclass(frozen=True)
class OverfitMetrics:
    """Metrics measured on the same small dataset used for optimization."""

    loss: float
    token_error_rate: float
    character_error_rate: float
    exact_token_accuracy: float
    exact_text_accuracy: float
    example_reference: str
    example_prediction: str
    ctc_loss: float = 0.0


TONE_ACTIVITY_LOSS_WEIGHT = 0.3


def select_device(requested: str) -> torch.device:
    """Select CUDA, MPS, or CPU with an optional explicit override."""

    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Seed all random sources used by the overfit experiment."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def greedy_decode_batch(logits: Tensor, lengths: Tensor) -> list[tuple[AudioToken, ...]]:
    """Greedily collapse frame-level model predictions for every batch element."""

    frame_predictions = logits.argmax(dim=-1).detach().cpu()
    decoded: list[tuple[AudioToken, ...]] = []
    for row, length in enumerate(lengths.detach().cpu().tolist()):
        decoded.append(collapse_ctc_path(frame_predictions[row, :length].tolist()))
    return decoded


def split_concatenated_targets(
    targets: Tensor, target_lengths: Tensor
) -> list[tuple[AudioToken, ...]]:
    """Split PyTorch's concatenated CTC targets back into individual sequences."""

    cpu_targets = targets.detach().cpu()
    sequences: list[tuple[AudioToken, ...]] = []
    offset = 0
    for length in target_lengths.detach().cpu().tolist():
        sequences.append(
            tuple(
                AudioToken(int(value))
                for value in cpu_targets[offset : offset + length]
            )
        )
        offset += length
    return sequences


def edit_distance(reference: tuple[AudioToken, ...], prediction: tuple[AudioToken, ...]) -> int:
    """Calculate token-level Levenshtein distance with linear auxiliary memory."""

    previous = list(range(len(prediction) + 1))
    for reference_index, reference_token in enumerate(reference, start=1):
        current = [reference_index]
        for prediction_index, prediction_token in enumerate(prediction, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[prediction_index] + 1,
                    previous[prediction_index - 1]
                    + int(reference_token != prediction_token),
                )
            )
        previous = current
    return previous[-1]


def character_edit_distance(reference: str, prediction: str) -> int:
    """Calculate character-level Levenshtein distance."""

    previous = list(range(len(prediction) + 1))
    for reference_index, reference_character in enumerate(reference, start=1):
        current = [reference_index]
        for prediction_index, prediction_character in enumerate(prediction, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[prediction_index] + 1,
                    previous[prediction_index - 1]
                    + int(reference_character != prediction_character),
                )
            )
        previous = current
    return previous[-1]


def train_epoch(
    model: MorseAudioCTCModel,
    loader: DataLoader[AudioBatch],
    optimizer: AdamW,
    device: torch.device,
    gradient_clip: float,
    log_interval: int,
    profile_batches: int = 0,
) -> float:
    """Optimize the model once with the fixed curriculum objective."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    started_at = perf_counter()
    profile_totals = {
        name: 0.0
        for name in (
            "data",
            "device_copy",
            "cnn",
            "projection",
            "sequence_encoder",
            "classifier",
            "loss",
            "backward",
            "optimizer",
        )
    }
    previous_batch_finished = started_at

    def synchronize() -> None:
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize(device)

    def timed(name: str, operation):
        synchronize()
        operation_started = perf_counter()
        result = operation()
        synchronize()
        profile_totals[name] += perf_counter() - operation_started
        return result

    for batch_index, cpu_batch in enumerate(loader, start=1):
        profiling = batch_index <= profile_batches
        if profiling:
            profile_totals["data"] += perf_counter() - previous_batch_finished
            batch = timed("device_copy", lambda: cpu_batch.to(device))
            timed("optimizer", lambda: optimizer.zero_grad(set_to_none=True))
            cnn_output = timed(
                "cnn", lambda: model.extract_frequency_features(batch.spectrograms)
            )
            projected = timed("projection", lambda: model.project_frames(cnn_output))
            recurrent_output = timed(
                "sequence_encoder",
                lambda: model.encode_frames(
                    projected, batch.input_lengths, batch.spectrograms.shape[1]
                ),
            )
            logits = timed(
                "classifier", lambda: model.classify_frames(recurrent_output)
            )
            tone_logits = model.classify_auxiliary(recurrent_output)
            output_lengths = batch.input_lengths
            loss = timed(
                "loss",
                lambda: _training_loss(
                    logits,
                    output_lengths,
                    batch,
                    tone_logits,
                ),
            )
            timed("backward", loss.backward)
        else:
            batch = cpu_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, output_lengths, tone_logits = model.forward_with_auxiliary(
                batch.spectrograms,
                batch.input_lengths,
            )
            loss = _training_loss(
                logits,
                output_lengths,
                batch,
                tone_logits,
            )
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        if profiling:
            timed("optimizer", optimizer.step)
        else:
            optimizer.step()
        batch_size = len(batch.texts)
        total_loss += float(loss.detach()) * batch_size
        total_samples += batch_size
        if log_interval > 0 and (
            batch_index % log_interval == 0 or batch_index == len(loader)
        ):
            print(
                f"  batch={batch_index}/{len(loader)} "
                f"loss={total_loss / total_samples:.5f} "
                f"elapsed={perf_counter() - started_at:.1f}s",
                flush=True,
            )
        previous_batch_finished = perf_counter()
    measured_batches = min(profile_batches, len(loader))
    if measured_batches:
        print(f"profile_batches={measured_batches} average_ms:", flush=True)
        for name, total_seconds in profile_totals.items():
            print(
                f"  {name}={total_seconds * 1_000.0 / measured_batches:.2f}",
                flush=True,
            )
    return total_loss / total_samples


@torch.inference_mode()
def evaluate_overfit_dataset(
    model: MorseAudioCTCModel,
    loader: DataLoader[AudioBatch],
    device: torch.device,
) -> OverfitMetrics:
    """Measure token and decoded-text quality with the training objective."""
    model.eval()
    total_loss = 0.0
    total_ctc_loss = 0.0
    sample_count = 0
    token_edits = 0
    target_token_count = 0
    character_edits = 0
    reference_character_count = 0
    exact_tokens = 0
    exact_texts = 0
    example_reference = ""
    example_prediction = ""
    for cpu_batch in loader:
        batch = cpu_batch.to(device)
        logits, output_lengths, tone_logits = model.forward_with_auxiliary(
            batch.spectrograms,
            batch.input_lengths,
        )
        ctc_loss = compute_ctc_loss(
            logits,
            batch.targets,
            output_lengths,
            batch.target_lengths,
        )
        loss = _training_loss(
            logits,
            output_lengths,
            batch,
            tone_logits,
            ctc_loss,
        )
        predictions = greedy_decode_batch(logits, output_lengths)
        references = split_concatenated_targets(batch.targets, batch.target_lengths)
        total_loss += float(loss.detach()) * len(batch.texts)
        total_ctc_loss += float(ctc_loss.detach()) * len(batch.texts)
        sample_count += len(batch.texts)
        for text, reference, prediction in zip(
            batch.texts, references, predictions, strict=True
        ):
            token_edits += edit_distance(reference, prediction)
            target_token_count += len(reference)
            exact_tokens += int(reference == prediction)
            try:
                predicted_text = decode_audio_tokens(
                    normalize_audio_tokens(prediction),
                    recognize_prosigns=False,
                ).text
            except ValueError:
                morse = format_audio_tokens_as_morse(
                    normalize_audio_tokens(prediction)
                )
                predicted_text = f"[{morse}]"
            character_edits += character_edit_distance(text, predicted_text)
            reference_character_count += len(text)
            exact_texts += int(predicted_text == text)
            if not example_reference:
                example_reference = text
                example_prediction = predicted_text
    return OverfitMetrics(
        loss=total_loss / sample_count,
        token_error_rate=token_edits / target_token_count,
        character_error_rate=character_edits / reference_character_count,
        exact_token_accuracy=exact_tokens / sample_count,
        exact_text_accuracy=exact_texts / sample_count,
        example_reference=example_reference,
        example_prediction=example_prediction,
        ctc_loss=total_ctc_loss / sample_count,
    )


def _training_loss(
    logits: Tensor,
    output_lengths: Tensor,
    batch: AudioBatch,
    tone_activity_logits: Tensor,
    ctc_loss: Tensor | None = None,
) -> Tensor:
    """Calculate the fixed CTC plus tone-activity training objective."""

    if ctc_loss is None:
        ctc_loss = compute_ctc_loss(
            logits,
            batch.targets,
            output_lengths,
            batch.target_lengths,
        )
    valid = ~batch.padding_mask
    tone_loss = F.binary_cross_entropy_with_logits(
        tone_activity_logits,
        batch.tone_activity,
        reduction="none",
    )[valid].mean()
    return ctc_loss + TONE_ACTIVITY_LOSS_WEIGHT * tone_loss.to(ctc_loss.device)


def save_checkpoint(
    path: Path,
    model: MorseAudioCTCModel,
    dataset_config: Stage1DatasetConfig,
    texts: list[str],
    epoch: int,
    metrics: OverfitMetrics,
) -> None:
    """Save an overfit checkpoint with architecture and experiment metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "purpose": "stage1_overfit",
            "training_objective": "ctc",
            "model_config": asdict(model.config),
            "model_state": model.state_dict(),
            "dataset_config": asdict(dataset_config),
            "texts": texts,
            "epoch": epoch,
            "metrics": asdict(metrics),
        },
        path,
    )
