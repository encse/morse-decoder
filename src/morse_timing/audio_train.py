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
    tone_activity_loss: float = 0.0
    event_timing_loss: float = 0.0


class TrainingLoss(float):
    """Backward-compatible scalar epoch loss with unweighted components."""

    ctc_loss: float
    tone_activity_loss: float
    event_timing_loss: float

    def __new__(
        cls,
        loss: float,
        ctc_loss: float,
        tone_activity_loss: float,
        event_timing_loss: float,
    ) -> TrainingLoss:
        value = super().__new__(cls, loss)
        value.ctc_loss = ctc_loss
        value.tone_activity_loss = tone_activity_loss
        value.event_timing_loss = event_timing_loss
        return value


TONE_ACTIVITY_LOSS_WEIGHT = 0.3
EVENT_TIMING_LOSS_WEIGHT = 0.3


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
) -> TrainingLoss:
    """Optimize the model once with the fixed curriculum objective."""
    model.train()
    total_loss = 0.0
    total_ctc_loss = 0.0
    total_tone_activity_loss = 0.0
    total_event_timing_loss = 0.0
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
            components = timed(
                "loss",
                lambda: _training_loss_components(
                    logits,
                    output_lengths,
                    batch,
                    tone_logits,
                ),
            )
            loss = components[0]
            timed("backward", loss.backward)
        else:
            batch = cpu_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, output_lengths, tone_logits = model.forward_with_auxiliary(
                batch.spectrograms,
                batch.input_lengths,
            )
            components = _training_loss_components(
                logits,
                output_lengths,
                batch,
                tone_logits,
            )
            loss = components[0]
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        if profiling:
            timed("optimizer", optimizer.step)
        else:
            optimizer.step()
        batch_size = len(batch.texts)
        total_loss += float(loss.detach()) * batch_size
        total_ctc_loss += float(components[1].detach()) * batch_size
        total_tone_activity_loss += float(components[2].detach()) * batch_size
        total_event_timing_loss += float(components[3].detach()) * batch_size
        total_samples += batch_size
        if log_interval > 0 and (
            batch_index % log_interval == 0 or batch_index == len(loader)
        ):
            print(
                f"  batch={batch_index}/{len(loader)} "
                f"loss={total_loss / total_samples:.5f} "
                f"ctc_loss={total_ctc_loss / total_samples:.5f} "
                f"tone_activity_loss="
                f"{total_tone_activity_loss / total_samples:.5f} "
                f"event_timing_loss="
                f"{total_event_timing_loss / total_samples:.5f} "
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
    return TrainingLoss(
        total_loss / total_samples,
        total_ctc_loss / total_samples,
        total_tone_activity_loss / total_samples,
        total_event_timing_loss / total_samples,
    )


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
    total_tone_activity_loss = 0.0
    total_event_timing_loss = 0.0
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
        components = _training_loss_components(
            logits,
            output_lengths,
            batch,
            tone_logits,
            ctc_loss,
        )
        loss = components[0]
        predictions = greedy_decode_batch(logits, output_lengths)
        references = split_concatenated_targets(batch.targets, batch.target_lengths)
        total_loss += float(loss.detach()) * len(batch.texts)
        total_ctc_loss += float(ctc_loss.detach()) * len(batch.texts)
        total_tone_activity_loss += float(components[2].detach()) * len(batch.texts)
        total_event_timing_loss += float(components[3].detach()) * len(batch.texts)
        sample_count += len(batch.texts)
        for text, reference, prediction in zip(
            batch.texts, references, predictions, strict=True
        ):
            token_edits += edit_distance(reference, prediction)
            target_token_count += len(reference)
            exact_tokens += int(reference == prediction)
            try:
                predicted_text = decode_audio_tokens(
                    normalize_audio_tokens(prediction)
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
        tone_activity_loss=total_tone_activity_loss / sample_count,
        event_timing_loss=total_event_timing_loss / sample_count,
    )


def _training_loss(
    logits: Tensor,
    output_lengths: Tensor,
    batch: AudioBatch,
    tone_activity_logits: Tensor,
    ctc_loss: Tensor | None = None,
) -> Tensor:
    """Calculate the fixed CTC and frame-level auxiliary objectives."""

    return _training_loss_components(
        logits,
        output_lengths,
        batch,
        tone_activity_logits,
        ctc_loss,
    )[0]


def _training_loss_components(
    logits: Tensor,
    output_lengths: Tensor,
    batch: AudioBatch,
    tone_activity_logits: Tensor,
    ctc_loss: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return total loss followed by its three unweighted components."""

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
    timing_loss = event_timing_loss(
        logits,
        batch.event_timing_targets,
        batch.padding_mask,
    )
    total_loss = (
        ctc_loss
        + TONE_ACTIVITY_LOSS_WEIGHT * tone_loss.to(ctc_loss.device)
        + EVENT_TIMING_LOSS_WEIGHT * timing_loss.to(ctc_loss.device)
    )
    return total_loss, ctc_loss, tone_loss, timing_loss


def event_timing_loss(
    logits: Tensor,
    timing_targets: Tensor,
    padding_mask: Tensor,
) -> Tensor:
    """Apply timing supervision to the second half of samples with enough Morse events."""

    if timing_targets.shape != logits.shape:
        raise ValueError("Timing targets must match the frame logits")

    expected = timing_targets.to(dtype=torch.bool).clone()
    expected[..., int(AudioToken.CTC_BLANK)] = False

    frame_count = logits.shape[1]
    frame_positions = torch.arange(
        frame_count,
        device=logits.device,
    ).view(1, -1)

    valid_frames = ~padding_mask
    valid_lengths = valid_frames.sum(dim=1)

    morse_event_count = (
        expected[..., int(AudioToken.DIT)].sum(dim=1)
        + expected[..., int(AudioToken.DAH)].sum(dim=1)
    )

    eligible_samples = morse_event_count >= 10
    supervised_starts = valid_lengths // 2

    supervised_frames = (
        valid_frames
        & eligible_samples.unsqueeze(1)
        & (frame_positions >= supervised_starts.unsqueeze(1))
    )

    expected &= supervised_frames.unsqueeze(-1)

    if not torch.any(expected):
        return logits.sum() * 0.0

    event_probabilities = logits.softmax(dim=-1)
    previous_probabilities = F.pad(
        event_probabilities[:, :-1],
        (0, 0, 1, 0),
        value=0.0,
    )
    event_onsets = event_probabilities * (1.0 - previous_probabilities)

    nonblank = torch.ones(
        logits.shape[-1],
        dtype=torch.bool,
        device=logits.device,
    )
    nonblank[int(AudioToken.CTC_BLANK)] = False

    valid = (
        supervised_frames.unsqueeze(-1)
        & nonblank.view(1, 1, -1)
    )

    unexpected = valid & ~expected

    positions = frame_positions.unsqueeze(-1)

    missing_position = torch.full_like(
        positions,
        -frame_count,
    ).expand_as(expected)

    previous_targets = torch.where(
        expected,
        positions,
        missing_position,
    ).cummax(dim=1).values

    next_candidates = torch.where(
        expected,
        positions,
        torch.full_like(
            positions,
            frame_count,
        ).expand_as(expected),
    )

    next_targets = torch.flip(
        torch.flip(next_candidates, dims=(1,))
        .cummin(dim=1)
        .values,
        dims=(1,),
    )

    distances = torch.minimum(
        (positions - previous_targets).abs(),
        (next_targets - positions).abs(),
    ).clamp_max(frame_count)

    distance_weights = (
        1.0
        + distances.to(logits.dtype)
        / max(1, frame_count - 1)
    )

    positive_terms = (
        -event_onsets.clamp_min(1e-6).log()
        * expected
    )

    negative_terms = (
        -torch.log1p(
            -event_onsets.clamp_max(1.0 - 1e-6)
        )
        * event_onsets.square()
        * unexpected
    )

    reduction_dimensions = (0, 1)

    positive_counts = expected.sum(dim=reduction_dimensions)

    positive_loss = (
        positive_terms.sum(dim=reduction_dimensions)
        / positive_counts.clamp_min(1)
    )

    negative_loss = (
        negative_terms * distance_weights
    ).sum(dim=reduction_dimensions) / positive_counts.clamp_min(1)

    represented_tokens = positive_counts > 0
    represented_tokens[int(AudioToken.CTC_BLANK)] = False

    return (
        positive_loss + negative_loss
    )[represented_tokens].mean()

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
