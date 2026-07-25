"""Sequence models for Morse audio event recognition."""

from __future__ import annotations

from dataclasses import dataclass

import numpy  # Load the Conda OpenMP runtime before PyTorch on macOS.
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from morse_timing.audio_tokens import CTC_BLANK_INDEX, NUM_AUDIO_TOKENS




@dataclass(frozen=True)
class AudioModelConfig:
    """Serializable architecture parameters for the Stage 1 model."""

    frequency_bins: int = 65
    first_conv_channels: int = 8
    second_conv_channels: int = 16
    projection_size: int = 384
    hidden_size: int = 384
    num_gru_layers: int = 2
    num_lstm_layers: int = 2
    dense_layers: int = 4
    bidirectional: bool = True
    sequence_model: str = "lstm"
    tcn_layers: int = 4
    auxiliary_heads: bool = True

    def __post_init__(self) -> None:
        dimensions = (
            self.frequency_bins,
            self.first_conv_channels,
            self.second_conv_channels,
            self.projection_size,
            self.hidden_size,
            self.num_gru_layers,
            self.num_lstm_layers,
            self.dense_layers,
            self.tcn_layers,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("Every model dimension must be positive")
        if self.frequency_bins < 4:
            raise ValueError("At least four frequency bins are required")
        if self.sequence_model not in {"gru", "lstm", "tcn"}:
            raise ValueError("Sequence model must be 'gru', 'lstm', or 'tcn'")

class CausalConv1d(nn.Module):
    """Preserve time length while reading only current and previous frames."""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.left_padding = 2 * dilation
        self.convolution = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            dilation=dilation,
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.convolution(F.pad(values, (self.left_padding, 0)))


class CausalConv2d(nn.Module):
    """Preserve time and frequency while reading no future time frames."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.convolution = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=3,
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.convolution(F.pad(values, (1, 1, 2, 0)))


class TemporalResidualBlock(nn.Module):
    """A residual, dilated temporal convolution block."""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            CausalConv1d(channels, dilation),
            nn.GELU(),
            CausalConv1d(channels, dilation),
            nn.GELU(),
        )
        self.normalization = nn.LayerNorm(channels)

    def forward(self, values: Tensor) -> Tensor:
        """Preserve time and channel dimensions while expanding context."""

        residual = values + self.network(values)
        return self.normalization(residual.transpose(1, 2)).transpose(1, 2)


class MorseAudioCTCModel(nn.Module):
    """Convert variable-length log-STFT sequences to frame-level CTC logits."""

    def __init__(self, config: AudioModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or AudioModelConfig()
        if self.config.sequence_model == "lstm":
            dense: list[nn.Module] = []
            input_size = self.config.frequency_bins
            for _ in range(self.config.dense_layers):
                dense.extend(
                    (nn.Linear(input_size, self.config.projection_size), nn.ReLU())
                )
                input_size = self.config.projection_size
            self.frequency_cnn = nn.Identity()
            self.frame_projection = nn.Sequential(*dense)
            self.sequence_encoder = nn.LSTM(
                input_size=self.config.projection_size,
                hidden_size=self.config.hidden_size,
                num_layers=self.config.num_lstm_layers,
                batch_first=True,
                bidirectional=False,
            )
            self.classifier = nn.Linear(self.config.hidden_size, NUM_AUDIO_TOKENS)
            recurrent_features = self.config.hidden_size
            self._build_auxiliary_heads(recurrent_features)
            return
        self.frequency_cnn = nn.Sequential(
            CausalConv2d(1, self.config.first_conv_channels),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            CausalConv2d(
                self.config.first_conv_channels,
                self.config.second_conv_channels,
            ),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )
        reduced_frequency_bins = self.config.frequency_bins // 2 // 2
        cnn_features = self.config.second_conv_channels * reduced_frequency_bins
        self.frame_projection = nn.Sequential(
            nn.Linear(cnn_features, self.config.projection_size),
            nn.LayerNorm(self.config.projection_size),
            nn.GELU(),
        )
        if self.config.sequence_model in {"gru", "lstm"}:
            self.sequence_encoder: nn.Module = nn.GRU(
                input_size=self.config.projection_size,
                hidden_size=self.config.hidden_size,
                num_layers=self.config.num_gru_layers,
                bidirectional=self.config.bidirectional,
                batch_first=True,
            )
            recurrent_features = self.config.hidden_size * (
                2 if self.config.bidirectional else 1
            )
        else:
            self.sequence_encoder = nn.Sequential(
                nn.Conv1d(
                    self.config.projection_size,
                    self.config.hidden_size,
                    kernel_size=1,
                ),
                *(
                    TemporalResidualBlock(
                        self.config.hidden_size,
                        dilation=2**layer,
                    )
                    for layer in range(self.config.tcn_layers)
                ),
            )
            recurrent_features = self.config.hidden_size
        self.classifier = nn.Linear(recurrent_features, NUM_AUDIO_TOKENS)
        self._build_auxiliary_heads(recurrent_features)

    def _build_auxiliary_heads(self, recurrent_features: int) -> None:
        """Create training-only heads without changing the inference output."""

        if self.config.auxiliary_heads:
            self.tone_activity_head: nn.Module | None = nn.Linear(
                recurrent_features, 1
            )
        else:
            self.tone_activity_head = None

    def forward(
        self,
        spectrograms: Tensor,
        input_lengths: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return ``[batch, time, token]`` logits and unchanged time lengths."""

        self._validate_inputs(spectrograms, input_lengths)
        cnn_output = self.extract_frequency_features(spectrograms)
        projected = self.project_frames(cnn_output)
        recurrent_output = self.encode_frames(
            projected,
            input_lengths,
            spectrograms.shape[1],
        )
        return self.classify_frames(recurrent_output), input_lengths

    def forward_with_auxiliary(
        self,
        spectrograms: Tensor,
        input_lengths: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return CTC logits plus training-only tone-activity estimates."""

        if self.tone_activity_head is None:
            raise ValueError("Auxiliary heads are disabled in this model configuration")
        self._validate_inputs(spectrograms, input_lengths)
        cnn_output = self.extract_frequency_features(spectrograms)
        projected = self.project_frames(cnn_output)
        recurrent_output = self.encode_frames(
            projected,
            input_lengths,
            spectrograms.shape[1],
        )
        logits = self.classify_frames(recurrent_output)
        tone_activity_logits = self.classify_auxiliary(recurrent_output)
        return logits, input_lengths, tone_activity_logits

    def classify_auxiliary(
        self,
        recurrent_output: Tensor,
    ) -> Tensor:
        """Apply the training-only tone head to encoded frame representations."""

        if self.tone_activity_head is None:
            raise ValueError("Auxiliary heads are disabled in this model configuration")
        return self.tone_activity_head(recurrent_output).squeeze(-1)

    def extract_frequency_features(self, spectrograms: Tensor) -> Tensor:
        """Run the frequency CNN while preserving the time dimension."""

        if self.config.sequence_model == "lstm":
            return spectrograms
        cnn_input = spectrograms.unsqueeze(1)
        return self.frequency_cnn(cnn_input)

    def project_frames(self, cnn_output: Tensor) -> Tensor:
        """Flatten compressed frequency features and project each time frame."""

        if self.config.sequence_model == "lstm":
            return self.frame_projection(cnn_output)
        frame_features = cnn_output.permute(0, 2, 1, 3).flatten(start_dim=2)
        return self.frame_projection(frame_features)

    def encode_frames(
        self,
        projected: Tensor,
        input_lengths: Tensor,
        total_length: int,
    ) -> Tensor:
        """Encode valid variable-length frame sequences with the GRU."""

        if self.config.sequence_model in {"gru", "lstm"}:
            packed = pack_padded_sequence(
                projected,
                input_lengths.detach().cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_output, _ = self.sequence_encoder(packed)
            recurrent_output, _ = pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=total_length,
            )
            return recurrent_output
        return self.sequence_encoder(projected.transpose(1, 2)).transpose(1, 2)

    def forward_stream(
        self,
        spectrograms: Tensor,
        state: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Process an LSTM chunk and return logits plus reusable hidden state."""

        if self.config.sequence_model != "lstm":
            raise ValueError("Stateful streaming is available only for the LSTM model")
        if spectrograms.ndim != 3 or spectrograms.shape[2] != self.config.frequency_bins:
            raise ValueError(
                "Streaming input must have shape [batch, time, frequency_bins]"
            )
        projected = self.frame_projection(spectrograms)
        encoded, next_state = self.sequence_encoder(projected, state)
        return self.classify_frames(encoded), next_state

    def classify_frames(self, recurrent_output: Tensor) -> Tensor:
        """Produce one CTC vocabulary logit vector per time frame."""

        return self.classifier(recurrent_output)

    def _validate_inputs(self, spectrograms: Tensor, input_lengths: Tensor) -> None:
        if spectrograms.ndim != 3:
            raise ValueError("Spectrograms must have shape [batch, time, frequency]")
        if spectrograms.shape[2] != self.config.frequency_bins:
            raise ValueError(
                f"Expected {self.config.frequency_bins} frequency bins, "
                f"received {spectrograms.shape[2]}"
            )
        if input_lengths.ndim != 1 or input_lengths.numel() != spectrograms.shape[0]:
            raise ValueError("Input lengths must contain one value per batch element")
        if torch.any(input_lengths <= 0) or torch.any(
            input_lengths > spectrograms.shape[1]
        ):
            raise ValueError("Input lengths must be within the padded time dimension")


def logits_to_ctc_log_probs(logits: Tensor) -> Tensor:
    """Convert batch-first logits to the time-first format required by CTC loss."""

    if logits.ndim != 3 or logits.shape[2] != NUM_AUDIO_TOKENS:
        raise ValueError("Logits must have shape [batch, time, five tokens]")
    return logits.log_softmax(dim=-1).transpose(0, 1)


def minimum_ctc_input_lengths(targets: Tensor, target_lengths: Tensor) -> Tensor:
    """Calculate target lengths plus separators required between repeated tokens."""

    if targets.ndim != 1 or target_lengths.ndim != 1:
        raise ValueError("CTC targets and target lengths must be one-dimensional")
    if int(target_lengths.sum()) != targets.numel():
        raise ValueError("Target lengths do not match the concatenated target tensor")
    required_lengths: list[int] = []
    offset = 0
    for target_length in target_lengths.detach().cpu().tolist():
        sequence = targets[offset : offset + target_length]
        repeated_neighbors = int((sequence[1:] == sequence[:-1]).sum().item())
        required_lengths.append(target_length + repeated_neighbors)
        offset += target_length
    return torch.tensor(required_lengths, dtype=torch.long)


def compute_ctc_loss(
    logits: Tensor,
    targets: Tensor,
    input_lengths: Tensor,
    target_lengths: Tensor,
) -> Tensor:
    """Validate sequence lengths and calculate mean CTC loss."""

    minimum_lengths = minimum_ctc_input_lengths(targets, target_lengths)
    cpu_input_lengths = input_lengths.detach().cpu()
    if torch.any(cpu_input_lengths < minimum_lengths):
        raise ValueError("CTC output does not contain enough alignment time steps")
    log_probs = logits_to_ctc_log_probs(logits)
    loss_targets = targets
    if logits.device.type == "mps":
        log_probs = log_probs.cpu()
        loss_targets = targets.cpu()
    return F.ctc_loss(
        log_probs,
        loss_targets,
        cpu_input_lengths,
        target_lengths.detach().cpu(),
        blank=CTC_BLANK_INDEX,
        reduction="mean",
        zero_infinity=True,
    )
