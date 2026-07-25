"""Morse audio features, CTC token sequences, and variable-length batching."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from morse_timing.audio import (
    AudioConfig,
    RenderedSegment,
    add_power_scaled_noise,
    add_white_noise,
    apply_recording_amplitude,
    apply_sinusoidal_fading,
    synthesize_morse_with_timing,
)
from morse_timing.audio_tokens import AudioToken, text_to_audio_tokens
from morse_timing.morse import SUPPORTED_CHARACTERS, normalize_text
from morse_timing.spectrogram import SpectrogramConfig, compute_log_magnitude_stft


@dataclass(frozen=True)
class Stage1DatasetConfig:
    """Configuration for clean, fixed-speed Stage 1 examples."""

    wpm: float = 20.0
    min_wpm: float | None = None
    max_wpm: float | None = None
    min_frequency_hz: float | None = None
    max_frequency_hz: float | None = None
    timing_jitter: float = 0.0
    noise_power: float = 0.0
    noise_percent: float = 0.0
    min_amplitude_percent: float = 100.0
    max_amplitude_percent: float = 100.0
    fade_depth_percent: float = 0.0
    min_fade_frequency_hz: float = 0.1
    max_fade_frequency_hz: float = 2.0
    min_rise_fall_ms: float = 0.0
    max_rise_fall_ms: float = 0.0
    min_characters: int = 2
    max_characters: int = 12
    space_probability: float = 0.12
    word_boundary_sample_probability: float = 0.5
    doubled_space_probability: float = 0.5
    leading_silence_seconds: float = 0.7
    trailing_silence_seconds: float = 0.7
    noise_only_probability: float = 0.0
    min_noise_only_seconds: float = 2.0
    max_noise_only_seconds: float = 4.0
    min_noise_only_power: float = 1.0
    audio: AudioConfig = AudioConfig()
    spectrogram: SpectrogramConfig = SpectrogramConfig()

    def __post_init__(self) -> None:
        if self.wpm <= 0.0:
            raise ValueError("WPM must be positive")
        if (self.min_wpm is None) != (self.max_wpm is None):
            raise ValueError("Minimum and maximum WPM must be provided together")
        if self.min_wpm is not None and (
            self.min_wpm <= 0.0 or self.max_wpm < self.min_wpm
        ):
            raise ValueError("WPM range must be positive and ordered")
        if (self.min_frequency_hz is None) != (self.max_frequency_hz is None):
            raise ValueError("Minimum and maximum frequency must be provided together")
        if self.min_frequency_hz is not None and (
            self.min_frequency_hz <= 0.0
            or self.max_frequency_hz < self.min_frequency_hz
            or self.max_frequency_hz >= self.audio.sample_rate / 2
        ):
            raise ValueError(
                "Frequency range must be positive, ordered, and below Nyquist"
            )
        if self.min_characters < 1 or self.max_characters < self.min_characters:
            raise ValueError("Character range must be positive and ordered")
        if not 0.0 <= self.space_probability <= 1.0:
            raise ValueError("Space probability must be between zero and one")
        if not 0.0 <= self.word_boundary_sample_probability <= 1.0:
            raise ValueError(
                "Word-boundary sample probability must be between zero and one"
            )
        if not 0.0 <= self.doubled_space_probability <= 1.0:
            raise ValueError("Doubled-space probability must be between zero and one")
        if self.timing_jitter < 0.0:
            raise ValueError("Timing jitter cannot be negative")
        if self.noise_power < 0.0:
            raise ValueError("Noise power cannot be negative")
        if self.noise_percent < 0.0:
            raise ValueError("Noise percentage cannot be negative")
        if (
            self.min_amplitude_percent < 0.0
            or self.max_amplitude_percent < self.min_amplitude_percent
        ):
            raise ValueError("Amplitude range must be non-negative and ordered")
        if not 0.0 <= self.fade_depth_percent <= 100.0:
            raise ValueError("Fade depth must be between zero and 100 percent")
        if (
            self.min_fade_frequency_hz <= 0.0
            or self.max_fade_frequency_hz < self.min_fade_frequency_hz
        ):
            raise ValueError("Fade frequency range must be positive and ordered")
        if self.min_rise_fall_ms < 0.0 or self.max_rise_fall_ms < self.min_rise_fall_ms:
            raise ValueError("Rise/fall range must be non-negative and ordered")
        if self.leading_silence_seconds < 0.0 or self.trailing_silence_seconds < 0.0:
            raise ValueError("Leading and trailing silence cannot be negative")
        if not 0.0 <= self.noise_only_probability <= 1.0:
            raise ValueError("Noise-only probability must be between zero and one")
        if (
            self.min_noise_only_seconds <= 0.0
            or self.max_noise_only_seconds < self.min_noise_only_seconds
        ):
            raise ValueError("Noise-only duration range must be positive and ordered")
        if self.min_noise_only_power <= 0.0:
            raise ValueError("Minimum noise-only power must be positive")


@dataclass(frozen=True)
class AudioSequenceSample:
    """One variable-length spectrogram with its CTC token sequence."""

    spectrogram: Tensor
    targets: Tensor
    tone_activity: Tensor
    text: str

    @property
    def input_length(self) -> int:
        return self.spectrogram.shape[0]

    @property
    def target_length(self) -> int:
        return self.targets.numel()


@dataclass(frozen=True)
class AudioBatch:
    """A padded spectrogram batch with concatenated CTC targets."""

    spectrograms: Tensor
    targets: Tensor
    tone_activity: Tensor
    input_lengths: Tensor
    target_lengths: Tensor
    padding_mask: Tensor
    texts: tuple[str, ...]

    def to(self, device: torch.device | str) -> AudioBatch:
        """Move all tensor fields to a training device."""

        return AudioBatch(
            spectrograms=self.spectrograms.to(device),
            targets=self.targets.to(device),
            tone_activity=self.tone_activity.to(device),
            input_lengths=self.input_lengths.to(device),
            target_lengths=self.target_lengths.to(device),
            padding_mask=self.padding_mask.to(device),
            texts=self.texts,
        )


class CleanAudioMorseDataset(Dataset[AudioSequenceSample]):
    """Generate reproducible clean Stage 1 audio and features on demand."""

    def __init__(
        self,
        size: int,
        config: Stage1DatasetConfig | None = None,
        seed: int = 0,
        texts: Sequence[str] | None = None,
    ) -> None:
        if size <= 0:
            raise ValueError("Dataset size must be positive")
        if texts is not None and len(texts) != size:
            raise ValueError("The number of supplied texts must equal dataset size")
        self.size = size
        self.config = config or Stage1DatasetConfig()
        self.seed = seed
        self.texts = tuple(normalize_text(text) for text in texts) if texts else None

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> AudioSequenceSample:
        if index < 0:
            index += self.size
        if index < 0 or index >= self.size:
            raise IndexError(index)
        if self.texts is None and self._is_noise_only(index):
            return build_noise_only_sequence_sample(
                self.config,
                duration_seconds=self._sample_noise_only_duration(index),
                noise_seed=int(
                    np.random.SeedSequence([self.seed, index, 4]).generate_state(1)[0]
                ),
                noise_power=max(
                    self.config.min_noise_only_power,
                    self._sample_noise_power(index),
                ),
                amplitude_percent=self._sample_amplitude_percent(index),
            )
        text = self.texts[index] if self.texts is not None else self._random_text(index)
        return build_audio_sequence_sample(
            text,
            self.config,
            wpm=self._sample_wpm(index),
            frequency_hz=self._sample_frequency(index),
            timing_seed=int(
                np.random.SeedSequence([self.seed, index, 3]).generate_state(1)[0]
            ),
            noise_seed=int(
                np.random.SeedSequence([self.seed, index, 4]).generate_state(1)[0]
            ),
            noise_percent=self._sample_noise_percent(index),
            noise_power=self._sample_noise_power(index),
            amplitude_percent=self._sample_amplitude_percent(index),
            fade_depth_percent=self._sample_fade_depth(index),
            fade_frequency_hz=self._sample_fade_frequency(index),
            fade_phase_radians=self._sample_fade_phase(index),
            rise_fall_ms=self._sample_rise_fall(index),
            doubled_word_gaps=(
                None
                if self.texts is not None
                else self._sample_doubled_word_gaps(index, text)
            ),
        )

    def _is_noise_only(self, index: int) -> bool:
        """Select reproducible negative examples without affecting text sampling."""

        rng = np.random.default_rng(
            int(np.random.SeedSequence([self.seed, index, 13]).generate_state(1)[0])
        )
        return bool(rng.random() < self.config.noise_only_probability)

    def _sample_noise_only_duration(self, index: int) -> float:
        rng = np.random.default_rng(
            int(np.random.SeedSequence([self.seed, index, 14]).generate_state(1)[0])
        )
        return float(
            rng.uniform(
                self.config.min_noise_only_seconds,
                self.config.max_noise_only_seconds,
            )
        )

    def _sample_doubled_word_gaps(
        self,
        index: int,
        text: str,
    ) -> tuple[bool, ...]:
        """Choose normal or doubled duration for every generated word boundary."""

        rng = np.random.default_rng(
            int(np.random.SeedSequence([self.seed, index, 12]).generate_state(1)[0])
        )
        return tuple(
            bool(rng.random() < self.config.doubled_space_probability)
            for _ in range(text.count(" "))
        )

    def _sample_rise_fall(self, index: int) -> float:
        rng = np.random.default_rng(
            int(np.random.SeedSequence([self.seed, index, 9]).generate_state(1)[0])
        )
        return float(rng.uniform(self.config.min_rise_fall_ms, self.config.max_rise_fall_ms))

    def _sample_fade_depth(self, index: int) -> float:
        if self.config.fade_depth_percent == 0.0:
            return 0.0
        rng = np.random.default_rng(
            int(np.random.SeedSequence([self.seed, index, 6]).generate_state(1)[0])
        )
        return float(rng.uniform(0.0, self.config.fade_depth_percent))

    def _sample_fade_frequency(self, index: int) -> float:
        rng = np.random.default_rng(
            int(np.random.SeedSequence([self.seed, index, 7]).generate_state(1)[0])
        )
        return float(
            rng.uniform(
                self.config.min_fade_frequency_hz,
                self.config.max_fade_frequency_hz,
            )
        )

    def _sample_fade_phase(self, index: int) -> float:
        rng = np.random.default_rng(
            int(np.random.SeedSequence([self.seed, index, 8]).generate_state(1)[0])
        )
        return float(rng.uniform(0.0, 2.0 * np.pi))

    def _sample_noise_percent(self, index: int) -> float:
        """Sample a reproducible noise level from zero to the configured maximum."""

        if self.config.noise_percent == 0.0:
            return 0.0
        child_seed = int(
            np.random.SeedSequence([self.seed, index, 5]).generate_state(1)[0]
        )
        rng = np.random.default_rng(child_seed)
        return float(rng.uniform(0.0, self.config.noise_percent))

    def _sample_noise_power(self, index: int) -> float:
        """Sample noise power from zero to its configured maximum."""

        if self.config.noise_power == 0.0:
            return 0.0
        rng = np.random.default_rng(
            int(np.random.SeedSequence([self.seed, index, 10]).generate_state(1)[0])
        )
        return float(rng.uniform(0.0, self.config.noise_power))

    def _sample_amplitude_percent(self, index: int) -> float:
        """Sample recording-wide gain independently for every recording."""

        rng = np.random.default_rng(
            int(np.random.SeedSequence([self.seed, index, 11]).generate_state(1)[0])
        )
        return float(
            rng.uniform(
                self.config.min_amplitude_percent,
                self.config.max_amplitude_percent,
            )
        )

    def _sample_wpm(self, index: int) -> float:
        """Select a reproducible speed for one dataset index."""

        if self.config.min_wpm is None:
            return self.config.wpm
        child_seed = int(
            np.random.SeedSequence([self.seed, index, 1]).generate_state(1)[0]
        )
        rng = np.random.default_rng(child_seed)
        return float(rng.uniform(self.config.min_wpm, self.config.max_wpm))

    def _sample_frequency(self, index: int) -> float:
        """Select a reproducible tone frequency for one dataset index."""

        if self.config.min_frequency_hz is None:
            return self.config.audio.frequency_hz
        child_seed = int(
            np.random.SeedSequence([self.seed, index, 2]).generate_state(1)[0]
        )
        rng = np.random.default_rng(child_seed)
        return float(
            rng.uniform(self.config.min_frequency_hz, self.config.max_frequency_hz)
        )

    def _random_text(self, index: int) -> str:
        child_seed = int(np.random.SeedSequence([self.seed, index]).generate_state(1)[0])
        rng = np.random.default_rng(child_seed)
        character_count = int(
            rng.integers(self.config.min_characters, self.config.max_characters + 1)
        )
        alphabet = SUPPORTED_CHARACTERS
        raw_characters = [str(rng.choice(alphabet)) for _ in range(character_count)]
        characters: list[str] = []
        force_boundary = (
            character_count >= 2
            and rng.random() < self.config.word_boundary_sample_probability
        )
        if force_boundary and character_count >= 4:
            forced_position = int(rng.integers(2, character_count - 1))
        elif force_boundary:
            forced_position = int(rng.integers(1, character_count))
        else:
            forced_position = None
        for position, character in enumerate(raw_characters):
            if position == forced_position:
                characters.append(" ")
            elif (
                not force_boundary
                and 0 < position < character_count
                and rng.random() < self.config.space_probability
            ):
                characters.append(" ")
            characters.append(character)
        return "".join(characters)


def normalize_log_spectrogram(values_db: Tensor, minimum_db: float) -> Tensor:
    """Map a fixed dBFS interval to [0, 1] without per-recording normalization."""

    if minimum_db >= 0.0:
        raise ValueError("Minimum dB value must be negative")
    return ((values_db - minimum_db) / -minimum_db).clamp(0.0, 1.0)


def prepare_spectrogram_features(values: Tensor, config: SpectrogramConfig) -> Tensor:
    """Convert the configured spectrogram scale into model input features."""

    if config.scale == "power":
        return values.clamp_min(0.0)
    return normalize_log_spectrogram(values, config.minimum_db)


def build_tone_activity(
    segments: Sequence[RenderedSegment],
    frame_count: int,
    leading_samples: int,
    spectrogram_config: SpectrogramConfig,
) -> Tensor:
    """Build fractional tone occupancy for every STFT frame."""

    tone_activity = torch.zeros(frame_count, dtype=torch.float32)
    window_length = spectrogram_config.win_length
    hop_length = spectrogram_config.hop_length
    frame_starts = torch.arange(frame_count, dtype=torch.long) * hop_length
    frame_ends = frame_starts + window_length
    for rendered in segments:
        if not rendered.segment.is_tone:
            continue
        start_sample = leading_samples + rendered.start_sample
        end_sample = leading_samples + rendered.end_sample
        overlap = (
            torch.minimum(frame_ends, torch.tensor(end_sample))
            - torch.maximum(frame_starts, torch.tensor(start_sample))
        ).clamp_min(0)
        tone_activity = torch.maximum(
            tone_activity,
            overlap.to(torch.float32) / window_length,
        )
    return tone_activity


def build_audio_sequence_sample(
    text: str,
    config: Stage1DatasetConfig | None = None,
    *,
    wpm: float | None = None,
    frequency_hz: float | None = None,
    timing_seed: int = 0,
    noise_seed: int = 0,
    noise_percent: float | None = None,
    noise_power: float | None = None,
    amplitude_percent: float | None = None,
    fade_depth_percent: float | None = None,
    fade_frequency_hz: float | None = None,
    fade_phase_radians: float = 0.0,
    rise_fall_ms: float | None = None,
    doubled_word_gaps: Sequence[bool] | None = None,
) -> AudioSequenceSample:
    """Create one model-ready clean audio sample from caller-supplied text."""

    selected_config = config or Stage1DatasetConfig()
    normalized_text = normalize_text(text)
    audio_config = replace(
        selected_config.audio,
        **{
            key: value
            for key, value in {
                "frequency_hz": frequency_hz,
                "rise_fall_ms": rise_fall_ms,
            }.items()
            if value is not None
        },
    )
    selected_wpm = selected_config.wpm if wpm is None else wpm
    waveform, rendered_segments = synthesize_morse_with_timing(
        normalized_text,
        selected_wpm,
        audio_config,
        timing_jitter=selected_config.timing_jitter,
        rng=np.random.default_rng(timing_seed),
        doubled_word_gaps=doubled_word_gaps,
    )
    leading_samples = round(
        selected_config.leading_silence_seconds * audio_config.sample_rate
    )
    trailing_samples = round(
        selected_config.trailing_silence_seconds * audio_config.sample_rate
    )
    if leading_samples or trailing_samples:
        waveform = np.concatenate(
            [
                np.zeros(leading_samples, dtype=np.float32),
                waveform,
                np.zeros(trailing_samples, dtype=np.float32),
            ]
        )
    selected_fade_depth = (
        selected_config.fade_depth_percent
        if fade_depth_percent is None
        else fade_depth_percent
    )
    selected_fade_frequency = (
        selected_config.min_fade_frequency_hz
        if fade_frequency_hz is None
        else fade_frequency_hz
    )
    waveform = apply_sinusoidal_fading(
        waveform,
        audio_config.sample_rate,
        selected_fade_depth,
        selected_fade_frequency,
        fade_phase_radians,
    )
    waveform = add_power_scaled_noise(
        waveform,
        audio_config.sample_rate,
        selected_config.noise_power if noise_power is None else noise_power,
        np.random.default_rng(noise_seed),
    )
    waveform = apply_recording_amplitude(
        waveform,
        (
            selected_config.max_amplitude_percent
            if amplitude_percent is None
            else amplitude_percent
        ),
    )
    waveform = add_white_noise(
        waveform,
        selected_config.noise_percent if noise_percent is None else noise_percent,
        np.random.default_rng(noise_seed + 1),
    )
    spectrogram = compute_log_magnitude_stft(
        torch.from_numpy(waveform),
        audio_config.sample_rate,
        selected_config.spectrogram,
    )
    features = prepare_spectrogram_features(
        spectrogram.values,
        selected_config.spectrogram,
    ).transpose(0, 1).contiguous()
    tone_activity = build_tone_activity(
        rendered_segments,
        features.shape[0],
        leading_samples,
        selected_config.spectrogram,
    )
    targets = torch.tensor(
        [int(token) for token in text_to_audio_tokens(normalized_text)],
        dtype=torch.long,
    )
    return AudioSequenceSample(
        spectrogram=features,
        targets=targets,
        tone_activity=tone_activity,
        text=normalized_text,
    )


def build_noise_only_sequence_sample(
    config: Stage1DatasetConfig,
    *,
    duration_seconds: float,
    noise_seed: int,
    noise_power: float,
    amplitude_percent: float,
) -> AudioSequenceSample:
    """Create one negative example containing noise without a Morse signal."""

    sample_count = round(duration_seconds * config.audio.sample_rate)
    waveform = np.zeros(sample_count, dtype=np.float32)
    waveform = add_power_scaled_noise(
        waveform,
        config.audio.sample_rate,
        noise_power,
        np.random.default_rng(noise_seed),
    )
    waveform = apply_recording_amplitude(waveform, amplitude_percent)
    spectrogram = compute_log_magnitude_stft(
        torch.from_numpy(waveform),
        config.audio.sample_rate,
        config.spectrogram,
    )
    features = prepare_spectrogram_features(
        spectrogram.values,
        config.spectrogram,
    ).transpose(0, 1).contiguous()
    return AudioSequenceSample(
        spectrogram=features,
        targets=torch.empty(0, dtype=torch.long),
        tone_activity=torch.zeros(features.shape[0], dtype=torch.float32),
        text="",
    )


def restore_stage1_dataset_config(values: dict[str, Any]) -> Stage1DatasetConfig:
    """Restore nested Stage 1 preprocessing dataclasses saved through ``asdict``."""

    restored = dict(values)
    restored.pop("synthesis_profile", None)
    if "noise_power" not in restored:
        restored["noise_power"] = 0.0
        restored["min_amplitude_percent"] = 100.0
        restored["max_amplitude_percent"] = 100.0
    restored["audio"] = AudioConfig(**restored["audio"])
    spectrogram = dict(restored["spectrogram"])
    if "scale" not in spectrogram:
        spectrogram["scale"] = "log_magnitude"
        spectrogram["frequency_bins"] = None
    restored["spectrogram"] = SpectrogramConfig(**spectrogram)
    return Stage1DatasetConfig(**restored)


def collate_audio_sequences(samples: Sequence[AudioSequenceSample]) -> AudioBatch:
    """Pad spectrograms and frame labels; concatenate event sequences."""

    if not samples:
        raise ValueError("Cannot collate an empty audio batch")
    frequency_bins = samples[0].spectrogram.shape[1]
    if any(sample.spectrogram.shape[1] != frequency_bins for sample in samples):
        raise ValueError("Every spectrogram in a batch must have the same frequency size")
    input_lengths = torch.tensor(
        [sample.input_length for sample in samples], dtype=torch.long
    )
    target_lengths = torch.tensor(
        [sample.target_length for sample in samples], dtype=torch.long
    )
    spectrograms = pad_sequence(
        [sample.spectrogram for sample in samples],
        batch_first=True,
        padding_value=0.0,
    )
    targets = torch.cat([sample.targets for sample in samples])
    tone_activity = pad_sequence(
        [sample.tone_activity for sample in samples],
        batch_first=True,
        padding_value=0.0,
    )
    positions = torch.arange(spectrograms.shape[1]).unsqueeze(0)
    padding_mask = positions >= input_lengths.unsqueeze(1)
    return AudioBatch(
        spectrograms=spectrograms,
        targets=targets,
        tone_activity=tone_activity,
        input_lengths=input_lengths,
        target_lengths=target_lengths,
        padding_mask=padding_mask,
        texts=tuple(sample.text for sample in samples),
    )
