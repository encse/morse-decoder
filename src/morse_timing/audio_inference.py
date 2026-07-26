"""Decode synthesized text or an external PCM WAV with a saved model."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy  # Load the Conda OpenMP runtime before PyTorch on macOS.
import torch
from torch.nn import functional as F

from morse_timing.audio import (
    AudioConfig,
    RenderedSegment,
    add_power_scaled_noise,
    add_white_noise,
    apply_recording_amplitude,
    apply_sinusoidal_fading,
    save_wav,
    synthesize_morse_with_timing,
)
from morse_timing.audio_dataset import (
    Stage1DatasetConfig,
    prepare_spectrogram_features,
    restore_stage1_dataset_config,
)
from morse_timing.audio_model import AudioModelConfig, MorseAudioCTCModel
from morse_timing.audio_tokens import (
    AudioToken,
    audio_tokens_to_morse,
    collapse_ctc_path,
    decode_audio_tokens,
    format_audio_tokens_as_morse,
    normalize_audio_tokens,
    text_to_audio_tokens,
)
from morse_timing.inference_report import (
    CharacterSpan,
    ctc_token_events,
    save_inference_report,
)
from morse_timing.morse import MORSE_TABLE, normalize_text
from morse_timing.spectrogram import (
    Spectrogram,
    compute_log_magnitude_stft,
    load_pcm_wav,
)


@dataclass(frozen=True)
class AudioDecodeResult:
    """Reference and greedy CTC token prediction for synthesized audio."""

    input_text: str
    expected_tokens: tuple[str, ...]
    frame_tokens: tuple[str, ...]
    predicted_tokens: tuple[str, ...]
    normalized_tokens: tuple[str, ...]
    predicted_morse: str
    decoded_text: str
    valid: bool
    exact_tokens: bool
    exact_text: bool
    error: str | None = None


@dataclass(frozen=True)
class WavDecodeResult:
    """Decoded events and text from an external WAV recording."""

    input_path: Path
    source_sample_rate: int
    model_sample_rate: int
    duration_seconds: float
    frame_tokens: tuple[str, ...]
    predicted_tokens: tuple[str, ...]
    normalized_tokens: tuple[str, ...]
    predicted_morse: str
    decoded_text: str
    valid: bool
    error: str | None = None


@dataclass(frozen=True)
class SynthesizedInferenceInput:
    """One deterministic waveform and every representation derived from it."""

    text: str
    waveform: numpy.ndarray
    spectrogram: Spectrogram
    features: torch.Tensor
    expected_tokens: tuple[AudioToken, ...]
    character_spans: tuple[CharacterSpan, ...]
    config: Stage1DatasetConfig


class MorseAudioDecoder:
    """Load an audio checkpoint and run stateful chunked inference."""

    def __init__(
        self,
        model: MorseAudioCTCModel,
        dataset_config: Stage1DatasetConfig,
        training_texts: tuple[str, ...],
        device: torch.device,
        checkpoint_path: Path | None = None,
    ) -> None:
        self.model = model.to(device).eval()
        self.dataset_config = dataset_config
        self.training_texts = training_texts
        self.device = device
        self.checkpoint_path = checkpoint_path

    @classmethod
    def load(
        cls,
        checkpoint_path: str | Path,
        device: str = "auto",
    ) -> MorseAudioDecoder:
        """Restore model weights and preprocessing parameters from a checkpoint."""

        selected_device = _select_device(device)
        checkpoint: dict[str, Any] = torch.load(
            checkpoint_path,
            map_location=selected_device,
            weights_only=True,
        )
        if checkpoint.get("format_version") != 1:
            raise ValueError("Unsupported checkpoint format")
        model = MorseAudioCTCModel(AudioModelConfig(**checkpoint["model_config"]))
        model.load_state_dict(checkpoint["model_state"])
        dataset_config = restore_stage1_dataset_config(checkpoint["dataset_config"])
        return cls(
            model,
            dataset_config,
            tuple(checkpoint.get("texts", ())),
            selected_device,
            Path(checkpoint_path),
        )

    @torch.inference_mode()
    def decode_text(
        self,
        text: str,
        wpm: float | None = None,
        frequency_hz: float | None = None,
        timing_jitter: float | None = None,
        noise_percent: float | None = None,
        fade_depth_percent: float | None = None,
        fade_frequency_hz: float | None = None,
        rise_fall_ms: float | None = None,
        chunk_frames: int = 25,
        noise_power: float | None = None,
        amplitude_percent: float | None = None,
        profile: str = "clean",
        random_seed: int | None = None,
        repetition_count: int = 1,
        noise_gap_seconds: float = 0.0,
        lowpass_cutoff_hz: float | None = None,
    ) -> AudioDecodeResult:
        """Synthesize caller text and greedily decode CTC token output."""

        config = self.effective_config(
            wpm,
            frequency_hz,
            timing_jitter,
            noise_percent,
            fade_depth_percent,
            fade_frequency_hz,
            rise_fall_ms,
            noise_power,
            amplitude_percent,
            profile,
            random_seed,
        )
        sample = self._build_synthesized_input(
            text,
            config,
            repetition_count,
            noise_gap_seconds,
            lowpass_cutoff_hz,
        )
        frame_tokens, predicted, normalized_prediction, morse, decoded_text, valid, error = (
            self._decode_features(sample.features, chunk_frames)
        )
        return AudioDecodeResult(
            input_text=sample.text,
            expected_tokens=tuple(token.name for token in sample.expected_tokens),
            frame_tokens=tuple(token.name for token in frame_tokens),
            predicted_tokens=tuple(token.name for token in predicted),
            normalized_tokens=tuple(token.name for token in normalized_prediction),
            predicted_morse=morse,
            decoded_text=decoded_text,
            valid=valid,
            exact_tokens=predicted == sample.expected_tokens,
            exact_text=decoded_text == sample.text,
            error=error,
        )

    @torch.inference_mode()
    def decode_wav(
        self,
        path: str | Path,
        chunk_frames: int = 25,
    ) -> WavDecodeResult:
        """Load, resample, preprocess, and decode an external PCM WAV file."""

        input_path = Path(path)
        waveform, source_sample_rate = load_pcm_wav(input_path)
        duration_seconds = waveform.numel() / source_sample_rate
        model_sample_rate = self.dataset_config.audio.sample_rate
        waveform = _resample_waveform(
            waveform, source_sample_rate, model_sample_rate
        )
        trailing_samples = round(
            self.dataset_config.trailing_silence_seconds * model_sample_rate
        )
        if trailing_samples:
            waveform = F.pad(waveform, (0, trailing_samples))
        spectrogram = compute_log_magnitude_stft(
            waveform,
            model_sample_rate,
            self.dataset_config.spectrogram,
        )
        features = prepare_spectrogram_features(
            spectrogram.values,
            self.dataset_config.spectrogram,
        ).transpose(0, 1).contiguous()
        frame_tokens, predicted, normalized_prediction, morse, decoded_text, valid, error = (
            self._decode_features(features, chunk_frames)
        )
        return WavDecodeResult(
            input_path=input_path,
            source_sample_rate=source_sample_rate,
            model_sample_rate=model_sample_rate,
            duration_seconds=duration_seconds,
            frame_tokens=tuple(token.name for token in frame_tokens),
            predicted_tokens=tuple(token.name for token in predicted),
            normalized_tokens=tuple(token.name for token in normalized_prediction),
            predicted_morse=morse,
            decoded_text=decoded_text,
            valid=valid,
            error=error,
        )

    def _decode_features(
        self,
        features: torch.Tensor,
        chunk_frames: int,
    ) -> tuple[
        tuple[AudioToken, ...],
        tuple[AudioToken, ...],
        tuple[AudioToken, ...],
        str,
        str,
        bool,
        str | None,
    ]:
        """Run stateful model chunks and parse their joined frame predictions."""

        frame_tokens = tuple(
            AudioToken(value)
            for value in self._predict_frame_tokens(features, chunk_frames)
        )
        predicted = collapse_ctc_path(frame_tokens)
        normalized_prediction = normalize_audio_tokens(predicted)
        try:
            morse = audio_tokens_to_morse(normalized_prediction)
            decoded = decode_audio_tokens(normalized_prediction)
            return (
                frame_tokens,
                predicted,
                normalized_prediction,
                morse,
                decoded.text,
                decoded.is_valid,
                None,
            )
        except ValueError as exception:
            morse = format_audio_tokens_as_morse(normalized_prediction)
            return (
                frame_tokens,
                predicted,
                normalized_prediction,
                morse,
                f"[{morse}]",
                False,
                str(exception),
            )

    def _predict_frame_tokens(
        self,
        features: torch.Tensor,
        chunk_frames: int,
    ) -> list[int]:
        """Predict every frame while retaining LSTM state between chunks."""

        if chunk_frames <= 0:
            raise ValueError("Chunk size must be a positive number of frames")
        if self.model.config.sequence_model != "lstm":
            raise ValueError("Streaming inference requires an LSTM checkpoint")
        frame_tokens: list[int] = []
        state: tuple[torch.Tensor, torch.Tensor] | None = None
        for start in range(0, features.shape[0], chunk_frames):
            chunk = features[start : start + chunk_frames].unsqueeze(0).to(self.device)
            logits, state = self.model.forward_stream(chunk, state)
            frame_tokens.extend(logits[0].argmax(dim=-1).cpu().tolist())
        return frame_tokens

    def effective_config(
        self,
        wpm: float | None,
        frequency_hz: float | None,
        timing_jitter: float | None = None,
        noise_percent: float | None = None,
        fade_depth_percent: float | None = None,
        fade_frequency_hz: float | None = None,
        rise_fall_ms: float | None = None,
        noise_power: float | None = None,
        amplitude_percent: float | None = None,
        profile: str = "clean",
        random_seed: int | None = None,
    ) -> Stage1DatasetConfig:
        """Apply explicit inference overrides to checkpoint preprocessing settings."""

        if profile not in {"clean", "random"}:
            raise ValueError("Inference profile must be 'clean' or 'random'")
        base_config = (
            self._sample_supported_config(random_seed)
            if profile == "random"
            else self.dataset_config
        )
        config = (
            replace(
                base_config,
                wpm=wpm,
                min_wpm=None,
                max_wpm=None,
            )
            if wpm is not None
            else base_config
        )
        if frequency_hz is not None:
            config = replace(
                config,
                min_frequency_hz=None,
                max_frequency_hz=None,
                audio=replace(config.audio, frequency_hz=frequency_hz),
            )
        config = replace(
            config,
            timing_jitter=(
                config.timing_jitter
                if profile == "random" and timing_jitter is None
                else 0.0 if timing_jitter is None else timing_jitter
            ),
            noise_percent=(
                config.noise_percent
                if profile == "random" and noise_percent is None
                else 0.0 if noise_percent is None else noise_percent
            ),
            noise_power=(
                config.noise_power
                if profile == "random" and noise_power is None
                else 0.0 if noise_power is None else noise_power
            ),
            fade_depth_percent=(
                config.fade_depth_percent
                if profile == "random" and fade_depth_percent is None
                else 0.0 if fade_depth_percent is None else fade_depth_percent
            ),
        )
        selected_amplitude = (
            config.max_amplitude_percent
            if profile == "random" and amplitude_percent is None
            else 100.0 if amplitude_percent is None else amplitude_percent
        )
        config = replace(
            config,
            min_amplitude_percent=selected_amplitude,
            max_amplitude_percent=selected_amplitude,
        )
        if fade_frequency_hz is not None:
            config = replace(
                config,
                min_fade_frequency_hz=fade_frequency_hz,
                max_fade_frequency_hz=fade_frequency_hz,
            )
        if rise_fall_ms is not None:
            config = replace(
                config,
                min_rise_fall_ms=rise_fall_ms,
                max_rise_fall_ms=rise_fall_ms,
            )
        config = replace(
            config,
            audio=replace(config.audio, rise_fall_ms=config.min_rise_fall_ms),
        )
        return config

    def _sample_supported_config(
        self,
        random_seed: int | None,
    ) -> Stage1DatasetConfig:
        """Choose one exact signal condition from every checkpoint range."""

        source = self.dataset_config
        rng = numpy.random.default_rng(random_seed)

        def uniform(lower: float, upper: float) -> float:
            return float(rng.uniform(lower, upper)) if upper > lower else float(lower)

        selected_wpm = (
            uniform(source.min_wpm, source.max_wpm)
            if source.min_wpm is not None
            else source.wpm
        )
        selected_frequency = (
            uniform(source.min_frequency_hz, source.max_frequency_hz)
            if source.min_frequency_hz is not None
            else source.audio.frequency_hz
        )
        selected_amplitude = uniform(
            source.min_amplitude_percent,
            source.max_amplitude_percent,
        )
        selected_fade_frequency = uniform(
            source.min_fade_frequency_hz,
            source.max_fade_frequency_hz,
        )
        selected_rise_fall = uniform(
            source.min_rise_fall_ms,
            source.max_rise_fall_ms,
        )
        return replace(
            source,
            wpm=selected_wpm,
            min_wpm=None,
            max_wpm=None,
            min_frequency_hz=None,
            max_frequency_hz=None,
            timing_jitter=uniform(0.0, source.timing_jitter),
            noise_percent=uniform(0.0, source.noise_percent),
            noise_power=uniform(0.0, source.noise_power),
            min_amplitude_percent=selected_amplitude,
            max_amplitude_percent=selected_amplitude,
            fade_depth_percent=uniform(0.0, source.fade_depth_percent),
            min_fade_frequency_hz=selected_fade_frequency,
            max_fade_frequency_hz=selected_fade_frequency,
            min_rise_fall_ms=selected_rise_fall,
            max_rise_fall_ms=selected_rise_fall,
            audio=replace(
                source.audio,
                frequency_hz=selected_frequency,
                rise_fall_ms=selected_rise_fall,
            ),
        )

    def save_input_artifacts(
        self,
        text: str,
        wpm: float | None,
        frequency_hz: float | None,
        output_directory: Path,
        timing_jitter: float | None = None,
        noise_percent: float | None = None,
        fade_depth_percent: float | None = None,
        fade_frequency_hz: float | None = None,
        rise_fall_ms: float | None = None,
        noise_power: float | None = None,
        amplitude_percent: float | None = None,
        chunk_frames: int = 25,
        profile: str = "clean",
        random_seed: int | None = None,
        output_path: Path | None = None,
        repetition_count: int = 1,
        noise_gap_seconds: float = 0.0,
        lowpass_cutoff_hz: float | None = None,
    ) -> tuple[Path, Path]:
        """Save the exact synthesized audio and its visual inference report."""

        config = self.effective_config(
            wpm,
            frequency_hz,
            timing_jitter,
            noise_percent,
            fade_depth_percent,
            fade_frequency_hz,
            rise_fall_ms,
            noise_power,
            amplitude_percent,
            profile,
            random_seed,
        )
        sample = self._build_synthesized_input(
            text,
            config,
            repetition_count,
            noise_gap_seconds,
            lowpass_cutoff_hz,
        )
        (
            frame_tokens,
            _,
            _,
            _,
            decoded_text,
            _,
            _,
        ) = self._decode_features(sample.features, chunk_frames)
        normalized = sample.text
        safe_text = re.sub(r"[^a-z0-9]+", "-", normalized.lower()).strip("-")
        safe_text = safe_text[:80] or "morse"
        noise_label = (
            f"{config.noise_power:g}noise-power-"
            f"{config.max_amplitude_percent:g}pct-amplitude"
        )
        if config.noise_percent > 0.0:
            noise_label += f"-{config.noise_percent:g}pct-noise"
        stem = (
            f"{safe_text}-{config.wpm:g}wpm-"
            f"{config.audio.frequency_hz:g}hz-"
            f"{config.timing_jitter * 100:g}pct-jitter"
            f"-{noise_label}"
        )
        if config.audio.rise_fall_ms != 5.0:
            stem += f"-{config.audio.rise_fall_ms:g}ms-edges"
        if config.fade_depth_percent > 0.0:
            stem += (
                f"-{config.fade_depth_percent:g}pct-fade"
                f"-{config.min_fade_frequency_hz:g}hz-fade"
            )
        if output_path is None:
            wav_path = output_directory / f"{stem}.wav"
            image_path = output_directory / f"{stem}.png"
        else:
            if output_path.suffix.lower() not in {"", ".png"}:
                raise ValueError("Custom analysis output must use a .png extension")
            image_path = (
                output_path
                if output_path.suffix
                else output_path.with_suffix(".png")
            )
            wav_path = image_path.with_suffix(".wav")
        save_wav(wav_path, sample.waveform, config.audio.sample_rate)
        events = ctc_token_events(
            frame_tokens,
            sample.spectrogram.times_seconds.cpu().tolist(),
        )
        save_inference_report(
            image_path,
            spectrogram=sample.spectrogram,
            character_spans=sample.character_spans,
            token_events=events,
            checkpoint_path=self.checkpoint_path,
            architecture=self._architecture_label(),
            decoded_text=decoded_text,
            parameters=(
                ("Profile", profile),
                *(
                    (("Seed", str(random_seed)),)
                    if profile == "random" and random_seed is not None
                    else ()
                ),
                ("WPM", f"{config.wpm:g}"),
                ("Tone", f"{config.audio.frequency_hz:g} Hz"),
                ("Jitter", f"{config.timing_jitter * 100:g}%"),
                ("Noise power", f"{config.noise_power:g}"),
                ("Amplitude", f"{config.max_amplitude_percent:g}%"),
                ("Fade", f"{config.fade_depth_percent:g}%"),
                ("Fade rate", f"{config.min_fade_frequency_hz:g} Hz"),
                ("Edges", f"{config.audio.rise_fall_ms:g} ms"),
                ("Sample rate", f"{config.audio.sample_rate:g} Hz"),
                ("Chunk", f"{chunk_frames} frames"),
                ("Repetitions", str(repetition_count)),
                ("Noise gap", f"{noise_gap_seconds:g} s"),
                (
                    "Low-pass",
                    (
                        "off"
                        if lowpass_cutoff_hz is None
                        else f"0–{lowpass_cutoff_hz:g} Hz"
                    ),
                ),
            ),
        )
        return wav_path, image_path

    def _build_synthesized_input(
        self,
        text: str,
        config: Stage1DatasetConfig,
        repetition_count: int = 1,
        noise_gap_seconds: float = 0.0,
        lowpass_cutoff_hz: float | None = None,
    ) -> SynthesizedInferenceInput:
        """Create the one deterministic signal used for decoding and reporting."""

        if repetition_count < 1:
            raise ValueError("Repetition count must be positive")
        if not numpy.isfinite(noise_gap_seconds) or noise_gap_seconds < 0.0:
            raise ValueError("Noise gap duration must be finite and non-negative")

        source_text = normalize_text(text)
        normalized = " ".join([source_text] * repetition_count)
        timing_rng = numpy.random.default_rng(0)
        waveform_chunks: list[numpy.ndarray] = []
        shifted_segments: list[RenderedSegment] = []
        gap_ranges: list[tuple[int, int]] = []
        sample_offset = 0
        gap_samples = round(noise_gap_seconds * config.audio.sample_rate)
        for repetition_index in range(repetition_count):
            repetition_waveform, repetition_segments = synthesize_morse_with_timing(
                source_text,
                config.wpm,
                config.audio,
                timing_jitter=config.timing_jitter,
                rng=timing_rng,
            )
            waveform_chunks.append(repetition_waveform)
            shifted_segments.extend(
                RenderedSegment(
                    rendered.segment,
                    rendered.start_sample + sample_offset,
                    rendered.end_sample + sample_offset,
                )
                for rendered in repetition_segments
            )
            sample_offset += repetition_waveform.size
            if repetition_index < repetition_count - 1 and gap_samples:
                gap_ranges.append((sample_offset, sample_offset + gap_samples))
                waveform_chunks.append(
                    numpy.zeros(gap_samples, dtype=numpy.float32)
                )
                sample_offset += gap_samples
        waveform = numpy.concatenate(waveform_chunks)
        rendered_segments = tuple(shifted_segments)

        leading_samples = round(
            config.leading_silence_seconds * config.audio.sample_rate
        )
        trailing_samples = round(
            config.trailing_silence_seconds * config.audio.sample_rate
        )
        if leading_samples or trailing_samples:
            waveform = numpy.concatenate(
                (
                    numpy.zeros(leading_samples, dtype=numpy.float32),
                    waveform,
                    numpy.zeros(trailing_samples, dtype=numpy.float32),
                )
            )

        waveform = apply_sinusoidal_fading(
            waveform,
            config.audio.sample_rate,
            config.fade_depth_percent,
            config.min_fade_frequency_hz,
        )
        waveform = add_power_scaled_noise(
            waveform,
            config.audio.sample_rate,
            config.noise_power,
            numpy.random.default_rng(0),
        )
        gap_noise_power = max(config.min_noise_only_power, config.noise_power)
        gap_rng = numpy.random.default_rng(2)
        for gap_start, gap_end in gap_ranges:
            waveform[leading_samples + gap_start : leading_samples + gap_end] = (
                add_power_scaled_noise(
                    numpy.zeros(gap_end - gap_start, dtype=numpy.float32),
                    config.audio.sample_rate,
                    gap_noise_power,
                    gap_rng,
                )
            )
        waveform = apply_recording_amplitude(
            waveform,
            config.max_amplitude_percent,
        )
        waveform = add_white_noise(
            waveform,
            config.noise_percent,
            numpy.random.default_rng(1),
        )
        if lowpass_cutoff_hz is not None:
            frequencies = numpy.fft.rfftfreq(
                waveform.size,
                d=1.0 / config.audio.sample_rate,
            )
            spectrum = numpy.fft.rfft(waveform)
            spectrum[frequencies > lowpass_cutoff_hz] = 0.0
            waveform = numpy.fft.irfft(
                spectrum,
                n=waveform.size,
            ).astype(numpy.float32)
        spectrogram = compute_log_magnitude_stft(
            torch.from_numpy(waveform),
            config.audio.sample_rate,
            config.spectrogram,
        )
        features = prepare_spectrogram_features(
            spectrogram.values,
            config.spectrogram,
        ).transpose(0, 1).contiguous()
        return SynthesizedInferenceInput(
            text=normalized,
            waveform=waveform,
            spectrogram=spectrogram,
            features=features,
            expected_tokens=text_to_audio_tokens(normalized),
            character_spans=_character_spans(
                normalized,
                rendered_segments,
                leading_samples,
                config.audio.sample_rate,
            ),
            config=config,
        )

    def _architecture_label(self) -> str:
        """Describe the inference-relevant model stack."""

        config = self.model.config
        return (
            f"{config.frequency_bins} STFT bins → "
            f"{config.dense_layers}× dense {config.projection_size} → "
            f"{config.num_lstm_layers}× LSTM {config.hidden_size} → "
            f"{len(AudioToken)} tokens"
        )


def _character_spans(
    text: str,
    rendered_segments: tuple[RenderedSegment, ...],
    leading_samples: int,
    sample_rate: int,
) -> tuple[CharacterSpan, ...]:
    """Map intended characters to the first and last tone of each letter."""

    spans: list[CharacterSpan] = []
    segment_index = 0
    for character in text:
        if character == " ":
            continue
        symbol_count = len(MORSE_TABLE[character])
        first_tone = rendered_segments[segment_index]
        last_tone = rendered_segments[segment_index + 2 * (symbol_count - 1)]
        spans.append(
            CharacterSpan(
                character=character,
                start_seconds=(
                    leading_samples + first_tone.start_sample
                ) / sample_rate,
                end_seconds=(
                    leading_samples + last_tone.end_sample
                ) / sample_rate,
            )
        )
        segment_index += 2 * symbol_count
    if segment_index != len(rendered_segments):
        raise RuntimeError("Rendered Morse segments do not match the input text")
    return tuple(spans)


def _select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resample_waveform(
    waveform: torch.Tensor,
    source_sample_rate: int,
    target_sample_rate: int,
) -> torch.Tensor:
    """Resample mono audio with PyTorch's antialiased interpolation."""

    if source_sample_rate <= 0 or target_sample_rate <= 0:
        raise ValueError("Sample rates must be positive")
    if source_sample_rate == target_sample_rate:
        return waveform.to(dtype=torch.float32)
    target_samples = max(
        1, round(waveform.numel() * target_sample_rate / source_sample_rate)
    )
    values = waveform.to(dtype=torch.float32).reshape(1, 1, 1, -1)
    return F.interpolate(
        values,
        size=(1, target_samples),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).reshape(-1)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the synthesized-text and WAV inference CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("text", nargs="?", help="Text to synthesize and decode")
    parser.add_argument("--wav", type=Path, help="Decode an external PCM WAV file")
    parser.add_argument("--wpm", type=float, help="Override checkpoint WPM")
    parser.add_argument(
        "--frequency",
        type=float,
        help="Override checkpoint tone frequency in Hz",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--timing-jitter", type=float)
    parser.add_argument("--noise-percent", type=float)
    parser.add_argument("--noise-power", type=float)
    parser.add_argument("--amplitude-percent", type=float)
    parser.add_argument("--fade-depth-percent", type=float)
    parser.add_argument("--fade-frequency", type=float)
    parser.add_argument("--rise-fall-ms", type=float)
    parser.add_argument(
        "--profile",
        choices=("clean", "random"),
        default="clean",
        help="Use clean defaults or sample one condition from checkpoint ranges",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Reproduce values selected by the random inference profile",
    )
    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=25,
        help="STFT frames per stateful LSTM call (25 frames = 500 ms by default)",
    )
    parser.add_argument("--artifacts-directory", type=Path, default=Path("audio"))
    parser.add_argument(
        "--output",
        type=Path,
        help="Exact analysis PNG path; the WAV uses the same filename stem",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Number of times to synthesize the input text",
    )
    parser.add_argument(
        "--noise-gap-seconds",
        type=float,
        default=0.0,
        help="Duration of noise inserted between repeated texts",
    )
    parser.add_argument(
        "--lowpass-cutoff-hz",
        type=float,
        help="Remove synthesized-input frequencies above this cutoff",
    )
    parser.add_argument("--list-training-texts", action="store_true")
    return parser



def main(argv: list[str] | None = None) -> None:
    """Load a checkpoint and print an inference result."""

    args = build_argument_parser().parse_args(argv)
    decoder = MorseAudioDecoder.load(args.checkpoint, args.device)

    wav_path = None
    image_path = None
    random_seed = args.seed

    if args.wav is not None:
        if args.text is not None:
            raise ValueError("Text and --wav cannot be used together")

        result = decoder.decode_wav(args.wav, args.chunk_frames)

        input_lines = [
            f"input_wav={result.input_path}",
            (
                f"source_sample_rate={result.source_sample_rate} "
                f"model_sample_rate={result.model_sample_rate} "
                f"duration={result.duration_seconds:.3f}s"
            ),
        ]
        validity_line = f"valid={result.valid}"

    else:
        if args.list_training_texts:
            for index, text in enumerate(decoder.training_texts):
                print(f"{index}: {text}")

            if args.text is None:
                return

        if args.text is None:
            raise ValueError(
                "Text is required unless --list-training-texts is used"
            )

        if args.profile == "random" and random_seed is None:
            random_seed = int(
                numpy.random.SeedSequence().generate_state(1)[0]
            )

        result = decoder.decode_text(
            args.text,
            args.wpm,
            args.frequency,
            args.timing_jitter,
            args.noise_percent,
            args.fade_depth_percent,
            args.fade_frequency,
            args.rise_fall_ms,
            args.chunk_frames,
            noise_power=args.noise_power,
            amplitude_percent=args.amplitude_percent,
            profile=args.profile,
            random_seed=random_seed,
            repetition_count=args.repetitions,
            noise_gap_seconds=args.noise_gap_seconds,
            lowpass_cutoff_hz=args.lowpass_cutoff_hz,
        )

        wav_path, image_path = decoder.save_input_artifacts(
            args.text,
            args.wpm,
            args.frequency,
            args.artifacts_directory,
            args.timing_jitter,
            args.noise_percent,
            args.fade_depth_percent,
            args.fade_frequency,
            args.rise_fall_ms,
            noise_power=args.noise_power,
            amplitude_percent=args.amplitude_percent,
            chunk_frames=args.chunk_frames,
            profile=args.profile,
            random_seed=random_seed,
            output_path=args.output,
            repetition_count=args.repetitions,
            noise_gap_seconds=args.noise_gap_seconds,
            lowpass_cutoff_hz=args.lowpass_cutoff_hz,
        )

        profile_suffix = (
            f" seed={random_seed}" if random_seed is not None else ""
        )

        input_lines = [
            f"input_text={result.input_text!r}",
            f"profile={args.profile}{profile_suffix}",
            f"expected_tokens={' '.join(result.expected_tokens)}",
        ]
        validity_line = (
            f"valid={result.valid} "
            f"exact_tokens={result.exact_tokens} "
            f"exact_text={result.exact_text}"
        )

    for line in input_lines:
        print(line)

    print(f"stream_chunk_frames={args.chunk_frames}")
    # print(f"predicted_tokens={' '.join(result.predicted_tokens)}")

    # if result.normalized_tokens != result.predicted_tokens:
    #     print(f"normalized_tokens={' '.join(result.normalized_tokens)}")

    # print(f"predicted_morse={result.predicted_morse}")
    print(f"decoded_text={result.decoded_text!r}")
    print(validity_line)

    if result.error:
        print(f"error={result.error}")

    if wav_path is not None:
        print(f"wav={wav_path}")

    if image_path is not None:
        print(f"analysis={image_path}")


if __name__ == "__main__":
    main()
