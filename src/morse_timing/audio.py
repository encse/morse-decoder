"""Generate clean, audible Morse code as a mono PCM WAV file."""

from __future__ import annotations

import argparse
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from morse_timing.morse import MORSE_TABLE, encode_text, normalize_text


@dataclass(frozen=True)
class AudioConfig:
    """Parameters for clean Morse waveform synthesis."""

    sample_rate: int = 8_000
    frequency_hz: float = 700.0
    amplitude: float = 0.5
    rise_fall_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
        if self.frequency_hz <= 0.0 or self.frequency_hz >= self.sample_rate / 2:
            raise ValueError("Tone frequency must be between zero and the Nyquist frequency")
        if not 0.0 < self.amplitude <= 1.0:
            raise ValueError("Amplitude must be in the interval (0, 1]")
        if self.rise_fall_ms < 0.0:
            raise ValueError("Rise/fall time cannot be negative")


@dataclass(frozen=True)
class AudioSegment:
    """One ideal tone or silence segment measured in Morse timing units."""

    is_tone: bool
    units: int


@dataclass(frozen=True)
class RenderedSegment:
    """One synthesized segment with exact sample boundaries."""

    segment: AudioSegment
    start_sample: int
    end_sample: int


def text_to_segments(text: str) -> tuple[AudioSegment, ...]:
    """Convert text to ideal 1:3:7 Morse tone and gap segments."""

    normalized = normalize_text(text)
    words = normalized.split(" ")
    segments: list[AudioSegment] = []
    for word_index, word in enumerate(words):
        for character_index, character in enumerate(word):
            symbols = MORSE_TABLE[character]
            for symbol_index, symbol in enumerate(symbols):
                segments.append(AudioSegment(is_tone=True, units=1 if symbol == "." else 3))
                if symbol_index < len(symbols) - 1:
                    segments.append(AudioSegment(is_tone=False, units=1))
            if character_index < len(word) - 1:
                segments.append(AudioSegment(is_tone=False, units=3))
            elif word_index < len(words) - 1:
                segments.append(AudioSegment(is_tone=False, units=7))
            else:
                segments.append(AudioSegment(is_tone=False, units=3))
    return tuple(segments)


def synthesize_morse(
    text: str,
    wpm: float,
    config: AudioConfig | None = None,
    *,
    timing_jitter: float = 0.0,
    rng: np.random.Generator | None = None,
    carrier_phase_radians: float | None = None,
) -> NDArray[np.float32]:
    """Synthesize clean Morse audio using the standard 1.2/WPM dit duration."""

    waveform, _ = synthesize_morse_with_timing(
        text,
        wpm,
        config,
        timing_jitter=timing_jitter,
        rng=rng,
        carrier_phase_radians=carrier_phase_radians,
    )
    return waveform


def synthesize_morse_with_timing(
    text: str,
    wpm: float,
    config: AudioConfig | None = None,
    *,
    timing_jitter: float = 0.0,
    rng: np.random.Generator | None = None,
    carrier_phase_radians: float | None = None,
) -> tuple[NDArray[np.float32], tuple[RenderedSegment, ...]]:
    """Synthesize Morse and return exact boundaries for supervised events."""

    if not np.isfinite(wpm) or wpm <= 0.0:
        raise ValueError("WPM must be finite and positive")
    if not np.isfinite(timing_jitter) or timing_jitter < 0.0:
        raise ValueError("Timing jitter must be finite and non-negative")
    selected_config = config or AudioConfig()
    selected_rng = rng or np.random.default_rng(0)
    initial_phase = (
        float(selected_rng.uniform(0.0, 2.0 * np.pi))
        if carrier_phase_radians is None
        else carrier_phase_radians
    )
    if not np.isfinite(initial_phase):
        raise ValueError("Carrier phase must be finite")
    unit_seconds = 1.2 / wpm
    chunks: list[NDArray[np.float32]] = []
    rendered_segments: list[RenderedSegment] = []
    sample_offset = 0
    base_samples = unit_seconds * selected_config.sample_rate

    def dot_samples(multiplier: int = 1) -> int:
        scale = (
            float(np.clip(selected_rng.normal(1.0, timing_jitter), 0.5, 2.0))
            if timing_jitter > 0.0
            else 1.0
        )
        return max(1, round(multiplier * base_samples * scale))

    for segment in text_to_segments(text):
        sample_count = dot_samples(segment.units)
        if segment.is_tone:
            chunks.append(
                _tone(
                    sample_count,
                    selected_config,
                    sample_offset,
                    initial_phase,
                )
            )
        else:
            chunks.append(np.zeros(sample_count, dtype=np.float32))
        rendered_segments.append(
            RenderedSegment(segment, sample_offset, sample_offset + sample_count)
        )
        sample_offset += sample_count
    return np.concatenate(chunks), tuple(rendered_segments)


def add_white_noise(
    samples: NDArray[np.float32],
    noise_percent: float,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float32]:
    """Add Gaussian noise whose RMS is a percentage of active signal RMS."""

    if not np.isfinite(noise_percent) or noise_percent < 0.0:
        raise ValueError("Noise percentage must be finite and non-negative")
    if noise_percent == 0.0:
        return samples.copy()
    active = samples[np.abs(samples) > 1e-6]
    if active.size == 0:
        raise ValueError("Cannot scale noise relative to a silent signal")
    signal_rms = float(np.sqrt(np.mean(active.astype(np.float64) ** 2)))
    noise_rms = signal_rms * noise_percent / 100.0
    selected_rng = rng or np.random.default_rng(0)
    noise = selected_rng.normal(0.0, noise_rms, samples.shape).astype(np.float32)
    return np.clip(samples + noise, -1.0, 1.0).astype(np.float32)


def add_power_scaled_noise(
    samples: NDArray[np.float32],
    sample_rate: int,
    noise_power: float,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float32]:
    """Add Gaussian noise using sample-rate-normalized noise power."""

    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive")
    if not np.isfinite(noise_power) or noise_power < 0.0:
        raise ValueError("Noise power must be finite and non-negative")
    if noise_power == 0.0:
        return samples.copy()
    variance = 1e-6 * noise_power * sample_rate / 2.0
    selected_rng = rng or np.random.default_rng(0)
    noise = selected_rng.normal(0.0, np.sqrt(variance), samples.shape)
    return samples + noise.astype(np.float32)


def apply_recording_amplitude(
    samples: NDArray[np.float32], amplitude_percent: float
) -> NDArray[np.float32]:
    """Apply recording-wide gain and clip the resulting waveform."""

    if not np.isfinite(amplitude_percent) or amplitude_percent < 0.0:
        raise ValueError("Amplitude percentage must be finite and non-negative")
    return np.clip(samples * amplitude_percent / 100.0, -1.0, 1.0).astype(np.float32)


def apply_sinusoidal_fading(
    samples: NDArray[np.float32],
    sample_rate: int,
    depth_percent: float,
    frequency_hz: float,
    phase_radians: float = 0.0,
) -> NDArray[np.float32]:
    """Apply sinusoidal fading with an envelope from full to reduced amplitude."""

    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive")
    if not np.isfinite(depth_percent) or not 0.0 <= depth_percent <= 100.0:
        raise ValueError("Fade depth must be between zero and 100 percent")
    if not np.isfinite(frequency_hz) or frequency_hz <= 0.0:
        raise ValueError("Fade frequency must be finite and positive")
    if depth_percent == 0.0:
        return samples.copy()
    time = np.arange(samples.size, dtype=np.float64) / sample_rate
    depth = depth_percent / 100.0
    envelope = 1.0 - depth * (1.0 + np.sin(
        2.0 * np.pi * frequency_hz * time + phase_radians
    )) / 2.0
    return (samples * envelope).astype(np.float32)


def _tone(
    sample_count: int,
    config: AudioConfig,
    start_sample: int,
    initial_phase: float,
) -> NDArray[np.float32]:
    """Gate one segment from a recording-wide continuous carrier oscillator."""

    sample_positions = start_sample + np.arange(sample_count, dtype=np.float64)
    phase = (
        2.0 * np.pi * config.frequency_hz * sample_positions / config.sample_rate
        + initial_phase
    )
    samples = np.sin(phase)
    fade_samples = min(
        round(config.rise_fall_ms * config.sample_rate / 1_000.0),
        sample_count // 2,
    )
    if fade_samples >= 2:
        edge_phase = np.linspace(0.0, np.pi / 2.0, fade_samples, dtype=np.float64)
        ramp = np.sin(edge_phase) ** 2
        envelope = np.ones(sample_count, dtype=np.float64)
        envelope[:fade_samples] = ramp
        envelope[-fade_samples:] = ramp[::-1]
        samples *= envelope
    return (samples * config.amplitude).astype(np.float32)


def save_wav(
    path: str | Path,
    samples: NDArray[np.float32],
    sample_rate: int,
) -> None:
    """Save normalized floating-point samples as mono 16-bit PCM WAV."""

    if samples.ndim != 1 or len(samples) == 0:
        raise ValueError("Audio samples must be a non-empty one-dimensional array")
    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.rint(np.clip(samples, -1.0, 1.0) * 32_767.0).astype("<i2")
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for clean audio generation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="Text to encode as Morse audio")
    parser.add_argument("--wpm", type=float, required=True, help="Morse speed in WPM")
    parser.add_argument("--output", type=Path, required=True, help="Output WAV path")
    parser.add_argument("--frequency", type=float, default=700.0, help="Tone frequency in Hz")
    parser.add_argument("--sample-rate", type=int, default=8_000, help="WAV sample rate")
    parser.add_argument("--amplitude", type=float, default=0.5, help="Peak amplitude in (0, 1]")
    parser.add_argument(
        "--rise-fall-ms",
        type=float,
        default=5.0,
        help="Raised-cosine tone edge duration in milliseconds",
    )
    parser.add_argument("--timing-jitter", type=float, default=0.0)
    parser.add_argument("--noise-percent", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Generate a WAV file from command-line arguments."""

    args = build_argument_parser().parse_args(argv)
    config = AudioConfig(
        sample_rate=args.sample_rate,
        frequency_hz=args.frequency,
        amplitude=args.amplitude,
        rise_fall_ms=args.rise_fall_ms,
    )
    normalized_text = normalize_text(args.text)
    samples = synthesize_morse(
        normalized_text,
        args.wpm,
        config,
        timing_jitter=args.timing_jitter,
        rng=np.random.default_rng(args.seed),
    )
    samples = add_white_noise(
        samples,
        args.noise_percent,
        np.random.default_rng(args.seed + 1),
    )
    save_wav(args.output, samples, config.sample_rate)
    print(f"text={normalized_text}")
    print(f"morse={encode_text(normalized_text)}")
    print(
        f"saved={args.output} duration={len(samples) / config.sample_rate:.3f}s "
        f"sample_rate={config.sample_rate}Hz frequency={config.frequency_hz:g}Hz "
        f"wpm={args.wpm:g}"
    )


if __name__ == "__main__":
    main()
