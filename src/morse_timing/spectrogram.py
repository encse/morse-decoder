"""Load PCM WAV audio and create log-magnitude STFT spectrograms."""

from __future__ import annotations

import argparse
import os
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

_MATPLOTLIB_CACHE = Path(tempfile.gettempdir()) / "morse-timing-matplotlib"
_MATPLOTLIB_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MATPLOTLIB_CACHE))

from torch import Tensor


@dataclass(frozen=True)
class SpectrogramConfig:
    """STFT parameters shared by feature extraction and visualization."""

    n_fft: int = 160
    hop_length: int = 160
    win_length: int = 160
    minimum_db: float = -80.0
    scale: str = "power"
    frequency_bins: int | None = 65

    def __post_init__(self) -> None:
        if self.n_fft <= 0:
            raise ValueError("FFT size must be positive")
        if self.hop_length <= 0:
            raise ValueError("Hop length must be positive")
        if self.win_length <= 0 or self.win_length > self.n_fft:
            raise ValueError("Window length must be positive and no larger than FFT size")
        if self.minimum_db >= 0.0:
            raise ValueError("Minimum dB value must be negative")
        if self.scale not in {"power", "log_magnitude"}:
            raise ValueError("Spectrogram scale must be power or log_magnitude")
        maximum_bins = self.n_fft // 2 + 1
        if self.frequency_bins is not None and not 1 <= self.frequency_bins <= maximum_bins:
            raise ValueError("Frequency-bin count must fit inside the FFT output")


@dataclass(frozen=True)
class Spectrogram:
    """A model spectrogram and its physical coordinate axes."""

    values: Tensor
    frequencies_hz: Tensor
    times_seconds: Tensor
    sample_rate: int
    duration_seconds: float
    scale: str
    minimum_db: float


def load_pcm_wav(path: str | Path) -> tuple[Tensor, int]:
    """Load an uncompressed PCM WAV file and downmix it to mono float32."""

    input_path = Path(path)
    try:
        with wave.open(str(input_path), "rb") as wav_file:
            if wav_file.getcomptype() != "NONE":
                raise ValueError("Only uncompressed PCM WAV files are supported")
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            raw_frames = wav_file.readframes(frame_count)
    except (wave.Error, EOFError) as error:
        raise ValueError(f"Invalid WAV file: {input_path}") from error

    if channels <= 0 or sample_rate <= 0 or frame_count <= 0:
        raise ValueError("WAV file must contain non-empty audio with a valid sample rate")
    decoded = _decode_pcm(raw_frames, sample_width)
    if decoded.size != frame_count * channels:
        raise ValueError("WAV frame data has an unexpected size")
    mono = decoded.reshape(frame_count, channels).mean(axis=1, dtype=np.float32)
    return torch.from_numpy(mono.copy()), sample_rate


def _decode_pcm(raw_frames: bytes, sample_width: int) -> np.ndarray:
    """Decode little-endian 8-, 16-, 24-, or 32-bit integer PCM samples."""

    if sample_width == 1:
        values = np.frombuffer(raw_frames, dtype=np.uint8).astype(np.float32)
        return (values - 128.0) / 128.0
    if sample_width == 2:
        values = np.frombuffer(raw_frames, dtype="<i2").astype(np.float32)
        return values / 32_768.0
    if sample_width == 3:
        bytes_by_sample = np.frombuffer(raw_frames, dtype=np.uint8).reshape(-1, 3)
        values = (
            bytes_by_sample[:, 0].astype(np.int32)
            | (bytes_by_sample[:, 1].astype(np.int32) << 8)
            | (bytes_by_sample[:, 2].astype(np.int32) << 16)
        )
        values = (values ^ 0x800000) - 0x800000
        return values.astype(np.float32) / 8_388_608.0
    if sample_width == 4:
        values = np.frombuffer(raw_frames, dtype="<i4").astype(np.float32)
        return values / 2_147_483_648.0
    raise ValueError(f"Unsupported PCM sample width: {sample_width} bytes")


def compute_log_magnitude_stft(
    samples: Tensor,
    sample_rate: int,
    config: SpectrogramConfig | None = None,
) -> Spectrogram:
    """Compute a Hann-windowed power or log-magnitude STFT."""

    selected_config = config or SpectrogramConfig()
    if samples.ndim != 1 or samples.numel() == 0:
        raise ValueError("Audio must be a non-empty one-dimensional tensor")
    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive")
    audio = samples.to(dtype=torch.float32)
    duration_seconds = audio.numel() / sample_rate
    if audio.numel() < selected_config.n_fft:
        audio = torch.nn.functional.pad(audio, (0, selected_config.n_fft - audio.numel()))

    window = torch.hann_window(
        selected_config.win_length,
        periodic=True,
        dtype=audio.dtype,
        device=audio.device,
    )
    complex_stft = torch.stft(
        audio,
        n_fft=selected_config.n_fft,
        hop_length=selected_config.hop_length,
        win_length=selected_config.win_length,
        window=window,
        center=False,
        return_complex=True,
    )
    coherent_gain = window.sum() / 2.0
    magnitude = complex_stft.abs() / coherent_gain
    if selected_config.scale == "power":
        values = magnitude.square()
    else:
        floor = 10.0 ** (selected_config.minimum_db / 20.0)
        values = 20.0 * torch.log10(magnitude.clamp_min(floor))
        values = values.clamp_min(selected_config.minimum_db)
    frequencies = torch.fft.rfftfreq(
        selected_config.n_fft,
        d=1.0 / sample_rate,
        device=audio.device,
    )
    if selected_config.frequency_bins is not None:
        values = values[: selected_config.frequency_bins]
        frequencies = frequencies[: selected_config.frequency_bins]
    frame_count = complex_stft.shape[1]
    times = (
        torch.arange(frame_count, dtype=torch.float32, device=audio.device)
        * selected_config.hop_length
        + selected_config.win_length / 2.0
    ) / sample_rate
    return Spectrogram(
        values=values,
        frequencies_hz=frequencies,
        times_seconds=times,
        sample_rate=sample_rate,
        duration_seconds=duration_seconds,
        scale=selected_config.scale,
        minimum_db=selected_config.minimum_db,
    )


def save_spectrogram_image(
    spectrogram: Spectrogram,
    output_path: str | Path,
    *,
    title: str = "STFT spectrogram",
    maximum_frequency_hz: float | None = 2_000.0,
) -> None:
    """Render a spectrogram with time, frequency, and dBFS axes to a PNG file."""

    nyquist = spectrogram.sample_rate / 2.0
    upper_frequency = nyquist if maximum_frequency_hz is None else maximum_frequency_hz
    if upper_frequency <= 0.0 or upper_frequency > nyquist:
        raise ValueError(f"Maximum frequency must be in (0, {nyquist:g}] Hz")

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(12.0, 4.8), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.add_subplot(1, 1, 1)
    display_values = (
        10.0
        * torch.log10(
            spectrogram.values.clamp_min(10.0 ** (spectrogram.minimum_db / 10.0))
        )
        if spectrogram.scale == "power"
        else spectrogram.values
    )
    image = axes.pcolormesh(
        spectrogram.times_seconds.cpu().numpy(),
        spectrogram.frequencies_hz.cpu().numpy(),
        display_values.cpu().numpy(),
        shading="auto",
        cmap="magma",
        vmin=float(display_values.min()),
        vmax=max(0.0, float(display_values.max())),
    )
    axes.set_title(title)
    axes.set_xlabel("Time (seconds)")
    axes.set_ylabel("Frequency (Hz)")
    axes.set_xlim(0.0, spectrogram.duration_seconds)
    axes.set_ylim(0.0, upper_frequency)
    colorbar = figure.colorbar(image, ax=axes)
    colorbar.set_label("Magnitude (dBFS)")
    figure.savefig(output, dpi=150)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for spectrogram visualization."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input PCM WAV file")
    parser.add_argument("--output", type=Path, help="Output PNG path")
    parser.add_argument("--n-fft", type=int, default=160, help="FFT size in samples")
    parser.add_argument("--hop-length", type=int, default=160, help="Frame hop in samples")
    parser.add_argument("--win-length", type=int, default=160, help="Window size in samples")
    parser.add_argument("--minimum-db", type=float, default=-80.0, help="Display floor in dBFS")
    parser.add_argument("--scale", choices=("power", "log_magnitude"), default="power")
    parser.add_argument("--frequency-bins", type=int, default=65)
    parser.add_argument(
        "--maximum-frequency",
        type=float,
        default=2_000.0,
        help="Highest displayed frequency in Hz",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Load a WAV file, calculate its STFT, and save a visualization."""

    args = build_argument_parser().parse_args(argv)
    output = args.output or args.input.with_suffix(".spectrogram.png")
    config = SpectrogramConfig(
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        minimum_db=args.minimum_db,
        scale=args.scale,
        frequency_bins=args.frequency_bins,
    )
    samples, sample_rate = load_pcm_wav(args.input)
    spectrogram = compute_log_magnitude_stft(samples, sample_rate, config)
    save_spectrogram_image(
        spectrogram,
        output,
        title=f"{args.scale} STFT: {args.input.name}",
        maximum_frequency_hz=args.maximum_frequency,
    )
    frequency_resolution = sample_rate / config.n_fft
    time_resolution_ms = 1_000.0 * config.hop_length / sample_rate
    print(
        f"input={args.input} duration={spectrogram.duration_seconds:.3f}s "
        f"sample_rate={sample_rate}Hz"
    )
    print(
        f"spectrogram_shape={tuple(spectrogram.values.shape)} "
        f"frequency_resolution={frequency_resolution:.2f}Hz "
        f"frame_step={time_resolution_ms:.2f}ms"
    )
    print(f"saved={output}")


if __name__ == "__main__":
    main()
