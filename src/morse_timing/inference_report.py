"""Render a compact visual report for synthesized Morse inference."""

from __future__ import annotations

import os
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from morse_timing.audio_tokens import AudioToken
from morse_timing.spectrogram import Spectrogram

_MATPLOTLIB_CACHE = Path(tempfile.gettempdir()) / "morse-timing-matplotlib"
_MATPLOTLIB_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MATPLOTLIB_CACHE))


@dataclass(frozen=True)
class CharacterSpan:
    """One intended character and its exact tone interval."""

    character: str
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class TimedToken:
    """One non-blank token emitted by greedy CTC collapse."""

    token: AudioToken
    time_seconds: float


def _wrap_parameter_lines(
    parameters: Sequence[tuple[str, str]],
    max_characters: int = 115,
) -> tuple[str, ...]:
    """Wrap complete parameter entries without splitting a label from its value."""

    if max_characters <= 0:
        raise ValueError("Maximum parameter line length must be positive")
    lines: list[str] = []
    current = ""
    for name, value in parameters:
        entry = f"{name} {value}"
        candidate = f"{current}   {entry}" if current else entry
        if current and len(candidate) > max_characters:
            lines.append(current)
            current = entry
        else:
            current = candidate
    if current:
        lines.append(current)
    return tuple(lines)


def ctc_token_events(
    frame_tokens: Sequence[AudioToken | int],
    frame_times_seconds: Sequence[float],
) -> tuple[TimedToken, ...]:
    """Locate every token retained by normal CTC repeat/blank collapse."""

    if len(frame_tokens) != len(frame_times_seconds):
        raise ValueError("Frame tokens and frame times must have equal length")
    events: list[TimedToken] = []
    previous: AudioToken | None = None
    for raw_token, time_seconds in zip(frame_tokens, frame_times_seconds, strict=True):
        token = AudioToken(raw_token)
        if token is not previous and token is not AudioToken.CTC_BLANK:
            events.append(TimedToken(token, float(time_seconds)))
        previous = token
    return tuple(events)


def save_inference_report(
    output_path: str | Path,
    *,
    spectrogram: Spectrogram,
    character_spans: Sequence[CharacterSpan],
    token_events: Sequence[TimedToken],
    checkpoint_path: Path | None,
    architecture: str,
    decoded_text: str,
    parameters: Sequence[tuple[str, str]],
) -> None:
    """Save a spectrogram, intended-character strip, and token timeline."""

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(15.0, 9.0), facecolor="#10151d")
    FigureCanvasAgg(figure)
    grid = figure.add_gridspec(
        2,
        1,
        height_ratios=(2.1, 6.3),
        hspace=0.12,
        left=0.075,
        right=0.94,
        top=0.94,
        bottom=0.18,
    )

    header = figure.add_subplot(grid[0])
    spectrum = figure.add_subplot(grid[1])

    checkpoint_label = str(checkpoint_path) if checkpoint_path else "(in-memory model)"
    header.set_facecolor("#10151d")
    header.axis("off")
    header.text(
        0.0,
        0.96,
        "MORSE AUDIO INFERENCE",
        color="#f3f5f7",
        fontsize=17,
        fontweight="bold",
        va="top",
    )
    header.text(
        0.0,
        0.70,
        f"Model  {checkpoint_label}",
        color="#aeb8c5",
        fontsize=9.5,
        va="top",
    )
    header.text(
        0.0,
        0.50,
        f"Architecture  {architecture}",
        color="#7f8b99",
        fontsize=9,
        va="top",
    )
    header.text(
        0.0,
        0.31,
        "\n".join(_wrap_parameter_lines(parameters)),
        color="#aeb8c5",
        fontsize=8.5,
        linespacing=1.3,
        va="top",
    )

    duration = spectrogram.duration_seconds
    display_values = (
        10.0
        * torch.log10(
            spectrogram.values.clamp_min(10.0 ** (spectrogram.minimum_db / 10.0))
        )
        if spectrogram.scale == "power"
        else spectrogram.values
    )
    image = spectrum.pcolormesh(
        spectrogram.times_seconds.cpu().numpy(),
        spectrogram.frequencies_hz.cpu().numpy(),
        display_values.cpu().numpy(),
        shading="auto",
        cmap="magma",
        vmin=float(display_values.min()),
        vmax=max(0.0, float(display_values.max())),
    )
    spectrum.set_facecolor("#10151d")
    spectrum.set_xlim(0.0, duration)
    spectrum.set_ylim(0.0, min(2_000.0, spectrogram.sample_rate / 2.0))
    spectrum.set_ylabel("Frequency (Hz)", color="#aeb8c5")
    spectrum.tick_params(colors="#8d99a8", labelsize=8)
    spectrum.grid(False)
    for spine in spectrum.spines.values():
        spine.set_color("#34404f")
    for span in character_spans:
        spectrum.axvline(
            span.start_seconds,
            ymin=0.0,
            ymax=1.0,
            color="#718096",
            linewidth=0.65,
            alpha=0.48,
        )
        spectrum.axvline(
            span.end_seconds,
            ymin=0.0,
            ymax=1.0,
            color="#718096",
            linewidth=0.65,
            alpha=0.48,
        )
        spectrum.text(
            (span.start_seconds + span.end_seconds) / 2.0,
            1.012,
            span.character,
            transform=spectrum.get_xaxis_transform(),
            color="#f3f5f7",
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="bottom",
            clip_on=False,
        )
    token_labels = {
        AudioToken.DIT: "DIT",
        AudioToken.DAH: "DAH",
        AudioToken.END_CHARACTER: "CHAR",
        AudioToken.END_WORD: "WORD",
    }
    for index, event in enumerate(token_events):
        label_position = -0.045 if index % 2 == 0 else -0.115
        spectrum.axvline(
            event.time_seconds,
            ymin=label_position,
            ymax=1.0,
            color="#8996a6",
            linewidth=0.55,
            linestyle=(0, (3, 5)),
            alpha=0.28,
            clip_on=False,
        )
        spectrum.text(
            event.time_seconds,
            label_position,
            token_labels[event.token],
            transform=spectrum.get_xaxis_transform(),
            color="#e6e9ed",
            fontsize=7.5,
            rotation=90,
            ha="center",
            va="top",
            clip_on=False,
        )

    colorbar_axes = spectrum.inset_axes((1.012, 0.0, 0.018, 1.0))
    colorbar = figure.colorbar(image, cax=colorbar_axes)
    colorbar.set_label("dB", color="#aeb8c5")
    colorbar.ax.tick_params(colors="#8d99a8", labelsize=7)
    colorbar.outline.set_edgecolor("#34404f")

    decoded_lines = "\n".join(textwrap.wrap(decoded_text or "(empty)", width=145))
    figure.text(
        0.075,
        0.025,
        f"Decoded text  {decoded_lines}",
        color="#aeb8c5",
        fontsize=8.5,
        va="bottom",
    )
    figure.savefig(output, dpi=150, facecolor=figure.get_facecolor())
