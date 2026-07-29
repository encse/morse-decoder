"""Continuously decode Morse audio from a microphone or radio audio input."""

from __future__ import annotations

import argparse
import queue
import sys
from pathlib import Path

import numpy as np
import torch

from morse_timing.audio_dataset import prepare_spectrogram_features
from morse_timing.audio_inference import MorseAudioDecoder
from morse_timing.audio_tokens import AudioToken
from morse_timing.morse import MORSE_DECODING_TABLE
from morse_timing.spectrogram import compute_log_magnitude_stft


class StreamingLinearResampler:
    """Convert arbitrary input chunks while retaining interpolation phase."""

    def __init__(self, source_rate: float, target_rate: float) -> None:
        if source_rate <= 0.0 or target_rate <= 0.0:
            raise ValueError("Sample rates must be positive")
        self.step = source_rate / target_rate
        self.buffer = np.empty(0, dtype=np.float32)
        self.next_position = 0.0

    def process(self, samples: np.ndarray) -> np.ndarray:
        """Return every target-rate sample available after appending one chunk."""

        values = np.asarray(samples, dtype=np.float32).reshape(-1)
        if values.size == 0:
            return values
        if self.step == 1.0:
            return values.copy()
        self.buffer = np.concatenate((self.buffer, values))
        if self.buffer.size < 2 or self.next_position >= self.buffer.size - 1:
            return np.empty(0, dtype=np.float32)
        positions = np.arange(
            self.next_position,
            self.buffer.size - 1,
            self.step,
            dtype=np.float64,
        )
        left = positions.astype(np.int64)
        fraction = positions - left
        output = (
            self.buffer[left] * (1.0 - fraction)
            + self.buffer[left + 1] * fraction
        ).astype(np.float32)
        self.next_position = float(positions[-1] + self.step)
        consumed = int(self.next_position)
        if consumed:
            self.buffer = self.buffer[consumed:]
            self.next_position -= consumed
        return output


class OnlineCTCCollapse:
    """Collapse a frame path incrementally without losing chunk boundaries."""

    def __init__(self) -> None:
        self.previous = AudioToken.CTC_BLANK

    def process(self, frame_path: list[int]) -> tuple[AudioToken, ...]:
        """Return newly emitted non-blank CTC tokens."""

        emitted: list[AudioToken] = []
        for value in frame_path:
            token = AudioToken(value)
            if token != self.previous and token != AudioToken.CTC_BLANK:
                emitted.append(token)
            self.previous = token
        return tuple(emitted)


class IncrementalMorseParser:
    """Convert completed streaming Morse token groups into printable text."""

    def __init__(self) -> None:
        self.symbols: list[str] = []
        self.last_output_was_space = True

    def process(self, tokens: tuple[AudioToken, ...]) -> str:
        """Return only text completed by the newly supplied tokens."""

        output: list[str] = []
        for token in tokens:
            if token == AudioToken.DIT:
                self.symbols.append(".")
            elif token == AudioToken.DAH:
                self.symbols.append("-")
            elif token == AudioToken.END_CHARACTER:
                character = self._finish_character()
                if character:
                    output.append(character)
                    self.last_output_was_space = False
            elif token == AudioToken.END_WORD:
                character = self._finish_character()
                if character:
                    output.append(character)
                    self.last_output_was_space = False
                if not self.last_output_was_space:
                    output.append(" ")
                    self.last_output_was_space = True
        return "".join(output)

    def _finish_character(self) -> str:
        if not self.symbols:
            return ""
        morse = "".join(self.symbols)
        self.symbols.clear()
        return MORSE_DECODING_TABLE.get(morse, f"[{morse}]")


def _load_sounddevice():
    try:
        import sounddevice
    except ImportError as exception:
        raise RuntimeError(
            "Live input requires sounddevice; install the project's live extra"
        ) from exception
    return sounddevice


def list_devices() -> None:
    """Print PortAudio input and output devices."""

    sounddevice = _load_sounddevice()
    print(sounddevice.query_devices())


def run_live_decoder(
    checkpoint: Path,
    input_device: int | str | None,
    capture_sample_rate: float | None,
    latency_ms: float,
    device: str,
) -> None:
    """Capture audio indefinitely and print decoded characters as they finish."""

    if latency_ms <= 0.0:
        raise ValueError("Latency must be positive")
    decoder = MorseAudioDecoder.load(checkpoint, device)
    if decoder.model.config.sequence_model != "lstm":
        raise ValueError("Live inference requires an LSTM checkpoint")
    sounddevice = _load_sounddevice()
    selected_input = None if input_device is None else input_device
    device_info = sounddevice.query_devices(selected_input, "input")
    source_rate = float(
        device_info["default_samplerate"]
        if capture_sample_rate is None
        else capture_sample_rate
    )
    model_rate = decoder.dataset_config.audio.sample_rate
    hop_length = decoder.dataset_config.spectrogram.hop_length
    frames_per_chunk = max(1, round(latency_ms * model_rate / 1000.0 / hop_length))
    model_chunk_samples = frames_per_chunk * hop_length
    capture_blocksize = max(1, round(source_rate * latency_ms / 1000.0))
    audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=32)
    dropped_blocks = 0

    def callback(indata, frames, time_info, status) -> None:
        del frames, time_info
        nonlocal dropped_blocks
        if status:
            print(f"audio_status={status}", file=sys.stderr, flush=True)
        try:
            audio_queue.put_nowait(np.asarray(indata[:, 0], dtype=np.float32).copy())
        except queue.Full:
            dropped_blocks += 1

    print(
        f"input_device={device_info['name']!r} capture_rate={source_rate:g} "
        f"model_rate={model_rate} latency_ms={latency_ms:g}",
        file=sys.stderr,
        flush=True,
    )
    print("Listening; press Ctrl-C to stop.", file=sys.stderr, flush=True)
    resampler = StreamingLinearResampler(source_rate, model_rate)
    pending = np.empty(0, dtype=np.float32)
    state: tuple[torch.Tensor, torch.Tensor] | None = None
    collapse = OnlineCTCCollapse()
    parser = IncrementalMorseParser()
    try:
        with sounddevice.InputStream(
            device=selected_input,
            channels=1,
            samplerate=source_rate,
            blocksize=capture_blocksize,
            dtype="float32",
            callback=callback,
        ):
            while True:
                captured = audio_queue.get()
                resampled = resampler.process(captured)
                if resampled.size:
                    pending = np.concatenate((pending, resampled))
                while pending.size >= model_chunk_samples:
                    waveform = torch.from_numpy(pending[:model_chunk_samples].copy())
                    pending = pending[model_chunk_samples:]
                    spectrogram = compute_log_magnitude_stft(
                        waveform,
                        model_rate,
                        decoder.dataset_config.spectrogram,
                    )
                    features = prepare_spectrogram_features(
                        spectrogram.values,
                        decoder.dataset_config.spectrogram,
                    ).transpose(0, 1).contiguous()
                    logits, state = decoder.model.forward_stream(
                        features.unsqueeze(0).to(decoder.device),
                        state,
                    )
                    frame_path = logits[0].argmax(dim=-1).cpu().tolist()
                    text = parser.process(collapse.process(frame_path))
                    if text:
                        print(text, end="", flush=True)
                if dropped_blocks:
                    print(
                        f"\naudio_blocks_dropped={dropped_blocks}",
                        file=sys.stderr,
                        flush=True,
                    )
                    dropped_blocks = 0
    except KeyboardInterrupt:
        print()
        print("Stopped.", file=sys.stderr, flush=True)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the live audio inference command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, nargs="?")
    parser.add_argument("--input-device", help="Input device index or name")
    parser.add_argument("--capture-sample-rate", type=float)
    parser.add_argument("--latency-ms", type=float, default=100.0)
    parser.add_argument("--device", default="cpu", help="PyTorch inference device")
    parser.add_argument("--list-devices", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    """List audio devices or start continuous decoding."""

    args = build_argument_parser().parse_args(argv)
    if args.list_devices:
        list_devices()
        return
    if args.checkpoint is None:
        raise ValueError("Checkpoint is required unless --list-devices is used")
    input_device: int | str | None = args.input_device
    if isinstance(input_device, str) and input_device.isdecimal():
        input_device = int(input_device)
    run_live_decoder(
        args.checkpoint,
        input_device,
        args.capture_sample_rate,
        args.latency_ms,
        args.device,
    )


if __name__ == "__main__":
    main()
