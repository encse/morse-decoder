import wave
from pathlib import Path

import torch

from morse_timing.audio import AudioConfig, save_wav, synthesize_morse
from morse_timing.spectrogram import (
    SpectrogramConfig,
    compute_log_magnitude_stft,
    load_pcm_wav,
    save_spectrogram_image,
)


def test_spectrogram_finds_the_generated_tone_frequency() -> None:
    audio_config = AudioConfig(sample_rate=8_000, frequency_hz=700.0)
    samples = torch.from_numpy(synthesize_morse("T", 20.0, audio_config))
    spectrogram = compute_log_magnitude_stft(samples, audio_config.sample_rate)
    frequency_energy = spectrogram.values.max(dim=1).values
    peak_frequency = spectrogram.frequencies_hz[frequency_energy.argmax()]

    assert spectrogram.values.shape[0] == 65
    assert abs(float(peak_frequency) - 700.0) <= 31.25
    assert float(spectrogram.values.min()) >= 0.0
    assert spectrogram.scale == "power"
    assert spectrogram.times_seconds[1] - spectrogram.times_seconds[0] == 0.02


def test_wav_loading_and_png_visualization(tmp_path: Path) -> None:
    audio_config = AudioConfig(sample_rate=8_000)
    wav_path = tmp_path / "test.wav"
    png_path = tmp_path / "test.png"
    save_wav(wav_path, synthesize_morse("SOS", 20.0, audio_config), 8_000)

    samples, sample_rate = load_pcm_wav(wav_path)
    spectrogram = compute_log_magnitude_stft(
        samples,
        sample_rate,
        SpectrogramConfig(n_fft=256, hop_length=80, win_length=256),
    )
    save_spectrogram_image(spectrogram, png_path)

    assert samples.ndim == 1
    assert sample_rate == 8_000
    assert png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_stereo_pcm_is_downmixed_to_mono(tmp_path: Path) -> None:
    wav_path = tmp_path / "stereo.wav"
    stereo_frames = torch.tensor([[1_000, -1_000], [2_000, 0]], dtype=torch.int16)
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(8_000)
        wav_file.writeframes(stereo_frames.numpy().astype("<i2").tobytes())

    samples, sample_rate = load_pcm_wav(wav_path)

    assert sample_rate == 8_000
    assert torch.allclose(samples, torch.tensor([0.0, 1_000.0 / 32_768.0]))
