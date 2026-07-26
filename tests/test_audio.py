import wave
from pathlib import Path

import numpy as np
import pytest

from morse_timing.audio import (
    AudioConfig,
    add_power_scaled_noise,
    apply_radio_noise_filter,
    apply_recording_amplitude,
    apply_sinusoidal_fading,
    add_white_noise,
    save_wav,
    synthesize_morse,
    text_to_segments,
)


def test_noise_uses_sample_rate_normalized_power() -> None:
    samples = np.zeros(200_000, dtype=np.float32)
    noisy = add_power_scaled_noise(samples, 8_000, 1.0, np.random.default_rng(7))

    assert float(noisy.std()) == pytest.approx(np.sqrt(0.004), rel=0.02)


def test_radio_noise_filter_shapes_spectrum_and_preserves_rms() -> None:
    samples = np.random.default_rng(7).normal(
        0.0,
        0.2,
        200_000,
    ).astype(np.float32)

    filtered = apply_radio_noise_filter(
        samples,
        8_000,
        low_cutoff_hz=500.0,
        high_cutoff_hz=1_000.0,
        order=4,
    )
    frequencies = np.fft.rfftfreq(filtered.size, d=1.0 / 8_000)
    power = np.abs(np.fft.rfft(filtered)) ** 2
    passband_power = power[
        (frequencies >= 600.0) & (frequencies <= 900.0)
    ].mean()
    rejected_power = power[
        (frequencies >= 2_000.0) & (frequencies <= 3_000.0)
    ].mean()

    assert float(filtered.std()) == pytest.approx(float(samples.std()), rel=0.001)
    assert rejected_power < passband_power * 0.005


def test_recording_amplitude_scales_signal_and_clips() -> None:
    samples = np.array([-1.0, 0.5, 1.0], dtype=np.float32)

    assert np.allclose(
        apply_recording_amplitude(samples, 50.0),
        np.array([-0.5, 0.25, 0.5], dtype=np.float32),
    )
    assert np.allclose(
        apply_recording_amplitude(samples, 150.0),
        np.array([-1.0, 0.75, 1.0], dtype=np.float32),
    )


def test_sinusoidal_fading_preserves_full_level_and_reaches_requested_depth() -> None:
    samples = np.ones(8_000, dtype=np.float32)

    faded = apply_sinusoidal_fading(samples, 8_000, 60.0, 1.0)

    assert np.isclose(faded.max(), 1.0, atol=1e-5)
    assert np.isclose(faded.min(), 0.4, atol=1e-5)


def test_text_to_segments_uses_ideal_morse_ratios() -> None:
    segments = text_to_segments("ET E")

    assert [(segment.is_tone, segment.units) for segment in segments] == [
        (True, 1),
        (False, 3),
        (True, 3),
        (False, 7),
        (True, 1),
        (False, 3),
    ]


def test_existing_word_gap_can_be_randomized_to_twice_its_normal_duration() -> None:
    normal = text_to_segments("E E", (False,))
    doubled = text_to_segments("E E", (True,))

    normal_gap = next(
        segment for segment in normal if not segment.is_tone and segment.units > 3
    )
    doubled_gap = next(
        segment for segment in doubled if not segment.is_tone and segment.units > 3
    )

    assert normal_gap.units == 7
    assert doubled_gap.units == 14


def test_synthesis_has_exact_expected_duration_and_bounded_amplitude() -> None:
    config = AudioConfig(sample_rate=8_000, frequency_hz=700.0, amplitude=0.5)
    samples = synthesize_morse("E", wpm=20.0, config=config)

    assert len(samples) == 1_920
    assert samples.dtype == np.float32
    assert np.max(np.abs(samples)) <= 0.5
    assert samples[-1] == 0.0


def test_synthesis_uses_standard_seven_unit_word_gap() -> None:
    samples = synthesize_morse(
        "E E",
        wpm=20.0,
        config=AudioConfig(sample_rate=8_000),
    )

    assert len(samples) == 12 * 480


def test_carrier_phase_continues_across_silence() -> None:
    config = AudioConfig(
        sample_rate=8_000,
        frequency_hz=703.0,
        amplitude=1.0,
        rise_fall_ms=0.0,
    )

    samples = synthesize_morse(
        "EE", 20.0, config, carrier_phase_radians=0.0
    )

    second_tone_start = 1_920
    expected = np.sin(
        2.0 * np.pi * config.frequency_hz * second_tone_start / config.sample_rate
    )
    assert samples[second_tone_start] == pytest.approx(expected, abs=1e-5)
    assert abs(samples[second_tone_start]) > 0.5


def test_save_wav_writes_mono_16_bit_pcm(tmp_path: Path) -> None:
    config = AudioConfig(sample_rate=8_000)
    samples = synthesize_morse("SOS", wpm=18.0, config=config)
    output = tmp_path / "nested" / "sos.wav"

    save_wav(output, samples, config.sample_rate)

    with wave.open(str(output), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 8_000
        assert wav_file.getnframes() == len(samples)


def test_white_noise_is_reproducible_and_scaled_to_active_signal_rms() -> None:
    samples = synthesize_morse("E", wpm=20.0)
    first = add_white_noise(samples, 10.0, np.random.default_rng(7))
    second = add_white_noise(samples, 10.0, np.random.default_rng(7))
    active = samples[np.abs(samples) > 1e-6]
    expected_noise_rms = np.sqrt(np.mean(active.astype(np.float64) ** 2)) * 0.10
    measured_noise_rms = np.sqrt(
        np.mean((first.astype(np.float64) - samples.astype(np.float64)) ** 2)
    )

    assert np.array_equal(first, second)
    assert measured_noise_rms == pytest.approx(expected_noise_rms, rel=0.1)


def test_invalid_audio_parameters_are_rejected() -> None:
    with pytest.raises(ValueError, match="WPM"):
        synthesize_morse("TEST", wpm=0.0)
    with pytest.raises(ValueError, match="Nyquist"):
        AudioConfig(sample_rate=8_000, frequency_hz=4_000.0)
