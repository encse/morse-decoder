from pathlib import Path

import numpy as np
import pytest
import torch

from morse_timing.audio import (
    AudioConfig,
    save_wav,
    synthesize_morse,
    synthesize_morse_with_timing,
)
from morse_timing.audio_dataset import Stage1DatasetConfig
from morse_timing.audio_inference import (
    MorseAudioDecoder,
    _character_spans,
    _input_filter_label,
)
from morse_timing.audio_model import AudioModelConfig, MorseAudioCTCModel
from morse_timing.audio_tokens import AudioToken
from morse_timing.audio_train import OverfitMetrics, save_checkpoint
from morse_timing.inference_report import _wrap_parameter_lines, ctc_token_events


def test_saved_checkpoint_can_decode_synthesized_text(tmp_path: Path) -> None:
    config = AudioModelConfig(
        first_conv_channels=2,
        second_conv_channels=4,
        projection_size=8,
        hidden_size=8,
        num_gru_layers=1,
    )
    model = MorseAudioCTCModel(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.classifier.bias[1] = 5.0
    checkpoint = tmp_path / "test.pt"
    metrics = OverfitMetrics(0.1, 0.0, 0.0, 1.0, 1.0, "E", "E")
    save_checkpoint(
        checkpoint,
        model,
        Stage1DatasetConfig(),
        ["E"],
        1,
        metrics,
    )

    decoder = MorseAudioDecoder.load(checkpoint, device="cpu")
    result = decoder.decode_text("E")

    assert decoder.training_texts == ("E",)
    assert result.frame_tokens
    assert result.predicted_tokens == ("DIT",)
    assert result.decoded_text == "E"
    assert result.exact_text

    wav_path, image_path = decoder.save_input_artifacts(
        "E",
        wpm=17.0,
        frequency_hz=850.0,
        output_directory=tmp_path / "audio",
    )

    assert wav_path.name == (
        "e-17wpm-850hz-0pct-jitter-0noise-power-"
        "100pct-amplitude-0ms-edges.wav"
    )
    assert image_path.name == (
        "e-17wpm-850hz-0pct-jitter-0noise-power-"
        "100pct-amplitude-0ms-edges.png"
    )
    assert wav_path.stat().st_size > 44
    assert image_path.stat().st_size > 0

    custom_wav, custom_image = decoder.save_input_artifacts(
        "E",
        wpm=17.0,
        frequency_hz=850.0,
        output_directory=tmp_path / "ignored",
        output_path=tmp_path / "named" / "analysis.png",
    )

    assert custom_image == tmp_path / "named" / "analysis.png"
    assert custom_wav == tmp_path / "named" / "analysis.wav"
    assert custom_image.is_file()
    assert custom_wav.is_file()


def test_character_spans_follow_exact_rendered_tone_boundaries() -> None:
    audio_config = AudioConfig(sample_rate=8_000, frequency_hz=700.0)
    _, segments = synthesize_morse_with_timing("E T", 20.0, audio_config)

    spans = _character_spans("E T", segments, 800, audio_config.sample_rate)

    assert tuple(span.character for span in spans) == ("E", "T")
    assert spans[0].start_seconds == 0.1
    assert spans[0].end_seconds == 0.16
    assert spans[1].start_seconds == 0.58
    assert spans[1].end_seconds == 0.76


def test_synthesized_analysis_repeats_text_with_the_same_noise_in_gaps() -> None:
    config = Stage1DatasetConfig(
        wpm=20.0,
        leading_silence_seconds=0.0,
        trailing_silence_seconds=0.0,
        noise_power=4.0,
        min_noise_only_power=1_000.0,
    )
    decoder = MorseAudioDecoder(
        MorseAudioCTCModel(
            AudioModelConfig(
                projection_size=8,
                hidden_size=8,
                dense_layers=2,
            )
        ),
        config,
        (),
        torch.device("cpu"),
    )

    sample = decoder._build_synthesized_input(
        "E",
        config,
        repetition_count=4,
        gap_seconds=5.0,
    )
    gap_samples = 5 * config.audio.sample_rate

    assert sample.text == "E E E E"
    assert len(sample.character_spans) == 4
    for span in sample.character_spans[1:]:
        next_tone_start = round(span.start_seconds * config.audio.sample_rate)
        gap_start = next_tone_start - gap_samples
        gap = sample.waveform[gap_start : gap_start + gap_samples]
        assert np.std(gap) > 0.0
    background = sample.waveform[
        round(sample.character_spans[0].end_seconds * config.audio.sample_rate)
        + 100:
        round(sample.character_spans[1].start_seconds * config.audio.sample_rate)
        - 100
    ]
    first_gap_end = round(
        sample.character_spans[1].start_seconds * config.audio.sample_rate
    )
    first_gap = sample.waveform[first_gap_end - gap_samples : first_gap_end]
    assert np.std(first_gap) == pytest.approx(np.std(background), rel=0.1)


def test_clean_analysis_gap_is_silent() -> None:
    config = Stage1DatasetConfig(
        leading_silence_seconds=0.0,
        trailing_silence_seconds=0.0,
        noise_power=0.0,
        apply_input_filter=False,
    )
    decoder = MorseAudioDecoder(
        MorseAudioCTCModel(
            AudioModelConfig(
                projection_size=8,
                hidden_size=8,
                dense_layers=2,
            )
        ),
        config,
        (),
        torch.device("cpu"),
    )

    sample = decoder._build_synthesized_input(
        "E",
        config,
        repetition_count=2,
        gap_seconds=1.0,
    )

    next_tone_start = round(
        sample.character_spans[1].start_seconds * config.audio.sample_rate
    )
    explicit_gap = sample.waveform[
        next_tone_start - config.audio.sample_rate : next_tone_start
    ]
    assert np.all(explicit_gap == 0.0)


def test_explicit_analysis_filters_apply_to_clean_profile() -> None:
    config = Stage1DatasetConfig(
        leading_silence_seconds=0.0,
        trailing_silence_seconds=0.0,
        apply_input_filter=False,
    )
    decoder = MorseAudioDecoder(
        MorseAudioCTCModel(
            AudioModelConfig(
                projection_size=8,
                hidden_size=8,
                dense_layers=2,
            )
        ),
        config,
        (),
        torch.device("cpu"),
    )

    clean = decoder._build_synthesized_input("E", config)
    lowpass = decoder._build_synthesized_input(
        "E",
        config,
        lowpass_cutoff_hz=1_500.0,
    )
    bandpass = decoder._build_synthesized_input(
        "E",
        config,
        bandpass_bandwidth_hz=750.0,
    )
    automatic_filter_config = Stage1DatasetConfig(
        leading_silence_seconds=0.0,
        trailing_silence_seconds=0.0,
        apply_input_filter=True,
    )
    explicit_overrides_automatic = decoder._build_synthesized_input(
        "E",
        automatic_filter_config,
        bandpass_bandwidth_hz=750.0,
    )

    assert not np.array_equal(clean.waveform, lowpass.waveform)
    assert not np.array_equal(clean.waveform, bandpass.waveform)
    assert np.array_equal(bandpass.waveform, explicit_overrides_automatic.waveform)
    assert bandpass.input_filter is not None
    assert (
        bandpass.input_filter.high_cutoff_hz
        - bandpass.input_filter.low_cutoff_hz
        == 750.0
    )
    assert (
        bandpass.input_filter.low_cutoff_hz
        <= config.audio.frequency_hz
        <= bandpass.input_filter.high_cutoff_hz
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        decoder._build_synthesized_input(
            "E",
            config,
            lowpass_cutoff_hz=1_500.0,
            bandpass_bandwidth_hz=750.0,
        )
    with pytest.raises(ValueError, match="at least 100 Hz above"):
        decoder._build_synthesized_input(
            "E",
            config,
            lowpass_cutoff_hz=config.audio.frequency_hz + 99.0,
        )
    with pytest.raises(ValueError, match="between 100 and 1000"):
        decoder._build_synthesized_input(
            "E",
            config,
            bandpass_bandwidth_hz=99.0,
        )


def test_ctc_token_events_ignore_blanks_and_repeated_frames() -> None:
    frame_tokens = (
        AudioToken.CTC_BLANK,
        AudioToken.DIT,
        AudioToken.DIT,
        AudioToken.CTC_BLANK,
        AudioToken.DIT,
        AudioToken.END_CHARACTER,
    )

    events = ctc_token_events(frame_tokens, (0.1, 0.2, 0.3, 0.4, 0.5, 0.6))

    assert tuple(event.token for event in events) == (
        AudioToken.DIT,
        AudioToken.DIT,
        AudioToken.END_CHARACTER,
    )
    assert tuple(event.time_seconds for event in events) == (0.2, 0.5, 0.6)


def test_report_parameters_wrap_without_splitting_entries() -> None:
    parameters = (
        ("Profile", "random"),
        ("Noise power", "30.4456"),
        ("Low-pass", "0–1500 Hz"),
        ("Band-pass", "100–1000 Hz"),
    )

    lines = _wrap_parameter_lines(parameters, max_characters=45)

    assert lines == (
        "Profile random   Noise power 30.4456",
        "Low-pass 0–1500 Hz   Band-pass 100–1000 Hz",
    )


def test_synthesized_input_reports_the_actual_sampled_filter() -> None:
    config = Stage1DatasetConfig(apply_input_filter=True)
    decoder = MorseAudioDecoder(
        MorseAudioCTCModel(
            AudioModelConfig(
                projection_size=8,
                hidden_size=8,
                dense_layers=2,
            )
        ),
        config,
        (),
        torch.device("cpu"),
    )

    sample = decoder._build_synthesized_input(
        "E",
        config,
        input_filter_seed=101,
    )

    assert sample.input_filter is not None
    label = _input_filter_label(sample.input_filter)
    assert sample.input_filter.kind.replace("pass", "-pass") in label
    assert f"order {sample.input_filter.order}" in label


def test_saved_checkpoint_decodes_and_resamples_external_wav(tmp_path: Path) -> None:
    config = AudioModelConfig(
        first_conv_channels=2,
        second_conv_channels=4,
        projection_size=8,
        hidden_size=8,
        num_gru_layers=1,
    )
    model = MorseAudioCTCModel(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.classifier.bias[1] = 5.0
    checkpoint = tmp_path / "wav-model.pt"
    save_checkpoint(
        checkpoint,
        model,
        Stage1DatasetConfig(),
        [],
        1,
        OverfitMetrics(0.1, 0.0, 0.0, 1.0, 1.0, "E", "E"),
    )
    wav_path = tmp_path / "external.wav"
    source_config = AudioConfig(sample_rate=16_000, frequency_hz=700.0)
    save_wav(
        wav_path,
        synthesize_morse("E", 20.0, source_config),
        source_config.sample_rate,
    )

    result = MorseAudioDecoder.load(checkpoint, device="cpu").decode_wav(wav_path)

    assert result.source_sample_rate == 16_000
    assert result.model_sample_rate == 8_000
    assert result.predicted_tokens == ("DIT",)
    assert result.decoded_text == "E"
    assert result.valid


def test_streaming_decode_matches_across_small_and_large_chunks(tmp_path: Path) -> None:
    config = AudioModelConfig(
        projection_size=8,
        hidden_size=8,
        dense_layers=2,
        num_lstm_layers=1,
        sequence_model="lstm",
    )
    model = MorseAudioCTCModel(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.classifier.bias[1] = 5.0
    checkpoint = tmp_path / "stream-model.pt"
    save_checkpoint(
        checkpoint,
        model,
        Stage1DatasetConfig(),
        [],
        1,
        OverfitMetrics(0.1, 0.0, 0.0, 1.0, 1.0, "E", "E"),
    )
    decoder = MorseAudioDecoder.load(checkpoint, device="cpu")

    one_frame = decoder.decode_text("E", chunk_frames=1)
    whole_sample = decoder.decode_text("E", chunk_frames=10_000)

    assert one_frame.predicted_tokens == whole_sample.predicted_tokens == ("DIT",)
    assert one_frame.decoded_text == whole_sample.decoded_text == "E"


def test_unspecified_augmentations_default_to_clean_inference() -> None:
    dataset_config = Stage1DatasetConfig(
        timing_jitter=0.2,
        noise_percent=180.0,
        fade_depth_percent=60.0,
    )
    decoder = MorseAudioDecoder(
        MorseAudioCTCModel(
            AudioModelConfig(
                projection_size=8,
                hidden_size=8,
                dense_layers=2,
            )
        ),
        dataset_config,
        (),
        torch.device("cpu"),
    )

    effective = decoder.effective_config(None, None)

    assert effective.timing_jitter == 0.0
    assert effective.gap_timing.min_character_units == 3.0
    assert effective.gap_timing.max_character_units == 3.0
    assert effective.noise_percent == 0.0
    assert effective.fade_depth_percent == 0.0
    assert not effective.apply_input_filter


def test_random_profile_samples_reproducibly_from_checkpoint_ranges() -> None:
    dataset_config = Stage1DatasetConfig(
        min_wpm=10.0,
        max_wpm=40.0,
        min_frequency_hz=100.0,
        max_frequency_hz=2_000.0,
        timing_jitter=0.1,
        noise_power=200.0,
        min_amplitude_percent=10.0,
        max_amplitude_percent=150.0,
        fade_depth_percent=60.0,
        min_fade_frequency_hz=0.1,
        max_fade_frequency_hz=2.0,
        min_rise_fall_ms=0.0,
        max_rise_fall_ms=10.0,
    )
    decoder = MorseAudioDecoder(
        MorseAudioCTCModel(
            AudioModelConfig(
                projection_size=8,
                hidden_size=8,
                dense_layers=2,
            )
        ),
        dataset_config,
        (),
        torch.device("cpu"),
    )

    first = decoder.effective_config(None, None, profile="random", random_seed=42)
    second = decoder.effective_config(None, None, profile="random", random_seed=42)

    assert first == second
    assert 10.0 <= first.wpm <= 40.0
    assert 100.0 <= first.audio.frequency_hz <= 2_000.0
    assert 0.0 <= first.timing_jitter <= 0.1
    assert first.gap_timing == dataset_config.gap_timing
    assert 0.0 <= first.noise_power <= 200.0
    assert first.apply_input_filter
    assert 10.0 <= first.max_amplitude_percent <= 150.0
    assert 0.0 <= first.fade_depth_percent <= 60.0
    assert 0.1 <= first.min_fade_frequency_hz <= 2.0
    assert 0.0 <= first.audio.rise_fall_ms <= 10.0
