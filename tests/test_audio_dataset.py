import torch

from morse_timing.audio_dataset import (
    CleanAudioMorseDataset,
    Stage1DatasetConfig,
    collate_audio_sequences,
)
from morse_timing.audio_tokens import (
    AudioToken,
    audio_tokens_to_morse,
    collapse_ctc_path,
    decode_audio_tokens,
    format_audio_tokens_as_morse,
    normalize_audio_tokens,
    text_to_audio_tokens,
)
from morse_timing.morse import SUPPORTED_CHARACTERS, decode_morse


def test_text_tokens_round_trip_through_deterministic_parser() -> None:
    tokens = text_to_audio_tokens("A B")

    assert tokens == (
        AudioToken.DIT,
        AudioToken.DAH,
        AudioToken.END_CHARACTER,
        AudioToken.END_WORD,
        AudioToken.DAH,
        AudioToken.DIT,
        AudioToken.DIT,
        AudioToken.DIT,
        AudioToken.END_CHARACTER,
        AudioToken.END_WORD,
    )
    assert audio_tokens_to_morse(tokens) == ".- / -..."
    assert decode_audio_tokens(tokens).text == "A B"


def test_invalid_morse_is_shown_without_an_invalid_label() -> None:
    result = decode_morse("......")

    assert result.text == "[......]"
    assert not result.is_valid
    assert result.invalid_codes == ("......",)


def test_malformed_token_boundaries_still_show_the_morse_symbols() -> None:
    tokens = (
        AudioToken.DIT,
        AudioToken.END_CHARACTER,
        AudioToken.END_CHARACTER,
        AudioToken.DAH,
    )

    assert format_audio_tokens_as_morse(tokens) == ". -"


def test_word_gap_follows_the_causal_character_boundary() -> None:
    tokens = text_to_audio_tokens("E T")

    assert tokens == (
        AudioToken.DIT,
        AudioToken.END_CHARACTER,
        AudioToken.END_WORD,
        AudioToken.DAH,
        AudioToken.END_CHARACTER,
        AudioToken.END_WORD,
    )
    assert decode_audio_tokens(tokens).text == "E T"

def test_greedy_ctc_collapse_removes_repeats_and_blanks() -> None:
    path = [
        AudioToken.CTC_BLANK,
        AudioToken.DIT,
        AudioToken.DIT,
        AudioToken.CTC_BLANK,
        AudioToken.DIT,
        AudioToken.END_CHARACTER,
        AudioToken.END_CHARACTER,
        AudioToken.CTC_BLANK,
    ]

    assert collapse_ctc_path(path) == (
        AudioToken.DIT,
        AudioToken.DIT,
        AudioToken.END_CHARACTER,
    )


def test_token_normalization_merges_consecutive_boundaries() -> None:
    tokens = (
        AudioToken.DIT,
        AudioToken.END_CHARACTER,
        AudioToken.END_CHARACTER,
        AudioToken.DAH,
        AudioToken.END_CHARACTER,
        AudioToken.END_WORD,
        AudioToken.DIT,
        AudioToken.END_CHARACTER,
    )

    normalized = normalize_audio_tokens(tokens)

    assert normalized == (
        AudioToken.DIT,
        AudioToken.END_CHARACTER,
        AudioToken.DAH,
        AudioToken.END_CHARACTER,
        AudioToken.END_WORD,
        AudioToken.DIT,
        AudioToken.END_CHARACTER,
    )
    assert decode_audio_tokens(normalized).text == "ET E"


def test_token_normalization_discards_leading_boundaries() -> None:
    tokens = (
        AudioToken.END_WORD,
        AudioToken.END_CHARACTER,
        AudioToken.DIT,
        AudioToken.END_CHARACTER,
    )

    normalized = normalize_audio_tokens(tokens)

    assert normalized == (AudioToken.DIT, AudioToken.END_CHARACTER)
    assert decode_audio_tokens(normalized).text == "E"


def test_clean_audio_dataset_produces_model_ready_features() -> None:
    dataset = CleanAudioMorseDataset(2, seed=7, texts=["E", "SOS"])
    first = dataset[0]

    assert first.spectrogram.ndim == 2
    assert first.spectrogram.shape[1] == 65
    assert first.targets.tolist() == [
        AudioToken.DIT,
        AudioToken.END_CHARACTER,
        AudioToken.END_WORD,
    ]
    assert first.tone_activity.shape == (first.input_length,)
    assert torch.all((first.spectrogram >= 0.0) & (first.spectrogram <= 1.0))
    assert first.input_length > first.target_length


def test_leading_silence_adds_the_expected_number_of_frames() -> None:
    without_leading = CleanAudioMorseDataset(
        1,
        Stage1DatasetConfig(
            leading_silence_seconds=0.0,
            trailing_silence_seconds=0.0,
        ),
        texts=["E"],
    )[0]
    with_leading = CleanAudioMorseDataset(
        1,
        Stage1DatasetConfig(
            leading_silence_seconds=0.7,
            trailing_silence_seconds=0.0,
        ),
        texts=["E"],
    )[0]

    assert with_leading.input_length - without_leading.input_length == 35


def test_audio_batch_padding_lengths_and_concatenated_targets() -> None:
    dataset = CleanAudioMorseDataset(2, seed=8, texts=["E", "SOS"])
    samples = [dataset[0], dataset[1]]
    batch = collate_audio_sequences(samples)

    assert batch.spectrograms.shape == (
        2,
        samples[1].input_length,
        65,
    )
    assert batch.input_lengths.tolist() == [
        samples[0].input_length,
        samples[1].input_length,
    ]
    assert batch.target_lengths.tolist() == [3, 13]
    assert torch.equal(batch.targets, torch.cat([samples[0].targets, samples[1].targets]))
    assert batch.tone_activity.shape == batch.spectrograms.shape[:2]
    assert torch.all(batch.padding_mask[0, samples[0].input_length :])
    assert not torch.any(batch.padding_mask[1])


def test_tone_activity_marks_frames_that_overlap_the_tone() -> None:
    sample = CleanAudioMorseDataset(
        1,
        Stage1DatasetConfig(
            wpm=20.0,
            leading_silence_seconds=0.0,
            trailing_silence_seconds=0.0,
        ),
        texts=["E"],
    )[0]

    assert sample.tone_activity[:3].tolist() == [1.0, 1.0, 1.0]
    assert sample.tone_activity[3:].sum() == 0.0


def test_dataset_is_reproducible_by_seed_and_index() -> None:
    first = CleanAudioMorseDataset(3, seed=123)[2]
    second = CleanAudioMorseDataset(3, seed=123)[2]

    assert first.text == second.text
    assert torch.equal(first.targets, second.targets)
    assert torch.equal(first.spectrogram, second.spectrogram)


def test_dataset_samples_rise_fall_time_inside_configured_range() -> None:
    dataset = CleanAudioMorseDataset(
        10,
        Stage1DatasetConfig(min_rise_fall_ms=0.0, max_rise_fall_ms=10.0),
        seed=321,
    )

    values = [dataset._sample_rise_fall(index) for index in range(10)]

    assert all(0.0 <= value <= 10.0 for value in values)
    assert len(set(values)) > 1


def test_random_text_uses_the_complete_supported_morse_alphabet() -> None:
    dataset = CleanAudioMorseDataset(1, seed=123)
    sampled = {
        character
        for index in range(1_000)
        for character in dataset._random_text(index)
        if character != " "
    }

    assert sampled == set(SUPPORTED_CHARACTERS)


def test_forced_word_boundary_keeps_words_longer_than_single_characters() -> None:
    dataset = CleanAudioMorseDataset(
        1,
        Stage1DatasetConfig(
            min_characters=10,
            max_characters=10,
            space_probability=0.0,
            word_boundary_sample_probability=1.0,
        ),
        seed=7,
    )

    words = dataset._random_text(0).split()

    assert len(words) == 2
    assert sum(map(len, words)) == 10
    assert min(map(len, words)) >= 2


def test_variable_wpm_is_reproducible_and_changes_audio_length() -> None:
    config = Stage1DatasetConfig(min_wpm=12.0, max_wpm=30.0)
    first_dataset = CleanAudioMorseDataset(2, config, seed=91, texts=["SOS", "SOS"])
    second_dataset = CleanAudioMorseDataset(2, config, seed=91, texts=["SOS", "SOS"])

    first = first_dataset[0]
    second = first_dataset[1]

    assert torch.equal(first.spectrogram, second_dataset[0].spectrogram)
    assert first.input_length != second.input_length


def test_variable_frequency_is_reproducible_and_changes_spectral_peak() -> None:
    config = Stage1DatasetConfig(
        min_frequency_hz=400.0,
        max_frequency_hz=1_200.0,
    )
    first_dataset = CleanAudioMorseDataset(2, config, seed=92, texts=["E", "E"])
    second_dataset = CleanAudioMorseDataset(2, config, seed=92, texts=["E", "E"])

    first = first_dataset[0]
    second = first_dataset[1]
    first_peak = int(first.spectrogram.max(dim=0).values.argmax())
    second_peak = int(second.spectrogram.max(dim=0).values.argmax())

    assert torch.equal(first.spectrogram, second_dataset[0].spectrogram)
    assert first_peak != second_peak


def test_timing_jitter_is_reproducible_and_changes_segment_duration() -> None:
    clean = CleanAudioMorseDataset(
        1, Stage1DatasetConfig(timing_jitter=0.0), seed=5, texts=["SOS"]
    )[0]
    jittered_dataset = CleanAudioMorseDataset(
        1, Stage1DatasetConfig(timing_jitter=0.15), seed=5, texts=["SOS"]
    )

    jittered = jittered_dataset[0]

    assert torch.equal(jittered.spectrogram, jittered_dataset[0].spectrogram)
    assert clean.input_length != jittered.input_length


def test_training_noise_level_is_sampled_up_to_configured_maximum() -> None:
    dataset = CleanAudioMorseDataset(
        3,
        Stage1DatasetConfig(noise_percent=50.0),
        seed=17,
        texts=["E", "E", "E"],
    )

    levels = [dataset._sample_noise_percent(index) for index in range(3)]

    assert all(0.0 <= level <= 50.0 for level in levels)
    assert len(set(levels)) == 3
    assert torch.equal(dataset[0].spectrogram, dataset[0].spectrogram)
