import numpy as np

from morse_timing.audio_tokens import AudioToken
from morse_timing.live_inference import (
    IncrementalMorseParser,
    OnlineCTCCollapse,
    StreamingLinearResampler,
)


def test_streaming_resampler_preserves_ratio_across_chunks() -> None:
    resampler = StreamingLinearResampler(48_000, 8_000)

    first = resampler.process(np.arange(480, dtype=np.float32))
    second = resampler.process(np.arange(480, 960, dtype=np.float32))

    assert first.shape == (80,)
    assert second.shape == (80,)
    assert np.allclose(np.concatenate((first, second)), np.arange(0, 960, 6))


def test_online_ctc_collapse_retains_previous_frame_between_chunks() -> None:
    collapse = OnlineCTCCollapse()

    first = collapse.process([0, 1, 1])
    second = collapse.process([1, 0, 1, 3, 3])

    assert first == (AudioToken.DIT,)
    assert second == (AudioToken.DIT, AudioToken.END_CHARACTER)


def test_incremental_parser_prints_only_completed_characters_and_words() -> None:
    parser = IncrementalMorseParser()

    assert parser.process((AudioToken.DIT, AudioToken.DIT)) == ""
    assert parser.process((AudioToken.END_CHARACTER,)) == "I"
    assert parser.process((AudioToken.END_WORD,)) == " "
    assert parser.process((AudioToken.DAH, AudioToken.END_CHARACTER)) == "T"


def test_incremental_parser_shows_invalid_morse_without_a_label() -> None:
    parser = IncrementalMorseParser()

    tokens = (AudioToken.DIT,) * 7 + (AudioToken.END_CHARACTER,)

    assert parser.process(tokens) == "[.......]"


def test_incremental_parser_decodes_bk_prosign_as_one_group() -> None:
    parser = IncrementalMorseParser()
    tokens = (
        AudioToken.DAH,
        AudioToken.DIT,
        AudioToken.DIT,
        AudioToken.DIT,
        AudioToken.DAH,
        AudioToken.DIT,
        AudioToken.DAH,
        AudioToken.END_CHARACTER,
    )

    assert parser.process(tokens) == "<BK>"
