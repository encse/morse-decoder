import pytest
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from morse_timing.audio_dataset import (
    CleanAudioMorseDataset,
    Stage1DatasetConfig,
    collate_audio_sequences,
)
from morse_timing.audio_model import AudioModelConfig, MorseAudioCTCModel
from morse_timing.audio_tokens import AudioToken
from morse_timing.audio_train import (
    EVENT_TIMING_LOSS_WEIGHT,
    TONE_ACTIVITY_LOSS_WEIGHT,
    character_edit_distance,
    event_timing_loss,
    greedy_decode_batch,
    split_concatenated_targets,
    train_epoch,
)


def _timing_logits(*events: tuple[int, AudioToken], frames: int = 10) -> torch.Tensor:
    logits = torch.full((1, frames, 5), -8.0)
    logits[..., int(AudioToken.CTC_BLANK)] = 8.0
    for frame, token in events:
        logits[0, frame, int(AudioToken.CTC_BLANK)] = -8.0
        logits[0, frame, int(token)] = 8.0
    return logits


def _timing_targets(
    *events: tuple[int, AudioToken],
    frames: int = 10,
) -> torch.Tensor:
    targets = torch.zeros((1, frames, 5), dtype=torch.bool)
    for frame, token in events:
        targets[0, frame, int(token)] = True
    return targets


def test_event_timing_loss_prefers_expected_frames_in_second_half() -> None:
    frames = 40

    # These make the sample eligible for event timing supervision.
    prefix_events = tuple(
        (frame, AudioToken.DIT)
        for frame in range(2, 20, 2)
    )
    assert len(prefix_events) == 9

    expected_events = (
        *prefix_events,
        (22, AudioToken.DAH),
        (26, AudioToken.END_CHARACTER),
        (32, AudioToken.END_WORD),
    )

    targets = _timing_targets(
        *expected_events,
        frames=frames,
    )
    padding_mask = torch.zeros((1, frames), dtype=torch.bool)

    on_time = event_timing_loss(
        _timing_logits(
            *expected_events,
            frames=frames,
        ),
        targets,
        padding_mask,
    )

    character_late = event_timing_loss(
        _timing_logits(
            *prefix_events,
            (22, AudioToken.DAH),
            (28, AudioToken.END_CHARACTER),
            (32, AudioToken.END_WORD),
            frames=frames,
        ),
        targets,
        padding_mask,
    )

    word_early = event_timing_loss(
        _timing_logits(
            *prefix_events,
            (22, AudioToken.DAH),
            (26, AudioToken.END_CHARACTER),
            (29, AudioToken.END_WORD),
            frames=frames,
        ),
        targets,
        padding_mask,
    )

    assert EVENT_TIMING_LOSS_WEIGHT == 0.3
    assert on_time < character_late
    assert on_time < word_early

def test_event_timing_loss_is_zero_for_fewer_than_ten_morse_events() -> None:
    frames = 30
    events = tuple(
        (frame, AudioToken.DIT)
        for frame in range(2, 20, 2)
    )
    assert len(events) == 9

    targets = _timing_targets(*events, frames=frames)
    padding_mask = torch.zeros((1, frames), dtype=torch.bool)

    badly_timed_logits = _timing_logits(
        (25, AudioToken.DIT),
        frames=frames,
    )

    loss = event_timing_loss(
        badly_timed_logits,
        targets,
        padding_mask,
    )

    assert float(loss) == pytest.approx(0.0)

def test_event_timing_loss_penalizes_later_and_duplicate_events_more() -> None:
    frames = 40
    target_frame = 22

    prefix_events = tuple(
        (frame, AudioToken.DIT)
        for frame in range(2, 20, 2)
    )
    assert len(prefix_events) == 9

    target_events = (
        *prefix_events,
        (target_frame, AudioToken.DIT),
    )

    targets = _timing_targets(
        *target_events,
        frames=frames,
    )
    padding_mask = torch.zeros((1, frames), dtype=torch.bool)

    on_time = event_timing_loss(
        _timing_logits(
            *prefix_events,
            (target_frame, AudioToken.DIT),
            frames=frames,
        ),
        targets,
        padding_mask,
    )

    one_frame_late = event_timing_loss(
        _timing_logits(
            *prefix_events,
            (target_frame + 1, AudioToken.DIT),
            frames=frames,
        ),
        targets,
        padding_mask,
    )

    three_frames_late = event_timing_loss(
        _timing_logits(
            *prefix_events,
            (target_frame + 3, AudioToken.DIT),
            frames=frames,
        ),
        targets,
        padding_mask,
    )

    repeated_event = event_timing_loss(
        _timing_logits(
            *prefix_events,
            (target_frame, AudioToken.DIT),
            (target_frame + 3, AudioToken.DIT),
            frames=frames,
        ),
        targets,
        padding_mask,
    )

    assert on_time < one_frame_late < three_frames_late
    assert on_time < repeated_event

def test_event_timing_loss_ignores_the_first_half() -> None:
    targets = _timing_targets((2, AudioToken.DIT))
    padding_mask = torch.zeros((1, 10), dtype=torch.bool)

    on_time = event_timing_loss(
        _timing_logits((2, AudioToken.DIT)),
        targets,
        padding_mask,
    )
    emitted_elsewhere_in_first_half = event_timing_loss(
        _timing_logits((4, AudioToken.DIT)),
        targets,
        padding_mask,
    )
    not_emitted = event_timing_loss(
        _timing_logits(),
        targets,
        padding_mask,
    )

    assert float(on_time) == pytest.approx(0.0)
    assert float(emitted_elsewhere_in_first_half) == pytest.approx(0.0)
    assert float(not_emitted) == pytest.approx(0.0)


def test_event_timing_loss_balances_rare_token_classes() -> None:
    dit_events = tuple(
        (frame, AudioToken.DIT)
        for frame in (10, 12, 14, 16)
    )
    word_event = (18, AudioToken.END_WORD)

    targets = _timing_targets(
        *dit_events,
        word_event,
        frames=20,
    )
    padding_mask = torch.zeros((1, 20), dtype=torch.bool)

    missing_every_dit = event_timing_loss(
        _timing_logits(word_event, frames=20),
        targets,
        padding_mask,
    )
    missing_the_word_end = event_timing_loss(
        _timing_logits(*dit_events, frames=20),
        targets,
        padding_mask,
    )

    assert float(missing_the_word_end) == pytest.approx(
        float(missing_every_dit),
        rel=0.05,
    )


def test_character_edit_distance_counts_insertions_deletions_and_substitutions() -> None:
    assert character_edit_distance("MORSE", "HORSE") == 1
    assert character_edit_distance("MORSE", "MORE") == 1
    assert character_edit_distance("MORSE", "MORSES") == 1


def test_greedy_batch_decoding_respects_input_lengths() -> None:
    logits = torch.full((2, 5, 5), -10.0)
    paths = [[0, 1, 1, 0, 3], [2, 2, 0, 4, 1]]
    for row, path in enumerate(paths):
        for time, token in enumerate(path):
            logits[row, time, token] = 10.0

    decoded = greedy_decode_batch(logits, torch.tensor([5, 3]))

    assert decoded == [
        (AudioToken.DIT, AudioToken.END_CHARACTER),
        (AudioToken.DAH,),
    ]


def test_one_training_epoch_updates_the_audio_model() -> None:
    dataset = CleanAudioMorseDataset(2, texts=["E", "T"])
    loader = DataLoader(
        [dataset[0], dataset[1]],
        batch_size=2,
        collate_fn=collate_audio_sequences,
    )
    model = MorseAudioCTCModel(
        AudioModelConfig(
            first_conv_channels=2,
            second_conv_channels=4,
            projection_size=8,
            hidden_size=8,
            num_gru_layers=1,
            auxiliary_heads=True,
        )
    )
    optimizer = AdamW(model.parameters(), lr=1e-3)
    original = model.classifier.weight.detach().clone()

    loss = train_epoch(model, loader, optimizer, torch.device("cpu"), 5.0, 0)

    assert torch.isfinite(torch.tensor(loss))
    assert torch.isfinite(
        torch.tensor(
            [
                loss.ctc_loss,
                loss.tone_activity_loss,
                loss.event_timing_loss,
            ]
        )
    ).all()
    assert float(loss) == pytest.approx(
        loss.ctc_loss
        + TONE_ACTIVITY_LOSS_WEIGHT * loss.tone_activity_loss
        + EVENT_TIMING_LOSS_WEIGHT * loss.event_timing_loss
    )
    assert not torch.equal(original, model.classifier.weight)
    assert model.tone_activity_head.weight.grad is not None


def test_noise_only_batch_with_empty_ctc_targets_can_be_trained() -> None:
    dataset = CleanAudioMorseDataset(
        2,
        Stage1DatasetConfig(
            noise_only_probability=1.0,
            min_noise_only_seconds=0.5,
            max_noise_only_seconds=0.5,
        ),
        seed=7,
    )
    loader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=collate_audio_sequences,
    )
    model = MorseAudioCTCModel(
        AudioModelConfig(
            first_conv_channels=2,
            second_conv_channels=4,
            projection_size=8,
            hidden_size=8,
            num_gru_layers=1,
        )
    )
    optimizer = AdamW(model.parameters(), lr=1e-3)

    loss = train_epoch(model, loader, optimizer, torch.device("cpu"), 5.0, 0)

    assert torch.isfinite(torch.tensor(loss))


def test_concatenated_targets_are_split_per_sample() -> None:
    targets = torch.tensor([1, 3, 2, 2, 3])
    lengths = torch.tensor([2, 3])

    assert split_concatenated_targets(targets, lengths) == [
        (AudioToken.DIT, AudioToken.END_CHARACTER),
        (AudioToken.DAH, AudioToken.DAH, AudioToken.END_CHARACTER),
    ]
