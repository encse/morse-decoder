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
    character_edit_distance,
    greedy_decode_batch,
    split_concatenated_targets,
    train_epoch,
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
    assert not torch.equal(original, model.classifier.weight)
    assert model.tone_activity_head.weight.grad is not None
    assert model.tone_length_head.weight.grad is not None


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
