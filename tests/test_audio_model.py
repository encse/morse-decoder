import pytest
import torch

from morse_timing.audio_dataset import CleanAudioMorseDataset, collate_audio_sequences
from morse_timing.audio_model import (
    AudioModelConfig,
    MorseAudioCTCModel,
    compute_ctc_loss,
    logits_to_ctc_log_probs,
    minimum_ctc_input_lengths,
)


def small_model() -> MorseAudioCTCModel:
    return MorseAudioCTCModel(
        AudioModelConfig(
            first_conv_channels=4,
            second_conv_channels=6,
            projection_size=12,
            hidden_size=10,
            num_gru_layers=1,
        )
    )


def test_model_preserves_variable_time_dimension_and_emits_five_logits() -> None:
    batch = collate_audio_sequences(
        [
            CleanAudioMorseDataset(2, texts=["E", "SOS"])[0],
            CleanAudioMorseDataset(2, texts=["E", "SOS"])[1],
        ]
    )

    logits, output_lengths = small_model()(batch.spectrograms, batch.input_lengths)

    assert logits.shape == (2, batch.spectrograms.shape[1], 5)
    assert torch.equal(output_lengths, batch.input_lengths)
    assert logits_to_ctc_log_probs(logits).shape == (
        batch.spectrograms.shape[1],
        2,
        5,
    )


def test_real_audio_batch_has_finite_ctc_loss_and_backward() -> None:
    dataset = CleanAudioMorseDataset(2, seed=11, texts=["E", "SOS"])
    batch = collate_audio_sequences([dataset[0], dataset[1]])
    model = small_model()
    logits, output_lengths = model(batch.spectrograms, batch.input_lengths)

    loss = compute_ctc_loss(
        logits,
        batch.targets,
        output_lengths,
        batch.target_lengths,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert model.classifier.weight.grad is not None
    assert torch.isfinite(model.classifier.weight.grad).all()
    assert torch.count_nonzero(model.classifier.weight.grad) > 0


def test_minimum_ctc_length_counts_repeated_neighbor_tokens() -> None:
    targets = torch.tensor([1, 1, 3, 2, 3], dtype=torch.long)
    lengths = torch.tensor([3, 2], dtype=torch.long)

    assert minimum_ctc_input_lengths(targets, lengths).tolist() == [4, 2]


def test_ctc_loss_rejects_an_output_sequence_without_enough_alignment_room() -> None:
    logits = torch.zeros(1, 2, 5)

    with pytest.raises(ValueError, match="enough alignment"):
        compute_ctc_loss(
            logits,
            torch.tensor([1, 1]),
            torch.tensor([2]),
            torch.tensor([2]),
        )


def test_model_rejects_wrong_frequency_dimension() -> None:
    model = small_model()

    with pytest.raises(ValueError, match="frequency bins"):
        model(torch.zeros(2, 10, 128), torch.tensor([10, 8]))


def test_tcn_model_preserves_time_and_supports_ctc_backward() -> None:
    dataset = CleanAudioMorseDataset(2, texts=["E", "T"])
    batch = collate_audio_sequences([dataset[0], dataset[1]])
    model = MorseAudioCTCModel(
        AudioModelConfig(
            first_conv_channels=2,
            second_conv_channels=4,
            projection_size=8,
            hidden_size=8,
            sequence_model="tcn",
            tcn_layers=2,
        )
    )

    logits, output_lengths = model(batch.spectrograms, batch.input_lengths)
    loss = compute_ctc_loss(
        logits, batch.targets, output_lengths, batch.target_lengths
    )
    loss.backward()

    assert logits.shape[:2] == batch.spectrograms.shape[:2]
    assert torch.isfinite(loss)
    assert model.classifier.weight.grad is not None


def test_tcn_output_does_not_depend_on_future_frames() -> None:
    model = MorseAudioCTCModel(
        AudioModelConfig(
            first_conv_channels=2,
            second_conv_channels=4,
            projection_size=8,
            hidden_size=8,
            sequence_model="tcn",
            tcn_layers=2,
        )
    )
    spectrograms = torch.randn(2, 50, 65)
    lengths = torch.tensor([50, 50])
    changed = spectrograms.clone()
    changed[:, 20:] = torch.randn_like(changed[:, 20:])

    original_logits, _ = model(spectrograms, lengths)
    changed_logits, _ = model(changed, lengths)

    assert torch.allclose(original_logits[:, :20], changed_logits[:, :20], atol=1e-6)


def test_lstm_streaming_chunks_match_whole_sequence() -> None:
    model = MorseAudioCTCModel(
        AudioModelConfig(
            projection_size=12,
            hidden_size=10,
            num_lstm_layers=1,
            dense_layers=2,
            sequence_model="lstm",
        )
    ).eval()
    spectrograms = torch.randn(2, 30, 65)
    lengths = torch.tensor([30, 30])

    whole, _ = model(spectrograms, lengths)
    first, state = model.forward_stream(spectrograms[:, :11])
    second, _ = model.forward_stream(spectrograms[:, 11:], state)

    assert torch.allclose(whole, torch.cat((first, second), dim=1), atol=1e-6)


def test_training_forward_emits_tone_activity_predictions() -> None:
    dataset = CleanAudioMorseDataset(2, texts=["E", "T"])
    batch = collate_audio_sequences([dataset[0], dataset[1]])
    model = MorseAudioCTCModel(
        AudioModelConfig(
            projection_size=8,
            hidden_size=8,
            dense_layers=2,
            auxiliary_heads=True,
        )
    )

    logits, lengths, tone_logits = model.forward_with_auxiliary(
        batch.spectrograms,
        batch.input_lengths,
    )

    assert logits.shape[:2] == batch.spectrograms.shape[:2]
    assert tone_logits.shape == batch.spectrograms.shape[:2]
    assert torch.equal(lengths, batch.input_lengths)
