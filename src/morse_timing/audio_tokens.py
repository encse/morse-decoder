"""CTC token vocabulary and deterministic parsing for audio Morse decoding."""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum

from morse_timing.morse import DecodeResult, MORSE_TABLE, decode_morse, normalize_text


class AudioToken(IntEnum):
    """Output vocabulary for frame-to-Morse CTC models."""

    CTC_BLANK = 0
    DIT = 1
    DAH = 2
    END_CHARACTER = 3
    END_WORD = 4


CTC_BLANK_INDEX = int(AudioToken.CTC_BLANK)
NUM_AUDIO_TOKENS = len(AudioToken)


def text_to_audio_tokens(text: str) -> tuple[AudioToken, ...]:
    """Convert text to one token per completed tone or semantic silence."""

    normalized = normalize_text(text)
    words = normalized.split(" ")
    tokens: list[AudioToken] = []
    for word in words:
        for character_index, character in enumerate(word):
            tokens.extend(
                AudioToken.DIT if symbol == "." else AudioToken.DAH
                for symbol in MORSE_TABLE[character]
            )
            tokens.append(AudioToken.END_CHARACTER)
            if character_index == len(word) - 1:
                tokens.append(AudioToken.END_WORD)
    return tuple(tokens)


def audio_tokens_to_morse(tokens: Sequence[AudioToken | int]) -> str:
    """Reconstruct Morse; END_WORD adds `/` after a closed character."""

    morse_groups: list[str] = []
    current_symbols: list[str] = []
    for raw_token in tokens:
        token = AudioToken(raw_token)
        if token is AudioToken.CTC_BLANK:
            raise ValueError("Collapsed token sequences cannot contain CTC blank")
        if token is AudioToken.DIT:
            current_symbols.append(".")
        elif token is AudioToken.DAH:
            current_symbols.append("-")
        elif token is AudioToken.END_CHARACTER:
            if not current_symbols:
                raise ValueError(f"Boundary {token.name} has no preceding Morse symbols")
            morse_groups.append("".join(current_symbols))
            current_symbols.clear()
        else:
            if current_symbols:
                morse_groups.append("".join(current_symbols))
                current_symbols.clear()
            if not morse_groups or morse_groups[-1] == "/":
                raise ValueError("END_WORD must follow a completed Morse character")
            morse_groups.append("/")
    if current_symbols:
        morse_groups.append("".join(current_symbols))
    if morse_groups and morse_groups[-1] == "/":
        morse_groups.pop()
    return " ".join(morse_groups)


def format_audio_tokens_as_morse(tokens: Sequence[AudioToken | int]) -> str:
    """Format predicted Morse symbols without rejecting malformed boundaries."""

    morse_groups: list[str] = []
    current_symbols: list[str] = []
    for raw_token in tokens:
        token = AudioToken(raw_token)
        if token is AudioToken.DIT:
            current_symbols.append(".")
        elif token is AudioToken.DAH:
            current_symbols.append("-")
        elif token is AudioToken.END_CHARACTER:
            if current_symbols:
                morse_groups.append("".join(current_symbols))
                current_symbols.clear()
        elif token is AudioToken.END_WORD:
            if current_symbols:
                morse_groups.append("".join(current_symbols))
                current_symbols.clear()
            if morse_groups and morse_groups[-1] != "/":
                morse_groups.append("/")
    if current_symbols:
        morse_groups.append("".join(current_symbols))
    if morse_groups and morse_groups[-1] == "/":
        morse_groups.pop()
    return " ".join(morse_groups)


def decode_audio_tokens(
    tokens: Sequence[AudioToken | int],
    *,
    recognize_prosigns: bool = False,
) -> DecodeResult:
    """Convert collapsed CTC tokens, optionally recognizing prosigns."""

    return decode_morse(
        audio_tokens_to_morse(tokens),
        recognize_prosigns=recognize_prosigns,
    )


def collapse_ctc_path(
    frame_tokens: Sequence[AudioToken | int],
    blank_index: int = CTC_BLANK_INDEX,
) -> tuple[AudioToken, ...]:
    """Greedily collapse repeats and blanks from frame-level CTC predictions."""

    collapsed: list[AudioToken] = []
    previous: int | None = None
    for raw_token in frame_tokens:
        token_index = int(raw_token)
        if token_index != previous and token_index != blank_index:
            collapsed.append(AudioToken(token_index))
        previous = token_index
    return tuple(collapsed)


def normalize_audio_tokens(
    tokens: Sequence[AudioToken | int],
) -> tuple[AudioToken, ...]:
    """Discard leading and duplicate boundaries before deterministic parsing."""

    normalized: list[AudioToken] = []
    for raw_token in tokens:
        token = AudioToken(raw_token)
        if not normalized and token in {
            AudioToken.END_CHARACTER,
            AudioToken.END_WORD,
        }:
            continue
        if normalized and token is normalized[-1] and token in {
            AudioToken.END_CHARACTER,
            AudioToken.END_WORD,
        }:
            continue
        if (
            token is AudioToken.END_CHARACTER
            and normalized
            and normalized[-1] is AudioToken.END_WORD
        ):
            continue
        normalized.append(token)
    return tuple(normalized)
