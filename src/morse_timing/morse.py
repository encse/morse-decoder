"""Deterministic Morse encoding, reconstruction, and decoding."""

from __future__ import annotations

from dataclasses import dataclass


MORSE_TABLE: dict[str, str] = {
    "A": ".-",
    "B": "-...",
    "C": "-.-.",
    "D": "-..",
    "E": ".",
    "F": "..-.",
    "G": "--.",
    "H": "....",
    "I": "..",
    "J": ".---",
    "K": "-.-",
    "L": ".-..",
    "M": "--",
    "N": "-.",
    "O": "---",
    "P": ".--.",
    "Q": "--.-",
    "R": ".-.",
    "S": "...",
    "T": "-",
    "U": "..-",
    "V": "...-",
    "W": ".--",
    "X": "-..-",
    "Y": "-.--",
    "Z": "--..",
    "0": "-----",
    "1": ".----",
    "2": "..---",
    "3": "...--",
    "4": "....-",
    "5": ".....",
    "6": "-....",
    "7": "--...",
    "8": "---..",
    "9": "----.",
    ".": ".-.-.-",
    ",": "--..--",
    "?": "..--..",
    "'": ".----.",
    "!": "-.-.--",
    "/": "-..-.",
    "(": "-.--.",
    ")": "-.--.-",
    "&": ".-...",
    ":": "---...",
    ";": "-.-.-.",
    "=": "-...-",
    "+": ".-.-.",
    "-": "-....-",
    "_": "..--.-",
    '"': ".-..-.",
    "$": "...-..-",
    "@": ".--.-.",
}
REVERSE_MORSE_TABLE = {value: key for key, value in MORSE_TABLE.items()}
SUPPORTED_CHARACTERS = tuple(MORSE_TABLE)


@dataclass(frozen=True)
class DecodeResult:
    """A decoded message and information about malformed Morse groups."""

    text: str
    is_valid: bool
    invalid_codes: tuple[str, ...] = ()


def normalize_text(text: str) -> str:
    """Uppercase text, collapse whitespace, and validate supported characters."""

    normalized = " ".join(text.upper().split())
    unsupported = sorted({char for char in normalized if char != " " and char not in MORSE_TABLE})
    if unsupported:
        raise ValueError(f"Unsupported Morse characters: {unsupported!r}")
    if not normalized:
        raise ValueError("Text must contain at least one supported character")
    return normalized


def encode_text(text: str) -> str:
    """Encode text using spaces between characters and a slash between words."""

    normalized = normalize_text(text)
    return " / ".join(
        " ".join(MORSE_TABLE[character] for character in word)
        for word in normalized.split(" ")
    )


def decode_morse(morse: str) -> DecodeResult:
    """Decode Morse and show unrecognized groups as their original symbols."""

    stripped = morse.strip()
    if not stripped:
        return DecodeResult(text="", is_valid=False, invalid_codes=("",))

    decoded_words: list[str] = []
    invalid_codes: list[str] = []
    for encoded_word in (part.strip() for part in stripped.split("/")):
        decoded_characters: list[str] = []
        if not encoded_word:
            invalid_codes.append("")
            decoded_characters.append("[]")
        else:
            for code in encoded_word.split():
                character = REVERSE_MORSE_TABLE.get(code)
                if character is None:
                    invalid_codes.append(code)
                    decoded_characters.append(f"[{code}]")
                else:
                    decoded_characters.append(character)
        decoded_words.append("".join(decoded_characters))
    return DecodeResult(
        text=" ".join(decoded_words),
        is_valid=not invalid_codes,
        invalid_codes=tuple(invalid_codes),
    )
