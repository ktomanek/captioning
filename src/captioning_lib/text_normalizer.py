# Extracted from transformers.models.whisper.english_normalizer
# This allows us to use BasicTextNormalizer without pulling in the full transformers library
# Original source: https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/english_normalizer.py

import re
import unicodedata

# Mapping for special characters that are not separated by NFKD normalization
ADDITIONAL_DIACRITICS = {
    "œ": "oe",
    "Œ": "OE",
    "ø": "o",
    "Ø": "O",
    "æ": "ae",
    "Æ": "AE",
    "ß": "ss",
    "ẞ": "SS",
    "đ": "d",
    "Đ": "D",
    "ð": "d",
    "Ð": "D",
    "þ": "th",
    "Þ": "th",
    "ł": "l",
    "Ł": "L",
}


def remove_symbols_and_diacritics(s: str, keep=""):
    """
    Replace any other markers, symbols, and punctuations with a space, and drop any diacritics (category 'Mn' and some
    manual mappings)
    """

    def replace_character(char):
        if char in keep:
            return char
        elif char in ADDITIONAL_DIACRITICS:
            return ADDITIONAL_DIACRITICS[char]
        elif unicodedata.category(char) == "Mn":
            return ""
        elif unicodedata.category(char)[0] in "MSP":
            return " "
        return char

    return "".join(replace_character(c) for c in unicodedata.normalize("NFKD", s))


def remove_symbols(s: str):
    """
    Replace any other markers, symbols, punctuations with a space, keeping diacritics
    """
    return "".join(" " if unicodedata.category(c)[0] in "MSP" else c for c in unicodedata.normalize("NFKC", s))


class BasicTextNormalizer:
    """
    Basic text normalizer for English transcripts.
    Equivalent to transformers.models.whisper.english_normalizer.BasicTextNormalizer

    This normalizer:
    - Converts to lowercase
    - Removes content in brackets and parentheses
    - Removes symbols and optionally diacritics
    - Normalizes whitespace
    """

    def __init__(self, remove_diacritics: bool = False, split_letters: bool = False):
        self.clean = remove_symbols_and_diacritics if remove_diacritics else remove_symbols
        self.split_letters = split_letters

    def __call__(self, s: str):
        s = s.lower()
        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)  # remove words between brackets
        s = re.sub(r"\(([^)]+?)\)", "", s)  # remove words between parenthesis
        s = self.clean(s).lower()

        if self.split_letters:
            # Note: This requires the regex package for proper Unicode handling
            # For basic usage without regex package, we skip this feature
            try:
                import regex
                s = " ".join(regex.findall(r"\X", s, regex.U))
            except ImportError:
                pass  # Skip letter splitting if regex package not available

        s = re.sub(r"\s+", " ", s)  # replace any successive whitespace characters with a space

        return s
