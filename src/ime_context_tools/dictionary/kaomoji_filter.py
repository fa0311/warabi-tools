"""Structural kaomoji/emoticon detection.

Flags symbol-structured faces only — Unicode emoji (pictographs, decorative
symbols) are legitimate IME conversions and are deliberately NOT flagged.
"""

from __future__ import annotations

import re
import unicodedata

ASCII_EMOTICON = re.compile(
    r"^(?:[:;=8xX]['’`\-^]?[()\[\]/\\DPpOo]|<3|orz|OTL|><|>_<|<_<|T_T|;_;)$",
    re.IGNORECASE,
)
BRACKETS = frozenset("()（）[]［］{}｛｝<>＜＞「」『』【】〔〕〘〙〚〛ʕʔ")
BRACKET_PAIRS = {"(": ")", "（": "）", "[": "]", "［": "］", "ʕ": "ʔ"}
FACE_MARKS = frozenset(
    "^＾*_＿・･•､｡´｀`'’‘°ﾟ¯￣˘ωΩ∀▽△Дд皿益ಠಥ눈ツノ口ㅅᴥ꒳ᵕᗜᐛᐡᐢ"
)
ARM_MARKS = frozenset("\\/／＼┌┐└┘┏┓┗┛╰╯ヽﾉノ乁งづっ")

# Emoji sequences legitimately carry ZWJ and variation selectors; every
# other format/control character in a candidate marks it as face junk.
_EMOJI_CONTROLS = frozenset({"‍", "︎", "️"})

# Letters a Japanese IME candidate legitimately uses. Anything else (Thai,
# Arabic, Canadian syllabics, combining marks, ...) inside a bracket is face
# material — kaomoji borrow exotic glyphs faster than FACE_MARKS can list.
_FAMILIAR_LETTERS = (
    ("ぁ", "ヿ"),  # hiragana + katakana
    ("ㇰ", "ㇿ"),  # small katakana extension
    ("ｦ", "ﾟ"),  # halfwidth katakana
    ("一", "鿿"),  # CJK unified
    ("㐀", "䶿"),  # CJK extension A
    ("Ａ", "ｚ"),  # fullwidth Latin
)


def _is_exotic(char: str) -> bool:
    category = unicodedata.category(char)
    if char.isascii() or char in _EMOJI_CONTROLS:
        return False
    if category == "Nd":  # exotic digits drawn as limbs: ๑ ੭ (０-９ are familiar)
        return not "０" <= char <= "９"
    if not category.startswith(("L", "M")):
        return False
    if char in "々〆ヶー゙゚゛゜":
        return False
    return not any(low <= char <= high for low, high in _FAMILIAR_LETTERS)


def _is_decoration(char: str) -> bool:
    """Face-building material: punctuation, non-pictographic symbols, spaces,
    combining marks, listed face/arm glyphs, and exotic digits used as limbs
    (੭). `So` stays out — that is where emoji and legitimate symbol
    conversions (→, ♪) live."""
    if char in FACE_MARKS or char in ARM_MARKS:
        return True
    category = unicodedata.category(char)
    if category.startswith(("P", "M")) or category in ("Zs", "Sk", "Sm", "Sc"):
        return True
    return category == "Nd" and not char.isascii() and not "０" <= char <= "９"


def is_kaomoji_like(text: str) -> bool:
    """Detect symbol-structured emoticons (kaomoji); emoji pass through."""
    value = text.strip()
    if not value:
        return False
    if ASCII_EMOTICON.fullmatch(value):
        return True
    if any(
        unicodedata.category(char) in ("Cf", "Cc") and char not in _EMOJI_CONTROLS
        for char in value
    ):
        return True
    if len(value) > 48:
        return False

    symbol_count = sum(
        unicodedata.category(char).startswith(("P", "S")) for char in value
    )
    face_count = sum(char in FACE_MARKS for char in value)
    distinct_face_count = len({char for char in value if char in FACE_MARKS})
    has_bracket = any(char in BRACKETS for char in value)
    has_arm = any(char in ARM_MARKS for char in value)

    for opening, closing in BRACKET_PAIRS.items():
        start = value.find(opening)
        while start >= 0:
            closing_index = value.find(closing, start + 1, start + 27)
            if closing_index > start + 1:
                inner = value[start + 1 : closing_index]
                inner_symbols = sum(
                    unicodedata.category(char).startswith(("P", "S")) for char in inner
                )
                if inner_symbols >= 3 or (
                    inner_symbols >= 2 and any(char in FACE_MARKS for char in inner)
                ):
                    return True
                if any(_is_exotic(char) for char in inner):
                    return True
            start = value.find(opening, start + 1)

    if face_count >= 1 and symbol_count >= 2 and (has_bracket or has_arm):
        return True
    arm_extra = sum(
        char in ARM_MARKS and not unicodedata.category(char).startswith(("P", "S"))
        for char in value
    )
    if symbol_count + arm_extra >= 2 and any(_is_exotic(char) for char in value):
        return True
    if len(value) >= 2 and all(_is_decoration(char) for char in value):
        return True
    return face_count >= 2 and distinct_face_count >= 2 and symbol_count >= 1
