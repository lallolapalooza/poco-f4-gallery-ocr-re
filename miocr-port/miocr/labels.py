"""Label mapping for Xiaomi's miocr recognizer (recovered from libmiocr.so).

The recognizer head has 7888 classes. Probing the carved model with rendered glyphs
established this mapping:

    class 0            -> CTC blank
    classes 1..10      -> '0'..'9'
    classes 11..36     -> 'A'..'Z'
    classes 37..62     -> 'a'..'z'
    classes 63..116    -> 54 additional labels that are not present in the extracted
                          charset table (special tokens / further ASCII); emitted as
                          UNKNOWN so nothing is silently invented
    classes 117..7863  -> charset.txt entry [class - 117]   (7747 entries)
    classes 7864..7887 -> unused padding

Evidence: single-glyph probes produced class == charset_index + 117 for every CJK
character (中 13->130, 国 27->144, 银 737->854, 的 3->120, 行 54->171), and
"Hello" decoded to classes 18/41/48/48/51 == 11+7, 37+4, 37+11, 37+11, 37+14.
"""

from __future__ import annotations

import os

BLANK = 0
_CJK_OFFSET = 117
NUM_CLASSES = 7888
UNKNOWN = "\ufffd"

# The model's dictionary spells ASCII out in code rather than in the .so's table, the
# same way classes 1..62 do. Probing glyph-by-glyph recovered the printable-ASCII block:
# class 6159 = ' ', 6160 = '!', 6161 = '"', ... 6191 = '~' (32 punctuation characters in
# ASCII order, alphanumerics excluded because they are already classes 1..62). The .so's
# charset array does not contain this block, so it must be applied before the CJK table.
_ASCII_BLOCK_START = 6159
_ASCII_BLOCK = " " + "".join(chr(c) for c in range(33, 127) if not chr(c).isalnum())

_charset: list[str] | None = None


def models_dir() -> str:
    return os.environ.get("MIOCR_MODELS", os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "models"))


def charset() -> list[str]:
    global _charset
    if _charset is None:
        with open(os.path.join(models_dir(), "charset.txt"), encoding="utf-8") as fh:
            _charset = fh.read().split("\n")
    return _charset


def label(cls: int) -> str:
    """Model class index -> character."""
    if cls == BLANK:
        return ""
    if 1 <= cls <= 10:
        return chr(ord("0") + cls - 1)
    if 11 <= cls <= 36:
        return chr(ord("A") + cls - 11)
    if 37 <= cls <= 62:
        return chr(ord("a") + cls - 37)
    if 63 <= cls <= 116:
        return UNKNOWN
    i = cls - _ASCII_BLOCK_START
    if 0 <= i < len(_ASCII_BLOCK):
        return _ASCII_BLOCK[i]
    cs = charset()
    i = cls - _CJK_OFFSET
    if 0 <= i < len(cs):
        return cs[i]
    return UNKNOWN
