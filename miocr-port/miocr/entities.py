"""Entity extraction (`OCRData.OCREntity`) — recovered from libmiocr.so.

The SDK does this with `std::regex` in native code, not with a neural model: the
pattern strings are embedded in `libmiocr.so` and a `regexPattens` table applies them
to each recognised line. Recovered patterns (file offsets in the Gallery copy of
libmiocr.so):

  0x0451ed0  URL / website
  0x0451fec  phone (CN mobile, landline, 400/800, 100xx, 11x/12x, 123xx, 95xxx)
  0x0452170  e-mail

Supporting evidence: libc++ `std::regex` symbols (`basic_regex`, `__parse_alternative`,
`__word_boundary`, `__lookahead`, ...) in the same region, the log strings
"ocr entity caught exception :" / " regex error happend in  regexPattens  " /
" LineText " / "ocr entity caught regex_error: ", and the JNI marshalling strings
`com/xiaomi/ocr/sdk_ocr/OCRData$OCREntity` / `OCRData$Location` at 0x45352f / 0x4535f4.

Type ids come from `com.xiaomi.ocr.sdk_ocr.OCRData`:
    0 = ENTITY_TYPE_WEBSITE, 1 = ENTITY_TYPE_PHONE_NUMBER,
    2 = ENTITY_TYPE_EMAIL,   3 = ENTITY_TYPE_DELIVERY_NUMBER

Delivery numbers (type 3) are **not** covered by any pattern present in the APKs: neither
the Gallery nor the MIUIAICR build of `libmiocr.so` contains a courier rule, and there is
no courier vocabulary in either binary (see README "Entity types"). The rules for type 3
in this module are therefore *ours*, deliberately prefix/keyword anchored so precision
stays high; they are marked below.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

ENTITY_WEBSITE = 0
ENTITY_PHONE = 1
ENTITY_EMAIL = 2
ENTITY_DELIVERY = 3

ENTITY_NAMES = {
    ENTITY_WEBSITE: "website",
    ENTITY_PHONE: "phone_number",
    ENTITY_EMAIL: "email",
    ENTITY_DELIVERY: "delivery_number",
}

# Patterns copied verbatim from libmiocr.so (std::regex / ECMAScript syntax).
URL_PATTERN = (
    r"(([@]|^)|(?!\d*\.))(https?:\/\/)?((([a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+"
    r"(com|cn|net|org|gov|mil|edu))|(www\.([a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+"
    r"([a-zA-Z0-9][a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])))((\/([\w#!:.?_+=~&%@!\\\-\/])*([a-zA-Z0-9\/#])+)|\/)?"
    r"(?!:.?_+=&%@!\-)"
)
PHONE_PATTERN = (
    r"(([^0-9\)]|^)(((((\(\+?0{0,2}86\))|(\+?0{0,2}86))-?)?(1[3-9]\d[ -]?\d{4}[ -]?\d{4}))|"
    r"(((((0[1-2]\d)|(\(0[1-2]\d\)))[ -]?)?(([2-8]\d{3}-?\d{4})|(\d{5})))|"
    r"((((0[3-9]\d{2})|(\(0[3-9]\d{2}\)))[ -]?)?(([2-8]\d{2,3}-?\d{4})|(\d{5}))))|"
    r"((400|800)-?\d{3}-?\d{4})|(100(([0156]0)|(86)))|(11[0-79])|(12([012]))|"
    r"(123((1[02589])|(2\d)|(33)|(4[58])|(58)|(6[569])))|(95[0135]\d{2})|(952\d{3})))(?![\d])"
)
EMAIL_PATTERN = (
    r"[a-zA-Z0-9]+([._\\-]*[a-zA-Z0-9])*@([a-zA-Z0-9]+[-a-zA-Z0-9]*[a-zA-Z0-9]+\.){1,63}[a-zA-Z0-9]+"
)

PATTERNS: list[tuple[int, "re.Pattern[str]"]] = []
for _type, _src in ((ENTITY_WEBSITE, URL_PATTERN),
                    (ENTITY_PHONE, PHONE_PATTERN),
                    (ENTITY_EMAIL, EMAIL_PATTERN)):
    try:
        PATTERNS.append((_type, re.compile(_src)))
    except re.error:                                    # mirrors the SDK's guard
        pass
# e-mail before URL: the URL pattern also matches the host part of an address
PATTERNS.sort(key=lambda p: (p[0] != ENTITY_EMAIL,))


@dataclass
class Entity:
    entity_type: int
    entity_text: str
    start_index: int          # index into the line text
    end_index: int            # exclusive
    line_id: int = 0
    char_start: int = 0       # index into the line's char list
    char_end: int = 0         # exclusive
    box: tuple[float, float, float, float] | None = None

    @property
    def name(self) -> str:
        return ENTITY_NAMES.get(self.entity_type, f"unknown({self.entity_type})")


def _trim(match_text: str, entity_type: int) -> int:
    """Drop the leading context character some patterns consume.

    The phone pattern starts with `([^0-9\\)]|^)` and the URL pattern with `([@]|^)`:
    std::regex_match consumption includes that character, but the entity itself must
    not. Returns how many characters to strip from the front.
    """
    if not match_text:
        return 0
    if entity_type == ENTITY_PHONE and match_text[0] not in "0123456789(+" :
        return 1
    if entity_type == ENTITY_WEBSITE and match_text[0] == "@":
        return 1
    return 0


# ---------------------------------------------------------------------------
# type 3, delivery number — OUR rules (the vendor ships none in these builds)
# ---------------------------------------------------------------------------
# Carrier-prefixed forms: unambiguous on their own, so no keyword is required.
DELIVERY_PREFIX_PATTERNS = [
    ("sf", r"SF\d{12,15}"),                       # SF Express
    ("yto", r"YT\d{13}"),                         # YTO
    ("jdl", r"JD[A-Z]{0,3}\d{10,15}"),            # JD Logistics
    ("jt", r"JT\d{10,13}"),                       # J&T / 极兔
    ("sto", r"STO\d{10,15}"),                     # STO / 申通
    ("ems", r"E[A-Z]\d{9}CN"),                    # EMS international
    ("ups", r"1Z[0-9A-Z]{16}"),                   # UPS
    ("dhl", r"JJD\d{10,15}"),                     # DHL eCommerce / 京东海外
]
DELIVERY_KEYWORD = (
    r"(?:快递单号|快递号|运单号|物流单号|发货单号|包裹号|单号|快递|物流|包裹|已发货|已寄出"
    r"|tracking(?:\s*(?:no\.?|number|#))?|waybill(?:\s*(?:no\.?|number))?|parcel|consignment)"
)
DELIVERY_BARE_NUMBER = r"\b\d{12,20}\b"

_DELIVERY_PREFIX_RE = [(k, re.compile(p)) for k, p in DELIVERY_PREFIX_PATTERNS]
_DELIVERY_ANCHOR_RE = re.compile(
    DELIVERY_KEYWORD + r"[^0-9A-Za-z]{0,6}(" + DELIVERY_BARE_NUMBER + r")", re.I)
_DELIVERY_BARE_RE = re.compile(DELIVERY_BARE_NUMBER)


def _delivery_candidates(text: str, line_id: int) -> list[Entity]:
    """Carrier-prefixed tracking numbers, plus bare long digit runs next to a keyword."""
    out: list[Entity] = []
    taken: list[tuple[int, int]] = []
    for _key, pat in _DELIVERY_PREFIX_RE:
        for m in pat.finditer(text):
            out.append(Entity(ENTITY_DELIVERY, m.group(0), m.start(), m.end(), line_id))
            taken.append((m.start(), m.end()))
    for m in _DELIVERY_ANCHOR_RE.finditer(text):
        for num in _DELIVERY_BARE_RE.finditer(m.group(1)):
            start = m.start(1) + num.start()
            end = m.start(1) + num.end()
            if any(start < e and s < end for s, e in taken):
                continue
            out.append(Entity(ENTITY_DELIVERY, text[start:end], start, end, line_id))
            taken.append((start, end))
    return out


def extract_entities(text: str, chars: list | None = None, line_id: int = 0) -> list[Entity]:
    """Find entities in one recognised line, non-overlapping, left to right."""
    candidates: list[Entity] = []
    for etype, pat in PATTERNS:
        for m in pat.finditer(text):
            strip = _trim(m.group(0), etype)
            start, end = m.start() + strip, m.end()
            if end <= start:
                continue
            candidates.append(Entity(etype, text[start:end], start, end, line_id))
    candidates.extend(_delivery_candidates(text, line_id))
    # earliest start wins; on ties take the longer match
    candidates.sort(key=lambda e: (e.start_index, -(e.end_index - e.start_index)))
    chosen: list[Entity] = []
    for cand in candidates:
        if any(cand.start_index < e.end_index and e.start_index < cand.end_index for e in chosen):
            continue
        chosen.append(cand)
    if chars and len(chars) == len(text):
        for ent in chosen:
            ent.char_start = ent.start_index
            ent.char_end = ent.end_index
            span = chars[ent.start_index:ent.end_index]
            if span:
                xs0 = min(c.box[0] for c in span)
                ys0 = min(c.box[1] for c in span)
                xs1 = max(c.box[2] for c in span)
                ys1 = max(c.box[3] for c in span)
                ent.box = (xs0, ys0, xs1, ys1)
    return sorted(chosen, key=lambda e: e.start_index)
