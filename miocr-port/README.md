# miocr-port — portable reimplementation of Xiaomi's on-device OCR

Turns the local ("Recognize text") OCR path of Xiaomi's MIUI Gallery into a
self-contained Python package that runs on any machine with MNN — no phone, no
Android, no cloud.

```
miocr-port/
  miocr/                  the portable implementation
    labels.py             model class -> character mapping (recovered, see below)
    recognizer.py         text-line recognizer (carved MNN model + CTC decode)
    orientation.py        line rotation classifier (carved MNN model)
    detect.py             text-line detection (classical stand-in, pluggable)
    pipeline.py           image -> lines -> text + boxes (OCRData-shaped result)
    __main__.py           CLI
  models/                 models carved out of libmiocr.so
    recognizer.mnn        3,017,516 B   in (N,1,32,W)  out (N,7888,W)
    orientation.mnn         926,320 B   in (1,1,48,192) out (1,4)
    charset.txt           7,747 labels, in model class order (minus the 117-class prefix)
    models.json           offsets/extents inside the source .so
  tools/                  reverse-engineering + verification tooling
    carve_models.py       re-carve models from a libmiocr.so
    scan_models.py        find MNN models in a .so (flatbuffer prefilter)
    mnnfb.py, aa64.py     flatbuffer walker / AArch64 static analysis helpers
    selftest.py           end-to-end check (renders text, asserts the result)
  docs/HOW-IT-WORKS.md    how the original feature works, with evidence
```

## Quick start

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt

.venv/bin/python tools/selftest.py                 # full end-to-end check
.venv/bin/python -m miocr photo.jpg                # lines with boxes
.venv/bin/python -m miocr photo.jpg --text         # text only
.venv/bin/python -m miocr photo.jpg --json out.json
```

```python
from miocr import OCR
ocr = OCR()
res = ocr.recognize_file("screenshot.png")
for line in res.lines:
    print(line.box, repr(line.text), line.confidence, [c.text for c in line.chars])
```

## Status

| Piece | State |
|---|---|
| Text-line recognizer (CJK + Latin + digits) | **reimplemented, runs locally, exact on tests** |
| Line orientation (0/90/180/270) | **reimplemented** |
| Character/line positions | **reimplemented** (per-character boxes from CTC timesteps) |
| Entity types (URL/phone/email) | **reimplemented** from the recovered `std::regex` rules (see below) |
| Entity type 3 (delivery number) | implemented with **our own** anchored rules — the vendor ships none in either build |
| Text-line **detector** | **not faithful** — the neural detector is not shipped in any `libmiocr.so` in the device dump; a classical OpenCV detector is used instead |

## Entity types

The SDK does entity extraction **in native code with `std::regex`**, not with a model.
The pattern table is embedded in `libmiocr.so` and applied to each recognised line:

| type | id | pattern (verbatim, file offset) |
|---|---|---|
| website | 0 | `0x0451ed0` — scheme-less or `http(s)://` host with TLD list `com\|cn\|net\|org\|gov\|mil\|edu`, `www.*`, optional path |
| phone number | 1 | `0x0451fec` — CN mobile (`1[3-9]\d[ -]?\d{4}[ -]?\d{4}`, optional `+86`), landline area codes `0[1-9]\d{2}`, `400/800-…`, `100xx`, `11[0-79]`, `12[012]`, `123xx`, `95xxx`, `952xxx` |
| email | 2 | `0x0452170` — `local@domain.tld` (dot-atom local part, 1–63 label repetitions) |
| delivery number | 3 | **vendor rule absent** — replaced by our own anchored rules (see below) |

Type 3 in the port is therefore **ours, not the vendor's**: neither the Gallery nor the
MIUIAICR `libmiocr.so` contains a courier rule or courier vocabulary, so `entities.py`
uses carrier-prefixed forms (`SF…`, `YT…`, `JD…`, `JT…`, `STO…`, `E…CN` EMS,
`1Z…` UPS, `JJD…` DHL) plus bare 12–20-digit runs directly anchored to a keyword
(`快递单号`, `运单号`, `物流单号`, `单号`, `已发货`, `tracking number`, `waybill`, `parcel`).
Prefix/keyword anchoring keeps precision high; it is deliberately conservative rather
than guessing at the vendor's unknown rule.

Evidence: libc++ regex symbols (`basic_regex`, `__parse_alternative`, `__word_boundary`,
`__lookahead`, `__bracket_expression`, …) clustered around the patterns; the log strings
`"ocr entity caught exception :"`, `" regex error happend in  regexPattens  "`,
`" LineText "`, `"ocr entity caught regex_error: "`; the JNI marshalling strings
`com/xiaomi/ocr/sdk_ocr/OCRData$OCREntity` / `OCRData$Location` at `0x453510`/`0x45352f`
and `[Lcom/xiaomi/ocr/sdk_ocr/OCRData$OCREntity;` at `0x4535d3`; the gate property
`persist.vendor.camera.ocr.entity` at `0x45380b`; and the code sites that reference the
patterns (`0xcd7d0`, `0xbe9a0`, `0xd297c`, `0xc13a0`, `0x341df4`, …) and the log strings
(`0xbc0d8`, `0xcddb4`, `0xda864`, `0xda884`, `0xc1468`, `0xc1508`, …), i.e. the entity
functions live in `0xbc000–0x0ee000`.

`miocr/entities.py` re-implements this: the three patterns are copied verbatim, matches
are made non-overlapping (earliest start, longest on ties), the leading context character
that the phone/URL patterns consume is stripped so `start_index`/`end_index` bound the
entity itself, and the entity box is the union of its characters' boxes. Two `re` vs
`std::regex` differences remain possible (both use ECMAScript syntax here).

Measured on rendered text (40 px CJK/Latin, Noto Sans CJK):

```
'中国银行'  -> '中国银行'  conf 1.00
'北京市'    -> '北京市'    conf 1.00
'ABC123'   -> 'ABC123'   conf 0.97
orientation: horizontal strip -> 0°, 90°-rotated strip -> 90°
3-line document -> 3 lines detected and recognised
```

## Label mapping (recovered)

The recognizer emits 7888 classes. Probing the carved model with rendered glyphs gave
an exact, self-consistent mapping:

| classes | meaning |
|---|---|
| 0 | CTC blank |
| 1–10 | `0`–`9` |
| 11–36 | `A`–`Z` |
| 37–62 | `a`–`z` |
| 63–116 | 54 labels not identified (emitted as U+FFFD) |
| **6159–6191** | **ASCII block**: `' '` then `!"#$%&'()*+,-./:;<=>?@[\]^_\`{|}~` |
| 117–7863 (outside that block) | `charset.txt[class - 117]` — 7,747 entries ('¨','¡','¿', CJK by frequency, symbols) |
| 7864–7887 | unused |

Evidence: single-glyph probes returned `class == charset_index + 117` for CJK characters
spanning the table (中 13→130, 国 27→144, 的 3→120, 行 54→171, 桥 1000→1117, 齿 2000→2117)
and `Hello` decoded as classes 18/41/48/48/51 = `11+7, 37+4, 37+11, 37+11, 37+14`.

The ASCII block is **not** in the `.so`'s charset array — the SDK spells ASCII out in code,
exactly as it does for the digit/letter prefix — so it was recovered glyph-by-glyph
(`'!'`→6160, `'"'`→6161 … `']'`→6184, `'_'`→6186, `'~'`→6191) and applied ahead of the CJK
table. With it, URLs and e-mail addresses decode exactly:

```
www.mi.com/store                   -> www.mi.com/store                    (website)
user.name@example.com              -> user.name@example.com               (email)
https://api.open.ai.xiaomi.com/v1  -> https://api.open.ai.xiaomi.com/v1   (website)
Tel: 13800138000                   -> Tel: 13800138000                    (phone)
```

## Provenance / licensing

`models/*.mnn` and `models/charset.txt` are extracted from Xiaomi's closed-source
`libmiocr.so` shipped in `MIUIGalleryGlobal.apk` on the device under analysis. They are
kept here only so the reimplementation can run; treat them as the vendor's content.

## Re-carving from your own dump

```bash
.venv/bin/python tools/carve_models.py /path/to/libmiocr.so models
```

The script finds model extents by bisecting the smallest buffer length that MNN's
flatbuffer verifier accepts (which is exactly the model's size), then writes
`recognizer.mnn`, `orientation.mnn`, `charset.txt` and `models.json`.
