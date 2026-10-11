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
| Entity types (URL/phone/email) | not reimplemented (see HOW-IT-WORKS §App layer) |
| Text-line **detector** | **not faithful** — the neural detector is not shipped in any `libmiocr.so` in the device dump; a classical OpenCV detector is used instead |

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
| 63–116 | 54 labels absent from the extracted charset (emitted as U+FFFD) |
| 117–7863 | `charset.txt[class - 117]` — 7,747 entries ('¨','¡','¿', CJK by frequency, symbols) |
| 7864–7887 | unused |

Evidence: single-glyph probes returned `class == charset_index + 117` for every
character (中 13→130, 国 27→144, 银 737→854, 的 3→120, 行 54→171) and `Hello`
decoded as classes 18/41/48/48/51 = `11+7, 37+4, 37+11, 37+11, 37+14`.

Known limitation: the model itself is unreliable for ASCII punctuation and spaces —
probed glyphs land on neighbouring charset symbols (space → `charset[6042]` = `#`,
`-` → `·`, `!` → `%`), while CJK, Latin letters and digits decode exactly. The mapping
is faithful; the model's ASCII-punctuation head is not.

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
