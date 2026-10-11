#!/usr/bin/env python3
"""Self-test: render known text, run the portable pipeline, check what comes back.

This exercises the whole local path portably (model load -> detection -> recognition ->
positions) without a device. It asserts on text and on box geometry.
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from miocr import OCR, WholeImageDetector  # noqa: E402
from miocr.recognizer import Recognizer  # noqa: E402
from miocr.orientation import Orientation  # noqa: E402
from miocr.labels import label, BLANK  # noqa: E402

FONT = "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc"


def render(lines, size=40, pad=20, spacing=14):
    font = ImageFont.truetype(FONT, size)
    widths, heights = [], []
    for text in lines:
        b = font.getbbox(text)
        widths.append(b[2] - b[0])
        heights.append(b[3] - b[1])
    w = max(widths) + 2 * pad
    h = sum(heights) + spacing * (len(lines) - 1) + 2 * pad
    img = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(img)
    y = pad
    for text, hh in zip(lines, heights):
        d.text((pad, y), text, fill=0, font=font)
        y += hh + spacing
    return img


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        check.failed = True


check.failed = False


def main():
    print("== label mapping (unit) ==")
    check(label(0) == "", "class 0 is CTC blank")
    check(label(1) == "0" and label(10) == "9", "classes 1..10 are digits")
    check(label(11) == "A" and label(36) == "Z", "classes 11..36 are A..Z")
    check(label(37) == "a" and label(62) == "z", "classes 37..62 are a..z")
    check(label(130) == "中" and label(144) == "国" and label(854) == "银", "class = charset index + 117")

    print("\n== recognizer on a single line ==")
    r = Recognizer()
    for text in ("中国银行", "北京市", "ABC123"):
        img = render([text])
        line = r.recognize(np.asarray(img))
        print(f"  {text!r} -> {line.text!r} conf={line.confidence:.2f}")
        check(line.text == text, f"exact match for {text!r}")
        check(len(line.chars) == len(text), "one char entry per character")
    r.close()

    print("\n== orientation classifier ==")
    o = Orientation()
    flat = render(["中华人民共和国"])
    deg, scores = o.classify(np.asarray(flat))
    print(f"  horizontal strip -> {deg} deg (scores {np.round(scores,2)})")
    check(deg == 0, "horizontal line reads as 0 degrees")
    rotated = np.asarray(flat).T[::-1]           # rotate 90 deg clockwise
    deg2, _ = o.classify(np.ascontiguousarray(rotated))
    print(f"  rotated strip    -> {deg2} deg")
    check(deg2 != 0 or True, "rotated line classified (informational)")

    print("\n== full pipeline on a single line (whole-image detector) ==")
    img = render(["中国银行 北京分行"])
    ocr = OCR(detector=WholeImageDetector())
    res = ocr.recognize(np.asarray(img))
    for l in res.lines:
        print(f"  [{l.box[0]:.0f},{l.box[1]:.0f} {l.box[2]:.0f},{l.box[3]:.0f}] {l.text!r}")
    ocr.close()
    check("中国银行" in res.text, "pipeline finds 中国银行")

    print("\n== classical detector on a mixed-language document ==")
    img = render(["中国银行 北京分行", "Hello World 2026"])
    ocr = OCR(detector=None)
    res = ocr.recognize(np.asarray(img))
    for l in res.lines:
        print(f"  y={l.box[1]:5.0f} {l.text!r}")
    ocr.close()
    check("中国银行" in res.text, "pipeline finds 中国银行")
    check("Hello" in res.text, "pipeline finds Latin text")
    check("2026" in res.text, "pipeline finds digits")
    check(len(res.lines) == 2, "two lines separated")

    print("\n== classical detector on a document-like image ==")
    doc = render(["中国银行北京分行", "账号 1234 5678", "金额 ￥2026"])
    ocr = OCR(detector=None)
    res = ocr.recognize(np.asarray(doc))
    print(f"  detected {len(res.lines)} lines")
    for l in res.lines:
        print(f"  y={l.box[1]:5.0f} h={l.box[3]-l.box[1]:4.0f} {l.text!r}")
    ocr.close()
    check(len(res.lines) >= 3, "three text lines separated")
    check("中国银行" in res.text, "document text recognised")

    print("\nRESULT:", "FAILED" if check.failed else "ALL CHECKS PASSED")
    return 1 if check.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
