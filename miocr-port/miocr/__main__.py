"""CLI: python -m miocr IMAGE [options]"""

from __future__ import annotations

import argparse
import json
import sys

from .detect import ClassicalDetector, WholeImageDetector
from .pipeline import OCR


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="miocr", description="Portable reimplementation of Xiaomi's miocr local OCR")
    ap.add_argument("image", help="input image (anything OpenCV can read)")
    ap.add_argument("--json", metavar="FILE", help="write full result (text + boxes) as JSON")
    ap.add_argument("--text", action="store_true", help="print only the recognised text")
    ap.add_argument("--detector", choices=("classical", "whole"), default="classical")
    ap.add_argument("--no-orientation", action="store_true", help="skip the orientation model")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)

    detector = WholeImageDetector() if args.detector == "whole" else ClassicalDetector()
    ocr = OCR(detector=detector, use_orientation=not args.no_orientation, threads=args.threads)
    try:
        result = ocr.recognize_file(args.image)
    finally:
        ocr.close()

    if args.text:
        print(result.text)
    else:
        for line in result.lines:
            x0, y0, x1, y1 = line.box
            print(f"[{x0:6.0f},{y0:6.0f} {x1:6.0f},{y1:6.0f}] rot={line.rotation:3d} "
                  f"conf={line.confidence:.2f}  {line.text}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(result.as_dict(), fh, ensure_ascii=False, indent=2)
        print(f"wrote {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
