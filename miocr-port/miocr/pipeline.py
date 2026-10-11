"""End-to-end portable OCR: image -> lines -> recognised text with positions.

Mirrors the structure the original SDK hands to Gallery
(`com.xiaomi.ocr.sdk_ocr.OCRData`: OCRResult -> OCRParagraph -> OCRLine -> OCRChar,
each with a Location), so consumers can work with the same shape of data.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np

from .detect import Box, ClassicalDetector, Detector, WholeImageDetector
from .orientation import Orientation
from .recognizer import Recognizer


@dataclass
class CharResult:
    text: str
    confidence: float
    box: tuple[float, float, float, float]   # absolute image coords


@dataclass
class LineResult:
    text: str
    confidence: float
    box: tuple[float, float, float, float]
    chars: list[CharResult] = field(default_factory=list)
    rotation: int = 0


@dataclass
class Result:
    lines: list[LineResult] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(l.text for l in self.lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "paragraphs": [
                {"lines": [{"text": l.text, "confidence": l.confidence, "box": list(l.box),
                            "rotation": l.rotation,
                            "chars": [{"text": c.text, "confidence": c.confidence,
                                       "box": list(c.box)} for c in l.chars]}
                          for l in self.lines]}
            ],
        }


class OCR:
    """Portable reimplementation of the miocr local path."""

    def __init__(self, detector: Detector | None = None, use_orientation: bool = True,
                 threads: int = 4):
        self.recognizer = Recognizer(threads=threads)
        self.orientation = Orientation(threads=min(2, threads)) if use_orientation else None
        self.detector: Detector = detector or ClassicalDetector()

    # ---- helpers -------------------------------------------------------
    @staticmethod
    def _to_gray(image: "np.ndarray") -> "np.ndarray":
        import cv2

        if image.ndim == 2:
            return image
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _rotate(self, gray: "np.ndarray", deg: int) -> "np.ndarray":
        import cv2

        if deg == 90:
            return cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
        if deg == 180:
            return cv2.rotate(gray, cv2.ROTATE_180)
        if deg == 270:
            return cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return gray

    # ---- public API ----------------------------------------------------
    def recognize(self, image: "np.ndarray") -> Result:
        gray = self._to_gray(image)
        boxes = self.detector.detect(gray)
        lines: list[LineResult] = []
        for b in boxes:
            x0, y0, x1, y1 = (int(b.x0), int(b.y0), int(np.ceil(b.x1)), int(np.ceil(b.y1)))
            crop = gray[max(0, y0):max(1, y1), max(0, x0):max(1, x1)]
            if crop.size == 0:
                continue
            rotation = 0
            if self.orientation is not None and crop.shape[0] < crop.shape[1] * 1.5:
                rotation, _ = self.orientation.classify(crop)
                if rotation:
                    crop = self._rotate(crop, rotation)
            line = self.recognizer.recognize(crop)
            if not line.text.strip():
                continue
            # chars: crop-local x -> image x (rotation re-mapping is approximate)
            chars = []
            for c in line.chars:
                cx0, cy0, cx1, cy1 = c.box
                if rotation in (0, 180):
                    ax0, ax1 = x0 + cx0, x0 + cx1
                    ay0, ay1 = y0, y1
                else:
                    span = max(1.0, y1 - y0)
                    ax0, ax1 = x0 + cy0 / max(1.0, crop.shape[0]) * span, x0 + cy1 / max(1.0, crop.shape[0]) * span
                    ay0, ay1 = y0, y1
                chars.append(CharResult(c.text, c.confidence, (ax0, ay0, ax1, ay1)))
            lines.append(LineResult(line.text, line.confidence, (x0, y0, x1, y1), chars, rotation))
        return Result(lines=lines)

    def recognize_file(self, path: str) -> Result:
        import cv2

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(path)
        return self.recognize(img)

    def close(self) -> None:
        self.recognizer.close()
