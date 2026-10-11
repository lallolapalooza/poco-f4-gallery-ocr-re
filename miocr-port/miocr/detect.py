"""Text-line detection.

Fidelity note: the original engine detects lines with an *embedded neural* detector
(`ocr_detector_model`) driven through MACE. That model is **not shipped in any
libmiocr.so in this device dump** (see docs/HOW-IT-WORKS.md §"Detector gap"), so this
module provides a classical OpenCV detector as a drop-in stand-in, plus a pluggable
interface so a faithful detector can be substituted when its weights are available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np


@dataclass
class Box:
    """Axis-aligned line box in source-image coordinates."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)


class Detector(Protocol):
    def detect(self, gray: "np.ndarray") -> list[Box]:  # pragma: no cover - protocol
        ...


class WholeImageDetector:
    """Treat the whole image as a single text line."""

    def detect(self, gray: "np.ndarray") -> list[Box]:
        h, w = gray.shape[:2]
        return [Box(0, 0, w, h)]


class ClassicalDetector:
    """Otsu binarisation + horizontal run merging + line grouping.

    Works well for screenshots/documents (the common case for "extract text from a
    photo of text"); it does not model scene text the way the original detector does.
    """

    def __init__(self, min_height: int = 8, max_lines: int = 400):
        self.min_height = min_height
        self.max_lines = max_lines

    def detect(self, gray: "np.ndarray") -> list[Box]:
        import cv2

        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        # dark text on light background -> text = white
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # estimate glyph height from connected components
        n, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        heights = [stats[i, cv2.CC_STAT_HEIGHT] for i in range(1, n)
                   if self.min_height <= stats[i, cv2.CC_STAT_HEIGHT] <= h // 2
                   and stats[i, cv2.CC_STAT_AREA] > 8]
        glyph = int(np.median(heights)) if heights else max(self.min_height, h // 40)
        glyph = max(self.min_height, min(glyph, h // 4))

        # merge characters into words/lines, then split lines vertically
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, glyph), 1))
        merged = cv2.dilate(binary, kernel, iterations=1)
        contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes: list[Box] = []
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            if bh < self.min_height or bw < 2:
                continue
            boxes.append(Box(x, y, x + bw, y + bh))
        return self._group(boxes)

    def _group(self, boxes: Sequence[Box]) -> list[Box]:
        """Merge boxes that share a text line (same vertical band, close in x)."""
        boxes = sorted(boxes, key=lambda b: (b.y0, b.x0))
        lines: list[list[Box]] = []
        for b in boxes:
            placed = False
            for ln in lines:
                ref = ln[-1]
                overlap = min(ref.y1, b.y1) - max(ref.y0, b.y0)
                if overlap > 0.4 * min(ref.height, b.height) and b.x0 - ref.x1 < 1.5 * ref.height:
                    ln.append(b)
                    placed = True
                    break
            if not placed:
                lines.append([b])
        out = []
        for ln in lines:
            out.append(Box(min(x.x0 for x in ln), min(x.y0 for x in ln),
                           max(x.x1 for x in ln), max(x.y1 for x in ln)))
        out.sort(key=lambda b: (round(b.y0 / max(1.0, b.height)), b.x0))
        return out[: self.max_lines]


DEFAULT = ClassicalDetector
