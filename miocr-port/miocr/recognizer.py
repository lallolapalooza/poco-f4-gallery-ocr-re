"""Text-line recognizer: the carved MNN model + recovered preprocessing/decoding.

The original feature calls, in order (see `docs/HOW-IT-WORKS.md`):
    OCREngine.nativeOCRLineDetectBitmap(bitmap)      -> line boxes
    OCREngine.nativeOCRRecognizeBitmap(bitmap, boxes)-> OCRResult{paragraphs->lines->chars}
This module reimplements the recognition half portably: MNN (x86/arm, CPU) runs the
model carved out of libmiocr.so, with the same glyph-level output.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from .labels import BLANK, NUM_CLASSES, label, models_dir

try:  # MNN is a hard dependency of the recognition path
    import MNN
except ImportError as exc:  # pragma: no cover
    raise ImportError("the portable recognizer needs MNN: pip install MNN") from exc

INPUT_NAME = "input_image"
OUTPUT_NAME = "output"
LINE_HEIGHT = 32          # model input height (N,1,32,W)
NORM = "pm1"              # (x/255 - 0.5)/0.5; the graph is insensitive to scale


@dataclass
class Char:
    text: str
    confidence: float
    box: tuple[float, float, float, float]   # x0,y0,x1,y1 in crop coordinates


@dataclass
class Line:
    text: str
    confidence: float
    chars: list[Char] = field(default_factory=list)


class Recognizer:
    """Single-text-line recognizer backed by the carved MNN model."""

    def __init__(self, model_path: str | None = None, threads: int = 4):
        self.model_path = model_path or os.path.join(models_dir(), "recognizer.mnn")
        self.interp = MNN.Interpreter(self.model_path)
        self.session = self.interp.createSession({"backend": "CPU", "numThread": threads})

    # ---- preprocessing -------------------------------------------------
    @staticmethod
    def _to_input(gray: "np.ndarray", height: int = LINE_HEIGHT) -> np.ndarray:
        """Resize to (1,1,32,W) and normalise; matches the SDK's decode path."""
        import cv2  # imported lazily so the package works without cv2 for boxes

        h, w = gray.shape[:2]
        new_w = max(1, int(round(w * height / h)))
        resized = cv2.resize(gray, (new_w, height), interpolation=cv2.INTER_LINEAR)
        arr = resized.astype(np.float32)
        arr = (arr / 255.0 - 0.5) / 0.5
        return arr.reshape(1, 1, height, new_w)

    # ---- inference -----------------------------------------------------
    def run(self, gray: "np.ndarray") -> np.ndarray:
        """Return per-timestep class logits with shape (T, 7888)."""
        arr = self._to_input(gray)
        t = self.interp.getSessionInput(self.session, INPUT_NAME)
        self.interp.resizeTensor(t, arr.shape)
        self.interp.resizeSession(self.session)
        t = self.interp.getSessionInput(self.session, INPUT_NAME)
        t.copyFrom(MNN.Tensor(t.getShape(), MNN.Halide_Type_Float, arr, t.getDimensionType()))
        self.interp.runSession(self.session)
        o = self.interp.getSessionOutput(self.session, OUTPUT_NAME)
        host = MNN.Tensor(o.getShape(), MNN.Halide_Type_Float,
                          np.zeros(o.getShape(), np.float32), o.getDimensionType())
        o.copyToHostTensor(host)
        flat = np.array(host.getData(), dtype=np.float32)
        return flat.reshape(flat.size // NUM_CLASSES, NUM_CLASSES)

    # ---- decoding ------------------------------------------------------
    def recognize(self, gray: "np.ndarray") -> Line:
        """Greedy CTC decode; per-character boxes come from the timestep span."""
        logits = self.run(gray)
        # log-softmax output -> probabilities for confidence reporting
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        idx = logits.argmax(axis=1)
        h, w = gray.shape[:2]
        t_total = idx.shape[0]
        chars: list[Char] = []
        prev = None
        start = 0
        for t, c in enumerate(idx):
            c = int(c)
            if c != prev:
                if prev is not None and prev != BLANK:
                    ch = label(prev)
                    x0 = start / t_total * w
                    x1 = t / t_total * w
                    chars.append(Char(ch, float(probs[start:t, prev].mean()), (x0, 0.0, x1, float(h))))
                start = t
                prev = c
        if prev is not None and prev != BLANK:
            ch = label(prev)
            chars.append(Char(ch, float(probs[start:t_total, prev].mean()),
                              (start / t_total * w, 0.0, float(w), float(h))))
        text = "".join(c.text for c in chars)
        conf = float(np.mean([c.confidence for c in chars])) if chars else 0.0
        return Line(text=text, confidence=conf, chars=chars)

    def close(self) -> None:
        try:
            self.interp.releaseSession(self.session)
        except Exception:
            pass
