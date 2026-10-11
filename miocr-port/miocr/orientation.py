"""Text-line orientation classifier (carved MNN model).

The SDK logs "ocr_orientation predict_orient use mnn" and exposes
OCRData.RotateFlags{ROTATE_0, ROTATE_90, ROTATE_180, ROTATE_270}; the carved model
takes one grayscale 1x48x192 line strip and emits 4 scores, i.e. the line's rotation.
"""

from __future__ import annotations

import os

import numpy as np

from .labels import models_dir

try:
    import MNN
except ImportError as exc:  # pragma: no cover
    raise ImportError("the orientation classifier needs MNN: pip install MNN") from exc

INPUT_NAME = "input_image"
OUTPUT_NAME = "output"
STRIP_H, STRIP_W = 48, 192
ROTATIONS = (0, 90, 180, 270)


class Orientation:
    def __init__(self, model_path: str | None = None, threads: int = 2):
        self.model_path = model_path or os.path.join(models_dir(), "orientation.mnn")
        self.interp = MNN.Interpreter(self.model_path)
        self.session = self.interp.createSession({"backend": "CPU", "numThread": threads})

    def classify(self, gray: "np.ndarray") -> tuple[int, np.ndarray]:
        """Return (rotation_degrees, scores) for a line crop."""
        import cv2

        resized = cv2.resize(gray, (STRIP_W, STRIP_H), interpolation=cv2.INTER_LINEAR)
        arr = ((resized.astype(np.float32) / 255.0 - 0.5) / 0.5).reshape(1, 1, STRIP_H, STRIP_W)
        t = self.interp.getSessionInput(self.session, INPUT_NAME)
        t.copyFrom(MNN.Tensor(t.getShape(), MNN.Halide_Type_Float, arr, t.getDimensionType()))
        self.interp.runSession(self.session)
        o = self.interp.getSessionOutput(self.session, OUTPUT_NAME)
        host = MNN.Tensor(o.getShape(), MNN.Halide_Type_Float,
                          np.zeros(o.getShape(), np.float32), o.getDimensionType())
        o.copyToHostTensor(host)
        scores = np.array(host.getData(), dtype=np.float32).reshape(-1)[:4]
        return ROTATIONS[int(scores.argmax())], scores
