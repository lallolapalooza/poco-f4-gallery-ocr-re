"""Portable reimplementation of Xiaomi's on-device miocr OCR path.

Models were carved out of `libmiocr.so` (MIUIGalleryGlobal.apk, Poco F4 / munch,
Android 14 Global). See README.md and docs/HOW-IT-WORKS.md.
"""

from .labels import charset, label
from .entities import (Entity, extract_entities, ENTITY_WEBSITE, ENTITY_PHONE,
                       ENTITY_EMAIL, ENTITY_DELIVERY, ENTITY_NAMES, PATTERNS)
from .orientation import Orientation
from .recognizer import Recognizer, Line, Char
from .pipeline import OCR, Result, LineResult, CharResult
from .detect import Box, ClassicalDetector, WholeImageDetector, Detector

__all__ = [
    "charset", "label", "Orientation", "Recognizer", "Line", "Char",
    "OCR", "Result", "LineResult", "CharResult",
    "Box", "ClassicalDetector", "WholeImageDetector", "Detector",
    "Entity", "extract_entities", "ENTITY_WEBSITE", "ENTITY_PHONE",
    "ENTITY_EMAIL", "ENTITY_DELIVERY", "ENTITY_NAMES", "PATTERNS",
]
__version__ = "0.1.0"
