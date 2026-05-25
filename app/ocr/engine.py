from __future__ import annotations

import io
import logging
import os
from pathlib import Path

import numpy as np
from PIL import Image

from app.config import settings
from app.models.schemas import OCRResult
from app.ocr.postprocessor import OCRPostprocessor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(PROJECT_ROOT / ".cache" / "paddlex"))
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from paddleocr import PaddleOCR  # noqa: E402

logger = logging.getLogger(__name__)


class OCREngine:
    _instance: PaddleOCR | None = None

    def __init__(self) -> None:
        self.postprocessor = OCRPostprocessor(
            confidence_threshold=settings.ocr_confidence_threshold,
        )

    @classmethod
    def _get_paddleocr(cls) -> PaddleOCR:
        if cls._instance is None:
            cls._instance = PaddleOCR(
                lang=settings.ocr_lang,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=True,
                enable_mkldnn=False,
                device="cpu",
            )
        return cls._instance

    def _parse_raw_result(self, raw: list) -> list[OCRResult]:
        results: list[OCRResult] = []
        if not raw:
            return results

        if self._is_paddleocr_v3_result(raw):
            return self._parse_v3_result(raw)

        for line_group in raw:
            if not line_group:
                continue
            for detection in line_group:
                bbox, (text, confidence) = detection
                results.append(
                    OCRResult(text=text, confidence=confidence, bbox=bbox)
                )
        return results

    def _is_paddleocr_v3_result(self, raw: list) -> bool:
        return bool(raw and hasattr(raw[0], "get") and raw[0].get("rec_texts") is not None)

    def _parse_v3_result(self, raw: list) -> list[OCRResult]:
        results: list[OCRResult] = []
        for page_result in raw:
            texts = page_result.get("rec_texts", [])
            scores = page_result.get("rec_scores", [])
            polygons = page_result.get("rec_polys") or page_result.get("dt_polys") or []

            for text, confidence, polygon in zip(texts, scores, polygons):
                bbox = np.asarray(polygon).astype(float).tolist()
                results.append(
                    OCRResult(text=text, confidence=float(confidence), bbox=bbox)
                )
        return results

    def extract(self, image_path: str) -> tuple[list[OCRResult], str]:
        """Extract text from an image file.

        Returns (raw_results, structured_text).
        """
        ocr = self._get_paddleocr()
        raw = ocr.predict(image_path)
        results = self._parse_raw_result(raw)
        filtered = self.postprocessor.filter_by_confidence(results)
        structured_text = self.postprocessor.to_structured_text(filtered)
        return results, structured_text

    def extract_from_bytes(self, img_bytes: bytes) -> tuple[list[OCRResult], str]:
        """Extract text from raw image bytes (e.g. rendered PDF page)."""
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_array = np.array(image)

        ocr = self._get_paddleocr()
        raw = ocr.predict(img_array)
        results = self._parse_raw_result(raw)
        filtered = self.postprocessor.filter_by_confidence(results)
        structured_text = self.postprocessor.to_structured_text(filtered)
        return results, structured_text
