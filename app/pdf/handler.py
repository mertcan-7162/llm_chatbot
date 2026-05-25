from __future__ import annotations

import logging

import fitz

from app.models.schemas import ExtractionMethod, PageResult
from app.ocr.engine import OCREngine

logger = logging.getLogger(__name__)

MIN_TEXT_LENGTH = 30
RENDER_DPI = 300


class PDFHandler:
    def __init__(self, ocr_engine: OCREngine) -> None:
        self.ocr_engine = ocr_engine

    def process(self, pdf_path: str) -> list[PageResult]:
        doc = fitz.open(pdf_path)
        results: list[PageResult] = []

        if doc.page_count == 0:
            doc.close()
            raise ValueError(
                f"PDF has no readable pages according to PyMuPDF: {pdf_path}"
            )

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text().strip()
            
            if len(text) >= MIN_TEXT_LENGTH:
                logger.debug("Page %d: native text extraction (%d chars)", page_num, len(text))
                results.append(
                    PageResult(page=page_num, text=text, method=ExtractionMethod.NATIVE)
                )
            else:
                logger.debug("Page %d: scanned, running OCR", page_num)
                pix = page.get_pixmap(dpi=RENDER_DPI)
                img_bytes = pix.tobytes("png")
                _, ocr_text = self.ocr_engine.extract_from_bytes(img_bytes)
                results.append(
                    PageResult(page=page_num, text=ocr_text, method=ExtractionMethod.OCR)
                )

        doc.close()
        return results

    @staticmethod
    def is_pdf(file_path: str) -> bool:
        return file_path.lower().endswith(".pdf")
