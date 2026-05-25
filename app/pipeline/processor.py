from __future__ import annotations

import logging
from pathlib import Path

from app.llm.client import LLMClient
from app.models.schemas import (
    Chunk,
    ExtractionMethod,
    IngestResponse,
    PageResult,
    QueryResponse,
    SourceReference,
)
from app.ocr.engine import OCREngine
from app.pdf.handler import PDFHandler
from app.rag.chunker import TextChunker
from app.rag.embedder import Embedder
from app.rag.store import VectorStore

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}


class Pipeline:
    def __init__(self) -> None:
        self.ocr_engine = OCREngine()
        self.pdf_handler = PDFHandler(self.ocr_engine)
        self.chunker = TextChunker()
        self.embedder = Embedder()
        self.store = VectorStore()
        self.llm_client = LLMClient()

    async def ingest(
        self,
        file_path: str,
        collection_name: str = "default",
        original_filename: str | None = None,
    ) -> IngestResponse:
        path = Path(file_path)
        suffix = path.suffix.lower()
        display_name = original_filename or path.name

        if suffix == ".pdf":
            pages = self.pdf_handler.process(file_path)
        elif suffix in IMAGE_EXTENSIONS:
            pages = self._process_image(file_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        all_chunks: list[Chunk] = []
        methods_used: set[ExtractionMethod] = set()

        for page in pages:
            if not page.text.strip():
                continue
            methods_used.add(page.method)
            chunks = self.chunker.split(
                text=page.text,
                source=display_name,
                page=page.page,
                method=page.method,
            )
            all_chunks.extend(chunks)

        if all_chunks:
            texts = [c.text for c in all_chunks]
            embeddings = self.embedder.embed_documents(texts)
            self.store.add_documents(collection_name, all_chunks, embeddings)

        logger.info(
            "Ingested '%s': %d pages, %d chunks", display_name, len(pages), len(all_chunks)
        )

        return IngestResponse(
            source=display_name,
            total_pages=len(pages),
            total_chunks=len(all_chunks),
            collection_name=collection_name,
            methods_used=sorted(methods_used, key=lambda m: m.value),
        )

    def _process_image(self, image_path: str) -> list[PageResult]:
        _, structured_text = self.ocr_engine.extract(image_path)
        return [
            PageResult(page=0, text=structured_text, method=ExtractionMethod.OCR)
        ]

    async def query(
        self,
        query: str,
        collection_name: str = "default",
        n_results: int = 5,
        min_score: float = 0.3,
    ) -> QueryResponse:
        if not self.store.has_documents(collection_name):
            answer = await self.llm_client.chat(query)
            return QueryResponse(query=query, answer=answer, sources=[])

        query_embedding = self.embedder.embed_query(query)
        search_results = self.store.query(collection_name, query_embedding, n_results)
        search_results = [r for r in search_results if r.score >= min_score]

        if not search_results:
            answer = await self.llm_client.chat(query)
            return QueryResponse(query=query, answer=answer, sources=[])

        retrieved_texts = [r.text for r in search_results]
        answer = await self.llm_client.answer_with_context(query, retrieved_texts)

        sources = [
            SourceReference(
                text=r.text[:200],
                source=r.metadata.get("source", "unknown"),
                page=r.metadata.get("page", 0),
                score=r.score,
            )
            for r in search_results
        ]

        return QueryResponse(query=query, answer=answer, sources=sources)
