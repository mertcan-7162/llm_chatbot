from __future__ import annotations

from app.config import settings
from app.models.schemas import Chunk, ChunkMetadata, ExtractionMethod


class TextChunker:
    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.separators = ["\n\n", "\n", ". ", " ", ""]

    def split(
        self,
        text: str,
        source: str,
        page: int,
        method: ExtractionMethod,
    ) -> list[Chunk]:
        raw_chunks = self._recursive_split(text, self.separators)
        prefix = f"[Source: {source} | Page: {page + 1}]\n"
        chunks: list[Chunk] = []
        for i, chunk_text in enumerate(raw_chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue
            chunks.append(
                Chunk(
                    text=prefix + chunk_text,
                    metadata=ChunkMetadata(
                        source=source,
                        page=page,
                        method=method,
                        chunk_index=i,
                    ),
                )
            )
        return chunks

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        if not separators:
            return self._split_by_size(text)

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            return self._split_by_size(text)

        parts = text.split(separator)
        chunks: list[str] = []
        current = ""

        for part in parts:
            candidate = separator.join([current, part]) if current else part

            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    if len(current) <= self.chunk_size:
                        chunks.append(current)
                    else:
                        chunks.extend(self._recursive_split(current, remaining_separators))
                current = part

        if current:
            if len(current) <= self.chunk_size:
                chunks.append(current)
            else:
                chunks.extend(self._recursive_split(current, remaining_separators))

        return self._apply_overlap(chunks)

    def _split_by_size(self, text: str) -> list[str]:
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        if len(chunks) <= 1 or self.chunk_overlap <= 0:
            return chunks

        overlapped: list[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            overlap_text = prev[-self.chunk_overlap :] if len(prev) > self.chunk_overlap else prev
            merged = overlap_text + chunks[i]
            if len(merged) <= self.chunk_size:
                overlapped.append(merged)
            else:
                overlapped.append(chunks[i])

        return overlapped
