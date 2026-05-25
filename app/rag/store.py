from __future__ import annotations

import logging
from dataclasses import dataclass

import chromadb

from app.config import settings
from app.models.schemas import Chunk

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    text: str
    metadata: dict
    score: float


class VectorStore:
    def __init__(self, persist_dir: str | None = None) -> None:
        path = persist_dir or settings.chroma_persist_dir
        self.client = chromadb.PersistentClient(path=path)

    def get_or_create_collection(self, name: str) -> chromadb.Collection:
        return self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(
        self,
        collection_name: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        collection = self.get_or_create_collection(collection_name)
        collection.upsert(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[c.metadata.model_dump() for c in chunks],
        )
        logger.info(
            "Upserted %d chunks into collection '%s'", len(chunks), collection_name
        )

    def query(
        self,
        collection_name: str,
        query_embedding: list[float],
        n_results: int = 5,
    ) -> list[SearchResult]:
        collection = self.get_or_create_collection(collection_name)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        search_results: list[SearchResult] = []
        if not results["documents"] or not results["documents"][0]:
            return search_results

        for doc, meta, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            score = 1.0 - distance
            search_results.append(SearchResult(text=doc, metadata=meta, score=score))

        return search_results

    def list_sources(self, collection_name: str) -> list[str]:
        try:
            collection = self.client.get_collection(collection_name)
        except Exception:
            return []
        result = collection.get(include=["metadatas"])
        if not result["metadatas"]:
            return []
        sources = sorted({m["source"] for m in result["metadatas"] if "source" in m})
        return sources

    def delete_by_source(self, collection_name: str, source: str) -> int:
        try:
            collection = self.client.get_collection(collection_name)
        except Exception:
            return 0
        before = collection.count()
        collection.delete(where={"source": source})
        after = collection.count()
        deleted = before - after
        logger.info("Deleted %d chunks for source '%s' from '%s'", deleted, source, collection_name)
        return deleted

    def has_documents(self, collection_name: str) -> bool:
        try:
            collection = self.client.get_collection(collection_name)
            return collection.count() > 0
        except Exception:
            return False

    def list_collections(self) -> list[str]:
        return [c.name for c in self.client.list_collections()]

    def delete_collection(self, name: str) -> None:
        self.client.delete_collection(name)
        logger.info("Deleted collection '%s'", name)
