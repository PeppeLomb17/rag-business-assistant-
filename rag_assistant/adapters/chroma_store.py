"""
Vector store basato su ChromaDB.

ChromaDB salva embedding su disco e li recupera per similarità.
Supporta upsert, ricerca per similarità e query sui metadati
per l'indicizzazione incrementale.
"""

import logging
from pathlib import Path

import chromadb

from rag_assistant.adapters.base import VectorStore
from rag_assistant.core.config import settings
from rag_assistant.core.models import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)


class ChromaStore(VectorStore):
    """Vector store persistente basato su ChromaDB."""

    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str | None = None,
    ):
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self.collection_name = collection_name or settings.collection_name

        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=self.persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            f"ChromaDB inizializzato: {self.persist_dir} "
            f"| collection: {self.collection_name} "
            f"| {self._collection.count()} chunk esistenti"
        )

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Salva chunk con upsert (sovrascrive se già esistono)."""
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunk ma {len(embeddings)} embedding"
            )

        if not chunks:
            return

        batch_size = 500

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_embeddings = embeddings[i : i + batch_size]

            self._collection.upsert(
                ids=[c.chunk_id for c in batch_chunks],
                embeddings=batch_embeddings,
                documents=[c.text for c in batch_chunks],
                metadatas=[
                    {
                        "source_name": c.source_name,
                        "doc_id": c.doc_id,
                        "chunk_index": c.chunk_index,
                        "word_count": c.word_count,
                        "chunker": c.metadata.get("chunker", "unknown"),
                        "category": c.metadata.get("category", "Generale"),
                    }
                    for c in batch_chunks
                ],
            )

            logger.info(
                f"Upsert batch: {min(i + batch_size, len(chunks))}/{len(chunks)}"
            )

    def search(self, embedding: list[float], top_k: int = 5) -> list[RetrievedChunk]:
        """Cerca i chunk più simili a un embedding."""
        if self._collection.count() == 0:
            logger.warning("Vector store vuoto, nessun risultato")
            return []

        actual_top_k = min(top_k, self._collection.count())

        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=actual_top_k,
            include=["documents", "metadatas", "distances"],
        )

        retrieved = []
        for i in range(len(results["documents"][0])):
            distance = results["distances"][0][i]
            score = 1.0 - distance
            metadata = results["metadatas"][0][i]

            retrieved.append(RetrievedChunk(
                chunk_id=results["ids"][0][i],
                text=results["documents"][0][i],
                source_name=metadata.get("source_name", "unknown"),
                score=score,
                retrieval_method="semantic",
                metadata=metadata,
            ))

        return retrieved

    def get_indexed_doc_ids(self) -> set[str]:
        """Restituisce l'insieme dei doc_id già indicizzati.

        Fondamentale per l'indicizzazione incrementale:
        se un doc_id è già presente, il file corrispondente
        può essere saltato.
        """
        if self._collection.count() == 0:
            return set()

        # Recupera tutti i metadati senza embedding né documenti
        all_data = self._collection.get(include=["metadatas"])

        doc_ids = set()
        for metadata in all_data["metadatas"]:
            if "doc_id" in metadata:
                doc_ids.add(metadata["doc_id"])

        return doc_ids

    def clear(self) -> None:
        """Svuota il vector store."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Vector store svuotato")

    def count(self) -> int:
        """Restituisce il numero di chunk indicizzati."""
        return self._collection.count()
