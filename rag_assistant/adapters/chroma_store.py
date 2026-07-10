"""
Vector store basato su ChromaDB.

Supporta ricerca per similarità con filtro opzionale sui metadati.
Il filtro permette di restringere la ricerca a una categoria
specifica (es: solo DDT PDF, solo Fatture Airone).
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

    def search(
        self,
        embedding: list[float],
        top_k: int = 5,
        category_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """Cerca i chunk più simili, con filtro opzionale per categoria.

        Args:
            embedding: vettore query.
            top_k: numero massimo di risultati.
            category_filter: se specificato, cerca solo in quella categoria.
                           Es: "DDT PDF", "Fatture Airone", "Generale".
        """
        if self._collection.count() == 0:
            logger.warning("Vector store vuoto, nessun risultato")
            return []

        actual_top_k = min(top_k, self._collection.count())

        # Costruisci il filtro per ChromaDB
        where_filter = None
        if category_filter:
            where_filter = {"category": category_filter}
            logger.info(f"Ricerca filtrata per categoria: {category_filter}")

        try:
            results = self._collection.query(
                query_embeddings=[embedding],
                n_results=actual_top_k,
                include=["documents", "metadatas", "distances"],
                where=where_filter,
            )
        except Exception as e:
            # Se il filtro non matcha niente, ChromaDB potrebbe dare errore
            # Riprova senza filtro
            logger.warning(f"Filtro '{category_filter}' fallito: {e}. Ricerca senza filtro.")
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
        """Restituisce l'insieme dei doc_id già indicizzati."""
        if self._collection.count() == 0:
            return set()

        all_data = self._collection.get(include=["metadatas"])

        doc_ids = set()
        for metadata in all_data["metadatas"]:
            if "doc_id" in metadata:
                doc_ids.add(metadata["doc_id"])

        return doc_ids

    def get_categories(self) -> set[str]:
        """Restituisce tutte le categorie presenti nel vector store.

        Utile per /status e per validare i filtri.
        """
        if self._collection.count() == 0:
            return set()

        all_data = self._collection.get(include=["metadatas"])

        categories = set()
        for metadata in all_data["metadatas"]:
            if "category" in metadata:
                categories.add(metadata["category"])

        return categories

    def clear(self) -> None:
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Vector store svuotato")

    def count(self) -> int:
        return self._collection.count()
