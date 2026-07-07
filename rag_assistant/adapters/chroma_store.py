"""
Vector store basato su ChromaDB.

ChromaDB è un database vettoriale che salva embedding su disco
e li recupera per similarità. In questa implementazione:

- I chunk vengono salvati con i loro embedding e metadati
- La ricerca usa cosine similarity (configurata nella collection)
- Il database è persistente: sopravvive al riavvio del processo
- Ogni chunk è identificato dal suo chunk_id (deterministico)

Persistenza:
    ChromaDB salva i dati in una cartella su disco (default: ./chroma_db).
    Al riavvio, basta riaprire il PersistentClient sulla stessa cartella
    e i dati sono ancora lì. Non serve re-indicizzare.

Perché cosine e non L2:
    L2 (distanza euclidea) misura la distanza "in linea d'aria" tra
    due punti. Cosine misura l'angolo tra due vettori, ignorando la
    magnitudine. Per gli embedding testuali, due testi con lo stesso
    significato possono avere vettori di lunghezza diversa (perché uno
    è più lungo dell'altro). Cosine li considera simili comunque.
    L2 li penalizzerebbe.

ChromaDB restituisce DISTANZE, non similarità:
    Con cosine, distanza = 1 - similarità.
    Distanza 0 = identici, distanza 2 = opposti.
    Noi convertiamo: score = 1 - distance, così score 1.0 = perfetto
    e score 0.0 = irrilevante. Più intuitivo per il debug.
"""

import logging
from pathlib import Path

import chromadb

from rag_assistant.adapters.base import VectorStore
from rag_assistant.core.config import settings
from rag_assistant.core.models import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)


class ChromaStore(VectorStore):
    """Vector store persistente basato su ChromaDB.

    Args:
        persist_dir: cartella dove salvare il database.
                     Default dal config.
        collection_name: nome della collection ChromaDB.
                        Una collection è l'equivalente di una "tabella"
                        in un database relazionale.
    """

    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str | None = None,
    ):
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self.collection_name = collection_name or settings.collection_name

        # Crea la cartella se non esiste
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

        # Client persistente: i dati sopravvivono al riavvio
        self._client = chromadb.PersistentClient(path=self.persist_dir)

        # get_or_create: se la collection esiste la apre,
        # altrimenti la crea. Evita errori al primo avvio
        # e al riavvio successivo.
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
        """Salva chunk con i loro embedding nel vector store.

        Usa upsert invece di add: se un chunk con lo stesso ID esiste
        già, lo sovrascrive invece di duplicarlo. Questo è fondamentale
        per il re-indexing: puoi rieseguire l'indicizzazione senza
        dover prima cancellare tutto.

        I metadati salvati per ogni chunk:
        - source_name: nome del file di origine (per le citazioni)
        - doc_id: ID del documento padre (per raggruppamenti)
        - word_count: numero di parole (per diagnostica)
        - chunker: tipo di chunker usato (per debug)

        ChromaDB accetta solo metadati con valori str, int, float, bool.
        I dict annidati non sono supportati, quindi serializziamo
        solo i campi rilevanti.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunk ma {len(embeddings)} embedding"
            )

        if not chunks:
            return

        # ChromaDB ha un limite di batch size (~5000).
        # Per sicurezza processiamo in batch da 500.
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
                    }
                    for c in batch_chunks
                ],
            )

            logger.info(
                f"Upsert batch: {min(i + batch_size, len(chunks))}/{len(chunks)}"
            )

    def search(self, embedding: list[float], top_k: int = 5) -> list[RetrievedChunk]:
        """Cerca i chunk più simili a un embedding.

        ChromaDB restituisce i risultati in ordine di distanza
        crescente (il più vicino prima). Noi convertiamo la
        distanza in score di similarità (1 - distance) per
        maggiore leggibilità.

        Se il vector store è vuoto, restituisce una lista vuota
        invece di crashare.
        """
        if self._collection.count() == 0:
            logger.warning("Vector store vuoto, nessun risultato")
            return []

        # Non chiedere più risultati di quanti ce ne sono
        actual_top_k = min(top_k, self._collection.count())

        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=actual_top_k,
            include=["documents", "metadatas", "distances"],
        )

        retrieved = []
        for i in range(len(results["documents"][0])):
            # Converti distanza cosine in score di similarità
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

    def clear(self) -> None:
        """Svuota il vector store eliminando e ricreando la collection.

        ChromaDB non ha un metodo "delete all". La strategia è:
        eliminare la collection e ricrearne una nuova con lo stesso
        nome e le stesse impostazioni.
        """
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Vector store svuotato")

    def count(self) -> int:
        """Restituisce il numero di chunk indicizzati."""
        return self._collection.count()
