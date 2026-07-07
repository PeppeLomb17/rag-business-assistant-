"""Test per il ChromaDB vector store.

Usa una directory temporanea per ogni test, così:
- Non interferiscono tra loro
- Non inquinano il vector store di sviluppo
- Si puliscono automaticamente dopo l'esecuzione

Nota: questi test usano ChromaDB reale (non mockato) perché
è un database in-process — non serve un server esterno.
A differenza di Ollama, ChromaDB gira dentro il processo Python.
"""

import pytest

from rag_assistant.adapters.chroma_store import ChromaStore
from rag_assistant.core.models import Chunk, RetrievedChunk


def _make_chunk(
    doc_id: str = "doc001",
    source_name: str = "test.pdf",
    text: str = "chunk di test",
    chunk_index: int = 0,
    chunker: str = "sentence",
) -> Chunk:
    """Helper: crea un Chunk di test."""
    return Chunk(
        doc_id=doc_id,
        source_name=source_name,
        text=text,
        chunk_index=chunk_index,
        metadata={"chunker": chunker},
    )


def _fake_embedding(dimensions: int = 10, seed: float = 0.1) -> list[float]:
    """Helper: crea un embedding finto deterministico.

    Non serve un vero modello per testare il vector store.
    Usiamo vettori semplici dove possiamo prevedere la similarità.
    """
    return [seed + (i * 0.01) for i in range(dimensions)]


class TestChromaStore:

    @pytest.fixture
    def store(self, tmp_path):
        """Crea un vector store in una directory temporanea."""
        return ChromaStore(
            persist_dir=str(tmp_path / "chroma_test"),
            collection_name="test_collection",
        )

    def test_initially_empty(self, store):
        """Un vector store appena creato ha zero chunk."""
        assert store.count() == 0

    def test_add_and_count(self, store):
        """Dopo aver aggiunto chunk, il count si aggiorna."""
        chunks = [
            _make_chunk(text="primo chunk", chunk_index=0),
            _make_chunk(text="secondo chunk", chunk_index=1),
            _make_chunk(text="terzo chunk", chunk_index=2),
        ]
        embeddings = [
            _fake_embedding(seed=0.1),
            _fake_embedding(seed=0.2),
            _fake_embedding(seed=0.3),
        ]

        store.add(chunks, embeddings)
        assert store.count() == 3

    def test_search_returns_results(self, store):
        """La ricerca restituisce RetrievedChunk con score."""
        chunks = [
            _make_chunk(text="Le mele costano 2 euro al kg", chunk_index=0),
            _make_chunk(text="Le pere costano 3 euro al kg", chunk_index=1),
            _make_chunk(text="Il meteo domani sarà soleggiato", chunk_index=2),
        ]
        embeddings = [
            _fake_embedding(seed=0.10),  # simile alla query
            _fake_embedding(seed=0.11),  # simile alla query
            _fake_embedding(seed=0.90),  # diverso dalla query
        ]

        store.add(chunks, embeddings)

        # Query con embedding simile ai primi due chunk
        query_embedding = _fake_embedding(seed=0.10)
        results = store.search(query_embedding, top_k=2)

        assert len(results) == 2
        assert all(isinstance(r, RetrievedChunk) for r in results)
        assert all(r.score > 0 for r in results)
        # Il primo risultato deve essere il più simile
        assert results[0].score >= results[1].score

    def test_search_respects_top_k(self, store):
        """La ricerca restituisce al massimo top_k risultati."""
        chunks = [_make_chunk(chunk_index=i) for i in range(10)]
        embeddings = [_fake_embedding(seed=i * 0.1) for i in range(10)]

        store.add(chunks, embeddings)

        results = store.search(_fake_embedding(seed=0.0), top_k=3)
        assert len(results) == 3

    def test_search_empty_store(self, store):
        """La ricerca su uno store vuoto restituisce lista vuota."""
        results = store.search(_fake_embedding(), top_k=5)
        assert results == []

    def test_search_top_k_exceeds_count(self, store):
        """Se top_k > numero di chunk, restituisce tutti i chunk."""
        chunks = [_make_chunk(chunk_index=i) for i in range(3)]
        embeddings = [_fake_embedding(seed=i * 0.1) for i in range(3)]

        store.add(chunks, embeddings)

        results = store.search(_fake_embedding(), top_k=10)
        assert len(results) == 3

    def test_search_returns_correct_metadata(self, store):
        """I metadati del chunk vengono preservati nel risultato."""
        chunk = _make_chunk(
            doc_id="doc_abc",
            source_name="fattura.pdf",
            text="Importo totale: 15000 euro",
            chunk_index=7,
        )
        store.add([chunk], [_fake_embedding()])

        results = store.search(_fake_embedding(), top_k=1)

        assert len(results) == 1
        assert results[0].source_name == "fattura.pdf"
        assert results[0].metadata["doc_id"] == "doc_abc"
        assert results[0].metadata["chunk_index"] == 7

    def test_search_returns_retrieval_method(self, store):
        """Il metodo di retrieval è 'semantic'."""
        store.add([_make_chunk()], [_fake_embedding()])
        results = store.search(_fake_embedding(), top_k=1)

        assert results[0].retrieval_method == "semantic"

    def test_upsert_overwrites(self, store):
        """Aggiungere lo stesso chunk due volte non crea duplicati."""
        chunk = _make_chunk(text="versione 1", chunk_index=0)
        store.add([chunk], [_fake_embedding(seed=0.1)])
        assert store.count() == 1

        # Stesso chunk_id, testo diverso
        chunk_updated = _make_chunk(text="versione 2", chunk_index=0)
        store.add([chunk_updated], [_fake_embedding(seed=0.2)])
        assert store.count() == 1  # Ancora 1, non 2

        # Verifica che il testo sia aggiornato
        results = store.search(_fake_embedding(seed=0.2), top_k=1)
        assert results[0].text == "versione 2"

    def test_clear(self, store):
        """clear() svuota completamente lo store."""
        chunks = [_make_chunk(chunk_index=i) for i in range(5)]
        embeddings = [_fake_embedding(seed=i * 0.1) for i in range(5)]

        store.add(chunks, embeddings)
        assert store.count() == 5

        store.clear()
        assert store.count() == 0

    def test_add_empty_list(self, store):
        """Aggiungere una lista vuota non causa errori."""
        store.add([], [])
        assert store.count() == 0

    def test_add_mismatched_lengths_raises(self, store):
        """Se chunks e embeddings hanno lunghezze diverse, errore."""
        chunks = [_make_chunk(chunk_index=0), _make_chunk(chunk_index=1)]
        embeddings = [_fake_embedding()]  # Solo 1 embedding per 2 chunk

        with pytest.raises(ValueError, match="Mismatch"):
            store.add(chunks, embeddings)

    def test_persistence(self, tmp_path):
        """I dati sopravvivono alla chiusura e riapertura dello store."""
        persist_dir = str(tmp_path / "persistence_test")

        # Sessione 1: crea e popola
        store1 = ChromaStore(persist_dir=persist_dir, collection_name="persist")
        store1.add(
            [_make_chunk(text="dato persistente", chunk_index=0)],
            [_fake_embedding()],
        )
        assert store1.count() == 1
        del store1  # Simula la chiusura

        # Sessione 2: riapri e verifica
        store2 = ChromaStore(persist_dir=persist_dir, collection_name="persist")
        assert store2.count() == 1

        results = store2.search(_fake_embedding(), top_k=1)
        assert results[0].text == "dato persistente"

    def test_multiple_documents(self, store):
        """Chunk da documenti diversi coesistono nello stesso store."""
        chunk_pdf = _make_chunk(
            doc_id="doc_pdf",
            source_name="contratto.pdf",
            text="Clausola di rescissione",
            chunk_index=0,
        )
        chunk_excel = _make_chunk(
            doc_id="doc_excel",
            source_name="listino.xlsx — Prezzi",
            text="Fornitore: Ferrara | Prezzo: 1.20",
            chunk_index=0,
        )

        store.add(
            [chunk_pdf, chunk_excel],
            [_fake_embedding(seed=0.1), _fake_embedding(seed=0.5)],
        )
        assert store.count() == 2

        # Cerca qualcosa vicino al chunk PDF
        results = store.search(_fake_embedding(seed=0.1), top_k=2)
        assert len(results) == 2
        # Il primo risultato dovrebbe essere il PDF (embedding più vicino)
        assert results[0].source_name == "contratto.pdf"
