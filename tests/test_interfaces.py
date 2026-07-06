"""Test per le interfacce astratte.

Verifica che:
1. Le ABC non siano istanziabili direttamente
2. Una sottoclasse incompleta non sia istanziabile
3. Una sottoclasse completa funzioni correttamente
"""

import pytest
from rag_assistant.adapters.base import (
    DocumentLoader, Chunker, Embedder, VectorStore, LLMProvider,
)
from rag_assistant.core.models import Document, Chunk, RetrievedChunk


class TestDocumentLoaderABC:
    """Verifica il contratto di DocumentLoader."""

    def test_cannot_instantiate(self):
        """Una ABC non può essere istanziata direttamente."""
        with pytest.raises(TypeError):
            DocumentLoader()

    def test_incomplete_subclass_fails(self):
        """Una sottoclasse che non implementa load() non è istanziabile."""
        class BadLoader(DocumentLoader):
            pass  # dimentico load()

        with pytest.raises(TypeError):
            BadLoader()

    def test_complete_subclass_works(self):
        """Una sottoclasse che implementa load() funziona."""
        class FakeLoader(DocumentLoader):
            def load(self, file_path: str) -> list[Document]:
                return [Document(
                    source_path=file_path,
                    source_name="fake.txt",
                    doc_type="txt",
                    text="contenuto fake",
                )]

            @staticmethod
            def supported_extensions() -> list[str]:
                return [".txt"]

        loader = FakeLoader()
        docs = loader.load("/fake/path.txt")

        assert len(docs) == 1
        assert docs[0].doc_type == "txt"
        assert docs[0].text == "contenuto fake"
        assert FakeLoader.supported_extensions() == [".txt"]


class TestChunkerABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            Chunker()

    def test_complete_subclass_works(self):
        """Un chunker che taglia ogni 3 parole — solo per test."""
        class SimpleChunker(Chunker):
            def chunk(self, document: Document) -> list[Chunk]:
                words = document.text.split()
                chunks = []
                for i in range(0, len(words), 3):
                    chunk_text = " ".join(words[i : i + 3])
                    chunks.append(Chunk(
                        doc_id=document.doc_id,
                        source_name=document.source_name,
                        text=chunk_text,
                        chunk_index=len(chunks),
                    ))
                return chunks

        doc = Document(
            source_path="/test",
            source_name="test.txt",
            doc_type="txt",
            text="uno due tre quattro cinque sei sette otto nove",
        )
        chunker = SimpleChunker()
        chunks = chunker.chunk(doc)

        assert len(chunks) == 3
        assert chunks[0].text == "uno due tre"
        assert chunks[1].text == "quattro cinque sei"
        assert chunks[2].text == "sette otto nove"
        assert chunks[0].chunk_index == 0
        assert chunks[1].chunk_index == 1


class TestEmbedderABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            Embedder()

    def test_complete_subclass_works(self):
        """Un embedder finto che restituisce vettori di lunghezza fissa."""
        class FakeEmbedder(Embedder):
            def embed(self, text: str) -> list[float]:
                # Vettore finto: la lunghezza del testo normalizzata
                return [len(text) / 100.0] * 4

            def embed_batch(self, texts: list[str]) -> list[list[float]]:
                return [self.embed(t) for t in texts]

        embedder = FakeEmbedder()
        vec = embedder.embed("ciao")
        batch = embedder.embed_batch(["ciao", "mondo"])

        assert len(vec) == 4
        assert len(batch) == 2
        assert len(batch[0]) == 4


class TestVectorStoreABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            VectorStore()

    def test_complete_subclass_works(self):
        """Un vector store in memoria — solo per test."""
        class MemoryStore(VectorStore):
            def __init__(self):
                self._data = []

            def add(self, chunks, embeddings):
                for c, e in zip(chunks, embeddings):
                    self._data.append((c, e))

            def search(self, embedding, top_k=5):
                # Restituisce tutto, senza ranking reale
                results = []
                for chunk, _ in self._data[:top_k]:
                    results.append(RetrievedChunk(
                        chunk_id=chunk.chunk_id,
                        text=chunk.text,
                        source_name=chunk.source_name,
                        score=0.99,
                    ))
                return results

            def clear(self):
                self._data = []

            def count(self):
                return len(self._data)

        store = MemoryStore()
        assert store.count() == 0

        chunk = Chunk(
            doc_id="abc", source_name="f.pdf",
            text="test", chunk_index=0,
        )
        store.add([chunk], [[0.1, 0.2, 0.3]])
        assert store.count() == 1

        results = store.search([0.1, 0.2, 0.3], top_k=1)
        assert len(results) == 1
        assert results[0].text == "test"

        store.clear()
        assert store.count() == 0


class TestLLMProviderABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            LLMProvider()

    def test_complete_subclass_works(self):
        """Un LLM finto che fa eco della domanda."""
        class EchoLLM(LLMProvider):
            def generate(self, system_prompt: str, user_prompt: str) -> str:
                return f"Echo: {user_prompt}"

        llm = EchoLLM()
        answer = llm.generate("system", "Quanti cluster?")
        assert answer == "Echo: Quanti cluster?"
