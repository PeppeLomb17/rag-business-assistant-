"""Test per il RAGService.

Usa fake adapters come nel Blocco 7. Il test verifica che il
service assembli correttamente la pipeline: embed → search →
prompt → generate → response.

Non testa la qualità delle risposte (quello dipende dal modello).
Testa che il flusso sia corretto, che gli errori siano gestiti,
e che le metriche siano calcolate.
"""

import pytest

from rag_assistant.adapters.base import Embedder, VectorStore, LLMProvider
from rag_assistant.core.models import Chunk, RetrievedChunk, RAGResponse
from rag_assistant.services.rag_service import RAGService, SYSTEM_PROMPT


# ─── Fake Adapters ────────────────────────────────────────────────────────────

class FakeEmbedder(Embedder):
    def __init__(self):
        self.last_text = None

    def embed(self, text: str) -> list[float]:
        self.last_text = text
        return [0.1] * 10

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 10 for _ in texts]


class FakeVectorStore(VectorStore):
    """Store finto che restituisce chunk predefiniti."""

    def __init__(self, fake_results: list[RetrievedChunk] | None = None):
        self._results = fake_results or []
        self._count = len(self._results)

    def add(self, chunks, embeddings):
        self._count += len(chunks)

    def search(self, embedding, top_k=5):
        return self._results[:top_k]

    def clear(self):
        self._results = []
        self._count = 0

    def count(self):
        return self._count


class FakeLLM(LLMProvider):
    """LLM finto che restituisce una risposta predefinita.

    Salva system_prompt e user_prompt per le assert nei test.
    """

    def __init__(self, answer: str = "Risposta di test."):
        self.answer = answer
        self.model = "fake-model"
        self.last_system_prompt = None
        self.last_user_prompt = None
        self.call_count = 0

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        self.call_count += 1
        return self.answer


class ErrorLLM(LLMProvider):
    """LLM che simula un errore di connessione."""

    def __init__(self, error_type: type = ConnectionError):
        self.error_type = error_type

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise self.error_type("Ollama non raggiungibile")


# ─── Helper ───────────────────────────────────────────────────────────────────

def _fake_chunks(n: int = 3) -> list[RetrievedChunk]:
    """Crea N chunk finti con score decrescente."""
    return [
        RetrievedChunk(
            chunk_id=f"chunk_{i}",
            text=f"Contenuto del chunk numero {i} con informazioni rilevanti.",
            source_name=f"documento_{i}.pdf",
            score=0.95 - (i * 0.1),
        )
        for i in range(n)
    ]


# ─── Test ─────────────────────────────────────────────────────────────────────

class TestRAGService:

    def test_query_returns_rag_response(self):
        """Una query restituisce un RAGResponse completo."""
        service = RAGService(
            embedder=FakeEmbedder(),
            store=FakeVectorStore(fake_results=_fake_chunks(3)),
            llm=FakeLLM(answer="Il costo è 1.20 euro/kg."),
        )

        response = service.query("Quanto costa l'uva?")

        assert isinstance(response, RAGResponse)
        assert response.success is True
        assert response.query == "Quanto costa l'uva?"
        assert response.answer == "Il costo è 1.20 euro/kg."
        assert len(response.chunks_used) == 3
        assert response.model == "fake-model"
        assert response.retrieval_time_ms >= 0
        assert response.generation_time_ms >= 0

    def test_query_embeds_the_question(self):
        """La domanda viene passata all'embedder."""
        embedder = FakeEmbedder()
        service = RAGService(
            embedder=embedder,
            store=FakeVectorStore(fake_results=_fake_chunks()),
            llm=FakeLLM(),
        )

        service.query("Qual è il prezzo dell'uva?")

        assert embedder.last_text == "Qual è il prezzo dell'uva?"

    def test_query_passes_context_to_llm(self):
        """Il LLM riceve il contesto con i chunk recuperati."""
        chunks = _fake_chunks(2)
        llm = FakeLLM()
        service = RAGService(
            embedder=FakeEmbedder(),
            store=FakeVectorStore(fake_results=chunks),
            llm=llm,
        )

        service.query("Dimmi il prezzo.")

        # Il user prompt deve contenere il testo dei chunk
        assert "Contenuto del chunk numero 0" in llm.last_user_prompt
        assert "Contenuto del chunk numero 1" in llm.last_user_prompt
        # E la domanda
        assert "Dimmi il prezzo." in llm.last_user_prompt
        # E il system prompt deve essere quello anti-allucinazione
        assert "SOLO" in llm.last_system_prompt

    def test_query_respects_top_k(self):
        """Il retrieval usa il top_k configurato."""
        service = RAGService(
            embedder=FakeEmbedder(),
            store=FakeVectorStore(fake_results=_fake_chunks(10)),
            llm=FakeLLM(),
            top_k=3,
        )

        response = service.query("Test")

        assert len(response.chunks_used) == 3

    def test_query_empty_store(self):
        """Con store vuoto, risponde senza chiamare il LLM."""
        llm = FakeLLM()
        service = RAGService(
            embedder=FakeEmbedder(),
            store=FakeVectorStore(fake_results=[]),
            llm=llm,
        )

        response = service.query("Test")

        assert response.success is True
        assert "Nessun documento" in response.answer
        assert llm.call_count == 0  # LLM non chiamato

    def test_query_connection_error(self):
        """Se il LLM non è raggiungibile, restituisce errore gestito."""
        service = RAGService(
            embedder=FakeEmbedder(),
            store=FakeVectorStore(fake_results=_fake_chunks()),
            llm=ErrorLLM(ConnectionError),
        )

        response = service.query("Test")

        assert response.success is False
        assert "connessione" in response.error.lower()

    def test_query_timeout_error(self):
        """Se il LLM va in timeout, restituisce errore gestito."""
        service = RAGService(
            embedder=FakeEmbedder(),
            store=FakeVectorStore(fake_results=_fake_chunks()),
            llm=ErrorLLM(TimeoutError),
        )

        response = service.query("Test")

        assert response.success is False
        assert "Timeout" in response.error

    def test_query_unexpected_error(self):
        """Errori imprevisti vengono catturati e restituiti."""
        service = RAGService(
            embedder=FakeEmbedder(),
            store=FakeVectorStore(fake_results=_fake_chunks()),
            llm=ErrorLLM(RuntimeError),
        )

        response = service.query("Test")

        assert response.success is False
        assert response.error is not None

    def test_prompt_contains_source_names(self):
        """Il prompt include i nomi dei file sorgente."""
        chunks = [
            RetrievedChunk(
                chunk_id="c1",
                text="Dati della fattura",
                source_name="fattura_marzo.pdf",
                score=0.92,
            ),
        ]
        llm = FakeLLM()
        service = RAGService(
            embedder=FakeEmbedder(),
            store=FakeVectorStore(fake_results=chunks),
            llm=llm,
        )

        service.query("Mostrami la fattura")

        assert "fattura_marzo.pdf" in llm.last_user_prompt
        assert "0.92" in llm.last_user_prompt

    def test_prompt_contains_relevance_scores(self):
        """Il prompt include i punteggi di rilevanza."""
        chunks = _fake_chunks(1)
        llm = FakeLLM()
        service = RAGService(
            embedder=FakeEmbedder(),
            store=FakeVectorStore(fake_results=chunks),
            llm=llm,
        )

        service.query("Test")

        assert "Rilevanza:" in llm.last_user_prompt

    def test_custom_system_prompt(self):
        """Il system prompt è personalizzabile."""
        custom_prompt = "Sei un esperto di agricoltura."
        llm = FakeLLM()
        service = RAGService(
            embedder=FakeEmbedder(),
            store=FakeVectorStore(fake_results=_fake_chunks()),
            llm=llm,
            system_prompt=custom_prompt,
        )

        service.query("Test")

        assert llm.last_system_prompt == custom_prompt

    def test_has_timestamp(self):
        """La risposta ha un timestamp."""
        service = RAGService(
            embedder=FakeEmbedder(),
            store=FakeVectorStore(fake_results=_fake_chunks()),
            llm=FakeLLM(),
        )

        response = service.query("Test")

        assert response.timestamp is not None

    def test_default_system_prompt_is_anti_hallucination(self):
        """Il system prompt di default contiene le regole anti-allucinazione."""
        assert "SOLO" in SYSTEM_PROMPT
        assert "Non inventare" in SYSTEM_PROMPT
        assert "dillo chiaramente" in SYSTEM_PROMPT.lower()
