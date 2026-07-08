"""
Embedder che usa Ollama per generare vettori localmente.

Include una safety net per testi troppo lunghi: se un chunk
supera il limite del modello, viene troncato automaticamente.
bge-m3 ha un context window di 8192 token.
"""

import logging

import requests

from rag_assistant.adapters.base import Embedder
from rag_assistant.core.config import settings

logger = logging.getLogger(__name__)

# Limite conservativo in caratteri.
# bge-m3: 8192 token. La tokenizzazione varia, ma
# 8000 caratteri è un limite sicuro per testi misti
# (numeri, simboli, testo) come DDT e fatture.
MAX_TEXT_CHARS = 3000


class OllamaEmbedder(Embedder):
    """Genera embedding tramite Ollama locale."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout: int = 30,
    ):
        self.model = model or settings.embed_model
        self.base_url = base_url or settings.ollama_base_url
        self.timeout = timeout

    def embed(self, text: str) -> list[float]:
        """Genera l'embedding di un singolo testo."""
        if len(text) > MAX_TEXT_CHARS:
            logger.warning(
                f"Testo troncato da {len(text)} a {MAX_TEXT_CHARS} caratteri"
            )
            text = text[:MAX_TEXT_CHARS]

        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=self.timeout,
            )
            response.raise_for_status()

        except requests.ConnectionError:
            raise ConnectionError(
                f"Ollama non raggiungibile su {self.base_url}. "
                f"Verifica che 'ollama serve' sia attivo."
            )
        except requests.HTTPError as e:
            raise RuntimeError(
                f"Errore HTTP da Ollama: {e.response.status_code} — "
                f"{e.response.text}"
            )

        data = response.json()

        if "embedding" not in data:
            raise RuntimeError(
                f"Risposta Ollama malformata: campo 'embedding' mancante. "
                f"Modello '{self.model}' potrebbe non supportare gli embedding."
            )

        return data["embedding"]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Genera embedding per una lista di testi.

        Se un singolo testo fallisce, logga l'errore e usa un
        vettore zero come fallback. Un chunk problematico non
        deve bloccare l'indicizzazione di 900 chunk validi.
        """
        embeddings = []
        total = len(texts)
        failures = 0

        for i, text in enumerate(texts):
            try:
                embedding = self.embed(text)
                embeddings.append(embedding)
            except (RuntimeError, ConnectionError) as e:
                logger.error(f"Embedding fallito per chunk {i}: {e}")
                if embeddings:
                    zero_vec = [0.0] * len(embeddings[0])
                else:
                    zero_vec = [0.0] * 1024
                embeddings.append(zero_vec)
                failures += 1

            if (i + 1) % 20 == 0 or (i + 1) == total:
                logger.info(f"Embedding: {i + 1}/{total}")

        if failures:
            logger.warning(f"Embedding completato con {failures} errori su {total}")

        return embeddings
