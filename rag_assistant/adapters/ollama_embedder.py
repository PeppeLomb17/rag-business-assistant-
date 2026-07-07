"""
Embedder che usa Ollama per generare vettori localmente.

Ollama espone un'API REST locale su http://localhost:11434.
L'endpoint /api/embeddings accetta un testo e restituisce un
vettore numerico. Tutto gira sulla macchina — nessun dato
esce verso server esterni.

Il modello bge-m3 è multilingua: mappa testi in italiano,
inglese e altre lingue nello stesso spazio vettoriale.
Questo significa che una query in italiano può trovare chunk
in inglese e viceversa — esattamente il problema che avevamo
nel progetto base con nomic-embed-text.

Nota tecnica: l'API di Ollama non supporta il batching nativo
per gli embedding (a differenza di OpenAI). Il metodo embed_batch()
chiama embed() in sequenza. Se in futuro Ollama aggiungesse il
batching, basterebbe modificare questo metodo senza toccare
nient'altro nel sistema.
"""

import logging

import requests

from rag_assistant.adapters.base import Embedder
from rag_assistant.core.config import settings

logger = logging.getLogger(__name__)


class OllamaEmbedder(Embedder):
    """Genera embedding tramite Ollama locale.

    Args:
        model: nome del modello embedding. Default dal config.
        base_url: URL del server Ollama. Default dal config.
        timeout: timeout in secondi per ogni richiesta.
                 Gli embedding sono veloci (~100ms per chunk),
                 ma su testi lunghi o macchine cariche può servire
                 più tempo.
    """

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
        """Genera l'embedding di un singolo testo.

        Chiama l'API /api/embeddings di Ollama e restituisce
        il vettore risultante.

        Args:
            text: testo da vettorizzare.

        Returns:
            Lista di float (es: 1024 dimensioni per bge-m3).

        Raises:
            ConnectionError: se Ollama non è raggiungibile.
            RuntimeError: se la risposta è malformata.
        """
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

        Ollama non supporta batching nativo per gli embedding,
        quindi processiamo in sequenza. Il logging mostra il
        progresso per batch lunghi.

        Args:
            texts: lista di testi da vettorizzare.

        Returns:
            Lista di vettori, uno per ogni testo.
        """
        embeddings = []
        total = len(texts)

        for i, text in enumerate(texts):
            embedding = self.embed(text)
            embeddings.append(embedding)

            # Log progresso ogni 20 chunk
            if (i + 1) % 20 == 0 or (i + 1) == total:
                logger.info(f"Embedding: {i + 1}/{total}")

        return embeddings
