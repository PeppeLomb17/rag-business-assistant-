"""Test per l'Ollama embedder.

Usa unittest.mock per simulare le risposte HTTP di Ollama.
Non serve Ollama in esecuzione per questi test.

Il mocking funziona così: invece di chiamare davvero requests.post(),
gli diciamo "quando qualcuno chiama requests.post(), restituisci
questo oggetto finto". Il test verifica che l'embedder gestisca
correttamente la risposta, senza fare richieste di rete reali.
"""

from unittest.mock import patch, MagicMock

import pytest
import requests

from rag_assistant.adapters.ollama_embedder import OllamaEmbedder


class TestOllamaEmbedder:

    def _mock_response(self, json_data: dict, status_code: int = 200):
        """Crea un oggetto Response finto."""
        mock = MagicMock()
        mock.json.return_value = json_data
        mock.status_code = status_code
        mock.raise_for_status.return_value = None
        return mock

    @patch("rag_assistant.adapters.ollama_embedder.requests.post")
    def test_embed_returns_vector(self, mock_post):
        """embed() restituisce un vettore di float."""
        fake_embedding = [0.1, 0.2, 0.3] * 341 + [0.1]  # 1024 dimensioni
        mock_post.return_value = self._mock_response({"embedding": fake_embedding})

        embedder = OllamaEmbedder(model="bge-m3", base_url="http://fake:11434")
        result = embedder.embed("testo di test")

        assert isinstance(result, list)
        assert len(result) == 1024
        assert all(isinstance(x, float) for x in result)

    @patch("rag_assistant.adapters.ollama_embedder.requests.post")
    def test_embed_calls_correct_endpoint(self, mock_post):
        """embed() chiama l'endpoint giusto con i parametri giusti."""
        mock_post.return_value = self._mock_response({"embedding": [0.1, 0.2]})

        embedder = OllamaEmbedder(model="bge-m3", base_url="http://localhost:11434")
        embedder.embed("ciao mondo")

        # Verifica che requests.post sia stato chiamato correttamente
        mock_post.assert_called_once_with(
            "http://localhost:11434/api/embeddings",
            json={"model": "bge-m3", "prompt": "ciao mondo"},
            timeout=30,
        )

    @patch("rag_assistant.adapters.ollama_embedder.requests.post")
    def test_embed_batch_returns_list_of_vectors(self, mock_post):
        """embed_batch() restituisce un vettore per ogni testo."""
        mock_post.return_value = self._mock_response({"embedding": [0.1, 0.2, 0.3]})

        embedder = OllamaEmbedder(model="bge-m3", base_url="http://fake:11434")
        results = embedder.embed_batch(["testo uno", "testo due", "testo tre"])

        assert len(results) == 3
        assert mock_post.call_count == 3  # Una chiamata per testo

    @patch("rag_assistant.adapters.ollama_embedder.requests.post")
    def test_connection_error(self, mock_post):
        """Se Ollama non è raggiungibile, solleva ConnectionError."""
        mock_post.side_effect = requests.ConnectionError("refused")

        embedder = OllamaEmbedder(model="bge-m3", base_url="http://fake:11434")

        with pytest.raises(ConnectionError, match="Ollama non raggiungibile"):
            embedder.embed("test")

    @patch("rag_assistant.adapters.ollama_embedder.requests.post")
    def test_malformed_response(self, mock_post):
        """Se la risposta non ha il campo 'embedding', solleva RuntimeError."""
        mock_post.return_value = self._mock_response({"error": "model not found"})

        embedder = OllamaEmbedder(model="modello-inesistente", base_url="http://fake:11434")

        with pytest.raises(RuntimeError, match="campo 'embedding' mancante"):
            embedder.embed("test")

    @patch("rag_assistant.adapters.ollama_embedder.requests.post")
    def test_http_error(self, mock_post):
        """Se Ollama restituisce un errore HTTP, solleva RuntimeError."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = requests.HTTPError(
            response=mock_response
        )
        mock_post.return_value = mock_response

        embedder = OllamaEmbedder(model="bge-m3", base_url="http://fake:11434")

        with pytest.raises(RuntimeError, match="Errore HTTP"):
            embedder.embed("test")

    def test_default_config(self):
        """Senza parametri, usa i valori dal config."""
        embedder = OllamaEmbedder()
        assert embedder.model == "bge-m3"
        assert "11434" in embedder.base_url

    def test_custom_params(self):
        """I parametri custom sovrascrivono il config."""
        embedder = OllamaEmbedder(
            model="custom-model",
            base_url="http://custom:9999",
            timeout=60,
        )
        assert embedder.model == "custom-model"
        assert embedder.base_url == "http://custom:9999"
        assert embedder.timeout == 60
