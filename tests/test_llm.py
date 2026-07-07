"""Test per l'Ollama LLM provider.

Come per l'embedder, mockiamo le chiamate HTTP per non dipendere
da Ollama in esecuzione. La struttura della risposta di /api/chat
è diversa da /api/embeddings:

/api/embeddings → {"embedding": [0.1, 0.2, ...]}
/api/chat       → {"message": {"role": "assistant", "content": "risposta"}}
"""

from unittest.mock import patch, MagicMock

import pytest
import requests

from rag_assistant.adapters.ollama_llm import OllamaLLM


class TestOllamaLLM:

    def _mock_chat_response(self, content: str, status_code: int = 200):
        """Crea una risposta finta dell'API /api/chat."""
        mock = MagicMock()
        mock.json.return_value = {
            "message": {
                "role": "assistant",
                "content": content,
            },
        }
        mock.status_code = status_code
        mock.raise_for_status.return_value = None
        return mock

    @patch("rag_assistant.adapters.ollama_llm.requests.post")
    def test_generate_returns_text(self, mock_post):
        """generate() restituisce il testo della risposta."""
        mock_post.return_value = self._mock_chat_response(
            "Sono stati identificati 3 cluster."
        )

        llm = OllamaLLM(model="qwen2.5:14b", base_url="http://fake:11434")
        answer = llm.generate(
            system_prompt="Rispondi dal contesto.",
            user_prompt="Quanti cluster ci sono?",
        )

        assert answer == "Sono stati identificati 3 cluster."

    @patch("rag_assistant.adapters.ollama_llm.requests.post")
    def test_generate_strips_whitespace(self, mock_post):
        """La risposta viene strippata da spazi iniziali/finali."""
        mock_post.return_value = self._mock_chat_response(
            "\n  Risposta con spazi  \n\n"
        )

        llm = OllamaLLM(model="test", base_url="http://fake:11434")
        answer = llm.generate("system", "user")

        assert answer == "Risposta con spazi"

    @patch("rag_assistant.adapters.ollama_llm.requests.post")
    def test_generate_sends_correct_payload(self, mock_post):
        """generate() manda il payload corretto a Ollama."""
        mock_post.return_value = self._mock_chat_response("ok")

        llm = OllamaLLM(
            model="qwen2.5:14b",
            base_url="http://localhost:11434",
            temperature=0.3,
        )
        llm.generate(
            system_prompt="Sei un assistente.",
            user_prompt="Ciao",
        )

        # Verifica la struttura della chiamata
        call_args = mock_post.call_args
        url = call_args[0][0]
        payload = call_args[1]["json"]

        assert url == "http://localhost:11434/api/chat"
        assert payload["model"] == "qwen2.5:14b"
        assert payload["stream"] is False
        assert len(payload["messages"]) == 2
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][0]["content"] == "Sei un assistente."
        assert payload["messages"][1]["role"] == "user"
        assert payload["messages"][1]["content"] == "Ciao"
        assert payload["options"]["temperature"] == 0.3

    @patch("rag_assistant.adapters.ollama_llm.requests.post")
    def test_connection_error(self, mock_post):
        """Se Ollama non è raggiungibile, solleva ConnectionError."""
        mock_post.side_effect = requests.ConnectionError("refused")

        llm = OllamaLLM(model="test", base_url="http://fake:11434")

        with pytest.raises(ConnectionError, match="Ollama non raggiungibile"):
            llm.generate("system", "user")

    @patch("rag_assistant.adapters.ollama_llm.requests.post")
    def test_timeout_error(self, mock_post):
        """Se la generazione va in timeout, solleva TimeoutError."""
        mock_post.side_effect = requests.Timeout("timeout")

        llm = OllamaLLM(model="test", base_url="http://fake:11434", timeout=10)

        with pytest.raises(TimeoutError, match="Timeout"):
            llm.generate("system", "user")

    @patch("rag_assistant.adapters.ollama_llm.requests.post")
    def test_http_error(self, mock_post):
        """Se Ollama restituisce un errore HTTP, solleva RuntimeError."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "model not found"
        mock_response.raise_for_status.side_effect = requests.HTTPError(
            response=mock_response
        )
        mock_post.return_value = mock_response

        llm = OllamaLLM(model="modello-inesistente", base_url="http://fake:11434")

        with pytest.raises(RuntimeError, match="Errore HTTP"):
            llm.generate("system", "user")

    @patch("rag_assistant.adapters.ollama_llm.requests.post")
    def test_malformed_response(self, mock_post):
        """Se la risposta non ha message.content, solleva RuntimeError."""
        mock = MagicMock()
        mock.json.return_value = {"error": "something went wrong"}
        mock.raise_for_status.return_value = None
        mock_post.return_value = mock

        llm = OllamaLLM(model="test", base_url="http://fake:11434")

        with pytest.raises(RuntimeError, match="malformata"):
            llm.generate("system", "user")

    def test_default_config(self):
        """Senza parametri, usa i valori dal config."""
        llm = OllamaLLM()
        assert llm.model == "qwen2.5:14b"
        assert llm.temperature == 0.2
        assert "11434" in llm.base_url

    def test_custom_params(self):
        """I parametri custom sovrascrivono il config."""
        llm = OllamaLLM(
            model="custom-model",
            base_url="http://custom:9999",
            temperature=0.8,
            timeout=60,
        )
        assert llm.model == "custom-model"
        assert llm.base_url == "http://custom:9999"
        assert llm.temperature == 0.8
        assert llm.timeout == 60

    def test_zero_temperature_preserved(self):
        """Temperatura 0.0 non viene sovrascritta dal config.

        Questo test verifica il bug 'falsy value':
        0.0 è un valore valido ma è falsy in Python.
        Se usassimo 'temperature or settings.temperature',
        0.0 verrebbe ignorato.
        """
        llm = OllamaLLM(temperature=0.0)
        assert llm.temperature == 0.0
