"""Test per la configurazione."""

from rag_assistant.core.config import Settings


class TestConfig:
    def test_defaults(self):
        s = Settings()
        assert s.llm_model == "qwen2.5:14b"
        assert s.embed_model == "bge-m3"
        assert s.top_k == 5

    def test_temperature_range(self):
        s = Settings(temperature=0.5)
        assert s.temperature == 0.5

    def test_token_masked(self):
        s = Settings(telegram_token="secret123")
        assert "secret123" not in str(s.telegram_token)
        assert s.telegram_token.get_secret_value() == "secret123"
