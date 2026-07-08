"""
Configurazione centralizzata.

Sovrascrivibile via variabili d'ambiente con prefisso RAG_.
Priorità: env var > .env > default.
"""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="RAG_",
        case_sensitive=False,
    )

    # LLM
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:14b"
    embed_model: str = "bge-m3"
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)

    # RAG
    chunk_size: int = Field(default=300, ge=50, le=2000)
    chunk_overlap: int = Field(default=50, ge=0, le=200)
    top_k: int = Field(default=5, ge=1, le=20)

    # ChromaDB
    chroma_persist_dir: str = "./chroma_db"
    collection_name: str = "documents"

    # Telegram
    telegram_token: SecretStr = SecretStr("")
    telegram_allowed_users: str = ""

    # Paths
    documents_dir: str = "./documents"

    # Logging
    log_level: str = "INFO"


settings = Settings()
