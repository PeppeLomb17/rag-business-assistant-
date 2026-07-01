"""
Domain models per la pipeline RAG.

Contratti tra i layer dell'architettura. Ogni modello usa Pydantic
per validazione automatica e calcolo di campi derivati.
"""

from datetime import datetime
from hashlib import sha256

from pydantic import BaseModel, Field


def _generate_id(value: str) -> str:
    """ID deterministico a 12 caratteri hex.
    
    SHA-256 troncato a 12 chars = 48 bit di entropia.
    Sufficiente per un corpus documentale aziendale.
    """
    return sha256(value.encode()).hexdigest()[:12]


class Document(BaseModel):
    """Documento caricato e pronto per il chunking.
    
    Un Document è un'unità logica di contenuto:
    - un PDF intero
    - un singolo foglio Excel
    - un file Word
    - un file di testo
    
    Per file Excel multi-foglio, ogni foglio è un Document separato.
    """
    doc_id: str = ""
    source_path: str
    source_name: str
    doc_type: str                                       # "pdf", "excel", "csv", "txt", "docx"
    text: str
    metadata: dict = Field(default_factory=dict)
    loaded_at: datetime = Field(default_factory=datetime.now)

    def model_post_init(self, __context) -> None:
        if not self.doc_id:
            self.doc_id = _generate_id(self.source_path)


class Chunk(BaseModel):
    """Singolo chunk pronto per l'embedding.
    
    chunk_id è composto da doc_id + indice progressivo zero-padded,
    garantendo unicità e tracciabilità verso il documento padre.
    """
    chunk_id: str = ""
    doc_id: str
    source_name: str
    text: str
    chunk_index: int
    char_count: int = 0
    word_count: int = 0
    metadata: dict = Field(default_factory=dict)

    def model_post_init(self, __context) -> None:
        if not self.chunk_id:
            self.chunk_id = f"{self.doc_id}_chunk_{self.chunk_index:04d}"
        if not self.char_count:
            self.char_count = len(self.text)
        if not self.word_count:
            self.word_count = len(self.text.split())


class RetrievedChunk(BaseModel):
    """Chunk restituito dal retrieval con score di rilevanza."""
    chunk_id: str
    text: str
    source_name: str
    score: float
    retrieval_method: str = "semantic"
    metadata: dict = Field(default_factory=dict)


class RAGResponse(BaseModel):
    """Risposta completa della pipeline RAG."""
    query: str
    answer: str
    success: bool = True
    error: str | None = None
    chunks_used: list[RetrievedChunk] = Field(default_factory=list)
    model: str = ""
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)
