"""
Registry dei chunker.

Mappa il doc_type del Document al chunker appropriato.
La logica è semplice: dati tabulari (excel, csv) → TabularChunker,
tutto il resto (pdf, docx, txt) → SentenceChunker.

A differenza del loader registry (che mappa per estensione file),
il chunker registry mappa per TIPO DI CONTENUTO. Questo perché
la strategia di chunking dipende dalla natura del contenuto,
non dal formato del file.
"""

from rag_assistant.adapters.base import Chunker
from rag_assistant.adapters.sentence_chunker import SentenceChunker
from rag_assistant.adapters.tabular_chunker import TabularChunker
from rag_assistant.core.config import settings
from rag_assistant.core.models import Document

# Tipi di documento considerati tabulari
_TABULAR_TYPES = {"excel", "csv"}


def get_chunker(document: Document) -> Chunker:
    """Restituisce il chunker appropriato per un documento.

    La scelta si basa sul doc_type:
    - "excel", "csv" → TabularChunker (row-based)
    - "pdf", "docx", "txt" e tutto il resto → SentenceChunker

    I parametri (chunk_size, overlap) vengono dal config centralizzato.

    Args:
        document: il Document da chunkare.

    Returns:
        Un'istanza del chunker appropriato.

    Esempio:
        doc_pdf = Document(..., doc_type="pdf")
        doc_xls = Document(..., doc_type="excel")

        get_chunker(doc_pdf)   # → SentenceChunker
        get_chunker(doc_xls)   # → TabularChunker
    """
    if document.doc_type in _TABULAR_TYPES:
        return TabularChunker(
            rows_per_chunk=20,
            overlap_rows=2,
        )

    return SentenceChunker(
        chunk_size=settings.chunk_size,
        overlap_sentences=2,
        min_chunk_words=settings.chunk_size // 10,
    )
