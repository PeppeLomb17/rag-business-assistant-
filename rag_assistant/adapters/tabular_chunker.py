"""
Chunker per contenuto tabulare (Excel, CSV).

I dati tabulari hanno una struttura diversa dalla prosa:
- Ogni riga è un'unità di informazione autonoma
- Le righe vicine non hanno necessariamente una relazione narrativa
- L'header (nome delle colonne) è il contesto cruciale

Strategia: raggruppa N righe per chunk. L'overlap non è a frasi
ma a righe.
"""

from rag_assistant.adapters.base import Chunker
from rag_assistant.core.models import Chunk, Document


class TabularChunker(Chunker):
    """Divide dati tabulari in chunk per gruppi di righe.

    Args:
        rows_per_chunk: quante righe per chunk.
        overlap_rows: quante righe di overlap tra chunk consecutivi.
    """

    def __init__(
        self,
        rows_per_chunk: int = 20,
        overlap_rows: int = 2,
    ):
        self.rows_per_chunk = rows_per_chunk
        self.overlap_rows = overlap_rows

    def chunk(self, document: Document) -> list[Chunk]:
        """Divide un documento tabulare in chunk per gruppi di righe."""
        lines = [
            line.strip()
            for line in document.text.split("\n")
            if line.strip()
        ]

        if not lines:
            return []

        raw_chunks = self._group_rows(lines)

        chunks = []
        for i, chunk_lines in enumerate(raw_chunks):
            chunk_text = "\n".join(chunk_lines)

            chunks.append(Chunk(
                doc_id=document.doc_id,
                source_name=document.source_name,
                text=chunk_text,
                chunk_index=i,
                metadata={
                    "chunker": "tabular",
                    "rows_in_chunk": len(chunk_lines),
                },
            ))

        return chunks

    def _group_rows(self, lines: list[str]) -> list[list[str]]:
        """Raggruppa le righe in blocchi con overlap."""
        chunks = []
        start = 0

        while start < len(lines):
            end = min(start + self.rows_per_chunk, len(lines))
            chunks.append(lines[start:end])

            # Se abbiamo raggiunto la fine, basta
            if end >= len(lines):
                break

            # Avanza con overlap
            next_start = end - self.overlap_rows
            if next_start <= start:
                next_start = start + 1

            start = next_start

        return chunks
