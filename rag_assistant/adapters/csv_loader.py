"""
Loader per file CSV (.csv).

Stessa logica dell'ExcelLoader ma per file CSV:
- Prima riga = header
- Righe successive convertite in "Header: Valore | Header: Valore"

Gestione separatore: usiamo csv.Sniffer per auto-detectare il
delimitatore (virgola, punto e virgola, tab). I file italiani
usano spesso il punto e virgola perché la virgola è il separatore
decimale. Lo Sniffer risolve il problema automaticamente.
"""

import csv
from pathlib import Path

from rag_assistant.adapters.base import DocumentLoader
from rag_assistant.core.models import Document


class CSVLoader(DocumentLoader):
    """Carica file CSV con auto-detection del separatore."""

    def load(self, file_path: str) -> list[Document]:
        """Carica un file CSV.

        Detecta automaticamente il separatore e l'encoding.
        Converte ogni riga in formato "Header: Valore | ..."

        Returns:
            Lista con un singolo Document.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File non trovato: {file_path}")

        content, encoding = self._read_with_fallback(path)
        dialect = self._detect_dialect(content)
        text = self._parse_csv(content, dialect, path.stem)

        return [Document(
            source_path=str(path.resolve()),
            source_name=path.name,
            doc_type="csv",
            text=text,
            metadata={
                "encoding": encoding,
                "delimiter": dialect.delimiter if dialect else ",",
            },
        )]

    def _read_with_fallback(self, path: Path) -> tuple[str, str]:
        """Legge il file provando diversi encoding."""
        for encoding in ["utf-8-sig", "utf-8", "latin-1"]:
            try:
                return path.read_text(encoding=encoding), encoding
            except (UnicodeDecodeError, ValueError):
                continue
        raise ValueError(f"Impossibile decodificare {path.name}")

    def _detect_dialect(self, content: str):
        """Auto-detecta il separatore usando csv.Sniffer.

        Lo Sniffer analizza le prime righe e indovina il delimitatore.
        Se fallisce (file malformato), ritorna None e useremo la virgola.
        """
        try:
            sample = "\n".join(content.split("\n")[:10])
            return csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except csv.Error:
            return None

    def _parse_csv(self, content: str, dialect, file_stem: str) -> str:
        """Converte il CSV in testo leggibile."""
        reader = csv.reader(
            content.strip().split("\n"),
            dialect=dialect if dialect else "excel",
        )

        rows = list(reader)
        if not rows:
            return ""

        headers = [h.strip() for h in rows[0]]

        lines = []
        for row in rows[1:]:
            parts = []
            for j, value in enumerate(row):
                value = value.strip()
                if not value:
                    continue
                header = headers[j] if j < len(headers) else f"Col_{j}"
                parts.append(f"{header}: {value}")

            if parts:
                lines.append(f"[{file_stem}] {' | '.join(parts)}")

        return "\n".join(lines)

    @staticmethod
    def supported_extensions() -> list[str]:
        return [".csv"]
