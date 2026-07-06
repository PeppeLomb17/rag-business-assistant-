"""
Loader per file di testo (.txt).

Il caso più semplice: il file È già testo. L'unica complessità
è gestire l'encoding — un file creato su Windows potrebbe usare
'cp1252' invece di 'utf-8', e un file con BOM (Byte Order Mark)
potrebbe avere 3 byte invisibili all'inizio che inquinano il testo.

La strategia: proviamo UTF-8 prima (lo standard), poi Latin-1 come
fallback (copre quasi tutti i file europei/italiani).
"""

from pathlib import Path

from rag_assistant.adapters.base import DocumentLoader
from rag_assistant.core.models import Document


class TXTLoader(DocumentLoader):
    """Carica file .txt con gestione automatica dell'encoding."""

    # Encoding da provare in ordine di priorità.
    # UTF-8 è lo standard moderno. Latin-1 (ISO 8859-1) è il fallback
    # perché non può MAI fallire: mappa ogni byte 0-255 a un carattere.
    # Il rischio è che caratteri multi-byte UTF-8 vengano letti come
    # sequenze Latin-1 sbagliate, ma almeno non crasha.
    ENCODINGS = ["utf-8-sig", "utf-8", "latin-1"]

    def load(self, file_path: str) -> list[Document]:
        """Carica un file di testo.

        Prova diversi encoding in sequenza. 'utf-8-sig' è UTF-8 con
        rimozione automatica del BOM (Byte Order Mark), quei 3 byte
        \xef\xbb\xbf che Windows aggiunge a volte.

        Returns:
            Lista con un singolo Document.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File non trovato: {file_path}")

        text = self._read_with_fallback(path)

        return [Document(
            source_path=str(path.resolve()),
            source_name=path.name,
            doc_type="txt",
            text=text,
            metadata={"encoding": self._detected_encoding},
        )]

    def _read_with_fallback(self, path: Path) -> str:
        """Prova ogni encoding finché uno funziona."""
        for encoding in self.ENCODINGS:
            try:
                text = path.read_text(encoding=encoding)
                self._detected_encoding = encoding
                return text
            except (UnicodeDecodeError, ValueError):
                continue

        raise ValueError(f"Impossibile decodificare {path.name} con nessun encoding noto")

    @staticmethod
    def supported_extensions() -> list[str]:
        return [".txt"]
