"""
Registry dei document loader.

Mappa ogni estensione file al loader corretto.
L'IngestionService usa il registry per scegliere automaticamente
il loader giusto senza dover conoscere le implementazioni.

Questo è il pattern Factory: il registry crea il loader corretto
basandosi sull'estensione, il codice chiamante non sa e non
gli importa quale classe concreta sta usando.
"""

from pathlib import Path

from rag_assistant.adapters.base import DocumentLoader
from rag_assistant.adapters.pdf_loader import PDFLoader
from rag_assistant.adapters.excel_loader import ExcelLoader
from rag_assistant.adapters.word_loader import WordLoader
from rag_assistant.adapters.csv_loader import CSVLoader
from rag_assistant.adapters.txt_loader import TXTLoader


# Tutti i loader disponibili
_LOADERS: list[type[DocumentLoader]] = [
    PDFLoader,
    ExcelLoader,
    WordLoader,
    CSVLoader,
    TXTLoader,
]

# Mappa estensione → classe loader, costruita automaticamente
_EXTENSION_MAP: dict[str, type[DocumentLoader]] = {}
for loader_class in _LOADERS:
    for ext in loader_class.supported_extensions():
        _EXTENSION_MAP[ext.lower()] = loader_class


def get_loader(file_path: str) -> DocumentLoader:
    """Restituisce il loader corretto per un file.

    Args:
        file_path: percorso al file.

    Returns:
        Un'istanza del loader appropriato.

    Raises:
        ValueError: se l'estensione non è supportata.

    Esempio:
        loader = get_loader("fattura.pdf")    # → PDFLoader
        loader = get_loader("listino.xlsx")   # → ExcelLoader
        loader = get_loader("contratto.docx") # → WordLoader
    """
    ext = Path(file_path).suffix.lower()

    if ext not in _EXTENSION_MAP:
        supported = ", ".join(sorted(_EXTENSION_MAP.keys()))
        raise ValueError(
            f"Formato '{ext}' non supportato. "
            f"Formati supportati: {supported}"
        )

    return _EXTENSION_MAP[ext]()


def supported_extensions() -> list[str]:
    """Restituisce tutte le estensioni supportate."""
    return sorted(_EXTENSION_MAP.keys())
