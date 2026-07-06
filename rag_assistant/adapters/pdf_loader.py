"""
Loader per file PDF (.pdf).

Un PDF non contiene "testo" nel senso tradizionale. Contiene istruzioni
di rendering: "disegna la lettera 'A' in Helvetica 12pt alle coordinate
(72, 700)". Estrarre testo significa:
1. Leggere queste istruzioni pagina per pagina
2. Ricostruire l'ordine di lettura dalle coordinate
3. Raggruppare le lettere in parole e le parole in righe

pymupdf (importato come 'fitz', dal nome del motore di rendering
MuPDF) è una delle librerie più affidabili per questo. Gestisce
anche PDF scansionati (immagini), ma per quelli servirebbe OCR
che non implementiamo in questa versione.

Nota: PDF con solo immagini (scansioni) produrranno testo vuoto.
Il loader lo segnala nei metadati con 'pages_with_text'.
"""

import fitz  # pymupdf
from pathlib import Path

from rag_assistant.adapters.base import DocumentLoader
from rag_assistant.core.models import Document


class PDFLoader(DocumentLoader):
    """Carica file PDF ed estrae il testo pagina per pagina."""

    def load(self, file_path: str) -> list[Document]:
        """Estrae il testo da un PDF.

        Concatena il testo di tutte le pagine separandole con doppio
        a capo. Salva nei metadati il numero di pagine totali e
        quante contenevano effettivamente testo (utile per identificare
        PDF scansionati che richiederebbero OCR).

        Returns:
            Lista con un singolo Document.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File non trovato: {file_path}")

        doc = fitz.open(str(path))

        pages_text = []
        pages_with_text = 0

        for page in doc:
            text = page.get_text()
            pages_text.append(text)
            if text.strip():
                pages_with_text += 1

        doc.close()

        full_text = "\n\n".join(pages_text)

        return [Document(
            source_path=str(path.resolve()),
            source_name=path.name,
            doc_type="pdf",
            text=full_text,
            metadata={
                "total_pages": len(pages_text),
                "pages_with_text": pages_with_text,
            },
        )]

    @staticmethod
    def supported_extensions() -> list[str]:
        return [".pdf"]
