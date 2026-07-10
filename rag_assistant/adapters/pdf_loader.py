"""
Loader per file PDF (.pdf).

Include post-processing per DDT: riordina i campi in formato
strutturato leggibile. NON include il testo originale duplicato.
"""

import re
import fitz
from pathlib import Path

from rag_assistant.adapters.base import DocumentLoader
from rag_assistant.core.models import Document


class PDFLoader(DocumentLoader):

    def load(self, file_path: str) -> list[Document]:
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

        if self._looks_like_ddt(full_text, path.name):
            full_text = self._restructure_ddt(full_text, path.name)

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

    def _looks_like_ddt(self, text: str, filename: str) -> bool:
        text_lower = text.lower()
        filename_lower = filename.lower()

        if "ddt" in filename_lower:
            return True
        if "documento di trasporto" in text_lower:
            return True
        if "d.d.t." in text_lower:
            return True

        return False

    def _restructure_ddt(self, text: str, filename: str) -> str:
        """Riordina il testo di un DDT in formato strutturato."""
        fields = {}

        # Numero DDT e data
        match = re.search(r'N\.?\s*(\d+\w*)\s+DEL\s+([\d/]+)', text)
        if match:
            fields["Numero DDT"] = match.group(1)
            fields["Data"] = match.group(2)
        else:
            name_match = re.search(r'DDT\s+(\w+)', filename, re.IGNORECASE)
            if name_match:
                fields["Numero DDT"] = name_match.group(1)

        # Mittente
        if "F.LLI LOMBARDIA" in text or "LOMBARDIA" in text:
            fields["Mittente"] = "SOC.COOP.AGR. F.LLI LOMBARDIA — VIA XX SETTEMBRE 133, 97011 ACATE (RG)"

        # 1° Cessionario (tipicamente l'OP)
        match = re.search(r'P\.?IVA\s*0168869088\d', text)
        if match:
            fields["1° Cessionario"] = "AIRONE OP SOC. COOP. AGR. — VIA E. CRISCIONE LUPIS N.23, 97100 RAGUSA (RG)"

        # Luogo di scarico / Destinatario
        match = re.search(r'LUOGO\s*DI\s*SCARICO\s*(?:MERCE)?\s*(.*?)(?=MITTENTE|P\.?IVA|DOCUMENTO|CAUSALE|$)', text, re.DOTALL)
        if match:
            fields["Luogo di scarico"] = self._clean_field(match.group(1))

        # Descrizione merce
        descriptions = re.findall(r'(?:Cartone|Cassetta|Cassa|Plateau|Imballaggio)\s+\w+.*?(?:kg|KG)\s*(?:netto)?', text)
        if descriptions:
            fields["Merce"] = " | ".join(d.strip() for d in descriptions)

        # Peso netto e lordo
        netto_matches = re.findall(r'(\d{2,})', text[text.find("PESO NETTO"):text.find("PESO NETTO")+100]) if "PESO NETTO" in text else []
        if netto_matches:
            fields["Peso netto (kg)"] = netto_matches[0]

        # Colli
        colli_matches = re.findall(r'(\d+)\s*(?:CHEP|EPAL|EUR)', text)
        if colli_matches:
            fields["Pedane"] = " + ".join(colli_matches)

        # Vettore
        if "Sicilsole" in text or "SICILSOLE" in text:
            fields["Vettore"] = "Sicilsole Trasporti S.r.l. — P.IVA 01196270886"

        # Causale
        if "C/VENDITA" in text:
            fields["Causale"] = "C/VENDITA"
        elif "C/LAVORAZIONE" in text:
            fields["Causale"] = "C/LAVORAZIONE"

        # Data e ora ritiro
        match = re.search(r'(\d{2}/\d{2}/\d{2})\s+(\d{2}:\d{2})', text)
        if match:
            fields["Data e ora ritiro"] = f"{match.group(1)} {match.group(2)}"

        # Origine
        match = re.search(r'ORIGINE\s+(.*?)(?=\n|$)', text)
        if match:
            fields["Origine"] = match.group(1).strip()

        # Certificazioni
        if "GLOBALG.A.P" in text:
            match = re.search(r'GLOBALG\.A\.P\.?\s*(?:n\.?)?\s*([\d]+)', text)
            cert = "GLOBALG.A.P."
            if match:
                cert += f" n.{match.group(1)}"
            fields["Certificazione"] = cert

        # Costruisci output strutturato
        if fields:
            num = fields.get("Numero DDT", "N/D")
            data = fields.get("Data", "N/D")
            lines = [f"DDT {num} del {data}"]

            for key, value in fields.items():
                if key not in ("Numero DDT", "Data"):
                    lines.append(f"{key}: {value}")

            return "\n".join(lines)

        return text

    def _clean_field(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text).strip()
        if len(text) > 300:
            text = text[:300] + "..."
        return text

    @staticmethod
    def supported_extensions() -> list[str]:
        return [".pdf"]
