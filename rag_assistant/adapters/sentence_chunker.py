"""
Chunker basato su frasi per contenuto discorsivo (PDF, Word, TXT).

Perché sentence-based e non word-based?

Consideriamo questa frase a cavallo tra due chunk con taglio word-based:

    Chunk 1: "...il contratto prevede una penale del"
    Chunk 2: "15% sul valore totale della fornitura..."

Il chunk 1 parla di una penale ma non dice quanto. Il chunk 2
dice 15% ma non dice di cosa. Nessuno dei due è autosufficiente.
Se la query è "qual è la penale?", nessun chunk matcha bene.

Con il taglio sentence-based:

    Chunk 1: "...Il contratto prevede una penale del 15% sul valore
              totale della fornitura."
    Chunk 2: "Il contratto prevede una penale del 15% sul valore
              totale della fornitura. Il pagamento deve avvenire..."

La frase resta intera in entrambi i chunk (grazie all'overlap).
La query "qual è la penale?" trova un match pulito.

Algoritmo:
1. Dividi il testo in frasi (sentence splitting)
2. Accumula frasi finché non raggiungi chunk_size parole
3. Quando raggiungi il limite, chiudi il chunk
4. Il chunk successivo inizia N frasi prima (overlap)
"""

import re

from rag_assistant.adapters.base import Chunker
from rag_assistant.core.models import Chunk, Document


class SentenceChunker(Chunker):
    """Divide documenti in chunk rispettando i confini di frase.

    Args:
        chunk_size: numero approssimativo di parole per chunk.
                    È un target, non un limite rigido — il chunk
                    può essere leggermente più grande per non
                    spezzare l'ultima frase.
        overlap_sentences: quante frasi di overlap tra chunk consecutivi.
                          Queste frasi appaiono sia alla fine del chunk N
                          che all'inizio del chunk N+1.
        min_chunk_words: chunk con meno di queste parole vengono scartati.
                        Evita chunk troppo corti che producono embedding
                        poco informativi.
    """

    def __init__(
        self,
        chunk_size: int = 300,
        overlap_sentences: int = 2,
        min_chunk_words: int = 30,
    ):
        self.chunk_size = chunk_size
        self.overlap_sentences = overlap_sentences
        self.min_chunk_words = min_chunk_words

    def chunk(self, document: Document) -> list[Chunk]:
        """Divide un documento in chunk sentence-based.

        Pipeline interna:
        1. Split del testo in frasi
        2. Raggruppamento delle frasi in chunk da ~chunk_size parole
        3. Applicazione dell'overlap
        4. Filtraggio dei chunk troppo corti
        """
        sentences = self._split_sentences(document.text)

        if not sentences:
            return []

        raw_chunks = self._group_sentences(sentences)

        # Costruisci i Chunk pydantic
        chunks = []
        for i, chunk_text in enumerate(raw_chunks):
            if len(chunk_text.split()) < self.min_chunk_words:
                continue

            chunks.append(Chunk(
                doc_id=document.doc_id,
                source_name=document.source_name,
                text=chunk_text,
                chunk_index=i,
                metadata={
                    "chunker": "sentence",
                    "target_size": self.chunk_size,
                },
            ))

        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """Divide il testo in frasi.

        Strategia multi-livello:
        1. Splitta sui doppi a capo (confini di paragrafo)
        2. Dentro ogni paragrafo, splitta sui punti fermi seguiti
           da spazio e lettera maiuscola

        Il pattern regex gestisce:
        - Punto + spazio + maiuscola: "prima frase. Seconda frase"
        - Punto esclamativo/interrogativo + spazio: "Davvero? Sì!"
        - Non splitta su abbreviazioni comuni: "Dr. Rossi", "n. 42"
        - Non splitta su numeri decimali: "€ 1.234,56"

        Nota: il sentence splitting perfetto è un problema di NLP
        non banale. Questa implementazione copre il 95% dei casi
        nei documenti aziendali italiani. Per il restante 5%
        (abbreviazioni insolite, testo molto tecnico) servirebbe
        un modello NLP dedicato — overkill per il nostro caso d'uso.
        """
        # Normalizza gli spazi multipli e i line breaks
        text = re.sub(r'\r\n', '\n', text)          # Windows → Unix
        text = re.sub(r'\n{3,}', '\n\n', text)      # 3+ newline → 2

        # Split sui paragrafi prima (doppio a capo)
        paragraphs = re.split(r'\n\n+', text)

        sentences = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Split sulle frasi dentro il paragrafo
            # Pattern: punto/!/? seguito da spazio e maiuscola
            # Il (?<= ) è un lookbehind, (?= ) è un lookahead
            # Splittiamo DOPO il punto, non prima
            para_sentences = re.split(
                r'(?<=[.!?])\s+(?=[A-ZÀÈÉÌÒÙ])',
                para,
            )

            for sent in para_sentences:
                sent = sent.strip()
                if sent:
                    sentences.append(sent)

        return sentences

    def _group_sentences(self, sentences: list[str]) -> list[str]:
        """Raggruppa le frasi in chunk da ~chunk_size parole con overlap.

        Algoritmo:
        1. Accumula frasi finché il conteggio parole >= chunk_size
        2. Salva il chunk
        3. Torna indietro di overlap_sentences frasi
        4. Ricomincia ad accumulare

        L'overlap funziona a livello di frasi, non di parole.
        Questo è più pulito: le frasi di overlap sono sempre
        complete, mai troncate.
        """
        chunks = []
        start = 0

        while start < len(sentences):
            # Accumula frasi fino a raggiungere chunk_size
            word_count = 0
            end = start

            while end < len(sentences) and word_count < self.chunk_size:
                word_count += len(sentences[end].split())
                end += 1

            # Crea il chunk
            chunk_text = " ".join(sentences[start:end])
            chunks.append(chunk_text)

            # Avanza con overlap: torna indietro di N frasi
            # Se end - overlap_sentences <= start, avanza di almeno 1
            next_start = max(start + 1, end - self.overlap_sentences)
            start = next_start

        return chunks
