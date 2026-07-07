"""Test per i chunker."""

import pytest

from rag_assistant.adapters.sentence_chunker import SentenceChunker
from rag_assistant.adapters.tabular_chunker import TabularChunker
from rag_assistant.adapters.chunker_registry import get_chunker
from rag_assistant.core.models import Document


def _make_doc(text: str, doc_type: str = "pdf") -> Document:
    return Document(
        source_path=f"/test/file.{doc_type}",
        source_name=f"file.{doc_type}",
        doc_type=doc_type,
        text=text,
    )


class TestSentenceChunker:

    def test_splits_on_sentence_boundary(self):
        text = (
            "Il contratto prevede una penale del 15% sul valore totale. "
            "Il pagamento deve avvenire entro 30 giorni. "
            "Le spese di trasporto sono a carico del compratore. "
            "La merce viene consegnata franco fabbrica."
        )
        doc = _make_doc(text)
        chunker = SentenceChunker(chunk_size=15, overlap_sentences=1)
        chunks = chunker.chunk(doc)
        for chunk in chunks:
            assert chunk.text.rstrip().endswith(".")

    def test_overlap_works(self):
        text = (
            "Prima frase del documento. "
            "Seconda frase importante. "
            "Terza frase con dettagli. "
            "Quarta frase conclusiva. "
            "Quinta frase finale."
        )
        doc = _make_doc(text)
        chunker = SentenceChunker(chunk_size=8, overlap_sentences=1)
        chunks = chunker.chunk(doc)
        if len(chunks) >= 2:
            last_sentence_chunk0 = chunks[0].text.split(".")[-2].strip()
            assert last_sentence_chunk0 in chunks[1].text

    def test_respects_paragraph_breaks(self):
        text = "Primo paragrafo con contenuto.\n\nSecondo paragrafo separato."
        doc = _make_doc(text)
        chunker = SentenceChunker(chunk_size=5, overlap_sentences=0, min_chunk_words=1)
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 1

    def test_empty_document(self):
        doc = _make_doc("")
        chunker = SentenceChunker()
        chunks = chunker.chunk(doc)
        assert len(chunks) == 0

    def test_min_chunk_words_filter(self):
        text = "Corto. Anche questo è corto."
        doc = _make_doc(text)
        chunker = SentenceChunker(chunk_size=100, min_chunk_words=50)
        chunks = chunker.chunk(doc)
        assert len(chunks) == 0

    def test_chunk_has_correct_metadata(self):
        text = "Una frase sufficientemente lunga per creare un chunk valido con abbastanza parole."
        doc = _make_doc(text)
        chunker = SentenceChunker(chunk_size=50, min_chunk_words=5)
        chunks = chunker.chunk(doc)
        if chunks:
            assert chunks[0].metadata["chunker"] == "sentence"

    def test_word_and_char_counts(self):
        text = "Questa è una frase di test con esattamente dieci parole dentro di essa."
        doc = _make_doc(text)
        chunker = SentenceChunker(chunk_size=50, min_chunk_words=5)
        chunks = chunker.chunk(doc)
        if chunks:
            assert chunks[0].word_count > 0
            assert chunks[0].char_count > 0

    def test_italian_accented_sentence_split(self):
        text = "La situazione è complessa. È necessario intervenire subito."
        doc = _make_doc(text)
        chunker = SentenceChunker(chunk_size=5, overlap_sentences=0, min_chunk_words=1)
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 1

    def test_preserves_doc_id_and_source(self):
        doc = _make_doc(
            "Abbastanza testo per generare almeno un chunk valido con un buon numero di parole."
        )
        chunker = SentenceChunker(chunk_size=50, min_chunk_words=5)
        chunks = chunker.chunk(doc)
        if chunks:
            assert chunks[0].doc_id == doc.doc_id
            assert chunks[0].source_name == doc.source_name


class TestTabularChunker:

    def _make_tabular_doc(self, num_rows: int) -> Document:
        lines = [
            f"[Vendite] Data: 2025-01-{i+1:02d} | Prodotto: Uva | Quantità: {100+i}"
            for i in range(num_rows)
        ]
        return _make_doc("\n".join(lines), doc_type="excel")

    def test_groups_rows(self):
        doc = self._make_tabular_doc(10)
        chunker = TabularChunker(rows_per_chunk=5, overlap_rows=0)
        chunks = chunker.chunk(doc)
        assert len(chunks) == 2
        assert chunks[0].metadata["rows_in_chunk"] == 5
        assert chunks[1].metadata["rows_in_chunk"] == 5

    def test_overlap_rows(self):
        doc = self._make_tabular_doc(10)
        chunker = TabularChunker(rows_per_chunk=5, overlap_rows=2)
        chunks = chunker.chunk(doc)
        if len(chunks) >= 2:
            lines_chunk0 = chunks[0].text.strip().split("\n")
            lines_chunk1 = chunks[1].text.strip().split("\n")
            overlap_line = lines_chunk0[-1]
            assert overlap_line in lines_chunk1[0] or overlap_line in lines_chunk1[1]

    def test_empty_document(self):
        doc = _make_doc("", doc_type="excel")
        chunker = TabularChunker()
        chunks = chunker.chunk(doc)
        assert len(chunks) == 0

    def test_fewer_rows_than_chunk_size(self):
        doc = self._make_tabular_doc(3)
        chunker = TabularChunker(rows_per_chunk=20)
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].metadata["rows_in_chunk"] == 3

    def test_has_correct_metadata(self):
        doc = self._make_tabular_doc(5)
        chunker = TabularChunker(rows_per_chunk=5)
        chunks = chunker.chunk(doc)
        assert chunks[0].metadata["chunker"] == "tabular"

    def test_preserves_doc_id(self):
        doc = self._make_tabular_doc(5)
        chunker = TabularChunker(rows_per_chunk=5)
        chunks = chunker.chunk(doc)
        assert chunks[0].doc_id == doc.doc_id


class TestChunkerRegistry:

    def test_pdf_gets_sentence_chunker(self):
        doc = _make_doc("testo", doc_type="pdf")
        assert isinstance(get_chunker(doc), SentenceChunker)

    def test_docx_gets_sentence_chunker(self):
        doc = _make_doc("testo", doc_type="docx")
        assert isinstance(get_chunker(doc), SentenceChunker)

    def test_txt_gets_sentence_chunker(self):
        doc = _make_doc("testo", doc_type="txt")
        assert isinstance(get_chunker(doc), SentenceChunker)

    def test_excel_gets_tabular_chunker(self):
        doc = _make_doc("testo", doc_type="excel")
        assert isinstance(get_chunker(doc), TabularChunker)

    def test_csv_gets_tabular_chunker(self):
        doc = _make_doc("testo", doc_type="csv")
        assert isinstance(get_chunker(doc), TabularChunker)
