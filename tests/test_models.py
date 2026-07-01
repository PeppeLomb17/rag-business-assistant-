"""Test per i domain models."""

from rag_assistant.core.models import (
    Document, Chunk, RetrievedChunk, RAGResponse, _generate_id,
)


class TestGenerateId:
    def test_deterministic(self):
        assert _generate_id("/path/a.pdf") == _generate_id("/path/a.pdf")

    def test_unique(self):
        assert _generate_id("/path/a.pdf") != _generate_id("/path/b.pdf")

    def test_length(self):
        assert len(_generate_id("/any/path")) == 12


class TestDocument:
    def test_auto_id(self):
        doc = Document(
            source_path="/test/file.pdf",
            source_name="file.pdf",
            doc_type="pdf",
            text="contenuto",
        )
        assert len(doc.doc_id) == 12

    def test_explicit_id_preserved(self):
        doc = Document(
            doc_id="my_custom_id",
            source_path="/test/file.pdf",
            source_name="file.pdf",
            doc_type="pdf",
            text="contenuto",
        )
        assert doc.doc_id == "my_custom_id"

    def test_metadata_isolation(self):
        d1 = Document(source_path="/a", source_name="a", doc_type="pdf", text="a")
        d2 = Document(source_path="/b", source_name="b", doc_type="pdf", text="b")
        d1.metadata["x"] = 1
        assert "x" not in d2.metadata


class TestChunk:
    def test_auto_id_format(self):
        c = Chunk(doc_id="abc123def456", source_name="f.pdf", text="testo", chunk_index=3)
        assert c.chunk_id == "abc123def456_chunk_0003"

    def test_word_count(self):
        c = Chunk(doc_id="abc", source_name="f.pdf", text="una due tre", chunk_index=0)
        assert c.word_count == 3

    def test_char_count(self):
        c = Chunk(doc_id="abc", source_name="f.pdf", text="ciao", chunk_index=0)
        assert c.char_count == 4


class TestRetrievedChunk:
    def test_default_method(self):
        rc = RetrievedChunk(chunk_id="x", text="t", source_name="f", score=0.9)
        assert rc.retrieval_method == "semantic"


class TestRAGResponse:
    def test_success_default(self):
        r = RAGResponse(query="q", answer="a")
        assert r.success is True
        assert r.error is None

    def test_error_state(self):
        r = RAGResponse(query="q", answer="", success=False, error="timeout")
        assert not r.success
        assert r.error == "timeout"

    def test_has_timestamp(self):
        r = RAGResponse(query="q", answer="a")
        assert r.timestamp is not None
