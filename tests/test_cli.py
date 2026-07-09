"""Test per la CLI."""

import pytest
from unittest.mock import patch, MagicMock

from rag_assistant.interfaces.cli import (
    _print_header,
    _print_help,
    _print_report,
)


class TestCLIFormatting:

    def test_print_header(self, capsys):
        _print_header()
        captured = capsys.readouterr()
        assert "RAG Business Assistant" in captured.out

    def test_print_help(self, capsys):
        _print_help()
        captured = capsys.readouterr()
        assert "/reindex" in captured.out
        assert "/update" in captured.out
        assert "/status" in captured.out

    def test_print_report_success(self, capsys):
        report = {
            "files_found": 5,
            "files_processed": 5,
            "files_failed": 0,
            "files_skipped": 0,
            "documents_created": 7,
            "chunks_created": 120,
            "errors": [],
            "elapsed_seconds": 12.5,
        }
        _print_report(report)
        captured = capsys.readouterr()
        assert "5" in captured.out
        assert "120" in captured.out

    def test_print_report_with_errors(self, capsys):
        report = {
            "files_found": 3,
            "files_processed": 2,
            "files_failed": 1,
            "files_skipped": 0,
            "documents_created": 2,
            "chunks_created": 45,
            "errors": [{"file": "corrotto.pdf", "error": "File danneggiato"}],
            "elapsed_seconds": 8.3,
        }
        _print_report(report)
        captured = capsys.readouterr()
        assert "corrotto.pdf" in captured.out

    def test_print_report_with_skipped(self, capsys):
        report = {
            "files_found": 10,
            "files_processed": 2,
            "files_failed": 0,
            "files_skipped": 8,
            "documents_created": 2,
            "chunks_created": 15,
            "errors": [],
            "elapsed_seconds": 3.1,
        }
        _print_report(report)
        captured = capsys.readouterr()
        assert "8" in captured.out


class TestCLIQueryFormatting:

    def test_do_query_success(self, capsys):
        from rag_assistant.interfaces.cli import _do_query
        from rag_assistant.core.models import RetrievedChunk, RAGResponse

        mock_rag = MagicMock()
        mock_rag.query.return_value = RAGResponse(
            query="test?",
            answer="La risposta è 42.",
            success=True,
            chunks_used=[
                RetrievedChunk(
                    chunk_id="c1",
                    text="chunk text",
                    source_name="doc.pdf",
                    score=0.85,
                    metadata={"category": "DDT PDF"},
                )
            ],
            model="test-model",
            retrieval_time_ms=50.0,
            generation_time_ms=3000.0,
        )

        _do_query(mock_rag, "test?")
        captured = capsys.readouterr()

        assert "La risposta è 42." in captured.out
        assert "doc.pdf" in captured.out

    def test_do_query_error(self, capsys):
        from rag_assistant.interfaces.cli import _do_query
        from rag_assistant.core.models import RAGResponse

        mock_rag = MagicMock()
        mock_rag.query.return_value = RAGResponse(
            query="test?",
            answer="",
            success=False,
            error="Ollama non raggiungibile",
        )

        _do_query(mock_rag, "test?")
        captured = capsys.readouterr()

        assert "Errore" in captured.out


class TestCLIStatus:

    def test_do_status(self, capsys):
        from rag_assistant.interfaces.cli import _do_status

        mock_store = MagicMock()
        mock_store.count.return_value = 142

        _do_status(mock_store)
        captured = capsys.readouterr()

        assert "142" in captured.out
