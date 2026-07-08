"""Test per la CLI.

Non testa il loop interattivo (richiederebbe simulare stdin),
ma verifica che:
1. Il wiring dei service funzioni
2. Le funzioni di formattazione non crashino
3. I comandi producano output corretto
"""

import pytest
from unittest.mock import patch, MagicMock
from io import StringIO

from rag_assistant.interfaces.cli import (
    _print_header,
    _print_help,
    _print_report,
)


class TestCLIFormatting:

    def test_print_header(self, capsys):
        """L'header si stampa senza errori."""
        _print_header()
        captured = capsys.readouterr()
        assert "RAG Business Assistant" in captured.out

    def test_print_help(self, capsys):
        """L'help mostra tutti i comandi."""
        _print_help()
        captured = capsys.readouterr()
        assert "/reindex" in captured.out
        assert "/status" in captured.out
        assert "/help" in captured.out

    def test_print_report_success(self, capsys):
        """Il report di successo si stampa correttamente."""
        report = {
            "files_found": 5,
            "files_processed": 5,
            "files_failed": 0,
            "documents_created": 7,
            "chunks_created": 120,
            "errors": [],
            "elapsed_seconds": 12.5,
        }
        _print_report(report)
        captured = capsys.readouterr()
        assert "5" in captured.out
        assert "120" in captured.out
        assert "12.5" in captured.out

    def test_print_report_with_errors(self, capsys):
        """Il report con errori mostra i file falliti."""
        report = {
            "files_found": 3,
            "files_processed": 2,
            "files_failed": 1,
            "documents_created": 2,
            "chunks_created": 45,
            "errors": [{"file": "corrotto.pdf", "error": "File danneggiato"}],
            "elapsed_seconds": 8.3,
        }
        _print_report(report)
        captured = capsys.readouterr()
        assert "corrotto.pdf" in captured.out
        assert "File danneggiato" in captured.out


class TestCLIQueryFormatting:

    def test_do_query_success(self, capsys):
        """Una query riuscita stampa chunk e risposta."""
        from rag_assistant.interfaces.cli import _do_query
        from rag_assistant.core.models import RetrievedChunk, RAGResponse

        # Crea un RAGService finto che restituisce una risposta predefinita
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
        assert "0.850" in captured.out

    def test_do_query_error(self, capsys):
        """Una query fallita mostra l'errore."""
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
        assert "Ollama non raggiungibile" in captured.out


class TestCLIStatus:

    def test_do_status(self, capsys):
        """Il comando /status mostra le informazioni corrette."""
        from rag_assistant.interfaces.cli import _do_status

        mock_store = MagicMock()
        mock_store.count.return_value = 142

        _do_status(mock_store)
        captured = capsys.readouterr()

        assert "142" in captured.out
