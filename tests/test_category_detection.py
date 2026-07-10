"""Test per category detection, format detection e document number extraction."""

from rag_assistant.services.rag_service import (
    _detect_category, _extract_doc_number, _has_explicit_format,
)


class TestCategoryDetection:

    def test_ddt_pdf_explicit(self):
        assert _detect_category("leggimi il DDT 240E in pdf") == "DDT PDF"

    def test_ddt_excel(self):
        assert _detect_category("apri il DDT 240E in excel") == "DDT EXCEL"

    def test_ddt_without_format_no_category(self):
        """DDT senza formato esplicito non produce categoria."""
        assert _detect_category("parlami del DDT 240E") is None

    def test_cmr_pdf(self):
        assert _detect_category("CMR 15 in pdf") == "CMR PDF"

    def test_cmr_word(self):
        assert _detect_category("CMR in word") == "CMR WORD"

    def test_fattura_airone(self):
        assert _detect_category("fattura Airone numero 15") == "Fatture Airone"

    def test_fatture_nostre(self):
        assert _detect_category("le nostre fatture di giugno") == "Fatture Nostre"

    def test_campagna(self):
        assert _detect_category("dati della campagna") == "Generale"

    def test_generic_no_filter(self):
        assert _detect_category("quanto abbiamo spedito?") is None


class TestFormatDetection:

    def test_pdf_explicit(self):
        assert _has_explicit_format("DDT 240E in pdf") is True

    def test_excel_explicit(self):
        assert _has_explicit_format("DDT in excel") is True

    def test_word_explicit(self):
        assert _has_explicit_format("CMR in word") is True

    def test_no_format(self):
        assert _has_explicit_format("leggimi il DDT 240E") is False

    def test_generic_question(self):
        assert _has_explicit_format("quanto abbiamo spedito?") is False


class TestDocNumberExtraction:

    def test_ddt_number(self):
        assert _extract_doc_number("leggimi il DDT 240E") == "240E"

    def test_ddt_number_lowercase(self):
        assert _extract_doc_number("ddt 100e") == "100E"

    def test_cmr_number(self):
        assert _extract_doc_number("CMR numero 15") == "15"

    def test_cmr_with_n(self):
        assert _extract_doc_number("CMR n. 22") == "22"

    def test_fattura_number(self):
        assert _extract_doc_number("fattura 2024-0847") == "2024-0847"

    def test_no_number(self):
        assert _extract_doc_number("quanto abbiamo spedito?") is None

    def test_generic_question(self):
        assert _extract_doc_number("mostrami le fatture di giugno") is None
