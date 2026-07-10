"""Test per la category detection."""

from rag_assistant.services.rag_service import _detect_category


class TestCategoryDetection:

    # DDT
    def test_ddt_pdf_explicit(self):
        assert _detect_category("leggimi il DDT 240E in pdf") == "DDT PDF"

    def test_ddt_pdf_reversed(self):
        assert _detect_category("il pdf del DDT 240E") == "DDT PDF"

    def test_ddt_excel(self):
        assert _detect_category("apri il DDT 240E in excel") == "DDT EXCEL"

    def test_ddt_xlsx(self):
        assert _detect_category("DDT 100E xlsx") == "DDT EXCEL"

    def test_ddt_default_pdf(self):
        """DDT senza formato → default PDF."""
        assert _detect_category("parlami del DDT 240E") == "DDT PDF"

    # CMR
    def test_cmr_pdf(self):
        assert _detect_category("CMR 15 in pdf") == "CMR PDF"

    def test_cmr_word(self):
        assert _detect_category("CMR in word") == "CMR WORD"

    def test_cmr_docx(self):
        assert _detect_category("CMR 22 docx") == "CMR WORD"

    def test_cmr_default_pdf(self):
        assert _detect_category("mostrami il CMR 15") == "CMR PDF"

    # Fatture
    def test_fattura_airone(self):
        assert _detect_category("fattura Airone numero 15") == "Fatture Airone"

    def test_fatture_nostre(self):
        assert _detect_category("le nostre fatture di giugno") == "Fatture Nostre"

    def test_fatture_nostra(self):
        assert _detect_category("fattura nostra 200") == "Fatture Nostre"

    def test_fattura_generic_no_filter(self):
        """Fattura senza specificare quale → nessun filtro."""
        assert _detect_category("quante fatture abbiamo emesso?") is None

    # Campagna
    def test_campagna(self):
        assert _detect_category("dati della campagna 2025") == "Generale"

    # Nessun filtro
    def test_generic_query_no_filter(self):
        assert _detect_category("quanto abbiamo spedito a giugno?") is None

    def test_empty_query(self):
        assert _detect_category("") is None

    # Case insensitive
    def test_case_insensitive(self):
        assert _detect_category("DDT 240E PDF") == "DDT PDF"
        assert _detect_category("ddt 240e pdf") == "DDT PDF"
