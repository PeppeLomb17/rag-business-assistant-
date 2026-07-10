"""
Servizio RAG con category detection e document number extraction.
"""

import logging
import re
import time

from rag_assistant.adapters.base import Embedder, VectorStore, LLMProvider
from rag_assistant.core.models import RAGResponse, RetrievedChunk

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Sei un assistente aziendale specializzato in documenti di trasporto e logistica ortofrutticola.

I documenti che ricevi come contesto possono essere:
- DDT (Documenti Di Trasporto): contengono mittente, destinatario, 1° cessionario, luogo di scarico, descrizione merce, peso, colli, vettore, data
- CMR (lettere di vettura internazionali): simili ai DDT ma per trasporti internazionali
- Fatture: documenti contabili con importi, IVA, date

IMPORTANTE: il testo dei DDT è estratto da PDF strutturati come form. Ogni chunk inizia con [Documento: nome_file] — quel nome identifica il documento.

Regole:
- Rispondi SOLO usando le informazioni presenti nel contesto
- Non inventare e non aggiungere informazioni esterne
- Se il contesto non contiene la risposta, dillo chiaramente
- Cita il documento di provenienza
- Rispondi in modo conciso e diretto
- Rispondi nella stessa lingua della domanda"""

CATEGORY_PATTERNS = [
    (r"\bddt\b.*\bpdf\b", "DDT PDF"),
    (r"\bpdf\b.*\bddt\b", "DDT PDF"),
    (r"\bddt\b.*\bexcel\b", "DDT EXCEL"),
    (r"\bexcel\b.*\bddt\b", "DDT EXCEL"),
    (r"\bddt\b.*\bxlsx?\b", "DDT EXCEL"),
    (r"\bcmr\b.*\bpdf\b", "CMR PDF"),
    (r"\bpdf\b.*\bcmr\b", "CMR PDF"),
    (r"\bcmr\b.*\bword\b", "CMR WORD"),
    (r"\bword\b.*\bcmr\b", "CMR WORD"),
    (r"\bcmr\b.*\bdocx?\b", "CMR WORD"),
    (r"\bfattur[ae]\b.*\bairone\b", "Fatture Airone"),
    (r"\bairone\b.*\bfattur[ae]\b", "Fatture Airone"),
    (r"\bfattur[ae]\b.*\bnostr[aei]\b", "Fatture Nostre"),
    (r"\bnostr[aei]\b.*\bfattur[ae]\b", "Fatture Nostre"),
    (r"\bcampagna\b", "Generale"),
]

# Pattern che indicano formato esplicito nella query
FORMAT_EXPLICIT_PATTERNS = [
    r"\bpdf\b", r"\bexcel\b", r"\bxlsx?\b",
    r"\bword\b", r"\bdocx?\b",
]


def _detect_category(query: str) -> str | None:
    query_lower = query.lower()
    for pattern, category in CATEGORY_PATTERNS:
        if re.search(pattern, query_lower):
            return category
    return None


def _has_explicit_format(query: str) -> bool:
    """Controlla se l'utente ha specificato un formato esplicito."""
    query_lower = query.lower()
    for pattern in FORMAT_EXPLICIT_PATTERNS:
        if re.search(pattern, query_lower):
            return True
    return False


def _extract_doc_number(query: str) -> str | None:
    match = re.search(
        r'(?:ddt|cmr|fattura|fatture)\s+(?:n\.?|numero|num\.?)?\s*(\d+[\w-]*)',
        query.lower(),
    )
    if match:
        return match.group(1).upper()
    return None


class RAGService:

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        llm: LLMProvider,
        system_prompt: str = SYSTEM_PROMPT,
        top_k: int = 5,
    ):
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.system_prompt = system_prompt
        self.top_k = top_k

    def query(self, question: str) -> RAGResponse:
        logger.info(f"Query: {question[:100]}...")

        try:
            # ── Step 0: Analisi query ─────────────────────────────
            category = _detect_category(question)
            doc_number = _extract_doc_number(question)
            explicit_format = _has_explicit_format(question)

            # Logica di filtraggio:
            # - Se c'è un numero documento → usa text_contains (preciso)
            #   + aggiungi categoria SOLO se l'utente ha specificato un formato esplicito
            # - Se non c'è numero → usa solo categoria per query generiche
            use_category = None
            use_text = None

            if doc_number:
                use_text = doc_number
                if explicit_format and category:
                    use_category = category
                logger.info(
                    f"Numero documento: {doc_number}"
                    f"{f' | Formato esplicito: {category}' if explicit_format and category else ' | Nessun filtro formato'}"
                )
            elif category:
                use_category = category
                logger.info(f"Categoria: {category}")

            # ── Step 1: Retrieval ─────────────────────────────────
            t_retrieval = time.perf_counter()

            query_embedding = self.embedder.embed(question)

            kwargs = {"top_k": self.top_k}
            if use_category:
                kwargs["category_filter"] = use_category
            if use_text:
                kwargs["text_contains"] = use_text

            try:
                chunks = self.store.search(query_embedding, **kwargs)
            except TypeError:
                chunks = self.store.search(query_embedding, top_k=self.top_k)

            retrieval_ms = (time.perf_counter() - t_retrieval) * 1000

            filters_log = ""
            if use_category:
                filters_log += f" [cat: {use_category}]"
            if use_text:
                filters_log += f" [doc: {use_text}]"

            logger.info(
                f"Retrieval: {len(chunks)} chunk in {retrieval_ms:.0f}ms{filters_log}"
            )

            if not chunks:
                return RAGResponse(
                    query=question,
                    answer="Nessun documento trovato con quei criteri.",
                    success=True,
                    chunks_used=[],
                    model=getattr(self.llm, 'model', 'unknown'),
                    retrieval_time_ms=retrieval_ms,
                    generation_time_ms=0.0,
                )

            # ── Step 2: Prompt Assembly ───────────────────────────
            user_prompt = self._build_user_prompt(question, chunks)

            # ── Step 3: Generation ────────────────────────────────
            t_generation = time.perf_counter()
            answer = self.llm.generate(self.system_prompt, user_prompt)
            generation_ms = (time.perf_counter() - t_generation) * 1000

            logger.info(f"Generazione: {len(answer)} chars in {generation_ms:.0f}ms")

            return RAGResponse(
                query=question,
                answer=answer,
                success=True,
                chunks_used=chunks,
                model=getattr(self.llm, 'model', 'unknown'),
                retrieval_time_ms=retrieval_ms,
                generation_time_ms=generation_ms,
            )

        except ConnectionError as e:
            logger.error(f"Connessione fallita: {e}")
            return RAGResponse(
                query=question, answer="", success=False,
                error=f"Errore di connessione: {e}",
            )
        except TimeoutError as e:
            logger.error(f"Timeout: {e}")
            return RAGResponse(
                query=question, answer="", success=False,
                error=f"Timeout: {e}",
            )
        except Exception as e:
            logger.error(f"Errore imprevisto: {e}", exc_info=True)
            return RAGResponse(
                query=question, answer="", success=False,
                error=f"Errore imprevisto: {e}",
            )

    def _build_user_prompt(self, question: str, chunks: list[RetrievedChunk]) -> str:
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            category = chunk.metadata.get("category", "")
            cat_str = f" | Categoria: {category}" if category else ""
            context_parts.append(
                f"[{i}] Fonte: {chunk.source_name}{cat_str} | Rilevanza: {chunk.score:.2f}\n"
                f"{chunk.text}"
            )

        context = "\n\n---\n\n".join(context_parts)

        return f"""CONTESTO:
{context}

DOMANDA: {question}

RISPOSTA:"""
