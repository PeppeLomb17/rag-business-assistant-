"""
Servizio RAG: query → retrieval → generation → risposta.

Include category detection: se la query menziona un formato
o una categoria specifica, il retrieval viene filtrato
per cercare solo in quella categoria.
"""

import logging
import re
import time

from rag_assistant.adapters.base import Embedder, VectorStore, LLMProvider
from rag_assistant.core.models import RAGResponse, RetrievedChunk

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Sei un assistente che risponde alle domande basandosi ESCLUSIVAMENTE sul contesto fornito.

Regole:
- Rispondi SOLO usando le informazioni presenti nel contesto
- Non inventare, non inferire, non aggiungere informazioni esterne
- Se il contesto non contiene la risposta, di' chiaramente: "Non ho trovato questa informazione nei documenti forniti"
- Cita il documento di provenienza quando possibile (es: "Secondo il file fattura.pdf...")
- Rispondi in modo conciso e diretto
- Rispondi nella stessa lingua della domanda"""

# ─── Category Detection ───────────────────────────────────────────────────────
# Mappa keyword nella query → nome categoria nelle cartelle.
# L'ordine conta: pattern più specifici prima dei generici.

CATEGORY_PATTERNS = [
    # DDT
    (r"\bddt\b.*\bpdf\b", "DDT PDF"),
    (r"\bpdf\b.*\bddt\b", "DDT PDF"),
    (r"\bddt\b.*\bexcel\b", "DDT EXCEL"),
    (r"\bexcel\b.*\bddt\b", "DDT EXCEL"),
    (r"\bddt\b.*\bxlsx?\b", "DDT EXCEL"),
    # CMR
    (r"\bcmr\b.*\bpdf\b", "CMR PDF"),
    (r"\bpdf\b.*\bcmr\b", "CMR PDF"),
    (r"\bcmr\b.*\bword\b", "CMR WORD"),
    (r"\bword\b.*\bcmr\b", "CMR WORD"),
    (r"\bcmr\b.*\bdocx?\b", "CMR WORD"),
    # Fatture
    (r"\bfattur[ae]\b.*\bairone\b", "Fatture Airone"),
    (r"\bairone\b.*\bfattur[ae]\b", "Fatture Airone"),
    (r"\bfattur[ae]\b.*\bnostr[aei]\b", "Fatture Nostre"),
    (r"\bnostr[aei]\b.*\bfattur[ae]\b", "Fatture Nostre"),
    # Campagna
    (r"\bcampagna\b", "Generale"),
    # Generici (meno specifici, in fondo)
    (r"\bddt\b", "DDT PDF"),       # DDT senza formato → default PDF
    (r"\bcmr\b", "CMR PDF"),       # CMR senza formato → default PDF
    (r"\bfattur[ae]\b", None),     # Fattura senza specificare quale → nessun filtro
]


def _detect_category(query: str) -> str | None:
    """Detecta la categoria dalla query dell'utente.

    Analizza la query cercando keyword che indicano un formato
    o un tipo di documento specifico.

    Esempi:
        "leggimi il DDT 240E in pdf"    → "DDT PDF"
        "mostrami la fattura Airone 15"  → "Fatture Airone"
        "CMR in word del 15 giugno"     → "CMR WORD"
        "DDT 240E"                       → "DDT PDF" (default)
        "quanto abbiamo fatturato?"      → None (nessun filtro)

    Returns:
        Nome della categoria o None se non detectata.
    """
    query_lower = query.lower()

    for pattern, category in CATEGORY_PATTERNS:
        if re.search(pattern, query_lower):
            return category

    return None


class RAGService:
    """Servizio di question-answering basato su RAG."""

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
        """Esegue una query RAG con category detection automatica."""
        logger.info(f"Query: {question[:100]}...")

        try:
            # ── Step 0: Category Detection ────────────────────────
            category = _detect_category(question)
            if category:
                logger.info(f"Categoria detectata: {category}")

            # ── Step 1: Retrieval ─────────────────────────────────
            t_retrieval = time.perf_counter()

            query_embedding = self.embedder.embed(question)

            # Passa il filtro allo store (se supportato)
            if category and hasattr(self.store, 'search'):
                try:
                    chunks = self.store.search(
                        query_embedding,
                        top_k=self.top_k,
                        category_filter=category,
                    )
                except TypeError:
                    # Se lo store non supporta category_filter, fallback
                    chunks = self.store.search(query_embedding, top_k=self.top_k)
            else:
                chunks = self.store.search(query_embedding, top_k=self.top_k)

            retrieval_ms = (time.perf_counter() - t_retrieval) * 1000

            logger.info(
                f"Retrieval: {len(chunks)} chunk in {retrieval_ms:.0f}ms"
                f"{f' [filtro: {category}]' if category else ''}"
            )

            if not chunks:
                return RAGResponse(
                    query=question,
                    answer="Nessun documento indicizzato. Usa il comando di indicizzazione prima di fare domande.",
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
