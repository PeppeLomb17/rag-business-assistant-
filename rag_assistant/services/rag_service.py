"""
Servizio RAG: query → retrieval → generation → risposta.

Questo service assembla la pipeline completa di question-answering:
1. Trasforma la domanda in un embedding
2. Cerca i chunk più simili nel vector store
3. Costruisce il prompt con il contesto recuperato
4. Chiede al LLM di generare una risposta

Il prompt engineering è centralizzato qui. Il system prompt
definisce il comportamento del modello: rispondere solo dal
contesto, citare le fonti, dichiarare quando non sa.

Metriche:
    Ogni risposta include tempi di retrieval e generazione.
    Questo permette di diagnosticare dove la pipeline è lenta:
    - retrieval lento → troppi chunk, embedding model pesante
    - generazione lenta → modello troppo grande, contesto troppo lungo
"""

import logging
import time

from rag_assistant.adapters.base import Embedder, VectorStore, LLMProvider
from rag_assistant.core.models import RAGResponse, RetrievedChunk

logger = logging.getLogger(__name__)

# ─── System Prompt ────────────────────────────────────────────────────────────
# Questo è il pezzo di prompt engineering più importante del progetto.
# Ogni riga è intenzionale:
#
# 1. "ESCLUSIVAMENTE sul contesto" → il vincolo principale
# 2. "Non inventare" → riduzione esplicita delle allucinazioni
# 3. "di' chiaramente" → preferire il silenzio all'invenzione
# 4. "Cita il documento" → tracciabilità della risposta
# 5. "conciso e diretto" → evita risposte prolisse
# 6. "nella stessa lingua" → risponde in italiano se chiedi in italiano

SYSTEM_PROMPT = """Sei un assistente che risponde alle domande basandosi ESCLUSIVAMENTE sul contesto fornito.

Regole:
- Rispondi SOLO usando le informazioni presenti nel contesto
- Non inventare, non inferire, non aggiungere informazioni esterne
- Se il contesto non contiene la risposta, di' chiaramente: "Non ho trovato questa informazione nei documenti forniti"
- Cita il documento di provenienza quando possibile (es: "Secondo il file fattura.pdf...")
- Rispondi in modo conciso e diretto
- Rispondi nella stessa lingua della domanda"""


class RAGService:
    """Servizio di question-answering basato su RAG.

    Combina retrieval semantico e generazione LLM per rispondere
    a domande basandosi su documenti indicizzati.

    Args:
        embedder: per trasformare la query in vettore.
                  DEVE essere lo stesso modello usato per indicizzare.
                  Se embeddi i chunk con bge-m3 e la query con
                  nomic-embed-text, i vettori vivono in spazi
                  diversi e la similarity non ha senso.
        store: vector store con i chunk indicizzati.
        llm: modello per la generazione della risposta.
        system_prompt: istruzioni permanenti per il modello.
                      Default: SYSTEM_PROMPT (anti-allucinazione).
        top_k: quanti chunk recuperare per ogni query.
    """

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
        """Esegue una query RAG completa.

        Flusso:
        1. Embedding della domanda
        2. Retrieval dei top-k chunk
        3. Assemblaggio del prompt con contesto
        4. Generazione della risposta
        5. Packaging in RAGResponse con metriche

        Se qualcosa fallisce (Ollama giù, store vuoto, timeout),
        restituisce un RAGResponse con success=False e il messaggio
        di errore, invece di crashare. L'interfaccia (CLI o Telegram)
        può gestire l'errore in modo appropriato.

        Args:
            question: domanda in linguaggio naturale.

        Returns:
            RAGResponse con risposta, chunk usati e metriche.
        """
        logger.info(f"Query: {question[:100]}...")

        try:
            # ── Step 1: Retrieval ─────────────────────────────────────
            t_retrieval = time.perf_counter()

            query_embedding = self.embedder.embed(question)
            chunks = self.store.search(query_embedding, top_k=self.top_k)

            retrieval_ms = (time.perf_counter() - t_retrieval) * 1000

            logger.info(
                f"Retrieval: {len(chunks)} chunk in {retrieval_ms:.0f}ms"
            )

            # Se non ci sono chunk, rispondi subito senza chiamare il LLM
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

            # ── Step 2: Prompt Assembly ───────────────────────────────
            user_prompt = self._build_user_prompt(question, chunks)

            # ── Step 3: Generation ────────────────────────────────────
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
                query=question,
                answer="",
                success=False,
                error=f"Errore di connessione: {e}",
            )

        except TimeoutError as e:
            logger.error(f"Timeout: {e}")
            return RAGResponse(
                query=question,
                answer="",
                success=False,
                error=f"Timeout: {e}",
            )

        except Exception as e:
            logger.error(f"Errore imprevisto: {e}", exc_info=True)
            return RAGResponse(
                query=question,
                answer="",
                success=False,
                error=f"Errore imprevisto: {e}",
            )

    def _build_user_prompt(self, question: str, chunks: list[RetrievedChunk]) -> str:
        """Assembla il prompt utente con contesto e domanda.

        Formato del contesto per ogni chunk:
            [1] Fonte: fattura.pdf | Rilevanza: 0.85
            Testo del chunk...

        Il numero progressivo [1], [2]... permette al modello di
        citare i chunk nella risposta ("Come indicato nel documento [1]...").

        Il punteggio di rilevanza è incluso come hint per il modello:
        un chunk con score 0.95 è molto rilevante, uno con 0.45 lo è poco.
        Modelli capaci useranno questa informazione per pesare le fonti.

        La struttura CONTESTO → DOMANDA → RISPOSTA è un pattern standard
        nel prompt engineering per RAG. Il tag RISPOSTA: alla fine è un
        "prompt cue" che spinge il modello a iniziare immediatamente
        la risposta senza preamboli.
        """
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[{i}] Fonte: {chunk.source_name} | Rilevanza: {chunk.score:.2f}\n"
                f"{chunk.text}"
            )

        context = "\n\n---\n\n".join(context_parts)

        return f"""CONTESTO:
{context}

DOMANDA: {question}

RISPOSTA:"""
