"""
Interfaccia a riga di comando per il RAG assistant.

Comandi:
    /reindex  → cancella tutto e re-indicizza da zero
    /update   → indicizza solo file nuovi (incrementale)
    /status   → statistiche
    /help     → comandi
    q         → esci
"""

import logging
import sys

from rag_assistant.adapters.ollama_embedder import OllamaEmbedder
from rag_assistant.adapters.chroma_store import ChromaStore
from rag_assistant.adapters.ollama_llm import OllamaLLM
from rag_assistant.core.config import settings
from rag_assistant.services.ingestion_service import IngestionService
from rag_assistant.services.rag_service import RAGService


def _setup_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _create_services() -> tuple[IngestionService, RAGService, ChromaStore]:
    embedder = OllamaEmbedder()
    store = ChromaStore()
    llm = OllamaLLM()

    ingestion = IngestionService(embedder=embedder, store=store)
    rag = RAGService(
        embedder=embedder,
        store=store,
        llm=llm,
        top_k=settings.top_k,
    )

    return ingestion, rag, store


def _print_header() -> None:
    print()
    print("=" * 60)
    print("  RAG Business Assistant")
    print("  Interroga i tuoi documenti aziendali")
    print("=" * 60)
    print()


def _print_help() -> None:
    print()
    print("Comandi disponibili:")
    print("  /update   — indicizza solo file nuovi (veloce)")
    print("  /reindex  — cancella tutto e re-indicizza da zero")
    print("  /status   — statistiche del vector store")
    print("  /help     — mostra questo messaggio")
    print("  q         — esci")
    print()
    print("Oppure scrivi una domanda in linguaggio naturale.")
    print()


def _print_report(report: dict) -> None:
    print()
    print(f"  File trovati:     {report['files_found']}")
    print(f"  File processati:  {report['files_processed']}")
    print(f"  File saltati:     {report['files_skipped']}")
    print(f"  File con errori:  {report['files_failed']}")
    print(f"  Documenti creati: {report['documents_created']}")
    print(f"  Chunk creati:     {report['chunks_created']}")
    print(f"  Tempo:            {report['elapsed_seconds']}s")

    if report["errors"]:
        print()
        print("  Errori:")
        for err in report["errors"]:
            print(f"    ✗ {err['file']}: {err['error']}")
    print()


def _do_reindex(ingestion: IngestionService, store: ChromaStore) -> None:
    """Re-indicizzazione completa: cancella tutto e riparte."""
    print()
    print(f"Indicizzazione COMPLETA di: {settings.documents_dir}")
    print("-" * 40)

    store.clear()
    report = ingestion.ingest_directory(settings.documents_dir, force=True)
    _print_report(report)


def _do_update(ingestion: IngestionService) -> None:
    """Indicizzazione incrementale: solo file nuovi."""
    print()
    print(f"Indicizzazione INCREMENTALE di: {settings.documents_dir}")
    print("-" * 40)

    report = ingestion.ingest_directory(settings.documents_dir, force=False)
    _print_report(report)


def _do_status(store: ChromaStore) -> None:
    print()
    print(f"  Cartella documenti: {settings.documents_dir}")
    print(f"  Vector store:       {settings.chroma_persist_dir}")
    print(f"  Chunk indicizzati:  {store.count()}")
    print(f"  Modello embedding:  {settings.embed_model}")
    print(f"  Modello LLM:        {settings.llm_model}")
    print(f"  Temperatura:        {settings.temperature}")
    print(f"  Top-K:              {settings.top_k}")
    print()


def _do_query(rag: RAGService, question: str) -> None:
    response = rag.query(question)

    if not response.success:
        print(f"\n  ✗ Errore: {response.error}\n")
        return

    print()
    print("  Chunk recuperati:")
    for i, chunk in enumerate(response.chunks_used, 1):
        category = chunk.metadata.get("category", "")
        cat_str = f" [{category}]" if category else ""
        print(f"    [{i}]{cat_str} {chunk.source_name} (score: {chunk.score:.3f})")

    print()
    print(f"  {response.answer}")

    print()
    print(
        f"  ⏱  retrieval: {response.retrieval_time_ms:.0f}ms | "
        f"generazione: {response.generation_time_ms:.0f}ms"
    )
    print()


def _query_loop(ingestion: IngestionService, rag: RAGService, store: ChromaStore) -> None:
    _print_help()

    while True:
        try:
            user_input = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nCiao.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("q", "quit", "exit"):
            print("Ciao.")
            break

        if user_input == "/reindex":
            _do_reindex(ingestion, store)
            continue

        if user_input == "/update":
            _do_update(ingestion)
            continue

        if user_input == "/status":
            _do_status(store)
            continue

        if user_input == "/help":
            _print_help()
            continue

        _do_query(rag, user_input)


def main() -> None:
    _setup_logging()
    _print_header()

    try:
        ingestion, rag, store = _create_services()
    except Exception as e:
        print(f"\n  ✗ Errore durante l'inizializzazione: {e}")
        print("  Verifica che Ollama sia attivo: ollama serve")
        print()
        sys.exit(1)

    if store.count() == 0:
        print(f"  Vector store vuoto.")
        print(f"  Cartella documenti: {settings.documents_dir}")
        print()

        try:
            answer = input("  Vuoi indicizzare i documenti ora? [S/n] ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\nCiao.")
            return

        if answer in ("", "s", "si", "sì", "y", "yes"):
            _do_reindex(ingestion, store)
        else:
            print("  OK, puoi usare /reindex o /update quando vuoi.\n")

    else:
        print(f"  Vector store caricato: {store.count()} chunk indicizzati.")
        print()

    _query_loop(ingestion, rag, store)
