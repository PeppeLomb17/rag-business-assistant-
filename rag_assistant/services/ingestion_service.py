"""
Servizio di indicizzazione documenti.

Orchestrazione completa: scansiona una cartella, carica ogni file
con il loader appropriato, lo chunka con la strategia giusta,
genera gli embedding e salva tutto nel vector store.
"""

import logging
import time
from pathlib import Path

from rag_assistant.adapters.base import Embedder, VectorStore
from rag_assistant.adapters.loader_registry import get_loader, supported_extensions
from rag_assistant.adapters.chunker_registry import get_chunker
from rag_assistant.core.models import Chunk, Document

logger = logging.getLogger(__name__)


class IngestionService:
    """Servizio per indicizzare documenti nel vector store."""

    def __init__(self, embedder: Embedder, store: VectorStore):
        self.embedder = embedder
        self.store = store

    def ingest_directory(self, directory: str) -> dict:
        """Indicizza tutti i documenti supportati in una directory."""
        t_start = time.perf_counter()

        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory non trovata: {directory}")
        if not dir_path.is_dir():
            raise ValueError(f"Non è una directory: {directory}")

        supported = set(supported_extensions())
        files = [
            f for f in sorted(dir_path.iterdir())
            if f.is_file() and f.suffix.lower() in supported
        ]

        logger.info(f"Trovati {len(files)} file supportati in {directory}")

        if not files:
            logger.warning(
                f"Nessun file supportato trovato in {directory}. "
                f"Formati supportati: {', '.join(sorted(supported))}"
            )
            return self._build_report(0, 0, 0, 0, 0, [], t_start)

        all_chunks: list[Chunk] = []
        files_processed = 0
        documents_created = 0
        errors = []

        for file_path in files:
            try:
                file_chunks = self._process_file(str(file_path))
                all_chunks.extend(file_chunks)
                files_processed += 1

                doc_ids = set(c.doc_id for c in file_chunks)
                documents_created += len(doc_ids)

                logger.info(
                    f"  ✓ {file_path.name}: "
                    f"{len(doc_ids)} doc, {len(file_chunks)} chunk"
                )

            except Exception as e:
                errors.append({
                    "file": file_path.name,
                    "error": str(e),
                })
                logger.error(f"  ✗ {file_path.name}: {e}")

        if all_chunks:
            self._embed_and_store(all_chunks)

        return self._build_report(
            files_found=len(files),
            files_processed=files_processed,
            documents_created=documents_created,
            chunks_created=len(all_chunks),
            files_failed=len(errors),
            errors=errors,
            t_start=t_start,
        )

    def ingest_file(self, file_path: str) -> dict:
        """Indicizza un singolo file."""
        t_start = time.perf_counter()

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File non trovato: {file_path}")

        try:
            chunks = self._process_file(file_path)
            self._embed_and_store(chunks)

            doc_ids = set(c.doc_id for c in chunks)

            logger.info(
                f"Indicizzato {path.name}: "
                f"{len(doc_ids)} doc, {len(chunks)} chunk"
            )

            return self._build_report(
                files_found=1,
                files_processed=1,
                documents_created=len(doc_ids),
                chunks_created=len(chunks),
                files_failed=0,
                errors=[],
                t_start=t_start,
            )

        except Exception as e:
            logger.error(f"Errore indicizzando {path.name}: {e}")
            return self._build_report(
                files_found=1,
                files_processed=0,
                documents_created=0,
                chunks_created=0,
                files_failed=1,
                errors=[{"file": path.name, "error": str(e)}],
                t_start=t_start,
            )

    def _process_file(self, file_path: str) -> list[Chunk]:
        """Pipeline per singolo file: load → chunk → inject source name."""
        loader = get_loader(file_path)
        documents = loader.load(file_path)

        all_chunks = []
        for doc in documents:
            if not doc.text.strip():
                logger.warning(
                    f"Documento vuoto saltato: {doc.source_name}"
                )
                continue

            chunker = get_chunker(doc)
            chunks = chunker.chunk(doc)

            # Inietta il nome del file all'inizio di ogni chunk.
            # Questo permette alla semantic search di trovare chunk
            # per nome documento (es: "DDT 219E") perché il nome
            # diventa parte del testo embeddato.
            for chunk in chunks:
                chunk.text = f"[Documento: {doc.source_name}]\n{chunk.text}"
                chunk.char_count = len(chunk.text)
                chunk.word_count = len(chunk.text.split())

            all_chunks.extend(chunks)

        return all_chunks

    def _embed_and_store(self, chunks: list[Chunk]) -> None:
        """Genera embedding e salva nel vector store."""
        logger.info(f"Embedding di {len(chunks)} chunk...")

        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_batch(texts)

        logger.info("Salvataggio nel vector store...")
        self.store.add(chunks, embeddings)

        logger.info(
            f"Indicizzazione completata: {len(chunks)} chunk, "
            f"{self.store.count()} totali nel vector store"
        )

    def _build_report(
        self,
        files_found: int,
        files_processed: int,
        documents_created: int,
        chunks_created: int,
        files_failed: int,
        errors: list,
        t_start: float,
    ) -> dict:
        """Costruisce il report dell'operazione."""
        elapsed = time.perf_counter() - t_start

        report = {
            "files_found": files_found,
            "files_processed": files_processed,
            "files_failed": files_failed,
            "documents_created": documents_created,
            "chunks_created": chunks_created,
            "errors": errors,
            "elapsed_seconds": round(elapsed, 2),
        }

        logger.info(
            f"Report: {files_processed}/{files_found} file, "
            f"{documents_created} documenti, {chunks_created} chunk "
            f"in {elapsed:.1f}s"
        )

        if errors:
            logger.warning(f"File con errori: {[e['file'] for e in errors]}")

        return report
