"""
Servizio di indicizzazione documenti.

Orchestrazione completa: scansiona una cartella, carica ogni file
con il loader appropriato, lo chunka con la strategia giusta,
genera gli embedding e salva tutto nel vector store.

Responsabilità:
- Scansionare una directory e filtrare i file supportati
- Scegliere loader e chunker giusti per ogni file
- Gestire errori su singoli file senza bloccare il resto
- Loggare il progresso e le statistiche finali

NON è responsabile di:
- Sapere come si legge un PDF (lo fa il loader)
- Sapere come si spezza un testo (lo fa il chunker)
- Sapere come si genera un embedding (lo fa l'embedder)
- Sapere come si salva un vettore (lo fa lo store)

Questa separazione è il cuore dell'architettura a layer:
il service sa COSA fare, gli adapter sanno COME farlo.
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
    """Servizio per indicizzare documenti nel vector store.

    Args:
        embedder: implementazione dell'Embedder (es: OllamaEmbedder).
        store: implementazione del VectorStore (es: ChromaStore).

    Nota: il service riceve INTERFACCE, non implementazioni concrete.
    Non sa se l'embedder è Ollama o OpenAI, non sa se lo store è
    ChromaDB o Qdrant. Questo è il Dependency Inversion Principle:
    i moduli di alto livello (service) dipendono dalle astrazioni
    (interfacce), non dai dettagli (implementazioni).
    """

    def __init__(self, embedder: Embedder, store: VectorStore):
        self.embedder = embedder
        self.store = store

    def ingest_directory(self, directory: str) -> dict:
        """Indicizza tutti i documenti supportati in una directory.

        Flusso:
        1. Scansiona la directory per file con estensioni supportate
        2. Per ogni file: load → chunk → embed → store
        3. Raccoglie statistiche e errori
        4. Restituisce un report dell'operazione

        I file che causano errori vengono saltati — un PDF corrotto
        non deve bloccare l'indicizzazione degli altri 50 file.

        Args:
            directory: percorso alla cartella dei documenti.

        Returns:
            Dizionario con statistiche:
            {
                "files_found": 12,
                "files_processed": 11,
                "files_failed": 1,
                "documents_created": 15,  (> files se Excel multi-foglio)
                "chunks_created": 234,
                "errors": [{"file": "corrotto.pdf", "error": "..."}],
                "elapsed_seconds": 45.2,
            }
        """
        t_start = time.perf_counter()

        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory non trovata: {directory}")
        if not dir_path.is_dir():
            raise ValueError(f"Non è una directory: {directory}")

        # Trova tutti i file supportati
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

        # Processa ogni file
        all_chunks: list[Chunk] = []
        files_processed = 0
        documents_created = 0
        errors = []

        for file_path in files:
            try:
                file_chunks = self._process_file(str(file_path))
                all_chunks.extend(file_chunks)
                files_processed += 1

                # Conta i documenti creati (Excel multi-foglio → più documenti)
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

        # Embedding + storage in batch
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
        """Indicizza un singolo file.

        Utile per il bot Telegram: l'utente manda un file,
        il bot lo salva e chiama ingest_file().

        Args:
            file_path: percorso al file da indicizzare.

        Returns:
            Report come ingest_directory() ma per un solo file.
        """
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
        """Pipeline per singolo file: load → chunk.

        L'embedding viene fatto separatamente in batch per
        efficienza — è più veloce embeddare 200 chunk in
        una volta che 10 chunk alla volta per 20 file.

        Args:
            file_path: percorso al file.

        Returns:
            Lista di Chunk pronti per l'embedding.
        """
        # 1. Scegli il loader giusto per l'estensione
        loader = get_loader(file_path)

        # 2. Carica il file → lista di Document
        documents = loader.load(file_path)

        # 3. Per ogni Document, scegli il chunker e chunka
        all_chunks = []
        for doc in documents:
            # Salta documenti vuoti (es: PDF solo immagini)
            if not doc.text.strip():
                logger.warning(
                    f"Documento vuoto saltato: {doc.source_name}"
                )
                continue

            chunker = get_chunker(doc)
            chunks = chunker.chunk(doc)
            all_chunks.extend(chunks)

        return all_chunks

    def _embed_and_store(self, chunks: list[Chunk]) -> None:
        """Genera embedding e salva nel vector store.

        Processa in batch per:
        - Mostrare il progresso (utile con centinaia di chunk)
        - Non sovraccaricare Ollama con migliaia di richieste
        - Poter interrompere e riprendere (futuro)

        Args:
            chunks: lista di Chunk da embeddare e salvare.
        """
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
