"""
Bot Telegram per il RAG Business Assistant.

Permette di interrogare i documenti aziendali direttamente
da Telegram. Funziona in polling mode: il bot controlla
periodicamente se ci sono nuovi messaggi, senza bisogno
di un server pubblico o porte aperte.

Comandi:
    /start    → messaggio di benvenuto
    /help     → comandi disponibili
    /status   → statistiche del vector store
    /reindex  → re-indicizza tutti i documenti

Messaggi normali → query RAG

Upload di documenti (PDF, Excel, Word, CSV, TXT):
    Il bot scarica il file, lo salva nella cartella documenti,
    lo indicizza e conferma.

Sicurezza:
    Il bot è accessibile solo agli utenti autorizzati.
    Gli ID autorizzati vanno nel .env come RAG_TELEGRAM_ALLOWED_USERS.
    Se la lista è vuota, il bot è aperto a tutti (comodo per test,
    da chiudere in produzione).
"""

import logging
import os
from pathlib import Path

from telegram import Update, BotCommand
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from rag_assistant.adapters.ollama_embedder import OllamaEmbedder
from rag_assistant.adapters.chroma_store import ChromaStore
from rag_assistant.adapters.ollama_llm import OllamaLLM
from rag_assistant.adapters.loader_registry import supported_extensions
from rag_assistant.core.config import settings
from rag_assistant.services.ingestion_service import IngestionService
from rag_assistant.services.rag_service import RAGService

logger = logging.getLogger(__name__)

# ─── Globals (inizializzati in main) ──────────────────────────────────────────
ingestion_service: IngestionService | None = None
rag_service: RAGService | None = None
vector_store: ChromaStore | None = None

# Utenti autorizzati (lista di Telegram user ID)
ALLOWED_USERS: set[int] = set()


def _parse_allowed_users() -> set[int]:
    """Legge gli ID utenti autorizzati dall'environment.

    Formato: RAG_TELEGRAM_ALLOWED_USERS=123456,789012
    Se vuoto, tutti sono autorizzati.
    """
    raw = os.environ.get("RAG_TELEGRAM_ALLOWED_USERS", "")
    if not raw.strip():
        return set()
    try:
        return {int(uid.strip()) for uid in raw.split(",") if uid.strip()}
    except ValueError:
        logger.warning("RAG_TELEGRAM_ALLOWED_USERS contiene valori non validi")
        return set()


def _is_authorized(user_id: int) -> bool:
    """Controlla se l'utente è autorizzato."""
    if not ALLOWED_USERS:
        return True  # Nessuna restrizione
    return user_id in ALLOWED_USERS


# ─── Command Handlers ─────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler per /start."""
    if not _is_authorized(update.effective_user.id):
        await update.message.reply_text("Non sei autorizzato a usare questo bot.")
        return

    await update.message.reply_text(
        "👋 *RAG Business Assistant*\n\n"
        "Mandami una domanda e cercherò la risposta nei documenti aziendali.\n\n"
        "Puoi anche inviarmi file (PDF, Excel, Word) per aggiungerli alla knowledge base.\n\n"
        "Scrivi /help per i comandi disponibili.",
        parse_mode="Markdown",
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler per /help."""
    if not _is_authorized(update.effective_user.id):
        return

    supported = ", ".join(sorted(supported_extensions()))
    await update.message.reply_text(
        "📖 *Comandi disponibili*\n\n"
        "/status — statistiche del vector store\n"
        "/reindex — re-indicizza tutti i documenti\n"
        "/help — questo messaggio\n\n"
        "📄 *Documenti*\n"
        f"Invia un file ({supported}) per aggiungerlo alla knowledge base.\n\n"
        "💬 *Domande*\n"
        "Scrivi qualsiasi domanda in linguaggio naturale.",
        parse_mode="Markdown",
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler per /status."""
    if not _is_authorized(update.effective_user.id):
        return

    count = vector_store.count() if vector_store else 0

    await update.message.reply_text(
        "📊 *Status*\n\n"
        f"Chunk indicizzati: {count}\n"
        f"Cartella documenti: `{settings.documents_dir}`\n"
        f"Modello embedding: `{settings.embed_model}`\n"
        f"Modello LLM: `{settings.llm_model}`\n"
        f"Temperatura: {settings.temperature}\n"
        f"Top-K: {settings.top_k}",
        parse_mode="Markdown",
    )


async def cmd_reindex(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler per /reindex."""
    if not _is_authorized(update.effective_user.id):
        return

    await update.message.reply_text("🔄 Re-indicizzazione in corso... Potrebbe richiedere qualche minuto.")

    try:
        vector_store.clear()
        report = ingestion_service.ingest_directory(settings.documents_dir)

        text = (
            "✅ *Re-indicizzazione completata*\n\n"
            f"File trovati: {report['files_found']}\n"
            f"File processati: {report['files_processed']}\n"
            f"File con errori: {report['files_failed']}\n"
            f"Documenti creati: {report['documents_created']}\n"
            f"Chunk creati: {report['chunks_created']}\n"
            f"Tempo: {report['elapsed_seconds']}s"
        )

        if report["errors"]:
            text += "\n\n⚠️ *Errori:*\n"
            for err in report["errors"][:5]:  # Max 5 errori nel messaggio
                text += f"• `{err['file']}`: {err['error']}\n"
            if len(report["errors"]) > 5:
                text += f"...e altri {len(report['errors']) - 5} errori"

        await update.message.reply_text(text, parse_mode="Markdown")

    except Exception as e:
        logger.error(f"Errore durante re-indicizzazione: {e}")
        await update.message.reply_text(f"❌ Errore: {e}")


# ─── Message Handlers ─────────────────────────────────────────────────────────

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler per messaggi di testo: query RAG."""
    if not _is_authorized(update.effective_user.id):
        return

    question = update.message.text.strip()
    if not question:
        return

    # Mostra "sta scrivendo..." mentre genera la risposta
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action="typing",
    )

    logger.info(f"Query Telegram da {update.effective_user.id}: {question[:80]}...")

    response = rag_service.query(question)

    if not response.success:
        await update.message.reply_text(f"❌ Errore: {response.error}")
        return

    # Formatta la risposta
    text = f"{response.answer}\n\n"

    # Fonti
    sources = set(c.source_name for c in response.chunks_used)
    if sources:
        text += "📄 *Fonti:*\n"
        for source in sorted(sources):
            best_score = max(
                c.score for c in response.chunks_used if c.source_name == source
            )
            text += f"• {source} ({best_score:.0%})\n"

    # Metriche
    text += (
        f"\n⏱ retrieval: {response.retrieval_time_ms:.0f}ms "
        f"| generazione: {response.generation_time_ms:.0f}ms"
    )

    await update.message.reply_text(text, parse_mode="Markdown")


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler per upload di documenti: indicizza il file."""
    if not _is_authorized(update.effective_user.id):
        return

    document = update.message.document
    if not document:
        return

    file_name = document.file_name or "unknown"
    file_ext = Path(file_name).suffix.lower()

    # Controlla che il formato sia supportato
    if file_ext not in supported_extensions():
        supported = ", ".join(sorted(supported_extensions()))
        await update.message.reply_text(
            f"❌ Formato `{file_ext}` non supportato.\n"
            f"Formati accettati: {supported}",
            parse_mode="Markdown",
        )
        return

    await update.message.reply_text(f"📥 Scaricamento di `{file_name}`...", parse_mode="Markdown")

    try:
        # Scarica il file
        docs_dir = Path(settings.documents_dir)
        docs_dir.mkdir(parents=True, exist_ok=True)
        file_path = docs_dir / file_name

        tg_file = await document.get_file()
        await tg_file.download_to_drive(str(file_path))

        logger.info(f"File scaricato: {file_path}")

        # Indicizza
        await update.message.reply_text(f"🔄 Indicizzazione di `{file_name}`...", parse_mode="Markdown")

        report = ingestion_service.ingest_file(str(file_path))

        if report["files_failed"] > 0:
            await update.message.reply_text(
                f"❌ Errore indicizzando `{file_name}`: {report['errors'][0]['error']}",
                parse_mode="Markdown",
            )
        else:
            await update.message.reply_text(
                f"✅ `{file_name}` indicizzato\n\n"
                f"Documenti: {report['documents_created']}\n"
                f"Chunk: {report['chunks_created']}\n"
                f"Tempo: {report['elapsed_seconds']}s\n\n"
                f"Totale chunk nel sistema: {vector_store.count()}",
                parse_mode="Markdown",
            )

    except Exception as e:
        logger.error(f"Errore upload documento: {e}")
        await update.message.reply_text(f"❌ Errore: {e}")


# ─── Setup & Main ─────────────────────────────────────────────────────────────

async def post_init(application: Application) -> None:
    """Registra i comandi nel menu del bot."""
    commands = [
        BotCommand("start", "Avvia il bot"),
        BotCommand("help", "Comandi disponibili"),
        BotCommand("status", "Statistiche del vector store"),
        BotCommand("reindex", "Re-indicizza tutti i documenti"),
    ]
    await application.bot.set_my_commands(commands)
    logger.info("Comandi bot registrati")


def run_bot() -> None:
    """Avvia il bot Telegram in polling mode."""
    global ingestion_service, rag_service, vector_store, ALLOWED_USERS

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Leggi token
    token = settings.telegram_token.get_secret_value()
    if not token:
        print("❌ RAG_TELEGRAM_TOKEN non configurato nel .env")
        print("   Crea un bot su @BotFather e inserisci il token.")
        return

    # Utenti autorizzati
    ALLOWED_USERS = _parse_allowed_users()
    if ALLOWED_USERS:
        logger.info(f"Utenti autorizzati: {ALLOWED_USERS}")
    else:
        logger.warning("Nessuna restrizione utenti — bot aperto a tutti")

    # Composition Root
    embedder = OllamaEmbedder()
    vector_store = ChromaStore()
    llm = OllamaLLM()

    ingestion_service = IngestionService(embedder=embedder, store=vector_store)
    rag_service = RAGService(
        embedder=embedder,
        store=vector_store,
        llm=llm,
        top_k=settings.top_k,
    )

    logger.info(f"Vector store: {vector_store.count()} chunk indicizzati")

    # Costruisci l'applicazione Telegram
    app = Application.builder().token(token).post_init(post_init).build()

    # Registra gli handler
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("reindex", cmd_reindex))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Avvia in polling
    print()
    print("=" * 50)
    print("  RAG Business Assistant — Telegram Bot")
    print(f"  Vector store: {vector_store.count()} chunk")
    print("  In attesa di messaggi...")
    print("=" * 50)
    print()

    app.run_polling(drop_pending_updates=True)
