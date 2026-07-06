"""
Interfacce astratte per tutti gli adapter.

Ogni classe definisce il CONTRATTO che un'implementazione deve rispettare.
Il codice upstream (services) lavora con queste interfacce, non con le
implementazioni concrete. Questo permette di sostituire qualsiasi
componente esterno senza toccare la logica di business.

Principio: Dependency Inversion (la D di SOLID)
    I moduli di alto livello (services) non dipendono dai moduli
    di basso livello (adapter concreti). Entrambi dipendono dalle
    astrazioni (queste interfacce).
"""

from abc import ABC, abstractmethod
from pathlib import Path

from rag_assistant.core.models import Chunk, Document, RetrievedChunk


# ─── Document Loading ─────────────────────────────────────────────────────────

class DocumentLoader(ABC):
    """Carica un file e ne estrae il contenuto testuale.

    Ogni formato (PDF, Excel, Word, TXT, CSV) ha il proprio loader.
    Il loader è responsabile di:
    - Aprire il file
    - Estrarre il testo in modo sensato per quel formato
    - Restituire uno o più Document (Excel multi-foglio → più Document)
    - Chiudere il file / liberare risorse

    Esempio d'uso (dopo aver implementato PDFLoader):
        loader = PDFLoader()
        documents = loader.load("/path/to/relazione.pdf")
        for doc in documents:
            print(doc.source_name, len(doc.text), "caratteri")
    """

    @abstractmethod
    def load(self, file_path: str) -> list[Document]:
        """Carica un file e restituisce una lista di Document.

        Restituisce una LISTA perché alcuni formati producono più documenti
        logici da un singolo file. Esempio: un file Excel con 3 fogli
        produce 3 Document, uno per foglio. Un PDF produce 1 Document.

        Args:
            file_path: percorso assoluto o relativo al file.

        Returns:
            Lista di Document con testo estratto e metadati.

        Raises:
            FileNotFoundError: se il file non esiste.
            ValueError: se il file è corrotto o illeggibile.
        """
        ...

    @staticmethod
    def supported_extensions() -> list[str]:
        """Estensioni file gestite da questo loader.

        Serve all'IngestionService per capire quale loader usare
        per ogni file. Esempio: PDFLoader restituisce [".pdf"],
        ExcelLoader restituisce [".xlsx", ".xls"].

        Returns:
            Lista di estensioni con il punto (es: [".pdf"]).
        """
        return []


# ─── Chunking ─────────────────────────────────────────────────────────────────

class Chunker(ABC):
    """Divide un Document in Chunk pronti per l'embedding.

    Strategie diverse per contenuti diversi:
    - SentenceChunker: per prosa (PDF, Word). Taglia ai confini di
      frase con overlap per preservare il contesto.
    - TabularChunker: per dati tabulari (Excel, CSV). Converte le
      righe in frasi leggibili e le raggruppa.

    La scelta del chunker impatta direttamente la qualità del retrieval.
    Chunk troppo piccoli perdono contesto. Chunk troppo grandi
    introducono rumore e diluiscono la rilevanza.

    Esempio d'uso:
        chunker = SentenceChunker(chunk_size=300, overlap=50)
        chunks = chunker.chunk(document)
        print(f"{len(chunks)} chunk generati")
        print(f"Primo chunk: {chunks[0].word_count} parole")
    """

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Divide un documento in chunk.

        Args:
            document: il Document da spezzare.

        Returns:
            Lista ordinata di Chunk con indice progressivo.
        """
        ...


# ─── Embedding ────────────────────────────────────────────────────────────────

class Embedder(ABC):
    """Trasforma testo in un vettore numerico (embedding).

    L'embedding cattura il "significato" del testo in uno spazio
    vettoriale dove testi simili hanno vettori vicini. Questo è
    il meccanismo che permette al retrieval di trovare chunk
    rilevanti anche se non contengono le stesse parole della query.

    Proprietà fondamentale: il modello di embedding deve essere
    lo STESSO in fase di indicizzazione e in fase di query.
    Se embeddi i chunk con bge-m3 e la query con nomic-embed-text,
    i vettori vivono in spazi diversi e la similarity non ha senso.

    Esempio d'uso:
        embedder = OllamaEmbedder(model="bge-m3")
        vector = embedder.embed("Fattura n. 2024-0847")
        print(f"Dimensioni: {len(vector)}")  # es: 1024
    """

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Genera l'embedding di un testo.

        Args:
            text: testo da vettorizzare.

        Returns:
            Lista di float che rappresenta il vettore embedding.
        """
        ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Genera embedding per una lista di testi.

        Più efficiente di chiamare embed() in loop perché alcune
        implementazioni (API remote) supportano il batching nativo.

        Args:
            texts: lista di testi da vettorizzare.

        Returns:
            Lista di vettori, uno per ogni testo in input.
        """
        ...


# ─── Vector Store ─────────────────────────────────────────────────────────────

class VectorStore(ABC):
    """Salva chunk vettorizzati e li recupera per similarità.

    Il vector store è il "database" della pipeline RAG. A differenza
    di un database tradizionale che cerca per match esatto (WHERE name = 'X'),
    il vector store cerca per SIMILARITÀ nello spazio vettoriale.

    Due operazioni fondamentali:
    - add: salva chunk con i loro embedding
    - search: trova i chunk più simili a un vettore query

    La metrica di similarità (cosine, L2, dot product) è una scelta
    architetturale. Noi usiamo cosine similarity perché è lo standard
    per embedding testuali e non dipende dalla magnitudine dei vettori.

    Esempio d'uso:
        store = ChromaStore(persist_dir="./chroma_db")
        store.add(chunks, embeddings)
        results = store.search(query_embedding, top_k=5)
        for r in results:
            print(f"{r.source_name}: {r.score:.3f}")
    """

    @abstractmethod
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Salva chunk con i loro embedding nel vector store.

        Args:
            chunks: lista di Chunk da salvare.
            embeddings: lista di vettori, uno per chunk.
                        len(embeddings) deve essere == len(chunks).

        Raises:
            ValueError: se len(chunks) != len(embeddings).
        """
        ...

    @abstractmethod
    def search(self, embedding: list[float], top_k: int = 5) -> list[RetrievedChunk]:
        """Cerca i chunk più simili a un embedding.

        Args:
            embedding: vettore query.
            top_k: numero massimo di risultati.

        Returns:
            Lista di RetrievedChunk ordinata per score decrescente.
        """
        ...

    @abstractmethod
    def clear(self) -> None:
        """Svuota il vector store. Usato per il re-indexing completo."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Restituisce il numero di chunk indicizzati."""
        ...


# ─── LLM Provider ─────────────────────────────────────────────────────────────

class LLMProvider(ABC):
    """Genera testo a partire da un prompt.

    Il LLM è l'ultimo anello della pipeline RAG: riceve il contesto
    (chunk recuperati) e la domanda dell'utente, e genera una risposta
    in linguaggio naturale.

    La separazione tra system prompt e user prompt è fondamentale:
    - System prompt: istruzioni permanenti ("rispondi solo dal contesto")
    - User prompt: la domanda specifica con il contesto allegato

    Il system prompt è dove si fa Prompt Engineering. È il pezzo
    che controlla se il modello allucina o resta ancorato ai dati.

    Esempio d'uso:
        llm = OllamaLLM(model="qwen2.5:14b", temperature=0.2)
        answer = llm.generate(
            system_prompt="Rispondi solo basandoti sul contesto.",
            user_prompt="CONTESTO: ... DOMANDA: quanti cluster?"
        )
        print(answer)
    """

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Genera una risposta testuale.

        Args:
            system_prompt: istruzioni di comportamento per il modello.
            user_prompt: la richiesta specifica con eventuale contesto.

        Returns:
            La risposta generata dal modello.

        Raises:
            ConnectionError: se il modello non è raggiungibile.
            TimeoutError: se la generazione supera il timeout.
        """
        ...
