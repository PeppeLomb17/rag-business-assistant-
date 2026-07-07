"""
LLM Provider che usa Ollama per la generazione locale.

Ollama espone l'endpoint /api/chat per generazione conversazionale.
Usiamo /api/chat invece di /api/generate perché supporta i ruoli
(system, user, assistant) — fondamentale per separare le istruzioni
dal contesto.

Streaming:
    L'API di Ollama supporta lo streaming (token per token).
    In questa implementazione usiamo stream=False per semplicità:
    aspettiamo la risposta completa. Per un'interfaccia real-time
    (Telegram, web) lo streaming migliorerebbe l'UX mostrando
    la risposta progressivamente. È un'estensione futura che
    non richiede modifiche all'interfaccia astratta.

Nota sui timeout:
    La generazione con un modello 14B su M4 Pro può richiedere
    10-30 secondi per risposte lunghe. Il timeout di default
    è 120 secondi — generoso ma necessario per query complesse
    o macchine sotto carico.
"""

import logging

import requests

from rag_assistant.adapters.base import LLMProvider
from rag_assistant.core.config import settings

logger = logging.getLogger(__name__)


class OllamaLLM(LLMProvider):
    """Genera risposte testuali tramite Ollama locale.

    Args:
        model: nome del modello LLM. Default dal config.
        base_url: URL del server Ollama. Default dal config.
        temperature: controllo della creatività (0.0 = deterministico,
                    1.0 = creativo). Default dal config.
        timeout: timeout in secondi per la generazione.
                I modelli grandi possono impiegare 10-30 secondi.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        timeout: int = 120,
    ):
        self.model = model or settings.llm_model
        self.base_url = base_url or settings.ollama_base_url
        self.temperature = temperature if temperature is not None else settings.temperature
        self.timeout = timeout

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Genera una risposta testuale.

        Invia il system prompt e il user prompt come messaggi separati
        all'API /api/chat di Ollama. La risposta viene estratta dal
        campo 'message.content'.

        Args:
            system_prompt: istruzioni di comportamento per il modello.
            user_prompt: la richiesta specifica con eventuale contesto.

        Returns:
            La risposta generata dal modello.

        Raises:
            ConnectionError: se Ollama non è raggiungibile.
            TimeoutError: se la generazione supera il timeout.
            RuntimeError: se la risposta è malformata.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        logger.info(
            f"Generazione con {self.model} | "
            f"temp={self.temperature} | "
            f"prompt={len(user_prompt)} chars"
        )

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "top_p": 0.9,
                    },
                },
                timeout=self.timeout,
            )
            response.raise_for_status()

        except requests.ConnectionError:
            raise ConnectionError(
                f"Ollama non raggiungibile su {self.base_url}. "
                f"Verifica che 'ollama serve' sia attivo."
            )
        except requests.Timeout:
            raise TimeoutError(
                f"Timeout dopo {self.timeout}s. Il modello {self.model} "
                f"potrebbe essere troppo lento per questa query. "
                f"Prova con un modello più piccolo o aumenta il timeout."
            )
        except requests.HTTPError as e:
            raise RuntimeError(
                f"Errore HTTP da Ollama: {e.response.status_code} — "
                f"{e.response.text}"
            )

        data = response.json()

        # L'API /api/chat restituisce la risposta in message.content
        if "message" not in data or "content" not in data["message"]:
            raise RuntimeError(
                f"Risposta Ollama malformata: struttura 'message.content' "
                f"mancante. Risposta ricevuta: {list(data.keys())}"
            )

        answer = data["message"]["content"].strip()

        logger.info(f"Risposta generata: {len(answer)} chars")

        return answer
