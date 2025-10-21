"""Hypothesis-link v2
=====================

This module provides a re-imagined version of the original Hypothesis-link
application.  The new implementation focuses on the following guiding
principles:

* **Resilience** – network failures are anticipated and handled with
  structured retries, exponential backoff and rate limiting safeguards.
* **Security** – user supplied data is sanitised before being shipped to the
  backend service.  Configuration secrets are kept out of source control and
  only read via environment variables or configuration files owned by the
  executing user.
* **Observability** – a lightweight metrics subsystem tracks performance data
  that can later be exported or logged, providing visibility into latency
  spikes and API usage anomalies.
* **Extensibility** – the implementation is modular and self-documenting,
  allowing future maintainers to extend each component independently without
  having to untangle global state.

The overall design is intentionally verbose; the user requested enterprise
level robustness, and we comply by investing into defensive programming, rich
inline documentation and a disciplined separation of concerns.

The module is designed to run on Python 3.13 (and is backwards compatible with
3.10+).  No deprecated APIs are used, and type hints are supplied to enable
static analysis with tools such as `pyright` or `mypy`.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import queue
import random
import re
import statistics
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Tuple

import requests
from requests import Response
from rich.console import Console

try:
    # Tkinter is part of the standard library.  Import errors usually indicate
    # a headless server without GUI capabilities.  We catch the error to raise
    # a more descriptive exception later on when the GUI is initialised.
    import tkinter as tk
    from tkinter import scrolledtext, messagebox
except ModuleNotFoundError as exc:  # pragma: no cover - depends on environment
    raise RuntimeError(
        "Tkinter is required to run the Hypothesis-link GUI. "
        "Install the Python Tk bindings for your platform."  # noqa: TRY002
    ) from exc


# ---------------------------------------------------------------------------
# Logging infrastructure
# ---------------------------------------------------------------------------

def _build_logger() -> logging.Logger:
    """Configure a module level logger with safe defaults.

    The logger writes to stdout and is intentionally conservative: INFO for
    regular events and DEBUG for verbose traces.  In high-volume deployments a
    JSON formatter or remote log shipper can be plugged in by replacing the
    handler configuration here.
    """

    logger = logging.getLogger("hypothesis_link")
    if logger.handlers:
        return logger

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(threadName)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


LOGGER = _build_logger()
CONSOLE = Console()


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class ConfigurationError(RuntimeError):
    """Raised when the application configuration is invalid."""


class ChatStorageError(RuntimeError):
    """Raised when the conversation history cannot be persisted."""


class APIError(RuntimeError):
    """Generic API failure encompassing HTTP or schema level issues."""


class RateLimitError(APIError):
    """Raised when API calls exceed the configured rate limits."""


class ResponseFormatError(APIError):
    """Raised when the language model returns an unexpected schema."""


# ---------------------------------------------------------------------------
# Configuration handling
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class AppPaths:
    """Container for important filesystem paths used by the application."""

    base_dir: Path = field(default_factory=lambda: Path.cwd())
    config_file_name: str = "hypothesis_link.json"
    history_file_name: str = "hypothesis_history.json"

    @property
    def config_path(self) -> Path:
        return self.base_dir / self.config_file_name

    @property
    def history_path(self) -> Path:
        return self.base_dir / self.history_file_name


@dataclass(slots=True)
class AppConfig:
    """Configuration data for the LM backend and runtime behaviour."""

    api_base_url: str
    model_name: str
    timeout: float = 45.0
    max_retries: int = 3
    request_backoff: float = 1.8
    history_limit: int = 32
    rate_limit_per_minute: int = 90
    verify_tls: bool = True

    @classmethod
    def from_env(cls, paths: AppPaths) -> "AppConfig":
        """Load configuration from environment variables or disk.

        Precedence order (highest to lowest): environment variables, JSON
        configuration file, sensible defaults.  Only a minimal JSON structure is
        required to bootstrap the application:

        ```json
        {
            "api_base_url": "http://localhost:1234/v1/chat/completions",
            "model_name": "liquid/lfm2-1.2b"
        }
        ```
        """

        env_url = os.getenv("HYPOTHESIS_API_URL")
        env_model = os.getenv("HYPOTHESIS_MODEL")
        config_data: Dict[str, Any] = {}

        if paths.config_path.exists():
            try:
                config_data = json.loads(paths.config_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ConfigurationError(
                    f"Configuration file {paths.config_path} contains invalid JSON"
                ) from exc

        data = {
            "api_base_url": env_url or config_data.get("api_base_url"),
            "model_name": env_model or config_data.get("model_name"),
            "timeout": float(os.getenv("HYPOTHESIS_TIMEOUT", config_data.get("timeout", 45))),
            "max_retries": int(os.getenv("HYPOTHESIS_RETRIES", config_data.get("max_retries", 3))),
            "request_backoff": float(
                os.getenv("HYPOTHESIS_BACKOFF", config_data.get("request_backoff", 1.8))
            ),
            "history_limit": int(
                os.getenv("HYPOTHESIS_HISTORY_LIMIT", config_data.get("history_limit", 32))
            ),
            "rate_limit_per_minute": int(
                os.getenv(
                    "HYPOTHESIS_RPM",
                    config_data.get("rate_limit_per_minute", 90),
                )
            ),
            "verify_tls": bool(
                json.loads(
                    os.getenv(
                        "HYPOTHESIS_VERIFY_TLS",
                        json.dumps(config_data.get("verify_tls", True)),
                    )
                )
            ),
        }

        config = cls(**data)
        config.validate()
        return config

    def validate(self) -> None:
        """Validate configuration fields and raise :class:`ConfigurationError`.

        A series of defensive checks ensures invalid data is surfaced early and
        in a descriptive fashion.
        """

        if not self.api_base_url:
            raise ConfigurationError("API base URL is required")
        if not self.model_name:
            raise ConfigurationError("Model name is required")
        if self.timeout <= 0:
            raise ConfigurationError("Timeout must be positive")
        if self.max_retries < 0:
            raise ConfigurationError("Retries cannot be negative")
        if self.request_backoff <= 1.0:
            raise ConfigurationError("Backoff multiplier must exceed 1.0")
        if self.history_limit < 1:
            raise ConfigurationError("History limit must be at least 1")
        if self.rate_limit_per_minute < 1:
            raise ConfigurationError("Rate limit must be positive")


# ---------------------------------------------------------------------------
# Metrics and analytics support
# ---------------------------------------------------------------------------

class MetricsTracker:
    """Tracks performance metrics for API calls.

    Metrics are kept in-memory for simplicity, but the design allows easy
    persistence to disk or export to Prometheus by providing additional hooks.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._durations: List[float] = []
        self._failures: int = 0

    def record(self, duration: float, success: bool) -> None:
        with self._lock:
            if success:
                self._durations.append(duration)
            else:
                self._failures += 1

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            durations = list(self._durations)
            failures = self._failures
        if durations:
            return {
                "count": len(durations),
                "mean": statistics.fmean(durations),
                "p95": statistics.quantiles(durations, n=100)[94],
                "last": durations[-1],
                "failures": failures,
            }
        return {"count": 0, "mean": 0.0, "p95": 0.0, "last": 0.0, "failures": failures}


# ---------------------------------------------------------------------------
# Rate limiting and sanitisation utilities
# ---------------------------------------------------------------------------

class RateLimiter:
    """A naive token bucket implementation suitable for short-lived clients."""

    def __init__(self, tokens_per_minute: int) -> None:
        self.tokens_per_minute = tokens_per_minute
        self._lock = threading.Lock()
        self._tokens = tokens_per_minute
        self._last_refill = time.monotonic()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            tokens_to_add = int(elapsed * self.tokens_per_minute / 60)
            if tokens_to_add:
                self._tokens = min(self.tokens_per_minute, self._tokens + tokens_to_add)
                self._last_refill = now

            if self._tokens <= 0:
                raise RateLimitError(
                    "Rate limit exceeded: too many requests per minute. "
                    "Try again shortly."
                )

            self._tokens -= 1


class InputSanitiser:
    """Ensure user supplied text is suitable for the target language model."""

    _CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

    @classmethod
    def sanitise(cls, text: str) -> str:
        """Remove control characters and trim suspicious content."""

        cleaned = cls._CONTROL_CHAR_PATTERN.sub("", text)
        cleaned = cleaned.strip()
        if not cleaned:
            raise ValueError("Input is empty or contains only invalid characters")
        return cleaned


# ---------------------------------------------------------------------------
# Conversation history persistence
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ChatMessage:
    """Represents a single turn in the conversation."""

    role: str
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: MutableMapping[str, Any]) -> "ChatMessage":
        timestamp_str = payload.get("timestamp")
        timestamp = (
            datetime.fromisoformat(timestamp_str)
            if isinstance(timestamp_str, str)
            else datetime.now(timezone.utc)
        )
        return cls(
            role=str(payload.get("role", "user")),
            content=str(payload.get("content", "")),
            timestamp=timestamp,
            metadata=dict(payload.get("metadata", {})),
        )


class ConversationHistory:
    """Thread-safe in-memory storage for conversation messages."""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._messages: List[ChatMessage] = []
        self._lock = threading.RLock()

    def append(self, message: ChatMessage) -> None:
        with self._lock:
            self._messages.append(message)
            if len(self._messages) > self._limit:
                # remove oldest
                del self._messages[0]

    def export(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dataclasses.asdict(msg) for msg in self._messages]

    def as_payload(self) -> List[Dict[str, str]]:
        with self._lock:
            return [{"role": msg.role, "content": msg.content} for msg in self._messages]

    def load_from(self, messages: Iterable[ChatMessage]) -> None:
        with self._lock:
            self._messages = list(messages)[-self._limit :]

    def __len__(self) -> int:
        with self._lock:
            return len(self._messages)


class PersistentHistoryStore:
    """Persists history to disk so users can resume sessions."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()

    def load(self) -> List[ChatMessage]:
        if not self._path.exists():
            return []
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:  # pragma: no cover - corrupted file
            raise ChatStorageError("History file is corrupted") from exc
        return [ChatMessage.from_dict(item) for item in data]

    def save(self, messages: Iterable[ChatMessage]) -> None:
        serialised = [msg.to_dict() for msg in messages]
        tmp_path = self._path.with_suffix(".tmp")
        with self._lock:
            try:
                tmp_path.write_text(json.dumps(serialised, indent=2), encoding="utf-8")
                tmp_path.replace(self._path)
            except OSError as exc:  # pragma: no cover - disk failure
                raise ChatStorageError("Could not persist conversation history") from exc


# ---------------------------------------------------------------------------
# Prompt enhancement layer
# ---------------------------------------------------------------------------

class PromptEnhancer:
    """A deterministic, lightweight prompt improvement engine.

    The enhancer analyses the user's prompt, referencing prior conversation
    turns for context.  It applies lexical normalisation, optional summarisation
    and adds clarifying instructions when the input seems underspecified.  The
    implementation avoids heavy ML dependencies for portability.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._observed_keywords: Dict[str, int] = {}
        self._max_keywords = 64

    def enhance(self, prompt: str, context: ConversationHistory) -> str:
        prompt = InputSanitiser.sanitise(prompt)
        with self._lock:
            baseline = prompt
            baseline = self._ensure_punctuation(baseline)
            baseline = self._normalise_whitespace(baseline)
            baseline = self._augment_with_context(baseline, context)
            self._remember_keywords(baseline)
            enriched = self._add_quality_instructions(baseline)
            return enriched

    @staticmethod
    def _ensure_punctuation(text: str) -> str:
        if text and text[-1].isalnum():
            return text + "."
        return text

    @staticmethod
    def _normalise_whitespace(text: str) -> str:
        return re.sub(r"\s+", " ", text)

    def _augment_with_context(self, text: str, context: ConversationHistory) -> str:
        """Include references to important entities from prior messages."""

        summary_terms: List[str] = []
        for message in context.export()[::-1]:
            if message["role"] != "assistant":
                continue
            terms = self._extract_keywords(message["content"])
            for term in terms:
                if term not in summary_terms:
                    summary_terms.append(term)
                if len(summary_terms) >= 4:
                    break
            if len(summary_terms) >= 4:
                break

        if summary_terms:
            joined = ", ".join(summary_terms)
            return f"{text} (Consider continuity with: {joined}.)"
        return text

    def _remember_keywords(self, text: str) -> None:
        keywords = self._extract_keywords(text)
        for word in keywords:
            self._observed_keywords[word] = self._observed_keywords.get(word, 0) + 1
        if len(self._observed_keywords) > self._max_keywords:
            # remove least frequent keyword
            least_common = min(self._observed_keywords.items(), key=lambda item: item[1])[0]
            self._observed_keywords.pop(least_common, None)

    def _add_quality_instructions(self, text: str) -> str:
        """If user prompt is short, append clarifying instructions."""

        word_count = len(text.split())
        if word_count < 8:
            return (
                f"{text} Please provide a detailed, structured explanation "
                "including numbered steps and relevant caveats."
            )
        if word_count < 20 and self._observed_keywords:
            emphasised = ", ".join(sorted(self._observed_keywords.keys())[:3])
            return (
                f"{text} Ensure the answer references the following key themes: "
                f"{emphasised}."
            )
        return text

    @staticmethod
    def _extract_keywords(text: str) -> List[str]:
        text = text.lower()
        candidates = re.findall(r"[a-zA-Z]{4,}", text)
        stopwords = {"this", "that", "have", "with", "from", "about", "which"}
        return [word for word in candidates if word not in stopwords]

    def register_feedback(self, prompt: str, response: str) -> None:
        """Record which keywords yielded rich responses to inform future prompts."""

        with self._lock:
            positive_terms = self._extract_keywords(prompt + " " + response)
            for term in positive_terms:
                self._observed_keywords[term] = self._observed_keywords.get(term, 0) + 2
            if len(self._observed_keywords) > self._max_keywords:
                sorted_terms = sorted(
                    self._observed_keywords.items(), key=lambda item: item[1], reverse=True
                )
                self._observed_keywords = dict(sorted_terms[: self._max_keywords])


# ---------------------------------------------------------------------------
# HTTP Client for interacting with the model API
# ---------------------------------------------------------------------------

class SecureHTTPClient:
    """Handles communication with the language model backend."""

    def __init__(self, config: AppConfig, metrics: MetricsTracker, limiter: RateLimiter) -> None:
        self._config = config
        self._metrics = metrics
        self._limiter = limiter
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})
        self._lock = threading.Lock()

    def close(self) -> None:
        self._session.close()

    def chat_completion(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """Send a chat completion request and return the assistant's reply."""

        payload = {
            "model": self._config.model_name,
            "messages": messages,
            "temperature": temperature,
        }

        retries = 0
        backoff = self._config.request_backoff
        while True:
            self._limiter.acquire()
            start = time.perf_counter()
            try:
                response = self._session.post(
                    self._config.api_base_url,
                    json=payload,
                    timeout=self._config.timeout,
                    verify=self._config.verify_tls,
                )
                response.raise_for_status()
                data = response.json()
                reply = self._extract_content(data)
                duration = time.perf_counter() - start
                self._metrics.record(duration, success=True)
                LOGGER.debug("Received response in %.2fs", duration)
                return reply
            except (requests.Timeout, requests.ConnectionError) as exc:
                self._metrics.record(time.perf_counter() - start, success=False)
                if retries >= self._config.max_retries:
                    raise APIError("Network failure contacting language model") from exc
                retries += 1
                sleep_for = backoff ** retries + random.uniform(0, 0.2)
                LOGGER.warning(
                    "Transient network failure (%s). Retrying in %.2fs (attempt %d/%d)",
                    exc,
                    sleep_for,
                    retries,
                    self._config.max_retries,
                )
                time.sleep(sleep_for)
            except requests.HTTPError as exc:
                self._metrics.record(time.perf_counter() - start, success=False)
                status = exc.response.status_code if isinstance(exc.response, Response) else None
                raise APIError(f"Language model returned HTTP {status}") from exc
            except ValueError as exc:
                self._metrics.record(time.perf_counter() - start, success=False)
                raise ResponseFormatError("Malformed JSON received from backend") from exc

    @staticmethod
    def _extract_content(data: Dict[str, Any]) -> str:
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, AttributeError) as exc:
            raise ResponseFormatError("Unexpected response structure from backend") from exc


# ---------------------------------------------------------------------------
# Chat orchestration layer
# ---------------------------------------------------------------------------

class ChatOrchestrator:
    """Encapsulates the end-to-end logic for running a chat session."""

    def __init__(
        self,
        config: AppConfig,
        history: ConversationHistory,
        store: PersistentHistoryStore,
        enhancer: PromptEnhancer,
        http_client: SecureHTTPClient,
        metrics: MetricsTracker,
    ) -> None:
        self._config = config
        self._history = history
        self._store = store
        self._enhancer = enhancer
        self._client = http_client
        self._metrics = metrics
        self._lock = threading.RLock()
        self._load_history()

    def _load_history(self) -> None:
        try:
            messages = self._store.load()
        except ChatStorageError as exc:
            LOGGER.error("Failed to load history: %s", exc)
            return
        self._history.load_from(messages)

    def submit_user_message(self, user_input: str) -> str:
        """Process user input, query the backend and persist the outcome."""

        with self._lock:
            enhanced_prompt = self._enhancer.enhance(user_input, self._history)
            messages_payload = self._history.as_payload()
            messages_payload.append({"role": "user", "content": enhanced_prompt})

            LOGGER.info("Dispatching prompt with %d prior turns", len(messages_payload))
            response_text = self._client.chat_completion(messages_payload)

            user_message = ChatMessage(role="user", content=user_input)
            assistant_message = ChatMessage(role="assistant", content=response_text)
            self._history.append(user_message)
            self._history.append(assistant_message)
            try:
                self._store.save(self._history.export())
            except ChatStorageError as exc:
                LOGGER.warning("Could not persist conversation history: %s", exc)

            self._enhancer.register_feedback(user_input, response_text)
            metrics = self._metrics.snapshot()
            LOGGER.debug("Metrics snapshot: %s", metrics)
            return response_text

    def export_history(self) -> List[Dict[str, Any]]:
        return self._history.export()

    def shutdown(self) -> None:
        LOGGER.info("Shutting down orchestrator")
        self._client.close()


# ---------------------------------------------------------------------------
# Tkinter based GUI
# ---------------------------------------------------------------------------

class ChatGUI:
    """Tkinter based desktop client for Hypothesis-link."""

    def __init__(self, orchestrator: ChatOrchestrator, metrics: MetricsTracker) -> None:
        self._orchestrator = orchestrator
        self._metrics = metrics
        self._root = tk.Tk()
        self._root.title("Hypothesis-link v2 – Enterprise Edition")
        self._root.geometry("900x620")
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._chat_display = scrolledtext.ScrolledText(
            self._root,
            wrap=tk.WORD,
            font=("Consolas", 11),
            state=tk.DISABLED,
        )
        self._chat_display.pack(padx=12, pady=12, fill=tk.BOTH, expand=True)

        self._input_frame = tk.Frame(self._root)
        self._input_frame.pack(fill=tk.X, padx=12, pady=(0, 12))

        self._user_entry = tk.Entry(self._input_frame, font=("Consolas", 11))
        self._user_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._user_entry.bind("<Return>", self._send_message_event)

        self._send_button = tk.Button(
            self._input_frame, text="Send", command=self._send_message_direct
        )
        self._send_button.pack(side=tk.LEFT, padx=(8, 0))

        self._clear_button = tk.Button(
            self._input_frame, text="Clear", command=self._clear_history
        )
        self._clear_button.pack(side=tk.LEFT, padx=(8, 0))

        self._status_var = tk.StringVar(value="Ready")
        self._status_label = tk.Label(
            self._root,
            textvariable=self._status_var,
            anchor=tk.W,
            relief=tk.SUNKEN,
        )
        self._status_label.pack(fill=tk.X, padx=12, pady=(0, 12))

        self._work_queue: "queue.Queue[Tuple[str, threading.Event]]" = queue.Queue()
        self._response_thread = threading.Thread(
            target=self._response_worker,
            name="ChatResponseWorker",
            daemon=True,
        )
        self._response_thread.start()
        self._append_initial_history()

    def _append_initial_history(self) -> None:
        history = self._orchestrator.export_history()
        if not history:
            self._append_to_display(
                "Session initialised. Messages will appear here.\n",
                tag="info",
            )
            return
        for entry in history:
            role = entry.get("role", "user").capitalize()
            content = entry.get("content", "")
            self._append_to_display(f"{role}: {content}\n", tag="history")

    def _send_message_event(self, event: tk.Event[Any]) -> None:  # pragma: no cover - GUI
        self._send_message_direct()

    def _send_message_direct(self) -> None:
        raw_input = self._user_entry.get()
        try:
            sanitised = InputSanitiser.sanitise(raw_input)
        except ValueError as exc:
            messagebox.showwarning("Invalid input", str(exc))
            return

        self._user_entry.delete(0, tk.END)
        self._append_to_display(f"You: {sanitised}\n", tag="user")

        completion_event = threading.Event()
        self._work_queue.put((sanitised, completion_event))
        self._send_button.configure(state=tk.DISABLED)
        self._status_var.set("Waiting for response...")

    def _response_worker(self) -> None:
        while True:
            prompt, completion_event = self._work_queue.get()
            try:
                response = self._orchestrator.submit_user_message(prompt)
                self._root.after(
                    0,
                    lambda resp=response: self._append_to_display(
                        f"Assistant: {resp}\n\n", tag="assistant"
                    ),
                )
            except RateLimitError as exc:
                self._root.after(0, lambda msg=str(exc): messagebox.showerror("Rate limit", msg))
            except APIError as exc:
                LOGGER.error("API failure: %s", exc)
                self._root.after(
                    0,
                    lambda msg=str(exc): messagebox.showerror(
                        "API Error", f"The language model could not be reached: {msg}"
                    ),
                )
            finally:
                self._root.after(0, self._on_response_complete)
                completion_event.set()
                self._work_queue.task_done()

    def _on_response_complete(self) -> None:
        snapshot = self._metrics.snapshot()
        status = (
            f"Ready | Responses: {snapshot['count']} | Failures: {snapshot['failures']} "
            f"| Mean latency: {snapshot['mean']:.2f}s"
        )
        self._status_var.set(status)
        self._send_button.configure(state=tk.NORMAL)

    def _append_to_display(self, text: str, tag: str) -> None:
        self._chat_display.configure(state=tk.NORMAL)
        self._chat_display.insert(tk.END, text, tag)
        self._chat_display.tag_configure("user", foreground="#1f77b4")
        self._chat_display.tag_configure("assistant", foreground="#2ca02c")
        self._chat_display.tag_configure("info", foreground="#7f7f7f")
        self._chat_display.tag_configure("history", foreground="#9467bd")
        self._chat_display.configure(state=tk.DISABLED)
        self._chat_display.yview(tk.END)

    def _clear_history(self) -> None:
        if messagebox.askyesno("Confirm", "Clear the conversation history from this session?"):
            self._chat_display.configure(state=tk.NORMAL)
            self._chat_display.delete("1.0", tk.END)
            self._chat_display.configure(state=tk.DISABLED)
            self._status_var.set("History cleared – Ready")

    def _on_close(self) -> None:
        if messagebox.askokcancel("Quit", "Do you really want to exit Hypothesis-link?"):
            self._orchestrator.shutdown()
            self._root.destroy()

    def run(self) -> None:  # pragma: no cover - GUI loop
        LOGGER.info("Starting GUI loop")
        self._root.mainloop()


# ---------------------------------------------------------------------------
# Application bootstrap
# ---------------------------------------------------------------------------

def build_application() -> ChatGUI:
    """Construct the application using dependency injection.

    This function wires together all components in a predictable order.  Doing
    so makes the codebase easier to unit test by allowing individual pieces to
    be swapped with fakes or mocks.
    """

    paths = AppPaths()
    config = AppConfig.from_env(paths)
    metrics = MetricsTracker()
    limiter = RateLimiter(config.rate_limit_per_minute)
    history = ConversationHistory(config.history_limit)
    store = PersistentHistoryStore(paths.history_path)
    enhancer = PromptEnhancer()
    client = SecureHTTPClient(config, metrics, limiter)
    orchestrator = ChatOrchestrator(config, history, store, enhancer, client, metrics)
    gui = ChatGUI(orchestrator, metrics)
    return gui


def main() -> None:  # pragma: no cover - entry point
    try:
        app = build_application()
    except ConfigurationError as exc:
        CONSOLE.print(f"[bold red]Configuration error:[/bold red] {exc}")
        return
    except Exception as exc:  # pragma: no cover - defensive catch-all
        LOGGER.exception("Fatal error during application startup")
        CONSOLE.print(f"[bold red]Unexpected error:[/bold red] {exc}")
        return

    app.run()


if __name__ == "__main__":  # pragma: no cover - module executed directly
    main()
