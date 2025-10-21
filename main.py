"""Hypothesis-link v2
=====================

This module provides a refreshed implementation of the Hypothesis-link desktop
client.  The goal is to deliver an enterprise-grade code base that prizes
robustness, correctness, maintainability, and clear observability hooks.

The guiding principles are summarised by the symbolic memory equation supplied
by the user:

    ∂M/∂t = D∇²M − λ(M − M³) + α∫K(x, y)σ(M(y, t))dy − μ(t)M + η(x, t)

Diffusion keeps meanings related (continuity), reaction stabilises identity
(coherence), kernel integration reactivates relations (recursion), the Lagrange
term bounds energy (integrity), and noise fosters creativity (freedom).  We
retain the equation as a reminder to balance resilience, security, and
adaptability in every subsystem.

The code targets Python 3.13 and avoids deprecated interfaces so that static
analysis and modern tooling can work without friction.  The implementation is
liberally documented, making it straightforward to audit or extend.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import logging
import os
import queue
import random
import re
import statistics
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterable, List, MutableMapping, Optional, Tuple, Union
from urllib.parse import urljoin, urlparse

if importlib.util.find_spec("requests") is not None:
    import requests  # type: ignore[assignment]
    from requests import Response
    REQUESTS_AVAILABLE = True
else:  # pragma: no cover - offline fallback when requests is unavailable
    REQUESTS_AVAILABLE = False

    class Response:  # type: ignore[override]
        """Lightweight stand-in for :class:`requests.Response`."""

        def __init__(self, status_code: int = 0) -> None:
            self.status_code = status_code
            self.ok = 200 <= status_code < 300


    class _RequestsRequestException(RuntimeError):
        """Fallback hierarchy mirroring requests' exception structure."""


    class _RequestsHTTPError(_RequestsRequestException):
        def __init__(self, message: str = "", response: Optional[Response] = None) -> None:
            super().__init__(message)
            self.response = response


    class _RequestsTimeout(_RequestsRequestException):
        pass


    class _RequestsConnectionError(_RequestsRequestException):
        pass


    class _OfflineNetworkUnavailable(RuntimeError):
        """Raised when attempting network IO without the requests dependency."""


    class _OfflineRequestsSession:
        def __init__(self) -> None:
            self.headers: Dict[str, str] = {}

        def close(self) -> None:
            return None

        def post(self, *args: Any, **kwargs: Any) -> "Response":
            raise _OfflineNetworkUnavailable(
                "The 'requests' package is required for network access."
            )

        def get(self, *args: Any, **kwargs: Any) -> "Response":
            raise _OfflineNetworkUnavailable(
                "The 'requests' package is required for network access."
            )


    class _OfflineRequestsModule:
        Session = _OfflineRequestsSession
        Timeout = _RequestsTimeout
        ConnectionError = _RequestsConnectionError
        HTTPError = _RequestsHTTPError
        RequestException = _RequestsRequestException


    requests = _OfflineRequestsModule()  # type: ignore[assignment]


if importlib.util.find_spec("rich") is not None:
    from rich.console import Console
else:  # pragma: no cover - graceful degradation when rich is unavailable

    class Console:  # type: ignore[override]
        """Minimal substitute for :class:`rich.console.Console`."""

        def print(self, message: str) -> None:
            cleaned = re.sub(r"\[/?[a-zA-Z0-9_\s-]+\]", "", message)
            print(cleaned)

try:
    import tkinter as tk
    from tkinter import messagebox, scrolledtext, ttk
except ModuleNotFoundError as exc:  # pragma: no cover - depends on environment
    raise RuntimeError(
        "Tkinter is required to run the Hypothesis-link GUI. "
        "Install the Python Tk bindings for your platform."
    ) from exc


# ---------------------------------------------------------------------------
# Logging infrastructure
# ---------------------------------------------------------------------------


def _build_logger() -> logging.Logger:
    """Configure a module-level logger with safe defaults."""

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
    """Generic API failure encompassing HTTP or schema-level issues."""


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
    offline_fallback_enabled: bool = True

    _DEFAULT_BASE_URL: ClassVar[str] = "http://127.0.0.1:1234"
    _DEFAULT_MODEL: ClassVar[str] = "liquid/lfm2-1.2b"

    @classmethod
    def from_env(cls, paths: AppPaths) -> "AppConfig":
        """Load configuration from environment variables or disk."""

        config_data: Dict[str, Any] = {}
        if paths.config_path.exists():
            try:
                config_data = json.loads(paths.config_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ConfigurationError(
                    f"Configuration file {paths.config_path} contains invalid JSON"
                ) from exc

        def _read_str(env_name: str, key: str, default: str) -> str:
            env_value = os.getenv(env_name)
            if env_value is not None and env_value.strip():
                return env_value.strip()
            raw_value = config_data.get(key)
            if raw_value is None:
                return default
            return str(raw_value)

        def _read_float(env_name: str, key: str, default: float) -> float:
            env_value = os.getenv(env_name)
            if env_value is not None:
                try:
                    return float(env_value)
                except ValueError as exc:
                    raise ConfigurationError(
                        f"Environment variable {env_name} must be a number"
                    ) from exc
            raw_value = config_data.get(key, default)
            try:
                return float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ConfigurationError(f"Configuration field {key} must be a number") from exc

        def _read_int(env_name: str, key: str, default: int) -> int:
            value = _read_float(env_name, key, float(default))
            if value < 0:
                raise ConfigurationError(f"Configuration field {key} cannot be negative")
            return int(value)

        def _read_bool(env_name: str, key: str, default: bool) -> bool:
            env_value = os.getenv(env_name)
            source = env_value if env_value is not None else config_data.get(key, default)
            if isinstance(source, bool):
                return source
            if isinstance(source, (int, float)):
                return bool(source)
            if source is None:
                return default
            text = str(source).strip().lower()
            if text in {"1", "true", "yes", "on"}:
                return True
            if text in {"0", "false", "no", "off"}:
                return False
            raise ConfigurationError(f"Configuration field {key} expects a boolean value")

        data = {
            "api_base_url": _read_str(
                "HYPOTHESIS_API_URL", "api_base_url", cls._DEFAULT_BASE_URL
            ),
            "model_name": _read_str(
                "HYPOTHESIS_MODEL", "model_name", cls._DEFAULT_MODEL
            ),
            "timeout": _read_float("HYPOTHESIS_TIMEOUT", "timeout", 45.0),
            "max_retries": _read_int("HYPOTHESIS_RETRIES", "max_retries", 3),
            "request_backoff": _read_float("HYPOTHESIS_BACKOFF", "request_backoff", 1.8),
            "history_limit": _read_int("HYPOTHESIS_HISTORY_LIMIT", "history_limit", 32),
            "rate_limit_per_minute": _read_int("HYPOTHESIS_RPM", "rate_limit_per_minute", 90),
            "verify_tls": _read_bool("HYPOTHESIS_VERIFY_TLS", "verify_tls", True),
            "offline_fallback_enabled": _read_bool(
                "HYPOTHESIS_OFFLINE_FALLBACK",
                "offline_fallback_enabled",
                True,
            ),
        }

        config = cls(**data)
        config.validate()
        return config

    def validate(self) -> None:
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
        LMStudioEndpoints(self.api_base_url)
        if not isinstance(self.offline_fallback_enabled, bool):
            raise ConfigurationError("Offline fallback flag must be boolean")


@dataclass(slots=True)
class LMStudioEndpoints:
    """Utility helper that exposes the canonical LM Studio REST endpoints."""

    base_url: str

    _PATHS: ClassVar[Dict[str, str]] = {
        "chat_completions": "/v1/chat/completions",
        "completions": "/v1/completions",
        "embeddings": "/v1/embeddings",
        "models": "/v1/models",
        "moderations": "/v1/moderations",
        "images_generations": "/v1/images/generations",
        "audio_transcriptions": "/v1/audio/transcriptions",
        "audio_speech": "/v1/audio/speech",
        "tokenize": "/v1/tokenize",
        "detokenize": "/v1/detokenize",
        "rerank": "/v1/rerank",
        "health": "/v1/health",
    }

    def __post_init__(self) -> None:
        if not self.base_url:
            raise ConfigurationError("API base URL is required to construct endpoints")

        parsed = urlparse(self.base_url)
        if not parsed.scheme or not parsed.netloc:
            raise ConfigurationError("API base URL must include a valid scheme and host")
        if parsed.scheme not in {"http", "https"}:
            raise ConfigurationError("API base URL must use HTTP or HTTPS")
        if parsed.query or parsed.fragment:
            raise ConfigurationError("API base URL must not contain query parameters")

        path = parsed.path.rstrip("/")
        normalised = f"{parsed.scheme}://{parsed.netloc}{path}"
        self.base_url = normalised.rstrip("/")

    def url_for(self, endpoint: str) -> str:
        try:
            path = self._PATHS[endpoint]
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise ConfigurationError(f"Unknown LM Studio endpoint: {endpoint}") from exc
        return urljoin(f"{self.base_url}/", path)

    def as_dict(self) -> Dict[str, str]:
        return {name: self.url_for(name) for name in self._PATHS}

    def __getattr__(self, item: str) -> str:
        """Allow attribute-style access to the canonical endpoint URLs."""

        try:
            return self.url_for(item)
        except ConfigurationError as exc:
            raise AttributeError(item) from exc

    def __dir__(self) -> List[str]:
        return sorted({*super().__dir__(), *self._PATHS.keys()})


# ---------------------------------------------------------------------------
# Metrics and analytics support
# ---------------------------------------------------------------------------


class MetricsTracker:
    """Tracks performance metrics for API calls in a thread-safe manner."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._durations: List[float] = []
        self._failures: int = 0
        self._max_samples = 512

    def record(self, duration: float, success: bool) -> None:
        with self._lock:
            if success:
                self._durations.append(duration)
                if len(self._durations) > self._max_samples:
                    self._durations = self._durations[-self._max_samples :]
            else:
                self._failures += 1

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            durations = list(self._durations)
            failures = self._failures

        if not durations:
            return {"count": 0, "mean": 0.0, "p95": 0.0, "last": 0.0, "failures": failures}

        durations.sort()
        count = len(durations)
        mean = statistics.fmean(durations)
        last = durations[-1]
        p95_index = max(0, min(count - 1, int(round(0.95 * (count - 1)))))
        p95 = durations[p95_index]
        return {"count": count, "mean": mean, "p95": p95, "last": last, "failures": failures}


# ---------------------------------------------------------------------------
# Rate limiting and sanitisation utilities
# ---------------------------------------------------------------------------


class RateLimiter:
    """A token-bucket based rate limiter with cooperative waiting."""

    def __init__(self, tokens_per_minute: int) -> None:
        if tokens_per_minute < 1:
            raise ConfigurationError("tokens_per_minute must be positive")
        self.tokens_per_minute = tokens_per_minute
        self._lock = threading.Lock()
        self._tokens = tokens_per_minute
        self._last_refill = time.monotonic()

    def _refill(self, now: float) -> None:
        elapsed = now - self._last_refill
        if elapsed <= 0:
            return
        tokens_to_add = int(elapsed * self.tokens_per_minute / 60)
        if tokens_to_add:
            self._tokens = min(self.tokens_per_minute, self._tokens + tokens_to_add)
            self._last_refill = now

    def acquire(self) -> None:
        deadline = time.monotonic() + 60
        while True:
            with self._lock:
                now = time.monotonic()
                self._refill(now)
                if self._tokens > 0:
                    self._tokens -= 1
                    return
                wait_time = max(0.01, 60 / self.tokens_per_minute)
            if time.monotonic() > deadline:
                raise RateLimitError("Rate limit exceeded: please retry later.")
            time.sleep(min(wait_time, 1.0))


class InputSanitiser:
    """Ensure user supplied text is suitable for the target language model."""

    _CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

    @classmethod
    def sanitise(cls, text: str) -> str:
        cleaned = cls._CONTROL_CHAR_PATTERN.sub("", text)
        cleaned = cleaned.strip()
        if not cleaned:
            raise ValueError("Input is empty or contains only invalid characters")
        return cleaned


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

    def clear(self) -> None:
        with self._lock:
            self._messages.clear()

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

    def save(self, messages: Iterable[Union[ChatMessage, MutableMapping[str, Any]]]) -> None:
        serialised: List[Dict[str, Any]] = []
        for item in messages:
            if isinstance(item, ChatMessage):
                serialised.append(item.to_dict())
            elif isinstance(item, MutableMapping):
                snapshot = dict(item)
                timestamp = snapshot.get("timestamp")
                if isinstance(timestamp, datetime):
                    snapshot["timestamp"] = timestamp.isoformat()
                serialised.append(snapshot)
            else:  # pragma: no cover - defensive guard
                raise ChatStorageError(
                    f"Unsupported history entry type: {type(item)!r}"
                )
        tmp_path = self._path.with_suffix(".tmp")
        with self._lock:
            try:
                tmp_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path.write_text(json.dumps(serialised, indent=2), encoding="utf-8")
                tmp_path.replace(self._path)
            except OSError as exc:  # pragma: no cover - disk failure
                raise ChatStorageError("Could not persist conversation history") from exc
# ---------------------------------------------------------------------------
# Prompt enhancement layer
# ---------------------------------------------------------------------------


class PromptEnhancer:
    """Deterministically refine prompts while preserving user intent."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_prompt: Optional[str] = None
        self._last_response: Optional[str] = None
        self._observed_keywords: Dict[str, int] = {}
        self._max_keywords: int = 32

    def enhance(self, prompt: str, context: ConversationHistory) -> str:  # noqa: D401
        """Return a sanitised, context-aware prompt."""

        sanitised = InputSanitiser.sanitise(prompt)
        contextualised = self._augment_with_context(sanitised, context)
        enriched = self._add_quality_instructions(contextualised)
        with self._lock:
            self._last_prompt = enriched
            self._last_response = None
        self._remember_keywords(sanitised)
        return enriched

    def last_exchange(self) -> Optional[Tuple[str, Optional[str]]]:
        """Expose the most recent prompt/response pair for diagnostics."""

        with self._lock:
            if self._last_prompt is None:
                return None
            return (self._last_prompt, self._last_response)
    def _augment_with_context(self, text: str, context: ConversationHistory) -> str:
        summary_terms: List[str] = []
        for message in reversed(context.export()):
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
        if not keywords:
            return
        with self._lock:
            for word in keywords:
                self._observed_keywords[word] = self._observed_keywords.get(word, 0) + 1
            if len(self._observed_keywords) > self._max_keywords:
                least_common = min(
                    self._observed_keywords.items(), key=lambda item: item[1]
                )[0]
                self._observed_keywords.pop(least_common, None)

    def _add_quality_instructions(self, text: str) -> str:
        word_count = len(text.split())
        if word_count < 8:
            return (
                f"{text} Please provide a detailed, structured explanation "
                "including numbered steps and relevant caveats."
            )
        with self._lock:
            observed_snapshot = dict(self._observed_keywords)
        if word_count < 20 and observed_snapshot:
            emphasised = ", ".join(sorted(observed_snapshot.keys())[:3])
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
        combined_keywords = self._extract_keywords(f"{prompt} {response}")
        sanitised_prompt = InputSanitiser.sanitise(prompt)
        with self._lock:
            if self._last_prompt is None:
                self._last_prompt = sanitised_prompt
            self._last_response = response
            for term in combined_keywords:
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

    def __init__(
        self,
        config: AppConfig,
        metrics: MetricsTracker,
        limiter: RateLimiter,
        session: Optional[Any] = None,
    ) -> None:
        self._config = config
        self._metrics = metrics
        self._limiter = limiter
        self._session = session or requests.Session()
        self._custom_session_provided = session is not None
        if not hasattr(self._session, "headers"):
            self._session.headers = {}
        try:
            self._session.headers.update({"Content-Type": "application/json"})
        except AttributeError:  # pragma: no cover - highly defensive
            self._session.headers["Content-Type"] = "application/json"
        self._lock = threading.Lock()
        self._offline_reason: Optional[str] = None
        self._endpoints = LMStudioEndpoints(config.api_base_url)
        if (
            self._config.offline_fallback_enabled
            and not self._custom_session_provided
        ):
            self._auto_enable_offline_mode()

    def close(self) -> None:
        self._session.close()

    def using_offline_simulator(self) -> bool:
        """Indicate whether requests are served by the offline simulator."""

        return isinstance(self._session, OfflineSession)

    @property
    def offline_reason(self) -> Optional[str]:
        """Return the reason the client switched to offline mode, if any."""

        return self._offline_reason

    def _auto_enable_offline_mode(self) -> None:
        """Run a health probe and enable the offline simulator when required."""

        if self.using_offline_simulator():
            return

        health_url = self._endpoints.url_for("health")
        try:
            response = self._session.get(
                health_url,
                timeout=min(self._config.timeout, 3.0),
                verify=self._config.verify_tls,
            )
        except (requests.Timeout, requests.ConnectionError) as exc:
            self._switch_to_offline_session(f"Health probe failed: {exc}")
            return
        except Exception as exc:  # pragma: no cover - defensive guard
            if not REQUESTS_AVAILABLE:
                self._switch_to_offline_session(
                    "The 'requests' package is unavailable; using offline simulator.",
                )
            else:
                LOGGER.debug("Health probe raised unexpected exception: %s", exc)
            return

        if getattr(response, "ok", False):
            return

        status = getattr(response, "status_code", None)
        if status is not None and status >= 500:
            self._switch_to_offline_session(
                f"Health probe returned HTTP {status}; enabling offline simulator."
            )

    def _switch_to_offline_session(self, reason: str) -> None:
        """Swap the active session with the offline simulator implementation."""

        with self._lock:
            if self.using_offline_simulator():
                if self._offline_reason is None:
                    self._offline_reason = reason
                return

            LOGGER.warning("Falling back to offline simulator: %s", reason)
            offline_session = OfflineSession()
            offline_session.headers.update(getattr(self._session, "headers", {}))
            offline_session.configure_models(
                [self._config.model_name, "offline-simulator"]
            )
            self._session = offline_session
            self._offline_reason = reason
            self._custom_session_provided = False
            if "offline" not in self._config.api_base_url:
                self._config.api_base_url = "http://offline.local"
                self._endpoints = LMStudioEndpoints(self._config.api_base_url)

    def _maybe_failover_to_offline(self, reason: str) -> bool:
        """Attempt to switch to offline mode, returning True on success."""

        if (
            not self._config.offline_fallback_enabled
            or self.using_offline_simulator()
            or self._custom_session_provided
        ):
            return False

        self._switch_to_offline_session(reason)
        return True

    def chat_completion(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
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
                    self._endpoints.url_for("chat_completions"),
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
                duration = time.perf_counter() - start
                self._metrics.record(duration, success=False)
                reason = f"Network failure contacting language model: {exc}"
                if self._maybe_failover_to_offline(reason):
                    retries = 0
                    continue
                if retries >= self._config.max_retries:
                    raise APIError("Network failure contacting language model") from exc
                retries += 1
                sleep_for = backoff**retries + random.uniform(0, 0.2)
                LOGGER.warning(
                    "Transient network failure (%s). Retrying in %.2fs (attempt %d/%d)",
                    exc,
                    sleep_for,
                    retries,
                    self._config.max_retries,
                )
                time.sleep(min(sleep_for, 30.0))
            except requests.HTTPError as exc:
                self._metrics.record(time.perf_counter() - start, success=False)
                status = exc.response.status_code if isinstance(exc.response, Response) else None
                detail = self._summarise_http_error(exc.response)
                if status in {400, 404} and self._should_retry_model_refresh(detail):
                    retries += 1
                    if retries > self._config.max_retries:
                        raise APIError(detail or f"Language model returned HTTP {status}") from exc
                    LOGGER.warning(
                        "Backend rejected request (%s). Attempting model refresh (%d/%d).",
                        status,
                        retries,
                        self._config.max_retries,
                    )
                    if not self._refresh_active_model():
                        raise APIError(detail or f"Language model returned HTTP {status}") from exc
                    time.sleep(min(backoff**retries, 5.0))
                    continue
                raise APIError(detail or f"Language model returned HTTP {status}") from exc
            except ValueError as exc:
                self._metrics.record(time.perf_counter() - start, success=False)
                raise ResponseFormatError("Malformed JSON received from backend") from exc

    def list_models(self) -> List[str]:
        self._limiter.acquire()
        start = time.perf_counter()
        try:
            response = self._session.get(
                self._endpoints.url_for("models"),
                timeout=self._config.timeout,
                verify=self._config.verify_tls,
            )
            response.raise_for_status()
            payload = response.json()
            models = self._coerce_models(payload)
            self._metrics.record(time.perf_counter() - start, success=True)
            return models
        except (requests.Timeout, requests.ConnectionError) as exc:
            self._metrics.record(time.perf_counter() - start, success=False)
            if self._maybe_failover_to_offline(
                f"Model listing failed due to network error: {exc}"
            ):
                return self.list_models()
            raise APIError("Network failure while listing models") from exc
        except requests.HTTPError as exc:
            self._metrics.record(time.perf_counter() - start, success=False)
            status = exc.response.status_code if isinstance(exc.response, Response) else None
            raise APIError(f"Model listing returned HTTP {status}") from exc
        except ValueError as exc:
            self._metrics.record(time.perf_counter() - start, success=False)
            raise ResponseFormatError("Malformed JSON payload from model listing") from exc

    def available_endpoints(self) -> Dict[str, str]:
        return self._endpoints.as_dict()

    def check_health(self) -> bool:
        try:
            response = self._session.get(
                self._endpoints.url_for("health"),
                timeout=min(self._config.timeout, 5.0),
                verify=self._config.verify_tls,
            )
            return response.ok
        except requests.RequestException:
            return False

    def _refresh_active_model(self) -> bool:
        """Refresh the cached model list and select a valid model if needed."""

        try:
            models = self.list_models()
        except APIError as exc:
            LOGGER.warning("Unable to refresh models after backend rejection: %s", exc)
            return False

        if not models:
            LOGGER.warning("Model refresh succeeded but returned no candidates")
            return False

        if self._config.model_name in models:
            return True

        selected = models[0]
        LOGGER.info(
            "Switching to available model '%s' after backend rejection of '%s'",
            selected,
            self._config.model_name,
        )
        self._config.model_name = selected
        return True

    @staticmethod
    def _should_retry_model_refresh(detail: Optional[str]) -> bool:
        if not detail:
            return False
        text = detail.lower()
        if "model" not in text:
            return False
        cues = ["not found", "load a model", "no model", "missing model", "model is not"]
        return any(cue in text for cue in cues)

    @staticmethod
    def _summarise_http_error(response: Optional[Response]) -> Optional[str]:
        if response is None:
            return None

        snippets: List[str] = []
        try:
            payload = response.json()
        except ValueError:
            payload = None

        def _normalise(value: Any) -> None:
            if isinstance(value, str):
                cleaned = value.strip()
                if cleaned:
                    snippets.append(cleaned)
            elif isinstance(value, MutableMapping):
                for candidate in value.values():
                    _normalise(candidate)
            elif isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
                for item in value:
                    _normalise(item)

        if payload is not None:
            _normalise(payload)

        if not snippets:
            text_content = getattr(response, "text", "")
            if text_content:
                snippets.append(str(text_content).strip())

        if not snippets:
            return None

        merged = "; ".join(dict.fromkeys(snippets))
        if len(merged) > 240:
            merged = merged[:237] + "…"
        return merged or None

    @staticmethod
    def _coerce_models(payload: Any) -> List[str]:
        models: List[str] = []
        if isinstance(payload, MutableMapping):
            data = payload.get("data")
            if isinstance(data, Iterable):
                for entry in data:
                    if isinstance(entry, MutableMapping):
                        candidate = entry.get("id") or entry.get("name")
                        if candidate:
                            models.append(str(candidate))
            elif "id" in payload:
                models.append(str(payload["id"]))

        seen = set()
        unique_models = []
        for model in models:
            if model not in seen:
                unique_models.append(model)
                seen.add(model)

        return unique_models

    @staticmethod
    def _extract_content(data: Dict[str, Any]) -> str:
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, AttributeError) as exc:
            raise ResponseFormatError("Unexpected response structure from backend") from exc


class OfflineResponse(Response):
    """Response surrogate used by the offline model simulator."""

    def __init__(self, status_code: int, payload: Any) -> None:
        super().__init__()
        self._payload = payload
        self.status_code = status_code
        self.ok = 200 <= status_code < 300

    def json(self) -> Any:
        return json.loads(json.dumps(self._payload))

    def raise_for_status(self) -> None:
        if not self.ok:
            raise requests.HTTPError(
                f"Offline simulator returned HTTP {self.status_code}", response=self
            )

    @property
    def text(self) -> str:
        try:
            return json.dumps(self._payload)
        except TypeError:
            return str(self._payload)


class OfflineModelSimulator:
    """In-memory LM Studio surrogate for offline tests."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._models = ["offline-simulator"]
        self._history: List[Dict[str, str]] = []

    def dispatch(self, method: str, url: str, payload: Optional[Dict[str, Any]]) -> OfflineResponse:
        path = urlparse(url).path
        if method == "POST" and path.endswith("/chat/completions"):
            return self._handle_chat_completion(payload or {})
        if method == "GET" and path.endswith("/models"):
            return OfflineResponse(200, {"data": [{"id": model} for model in self._models]})
        if method == "GET" and path.endswith("/health"):
            return OfflineResponse(200, {"status": "ok"})
        return OfflineResponse(404, {"error": "Unsupported offline endpoint", "path": path})

    def _handle_chat_completion(self, payload: Dict[str, Any]) -> OfflineResponse:
        messages = payload.get("messages") or []
        if not messages:
            return OfflineResponse(400, {"error": "messages payload is required"})
        model = str(payload.get("model", "")).strip()
        with self._lock:
            available_models = set(self._models)
        if model and model not in available_models:
            return OfflineResponse(
                400,
                {
                    "error": {
                        "message": (
                            f"Model '{model}' not found. Please load a model or choose one of: "
                            + ", ".join(sorted(available_models))
                        )
                    }
                },
            )
        prompt = str(messages[-1].get("content", ""))
        with self._lock:
            self._history.append({"role": "user", "content": prompt})
            response_text = f"offline-response::{prompt}"
            self._history.append({"role": "assistant", "content": response_text})
        return OfflineResponse(
            200,
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": response_text,
                        }
                    }
                ]
            },
        )

    @property
    def history(self) -> List[Dict[str, str]]:
        with self._lock:
            return list(self._history)

    def set_models(self, models: Iterable[str]) -> None:
        with self._lock:
            cleaned = [str(model).strip() for model in models if str(model).strip()]
            self._models = cleaned or ["offline-simulator"]


class OfflineSession:
    """requests.Session compatible stub for offline diagnostics."""

    def __init__(self) -> None:
        self.headers: Dict[str, str] = {}
        self._simulator = OfflineModelSimulator()

    def close(self) -> None:
        return None

    def post(self, url: str, json: Dict[str, Any], timeout: float, verify: bool) -> OfflineResponse:
        return self._simulator.dispatch("POST", url, json)

    def get(self, url: str, timeout: float, verify: bool) -> OfflineResponse:
        return self._simulator.dispatch("GET", url, None)

    def configure_models(self, models: Iterable[str]) -> None:
        self._simulator.set_models(models)

    @property
    def simulator(self) -> OfflineModelSimulator:
        return self._simulator

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
        self._synchronise_active_model()
        self._announce_offline_mode()

    def _load_history(self) -> None:
        try:
            messages = self._store.load()
        except ChatStorageError as exc:
            LOGGER.error("Failed to load history: %s", exc)
            return
        self._history.load_from(messages)

    def _synchronise_active_model(self) -> None:
        try:
            models = self._client.list_models()
        except APIError as exc:
            LOGGER.warning("Could not verify active model availability: %s", exc)
            return

        if not models:
            LOGGER.warning(
                "Model listing did not return any entries; retaining configured model '%s'",
                self._config.model_name,
            )
            return

        if self._config.model_name not in models:
            fallback = models[0]
            LOGGER.info(
                "Configured model '%s' not available. Switching to '%s' instead.",
                self._config.model_name,
                fallback,
            )
            self._config.model_name = fallback

    def _announce_offline_mode(self) -> None:
        offline, reason = self.offline_status()
        if offline:
            detail = reason or "Network connectivity unavailable"
            LOGGER.info("Offline simulator engaged: %s", detail)

    def submit_user_message(self, user_input: str) -> str:
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

    def available_models(self) -> List[str]:
        try:
            models = self._client.list_models()
        except APIError as exc:
            LOGGER.warning("Falling back to configured model due to listing failure: %s", exc)
            return [self._config.model_name]
        if not models:
            return [self._config.model_name]
        if self._config.model_name not in models:
            fallback = models[0]
            LOGGER.info(
                "Configured model '%s' missing from listing. Adopting '%s'.",
                self._config.model_name,
                fallback,
            )
            self._config.model_name = fallback
        return models

    def select_model(self, model_name: str) -> None:
        with self._lock:
            LOGGER.info("Switching active model to %s", model_name)
            self._config.model_name = model_name

    def current_model(self) -> str:
        return self._config.model_name

    def endpoints(self) -> Dict[str, str]:
        return self._client.available_endpoints()

    def offline_status(self) -> Tuple[bool, Optional[str]]:
        return (self._client.using_offline_simulator(), self._client.offline_reason)

    def using_offline_mode(self) -> bool:
        offline, _ = self.offline_status()
        return offline

    def health_status(self) -> bool:
        return self._client.check_health()

    def clear_history(self) -> None:
        with self._lock:
            self._history.clear()
            try:
                self._store.save(self._history.export())
            except ChatStorageError as exc:
                LOGGER.warning("Could not clear stored history: %s", exc)

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
        self._palette = self._build_palette()
        self._root = tk.Tk()
        self._root.title("Hypothesis-link v2 – Enterprise Edition")
        self._root.geometry("1024x720")
        self._root.minsize(920, 640)
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._status_var = tk.StringVar(value="Initialising…")
        self._metrics_var = tk.StringVar(value=self._format_metrics(self._metrics.snapshot()))
        self._health_var = tk.StringVar(value="Health: checking…")
        self._model_var = tk.StringVar(value=self._orchestrator.current_model())

        self._apply_theme()
        self._build_layout()

        self._shutdown = threading.Event()
        self._work_queue: queue.Queue[Optional[Tuple[str, threading.Event]]] = queue.Queue()
        self._response_thread = threading.Thread(
            target=self._response_worker,
            name="ChatResponseWorker",
            daemon=True,
        )
        self._response_thread.start()

        self._populate_endpoints()
        self._append_initial_history()
        self._announce_offline_status()
        self._refresh_models_async()
        self._refresh_health_async()

    def _build_palette(self) -> Dict[str, str]:
        return {
            "background": "#0f172a",
            "card": "#111c3a",
            "accent": "#38bdf8",
            "accent_dark": "#0ea5e9",
            "text": "#f8fafc",
            "muted": "#94a3b8",
            "success": "#22c55e",
            "danger": "#ef4444",
        }

    def _apply_theme(self) -> None:
        palette = self._palette
        self._root.configure(bg=palette["background"])
        style = ttk.Style(self._root)
        try:
            style.theme_use("clam")
        except tk.TclError:  # pragma: no cover - depends on platform themes
            pass

        style.configure("Background.TFrame", background=palette["background"])
        style.configure("Card.TFrame", background=palette["card"], borderwidth=0)
        style.configure("CardInner.TFrame", background=palette["card"], borderwidth=0)
        style.configure(
            "Title.TLabel",
            background=palette["background"],
            foreground=palette["text"],
            font=("Segoe UI", 22, "bold"),
        )
        style.configure(
            "Subtitle.TLabel",
            background=palette["background"],
            foreground=palette["muted"],
            font=("Segoe UI", 12),
        )
        style.configure(
            "Status.TLabel",
            background=palette["background"],
            foreground=palette["accent"],
            font=("Segoe UI", 11, "bold"),
        )
        style.configure(
            "Metrics.TLabel",
            background=palette["background"],
            foreground=palette["muted"],
            font=("Segoe UI", 11),
        )
        style.configure(
            "HealthGood.TLabel",
            background=palette["background"],
            foreground=palette["success"],
            font=("Segoe UI", 11, "bold"),
        )
        style.configure(
            "HealthBad.TLabel",
            background=palette["background"],
            foreground=palette["danger"],
            font=("Segoe UI", 11, "bold"),
        )
        style.configure(
            "HealthUnknown.TLabel",
            background=palette["background"],
            foreground=palette["muted"],
            font=("Segoe UI", 11, "bold"),
        )
        style.configure(
            "Section.TLabel",
            background=palette["card"],
            foreground=palette["text"],
            font=("Segoe UI", 13, "bold"),
        )
        style.configure(
            "Caption.TLabel",
            background=palette["card"],
            foreground=palette["muted"],
            font=("Segoe UI", 10),
            wraplength=260,
            justify=tk.LEFT,
        )
        style.configure(
            "Accent.TButton",
            background=palette["accent"],
            foreground=palette["background"],
            padding=(10, 6),
            font=("Segoe UI", 11, "bold"),
            borderwidth=0,
        )
        style.map(
            "Accent.TButton",
            background=[("active", palette["accent_dark"]), ("disabled", "#1f2937")],
            foreground=[("disabled", palette["muted"])],
        )
        style.configure(
            "Ghost.TButton",
            background="#1e293b",
            foreground=palette["text"],
            padding=(10, 6),
            font=("Segoe UI", 11),
            borderwidth=0,
        )
        style.map(
            "Ghost.TButton",
            background=[("active", "#334155"), ("disabled", "#1f2937")],
            foreground=[("disabled", palette["muted"])],
        )
        style.configure(
            "Treeview",
            background=palette["card"],
            fieldbackground=palette["card"],
            foreground=palette["text"],
            bordercolor=palette["card"],
            rowheight=24,
        )
        style.configure(
            "Treeview.Heading",
            background=palette["accent"],
            foreground=palette["background"],
            bordercolor=palette["accent"],
            relief="flat",
            font=("Segoe UI", 11, "bold"),
        )
        style.map(
            "Treeview",
            background=[("selected", palette["accent_dark"])],
            foreground=[("selected", palette["background"])],
        )
        style.configure(
            "Activity.Horizontal.TProgressbar",
            troughcolor=palette["card"],
            background=palette["accent"],
            bordercolor=palette["card"],
            lightcolor=palette["accent"],
            darkcolor=palette["accent_dark"],
        )

    def _build_layout(self) -> None:
        palette = self._palette
        outer = ttk.Frame(self._root, padding=(24, 20, 24, 20), style="Background.TFrame")
        outer.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(outer, style="Background.TFrame")
        header.pack(fill=tk.X)
        ttk.Label(header, text="Hypothesis-link Enterprise", style="Title.TLabel").pack(
            anchor=tk.W
        )
        ttk.Label(
            header,
            text="Secure, beautiful interface for LM Studio",
            style="Subtitle.TLabel",
        ).pack(anchor=tk.W, pady=(4, 0))

        info_row = ttk.Frame(header, style="Background.TFrame")
        info_row.pack(fill=tk.X, pady=(12, 0))
        info_row.columnconfigure(0, weight=1)
        info_row.columnconfigure(1, weight=1)
        info_row.columnconfigure(2, weight=0)
        self._status_label = ttk.Label(info_row, textvariable=self._status_var, style="Status.TLabel")
        self._status_label.grid(row=0, column=0, sticky=tk.W)
        self._metrics_label = ttk.Label(info_row, textvariable=self._metrics_var, style="Metrics.TLabel")
        self._metrics_label.grid(row=0, column=1, sticky=tk.E, padx=(0, 12))
        self._health_label = ttk.Label(info_row, textvariable=self._health_var, style="HealthUnknown.TLabel")
        self._health_label.grid(row=0, column=2, sticky=tk.E)

        content = ttk.Frame(outer, style="Background.TFrame")
        content.pack(fill=tk.BOTH, expand=True, pady=(16, 0))
        content.columnconfigure(0, weight=3)
        content.columnconfigure(1, weight=2)

        chat_card = ttk.Frame(content, padding=18, style="Card.TFrame")
        chat_card.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        chat_card.columnconfigure(0, weight=1)

        insight_card = ttk.Frame(content, padding=18, style="Card.TFrame")
        insight_card.grid(row=0, column=1, sticky="nsew")
        insight_card.columnconfigure(0, weight=1)
        insight_card.rowconfigure(2, weight=1)

        model_row = ttk.Frame(chat_card, style="CardInner.TFrame")
        model_row.grid(row=0, column=0, sticky="ew")
        model_row.columnconfigure(1, weight=1)
        ttk.Label(model_row, text="Active model", style="Section.TLabel").grid(
            row=0, column=0, sticky=tk.W
        )
        self._model_dropdown = ttk.Combobox(
            model_row,
            textvariable=self._model_var,
            values=[self._model_var.get()],
            state="readonly",
        )
        self._model_dropdown.grid(row=0, column=1, sticky="ew", padx=(12, 12))
        self._model_dropdown.bind("<<ComboboxSelected>>", self._on_model_changed)
        self._model_refresh = ttk.Button(
            model_row,
            text="Refresh",
            style="Ghost.TButton",
            command=self._refresh_models_async,
        )
        self._model_refresh.grid(row=0, column=2, sticky=tk.E)

        ttk.Separator(chat_card, orient=tk.HORIZONTAL).grid(row=1, column=0, sticky="ew", pady=(12, 12))

        self._chat_display = scrolledtext.ScrolledText(
            chat_card,
            wrap=tk.WORD,
            font=("Segoe UI", 11),
            height=20,
            state=tk.DISABLED,
            bg=palette["background"],
            fg=palette["text"],
            insertbackground=palette["accent"],
            relief=tk.FLAT,
            borderwidth=0,
            highlightthickness=0,
        )
        self._chat_display.grid(row=2, column=0, sticky="nsew")
        chat_card.rowconfigure(2, weight=1)
        self._configure_chat_display()

        ttk.Separator(chat_card, orient=tk.HORIZONTAL).grid(row=3, column=0, sticky="ew", pady=(12, 12))

        input_row = ttk.Frame(chat_card, style="CardInner.TFrame")
        input_row.grid(row=4, column=0, sticky="ew")
        input_row.columnconfigure(0, weight=1)

        self._user_entry = tk.Entry(
            input_row,
            font=("Segoe UI", 12),
            relief=tk.FLAT,
            bg=palette["background"],
            fg=palette["text"],
            insertbackground=palette["accent"],
            highlightthickness=1,
            highlightbackground="#1e293b",
            highlightcolor=palette["accent"],
        )
        self._user_entry.grid(row=0, column=0, sticky="ew")
        self._user_entry.bind("<Return>", self._send_message_event)

        self._send_button = ttk.Button(
            input_row,
            text="Send",
            style="Accent.TButton",
            command=self._send_message_direct,
        )
        self._send_button.grid(row=0, column=1, padx=(12, 0))

        self._clear_button = ttk.Button(
            input_row,
            text="Clear",
            style="Ghost.TButton",
            command=self._clear_history,
        )
        self._clear_button.grid(row=0, column=2, padx=(12, 0))

        self._activity = ttk.Progressbar(
            chat_card,
            mode="indeterminate",
            style="Activity.Horizontal.TProgressbar",
        )
        self._activity.grid(row=5, column=0, sticky="ew", pady=(12, 0))

        ttk.Label(insight_card, text="LM Studio Diagnostics", style="Section.TLabel").grid(
            row=0, column=0, sticky=tk.W
        )
        ttk.Separator(insight_card, orient=tk.HORIZONTAL).grid(row=1, column=0, sticky="ew", pady=(10, 12))

        tree_container = ttk.Frame(insight_card, style="CardInner.TFrame")
        tree_container.grid(row=2, column=0, sticky="nsew")
        tree_container.columnconfigure(0, weight=1)
        tree_container.rowconfigure(0, weight=1)

        self._endpoint_tree = ttk.Treeview(
            tree_container,
            columns=("endpoint", "url"),
            show="headings",
            height=11,
        )
        self._endpoint_tree.heading("endpoint", text="Endpoint")
        self._endpoint_tree.heading("url", text="URL")
        self._endpoint_tree.column("endpoint", width=160, anchor=tk.W)
        self._endpoint_tree.column("url", width=280, anchor=tk.W)
        self._endpoint_tree.grid(row=0, column=0, sticky="nsew")

        endpoint_scroll = ttk.Scrollbar(
            tree_container, orient=tk.VERTICAL, command=self._endpoint_tree.yview
        )
        endpoint_scroll.grid(row=0, column=1, sticky="ns")
        self._endpoint_tree.configure(yscrollcommand=endpoint_scroll.set)

        ttk.Separator(insight_card, orient=tk.HORIZONTAL).grid(row=3, column=0, sticky="ew", pady=(12, 12))
        ttk.Label(
            insight_card,
            text="Endpoints mirror LM Studio's REST surface. Refresh models to stay in sync.",
            style="Caption.TLabel",
        ).grid(row=4, column=0, sticky=tk.W)

        action_row = ttk.Frame(insight_card, style="CardInner.TFrame")
        action_row.grid(row=5, column=0, sticky="ew", pady=(12, 0))
        action_row.columnconfigure(0, weight=1)
        ttk.Button(
            action_row,
            text="Check health",
            style="Ghost.TButton",
            command=self._refresh_health_async,
        ).grid(row=0, column=0, sticky=tk.E)

    def _configure_chat_display(self) -> None:
        palette = self._palette
        self._chat_display.tag_configure(
            "user",
            foreground=palette["accent"],
            font=("Segoe UI", 11, "bold"),
            spacing3=8,
        )
        self._chat_display.tag_configure(
            "assistant",
            foreground=palette["success"],
            font=("Segoe UI", 11),
            spacing3=12,
        )
        self._chat_display.tag_configure(
            "info",
            foreground=palette["muted"],
            font=("Segoe UI", 10, "italic"),
            spacing3=8,
        )
        self._chat_display.tag_configure(
            "history",
            foreground="#c084fc",
            font=("Segoe UI", 10),
            spacing3=8,
        )

    def _append_initial_history(self) -> None:
        history = self._orchestrator.export_history()
        if not history:
            self._append_to_display(
                "Session initialised. Messages will appear here.",
                tag="info",
                timestamp=False,
            )
            self._status_var.set("Ready")
            return
        for entry in history:
            role = entry.get("role", "user").capitalize()
            content = entry.get("content", "")
            self._append_to_display(f"{role}: {content}", tag="history", timestamp=False)
        self._status_var.set("History loaded")

    def _announce_offline_status(self) -> None:
        offline, reason = self._orchestrator.offline_status()
        if not offline:
            return
        detail = reason or "Network connectivity to the LM backend is unavailable."
        self._append_to_display(
            f"Offline simulator active. {detail}",
            tag="info",
            timestamp=False,
        )
        self._status_var.set("Offline simulator active")

    def _populate_endpoints(self) -> None:
        endpoints = self._orchestrator.endpoints()
        for item in self._endpoint_tree.get_children():
            self._endpoint_tree.delete(item)
        for name, url in sorted(endpoints.items()):
            pretty = name.replace("_", " ").title()
            self._endpoint_tree.insert("", tk.END, values=(pretty, url))

    def _refresh_models_async(self) -> None:
        self._model_refresh.state(["disabled"])

        def worker() -> None:
            try:
                models = self._orchestrator.available_models()
            except Exception as exc:  # pragma: no cover - defensive guard
                LOGGER.exception("Failed to refresh models: %s", exc)
                models = [self._orchestrator.current_model()]
            self._root.after(0, lambda m=models: self._update_models(m))

        threading.Thread(target=worker, name="ModelRefresh", daemon=True).start()

    def _refresh_health_async(self) -> None:
        self._health_var.set("Health: checking…")
        self._set_health_style("unknown")

        def worker() -> None:
            healthy = self._orchestrator.health_status()
            self._root.after(0, lambda: self._apply_health_result(healthy))

        threading.Thread(target=worker, name="HealthProbe", daemon=True).start()

    def _apply_health_result(self, healthy: bool) -> None:
        if healthy:
            self._health_var.set("Health: online")
            self._set_health_style("good")
        else:
            self._health_var.set("Health: offline or unsupported")
            self._set_health_style("bad")

    def _set_health_style(self, status: str) -> None:
        mapping = {
            "good": "HealthGood.TLabel",
            "bad": "HealthBad.TLabel",
            "unknown": "HealthUnknown.TLabel",
        }
        self._health_label.configure(style=mapping.get(status, "HealthUnknown.TLabel"))

    def _update_models(self, models: List[str]) -> None:
        unique_models = models or [self._orchestrator.current_model()]
        self._model_dropdown.configure(values=unique_models)
        current = self._orchestrator.current_model()
        if current not in unique_models:
            current = unique_models[0]
            self._orchestrator.select_model(current)
        self._model_var.set(current)
        self._model_refresh.state(["!disabled"])
        self._status_var.set(f"Active model set to {current}")

    def _format_metrics(self, snapshot: Dict[str, Any]) -> str:
        return (
            f"Responses: {snapshot['count']} • Failures: {snapshot['failures']} "
            f"• Mean latency: {snapshot['mean']:.2f}s"
        )

    def _send_message_event(self, _event: tk.Event[Any]) -> None:  # pragma: no cover - GUI
        self._send_message_direct()

    def _send_message_direct(self) -> None:
        raw_input = self._user_entry.get()
        try:
            sanitised = InputSanitiser.sanitise(raw_input)
        except ValueError as exc:
            messagebox.showwarning("Invalid input", str(exc))
            return

        self._user_entry.delete(0, tk.END)
        self._append_to_display(f"You: {sanitised}", tag="user")
        self._status_var.set("Dispatching prompt…")
        self._send_button.state(["disabled"])
        self._clear_button.state(["disabled"])
        self._activity.start(12)

        completion_event = threading.Event()
        self._work_queue.put((sanitised, completion_event))

    def _response_worker(self) -> None:
        while not self._shutdown.is_set():
            try:
                item = self._work_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if item is None:
                self._work_queue.task_done()
                break

            prompt, completion_event = item
            try:
                response = self._orchestrator.submit_user_message(prompt)
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
            except Exception as exc:  # pragma: no cover - defensive guard
                LOGGER.exception("Unexpected failure while processing prompt")
                self._root.after(
                    0,
                    lambda msg=str(exc): messagebox.showerror(
                        "Unexpected error", f"An unexpected error occurred: {msg}"
                    ),
                )
            else:
                self._root.after(
                    0,
                    lambda resp=response: self._append_to_display(
                        f"Assistant: {resp}", tag="assistant"
                    ),
                )
            finally:
                self._root.after(0, self._on_response_complete)
                completion_event.set()
                self._work_queue.task_done()

    def _on_response_complete(self) -> None:
        snapshot = self._metrics.snapshot()
        self._metrics_var.set(self._format_metrics(snapshot))
        self._status_var.set("Ready")
        self._activity.stop()
        self._send_button.state(["!disabled"])
        self._clear_button.state(["!disabled"])
        self._refresh_health_async()

    def _append_to_display(self, text: str, tag: str, timestamp: bool = True) -> None:
        message = text.strip()
        if timestamp:
            stamp = datetime.now().strftime("%H:%M:%S")
            message = f"[{stamp}] {message}"
        if not message.endswith("\n"):
            message = f"{message}\n"
        self._chat_display.configure(state=tk.NORMAL)
        self._chat_display.insert(tk.END, message, tag)
        self._chat_display.configure(state=tk.DISABLED)
        self._chat_display.yview(tk.END)

    def _clear_history(self) -> None:
        if not messagebox.askyesno("Confirm", "Clear the conversation history from this session?"):
            return
        self._orchestrator.clear_history()
        self._chat_display.configure(state=tk.NORMAL)
        self._chat_display.delete("1.0", tk.END)
        self._chat_display.configure(state=tk.DISABLED)
        self._append_to_display("History cleared.", tag="info")
        self._status_var.set("History cleared – Ready")

    def _on_model_changed(self, _event: tk.Event[Any]) -> None:
        selected = self._model_var.get()
        if selected:
            self._orchestrator.select_model(selected)
            self._status_var.set(f"Active model set to {selected}")

    def _on_close(self) -> None:
        if not messagebox.askokcancel("Quit", "Do you really want to exit Hypothesis-link?"):
            return
        self._shutdown.set()
        self._work_queue.put(None)
        self._response_thread.join(timeout=2)
        self._orchestrator.shutdown()
        self._root.destroy()

    def run(self) -> None:  # pragma: no cover - GUI loop
        LOGGER.info("Starting GUI loop")
        self._root.mainloop()


# ---------------------------------------------------------------------------
# Application bootstrap
# ---------------------------------------------------------------------------


def run_offline_self_test(rounds: int = 3) -> None:
    """Execute an offline send/receive diagnostic without network access."""

    CONSOLE.print("[bold cyan]Starting offline self-test…[/bold cyan]")

    config = AppConfig(
        api_base_url="http://offline.local",
        model_name="offline-simulator",
        timeout=5.0,
        max_retries=1,
        request_backoff=1.5,
        history_limit=8,
        rate_limit_per_minute=120,
        verify_tls=False,
    )
    config.validate()

    metrics = MetricsTracker()
    limiter = RateLimiter(config.rate_limit_per_minute)
    history = ConversationHistory(config.history_limit)
    temp_history_path = Path(tempfile.gettempdir()) / "hypothesis_link_offline_history.json"
    store = PersistentHistoryStore(temp_history_path)
    enhancer = PromptEnhancer()
    client = SecureHTTPClient(config, metrics, limiter, session=OfflineSession())
    orchestrator = ChatOrchestrator(config, history, store, enhancer, client, metrics)

    try:
        endpoints = orchestrator.endpoints()
        CONSOLE.print(
            "[bold green]Offline endpoints available:[/bold green] "
            + ", ".join(sorted(endpoints))
        )

        required_endpoints = {"chat_completions", "models", "health"}
        missing = required_endpoints.difference(endpoints)
        if missing:
            raise RuntimeError(f"Offline simulator is missing endpoints: {sorted(missing)}")

        if not orchestrator.health_status():
            raise RuntimeError("Offline simulator reported an unhealthy status")

        models = orchestrator.available_models()
        if not models:
            raise RuntimeError("Offline simulator returned no models")
        CONSOLE.print(
            "[bold green]Offline models detected:[/bold green] " + ", ".join(models)
        )

        for idx in range(rounds):
            prompt = f"offline diagnostic round {idx + 1}"
            response = orchestrator.submit_user_message(prompt)
            if prompt not in response:
                raise RuntimeError(
                    "Offline simulator response did not echo the supplied prompt"
                )
            CONSOLE.print(
                f"[cyan]Prompt:[/cyan] {prompt!r} -> [magenta]Response:[/magenta] {response!r}"
            )

        CONSOLE.print("[bold cyan]Offline self-test completed successfully.[/bold cyan]")
    finally:
        orchestrator.shutdown()
        temp_history_path.unlink(missing_ok=True)



def build_application() -> ChatGUI:
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


def main(argv: Optional[List[str]] = None) -> None:  # pragma: no cover - entry point
    parser = argparse.ArgumentParser(description="Hypothesis-link desktop client")
    parser.add_argument(
        "--offline-self-test",
        action="store_true",
        help="run the built-in offline send/receive diagnostic and exit",
    )
    args = parser.parse_args(argv)

    if args.offline_self_test:
        try:
            run_offline_self_test()
        except Exception as exc:  # pragma: no cover - diagnostic failure surface
            LOGGER.exception("Offline self-test failed")
            CONSOLE.print(f"[bold red]Offline self-test failed:[/bold red] {exc}")
            raise
        return

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
