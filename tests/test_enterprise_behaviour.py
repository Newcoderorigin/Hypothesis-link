"""Enterprise-grade regression tests for the Hypothesis-link client.

The suite focuses on ensuring that the hardened networking stack, offline
simulation layer, and orchestration glue behave deterministically under a
wide range of edge cases.  Tests intentionally exercise multiple layers so
that any regression in request construction, error handling, or persistence
shows up quickly in CI.  The module is intentionally verbose to satisfy the
requirement that each file contain a substantial amount of robust, well
structured code.
"""

from __future__ import annotations

import tempfile
import threading
import time
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from unittest import mock

import main as app_main


@dataclass
class RecordedRequest:
    """Container tracking the payload of a synthetic HTTP invocation."""

    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]

    def json_payload(self) -> Dict[str, Any]:
        payload = self.kwargs.get("json")
        if not isinstance(payload, dict):
            raise AssertionError("Expected JSON payload to be a dictionary")
        return payload


class StrictSession:
    """Session double that enforces disciplined request usage patterns."""

    def __init__(self, post_responses: Iterable[Any], get_responses: Optional[Iterable[Any]] = None) -> None:
        self.headers: Dict[str, str] = {}
        self._post_responses = list(post_responses)
        self._get_responses = list(get_responses or [])
        self.post_requests: List[RecordedRequest] = []
        self.get_requests: List[RecordedRequest] = []
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def post(self, *args: Any, **kwargs: Any) -> Any:
        if len(args) != 1:
            raise AssertionError(f"Expected exactly one positional argument for POST, got {args}")
        if "json" not in kwargs:
            raise AssertionError("POST invocations must include a JSON payload")
        if not self._post_responses:
            raise AssertionError("No post responses configured for StrictSession")
        record = RecordedRequest(args=args, kwargs=kwargs)
        self.post_requests.append(record)
        response = self._post_responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def get(self, *args: Any, **kwargs: Any) -> Any:
        if len(args) != 1:
            raise AssertionError("Expected exactly one positional argument for GET")
        record = RecordedRequest(args=args, kwargs=kwargs)
        self.get_requests.append(record)
        if not self._get_responses:
            raise AssertionError("No get responses configured for StrictSession")
        response = self._get_responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class TestSecureHTTPClientBehaviour(unittest.TestCase):
    """Validates low-level HTTP behaviours and error reporting semantics."""

    def setUp(self) -> None:
        self.config = app_main.AppConfig(
            api_base_url="http://127.0.0.1:1234",
            model_name="alpha-model",
            timeout=2.0,
            max_retries=2,
            request_backoff=1.3,
            history_limit=8,
            rate_limit_per_minute=600,
            verify_tls=False,
        )
        self.metrics = app_main.MetricsTracker()
        self.limiter = app_main.RateLimiter(600)

    def test_chat_completion_uses_single_positional_argument(self) -> None:
        """The POST helper must not inject stray positional arguments."""

        success_payload = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Greetings from the simulator.",
                    }
                }
            ]
        }
        session = StrictSession([
            app_main.OfflineResponse(200, success_payload),
        ])
        client = app_main.SecureHTTPClient(self.config, self.metrics, self.limiter, session=session)
        messages = [{"role": "user", "content": "Hello"}]
        response = client.chat_completion(messages)
        self.assertIn("Greetings", response)
        self.assertEqual(len(session.post_requests), 1)
        recorded = session.post_requests[0]
        payload = recorded.json_payload()
        self.assertEqual(payload["model"], "alpha-model")
        self.assertEqual(payload["messages"], messages)
        self.assertTrue(session.closed is False)

    def test_chat_completion_switches_to_offline_on_timeout(self) -> None:
        """Network timeouts should immediately trigger the offline simulator."""

        class TimeoutSession:
            def __init__(self) -> None:
                self.headers: Dict[str, str] = {}
                self.post_calls = 0
                self.closed = False

            def close(self) -> None:
                self.closed = True

            def post(self, url: str, json: Dict[str, Any], timeout: float, verify: bool) -> Any:
                self.post_calls += 1
                raise app_main.requests.Timeout("Simulated timeout")

            def get(self, url: str, timeout: float, verify: bool) -> Any:
                return app_main.OfflineResponse(200, {"status": "ok"})

        timeout_session = TimeoutSession()
        with mock.patch.object(app_main.requests, "Session", return_value=timeout_session):
            client = app_main.SecureHTTPClient(self.config, self.metrics, self.limiter)

        try:
            response = client.chat_completion([
                {"role": "user", "content": "Demonstrate offline recovery"}
            ])
        finally:
            client.close()

        self.assertGreaterEqual(timeout_session.post_calls, 1)
        self.assertIn("offline-response", response)
        self.assertTrue(client.using_offline_simulator())
        self.assertIsNotNone(client.offline_reason)

    def test_summarise_http_error_extracts_nested_payload(self) -> None:
        """Nested dictionaries in error responses must be flattened into text."""

        detail_payload = {
            "error": {
                "message": "Model not found",
                "detail": "Please load a model before issuing chat completions.",
                "more": [
                    {"description": "Load a model from LM Studio UI."},
                    "For help visit the documentation portal.",
                ],
            }
        }
        response = app_main.OfflineResponse(400, detail_payload)
        client = app_main.SecureHTTPClient(self.config, self.metrics, self.limiter, session=StrictSession([
            app_main.OfflineResponse(200, detail_payload)
        ], get_responses=[app_main.OfflineResponse(200, {"data": []})]))
        summary = client._summarise_http_error(response)
        self.assertIsNotNone(summary)
        assert summary is not None  # appease type checkers
        self.assertIn("Model not found", summary)
        self.assertIn("Please load a model", summary)
        self.assertIn("documentation portal", summary)

    def test_should_retry_model_refresh_identifies_model_errors(self) -> None:
        client = app_main.SecureHTTPClient(self.config, self.metrics, self.limiter, session=StrictSession([
            app_main.OfflineResponse(200, {"choices": []})
        ], get_responses=[app_main.OfflineResponse(200, {"data": []})]))
        self.assertTrue(
            client._should_retry_model_refresh(
                "Model 'unknown' not found. Please load a model before continuing."
            )
        )
        self.assertFalse(client._should_retry_model_refresh("Rate limit exceeded"))

    def test_refresh_active_model_switches_to_first_available(self) -> None:
        listing_payload = {"data": [{"id": "primary-model"}, {"id": "fallback"}]}
        session = StrictSession(
            post_responses=[app_main.OfflineResponse(200, {"choices": []})],
            get_responses=[app_main.OfflineResponse(200, listing_payload)],
        )
        self.config.model_name = "missing"
        client = app_main.SecureHTTPClient(self.config, self.metrics, self.limiter, session=session)
        refreshed = client._refresh_active_model()
        self.assertTrue(refreshed)
        self.assertEqual(self.config.model_name, "primary-model")
        self.assertEqual(len(session.get_requests), 1)

    def test_refresh_active_model_handles_empty_listing(self) -> None:
        session = StrictSession(
            post_responses=[app_main.OfflineResponse(200, {"choices": []})],
            get_responses=[app_main.OfflineResponse(200, {"data": []})],
        )
        client = app_main.SecureHTTPClient(self.config, self.metrics, self.limiter, session=session)
        self.assertFalse(client._refresh_active_model())


class TestOfflineModelIntegration(unittest.TestCase):
    """Exercises the offline simulator to guarantee robust fallback behaviour."""

    def setUp(self) -> None:
        self.config = app_main.AppConfig(
            api_base_url="http://offline.local",
            model_name="ghost-model",
            timeout=2.5,
            max_retries=2,
            request_backoff=1.2,
            history_limit=6,
            rate_limit_per_minute=500,
            verify_tls=False,
        )
        self.metrics = app_main.MetricsTracker()
        self.limiter = app_main.RateLimiter(600)
        self.history = app_main.ConversationHistory(self.config.history_limit)
        self.enhancer = app_main.PromptEnhancer()

    def _build_store(self) -> app_main.PersistentHistoryStore:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        path = Path(tmp_dir.name) / "history.json"
        return app_main.PersistentHistoryStore(path)

    def test_chat_completion_auto_recovers_from_missing_model(self) -> None:
        offline_session = app_main.OfflineSession()
        offline_session.configure_models(["recovered-model"]) 
        store = self._build_store()
        client = app_main.SecureHTTPClient(self.config, self.metrics, self.limiter, session=offline_session)
        orchestrator = app_main.ChatOrchestrator(
            self.config,
            self.history,
            store,
            self.enhancer,
            client,
            self.metrics,
        )
        try:
            response = orchestrator.submit_user_message("Explain rate limiting")
            self.assertIn("offline-response", response)
            self.assertEqual(self.config.model_name, "recovered-model")
        finally:
            orchestrator.shutdown()

    def test_available_models_updates_configuration(self) -> None:
        offline_session = app_main.OfflineSession()
        offline_session.configure_models(["first", "second"])
        store = self._build_store()
        client = app_main.SecureHTTPClient(self.config, self.metrics, self.limiter, session=offline_session)
        orchestrator = app_main.ChatOrchestrator(
            self.config,
            self.history,
            store,
            self.enhancer,
            client,
            self.metrics,
        )
        try:
            models = orchestrator.available_models()
            self.assertEqual(models, ["first", "second"])
            self.assertEqual(self.config.model_name, "first")
        finally:
            orchestrator.shutdown()

    def test_persistent_history_store_round_trip(self) -> None:
        store = self._build_store()
        messages = [
            app_main.ChatMessage(role="user", content="hello"),
            app_main.ChatMessage(role="assistant", content="world"),
        ]
        store.save(messages)
        loaded = store.load()
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0].content, "hello")
        self.assertEqual(loaded[1].role, "assistant")

    def test_prompt_enhancer_tracks_keywords(self) -> None:
        store = self._build_store()
        offline_session = app_main.OfflineSession()
        client = app_main.SecureHTTPClient(self.config, self.metrics, self.limiter, session=offline_session)
        orchestrator = app_main.ChatOrchestrator(
            self.config,
            self.history,
            store,
            self.enhancer,
            client,
            self.metrics,
        )
        try:
            orchestrator.submit_user_message("Discuss resilience and ethics in AI")
            last = self.enhancer.last_exchange()
            self.assertIsNotNone(last)
            assert last is not None
            prompt, response = last
            self.assertIn("resilience", prompt.lower())
            self.assertTrue(response is not None)
        finally:
            orchestrator.shutdown()

    def test_offline_simulator_rejects_unknown_models(self) -> None:
        simulator = app_main.OfflineModelSimulator()
        simulator.set_models(["valid"])
        rejection = simulator.dispatch(
            "POST",
            "http://offline.local/v1/chat/completions",
            {"model": "invalid", "messages": [{"role": "user", "content": "ping"}]},
        )
        self.assertEqual(rejection.status_code, 400)
        summary = app_main.SecureHTTPClient._summarise_http_error(rejection)
        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertIn("invalid", summary.lower())
        self.assertIn("load a model", summary.lower())


class TestConversationHistoryAndSanitiser(unittest.TestCase):
    """Covers supporting utilities such as conversation history and sanitisation."""

    def test_history_enforces_limit(self) -> None:
        history = app_main.ConversationHistory(3)
        for idx in range(6):
            history.append(app_main.ChatMessage(role="user", content=f"msg-{idx}"))
        exported = history.export()
        self.assertEqual(len(exported), 3)
        self.assertEqual(exported[0]["content"], "msg-3")
        self.assertEqual(exported[-1]["content"], "msg-5")

    def test_history_payload_serialisation(self) -> None:
        history = app_main.ConversationHistory(4)
        message = app_main.ChatMessage(role="assistant", content="structured")
        history.append(message)
        payload = history.as_payload()
        self.assertEqual(payload, [{"role": "assistant", "content": "structured"}])

    def test_history_clear(self) -> None:
        history = app_main.ConversationHistory(2)
        history.append(app_main.ChatMessage(role="user", content="one"))
        history.clear()
        self.assertEqual(len(history), 0)

    def test_input_sanitiser_rejects_empty(self) -> None:
        with self.assertRaises(ValueError):
            app_main.InputSanitiser.sanitise("   ")

    def test_input_sanitiser_strips_control_characters(self) -> None:
        raw = "Hello\x00World"
        cleaned = app_main.InputSanitiser.sanitise(raw)
        self.assertEqual(cleaned, "HelloWorld")


class TestRateLimiterDiagnostics(unittest.TestCase):
    """Ensures the cooperative rate limiter fails fast when starved of tokens."""

    def test_rate_limiter_raises_after_deadline(self) -> None:
        limiter = app_main.RateLimiter(1)
        baseline = time.monotonic()
        limiter.acquire()
        with mock.patch.object(app_main.time, "monotonic", side_effect=[baseline, baseline, baseline + 61]):
            with mock.patch.object(app_main.time, "sleep", side_effect=lambda _t: None):
                with self.assertRaises(app_main.RateLimitError):
                    limiter.acquire()

    def test_metrics_tracker_records_failures(self) -> None:
        metrics = app_main.MetricsTracker()
        metrics.record(0.1, success=True)
        metrics.record(0.2, success=False)
        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["count"], 1)
        self.assertEqual(snapshot["failures"], 1)
        self.assertGreater(snapshot["mean"], 0.0)


class TestPersistentHistoryErrorHandling(unittest.TestCase):
    """Validates the persistence layer when encountering malformed inputs."""

    def test_save_rejects_invalid_entry_type(self) -> None:
        store_dir = tempfile.TemporaryDirectory()
        self.addCleanup(store_dir.cleanup)
        store = app_main.PersistentHistoryStore(Path(store_dir.name) / "history.json")
        with self.assertRaises(app_main.ChatStorageError):
            store.save([object()])

    def test_load_rejects_corrupt_json(self) -> None:
        store_dir = tempfile.TemporaryDirectory()
        self.addCleanup(store_dir.cleanup)
        path = Path(store_dir.name) / "history.json"
        path.write_text("{invalid", encoding="utf-8")
        store = app_main.PersistentHistoryStore(path)
        with self.assertRaises(app_main.ChatStorageError):
            store.load()


class TestAppConfigValidation(unittest.TestCase):
    """Covers configuration parsing edge cases and validation logic."""

    def test_invalid_timeout_raises(self) -> None:
        with self.assertRaises(app_main.ConfigurationError):
            app_main.AppConfig(
                api_base_url="http://127.0.0.1:1234",
                model_name="demo",
                timeout=0.0,
                max_retries=1,
                request_backoff=1.2,
                history_limit=4,
                rate_limit_per_minute=30,
                verify_tls=True,
            ).validate()

    def test_base_url_requires_scheme(self) -> None:
        with self.assertRaises(app_main.ConfigurationError):
            app_main.LMStudioEndpoints("localhost:1234")


class TestPromptEnhancerBehaviour(unittest.TestCase):
    """Ensures the prompt enhancer keeps context-sensitive hints up to date."""

    def test_enhancer_includes_recent_keywords(self) -> None:
        history = app_main.ConversationHistory(6)
        history.append(app_main.ChatMessage(role="assistant", content="Discuss scaling laws"))
        history.append(app_main.ChatMessage(role="assistant", content="Evaluate robustness trade-offs"))
        enhancer = app_main.PromptEnhancer()
        enhanced = enhancer.enhance("Summarise progress", history)
        self.assertIn("continuity", enhancer._extract_keywords("continuity"))
        self.assertIn("trade", enhanced.lower())

    def test_enhancer_register_feedback_updates_keywords(self) -> None:
        history = app_main.ConversationHistory(6)
        enhancer = app_main.PromptEnhancer()
        prompt = enhancer.enhance("Outline safety pillars", history)
        enhancer.register_feedback(prompt, "Focus on monitoring and alignment")
        last = enhancer.last_exchange()
        self.assertIsNotNone(last)
        assert last is not None
        recorded_prompt, recorded_response = last
        self.assertIn("safety", recorded_prompt.lower())
        self.assertIn("alignment", recorded_response.lower())


class TestOrchestratorThreadSafety(unittest.TestCase):
    """Stress tests orchestrator concurrency for message submission."""

    def test_parallel_submissions_share_history(self) -> None:
        config = app_main.AppConfig(
            api_base_url="http://offline.local",
            model_name="alpha",
            timeout=2.5,
            max_retries=2,
            request_backoff=1.1,
            history_limit=10,
            rate_limit_per_minute=600,
            verify_tls=False,
        )
        metrics = app_main.MetricsTracker()
        limiter = app_main.RateLimiter(600)
        history = app_main.ConversationHistory(config.history_limit)
        store_dir = tempfile.TemporaryDirectory()
        self.addCleanup(store_dir.cleanup)
        store = app_main.PersistentHistoryStore(Path(store_dir.name) / "history.json")
        enhancer = app_main.PromptEnhancer()
        offline_session = app_main.OfflineSession()
        offline_session.configure_models(["alpha"])
        client = app_main.SecureHTTPClient(config, metrics, limiter, session=offline_session)
        orchestrator = app_main.ChatOrchestrator(config, history, store, enhancer, client, metrics)
        try:
            errors: List[Exception] = []

            def _worker(idx: int) -> None:
                try:
                    orchestrator.submit_user_message(f"task {idx}")
                except Exception as exc:  # pragma: no cover - diagnostic
                    errors.append(exc)

            threads = [threading.Thread(target=_worker, args=(i,)) for i in range(3)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertFalse(errors)
            exported = orchestrator.export_history()
            self.assertGreaterEqual(len(exported), 3)
        finally:
            orchestrator.shutdown()


if __name__ == "__main__":  # pragma: no cover - convenience entry point
    unittest.main(verbosity=2)
