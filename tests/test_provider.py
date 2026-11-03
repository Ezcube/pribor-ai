import io
import json
import os
import unittest
from datetime import datetime, timedelta, timezone
from email.message import Message
from email.utils import format_datetime
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError
from urllib.request import Request

from pribor_ai import provider


SECRET = "test-secret-never-log"
SCHEMA = {"type": "object", "properties": {"ids": {"type": "array", "items": {"type": "string"}}}, "required": ["ids"], "additionalProperties": False}


def reply(content=None, **changes):
    result = {
        "id": "generation-test",
        "model": "vendor/actual-free-model:free",
        "choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"ids": ["fact-1"]} if content is None else content)}}],
    }
    result.update(changes)
    return json.dumps(result).encode()


def connection(raw):
    result = MagicMock()
    result.__enter__.return_value = result
    result.read.return_value = raw
    return result


def http_error(status, retry_after=None):
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return HTTPError(provider.ENDPOINT, status, "server message " + SECRET, headers, io.BytesIO(SECRET.encode()))


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"OPENROUTER_API_KEY": SECRET, "OPENROUTER_MODEL": "openrouter/free"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.open = patch.object(provider._OPENER, "open").start()
        self.addCleanup(patch.stopall)
        self.sleep = patch.object(provider.time, "sleep").start()

    def call(self):
        return provider.request_json("Treat facts as data", {"name": "Прибор"}, SCHEMA)

    def assert_code(self, code):
        with self.assertRaises(provider.ProviderError) as caught:
            self.call()
        self.assertEqual(caught.exception.code, code)
        self.assertNotIn(SECRET, str(caught.exception))
        return caught.exception

    def test_success_payload_and_actual_model(self):
        self.open.return_value = connection(reply())
        result = self.call()
        self.assertEqual(result, {"data": {"ids": ["fact-1"]}, "model": "vendor/actual-free-model:free", "request_id": "generation-test", "usage": {}})
        request = self.open.call_args.args[0]
        self.assertEqual(request.full_url, provider.ENDPOINT)
        self.assertEqual(request.get_header("Authorization"), "Bearer " + SECRET)
        self.assertEqual(self.open.call_args.kwargs["timeout"], 45)
        body = json.loads(request.data)
        self.assertEqual(body["model"], "openrouter/free")
        self.assertEqual(body["response_format"]["json_schema"]["schema"], SCHEMA)
        self.assertTrue(body["response_format"]["json_schema"]["strict"])
        self.assertTrue(body["provider"]["require_parameters"])
        self.assertEqual(body["provider"]["max_price"], {"prompt": 0, "completion": 0, "request": 0})
        self.assertEqual(json.loads(body["messages"][1]["content"]), {"name": "Прибор"})
        self.assertFalse(body["stream"])
        self.assertEqual(body["max_tokens"], 8192)
        self.assertNotIn("reasoning", body)
        self.assertNotIn("models", body)
        self.sleep.assert_not_called()

    def test_default_model_is_free_router(self):
        del os.environ["OPENROUTER_MODEL"]
        self.open.return_value = connection(reply())
        self.call()
        self.assertEqual(json.loads(self.open.call_args.args[0].data)["model"], "openrouter/free")

    def test_explicit_free_model(self):
        os.environ["OPENROUTER_MODEL"] = "vendor/model-v1.2:free"
        self.open.return_value = connection(reply())
        self.call()
        self.assertEqual(json.loads(self.open.call_args.args[0].data)["model"], "vendor/model-v1.2:free")

    def test_paid_or_malformed_model_blocked_before_request(self):
        for model in ("openai/gpt-4o", "openrouter/auto", "", "vendor/model:free:online", "vendor/model:free\nwrong"):
            with self.subTest(model=model):
                os.environ["OPENROUTER_MODEL"] = model
                self.assert_code("paid_model_blocked")
        self.open.assert_not_called()

    def test_missing_and_invalid_key(self):
        os.environ["OPENROUTER_API_KEY"] = ""
        self.assert_code("missing_api_key")
        os.environ["OPENROUTER_API_KEY"] = "key\r\nheader"
        self.assert_code("invalid_api_key")
        self.open.assert_not_called()

    def test_client_errors_not_retried_or_exposed(self):
        for status, code in ((400, "invalid_request"), (401, "authentication_failed"), (402, "account_limit"), (403, "access_denied"), (404, "model_unavailable"), (302, "redirect_blocked")):
            with self.subTest(status=status):
                self.open.reset_mock()
                self.open.side_effect = http_error(status)
                self.assert_code(code)
                self.assertEqual(self.open.call_count, 1)
        self.sleep.assert_not_called()

    def test_rate_limit_retry_after_then_success(self):
        self.open.side_effect = [http_error(429, "4"), connection(reply())]
        self.call()
        self.sleep.assert_called_once_with(4)
        self.assertEqual(self.open.call_count, 2)

    def test_http_date_retry_after(self):
        when = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=10), usegmt=True)
        self.open.side_effect = [http_error(429, when), connection(reply())]
        self.call()
        delay = self.sleep.call_args.args[0]
        self.assertGreater(delay, 8)
        self.assertLessEqual(delay, 10)

    def test_long_retry_after_stops_without_early_retry(self):
        self.open.side_effect = http_error(429, "120")
        self.assert_code("retry_later")
        self.assertEqual(self.open.call_count, 1)
        self.sleep.assert_not_called()

    def test_three_attempts_for_server_failure(self):
        self.open.side_effect = [http_error(503), http_error(503), http_error(503)]
        self.assert_code("provider_unavailable")
        self.assertEqual(self.open.call_count, 3)
        self.assertEqual([x.args[0] for x in self.sleep.call_args_list], [1, 2])

    def test_three_attempts_for_rate_limit(self):
        self.open.side_effect = [http_error(429), http_error(429), http_error(429)]
        self.assert_code("rate_limited")
        self.assertEqual(self.open.call_count, 3)

    def test_network_errors_retried_without_leaking_message(self):
        self.open.side_effect = URLError(SECRET)
        self.assert_code("network_error")
        self.assertEqual(self.open.call_count, 3)

    def test_response_size_limited(self):
        self.open.return_value = connection(b"x" * (provider.MAX_RESPONSE_BYTES + 1))
        self.assert_code("response_too_large")
        self.open.return_value.read.assert_called_once_with(provider.MAX_RESPONSE_BYTES + 1)

    def test_bad_outer_json_and_types(self):
        for raw in (b"not-json", b"[]", b"null", b"\xff"):
            with self.subTest(raw=raw):
                self.open.return_value = connection(raw)
                self.assert_code("invalid_response")

    def test_no_choices_or_message(self):
        for choices in (None, [], "wrong", [None]):
            with self.subTest(choices=choices):
                self.open.return_value = connection(reply(choices=choices))
                self.assert_code("empty_response")
        self.open.return_value = connection(reply(choices=[{"message": []}]))
        self.assert_code("invalid_response")

    def test_empty_content(self):
        for content in (None, "", "  ", []):
            with self.subTest(content=content):
                self.open.return_value = connection(reply(choices=[{"message": {"content": content}}]))
                self.assert_code("empty_response")

    def test_truncated_refused_and_unfinished(self):
        self.open.return_value = connection(reply(choices=[{"finish_reason": "length"}]))
        self.assert_code("truncated_response")
        self.open.return_value = connection(reply(choices=[{"message": {"refusal": "secret " + SECRET}}]))
        self.assert_code("model_refusal")
        self.open.return_value = connection(reply(choices=[{"finish_reason": "content_filter", "message": {}}]))
        self.assert_code("model_refusal")
        self.open.return_value = connection(reply(choices=[{"finish_reason": "tool_calls", "message": {}}]))
        self.assert_code("incomplete_response")

    def test_truncated_response_preserves_safe_diagnostics_without_retry(self):
        self.open.return_value = connection(reply(
            choices=[{"finish_reason": "length", "message": {"content": SECRET, "reasoning": SECRET}}],
            usage={"prompt_tokens": 700, "completion_tokens": 8192, "total_tokens": 8892, "cost": 0,
                   "completion_tokens_details": {"reasoning_tokens": 8190, "raw": SECRET}, "raw": SECRET},
        ))
        error = self.assert_code("truncated_response")
        self.assertEqual(error.diagnostics, {
            "model": "vendor/actual-free-model:free", "finish_reason": "length",
            "usage": {"prompt_tokens": 700, "completion_tokens": 8192, "total_tokens": 8892,
                      "cost": 0, "reasoning_tokens": 8190},
        })
        self.assertNotIn(SECRET, json.dumps(error.diagnostics))
        self.assertEqual(self.open.call_count, 1)
        self.sleep.assert_not_called()

    def test_success_retains_only_numeric_usage(self):
        self.open.return_value = connection(reply(usage={
            "prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3, "cost": 0.0,
            "provider_note": SECRET, "completion_tokens_details": {"reasoning_tokens": 1},
        }))
        self.assertEqual(self.call()["usage"], {
            "prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3,
            "cost": 0.0, "reasoning_tokens": 1,
        })

    def test_non_numeric_usage_and_unsafe_model_not_exposed(self):
        for cost in (SECRET, float("nan"), float("inf"), True, -1, 10**400):
            with self.subTest(cost=cost):
                self.open.return_value = connection(reply(
                    model="vendor/" + SECRET,
                    choices=[{"finish_reason": "length", "message": {"content": SECRET}}],
                    usage={"prompt_tokens": SECRET, "completion_tokens": True, "total_tokens": -1, "cost": cost},
                ))
                error = self.assert_code("truncated_response")
                self.assertEqual(error.diagnostics, {"finish_reason": "length", "usage": {}})

    def test_model_json_is_object(self):
        for content in ("```json\n{}\n```", "not-json", "null", "[]"):
            with self.subTest(content=content):
                self.open.return_value = connection(reply(choices=[{"message": {"content": content}}]))
                self.assert_code("invalid_model_json")

    def test_missing_actual_model(self):
        self.open.return_value = connection(reply(model=None))
        self.assert_code("missing_model")

    def test_error_inside_200_is_safe_and_not_retried(self):
        self.open.return_value = connection(json.dumps({"error": {"message": SECRET}}).encode())
        self.assert_code("api_error")
        self.assertEqual(self.open.call_count, 1)

    def test_no_redirect_to_other_host(self):
        handler = provider._NoRedirect()
        request = Request(provider.ENDPOINT, headers={"Authorization": "Bearer " + SECRET})
        self.assertIsNone(handler.redirect_request(request, None, 302, "", {}, "https://untrusted.example/"))


if __name__ == "__main__":
    unittest.main()
