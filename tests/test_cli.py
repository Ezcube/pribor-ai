import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pribor_ai import __main__ as cli
from pribor_ai import core
from pribor_ai.provider import ProviderError


SECRET = "synthetic-test-key-do-not-log"


class CliTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.input = self.root / "products.json"
        self.output = self.root / "cards.json"
        self.md = self.root / "cards.md"
        # This is synthetic data in a temporary directory, never the actual project .env.
        self.env_file = self.root / ".env"
        self.env_file.write_text("OPENROUTER_API_KEY=" + SECRET + "\n", encoding="utf-8")
        self.input.write_text(json.dumps({"name": "Прибор"}), encoding="utf-8")
        self.clean_environment = patch.dict(os.environ, {}, clear=True)
        self.clean_environment.start()
        self.addCleanup(self.clean_environment.stop)

    def run_cli(self, *options):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = cli.main([str(self.input), "--env-file", str(self.env_file), *map(str, options)])
        combined = stdout.getvalue() + stderr.getvalue()
        self.assertNotIn(SECRET, combined)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_demo_stdout_is_json_and_does_not_load_key(self):
        with patch.object(cli, "load_env", side_effect=AssertionError("No env reads in demo")):
            code, stdout, stderr = self.run_cli("--demo")
        self.assertEqual(code, 0)
        cards = json.loads(stdout)
        self.assertEqual(cards[0]["generation"]["mode"], "demo")
        self.assertIn("без API", stderr)
        self.assertNotIn("OPENROUTER_API_KEY", os.environ)

    def test_demo_writes_json_and_readable_markdown(self):
        code, stdout, _ = self.run_cli("--demo", "--output", self.output, "--markdown", self.md)
        self.assertEqual(code, 0)
        self.assertEqual(stdout, "")
        self.assertEqual(json.loads(self.output.read_text(encoding="utf-8"))[0]["generation"]["mode"], "demo")
        self.assertIn("**demo**", self.md.read_text(encoding="utf-8"))

    def test_bad_input_has_no_traceback_or_partial_output(self):
        self.input.write_text('{"name":', encoding="utf-8")
        code, stdout, stderr = self.run_cli("--demo", "--output", self.output)
        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertIn("Некорректный JSON", stderr)
        self.assertNotIn("Traceback", stderr)
        self.assertFalse(self.output.exists())

    def test_api_batch_failure_preserves_both_previous_outputs(self):
        self.input.write_text(json.dumps([{"name": "Первый"}, {"name": "Второй"}]), encoding="utf-8")
        old_json, old_md = "previous JSON", "previous Markdown"
        self.output.write_text(old_json, encoding="utf-8")
        self.md.write_text(old_md, encoding="utf-8")
        first = core.generate(core.normalize_product({"name": "Первый"}), demo=True)
        failure = ProviderError("network_error", "Нет соединения")
        with patch.object(cli, "generate", side_effect=[first, failure]) as generate:
            code, stdout, stderr = self.run_cli("--output", self.output, "--markdown", self.md)
        self.assertEqual(code, 2)
        self.assertEqual(generate.call_count, 2)
        self.assertTrue(all(call.kwargs["demo"] is False for call in generate.call_args_list))
        self.assertEqual(stdout, "")
        self.assertEqual(self.output.read_text(encoding="utf-8"), old_json)
        self.assertEqual(self.md.read_text(encoding="utf-8"), old_md)
        self.assertIn("Нет соединения", stderr)
        self.assertNotIn("Готово", stderr)

    def test_failed_single_request_does_not_fallback_or_create_output(self):
        failure = ProviderError("rate_limited", "Лимит запросов")
        with patch.object(cli, "generate", side_effect=failure) as generate:
            code, stdout, _ = self.run_cli("--output", self.output)
        self.assertEqual(code, 2)
        generate.assert_called_once()
        self.assertEqual(generate.call_args.kwargs, {"demo": False})
        self.assertEqual(stdout, "")
        self.assertFalse(self.output.exists())

    def test_output_cannot_replace_env_input_or_other_output(self):
        env_before = self.env_file.read_bytes()
        input_before = self.input.read_bytes()
        cases = [
            ("--output", self.env_file), ("--markdown", self.env_file),
            ("--output", self.input), ("--markdown", self.input),
            ("--output", self.output, "--markdown", self.output),
            ("--output", self.root / "sub" / ".." / ".env"),
        ]
        for options in cases:
            with self.subTest(options=options), patch.object(cli, "load_env") as load, patch.object(cli, "generate") as generate:
                code, stdout, stderr = self.run_cli(*options)
                self.assertEqual(code, 2)
                self.assertEqual(stdout, "")
                self.assertIn("разные пути", stderr)
                load.assert_not_called()
                generate.assert_not_called()
                self.assertEqual(self.env_file.read_bytes(), env_before)
                self.assertEqual(self.input.read_bytes(), input_before)

    def test_env_errors_do_not_print_contents(self):
        self.env_file.write_text("UNKNOWN=" + SECRET, encoding="utf-8")
        with patch.object(cli, "generate") as generate:
            code, stdout, stderr = self.run_cli()
        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertIn("В .env допустимы", stderr)
        generate.assert_not_called()

    def test_process_environment_takes_precedence_over_file(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "existing-synthetic-key"}):
            cli.load_env(self.env_file)
            self.assertEqual(os.environ["OPENROUTER_API_KEY"], "existing-synthetic-key")


if __name__ == "__main__":
    unittest.main()
