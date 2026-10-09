"""Regression tests for the baseline gate, using only the standard library."""

import importlib.util
from io import StringIO
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import patch
import subprocess
import sys

sys.dont_write_bytecode = True

spec = importlib.util.spec_from_file_location(
    "check_ast_grep", Path(__file__).with_name("check-ast-grep.py")
)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class BaselineTests(TestCase):
    def test_existing_match_and_line_movement(self):
        diagnostic = {
            "ruleId": "rule", "file": "a.rs", "text": 'Symbol::new("func")',
            "range": {"start": {"line": 99}},
        }
        self.assertEqual(
            gate.new_diagnostics(
                [diagnostic], [["rule", "a.rs", diagnostic["text"], 1]]
            ),
            [],
        )

    def test_duplicate_is_new(self):
        diagnostic = {"ruleId": "rule", "file": "a.rs", "text": "Symbol::new(\"func\")"}
        self.assertEqual(
            gate.new_diagnostics(
                [diagnostic, diagnostic], [["rule", "a.rs", diagnostic["text"], 1]]
            ),
            [diagnostic],
        )

    def test_new_file_or_literal_is_new(self):
        diagnostics = [
            {"ruleId": "rule", "file": "b.rs", "text": "Symbol::new(\"func\")"},
            {"ruleId": "rule", "file": "a.rs", "text": "Symbol::new(\"other\")"},
        ]
        self.assertEqual(
            gate.new_diagnostics(diagnostics, [["rule", "a.rs", 'Symbol::new("func")', 1]]),
            diagnostics,
        )

    def test_different_rule_is_new(self):
        diagnostic = {"ruleId": "new-rule", "file": "a.rs", "text": "same"}
        self.assertEqual(
            gate.new_diagnostics([diagnostic], [["rule", "a.rs", "same", 1]]),
            [diagnostic],
        )

    def test_removing_existing_match_is_allowed(self):
        self.assertEqual(gate.new_diagnostics([], [["rule", "a.rs", "old", 1]]), [])

    def test_scan_failure_is_not_baselined(self):
        for status, output in [(2, "[]"), (1, "[]"), (1, "invalid json")]:
            with self.subTest(status=status, output=output), patch.object(
                gate.subprocess, "run", return_value=subprocess.CompletedProcess(
                    [], status, output, "SCAN_FAILURE"
                )
            ), patch.object(gate.Path, "read_text", return_value="[]"), patch.object(
                gate.sys, "stderr"
            ):
                self.assertEqual(gate.main(), 2)

    def test_launch_failure_reports_error(self):
        for error in [FileNotFoundError("not found"), PermissionError("not executable")]:
            with self.subTest(error=error), patch.object(
                gate.subprocess, "run", side_effect=error
            ), patch.object(gate.sys, "stderr", new_callable=StringIO) as stderr:
                self.assertEqual(gate.main(), 2)
                self.assertIn(f"Cannot launch ast-grep: {error}", stderr.getvalue())


if __name__ == "__main__":
    main()
