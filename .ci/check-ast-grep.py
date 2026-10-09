"""Reject ast-grep diagnostics exceeding the checked-in baseline."""

import json
from collections import Counter
from pathlib import Path
import subprocess
import sys


def new_diagnostics(diagnostics, baseline):
    remaining = Counter(
        {(rule, file, text): count for rule, file, text, count in baseline}
    )
    new = []
    for diagnostic in diagnostics:
        key = (diagnostic["ruleId"], diagnostic["file"], diagnostic["text"])
        if remaining[key] > 0:
            remaining[key] -= 1
        else:
            new.append(diagnostic)
    return new


def main():
    try:
        result = subprocess.run(
            ["ast-grep", "scan", "--json=compact", "crates"],
            capture_output=True,
            text=True,
        )
    except OSError as error:
        print(f"Cannot launch ast-grep: {error}", file=sys.stderr)
        return 2
    if result.returncode not in (0, 1):
        print(result.stderr, file=sys.stderr)
        return 2
    try:
        diagnostics = json.loads(result.stdout)
        baseline = json.loads(Path(".ast-grep/baseline.json").read_text())
        if not isinstance(diagnostics, list):
            raise ValueError("expected a diagnostic list")
        if result.returncode == 1 and not diagnostics:
            raise ValueError(result.stderr or "scan failed without diagnostics")
        new = new_diagnostics(diagnostics, baseline)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Cannot check ast-grep baseline: {error}", file=sys.stderr)
        return 2
    for diagnostic in new:
        start = diagnostic["range"]["start"]
        print(
            f'{diagnostic["file"]}:{start["line"] + 1}:{start["column"] + 1}: '
            f'{diagnostic["ruleId"]}: {diagnostic["message"]}',
            file=sys.stderr,
        )
    if new:
        print(f"ast-grep: {len(new)} new diagnostic(s)", file=sys.stderr)
        return 1
    print(f"ast-grep: no new diagnostics ({len(diagnostics)} baseline matches)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
