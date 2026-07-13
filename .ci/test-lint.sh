#!/bin/sh

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TEMP_DIR"' EXIT HUP INT TERM

FIXTURE_BIN="$SCRIPT_DIR/test-fixtures/lint/bin"
export LINT_TEST_NPX_MARKER="$TEMP_DIR/npx-ran"

if PATH="$FIXTURE_BIN:$PATH" "$SCRIPT_DIR/lint.sh" >"$TEMP_DIR/output" 2>&1; then
    echo "lint.sh unexpectedly succeeded" >&2
    exit 1
else
    status=$?
fi

if [ "$status" -ne 2 ]; then
    echo "lint.sh returned $status instead of 2" >&2
    exit 1
fi

if ! grep -q "CLIPPY_FAILURE_MARKER" "$TEMP_DIR/output"; then
    echo "lint.sh did not print the tail of clippy output" >&2
    exit 1
fi

if grep -q "clippy output 1$" "$TEMP_DIR/output"; then
    echo "lint.sh printed unbounded clippy output" >&2
    exit 1
fi

if [ -e "$LINT_TEST_NPX_MARKER" ]; then
    echo "lint.sh continued after clippy failed" >&2
    exit 1
fi
