#!/bin/sh

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TEMP_DIR"' EXIT HUP INT TERM

FIXTURE_BIN="$SCRIPT_DIR/test-fixtures/lint/bin"
export LINT_TEST_NPX_MARKER="$TEMP_DIR/npx-ran"
export LINT_TEST_COMMAND_LOG="$TEMP_DIR/commands"

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

if ! grep -q "clippy output 62$" "$TEMP_DIR/output"; then
    echo "lint.sh did not print the complete 40-line clippy tail" >&2
    exit 1
fi

if grep -q "clippy output 61$" "$TEMP_DIR/output"; then
    echo "lint.sh printed more than the final 40 clippy lines" >&2
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

export PATH="$FIXTURE_BIN:$PATH"

expect_status() {
    expected_status="$1"
    shift
    : >"$LINT_TEST_COMMAND_LOG"
    if "$@" >"$TEMP_DIR/output" 2>&1; then
        actual_status=0
    else
        actual_status=$?
    fi
    if [ "$actual_status" -ne "$expected_status" ]; then
        cat "$TEMP_DIR/output" >&2
        echo "$* returned $actual_status instead of $expected_status" >&2
        exit 1
    fi
}

expect_commands() {
    : >"$TEMP_DIR/expected"
    for command in "$@"; do
        printf '%s\n' "$command" >>"$TEMP_DIR/expected"
    done
    diff -u "$TEMP_DIR/expected" "$LINT_TEST_COMMAND_LOG"
}

FMT='cargo fmt --all --check'
CLIPPY='cargo clippy --workspace --all-targets --message-format=short -- -D warnings'
MARKDOWN='npx markdownlint-cli2 **/*.md #node_modules'
TESTS='cargo nextest run --workspace'

# Quick lint succeeds even though the Clippy fixture fails by default.
expect_status 0 "$SCRIPT_DIR/lint.sh" --quick
expect_commands "$FMT" "$MARKDOWN"

expect_status 2 env LINT_TEST_FMT_STATUS=1 "$SCRIPT_DIR/lint.sh" --quick
expect_commands "$FMT"

expect_status 2 env LINT_TEST_NPX_STATUS=1 "$SCRIPT_DIR/lint.sh" --quick
expect_commands "$FMT" "$MARKDOWN"

expect_status 2 "$SCRIPT_DIR/lint.sh" --unknown
expect_commands

expect_status 2 "$SCRIPT_DIR/lint.sh" --quick extra
expect_commands

expect_status 0 env LINT_TEST_CLIPPY_STATUS=0 "$SCRIPT_DIR/lint.sh"
expect_commands "$FMT" "$CLIPPY" "$MARKDOWN"

# Full validation runs tests only after every lint check passes.
expect_status 0 env LINT_TEST_CLIPPY_STATUS=0 "$SCRIPT_DIR/check.sh"
expect_commands "$FMT" "$CLIPPY" "$MARKDOWN" "$TESTS"

expect_status 2 "$SCRIPT_DIR/check.sh"
expect_commands "$FMT" "$CLIPPY"

expect_status 2 env LINT_TEST_CLIPPY_STATUS=0 LINT_TEST_NPX_STATUS=1 \
    "$SCRIPT_DIR/check.sh"
expect_commands "$FMT" "$CLIPPY" "$MARKDOWN"

expect_status 7 env LINT_TEST_CLIPPY_STATUS=0 LINT_TEST_NEXTEST_STATUS=7 \
    "$SCRIPT_DIR/check.sh"
expect_commands "$FMT" "$CLIPPY" "$MARKDOWN" "$TESTS"

# Exercise real commits without touching the caller's index or hooks.
(
    cd "$TEMP_DIR"
    git init -q repo
    cd repo
    git config user.name 'Hook Test'
    git config user.email 'hook-test@example.invalid'
    git config commit.gpgsign false
    git config core.hooksPath .ci/githooks
    git config core.whitespace trailing-space,space-before-tab
    mkdir -p .ci/githooks
    cp "$SCRIPT_DIR/lint.sh" .ci/lint.sh
    cp "$SCRIPT_DIR/githooks/pre-commit" .ci/githooks/pre-commit

    printf 'clean\n' >sample.txt
    git add sample.txt
    expect_status 0 git commit -qm 'Clean commit'
    expect_commands "$FMT" "$MARKDOWN"
    initial_head="$(git rev-parse HEAD)"

    printf 'trailing space \n' >sample.txt
    git add sample.txt
    expect_status 1 git commit -qm 'Whitespace must fail'
    expect_commands

    printf 'clean change\n' >sample.txt
    git add sample.txt
    expect_status 1 env LINT_TEST_FMT_STATUS=1 git commit -qm 'Format must fail'
    expect_commands "$FMT"

    expect_status 1 env LINT_TEST_NPX_STATUS=1 git commit -qm 'Markdown must fail'
    expect_commands "$FMT" "$MARKDOWN"
    test "$(git rev-parse HEAD)" = "$initial_head"
)

echo "Lint, full validation, and pre-commit tests passed!"
