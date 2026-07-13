#!/bin/sh

set -e

echo "Running cargo fmt..."
if ! cargo fmt --all --check; then
    echo "" >&2
    echo "Code formatting check failed." >&2
    echo "Run 'cargo fmt --all' to fix formatting issues." >&2
    exit 2
fi

echo "Running clippy..."
CLIPPY_OUTPUT="$(mktemp)"
trap 'rm -f "$CLIPPY_OUTPUT"' EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
if cargo clippy --workspace --all-targets --message-format=short -- -D warnings \
    >"$CLIPPY_OUTPUT" 2>&1; then
    rm -f "$CLIPPY_OUTPUT"
    trap - EXIT HUP INT TERM
else
    tail -n 40 "$CLIPPY_OUTPUT" >&2
    rm -f "$CLIPPY_OUTPUT"
    trap - EXIT HUP INT TERM
    echo "" >&2
    echo "Clippy failed with warnings/errors. Fix the issues above and try again." >&2
    echo "Run 'cargo clippy --workspace --all-targets' to see all issues." >&2
    exit 2
fi

echo "Running markdownlint..."
if ! npx markdownlint-cli2 "**/*.md" "#node_modules"; then
    echo "" >&2
    echo "Markdown lint check failed." >&2
    echo "Run 'npx markdownlint-cli2 \"**/*.md\" \"#node_modules\"' to see all issues." >&2
    exit 2
fi

echo "Lint checks passed!"
