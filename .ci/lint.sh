#!/bin/sh

set -e

echo "Running cargo fmt..."
if ! cargo fmt --all --check; then
    echo ""
    echo "❌ Code formatting check failed." >&2
    echo "   Run 'cargo fmt --all' to fix formatting issues." >&2
    exit 2
fi

echo "Running clippy..."
if ! cargo clippy --workspace --all-targets --message-format=short -- -D warnings 2>&1 | head -n 40; then
    echo ""
    echo "❌ Clippy failed with warnings/errors. Fix the issues above and try again." >&2
    echo "   Run 'cargo clippy --workspace --all-targets' to see all issues." >&2
    exit 2
fi

echo "✅ Lint checks passed!"
