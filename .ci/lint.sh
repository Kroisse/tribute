#!/bin/sh

set -e

echo "Running cargo fmt..."
if ! cargo fmt --all --check; then
    echo ""
    echo "❌ Code formatting check failed."
    echo "   Run 'cargo fmt --all' to fix formatting issues."
    exit 2
fi

echo "Running clippy..."
if ! cargo clippy --workspace --all-targets --message-format=short -- -D warnings 2>&1 | head -n 40; then
    echo ""
    echo "❌ Clippy failed with warnings/errors. Fix the issues above and try again."
    echo "   Run 'cargo clippy --workspace --all-targets' to see all issues."
    exit 2
fi

echo "✅ Lint checks passed!"
