#!/bin/sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

"$SCRIPT_DIR/lint.sh"

echo "Running tests..."
cargo nextest run --workspace

echo "All checks passed!"
