#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRAMMAR_DIR="$SCRIPT_DIR/../../crates/tree-sitter-tribute"
OUTPUT_DIR="$SCRIPT_DIR/grammars"

echo "Building tree-sitter WASM for Tribute..."

cd "$GRAMMAR_DIR"
tree-sitter build --wasm -o "$OUTPUT_DIR/tribute.wasm"

echo "Done! WASM built at: $OUTPUT_DIR/tribute.wasm"
