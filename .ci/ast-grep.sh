#!/bin/sh

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

if ! command -v ast-grep >/dev/null 2>&1; then
    echo "Install ast-grep: cargo binstall ast-grep --version 0.45.3 --no-confirm" >&2
    exit 2
fi

ast-grep test --skip-snapshot-tests
ast-grep scan src tests crates
