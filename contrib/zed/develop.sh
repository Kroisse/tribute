#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXT_NAME="tribute"

# Determine Zed extensions directory
if [[ "$OSTYPE" == "darwin"* ]]; then
    ZED_EXT_DIR="$HOME/.config/zed/extensions/installed"
else
    ZED_EXT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/zed/extensions/installed"
fi

# Check if WASM exists
if [[ ! -f "$SCRIPT_DIR/grammars/tribute.wasm" ]]; then
    echo "Error: tribute.wasm not found. Run ./build.sh first."
    exit 1
fi

# Create extensions directory if needed
mkdir -p "$ZED_EXT_DIR"

# Remove existing installation
if [[ -e "$ZED_EXT_DIR/$EXT_NAME" ]]; then
    echo "Removing existing installation..."
    rm -rf "$ZED_EXT_DIR/$EXT_NAME"
fi

# Symlink for development
ln -s "$SCRIPT_DIR" "$ZED_EXT_DIR/$EXT_NAME"

echo "Linked! Extension at: $ZED_EXT_DIR/$EXT_NAME -> $SCRIPT_DIR"
echo "Restart Zed or run 'zed: reload extensions' to activate."
