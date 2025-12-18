# Tribute Language Support for Zed

Zed editor extension providing syntax highlighting for the Tribute language.

## Installation

```bash
cd contrib/zed
./build.sh    # Build tree-sitter WASM
./develop.sh  # Symlink to Zed extensions directory
```

Then restart Zed or run `zed: reload extensions`.

## File Types

- `.trb` - Tribute source files

## Features

- Syntax highlighting via Tree-sitter
- Bracket matching and auto-closing
- Line and block comment support

## Development

To update syntax highlighting:

1. Edit `crates/tree-sitter-tribute/queries/highlights.scm`
2. Copy to `contrib/zed/languages/tribute/highlights.scm`
3. Run `./build.sh` and reload the extension in Zed

## Structure

```
contrib/zed/
├── extension.toml           # Extension metadata
├── build.sh                 # Build WASM from tree-sitter grammar
├── develop.sh               # Symlink extension for development
├── grammars/
│   └── tribute.wasm         # Tree-sitter WASM (built, not committed)
└── languages/
    └── tribute/
        ├── config.toml      # Language configuration
        └── highlights.scm   # Syntax highlighting queries
```
