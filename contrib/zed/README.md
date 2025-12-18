# Tribute Language Support for Zed

Zed editor extension providing syntax highlighting for the Tribute language.

## Installation

```bash
cd contrib/zed
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

The tree-sitter grammar is fetched from the external repository:
https://github.com/Kroisse/tree-sitter-tribute

To update syntax highlighting queries, edit `languages/tribute/highlights.scm`.

## Structure

```
contrib/zed/
├── extension.toml           # Extension metadata (references external grammar)
├── develop.sh               # Symlink extension for development
└── languages/
    └── tribute/
        ├── config.toml      # Language configuration
        └── highlights.scm   # Syntax highlighting queries
```
