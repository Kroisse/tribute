# Tribute Language Support for Zed

Zed editor extension providing syntax highlighting and LSP support for the Tribute language.

## Installation

### For Development

1. Build the tribute binary and ensure it's in your PATH:
   ```bash
   cargo install --path .
   ```

2. Install the extension as a dev extension:
   ```bash
   cd contrib/zed
   ./develop.sh  # Symlink to Zed extensions directory
   ```

3. Restart Zed or run `zed: reload extensions`.

## File Types

- `.trb` - Tribute source files

## Features

- Syntax highlighting via Tree-sitter
- Bracket matching and auto-closing
- Line and block comment support
- Language Server Protocol (LSP) support:
  - Hover information (type display)
  - Diagnostics (errors and warnings)

## Development

The tree-sitter grammar is fetched from the external repository:
https://github.com/Kroisse/tree-sitter-tribute

To update syntax highlighting queries, edit `languages/tribute/highlights.scm`.

## Structure

```
contrib/zed/
├── Cargo.toml               # Rust extension dependencies
├── extension.toml           # Extension metadata
├── develop.sh               # Symlink extension for development
├── src/
│   └── lib.rs               # Language server integration
└── languages/
    └── tribute/
        ├── config.toml      # Language configuration
        └── highlights.scm   # Syntax highlighting queries
```
