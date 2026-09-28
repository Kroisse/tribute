# Tribute Language Support for Zed

Zed editor extension providing syntax highlighting and LSP support for the
Tribute language.

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
  - Document symbols (outline view)

## Configuration

### LSP Log Level

To adjust the LSP server log level, add the following to your Zed
`settings.json` (open with `Cmd+,` or `Ctrl+,`):

```json
{
  "lsp": {
    "tribute-lsp": {
      "settings": {
        "log_level": "debug"
      }
    }
  }
}
```

Available log levels:

- `"error"` - Only errors
- `"warn"` - Warnings and errors (default)
- `"info"` - General information
- `"debug"` - Detailed debugging information
- `"trace"` - Very detailed trace information

You can also use module-specific filters:

```json
{
  "lsp": {
    "tribute-lsp": {
      "settings": {
        "log_level": "tribute::lsp=trace,info"
      }
    }
  }
}
```

LSP logs can be viewed via `Cmd+Shift+P` → "lsp: open log".

## Development

The tree-sitter grammar is fetched from the external repository:
<https://github.com/Kroisse/tree-sitter-tribute>

The `rev` pinned under `[grammars.tribute]` in `extension.toml` must track the
grammar version used by the root `Cargo.toml` (`tree-sitter-tribute` `tag`).
When bumping the grammar, update both, then check that every query in
`languages/tribute/*.scm` still compiles against the new grammar — a query that
references a missing node or field fails to load, and Zed drops highlighting.
For example, from a checkout of the grammar at the pinned tag:

```bash
tree-sitter build
tree-sitter query path/to/contrib/zed/languages/tribute/highlights.scm path/to/file.trb
```

(`tree-sitter query` locates the grammar via the `parser-directories` in its
config; pass `--config-path` to point it at the checkout's parent directory.)
Compare node and field names against the grammar's `src/node-types.json`.

To update syntax highlighting queries, edit `languages/tribute/highlights.scm`.

## Structure

```text
contrib/zed/
├── Cargo.toml               # Rust extension dependencies
├── extension.toml           # Extension metadata
├── develop.sh               # Symlink extension for development
├── src/
│   └── lib.rs               # Language server integration
└── languages/
    └── tribute/
        ├── config.toml      # Language configuration
        ├── highlights.scm   # Syntax highlighting queries
        └── outline.scm      # Outline/symbol queries
```
