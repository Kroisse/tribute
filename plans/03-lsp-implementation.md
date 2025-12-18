# LSP Implementation Plan

## Overview

Implementation of a Language Server Protocol (LSP) server for Tribute to provide IDE support in VSCode and other editors.

## Status: 🚧 In Progress

Located at `src/lsp/`.

## Current Implementation

Using `lsp-server` crate (synchronous, simple).

### ✅ Completed Features

- **Server lifecycle**: Initialize/shutdown handling
- **Document synchronization**: Open/change/close with full sync
- **Diagnostics**: Real-time error/warning reporting via Salsa
- **Hover**: Type information display at cursor position
- **Salsa integration**: Incremental compilation for diagnostics

### 🔲 Planned Features

**Navigation:**
- Go to definition
- Find references
- Document symbols
- Workspace symbols

**Code Intelligence:**
- Auto-completion
- Signature help
- Effect information in hover
- Code actions (quick fixes)

**Refactoring:**
- Rename symbol

**Editor Extensions:**
- VSCode extension
- Zed extension exists at `contrib/zed/` (syntax highlighting via Tree-sitter)

## Architecture

```
Editor ←→ LSP Client ←→ Tribute LSP Server (src/lsp/)
                              ↓
                       Salsa Database
                       ↙           ↘
               tribute-ast    tribute-passes
```

Key modules:
- `server.rs` - Main LSP server loop and handlers
- `line_index.rs` - Line/column ↔ byte offset conversion
- `type_index.rs` - Type information index for hover
- `pretty.rs` - Type pretty-printing

## Future: Ability-aware Features

Once the ability system is implemented:
- Effect information in hover
- Handler suggestions for unhandled effects
- Effect propagation visualization
