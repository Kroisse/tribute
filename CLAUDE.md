# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tribute is a Lisp-like programming language interpreter written in Rust. The project uses Tree-sitter for parsing and Salsa for incremental computation. It is organized as a Cargo workspace with multiple crates implementing a multi-level IR architecture.

## Commit Guidelines

- Please follow the Conventional Commit format for commit messages.

## Common Commands

### Building and Testing
```bash
# Build the entire workspace
cargo build

# Run all tests (includes snapshot tests for language examples)
cargo test

# Run tests for specific package
cargo test -p tribute
cargo test -p tree-sitter-tribute

# Run the interpreter on a .trb file
cargo run --bin trbi <file.trb>

# Run the compiler (planned, not yet implemented)
cargo run --bin trbc <file.trb>
```

### Tree-sitter Grammar Development
```bash
# Generate parser from grammar.js (automatic during build)
# Manual generation: run from tree-sitter-tribute/
tree-sitter generate

# Test the grammar against corpus
tree-sitter test  # or: npm run test

# Build WASM parser for web use
tree-sitter build-wasm  # or: npm run build
```

### Testing Language Examples
The `lang-examples/` directory contains `.trb` files that are automatically tested via snapshot tests. After modifying the parser or AST, run `cargo test` and review snapshot changes with `cargo insta review` if needed.

## Architecture

### Workspace Structure
- **Main package (/)**: Core interpreter (`tribute` crate)
- **tribute-ast/**: AST definitions with Salsa integration and unified diagnostics
- **tribute-hir/**: High-level IR with Salsa-integrated lowering
- **tree-sitter-tribute/**: Grammar definition and Tree-sitter integration

### Core Modules

#### Main Crate (`tribute`)
- **`src/eval.rs`**: Environment-based evaluator with lexical scoping
- **`src/builtins.rs`**: Built-in functions (print_line, input_line)
- **`src/lib.rs`**: Main API exposing `parse()` and `parse_with_database()` functions
- **`src/bin/trbi.rs`**: Interpreter binary
- **`src/bin/trbc.rs`**: Compiler binary (planned)

#### AST Crate (`tribute-ast`)
- **`ast.rs`**: Salsa-tracked AST types (`Program`, `Item`, `Expr`)
- **`parser.rs`**: `TributeParser` wrapper around Tree-sitter
- **`database.rs`**: Salsa database definitions with unified diagnostics

#### HIR Crate (`tribute-hir`)
- **`hir.rs`**: High-level IR types (`HirProgram`, `HirFunction`, `HirExpr`)
- **`lower.rs`**: AST to HIR lowering implementation
- **`queries.rs`**: Salsa query definitions for HIR operations

### Language Characteristics
- **Syntax**: Lisp-style S-expressions `(function arg1 arg2)`
- **File extension**: `.trb`
- **Types**: Numbers, strings, identifiers, lists
- **Special forms**: `fn`, `let` (function definitions partially implemented)
- **Built-ins**: `print_line`, `input_line`

### Tree-sitter Integration
- **Grammar location**: `tree-sitter-tribute/grammar.js`
- **Rust bindings**: `tree-sitter-tribute/src-rs/lib.rs` (note: not in `src/` to avoid .gitignore)
- **Scope**: `source.tribute` for editor integration
- **Identifier pattern**: Allows operators as valid names: `[a-zA-Z_+\-*\/][a-zA-Z0-9_+\-*\/]*`

## Development Notes

### Testing Strategy
- **Snapshot testing**: Uses `insta` crate for regression testing of parsed ASTs
- **Tree-sitter corpus**: Tests in `tree-sitter-tribute/test/corpus/` for grammar validation
- **Language examples**: `hello.trb` and `calc.trb` serve as integration test cases

### Recent Architecture Changes
- **Salsa Integration**: The project now uses Salsa for incremental computation with tracked AST/HIR types
- **Multi-level IR**: Implemented AST → HIR transformation pipeline for future optimizations
- **Unified Diagnostics**: All compilation phases report errors through a centralized `Diagnostic` system
- **Automatic Parser Generation**: Tree-sitter parser is now generated automatically during build

### Dependencies
- **`tree-sitter`**: Core parsing infrastructure
- **`insta`**: Snapshot testing framework
- **`salsa`**: Incremental computation framework for compiler infrastructure
- **`serde`**: Serialization support

### API Usage
```rust
// Legacy API (still supported)
let ast = tribute::parse("(print_line \"Hello\")")?;

// Salsa-based API (recommended)
use tribute::TributeDatabaseImpl;
let mut db = TributeDatabaseImpl::default();
let program = tribute::parse_with_database(&mut db, "program.trb", source)?;
for item in program.items(&db) {
    // Process items
}
```

### Additional Documentation
- **`SALSA_INTEGRATION.md`**: Detailed guide on Salsa integration with architecture diagrams and examples

When working on this codebase, pay attention to:
- The separation between parsing (Tree-sitter), AST, HIR, and evaluation phases
- Using the Salsa database for any new compiler features
- Maintaining compatibility with the legacy `parse()` API while preferring the database-based approach
