# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tribute is a Lisp-like programming language interpreter written in Rust. The project uses Tree-sitter for parsing and is organized as a Cargo workspace with two main packages: the core interpreter and the Tree-sitter grammar definition.

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

# Run the compiler on a .trb file
cargo run --bin trbc <file.trb>
```

### Tree-sitter Grammar Development
```bash
# Generate parser from grammar.js (run from tree-sitter-tribute/)
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
- **tree-sitter-tribute/**: Grammar definition and Tree-sitter integration

### Core Modules
- **`src/ast.rs`**: AST definitions (`Expr` enum, `SimpleSpan` for source locations)
- **`src/parser.rs`**: `TributeParser` wrapper around Tree-sitter 
- **`src/eval.rs`**: Environment-based evaluator with lexical scoping
- **`src/lib.rs`**: Main API exposing `parse()` function

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

### Parser Migration Status
The project recently migrated from Chumsky to Tree-sitter. The `TributeParser` in `src/parser.rs` converts Tree-sitter nodes to Tribute AST types. The evaluator (`src/eval.rs`) has basic functionality but function definitions are not fully implemented.

### Dependencies
- **`tree-sitter`**: Core parsing infrastructure
- **`insta`**: Snapshot testing framework  
- **`salsa`**: Incremental computation (likely for future IDE features)
- **`serde`**: Serialization support

When working on this codebase, pay attention to the separation between parsing (Tree-sitter) and evaluation (interpreter) concerns. The Tree-sitter grammar provides a solid foundation for editor tooling and robust parsing.