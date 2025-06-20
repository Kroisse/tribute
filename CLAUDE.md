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
cargo test -p tribute-runtime

# Run the interpreter on a .trb file
cargo run --bin trbi <file.trb>

# Run the compiler (planned, not yet implemented)
cargo run --bin trbc <file.trb>
```

### Tree-sitter Grammar Development
```bash
# Generate parser from grammar.js (automatic during build)
# Manual generation: run from crates/tree-sitter-tribute/
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
- **crates/tribute-ast/**: AST definitions with Salsa integration and unified diagnostics
- **crates/tribute-hir/**: High-level IR with Salsa-integrated lowering
- **crates/tribute-cranelift/**: Cranelift-based native code compiler (in development)
- **crates/tribute-runtime/**: Runtime library for compiled Tribute programs
- **crates/tree-sitter-tribute/**: Grammar definition and Tree-sitter integration

### Core Modules

#### Main Crate (`tribute`)
- **`src/eval.rs`**: HIR-based evaluator with complete function call support and lexical scoping
- **`src/builtins.rs`**: Built-in functions (print_line, input_line)
- **`src/lib.rs`**: Main API exposing `parse_str()` and `eval_str()` functions for HIR-based operations
- **`src/bin/trbi.rs`**: Interpreter binary (uses HIR-based evaluation)
- **`src/bin/trbc.rs`**: Compiler binary (in development, uses Cranelift backend)

#### AST Crate (`tribute-ast`)
- **`ast.rs`**: Salsa-tracked AST types (`Program`, `Item`, `Expr`)
- **`parser.rs`**: `TributeParser` wrapper around Tree-sitter
- **`database.rs`**: Salsa database definitions with unified diagnostics

#### HIR Crate (`tribute-hir`)
- **`hir.rs`**: High-level IR types (`HirProgram`, `HirFunction`, `HirExpr`)
- **`lower.rs`**: AST to HIR lowering implementation
- **`queries.rs`**: Salsa query definitions for HIR operations

#### Cranelift Crate (`tribute-cranelift`)
- **`compiler.rs`**: Main compiler interface and module creation
- **`codegen.rs`**: HIR to Cranelift IR translation
- **`runtime.rs`**: Runtime function declarations
- **`types.rs`**: Type system mapping for dynamic values
- **`errors.rs`**: Compilation error types

#### Runtime Crate (`tribute-runtime`)
- **`value.rs`**: TrValue struct and dynamic type system with TrHandle for GC compatibility
- **`memory.rs`**: Memory management and allocation functions (tr_value_* API)
- **`arithmetic.rs`**: Dynamic arithmetic operations (tr_value_add, tr_value_sub, etc.)
- **`string_ops.rs`**: String manipulation and interpolation (tr_string_* API)
- **`builtins.rs`**: Built-in functions (tr_builtin_* API)

### Language Characteristics
- **Syntax**: Modern C-like syntax `fn name(args) { body }` (transitioned from Lisp S-expressions)
- **File extension**: `.trb`
- **Types**: Numbers, strings with interpolation, identifiers
- **Core features**: Function definitions, let bindings, arithmetic, pattern matching
- **String interpolation**: `"\{expr}"` syntax
- **Built-ins**: `print_line`, `input_line`

### Tree-sitter Integration
- **Grammar location**: `crates/tree-sitter-tribute/grammar.js`
- **Rust bindings**: `crates/tree-sitter-tribute/src-rs/lib.rs` (note: not in `src/` to avoid .gitignore)
- **Scope**: `source.tribute` for editor integration
- **Automatic generation**: Parser is generated automatically during build
- **String interpolation support**: Full grammar support for `"\{expr}"` syntax

## Development Notes

### Testing Strategy
- **HIR Snapshot testing**: Uses `insta` crate for regression testing of HIR structures (preferred)
- **HIR Evaluation tests**: Unit tests in `src/eval.rs` using `TributeDatabaseImpl::default().attach()`
- **Tree-sitter corpus**: Tests in `crates/tree-sitter-tribute/test/corpus/` for grammar validation
- **Language examples**: All `.trb` files in `lang-examples/` are tested via HIR snapshot tests
- **Comprehensive coverage**: 39 tests across all crates with dedicated string interpolation testing

### Recent Architecture Changes
- **Salsa Integration**: The project now uses Salsa for incremental computation with tracked AST/HIR types
- **Multi-level IR**: Implemented AST → HIR transformation pipeline for future optimizations
- **HIR-based Evaluation**: Complete transition to HIR-based evaluation as the primary execution path
- **Unified Diagnostics**: All compilation phases report errors through a centralized `Diagnostic` system
- **Automatic Parser Generation**: Tree-sitter parser is now generated automatically during build
- **Workspace Reorganization**: Moved to conventional `crates/` directory structure
- **Cranelift Compiler**: Replaced MLIR with Cranelift for native code compilation (in development)
- **Runtime Library Refactoring**: Adopted handle-based API with `Tr` prefix for GC compatibility

### Dependencies
- **`tree-sitter`**: Core parsing infrastructure
- **`insta`**: Snapshot testing framework
- **`salsa`**: Incremental computation framework for compiler infrastructure
- **`serde`**: Serialization support
- **`cranelift`**: Code generation backend for native compilation
- **`target-lexicon`**: Target triple handling for cross-compilation

### Dependency Management
- **Workspace Dependencies**: All shared dependencies are centralized in the root `Cargo.toml` under `[workspace.dependencies]`
- **Local References**: Individual crates reference workspace dependencies using `.workspace = true` (e.g., `cranelift-codegen.workspace = true`)
- **Consistency**: This ensures consistent versions across all crates and simplifies dependency management

### Error Handling
- **Error Derive**: Use `derive_more::Display` instead of `thiserror` for error types
- **Standard Pattern**: Combine `#[derive(Display, Debug)]` for custom error enums plus manual `std::error::Error` impl
- **Display Format**: Use `#[display("Message: {_0}")]` syntax for error formatting
- **Manual Implementations**: Implement `From` traits and `std::error::Error` manually for proper error chaining

### API Usage
```rust
// Primary API (HIR-based)
use tribute::TributeDatabaseImpl;

// For parsing
let db = TributeDatabaseImpl::default();
let (program, diagnostics) = tribute::parse_str(&db, "program.trb", source);
for item in program.items(&db) {
    // Process items
}

// For evaluation
let db = TributeDatabaseImpl::default();
match tribute::eval_str(&db, "program.trb", source) {
    Ok(result) => println!("Result: {:?}", result),
    Err(e) => eprintln!("Error: {}", e),
}

// For tests (using attach pattern)
use salsa::Database;
TributeDatabaseImpl::default().attach(|db| {
    let result = tribute::eval_str(db, "test.trb", source);
    assert_debug_snapshot!(result);
});
```

### Additional Documentation
- **`guides/salsa.md`**: Detailed guide on Salsa integration with architecture diagrams and examples
- **`plans/`**: Development plans for major features:
  - `01-syntax-modernization.md`: Modernizing syntax from S-expressions (✅ **COMPLETED**)
    - `01.01-string-interpolation.md`: String interpolation feature with `"\{expr}"` syntax (✅ **COMPLETED**)
  - `02-compiler-implementation.md`: MLIR-based native compilation (📋 **PLANNED**)
  - `03-lsp-implementation.md`: Language Server Protocol for IDE support (📋 **PLANNED**)
  - `04-static-type-system.md`: Gradual static typing system (📋 **PLANNED**)
  - `05-standard-library.md`: Core library implementation (📋 **PLANNED**)
  - `06-package-manager.md`: Package management system (📋 **PLANNED**)

## Implementation Status

### ✅ **Completed Features** (Ready for Production Use)
- **Modern Syntax**: Complete transition from Lisp S-expressions to C-like syntax
- **String Interpolation**: Full `"\{expr}"` syntax with expression evaluation
- **Multi-level IR**: AST → HIR transformation pipeline with Salsa integration
- **Core Language**: Function definitions, let bindings, arithmetic, built-ins
- **Infrastructure**: Tree-sitter grammar, comprehensive testing, workspace organization

### 🚧 **In Development**
- **Native Compilation**: Cranelift-based AOT compiler (replacing MLIR approach)
  - ✅ Basic infrastructure complete
  - ✅ Runtime function declarations defined
  - ✅ Complete runtime library with C-compatible ABI and handle-based API
  - ✅ Dynamic value system (TrValue/TrHandle) and memory management
  - ✅ Arithmetic and string operations with `tr_` prefix convention
  - 🔄 HIR → Cranelift IR translation in progress

### 📋 **Planned Features** (Detailed Plans Available)
- **LSP Support**: IDE integration with language server protocol (Plan 03)
- **Static Types**: Gradual static typing system (Plan 04)
- **Standard Library**: Collections, I/O, and common utilities (Plan 05)
- **Package Manager**: Dependency management and module system (Plan 06)

### ⚠️ **Known Issues**
- Tree-sitter grammar tests may have token naming inconsistencies
- Pattern matching evaluation is basic (parsing works, evaluation needs completion)

## Runtime Library Conventions

### Naming Convention
The runtime library uses a consistent `tr_` prefix for all public C ABI functions:
- **Value Operations**: `tr_value_*` (e.g., `tr_value_new`, `tr_value_free`, `tr_value_add`)
- **String Operations**: `tr_string_*` (e.g., `tr_string_concat`, `tr_string_length`)  
- **Built-in Functions**: `tr_builtin_*` (e.g., `tr_builtin_print_line`, `tr_builtin_input_line`)
- **Runtime Management**: `tr_runtime_*` (e.g., `tr_runtime_init`, `tr_runtime_cleanup`)

### Handle-Based API
- **TrHandle**: GC-compatible handle type that wraps `*mut TrValue`
- **Future-Proof**: When GC is implemented, handles can be updated to use indirection tables
- **Safety**: All C functions accept/return handles instead of raw pointers
- **Memory Management**: `tr_value_free()` handles proper cleanup including string deallocation

### Core Types
- **TrValue**: Main runtime value type (replaces TributeValue)
- **TrString**: Runtime string representation (replaces TributeString)
- **TrHandle**: GC-compatible handle wrapper
- **ValueTag**: Type discrimination enum (Number, String, Unit)

When working on this codebase, pay attention to:
- The separation between parsing (Tree-sitter), AST, HIR, and evaluation phases
- Using the Salsa database for any new compiler features
- HIR evaluation is the primary and only execution path (AST evaluation has been fully replaced)
- Use `TributeDatabaseImpl::default().attach()` pattern only for tests and snapshot testing
- For production code, create database instances normally: `TributeDatabaseImpl::default()`
- Use the simplified `eval_str()` and `parse_str()` API for all HIR-based operations
- Runtime library functions use `tr_` prefix and handle-based API for GC compatibility
