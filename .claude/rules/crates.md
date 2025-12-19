# Crate Structure

The Tribute compiler is organized as a Rust Cargo workspace with clearly separated responsibilities.

## tribute-core

**Role**: Shared compiler infrastructure and Salsa database

**Key Types**:
- `TributeDatabaseImpl` - Salsa database implementation
- `Diagnostic` - Compilation errors/warnings (phases: Parsing, TirGeneration, NameResolution, TypeChecking, Optimization)
- `SourceFile` - Salsa input for source file content
- `Span` - Source location (start, end character offsets)
- `PathId` - Interned file paths for efficient comparison
- `TargetInfo` - Platform info (triple, pointer size, endianness)

**Location**: `crates/tribute-core/`

## tribute-passes

**Role**: Compiler transformation passes from CST to TrunkIR

**Key Modules**:
- `pipeline.rs` - Compilation pipeline orchestration (5 stages)
- `tirgen/` - TrunkIR generation from CST
- `resolve.rs` - Name resolution (`src.*` ops → concrete ops)
- `typeck/` - Type checking (bidirectional, row-polymorphic effects)
- `tdnr.rs` - Type-directed name resolution (UFCS method calls)

**Location**: `crates/tribute-passes/`

## tribute-trunk-ir

**Role**: Multi-level dialect IR system (central IR representation)

**Key Modules**:
- `ir.rs` - Core IR structures (Operation, Value, Block, Region)
- `ops.rs` - Dialect operation traits and macros
- `types.rs` - Type system (interned types with attributes)
- `dialect/` - All dialect definitions

**Location**: `crates/tribute-trunk-ir/`

## tribute-cranelift

**Role**: Native code generation backend (work in progress)

**Status**: Pending TrunkIR → Cranelift IR lowering

**Location**: `crates/tribute-cranelift/`

## tribute (main crate)

**Role**: CLI entry point and LSP server

**Key Modules**:
- `cli.rs` - Command-line argument parsing (serve command)
- `lsp/` - LSP server (hover, diagnostics, document sync)

**Location**: `src/`

## Dependency Graph

```
tribute (main)
├── tribute-core
├── tribute-passes
│   ├── tribute-core
│   ├── tribute-trunk-ir
│   └── tree-sitter-tribute
└── tribute-trunk-ir
    └── tribute-core
```
