# Crate Structure

The Tribute compiler is organized as a Rust Cargo workspace with clearly
separated responsibilities.

## tribute-core

**Role**: Shared compiler utilities (target info, future utilities)

**Key Types**:

- `TargetInfo` - Platform info (triple, pointer size, endianness)
- `Endianness` - Target byte order

**Location**: `crates/tribute-core/`

## tribute-passes

**Role**: TrunkIR transformation passes (boxing, closures, effects, continuations)

**Key Modules**:

- `boxing.rs` - Insert explicit box/unbox operations for polymorphism
- `closure_lower.rs` - Lower closure operations to function calls
- `evidence.rs` - Evidence parameter insertion for effect handling
- `resolve_evidence.rs` - Resolve evidence dispatch and lower `ability.*` to `cont.*`
- `cont_to_trampoline.rs` - Lower continuations to trampoline implementation

**Location**: `crates/tribute-passes/`

## tribute-front

**Role**: Front-end utilities (CST parsing, AST lowering, and text helpers)

**Key Modules**:

- `astgen/` - CST to AST lowering
- `ast/` - Salsa-tracked AST types with phase-parameterized name resolution
- `resolve.rs` - Name resolution (AST → AST)
- `typeck/` - Type checking (AST → AST)
- `tdnr.rs` - Type-directed name resolution (AST → AST)
- `ast_to_ir/` - AST to TrunkIR lowering
- `query.rs` - Salsa-tracked query functions (including CST parsing)
- `source_file.rs` - `SourceCst` input and URI helpers

**Location**: `crates/tribute-front/`

## trunk-ir

**Role**: Multi-level dialect IR system (central IR representation)

**Key Modules**:

- `ir.rs` - Core IR structures (Operation, Value, Block, Region)
- `ops.rs` - Dialect operation traits and macros
- `types.rs` - Type system (interned types with attributes)
- `dialect/` - All dialect definitions

**Location**: `crates/trunk-ir/`

## tribute-cranelift

**Role**: Native code generation backend (work in progress)

**Status**: Pending TrunkIR → Cranelift IR lowering

**Location**: `crates/tribute-cranelift/`

## tribute (main crate)

**Role**: CLI entry point, LSP server, and pipeline orchestration

**Key Modules**:

- `cli.rs` - Command-line argument parsing (serve command)
- `database.rs` - `TributeDatabaseImpl` and file loading cache
- `lsp/` - LSP server (hover, diagnostics, document sync)
- `pipeline.rs` - Compilation pipeline orchestration

**Location**: `src/`

## Dependency Graph

```text
tribute (main)
├── tribute-front
├── tribute-passes
│   ├── trunk-ir
├── trunk-ir
└── tree-sitter-tribute

tribute-front
├── trunk-ir
└── tree-sitter-tribute
```

Note: `trunk-ir` is now fully independent with no dependencies on other
tribute crates (not even as dev-dependencies). It's a standalone IR system.
