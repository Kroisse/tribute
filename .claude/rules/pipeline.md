# Compilation Pipeline

The Tribute compiler uses a 5-stage pipeline, orchestrated in `src/pipeline.rs`.

## Pipeline Stages

### 1. parse_cst
**Function**: `parse_cst(db, source_file) -> ParsedCst`

Parses source code into a Concrete Syntax Tree using Tree-sitter.
- Input: `SourceFile` (Salsa input)
- Output: `ParsedCst` (reference-counted, cached)
- Parser: `tree-sitter-tribute` grammar

### 2. lower_cst
**Function**: `lower_cst(db, source_file, cst) -> Module`

Converts CST to TrunkIR with unresolved `src.*` operations.
- Input: `ParsedCst`
- Output: `Module` with `src.var`, `src.call`, `src.path`, `src.binop`, etc.
- Location: `crates/tribute-front/src/tirgen/`

### 3. stage_resolve
**Function**: `stage_resolve(db, source_file) -> Module`

Resolves names and converts `src.*` operations to concrete ops.
- Builds `ModuleEnv` (collects definitions)
- Resolves bindings: Function, Constructor, TypeDef
- Transforms: `src.var` → `func.call`, `src.call` → `func.call`, etc.
- Location: `crates/tribute-passes/src/resolve.rs`

### 4. stage_typecheck
**Function**: `stage_typecheck(db, source_file) -> Module`

Performs bidirectional type inference with row-polymorphic effects.
- `TypeChecker` walks module, collects `Constraint`s
- `TypeSolver` unifies constraints (union-find algorithm)
- Applies substitutions to get concrete types
- Location: `crates/tribute-passes/src/typeck/`

### 5. stage_tdnr
**Function**: `stage_tdnr(db, source_file) -> Module`

Type-directed name resolution for UFCS method calls.
- Resolves `expr.method(args)` → `Type::method(expr, args)`
- Uses inferred type information from stage 4
- Location: `crates/tribute-passes/src/tdnr.rs`

## Pipeline Flow

```
Source File (.trb)
    │
    ▼ parse_cst
ParsedCst (Tree-sitter tree)
    │
    ▼ lower_cst
Module [src.*, func.*, adt.*, arith.*]
    │
    ▼ stage_resolve
Module [func.*, adt.*, arith.*, case.*]
    │
    ▼ stage_typecheck
Module (with inferred types)
    │
    ▼ stage_tdnr
Fully Typed, Resolved Module
```

## Entry Point

The main compilation entry points are:
```rust
pub fn compile(db: &dyn salsa::Database, source_file: SourceFile) -> Module
pub fn compile_with_diagnostics(db: &dyn salsa::Database, source_file: SourceFile) -> CompilationResult
```

Returns `CompilationResult` containing:
- `module`: Final TrunkIR module
- `solver`: TypeSolver with type information
- `diagnostics`: Collected errors and warnings
