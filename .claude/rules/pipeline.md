# Compilation Pipeline

The Tribute compiler uses a multi-stage pipeline, orchestrated in `src/pipeline.rs`.

See the detailed pipeline diagram in the doc comment at the top of `src/pipeline.rs`.

## Key Stages

| Stage | Function | Description |
|-------|----------|-------------|
| parse_cst | `parse_cst(db, source)` | Parse source to CST using Tree-sitter |
| lower_cst | `lower_cst(db, source, cst)` | Convert CST to TrunkIR with `tribute.*` ops |
| resolve | `stage_resolve(db, module)` | Resolve names: `tribute.path` → `func.call`, etc. |
| typecheck | `stage_typecheck(db, module)` | Bidirectional type inference with effects |
| tdnr | `stage_tdnr(db, module)` | Type-directed name resolution for UFCS |
| const_inline | `stage_const_inline(db, module)` | Inline constant values |
| inline_refs | `stage_inline_refs(db, module)` | Inline `tribute.ref` operations |

## Entry Point

The main compilation entry points are:
```rust
pub fn compile(db: &dyn salsa::Database, source_file: SourceCst) -> Module
pub fn compile_with_diagnostics(db: &dyn salsa::Database, source_file: SourceCst) -> CompilationResult
```

Returns `CompilationResult` containing:
- `module`: Final TrunkIR module
- `solver`: TypeSolver with type information
- `diagnostics`: Collected errors and warnings
