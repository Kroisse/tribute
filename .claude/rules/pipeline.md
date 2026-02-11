# Compilation Pipeline

The Tribute compiler uses a multi-stage pipeline, orchestrated in `src/pipeline.rs`.

See the detailed pipeline diagram in the doc comment at the top of `src/pipeline.rs`.

## Pipeline Overview

The pipeline is divided into three main phases:

1. **Frontend (tribute-front)**: CST → AST → resolve → typecheck → TDNR → TrunkIR
2. **TrunkIR Passes (tribute-passes)**:
   Boxing → Evidence → Closures → Continuations
3. **Backend** (target-specific):
   - **WASM**: cont_to_trampoline → lower_to_wasm → emit_wasm
   - **Native**: cont_to_libmprompt → lower_to_clif → emit_native (Cranelift)

## Frontend Stages (tribute-front)

| Stage | Function | Description |
| ----- | -------- | ----------- |
| parse_cst | `parse_cst(db, source)` | Parse source to CST using Tree-sitter |
| astgen | `parsed_ast(db, source)` | Convert CST to AST |
| resolve | `resolve_with_env(db, ast, env, spans)` | Resolve names at AST level |
| typecheck | `TypeChecker::check_module(ast)` | Bidirectional type inference with effects |
| tdnr | `resolve_tdnr(db, ast)` | Type-directed name resolution for UFCS |
| ast_to_ir | `lower_ast_to_ir(db, ast, ...)` | Lower AST to TrunkIR |

## TrunkIR Passes (tribute-passes)

| Stage | Function | Description |
| ----- | -------- | ----------- |
| boxing | `insert_boxing(db, module)` | Insert explicit box/unbox operations |
| evidence_params | `add_evidence_params(db, module)` | Add evidence params to effectful functions |
| closure_lower | `lower_closures(db, module)` | Lower `closure.*` to function calls |
| evidence_calls | `transform_evidence_calls(db, module)` | Pass evidence through call sites |
| resolve_evidence | `resolve_evidence_dispatch(db, module)` | Resolve evidence and lower `ability.*` to `cont.*` |
| cont_to_trampoline | `lower_cont_to_trampoline(db, module)` | Lower `cont.*` to trampoline ops |
| dce | `eliminate_dead_functions(db, module)` | Dead code elimination |
| resolve_casts | `resolve_unrealized_casts(db, module, ...)` | Resolve type casts |

## Native Pipeline (trunk-ir-cranelift-backend)

| Stage | Function | Description |
| ----- | -------- | ----------- |
| cont_to_libmprompt | `lower_cont_to_libmprompt(db, module)` | Lower `cont.*` via libmprompt FFI calls |
| lower_to_clif | `lower_to_clif(db, module)` | Lower `func.*`/`arith.*`/`scf.*`/`adt.*` to `clif.*` |
| compile_to_native | `emit_module_to_native(db, module)` | Emit `clif.*` to native object file via Cranelift |

## Entry Point

The main compilation entry points are:

```rust
pub fn compile_ast(db: &dyn salsa::Database, source_file: SourceCst) -> Result<Module, ConversionError>
pub fn compile_with_diagnostics(db: &dyn salsa::Database, source_file: SourceCst) -> CompilationResult
pub fn compile_to_native_binary(db: &dyn salsa::Database, source: SourceCst) -> Option<Vec<u8>>
```

Returns `CompilationResult` containing:

- `module`: Final TrunkIR module
- `solver`: TypeSolver with type information
- `diagnostics`: Collected errors and warnings
