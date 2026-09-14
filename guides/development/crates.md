# Crate Structure

The Tribute compiler is organized as a Rust Cargo workspace.

## Design Principles

- **trunk-ir** is language-agnostic and must NOT depend on any tribute crate.
- **tribute-ir** contains Tribute-specific dialects and depends only on trunk-ir.
- **Backend crates** (trunk-ir-wasm-backend, trunk-ir-cranelift-backend)
  depend only on trunk-ir, keeping them language-agnostic.
- **tribute-passes** contains both shared and target-specific passes
  (native/, wasm/ subdirectories).

## Crates

| Crate | Role |
| ----- | ---- |
| `tribute` (src/) | CLI, LSP server, pipeline orchestration |
| `tribute-front` | Frontend: CST → AST → resolve → typecheck → TDNR → TrunkIR |
| `tribute-passes` | TrunkIR transformation passes (boxing, closures, effects, continuations, target-specific lowering) |
| `trunk-ir` | Language-agnostic multi-level dialect IR system |
| `tribute-ir` | Tribute-specific high-level dialects (ability, closure, tribute_rt) |
| `tribute-core` | Shared compiler utilities (TargetInfo, diagnostics) |
| `trunk-ir-wasm-backend` | WASM code generation via TrunkIR |
| `trunk-ir-cranelift-backend` | Native code generation via Cranelift |
| `tribute-runtime` | Runtime library for abilities/effects (static lib) |
| `trunk-ir-macros` | Proc macro for `#[dialect]` definitions |
| `tree-sitter-tribute` | Tree-sitter parser (external git dependency) |

## Dependency Graph

```text
tribute (main)
├── tribute-front
│   ├── trunk-ir
│   ├── tribute-ir
│   └── tree-sitter-tribute
├── tribute-passes
│   ├── trunk-ir
│   ├── tribute-ir
│   └── trunk-ir-wasm-backend
├── trunk-ir
├── trunk-ir-wasm-backend
│   └── trunk-ir
├── trunk-ir-cranelift-backend
│   └── trunk-ir
└── tribute-ir
    └── trunk-ir
```

Pipeline structure is documented in `src/pipeline.rs` (see top-of-file doc comment).
