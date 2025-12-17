# Tribute (Compiler) — Agent Notes

## Source of Truth (Language + Compiler Design)

The documents under `new-plans/` are the **source of truth** for Tribute’s current language and compiler design. If code/tests/older docs conflict, **treat `new-plans/` as authoritative** and align implementation afterwards.

- `new-plans/design.md`: high-level goals and direction
- `new-plans/syntax.md`: surface syntax (keywords, literals, `case`, `handle`, UFCS, etc.)
- `new-plans/types.md`: `struct`/`enum`, records, UFCS/field access rules
- `new-plans/modules.md`: module system (`::`), `use`, name resolution
- `new-plans/abilities.md`: abilities (= algebraic effects), syntax and semantics
- `new-plans/type-inference.md`: row-polymorphic effect typing + bidirectional typing
- `new-plans/ir.md`: TrunkIR (multi-level dialect IR) and pipeline
- `new-plans/implementation.md`: ability implementation strategy (evidence passing, prompts/continuations)

## Project Overview (Updated)

Tribute targets a functional language with **static types, type inference, and abilities (algebraic effects)**.

- **Syntax style**: ML semantics with C/Rust-style syntax (`fn`, `{}`, `;`, `struct`/`enum`, `::`, UFCS `.`)
- **Effects/Abilities**: Unison-style abilities (delimited, one-shot continuations)
- **IR**: TrunkIR multi-level dialect IR (`src`/`ability`/`adt` → `cont`/`func`/`scf` …)
- **Targets**: Cranelift (native) + WasmGC
- **File extension (spec)**: `.tr`
  - This repo may still contain legacy `.trb` examples (migration in progress).

## Repo Layout (Current Workspace)

This is a Rust Cargo workspace; the codebase is being incrementally aligned to the `new-plans/` spec.

- `src/bin/trbi.rs`: (legacy) interpreter / runner driver
- `src/bin/trbc.rs`: (legacy/experimental) compiler driver
- `crates/tree-sitter-tribute/`: Tree-sitter grammar and bindings
- `crates/tribute-core/`: shared infra (Salsa DB `TributeDatabaseImpl`, `TargetInfo`, `Diagnostic`, etc.)
- `crates/tribute-ast/`: AST definitions and parsing
- `crates/tribute-passes/`: compiler passes (AST → TrunkIR lowering)
- `crates/tribute-trunk-ir/`: TrunkIR dialect system and IR definitions
- `crates/tribute-cranelift/`: Cranelift backend (in progress)

## Commit Guidelines

- Follow the Conventional Commits format.

## Common Commands

```bash
# Build / Test
cargo build
cargo test

# If snapshot tests fail (insta)
cargo insta review

# Package-specific tests
cargo test -p tribute
cargo test -p tree-sitter-tribute
cargo test -p tribute-passes
cargo test -p tribute-trunk-ir

# Legacy drivers (example inputs may still be .trb today)
cargo run --bin trbi -- <file.trb>
cargo run --bin trbc -- <file.trb>
```

## Tree-sitter Grammar Development (Repo Tooling)

```bash
# Manual generation: run from crates/tree-sitter-tribute/
tree-sitter generate
tree-sitter test
tree-sitter build-wasm
```

## Conventions / Notes for Agents

- When changing language/type/ability/module rules: update `new-plans/*.md` first, then align implementation/tests.
- Type/effect terminology:
  - Omitting an effect annotation: `fn(a) -> b` desugars to `fn(a) ->{e} b` (fresh row variable).
  - Pure functions are written explicitly as `->{}`.
  - Module separator is `::`; method-style calls use UFCS `.`.
- Prefer `derive_more` (`Display`/`Error`/`From`) for Rust error types when appropriate.
