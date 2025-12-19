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
- **File extension (spec)**: `.trb`
  - This repo may still contain legacy `.trb` examples (migration in progress).

## Codebase Structure

See `.claude/rules/` for detailed documentation:
- `crates.md` - Crate structure and responsibilities
- `pipeline.md` - Compilation pipeline stages
- `ir.md` - TrunkIR dialect system
- `conventions.md` - Code conventions and patterns

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

## Notes for Agents

- When changing language/type/ability/module rules: update `new-plans/*.md` first, then align implementation/tests.
- See `.claude/rules/conventions.md` for code patterns and conventions.
