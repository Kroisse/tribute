# Tribute Agent Notes

## Source of Truth

The documents under `new-plans/` are authoritative for Tribute's current
language and compiler design. If code, tests, or older docs conflict with
`new-plans/`, follow `new-plans/` and align the implementation afterward.

One IR operation's own contract (its shape, meaning, verification, ownership,
and location) is the doc comment on its `#[dialect]` declaration, which is
authoritative for that operation. Rules that span operations stay in
`new-plans/`.

Key design docs:

- `new-plans/design.md`
- `new-plans/syntax.md`
- `new-plans/types.md`
- `new-plans/io.md`
- `new-plans/host-interface.md`
- `new-plans/modules.md`
- `new-plans/abilities.md`
- `new-plans/type-inference.md`
- `new-plans/ir.md`
- `new-plans/implementation.md`
- `new-plans/cranelift-backend.md`
- `new-plans/runtime-types.md`

## Working Rules

- Prefer Korean prose in `new-plans/`; retain code identifiers and technical
  terms where useful.
- When changing language, type, ability, module, IR, or backend rules, update
  the relevant `new-plans/*.md` file first. When changing one operation's
  contract, update its `#[dialect]` declaration's doc comment first.
- Design documents under `new-plans/` must stand independently of work progress:
  state durable contracts and compiler-phase ownership, not issue or PR numbers,
  temporary branches or commits, or task sequencing and status. Track
  implementation and handoff work in GitHub.
- Use `guides/development/` for crate structure, pipeline, IR, and code
  convention reference material. `.claude/rules` links to this directory.
- Use the `tribute-testing` skill for test commands, Salsa test patterns,
  nextest, and snapshot testing.
- Implementation work is tracked in GitHub Issues. Use `gh issue list` when
  looking for open work.

## Common Commands

```bash
cargo build
cargo test
cargo run -- --log=debug compile file.trb
cargo run -- --log=tribute_front::typeck=trace compile file.trb
```

## Review guidelines

- Flag code that contradicts `new-plans/`, and rule changes that do not update
  the relevant `new-plans/*.md` file in the same PR. For one operation's
  contract, the doc comment on its `#[dialect]` declaration plays that role.
- Flag `new-plans/` text that cites issue or PR numbers, branches, or task
  status.
- Flag language or frontend rules stated in terms of calling convention,
  convention class, or convention entry; those belong to the execution
  strategy (`new-plans/design.md`).
- Flag ownership or callable ABI inferred from symbol spelling, body shape, or
  an erased pointer type. Typed wrappers, interfaces, and declared signatures
  carry that information (`guides/development/conventions.md`).
- For native RC changes, check that every acquire has a matching release or
  transfer on each path, and that the change keeps the acyclic object graph
  contract in `new-plans/rc.md`. Ask for a sanitizer test when ownership
  changes.
- Flag language rules special-cased for the prelude instead of written in the
  prelude source.
- Prefer textual IR fixtures over programmatic IR construction in tests.
- Do not comment on formatting or lint findings; `.ci/lint.sh` checks them.

## Tree-sitter

The tree-sitter grammar source of truth is maintained separately:
<https://github.com/Kroisse/tree-sitter-tribute/blob/main/grammar.js>
