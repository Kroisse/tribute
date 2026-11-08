# Tribute Agent Notes

## Source of Truth

The documents under `design/` are authoritative for Tribute's current
language and compiler design. If code, tests, or older docs conflict with
`design/`, follow `design/` and align the implementation afterward.

One IR operation's own contract (its shape, meaning, verification, ownership,
and location) is the doc comment on its `#[dialect]` declaration, which is
authoritative for that operation. Rules that span operations stay in
`design/`.

Key design docs:

- `design/design.md`
- `design/syntax.md`
- `design/types.md`
- `design/io.md`
- `design/host-interface.md`
- `design/modules.md`
- `design/abilities.md`
- `design/type-inference.md`
- `design/ir.md`
- `design/tribute-dialects.md`
- `design/abi-boundary.md`
- `design/lowering.md`
- `design/implementation.md`
- `design/lsp.md`
- `design/cranelift-backend.md`
- `design/runtime-types.md`

## Working Rules

- Prefer Korean prose in `design/`; retain code identifiers and technical
  terms where useful.
- When changing language, type, ability, module, IR, or backend rules, update
  the relevant `design/*.md` file first. When changing one operation's
  contract, update its `#[dialect]` declaration's doc comment first.
- Design documents under `design/` must stand independently of work progress:
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

## Design principles

- Prefer principled, general mechanisms modeled on MLIR, LLVM, or rustc
  (e.g., RegionBranch-style interface queries, op traits) over hardcoded
  op-name checks or ad-hoc flags.
- Avoid `Rc<RefCell<...>>` shared-state designs. Prefer explicit ownership or
  borrowing (`&mut`).
- Prefer idiomatic Rust APIs (`Display`, `returns(as_ref)` for Option queries)
  over String-returning helpers.
- Package boundaries are isolation boundaries. Do not let passes such as
  specialization cross them implicitly.

## PRs and edits

- Base new PRs on origin/main by default. Only stack on an unmerged PR if the
  user asks.
- Use the Edit tool rather than `sed` for source changes, to avoid touching
  unrelated lines.
- When the user asks for a specific small change, make it first, then share
  any concerns about its value.

## Review guidelines

- Flag code that contradicts `design/`, and rule changes that do not update
  the relevant `design/*.md` file in the same PR. For one operation's
  contract, the doc comment on its `#[dialect]` declaration plays that role.
- Flag `design/` text that cites issue or PR numbers, branches, or task
  status.
- Flag language or frontend rules stated in terms of calling convention,
  convention class, or convention entry; those belong to the execution
  strategy (`design/design.md`).
- Flag ownership or callable ABI inferred from symbol spelling, body shape, or
  an erased pointer type. Typed wrappers, interfaces, and declared signatures
  carry that information (`guides/development/conventions.md`).
- For native RC changes, check that every acquire has a matching release or
  transfer on each path, and that the change keeps the acyclic object graph
  contract in `design/rc.md`. Ask for a sanitizer test when ownership
  changes.
- Flag language rules special-cased for the prelude instead of written in the
  prelude source.
- Prefer textual IR fixtures over programmatic IR construction in tests.
- Do not comment on formatting or lint findings; `.ci/lint.sh` checks them.

## Tree-sitter

The tree-sitter grammar source of truth is maintained separately:
<https://github.com/Kroisse/tree-sitter-tribute/blob/main/grammar.js>
