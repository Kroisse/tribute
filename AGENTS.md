# Tribute Agent Notes

## Source of Truth

The documents under `new-plans/` are authoritative for Tribute's current
language and compiler design. If code, tests, or older docs conflict with
`new-plans/`, follow `new-plans/` and align the implementation afterward.

Key design docs:

- `new-plans/design.md`
- `new-plans/syntax.md`
- `new-plans/types.md`
- `new-plans/io.md`
- `new-plans/modules.md`
- `new-plans/abilities.md`
- `new-plans/type-inference.md`
- `new-plans/ir.md`
- `new-plans/implementation.md`
- `new-plans/cranelift-backend.md`

## Working Rules

- When changing language, type, ability, module, IR, or backend rules, update
  the relevant `new-plans/*.md` file first.
- Use `.claude/rules/` for crate structure, pipeline, IR, and code convention
  reference material.
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

## Tree-sitter

The tree-sitter grammar source of truth is maintained separately:
<https://github.com/Kroisse/tree-sitter-tribute/blob/main/grammar.js>
