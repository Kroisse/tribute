# Development Guides

These documents are the shared implementation reference for contributors and
coding agents. Edit them here; `.claude/rules` is a relative symbolic link to
this directory.

- [Code conventions](conventions.md): preferred implementation patterns,
  formatting helpers, and error handling.
- [Crate structure](crates.md): crate responsibilities and dependency direction.
- [TrunkIR](ir.md): dialect definitions and typed IR APIs.
- [Salsa](../salsa.md): query and incremental computation guidance.

Language and compiler design contracts are authoritative under
[`new-plans/`](../../new-plans/). These guides explain implementation practices;
[`AGENTS.md`](../../AGENTS.md) provides working rules and entry points.

## Further Reading

- [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/checklist.html)
  for naming, traits, and public API design.
- [The Rust Performance Book](https://nnethercote.github.io/perf-book/iterators.html)
  for iterator and allocation practices.
- [Pragmatic Rust Guidelines](https://microsoft.github.io/rust-guidelines/)
  for application and library engineering practices.
- [Clippy](https://doc.rust-lang.org/clippy/index.html)
  for mechanically checked idioms and common mistakes.
