# Rust Lints

Run the shared lint entry point from the repository root:

```bash
.ci/lint.sh
```

Codex and Claude Stop hooks use this entry point. The Git pre-commit hook uses
`.ci/lint.sh --quick`, which skips Clippy. CI runs the same checks in separate
jobs.

Project-specific checks:

- [Symbol comparisons](symbol-lint.md): ast-grep rules, installation, and scope.
- [Hash maps and sets](hash-collections.md): collection conventions and the
  restrictions configured in the root `clippy.toml`.

To run Clippy alone:

```bash
cargo clippy --workspace --all-targets -- -D warnings
```
