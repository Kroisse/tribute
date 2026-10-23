# Rust Lints

Run the shared lint entry point from the repository root:

```bash
.ci/lint.sh
```

Codex and Claude Stop hooks use this entry point. The Git pre-commit hook uses
`.ci/lint.sh --quick`, which skips Clippy and rustdoc. It runs the script of
the work tree being committed, so a linked worktree that shares the main
checkout's `core.hooksPath` lints its own files. CI runs the same checks in
separate jobs.

Project-specific checks:

- [Symbol comparisons](symbol-lint.md): ast-grep rules, installation, and scope.
- [Hash maps and sets](hash-collections.md): collection conventions and the
  restrictions configured in the root `clippy.toml`.

To run Clippy alone:

```bash
cargo clippy --workspace --all-targets -- -D warnings
```

Rustdoc warnings, such as broken intra-doc links or public docs linking to
private items, fail the full lint and the CI lint job. To run the check alone:

```bash
RUSTDOCFLAGS=-Dwarnings cargo doc --workspace --no-deps
```
