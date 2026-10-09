# Symbol Comparison Lint

An [ast-grep](https://ast-grep.github.io/) rule detects temporary
`Symbol::new("...")` calls used as operands of `==` or `!=`. The prototype is
tested with ast-grep 0.45.3 and needs no nightly compiler. Install that version
of the CLI, then run from the repository root:

```bash
cargo binstall ast-grep --version 0.45.3 --no-confirm
.ci/ast-grep.sh
```

The rule is in `.ast-grep/rules/no-symbol-new-in-comparison.yml`; positive and
negative cases are in `.ast-grep/tests/`. The tests check detection without
diagnostic snapshots. CI and `.ci/lint.sh` (including `--quick`) run the shared
`.ci/ast-grep.sh` entry point. Codex and Claude Stop hooks and Git pre-commit
therefore run the same checks. A missing CLI or failed check is an error.

## Existing comparisons

`.ast-grep/baseline.json` records the existing diagnostics as
`[rule ID, file, constructor expression, count]` rows. The gate rejects counts
exceeding this baseline, including another occurrence of the same expression
in the same file. Line numbers are omitted so unrelated line movement does not
invalidate the baseline. Diagnostic tool failures are never treated as existing
violations. The gate uses Python 3's standard library.

When removing existing comparisons, reduce or remove the corresponding baseline
counts as well; otherwise the old allowance remains available. New rules and
new expressions fail without a baseline entry. Baseline increases must be
reviewed explicitly, not regenerated as part of normal lint runs.

For raw diagnostics, run `ast-grep scan crates`; it exits nonzero when matches
exist. Test the baseline gate with `python3 .ci/test-ast-grep.py`, and hook
failure propagation with `.ci/test-lint.sh`.

## Scope and limitations

It covers both operand orders, parentheses, multiline calls, ordinary and raw
string literals, and the paths `Symbol::new`, `trunk_ir::Symbol::new`, and
`trunk_ir::symbol::Symbol::new` (including a leading `::`). Runtime arguments,
standalone construction, and ordered comparisons are outside its scope.

This is a syntax check: renamed imports such as `Sym::new` and macro token
trees such as `assert!(sym == Symbol::new("func"))` or `assert_eq!` are not
checked. A different type also spelled `Symbol` can trigger a false positive.
The tests record these blind spots without endorsing those comparison styles.

For registered literals, prefer `trunk_ir::symbol!("...")`; otherwise direct
string comparison avoids constructing a temporary symbol when text comparison
is intended. There is no automatic fix because the rule cannot determine
whether a literal belongs to trunk-ir's generated set. See
[symbol conventions](conventions.md#symbols).
