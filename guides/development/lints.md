# Rust Lints

Project-specific type restrictions are configured in the root `clippy.toml`.
Run them with the existing Clippy command; no nightly toolchain or custom lint
runner is required:

```bash
cargo clippy --workspace --all-targets -- -D warnings
```

## Symbol comparison experiment

An optional [ast-grep](https://ast-grep.github.io/) rule detects temporary
`Symbol::new("...")` calls used as operands of `==` or `!=`. The prototype is
tested with ast-grep 0.45.3 and needs no nightly compiler. Install that version
of the CLI, then run from the repository root:

```bash
cargo binstall ast-grep --version 0.45.3 --no-confirm
ast-grep test --skip-snapshot-tests
ast-grep scan crates
```

The rule is in `.ast-grep/rules/no-symbol-new-in-comparison.yml`; positive and
negative cases are in `.ast-grep/tests/`. The tests check detection without
diagnostic snapshots. Scanning reports error diagnostics and exits nonzero when
matches exist. This experiment runs manually; it is not part of CI or hooks.

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

## Hash maps and sets

Use `rustc_hash::FxHashMap` as the project's `HashMap`, including Salsa's
module environment. Import `rustc_hash::FxHashSet` as `HashSet` for sets:

```rust
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

let mut map = HashMap::default();
map.insert("key", 1);
let mut set = HashSet::default();
set.insert("key");
```

Construct empty maps with `HashMap::default()` and initialize maps from entries
with `entries.into_iter().collect::<HashMap<_, _>>()`. `HashMap::new()` and the
standard map's array `From` implementation do not support this hasher. To reserve
capacity, use `HashMap::with_capacity_and_hasher(capacity, Default::default())`.
For sets, use `HashSet::default()`,
`entries.into_iter().collect::<HashSet<_>>()`, or
`HashSet::with_capacity_and_hasher(capacity, Default::default())`; the same
constructor restrictions apply. Keep hashbrown for the existing `HashTable`
interner.

Passes that need borrowed entry APIs can use the crate-private
`tribute-passes::collections` aliases. They use hashbrown with `FxBuildHasher`
so the hashing policy stays the same. Use `entry_ref()` for borrowed map keys
and `get_or_insert_with()` for borrowed set values to construct owned keys only
on a miss. Keep public collection types compatible with their callers.

Clippy's `disallowed_types` lint rejects paths resolving to
`std::collections::HashMap` and `std::collections::HashSet`, including renamed
imports and explicit custom hasher parameters. It checks named definitions rather
than expanding aliases,
so `rustc_hash::FxHashMap as HashMap` is allowed even though its underlying type
is a standard-library HashMap. Defining a local alias for the standard type still
warns at the alias definition. The same rule allows `FxHashSet as HashSet`.

If an external API requires the standard map, allow it only at the boundary:

```rust
#[allow(clippy::disallowed_types, reason = "the external API requires a standard HashMap")]
let mut map = std::collections::HashMap::new();
```

FxHasher is not designed to defend against deliberate hash collisions.
Review the hasher choice for externally controlled
keys; retaining the standard hasher at those boundaries is valid.
