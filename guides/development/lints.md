# Rust Lints

Project-specific type restrictions are configured in the root `clippy.toml`.
Run them with the existing Clippy command; no nightly toolchain or custom lint
runner is required:

```bash
cargo clippy --workspace --all-targets -- -D warnings
```

## Hash maps

Use `rustc_hash::FxHashMap` as the project's `HashMap`, including Salsa's
module environment:

```rust
use rustc_hash::FxHashMap as HashMap;

let mut map = HashMap::default();
map.insert("key", 1);
```

Construct empty maps with `HashMap::default()` and initialize maps from entries
with `entries.into_iter().collect::<HashMap<_, _>>()`. `HashMap::new()` and the
standard map's array `From` implementation do not support this hasher. To reserve
capacity, use `HashMap::with_capacity_and_hasher(capacity, Default::default())`.
Keep hashbrown for the existing `HashTable` interner.

Clippy's `disallowed_types` lint rejects paths resolving to
`std::collections::HashMap`, including renamed imports and explicit custom
hasher parameters. It checks named definitions rather than expanding aliases,
so `rustc_hash::FxHashMap as HashMap` is allowed even though its underlying type
is a standard-library HashMap. Defining a local alias for the standard type still
warns at the alias definition. `HashSet` is outside this restriction's scope.

If an external API requires the standard map, allow it only at the boundary:

```rust
#[allow(clippy::disallowed_types, reason = "the external API requires a standard HashMap")]
let mut map = std::collections::HashMap::new();
```

FxHasher is not designed to defend against deliberate hash collisions.
Review the hasher choice for externally controlled
keys; retaining the standard hasher at those boundaries is valid.
