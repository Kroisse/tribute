# Rust Lints

Project-specific type restrictions are configured in the root `clippy.toml`.
Run them with the existing Clippy command; no nightly toolchain or custom lint
runner is required:

```bash
cargo clippy --workspace --all-targets -- -D warnings
```

## Hash maps

Use `hashbrown::HashMap` for compiler data structures. The workspace enables
its `default-hasher` feature, so standard constructors such as `HashMap::new()`
remain available. Existing `rustc_hash::FxHashMap` uses are also supported.
The Salsa module environment uses `FxHashMap` because Salsa supports its hasher
as a `SalsaValue`; hashbrown's default hasher does not satisfy that bound.

Clippy's `disallowed_types` lint rejects paths resolving to
`std::collections::HashMap`, including renamed imports and explicit custom
hasher parameters. It checks named definitions rather than expanding aliases,
so `rustc_hash::FxHashMap` is allowed even though its underlying type is a
standard-library HashMap. Defining a local alias for the standard type still
warns at the alias definition. `HashSet` is outside this restriction's scope.

If an external API requires the standard map, allow it only at the boundary:

```rust
#[allow(clippy::disallowed_types, reason = "the external API requires a standard HashMap")]
let mut map = std::collections::HashMap::new();
```

Neither hashbrown's default hasher nor FxHasher is designed to defend against
deliberate hash collisions. Review the hasher choice for externally controlled
keys; retaining the standard hasher at those boundaries is valid.
