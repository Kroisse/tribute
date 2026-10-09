# Hash Maps and Sets

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
