//! Fx-hashed collections for passes that need hashbrown's borrowed entry APIs.

pub(crate) type HashMap<K, V> = hashbrown::HashMap<K, V, rustc_hash::FxBuildHasher>;
pub(crate) type HashSet<T> = hashbrown::HashSet<T, rustc_hash::FxBuildHasher>;
