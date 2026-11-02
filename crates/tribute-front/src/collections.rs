//! Fx-hashed collections for code that needs hashbrown's borrowed entry APIs.

pub(crate) type HashMap<K, V> = hashbrown::HashMap<K, V, rustc_hash::FxBuildHasher>;
