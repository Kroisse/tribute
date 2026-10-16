//! Lookup tables of hooks that dialects register per `(dialect, name)`.

use rustc_hash::FxHashMap as HashMap;

/// A hook registered for one `(dialect, name)`.
pub(crate) trait Registered: inventory::Collect {
    /// The hook kind, for duplicate-registration diagnostics.
    const KIND: &'static str;

    fn key(&self) -> (&'static str, &'static str);
}

/// Hooks of one kind by `(dialect, name)`, built once from `inventory`.
pub(crate) struct Registry<F: 'static>(HashMap<(&'static str, &'static str), &'static F>);

impl<F: Registered> Registry<F> {
    pub(crate) fn collect() -> Self {
        let mut map = HashMap::default();
        for hook in inventory::iter::<F> {
            let (dialect, name) = hook.key();
            if map.insert((dialect, name), hook).is_some() {
                panic!("duplicate {} registration for '{dialect}.{name}'", F::KIND);
            }
        }
        Self(map)
    }

    pub(crate) fn get(&self, dialect: &str, name: &str) -> Option<&'static F> {
        self.0.get(&(dialect, name)).copied()
    }
}
