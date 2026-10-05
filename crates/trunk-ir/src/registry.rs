//! Lookup tables of hooks that dialects register per `(dialect, name)`.

use rustc_hash::FxHashMap as HashMap;

use crate::Symbol;

/// A hook registered for one `(dialect, name)`.
pub(crate) trait Registered: inventory::Collect {
    /// The hook kind, for duplicate-registration diagnostics.
    const KIND: &'static str;

    fn key(&self) -> (&'static str, &'static str);
}

/// Hooks of one kind by `(dialect, name)`, built once from `inventory`.
pub(crate) struct Registry<F: 'static>(HashMap<(Symbol, Symbol), &'static F>);

impl<F: Registered> Registry<F> {
    pub(crate) fn collect() -> Self {
        let mut map = HashMap::default();
        for hook in inventory::iter::<F> {
            let (dialect, name) = hook.key();
            let key = (Symbol::new(dialect), Symbol::new(name));
            if map.insert(key, hook).is_some() {
                panic!("duplicate {} registration for '{dialect}.{name}'", F::KIND);
            }
        }
        Self(map)
    }

    pub(crate) fn get(&self, dialect: Symbol, name: Symbol) -> Option<&'static F> {
        self.0.get(&(dialect, name)).copied()
    }
}
