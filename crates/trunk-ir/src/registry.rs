//! Lookup tables of hooks that dialects register per definition.

use rustc_hash::FxHashMap as HashMap;

use crate::name::{Definition, Name};

/// A hook registered for one operation or type definition.
pub(crate) trait Registered: inventory::Collect {
    type Def: Definition;

    /// The hook kind, for duplicate-registration diagnostics.
    const KIND: &'static str;

    fn def(&self) -> &'static Self::Def;
}

/// Hooks of one kind by the name of the definition they belong to, built
/// once from `inventory`.
pub(crate) struct Registry<F: Registered + 'static>(HashMap<Name<F::Def>, &'static F>);

impl<F: Registered> Registry<F> {
    pub(crate) fn collect() -> Self {
        let mut map = HashMap::default();
        for hook in inventory::iter::<F> {
            let def = hook.def();
            if map.insert(Name::registered(def), hook).is_some() {
                panic!(
                    "duplicate {} registration for '{}'",
                    F::KIND,
                    def.qualified_name().as_str()
                );
            }
        }
        Self(map)
    }

    pub(crate) fn get(&self, name: Name<F::Def>) -> Option<&'static F> {
        self.0.get(&name).copied()
    }
}
