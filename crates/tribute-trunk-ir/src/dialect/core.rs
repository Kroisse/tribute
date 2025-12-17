//! Core dialect operations.

use crate::{Attribute, Region, dialect};
use tribute_core::Location;

dialect! {
    core {
        /// `core.module` operation: top-level module container.
        op module[sym_name]() @body {};

        /// `core.unrealized_conversion_cast` operation: temporary cast during dialect conversion.
        /// Must be eliminated after lowering is complete.
        op unrealized_conversion_cast(value) -> result;
    }
}

impl<'db> Module<'db> {
    /// Create a new module with explicit body region.
    pub fn create(
        db: &'db dyn salsa::Database,
        location: Location<'db>,
        name: &str,
        body: Region<'db>,
    ) -> Self {
        Self::new(db, location, Attribute::String(name.to_string()), body)
    }

    /// Build a module with a closure that constructs the top-level block.
    pub fn build(
        db: &'db dyn salsa::Database,
        location: Location<'db>,
        name: &str,
        f: impl FnOnce(&mut crate::BlockBuilder<'db>),
    ) -> Self {
        let mut top = crate::BlockBuilder::new(db, location);
        f(&mut top);
        let region = Region::new(db, location, vec![top.build()]);
        Self::create(db, location, name, region)
    }

    /// Get the module name.
    pub fn name(&self, db: &'db dyn salsa::Database) -> &str {
        let Attribute::String(name) = self.sym_name(db) else {
            panic!("core.module missing sym_name attribute")
        };
        name
    }
}
