//! Core dialect operations.

use crate::{Attribute, OpNameId, Operation, Region, Symbol, define_dialect_op};
use tribute_core::Location;

define_dialect_op! {
    /// `core.module` operation: top-level module container.
    pub struct Module("core", "module") {
        has_attr("sym_name"),
        has_region,
    }
}

impl<'db> Module<'db> {
    /// Create a new module.
    pub fn new(
        db: &'db dyn salsa::Database,
        location: Location<'db>,
        name: &str,
        body: Region<'db>,
    ) -> Self {
        let op_name = OpNameId::new(db, "core", "module");
        let op = Operation::of(db, location, op_name)
            .attr("sym_name", Attribute::String(name.to_string()))
            .region(body)
            .build();
        Self::wrap_unchecked(op)
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
        let body = Region::new(db, location, vec![top.build()]);
        Self::new(db, location, name, body)
    }

    /// Get the module name.
    pub fn name(&self, db: &'db dyn salsa::Database) -> String {
        let key = Symbol::new(db, "sym_name");
        match self.op.attributes(db).get(&key) {
            Some(Attribute::String(s)) => s.clone(),
            _ => panic!("core.module missing sym_name attribute"),
        }
    }

    /// Get the module body region.
    pub fn body(&self, db: &'db dyn salsa::Database) -> Region<'db> {
        self.op.regions(db)[0]
    }
}
