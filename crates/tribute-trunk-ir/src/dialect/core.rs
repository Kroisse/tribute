//! Core dialect operations and types.
//!
//! This dialect provides fundamental types:
//! - `core.unit` - unit type (void/empty)
//! - `core.never` - never/bottom type (no values)
//! - `core.string` - string type
//! - `core.bytes` - byte sequence type
//! - `core.ptr` - raw pointer type

use crate::{Attribute, IdVec, Region, Type, dialect, idvec, ir::BlockBuilder};
use tribute_core::Location;

dialect! {
    mod core {
        /// `core.module` operation: top-level module container.
        #[attr(sym_name)]
        fn module() {
            #[region(body)] {}
        };

        /// `core.unrealized_conversion_cast` operation: temporary cast during dialect conversion.
        /// Must be eliminated after lowering is complete.
        fn unrealized_conversion_cast(value) -> result;
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
        module(db, location, Attribute::String(name.to_string()), body)
    }

    /// Build a module with a closure that constructs the top-level block.
    pub fn build(
        db: &'db dyn salsa::Database,
        location: Location<'db>,
        name: &str,
        f: impl FnOnce(&mut BlockBuilder<'db>),
    ) -> Self {
        let mut top = BlockBuilder::new(db, location);
        f(&mut top);
        let region = Region::new(db, location, idvec![top.build()]);
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

// === Core type constructors ===

/// Create a unit type (`core.unit`).
pub fn unit_type(db: &dyn salsa::Database) -> Type<'_> {
    Type::dialect(db, "core", "unit", IdVec::new(), Attribute::Unit)
}

/// Create a never type (`core.never`).
pub fn never_type(db: &dyn salsa::Database) -> Type<'_> {
    Type::dialect(db, "core", "never", IdVec::new(), Attribute::Unit)
}

/// Create a string type (`core.string`).
pub fn string_type(db: &dyn salsa::Database) -> Type<'_> {
    Type::dialect(db, "core", "string", IdVec::new(), Attribute::Unit)
}

/// Create a bytes type (`core.bytes`).
pub fn bytes_type(db: &dyn salsa::Database) -> Type<'_> {
    Type::dialect(db, "core", "bytes", IdVec::new(), Attribute::Unit)
}

/// Create a pointer type (`core.ptr`).
pub fn ptr_type(db: &dyn salsa::Database) -> Type<'_> {
    Type::dialect(db, "core", "ptr", IdVec::new(), Attribute::Unit)
}
