//! Core dialect operations and types.
//!
//! This dialect provides fundamental types:
//! - `core.i` - integer type (with bit width in attr)
//! - `core.f` - floating-point type (with bit width in attr)
//! - `core.unit` - unit type (void/empty)
//! - `core.never` - never/bottom type (no values)
//! - `core.string` - string type
//! - `core.bytes` - byte sequence type
//! - `core.ptr` - raw pointer type
use std::collections::BTreeMap;

use crate::{Attribute, IdVec, Region, Symbol, Type, dialect, idvec, ir::BlockBuilder};
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

/// Create an integer type (`core.i{bits}`) with the given bit width.
pub fn i(db: &dyn salsa::Database, bits: u16) -> Type<'_> {
    Type::new(
        db,
        Symbol::new(db, "core"),
        Symbol::new(db, format!("i{bits}")),
        IdVec::new(),
        BTreeMap::new(),
    )
}

/// Create a floating-point type (`core.f{bits}`) with the given bit width.
pub fn f(db: &dyn salsa::Database, bits: u16) -> Type<'_> {
    Type::new(
        db,
        Symbol::new(db, "core"),
        Symbol::new(db, format!("f{bits}")),
        IdVec::new(),
        BTreeMap::new(),
    )
}

/// Create a unit type (`core.unit`).
pub fn unit(db: &dyn salsa::Database) -> Type<'_> {
    Type::new(
        db,
        Symbol::new(db, "core"),
        Symbol::new(db, "unit"),
        IdVec::new(),
        BTreeMap::new(),
    )
}

/// Create a never type (`core.never`).
pub fn never(db: &dyn salsa::Database) -> Type<'_> {
    Type::new(
        db,
        Symbol::new(db, "core"),
        Symbol::new(db, "never"),
        IdVec::new(),
        BTreeMap::new(),
    )
}

/// Create a string type (`core.string`).
pub fn string(db: &dyn salsa::Database) -> Type<'_> {
    Type::new(
        db,
        Symbol::new(db, "core"),
        Symbol::new(db, "string"),
        IdVec::new(),
        BTreeMap::new(),
    )
}

/// Create a bytes type (`core.bytes`).
pub fn bytes(db: &dyn salsa::Database) -> Type<'_> {
    Type::new(
        db,
        Symbol::new(db, "core"),
        Symbol::new(db, "bytes"),
        IdVec::new(),
        BTreeMap::new(),
    )
}

/// Create a pointer type (`core.ptr`).
pub fn ptr(db: &dyn salsa::Database) -> Type<'_> {
    Type::new(
        db,
        Symbol::new(db, "core"),
        Symbol::new(db, "ptr"),
        IdVec::new(),
        BTreeMap::new(),
    )
}

/// Create an array type (`core.array`).
pub fn array<'db>(db: &'db dyn salsa::Database, element: Type<'db>) -> Type<'db> {
    Type::new(
        db,
        Symbol::new(db, "core"),
        Symbol::new(db, "array"),
        idvec![element],
        BTreeMap::new(),
    )
}

/// Create a reference type (`core.ref`).
pub fn ref_<'db>(db: &'db dyn salsa::Database, pointee: Type<'db>, nullable: bool) -> Type<'db> {
    Type::new(
        db,
        Symbol::new(db, "core"),
        Symbol::new(db, "ref"),
        idvec![pointee],
        BTreeMap::from([(Symbol::new(db, "nullable"), Attribute::Bool(nullable))]),
    )
}

/// Create a tuple type (`core.tuple`).
pub fn tuple<'db>(db: &'db dyn salsa::Database, elements: IdVec<Type<'db>>) -> Type<'db> {
    Type::new(
        db,
        Symbol::new(db, "core"),
        Symbol::new(db, "tuple"),
        elements,
        BTreeMap::new(),
    )
}

/// Create a function type (`core.func`).
///
/// Layout: `params[0]` = return type, `params[1..]` = parameter types.
pub fn func<'db>(
    db: &'db dyn salsa::Database,
    params: IdVec<Type<'db>>,
    result: Type<'db>,
) -> Type<'db> {
    func_with_effect(db, params, result, None)
}

/// Create a function type with an explicit effect.
///
/// The `effect` parameter is the effect row type (abilities this function may perform).
/// Pass `None` for pure functions or when the effect is not yet known.
pub fn func_with_effect<'db>(
    db: &'db dyn salsa::Database,
    params: IdVec<Type<'db>>,
    result: Type<'db>,
    effect: Option<Type<'db>>,
) -> Type<'db> {
    let mut all_types = IdVec::with_capacity(params.len() + 1);
    all_types.push(result);
    all_types.extend(params.iter().copied());
    let attrs = match effect {
        Some(eff) => BTreeMap::from([(Symbol::new(db, "effect"), Attribute::Type(eff))]),
        None => BTreeMap::new(),
    };
    Type::new(
        db,
        Symbol::new(db, "core"),
        Symbol::new(db, "func"),
        all_types,
        attrs,
    )
}
