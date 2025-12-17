//! Core dialect operations and types.
//!
//! This dialect provides fundamental types:
//! - `core.i{bits}` - integer type (e.g., `core.i32`, `core.i64`)
//! - `core.f{bits}` - floating-point type (e.g., `core.f32`, `core.f64`)
//! - `core.nil` - nil/unit type (empty tuple terminator)
//! - `core.tuple` - tuple cons cell (head, tail)
//! - `core.never` - never/bottom type (no values)
//! - `core.string` - string type
//! - `core.bytes` - byte sequence type
//! - `core.ptr` - raw pointer type
use std::collections::BTreeMap;
use std::ops::Deref;

use crate::{
    Attribute, DialectType, IdVec, Region, Symbol, Type, dialect, idvec, ir::BlockBuilder,
};
use tribute_core::Location;

dialect! {
    mod core {
        // === Operations ===

        /// `core.module` operation: top-level module container.
        #[attr(sym_name)]
        fn module() {
            #[region(body)] {}
        };

        /// `core.unrealized_conversion_cast` operation: temporary cast during dialect conversion.
        /// Must be eliminated after lowering is complete.
        fn unrealized_conversion_cast(value) -> result;

        // === Types ===

        /// `core.nil` type: empty tuple terminator / unit type.
        type nil;

        /// `core.never` type: bottom type with no values.
        type never;

        /// `core.string` type: string type.
        type string;

        /// `core.bytes` type: byte sequence type.
        type bytes;

        /// `core.ptr` type: raw pointer type.
        type ptr;

        /// `core.array` type: array with element type.
        type array(element);

        /// `core.tuple` type: cons cell (head, tail).
        /// Use `Nil` as the tail terminator.
        /// Example: `(a, b, c)` → `Tuple(a, Tuple(b, Tuple(c, Nil)))`
        type tuple(head, tail);

        /// `core.ref_` type: GC-managed reference.
        #[attr(nullable: bool)]
        type ref_(pointee);
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

// === Integer type wrapper ===

/// Integer type wrapper (`core.i{BITS}`).
///
/// Use `I::<32>::new(db)` or the type alias `I32::new(db)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub struct I<'db, const BITS: u16>(Type<'db>);

impl<'db, const BITS: u16> I<'db, BITS> {
    /// Create a new integer type with the specified bit width.
    pub fn new(db: &'db dyn salsa::Database) -> Self {
        Self(i(db, BITS))
    }
}

impl<'db, const BITS: u16> Deref for I<'db, BITS> {
    type Target = Type<'db>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'db, const BITS: u16> DialectType<'db> for I<'db, BITS> {
    fn as_type(&self) -> Type<'db> {
        self.0
    }

    fn from_type(db: &'db dyn salsa::Database, ty: Type<'db>) -> Option<Self> {
        let expected_name = format!("i{BITS}");
        if ty.dialect(db).text(db) == "core" && ty.name(db).text(db) == expected_name {
            Some(Self(ty))
        } else {
            None
        }
    }
}

/// 1-bit integer type (boolean).
pub type I1<'db> = I<'db, 1>;
/// 8-bit integer type.
pub type I8<'db> = I<'db, 8>;
/// 16-bit integer type.
pub type I16<'db> = I<'db, 16>;
/// 32-bit integer type.
pub type I32<'db> = I<'db, 32>;
/// 64-bit integer type.
pub type I64<'db> = I<'db, 64>;

/// Create an integer type (`core.i{bits}`) with the given bit width.
fn i(db: &dyn salsa::Database, bits: u16) -> Type<'_> {
    Type::new(
        db,
        Symbol::new(db, "core"),
        Symbol::new(db, format!("i{bits}")),
        IdVec::new(),
        BTreeMap::new(),
    )
}

// === Floating-point type wrapper ===

/// Floating-point type wrapper (`core.f{BITS}`).
///
/// Use `F::<32>::new(db)` or the type alias `F32::new(db)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub struct F<'db, const BITS: u16>(Type<'db>);

impl<'db, const BITS: u16> F<'db, BITS> {
    /// Create a new floating-point type with the specified bit width.
    pub fn new(db: &'db dyn salsa::Database) -> Self {
        Self(f(db, BITS))
    }
}

impl<'db, const BITS: u16> Deref for F<'db, BITS> {
    type Target = Type<'db>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'db, const BITS: u16> DialectType<'db> for F<'db, BITS> {
    fn as_type(&self) -> Type<'db> {
        self.0
    }

    fn from_type(db: &'db dyn salsa::Database, ty: Type<'db>) -> Option<Self> {
        let expected_name = format!("f{BITS}");
        if ty.dialect(db).text(db) == "core" && ty.name(db).text(db) == expected_name {
            Some(Self(ty))
        } else {
            None
        }
    }
}

/// 32-bit floating-point type.
pub type F32<'db> = F<'db, 32>;
/// 64-bit floating-point type.
pub type F64<'db> = F<'db, 64>;

/// Create a floating-point type (`core.f{bits}`) with the given bit width.
fn f(db: &dyn salsa::Database, bits: u16) -> Type<'_> {
    Type::new(
        db,
        Symbol::new(db, "core"),
        Symbol::new(db, format!("f{bits}")),
        IdVec::new(),
        BTreeMap::new(),
    )
}

// === Function type wrapper ===

/// Function type wrapper (`core.func`).
///
/// Layout: `params[0]` = return type, `params[1..]` = parameter types.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub struct Func<'db>(Type<'db>);

impl<'db> Func<'db> {
    /// Create a pure function type (no effects).
    pub fn new(db: &'db dyn salsa::Database, params: IdVec<Type<'db>>, result: Type<'db>) -> Self {
        Self::with_effect(db, params, result, None)
    }

    /// Create a function type with an explicit effect.
    pub fn with_effect(
        db: &'db dyn salsa::Database,
        params: IdVec<Type<'db>>,
        result: Type<'db>,
        effect: Option<Type<'db>>,
    ) -> Self {
        let mut all_types = IdVec::with_capacity(params.len() + 1);
        all_types.push(result);
        all_types.extend(params.iter().copied());
        let attrs = match effect {
            Some(eff) => BTreeMap::from([(Symbol::new(db, "effect"), Attribute::Type(eff))]),
            None => BTreeMap::new(),
        };
        Self(Type::new(
            db,
            Symbol::new(db, "core"),
            Symbol::new(db, "func"),
            all_types,
            attrs,
        ))
    }

    /// Get the return type.
    pub fn result(&self, db: &'db dyn salsa::Database) -> Type<'db> {
        self.0.params(db)[0]
    }

    /// Get the parameter types.
    pub fn params(&self, db: &'db dyn salsa::Database) -> IdVec<Type<'db>> {
        self.0.params(db).iter().skip(1).copied().collect()
    }

    /// Get the effect type, if any.
    pub fn effect(&self, db: &'db dyn salsa::Database) -> Option<Type<'db>> {
        match self.0.get_attr(db, "effect") {
            Some(Attribute::Type(ty)) => Some(*ty),
            _ => None,
        }
    }
}

impl<'db> Deref for Func<'db> {
    type Target = Type<'db>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'db> DialectType<'db> for Func<'db> {
    fn as_type(&self) -> Type<'db> {
        self.0
    }

    fn from_type(db: &'db dyn salsa::Database, ty: Type<'db>) -> Option<Self> {
        if ty.dialect(db).text(db) == "core" && ty.name(db).text(db) == "func" {
            Some(Self(ty))
        } else {
            None
        }
    }
}
