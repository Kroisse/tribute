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
        #[attr(sym_name: Symbol)]
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
        module(db, location, Symbol::new(db, name), body)
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
    pub fn name(&self, db: &'db dyn salsa::Database) -> &'db str {
        self.sym_name(db).text(db)
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

// === Effect Row type wrapper ===

/// Effect row type wrapper (`core.effect_row`).
///
/// Represents an effect row for row-polymorphic effect typing.
/// Layout:
/// - `params`: Ability types (each ability is a type like `State(Int)`)
/// - `attrs.tail`: Optional row variable ID for the tail (open row)
///
/// Examples:
/// - `{}` - empty row (pure)
/// - `{State(Int)}` - concrete row with one ability
/// - `{Console | e}` - row with ability and tail variable
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub struct EffectRowType<'db>(Type<'db>);

impl<'db> EffectRowType<'db> {
    /// Create an empty effect row (pure function).
    pub fn empty(db: &'db dyn salsa::Database) -> Self {
        Self(Type::new(
            db,
            Symbol::new(db, "core"),
            Symbol::new(db, "effect_row"),
            IdVec::new(),
            BTreeMap::new(),
        ))
    }

    /// Create an effect row with abilities and no tail (closed row).
    pub fn concrete(db: &'db dyn salsa::Database, abilities: IdVec<Type<'db>>) -> Self {
        Self(Type::new(
            db,
            Symbol::new(db, "core"),
            Symbol::new(db, "effect_row"),
            abilities,
            BTreeMap::new(),
        ))
    }

    /// Create an effect row with a tail variable (open row).
    pub fn with_tail(
        db: &'db dyn salsa::Database,
        abilities: IdVec<Type<'db>>,
        tail_var_id: u64,
    ) -> Self {
        Self(Type::new(
            db,
            Symbol::new(db, "core"),
            Symbol::new(db, "effect_row"),
            abilities,
            BTreeMap::from([(Symbol::new(db, "tail"), Attribute::IntBits(tail_var_id))]),
        ))
    }

    /// Create an effect row with just a tail variable (polymorphic row).
    pub fn var(db: &'db dyn salsa::Database, tail_var_id: u64) -> Self {
        Self::with_tail(db, IdVec::new(), tail_var_id)
    }

    /// Check if this is an empty row (pure).
    pub fn is_empty(&self, db: &'db dyn salsa::Database) -> bool {
        self.0.params(db).is_empty() && self.tail_var(db).is_none()
    }

    /// Get the ability types in this row.
    pub fn abilities(&self, db: &'db dyn salsa::Database) -> &[Type<'db>] {
        self.0.params(db)
    }

    /// Get the tail variable ID, if any.
    pub fn tail_var(&self, db: &'db dyn salsa::Database) -> Option<u64> {
        match self.0.get_attr(db, "tail") {
            Some(Attribute::IntBits(id)) => Some(*id),
            _ => None,
        }
    }
}

impl<'db> Deref for EffectRowType<'db> {
    type Target = Type<'db>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'db> DialectType<'db> for EffectRowType<'db> {
    fn as_type(&self) -> Type<'db> {
        self.0
    }

    fn from_type(db: &'db dyn salsa::Database, ty: Type<'db>) -> Option<Self> {
        if ty.dialect(db).text(db) == "core" && ty.name(db).text(db) == "effect_row" {
            Some(Self(ty))
        } else {
            None
        }
    }
}

// === Ability type wrapper ===

/// Ability type wrapper (`core.ability_ref`).
///
/// Represents an ability (effect) reference like `State(Int)` or `Console`.
/// Layout:
/// - `attrs.name`: The ability name as a symbol
/// - `params`: Type parameters for the ability
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub struct AbilityRefType<'db>(Type<'db>);

impl<'db> AbilityRefType<'db> {
    /// Create an ability type with no type parameters.
    pub fn simple(db: &'db dyn salsa::Database, name: Symbol<'db>) -> Self {
        Self(Type::new(
            db,
            Symbol::new(db, "core"),
            Symbol::new(db, "ability_ref"),
            IdVec::new(),
            BTreeMap::from([(Symbol::new(db, "name"), Attribute::Symbol(name))]),
        ))
    }

    /// Create an ability type with type parameters.
    pub fn with_params(
        db: &'db dyn salsa::Database,
        name: Symbol<'db>,
        params: IdVec<Type<'db>>,
    ) -> Self {
        Self(Type::new(
            db,
            Symbol::new(db, "core"),
            Symbol::new(db, "ability_ref"),
            params,
            BTreeMap::from([(Symbol::new(db, "name"), Attribute::Symbol(name))]),
        ))
    }

    /// Get the ability name.
    pub fn name(&self, db: &'db dyn salsa::Database) -> Option<Symbol<'db>> {
        match self.0.get_attr(db, "name") {
            Some(Attribute::Symbol(sym)) => Some(*sym),
            _ => None,
        }
    }

    /// Get the type parameters.
    pub fn params(&self, db: &'db dyn salsa::Database) -> &[Type<'db>] {
        self.0.params(db)
    }
}

impl<'db> Deref for AbilityRefType<'db> {
    type Target = Type<'db>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'db> DialectType<'db> for AbilityRefType<'db> {
    fn as_type(&self) -> Type<'db> {
        self.0
    }

    fn from_type(db: &'db dyn salsa::Database, ty: Type<'db>) -> Option<Self> {
        if ty.dialect(db).text(db) == "core" && ty.name(db).text(db) == "ability_ref" {
            Some(Self(ty))
        } else {
            None
        }
    }
}
