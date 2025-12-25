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
    Attribute, DialectType, IdVec, Location, Region, Symbol, Type, dialect, idvec, ir::BlockBuilder,
};

crate::symbols! {
    ABILITY_REF => "ability_ref",
    EFFECT_ROW => "effect_row",
    FUNC => "func",
}

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
        name: Symbol,
        body: Region<'db>,
    ) -> Self {
        module(db, location, name, body)
    }

    /// Build a module with a closure that constructs the top-level block.
    pub fn build(
        db: &'db dyn salsa::Database,
        location: Location<'db>,
        name: Symbol,
        f: impl FnOnce(&mut BlockBuilder<'db>),
    ) -> Self {
        let mut top = BlockBuilder::new(db, location);
        f(&mut top);
        let region = Region::new(db, location, idvec![top.build()]);
        Self::create(db, location, name, region)
    }

    /// Get the module name.
    pub fn name(&self, db: &'db dyn salsa::Database) -> Symbol {
        self.sym_name(db)
    }
}

impl std::fmt::Debug for Module<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        salsa::with_attached_database(|db| {
            let name = self.name(db);
            let body = self.body(db);

            let mut operations = Vec::new();
            for block in body.blocks(db).iter() {
                for op in block.operations(db).iter() {
                    operations.push(*op);
                }
            }

            f.debug_struct(&format!("Module({})", name))
                .field(
                    "operations",
                    &debug_helpers::OperationList {
                        db,
                        ops: &operations,
                    },
                )
                .finish()
        })
        .unwrap_or_else(|| write!(f, "Module(<no database attached>)"))
    }
}

mod debug_helpers {
    use super::*;

    pub(super) struct OperationList<'a, 'db> {
        pub(super) db: &'db dyn salsa::Database,
        pub(super) ops: &'a [crate::Operation<'db>],
    }

    impl std::fmt::Debug for OperationList<'_, '_> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let mut list = f.debug_list();
            for op in self.ops {
                list.entry(&OpDebug {
                    db: self.db,
                    op: *op,
                });
            }
            list.finish()
        }
    }

    struct OpDebug<'db> {
        db: &'db dyn salsa::Database,
        op: crate::Operation<'db>,
    }

    impl std::fmt::Debug for OpDebug<'_> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            use crate::DialectOp;

            // Try func.func first
            if let Ok(func_op) = crate::dialect::func::Func::from_operation(self.db, self.op) {
                return func_op.fmt(f);
            }

            // Try nested module
            if let Ok(module_op) = Module::from_operation(self.db, self.op) {
                return module_op.fmt(f);
            }

            // Default: show operation name
            write!(f, "{}.{}", self.op.dialect(self.db), self.op.name(self.db))
        }
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
        if ty.dialect(db) == DIALECT_NAME()
            && ty.name(db).with_str(|n| n == format!("i{BITS}").as_str())
        {
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
        DIALECT_NAME(),
        Symbol::from_dynamic(&format!("i{bits}")),
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
        if ty.dialect(db) == DIALECT_NAME()
            && ty.name(db).with_str(|n| n == format!("f{BITS}").as_str())
        {
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
        DIALECT_NAME(),
        Symbol::from_dynamic(&format!("f{bits}")),
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
            Some(eff) => BTreeMap::from([(Self::effect_sym(), Attribute::Type(eff))]),
            None => BTreeMap::new(),
        };
        Self(Type::new(db, DIALECT_NAME(), FUNC(), all_types, attrs))
    }

    /// Get the return type.
    pub fn result(&self, db: &'db dyn salsa::Database) -> Type<'db> {
        self.0.params(db)[0]
    }

    /// Get the parameter types.
    pub fn params(&self, db: &'db dyn salsa::Database) -> IdVec<Type<'db>> {
        self.0.params(db).iter().skip(1).copied().collect()
    }

    pub fn effect_sym() -> Symbol {
        Symbol::new("effect")
    }

    /// Get the effect type, if any.
    pub fn effect(&self, db: &'db dyn salsa::Database) -> Option<Type<'db>> {
        match self.0.get_attr(db, Self::effect_sym()) {
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
        if ty.dialect(db) == DIALECT_NAME() && ty.name(db) == FUNC() {
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
    /// Create an effect row with abilities and a tail variable (open row).
    pub fn new(
        db: &'db dyn salsa::Database,
        abilities: IdVec<Type<'db>>,
        tail_var_id: u64,
    ) -> Self {
        let mut attrs = BTreeMap::new();
        if tail_var_id != 0 {
            attrs.insert(Self::tail_sym(), Attribute::IntBits(tail_var_id));
        }
        Self(Type::new(
            db,
            DIALECT_NAME(),
            EFFECT_ROW(),
            abilities,
            attrs,
        ))
    }

    /// Create an empty effect row (pure function).
    #[inline]
    pub fn empty(db: &'db dyn salsa::Database) -> Self {
        Self::new(db, Default::default(), Default::default())
    }

    /// Create an effect row with abilities and no tail (closed row).
    #[inline]
    pub fn concrete(db: &'db dyn salsa::Database, abilities: IdVec<Type<'db>>) -> Self {
        Self::new(db, abilities, Default::default())
    }

    /// Create an effect row with a tail variable (open row).
    #[inline]
    pub fn with_tail(
        db: &'db dyn salsa::Database,
        abilities: IdVec<Type<'db>>,
        tail_var_id: u64,
    ) -> Self {
        Self::new(db, abilities, tail_var_id)
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

    pub fn tail_sym() -> Symbol {
        Symbol::new("tail")
    }

    /// Get the tail variable ID, if any.
    pub fn tail_var(&self, db: &'db dyn salsa::Database) -> Option<u64> {
        match self.0.get_attr(db, Self::tail_sym()) {
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
        if ty.dialect(db) == DIALECT_NAME() && ty.name(db) == EFFECT_ROW() {
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
    pub fn simple(db: &'db dyn salsa::Database, name: Symbol) -> Self {
        Self(Type::new(
            db,
            DIALECT_NAME(),
            ABILITY_REF(),
            IdVec::new(),
            BTreeMap::from([(Self::name_sym(), Attribute::Symbol(name))]),
        ))
    }

    /// Create an ability type with type parameters.
    pub fn with_params(
        db: &'db dyn salsa::Database,
        name: Symbol,
        params: IdVec<Type<'db>>,
    ) -> Self {
        Self(Type::new(
            db,
            DIALECT_NAME(),
            ABILITY_REF(),
            params,
            BTreeMap::from([(Self::name_sym(), Attribute::Symbol(name))]),
        ))
    }

    pub fn name_sym() -> Symbol {
        Symbol::new("name")
    }

    /// Get the ability name.
    pub fn name(&self, db: &'db dyn salsa::Database) -> Option<Symbol> {
        match self.0.get_attr(db, Self::name_sym()) {
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
        if ty.dialect(db) == DIALECT_NAME() && ty.name(db) == ABILITY_REF() {
            Some(Self(ty))
        } else {
            None
        }
    }
}

// === Printable interface registrations ===

use std::fmt::{self, Formatter, Write};

use crate::type_interface::Printable;

// nil -> "()"
inventory::submit! { Printable::implement("core", "nil", |_, _, f| f.write_str("()")) }

// never -> "Never"
inventory::submit! { Printable::implement("core", "never", |_, _, f| f.write_str("Never")) }

// string -> "String"
inventory::submit! { Printable::implement("core", "string", |_, _, f| f.write_str("String")) }

// bytes -> "Bytes"
inventory::submit! { Printable::implement("core", "bytes", |_, _, f| f.write_str("Bytes")) }

// ptr -> "Ptr"
inventory::submit! { Printable::implement("core", "ptr", |_, _, f| f.write_str("Ptr")) }

// Integer types: i1 -> "Bool", i{N} -> "I{N}"
inventory::submit! {
    Printable::implement_prefix("core", "i", |db, ty, f| {
        ty.name(db).with_str(|name| match name {
            "i1" => f.write_str("Bool"),
            _ => write!(f, "I{}", &name[1..]),
        })
    })
}

// Floating-point types: f64 -> "Float", f{N} -> "F{N}"
inventory::submit! {
    Printable::implement_prefix("core", "f", |db, ty, f| {
        ty.name(db).with_str(|name| match name {
            "f64" => f.write_str("Float"),
            _ => write!(f, "F{}", &name[1..]),
        })
    })
}

// func -> "fn(a, b) ->{eff} c"
inventory::submit! { Printable::implement("core", "func", print_func) }

fn print_func(db: &dyn salsa::Database, ty: Type<'_>, f: &mut Formatter<'_>) -> fmt::Result {
    let Some(func) = Func::from_type(db, ty) else {
        return f.write_str("fn(?)");
    };

    let params = func.params(db);
    let result = func.result(db);
    let effect = func.effect(db);

    // Format parameters
    f.write_str("fn(")?;
    for (i, p) in params.iter().enumerate() {
        if i > 0 {
            f.write_str(", ")?;
        }
        Printable::print_type(db, *p, f)?;
    }
    f.write_char(')')?;

    // Format arrow with effect
    if let Some(eff) = effect {
        if let Some(row) = EffectRowType::from_type(db, eff) {
            if row.is_empty(db) {
                f.write_str(" -> ")?;
            } else {
                f.write_str(" ->{")?;
                print_effect_row_inner(db, &row, f)?;
                f.write_str("} ")?;
            }
        } else {
            f.write_str(" -> ")?;
        }
    } else {
        f.write_str(" -> ")?;
    }

    Printable::print_type(db, result, f)
}

// tuple -> "(a, b, c)"
inventory::submit! { Printable::implement("core", "tuple", print_tuple) }

fn print_tuple(db: &dyn salsa::Database, ty: Type<'_>, f: &mut Formatter<'_>) -> fmt::Result {
    let params = ty.params(db);
    if params.is_empty() {
        return f.write_str("()");
    }

    // Flatten cons cells into a list
    let mut elements = Vec::new();
    let mut current = ty;

    while current.is_dialect(db, DIALECT_NAME(), TUPLE()) {
        let params = current.params(db);
        if params.len() >= 2 {
            elements.push(params[0]); // head
            current = params[1]; // tail
        } else {
            break;
        }
    }

    // Check if tail is nil (complete tuple)
    let has_tail = !current.is_dialect(db, DIALECT_NAME(), NIL());

    f.write_char('(')?;
    for (i, &elem) in elements.iter().enumerate() {
        if i > 0 {
            f.write_str(", ")?;
        }
        Printable::print_type(db, elem, f)?;
    }
    if has_tail {
        if !elements.is_empty() {
            f.write_str(", ")?;
        }
        Printable::print_type(db, current, f)?;
    }
    f.write_char(')')
}

// array -> "Array(elem)"
inventory::submit! {
    Printable::implement("core", "array", |db, ty, f| {
        let params = ty.params(db);
        if let Some(&elem) = params.first() {
            f.write_str("Array(")?;
            Printable::print_type(db, elem, f)?;
            f.write_char(')')
        } else {
            f.write_str("Array(?)")
        }
    })
}

// effect_row -> "{Ability1, Ability2 | e}"
inventory::submit! {
    Printable::implement("core", "effect_row", |db, ty, f| {
        if let Some(row) = EffectRowType::from_type(db, ty) {
            f.write_char('{')?;
            print_effect_row_inner(db, &row, f)?;
            f.write_char('}')
        } else {
            f.write_str("{}")
        }
    })
}

fn print_effect_row_inner(
    db: &dyn salsa::Database,
    row: &EffectRowType<'_>,
    f: &mut Formatter<'_>,
) -> fmt::Result {
    if row.is_empty(db) {
        return Ok(());
    }

    let abilities = row.abilities(db);
    for (i, &a) in abilities.iter().enumerate() {
        if i > 0 {
            f.write_str(", ")?;
        }
        Printable::print_type(db, a, f)?;
    }

    if let Some(var_id) = row.tail_var(db) {
        if !abilities.is_empty() {
            f.write_str(" | ")?;
        }
        fmt_var_id(f, var_id)?;
    }

    Ok(())
}

// ability_ref -> "Console" or "Console(Int)"
inventory::submit! {
    Printable::implement("core", "ability_ref", |db, ty, f| {
        let Some(ability) = AbilityRefType::from_type(db, ty) else {
            return f.write_str("?ability");
        };

        let Some(name) = ability.name(db) else {
            return f.write_str("?ability");
        };

        let params = ability.params(db);
        // Get name string outside of lock to avoid nested lock when printing params
        let name_str = name.to_string();

        if params.is_empty() {
            f.write_str(&name_str)
        } else {
            f.write_str(&name_str)?;
            f.write_char('(')?;
            for (i, &p) in params.iter().enumerate() {
                if i > 0 {
                    f.write_str(", ")?;
                }
                Printable::print_type(db, p, f)?;
            }
            f.write_char(')')
        }
    })
}

// ref_ -> "Type?" (nullable) or "Type"
inventory::submit! {
    Printable::implement("core", "ref_", |db, ty, f| {
        let params = ty.params(db);
        let nullable = matches!(
            ty.get_attr(db, Ref::nullable_sym()),
            Some(Attribute::Bool(true))
        );

        if let Some(&pointee) = params.first() {
            Printable::print_type(db, pointee, f)?;
            if nullable {
                f.write_char('?')?;
            }
            Ok(())
        } else if nullable {
            f.write_char('?')
        } else {
            f.write_str("Ref(?)")
        }
    })
}

/// Convert a variable ID to a readable name (a, b, c, ..., t0, t1, ...).
fn fmt_var_id(f: &mut Formatter<'_>, id: u64) -> fmt::Result {
    if id < 26 {
        f.write_char((b'a' + id as u8) as char)
    } else {
        write!(f, "t{}", id - 26)
    }
}
