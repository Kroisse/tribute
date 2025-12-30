//! Tribute language AST/HIR dialect.
//!
//! This dialect represents ALL Tribute-specific high-level operations that exist
//! before lowering to target-independent IR (cont, func, scf, etc.).
//!
//! ## Dialect Organization
//!
//! **Unresolved operations** (eliminated after resolve/TDNR):
//! - `tribute.var`, `tribute.call`, `tribute.path`, `tribute.binop`
//! - `tribute.type` (unresolved type reference)
//!
//! **Type definitions** (metadata):
//! - `tribute.struct_def`, `tribute.enum_def`, `tribute.ability_def`
//! - `tribute.field_def`, `tribute.variant_def` (helpers inside definitions)
//!
//! **Effect definitions** (inside ability_def):
//! - `tribute.op` (ability operation declaration)
//! - `tribute.prompt` (handler expression)
//!
//! **Block and control flow**:
//! - `tribute.block`, `tribute.yield` (unified src.yield + case.yield)
//! - `tribute.lambda`, `tribute.tuple`, `tribute.const`, `tribute.use`
//!
//! **Pattern matching** (lowered to scf dialect):
//! - `tribute.case`, `tribute.arm`, `tribute.let`, `tribute.bind`, `tribute.guard`
//!
//! **Types**:
//! - `tribute.type` (unresolved type reference)
//! - `tribute.int`, `tribute.nat` (primitive types)
//! - `tribute.type_var`, `tribute.error_type` (type inference)
//!
//! ## Pattern Region
//!
//! Patterns are represented using the `tribute_pat` dialect operations
//! within a region of `tribute.arm` or `tribute.let`.

use std::collections::BTreeMap;
use std::fmt::Write;

use trunk_ir::type_interface::Printable;
use trunk_ir::{Attribute, Attrs, IdVec, Location, Symbol, dialect};

trunk_ir::symbols! {
    VAR_ID_ATTR => "id",
}

dialect! {
    mod tribute {
        // === Unresolved operations (eliminated after resolve/TDNR) ===

        /// `tribute.call` operation: unresolved function call.
        /// The callee name will be resolved to a concrete function reference.
        #[attr(name: QualifiedName)]
        fn call(#[rest] args) -> result;

        /// `tribute.cons` operation: unresolved constructor application.
        /// The constructor name will be resolved to a struct/variant constructor.
        #[attr(name: QualifiedName)]
        fn cons(#[rest] args) -> result;

        /// `tribute.var` operation: unresolved variable reference (single name).
        /// May resolve to local binding or module-level definition.
        #[attr(name: Symbol)]
        fn var() -> result;

        /// `tribute.path` operation: explicitly qualified path reference.
        /// Always refers to a module-level or type-level definition, never local.
        #[attr(path: QualifiedName)]
        fn path() -> result;

        /// `tribute.binop` operation: unresolved binary operation.
        /// Used for operators that need type-directed resolution (e.g., `<>` concat).
        /// The `op` attribute holds the operator name.
        #[attr(op: Symbol)]
        fn binop(lhs, rhs) -> result;

        // === Block and control flow ===

        /// `tribute.block` operation: block expression.
        /// Preserves block structure for source mapping and analysis.
        /// The body region contains the statements, and the result is the block's value.
        fn block() -> result {
            #[region(body)] {}
        };

        /// `tribute.yield` operation: yields a value from a block or case arm.
        /// Used to specify the result value of a `tribute.block` or `tribute.arm`.
        fn r#yield(value);

        /// `tribute.let` operation: let binding with pattern matching.
        /// Binds the value operand to names defined in the pattern region.
        /// The pattern region uses `tribute_pat.*` operations.
        fn r#let(value) {
            #[region(pattern)] {}
        };

        /// `tribute.lambda` operation: lambda expression.
        /// Represents an anonymous function before capture analysis.
        /// The `type` attribute holds the function type (params -> result).
        /// The body region contains the lambda body, ending with `tribute.yield`.
        #[attr(r#type: Type)]
        fn lambda() -> result {
            #[region(body)] {}
        };

        /// `tribute.tuple` operation: tuple construction.
        /// Takes variadic operands (tuple elements) and produces a tuple value.
        fn tuple(#[rest] elements) -> result;

        /// `tribute.const` operation: constant definition.
        /// Represents a named constant value before resolution.
        /// Unlike functions, constants are evaluated once and their value is inlined at use sites.
        /// The `value` attribute holds the literal value (IntBits, FloatBits, String, etc.).
        #[attr(name: Symbol, value: any)]
        fn r#const() -> result;

        /// `tribute.use` operation: import declaration.
        /// Carries the fully qualified path and an optional local alias.
        #[attr(path: QualifiedName, alias: Symbol, is_pub: bool)]
        fn r#use();

        // === Type declarations (metadata) ===

        /// `tribute.struct_def` operation: defines a struct type.
        ///
        /// Attributes:
        /// - `sym_name`: The name of the struct type
        /// - `fields`: Field definitions as [(name, type)] pairs
        #[attr(sym_name, fields)]
        fn struct_def() -> result;

        /// `tribute.enum_def` operation: defines an enum (sum) type.
        ///
        /// Attributes:
        /// - `sym_name`: The name of the enum type
        /// - `variants`: Variant definitions as [(name, fields)] pairs
        #[attr(sym_name, variants)]
        fn enum_def() -> result;

        /// `tribute.ability_def` operation: defines an ability (effect) type.
        ///
        /// The operations region contains `tribute.op` operations defining the signatures.
        ///
        /// Attributes:
        /// - `sym_name`: The name of the ability
        #[attr(sym_name)]
        fn ability_def() -> result {
            #[region(operations)] {}
        };

        // === Effect definition operations (metadata) ===

        /// `tribute.op` operation: declares an operation signature within an ability.
        ///
        /// Used inside `tribute.ability_def` operations region to define what
        /// operations the ability provides.
        ///
        /// Attributes:
        /// - `sym_name`: The operation name
        /// - `type`: The operation's function type (func.Fn)
        #[attr(sym_name: Symbol, r#type: Type)]
        fn op();

        /// `tribute.prompt` operation: runs body in a delimited context.
        ///
        /// Executes the body region until it either:
        /// - Completes with a value → returns `Request::Done(value)`
        /// - Performs an ability operation → returns `Request::Suspend(op, args, continuation)`
        ///
        /// The returned `Request` is typically pattern-matched using `tribute.case`
        /// with handler patterns.
        fn prompt() -> request {
            #[region(body)] {}
        };

        // === Pattern matching (lowered to scf dialect) ===

        /// `tribute.case` operation: pattern matching expression.
        /// Takes a scrutinee and has a body region containing `tribute.arm` operations.
        /// All arms must yield the same type.
        fn case(scrutinee) -> result {
            #[region(body)] {}
        };

        /// `tribute.arm` operation: a single pattern-matching arm.
        /// The pattern region contains a tree of `tribute_pat.*` operations.
        /// The body region contains the arm's expression (ends with `tribute.yield`).
        /// Guards are represented as nested `tribute.guard` operations within the body.
        fn arm() {
            #[region(pattern)] {}
            #[region(body)] {}
        };

        /// `tribute.guard` operation: conditional guard on a pattern arm.
        /// Executes the condition; if true, continues with the guarded body.
        /// The `else` region is for fallthrough to next arm/guard.
        fn guard(cond) {
            #[region(then)] {}
            #[region(r#else)] {}
        };

        /// `tribute.bind` operation: extracts a bound variable from pattern matching.
        /// The `name` attribute is the binding name.
        /// This is placed at the start of arm body to introduce bindings.
        // NOTE: Consider whether tribute.var could replace this by referencing pattern bindings.
        // Currently kept separate because pattern bindings don't have concrete values at resolve time.
        #[attr(name: Symbol)]
        fn bind() -> result;

        // === Types ===

        /// `tribute.type`: an unresolved type reference that needs name resolution.
        /// The `name` attribute holds the type name (e.g., "Int", "List").
        /// The `params` hold type arguments for generic types (e.g., `List(a)`).
        #[attr(name: Symbol)]
        type r#type(#[rest] params);

        /// `tribute.int` type: arbitrary precision integer (Fixnum/Bignum hybrid).
        /// At runtime, represented as i31ref (fixnum) or BigInt struct (bignum).
        type int;

        /// `tribute.nat` type: arbitrary precision natural number (non-negative).
        /// Semantically a subset of Int, but may have optimized representation.
        type nat;

        /// `tribute.type_var` type: a type variable to be resolved during type inference.
        /// The `id` attribute holds a unique variable ID.
        #[attr(id: any)]
        type type_var;

        /// `tribute.error_type` type: an error type indicating type resolution failed.
        type error_type;
    }
}

// === Type variable helper functions ===

/// Create a type variable (`tribute.type_var`) to be resolved during type inference.
///
/// The `attrs` can carry metadata such as a unique variable ID or constraints.
pub fn new_type_var<'db>(db: &'db dyn salsa::Database, attrs: Attrs<'db>) -> trunk_ir::Type<'db> {
    trunk_ir::Type::new(db, DIALECT_NAME(), TYPE_VAR(), IdVec::new(), attrs)
}

/// Create a type variable with a numeric ID.
pub fn type_var_with_id<'db>(db: &'db dyn salsa::Database, id: u64) -> trunk_ir::Type<'db> {
    new_type_var(
        db,
        BTreeMap::from([(VAR_ID_ATTR(), Attribute::IntBits(id))]),
    )
}

/// Create an error type (`tribute.error_type`) indicating type resolution failed.
///
/// The `attrs` can carry error information or source location.
pub fn new_error_type<'db>(db: &'db dyn salsa::Database, attrs: Attrs<'db>) -> trunk_ir::Type<'db> {
    trunk_ir::Type::new(db, DIALECT_NAME(), ERROR_TYPE(), IdVec::new(), attrs)
}

/// Check if a type is a type variable (`tribute.type_var`).
pub fn is_type_var(db: &dyn salsa::Database, ty: trunk_ir::Type<'_>) -> bool {
    ty.is_dialect(db, DIALECT_NAME(), TYPE_VAR())
}

/// Check if a type is an error type (`tribute.error_type`).
pub fn is_error_type(db: &dyn salsa::Database, ty: trunk_ir::Type<'_>) -> bool {
    ty.is_dialect(db, DIALECT_NAME(), ERROR_TYPE())
}

// === Convenience function for creating unresolved types ===

/// Create an unresolved type reference (`tribute.type`).
///
/// This is a convenience wrapper around `Type::new` that takes a string name.
pub fn unresolved_type<'db>(
    db: &'db dyn salsa::Database,
    name: Symbol,
    params: IdVec<trunk_ir::Type<'db>>,
) -> trunk_ir::Type<'db> {
    // Use the macro-generated Type struct
    *Type::new(db, params, name)
}

// === Pattern Region Builders ===

/// Re-export pattern helpers from the `tribute_pat` dialect.
pub use super::tribute_pat::helpers as pattern;

impl<'db> Arm<'db> {
    /// Create a wildcard arm that matches anything.
    pub fn wildcard(
        db: &'db dyn salsa::Database,
        location: Location<'db>,
        body: trunk_ir::Region<'db>,
    ) -> Self {
        let pattern_region = pattern::wildcard_region(db, location);
        arm(db, location, pattern_region, body)
    }

    /// Create an arm that binds the scrutinee to a name.
    pub fn binding(
        db: &'db dyn salsa::Database,
        location: Location<'db>,
        name: Symbol,
        body: trunk_ir::Region<'db>,
    ) -> Self {
        let pattern_region = pattern::bind_region(db, location, name);
        arm(db, location, pattern_region, body)
    }

    /// Create an arm matching a unit variant (e.g., `None`).
    pub fn unit_variant(
        db: &'db dyn salsa::Database,
        location: Location<'db>,
        variant_name: Symbol,
        body: trunk_ir::Region<'db>,
    ) -> Self {
        let variant_path = trunk_ir::QualifiedName::simple(variant_name);
        let fields = pattern::empty_region(db, location);
        let pattern_region = pattern::variant_region(db, location, variant_path, fields);
        arm(db, location, pattern_region, body)
    }
}

// === Printable interface registrations ===

use std::fmt::Formatter;

// tribute.type -> "Name" or "Name(params...)"
inventory::submit! {
    Printable::implement("tribute", "type", |db, ty, f| {
        let Some(Attribute::Symbol(name)) = ty.get_attr(db, Type::name_sym()) else {
            return f.write_str("?unresolved");
        };

        let params = ty.params(db);

        // Capitalize first letter
        let name_text = name.to_string();
        let mut chars = name_text.chars();
        if let Some(c) = chars.next() {
            for ch in c.to_uppercase() {
                f.write_char(ch)?;
            }
            f.write_str(chars.as_str())?;
        }

        if !params.is_empty() {
            f.write_char('(')?;
            for (i, &p) in params.iter().enumerate() {
                if i > 0 {
                    f.write_str(", ")?;
                }
                Printable::print_type(db, p, f)?;
            }
            f.write_char(')')?;
        }

        Ok(())
    })
}

// tribute.type_var -> "a", "b", ..., "t0", "t1", ...
inventory::submit! {
    Printable::implement("tribute", "type_var", |db, ty, f| {
        if let Some(Attribute::IntBits(id)) = ty.get_attr(db, VAR_ID_ATTR()) {
            fmt_var_id(f, *id)
        } else {
            f.write_char('?')
        }
    })
}

// tribute.error_type -> "<error>"
inventory::submit! { Printable::implement("tribute", "error_type", |_, _, f| f.write_str("<error>")) }

// tribute.int -> "Int"
inventory::submit! { Printable::implement("tribute", "int", |_, _, f| f.write_str("Int")) }

// tribute.nat -> "Nat"
inventory::submit! { Printable::implement("tribute", "nat", |_, _, f| f.write_str("Nat")) }

/// Convert a variable ID to a readable name (a, b, c, ..., t0, t1, ...).
fn fmt_var_id(f: &mut Formatter<'_>, id: u64) -> std::fmt::Result {
    if id < 26 {
        f.write_char((b'a' + id as u8) as char)
    } else {
        write!(f, "t{}", id - 26)
    }
}
