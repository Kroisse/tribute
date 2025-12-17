//! Case dialect operations.
//!
//! High-level pattern matching dialect that preserves source-level structure.
//! This dialect is lowered to `scf` primitives during compilation.
//!
//! Pattern matching in Tribute supports:
//! - Literal patterns (numbers, strings, runes, bools, Nil)
//! - Wildcard patterns (`_`)
//! - Identifier patterns (variable binding)
//! - Variant patterns (`Some(x)`, `None`, `Ok { value }`)
//! - Record patterns (`User { name, age }`)
//! - List patterns (`[]`, `[head, ..tail]`)
//! - Tuple patterns (`#(a, b)`)
//! - Handler patterns (`{ result }`, `{ Op(x) -> k }`)
//! - As patterns (`pat as x`)
//! - Guards (`if condition`)

use crate::{Attribute, dialect};
use tribute_core::Location;

dialect! {
    mod case {
        // === Main Pattern Matching ===

        /// `case.case` operation: pattern matching expression.
        /// Takes a scrutinee and has a body region containing `case.arm` operations.
        /// All arms must yield the same type.
        fn r#case(scrutinee) -> result {
            #[region(body)] {}
        };

        /// `case.arm` operation: a single pattern-matching arm.
        /// The `pattern` attribute describes what to match.
        /// The body region contains the arm's expression (ends with `case.yield`).
        /// Guards are represented as nested `case.guard` operations within the body.
        #[attr(pattern: any)]
        fn arm() {
            #[region(body)] {}
        };

        /// `case.guard` operation: conditional guard on a pattern arm.
        /// Executes the condition; if true, continues with the guarded body.
        /// The `else` region is for fallthrough to next arm/guard.
        fn guard(cond) {
            #[region(then)] {}
            #[region(r#else)] {}
        };

        /// `case.yield` operation: returns a value from a case arm.
        fn r#yield(value);

        // === Pattern Binding ===

        /// `case.bind` operation: extracts a bound variable from pattern matching.
        /// The `name` attribute is the binding name.
        /// This is placed at the start of arm body to introduce bindings.
        #[attr(name: Symbol)]
        fn bind() -> result;

        // === Variant/Record Destructuring ===

        /// `case.destruct_variant` operation: extracts variant payload.
        /// Used when matching `Some(x)` to extract `x`.
        #[attr(tag: Symbol)]
        fn destruct_variant(scrutinee) -> result;

        /// `case.destruct_field` operation: extracts a field from record/struct.
        /// Used when matching `User { name, age }` to extract fields.
        #[attr(field: Symbol)]
        fn destruct_field(scrutinee) -> result;

        /// `case.destruct_tuple` operation: extracts element from tuple.
        /// The `index` attribute specifies which element.
        #[attr(index: any)]
        fn destruct_tuple(scrutinee) -> result;

        // === List Destructuring ===

        /// `case.list_head` operation: extracts the head of a list.
        /// Used for `[head, ..tail]` patterns.
        #[attr(elem_type: Type)]
        fn list_head(list) -> result;

        /// `case.list_tail` operation: extracts the tail of a list.
        /// Used for `[head, ..tail]` patterns.
        #[attr(elem_type: Type)]
        fn list_tail(list) -> result;

        /// `case.list_len` operation: gets the length of a list.
        /// Used for checking list pattern length requirements.
        fn list_len(list) -> result;

        // === Handler Patterns ===

        /// `case.handler_done` operation: checks if Request is Done.
        /// Returns the result value if Done, otherwise control flows to else.
        fn handler_done(request) -> result {
            #[region(then)] {}
            #[region(r#else)] {}
        };

        /// `case.handler_suspend` operation: checks if Request is Suspend for a specific effect op.
        /// If matched, extracts the operation arguments and continuation.
        /// The `effect_op` attribute identifies which ability operation to match.
        #[attr(effect_op: Symbol)]
        fn handler_suspend(request) -> (args, continuation) {
            #[region(then)] {}
            #[region(r#else)] {}
        };
    }
}

// === Pattern Attribute Builders ===

/// Pattern representation for case.arm attributes.
///
/// Patterns are stored as `Attribute` values. This module provides
/// builders for creating well-formed pattern attributes.
pub mod pattern {
    use super::*;

    /// Create a wildcard pattern (`_`).
    pub fn wildcard<'db>() -> Attribute<'db> {
        Attribute::String("_".to_string())
    }

    /// Create an identifier/binding pattern (`x`).
    pub fn ident<'db>(name: &str) -> Attribute<'db> {
        Attribute::String(format!("${}", name))
    }

    /// Create a literal integer pattern.
    pub fn int<'db>(value: i64) -> Attribute<'db> {
        Attribute::String(format!("lit:{}", value))
    }

    /// Create a literal string pattern.
    pub fn string<'db>(value: &str) -> Attribute<'db> {
        Attribute::String(format!("lit:\"{}\"", value))
    }

    /// Create a literal boolean pattern.
    pub fn bool<'db>(value: bool) -> Attribute<'db> {
        Attribute::String(format!("lit:{}", if value { "True" } else { "False" }))
    }

    /// Create a unit variant pattern (e.g., `None`).
    pub fn unit_variant<'db>(name: &str) -> Attribute<'db> {
        Attribute::String(name.to_string())
    }

    /// Create a variant pattern with positional fields (e.g., `Some(x)`).
    pub fn variant<'db>(name: &str, fields: &[Attribute<'db>]) -> Attribute<'db> {
        let field_strs: Vec<_> = fields
            .iter()
            .map(|a| match a {
                Attribute::String(s) => s.clone(),
                _ => format!("{:?}", a),
            })
            .collect();
        Attribute::String(format!("{}({})", name, field_strs.join(", ")))
    }

    /// Create a tuple pattern (e.g., `#(a, b)`).
    pub fn tuple<'db>(elements: &[Attribute<'db>]) -> Attribute<'db> {
        let elem_strs: Vec<_> = elements
            .iter()
            .map(|a| match a {
                Attribute::String(s) => s.clone(),
                _ => format!("{:?}", a),
            })
            .collect();
        Attribute::String(format!("#({})", elem_strs.join(", ")))
    }

    /// Create a list pattern (e.g., `[a, b, c]`).
    pub fn list<'db>(elements: &[Attribute<'db>]) -> Attribute<'db> {
        let elem_strs: Vec<_> = elements
            .iter()
            .map(|a| match a {
                Attribute::String(s) => s.clone(),
                _ => format!("{:?}", a),
            })
            .collect();
        Attribute::String(format!("[{}]", elem_strs.join(", ")))
    }

    /// Create a list pattern with rest (e.g., `[head, ..tail]`).
    pub fn list_rest<'db>(head: &[Attribute<'db>], rest: Option<&str>) -> Attribute<'db> {
        let mut parts: Vec<_> = head
            .iter()
            .map(|a| match a {
                Attribute::String(s) => s.clone(),
                _ => format!("{:?}", a),
            })
            .collect();
        match rest {
            Some(name) => parts.push(format!("..{}", name)),
            None => parts.push("..".to_string()),
        }
        Attribute::String(format!("[{}]", parts.join(", ")))
    }

    /// Create an as pattern (e.g., `Some(x) as opt`).
    pub fn as_pattern<'db>(inner: Attribute<'db>, name: &str) -> Attribute<'db> {
        let inner_str = match &inner {
            Attribute::String(s) => s.clone(),
            _ => format!("{:?}", inner),
        };
        Attribute::String(format!("{} as {}", inner_str, name))
    }
}

impl<'db> Arm<'db> {
    /// Create a wildcard arm that matches anything.
    pub fn wildcard(
        db: &'db dyn salsa::Database,
        location: Location<'db>,
        body: crate::Region<'db>,
    ) -> Self {
        arm(db, location, pattern::wildcard(), body)
    }

    /// Create an arm that binds the scrutinee to a name.
    pub fn binding(
        db: &'db dyn salsa::Database,
        location: Location<'db>,
        name: &str,
        body: crate::Region<'db>,
    ) -> Self {
        arm(db, location, pattern::ident(name), body)
    }

    /// Create an arm matching a unit variant (e.g., `None`).
    pub fn unit_variant(
        db: &'db dyn salsa::Database,
        location: Location<'db>,
        variant_name: &str,
        body: crate::Region<'db>,
    ) -> Self {
        arm(db, location, pattern::unit_variant(variant_name), body)
    }
}
