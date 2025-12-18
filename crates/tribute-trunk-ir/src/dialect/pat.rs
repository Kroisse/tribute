//! Pattern dialect operations.
//!
//! High-level pattern representation that preserves source-level structure.
//! Patterns are used in `case` expressions, `let` bindings, and function parameters.
//!
//! ## Pattern Types
//!
//! - Wildcard patterns (`_`)
//! - Identifier patterns (variable binding)
//! - Literal patterns (numbers, strings, bools)
//! - Variant patterns (`Some(x)`, `None`)
//! - Record patterns (`{ name, age }`)
//! - List patterns (`[]`, `[head, ..tail]`)
//! - Tuple patterns (`#(a, b)`)
//! - As patterns (`pat as x`)
//! - Or patterns (`None | Some(0)`)
//!
//! ## Usage
//!
//! Patterns are represented as a tree of operations within a region.
//! Each `case.arm` has a pattern region containing these operations.
//!
//! ```text
//! case.arm {
//!     pattern {
//!         pat.variant("Some") {
//!             fields { pat.bind("x") }
//!         }
//!     }
//!     body { ... }
//! }
//! ```

use crate::dialect;

dialect! {
    mod pat {
        /// `pat.wildcard` operation: wildcard pattern (`_`).
        /// Matches anything without binding.
        fn wildcard();

        /// `pat.bind` operation: identifier pattern that binds a value.
        /// Matches anything and binds it to the given name.
        /// The binding is accessible via `case.bind` in the arm body.
        #[attr(name: Symbol)]
        fn bind();

        /// `pat.literal` operation: literal pattern.
        /// Matches a specific literal value (int, string, bool, nil, rune).
        #[attr(value: any)]
        fn literal();

        /// `pat.variant` operation: variant/constructor pattern.
        /// Matches a specific variant and destructures its fields.
        /// The fields region contains patterns for each field.
        #[attr(variant: SymbolRef)]
        fn variant() {
            #[region(fields)] {}
        };

        /// `pat.tuple` operation: tuple pattern.
        /// Matches a tuple and destructures its elements.
        fn tuple() {
            #[region(elements)] {}
        };

        /// `pat.list` operation: exact list pattern.
        /// Matches a list with exactly the given elements.
        fn list() {
            #[region(elements)] {}
        };

        /// `pat.list_rest` operation: list pattern with rest.
        /// Matches a list with the given head elements and binds the rest.
        /// `[head1, head2, ..rest]` or `[head1, head2, ..]`
        #[attr(rest_name: Symbol)]
        fn list_rest() {
            #[region(head)] {}
        };

        /// `pat.record` operation: record pattern.
        /// Matches a record and destructures specified fields.
        fn record() {
            #[region(fields)] {}
        };

        /// `pat.field` operation: field pattern within a record pattern.
        /// Matches a specific field with the given pattern.
        #[attr(field: Symbol)]
        fn field() {
            #[region(pattern)] {}
        };

        /// `pat.as_pat` operation: as pattern.
        /// Matches the inner pattern and also binds the whole value to a name.
        /// `Some(x) as opt`
        #[attr(name: Symbol)]
        fn as_pat() {
            #[region(inner)] {}
        };

        /// `pat.or` operation: or pattern (alternative patterns).
        /// Matches if any of the alternative patterns match.
        /// `None | Some(0)`
        fn or() {
            #[region(alternatives)] {}
        };

        // === Handler Patterns (for ability effect handling) ===

        /// `pat.handler_done` operation: matches Done variant of Request.
        /// Used for `{ result }` patterns in handle expressions.
        /// The result region contains the pattern for the result value.
        fn handler_done() {
            #[region(result)] {}
        };

        /// `pat.handler_suspend` operation: matches Suspend variant for a specific ability operation.
        /// Used for `{ Op(args) -> k }` patterns in handle expressions.
        /// - `ability_ref`: path to the ability being handled
        /// - `op`: the operation name within the ability
        /// - `args`: region containing patterns for operation arguments
        /// - `continuation`: name to bind the continuation (or empty for discard)
        #[attr(ability_ref: SymbolRef, op: Symbol, continuation: Symbol)]
        fn handler_suspend() {
            #[region(args)] {}
        };
    }
}

// === Pattern Region Builders ===

/// Helper functions for building pattern regions.
///
/// These helpers create single-operation pattern regions for simple cases.
pub mod helpers {
    use super::*;
    use crate::{Attribute, Block, DialectOp, IdVec, Operation, Region, Symbol};
    use tribute_core::Location;

    /// Create a wildcard pattern region (`_`).
    pub fn wildcard_region<'db>(
        db: &'db dyn salsa::Database,
        location: Location<'db>,
    ) -> Region<'db> {
        let op = wildcard(db, location);
        single_op_region(db, location, op.as_operation())
    }

    /// Create an identifier/binding pattern region (`x`).
    pub fn bind_region<'db>(
        db: &'db dyn salsa::Database,
        location: Location<'db>,
        name: &str,
    ) -> Region<'db> {
        let op = bind(db, location, Symbol::new(db, name));
        single_op_region(db, location, op.as_operation())
    }

    /// Create a literal integer pattern region.
    pub fn int_region<'db>(
        db: &'db dyn salsa::Database,
        location: Location<'db>,
        value: i64,
    ) -> Region<'db> {
        let op = literal(db, location, Attribute::from(value));
        single_op_region(db, location, op.as_operation())
    }

    /// Create a literal string pattern region.
    pub fn string_region<'db>(
        db: &'db dyn salsa::Database,
        location: Location<'db>,
        value: &str,
    ) -> Region<'db> {
        let op = literal(db, location, Attribute::String(value.to_string()));
        single_op_region(db, location, op.as_operation())
    }

    /// Create a literal boolean pattern region.
    pub fn bool_region<'db>(
        db: &'db dyn salsa::Database,
        location: Location<'db>,
        value: bool,
    ) -> Region<'db> {
        let op = literal(db, location, Attribute::Bool(value));
        single_op_region(db, location, op.as_operation())
    }

    /// Create a variant pattern region with nested field patterns.
    pub fn variant_region<'db>(
        db: &'db dyn salsa::Database,
        location: Location<'db>,
        variant_path: IdVec<Symbol<'db>>,
        fields: Region<'db>,
    ) -> Region<'db> {
        let op = variant(db, location, variant_path, fields);
        single_op_region(db, location, op.as_operation())
    }

    /// Create a tuple pattern region with nested element patterns.
    pub fn tuple_region<'db>(
        db: &'db dyn salsa::Database,
        location: Location<'db>,
        elements: Region<'db>,
    ) -> Region<'db> {
        let op = tuple(db, location, elements);
        single_op_region(db, location, op.as_operation())
    }

    /// Create a list pattern region with nested element patterns.
    pub fn list_region<'db>(
        db: &'db dyn salsa::Database,
        location: Location<'db>,
        elements: Region<'db>,
    ) -> Region<'db> {
        let op = list(db, location, elements);
        single_op_region(db, location, op.as_operation())
    }

    /// Create a list pattern with rest.
    pub fn list_rest_region<'db>(
        db: &'db dyn salsa::Database,
        location: Location<'db>,
        rest_name: Symbol<'db>,
        head: Region<'db>,
    ) -> Region<'db> {
        let op = list_rest(db, location, rest_name, head);
        single_op_region(db, location, op.as_operation())
    }

    /// Create an empty pattern region (for empty field lists, etc.)
    pub fn empty_region<'db>(db: &'db dyn salsa::Database, location: Location<'db>) -> Region<'db> {
        Region::new(
            db,
            location,
            IdVec::from(vec![Block::new(db, location, IdVec::new(), IdVec::new())]),
        )
    }

    /// Helper to create a region containing a single operation.
    pub fn single_op_region<'db>(
        db: &'db dyn salsa::Database,
        location: Location<'db>,
        op: Operation<'db>,
    ) -> Region<'db> {
        let block = Block::new(db, location, IdVec::new(), IdVec::from(vec![op]));
        Region::new(db, location, IdVec::from(vec![block]))
    }

    // === Handler Pattern Helpers ===

    /// Create a handler_done pattern region for `{ result }` patterns.
    pub fn handler_done_region<'db>(
        db: &'db dyn salsa::Database,
        location: Location<'db>,
        result_pattern: Region<'db>,
    ) -> Region<'db> {
        let op = handler_done(db, location, result_pattern);
        single_op_region(db, location, op.as_operation())
    }

    /// Create a handler_suspend pattern region for `{ Op(args) -> k }` patterns.
    pub fn handler_suspend_region<'db>(
        db: &'db dyn salsa::Database,
        location: Location<'db>,
        ability_ref: IdVec<Symbol<'db>>,
        op_name: Symbol<'db>,
        args_pattern: Region<'db>,
        continuation_name: Symbol<'db>,
    ) -> Region<'db> {
        let op = handler_suspend(
            db,
            location,
            ability_ref,
            op_name,
            continuation_name,
            args_pattern,
        );
        single_op_region(db, location, op.as_operation())
    }
}
