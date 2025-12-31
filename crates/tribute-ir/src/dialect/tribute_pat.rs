//! Tribute pattern dialect operations.
//!
//! High-level pattern representation that preserves source-level structure.
//! Patterns are used in `tribute.case` expressions, `tribute.let` bindings,
//! and function parameters.
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
//! Each `tribute.arm` has a pattern region containing these operations.
//!
//! ```text
//! tribute.arm {
//!     pattern {
//!         tribute_pat.variant("Some") {
//!             fields { tribute_pat.bind("x") }
//!         }
//!     }
//!     body { ... }
//! }
//! ```

use trunk_ir::dialect;

dialect! {
    mod tribute_pat {
        /// `tribute_pat.wildcard` operation: wildcard pattern (`_`).
        /// Matches anything without binding.
        fn wildcard();

        /// `tribute_pat.bind` operation: identifier pattern that binds a value.
        /// Matches anything and binds it to the given name.
        /// The binding is accessible via `tribute.bind` in the arm body.
        #[attr(name: Symbol)]
        fn bind();

        /// `tribute_pat.literal` operation: literal pattern.
        /// Matches a specific literal value (int, string, bool, nil, rune).
        #[attr(value: any)]
        fn literal();

        /// `tribute_pat.variant` operation: variant/constructor pattern.
        /// Matches a specific variant and destructures its fields.
        /// The fields region contains patterns for each field.
        #[attr(variant: QualifiedName)]
        fn variant() {
            #[region(fields)] {}
        };

        /// `tribute_pat.tuple` operation: tuple pattern.
        /// Matches a tuple and destructures its elements.
        fn tuple() {
            #[region(elements)] {}
        };

        /// `tribute_pat.list` operation: exact list pattern.
        /// Matches a list with exactly the given elements.
        fn list() {
            #[region(elements)] {}
        };

        /// `tribute_pat.list_rest` operation: list pattern with rest.
        /// Matches a list with the given head elements and binds the rest.
        /// `[head1, head2, ..rest]` or `[head1, head2, ..]`
        #[attr(rest_name: Symbol)]
        fn list_rest() {
            #[region(head)] {}
        };

        /// `tribute_pat.record` operation: record pattern.
        /// Matches a record and destructures specified fields.
        fn record() {
            #[region(fields)] {}
        };

        /// `tribute_pat.field` operation: field pattern within a record pattern.
        /// Matches a specific field with the given pattern.
        #[attr(field: Symbol)]
        fn field() {
            #[region(pattern)] {}
        };

        /// `tribute_pat.as_pat` operation: as pattern.
        /// Matches the inner pattern and also binds the whole value to a name.
        /// `Some(x) as opt`
        #[attr(name: Symbol)]
        fn as_pat() {
            #[region(inner)] {}
        };

        /// `tribute_pat.or` operation: or pattern (alternative patterns).
        /// Matches if any of the alternative patterns match.
        /// `None | Some(0)`
        fn or() {
            #[region(alternatives)] {}
        };

        // === Handler Patterns (for ability effect handling) ===

        /// `tribute_pat.handler_done` operation: matches Done variant of Request.
        /// Used for `{ result }` patterns in handle expressions.
        /// The result region contains the pattern for the result value.
        fn handler_done() {
            #[region(result)] {}
        };

        /// `tribute_pat.handler_suspend` operation: matches Suspend variant for a specific ability operation.
        /// Used for `{ Op(args) -> k }` patterns in handle expressions.
        /// - `ability_ref`: ability type (core.ability_ref) to support parameterized abilities
        /// - `op`: the operation name within the ability
        /// - `args`: region containing patterns for operation arguments
        /// - `continuation`: region containing pattern for continuation (tribute_pat.bind or wildcard)
        #[attr(ability_ref: Type, op: Symbol)]
        fn handler_suspend() {
            #[region(args)] {}
            #[region(continuation)] {}
        };
    }
}

// === Pattern Region Builders ===

/// Helper functions for building pattern regions.
///
/// These helpers create single-operation pattern regions for simple cases.
///
/// ```
/// # use salsa::Database;
/// # use salsa::DatabaseImpl;
/// # use trunk_ir::{Location, PathId, Region, Span};
/// # use tribute_ir::dialect::tribute_pat::helpers;
/// # #[salsa::tracked]
/// # fn build_pattern(db: &dyn salsa::Database) -> Region<'_> {
/// #     let path = PathId::new(db, "file:///test.trb".to_owned());
/// #     let location = Location::new(path, Span::new(0, 0));
///       helpers::wildcard_region(db, location)
/// # }
/// # DatabaseImpl::default().attach(|db| {
/// #     let pattern = build_pattern(db);
/// #     let _ = pattern;
/// # });
/// ```
pub mod helpers {
    use super::*;
    use trunk_ir::Location;
    use trunk_ir::{
        Attribute, Block, BlockId, DialectOp, IdVec, Operation, QualifiedName, Region, Symbol, Type,
    };

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
        name: Symbol,
    ) -> Region<'db> {
        let op = bind(db, location, name);
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
        variant_path: QualifiedName,
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
        rest_name: Symbol,
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
            IdVec::from(vec![Block::new(
                db,
                BlockId::fresh(),
                location,
                IdVec::new(),
                IdVec::new(),
            )]),
        )
    }

    /// Helper to create a region containing a single operation.
    pub fn single_op_region<'db>(
        db: &'db dyn salsa::Database,
        location: Location<'db>,
        op: Operation<'db>,
    ) -> Region<'db> {
        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            IdVec::new(),
            IdVec::from(vec![op]),
        );
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
    ///
    /// The `ability_ref` should be a `core.ability_ref` type created via
    /// `AbilityRefType::simple()` or `AbilityRefType::with_params()`.
    ///
    /// The `continuation_pattern` should be a region containing `tribute_pat.bind`
    /// for named continuations or `tribute_pat.wildcard` for discarded continuations.
    pub fn handler_suspend_region<'db>(
        db: &'db dyn salsa::Database,
        location: Location<'db>,
        ability_ref: Type<'db>,
        op_name: Symbol,
        args_pattern: Region<'db>,
        continuation_pattern: Region<'db>,
    ) -> Region<'db> {
        let op = handler_suspend(
            db,
            location,
            ability_ref,
            op_name,
            args_pattern,
            continuation_pattern,
        );
        single_op_region(db, location, op.as_operation())
    }
}
