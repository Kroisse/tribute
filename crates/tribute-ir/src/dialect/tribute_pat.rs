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
//! - Variant patterns (`Some(x)`, `None`)
//! - Tuple patterns (`#(a, b)`)
//! - As patterns (`pat as x`)
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
        /// The binding is accessible via `tribute.var` in the arm body,
        /// resolved by the resolver and case_lowering passes.
        #[attr(name: Symbol)]
        fn bind();

        /// `tribute_pat.variant` operation: variant/constructor pattern.
        /// Matches a specific variant and destructures its fields.
        /// The fields region contains patterns for each field.
        #[attr(variant: Symbol)]
        fn variant() {
            #[region(fields)] {}
        };

        /// `tribute_pat.tuple` operation: tuple pattern.
        /// Matches a tuple and destructures its elements.
        fn tuple() {
            #[region(elements)] {}
        };

        /// `tribute_pat.as_pat` operation: as pattern.
        /// Matches the inner pattern and also binds the whole value to a name.
        /// `Some(x) as opt`
        #[attr(name: Symbol)]
        fn as_pat() {
            #[region(inner)] {}
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

// === Block Argument Attributes ===

/// Attribute keys for block arguments in pattern matching contexts.
///
/// When pattern matching extracts bindings, the bound variables become
/// block arguments. These attributes store metadata about the bindings.
///
/// # Example
///
/// ```
/// # use trunk_ir::{Attribute, BlockArg, Symbol};
/// # use trunk_ir::dialect::core;
/// # use trunk_ir::DialectType;
/// use tribute_ir::dialect::tribute_pat::block_arg_attrs;
///
/// # let db = salsa::DatabaseImpl::default();
/// # let i32_ty = core::I32::new(&db).as_type();
/// // Pattern: Some(value) => ...
/// // The 'value' binding becomes a block argument with name attribute
/// let block_arg = BlockArg::with_attr(
///     &db,
///     i32_ty,
///     block_arg_attrs::BIND_NAME(),
///     Attribute::Symbol(Symbol::new("value")),
/// );
/// # let name = block_arg.get_attr(&db, block_arg_attrs::BIND_NAME());
/// # assert!(matches!(name, Some(Attribute::Symbol(_))));
/// ```
pub mod block_arg_attrs {
    use trunk_ir::Symbol;

    /// The source-level name of the bound variable.
    ///
    /// Used for:
    /// - Error messages: "expected i32, got String for 'value'"
    /// - LSP hover/go-to-definition
    /// - Debug output
    #[allow(non_snake_case)]
    #[inline]
    pub fn BIND_NAME() -> Symbol {
        Symbol::new("bind_name")
    }

    /// The source location of the binding.
    ///
    /// Used for:
    /// - LSP hover: show type at binding site
    /// - Go-to-definition: navigate to binding
    /// - Diagnostics: precise error locations
    #[allow(non_snake_case)]
    #[inline]
    pub fn BIND_LOCATION() -> Symbol {
        Symbol::new("bind_location")
    }
}

// === Handler Suspend Attributes ===

/// Attribute keys for `tribute_pat.handler_suspend` operations.
///
/// These attributes store type information computed during type checking
/// that is needed for lowering handler patterns.
pub mod handler_suspend_attrs {
    use trunk_ir::Symbol;

    /// The continuation type for this handler suspend arm.
    ///
    /// Set by tirgen as a type variable, then constrained by typeck to the
    /// concrete `cont.continuation` type. After type substitution, this
    /// contains the fully resolved continuation type.
    ///
    /// Used by:
    /// - tribute_to_cont: to create typed `cont.get_continuation` operations
    /// - wasm_type_concrete: to infer call result types
    #[allow(non_snake_case)]
    #[inline]
    pub fn CONTINUATION_TYPE() -> Symbol {
        Symbol::new("continuation_type")
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
    use trunk_ir::{Block, BlockId, DialectOp, IdVec, Operation, Region, Symbol, Type};

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

    /// Create a variant pattern region with nested field patterns.
    pub fn variant_region<'db>(
        db: &'db dyn salsa::Database,
        location: Location<'db>,
        variant_path: Symbol,
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
