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
//!
//! ## Pattern Region
//!
//! Patterns are represented as a tree of `pat.*` operations in a region.
//! See the `pat` dialect for pattern operations.
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

use crate::{Location, dialect};

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
        /// The pattern region contains a tree of `pat_*` operations.
        /// The body region contains the arm's expression (ends with `case.yield`).
        /// Guards are represented as nested `case.guard` operations within the body.
        fn arm() {
            #[region(pattern)] {}
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

// === Pattern Region Builders ===

/// Re-export pattern helpers from the `pat` dialect.
pub use super::pat::helpers as pattern;

impl<'db> Arm<'db> {
    /// Create a wildcard arm that matches anything.
    pub fn wildcard(
        db: &'db dyn salsa::Database,
        location: Location<'db>,
        body: crate::Region<'db>,
    ) -> Self {
        let pattern_region = pattern::wildcard_region(db, location);
        arm(db, location, pattern_region, body)
    }

    /// Create an arm that binds the scrutinee to a name.
    pub fn binding(
        db: &'db dyn salsa::Database,
        location: Location<'db>,
        name: &str,
        body: crate::Region<'db>,
    ) -> Self {
        let pattern_region = pattern::bind_region(db, location, name);
        arm(db, location, pattern_region, body)
    }

    /// Create an arm matching a unit variant (e.g., `None`).
    pub fn unit_variant(
        db: &'db dyn salsa::Database,
        location: Location<'db>,
        variant_name: &str,
        body: crate::Region<'db>,
    ) -> Self {
        use crate::{IdVec, Symbol};
        let variant_path = IdVec::from(vec![Symbol::new(variant_name)]);
        let fields = pattern::empty_region(db, location);
        let pattern_region = pattern::variant_region(db, location, variant_path, fields);
        arm(db, location, pattern_region, body)
    }
}
