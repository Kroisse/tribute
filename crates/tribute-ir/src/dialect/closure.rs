//! Closure dialect operations and types.
//!
//! High-level operations for creating and destructuring closures.
//! Closures combine a function reference with captured environment values.
//! This dialect is lowered to target-specific representations (wasm/clif).
//!
//! ## Types
//!
//! - `closure.closure` - Runtime closure representation wrapping a function type.
//!   This type distinguishes closures from bare function references (`core.func`).
//!
//! ## Usage
//!
//! ```ignore
//! // Create a closure type wrapping a function type
//! let closure_ty = closure::Closure::new(db, func_type);
//!
//! // Check if a type is a closure
//! if let Some(closure) = closure::Closure::from_type(db, ty) {
//!     let inner_fn = closure.func_type(db);
//! }
//! ```

use trunk_ir::dialect;

dialect! {
    mod closure {
        /// `closure.new` operation: creates a closure from a function reference and environment.
        ///
        /// The func_ref points to a lifted lambda function that takes env as first arg.
        /// The env operand is typically an `adt.struct` containing captured values.
        #[attr(func_ref: QualifiedName)]
        fn new(env) -> result;

        /// `closure.func` operation: extracts funcref from closure.
        /// Returns a function reference that can be used with `func.call_indirect`.
        fn func(closure) -> result;

        /// `closure.env` operation: extracts environment struct from closure.
        /// Returns the same env value that was passed to `closure.new`.
        fn env(closure) -> result;

        // === Types ===

        /// `closure.closure` type: runtime closure representation.
        ///
        /// Wraps a function type (`core.func`) to distinguish closures from bare
        /// function references at the type level.
        type closure(func_type);
    }
}

// ============================================================================
// Printable Registration
// ============================================================================

use trunk_ir::type_interface::Printable;

// closure.closure -> "Closure(fn(I64) -> I64)"
inventory::submit! {
    Printable::implement("closure", "closure", |db, ty, f| {
        f.write_str("Closure(")?;
        if let Some(&func_ty) = ty.params(db).first() {
            Printable::print_type(db, func_ty, f)?;
        } else {
            f.write_str("?")?;
        }
        f.write_str(")")
    })
}

// === Pure operation registrations ===
// All closure operations are pure

trunk_ir::register_pure_op!(New<'_>);
trunk_ir::register_pure_op!(Func<'_>);
trunk_ir::register_pure_op!(Env<'_>);
