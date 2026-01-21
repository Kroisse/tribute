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
//! ```
//! # use trunk_ir::dialect::core;
//! # use trunk_ir::DialectType;
//! use tribute_ir::dialect::closure;
//!
//! # let db = salsa::DatabaseImpl::default();
//! # let func_type = core::I32::new(&db).as_type(); // simplified example
//! // Create a closure type wrapping a function type
//! let closure_ty = closure::Closure::new(&db, func_type);
//!
//! // Check if a type is a closure
//! # let ty = closure_ty.as_type();
//! if let Some(closure) = closure::Closure::from_type(&db, ty) {
//!     let inner_fn = closure.func_type(&db);
//!     # assert_eq!(inner_fn, func_type);
//! }
//! ```

use trunk_ir::dialect;

dialect! {
    mod closure {
        /// `closure.new` operation: creates a closure from a function reference and environment.
        ///
        /// The func_ref points to a lifted lambda function that takes env as first arg.
        /// The env operand is typically an `adt.struct` containing captured values.
        #[attr(func_ref: Symbol)]
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

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use trunk_ir::dialect::{arith, core};
    use trunk_ir::type_interface::print_type;
    use trunk_ir::{Attribute, DialectOp, DialectType, IdVec, Location, PathId, Span, Symbol};

    fn test_location(db: &dyn salsa::Database) -> Location<'_> {
        let path = PathId::new(db, "test.trb".to_owned());
        Location::new(path, Span::new(0, 0))
    }

    #[salsa_test]
    fn test_closure_type(db: &salsa::DatabaseImpl) {
        let int_ty = core::I32::new(db).as_type();
        let func_ty = core::Func::new(db, IdVec::from(vec![int_ty]), int_ty).as_type();
        let closure_ty = Closure::new(db, func_ty);

        assert_eq!(closure_ty.as_type().dialect(db), DIALECT_NAME());
        assert_eq!(closure_ty.as_type().name(db), CLOSURE());
        assert_eq!(closure_ty.func_type(db), func_ty);
    }

    #[salsa_test]
    fn test_closure_type_printable(db: &salsa::DatabaseImpl) {
        let int_ty = core::I32::new(db).as_type();
        let func_ty = core::Func::new(db, IdVec::from(vec![int_ty]), int_ty).as_type();
        let closure_ty = Closure::new(db, func_ty);

        let printed = print_type(db, closure_ty.as_type());
        assert!(
            printed.starts_with("Closure("),
            "Expected 'Closure(...)', got '{}'",
            printed
        );
    }

    #[salsa_test]
    fn test_closure_from_type(db: &salsa::DatabaseImpl) {
        let int_ty = core::I32::new(db).as_type();
        let func_ty = core::Func::new(db, IdVec::from(vec![int_ty]), int_ty).as_type();
        let closure_ty = Closure::new(db, func_ty);

        // Should successfully convert back from Type
        let recovered = Closure::from_type(db, closure_ty.as_type());
        assert!(recovered.is_some());
        assert_eq!(recovered.unwrap().func_type(db), func_ty);

        // Non-closure type should return None
        let non_closure = Closure::from_type(db, int_ty);
        assert!(non_closure.is_none());
    }

    #[salsa::tracked]
    fn closure_new_test(db: &dyn salsa::Database) -> Symbol {
        let location = test_location(db);
        let int_ty = core::I32::new(db).as_type();
        let func_ty = core::Func::new(db, IdVec::from(vec![int_ty]), int_ty).as_type();
        let closure_ty = Closure::new(db, func_ty).as_type();

        let env = arith::r#const(db, location, int_ty, Attribute::IntBits(0)).result(db);
        let op = new(db, location, env, closure_ty, Symbol::new("lambda_0"));

        New::from_operation(db, op.as_operation())
            .unwrap()
            .func_ref(db)
    }

    #[salsa_test]
    fn test_closure_new_operation(db: &salsa::DatabaseImpl) {
        let func_ref = closure_new_test(db);
        assert_eq!(func_ref, Symbol::new("lambda_0"));
    }

    #[salsa::tracked]
    fn closure_func_test(db: &dyn salsa::Database) -> (Symbol, Symbol) {
        let location = test_location(db);
        let int_ty = core::I32::new(db).as_type();
        let func_ty = core::Func::new(db, IdVec::from(vec![int_ty]), int_ty).as_type();
        let closure_ty = Closure::new(db, func_ty).as_type();

        let closure_val =
            arith::r#const(db, location, closure_ty, Attribute::IntBits(0)).result(db);
        let op = func(db, location, closure_val, func_ty);

        let adapted = Func::from_operation(db, op.as_operation()).unwrap();
        (
            adapted.as_operation().dialect(db),
            adapted.as_operation().name(db),
        )
    }

    #[salsa_test]
    fn test_closure_func_operation(db: &salsa::DatabaseImpl) {
        let (dialect, name) = closure_func_test(db);
        assert_eq!(dialect, DIALECT_NAME());
        assert_eq!(name, FUNC());
    }

    #[salsa::tracked]
    fn closure_env_test(db: &dyn salsa::Database) -> (Symbol, Symbol) {
        let location = test_location(db);
        let int_ty = core::I32::new(db).as_type();
        let func_ty = core::Func::new(db, IdVec::from(vec![int_ty]), int_ty).as_type();
        let closure_ty = Closure::new(db, func_ty).as_type();

        let closure_val =
            arith::r#const(db, location, closure_ty, Attribute::IntBits(0)).result(db);
        let op = env(db, location, closure_val, int_ty);

        let adapted = Env::from_operation(db, op.as_operation()).unwrap();
        (
            adapted.as_operation().dialect(db),
            adapted.as_operation().name(db),
        )
    }

    #[salsa_test]
    fn test_closure_env_operation(db: &salsa::DatabaseImpl) {
        let (dialect, name) = closure_env_test(db);
        assert_eq!(dialect, DIALECT_NAME());
        assert_eq!(name, ENV());
    }
}
