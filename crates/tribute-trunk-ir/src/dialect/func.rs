//! Function dialect operations.

use crate::{Attribute, Region, Type, dialect, ir::BlockBuilder};
use tribute_core::Location;

dialect! {
    mod func {
        /// `func.func` operation: defines a function.
        #[attr(sym_name, r#type)]
        fn func() {
            #[region(body)] {}
        };

        /// `func.call` operation: calls a function.
        #[attr(callee)]
        fn call(#[rest] args) -> result;

        /// `func.tail_call` operation: tail call (does not return).
        #[attr(callee)]
        fn tail_call(#[rest] args);

        /// `func.return` operation: returns values from a function.
        fn r#return(#[rest] operands);

        /// `func.closure_new` operation: creates a closure with captured values.
        #[attr(func_ref)]
        fn closure_new(#[rest] captures) -> result;

        /// `func.closure_call` operation: calls a closure.
        fn closure_call(closure, #[rest] args) -> result;

        /// `func.unreachable` operation: marks unreachable code (trap).
        fn unreachable();
    }
}

impl<'db> Func<'db> {
    /// Build a function with a closure that constructs the entry block.
    pub fn build(
        db: &'db dyn salsa::Database,
        location: Location<'db>,
        name: &str,
        params: Vec<Type>,
        results: Vec<Type>,
        f: impl FnOnce(&mut BlockBuilder<'db>),
    ) -> Self {
        let mut entry = BlockBuilder::new(db, location).args(params.clone());
        f(&mut entry);
        let region = Region::new(db, location, vec![entry.build()]);
        func(
            db,
            location,
            Attribute::String(name.to_string()),
            Attribute::Type(Type::Function { params, results }),
            region,
        )
    }

    /// Get the function name.
    pub fn name(&self, db: &'db dyn salsa::Database) -> &str {
        let Attribute::String(name) = self.sym_name(db) else {
            panic!("func.func missing sym_name attribute")
        };
        name
    }

    /// Get the function type.
    pub fn ty(&self, db: &'db dyn salsa::Database) -> &'db Type {
        let Attribute::Type(t) = self.r#type(db) else {
            panic!("func.func missing type attribute")
        };
        t
    }
}

impl<'db> Return<'db> {
    /// Create a new return with no values.
    pub fn empty(db: &'db dyn salsa::Database, location: Location<'db>) -> Self {
        r#return(db, location, vec![])
    }

    /// Create a new return with a single value.
    pub fn value(
        db: &'db dyn salsa::Database,
        location: Location<'db>,
        value: crate::Value<'db>,
    ) -> Self {
        r#return(db, location, vec![value])
    }
}
