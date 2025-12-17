//! Function dialect operations.
use super::core;
use crate::{DialectType, IdVec, Region, Symbol, Type, dialect, idvec, ir::BlockBuilder};
use tribute_core::Location;

dialect! {
    mod func {
        /// `func.func` operation: defines a function.
        #[attr(sym_name: Symbol, r#type: Type)]
        fn func() {
            #[region(body)] {}
        };

        /// `func.call` operation: calls a function.
        #[attr(callee: SymbolRef)]
        fn call(#[rest] args) -> result;

        /// `func.tail_call` operation: tail call (does not return).
        #[attr(callee: SymbolRef)]
        fn tail_call(#[rest] args);

        /// `func.return` operation: returns values from a function.
        fn r#return(#[rest] operands);

        /// `func.closure_new` operation: creates a closure with captured values.
        #[attr(func_ref: SymbolRef)]
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
        params: IdVec<Type<'db>>,
        result: Type<'db>,
        f: impl FnOnce(&mut BlockBuilder<'db>),
    ) -> Self {
        let mut entry = BlockBuilder::new(db, location).args(params.clone());
        f(&mut entry);
        let region = Region::new(db, location, idvec![entry.build()]);
        func(
            db,
            location,
            Symbol::new(db, name),
            core::Func::new(db, params, result).as_type(),
            region,
        )
    }

    /// Get the function name.
    pub fn name(&self, db: &'db dyn salsa::Database) -> &'db str {
        self.sym_name(db).text(db)
    }

    /// Get the function type.
    pub fn ty(&self, db: &'db dyn salsa::Database) -> Type<'db> {
        self.r#type(db)
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
