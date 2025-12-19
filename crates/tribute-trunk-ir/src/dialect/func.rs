//! Function dialect operations.
use super::core;
use crate::{
    Attribute, DialectOp, DialectType, IdVec, Operation, Region, Symbol, Type, dialect, idvec,
    ir::BlockBuilder,
};
use tribute_core::{Location, Span};

dialect! {
    mod func {
        /// `func.func` operation: defines a function.
        #[attr(sym_name: Symbol, r#type: Type)]
        fn func() {
            #[region(body)] {}
        };

        /// `func.call` operation: direct call to a function symbol.
        #[attr(callee: SymbolRef)]
        fn call(#[rest] args) -> result;

        /// `func.call_indirect` operation: indirect call via function value.
        /// Callee can be from `func.constant` or a closure.
        fn call_indirect(callee, #[rest] args) -> result;

        /// `func.tail_call` operation: tail call (does not return).
        #[attr(callee: SymbolRef)]
        fn tail_call(#[rest] args);

        /// `func.return` operation: returns values from a function.
        fn r#return(#[rest] operands);

        /// `func.constant` operation: creates a function value from symbol.
        /// Used to get a first-class function reference for indirect calls.
        #[attr(func_ref: SymbolRef)]
        fn constant() -> result;

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
        Self::build_with_name_span_and_effect(db, location, name, None, params, result, None, f)
    }

    /// Build a function with an explicit effect type.
    pub fn build_with_effect(
        db: &'db dyn salsa::Database,
        location: Location<'db>,
        name: &str,
        params: IdVec<Type<'db>>,
        result: Type<'db>,
        effect: Option<Type<'db>>,
        f: impl FnOnce(&mut BlockBuilder<'db>),
    ) -> Self {
        Self::build_with_name_span_and_effect(db, location, name, None, params, result, effect, f)
    }

    /// Build a function with an explicit name span for hover support.
    pub fn build_with_name_span(
        db: &'db dyn salsa::Database,
        location: Location<'db>,
        name: &str,
        name_span: Option<Span>,
        params: IdVec<Type<'db>>,
        result: Type<'db>,
        f: impl FnOnce(&mut BlockBuilder<'db>),
    ) -> Self {
        Self::build_with_name_span_and_effect(
            db, location, name, name_span, params, result, None, f,
        )
    }

    /// Build a function with an explicit name span and effect type.
    pub fn build_with_name_span_and_effect(
        db: &'db dyn salsa::Database,
        location: Location<'db>,
        name: &str,
        name_span: Option<Span>,
        params: IdVec<Type<'db>>,
        result: Type<'db>,
        effect: Option<Type<'db>>,
        f: impl FnOnce(&mut BlockBuilder<'db>),
    ) -> Self {
        let mut entry = BlockBuilder::new(db, location).args(params.clone());
        f(&mut entry);
        let region = Region::new(db, location, idvec![entry.build()]);

        let mut builder = Operation::of_name(db, location, "func.func")
            .attr("sym_name", Attribute::Symbol(Symbol::new(db, name)))
            .attr(
                "type",
                Attribute::Type(core::Func::with_effect(db, params, result, effect).as_type()),
            )
            .region(region);

        if let Some(span) = name_span {
            builder = builder.attr("name_span", Attribute::Span(span));
        }

        Func::from_operation(db, builder.build()).expect("valid func.func operation")
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
