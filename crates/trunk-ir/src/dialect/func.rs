//! Function dialect operations.
use super::core;
use crate::{
    Attribute, DialectOp, DialectType, IdVec, Location, Operation, Region, Span, Symbol, Type,
    dialect, idvec, ir::BlockBuilder,
};

dialect! {
    mod func {
        /// `func.func` operation: defines a function.
        /// sym_name contains the full qualified name as a Symbol (e.g., "module::func").
        #[attr(sym_name: Symbol, r#type: Type)]
        fn func() {
            #[region(body)] {}
        };

        /// `func.call` operation: direct call to a function symbol.
        /// callee contains the full qualified name as a Symbol.
        #[attr(callee: Symbol)]
        fn call(#[rest] args) -> result;

        /// `func.call_indirect` operation: indirect call via function value.
        /// Callee can be from `func.constant` or a closure.
        fn call_indirect(callee, #[rest] args) -> result;

        /// `func.tail_call` operation: tail call (does not return).
        #[attr(callee: Symbol)]
        fn tail_call(#[rest] args);

        /// `func.return` operation: returns values from a function.
        fn r#return(#[rest] operands);

        /// `func.constant` operation: creates a function value from symbol.
        /// Used to get a first-class function reference for indirect calls.
        #[attr(func_ref: Symbol)]
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
        name: impl Into<Symbol>,
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
        name: impl Into<Symbol>,
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
        name: impl Into<Symbol>,
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
    #[allow(clippy::too_many_arguments)]
    pub fn build_with_name_span_and_effect(
        db: &'db dyn salsa::Database,
        location: Location<'db>,
        name: impl Into<Symbol>,
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
            .attr("sym_name", Attribute::Symbol(name.into()))
            .attr(
                "type",
                Attribute::Type(core::Func::with_effect(db, params, result, effect).as_type()),
            )
            .region(region);

        if let Some(span) = name_span {
            let name_loc = crate::Location::new(location.path, span);
            builder = builder.attr("name_location", Attribute::Location(name_loc));
        }

        Func::from_operation(db, builder.build()).expect("valid func.func operation")
    }

    /// Build a function with named parameters.
    ///
    /// Each parameter can optionally have a name that will be attached to the
    /// block argument as a `bind_name` attribute. This is useful for debugging
    /// and error messages.
    #[allow(clippy::too_many_arguments)]
    pub fn build_with_named_params(
        db: &'db dyn salsa::Database,
        location: Location<'db>,
        name: impl Into<Symbol>,
        name_span: Option<Span>,
        params: impl IntoIterator<Item = (Type<'db>, Option<Symbol>)>,
        result: Type<'db>,
        effect: Option<Type<'db>>,
        f: impl FnOnce(&mut BlockBuilder<'db>),
    ) -> Self {
        let mut entry = BlockBuilder::new(db, location);
        let mut param_types = IdVec::new();

        for (ty, param_name) in params {
            param_types.push(ty);
            entry = entry.arg(ty);
            if let Some(name) = param_name {
                entry = entry.attr(Symbol::new("bind_name"), name);
            }
        }

        f(&mut entry);
        let region = Region::new(db, location, idvec![entry.build()]);

        let mut builder = Operation::of_name(db, location, "func.func")
            .attr("sym_name", Attribute::Symbol(name.into()))
            .attr(
                "type",
                Attribute::Type(core::Func::with_effect(db, param_types, result, effect).as_type()),
            )
            .region(region);

        if let Some(span) = name_span {
            let name_loc = crate::Location::new(location.path, span);
            builder = builder.attr("name_location", Attribute::Location(name_loc));
        }

        Func::from_operation(db, builder.build()).expect("valid func.func operation")
    }

    /// Get the function's simple name.
    pub fn name(&self, db: &'db dyn salsa::Database) -> Symbol {
        self.sym_name(db)
    }

    /// Get the function type.
    pub fn ty(&self, db: &'db dyn salsa::Database) -> Type<'db> {
        self.r#type(db)
    }
}

impl std::fmt::Debug for Func<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        salsa::with_attached_database(|db| {
            let name = self.name(db);
            let ty = self.ty(db);

            // Build signature string
            let signature = if let Some(func_ty) = core::Func::from_type(db, ty) {
                let params = func_ty.params(db);
                let result = func_ty.result(db);

                let params_str = params
                    .iter()
                    .map(|p| format!("{}.{}", p.dialect(db), p.name(db)))
                    .collect::<Vec<_>>()
                    .join(", ");

                format!(
                    "@{}({}) -> {}.{}",
                    name,
                    params_str,
                    result.dialect(db),
                    result.name(db)
                )
            } else {
                format!("@{} : {:?}", name, ty)
            };

            // Collect body operations
            let body = self.body(db);
            let ops: Vec<_> = body
                .blocks(db)
                .first()
                .map(|block| {
                    block
                        .operations(db)
                        .iter()
                        .map(|op| format!("{}.{}", op.dialect(db), op.name(db)))
                        .collect()
                })
                .unwrap_or_default();

            f.debug_struct(&format!("func {}", signature))
                .field("body", &ops)
                .finish()
        })
        .unwrap_or_else(|| write!(f, "func @<no database attached>"))
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

// === Pure operation registrations ===
// Only func.constant is pure (it just creates a reference)

crate::register_pure_op!(func.constant);
