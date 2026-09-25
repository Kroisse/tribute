//! Target-independent effect ABI dialect.
//!
//! The `effect` dialect sits between high-level `ability` operations and
//! backend-specific evidence/callable representations. It preserves effect
//! dispatch semantics without exposing Marker fields, handler-table storage, or
//! closure function/environment layout to shared lowering passes.

#[trunk_ir::dialect]
mod effect {
    /// Allocate the runtime-unique prompt token for one dynamic handler
    /// installation. `resolve_evidence` lowers this before target lowering.
    fn fresh_prompt_tag() -> Value<_> {}

    /// Extend the current evidence with a handler for one ability.
    ///
    /// Operands are semantic ABI values:
    /// - `evidence`: current evidence value.
    /// - `prompt_tag`: runtime tag associated with the handler installation.
    /// - `tr_dispatch_fn`: tail-resumptive dispatch closure, or null.
    /// - `handler_dispatch`: full CPS dispatch closure, or null.
    fn extend(
        ability_ref: Attr<Type>,
        evidence: Value<_>,
        prompt_tag: Value<_>,
        tr_dispatch_fn: Value<_>,
        handler_dispatch: Value<_>,
    ) -> Value<_> {
    }

    /// Dispatch a tail-resumptive `fn` ability operation.
    ///
    /// The operation carries ability identity and operation name as attributes,
    /// while the backend chooses the concrete lookup and callable layout.
    fn dispatch_tail(
        ability_ref: Attr<Type>,
        op_name: Attr<Symbol>,
        evidence: Value<_>,
        payload: Value<_>,
    ) -> Value<_> {
    }

    /// Dispatch a general CPS `op` ability operation.
    ///
    /// `dispatch` and `resume` retain their exact result-indexed CPS closure
    /// types. `payload` is the single packed operation argument value. The
    /// operation is resultless: backend lowering performs the final proper tail
    /// transfer.
    fn dispatch_cps(
        ability_ref: Attr<Type>,
        op_name: Attr<Symbol>,
        answer_type: Attr<Type>,
        evidence: Value<_>,
        dispatch: Value<_>,
        resume: Value<_>,
        payload: Value<_>,
    ) {
    }
}

inventory::submit! { trunk_ir::op_interface::PureOps::register::<Extend>() }

impl trunk_ir::op_interface::CallableExitModel for DispatchCps {
    fn verify_callable_exit(
        &self,
        ctx: &trunk_ir::IrContext,
    ) -> Result<(), trunk_ir::op_interface::ControlFlowInterfaceError> {
        let data = ctx.op(self.op_ref());
        if ctx.op_operands(self.op_ref()).len() == 4
            && data.attributes.get_type("ability_ref").is_some()
            && data.attributes.get_symbol("op_name").is_some()
            && data.attributes.get_type("answer_type").is_some()
        {
            Ok(())
        } else {
            Err(trunk_ir::op_interface::ControlFlowInterfaceError::new(
                "effect.dispatch_cps CallableExit has an invalid final CPS shape",
            ))
        }
    }
}

inventory::submit! { trunk_ir::op_interface::CallableExitOps::register::<DispatchCps>() }

#[cfg(test)]
mod tests {
    use trunk_ir::ops::DialectOp;
    use trunk_ir::printer::print_op;
    use trunk_ir::refs::PathRef;
    use trunk_ir::types::{Attribute, Location, TypeDataBuilder};
    use trunk_ir::{IrContext, Span, Symbol};

    fn dummy_location() -> Location {
        Location::new(PathRef::from_u32(0), Span::default())
    }

    fn type_ref(ctx: &mut IrContext, dialect: &str, name: &str) -> trunk_ir::TypeRef {
        ctx.intern_type(
            TypeDataBuilder::new(Symbol::from_dynamic(dialect), Symbol::from_dynamic(name)).build(),
        )
    }

    fn ability_ref(ctx: &mut IrContext, name: &str) -> trunk_ir::TypeRef {
        ctx.intern_type(
            TypeDataBuilder::new(Symbol::new("core"), Symbol::new("ability_ref"))
                .attr("name", Attribute::Symbol(Symbol::from_dynamic(name)))
                .build(),
        )
    }

    fn const_i32(
        ctx: &mut IrContext,
        loc: Location,
        ty: trunk_ir::TypeRef,
        value: i128,
    ) -> trunk_ir::ValueRef {
        trunk_ir::dialect::arith::Const::operands()
            .value(Attribute::Int(value))
            .results(ty)
            .build(ctx, loc)
            .result(ctx)
    }

    #[test]
    fn extend_round_trips_through_typed_wrapper() {
        let mut ctx = IrContext::new();
        let loc = dummy_location();
        let i32_ty = type_ref(&mut ctx, "core", "i32");
        let ptr_ty = type_ref(&mut ctx, "core", "ptr");
        let evidence_ty = type_ref(&mut ctx, "core", "ptr");
        let ability = ability_ref(&mut ctx, "State");

        let evidence = const_i32(&mut ctx, loc, ptr_ty, 0);
        let prompt_tag = const_i32(&mut ctx, loc, i32_ty, 7);
        let tr_dispatch_fn = const_i32(&mut ctx, loc, ptr_ty, 0);
        let handler_dispatch = const_i32(&mut ctx, loc, ptr_ty, 1);

        let op = super::Extend::operands(evidence, prompt_tag, tr_dispatch_fn, handler_dispatch)
            .ability_ref(ability)
            .results(evidence_ty)
            .build(&mut ctx, loc);
        let wrapper = super::Extend::from_op(&ctx, op.op_ref()).expect("effect.extend matches");

        assert_eq!(wrapper.evidence(&ctx), evidence);
        assert_eq!(wrapper.prompt_tag(&ctx), prompt_tag);
        assert_eq!(wrapper.tr_dispatch_fn(&ctx), tr_dispatch_fn);
        assert_eq!(wrapper.handler_dispatch(&ctx), handler_dispatch);
        assert_eq!(wrapper.ability_ref(&ctx), ability);
        assert_eq!(ctx.value_ty(wrapper.result(&ctx)), evidence_ty);
    }

    #[test]
    fn dispatch_ops_round_trip_and_print_generically() {
        let mut ctx = IrContext::new();
        let loc = dummy_location();
        let ptr_ty = type_ref(&mut ctx, "core", "ptr");
        let anyref_ty = type_ref(&mut ctx, "tribute_rt", "anyref");
        let ability = ability_ref(&mut ctx, "Console");
        let evidence = const_i32(&mut ctx, loc, ptr_ty, 0);
        let payload = const_i32(&mut ctx, loc, anyref_ty, 1);
        let dispatch = const_i32(&mut ctx, loc, anyref_ty, 2);
        let resume = const_i32(&mut ctx, loc, anyref_ty, 3);

        let tail = super::DispatchTail::operands(evidence, payload)
            .ability_ref(ability)
            .op_name(Symbol::new("print"))
            .results(anyref_ty)
            .build(&mut ctx, loc);
        let cps = super::DispatchCps::operands(evidence, dispatch, resume, payload)
            .ability_ref(ability)
            .op_name(Symbol::new("get"))
            .answer_type(anyref_ty)
            .build(&mut ctx, loc);

        let tail_wrapper =
            super::DispatchTail::from_op(&ctx, tail.op_ref()).expect("effect.dispatch_tail");
        let cps_wrapper =
            super::DispatchCps::from_op(&ctx, cps.op_ref()).expect("effect.dispatch_cps");

        assert_eq!(tail_wrapper.evidence(&ctx), evidence);
        assert_eq!(tail_wrapper.payload(&ctx), payload);
        assert_eq!(tail_wrapper.ability_ref(&ctx), ability);
        assert_eq!(tail_wrapper.op_name(&ctx), Symbol::new("print"));
        assert_eq!(cps_wrapper.dispatch(&ctx), dispatch);
        assert_eq!(cps_wrapper.resume(&ctx), resume);
        assert_eq!(cps_wrapper.op_name(&ctx), Symbol::new("get"));

        let tail_printed = print_op(&ctx, tail.op_ref());
        assert!(tail_printed.contains("effect.dispatch_tail"));
        assert!(tail_printed.contains("ability_ref"));
        assert!(tail_printed.contains("op_name = @print"));
    }
}
