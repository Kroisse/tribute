//! Target-independent effect ABI dialect.
//!
//! The `effect` dialect sits between high-level `ability` operations and
//! backend-specific evidence/callable representations. It preserves effect
//! dispatch semantics without exposing Marker fields, handler-table storage, or
//! closure function/environment layout to shared lowering passes.

#[trunk_ir::dialect]
mod effect {
    /// Extend the current evidence with a handler for one ability.
    ///
    /// Operands are semantic ABI values:
    /// - `evidence`: current evidence value.
    /// - `prompt_tag`: runtime tag associated with the handler installation.
    /// - `tr_dispatch_fn`: tail-resumptive dispatch closure, or null.
    /// - `handler_dispatch`: full CPS dispatch closure, or null.
    #[attr(ability_ref: Type)]
    fn extend(evidence: (), prompt_tag: (), tr_dispatch_fn: (), handler_dispatch: ()) -> result {}

    /// Dispatch a tail-resumptive `fn` ability operation.
    ///
    /// The operation carries ability identity and operation name as attributes,
    /// while the backend chooses the concrete lookup and callable layout.
    #[attr(ability_ref: Type, op_name: Symbol)]
    fn dispatch_tail(evidence: (), payload: ()) -> result {}

    /// Dispatch a general CPS `op` ability operation.
    ///
    /// `continuation` is the already-constructed continuation closure and
    /// `payload` is the single packed operation argument value.
    #[attr(ability_ref: Type, op_name: Symbol)]
    fn dispatch_cps(evidence: (), continuation: (), payload: ()) -> result {}
}

inventory::submit! { trunk_ir::op_interface::PureOps::register("effect", "extend") }

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
        ctx.types.intern(
            TypeDataBuilder::new(Symbol::from_dynamic(dialect), Symbol::from_dynamic(name)).build(),
        )
    }

    fn ability_ref(ctx: &mut IrContext, name: &str) -> trunk_ir::TypeRef {
        ctx.types.intern(
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
        trunk_ir::dialect::arith::r#const(ctx, loc, ty, Attribute::Int(value)).result(ctx)
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

        let op = super::extend(
            &mut ctx,
            loc,
            evidence,
            prompt_tag,
            tr_dispatch_fn,
            handler_dispatch,
            evidence_ty,
            ability,
        );
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
        let continuation = const_i32(&mut ctx, loc, anyref_ty, 2);

        let tail = super::dispatch_tail(
            &mut ctx,
            loc,
            evidence,
            payload,
            anyref_ty,
            ability,
            Symbol::new("print"),
        );
        let cps = super::dispatch_cps(
            &mut ctx,
            loc,
            evidence,
            continuation,
            payload,
            anyref_ty,
            ability,
            Symbol::new("get"),
        );

        let tail_wrapper =
            super::DispatchTail::from_op(&ctx, tail.op_ref()).expect("effect.dispatch_tail");
        let cps_wrapper =
            super::DispatchCps::from_op(&ctx, cps.op_ref()).expect("effect.dispatch_cps");

        assert_eq!(tail_wrapper.evidence(&ctx), evidence);
        assert_eq!(tail_wrapper.payload(&ctx), payload);
        assert_eq!(tail_wrapper.ability_ref(&ctx), ability);
        assert_eq!(tail_wrapper.op_name(&ctx), Symbol::new("print"));
        assert_eq!(cps_wrapper.continuation(&ctx), continuation);
        assert_eq!(cps_wrapper.op_name(&ctx), Symbol::new("get"));

        let tail_printed = print_op(&ctx, tail.op_ref());
        assert!(tail_printed.contains("effect.dispatch_tail"));
        assert!(tail_printed.contains("ability_ref"));
        assert!(tail_printed.contains("op_name = @print"));
    }
}
