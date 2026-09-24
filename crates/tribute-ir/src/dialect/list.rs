//! Target-independent operations for the opaque persistent `List` sequence.

#[trunk_ir::dialect]
mod list {
    fn empty(element_type: Attr<Type>) -> Value<_> {}

    fn prepend(element_type: Attr<Type>, element: Value<_>, tail: Value<_>) -> Value<_> {}

    fn is_empty(element_type: Attr<Type>, list: Value<_>) -> Value<_> {}

    fn head(element_type: Attr<Type>, list: Value<_>) -> Value<_> {}

    fn tail(element_type: Attr<Type>, list: Value<_>) -> Value<_> {}
}

inventory::submit! { trunk_ir::op_interface::PureOps::register("list", "empty") }
inventory::submit! { trunk_ir::op_interface::PureOps::register("list", "prepend") }
inventory::submit! { trunk_ir::op_interface::PureOps::register("list", "is_empty") }
inventory::submit! { trunk_ir::op_interface::PureOps::register("list", "head") }
inventory::submit! { trunk_ir::op_interface::PureOps::register("list", "tail") }

#[cfg(test)]
mod tests {
    use trunk_ir::ops::DialectOp;
    use trunk_ir::refs::PathRef;
    use trunk_ir::types::{Attribute, Location, TypeDataBuilder};
    use trunk_ir::{IrContext, Span};

    fn location() -> Location {
        Location::new(PathRef::from_u32(0), Span::default())
    }

    #[test]
    fn sequence_ops_round_trip() {
        let mut ctx = IrContext::new();
        let loc = location();
        let element_ty = ctx.intern_type(TypeDataBuilder::new("core", "i32").build());
        let list_ty = ctx.intern_type(TypeDataBuilder::new("tribute_rt", "anyref").build());
        let bool_ty = ctx.intern_type(TypeDataBuilder::new("core", "i1").build());
        let element = trunk_ir::dialect::arith::Const::operands()
            .value(Attribute::Int(1))
            .results(element_ty)
            .build(&mut ctx, loc)
            .result(&ctx);

        let empty = super::Empty::operands()
            .element_type(element_ty)
            .results(list_ty)
            .build(&mut ctx, loc);
        let empty_value = empty.result(&ctx);
        let prepend = super::Prepend::operands(element, empty_value)
            .element_type(element_ty)
            .results(list_ty)
            .build(&mut ctx, loc);
        let list_value = prepend.result(&ctx);
        let is_empty = super::IsEmpty::operands(list_value)
            .element_type(element_ty)
            .results(bool_ty)
            .build(&mut ctx, loc);
        let head = super::Head::operands(list_value)
            .element_type(element_ty)
            .results(element_ty)
            .build(&mut ctx, loc);
        let tail = super::Tail::operands(list_value)
            .element_type(element_ty)
            .results(list_ty)
            .build(&mut ctx, loc);

        assert!(super::Empty::from_op(&ctx, empty.op_ref()).is_ok());
        assert!(super::Prepend::from_op(&ctx, prepend.op_ref()).is_ok());
        assert!(super::IsEmpty::from_op(&ctx, is_empty.op_ref()).is_ok());
        assert!(super::Head::from_op(&ctx, head.op_ref()).is_ok());
        assert!(super::Tail::from_op(&ctx, tail.op_ref()).is_ok());
    }
}
