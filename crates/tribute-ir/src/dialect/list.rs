//! Target-independent operations for the opaque persistent `List` sequence.

#[trunk_ir::dialect]
mod list {
    #[attr(element_type: Type)]
    fn empty() -> result {}

    #[attr(element_type: Type)]
    fn prepend(element: (), tail: ()) -> result {}

    #[attr(element_type: Type)]
    fn is_empty(list: ()) -> result {}

    #[attr(element_type: Type)]
    fn head(list: ()) -> result {}

    #[attr(element_type: Type)]
    fn tail(list: ()) -> result {}
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
    use trunk_ir::{IrContext, Span, Symbol};

    fn location() -> Location {
        Location::new(PathRef::from_u32(0), Span::default())
    }

    #[test]
    fn sequence_ops_round_trip() {
        let mut ctx = IrContext::new();
        let loc = location();
        let element_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());
        let list_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("tribute_rt"), Symbol::new("anyref")).build());
        let bool_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i1")).build());
        let element =
            trunk_ir::dialect::arith::r#const(&mut ctx, loc, element_ty, Attribute::Int(1))
                .result(&ctx);

        let empty = super::empty(&mut ctx, loc, list_ty, element_ty);
        let empty_value = empty.result(&ctx);
        let prepend = super::prepend(&mut ctx, loc, element, empty_value, list_ty, element_ty);
        let list_value = prepend.result(&ctx);
        let is_empty = super::is_empty(&mut ctx, loc, list_value, bool_ty, element_ty);
        let head = super::head(&mut ctx, loc, list_value, element_ty, element_ty);
        let tail = super::tail(&mut ctx, loc, list_value, list_ty, element_ty);

        assert!(super::Empty::from_op(&ctx, empty.op_ref()).is_ok());
        assert!(super::Prepend::from_op(&ctx, prepend.op_ref()).is_ok());
        assert!(super::IsEmpty::from_op(&ctx, is_empty.op_ref()).is_ok());
        assert!(super::Head::from_op(&ctx, head.op_ref()).is_ok());
        assert!(super::Tail::from_op(&ctx, tail.op_ref()).is_ok());
    }
}
