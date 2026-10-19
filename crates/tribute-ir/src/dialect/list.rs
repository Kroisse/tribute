//! Target-independent operations for the opaque persistent `List` sequence.

use trunk_ir::attr_kind::Type;
use trunk_ir::op_flag::Pure;

#[trunk_ir::dialect]
mod list {
    #[flags(Pure)]
    fn empty(element_type: Attr<Type>) -> Value<_> {}

    #[flags(Pure)]
    fn prepend<L>(element_type: Attr<Type>, element: Value<_>, tail: Value<L>) -> Value<L> {}

    #[flags(Pure)]
    fn is_empty(element_type: Attr<Type>, list: Value<_>) -> Value<_> {}

    #[flags(Pure)]
    fn head<T>(element_type: Attr<TypeOf<T>>, list: Value<_>) -> Value<T> {}

    #[flags(Pure)]
    fn tail<L>(element_type: Attr<Type>, list: Value<L>) -> Value<L> {}
}

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
        let element_ty = TypeDataBuilder::new("core.i32").intern(&mut ctx);
        let list_ty = TypeDataBuilder::new("tribute_rt.anyref").intern(&mut ctx);
        let bool_ty = TypeDataBuilder::new("core.i1").intern(&mut ctx);
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
            .build(&mut ctx, loc);
        assert_eq!(prepend.result_ty(&ctx), list_ty);
        let list_value = prepend.result(&ctx);
        let is_empty = super::IsEmpty::operands(list_value)
            .element_type(element_ty)
            .results(bool_ty)
            .build(&mut ctx, loc);
        let head = super::Head::operands(list_value)
            .element_type(element_ty)
            .build(&mut ctx, loc);
        assert_eq!(head.result_ty(&ctx), element_ty);
        let tail = super::Tail::operands(list_value)
            .element_type(element_ty)
            .build(&mut ctx, loc);
        assert_eq!(tail.result_ty(&ctx), list_ty);

        assert!(super::Empty::from_op(&ctx, empty.op_ref()).is_ok());
        assert!(super::Prepend::from_op(&ctx, prepend.op_ref()).is_ok());
        assert!(super::IsEmpty::from_op(&ctx, is_empty.op_ref()).is_ok());
        assert!(super::Head::from_op(&ctx, head.op_ref()).is_ok());
        assert!(super::Tail::from_op(&ctx, tail.op_ref()).is_ok());
    }
}
