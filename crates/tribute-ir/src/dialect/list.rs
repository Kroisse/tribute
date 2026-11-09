//! Target-independent operations for the opaque persistent `List` sequence.

use trunk_ir::attr_kind::Type;
use trunk_ir::op_flag::Pure;

#[trunk_ir::dialect]
mod list {
    /// The empty `List(a)` whose element type is `element_type`.
    #[flags(Pure)]
    #[asm_format(attr_dict_bare ":" type(result))]
    fn empty(element_type: Attr<Type>) -> Value<_> {}

    /// The list whose head is `element` and whose rest is `tail`. It is
    /// persistent: it returns a new sequence and leaves `tail` unchanged.
    #[flags(Pure)]
    #[asm_format(element "," tail attr_dict_bare ":" type(result))]
    fn prepend<L>(element_type: Attr<Type>, element: Value<_>, tail: Value<L>) -> Value<L> {}

    /// Whether `list` has no elements.
    #[flags(Pure)]
    #[asm_format(list attr_dict_bare ":" type(result))]
    fn is_empty(element_type: Attr<Type>, list: Value<_>) -> Value<_> {}

    /// The first element of `list`, whose type is `element_type`. An
    /// internal observation with the precondition that `list` is not empty:
    /// compiler-generated uses establish that before executing it, and a
    /// backend traps on a violation instead of returning a fallback such as
    /// the type's default value.
    #[flags(Pure)]
    #[asm_format(list attr_dict_bare ":" type(result))]
    fn head<T>(element_type: Attr<TypeOf<T>>, list: Value<_>) -> Value<T> {}

    /// `list` without its first element. An internal observation with the
    /// precondition that `list` is not empty: compiler-generated uses
    /// establish that before executing it, and a backend traps on a
    /// violation instead of returning a fallback such as a null tail.
    #[flags(Pure)]
    #[asm_format(list attr_dict_bare ":" type(result))]
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
        let list_ty = crate::dialect::tribute_rt::Anyref::params()
            .intern(&mut ctx)
            .as_type_ref();
        let bool_ty = TypeDataBuilder::new("core.i1").intern(&mut ctx);
        let element = trunk_ir::dialect::arith::Constant::operands()
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
