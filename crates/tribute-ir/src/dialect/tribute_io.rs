//! Target-independent standard I/O boundary.
//!
//! These operations carry flattened bytes and target-neutral result values.
//! Native and Wasm pipelines lower them to their respective host interfaces.

#[trunk_ir::dialect]
mod tribute_io {
    fn write(bytes: Value<_>, newline: Value<_>) -> Value<_> {}
    fn read_line() -> Value<_> {}
}

#[cfg(test)]
mod tests {
    use trunk_ir::ops::DialectOp;
    use trunk_ir::refs::PathRef;
    use trunk_ir::types::Location;
    use trunk_ir::{Attribute, IrContext, Span, TypeDataBuilder};

    fn location() -> Location {
        Location::new(PathRef::from_u32(0), Span::default())
    }

    #[test]
    fn io_ops_round_trip() {
        let mut ctx = IrContext::new();
        let loc = location();
        let ty = ctx.intern_type(TypeDataBuilder::new("core", "ptr").build());
        let bool_ty = ctx.intern_type(TypeDataBuilder::new("core", "i1").build());
        let bytes = trunk_ir::dialect::arith::Const::operands()
            .value(Attribute::Int(0))
            .results(ty)
            .build(&mut ctx, loc)
            .result(&ctx);
        let newline = trunk_ir::dialect::arith::Const::operands()
            .value(Attribute::Int(1))
            .results(bool_ty)
            .build(&mut ctx, loc)
            .result(&ctx);

        let write = super::Write::operands(bytes, newline)
            .results(ty)
            .build(&mut ctx, loc);
        let parsed = super::Write::from_op(&ctx, write.op_ref()).expect("tribute_io.write");
        assert_eq!(parsed.bytes(&ctx), bytes);
        assert_eq!(parsed.newline(&ctx), newline);

        let read = super::ReadLine::operands().results(ty).build(&mut ctx, loc);
        assert!(super::ReadLine::from_op(&ctx, read.op_ref()).is_ok());
        assert_eq!(ctx.value_ty(read.result(&ctx)), ty);
    }
}
