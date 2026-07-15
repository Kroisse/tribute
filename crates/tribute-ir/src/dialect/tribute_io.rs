//! Target-independent standard I/O boundary.
//!
//! These operations carry flattened bytes and target-neutral result values.
//! Native and Wasm pipelines lower them to their respective host interfaces.

#[trunk_ir::dialect]
mod tribute_io {
    fn write(bytes: (), newline: ()) -> result {}
    fn read_line() -> result {}
}

#[cfg(test)]
mod tests {
    use trunk_ir::ops::DialectOp;
    use trunk_ir::refs::PathRef;
    use trunk_ir::types::Location;
    use trunk_ir::{Attribute, IrContext, Span, Symbol, TypeDataBuilder};

    fn location() -> Location {
        Location::new(PathRef::from_u32(0), Span::default())
    }

    #[test]
    fn io_ops_round_trip() {
        let mut ctx = IrContext::new();
        let loc = location();
        let ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("ptr")).build());
        let bool_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i1")).build());
        let bytes =
            trunk_ir::dialect::arith::r#const(&mut ctx, loc, ty, Attribute::Int(0)).result(&ctx);
        let newline = trunk_ir::dialect::arith::r#const(&mut ctx, loc, bool_ty, Attribute::Int(1))
            .result(&ctx);

        let write = super::write(&mut ctx, loc, bytes, newline, ty);
        let parsed = super::Write::from_op(&ctx, write.op_ref()).expect("tribute_io.write");
        assert_eq!(parsed.bytes(&ctx), bytes);
        assert_eq!(parsed.newline(&ctx), newline);

        let read = super::read_line(&mut ctx, loc, ty);
        assert!(super::ReadLine::from_op(&ctx, read.op_ref()).is_ok());
        assert_eq!(ctx.value_ty(read.result(&ctx)), ty);
    }
}
