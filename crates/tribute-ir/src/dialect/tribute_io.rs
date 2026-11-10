//! Target-independent standard I/O boundary.
//!
//! These operations carry byte arrays and target-neutral result values.
//! Native and Wasm pipelines lower them to their respective host interfaces.

#[trunk_ir::dialect]
mod tribute_io {
    /// Write the `len` bytes of the byte array `data` from `start` to
    /// standard output, then one newline if `newline` is true. The operation
    /// borrows `data`; a range outside it ends the execution
    /// (`design/io.md` § "Runtime Boundary").
    ///
    /// ```text
    /// %r = tribute_io.write %data, %start, %len, %newline : core.nil
    /// ```
    #[asm_format(data "," start "," len "," newline attr_dict_bare ":" type(result))]
    fn write(data: Value<_>, start: Value<_>, len: Value<_>, newline: Value<_>) -> Value<_> {}
    #[asm_format(attr_dict_bare ":" type(result))]
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
        let ty = trunk_ir::dialect::core::Ptr::params()
            .intern(&mut ctx)
            .as_type_ref();
        let bool_ty = TypeDataBuilder::new("core.i1").intern(&mut ctx);
        let bytes = trunk_ir::dialect::arith::Constant::operands()
            .value(Attribute::Int(0))
            .results(ty)
            .build(&mut ctx, loc)
            .result(&ctx);
        let newline = trunk_ir::dialect::arith::Constant::operands()
            .value(Attribute::Int(1))
            .results(bool_ty)
            .build(&mut ctx, loc)
            .result(&ctx);

        let write = super::Write::operands(bytes, newline, newline, newline)
            .results(ty)
            .build(&mut ctx, loc);
        let parsed = super::Write::from_op(&ctx, write.op_ref()).expect("tribute_io.write");
        assert_eq!(parsed.data(&ctx), bytes);
        assert_eq!(parsed.newline(&ctx), newline);

        let read = super::ReadLine::operands().results(ty).build(&mut ctx, loc);
        assert!(super::ReadLine::from_op(&ctx, read.op_ref()).is_ok());
        assert_eq!(ctx.value_ty(read.result(&ctx)), ty);
    }
}
