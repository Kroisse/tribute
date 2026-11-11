//! Target-independent standard I/O boundary.
//!
//! These operations carry byte arrays and target-neutral result values.
//! Native and Wasm pipelines lower them to their respective host interfaces.

use crate::dialect::adt::Array;
use trunk_ir::dialect::core::{I1, I8, Index};

#[trunk_ir::dialect]
mod tribute_io {
    /// Write the `len` bytes of the byte array `data` from `start` to
    /// standard output, then one newline if `newline` is true. The operation
    /// borrows `data`, has no result, and ends the execution when
    /// `[start, start + len)` is not inside `data`
    /// (`design/io.md` § "Runtime Boundary").
    ///
    /// ```text
    /// tribute_io.write %data, %start, %len, %newline
    /// ```
    #[asm_format(data "," start "," len "," newline attr_dict_bare)]
    fn write<A: Array<Element = I8>>(
        data: Value<A>,
        start: Value<Index>,
        len: Value<Index>,
        newline: Value<I1>,
    ) {
    }
    #[asm_format(attr_dict_bare ":" type(result))]
    fn read_line() -> Value<_> {}
}

#[cfg(test)]
mod tests {
    use trunk_ir::ops::DialectOp;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::refs::PathRef;
    use trunk_ir::types::Location;
    use trunk_ir::validation::validate_op_schemas;
    use trunk_ir::{IrContext, Span};

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
        let read = super::ReadLine::operands().results(ty).build(&mut ctx, loc);
        assert!(super::ReadLine::from_op(&ctx, read.op_ref()).is_ok());
        assert_eq!(ctx.value_ty(read.result(&ctx)), ty);
    }

    #[test]
    fn write_takes_a_byte_array_range_and_has_no_result() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @f(%data: adt.array<core.i8>, %start: core.index, %len: core.index, %newline: core.i1) -> core.nil {
    tribute_io.write %data, %start, %len, %newline
    func.return
  }
}"#,
        );
        let result = validate_op_schemas(&ctx, module.op());
        assert!(result.is_ok(), "{result}");
        let printed = module.op().display(&ctx).to_string();
        let write = printed
            .lines()
            .find(|line| line.contains("tribute_io.write"))
            .expect("the write");
        assert_eq!(write.trim(), "tribute_io.write %0, %1, %2, %3", "{printed}");
    }

    #[test]
    fn write_rejects_a_position_that_is_not_an_index() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @f(%data: adt.array<core.i8>, %start: core.i32, %newline: core.i1) -> core.nil {
    tribute_io.write %data, %start, %start, %newline
    func.return
  }
}"#,
        );
        assert!(!validate_op_schemas(&ctx, module.op()).is_ok());
    }
}
