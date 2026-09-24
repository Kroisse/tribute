//! Lower target-independent I/O operations to the native runtime ABI.

use trunk_ir::Symbol;
use trunk_ir::context::{BlockData, IrContext, RegionData};
use trunk_ir::dialect::{adt, arith, core, func, mem, scf};
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{OpRef, RegionRef, TypeRef, ValueRef};
use trunk_ir::rewrite::{
    ConversionError, ConversionTarget, Module, PatternApplicator, PatternRewriter, RewritePattern,
    TypeConverter,
};
use trunk_ir::smallvec::smallvec;
use trunk_ir::types::{Attribute, Location, TypeDataBuilder};

use tribute_ir::dialect::tribute_io;

const WRITE_FN: &str = "__tribute_io_write";
const READ_LINE_FN: &str = "__tribute_io_read_line";
const DEALLOC_RESULT_FN: &str = "__tribute_io_read_line_result_dealloc";
const READ_LINE_RESULT_TYPE: &str = "std::io::ReadLineResult";

const TAG_LINE: i128 = 0;
const TAG_END_OF_FILE: i128 = 1;
const TAG_INVALID_ENCODING: i128 = 2;

const TAG_OFFSET: u32 = 0;
const CODE_OFFSET: u32 = 4;
const BYTES_OFFSET: u32 = 8;
const MESSAGE_OFFSET: u32 = 16;

/// Lower every `tribute_io` operation and add the required runtime declarations.
pub fn lower(ctx: &mut IrContext, module: Module) -> Result<(), ConversionError> {
    ensure_runtime_declarations(ctx, module);
    let read_line_result_ty = find_read_line_result_type(ctx);

    PatternApplicator::new(TypeConverter::new())
        .add_pattern(NativeWritePattern)
        .add_pattern(NativeReadLinePattern {
            read_line_result_ty,
        })
        .with_target(ConversionTarget::new().illegal_dialect("tribute_io"))
        .apply_partial_conversion(ctx, module, "io-to-native")?;
    Ok(())
}

fn find_read_line_result_type(ctx: &IrContext) -> Option<TypeRef> {
    let adt = Symbol::new("adt");
    let enum_name = Symbol::new("enum");
    let name_attr = Symbol::new("name");
    let expected = Symbol::new(READ_LINE_RESULT_TYPE);
    ctx.types().iter().find_map(|(ty, data)| {
        (data.dialect == adt
            && data.name == enum_name
            && data.attrs.get_symbol(name_attr) == Some(expected))
        .then_some(ty)
    })
}

fn ensure_runtime_declarations(ctx: &mut IrContext, module: Module) {
    let Some(block) = module.first_block(ctx) else {
        return;
    };
    let loc = ctx.op(module.op()).location;
    let bytes_ty = core::bytes(ctx).as_type_ref();
    let bool_ty = intern_type(ctx, "core", "i1");
    let ptr_ty = core::ptr(ctx).as_type_ref();
    let nil_ty = core::nil(ctx).as_type_ref();

    for (name, params, result) in [
        (WRITE_FN, vec![bytes_ty, bool_ty], nil_ty),
        (READ_LINE_FN, vec![], ptr_ty),
        (DEALLOC_RESULT_FN, vec![ptr_ty], nil_ty),
    ] {
        if has_function(ctx, block, name) {
            continue;
        }
        let declaration = super::build_extern_func(ctx, loc, name, &params, result);
        let first = ctx.block(block).ops.first().copied();
        if let Some(first) = first {
            ctx.insert_op_before(block, first, declaration);
        } else {
            ctx.push_op(block, declaration);
        }
    }
}

fn has_function(ctx: &IrContext, block: trunk_ir::BlockRef, name: &str) -> bool {
    ctx.block(block)
        .ops
        .iter()
        .copied()
        .any(|op| func::Func::from_op(ctx, op).is_ok_and(|function| function.sym_name(ctx) == name))
}

fn intern_type(ctx: &mut IrContext, dialect: &'static str, name: &'static str) -> TypeRef {
    ctx.intern_type(TypeDataBuilder::new(dialect, name).build())
}

struct NativeWritePattern;

impl RewritePattern for NativeWritePattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(write) = tribute_io::Write::from_op(ctx, op) else {
            return false;
        };
        let loc = ctx.op(op).location;
        let result_ty = ctx.op_result_types(op)[0];
        let call = func::Call::operands([write.bytes(ctx), write.newline(ctx)])
            .callee(Symbol::new(WRITE_FN))
            .results([result_ty])
            .build(ctx, loc);
        rewriter.replace_op(call.op_ref());
        true
    }
}

struct NativeReadLinePattern {
    read_line_result_ty: Option<TypeRef>,
}

impl RewritePattern for NativeReadLinePattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if tribute_io::ReadLine::from_op(ctx, op).is_err() {
            return false;
        }
        let Some(enum_ty) = self.read_line_result_ty else {
            return false;
        };

        let loc = ctx.op(op).location;
        let result_ty = ctx.op_result_types(op)[0];
        let ptr_ty = core::ptr(ctx).as_type_ref();
        let bytes_ty = core::bytes(ctx).as_type_ref();
        let i32_ty = intern_type(ctx, "core", "i32");
        let nil_ty = core::nil(ctx).as_type_ref();

        let descriptor = func::Call::operands([])
            .callee(Symbol::new(READ_LINE_FN))
            .results([ptr_ty])
            .build(ctx, loc);
        let descriptor_value = descriptor.result(ctx);
        let tag = mem::Load::operands(descriptor_value)
            .offset(TAG_OFFSET)
            .results(i32_ty)
            .build(ctx, loc);
        let code = mem::Load::operands(descriptor_value)
            .offset(CODE_OFFSET)
            .results(i32_ty)
            .build(ctx, loc);
        let bytes = mem::Load::operands(descriptor_value)
            .offset(BYTES_OFFSET)
            .results(bytes_ty)
            .build(ctx, loc);
        let message = mem::Load::operands(descriptor_value)
            .offset(MESSAGE_OFFSET)
            .results(bytes_ty)
            .build(ctx, loc);
        let dealloc = func::Call::operands([descriptor_value])
            .callee(Symbol::new(DEALLOC_RESULT_FN))
            .results([nil_ty])
            .build(ctx, loc);

        let (line_tag, line_cond) = tag_equals(ctx, loc, tag.result(ctx), TAG_LINE, i32_ty);
        let (eof_tag, eof_cond) = tag_equals(ctx, loc, tag.result(ctx), TAG_END_OF_FILE, i32_ty);
        let (invalid_tag, invalid_cond) =
            tag_equals(ctx, loc, tag.result(ctx), TAG_INVALID_ENCODING, i32_ty);

        let system_region = variant_region(
            ctx,
            loc,
            enum_ty,
            result_ty,
            "ReadSystem",
            [code.result(ctx), message.result(ctx)],
        );
        let invalid_region =
            variant_region(ctx, loc, enum_ty, result_ty, "ReadInvalidEncoding", []);
        let invalid_if = scf::If::operands(invalid_cond.result(ctx))
            .results(result_ty)
            .regions(invalid_region, system_region)
            .build(ctx, loc);

        let eof_region = variant_region(ctx, loc, enum_ty, result_ty, "ReadEndOfFile", []);
        let eof_else = op_region(ctx, loc, invalid_if.op_ref(), invalid_if.result(ctx));
        let eof_if = scf::If::operands(eof_cond.result(ctx))
            .results(result_ty)
            .regions(eof_region, eof_else)
            .build(ctx, loc);

        let line_region = variant_region(
            ctx,
            loc,
            enum_ty,
            result_ty,
            "ReadLine",
            [bytes.result(ctx)],
        );
        let line_else = op_region(ctx, loc, eof_if.op_ref(), eof_if.result(ctx));
        let line_if = scf::If::operands(line_cond.result(ctx))
            .results(result_ty)
            .regions(line_region, line_else)
            .build(ctx, loc);

        for inserted in [
            descriptor.op_ref(),
            tag.op_ref(),
            code.op_ref(),
            bytes.op_ref(),
            message.op_ref(),
            dealloc.op_ref(),
            line_tag.op_ref(),
            line_cond.op_ref(),
            eof_tag.op_ref(),
            eof_cond.op_ref(),
            invalid_tag.op_ref(),
            invalid_cond.op_ref(),
        ] {
            rewriter.insert_op(inserted);
        }
        rewriter.replace_op(line_if.op_ref());
        true
    }
}

fn tag_equals(
    ctx: &mut IrContext,
    loc: Location,
    tag: ValueRef,
    expected: i128,
    i32_ty: TypeRef,
) -> (arith::Const, arith::Cmpi) {
    let constant = arith::Const::builder()
        .value(Attribute::Int(expected))
        .results(i32_ty)
        .build(ctx, loc);
    let comparison = arith::Cmpi::operands(tag, constant.result(ctx))
        .predicate(Symbol::new("eq"))
        .build(ctx, loc);
    (constant, comparison)
}

fn variant_region<const N: usize>(
    ctx: &mut IrContext,
    loc: Location,
    enum_ty: TypeRef,
    result_ty: TypeRef,
    tag: &'static str,
    fields: [ValueRef; N],
) -> RegionRef {
    let variant = adt::VariantNew::operands(fields)
        .r#type(enum_ty)
        .tag(Symbol::new(tag))
        .results(result_ty)
        .build(ctx, loc);
    op_region(ctx, loc, variant.op_ref(), variant.result(ctx))
}

fn op_region(ctx: &mut IrContext, loc: Location, op: OpRef, result: ValueRef) -> RegionRef {
    let yield_op = scf::Yield::operands([result]).build(ctx, loc);
    let block = ctx.create_block(BlockData {
        location: loc,
        args: vec![],
        ops: smallvec![],
        parent_region: None,
    });
    ctx.push_op(block, op);
    ctx.push_op(block, yield_op.op_ref());
    ctx.create_region(RegionData {
        location: loc,
        blocks: smallvec![block],
        parent_op: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;

    #[test]
    fn lowers_native_io_and_builds_read_line_variants() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"
            core.module @test {
                !"std::io::ReadLineResult" = adt.enum() {name = @"std::io::ReadLineResult", variants = [[@ReadLine, [core.bytes]], [@ReadEndOfFile, []], [@ReadInvalidEncoding, []], [@ReadSystem, [core.i32, core.bytes]]]}

                func.func @caller(%0: core.bytes, %1: core.i1) -> tribute_rt.anyref {
                ^bb0:
                    %2 = tribute_io.write %0, %1 : core.nil
                    %3 = tribute_io.read_line : tribute_rt.anyref
                    func.return %3
                }
            }
            "#,
        );

        lower(&mut ctx, module).expect("native I/O lowering");

        let output = print_module(&ctx, module.op());
        assert!(!output.contains("tribute_io."), "{output}");
        assert!(output.contains("callee = @__tribute_io_write"), "{output}");
        assert!(
            output.contains("callee = @__tribute_io_read_line"),
            "{output}"
        );
        assert!(output.contains("tag = @ReadLine"), "{output}");
        assert!(output.contains("tag = @ReadEndOfFile"), "{output}");
        assert!(output.contains("tag = @ReadInvalidEncoding"), "{output}");
        assert!(output.contains("tag = @ReadSystem"), "{output}");

        let validation = trunk_ir::validation::validate_value_integrity(&ctx, module);
        assert!(validation.is_ok(), "{:?}", validation.errors);
    }
}
