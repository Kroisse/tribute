//! Lower target-independent I/O operations to the native runtime ABI.

use tribute_ir::dialect::adt;
use trunk_ir::context::{BlockData, IrContext, RegionData};
use trunk_ir::dialect::{arith, core, func, mem, scf};
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{OpRef, RegionRef, TypeRef, ValueRef};
use trunk_ir::rewrite::{
    ConversionError, ConversionTarget, Module, OpRewritePattern, PatternApplicator,
    PatternRewriter, TypeConverter,
};
use trunk_ir::smallvec::smallvec;
use trunk_ir::types::{Attribute, Location};

use tribute_ir::dialect::{tribute_io, tribute_rt};
use trunk_ir::SymbolPath;

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
fn lower(ctx: &mut IrContext, module: Module) -> Result<(), ConversionError> {
    ensure_runtime_declarations(ctx, module);
    let read_line_result = find_read_line_result(ctx, module);

    PatternApplicator::new(TypeConverter::new())
        .add_pattern(NativeWritePattern)
        .add_pattern(NativeReadLinePattern { read_line_result })
        .with_target(ConversionTarget::new().illegal_dialect(tribute_io::DIALECT_NAME))
        .apply_partial_conversion(ctx, module, "io-to-native")?;
    Ok(())
}

/// The `std::io::ReadLineResult` layout and the position of each variant.
#[derive(Clone, Copy)]
struct ReadLineResult {
    ty: TypeRef,
    line: u32,
    end_of_file: u32,
    invalid_encoding: u32,
    system: u32,
}

/// The layout `std::io` declares as `ReadLineResult`, found by its
/// declaration's source name, which a conversion's tag on the layout's name
/// leaves alone (`design/tribute-dialects.md` § "Nominal 선언").
fn find_read_line_result(ctx: &IrContext, module: Module) -> Option<ReadLineResult> {
    let block = module.first_block(ctx)?;
    let decl = ctx.block(block).ops.iter().find_map(|&op| {
        adt::EnumDecl::from_op(ctx, op)
            .ok()
            .filter(|decl| decl.name(ctx) == READ_LINE_RESULT_TYPE)
    })?;
    let ty = decl.r#type(ctx);
    let position = |variant| {
        decl.position_of(ctx, variant)
            .unwrap_or_else(|| panic!("std::io declares the `{variant}` of ReadLineResult"))
    };
    Some(ReadLineResult {
        ty,
        line: position("ReadLine"),
        end_of_file: position("ReadEndOfFile"),
        invalid_encoding: position("ReadInvalidEncoding"),
        system: position("ReadSystem"),
    })
}

fn ensure_runtime_declarations(ctx: &mut IrContext, module: Module) {
    let Some(block) = module.first_block(ctx) else {
        return;
    };
    let loc = ctx.op(module.op()).location;
    let bytes_ty = core::Bytes::params().intern(ctx).as_type_ref();
    let bool_ty = core::I1::type_ref(ctx);
    let ptr_ty = core::Ptr::params().intern(ctx).as_type_ref();
    let nil_ty = core::Nil::params().intern(ctx).as_type_ref();

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

struct NativeWritePattern;

impl OpRewritePattern for NativeWritePattern {
    type Op = tribute_io::Write;

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        write: Self::Op,
        rewriter: &mut PatternRewriter<'_>,
    ) -> Option<()> {
        let op = write.op_ref();
        let loc = ctx.op(op).location;
        let result_ty = ctx.op_result_types(op)[0];
        let call = func::Call::operands([write.bytes(ctx), write.newline(ctx)])
            .callee(SymbolPath::new([WRITE_FN]))
            .results([result_ty])
            .build(ctx, loc);
        rewriter.replace_op(call.op_ref());
        Some(())
    }
}

struct NativeReadLinePattern {
    read_line_result: Option<ReadLineResult>,
}

impl OpRewritePattern for NativeReadLinePattern {
    type Op = tribute_io::ReadLine;

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        read_line: Self::Op,
        rewriter: &mut PatternRewriter<'_>,
    ) -> Option<()> {
        let op = read_line.op_ref();
        let variants = self.read_line_result?;
        let enum_ty = variants.ty;

        let loc = ctx.op(op).location;
        let result_ty = ctx.op_result_types(op)[0];
        let ptr_ty = core::Ptr::params().intern(ctx).as_type_ref();
        let bytes_ty = core::Bytes::params().intern(ctx).as_type_ref();
        let i32_ty = core::I32::type_ref(ctx);
        let nil_ty = core::Nil::params().intern(ctx).as_type_ref();

        let descriptor = func::Call::operands([])
            .callee(SymbolPath::new([READ_LINE_FN]))
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
        // The descriptor's Bytes fields are runtime pointers that the
        // descriptor transfers to the caller; load them as pointers and take
        // over the unit of each.
        let bytes_ptr = mem::Load::operands(descriptor_value)
            .offset(BYTES_OFFSET)
            .results(ptr_ty)
            .build(ctx, loc);
        let bytes = tribute_rt::FromRaw::operands(bytes_ptr.result(ctx))
            .results(bytes_ty)
            .build(ctx, loc);
        let message_ptr = mem::Load::operands(descriptor_value)
            .offset(MESSAGE_OFFSET)
            .results(ptr_ty)
            .build(ctx, loc);
        let message = tribute_rt::FromRaw::operands(message_ptr.result(ctx))
            .results(bytes_ty)
            .build(ctx, loc);
        let dealloc = func::Call::operands([descriptor_value])
            .callee(SymbolPath::new([DEALLOC_RESULT_FN]))
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
            variants.system,
            [code.result(ctx), message.result(ctx)],
        );
        let invalid_region =
            variant_region(ctx, loc, enum_ty, result_ty, variants.invalid_encoding, []);
        let invalid_if = scf::If::operands(invalid_cond.result(ctx))
            .results(result_ty)
            .regions(invalid_region, system_region)
            .build(ctx, loc);

        let eof_region = variant_region(ctx, loc, enum_ty, result_ty, variants.end_of_file, []);
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
            variants.line,
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
            bytes_ptr.op_ref(),
            bytes.op_ref(),
            message_ptr.op_ref(),
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
        Some(())
    }
}

fn tag_equals(
    ctx: &mut IrContext,
    loc: Location,
    tag: ValueRef,
    expected: i128,
    i32_ty: TypeRef,
) -> (arith::Constant, arith::Cmpi) {
    let constant = arith::Constant::operands()
        .value(Attribute::Int(expected))
        .results(i32_ty)
        .build(ctx, loc);
    let comparison = arith::Cmpi::operands(tag, constant.result(ctx))
        .predicate(arith::CmpiPredicate::Eq)
        .build(ctx, loc);
    (constant, comparison)
}

fn variant_region<const N: usize>(
    ctx: &mut IrContext,
    loc: Location,
    enum_ty: TypeRef,
    result_ty: TypeRef,
    tag: u32,
    fields: [ValueRef; N],
) -> RegionRef {
    let variant = adt::VariantNew::operands(fields)
        .r#type(enum_ty)
        .tag(tag)
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

/// Lowers target-independent I/O operations to the native runtime ABI.
pub struct IoToNative;

impl trunk_ir::pass::Pass for IoToNative {
    type Target = trunk_ir::dialect::core::Module;

    fn name(&self) -> &'static str {
        "io-to-native"
    }

    fn run(
        &mut self,
        ctx: &mut trunk_ir::context::IrContext,
        target: trunk_ir::dialect::core::Module,
        _analyses: &mut trunk_ir::analysis::AnalysisCache,
    ) -> trunk_ir::pass::PassRunResult {
        lower(ctx, target.into())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::parser::parse_test_module;

    #[test]
    fn lowers_native_io_and_builds_read_line_variants() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"
            core.module @test {
                !"std::io::ReadLineResult" = adt.enum<"std::io::ReadLineResult", [(core.bytes), (), (), (core.i32, core.bytes)]>
                adt.enum_decl "std::io::ReadLineResult" !"std::io::ReadLineResult" ["ReadLine", "ReadEndOfFile", "ReadInvalidEncoding", "ReadSystem"]

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

        let output = module.op().display(&ctx).to_string();
        assert!(!output.contains("tribute_io."), "{output}");
        assert!(
            output.contains("func.call @__tribute_io_write("),
            "{output}"
        );
        assert!(
            output.contains("func.call @__tribute_io_read_line("),
            "{output}"
        );
        for tag in 0..4 {
            assert!(output.contains(&format!("tag = {tag}, ")), "{output}");
        }

        let validation = trunk_ir::validation::validate_value_integrity(&ctx, module);
        assert!(validation.is_ok(), "{:?}", validation.errors);
    }
}
