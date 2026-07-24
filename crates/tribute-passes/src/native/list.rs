//! Lower representation-independent `list.*` operations to private native nodes.

use tribute_ir::dialect::list;
use trunk_ir::Symbol;
use trunk_ir::context::{BlockData, IrContext, RegionData};
use trunk_ir::dialect::{adt, arith, scf};
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{OpRef, RegionRef, TypeRef, ValueRef};
use trunk_ir::rewrite::{
    ConversionError, ConversionTarget, Module, PatternApplicator, PatternRewriter, RewritePattern,
    TypeConverter,
};
use trunk_ir::smallvec::smallvec;
use trunk_ir::types::{Attribute, Location, TypeDataBuilder};

pub fn lower(ctx: &mut IrContext, module: Module) -> Result<(), ConversionError> {
    PatternApplicator::new(TypeConverter::new())
        .add_pattern(EmptyPattern)
        .add_pattern(PrependPattern)
        .add_pattern(IsEmptyPattern)
        .add_pattern(HeadPattern)
        .add_pattern(TailPattern)
        .with_target(ConversionTarget::new().illegal_dialect("list"))
        .apply_partial_conversion(ctx, module, "list-to-native")?;
    Ok(())
}

fn node_type(ctx: &mut IrContext, element_ty: TypeRef) -> TypeRef {
    let list_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("tribute_rt"), Symbol::new("anyref")).build());
    ctx.types.intern(
        TypeDataBuilder::new(Symbol::new("adt"), Symbol::new("struct"))
            .attr("name", Attribute::Symbol(Symbol::new("__native_list_node")))
            .attr(
                "fields",
                Attribute::List(vec![
                    Attribute::List(vec![
                        Attribute::Symbol(Symbol::new("element")),
                        Attribute::Type(element_ty),
                    ]),
                    Attribute::List(vec![
                        Attribute::Symbol(Symbol::new("tail")),
                        Attribute::Type(list_ty),
                    ]),
                ]),
            )
            .build(),
    )
}

fn single_value_region(
    ctx: &mut IrContext,
    location: Location,
    op: OpRef,
    value: ValueRef,
) -> RegionRef {
    let yield_op = scf::r#yield(ctx, location, [value]);
    let block = ctx.create_block(BlockData {
        location,
        args: vec![],
        ops: smallvec![],
        parent_region: None,
    });
    ctx.push_op(block, op);
    ctx.push_op(block, yield_op.op_ref());
    ctx.create_region(RegionData {
        location,
        blocks: smallvec![block],
        parent_op: None,
    })
}

fn default_value(ctx: &mut IrContext, location: Location, ty: TypeRef) -> OpRef {
    let data = ctx.types.get(ty);
    let value = if (data.dialect == "core" && data.name == "f64")
        || (data.dialect == "tribute_rt" && data.name == "float")
    {
        Attribute::FloatBits(0.0f64.to_bits())
    } else if data.dialect == "core" && data.name == "i1" {
        Attribute::Bool(false)
    } else if data.dialect == "core" && data.name == "nil" {
        Attribute::Unit
    } else if (data.dialect == "core"
        && matches!(
            data.name,
            name if name == Symbol::new("i8")
                || name == Symbol::new("i32")
                || name == Symbol::new("i64")
        ))
        || (data.dialect == "tribute_rt"
            && matches!(
                data.name,
                name if name == Symbol::new("int")
                    || name == Symbol::new("nat")
                    || name == Symbol::new("bool")
            ))
    {
        Attribute::Int(0)
    } else {
        return adt::ref_null(ctx, location, ty, ty).op_ref();
    };
    arith::r#const(ctx, location, ty, value).op_ref()
}

struct EmptyPattern;

impl RewritePattern for EmptyPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(empty) = list::Empty::from_op(ctx, op) else {
            return false;
        };
        let result_ty = ctx.op_result_types(op)[0];
        let node_ty = node_type(ctx, empty.element_type(ctx));
        let null = adt::ref_null(ctx, ctx.op(op).location, result_ty, node_ty);
        rewriter.replace_op(null.op_ref());
        true
    }
}

struct PrependPattern;

impl RewritePattern for PrependPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(prepend) = list::Prepend::from_op(ctx, op) else {
            return false;
        };
        let result_ty = ctx.op_result_types(op)[0];
        let node_ty = node_type(ctx, prepend.element_type(ctx));
        let node = adt::struct_new(
            ctx,
            ctx.op(op).location,
            [prepend.element(ctx), prepend.tail(ctx)],
            result_ty,
            node_ty,
        );
        rewriter.replace_op(node.op_ref());
        true
    }
}

struct IsEmptyPattern;

impl RewritePattern for IsEmptyPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(is_empty) = list::IsEmpty::from_op(ctx, op) else {
            return false;
        };
        let result_ty = ctx.op_result_types(op)[0];
        let check = adt::ref_is_null(ctx, ctx.op(op).location, is_empty.list(ctx), result_ty);
        rewriter.replace_op(check.op_ref());
        true
    }
}

struct HeadPattern;

impl RewritePattern for HeadPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(head) = list::Head::from_op(ctx, op) else {
            return false;
        };
        let location = ctx.op(op).location;
        let result_ty = ctx.op_result_types(op)[0];
        let bool_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i1")).build());
        let node_ty = node_type(ctx, head.element_type(ctx));
        let empty = adt::ref_is_null(ctx, location, head.list(ctx), bool_ty);
        let fallback = default_value(ctx, location, result_ty);
        let fallback_value = ctx.op_results(fallback)[0];
        let then_region = single_value_region(ctx, location, fallback, fallback_value);
        let get = adt::struct_get(ctx, location, head.list(ctx), result_ty, node_ty, 0);
        let else_region = single_value_region(ctx, location, get.op_ref(), get.result(ctx));
        let select = scf::r#if(
            ctx,
            location,
            empty.result(ctx),
            result_ty,
            then_region,
            else_region,
        );
        rewriter.insert_op(empty.op_ref());
        rewriter.replace_op(select.op_ref());
        true
    }
}

struct TailPattern;

impl RewritePattern for TailPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(tail) = list::Tail::from_op(ctx, op) else {
            return false;
        };
        let location = ctx.op(op).location;
        let result_ty = ctx.op_result_types(op)[0];
        let bool_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i1")).build());
        let node_ty = node_type(ctx, tail.element_type(ctx));
        let empty = adt::ref_is_null(ctx, location, tail.list(ctx), bool_ty);
        let fallback = adt::ref_null(ctx, location, result_ty, node_ty);
        let then_region =
            single_value_region(ctx, location, fallback.op_ref(), fallback.result(ctx));
        let get = adt::struct_get(ctx, location, tail.list(ctx), result_ty, node_ty, 1);
        let else_region = single_value_region(ctx, location, get.op_ref(), get.result(ctx));
        let select = scf::r#if(
            ctx,
            location,
            empty.result(ctx),
            result_ty,
            then_region,
            else_region,
        );
        rewriter.insert_op(empty.op_ref());
        rewriter.replace_op(select.op_ref());
        true
    }
}
