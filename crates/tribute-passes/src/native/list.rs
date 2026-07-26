//! Lower representation-independent `list.*` operations to private native nodes.

use std::ops::ControlFlow;

use tribute_ir::dialect::list;
use trunk_ir::Symbol;
use trunk_ir::context::{BlockArgData, BlockData, IrContext};
use trunk_ir::dialect::{adt, cf, func};
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{OpRef, TypeRef, ValueRef};
use trunk_ir::rewrite::helpers::split_block;
use trunk_ir::rewrite::{
    ConversionError, ConversionTarget, Module, PatternApplicator, PatternRewriter, RewritePattern,
    TypeConverter,
};
use trunk_ir::smallvec::smallvec;
use trunk_ir::types::{Attribute, TypeDataBuilder};
use trunk_ir::walk::{WalkAction, walk_op};

pub fn lower(ctx: &mut IrContext, module: Module) -> Result<(), ConversionError> {
    lower_observations(ctx, module);
    PatternApplicator::new(TypeConverter::new())
        .add_pattern(EmptyPattern)
        .add_pattern(PrependPattern)
        .add_pattern(IsEmptyPattern)
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

fn lower_observations(ctx: &mut IrContext, module: Module) {
    let mut observations = Vec::new();
    let _ = walk_op::<()>(ctx, module.op(), &mut |op| {
        if let Ok(head) = list::Head::from_op(ctx, op) {
            observations.push((op, head.list(ctx), head.element_type(ctx), 0));
        } else if let Ok(tail) = list::Tail::from_op(ctx, op) {
            observations.push((op, tail.list(ctx), tail.element_type(ctx), 1));
        }
        ControlFlow::Continue(WalkAction::Advance)
    });

    // Lower from last to first so splitting a block never hides a later
    // observation from this pass.
    for (op, list, element_ty, field_index) in observations.into_iter().rev() {
        lower_observation(ctx, op, list, element_ty, field_index);
    }
}

fn lower_observation(
    ctx: &mut IrContext,
    op: OpRef,
    list: ValueRef,
    element_ty: TypeRef,
    field_index: u32,
) {
    let location = ctx.op(op).location;
    let result_ty = ctx.op_result_types(op)[0];
    let block = ctx
        .op(op)
        .parent_block
        .expect("list observation must belong to a block");
    let region = ctx
        .block(block)
        .parent_region
        .expect("list observation block must belong to a region");
    let continuation = split_block(ctx, block, op);
    let result = ctx.add_block_arg(
        continuation,
        BlockArgData {
            ty: result_ty,
            attrs: Default::default(),
        },
    );
    ctx.replace_all_uses(ctx.op_results(op)[0], result);
    ctx.detach_op(op);
    ctx.remove_op(op);

    let trap_block = ctx.create_block(BlockData {
        location,
        args: vec![],
        ops: smallvec![],
        parent_region: None,
    });
    let projection_block = ctx.create_block(BlockData {
        location,
        args: vec![],
        ops: smallvec![],
        parent_region: None,
    });
    let continuation_index = ctx
        .region(region)
        .blocks
        .iter()
        .position(|&candidate| candidate == continuation)
        .expect("continuation must belong to the observation region");
    ctx.region_mut(region)
        .blocks
        .insert(continuation_index, trap_block);
    ctx.region_mut(region)
        .blocks
        .insert(continuation_index + 1, projection_block);
    ctx.block_mut(trap_block).parent_region = Some(region);
    ctx.block_mut(projection_block).parent_region = Some(region);

    let bool_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i1")).build());
    let empty = adt::ref_is_null(ctx, location, list, bool_ty);
    let branch = cf::cond_br(
        ctx,
        location,
        empty.result(ctx),
        trap_block,
        projection_block,
    );
    ctx.push_op(block, empty.op_ref());
    ctx.push_op(block, branch.op_ref());

    let trap = func::unreachable(ctx, location);
    ctx.push_op(trap_block, trap.op_ref());

    let node_ty = node_type(ctx, element_ty);
    let projection = adt::struct_get(ctx, location, list, result_ty, node_ty, field_index);
    let continue_with = cf::br(ctx, location, [projection.result(ctx)], continuation);
    ctx.push_op(projection_block, projection.op_ref());
    ctx.push_op(projection_block, continue_with.op_ref());
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

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;

    #[test]
    fn lowers_all_shared_list_ops_to_private_native_representation() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"
            core.module @test {
                func.func @scalar(%0: core.i32) -> tribute_rt.anyref {
                ^bb0:
                    %1 = list.empty {element_type = core.i32} : tribute_rt.anyref
                    %2 = list.prepend %0, %1 {element_type = core.i32} : tribute_rt.anyref
                    %3 = list.is_empty %2 {element_type = core.i32} : core.i1
                    %4 = list.head %2 {element_type = core.i32} : core.i32
                    %5 = list.tail %2 {element_type = core.i32} : tribute_rt.anyref
                    func.return %5
                }
                func.func @reference(%0: tribute_rt.anyref) -> tribute_rt.anyref {
                ^bb0:
                    %1 = list.empty {element_type = tribute_rt.anyref} : tribute_rt.anyref
                    %2 = list.prepend %0, %1 {element_type = tribute_rt.anyref} : tribute_rt.anyref
                    %3 = list.head %2 {element_type = tribute_rt.anyref} : tribute_rt.anyref
                    %4 = list.tail %2 {element_type = tribute_rt.anyref} : tribute_rt.anyref
                    func.return %4
                }
            }
            "#,
        );

        lower(&mut ctx, module).expect("native List lowering");

        let output = print_module(&ctx, module.op());
        assert!(!output.contains("list."), "{output}");
        assert!(output.contains("adt.struct_new"), "{output}");
        assert!(output.contains("adt.ref_is_null"), "{output}");
        assert!(output.contains("adt.struct_get"), "{output}");
        assert!(output.contains("[@element, core.i32]"), "{output}");
        assert!(output.contains("[@element, tribute_rt.anyref]"), "{output}");
        assert_eq!(output.matches("{field = 0,").count(), 2, "{output}");
        assert_eq!(output.matches("{field = 1,").count(), 2, "{output}");
        assert_eq!(output.matches("func.unreachable").count(), 4, "{output}");
        assert_eq!(output.matches("adt.ref_null").count(), 2, "{output}");
        assert!(!output.contains("arith.const {value = 0}"), "{output}");

        let validation = trunk_ir::validation::validate_all(&ctx, module);
        assert!(validation.is_ok(), "{:?}", validation.errors);
    }

    #[test]
    fn empty_head_and_tail_lower_to_traps_without_fallback_values() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"
            core.module @test {
                func.func @empty_scalar_head() -> core.i32 {
                ^bb0:
                    %0 = list.empty {element_type = core.i32} : tribute_rt.anyref
                    %1 = list.head %0 {element_type = core.i32} : core.i32
                    func.return %1
                }
                func.func @empty_reference_head() -> tribute_rt.anyref {
                ^bb0:
                    %0 = list.empty {element_type = tribute_rt.anyref} : tribute_rt.anyref
                    %1 = list.head %0 {element_type = tribute_rt.anyref} : tribute_rt.anyref
                    func.return %1
                }
                func.func @empty_scalar_tail() -> tribute_rt.anyref {
                ^bb0:
                    %0 = list.empty {element_type = core.i32} : tribute_rt.anyref
                    %1 = list.tail %0 {element_type = core.i32} : tribute_rt.anyref
                    func.return %1
                }
                func.func @empty_reference_tail() -> tribute_rt.anyref {
                ^bb0:
                    %0 = list.empty {element_type = tribute_rt.anyref} : tribute_rt.anyref
                    %1 = list.tail %0 {element_type = tribute_rt.anyref} : tribute_rt.anyref
                    func.return %1
                }
            }
            "#,
        );

        lower(&mut ctx, module).expect("native List lowering");

        let output = print_module(&ctx, module.op());
        assert!(!output.contains("list."), "{output}");
        assert_eq!(output.matches("func.unreachable").count(), 4, "{output}");
        assert_eq!(output.matches("cf.cond_br").count(), 4, "{output}");
        assert_eq!(output.matches("{field = 0,").count(), 2, "{output}");
        assert_eq!(output.matches("{field = 1,").count(), 2, "{output}");
        assert_eq!(output.matches("adt.ref_null").count(), 4, "{output}");
        assert!(!output.contains("arith.const"), "{output}");
        assert!(output.contains("[@element, core.i32]"), "{output}");
        assert!(output.contains("[@element, tribute_rt.anyref]"), "{output}");

        let validation = trunk_ir::validation::validate_all(&ctx, module);
        assert!(validation.is_ok(), "{:?}", validation.errors);
    }
}
