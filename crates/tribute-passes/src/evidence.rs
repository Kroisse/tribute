//! Evidence utilities for the ability system.
//!
//! Helpers for working with evidence parameters on function types.

use tribute_ir::dialect::ability;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::{core, func};
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::refs::{OpRef, TypeRef, ValueRef};

/// Check if a `core.func` type has evidence as its first parameter.
pub fn has_evidence_first_param(ctx: &IrContext, func_ty: TypeRef) -> bool {
    let Some(function) = core::Func::from_type_ref(ctx, func_ty) else {
        return false;
    };
    function
        .inputs(ctx)
        .first()
        .is_some_and(|&input| ability::is_evidence_type_ref(ctx, input))
}

/// Build a new `core.func` TypeRef with evidence prepended to params.
pub fn build_func_type_with_evidence(
    ctx: &mut IrContext,
    old_func_ty: TypeRef,
    ev_ty: TypeRef,
) -> TypeRef {
    let function = core::Func::from_type_ref(ctx, old_func_ty)
        .expect("build_func_type_with_evidence requires a valid core.func type");
    let results = function.results(ctx).to_vec();
    let old_params = function.inputs(ctx);
    let mut type_attrs = ctx.types.get(old_func_ty).attrs.clone();
    type_attrs.remove(core::NUM_INPUTS_ATTR);
    type_attrs.remove(core::NUM_RESULTS_ATTR);

    let mut new_params = Vec::with_capacity(old_params.len() + 1);
    new_params.push(ev_ty);
    new_params.extend_from_slice(old_params);

    core::func_with_attrs(ctx, new_params, results, type_attrs).as_type_ref()
}

/// Find the evidence value from the enclosing `func.func`'s entry block.
///
/// Walks up the parent chain from the given op to find the containing
/// `func.func`, then returns its first block argument if it is an evidence type.
pub fn find_enclosing_evidence(ctx: &IrContext, op: OpRef) -> Option<ValueRef> {
    let mut current = op;
    loop {
        let block = ctx.op(current).parent_block?;
        let region = ctx.block(block).parent_region?;
        let parent_op = ctx.region(region).parent_op?;
        if let Ok(func_op) = func::Func::from_op(ctx, parent_op) {
            let body = func_op.body(ctx);
            let entry = ctx.region(body).blocks[0];
            let args = ctx.block_args(entry);
            if !args.is_empty() && ability::is_evidence_type_ref(ctx, ctx.value_ty(args[0])) {
                return Some(args[0]);
            }
            return None;
        }
        current = parent_op;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::{Attribute, AttributeMap, Symbol};

    #[test]
    fn evidence_prefix_preserves_result_cardinality_and_type_attributes() {
        let mut ctx = IrContext::new();
        let i32_ty = ctx.types.intern(
            trunk_ir::types::TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build(),
        );
        let nil = core::nil(&mut ctx).as_type_ref();
        let evidence = ability::evidence_adt_type_ref(&mut ctx);
        for results in [vec![], vec![nil], vec![i32_ty]] {
            let attrs =
                AttributeMap::from_iter([(Symbol::new("metadata"), Attribute::Type(i32_ty))]);
            let source =
                core::func_with_attrs(&mut ctx, [i32_ty], results.clone(), attrs).as_type_ref();
            assert!(!has_evidence_first_param(&ctx, source));
            let converted = build_func_type_with_evidence(&mut ctx, source, evidence);
            let function = core::Func::from_type_ref(&ctx, converted).unwrap();
            assert_eq!(function.inputs(&ctx), [evidence, i32_ty]);
            assert_eq!(function.results(&ctx), results);
            assert_eq!(
                ctx.types.get(converted).attrs.get_type("metadata"),
                Some(i32_ty)
            );
            assert!(has_evidence_first_param(&ctx, converted));
        }
    }
}
