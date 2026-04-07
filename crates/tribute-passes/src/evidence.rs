//! Evidence insertion pass for ability system.
//!
//! This pass transforms effectful functions to receive an evidence pointer parameter
//! and transforms ability operations to evidence lookups.
//!
//! ## Transformations
//!
//! 1. **Effectful function signatures**: Add evidence parameter as first argument
//!    ```text
//!    // Before
//!    fn foo(x: Int) ->{State(Int)} Int
//!
//!    // After
//!    fn foo(ev: Evidence, x: Int) -> Int
//!    ```
//!
//! 2. **Call sites**: Pass evidence through call chains
//!    ```text
//!    // Before
//!    func.call @effectful_fn(%arg)
//!
//!    // After
//!    func.call @effectful_fn(%ev, %arg)
//!    ```
//!
//! 3. **Ability operations**: Transform to evidence lookups (placeholder for now)
//!    ```text
//!    // Before
//!    %result = ability.perform { ability_ref: State, op: "get" }
//!
//!    // After (conceptual - actual lowering in handler_lower pass)
//!    %marker = evidence.lookup %ev, STATE_ID
//!    %result = evidence.dispatch %marker, ...
//!    ```
//!
//! ## Design
//!
//! Per `new-plans/implementation.md`:
//! - Evidence is passed as a pointer (8 bytes) to all effectful functions
//! - Pure functions (empty effect row `{}`) don't receive evidence
//! - Handler installation creates new Evidence (handled in handler_lower pass)

use std::collections::{BTreeMap, HashSet};

use tribute_ir::dialect::ability as arena_ability;
use trunk_ir::Symbol;
use trunk_ir::context::{BlockArgData, IrContext};
use trunk_ir::dialect::func as arena_func;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{OpRef, TypeRef, ValueRef};
use trunk_ir::rewrite::{
    Module, PatternApplicator, PatternRewriter, RewritePattern, TypeConverter,
};
use trunk_ir::types::TypeDataBuilder;

// ============================================================================
// Arena-based evidence pass implementation
// ============================================================================

/// Check if an arena `core.func` type has concrete abilities in its effect row.
///
/// Note: Effect information is no longer stored on `core.func` types, so this
/// always returns `false`. Retained for API compatibility.
pub fn is_effectful_type(_ctx: &IrContext, _ty: TypeRef) -> bool {
    false
}

/// Check if a `core.func` type has a tail effect variable (effect-polymorphic).
///
/// Note: Effect information is no longer stored on `core.func` types, so this
/// always returns `false`. Retained for API compatibility.
pub fn has_tail_effect_variable(_ctx: &IrContext, _ty: TypeRef) -> bool {
    false
}

/// Check if a `core.func` type has evidence as its first parameter.
pub(crate) fn has_evidence_first_param(ctx: &IrContext, func_ty: TypeRef) -> bool {
    let data = ctx.types.get(func_ty);
    if data.dialect != Symbol::new("core") || data.name != Symbol::new("func") {
        return false;
    }
    // params[0] = return, params[1..] = param types
    if data.params.len() < 2 {
        return false;
    }
    arena_ability::is_evidence_type_ref(ctx, data.params[1])
}

/// Collect names of all effectful functions in the module.
pub fn collect_effectful_functions(ctx: &IrContext, module: Module) -> HashSet<Symbol> {
    let mut effectful = HashSet::new();
    for op in module.ops(ctx) {
        if let Ok(func_op) = arena_func::Func::from_op(ctx, op) {
            let func_ty = func_op.r#type(ctx);
            if is_effectful_type(ctx, func_ty) {
                effectful.insert(func_op.sym_name(ctx));
            }
        }
    }
    effectful
}

/// Collect names of functions whose first parameter is evidence type.
fn collect_functions_with_evidence_param(ctx: &IrContext, module: Module) -> HashSet<Symbol> {
    let mut fns_with_evidence = HashSet::new();
    for op in module.ops(ctx) {
        if let Ok(func_op) = arena_func::Func::from_op(ctx, op) {
            let func_ty = func_op.r#type(ctx);
            if has_evidence_first_param(ctx, func_ty) {
                fns_with_evidence.insert(func_op.sym_name(ctx));
            }
        }
    }
    fns_with_evidence
}

/// Build a new `core.func` TypeRef with evidence prepended to params.
pub fn build_func_type_with_evidence(
    ctx: &mut IrContext,
    old_func_ty: TypeRef,
    ev_ty: TypeRef,
) -> TypeRef {
    let data = ctx.types.get(old_func_ty);
    // params[0] = return, params[1..] = param types
    let result_ty = data.params[0];
    let old_params = &data.params[1..];

    let mut new_params = Vec::with_capacity(old_params.len() + 1);
    new_params.push(ev_ty);
    new_params.extend_from_slice(old_params);

    trunk_ir::dialect::core::func(ctx, result_ty, new_params).as_type_ref()
}

/// Pattern that adds evidence parameters to effectful `func.func` signatures.
struct AddEvidenceParamPattern {
    effectful_fns: HashSet<Symbol>,
    ev_ty: TypeRef,
}

impl RewritePattern for AddEvidenceParamPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(func_op) = arena_func::Func::from_op(ctx, op) else {
            return false;
        };

        let func_name = func_op.sym_name(ctx);
        if !self.effectful_fns.contains(&func_name) {
            return false;
        }

        let func_ty = func_op.r#type(ctx);
        if has_evidence_first_param(ctx, func_ty) {
            return false;
        }

        let new_func_ty = build_func_type_with_evidence(ctx, func_ty, self.ev_ty);

        let body = func_op.body(ctx);
        let blocks = &ctx.region(body).blocks;
        if blocks.is_empty() {
            return false;
        }
        let entry_block = blocks[0];

        ctx.prepend_block_arg(
            entry_block,
            BlockArgData {
                ty: self.ev_ty,
                attrs: BTreeMap::new(),
            },
        );

        let loc = ctx.op(op).location;
        ctx.detach_region(body);
        let new_op = arena_func::func(ctx, loc, func_name, new_func_ty, body).op_ref();
        rewriter.replace_op(new_op);
        true
    }

    fn name(&self) -> &'static str {
        "AddEvidenceParamPattern"
    }
}

/// Phase 1 (arena): Add evidence parameters to effectful function signatures.
pub fn add_evidence_params(ctx: &mut IrContext, module: Module) {
    let effectful_fns = collect_effectful_functions(ctx, module);
    if effectful_fns.is_empty() {
        return;
    }

    let ev_ty = arena_ability::evidence_adt_type_ref(ctx);

    let applicator =
        PatternApplicator::new(TypeConverter::new()).add_pattern(AddEvidenceParamPattern {
            effectful_fns,
            ev_ty,
        });
    applicator.apply_partial(ctx, module);
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
        if let Ok(func_op) = arena_func::Func::from_op(ctx, parent_op) {
            let body = func_op.body(ctx);
            let entry = ctx.region(body).blocks[0];
            let args = ctx.block_args(entry);
            if !args.is_empty() && arena_ability::is_evidence_type_ref(ctx, ctx.value_ty(args[0])) {
                return Some(args[0]);
            }
            return None;
        }
        current = parent_op;
    }
}

/// Pattern that transforms `func.call` ops to pass evidence to effectful callees.
struct TransformEvidenceCallPattern {
    effectful_fns: HashSet<Symbol>,
}

impl RewritePattern for TransformEvidenceCallPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(call_op) = arena_func::Call::from_op(ctx, op) else {
            return false;
        };

        let callee = call_op.callee(ctx);
        if !self.effectful_fns.contains(&callee) {
            return false;
        }

        let loc = ctx.op(op).location;

        let Some(ev_value) = find_enclosing_evidence(ctx, op) else {
            return false;
        };

        // Check if evidence is already the first argument (by value or by type)
        let operands = ctx.op_operands(op);
        if !operands.is_empty()
            && (operands[0] == ev_value
                || arena_ability::is_evidence_type_ref(ctx, ctx.value_ty(operands[0])))
        {
            return false;
        }

        let old_args: Vec<ValueRef> = operands.to_vec();
        let result_types: Vec<TypeRef> = ctx.op_result_types(op).to_vec();

        let mut new_args = vec![ev_value];
        new_args.extend(old_args.iter().copied());

        let result_ty = result_types.first().copied().unwrap_or_else(|| {
            ctx.types
                .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("nil")).build())
        });

        let new_call = arena_func::call(ctx, loc, new_args, result_ty, callee);
        rewriter.replace_op(new_call.op_ref());
        true
    }

    fn name(&self) -> &'static str {
        "TransformEvidenceCallPattern"
    }
}

/// Phase 2 (arena): Transform calls to pass evidence through call sites.
pub fn transform_evidence_calls(ctx: &mut IrContext, module: Module) {
    let effectful_fns = collect_effectful_functions(ctx, module);
    let fns_with_evidence = collect_functions_with_evidence_param(ctx, module);

    if effectful_fns.is_empty() && fns_with_evidence.is_empty() {
        return;
    }

    let all_effectful: HashSet<Symbol> = effectful_fns.union(&fns_with_evidence).copied().collect();

    let applicator =
        PatternApplicator::new(TypeConverter::new()).add_pattern(TransformEvidenceCallPattern {
            effectful_fns: all_effectful,
        });
    applicator.apply_partial(ctx, module);
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::context::{BlockData, OperationDataBuilder, RegionData};
    use trunk_ir::location::Span;
    use trunk_ir::smallvec::smallvec;
    use trunk_ir::types::{Attribute, Location};

    fn test_ctx() -> (IrContext, Location) {
        let mut ctx = IrContext::new();
        let path = ctx.paths.intern("test.trb".to_owned());
        let loc = Location::new(path, Span::new(0, 0));
        (ctx, loc)
    }

    fn i32_type(ctx: &mut IrContext) -> trunk_ir::refs::TypeRef {
        ctx.types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build())
    }

    fn make_func_type(
        ctx: &mut IrContext,
        params: &[trunk_ir::refs::TypeRef],
        ret: trunk_ir::refs::TypeRef,
    ) -> trunk_ir::refs::TypeRef {
        ctx.types.intern(
            TypeDataBuilder::new(Symbol::new("core"), Symbol::new("func"))
                .param(ret)
                .params(params.iter().copied())
                .build(),
        )
    }

    fn make_func_op(
        ctx: &mut IrContext,
        loc: Location,
        name: &'static str,
        func_type: trunk_ir::refs::TypeRef,
        param_types: &[trunk_ir::refs::TypeRef],
    ) -> trunk_ir::refs::OpRef {
        let entry_block = ctx.create_block(BlockData {
            location: loc,
            args: param_types
                .iter()
                .map(|&ty| BlockArgData {
                    ty,
                    attrs: BTreeMap::new(),
                })
                .collect(),
            ops: smallvec![],
            parent_region: None,
        });
        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry_block],
            parent_op: None,
        });
        let f = arena_func::func(ctx, loc, Symbol::new(name), func_type, body);
        f.op_ref()
    }

    fn make_module(ctx: &mut IrContext, loc: Location, ops: Vec<trunk_ir::refs::OpRef>) -> Module {
        let block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        for op in ops {
            ctx.push_op(block, op);
        }
        let region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![block],
            parent_op: None,
        });
        let module_data =
            OperationDataBuilder::new(loc, Symbol::new("core"), Symbol::new("module"))
                .attr("sym_name", Attribute::Symbol(Symbol::new("test")))
                .region(region)
                .build(ctx);
        let module_op = ctx.create_op(module_data);
        Module::new(ctx, module_op).expect("test module should be valid")
    }

    // === is_effectful_type tests ===
    // Effect information is no longer stored on core.func types,
    // so is_effectful_type always returns false.

    #[test]
    fn test_is_effectful_type_always_false() {
        let (mut ctx, _) = test_ctx();
        let i32_ty = i32_type(&mut ctx);
        let pure_ty = make_func_type(&mut ctx, &[i32_ty], i32_ty);
        assert!(!is_effectful_type(&ctx, pure_ty));
        assert!(!is_effectful_type(&ctx, i32_ty));
    }

    // === add_evidence_params tests ===
    // Since is_effectful_type always returns false (effect attribute removed from
    // core.func), add_evidence_params is a no-op. These tests verify that behavior.

    #[test]
    fn test_add_evidence_params_no_effectful() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);
        let pure_ty = make_func_type(&mut ctx, &[i32_ty], i32_ty);
        let func_op = make_func_op(&mut ctx, loc, "bar", pure_ty, &[i32_ty]);
        let module = make_module(&mut ctx, loc, vec![func_op]);

        add_evidence_params(&mut ctx, module);

        // Function should be unchanged
        let ops = module.ops(&ctx);
        let bar = arena_func::Func::from_op(&ctx, ops[0]).unwrap();
        assert_eq!(bar.r#type(&ctx), pure_ty);
    }

    // === transform_evidence_calls tests ===

    #[test]
    fn test_arena_evidence_no_calls_to_transform() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);

        // Pure caller calling pure function — nothing to transform
        let pure_ty = make_func_type(&mut ctx, &[i32_ty], i32_ty);
        let func_op = make_func_op(&mut ctx, loc, "pure_fn", pure_ty, &[i32_ty]);
        let module = make_module(&mut ctx, loc, vec![func_op]);

        // Should be no-op
        transform_evidence_calls(&mut ctx, module);

        let ops = module.ops(&ctx);
        let f = arena_func::Func::from_op(&ctx, ops[0]).unwrap();
        assert_eq!(f.r#type(&ctx), pure_ty);
    }
}
