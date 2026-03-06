//! Lower closure operations in indirect calls.
//!
//! This pass transforms `func.call_indirect` operations when the callee
//! is a closure:
//!
//! Before:
//! ```text
//! %closure = closure.new @lifted_func, %env
//! %result = func.call_indirect %closure, %args...
//! ```
//!
//! After:
//! ```text
//! %closure = closure.new @lifted_func, %env
//! %funcref = closure.func %closure
//! %env = closure.env %closure
//! %result = func.call_indirect %funcref, %env, %args...
//! ```
//!
//! Uses `ArenaRewritePattern` + `PatternApplicator` for declarative transformation.

use std::collections::HashSet;

use tribute_ir::arena::dialect::closure as arena_closure;
use trunk_ir::Symbol;
use trunk_ir::arena::context::IrContext;
use trunk_ir::arena::dialect::adt as arena_adt;
use trunk_ir::arena::dialect::func as arena_func;
use trunk_ir::arena::ops::ArenaDialectOp;
use trunk_ir::arena::refs::{OpRef, TypeRef, ValueRef};
use trunk_ir::arena::rewrite::{
    ArenaModule, ArenaRewritePattern, ArenaTypeConverter,
    PatternApplicator as ArenaPatternApplicator, PatternRewriter as ArenaPatternRewriter,
};
use trunk_ir::arena::types::{Attribute as ArenaAttribute, TypeDataBuilder};

use crate::evidence::collect_effectful_functions_arena;

/// Create the unified closure struct type in arena: `{ table_idx: i32, env: anyref }`.
fn closure_struct_type_ref(ctx: &mut IrContext) -> TypeRef {
    let i32_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());
    let anyref_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("tribute_rt"), Symbol::new("any")).build());
    ctx.types.intern(
        TypeDataBuilder::new(Symbol::new("adt"), Symbol::new("struct"))
            .param(i32_ty)
            .param(anyref_ty)
            .attr("name", ArenaAttribute::Symbol(Symbol::new("_closure")))
            .build(),
    )
}

/// Create an `i32` type ref in arena.
fn i32_type_ref(ctx: &mut IrContext) -> TypeRef {
    ctx.types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build())
}

/// Create a `tribute_rt.any` (anyref) type ref in arena.
fn anyref_type_ref(ctx: &mut IrContext) -> TypeRef {
    ctx.types
        .intern(TypeDataBuilder::new(Symbol::new("tribute_rt"), Symbol::new("any")).build())
}

/// Check if a TypeRef is a closure.closure type.
fn is_closure_type_ref(ctx: &IrContext, ty: TypeRef) -> bool {
    let data = ctx.types.get(ty);
    data.dialect == Symbol::new("closure") && data.name == Symbol::new("closure")
}

/// Check if a TypeRef is a core.func type.
fn is_core_func_type_ref(ctx: &IrContext, ty: TypeRef) -> bool {
    let data = ctx.types.get(ty);
    data.dialect == Symbol::new("core") && data.name == Symbol::new("func")
}

/// Check if a TypeRef is a cont.continuation type.
fn is_continuation_type_ref(ctx: &IrContext, ty: TypeRef) -> bool {
    let data = ctx.types.get(ty);
    data.dialect == Symbol::new("cont") && data.name == Symbol::new("continuation")
}

/// Check if a TypeRef is an adt.struct with name "_closure".
fn is_closure_struct_type_ref(ctx: &IrContext, ty: TypeRef) -> bool {
    let data = ctx.types.get(ty);
    if data.dialect != Symbol::new("adt") || data.name != Symbol::new("struct") {
        return false;
    }
    matches!(
        data.attrs.get(&Symbol::new("name")),
        Some(ArenaAttribute::Symbol(s)) if *s == Symbol::new("_closure")
    )
}

/// Extract the inner func type from a closure.closure TypeRef.
fn extract_closure_func_type(ctx: &IrContext, ty: TypeRef) -> Option<TypeRef> {
    let data = ctx.types.get(ty);
    if data.dialect != Symbol::new("closure") || data.name != Symbol::new("closure") {
        return None;
    }
    // closure.closure type has the inner func type as first param
    data.params.first().copied()
}

/// Get the type of an arena value.
fn arena_value_type(ctx: &IrContext, value: ValueRef) -> TypeRef {
    ctx.value_ty(value)
}

/// Check if an arena value is a closure value (for pre-lowering collection).
fn is_any_closure_value_arena(ctx: &IrContext, value: ValueRef) -> bool {
    use trunk_ir::arena::refs::ValueDef as ArenaValueDef;

    let ty = arena_value_type(ctx, value);

    // Direct check for closure.new result
    if let ArenaValueDef::OpResult(op, _) = ctx.value_def(value)
        && arena_closure::New::from_op(ctx, op).is_ok()
    {
        return true;
    }

    // Check type
    if is_closure_type_ref(ctx, ty) {
        return true;
    }
    if is_core_func_type_ref(ctx, ty) {
        // Only treat core.func as closure if it's a block arg (matching LowerClosureCallArena)
        return matches!(
            ctx.value_def(value),
            trunk_ir::arena::refs::ValueDef::BlockArg(_, _)
        );
    }
    if is_continuation_type_ref(ctx, ty) {
        return true;
    }
    false
}

/// Collect ALL closure call_indirect operations as OpRefs.
/// Collect all closure calls by location span (stable across Phase 1 rewrites).
///
/// We use `(span.start, span.end)` instead of `OpRef` because Phase 1 pattern
/// application destroys original ops and creates new ones with different OpRefs.
fn collect_all_closure_calls_arena(
    ctx: &IrContext,
    module: ArenaModule,
) -> HashSet<(usize, usize)> {
    let mut closure_calls = HashSet::new();
    for op in module.ops(ctx) {
        if let Ok(func_op) = arena_func::Func::from_op(ctx, op) {
            let body = func_op.body(ctx);
            collect_closure_calls_in_region_arena(ctx, body, &mut closure_calls);
        }
    }
    closure_calls
}

fn collect_closure_calls_in_region_arena(
    ctx: &IrContext,
    region: trunk_ir::arena::refs::RegionRef,
    closure_calls: &mut HashSet<(usize, usize)>,
) {
    for &block in ctx.region(region).blocks.iter() {
        for &op in ctx.block(block).ops.iter() {
            collect_closure_calls_in_op_arena(ctx, op, closure_calls);
        }
    }
}

fn collect_closure_calls_in_op_arena(
    ctx: &IrContext,
    op: OpRef,
    closure_calls: &mut HashSet<(usize, usize)>,
) {
    // Check if this is a call_indirect with a closure callee
    if arena_func::CallIndirect::from_op(ctx, op).is_ok() {
        let operands = ctx.op_operands(op);
        if let Some(&callee) = operands.first()
            && is_any_closure_value_arena(ctx, callee)
        {
            let loc = ctx.op(op).location;
            closure_calls.insert((loc.span.start, loc.span.end));
        }
    }

    // Recurse into regions
    for &region in ctx.op(op).regions.iter() {
        collect_closure_calls_in_region_arena(ctx, region, closure_calls);
    }
}

// ============================================================================
// Rewrite Patterns
// ============================================================================

/// Update function signatures to convert `core.func` params to `closure.closure`.
struct UpdateFuncSignatureArena;

impl ArenaRewritePattern for UpdateFuncSignatureArena {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut ArenaPatternRewriter<'_>,
    ) -> bool {
        let Ok(func_op) = arena_func::Func::from_op(ctx, op) else {
            return false;
        };

        let func_ty = func_op.r#type(ctx);
        let func_data = ctx.types.get(func_ty);
        if func_data.dialect != Symbol::new("core") || func_data.name != Symbol::new("func") {
            return false;
        }

        // params[0] = return, params[1..] = param types
        if func_data.params.is_empty() {
            return false;
        }

        // Clone data we need before mutating ctx.types
        let params: Vec<TypeRef> = func_data.params.to_vec();
        let effect_attr = func_data.attrs.get(&Symbol::new("effect")).cloned();

        let mut needs_update = false;
        let mut new_params = Vec::with_capacity(params.len());
        new_params.push(params[0]); // return type

        for &param_ty in &params[1..] {
            if is_core_func_type_ref(ctx, param_ty) {
                // Convert core.func to closure.closure wrapping the func type
                let closure_ty = ctx.types.intern(
                    TypeDataBuilder::new(Symbol::new("closure"), Symbol::new("closure"))
                        .param(param_ty)
                        .build(),
                );
                new_params.push(closure_ty);
                needs_update = true;
            } else {
                new_params.push(param_ty);
            }
        }

        if !needs_update {
            return false;
        }

        // Build new func type preserving effect attribute
        let mut builder =
            TypeDataBuilder::new(Symbol::new("core"), Symbol::new("func")).params(new_params);
        if let Some(eff) = effect_attr {
            builder = builder.attr("effect", eff);
        }
        let new_func_ty = ctx.types.intern(builder.build());

        // Rebuild the function with new type
        let func_name = func_op.sym_name(ctx);
        let body = func_op.body(ctx);
        let loc = ctx.op(op).location;
        ctx.detach_region(body);
        let new_op = arena_func::func(ctx, loc, func_name, new_func_ty, body).op_ref();
        rewriter.replace_op(new_op);
        true
    }

    fn name(&self) -> &'static str {
        "UpdateFuncSignatureArena"
    }
}

/// Lower `closure.new` to `func.constant` + `adt.struct_new`.
struct LowerClosureNewArena;

impl ArenaRewritePattern for LowerClosureNewArena {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut ArenaPatternRewriter<'_>,
    ) -> bool {
        let Ok(closure_new) = arena_closure::New::from_op(ctx, op) else {
            return false;
        };

        let loc = ctx.op(op).location;
        let func_ref = closure_new.func_ref(ctx);
        let env = closure_new.env(ctx);

        // Extract function type from closure.closure result type
        let result_ty = ctx.op_result_types(op)[0];
        let func_ty = extract_closure_func_type(ctx, result_ty)
            .expect("closure.new result type must contain a valid func type (from func.constant)");

        // Generate: %funcref = func.constant @func_ref : func_type
        let constant_op = arena_func::constant(ctx, loc, func_ty, func_ref);
        let funcref = ctx.op_result(constant_op.op_ref(), 0);

        // Generate: %closure = adt.struct_new(%funcref, %env) : closure_struct_type
        let struct_ty = closure_struct_type_ref(ctx);
        let struct_new_op =
            arena_adt::struct_new(ctx, loc, vec![funcref, env], struct_ty, struct_ty);

        rewriter.insert_op(constant_op.op_ref());
        rewriter.replace_op(struct_new_op.op_ref());
        true
    }

    fn name(&self) -> &'static str {
        "LowerClosureNewArena"
    }
}

/// Lower `func.call_indirect` on closure values.
struct LowerClosureCallArena;

impl ArenaRewritePattern for LowerClosureCallArena {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut ArenaPatternRewriter<'_>,
    ) -> bool {
        if arena_func::CallIndirect::from_op(ctx, op).is_err() {
            return false;
        }

        let operands = ctx.op_operands(op);
        if operands.is_empty() {
            return false;
        }
        let callee = operands[0];
        let callee_ty = arena_value_type(ctx, callee);

        // Determine if callee is a closure
        let callee_is_closure = if is_closure_type_ref(ctx, callee_ty) {
            true
        } else if is_closure_struct_type_ref(ctx, callee_ty) {
            // Already lowered closure struct
            true
        } else if is_core_func_type_ref(ctx, callee_ty) {
            // core.func: only treat as closure if it's a block arg
            matches!(
                ctx.value_def(callee),
                trunk_ir::arena::refs::ValueDef::BlockArg(_, _)
            )
        } else {
            // Fallback: check if result of closure.new
            if let trunk_ir::arena::refs::ValueDef::OpResult(def_op, _) = ctx.value_def(callee) {
                arena_closure::New::from_op(ctx, def_op).is_ok()
            } else {
                false
            }
        };

        if !callee_is_closure {
            return false;
        }

        let loc = ctx.op(op).location;
        let args: Vec<ValueRef> = operands[1..].to_vec();
        let result_ty = ctx.op_result_types(op)[0];

        let i32_ty = i32_type_ref(ctx);
        let anyref_ty = anyref_type_ref(ctx);

        // Generate: %table_idx = closure.func %closure
        let table_idx_op = arena_closure::func(ctx, loc, callee, i32_ty);
        let table_idx = ctx.op_result(table_idx_op.op_ref(), 0);

        // Generate: %env = closure.env %closure
        let env_op = arena_closure::env(ctx, loc, callee, anyref_ty);
        let env = ctx.op_result(env_op.op_ref(), 0);

        // Generate: %result = func.call_indirect %table_idx, [%env, %args...]
        let mut new_args = vec![env];
        new_args.extend(args);
        let new_call = arena_func::call_indirect(ctx, loc, table_idx, new_args, result_ty);

        rewriter.insert_op(table_idx_op.op_ref());
        rewriter.insert_op(env_op.op_ref());
        rewriter.replace_op(new_call.op_ref());
        true
    }

    fn name(&self) -> &'static str {
        "LowerClosureCallArena"
    }
}

/// Lower `closure.func` to `adt.struct_get` field 0.
struct LowerClosureFuncArena;

impl ArenaRewritePattern for LowerClosureFuncArena {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut ArenaPatternRewriter<'_>,
    ) -> bool {
        if arena_closure::Func::from_op(ctx, op).is_err() {
            return false;
        }

        let loc = ctx.op(op).location;
        let closure_value = ctx.op_operands(op)[0];
        let i32_ty = i32_type_ref(ctx);
        let struct_ty = closure_struct_type_ref(ctx);

        let get_op = arena_adt::struct_get(ctx, loc, closure_value, i32_ty, struct_ty, 0);
        rewriter.replace_op(get_op.op_ref());
        true
    }

    fn name(&self) -> &'static str {
        "LowerClosureFuncArena"
    }
}

/// Lower `closure.env` to `adt.struct_get` field 1.
struct LowerClosureEnvArena;

impl ArenaRewritePattern for LowerClosureEnvArena {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut ArenaPatternRewriter<'_>,
    ) -> bool {
        if arena_closure::Env::from_op(ctx, op).is_err() {
            return false;
        }

        let loc = ctx.op(op).location;
        let closure_value = ctx.op_operands(op)[0];
        let result_ty = ctx.op_result_types(op)[0];
        let struct_ty = closure_struct_type_ref(ctx);

        let get_op = arena_adt::struct_get(ctx, loc, closure_value, result_ty, struct_ty, 1);
        rewriter.replace_op(get_op.op_ref());
        true
    }

    fn name(&self) -> &'static str {
        "LowerClosureEnvArena"
    }
}

// ============================================================================
// Phase 2: Evidence passing for closure calls
// ============================================================================

/// Transform closure calls to pass evidence in arena IR.
///
/// After pattern application, closure calls have been expanded. Now we insert
/// evidence as the first argument to all closure call_indirect operations.
fn transform_closure_calls_with_evidence_arena(
    ctx: &mut IrContext,
    module: ArenaModule,
    effectful_fns: &HashSet<Symbol>,
    closure_calls: &HashSet<(usize, usize)>,
) {
    if closure_calls.is_empty() {
        return;
    }

    let func_ops: Vec<OpRef> = module.ops(ctx);
    for func_op_ref in func_ops {
        let Ok(func_op) = arena_func::Func::from_op(ctx, func_op_ref) else {
            continue;
        };

        let func_name = func_op.sym_name(ctx);
        let is_effectful = effectful_fns.contains(&func_name);

        let body = func_op.body(ctx);
        let blocks: Vec<_> = ctx.region(body).blocks.to_vec();
        if blocks.is_empty() {
            continue;
        }

        // For effectful functions, get evidence from first block argument
        let evidence_from_param = if is_effectful {
            let entry = blocks[0];
            let args = ctx.block_args(entry);
            if !args.is_empty() {
                let ev = args[0];
                if tribute_ir::arena::dialect::ability::is_evidence_type_ref(ctx, ctx.value_ty(ev))
                {
                    Some(ev)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        let loc = ctx.op(func_op_ref).location;
        transform_closure_calls_in_region_arena(ctx, body, evidence_from_param, closure_calls, loc);
    }
}

/// Transform closure calls in a region, inserting evidence arguments.
fn transform_closure_calls_in_region_arena(
    ctx: &mut IrContext,
    region: trunk_ir::arena::refs::RegionRef,
    evidence_from_param: Option<ValueRef>,
    closure_calls: &HashSet<(usize, usize)>,
    func_location: trunk_ir::arena::types::Location,
) {
    let blocks: Vec<_> = ctx.region(region).blocks.to_vec();
    for block in blocks {
        transform_closure_calls_in_block_arena(
            ctx,
            block,
            evidence_from_param,
            closure_calls,
            func_location,
        );
    }
}

/// Transform closure calls in a block, inserting evidence arguments.
fn transform_closure_calls_in_block_arena(
    ctx: &mut IrContext,
    block: trunk_ir::arena::refs::BlockRef,
    evidence_from_param: Option<ValueRef>,
    closure_calls: &HashSet<(usize, usize)>,
    func_location: trunk_ir::arena::types::Location,
) {
    // We need to track null evidence creation (lazy)
    let mut null_ev_value: Option<ValueRef> = None;

    // Collect ops to process (snapshot since we'll mutate)
    let ops: Vec<OpRef> = ctx.block(block).ops.to_vec();

    for op in ops {
        // Process nested regions first
        let regions: Vec<_> = ctx.op(op).regions.to_vec();
        for region in regions {
            transform_closure_calls_in_region_arena(
                ctx,
                region,
                evidence_from_param,
                closure_calls,
                func_location,
            );
        }

        // Check if this is a closure call_indirect that needs evidence
        if arena_func::CallIndirect::from_op(ctx, op).is_err() {
            continue;
        }

        let loc_key = {
            let loc = ctx.op(op).location;
            (loc.span.start, loc.span.end)
        };
        if !closure_calls.contains(&loc_key) {
            continue;
        }

        let evidence = if let Some(ev) = evidence_from_param {
            ev
        } else {
            // Create null evidence lazily
            if null_ev_value.is_none() {
                let evidence_ty = tribute_ir::arena::dialect::ability::evidence_adt_type_ref(ctx);
                let null_op = arena_adt::ref_null(ctx, func_location, evidence_ty, evidence_ty);
                let ev = ctx.op_result(null_op.op_ref(), 0);
                // Insert null evidence at the beginning of the block
                let first_op_in_block = ctx.block(block).ops.first().copied();
                if let Some(first_op) = first_op_in_block {
                    ctx.insert_op_before(block, first_op, null_op.op_ref());
                } else {
                    ctx.push_op(block, null_op.op_ref());
                }
                null_ev_value = Some(ev);
            }
            null_ev_value.unwrap()
        };

        // Transform: add evidence as first argument after table_idx
        let operands = ctx.op_operands(op).to_vec();
        let result_ty = ctx.op_result_types(op)[0];
        let loc = ctx.op(op).location;

        // operands[0] = table_idx, operands[1..] = env + args
        let table_idx = operands[0];
        let rest_args: Vec<ValueRef> = operands[1..].to_vec();

        // Build new args: [evidence, env, args...]
        let mut new_args = vec![evidence];
        new_args.extend(rest_args);

        let new_call = arena_func::call_indirect(ctx, loc, table_idx, new_args, result_ty);

        // Replace old result uses with new result
        let old_result = ctx.op_result(op, 0);
        let new_result = ctx.op_result(new_call.op_ref(), 0);
        ctx.replace_all_uses(old_result, new_result);

        // Insert new call and remove old one
        ctx.insert_op_before(block, op, new_call.op_ref());
        ctx.remove_op_from_block(block, op);
    }
}

/// Lower closures using arena IR.
///
/// This pass has two phases:
///
/// Phase 1 (PatternApplicator):
/// 1. UpdateFuncSignatureArena - updates function signatures: core.func params → closure.closure
/// 2. LowerClosureCallArena - expands call_indirect to use closure.func/closure.env
/// 3. LowerClosureNewArena - expands closure.new to func.constant + adt.struct_new
/// 4. LowerClosureFuncArena - extracts i32 table index from struct (field 0)
/// 5. LowerClosureEnvArena - extracts env from struct (field 1)
///
/// Phase 2 (Post-processing):
/// - Transform ALL closure calls to pass evidence from the enclosing function
pub fn lower_closures(ctx: &mut IrContext, module: ArenaModule) {
    // Collect effectful functions BEFORE lowering (while closure types are intact)
    let effectful_fns = collect_effectful_functions_arena(ctx, module);

    // Collect ALL closure calls before pattern application
    let all_closure_calls = collect_all_closure_calls_arena(ctx, module);

    // Phase 1: Pattern application
    let applicator = ArenaPatternApplicator::new(ArenaTypeConverter::new())
        .add_pattern(UpdateFuncSignatureArena)
        .add_pattern(LowerClosureCallArena)
        .add_pattern(LowerClosureNewArena)
        .add_pattern(LowerClosureFuncArena)
        .add_pattern(LowerClosureEnvArena);
    applicator.apply_partial(ctx, module);

    // Phase 2: Evidence passing for closure calls
    transform_closure_calls_with_evidence_arena(ctx, module, &effectful_fns, &all_closure_calls);
}
