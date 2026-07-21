//! Evidence runtime lowering for the native backend.
//!
//! This pass adapts evidence-related IR from `resolve_evidence` for native
//! (Cranelift) code generation.  The WASM backend (`evidence_to_wasm.rs`)
//! replaces the stubs with inline binary-search IR; the native backend instead
//! delegates to `extern "C"` functions in `tribute-runtime`.
//!
//! ## Transformations
//!
//! 1. **Stub replacement** — `func.func @__tribute_evidence_lookup` /
//!    `@__tribute_evidence_extend` stubs (with `unreachable` body) are replaced
//!    by extern declarations with native signatures.
//!
//! 2. **Empty evidence** — `adt.array_new(0, evidence_ty)` →
//!    `func.call @__tribute_evidence_empty()`.
//!
//! 3. **Effect ABI lowering** — `effect.extend`, `effect.dispatch_tail`, and
//!    `effect.dispatch_cps` are lowered to the native evidence runtime ABI and
//!    closure indirect calls.
//!
//! 4. **TR dispatch field** — `adt.struct_get(marker, MarkerField::TrDispatchFn)` on evidence_lookup
//!    results is rewritten to `func.call @__tribute_evidence_lookup_tr(ev, ability_id)`.

use std::collections::HashMap;
use std::ops::ControlFlow;

use tribute_ir::dialect::ability::{
    self, MarkerField, compute_op_idx, evidence_abi, evidence_runtime_symbols,
};
use tribute_ir::dialect::{effect, tribute_rt};
use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::func;
use trunk_ir::dialect::{adt, arith, core};
use trunk_ir::ops::DialectOp;
use trunk_ir::pass::{Pass, PassRunError, PassRunResult};
use trunk_ir::refs::{BlockRef, OpRef, RegionRef, TypeRef, ValueRef};
use trunk_ir::rewrite::helpers::erase_op;
use trunk_ir::rewrite::{
    ConversionError, ConversionTarget, Module, PatternApplicator, PatternRewriter, RewritePattern,
    TypeConverter,
};
use trunk_ir::types::{Attribute, Location, TypeDataBuilder};
use trunk_ir::walk::{WalkAction, walk_op};

/// Lower evidence operations for the native backend.
///
/// Must run AFTER effect lowering passes and BEFORE DCE.
pub fn lower_evidence_to_native(ctx: &mut IrContext, module: Module) {
    prepare_native_evidence_runtime(ctx, module);
    rewrite_evidence_ops_in_module(ctx, module);
}

/// Prepare native evidence runtime declarations at module scope.
pub fn prepare_native_evidence_runtime(ctx: &mut IrContext, module: Module) {
    replace_stubs_and_add_empty(ctx, module);
}

/// Lower evidence operations inside one function for the native backend.
pub fn lower_evidence_to_native_func(ctx: &mut IrContext, func_op: func::Func) {
    try_lower_evidence_to_native_func(ctx, func_op).expect("native evidence lowering failed");
}

fn try_lower_evidence_to_native_func(ctx: &mut IrContext, func_op: func::Func) -> PassRunResult {
    if is_evidence_runtime_fn(func_op.sym_name(ctx)) {
        return Ok(());
    }
    lower_effect_abi_to_native(ctx, func_op)?;
    rewrite_evidence_ops_in_region(ctx, func_op.body(ctx))?;
    Ok(())
}

/// PassManager-friendly native evidence lowering pass.
pub struct LowerEvidenceToNative;

impl Pass for LowerEvidenceToNative {
    type Target = func::Func;

    fn name(&self) -> &'static str {
        "lower-evidence-to-native"
    }

    fn run(&mut self, ctx: &mut IrContext, target: func::Func) -> PassRunResult {
        try_lower_evidence_to_native_func(ctx, target)
    }
}

// =============================================================================
// Phase 1: Replace stubs + add __tribute_evidence_empty declaration
// =============================================================================

fn replace_stubs_and_add_empty(ctx: &mut IrContext, module: Module) {
    let first_block = match module.first_block(ctx) {
        Some(b) => b,
        None => return,
    };

    let loc = ctx.op(module.op()).location;
    let ops: Vec<OpRef> = ctx.block(first_block).ops.to_vec();

    let mut has_evidence_empty = false;
    let mut has_lookup_tr = false;
    let mut has_lookup_handler = false;
    let mut stubs_to_replace: Vec<(OpRef, &'static str)> = Vec::new();

    let lookup_sym = Symbol::new(evidence_abi::LOOKUP);
    let extend_sym = Symbol::new(evidence_abi::EXTEND);
    let empty_sym = Symbol::new(evidence_abi::EMPTY);
    let lookup_tr_sym = Symbol::new(evidence_abi::LOOKUP_TR);
    let lookup_handler_sym = Symbol::new(evidence_abi::LOOKUP_HANDLER);

    for &op in &ops {
        if let Ok(func_op) = func::Func::from_op(ctx, op) {
            let name = func_op.sym_name(ctx);
            if name == lookup_sym {
                stubs_to_replace.push((op, evidence_abi::LOOKUP));
            } else if name == extend_sym {
                stubs_to_replace.push((op, evidence_abi::EXTEND));
            } else if name == empty_sym {
                has_evidence_empty = true;
            } else if name == lookup_tr_sym {
                has_lookup_tr = true;
            } else if name == lookup_handler_sym {
                has_lookup_handler = true;
            }
        }
    }

    let ptr_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("ptr")).build());
    let i32_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());

    // Replace stubs with extern declarations
    for (old_op, name) in stubs_to_replace {
        let new_op = match name {
            evidence_abi::LOOKUP => make_evidence_lookup_extern(ctx, loc, ptr_ty, i32_ty),
            evidence_abi::EXTEND => make_evidence_extend_extern(ctx, loc, ptr_ty, i32_ty),
            _ => unreachable!(),
        };
        // Insert new before old, then remove old
        ctx.insert_op_before(first_block, old_op, new_op);
        ctx.remove_op_from_block(first_block, old_op);
        ctx.remove_op(old_op);
    }

    // Add __tribute_evidence_empty if missing
    if !has_evidence_empty {
        let empty_op = make_evidence_empty_extern(ctx, loc, ptr_ty);
        // Insert at front of module block
        let block_ops = &ctx.block(first_block).ops;
        if block_ops.is_empty() {
            ctx.push_op(first_block, empty_op);
        } else {
            let first_op = block_ops[0];
            ctx.insert_op_before(first_block, first_op, empty_op);
        }
    }

    // Add __tribute_evidence_lookup_tr if missing
    if !has_lookup_tr {
        let lookup_tr_op = make_evidence_lookup_tr_extern(ctx, loc, ptr_ty, i32_ty);
        let block_ops = &ctx.block(first_block).ops;
        if block_ops.is_empty() {
            ctx.push_op(first_block, lookup_tr_op);
        } else {
            let first_op = block_ops[0];
            ctx.insert_op_before(first_block, first_op, lookup_tr_op);
        }
    }

    // Add __tribute_evidence_lookup_handler if missing
    if !has_lookup_handler {
        let lookup_handler_op = make_evidence_lookup_handler_extern(ctx, loc, ptr_ty, i32_ty);
        let block_ops = &ctx.block(first_block).ops;
        if block_ops.is_empty() {
            ctx.push_op(first_block, lookup_handler_op);
        } else {
            let first_op = block_ops[0];
            ctx.insert_op_before(first_block, first_op, lookup_handler_op);
        }
    }
}

/// Build extern `fn __tribute_evidence_empty() -> ptr`
fn make_evidence_empty_extern(ctx: &mut IrContext, loc: Location, ptr_ty: TypeRef) -> OpRef {
    super::build_extern_func(ctx, loc, evidence_abi::EMPTY, &[], ptr_ty)
}

/// Build extern `fn __tribute_evidence_lookup(ev: ptr, ability_id: i32) -> i32`
fn make_evidence_lookup_extern(
    ctx: &mut IrContext,
    loc: Location,
    ptr_ty: TypeRef,
    i32_ty: TypeRef,
) -> OpRef {
    super::build_extern_func(ctx, loc, evidence_abi::LOOKUP, &[ptr_ty, i32_ty], i32_ty)
}

/// Build extern `fn __tribute_evidence_extend(ev: ptr, ability_id: i32, prompt_tag: i32, tr_dispatch_fn: ptr, handler_dispatch: ptr) -> ptr`
fn make_evidence_extend_extern(
    ctx: &mut IrContext,
    loc: Location,
    ptr_ty: TypeRef,
    i32_ty: TypeRef,
) -> OpRef {
    super::build_extern_func(
        ctx,
        loc,
        evidence_abi::EXTEND,
        &[ptr_ty, i32_ty, i32_ty, ptr_ty, ptr_ty],
        ptr_ty,
    )
}

/// Build extern `fn __tribute_evidence_lookup_tr(ev: ptr, ability_id: i32) -> ptr`
fn make_evidence_lookup_tr_extern(
    ctx: &mut IrContext,
    loc: Location,
    ptr_ty: TypeRef,
    i32_ty: TypeRef,
) -> OpRef {
    super::build_extern_func(ctx, loc, evidence_abi::LOOKUP_TR, &[ptr_ty, i32_ty], ptr_ty)
}

/// Build extern `fn __tribute_evidence_lookup_handler(ev: ptr, ability_id: i32) -> ptr`
fn make_evidence_lookup_handler_extern(
    ctx: &mut IrContext,
    loc: Location,
    ptr_ty: TypeRef,
    i32_ty: TypeRef,
) -> OpRef {
    super::build_extern_func(
        ctx,
        loc,
        evidence_abi::LOOKUP_HANDLER,
        &[ptr_ty, i32_ty],
        ptr_ty,
    )
}

// =============================================================================
// Phase 2: Rewrite evidence ops inside function bodies
// =============================================================================

fn is_evidence_runtime_fn(name: Symbol) -> bool {
    evidence_runtime_symbols().contains(&name)
}

fn rewrite_evidence_ops_in_module(ctx: &mut IrContext, module: Module) {
    let mut funcs = Vec::new();
    let _ = walk_op::<()>(ctx, module.op(), &mut |op| {
        if let Ok(func_op) = func::Func::from_op(ctx, op) {
            funcs.push(func_op);
        }
        ControlFlow::Continue(WalkAction::Advance)
    });

    for func_op in funcs {
        lower_evidence_to_native_func(ctx, func_op);
    }
}

fn native_effect_abi_target() -> ConversionTarget {
    ConversionTarget::new()
        .legal_op("func", "func")
        .recursive_legal_op("func", "func")
        .illegal_op("effect", "extend")
        .illegal_op("effect", "dispatch_tail")
        .illegal_op("effect", "dispatch_cps")
}

fn lower_effect_abi_to_native(
    ctx: &mut IrContext,
    func_op: func::Func,
) -> Result<(), ConversionError> {
    PatternApplicator::new(TypeConverter::new())
        .with_target(native_effect_abi_target())
        .add_pattern(LowerEffectExtendToNative)
        .add_pattern(LowerEffectDispatchTailToNative)
        .add_pattern(LowerEffectDispatchCpsToNative)
        .apply_partial_conversion(ctx, func_op, "native-evidence-effect-abi")?;
    Ok(())
}

#[derive(Debug)]
struct NativeEvidenceRewriteError {
    op: OpRef,
    loc: Location,
    message: String,
}

impl std::fmt::Display for NativeEvidenceRewriteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "native evidence rewrite failed at op {:?} (loc {:?}): {}",
            self.op, self.loc, self.message
        )
    }
}

impl std::error::Error for NativeEvidenceRewriteError {}

fn native_evidence_rewrite_error(
    op: OpRef,
    loc: Location,
    message: impl Into<String>,
) -> PassRunError {
    Box::new(NativeEvidenceRewriteError {
        op,
        loc,
        message: message.into(),
    })
}

fn rewrite_evidence_ops_in_region(ctx: &mut IrContext, region: RegionRef) -> PassRunResult {
    let blocks: Vec<BlockRef> = ctx.region(region).blocks.to_vec();
    for block in blocks {
        rewrite_evidence_ops_in_block(ctx, block)?;
    }
    Ok(())
}

/// Check if a type is an evidence type in arena (adt.array<ability.evidence>).
fn is_evidence_type(ctx: &IrContext, ty: TypeRef) -> bool {
    tribute_ir::dialect::ability::is_evidence_type_ref(ctx, ty)
}

/// Check if a type is a Marker type in arena.
fn is_marker_type(ctx: &IrContext, ty: TypeRef) -> bool {
    tribute_ir::dialect::ability::is_marker_type_ref(ctx, ty)
}

fn op_idx_const(
    ctx: &mut IrContext,
    loc: Location,
    i32_ty: TypeRef,
    ability_ref: TypeRef,
    op_name: Symbol,
) -> arith::Const {
    let op_idx = compute_op_idx(ability::ability_name(ctx, ability_ref), Some(op_name));
    arith::r#const(ctx, loc, i32_ty, Attribute::Int(op_idx as i128))
}

fn core_ptr_type(ctx: &mut IrContext) -> TypeRef {
    ctx.types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("ptr")).build())
}

fn core_i32_type(ctx: &mut IrContext) -> TypeRef {
    ctx.types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build())
}

struct LowerEffectExtendToNative;

impl RewritePattern for LowerEffectExtendToNative {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(extend_op) = effect::Extend::from_op(ctx, op) else {
            return false;
        };

        let loc = ctx.op(op).location;
        let ptr_ty = core_ptr_type(ctx);
        let i32_ty = core_i32_type(ctx);

        let ability_id_op = ability::ability_id_const(ctx, loc, i32_ty, extend_op.ability_ref(ctx));
        let ability_id_val = ability_id_op.result(ctx);
        rewriter.insert_op(ability_id_op.op_ref());

        let tr_dispatch_ptr =
            core::unrealized_conversion_cast(ctx, loc, extend_op.tr_dispatch_fn(ctx), ptr_ty);
        rewriter.insert_op(tr_dispatch_ptr.op_ref());

        let handler_dispatch_ptr =
            core::unrealized_conversion_cast(ctx, loc, extend_op.handler_dispatch(ctx), ptr_ty);
        rewriter.insert_op(handler_dispatch_ptr.op_ref());

        let extend_call = func::call(
            ctx,
            loc,
            [
                extend_op.evidence(ctx),
                ability_id_val,
                extend_op.prompt_tag(ctx),
                tr_dispatch_ptr.result(ctx),
                handler_dispatch_ptr.result(ctx),
            ],
            ptr_ty,
            Symbol::new(evidence_abi::EXTEND),
        );
        let new_result = extend_call.result(ctx);
        rewriter.insert_op(extend_call.op_ref());
        rewriter.erase_op(vec![new_result]);
        true
    }
}

struct LowerEffectDispatchTailToNative;

impl RewritePattern for LowerEffectDispatchTailToNative {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(dispatch_op) = effect::DispatchTail::from_op(ctx, op) else {
            return false;
        };

        let loc = ctx.op(op).location;
        let ptr_ty = core_ptr_type(ctx);
        let i32_ty = core_i32_type(ctx);
        let anyref_ty = tribute_rt::anyref(ctx).as_type_ref();
        let closure_ty = crate::closure_lower::closure_struct_type_ref(ctx);
        let ability_ref = dispatch_op.ability_ref(ctx);

        let ability_id_op = ability::ability_id_const(ctx, loc, i32_ty, ability_ref);
        let ability_id_val = ability_id_op.result(ctx);
        rewriter.insert_op(ability_id_op.op_ref());

        let dispatch_closure = func::call(
            ctx,
            loc,
            [dispatch_op.evidence(ctx), ability_id_val],
            ptr_ty,
            Symbol::new(evidence_abi::LOOKUP_TR),
        );
        let dispatch_val = dispatch_closure.result(ctx);
        rewriter.insert_op(dispatch_closure.op_ref());

        let op_idx_op = op_idx_const(ctx, loc, i32_ty, ability_ref, dispatch_op.op_name(ctx));
        let op_idx_val = op_idx_op.result(ctx);
        rewriter.insert_op(op_idx_op.op_ref());

        let fn_ptr_get = adt::struct_get(ctx, loc, dispatch_val, i32_ty, closure_ty, 0);
        let fn_ptr = fn_ptr_get.result(ctx);
        rewriter.insert_op(fn_ptr_get.op_ref());

        let env_get = adt::struct_get(ctx, loc, dispatch_val, anyref_ty, closure_ty, 1);
        let env_val = env_get.result(ctx);
        rewriter.insert_op(env_get.op_ref());

        let result_ty = ctx.op_result_types(op)[0];
        let call = func::call_indirect(
            ctx,
            loc,
            fn_ptr,
            [
                dispatch_op.evidence(ctx),
                env_val,
                op_idx_val,
                dispatch_op.payload(ctx),
            ],
            result_ty,
        );
        let new_result = call.result(ctx);
        rewriter.insert_op(call.op_ref());
        rewriter.erase_op(vec![new_result]);
        true
    }
}

struct LowerEffectDispatchCpsToNative;

impl RewritePattern for LowerEffectDispatchCpsToNative {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(dispatch_op) = effect::DispatchCps::from_op(ctx, op) else {
            return false;
        };

        let loc = ctx.op(op).location;
        let ptr_ty = core_ptr_type(ctx);
        let i32_ty = core_i32_type(ctx);
        let anyref_ty = tribute_rt::anyref(ctx).as_type_ref();
        let closure_ty = crate::closure_lower::closure_struct_type_ref(ctx);
        let ability_ref = dispatch_op.ability_ref(ctx);

        let ability_id_op = ability::ability_id_const(ctx, loc, i32_ty, ability_ref);
        let ability_id_val = ability_id_op.result(ctx);
        rewriter.insert_op(ability_id_op.op_ref());

        let dispatch_closure = func::call(
            ctx,
            loc,
            [dispatch_op.evidence(ctx), ability_id_val],
            ptr_ty,
            Symbol::new(evidence_abi::LOOKUP_HANDLER),
        );
        let dispatch_val = dispatch_closure.result(ctx);
        rewriter.insert_op(dispatch_closure.op_ref());

        let op_idx_op = op_idx_const(ctx, loc, i32_ty, ability_ref, dispatch_op.op_name(ctx));
        let op_idx_val = op_idx_op.result(ctx);
        rewriter.insert_op(op_idx_op.op_ref());

        let fn_ptr_get = adt::struct_get(ctx, loc, dispatch_val, i32_ty, closure_ty, 0);
        let fn_ptr = fn_ptr_get.result(ctx);
        rewriter.insert_op(fn_ptr_get.op_ref());

        let env_get = adt::struct_get(ctx, loc, dispatch_val, anyref_ty, closure_ty, 1);
        let env_val = env_get.result(ctx);
        rewriter.insert_op(env_get.op_ref());

        let result_ty = ctx.op_result_types(op)[0];
        let call = func::call_indirect(
            ctx,
            loc,
            fn_ptr,
            [
                dispatch_op.evidence(ctx),
                env_val,
                dispatch_op.continuation(ctx),
                op_idx_val,
                dispatch_op.payload(ctx),
            ],
            result_ty,
        );
        let new_result = call.result(ctx);
        rewriter.insert_op(call.op_ref());
        rewriter.erase_op(vec![new_result]);
        true
    }
}

fn rewrite_evidence_ops_in_block(ctx: &mut IrContext, block: BlockRef) -> PassRunResult {
    let ptr_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("ptr")).build());
    let i32_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());

    // Track Marker struct_new results → their operands
    let mut marker_struct_operands: HashMap<ValueRef, Vec<ValueRef>> = HashMap::new();
    // Track evidence_lookup results for struct_get elimination.
    // Maps result value → (ev, ability_id) operands for __tribute_evidence_lookup_tr calls.
    let mut evidence_lookup_results: HashMap<ValueRef, (ValueRef, ValueRef)> = HashMap::new();
    // Ops to erase after processing
    let mut ops_to_erase: Vec<OpRef> = Vec::new();

    let ops: Vec<OpRef> = ctx.block(block).ops.to_vec();

    for op in ops {
        let op_data = ctx.op(op);
        let dialect = op_data.dialect;
        let name = op_data.name;
        let loc = op_data.location;

        // --- adt.ref_null with evidence type → func.call @__tribute_evidence_empty ---
        // Closure lowering creates `adt.ref_null {type = evidence}` for null evidence.
        // Without this, the null ptr gets unboxed via `clif.load` which dereferences null.
        if dialect == Symbol::new("adt") && name == Symbol::new("ref_null") {
            let result_types = ctx.op_result_types(op).to_vec();
            if !result_types.is_empty() && is_evidence_type(ctx, result_types[0]) {
                let old_result = ctx.op_result(op, 0);
                let call = func::call(ctx, loc, [], ptr_ty, Symbol::new(evidence_abi::EMPTY));
                let new_result = call.result(ctx);
                ctx.insert_op_before(block, op, call.op_ref());
                ctx.replace_all_uses(old_result, new_result);
                ops_to_erase.push(op);
                continue;
            }
        }

        // --- adt.array_new with evidence type → func.call @__tribute_evidence_empty ---
        if dialect == Symbol::new("adt") && name == Symbol::new("array_new") {
            let result_types = ctx.op_result_types(op).to_vec();
            if !result_types.is_empty() && is_evidence_type(ctx, result_types[0]) {
                // Evidence array_new must represent an empty evidence vector.
                // The only operand should be the size hint (arith.const 0).
                let operand_count = ctx.op_operands(op).len();
                if operand_count > 1 {
                    return Err(native_evidence_rewrite_error(
                        op,
                        loc,
                        format!(
                            "adt.array_new with evidence type has {operand_count} operands; \
                             expected at most 1 (the size hint). Non-empty evidence arrays \
                             should not reach this pass."
                        ),
                    ));
                }
                let old_result = ctx.op_result(op, 0);
                let call = func::call(ctx, loc, [], ptr_ty, Symbol::new(evidence_abi::EMPTY));
                let new_result = call.result(ctx);
                ctx.insert_op_before(block, op, call.op_ref());
                ctx.replace_all_uses(old_result, new_result);
                ops_to_erase.push(op);
                continue;
            }
        }

        // --- Track adt.struct_new that produces a Marker ---
        if dialect == Symbol::new("adt") && name == Symbol::new("struct_new") {
            let result_types = ctx.op_result_types(op).to_vec();
            if !result_types.is_empty() && is_marker_type(ctx, result_types[0]) {
                let operands: Vec<ValueRef> = ctx.op_operands(op).to_vec();
                let result_val = ctx.op_result(op, 0);
                marker_struct_operands.insert(result_val, operands);
                ops_to_erase.push(op);
                continue;
            }
        }

        // --- Rewrite func.call @__tribute_evidence_lookup → returns i32 ---
        if dialect == Symbol::new("func")
            && name == Symbol::new("call")
            && let Ok(call_op) = func::Call::from_op(ctx, op)
        {
            let callee = call_op.callee(ctx);

            if callee == Symbol::new(evidence_abi::LOOKUP) {
                let operands: Vec<ValueRef> = ctx.op_operands(op).to_vec();
                if operands.len() != 2 {
                    return Err(native_evidence_rewrite_error(
                        op,
                        loc,
                        format!(
                            "{} expects 2 operands, got {}",
                            evidence_abi::LOOKUP,
                            operands.len()
                        ),
                    ));
                }
                let ev_val = operands[0];
                let ability_id_val = operands[1];
                let old_result = ctx.op_result(op, 0);
                let new_call = func::call(
                    ctx,
                    loc,
                    operands,
                    i32_ty,
                    Symbol::new(evidence_abi::LOOKUP),
                );
                let new_result = new_call.result(ctx);
                ctx.insert_op_before(block, op, new_call.op_ref());
                evidence_lookup_results.insert(new_result, (ev_val, ability_id_val));
                ctx.replace_all_uses(old_result, new_result);
                ops_to_erase.push(op);
                continue;
            }

            // --- Rewrite func.call @__tribute_evidence_extend(ev, marker) → native ABI args ---
            if callee == Symbol::new(evidence_abi::EXTEND) {
                let operands: Vec<ValueRef> = ctx.op_operands(op).to_vec();
                if operands.len() != 2 {
                    continue;
                }

                let ev_val = operands[0];
                let marker_val = operands[1];

                let Some(fields) = marker_struct_operands.get(&marker_val) else {
                    return Err(native_evidence_rewrite_error(
                        op,
                        loc,
                        format!(
                            "missing marker decomposition for marker_val={marker_val:?}; \
                             the adt.struct_new that produced this marker was not recorded"
                        ),
                    ));
                };
                if fields.len() != tribute_ir::dialect::ability::MARKER_FIELD_COUNT {
                    return Err(native_evidence_rewrite_error(
                        op,
                        loc,
                        format!(
                            "marker operand count must match canonical layout; expected {}, got {}",
                            tribute_ir::dialect::ability::MARKER_FIELD_COUNT,
                            fields.len()
                        ),
                    ));
                }
                let mut args = vec![ev_val];
                args.extend_from_slice(fields);
                let old_result = ctx.op_result(op, 0);
                let new_call =
                    func::call(ctx, loc, args, ptr_ty, Symbol::new(evidence_abi::EXTEND));
                let new_result = new_call.result(ctx);
                ctx.insert_op_before(block, op, new_call.op_ref());
                ctx.replace_all_uses(old_result, new_result);
                ops_to_erase.push(op);
                continue;
            }
        }

        // --- Eliminate adt.struct_get on evidence_lookup results ---
        if dialect == Symbol::new("adt") && name == Symbol::new("struct_get") {
            let operands: Vec<ValueRef> = ctx.op_operands(op).to_vec();
            if !operands.is_empty() {
                let base_val = operands[0];
                if let Some(&(ev_val, ability_id_val)) = evidence_lookup_results.get(&base_val) {
                    let field_attr = ctx.op(op).attributes.get("field");
                    let field_idx = match field_attr {
                        Some(Attribute::Int(bits)) => *bits,
                        other => {
                            return Err(native_evidence_rewrite_error(
                                op,
                                loc,
                                format!(
                                    "expected Int field attribute on adt.struct_get, got {other:?}"
                                ),
                            ));
                        }
                    };
                    match field_idx {
                        field if field == i128::from(MarkerField::PromptTag.index()) => {
                            // prompt_tag — __tribute_evidence_lookup already returns this
                            let old_result = ctx.op_result(op, 0);
                            ctx.replace_all_uses(old_result, base_val);
                            evidence_lookup_results.insert(old_result, (ev_val, ability_id_val));
                            ops_to_erase.push(op);
                        }
                        field if field == i128::from(MarkerField::TrDispatchFn.index()) => {
                            // tr_dispatch_fn — call __tribute_evidence_lookup_tr
                            let old_result = ctx.op_result(op, 0);
                            let tr_call = func::call(
                                ctx,
                                loc,
                                [ev_val, ability_id_val],
                                ptr_ty,
                                Symbol::new(evidence_abi::LOOKUP_TR),
                            );
                            let new_result = tr_call.result(ctx);
                            ctx.insert_op_before(block, op, tr_call.op_ref());
                            ctx.replace_all_uses(old_result, new_result);
                            ops_to_erase.push(op);
                        }
                        field if field == i128::from(MarkerField::HandlerDispatch.index()) => {
                            // handler_dispatch — call __tribute_evidence_lookup_handler
                            let old_result = ctx.op_result(op, 0);
                            let handler_call = func::call(
                                ctx,
                                loc,
                                [ev_val, ability_id_val],
                                ptr_ty,
                                Symbol::new(evidence_abi::LOOKUP_HANDLER),
                            );
                            let new_result = handler_call.result(ctx);
                            ctx.insert_op_before(block, op, handler_call.op_ref());
                            ctx.replace_all_uses(old_result, new_result);
                            ops_to_erase.push(op);
                        }
                        _ => {
                            return Err(native_evidence_rewrite_error(
                                op,
                                loc,
                                format!(
                                    "unexpected struct_get field {field_idx} on evidence_lookup \
                                     result"
                                ),
                            ));
                        }
                    }
                    continue;
                }
            }
        }

        // --- Recurse into nested regions, but leave nested functions for their
        // own function-scoped pass invocation.
        if func::Func::from_op(ctx, op).is_ok() {
            continue;
        }
        let regions: Vec<RegionRef> = ctx.op(op).regions.to_vec();
        for region in regions {
            rewrite_evidence_ops_in_region(ctx, region)?;
        }
    }

    // Erase dead ops (in reverse to handle dependencies)
    for op in ops_to_erase.into_iter().rev() {
        erase_op(ctx, op);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ops::ControlFlow;
    use trunk_ir::Span;
    use trunk_ir::context::{BlockArgData, BlockData, RegionData};
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;
    use trunk_ir::smallvec::smallvec;
    use trunk_ir::walk::{WalkAction, walk_op};

    fn dispatch_module() -> &'static str {
        r#"core.module @test {
  func.func @selected(%ev: core.ptr, %payload: tribute_rt.anyref) -> core.ptr {
    %result = effect.dispatch_tail %ev, %payload {ability_ref = core.ability_ref() {name = @Console}, op_name = @read} : core.ptr
    func.return %result
  }
  func.func @untouched(%ev: core.ptr, %payload: tribute_rt.anyref) -> core.ptr {
    %result = effect.dispatch_tail %ev, %payload {ability_ref = core.ability_ref() {name = @Console}, op_name = @print} : core.ptr
    func.return %result
  }
}"#
    }

    fn func_by_name_recursive(ctx: &IrContext, module: Module, name: &'static str) -> func::Func {
        let name = Symbol::new(name);
        let mut found = None;
        let _ = walk_op::<()>(ctx, module.op(), &mut |op| {
            if let Ok(func_op) = func::Func::from_op(ctx, op)
                && func_op.sym_name(ctx) == name
            {
                found = Some(func_op);
                return ControlFlow::Break(());
            }
            ControlFlow::Continue(WalkAction::Advance)
        });
        found.expect("test function should exist")
    }

    fn call_indirect_operands_in_func(ctx: &IrContext, func_op: func::Func) -> Vec<Vec<ValueRef>> {
        let mut calls = Vec::new();
        for &block in &ctx.region(func_op.body(ctx)).blocks {
            for &op in &ctx.block(block).ops {
                if func::CallIndirect::from_op(ctx, op).is_ok() {
                    calls.push(ctx.op_operands(op).to_vec());
                }
            }
        }
        calls
    }

    fn entry_arg(ctx: &IrContext, func_op: func::Func, index: usize) -> ValueRef {
        let entry = ctx.region(func_op.body(ctx)).blocks[0];
        ctx.block_args(entry)[index]
    }

    fn test_ctx() -> (IrContext, Location) {
        let mut ctx = IrContext::new();
        let path = ctx.paths.intern("file:///test.trb".to_owned());
        let loc = Location::new(path, Span::new(0, 0));
        (ctx, loc)
    }

    #[test]
    fn function_scope_rewrites_only_selected_function() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, dispatch_module());
        let selected = module
            .ops(&ctx)
            .into_iter()
            .filter_map(|op| func::Func::from_op(&ctx, op).ok())
            .next()
            .expect("test module should contain a selected function");

        lower_evidence_to_native_func(&mut ctx, selected);

        let ir_text = print_module(&ctx, module.op());
        assert_eq!(ir_text.matches("effect.dispatch_tail").count(), 1);
        assert!(ir_text.contains("func.func @untouched"));
        assert!(ir_text.contains("op_name = @print"));
        assert!(ir_text.contains("__tribute_evidence_lookup_tr"));
    }

    #[test]
    fn module_entrypoint_prepares_runtime_and_rewrites_all_functions() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, dispatch_module());

        lower_evidence_to_native(&mut ctx, module);

        let ir_text = print_module(&ctx, module.op());
        assert!(!ir_text.contains("effect.dispatch_tail"));
        assert!(ir_text.contains("__tribute_evidence_empty"));
        assert!(ir_text.contains("__tribute_evidence_lookup_tr"));
        assert!(ir_text.contains("__tribute_evidence_lookup_handler"));
    }

    #[test]
    fn pass_adapter_runs_function_lowering() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, dispatch_module());
        let selected = module
            .ops(&ctx)
            .into_iter()
            .filter_map(|op| func::Func::from_op(&ctx, op).ok())
            .next()
            .expect("test module should contain a selected function");
        let mut pass = LowerEvidenceToNative;

        assert_eq!(pass.name(), "lower-evidence-to-native");
        pass.run(&mut ctx, selected).unwrap();

        let ir_text = print_module(&ctx, module.op());
        assert_eq!(ir_text.matches("effect.dispatch_tail").count(), 1);
        assert!(ir_text.contains("__tribute_evidence_lookup"));
    }

    #[test]
    fn pass_adapter_reports_malformed_legacy_extend() {
        let (mut ctx, loc) = test_ctx();
        let evidence_ty = ability::evidence_adt_type_ref(&mut ctx);
        let marker_ty = ability::marker_adt_type_ref(&mut ctx);
        let func_ty = core::func(&mut ctx, evidence_ty, [evidence_ty, marker_ty]).as_type_ref();

        let entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![
                BlockArgData {
                    ty: evidence_ty,
                    attrs: Default::default(),
                },
                BlockArgData {
                    ty: marker_ty,
                    attrs: Default::default(),
                },
            ],
            ops: smallvec![],
            parent_region: None,
        });
        let ev = ctx.block_arg(entry, 0);
        let marker = ctx.block_arg(entry, 1);
        let extend_call = func::call(
            &mut ctx,
            loc,
            [ev, marker],
            evidence_ty,
            Symbol::new(evidence_abi::EXTEND),
        );
        let extend_result = extend_call.result(&ctx);
        let ret = func::r#return(&mut ctx, loc, [extend_result]);
        ctx.push_op(entry, extend_call.op_ref());
        ctx.push_op(entry, ret.op_ref());

        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });
        let func_op = func::func(&mut ctx, loc, Symbol::new("malformed"), func_ty, body);

        let mut pass = LowerEvidenceToNative;
        let err = pass
            .run(&mut ctx, func_op)
            .expect_err("malformed legacy evidence extend should report a pass error");

        assert!(err.to_string().contains("missing marker decomposition"));
    }

    #[test]
    fn function_scope_skips_evidence_runtime_functions() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @__tribute_evidence_empty(%ev: core.ptr, %payload: tribute_rt.anyref) -> core.ptr {
    %result = effect.dispatch_tail %ev, %payload {ability_ref = core.ability_ref() {name = @Console}, op_name = @read} : core.ptr
    func.return %result
  }
}"#,
        );
        let runtime_func = module
            .ops(&ctx)
            .into_iter()
            .filter_map(|op| func::Func::from_op(&ctx, op).ok())
            .next()
            .expect("test module should contain a runtime function");

        lower_evidence_to_native_func(&mut ctx, runtime_func);

        let ir_text = print_module(&ctx, module.op());
        assert!(ir_text.contains("func.func @__tribute_evidence_empty"));
        assert!(ir_text.contains("effect.dispatch_tail"));
    }

    #[test]
    fn function_scope_leaves_nested_func_for_own_lowering() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @outer(%outer_ev: core.ptr, %payload: tribute_rt.anyref) -> tribute_rt.anyref {
    func.func @inner(%inner_ev: core.ptr, %inner_payload: tribute_rt.anyref) -> core.ptr {
      %result = effect.dispatch_tail %inner_ev, %inner_payload {ability_ref = core.ability_ref() {name = @Console}, op_name = @read} : core.ptr
      func.return %result
    }
    func.return %payload
  }
}"#,
        );
        let outer = func_by_name_recursive(&ctx, module, "outer");

        lower_evidence_to_native_func(&mut ctx, outer);

        let inner = func_by_name_recursive(&ctx, module, "inner");
        let inner_after_outer = print_module(&ctx, inner.op_ref());
        assert!(
            inner_after_outer.contains("effect.dispatch_tail"),
            "outer function pass should not lower nested function body:\n{inner_after_outer}"
        );

        lower_evidence_to_native_func(&mut ctx, inner);

        let inner_calls = call_indirect_operands_in_func(&ctx, inner);
        assert_eq!(inner_calls.len(), 1);
        assert_eq!(
            inner_calls[0][1],
            entry_arg(&ctx, inner, 0),
            "nested native evidence lowering should pass the nested function's own evidence"
        );
    }
}
