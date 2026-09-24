//! Evidence runtime functions to WASM lowering (arena-based).
//!
//! This pass generates the evidence helpers required by target effect operations:
//!
//! - `__tribute_evidence_lookup(ev, ability_id)` -> binary search for marker
//! - `__tribute_evidence_extend(ev, marker)` -> sorted insertion with binary search
//!
//! ## Evidence Structure
//!
//! Evidence is represented as a WasmGC array of Marker structs, sorted by ability_id:
//!
//! ```text
//! Evidence = Array(Marker)
//! Marker = struct { ability_id: i32, prompt_tag: i32, tr_dispatch_fn: anyref, handler_dispatch: anyref }
//! ```
//!
//! ## Implementation Strategy
//!
//! The pass generates helpers on demand and binds existing runtime declarations
//! to implementations that use binary search (O(log n)). The evidence array is
//! maintained in sorted order by ability_id.

use tribute_core::{CallingConvention, set_calling_convention};
use tribute_ir::dialect::ability::{self as ability, MarkerField, evidence_abi};
use tribute_ir::dialect::effect;
use trunk_ir::Symbol;
use trunk_ir::context::{BlockArgData, BlockData, IrContext, RegionData};
use trunk_ir::dialect::wasm as wasm_dialect;
use trunk_ir::ops::DialectOp;
use trunk_ir::ops::DialectType;
use trunk_ir::pass::{Pass, PassRunResult};
use trunk_ir::refs::{OpRef, RegionRef, TypeRef, ValueRef};
use trunk_ir::rewrite::{Module, PatternApplicator, PatternRewriter, RewritePattern, RewriteScope};
use trunk_ir::smallvec::smallvec;
use trunk_ir::types::{Location, TypeDataBuilder};
use trunk_ir_wasm_backend::gc_types::{CLOSURE_STRUCT_IDX, EVIDENCE_IDX, MARKER_IDX};

#[derive(Debug, PartialEq, Eq)]
pub enum EvidenceValidationError {
    InvalidDispatchMetadata,
    DispatchOperandMismatch,
    InvalidTailDispatch,
}

impl std::fmt::Display for EvidenceValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidTailDispatch => {
                f.write_str("Wasm tail dispatch differs from its fixed evidence/payload/result ABI")
            }
            Self::InvalidDispatchMetadata => f.write_str(
                "Wasm CPS dispatch requires four operands, no results and typed metadata",
            ),
            Self::DispatchOperandMismatch => {
                f.write_str("Wasm CPS dispatch operands differ from the fixed target ABI")
            }
        }
    }
}

impl std::error::Error for EvidenceValidationError {}

/// Lower evidence runtime functions to WASM implementations.
///
/// This pass:
/// 1. Replaces stub function declarations with real binary search implementations
/// 2. Lowers `effect.extend` and dispatch operations using the generated functions.
pub fn lower_evidence_to_wasm(
    ctx: &mut IrContext,
    module: Module,
) -> Result<(), EvidenceValidationError> {
    prepare_wasm_evidence_runtime(ctx, module)?;
    rewrite_evidence_ops_in_scope(ctx, module);
    Ok(())
}

/// Prepare module-scope WASM evidence runtime helper functions.
pub fn prepare_wasm_evidence_runtime(
    ctx: &mut IrContext,
    module: Module,
) -> Result<(), EvidenceValidationError> {
    validate_final_dispatches(ctx, module.op())?;
    materialize_evidence_runtime(ctx, module);
    Ok(())
}

/// Lower evidence operations in one WASM function body.
///
/// Precondition: [`prepare_wasm_evidence_runtime`] must already have run for
/// the containing module so the `__tribute_evidence_lookup` and
/// `__tribute_evidence_extend` implementations exist as WASM runtime helpers.
pub fn lower_evidence_to_wasm_func(
    ctx: &mut IrContext,
    func_op: wasm_dialect::Func,
) -> Result<(), EvidenceValidationError> {
    validate_final_dispatches(ctx, func_op.op_ref())?;
    rewrite_evidence_ops_in_scope(ctx, func_op);
    Ok(())
}

/// PassManager-friendly WASM evidence lowering pass.
///
/// This pass is function-scoped and does not prepare module-scope runtime
/// helpers. Run [`prepare_wasm_evidence_runtime`] on the module before adding
/// this pass to a `wasm.func` pipeline.
pub struct LowerEvidenceToWasm;

impl Pass for LowerEvidenceToWasm {
    type Target = wasm_dialect::Func;

    fn name(&self) -> &'static str {
        "lower-evidence-to-wasm"
    }

    fn run(&mut self, ctx: &mut IrContext, target: wasm_dialect::Func) -> PassRunResult {
        lower_evidence_to_wasm_func(ctx, target).map_err(Into::into)
    }
}

fn rewrite_evidence_ops_in_scope<S: RewriteScope>(ctx: &mut IrContext, scope: S) {
    // The operations created here are `wasm.*`, so they must declare target
    // types. Evidence lowering therefore runs with the shared WASM type
    // converter instead of an identity converter.
    let type_converter = crate::wasm::type_converter::wasm_type_converter(ctx);
    let applicator = PatternApplicator::new(type_converter)
        .add_pattern(EffectExtendPattern)
        .add_pattern(EffectDispatchTailPattern)
        .add_pattern(EffectDispatchCpsPattern);
    applicator.apply_partial(ctx, scope);
}

/// Generate required helpers and bind existing evidence runtime declarations.
fn materialize_evidence_runtime(ctx: &mut IrContext, module: Module) {
    let (needs_lookup, needs_extend) = evidence_helper_requirements(ctx, module);
    let ops = module.ops(ctx);
    let mut has_lookup = false;
    let mut has_extend = false;
    let location = ctx.op(module.op()).location;

    for op in ops {
        let data = ctx.op(op);
        let is_wasm_func = data.dialect == Symbol::new("wasm") && data.name == Symbol::new("func");
        let is_func_func = data.dialect == Symbol::new("func") && data.name == Symbol::new("func");

        if !is_wasm_func && !is_func_func {
            continue;
        }

        let sym_name = data.attributes.get_symbol("sym_name");

        if sym_name == Some(Symbol::new(evidence_abi::LOOKUP)) {
            has_lookup = true;
            let location = data.location;
            let new_op = generate_evidence_lookup_function(ctx, location);
            replace_module_op(ctx, module, op, new_op);
        } else if sym_name == Some(Symbol::new(evidence_abi::EXTEND)) {
            has_extend = true;
            let location = data.location;
            let new_op = generate_evidence_extend_function(ctx, location);
            replace_module_op(ctx, module, op, new_op);
        }
    }

    if needs_lookup && !has_lookup {
        let new_op = generate_evidence_lookup_function(ctx, location);
        prepend_module_op(ctx, module, new_op);
    }

    if needs_extend && !has_extend {
        let new_op = generate_evidence_extend_function(ctx, location);
        prepend_module_op(ctx, module, new_op);
    }
}

fn evidence_helper_requirements(ctx: &IrContext, module: Module) -> (bool, bool) {
    fn visit(ctx: &IrContext, region: RegionRef, lookup: &mut bool, extend: &mut bool) {
        for &block in &ctx.region(region).blocks {
            for &op in &ctx.block(block).ops {
                *lookup |= effect::DispatchTail::from_op(ctx, op).is_ok()
                    || effect::DispatchCps::from_op(ctx, op).is_ok();
                *extend |= effect::Extend::from_op(ctx, op).is_ok();
                for &nested in &ctx.op(op).regions {
                    visit(ctx, nested, lookup, extend);
                }
            }
        }
    }

    let mut lookup = false;
    let mut extend = false;
    if let Some(body) = module.body(ctx) {
        visit(ctx, body, &mut lookup, &mut extend);
    }
    (lookup, extend)
}

/// Replace a top-level module operation with a new one.
fn replace_module_op(ctx: &mut IrContext, module: Module, old_op: OpRef, new_op: OpRef) {
    let Some(first_block) = module.first_block(ctx) else {
        return;
    };

    // Insert new op before old, then remove old
    ctx.insert_op_before(first_block, old_op, new_op);
    ctx.remove_op_from_block(first_block, old_op);
    ctx.remove_op(old_op);
}

fn prepend_module_op(ctx: &mut IrContext, module: Module, op: OpRef) {
    let Some(first_block) = module.first_block(ctx) else {
        return;
    };

    if let Some(first_op) = ctx.block(first_block).ops.first().copied() {
        ctx.insert_op_before(first_block, first_op, op);
    } else {
        ctx.push_op(first_block, op);
    }
}

// =============================================================================
// Evidence Lookup Pattern
// =============================================================================

/// Pattern that matches `effect.extend` and replaces it with
/// `wasm.call @__tribute_evidence_extend`.
struct EffectExtendPattern;

impl RewritePattern for EffectExtendPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(extend_op) = effect::Extend::from_op(ctx, op) else {
            return false;
        };

        let Some(result_ty) = rewriter.result_type(ctx, op, 0) else {
            return false;
        };
        let loc = ctx.op(op).location;
        let call_result = insert_evidence_extend_call(
            ctx,
            loc,
            EvidenceExtendCall {
                evidence: extend_op.evidence(ctx),
                result_ty,
                ability_ref_ty: extend_op.ability_ref(ctx),
                prompt_tag: extend_op.prompt_tag(ctx),
                tr_dispatch_fn: extend_op.tr_dispatch_fn(ctx),
                handler_dispatch: extend_op.handler_dispatch(ctx),
            },
            rewriter,
        );

        rewriter.erase_op(vec![call_result]);
        true
    }

    fn name(&self) -> &'static str {
        "EffectExtendPattern"
    }
}

/// Pattern that matches `effect.dispatch_tail` and replaces it with evidence
/// lookup plus a wasm indirect call through the stored tail-dispatch closure.
struct EffectDispatchTailPattern;

impl RewritePattern for EffectDispatchTailPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(dispatch_op) = effect::DispatchTail::from_op(ctx, op) else {
            return false;
        };

        let Ok(signature) = tail_dispatch_signature(ctx, op) else {
            return false;
        };
        let loc = ctx.op(op).location;
        let Some(result_ty) = rewriter.result_type(ctx, op, 0) else {
            return false;
        };
        let ability_ref = dispatch_op.ability_ref(ctx);
        let dispatch_closure = insert_dispatch_closure_lookup(
            ctx,
            loc,
            dispatch_op.evidence(ctx),
            ability_ref,
            MarkerField::TrDispatchFn,
            rewriter,
        );
        let (table_idx, env) = insert_closure_parts(ctx, loc, dispatch_closure, rewriter);
        let op_idx = insert_op_idx_const(ctx, loc, ability_ref, dispatch_op.op_name(ctx), rewriter);

        let call = wasm_dialect::CallIndirect::operands([
            table_idx,
            dispatch_op.evidence(ctx),
            env,
            op_idx,
            dispatch_op.payload(ctx),
        ])
        .type_idx(0)
        .table(0)
        .signature(Some(signature))
        .results([result_ty])
        .build(ctx, loc);
        let call_result = call.results(ctx)[0];
        rewriter.insert_op(call.op_ref());
        rewriter.erase_op(vec![call_result]);
        true
    }

    fn name(&self) -> &'static str {
        "EffectDispatchTailPattern"
    }
}

/// Pattern that matches `effect.dispatch_cps` and replaces it with evidence
/// lookup plus a wasm indirect call through the stored CPS dispatch closure.
struct EffectDispatchCpsPattern;

fn validate_final_dispatches(
    ctx: &mut IrContext,
    root: OpRef,
) -> Result<(), EvidenceValidationError> {
    let mut dispatches = Vec::new();
    let _ = trunk_ir::walk::walk_op::<()>(ctx, root, &mut |op| {
        if effect::DispatchCps::matches(ctx, op) || effect::DispatchTail::matches(ctx, op) {
            dispatches.push(op);
        }
        std::ops::ControlFlow::Continue(trunk_ir::walk::WalkAction::Advance)
    });
    for op in dispatches {
        if effect::DispatchTail::matches(ctx, op) {
            tail_dispatch_signature(ctx, op)?;
        } else {
            final_dispatch_signature(ctx, op)?;
        }
    }
    Ok(())
}

fn tail_dispatch_signature(
    ctx: &mut IrContext,
    op: OpRef,
) -> Result<TypeRef, EvidenceValidationError> {
    let evidence = evidence_ref_type(ctx);
    let anyref = wasm_dialect::anyref(ctx).as_type_ref();
    let i32_ty = intern_i32(ctx);
    // Extension results can still carry the exact shared Evidence type before
    // the pattern applicator converts their uses to the Wasm array ABI.
    let shared_evidence = ability::evidence_adt_type_ref(ctx);
    let [ev, payload] = ctx.op_operands(op) else {
        return Err(EvidenceValidationError::InvalidTailDispatch);
    };
    if (ctx.value_ty(*ev) != evidence && ctx.value_ty(*ev) != shared_evidence)
        || ctx.op_result_types(op) != [anyref]
        || ctx.op(op).attributes.get_type("ability_ref").is_none()
        || ctx.op(op).attributes.get_symbol("op_name").is_none()
        || !trunk_ir_wasm_backend::is_wasm_physical_argument_assignable(
            ctx,
            ctx.value_ty(*payload),
            anyref,
        )
    {
        return Err(EvidenceValidationError::InvalidTailDispatch);
    }
    Ok(intern_func_type(
        ctx,
        &[evidence, anyref, i32_ty, anyref],
        anyref,
    ))
}

fn final_dispatch_signature(
    ctx: &mut IrContext,
    op: OpRef,
) -> Result<TypeRef, EvidenceValidationError> {
    if !ctx.op_result_types(op).is_empty()
        || ctx.op(op).attributes.get_type("answer_type").is_none()
        || ctx.op(op).attributes.get_type("ability_ref").is_none()
        || ctx.op(op).attributes.get_symbol("op_name").is_none()
        || ctx.op_operands(op).len() != 4
    {
        return Err(EvidenceValidationError::InvalidDispatchMetadata);
    }
    let evidence_ty = evidence_ref_type(ctx);
    let anyref_ty = wasm_dialect::anyref(ctx).as_type_ref();
    let closure_ty = super::type_converter::closure_adt_type(ctx);
    let i32_ty = intern_i32(ctx);
    // The canonical operands keep their exact identity: another array spelling
    // is not the evidence array, and a plain `wasm.structref` is not the shared
    // `_closure` layout. Only the payload slot accepts the verified physical
    // widening, because its producer may leave a concrete reference after a
    // no-op `anyref` upcast.
    let operands = ctx.op_operands(op);
    let canonical_match = operands
        .iter()
        .zip([evidence_ty, closure_ty, closure_ty])
        .all(|(value, expected)| ctx.value_ty(*value) == expected);
    let payload_match = trunk_ir_wasm_backend::is_wasm_physical_argument_assignable(
        ctx,
        ctx.value_ty(operands[3]),
        anyref_ty,
    );
    if !canonical_match || !payload_match {
        return Err(EvidenceValidationError::DispatchOperandMismatch);
    }
    Ok(wasm_dialect::func_sig(
        ctx,
        [
            evidence_ty,
            anyref_ty,
            closure_ty,
            i32_ty,
            i32_ty,
            i32_ty,
            anyref_ty,
        ],
        [],
    )
    .as_type_ref())
}

impl RewritePattern for EffectDispatchCpsPattern {
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
        let Ok(signature) = final_dispatch_signature(ctx, op) else {
            return false;
        };
        let ability_ref = dispatch_op.ability_ref(ctx);
        let (table_idx, env) = insert_closure_parts(ctx, loc, dispatch_op.dispatch(ctx), rewriter);
        let i32_ty = intern_i32(ctx);
        let ability_id = wasm_dialect::I32Const::builder()
            .value(compute_ability_id(ctx, ability_ref))
            .results(i32_ty)
            .build(ctx, loc);
        let ability_id_value = ability_id.result(ctx);
        rewriter.insert_op(ability_id.op_ref());
        let marker_ty = ability::marker_adt_type_ref(ctx);
        let marker = wasm_dialect::Call::operands([dispatch_op.evidence(ctx), ability_id_value])
            .callee(Symbol::new(evidence_abi::LOOKUP))
            .results([marker_ty])
            .build(ctx, loc);
        let prompt = wasm_dialect::StructGet::operands(marker.results(ctx)[0])
            .type_idx(MARKER_IDX)
            .field_idx(MarkerField::PromptTag.index())
            .results(i32_ty)
            .build(ctx, loc);
        let prompt_value = prompt.result(ctx);
        rewriter.insert_op(marker.op_ref());
        rewriter.insert_op(prompt.op_ref());
        let op_idx = insert_op_idx_const(ctx, loc, ability_ref, dispatch_op.op_name(ctx), rewriter);

        let tail = wasm_dialect::ReturnCallIndirect::operands([
            table_idx,
            dispatch_op.evidence(ctx),
            env,
            dispatch_op.resume(ctx),
            prompt_value,
            ability_id_value,
            op_idx,
            dispatch_op.payload(ctx),
        ])
        .type_idx(0)
        .table(0)
        .signature(Some(signature))
        .build(ctx, loc);
        set_calling_convention(ctx, tail.op_ref(), CallingConvention::Cps);
        rewriter.replace_op(tail.op_ref());
        true
    }

    fn name(&self) -> &'static str {
        "EffectDispatchCpsPattern"
    }
}

fn insert_dispatch_closure_lookup(
    ctx: &mut IrContext,
    loc: Location,
    evidence: ValueRef,
    ability_ref_ty: TypeRef,
    field: MarkerField,
    rewriter: &mut PatternRewriter<'_>,
) -> ValueRef {
    let ability_id = compute_ability_id(ctx, ability_ref_ty);
    let i32_ty = intern_i32(ctx);
    let marker_ty = ability::marker_adt_type_ref(ctx);
    let closure_ty = crate::wasm::type_converter::closure_adt_type(ctx);

    let ability_id_const = wasm_dialect::I32Const::builder()
        .value(ability_id)
        .results(i32_ty)
        .build(ctx, loc);
    let lookup = wasm_dialect::Call::operands([evidence, ability_id_const.result(ctx)])
        .callee(Symbol::new(evidence_abi::LOOKUP))
        .results([marker_ty])
        .build(ctx, loc);
    let marker = lookup.results(ctx)[0];
    let closure_get = wasm_dialect::StructGet::operands(marker)
        .type_idx(MARKER_IDX)
        .field_idx(field.index())
        .results(closure_ty)
        .build(ctx, loc);
    let closure = closure_get.result(ctx);

    rewriter.insert_op(ability_id_const.op_ref());
    rewriter.insert_op(lookup.op_ref());
    rewriter.insert_op(closure_get.op_ref());
    closure
}

fn insert_closure_parts(
    ctx: &mut IrContext,
    loc: Location,
    closure: ValueRef,
    rewriter: &mut PatternRewriter<'_>,
) -> (ValueRef, ValueRef) {
    let i32_ty = intern_i32(ctx);
    let anyref_ty = trunk_ir::dialect::wasm::anyref(ctx).as_type_ref();

    let table_idx_get = wasm_dialect::StructGet::operands(closure)
        .type_idx(CLOSURE_STRUCT_IDX)
        .field_idx(0)
        .results(i32_ty)
        .build(ctx, loc);
    let table_idx = table_idx_get.result(ctx);
    let env_get = wasm_dialect::StructGet::operands(closure)
        .type_idx(CLOSURE_STRUCT_IDX)
        .field_idx(1)
        .results(anyref_ty)
        .build(ctx, loc);
    let env = env_get.result(ctx);

    rewriter.insert_op(table_idx_get.op_ref());
    rewriter.insert_op(env_get.op_ref());
    (table_idx, env)
}

fn insert_op_idx_const(
    ctx: &mut IrContext,
    loc: Location,
    ability_ref_ty: TypeRef,
    op_name: Symbol,
    rewriter: &mut PatternRewriter<'_>,
) -> ValueRef {
    let ability_name = ability::ability_name(ctx, ability_ref_ty);
    let op_idx = ability::compute_op_idx(ability_name, Some(op_name));
    let i32_ty = intern_i32(ctx);
    let op_idx_const = wasm_dialect::I32Const::builder()
        .value(op_idx as i32)
        .results(i32_ty)
        .build(ctx, loc);
    let op_idx = op_idx_const.result(ctx);
    rewriter.insert_op(op_idx_const.op_ref());
    op_idx
}

struct EvidenceExtendCall {
    evidence: ValueRef,
    result_ty: TypeRef,
    ability_ref_ty: TypeRef,
    prompt_tag: ValueRef,
    tr_dispatch_fn: ValueRef,
    handler_dispatch: ValueRef,
}

fn insert_evidence_extend_call(
    ctx: &mut IrContext,
    loc: Location,
    call: EvidenceExtendCall,
    rewriter: &mut PatternRewriter<'_>,
) -> ValueRef {
    let ability_id = compute_ability_id(ctx, call.ability_ref_ty);
    let i32_ty = intern_i32(ctx);
    let marker_ty = ability::marker_adt_type_ref(ctx);

    // Create: %ability_id = wasm.i32_const(ability_id)
    let ability_id_const = wasm_dialect::I32Const::builder()
        .value(ability_id)
        .results(i32_ty)
        .build(ctx, loc);

    // Create: %marker = wasm.struct_new(MARKER_IDX, %ability_id, %prompt_tag, %tr_dispatch_fn, %handler_dispatch)
    let marker_op = wasm_dialect::StructNew::operands([
        ability_id_const.result(ctx),
        call.prompt_tag,
        call.tr_dispatch_fn,
        call.handler_dispatch,
    ])
    .type_idx(MARKER_IDX)
    .results(marker_ty)
    .build(ctx, loc);

    // Create: %result = wasm.call @__tribute_evidence_extend(%ev, %marker)
    let call_op = wasm_dialect::Call::operands([call.evidence, marker_op.result(ctx)])
        .callee(Symbol::new(evidence_abi::EXTEND))
        .results([call.result_ty])
        .build(ctx, loc);

    let call_result = call_op.results(ctx)[0];
    rewriter.insert_op(ability_id_const.op_ref());
    rewriter.insert_op(marker_op.op_ref());
    rewriter.insert_op(call_op.op_ref());
    call_result
}

// =============================================================================
// Function Generation
// =============================================================================

/// Local variable indices for binary search.
/// These start after function parameters (0, 1).
mod locals {
    pub const LOW: u32 = 2;
    pub const HIGH: u32 = 3;
}

/// Generate the `__tribute_evidence_lookup` function implementation.
///
/// Uses binary search to find a marker with the given ability_id.
/// Returns the marker, or traps with unreachable if not found (compiler bug).
fn generate_evidence_lookup_function(ctx: &mut IrContext, location: Location) -> OpRef {
    let evidence_ty = evidence_ref_type(ctx);
    let i32_ty = intern_i32(ctx);
    let marker_sig_ty = ability::marker_adt_type_ref(ctx);

    let func_ty = intern_func_type(ctx, &[evidence_ty, i32_ty], marker_sig_ty);

    // Create the function body block with arguments
    let body_block = ctx.create_block(BlockData {
        location,
        args: vec![
            BlockArgData {
                ty: evidence_ty,
                attrs: Default::default(),
            },
            BlockArgData {
                ty: i32_ty,
                attrs: Default::default(),
            },
        ],
        ops: smallvec![],
        parent_region: None,
    });

    let ev_val = ctx.block_arg(body_block, 0);
    let target_id_val = ctx.block_arg(body_block, 1);

    // Initialize low = 0
    let zero = wasm_dialect::I32Const::builder()
        .value(0)
        .results(i32_ty)
        .build(ctx, location);
    ctx.push_op(body_block, zero.op_ref());
    let low_init = wasm_dialect::LocalSet::operands(zero.result(ctx))
        .index(locals::LOW)
        .build(ctx, location);
    ctx.push_op(body_block, low_init.op_ref());

    // Initialize high = array.len(ev)
    let len_op = wasm_dialect::ArrayLen::operands(ev_val)
        .results(i32_ty)
        .build(ctx, location);
    ctx.push_op(body_block, len_op.op_ref());
    let high_init = wasm_dialect::LocalSet::operands(len_op.result(ctx))
        .index(locals::HIGH)
        .build(ctx, location);
    ctx.push_op(body_block, high_init.op_ref());

    // Build the search loop
    let nil_ty = trunk_ir::dialect::core::nil(ctx).as_type_ref();
    let loop_region = build_lookup_loop_body(ctx, location, ev_val, target_id_val, i32_ty);
    let loop_op = wasm_dialect::Loop::operands([])
        .results([nil_ty])
        .regions(loop_region)
        .build(ctx, location);
    ctx.push_op(body_block, loop_op.op_ref());

    // unreachable after loop (should never reach here - loop always returns)
    let unreachable_op = wasm_dialect::unreachable(ctx, location);
    ctx.push_op(body_block, unreachable_op.op_ref());

    let body = ctx.create_region(RegionData {
        location,
        blocks: smallvec![body_block],
        parent_op: None,
    });

    let func_op = wasm_dialect::Func::builder()
        .sym_name(Symbol::new(evidence_abi::LOOKUP))
        .r#type(func_ty)
        .regions(body)
        .build(ctx, location);
    func_op.op_ref()
}

/// Build the loop body for binary search lookup.
fn build_lookup_loop_body(
    ctx: &mut IrContext,
    location: Location,
    ev_val: ValueRef,
    target_id_val: ValueRef,
    i32_ty: TypeRef,
) -> RegionRef {
    let nil_ty = trunk_ir::dialect::core::nil(ctx).as_type_ref();
    let marker_ty = ability::marker_adt_type_ref(ctx);

    let block = ctx.create_block(BlockData {
        location,
        args: vec![],
        ops: smallvec![],
        parent_region: None,
    });

    // low = local.get LOW
    let get_low = wasm_dialect::LocalGet::builder()
        .index(locals::LOW)
        .results(i32_ty)
        .build(ctx, location);
    let low = get_low.result(ctx);
    ctx.push_op(block, get_low.op_ref());

    // high = local.get HIGH
    let get_high = wasm_dialect::LocalGet::builder()
        .index(locals::HIGH)
        .results(i32_ty)
        .build(ctx, location);
    let high = get_high.result(ctx);
    ctx.push_op(block, get_high.op_ref());

    // Check low >= high -> unreachable (ability not found = compiler bug)
    let ge_check = wasm_dialect::I32GeS::operands(low, high)
        .results(i32_ty)
        .build(ctx, location);
    ctx.push_op(block, ge_check.op_ref());

    let unreachable_then = {
        let inner_block = ctx.create_block(BlockData {
            location,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let unreachable_op = wasm_dialect::unreachable(ctx, location);
        ctx.push_op(inner_block, unreachable_op.op_ref());
        ctx.create_region(RegionData {
            location,
            blocks: smallvec![inner_block],
            parent_op: None,
        })
    };
    let empty_else = {
        let inner_block = ctx.create_block(BlockData {
            location,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        ctx.create_region(RegionData {
            location,
            blocks: smallvec![inner_block],
            parent_op: None,
        })
    };
    let bound_check_if = wasm_dialect::If::operands(ge_check.result(ctx))
        .results([nil_ty])
        .regions(unreachable_then, empty_else)
        .build(ctx, location);
    ctx.push_op(block, bound_check_if.op_ref());

    // mid = (low + high) / 2
    let add_op = wasm_dialect::I32Add::operands(low, high).build(ctx, location);
    ctx.push_op(block, add_op.op_ref());
    let two = wasm_dialect::I32Const::builder()
        .value(2)
        .results(i32_ty)
        .build(ctx, location);
    ctx.push_op(block, two.op_ref());
    let mid_op = wasm_dialect::I32DivU::operands(add_op.result(ctx), two.result(ctx))
        .results(i32_ty)
        .build(ctx, location);
    let mid = mid_op.result(ctx);
    ctx.push_op(block, mid_op.op_ref());

    // marker = array.get(ev, mid)
    let marker_op = wasm_dialect::ArrayGet::operands(ev_val, mid)
        .type_idx(EVIDENCE_IDX)
        .results(marker_ty)
        .build(ctx, location);
    let marker = marker_op.result(ctx);
    ctx.push_op(block, marker_op.op_ref());

    // marker_ability_id = struct.get(marker, MarkerField::AbilityId)
    let marker_id_op = wasm_dialect::StructGet::operands(marker)
        .type_idx(MARKER_IDX)
        .field_idx(MarkerField::AbilityId.index())
        .results(i32_ty)
        .build(ctx, location);
    let marker_id = marker_id_op.result(ctx);
    ctx.push_op(block, marker_id_op.op_ref());

    // Check marker_ability_id == target -> return marker
    let eq_check = wasm_dialect::I32Eq::operands(marker_id, target_id_val)
        .results(i32_ty)
        .build(ctx, location);
    ctx.push_op(block, eq_check.op_ref());

    let found_then = {
        // Return marker
        let inner_block = ctx.create_block(BlockData {
            location,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let return_op = wasm_dialect::Return::operands([marker]).build(ctx, location);
        ctx.push_op(inner_block, return_op.op_ref());
        ctx.create_region(RegionData {
            location,
            blocks: smallvec![inner_block],
            parent_op: None,
        })
    };

    let continue_else = {
        // marker_ability_id < target ? low = mid + 1 : high = mid
        let inner_block = ctx.create_block(BlockData {
            location,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });

        let lt_check = wasm_dialect::I32LtS::operands(marker_id, target_id_val)
            .results(i32_ty)
            .build(ctx, location);
        ctx.push_op(inner_block, lt_check.op_ref());

        let update_low = {
            // low = mid + 1
            let ub = ctx.create_block(BlockData {
                location,
                args: vec![],
                ops: smallvec![],
                parent_region: None,
            });
            let one = wasm_dialect::I32Const::builder()
                .value(1)
                .results(i32_ty)
                .build(ctx, location);
            ctx.push_op(ub, one.op_ref());
            let add_one = wasm_dialect::I32Add::operands(mid, one.result(ctx)).build(ctx, location);
            ctx.push_op(ub, add_one.op_ref());
            let set_low = wasm_dialect::LocalSet::operands(add_one.result(ctx))
                .index(locals::LOW)
                .build(ctx, location);
            ctx.push_op(ub, set_low.op_ref());
            ctx.create_region(RegionData {
                location,
                blocks: smallvec![ub],
                parent_op: None,
            })
        };

        let update_high = {
            // high = mid
            let ub = ctx.create_block(BlockData {
                location,
                args: vec![],
                ops: smallvec![],
                parent_region: None,
            });
            let set_high = wasm_dialect::LocalSet::operands(mid)
                .index(locals::HIGH)
                .build(ctx, location);
            ctx.push_op(ub, set_high.op_ref());
            ctx.create_region(RegionData {
                location,
                blocks: smallvec![ub],
                parent_op: None,
            })
        };

        let update_if = wasm_dialect::If::operands(lt_check.result(ctx))
            .results([nil_ty])
            .regions(update_low, update_high)
            .build(ctx, location);
        ctx.push_op(inner_block, update_if.op_ref());

        // br $loop (target = 0, since loop is innermost)
        let br_loop = wasm_dialect::Br::builder().target(0).build(ctx, location);
        ctx.push_op(inner_block, br_loop.op_ref());

        ctx.create_region(RegionData {
            location,
            blocks: smallvec![inner_block],
            parent_op: None,
        })
    };

    let found_if = wasm_dialect::If::operands(eq_check.result(ctx))
        .results([nil_ty])
        .regions(found_then, continue_else)
        .build(ctx, location);
    ctx.push_op(block, found_if.op_ref());

    ctx.create_region(RegionData {
        location,
        blocks: smallvec![block],
        parent_op: None,
    })
}

/// Generate the `__tribute_evidence_extend` function implementation.
///
/// Uses binary search to find the insertion point, then creates a new array
/// with the marker inserted at the correct position to maintain sorted order.
fn generate_evidence_extend_function(ctx: &mut IrContext, location: Location) -> OpRef {
    let evidence_ty = evidence_ref_type(ctx);
    let i32_ty = intern_i32(ctx);
    let marker_sig_ty = ability::marker_adt_type_ref(ctx);

    let func_ty = intern_func_type(ctx, &[evidence_ty, marker_sig_ty], evidence_ty);

    // Create the function body block with arguments
    let body_block = ctx.create_block(BlockData {
        location,
        args: vec![
            BlockArgData {
                ty: evidence_ty,
                attrs: Default::default(),
            },
            BlockArgData {
                ty: marker_sig_ty,
                attrs: Default::default(),
            },
        ],
        ops: smallvec![],
        parent_region: None,
    });

    let ev_val = ctx.block_arg(body_block, 0);
    let marker_val = ctx.block_arg(body_block, 1);

    let nil_ty = trunk_ir::dialect::core::nil(ctx).as_type_ref();

    // Get marker's ability_id for binary search
    let marker_id_op = wasm_dialect::StructGet::operands(marker_val)
        .type_idx(MARKER_IDX)
        .field_idx(MarkerField::AbilityId.index())
        .results(i32_ty)
        .build(ctx, location);
    let marker_id = marker_id_op.result(ctx);
    ctx.push_op(body_block, marker_id_op.op_ref());

    // old_len = array.len(ev)
    let len_op = wasm_dialect::ArrayLen::operands(ev_val)
        .results(i32_ty)
        .build(ctx, location);
    let old_len = len_op.result(ctx);
    ctx.push_op(body_block, len_op.op_ref());

    // Initialize low = 0
    let zero = wasm_dialect::I32Const::builder()
        .value(0)
        .results(i32_ty)
        .build(ctx, location);
    ctx.push_op(body_block, zero.op_ref());
    let low_init = wasm_dialect::LocalSet::operands(zero.result(ctx))
        .index(locals::LOW)
        .build(ctx, location);
    ctx.push_op(body_block, low_init.op_ref());

    // Initialize high = old_len
    let high_init = wasm_dialect::LocalSet::operands(old_len)
        .index(locals::HIGH)
        .build(ctx, location);
    ctx.push_op(body_block, high_init.op_ref());

    // Binary search loop to find insertion point
    // After loop, LOW contains the insertion index
    let search_loop = build_extend_search_loop(ctx, location, ev_val, marker_id, i32_ty);
    let loop_op = wasm_dialect::Loop::operands([])
        .results([nil_ty])
        .regions(search_loop)
        .build(ctx, location);

    // Wrap loop in block for br_if(..., 1) target
    let loop_wrapper_block = ctx.create_block(BlockData {
        location,
        args: vec![],
        ops: smallvec![],
        parent_region: None,
    });
    ctx.push_op(loop_wrapper_block, loop_op.op_ref());
    let loop_region = ctx.create_region(RegionData {
        location,
        blocks: smallvec![loop_wrapper_block],
        parent_op: None,
    });
    let block_op = wasm_dialect::Block::builder()
        .results([nil_ty])
        .regions(loop_region)
        .build(ctx, location);
    ctx.push_op(body_block, block_op.op_ref());

    // insert_idx = local.get LOW
    let get_insert_idx = wasm_dialect::LocalGet::builder()
        .index(locals::LOW)
        .results(i32_ty)
        .build(ctx, location);
    let insert_idx = get_insert_idx.result(ctx);
    ctx.push_op(body_block, get_insert_idx.op_ref());

    // new_len = old_len + 1
    let one = wasm_dialect::I32Const::builder()
        .value(1)
        .results(i32_ty)
        .build(ctx, location);
    ctx.push_op(body_block, one.op_ref());
    let add_len_op = wasm_dialect::I32Add::operands(old_len, one.result(ctx)).build(ctx, location);
    let new_len = add_len_op.result(ctx);
    ctx.push_op(body_block, add_len_op.op_ref());

    // new_ev = array.new_default(new_len)
    let new_array_op = wasm_dialect::ArrayNewDefault::operands(new_len)
        .type_idx(EVIDENCE_IDX)
        .results(evidence_ty)
        .build(ctx, location);
    let new_ev = new_array_op.result(ctx);
    ctx.push_op(body_block, new_array_op.op_ref());

    // Copy elements before insertion point: array.copy(new_ev, 0, ev, 0, insert_idx)
    // Only if insert_idx > 0
    let zero2 = wasm_dialect::I32Const::builder()
        .value(0)
        .results(i32_ty)
        .build(ctx, location);
    ctx.push_op(body_block, zero2.op_ref());

    let gt_zero = wasm_dialect::I32GtS::operands(insert_idx, zero2.result(ctx))
        .results(i32_ty)
        .build(ctx, location);
    ctx.push_op(body_block, gt_zero.op_ref());

    let copy_prefix_then = {
        let inner_block = ctx.create_block(BlockData {
            location,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let zero3 = wasm_dialect::I32Const::builder()
            .value(0)
            .results(i32_ty)
            .build(ctx, location);
        ctx.push_op(inner_block, zero3.op_ref());
        let copy_op = wasm_dialect::ArrayCopy::operands(
            new_ev,
            zero3.result(ctx),
            ev_val,
            zero3.result(ctx),
            insert_idx,
        )
        .dst_type_idx(EVIDENCE_IDX)
        .src_type_idx(EVIDENCE_IDX)
        .build(ctx, location);
        ctx.push_op(inner_block, copy_op.op_ref());
        ctx.create_region(RegionData {
            location,
            blocks: smallvec![inner_block],
            parent_op: None,
        })
    };
    let empty_else1 = {
        let inner_block = ctx.create_block(BlockData {
            location,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        ctx.create_region(RegionData {
            location,
            blocks: smallvec![inner_block],
            parent_op: None,
        })
    };
    let copy_prefix_if = wasm_dialect::If::operands(gt_zero.result(ctx))
        .results([nil_ty])
        .regions(copy_prefix_then, empty_else1)
        .build(ctx, location);
    ctx.push_op(body_block, copy_prefix_if.op_ref());

    // Set marker at insert_idx: array.set(new_ev, insert_idx, marker)
    let set_op = wasm_dialect::ArraySet::operands(new_ev, insert_idx, marker_val)
        .type_idx(EVIDENCE_IDX)
        .build(ctx, location);
    ctx.push_op(body_block, set_op.op_ref());

    // Copy elements after insertion point: array.copy(new_ev, insert_idx+1, ev, insert_idx, old_len - insert_idx)
    // suffix_len = old_len - insert_idx
    let suffix_len_op = wasm_dialect::I32Sub::operands(old_len, insert_idx)
        .results(i32_ty)
        .build(ctx, location);
    let suffix_len = suffix_len_op.result(ctx);
    ctx.push_op(body_block, suffix_len_op.op_ref());

    // Only if suffix_len > 0
    let zero4 = wasm_dialect::I32Const::builder()
        .value(0)
        .results(i32_ty)
        .build(ctx, location);
    ctx.push_op(body_block, zero4.op_ref());
    let gt_zero2 = wasm_dialect::I32GtS::operands(suffix_len, zero4.result(ctx))
        .results(i32_ty)
        .build(ctx, location);
    ctx.push_op(body_block, gt_zero2.op_ref());

    let copy_suffix_then = {
        let inner_block = ctx.create_block(BlockData {
            location,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let one2 = wasm_dialect::I32Const::builder()
            .value(1)
            .results(i32_ty)
            .build(ctx, location);
        ctx.push_op(inner_block, one2.op_ref());
        let dst_offset_op =
            wasm_dialect::I32Add::operands(insert_idx, one2.result(ctx)).build(ctx, location);
        ctx.push_op(inner_block, dst_offset_op.op_ref());
        let copy_op = wasm_dialect::ArrayCopy::operands(
            new_ev,
            dst_offset_op.result(ctx),
            ev_val,
            insert_idx,
            suffix_len,
        )
        .dst_type_idx(EVIDENCE_IDX)
        .src_type_idx(EVIDENCE_IDX)
        .build(ctx, location);
        ctx.push_op(inner_block, copy_op.op_ref());
        ctx.create_region(RegionData {
            location,
            blocks: smallvec![inner_block],
            parent_op: None,
        })
    };
    let empty_else2 = {
        let inner_block = ctx.create_block(BlockData {
            location,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        ctx.create_region(RegionData {
            location,
            blocks: smallvec![inner_block],
            parent_op: None,
        })
    };
    let copy_suffix_if = wasm_dialect::If::operands(gt_zero2.result(ctx))
        .results([nil_ty])
        .regions(copy_suffix_then, empty_else2)
        .build(ctx, location);
    ctx.push_op(body_block, copy_suffix_if.op_ref());

    // Return new_ev
    let return_op = wasm_dialect::Return::operands([new_ev]).build(ctx, location);
    ctx.push_op(body_block, return_op.op_ref());

    let body = ctx.create_region(RegionData {
        location,
        blocks: smallvec![body_block],
        parent_op: None,
    });

    let func_op = wasm_dialect::Func::builder()
        .sym_name(Symbol::new(evidence_abi::EXTEND))
        .r#type(func_ty)
        .regions(body)
        .build(ctx, location);
    func_op.op_ref()
}

/// Build the search loop for finding insertion position in evidence_extend.
///
/// This is a binary search that finds the first index where ev[i].ability_id >= marker_id.
/// After the loop, LOW contains the insertion index.
fn build_extend_search_loop(
    ctx: &mut IrContext,
    location: Location,
    ev_val: ValueRef,
    marker_id: ValueRef,
    i32_ty: TypeRef,
) -> RegionRef {
    let nil_ty = trunk_ir::dialect::core::nil(ctx).as_type_ref();
    let marker_ty = ability::marker_adt_type_ref(ctx);

    let block = ctx.create_block(BlockData {
        location,
        args: vec![],
        ops: smallvec![],
        parent_region: None,
    });

    // low = local.get LOW
    let get_low = wasm_dialect::LocalGet::builder()
        .index(locals::LOW)
        .results(i32_ty)
        .build(ctx, location);
    let low = get_low.result(ctx);
    ctx.push_op(block, get_low.op_ref());

    // high = local.get HIGH
    let get_high = wasm_dialect::LocalGet::builder()
        .index(locals::HIGH)
        .results(i32_ty)
        .build(ctx, location);
    let high = get_high.result(ctx);
    ctx.push_op(block, get_high.op_ref());

    // if low >= high: break (insertion point found at LOW)
    let ge_check = wasm_dialect::I32GeS::operands(low, high)
        .results(i32_ty)
        .build(ctx, location);
    ctx.push_op(block, ge_check.op_ref());

    // br_if to exit loop (target = 1 to break out of loop to outer block)
    let br_if_done = wasm_dialect::BrIf::operands(ge_check.result(ctx))
        .target(1)
        .build(ctx, location);
    ctx.push_op(block, br_if_done.op_ref());

    // mid = (low + high) / 2
    let add_op = wasm_dialect::I32Add::operands(low, high).build(ctx, location);
    ctx.push_op(block, add_op.op_ref());
    let two = wasm_dialect::I32Const::builder()
        .value(2)
        .results(i32_ty)
        .build(ctx, location);
    ctx.push_op(block, two.op_ref());
    let mid_op = wasm_dialect::I32DivU::operands(add_op.result(ctx), two.result(ctx))
        .results(i32_ty)
        .build(ctx, location);
    let mid = mid_op.result(ctx);
    ctx.push_op(block, mid_op.op_ref());

    // mid_marker = array.get(ev, mid)
    let mid_marker_op = wasm_dialect::ArrayGet::operands(ev_val, mid)
        .type_idx(EVIDENCE_IDX)
        .results(marker_ty)
        .build(ctx, location);
    let mid_marker = mid_marker_op.result(ctx);
    ctx.push_op(block, mid_marker_op.op_ref());

    // mid_ability_id = struct.get(mid_marker, MarkerField::AbilityId)
    let mid_id_op = wasm_dialect::StructGet::operands(mid_marker)
        .type_idx(MARKER_IDX)
        .field_idx(MarkerField::AbilityId.index())
        .results(i32_ty)
        .build(ctx, location);
    let mid_id = mid_id_op.result(ctx);
    ctx.push_op(block, mid_id_op.op_ref());

    // if mid_ability_id < marker_id: low = mid + 1
    // else: high = mid
    let lt_check = wasm_dialect::I32LtS::operands(mid_id, marker_id)
        .results(i32_ty)
        .build(ctx, location);
    ctx.push_op(block, lt_check.op_ref());

    let update_low = {
        let inner_block = ctx.create_block(BlockData {
            location,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let one = wasm_dialect::I32Const::builder()
            .value(1)
            .results(i32_ty)
            .build(ctx, location);
        ctx.push_op(inner_block, one.op_ref());
        let add_one = wasm_dialect::I32Add::operands(mid, one.result(ctx)).build(ctx, location);
        ctx.push_op(inner_block, add_one.op_ref());
        let set_low = wasm_dialect::LocalSet::operands(add_one.result(ctx))
            .index(locals::LOW)
            .build(ctx, location);
        ctx.push_op(inner_block, set_low.op_ref());
        ctx.create_region(RegionData {
            location,
            blocks: smallvec![inner_block],
            parent_op: None,
        })
    };

    let update_high = {
        let inner_block = ctx.create_block(BlockData {
            location,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let set_high = wasm_dialect::LocalSet::operands(mid)
            .index(locals::HIGH)
            .build(ctx, location);
        ctx.push_op(inner_block, set_high.op_ref());
        ctx.create_region(RegionData {
            location,
            blocks: smallvec![inner_block],
            parent_op: None,
        })
    };

    let update_if = wasm_dialect::If::operands(lt_check.result(ctx))
        .results([nil_ty])
        .regions(update_low, update_high)
        .build(ctx, location);
    ctx.push_op(block, update_if.op_ref());

    // br $loop (continue searching)
    let br_loop = wasm_dialect::Br::builder().target(0).build(ctx, location);
    ctx.push_op(block, br_loop.op_ref());

    ctx.create_region(RegionData {
        location,
        blocks: smallvec![block],
        parent_op: None,
    })
}

// =============================================================================
// Helper functions
// =============================================================================

/// Get the WASM reference type for Evidence (wasm.arrayref).
fn evidence_ref_type(ctx: &mut IrContext) -> TypeRef {
    trunk_ir::dialect::wasm::arrayref(ctx).as_type_ref()
}

/// Intern a `core.i32` type.
fn intern_i32(ctx: &mut IrContext) -> TypeRef {
    ctx.intern_type(TypeDataBuilder::new("core", "i32").build())
}

/// Intern a target-owned `wasm.func_sig` type with the given parameter and return types.
fn intern_func_type(ctx: &mut IrContext, params: &[TypeRef], ret: TypeRef) -> TypeRef {
    wasm_dialect::func_sig(ctx, params.iter().copied(), [ret]).as_type_ref()
}

/// Compute a stable ability ID as a WASM i32 immediate.
fn compute_ability_id(ctx: &IrContext, ability_ty: TypeRef) -> i32 {
    ability_id_as_wasm_i32(ability::compute_ability_id(ctx, ability_ty))
}

fn ability_id_as_wasm_i32(ability_id: u32) -> i32 {
    i32::from_ne_bytes(ability_id.to_ne_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;

    fn lower_text(ir: &str) -> String {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, ir);
        lower_evidence_to_wasm(&mut ctx, module).unwrap();
        print_module(&ctx, module.op())
    }

    fn dispatch_module() -> &'static str {
        r#"core.module @test {
  func.func @selected(%ev: wasm.arrayref, %payload: wasm.anyref) -> wasm.anyref {
    %result = effect.dispatch_tail %ev, %payload {ability_ref = core.ability_ref() {name = @Console}, op_name = @read} : wasm.anyref
    func.return %result
  }
  func.func @untouched(%ev: wasm.arrayref, %payload: wasm.anyref) -> wasm.anyref {
    %result = effect.dispatch_tail %ev, %payload {ability_ref = core.ability_ref() {name = @Console}, op_name = @print} : wasm.anyref
    func.return %result
  }
}"#
    }

    fn lower_funcs_to_wasm(ctx: &mut IrContext, module: Module) {
        let tc = super::super::type_converter::wasm_type_converter(ctx);
        trunk_ir_wasm_backend::passes::func_to_wasm::lower(ctx, module, tc);
    }

    #[test]
    fn textual_dispatch_tail_lowers_to_wasm_indirect_call() {
        let output = lower_text(
            r#"core.module @test {
  func.func @run(%ev: wasm.arrayref, %payload: wasm.anyref) -> wasm.anyref {
    %result = effect.dispatch_tail %ev, %payload {ability_ref = core.ability_ref() {name = @Console}, op_name = @read} : wasm.anyref
    func.return %result
  }
}"#,
        );

        assert!(!output.contains("effect.dispatch_tail"), "{output}");
        assert!(output.contains("__tribute_evidence_lookup"), "{output}");
        assert!(output.contains("wasm.struct_get"), "{output}");
        assert!(output.contains("wasm.call_indirect"), "{output}");
    }

    #[test]
    fn tail_dispatch_preserves_fixed_signature_and_rejects_malformed_contracts() {
        let template = r#"core.module @test {
            func.func @run(%ev: wasm.arrayref, %payload: wasm.anyref) -> wasm.anyref {
                %result = effect.dispatch_tail %ev, %payload {ability_ref = core.ability_ref() {name = @Console}, op_name = @read} : wasm.anyref
                func.return %result
            }
        }"#;
        let output = lower_text(template);
        assert!(output.contains("signature = wasm.func_sig<(wasm.arrayref, wasm.anyref, core.i32, wasm.anyref) -> wasm.anyref>"), "{output}");
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, template);
        let function = module.ops(&ctx)[0];
        let block = ctx.region(ctx.op(function).regions[0]).blocks[0];
        let shared_evidence = ability::evidence_adt_type_ref(&mut ctx);
        ctx.set_block_arg_type(block, 0, shared_evidence);
        lower_evidence_to_wasm(&mut ctx, module)
            .expect("exact shared Evidence is accepted before target conversion");

        for source in [
            template.replace("%ev: wasm.arrayref", "%ev: core.array(core.i32)"),
            template.replace("%ev: wasm.arrayref", "%ev: wasm.anyref"),
            template.replace("%payload: wasm.anyref", "%payload: core.i32"),
            template
                .replace("} : wasm.anyref", "} : core.i32")
                .replace("-> wasm.anyref", "-> core.i32"),
        ] {
            let mut ctx = IrContext::new();
            let module = parse_test_module(&mut ctx, &source);
            let before = print_module(&ctx, module.op());
            assert_eq!(
                lower_evidence_to_wasm(&mut ctx, module),
                Err(EvidenceValidationError::InvalidTailDispatch)
            );
            assert_eq!(print_module(&ctx, module.op()), before);
        }
    }

    #[test]
    fn fixed_dispatch_signature_is_independent_of_answer_and_rejects_operand_mutations() {
        let source = r#"core.module @test {
          !closure = adt.struct() {fields = [[@table_idx, core.i32], [@env, wasm.anyref]], name = @_closure}
          func.func @run(%ev: wasm.arrayref, %dispatch: !closure, %resume: !closure, %payload: wasm.anyref) {
            effect.dispatch_cps %ev, %dispatch, %resume, %payload {ability_ref = core.ability_ref() {name = @State}, op_name = @get, answer_type = core.i32}
          }
        }"#;
        let first = lower_text(source);
        let second =
            lower_text(&source.replace("answer_type = core.i32", "answer_type = core.i64"));
        assert_eq!(first, second);
        assert!(first.contains("wasm.func_sig<(wasm.arrayref, wasm.anyref, !closure, core.i32, core.i32, core.i32, wasm.anyref) -> ()>"), "{first}");
        assert!(!first.contains("answer_type"));
        let mut invalid_inputs: Vec<_> = [
            source.replace("answer_type = core.i32", "answer_type = 0"),
            source.replace(", answer_type = core.i32", ""),
            source.replace(
                "%ev, %dispatch, %resume, %payload",
                "%ev, %dispatch, %resume",
            ),
        ]
        .into_iter()
        .map(|source| (source, EvidenceValidationError::InvalidDispatchMetadata))
        .collect();
        for original in [
            "%ev: wasm.arrayref",
            "%dispatch: !closure",
            "%resume: !closure",
            "%payload: wasm.anyref",
        ] {
            let changed = source.replace(
                original,
                &format!("{}: core.i64", original.split(':').next().unwrap()),
            );
            invalid_inputs.push((changed, EvidenceValidationError::DispatchOperandMismatch));
        }
        for (changed, expected) in invalid_inputs {
            for stub in [
                "",
                "func.func @__tribute_evidence_lookup(%ev: wasm.arrayref, %id: core.i32) -> core.i32 { func.unreachable }",
            ] {
                let source =
                    changed.replacen("func.func @run", &format!("{stub}\nfunc.func @run"), 1);
                let mut ctx = IrContext::new();
                let module = parse_test_module(&mut ctx, &source);
                let before = print_module(&ctx, module.op());
                let ops = module.ops(&ctx);
                assert_eq!(
                    prepare_wasm_evidence_runtime(&mut ctx, module).unwrap_err(),
                    expected
                );
                assert_eq!(
                    lower_evidence_to_wasm(&mut ctx, module).unwrap_err(),
                    expected
                );
                assert_eq!(print_module(&ctx, module.op()), before);
                assert_eq!(module.ops(&ctx), ops);
            }
        }
    }

    #[test]
    fn dispatch_payload_accepts_registered_references_but_canonical_operands_stay_exact() {
        let dispatch_source = |evidence_ty: &str, payload_ty: &str| {
            format!(
                r#"core.module @test {{
          !closure = adt.struct() {{fields = [[@table_idx, core.i32], [@env, wasm.anyref]], name = @_closure}}
          !Payload = adt.struct() {{fields = [[@value, core.i32]], name = @Payload}}
          !TagOnly = adt.enum() {{is_variant = true, variant_tag = @Leaf}}
          !Array = core.array(core.i32)
          func.func @run(%ev: {evidence_ty}, %dispatch: !closure, %resume: !closure, %payload: {payload_ty}) {{
            effect.dispatch_cps %ev, %dispatch, %resume, %payload {{ability_ref = core.ability_ref() {{name = @State}}, op_name = @get, answer_type = core.i32}}
          }}
        }}"#
            )
        };
        let fixed_signature = "wasm.func_sig<(wasm.arrayref, wasm.anyref, !closure, core.i32, core.i32, core.i32, wasm.anyref) -> ()>";

        for payload_ty in [
            "wasm.anyref",
            "adt.typeref",
            "!Payload",
            "core.bytes",
            "!Array",
            "wasm.i31ref",
            "wasm.arrayref",
        ] {
            let output = lower_text(&dispatch_source("wasm.arrayref", payload_ty));
            assert!(
                output.contains(fixed_signature),
                "payload {payload_ty}: {output}"
            );
        }

        for (evidence_ty, payload_ty) in [
            ("!Array", "wasm.anyref"),
            ("wasm.anyref", "wasm.anyref"),
            ("core.i64", "wasm.anyref"),
            ("wasm.arrayref", "core.i64"),
            ("wasm.arrayref", "wasm.funcref"),
            ("wasm.arrayref", "!TagOnly"),
        ] {
            let mut ctx = IrContext::new();
            let module = parse_test_module(&mut ctx, &dispatch_source(evidence_ty, payload_ty));
            assert_eq!(
                lower_evidence_to_wasm(&mut ctx, module).unwrap_err(),
                EvidenceValidationError::DispatchOperandMismatch,
                "evidence {evidence_ty}, payload {payload_ty}"
            );
        }
    }

    #[test]
    fn pass_adapter_preserves_concrete_validation_error() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
            !Closure = adt.struct() {name = @_closure, fields = [[@table_idx, core.i32], [@env, wasm.anyref]]}
            wasm.func @run(%ev: wasm.arrayref, %dispatch: !Closure, %resume: !Closure, %payload: core.i64) {
                effect.dispatch_cps %ev, %dispatch, %resume, %payload {ability_ref = core.ability_ref() {name = @State}, op_name = @get, answer_type = core.i32}
            }
        }"#,
        );
        let function = wasm_dialect::Func::from_op(&ctx, module.ops(&ctx)[0]).unwrap();
        let before = print_module(&ctx, module.op());
        let error = LowerEvidenceToWasm.run(&mut ctx, function).unwrap_err();
        assert_eq!(
            error.downcast_ref::<EvidenceValidationError>(),
            Some(&EvidenceValidationError::DispatchOperandMismatch)
        );
        assert_eq!(print_module(&ctx, module.op()), before);
    }

    #[test]
    fn textual_dispatch_cps_lowers_to_wasm_indirect_call() {
        let output = lower_text(
            r#"core.module @test {
          !closure = adt.struct() {fields = [[@table_idx, core.i32], [@env, wasm.anyref]], name = @_closure}
  func.func @run(%ev: wasm.arrayref, %dispatch: !closure, %resume: !closure, %payload: wasm.anyref) {
    effect.dispatch_cps %ev, %dispatch, %resume, %payload {ability_ref = core.ability_ref() {name = @State}, op_name = @get, answer_type = core.i32}
  }
}"#,
        );

        assert!(!output.contains("effect.dispatch_cps"), "{output}");
        assert!(output.contains("__tribute_evidence_lookup"), "{output}");
        assert!(
            output.contains("wasm.struct_get %1"),
            "the Wasm tail must decompose the explicit dispatch operand: {output}"
        );
        assert!(
            !output.contains("core.unrealized_conversion_cast"),
            "the Wasm tail must not cast the explicit dispatch operand: {output}"
        );

        assert!(output.contains("wasm.return_call_indirect"), "{output}");
        assert!(
            output.contains("tribute.calling_convention = 2"),
            "{output}"
        );
        assert!(output.contains("signature ="), "{output}");
        assert!(!output.contains("func.indirect_call_signature"), "{output}");
    }

    #[test]
    fn result_bearing_final_dispatch_remains_unchanged() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @run(%ev: wasm.arrayref, %continuation: wasm.anyref, %payload: wasm.anyref) -> wasm.anyref {
    %result = effect.dispatch_cps %ev, %continuation, %payload {ability_ref = core.ability_ref() {name = @State}, op_name = @get} : wasm.anyref
    func.return %result
  }
}"#,
        );
        let before = print_module(&ctx, module.op());

        rewrite_evidence_ops_in_scope(&mut ctx, module);

        assert_eq!(print_module(&ctx, module.op()), before);
    }

    #[test]
    fn textual_extend_lowers_to_wasm_evidence_extend_call() {
        let output = lower_text(
            r#"core.module @test {
  func.func @run(%ev: wasm.arrayref, %prompt: core.i32, %tr: wasm.anyref, %handler: wasm.anyref) -> wasm.arrayref {
    %result = effect.extend %ev, %prompt, %tr, %handler {ability_ref = core.ability_ref() {name = @State}} : wasm.arrayref
    func.return %result
  }
}"#,
        );

        assert!(!output.contains("effect.extend"), "{output}");
        assert!(output.contains("wasm.struct_new"), "{output}");
        assert!(output.contains("__tribute_evidence_extend"), "{output}");
    }

    #[test]
    fn extend_result_is_produced_as_a_target_type() {
        // `effect.extend` lowers to a `wasm.call`, so its declared result must be
        // the converted evidence reference even when the shared IR still spells
        // the evidence array logically.
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !Evidence = core.array(adt.struct() {name = @_Marker, fields = [[@ability_id, core.i32], [@prompt_tag, core.i32], [@tr_dispatch_fn, core.ptr], [@handler_dispatch, core.ptr]]})
  func.func @run(%ev: !Evidence, %prompt: core.i32, %tr: wasm.anyref, %handler: wasm.anyref) {
    %result = effect.extend %ev, %prompt, %tr, %handler {ability_ref = core.ability_ref() {name = @State}} : !Evidence
  }
}"#,
        );
        lower_evidence_to_wasm(&mut ctx, module).unwrap();

        let fixture = module
            .ops(&ctx)
            .into_iter()
            .find(|&op| {
                ctx.op(op).dialect == Symbol::new("func")
                    && ctx.op(op).attributes.get_symbol("sym_name") == Some(Symbol::new("run"))
            })
            .expect("the fixture function must survive lowering");
        let body = ctx.op(fixture).regions[0];
        let block = ctx.region(body).blocks[0];
        let calls: Vec<_> = ctx
            .block(block)
            .ops
            .iter()
            .copied()
            .filter_map(|op| wasm_dialect::Call::from_op(&ctx, op).ok())
            .collect();

        assert_eq!(calls.len(), 1, "{}", print_module(&ctx, module.op()));
        let expected = crate::wasm::type_converter::evidence_wasm_type(&mut ctx);
        assert_eq!(ctx.op_result_types(calls[0].op_ref()), [expected]);
    }

    #[test]
    fn wasm_function_scope_rewrites_only_selected_function() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, dispatch_module());
        lower_funcs_to_wasm(&mut ctx, module);
        let selected = module
            .ops(&ctx)
            .into_iter()
            .filter_map(|op| wasm_dialect::Func::from_op(&ctx, op).ok())
            .next()
            .expect("test module should contain a selected wasm function");

        lower_evidence_to_wasm_func(&mut ctx, selected).unwrap();

        let output = print_module(&ctx, module.op());
        assert_eq!(output.matches("effect.dispatch_tail").count(), 1);
        assert!(output.contains("sym_name = @selected"), "{output}");
        assert!(output.contains("sym_name = @untouched"), "{output}");
        assert!(output.contains("op_name = @print"), "{output}");
        assert!(output.contains("wasm.call_indirect"), "{output}");
    }

    #[test]
    fn pass_adapter_runs_wasm_function_lowering() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, dispatch_module());
        lower_funcs_to_wasm(&mut ctx, module);
        let selected = module
            .ops(&ctx)
            .into_iter()
            .filter_map(|op| wasm_dialect::Func::from_op(&ctx, op).ok())
            .next()
            .expect("test module should contain a selected wasm function");
        let mut pass = LowerEvidenceToWasm;

        assert_eq!(pass.name(), "lower-evidence-to-wasm");
        pass.run(&mut ctx, selected).unwrap();

        let output = print_module(&ctx, module.op());
        assert_eq!(output.matches("effect.dispatch_tail").count(), 1);
        assert!(output.contains("__tribute_evidence_lookup"), "{output}");
    }
}
