//! Wasm evidence lowering and evidence runtime helpers (arena-based).
//!
//! Evidence lowering runs inside the representation/ABI boundary. It lowers
//! `effect.extend`, `effect.dispatch_tail`, and `effect.dispatch_cps` to shared
//! value/control operations that call the evidence runtime helper ABI shared
//! with native (`tribute_ir::dialect::ability::evidence_abi`), through bodyless
//! helper declarations.
//!
//! After the boundary exit, [`bind_wasm_evidence_runtime`] replaces those
//! declarations with the Wasm target runtime: helper implementations over a
//! WasmGC evidence array, built on two internal helpers:
//!
//! - `__tribute_evidence_find_marker(ev, ability_id)` -> binary search for a marker
//! - `__tribute_evidence_insert_marker(ev, marker)` -> sorted insertion
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
//! Marker construction and field access stay inside the helper
//! implementations.

use tribute_ir::dialect::ability::{self as ability, MarkerField, evidence_abi};
use tribute_ir::dialect::effect;
use trunk_ir::Symbol;
use trunk_ir::analysis::AnalysisCache;
use trunk_ir::context::{BlockArgData, BlockData, IrContext, RegionData};
use trunk_ir::dialect::{core, func, wasm as wasm_dialect};
use trunk_ir::ops::DialectOp;
use trunk_ir::ops::DialectType;
use trunk_ir::pass::{Pass, PassRunResult};
use trunk_ir::refs::{OpRef, RegionRef, TypeRef, ValueRef};
use trunk_ir::rewrite::{
    ConversionError, ConversionTarget, Module, PatternApplicator, PatternRewriter, RewritePattern,
    TypeConverter,
};
use trunk_ir::smallvec::smallvec;
use trunk_ir::types::{Attribute, Location, TypeDataBuilder};
use trunk_ir_wasm_backend::gc_types::{EVIDENCE_IDX, MARKER_IDX};

use crate::effect_dispatch;

/// Internal Wasm runtime helper: binary search for a marker.
const FIND_MARKER: &str = "__tribute_evidence_find_marker";
/// Internal Wasm runtime helper: sorted marker insertion.
const INSERT_MARKER: &str = "__tribute_evidence_insert_marker";
/// C link name of the helper that allocates a fresh prompt tag.
pub const NEXT_TAG: &str = "__tribute_next_tag";

// =============================================================================
// Boundary: effect lowering
// =============================================================================

/// Declare the evidence runtime helpers that the module's `effect.*`
/// operations will call.
///
/// Run once on the module before [`LowerEvidenceToWasm`]. Only helpers that
/// some operation needs are declared, so the target binds no unused runtime.
pub fn prepare_wasm_evidence_runtime(ctx: &mut IrContext, module: Module) {
    let (lookup, lookup_tr, extend) = evidence_helper_requirements(ctx, module);
    let evidence_ty = ability::evidence_adt_type_ref(ctx);
    let closure_ty = crate::closure_lower::closure_struct_type_ref(ctx);
    let i32_ty = effect_dispatch::i32_type(ctx);
    let declarations = [
        (
            lookup,
            evidence_abi::LOOKUP,
            vec![evidence_ty, i32_ty],
            i32_ty,
        ),
        (
            lookup_tr,
            evidence_abi::LOOKUP_TR,
            vec![evidence_ty, i32_ty],
            closure_ty,
        ),
        (
            extend,
            evidence_abi::EXTEND,
            vec![evidence_ty, i32_ty, i32_ty, closure_ty, closure_ty],
            evidence_ty,
        ),
    ];
    for (needed, name, params, result) in declarations {
        if !needed || has_function(ctx, module, name) {
            continue;
        }
        let location = ctx.op(module.op()).location;
        let declaration = crate::native::build_extern_func(ctx, location, name, &params, result);
        prepend_module_op(ctx, module, declaration);
    }
}

/// Lower evidence operations in one function body.
///
/// Precondition: [`prepare_wasm_evidence_runtime`] already declared the
/// helpers for the containing module.
pub fn lower_evidence_to_wasm_func(
    ctx: &mut IrContext,
    func_op: func::Func,
) -> Result<(), ConversionError> {
    PatternApplicator::new(TypeConverter::new())
        .with_target(wasm_effect_abi_target())
        .add_pattern(LowerEffectExtendToWasm)
        .add_pattern(LowerEffectDispatchTailToWasm)
        .add_pattern(LowerEffectDispatchCpsToWasm)
        .apply_partial_conversion(ctx, func_op, "wasm-evidence-effect-abi")?;
    Ok(())
}

/// PassManager-friendly Wasm evidence lowering pass.
///
/// This pass is function-scoped and does not declare module-scope runtime
/// helpers. Run [`prepare_wasm_evidence_runtime`] on the module first.
pub struct LowerEvidenceToWasm;

impl Pass for LowerEvidenceToWasm {
    type Target = func::Func;

    fn name(&self) -> &'static str {
        "lower-evidence-to-wasm"
    }

    fn run(
        &mut self,
        ctx: &mut IrContext,
        target: func::Func,
        _analyses: &mut AnalysisCache,
    ) -> PassRunResult {
        lower_evidence_to_wasm_func(ctx, target)?;
        Ok(())
    }
}

fn wasm_effect_abi_target() -> ConversionTarget {
    ConversionTarget::new()
        .legal_op("func", "func")
        .recursive_legal_op("func", "func")
        .illegal_op("effect", "extend")
        .illegal_op("effect", "dispatch_tail")
        .illegal_op("effect", "dispatch_cps")
}

/// Which helpers (`lookup`, `lookup_tr`, `extend`) the module's operations need.
fn evidence_helper_requirements(ctx: &IrContext, module: Module) -> (bool, bool, bool) {
    fn visit(ctx: &IrContext, region: RegionRef, needs: &mut (bool, bool, bool)) {
        for &block in &ctx.region(region).blocks {
            for &op in &ctx.block(block).ops {
                needs.0 |= effect::DispatchCps::matches(ctx, op);
                needs.1 |= effect::DispatchTail::matches(ctx, op);
                needs.2 |= effect::Extend::matches(ctx, op);
                for nested in ctx.op_regions(op) {
                    visit(ctx, nested, needs);
                }
            }
        }
    }

    let mut needs = (false, false, false);
    if let Some(body) = module.body(ctx) {
        visit(ctx, body, &mut needs);
    }
    needs
}

fn has_function(ctx: &IrContext, module: Module, name: &'static str) -> bool {
    module.ops(ctx).iter().copied().any(|op| {
        ctx.op(op).attributes.get_symbol("sym_name") == Some(Symbol::new(name))
            && (func::Func::matches(ctx, op) || wasm_dialect::Func::matches(ctx, op))
    })
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

/// Retype a closure operand to the canonical closure slot of a helper call.
fn as_canonical_closure(
    ctx: &mut IrContext,
    loc: Location,
    value: ValueRef,
    rewriter: &mut PatternRewriter<'_>,
) -> ValueRef {
    let closure_ty = crate::closure_lower::closure_struct_type_ref(ctx);
    if ctx.value_ty(value) == closure_ty {
        return value;
    }
    let cast = core::UnrealizedConversionCast::operands(value)
        .results(closure_ty)
        .build(ctx, loc);
    rewriter.insert_op(cast.op_ref());
    cast.result(ctx)
}

/// `effect.extend` → `func.call @__tribute_evidence_extend`.
struct LowerEffectExtendToWasm;

impl RewritePattern for LowerEffectExtendToWasm {
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
        let ability_id =
            effect_dispatch::insert_ability_id(ctx, loc, extend_op.ability_ref(ctx), rewriter);
        let tr_dispatch = as_canonical_closure(ctx, loc, extend_op.tr_dispatch_fn(ctx), rewriter);
        let handler_dispatch =
            as_canonical_closure(ctx, loc, extend_op.handler_dispatch(ctx), rewriter);
        let result_ty = ctx.op_result_types(op)[0];
        let call = func::Call::operands([
            extend_op.evidence(ctx),
            ability_id,
            extend_op.prompt_tag(ctx),
            tr_dispatch,
            handler_dispatch,
        ])
        .callee(Symbol::new(evidence_abi::EXTEND))
        .results([result_ty])
        .build(ctx, loc);
        rewriter.insert_op(call.op_ref());
        rewriter.erase_op(vec![call.result(ctx)]);
        true
    }
}

/// `effect.dispatch_tail` → `__tribute_evidence_lookup_tr` + indirect call.
struct LowerEffectDispatchTailToWasm;

impl RewritePattern for LowerEffectDispatchTailToWasm {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(dispatch_op) = effect::DispatchTail::from_op(ctx, op) else {
            return false;
        };
        let converter = super::type_converter::wasm_type_converter(ctx);
        if !effect_dispatch::is_valid_tail_dispatch(ctx, op, &converter) {
            return false;
        }

        let loc = ctx.op(op).location;
        let closure_ty = crate::closure_lower::closure_struct_type_ref(ctx);
        let ability_id =
            effect_dispatch::insert_ability_id(ctx, loc, dispatch_op.ability_ref(ctx), rewriter);
        let dispatch_closure = func::Call::operands([dispatch_op.evidence(ctx), ability_id])
            .callee(Symbol::new(evidence_abi::LOOKUP_TR))
            .results([closure_ty])
            .build(ctx, loc);
        rewriter.insert_op(dispatch_closure.op_ref());
        effect_dispatch::lower_tail_dispatch(ctx, op, dispatch_closure.result(ctx), rewriter);
        true
    }
}

/// `effect.dispatch_cps` → `__tribute_evidence_lookup` + proper-tail transfer.
struct LowerEffectDispatchCpsToWasm;

impl RewritePattern for LowerEffectDispatchCpsToWasm {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(dispatch_op) = effect::DispatchCps::from_op(ctx, op) else {
            return false;
        };
        let converter = super::type_converter::wasm_type_converter(ctx);
        if !effect_dispatch::is_valid_cps_dispatch(ctx, op, &converter) {
            return false;
        }

        let loc = ctx.op(op).location;
        let i32_ty = effect_dispatch::i32_type(ctx);
        let ability_id =
            effect_dispatch::insert_ability_id(ctx, loc, dispatch_op.ability_ref(ctx), rewriter);
        let prompt = func::Call::operands([dispatch_op.evidence(ctx), ability_id])
            .callee(Symbol::new(evidence_abi::LOOKUP))
            .results([i32_ty])
            .build(ctx, loc);
        rewriter.insert_op(prompt.op_ref());
        effect_dispatch::lower_cps_dispatch(ctx, op, ability_id, prompt.result(ctx), rewriter);
        true
    }
}

// =============================================================================
// After the exit: Wasm runtime helper binding
// =============================================================================

/// Bind bodyless evidence runtime helper declarations to their Wasm
/// implementations.
///
/// Runs after Wasm dialect lowering has converted the declarations'
/// signatures; each implementation adopts the declaration's signature.
pub fn bind_wasm_evidence_runtime(ctx: &mut IrContext, module: Module) {
    let Some(block) = module.first_block(ctx) else {
        return;
    };
    let mut needs_find = false;
    let mut needs_insert = false;
    for op in module.ops_snapshot(ctx) {
        let data = ctx.op(op);
        let is_function = wasm_dialect::Func::matches(ctx, op) || func::Func::matches(ctx, op);
        if !is_function || ctx.op_has_regions(op) {
            continue;
        }
        let Some(name) = data.attributes.get_symbol("sym_name") else {
            continue;
        };
        let Some(signature) = data.attributes.get_type("type") else {
            continue;
        };
        let location = data.location;
        let helper = if name == Symbol::new(evidence_abi::LOOKUP) {
            needs_find = true;
            Helper::MarkerField(MarkerField::PromptTag)
        } else if name == Symbol::new(evidence_abi::LOOKUP_TR) {
            needs_find = true;
            Helper::MarkerField(MarkerField::TrDispatchFn)
        } else if name == Symbol::new(evidence_abi::LOOKUP_HANDLER) {
            needs_find = true;
            Helper::MarkerField(MarkerField::HandlerDispatch)
        } else if name == Symbol::new(evidence_abi::EXTEND) {
            needs_insert = true;
            Helper::Extend
        } else if name == Symbol::new(NEXT_TAG) {
            Helper::NextTag(add_tag_counter(ctx, module, location))
        } else {
            continue;
        };
        let implementation = build_helper(ctx, location, name, signature, helper);
        ctx.insert_op_before(block, op, implementation);
        ctx.remove_op_from_block(block, op);
        ctx.remove_op(op);
    }

    let location = ctx.op(module.op()).location;
    if needs_find && !has_function(ctx, module, FIND_MARKER) {
        let op = generate_evidence_lookup_function(ctx, location);
        prepend_module_op(ctx, module, op);
    }
    if needs_insert && !has_function(ctx, module, INSERT_MARKER) {
        let op = generate_evidence_extend_function(ctx, location);
        prepend_module_op(ctx, module, op);
    }
}

#[derive(Clone, Copy)]
enum Helper {
    /// Look up the marker for an ability and return one of its fields.
    MarkerField(MarkerField),
    /// Build a marker from its fields and insert it.
    Extend,
    /// Return the tag counter global at this index and increment it.
    NextTag(u32),
}

/// Append a mutable `i32` tag counter global, returning its index.
fn add_tag_counter(ctx: &mut IrContext, module: Module, location: Location) -> u32 {
    let index = module
        .ops(ctx)
        .iter()
        .filter(|&&op| wasm_dialect::Global::matches(ctx, op))
        .count() as u32;
    let global = wasm_dialect::Global::operands()
        .valtype("i32")
        .mutable(true)
        .init(Attribute::Int(0))
        .build(ctx, location);
    let block = module
        .first_block(ctx)
        .expect("a module with declarations has a body block");
    ctx.push_op(block, global.op_ref());
    index
}

/// Build a helper implementation with the declaration's (converted)
/// signature.
fn build_helper(
    ctx: &mut IrContext,
    location: Location,
    name: Symbol,
    signature: TypeRef,
    helper: Helper,
) -> OpRef {
    let (inputs, results) = if let Some(sig) = wasm_dialect::FuncSig::from_type_ref(ctx, signature)
    {
        (sig.inputs(ctx).to_vec(), sig.results(ctx).to_vec())
    } else {
        let sig = func::FuncSig::from_type_ref(ctx, signature)
            .expect("runtime helper declaration must have a function signature");
        (sig.inputs(ctx).to_vec(), sig.results(ctx).to_vec())
    };
    let result_ty = *results
        .first()
        .expect("runtime helper declaration must have one result");
    let block = ctx.create_block(BlockData {
        location,
        args: inputs
            .iter()
            .map(|&ty| BlockArgData {
                ty,
                attrs: Default::default(),
            })
            .collect(),
        ops: smallvec![],
        parent_region: None,
    });
    let args = ctx.block_args(block).to_vec();
    let marker_ty = ability::marker_adt_type_ref(ctx);
    let returned = match helper {
        Helper::MarkerField(field) => {
            let marker = wasm_dialect::Call::operands([args[0], args[1]])
                .callee(Symbol::new(FIND_MARKER))
                .results([marker_ty])
                .build(ctx, location);
            ctx.push_op(block, marker.op_ref());
            let get = wasm_dialect::StructGet::operands(marker.results(ctx)[0])
                .type_idx(MARKER_IDX)
                .field_idx(field.index())
                .results(result_ty)
                .build(ctx, location);
            ctx.push_op(block, get.op_ref());
            get.result(ctx)
        }
        Helper::Extend => {
            let marker = wasm_dialect::StructNew::operands([args[1], args[2], args[3], args[4]])
                .type_idx(MARKER_IDX)
                .results(marker_ty)
                .build(ctx, location);
            ctx.push_op(block, marker.op_ref());
            let call = wasm_dialect::Call::operands([args[0], marker.result(ctx)])
                .callee(Symbol::new(INSERT_MARKER))
                .results([result_ty])
                .build(ctx, location);
            ctx.push_op(block, call.op_ref());
            call.results(ctx)[0]
        }
        Helper::NextTag(global) => {
            let current = wasm_dialect::GlobalGet::operands()
                .index(global)
                .results(result_ty)
                .build(ctx, location);
            ctx.push_op(block, current.op_ref());
            let one = wasm_dialect::I32Const::operands()
                .value(1)
                .results(result_ty)
                .build(ctx, location);
            ctx.push_op(block, one.op_ref());
            let next = wasm_dialect::I32Add::operands(current.result(ctx), one.result(ctx))
                .build(ctx, location);
            ctx.push_op(block, next.op_ref());
            let set = wasm_dialect::GlobalSet::operands(next.result(ctx))
                .index(global)
                .build(ctx, location);
            ctx.push_op(block, set.op_ref());
            current.result(ctx)
        }
    };
    let ret = wasm_dialect::Return::operands(vec![returned]).build(ctx, location);
    ctx.push_op(block, ret.op_ref());
    let body = ctx.create_region(RegionData {
        location,
        blocks: smallvec![block],
        parent_op: None,
    });
    let func_ty = wasm_dialect::func_sig(ctx, inputs, [result_ty]).as_type_ref();
    wasm_dialect::Func::operands()
        .sym_name(name)
        .r#type(func_ty)
        .regions(body)
        .build(ctx, location)
        .op_ref()
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

/// Generate the internal `__tribute_evidence_find_marker` helper.
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
    let zero = wasm_dialect::I32Const::operands()
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
    let unreachable_op = wasm_dialect::Unreachable::operands().build(ctx, location);
    ctx.push_op(body_block, unreachable_op.op_ref());

    let body = ctx.create_region(RegionData {
        location,
        blocks: smallvec![body_block],
        parent_op: None,
    });

    let func_op = wasm_dialect::Func::operands()
        .sym_name(Symbol::new(FIND_MARKER))
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
    let get_low = wasm_dialect::LocalGet::operands()
        .index(locals::LOW)
        .results(i32_ty)
        .build(ctx, location);
    let low = get_low.result(ctx);
    ctx.push_op(block, get_low.op_ref());

    // high = local.get HIGH
    let get_high = wasm_dialect::LocalGet::operands()
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
        let unreachable_op = wasm_dialect::Unreachable::operands().build(ctx, location);
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
    let two = wasm_dialect::I32Const::operands()
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
            let one = wasm_dialect::I32Const::operands()
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
        let br_loop = wasm_dialect::Br::operands().target(0).build(ctx, location);
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

/// Generate the internal `__tribute_evidence_insert_marker` helper.
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
    let zero = wasm_dialect::I32Const::operands()
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
    let block_op = wasm_dialect::Block::operands()
        .results([nil_ty])
        .regions(loop_region)
        .build(ctx, location);
    ctx.push_op(body_block, block_op.op_ref());

    // insert_idx = local.get LOW
    let get_insert_idx = wasm_dialect::LocalGet::operands()
        .index(locals::LOW)
        .results(i32_ty)
        .build(ctx, location);
    let insert_idx = get_insert_idx.result(ctx);
    ctx.push_op(body_block, get_insert_idx.op_ref());

    // new_len = old_len + 1
    let one = wasm_dialect::I32Const::operands()
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
    let zero2 = wasm_dialect::I32Const::operands()
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
        let zero3 = wasm_dialect::I32Const::operands()
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
    let zero4 = wasm_dialect::I32Const::operands()
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
        let one2 = wasm_dialect::I32Const::operands()
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

    let func_op = wasm_dialect::Func::operands()
        .sym_name(Symbol::new(INSERT_MARKER))
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
    let get_low = wasm_dialect::LocalGet::operands()
        .index(locals::LOW)
        .results(i32_ty)
        .build(ctx, location);
    let low = get_low.result(ctx);
    ctx.push_op(block, get_low.op_ref());

    // high = local.get HIGH
    let get_high = wasm_dialect::LocalGet::operands()
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
    let two = wasm_dialect::I32Const::operands()
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
        let one = wasm_dialect::I32Const::operands()
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
    let br_loop = wasm_dialect::Br::operands().target(0).build(ctx, location);
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
    super::type_converter::evidence_wasm_type(ctx)
}

/// Intern a `core.i32` type.
fn intern_i32(ctx: &mut IrContext) -> TypeRef {
    ctx.intern_type(TypeDataBuilder::new("core", "i32").build())
}

/// Intern a target-owned `wasm.func_sig` type with the given parameter and return types.
fn intern_func_type(ctx: &mut IrContext, params: &[TypeRef], ret: TypeRef) -> TypeRef {
    wasm_dialect::func_sig(ctx, params.iter().copied(), [ret]).as_type_ref()
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;

    const TYPES: &str = r#"  !Evidence = core.array<adt.struct<_Marker(ability_id: core.i32, prompt_tag: core.i32, tr_dispatch_fn: core.ptr, handler_dispatch: core.ptr), {layout = "evidence_marker"}>, {layout = "evidence"}>
  !Closure = adt.struct<_closure(func_ptr: core.i32, env: tribute_rt.anyref), {layout = "closure"}>"#;

    fn module_text(body: &str) -> String {
        format!("core.module @test {{\n{TYPES}\n{body}\n}}")
    }

    fn lower(ctx: &mut IrContext, module: Module) -> Result<(), ConversionError> {
        prepare_wasm_evidence_runtime(ctx, module);
        for op in module.ops_snapshot(ctx) {
            if let Ok(function) = func::Func::from_op(ctx, op) {
                lower_evidence_to_wasm_func(ctx, function)?;
            }
        }
        Ok(())
    }

    fn lower_text(body: &str) -> String {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, &module_text(body));
        lower(&mut ctx, module).expect("evidence lowering");
        print_module(&ctx, module.op())
    }

    fn assert_shared_dialect_only(printed: &str) {
        assert!(!printed.contains("effect."), "{printed}");
        assert!(!printed.contains("wasm."), "{printed}");
        assert!(!printed.contains("tribute.calling_convention"), "{printed}");
    }

    const TAIL: &str = r#"  func.func @tail(%ev: !Evidence, %payload: tribute_rt.anyref) -> tribute_rt.anyref {
    %result = effect.dispatch_tail %ev, %payload {ability_ref = core.ability_ref<{name = @Console}>, op_name = @read} : tribute_rt.anyref
    func.return %result
  }"#;

    const CPS: &str = r#"  func.func @cps(%ev: !Evidence, %dispatch: !Closure, %resume: !Closure, %payload: tribute_rt.anyref) {
    effect.dispatch_cps %ev, %dispatch, %resume, %payload {ability_ref = core.ability_ref<{name = @State}>, op_name = @get, answer_type = core.i32}
  }"#;

    const EXTEND: &str = r#"  func.func @install(%ev: !Evidence, %prompt: core.i32, %tr: !Closure, %handler: !Closure) -> !Evidence {
    %extended = effect.extend %ev, %prompt, %tr, %handler {ability_ref = core.ability_ref<{name = @State}>} : !Evidence
    func.return %extended
  }"#;

    #[test]
    fn tail_dispatch_calls_the_selected_closure_indirectly() {
        let printed = lower_text(TAIL);

        assert_shared_dialect_only(&printed);
        assert!(
            printed.contains("func.func @__tribute_evidence_lookup_tr("),
            "{printed}"
        );
        assert!(
            printed.contains("callee = @__tribute_evidence_lookup_tr"),
            "{printed}"
        );
        assert!(printed.contains("func.call_indirect"), "{printed}");
        assert!(
            !printed.contains("@__tribute_evidence_lookup("),
            "{printed}"
        );
        assert!(!printed.contains("@__tribute_evidence_extend"), "{printed}");
    }

    #[test]
    fn cps_dispatch_transfers_through_a_tail_signature() {
        let printed = lower_text(CPS);

        assert_shared_dialect_only(&printed);
        assert!(
            printed.contains("callee = @__tribute_evidence_lookup}"),
            "{printed}"
        );
        assert!(printed.contains("func.tail_call_indirect"), "{printed}");
        assert!(printed.contains("call_conv = @tail"), "{printed}");
    }

    #[test]
    fn extend_calls_the_runtime_with_marker_fields() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, &module_text(EXTEND));
        lower(&mut ctx, module).unwrap();
        let printed = print_module(&ctx, module.op());

        assert_shared_dialect_only(&printed);
        let mut calls = Vec::new();
        let _ = trunk_ir::walk::walk_op::<()>(&ctx, module.op(), &mut |op| {
            if let Ok(call) = func::Call::from_op(&ctx, op) {
                calls.push((call.callee(&ctx), ctx.op_operands(op).len()));
            }
            std::ops::ControlFlow::Continue(trunk_ir::walk::WalkAction::Advance)
        });
        assert_eq!(calls, [(Symbol::new(evidence_abi::EXTEND), 5)], "{printed}");
    }

    #[test]
    fn malformed_cps_dispatch_is_rejected_without_mutation() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            &module_text(
                r#"  func.func @cps(%ev: !Evidence, %dispatch: !Closure, %resume: !Closure, %payload: tribute_rt.anyref) {
    %bad = effect.dispatch_cps %ev, %dispatch, %resume, %payload {ability_ref = core.ability_ref<{name = @State}>, op_name = @get, answer_type = core.i32} : core.i32
  }"#,
            ),
        );
        prepare_wasm_evidence_runtime(&mut ctx, module);
        let before = print_module(&ctx, module.op());
        let function = module
            .ops(&ctx)
            .iter()
            .copied()
            .find_map(|op| {
                func::Func::from_op(&ctx, op)
                    .ok()
                    .filter(|f| f.sym_name(&ctx) == Symbol::new("cps"))
            })
            .unwrap();

        let error = lower_evidence_to_wasm_func(&mut ctx, function)
            .expect_err("a result-bearing final dispatch is malformed");

        assert!(error.to_string().contains("effect.dispatch_cps"), "{error}");
        assert_eq!(print_module(&ctx, module.op()), before);
    }

    #[test]
    fn only_needed_helpers_are_declared() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, &module_text(EXTEND));
        prepare_wasm_evidence_runtime(&mut ctx, module);
        let printed = print_module(&ctx, module.op());

        assert!(printed.contains("@__tribute_evidence_extend("), "{printed}");
        assert!(!printed.contains("@__tribute_evidence_lookup"), "{printed}");
    }

    #[test]
    fn binding_replaces_declarations_with_implementations() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @__tribute_evidence_lookup(%ev: wasm.arrayref, %id: core.i32) -> core.i32 attributes {abi = "C"}
  func.func @__tribute_evidence_lookup_tr(%ev: wasm.arrayref, %id: core.i32) -> wasm.anyref attributes {abi = "C"}
  func.func @__tribute_evidence_extend(%ev: wasm.arrayref, %id: core.i32, %prompt: core.i32, %tr: wasm.anyref, %handler: wasm.anyref) -> wasm.arrayref attributes {abi = "C"}
}"#,
        );

        bind_wasm_evidence_runtime(&mut ctx, module);

        let mut functions = Vec::new();
        for &op in module.ops(&ctx) {
            let function = wasm_dialect::Func::from_op(&ctx, op).expect("bound wasm.func");
            assert!(ctx.op_has_regions(op), "helpers must have bodies");
            functions.push(function.sym_name(&ctx).to_string());
        }
        functions.sort();
        assert_eq!(
            functions,
            [
                "__tribute_evidence_extend",
                "__tribute_evidence_find_marker",
                "__tribute_evidence_insert_marker",
                "__tribute_evidence_lookup",
                "__tribute_evidence_lookup_tr",
            ]
        );
    }

    #[test]
    fn next_tag_counts_in_a_global_appended_after_existing_ones() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  wasm.global {valtype = "i32", mutable = false, init = 7}
  func.func @__tribute_next_tag() -> core.i32 attributes {abi = "C"}
}"#,
        );

        bind_wasm_evidence_runtime(&mut ctx, module);

        assert_eq!(
            print_module(&ctx, module.op()),
            r#"core.module @test {
  wasm.global {init = 7, mutable = false, valtype = "i32"}
  wasm.func {sym_name = @__tribute_next_tag, type = wasm.func_sig<() -> core.i32>} {
      %0 = wasm.global_get {index = 1} : core.i32
      %1 = wasm.i32_const {value = 1} : core.i32
      %2 = wasm.i32_add %0, %1 : core.i32
      wasm.global_set %2 {index = 1}
      wasm.return %0
  }
  wasm.global {init = 0, mutable = true, valtype = "i32"}
}
"#
        );
    }

    #[test]
    fn binding_leaves_modules_without_helper_declarations_unchanged() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @main() -> core.nil {
    %nil = core.nil_value : core.nil
    func.return %nil
  }
}"#,
        );
        let before = print_module(&ctx, module.op());

        bind_wasm_evidence_runtime(&mut ctx, module);

        assert_eq!(print_module(&ctx, module.op()), before);
    }
}
