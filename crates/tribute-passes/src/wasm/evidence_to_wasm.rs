//! Wasm evidence lowering and evidence runtime helpers (arena-based).
//!
//! Evidence lowering runs inside the representation/ABI boundary. It lowers
//! `effect.extend`, `effect.mask`, `effect.dup`, `effect.dispatch_tail`, and
//! `effect.dispatch_cps` to shared value/control operations that call the
//! evidence runtime helper ABI shared with native
//! (`tribute_ir::dialect::ability::evidence_abi`), through bodyless helper
//! declarations.
//!
//! After the boundary exit, [`bind_wasm_evidence_runtime`] replaces those
//! declarations with the Wasm target runtime: helper implementations over a
//! WasmGC evidence array, built on four internal helpers:
//!
//! - `__tribute_evidence_find_slot(ev, ability_id, low, high)` -> binary search
//!   for the index of an ability's slot, or of where it would be inserted
//! - `__tribute_evidence_find_marker(ev, ability_id)` -> the top marker of an
//!   ability, or null
//! - `__tribute_evidence_set_marker(ev, marker)` -> copy with the marker as the
//!   top marker of its ability
//! - `__tribute_evidence_remove_marker(ev, ability_id)` -> copy without the
//!   ability's slot
//!
//! ## Evidence Structure
//!
//! Evidence is represented as a WasmGC array of Marker structs, sorted by
//! ability_id. Each element is the top marker of its ability, and `shadowed`
//! refers to the marker of the same ability it shadows:
//!
//! ```text
//! Evidence = Array(Marker)
//! Marker = struct { ability_id: i32, prompt_tag: i32, tr_dispatch_fn: anyref, shadowed: anyref }
//! ```
//!
//! Marker construction and field access stay inside the helper
//! implementations.

use tribute_ir::dialect::ability::{self as ability, MarkerField, evidence_abi};
use tribute_ir::dialect::effect;
use trunk_ir::analysis::AnalysisCache;
use trunk_ir::context::{BlockArgData, BlockData, IrContext, RegionData};
use trunk_ir::dialect::{core, func, wasm as wasm_dialect};
use trunk_ir::ops::DialectOp;
use trunk_ir::ops::DialectType;
use trunk_ir::pass::{Pass, PassRunResult};
use trunk_ir::refs::{BlockRef, OpRef, RegionRef, TypeRef, ValueRef};
use trunk_ir::rewrite::{
    ConversionError, ConversionTarget, Module, PatternApplicator, PatternRewriter, RewritePattern,
    TypeConverter,
};
use trunk_ir::smallvec::smallvec;
use trunk_ir::types::{Attribute, Location, TypeDataBuilder};
use trunk_ir::{StringRef, Symbol, SymbolPath};
use trunk_ir_wasm_backend::gc_types::{EVIDENCE_IDX, MARKER_IDX};

use crate::effect_dispatch;

/// Internal Wasm runtime helper: binary search for an ability's slot index.
const FIND_SLOT: &str = "__tribute_evidence_find_slot";
/// Internal Wasm runtime helper: lookup of an ability's top marker.
const FIND_MARKER: &str = "__tribute_evidence_find_marker";
/// Internal Wasm runtime helper: sorted marker insertion or replacement.
const SET_MARKER: &str = "__tribute_evidence_set_marker";
/// Internal Wasm runtime helper: removal of an ability's slot.
const REMOVE_MARKER: &str = "__tribute_evidence_remove_marker";
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
    let needs = evidence_helper_requirements(ctx, module);
    let evidence_ty = ability::evidence_adt_type_ref(ctx);
    let closure_ty = crate::closure_lower::closure_struct_type_ref(ctx);
    let i32_ty = effect_dispatch::i32_type(ctx);
    let declarations = [
        (
            needs.lookup,
            evidence_abi::LOOKUP,
            vec![evidence_ty, i32_ty],
            i32_ty,
        ),
        (
            needs.lookup_tr,
            evidence_abi::LOOKUP_TR,
            vec![evidence_ty, i32_ty],
            closure_ty,
        ),
        (
            needs.extend,
            evidence_abi::EXTEND,
            vec![evidence_ty, i32_ty, i32_ty, closure_ty],
            evidence_ty,
        ),
        (
            needs.mask,
            evidence_abi::MASK,
            vec![evidence_ty, i32_ty],
            evidence_ty,
        ),
        (
            needs.dup,
            evidence_abi::DUP,
            vec![evidence_ty, i32_ty],
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
        .add_pattern(LowerEffectStackOpToWasm)
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
        .illegal_op("effect", "mask")
        .illegal_op("effect", "dup")
        .illegal_op("effect", "dispatch_tail")
        .illegal_op("effect", "dispatch_cps")
}

/// The helpers the module's `effect.*` operations call.
#[derive(Default)]
struct HelperRequirements {
    lookup: bool,
    lookup_tr: bool,
    extend: bool,
    mask: bool,
    dup: bool,
}

fn evidence_helper_requirements(ctx: &IrContext, module: Module) -> HelperRequirements {
    fn visit(ctx: &IrContext, region: RegionRef, needs: &mut HelperRequirements) {
        for &block in &ctx.region(region).blocks {
            for &op in &ctx.block(block).ops {
                needs.lookup |= effect::DispatchCps::matches(ctx, op);
                needs.lookup_tr |= effect::DispatchTail::matches(ctx, op);
                needs.extend |= effect::Extend::matches(ctx, op);
                needs.mask |= effect::Mask::matches(ctx, op);
                needs.dup |= effect::Dup::matches(ctx, op);
                for nested in ctx.op_regions(op) {
                    visit(ctx, nested, needs);
                }
            }
        }
    }

    let mut needs = HelperRequirements::default();
    if let Some(body) = module.body(ctx) {
        visit(ctx, body, &mut needs);
    }
    needs
}

fn has_function(ctx: &IrContext, module: Module, name: &'static str) -> bool {
    module.ops(ctx).iter().copied().any(|op| {
        ctx.op(op)
            .attributes
            .get_str(ctx, "sym_name")
            .map(Symbol::new)
            == Some(Symbol::new(name))
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
        let result_ty = ctx.op_result_types(op)[0];
        let call = func::Call::operands([
            extend_op.evidence(ctx),
            ability_id,
            extend_op.prompt_tag(ctx),
            tr_dispatch,
        ])
        .callee(SymbolPath::from(evidence_abi::EXTEND))
        .results([result_ty])
        .build(ctx, loc);
        rewriter.insert_op(call.op_ref());
        rewriter.erase_op(vec![call.result(ctx)]);
        true
    }
}

/// `effect.mask` / `effect.dup` → the runtime call of the same name.
struct LowerEffectStackOpToWasm;

impl RewritePattern for LowerEffectStackOpToWasm {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let (helper, ability_ref, evidence) = if let Ok(mask) = effect::Mask::from_op(ctx, op) {
            (
                evidence_abi::MASK,
                mask.ability_ref(ctx),
                mask.evidence(ctx),
            )
        } else if let Ok(dup) = effect::Dup::from_op(ctx, op) {
            (evidence_abi::DUP, dup.ability_ref(ctx), dup.evidence(ctx))
        } else {
            return false;
        };
        let loc = ctx.op(op).location;
        let ability_id = effect_dispatch::insert_ability_id(ctx, loc, ability_ref, rewriter);
        let result_ty = ctx.op_result_types(op)[0];
        let call = func::Call::operands([evidence, ability_id])
            .callee(SymbolPath::from(helper))
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
            .callee(SymbolPath::from(evidence_abi::LOOKUP_TR))
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
            .callee(SymbolPath::from(evidence_abi::LOOKUP))
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
    let mut needs_set = false;
    let mut needs_remove = false;
    for op in module.ops_snapshot(ctx) {
        let data = ctx.op(op);
        let is_function = wasm_dialect::Func::matches(ctx, op) || func::Func::matches(ctx, op);
        if !is_function || ctx.op_has_regions(op) {
            continue;
        }
        let Some(name_ref) = data.attributes.get_string_ref("sym_name") else {
            continue;
        };
        let Some(signature) = data.attributes.get_type("type") else {
            continue;
        };
        let location = data.location;
        let name = ctx.str(name_ref);
        let helper = if name == evidence_abi::LOOKUP {
            needs_find = true;
            Helper::MarkerField(MarkerField::PromptTag)
        } else if name == evidence_abi::LOOKUP_TR {
            needs_find = true;
            Helper::MarkerField(MarkerField::TrDispatchFn)
        } else if name == evidence_abi::EXTEND {
            needs_find = true;
            needs_set = true;
            Helper::Extend
        } else if name == evidence_abi::MASK {
            needs_find = true;
            needs_set = true;
            needs_remove = true;
            Helper::Mask
        } else if name == evidence_abi::DUP {
            needs_find = true;
            needs_set = true;
            Helper::Dup
        } else if name == NEXT_TAG {
            Helper::NextTag(add_tag_counter(ctx, module, location))
        } else {
            continue;
        };
        let implementation = build_helper(ctx, location, name_ref, signature, helper);
        ctx.insert_op_before(block, op, implementation);
        ctx.remove_op_from_block(block, op);
        ctx.remove_op(op);
    }

    let location = ctx.op(module.op()).location;
    if needs_find && !has_function(ctx, module, FIND_MARKER) {
        let op = generate_evidence_lookup_function(ctx, location);
        prepend_module_op(ctx, module, op);
    }
    if needs_set && !has_function(ctx, module, SET_MARKER) {
        let op = generate_evidence_set_function(ctx, location);
        prepend_module_op(ctx, module, op);
    }
    if needs_remove && !has_function(ctx, module, REMOVE_MARKER) {
        let op = generate_evidence_remove_function(ctx, location);
        prepend_module_op(ctx, module, op);
    }
    if needs_find && !has_function(ctx, module, FIND_SLOT) {
        let op = generate_evidence_slot_function(ctx, location);
        prepend_module_op(ctx, module, op);
    }
}

#[derive(Clone, Copy)]
enum Helper {
    /// Look up the marker for an ability and return one of its fields.
    MarkerField(MarkerField),
    /// Build a marker from its fields and push it onto its ability's stack.
    Extend,
    /// Pop the top marker of an ability.
    Mask,
    /// Push a copy of the top marker of an ability.
    Dup,
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
    name: StringRef,
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
    // The top marker of the ability in `args[1]`, or null.
    let find_marker = |ctx: &mut IrContext| {
        let marker = wasm_dialect::Call::operands([args[0], args[1]])
            .callee(SymbolPath::from(FIND_MARKER))
            .results([marker_ty])
            .build(ctx, location);
        ctx.push_op(block, marker.op_ref());
        marker.results(ctx)[0]
    };
    let marker_field = |ctx: &mut IrContext, marker: ValueRef, field: MarkerField, ty: TypeRef| {
        let get = wasm_dialect::StructGet::operands(marker)
            .type_idx(MARKER_IDX)
            .field_idx(field.index())
            .results(ty)
            .build(ctx, location);
        ctx.push_op(block, get.op_ref());
        get.result(ctx)
    };
    // The evidence with `fields` as the top marker of its ability.
    let set_marker = |ctx: &mut IrContext, fields: [ValueRef; ability::MARKER_FIELD_COUNT]| {
        let marker = wasm_dialect::StructNew::operands(fields)
            .type_idx(MARKER_IDX)
            .results(marker_ty)
            .build(ctx, location);
        ctx.push_op(block, marker.op_ref());
        let call = wasm_dialect::Call::operands([args[0], marker.result(ctx)])
            .callee(SymbolPath::from(SET_MARKER))
            .results([result_ty])
            .build(ctx, location);
        ctx.push_op(block, call.op_ref());
        call.results(ctx)[0]
    };
    let returned = match helper {
        Helper::MarkerField(field) => {
            let marker = find_marker(ctx);
            marker_field(ctx, marker, field, result_ty)
        }
        Helper::Extend => {
            let shadowed = find_marker(ctx);
            set_marker(ctx, [args[1], args[2], args[3], shadowed])
        }
        Helper::Dup => {
            let i32_ty = intern_i32(ctx);
            let anyref_ty = intern_anyref(ctx);
            let top = find_marker(ctx);
            let fields = [
                marker_field(ctx, top, MarkerField::AbilityId, i32_ty),
                marker_field(ctx, top, MarkerField::PromptTag, i32_ty),
                marker_field(ctx, top, MarkerField::TrDispatchFn, anyref_ty),
                top,
            ];
            set_marker(ctx, fields)
        }
        Helper::Mask => {
            let i32_ty = intern_i32(ctx);
            let anyref_ty = intern_anyref(ctx);
            let nil_ty = core::nil(ctx).as_type_ref();
            let top = find_marker(ctx);
            let shadowed = marker_field(ctx, top, MarkerField::Shadowed, anyref_ty);
            let unshadowed = wasm_dialect::RefIsNull::operands(shadowed)
                .results(i32_ty)
                .build(ctx, location);
            ctx.push_op(block, unshadowed.op_ref());

            let remove_slot = {
                let inner = empty_block(ctx, location);
                let call = wasm_dialect::Call::operands([args[0], args[1]])
                    .callee(SymbolPath::from(REMOVE_MARKER))
                    .results([result_ty])
                    .build(ctx, location);
                ctx.push_op(inner, call.op_ref());
                let ret =
                    wasm_dialect::Return::operands(vec![call.results(ctx)[0]]).build(ctx, location);
                ctx.push_op(inner, ret.op_ref());
                single_block_region(ctx, location, inner)
            };
            let restore_shadowed = {
                let inner = empty_block(ctx, location);
                let marker = wasm_dialect::RefCast::operands(shadowed)
                    .target_type(marker_ty)
                    .type_idx(MARKER_IDX)
                    .results(marker_ty)
                    .build(ctx, location);
                ctx.push_op(inner, marker.op_ref());
                let call = wasm_dialect::Call::operands([args[0], marker.result(ctx)])
                    .callee(SymbolPath::from(SET_MARKER))
                    .results([result_ty])
                    .build(ctx, location);
                ctx.push_op(inner, call.op_ref());
                let ret =
                    wasm_dialect::Return::operands(vec![call.results(ctx)[0]]).build(ctx, location);
                ctx.push_op(inner, ret.op_ref());
                single_block_region(ctx, location, inner)
            };
            let branch = wasm_dialect::If::operands(unshadowed.result(ctx))
                .results([nil_ty])
                .regions(remove_slot, restore_shadowed)
                .build(ctx, location);
            ctx.push_op(block, branch.op_ref());
            let unreachable = wasm_dialect::Unreachable::operands().build(ctx, location);
            ctx.push_op(block, unreachable.op_ref());
            return finish_helper(ctx, location, name, inputs, result_ty, block);
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
    finish_helper(ctx, location, name, inputs, result_ty, block)
}

/// Wrap a terminated body block into the helper's `wasm.func`.
fn finish_helper(
    ctx: &mut IrContext,
    location: Location,
    name: StringRef,
    inputs: Vec<TypeRef>,
    result_ty: TypeRef,
    block: BlockRef,
) -> OpRef {
    let body = single_block_region(ctx, location, block);
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

/// Generate the internal `__tribute_evidence_find_slot` helper.
///
/// `(ev, ability_id, low, high) -> i32` is a recursive binary search over
/// `ev[low..high]`. It returns the first index whose marker has an ability id
/// not less than `ability_id`: the ability's slot if the evidence holds it,
/// otherwise the position that keeps the array sorted.
fn generate_evidence_slot_function(ctx: &mut IrContext, location: Location) -> OpRef {
    let evidence_ty = evidence_ref_type(ctx);
    let i32_ty = intern_i32(ctx);
    let params = [evidence_ty, i32_ty, i32_ty, i32_ty];
    let body = entry_block(ctx, location, &params);
    let [ev, ability_id, low, high] = ctx.block_args(body)[..] else {
        unreachable!("entry block has one argument per parameter")
    };

    // if low >= high: return low
    let done = wasm_dialect::I32GeS::operands(low, high)
        .results(i32_ty)
        .build(ctx, location);
    ctx.push_op(body, done.op_ref());
    let done_then = {
        let block = empty_block(ctx, location);
        push_return(ctx, location, block, low);
        single_block_region(ctx, location, block)
    };
    push_if(ctx, location, body, done.result(ctx), done_then, None);

    // mid = (low + high) / 2
    let sum = wasm_dialect::I32Add::operands(low, high).build(ctx, location);
    ctx.push_op(body, sum.op_ref());
    let two = i32_const(ctx, location, body, 2);
    let mid_op = wasm_dialect::I32DivU::operands(sum.result(ctx), two)
        .results(i32_ty)
        .build(ctx, location);
    let mid = mid_op.result(ctx);
    ctx.push_op(body, mid_op.op_ref());

    // if ev[mid].ability_id < ability_id: search ev[mid + 1..high]
    // else: search ev[low..mid]
    let mid_marker = marker_at(ctx, location, body, ev, mid);
    let mid_id = marker_ability_id(ctx, location, body, mid_marker);
    let below = wasm_dialect::I32LtS::operands(mid_id, ability_id)
        .results(i32_ty)
        .build(ctx, location);
    ctx.push_op(body, below.op_ref());
    let upper_half = {
        let block = empty_block(ctx, location);
        let one = i32_const(ctx, location, block, 1);
        let next = wasm_dialect::I32Add::operands(mid, one).build(ctx, location);
        ctx.push_op(block, next.op_ref());
        let slot = find_slot(ctx, location, block, ev, ability_id, next.result(ctx), high);
        push_return(ctx, location, block, slot);
        single_block_region(ctx, location, block)
    };
    let lower_half = {
        let block = empty_block(ctx, location);
        let slot = find_slot(ctx, location, block, ev, ability_id, low, mid);
        push_return(ctx, location, block, slot);
        single_block_region(ctx, location, block)
    };
    push_if(
        ctx,
        location,
        body,
        below.result(ctx),
        upper_half,
        Some(lower_half),
    );
    push_unreachable(ctx, location, body);

    helper_function(ctx, location, FIND_SLOT, &params, i32_ty, body)
}

/// Generate the internal `__tribute_evidence_find_marker` helper.
///
/// Returns the top marker with the given ability_id, or null if the evidence
/// has none for that ability.
fn generate_evidence_lookup_function(ctx: &mut IrContext, location: Location) -> OpRef {
    let evidence_ty = evidence_ref_type(ctx);
    let i32_ty = intern_i32(ctx);
    let marker_ty = ability::marker_adt_type_ref(ctx);
    let params = [evidence_ty, i32_ty];
    let body = entry_block(ctx, location, &params);
    let [ev, ability_id] = ctx.block_args(body)[..] else {
        unreachable!("entry block has one argument per parameter")
    };

    let len = evidence_len(ctx, location, body, ev);
    let slot = find_whole_slot(ctx, location, body, ev, ability_id, len);
    let found = {
        let block = empty_block(ctx, location);
        let marker = marker_at(ctx, location, block, ev, slot);
        push_return(ctx, location, block, marker);
        single_block_region(ctx, location, block)
    };
    push_if_slot_holds(ctx, location, body, ev, slot, ability_id, found);

    let null = wasm_dialect::RefNull::operands()
        .heap_type("struct")
        .type_idx(MARKER_IDX)
        .results(marker_ty)
        .build(ctx, location);
    ctx.push_op(body, null.op_ref());
    push_return(ctx, location, body, null.result(ctx));

    helper_function(ctx, location, FIND_MARKER, &params, marker_ty, body)
}

/// Generate the internal `__tribute_evidence_set_marker` helper.
///
/// Creates a new array with the marker in its ability's slot. A slot of the
/// same ability is replaced; otherwise the marker is inserted at the position
/// that keeps the array sorted.
fn generate_evidence_set_function(ctx: &mut IrContext, location: Location) -> OpRef {
    let evidence_ty = evidence_ref_type(ctx);
    let marker_ty = ability::marker_adt_type_ref(ctx);
    let params = [evidence_ty, marker_ty];
    let body = entry_block(ctx, location, &params);
    let [ev, marker] = ctx.block_args(body)[..] else {
        unreachable!("entry block has one argument per parameter")
    };

    let ability_id = marker_ability_id(ctx, location, body, marker);
    let old_len = evidence_len(ctx, location, body, ev);
    let slot = find_whole_slot(ctx, location, body, ev, ability_id, old_len);
    let zero = i32_const(ctx, location, body, 0);

    // The ability's slot exists: copy the array and replace the slot.
    let replace = {
        let block = empty_block(ctx, location);
        let new_ev = new_evidence(ctx, location, block, old_len);
        copy_markers(ctx, location, block, (new_ev, zero), (ev, zero), old_len);
        set_marker_at(ctx, location, block, new_ev, slot, marker);
        push_return(ctx, location, block, new_ev);
        single_block_region(ctx, location, block)
    };
    push_if_slot_holds(ctx, location, body, ev, slot, ability_id, replace);

    // Otherwise insert the marker at `slot`.
    let one = i32_const(ctx, location, body, 1);
    let new_len = wasm_dialect::I32Add::operands(old_len, one).build(ctx, location);
    ctx.push_op(body, new_len.op_ref());
    let new_ev = new_evidence(ctx, location, body, new_len.result(ctx));
    copy_markers(ctx, location, body, (new_ev, zero), (ev, zero), slot);
    set_marker_at(ctx, location, body, new_ev, slot, marker);
    let after_slot = wasm_dialect::I32Add::operands(slot, one).build(ctx, location);
    ctx.push_op(body, after_slot.op_ref());
    let suffix_len = wasm_dialect::I32Sub::operands(old_len, slot)
        .results(intern_i32(ctx))
        .build(ctx, location);
    ctx.push_op(body, suffix_len.op_ref());
    copy_markers(
        ctx,
        location,
        body,
        (new_ev, after_slot.result(ctx)),
        (ev, slot),
        suffix_len.result(ctx),
    );
    push_return(ctx, location, body, new_ev);

    helper_function(ctx, location, SET_MARKER, &params, evidence_ty, body)
}

/// Generate the internal `__tribute_evidence_remove_marker` helper.
///
/// Creates a new array without the slot of the given ability_id. The evidence
/// must hold the ability.
fn generate_evidence_remove_function(ctx: &mut IrContext, location: Location) -> OpRef {
    let evidence_ty = evidence_ref_type(ctx);
    let i32_ty = intern_i32(ctx);
    let params = [evidence_ty, i32_ty];
    let body = entry_block(ctx, location, &params);
    let [ev, ability_id] = ctx.block_args(body)[..] else {
        unreachable!("entry block has one argument per parameter")
    };

    let old_len = evidence_len(ctx, location, body, ev);
    let slot = find_whole_slot(ctx, location, body, ev, ability_id, old_len);
    let zero = i32_const(ctx, location, body, 0);
    let one = i32_const(ctx, location, body, 1);
    let new_len = wasm_dialect::I32Sub::operands(old_len, one)
        .results(i32_ty)
        .build(ctx, location);
    ctx.push_op(body, new_len.op_ref());
    let new_ev = new_evidence(ctx, location, body, new_len.result(ctx));
    copy_markers(ctx, location, body, (new_ev, zero), (ev, zero), slot);
    let after_slot = wasm_dialect::I32Add::operands(slot, one).build(ctx, location);
    ctx.push_op(body, after_slot.op_ref());
    let suffix_len = wasm_dialect::I32Sub::operands(new_len.result(ctx), slot)
        .results(i32_ty)
        .build(ctx, location);
    ctx.push_op(body, suffix_len.op_ref());
    copy_markers(
        ctx,
        location,
        body,
        (new_ev, slot),
        (ev, after_slot.result(ctx)),
        suffix_len.result(ctx),
    );
    push_return(ctx, location, body, new_ev);

    helper_function(ctx, location, REMOVE_MARKER, &params, evidence_ty, body)
}

// =============================================================================
// Helper functions
// =============================================================================

/// Create the entry block of a generated helper with the given parameters.
fn entry_block(ctx: &mut IrContext, location: Location, params: &[TypeRef]) -> BlockRef {
    ctx.create_block(BlockData {
        location,
        args: params
            .iter()
            .map(|&ty| BlockArgData {
                ty,
                attrs: Default::default(),
            })
            .collect(),
        ops: smallvec![],
        parent_region: None,
    })
}

/// Wrap a terminated entry block into a generated helper's `wasm.func`.
fn helper_function(
    ctx: &mut IrContext,
    location: Location,
    name: &'static str,
    params: &[TypeRef],
    result: TypeRef,
    body: BlockRef,
) -> OpRef {
    let func_ty = wasm_dialect::func_sig(ctx, params.iter().copied(), [result]).as_type_ref();
    let body = single_block_region(ctx, location, body);
    wasm_dialect::Func::operands()
        .sym_name(name)
        .r#type(func_ty)
        .regions(body)
        .build(ctx, location)
        .op_ref()
}

fn i32_const(ctx: &mut IrContext, location: Location, block: BlockRef, value: i32) -> ValueRef {
    let i32_ty = intern_i32(ctx);
    let op = wasm_dialect::I32Const::operands()
        .value(value)
        .results(i32_ty)
        .build(ctx, location);
    ctx.push_op(block, op.op_ref());
    op.result(ctx)
}

fn evidence_len(
    ctx: &mut IrContext,
    location: Location,
    block: BlockRef,
    ev: ValueRef,
) -> ValueRef {
    let i32_ty = intern_i32(ctx);
    let op = wasm_dialect::ArrayLen::operands(ev)
        .results(i32_ty)
        .build(ctx, location);
    ctx.push_op(block, op.op_ref());
    op.result(ctx)
}

fn new_evidence(
    ctx: &mut IrContext,
    location: Location,
    block: BlockRef,
    len: ValueRef,
) -> ValueRef {
    let evidence_ty = evidence_ref_type(ctx);
    let op = wasm_dialect::ArrayNewDefault::operands(len)
        .type_idx(EVIDENCE_IDX)
        .results(evidence_ty)
        .build(ctx, location);
    ctx.push_op(block, op.op_ref());
    op.result(ctx)
}

fn marker_at(
    ctx: &mut IrContext,
    location: Location,
    block: BlockRef,
    ev: ValueRef,
    index: ValueRef,
) -> ValueRef {
    let marker_ty = ability::marker_adt_type_ref(ctx);
    let op = wasm_dialect::ArrayGet::operands(ev, index)
        .type_idx(EVIDENCE_IDX)
        .results(marker_ty)
        .build(ctx, location);
    ctx.push_op(block, op.op_ref());
    op.result(ctx)
}

fn set_marker_at(
    ctx: &mut IrContext,
    location: Location,
    block: BlockRef,
    ev: ValueRef,
    index: ValueRef,
    marker: ValueRef,
) {
    let op = wasm_dialect::ArraySet::operands(ev, index, marker)
        .type_idx(EVIDENCE_IDX)
        .build(ctx, location);
    ctx.push_op(block, op.op_ref());
}

fn marker_ability_id(
    ctx: &mut IrContext,
    location: Location,
    block: BlockRef,
    marker: ValueRef,
) -> ValueRef {
    let i32_ty = intern_i32(ctx);
    let op = wasm_dialect::StructGet::operands(marker)
        .type_idx(MARKER_IDX)
        .field_idx(MarkerField::AbilityId.index())
        .results(i32_ty)
        .build(ctx, location);
    ctx.push_op(block, op.op_ref());
    op.result(ctx)
}

/// `array.copy` of `len` markers between `(array, offset)` positions.
fn copy_markers(
    ctx: &mut IrContext,
    location: Location,
    block: BlockRef,
    (dst, dst_offset): (ValueRef, ValueRef),
    (src, src_offset): (ValueRef, ValueRef),
    len: ValueRef,
) {
    let op = wasm_dialect::ArrayCopy::operands(dst, dst_offset, src, src_offset, len)
        .dst_type_idx(EVIDENCE_IDX)
        .src_type_idx(EVIDENCE_IDX)
        .build(ctx, location);
    ctx.push_op(block, op.op_ref());
}

/// Call `__tribute_evidence_find_slot` over `ev[low..high]`.
fn find_slot(
    ctx: &mut IrContext,
    location: Location,
    block: BlockRef,
    ev: ValueRef,
    ability_id: ValueRef,
    low: ValueRef,
    high: ValueRef,
) -> ValueRef {
    let i32_ty = intern_i32(ctx);
    let call = wasm_dialect::Call::operands([ev, ability_id, low, high])
        .callee(SymbolPath::from(FIND_SLOT))
        .results([i32_ty])
        .build(ctx, location);
    ctx.push_op(block, call.op_ref());
    call.results(ctx)[0]
}

/// Call `__tribute_evidence_find_slot` over the whole evidence of length `len`.
fn find_whole_slot(
    ctx: &mut IrContext,
    location: Location,
    block: BlockRef,
    ev: ValueRef,
    ability_id: ValueRef,
    len: ValueRef,
) -> ValueRef {
    let zero = i32_const(ctx, location, block, 0);
    find_slot(ctx, location, block, ev, ability_id, zero, len)
}

/// Run `then` if `ev[slot]` exists and is the slot of `ability_id`:
/// `if slot < len { if ev[slot].ability_id == ability_id { then } }`.
fn push_if_slot_holds(
    ctx: &mut IrContext,
    location: Location,
    block: BlockRef,
    ev: ValueRef,
    slot: ValueRef,
    ability_id: ValueRef,
    then: RegionRef,
) {
    let i32_ty = intern_i32(ctx);
    let len = evidence_len(ctx, location, block, ev);
    let in_bounds = wasm_dialect::I32LtS::operands(slot, len)
        .results(i32_ty)
        .build(ctx, location);
    ctx.push_op(block, in_bounds.op_ref());
    let check_ability = {
        let inner = empty_block(ctx, location);
        let existing = marker_at(ctx, location, inner, ev, slot);
        let existing_id = marker_ability_id(ctx, location, inner, existing);
        let same = wasm_dialect::I32Eq::operands(existing_id, ability_id)
            .results(i32_ty)
            .build(ctx, location);
        ctx.push_op(inner, same.op_ref());
        push_if(ctx, location, inner, same.result(ctx), then, None);
        single_block_region(ctx, location, inner)
    };
    push_if(
        ctx,
        location,
        block,
        in_bounds.result(ctx),
        check_ability,
        None,
    );
}

/// Append a resultless `wasm.if`; a missing `otherwise` is an empty region.
fn push_if(
    ctx: &mut IrContext,
    location: Location,
    block: BlockRef,
    condition: ValueRef,
    then: RegionRef,
    otherwise: Option<RegionRef>,
) {
    let nil_ty = core::nil(ctx).as_type_ref();
    let otherwise = otherwise.unwrap_or_else(|| {
        let empty = empty_block(ctx, location);
        single_block_region(ctx, location, empty)
    });
    let op = wasm_dialect::If::operands(condition)
        .results([nil_ty])
        .regions(then, otherwise)
        .build(ctx, location);
    ctx.push_op(block, op.op_ref());
}

fn push_return(ctx: &mut IrContext, location: Location, block: BlockRef, value: ValueRef) {
    let op = wasm_dialect::Return::operands([value]).build(ctx, location);
    ctx.push_op(block, op.op_ref());
}

fn push_unreachable(ctx: &mut IrContext, location: Location, block: BlockRef) {
    let op = wasm_dialect::Unreachable::operands().build(ctx, location);
    ctx.push_op(block, op.op_ref());
}

/// Get the WASM reference type for Evidence (wasm.arrayref).
fn evidence_ref_type(ctx: &mut IrContext) -> TypeRef {
    super::type_converter::evidence_wasm_type(ctx)
}

/// Intern a `core.i32` type.
fn intern_i32(ctx: &mut IrContext) -> TypeRef {
    ctx.intern_type(TypeDataBuilder::new("core", "i32").build())
}

/// Intern a `wasm.anyref` type.
fn intern_anyref(ctx: &mut IrContext) -> TypeRef {
    ctx.intern_type(TypeDataBuilder::new("wasm", "anyref").build())
}

fn empty_block(ctx: &mut IrContext, location: Location) -> BlockRef {
    ctx.create_block(BlockData {
        location,
        args: vec![],
        ops: smallvec![],
        parent_region: None,
    })
}

fn single_block_region(ctx: &mut IrContext, location: Location, block: BlockRef) -> RegionRef {
    ctx.create_region(RegionData {
        location,
        blocks: smallvec![block],
        parent_op: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;

    const TYPES: &str = r#"  !Evidence = core.array<adt.struct<_Marker(ability_id: core.i32, prompt_tag: core.i32, tr_dispatch_fn: core.ptr, shadowed: core.ptr), {layout = "evidence_marker"}>, {layout = "evidence"}>
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
    %result = effect.dispatch_tail %ev, %payload {ability_ref = core.ability_ref<{name = "Console"}>, op_name = "read"} : tribute_rt.anyref
    func.return %result
  }"#;

    const CPS: &str = r#"  func.func @cps(%ev: !Evidence, %dispatch: !Closure, %resume: !Closure, %payload: tribute_rt.anyref) {
    effect.dispatch_cps %ev, %dispatch, %resume, %payload {ability_ref = core.ability_ref<{name = "State"}>, op_name = "get", answer_type = core.i32}
  }"#;

    const EXTEND: &str = r#"  func.func @install(%ev: !Evidence, %prompt: core.i32, %tr: !Closure) -> !Evidence {
    %extended = effect.extend %ev, %prompt, %tr {ability_ref = core.ability_ref<{name = "State"}>} : !Evidence
    func.return %extended
  }"#;

    const SELECT: &str = r#"  func.func @select(%ev: !Evidence) -> !Evidence {
    %masked = effect.mask %ev {ability_ref = core.ability_ref<{name = "State"}>} : !Evidence
    %dup = effect.dup %masked {ability_ref = core.ability_ref<{name = "State"}>} : !Evidence
    func.return %dup
  }"#;

    #[test]
    fn mask_and_dup_call_the_runtime_with_the_ability_id() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, &module_text(SELECT));
        lower(&mut ctx, module).unwrap();
        let printed = print_module(&ctx, module.op());

        assert_shared_dialect_only(&printed);
        let mut calls = Vec::new();
        let _ = trunk_ir::walk::walk_op::<()>(&ctx, module.op(), &mut |op| {
            if let Ok(call) = func::Call::from_op(&ctx, op) {
                calls.push((call.callee(&ctx).clone(), ctx.op_operands(op).len()));
            }
            std::ops::ControlFlow::Continue(trunk_ir::walk::WalkAction::Advance)
        });
        assert_eq!(
            calls,
            [
                (SymbolPath::from(evidence_abi::MASK), 2),
                (SymbolPath::from(evidence_abi::DUP), 2)
            ],
            "{printed}"
        );
        assert!(!printed.contains("@__tribute_evidence_extend"), "{printed}");
        assert!(!printed.contains("@__tribute_evidence_lookup"), "{printed}");
    }

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
        assert!(printed.contains("call_conv = \"tail\""), "{printed}");
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
                calls.push((call.callee(&ctx).clone(), ctx.op_operands(op).len()));
            }
            std::ops::ControlFlow::Continue(trunk_ir::walk::WalkAction::Advance)
        });
        assert_eq!(
            calls,
            [(SymbolPath::from(evidence_abi::EXTEND), 4)],
            "{printed}"
        );
    }

    #[test]
    fn malformed_cps_dispatch_is_rejected_without_mutation() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            &module_text(
                r#"  func.func @cps(%ev: !Evidence, %dispatch: !Closure, %resume: !Closure, %payload: tribute_rt.anyref) {
    %bad = effect.dispatch_cps %ev, %dispatch, %resume, %payload {ability_ref = core.ability_ref<{name = "State"}>, op_name = "get", answer_type = core.i32} : core.i32
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
                    .filter(|f| f.sym_name(&ctx) == "cps")
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
  func.func @__tribute_evidence_extend(%ev: wasm.arrayref, %id: core.i32, %prompt: core.i32, %tr: wasm.anyref) -> wasm.arrayref attributes {abi = "C"}
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
                "__tribute_evidence_find_slot",
                "__tribute_evidence_lookup",
                "__tribute_evidence_lookup_tr",
                "__tribute_evidence_set_marker",
            ]
        );
    }

    #[test]
    fn binding_mask_and_dup_adds_the_internal_helpers_they_use() {
        let bound = |declarations: &str| {
            let mut ctx = IrContext::new();
            let module = parse_test_module(
                &mut ctx,
                &format!("core.module @test {{\n{declarations}\n}}"),
            );
            bind_wasm_evidence_runtime(&mut ctx, module);
            let mut functions: Vec<_> = module
                .ops(&ctx)
                .iter()
                .map(|&op| {
                    assert!(ctx.op_has_regions(op), "helpers must have bodies");
                    let function = wasm_dialect::Func::from_op(&ctx, op).expect("bound wasm.func");
                    function.sym_name(&ctx).to_string()
                })
                .collect();
            functions.sort();
            functions
        };

        assert_eq!(
            bound(
                r#"  func.func @__tribute_evidence_dup(%ev: wasm.arrayref, %id: core.i32) -> wasm.arrayref attributes {abi = "C"}"#
            ),
            [
                "__tribute_evidence_dup",
                "__tribute_evidence_find_marker",
                "__tribute_evidence_find_slot",
                "__tribute_evidence_set_marker",
            ]
        );
        assert_eq!(
            bound(
                r#"  func.func @__tribute_evidence_mask(%ev: wasm.arrayref, %id: core.i32) -> wasm.arrayref attributes {abi = "C"}"#
            ),
            [
                "__tribute_evidence_find_marker",
                "__tribute_evidence_find_slot",
                "__tribute_evidence_mask",
                "__tribute_evidence_remove_marker",
                "__tribute_evidence_set_marker",
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
  wasm.func {sym_name = "__tribute_next_tag", type = wasm.func_sig<() -> core.i32>} {
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
