//! Shared target-neutral physicalization of convention-proven CPS signatures.
//!
//! The pass consumes exact callable metadata, validates the whole transfer surface, and
//! only then maps logical CPS `core.never` results to the shared empty-result
//! list used by target backends.

use rustc_hash::FxHashMap as HashMap;
use std::error::Error;
use std::fmt;
use std::ops::ControlFlow;
use tribute_ir::continuation_frame;

use tribute_ir::calling_convention::{
    CLOSURE_ENVIRONMENT_INDEX_ATTR, cps_closure_function_type,
    get_physical_closure_environment_index,
};
use tribute_ir::calling_convention::{
    CallingConvention, ConventionBearing, DynConventionBearing, get_physical_closure_convention,
};
use tribute_ir::dialect::adt;
use tribute_ir::dialect::{ability, closure, effect, tribute_rt};
use trunk_ir::context::{BlockArgData, BlockData, IrContext, RegionData};
use trunk_ir::dialect::{core, func};
use trunk_ir::op_interface::DynIndirectCallLike;
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::refs::{BlockRef, OpRef, TypeRef, ValueRef};
use trunk_ir::rewrite::Module;
use trunk_ir::smallvec::smallvec;
use trunk_ir::symbol_table::qualified_name;
use trunk_ir::types::{Attribute, AttributeMap, Location, TypeData};
use trunk_ir::walk::{WalkAction, walk_op};
use trunk_ir::{Symbol, SymbolPath};

const ROOT_MAIN_SYMBOL: &str = "__tribute_main";
const ROOT_DONE_K_SYMBOL: &str = "__tribute_done_k";
const ROOT_UNHANDLED_SYMBOL: &str = "__tribute_unhandled";
const ROOT_COMPLETION_CELL_NAME: &str = "__tribute_completion_cell";
const ERASED_COMPLETION_CELL_NAME: &str = "__tribute_erased_completion_cell";
const ROOT_COMPLETION_CELL_VALUE_FIELD: &str = "value";
const DELIMIT_DONE_PREFIX: &str = "__tribute_delimit_done";
const DELIMIT_UNHANDLED_PREFIX: &str = "__tribute_delimit_unhandled";

/// Per-parameter attribute recording a physical callable's entry ownership
/// contract; see `new-plans/rc.md` (proper-tail ownership transfer).
pub(crate) const OWNERSHIP_ATTR: &str = "tribute.ownership";
/// The [`OWNERSHIP_ATTR`] value: the caller supplies one ownership unit.
pub(crate) const CONSUMED: &str = "consumed";

/// The parameter attributes this boundary assigns to every input of a
/// module-internal physical callable with `convention`.
///
/// Every module-internal callable may be the target of a proper tail
/// transfer, so it consumes every parameter whatever its convention; the
/// marker is inert on unmanaged ones. Callables with a platform `abi` keep
/// the platform contract and take no marker.
pub(crate) fn physical_parameter_attrs(
    ctx: &mut IrContext,
    _convention: CallingConvention,
) -> AttributeMap {
    let mut attrs = AttributeMap::new();
    let ownership = ctx.string_attr(CONSUMED);
    attrs.view_mut(ctx).insert(OWNERSHIP_ATTR, ownership);
    attrs
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TargetAbiError(String);

impl TargetAbiError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for TargetAbiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for TargetAbiError {}

#[derive(Clone, Copy)]
struct FunctionIdentity {
    signature: TypeRef,
    convention: CallingConvention,
    environment_index: Option<usize>,
}

/// Whether a contract check sees logical CPS callables or their physicalized
/// form, which lacks the provenance physicalization consumed.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ContractPhase {
    Logical,
    Physical,
}

#[derive(Clone, Copy)]
struct RootFrameContract {
    reference: TypeRef,
    layout: TypeRef,
    done: TypeRef,
    dispatch: TypeRef,
}

/// Physicalize exact CPS callables without selecting target instructions.
///
/// Validation and conversion planning finish before existing IR is mutated, so
/// rejected modules remain textually unchanged.
pub fn lower_cps_signatures_to_physical(
    ctx: &mut IrContext,
    module: Module,
) -> Result<(), TargetAbiError> {
    if let Some(violation) =
        tribute_ir::calling_convention::verify_conventions(ctx, module.op()).first()
    {
        return Err(TargetAbiError::new(format!(
            "target ABI: {}: {}",
            ctx.op(violation.op).name.display(ctx),
            violation.error
        )));
    }
    let ops = collect_ops(ctx, module.op());
    let never = core::Never::params().intern(ctx).as_type_ref();
    let functions = collect_functions(ctx, &ops, never)?;
    validate_transfers(ctx, &ops, &functions, never)?;
    validate_dispatch_contracts(ctx, module)?;
    validate_root_entry(ctx, module, ContractPhase::Logical)?;

    let aliases = ctx.type_aliases().to_vec();
    let mut converter = PhysicalTypeConverter::new(ctx, never);
    let mut alias_updates = Vec::new();
    let mut function_types = Vec::new();
    let mut result_types = Vec::new();
    let mut attributes = Vec::new();
    let mut consumed_attributes = Vec::new();
    let mut consumed_conventions = Vec::new();
    let mut indirect_signatures = Vec::new();
    let mut block_args = Vec::new();
    let mut block_attributes = Vec::new();

    for (name, ty) in aliases {
        let converted = converter.convert_embedded(ty)?;
        if converted != ty {
            alias_updates.push((name, converted));
        }
    }

    for &op in &ops {
        if let Ok(function) = func::Func::from_op(converter.ctx, op) {
            let signature = function.r#type(converter.ctx);
            let convention = exact_convention(converter.ctx, op)?;
            let platform = converter.ctx.op_attrs(op).contains_key("abi");
            let converted = match convention {
                Some(convention) if platform => {
                    converter.convert_platform_callable(signature, convention)?
                }
                Some(convention) => converter.convert_callable(signature, convention)?,
                None => converter.convert_embedded(signature)?,
            };
            if converted != signature {
                function_types.push((op, converted));
            }
            // Environment provenance was validated against the signature
            // above; the physical signature carries the environment as an
            // ordinary input, so no later pass reads the recorded position.
            if converter
                .ctx
                .op_attrs(op)
                .contains_key(CLOSURE_ENVIRONMENT_INDEX_ATTR)
            {
                consumed_attributes.push((op, CLOSURE_ENVIRONMENT_INDEX_ATTR));
            }
        }

        for (index, ty) in converter
            .ctx
            .op_result_types(op)
            .to_vec()
            .into_iter()
            .enumerate()
        {
            let converted = if let Ok(constant) = func::Constant::from_op(converter.ctx, op) {
                if let Some(identity) =
                    function_for_symbol_optional(constant.func_ref(converter.ctx), &functions)
                {
                    validate_constant(converter.ctx, constant, identity, never)?;
                    converter.convert_callable(ty, identity.convention)?
                } else {
                    converter.convert_embedded(ty)?
                }
            } else {
                converter.convert_embedded(ty)?
            };
            if converted != ty {
                result_types.push((op, index as u32, converted));
            }
        }

        let original_attributes = converter.ctx.op(op).attributes.clone();
        let mut converted_attributes = original_attributes.clone();
        let indirect = DynIndirectCallLike::get(converter.ctx, op);
        let indirect_signature = indirect.and_then(|call| call.exact_signature(converter.ctx));
        if let Some(signature) = indirect_signature {
            let convention = exact_convention(converter.ctx, op)?.ok_or_else(|| {
                TargetAbiError::new(
                    "target ABI: indirect callable signature has no convention metadata",
                )
            })?;
            let signature = converter.convert_callable(signature, convention)?;
            if !func::CallIndirect::matches(converter.ctx, op)
                && !func::TailCallIndirect::matches(converter.ctx, op)
            {
                return Err(TargetAbiError::new(
                    "target ABI: indirect callable signature cannot be set",
                ));
            }
            func::remove_indirect_call_signature(converter.ctx, &mut converted_attributes);
            indirect_signatures.push((op, signature));
        }
        // Transfers were validated against their callee above. Closure
        // lowering still reads the convention of a closure transfer; no later
        // pass reads any other transfer's.
        if is_transfer(converter.ctx, op)
            && exact_convention(converter.ctx, op)?.is_some()
            && indirect
                .and_then(|call| call.indirect_callee(converter.ctx))
                .is_none_or(|callee| {
                    crate::closure_lower::physical_closure_type_for_callee(converter.ctx, callee)
                        .is_none()
                })
        {
            consumed_conventions.push(op);
        }
        let op_attributes: Vec<_> = converted_attributes
            .iter()
            .map(|(name, value)| (name, value.clone()))
            .collect();
        for (name, value) in op_attributes {
            if func::Func::matches(converter.ctx, op) && converter.ctx.str(name) == "type" {
                continue;
            }
            let converted = converter.convert_attribute(value)?;
            converted_attributes
                .view_mut(converter.ctx)
                .insert(name, converted);
        }
        for (name, value) in converted_attributes {
            if original_attributes.view(converter.ctx).get(name) != Some(&value) {
                attributes.push((op, name, value));
            }
        }

        let regions = converter
            .ctx
            .op_regions(op)
            .collect::<trunk_ir::RegionList>();
        for region in regions {
            let block_count = converter.ctx.region(region).blocks.len();
            for block_index in 0..block_count {
                let block = converter.ctx.region(region).blocks[block_index];
                for (index, argument) in converter
                    .ctx
                    .block(block)
                    .args
                    .clone()
                    .into_iter()
                    .enumerate()
                {
                    let mut converted_attrs = argument.attrs.clone();
                    for (name, value) in argument.attrs.iter() {
                        let value = converter.convert_attribute(value.clone())?;
                        converted_attrs.view_mut(converter.ctx).insert(name, value);
                    }
                    if converted_attrs != argument.attrs {
                        block_attributes.push((block, index, converted_attrs));
                    }
                    let converted = converter.convert_embedded(argument.ty)?;
                    if converted != argument.ty {
                        block_args.push((block, index as u32, converted));
                    }
                }
            }
        }
    }

    drop(converter);
    for (name, ty) in alias_updates {
        ctx.register_type_alias(name, ty);
    }
    for (op, ty) in function_types {
        ctx.op_attrs_mut(op).insert("type", Attribute::Type(ty));
    }
    for (op, index, ty) in result_types {
        ctx.set_op_result_type(op, index, ty);
    }
    for (op, name, value) in attributes {
        ctx.op_attrs_mut(op).insert(name, value);
    }
    for (op, name) in consumed_attributes {
        ctx.op_attrs_mut(op).remove(name);
    }
    for op in consumed_conventions {
        DynConventionBearing::get(ctx, op)
            .expect("only a ConventionBearing operation records a calling convention")
            .remove_calling_convention(ctx);
    }
    for (op, signature) in indirect_signatures {
        assert!(
            DynIndirectCallLike::get(ctx, op)
                .is_some_and(|call| call.set_exact_signature(ctx, signature))
        );
    }
    for (block, index, ty) in block_args {
        ctx.set_block_arg_type(block, index, ty);
    }
    for (block, index, attributes) in block_attributes {
        ctx.block_mut(block).args[index].attrs = attributes;
    }
    Ok(())
}

struct RootEntryContract {
    worker_op: OpRef,
    source_result: TypeRef,
    evidence_ty: TypeRef,
    frame: RootFrameContract,
}

fn validate_root_entry(
    ctx: &mut IrContext,
    module: Module,
    phase: ContractPhase,
) -> Result<Option<RootEntryContract>, TargetAbiError> {
    let expected_results = match phase {
        ContractPhase::Logical => vec![core::Never::params().intern(ctx).as_type_ref()],
        ContractPhase::Physical => vec![],
    };
    let expected_results = expected_results.as_slice();
    let Some(module_block) = module.first_block(ctx) else {
        return Ok(None);
    };
    let top_level_ops = ctx.block(module_block).ops.clone();
    let roots: Vec<_> = top_level_ops
        .iter()
        .copied()
        .filter(|&op| {
            func::Func::from_op(ctx, op).is_ok_and(|function| function.sym_name(ctx) == "main")
        })
        .collect();
    if roots.len() > 1 {
        return Err(TargetAbiError::new(
            "target root bridge: multiple immediate root `main` definitions",
        ));
    }
    let Some(&worker_op) = roots.first() else {
        return Ok(None);
    };

    if DynConventionBearing::of(ctx, worker_op) != Some(CallingConvention::Cps) {
        return Ok(None);
    }
    // The root contract fixes the source result of `main`.
    let source_result = core::Nil::params().intern(ctx).as_type_ref();
    let worker = func::Func::from_op(ctx, worker_op)
        .map_err(|_| TargetAbiError::new("target root bridge: main is not func.func"))?;
    let worker_callable = func::FuncSig::from_type_ref(ctx, worker.r#type(ctx))
        .ok_or_else(|| TargetAbiError::new("target root bridge: main is not func.func_sig"))?;
    let evidence_ty = ability::evidence_adt_type_ref(ctx);
    let worker_params = worker_callable.inputs(ctx);
    if worker_callable.results(ctx) != expected_results
        || worker_params.len() != 2
        || worker_params[0] != evidence_ty
    {
        return Err(TargetAbiError::new(
            "target root bridge: Cps root must have exact evidence and ContinuationFrame ABI",
        ));
    }
    let frame = validate_root_continuation_frame(
        ctx,
        worker_params[1],
        source_result,
        evidence_ty,
        expected_results,
        phase,
    )?;
    if !ctx.op_has_regions(worker_op) {
        return Err(TargetAbiError::new(
            "target root bridge: root worker must be a definition",
        ));
    }

    let root_done_k = Symbol::new(ROOT_DONE_K_SYMBOL);
    let root_dispatch = Symbol::new(ROOT_UNHANDLED_SYMBOL);
    for &op in &top_level_ops {
        let Ok(function) = func::Func::from_op(ctx, op) else {
            continue;
        };
        if matches!(function.sym_name(ctx), name if name == root_done_k || name == root_dispatch) {
            return Err(TargetAbiError::new(
                "target root bridge: reserved root symbol collision",
            ));
        }
    }

    Ok(Some(RootEntryContract {
        worker_op,
        source_result,
        evidence_ty,
        frame,
    }))
}

/// Compose the root entry bridge after physicalization.
///
/// The source root `main` becomes the root worker whatever its calling
/// convention, and a synthesized parameterless Direct `main` supplies the
/// worker's convention-specific inputs. Nothing in the module references the
/// wrapper, so target entry generation adapts it in place and never reads the
/// source calling convention.
pub fn compose_root_entry_bridge(
    ctx: &mut IrContext,
    module: Module,
) -> Result<(), TargetAbiError> {
    let cps_root = validate_root_entry(ctx, module, ContractPhase::Physical)?;
    let Some(module_block) = module.first_block(ctx) else {
        return Ok(());
    };
    let top_level_ops = ctx.block(module_block).ops.clone();
    let main = SymbolPath::new(["main"]);
    let root_main = SymbolPath::new([ROOT_MAIN_SYMBOL]);
    let mut roots = top_level_ops.iter().copied().filter(|&op| {
        func::Func::from_op(ctx, op).is_ok_and(|function| function.sym_name(ctx) == "main")
    });
    let Some(worker_op) = roots.next() else {
        return Ok(());
    };
    if top_level_ops.iter().any(|&op| {
        func::Func::from_op(ctx, op)
            .is_ok_and(|function| function.sym_name(ctx) == ROOT_MAIN_SYMBOL)
    }) {
        return Err(TargetAbiError::new(
            "target root bridge: reserved root symbol collision",
        ));
    }
    if !ctx.op_has_regions(worker_op) {
        return Err(TargetAbiError::new(
            "target root bridge: root `main` must be a definition",
        ));
    }
    let worker = func::Func::from_op(ctx, worker_op).expect("filtered root function");
    let signature = func::FuncSig::from_type_ref(ctx, worker.r#type(ctx))
        .ok_or_else(|| TargetAbiError::new("target root bridge: main is not func.func_sig"))?;
    let inputs = signature.inputs(ctx).to_vec();
    let results = signature.results(ctx).to_vec();
    let nil_ty = core::Nil::params().intern(ctx).as_type_ref();
    let evidence_ty = ability::evidence_adt_type_ref(ctx);
    let convention = DynConventionBearing::of(ctx, worker_op);
    match (convention, &cps_root) {
        (Some(CallingConvention::Cps), Some(_)) => {}
        (Some(CallingConvention::EvidenceDirect), None) if inputs == [evidence_ty] => {}
        (None | Some(CallingConvention::Direct), None) if inputs.is_empty() => {}
        _ => {
            return Err(TargetAbiError::new(
                "target root bridge: root `main` has no exact entry ABI for its calling convention",
            ));
        }
    }
    if cps_root.is_none() && results != [nil_ty] {
        return Err(TargetAbiError::new(
            "target root bridge: root `main` must return core.nil",
        ));
    }

    let location = ctx.op(worker_op).location;
    let root_main_name = ctx.intern_str(ROOT_MAIN_SYMBOL);
    ctx.op_attrs_mut(worker_op)
        .insert("sym_name", Attribute::String(root_main_name));
    for &op in &top_level_ops {
        rewrite_symbol_refs(ctx, op, &main, &root_main);
    }

    let entry = ctx.create_block(BlockData {
        location,
        args: vec![],
        ops: smallvec![],
        parent_region: None,
    });
    let returned = match cps_root {
        Some(contract) => build_cps_root_call(ctx, module_block, entry, contract, root_main)?,
        None => {
            let args = if convention == Some(CallingConvention::EvidenceDirect) {
                vec![build_initial_evidence(ctx, entry, location, evidence_ty)]
            } else {
                vec![]
            };
            let call = func::Call::operands(args)
                .callee(root_main)
                .results([nil_ty])
                .build(ctx, location);
            ctx.push_op(entry, call.op_ref());
            ctx.op_results(call.op_ref())[0]
        }
    };
    let ret = func::Return::operands([returned]).build(ctx, location);
    ctx.push_op(entry, ret.op_ref());
    let body = ctx.create_region(RegionData {
        location,
        blocks: smallvec![entry],
        parent_op: None,
    });
    let wrapper_ty = func::func_sig(ctx, [], [nil_ty]).as_type_ref();
    let wrapper = func::Func::operands()
        .sym_name(main.to_string())
        .r#type(wrapper_ty)
        .regions(body)
        .build(ctx, location);
    wrapper.set_calling_convention(ctx, CallingConvention::Direct);
    ctx.push_op(module_block, wrapper.op_ref());
    Ok(())
}

/// Call a Cps root worker through the target-independent export delimiter.
///
/// The worker and exact frame members have empty results after
/// physicalization. The caller owns a completion cell, passes the worker a
/// frame capturing it, and reads the source result once the ordinary call
/// returns.
fn build_cps_root_call(
    ctx: &mut IrContext,
    module_block: BlockRef,
    entry: BlockRef,
    contract: RootEntryContract,
    worker: SymbolPath,
) -> Result<ValueRef, TargetAbiError> {
    let RootEntryContract {
        worker_op,
        source_result,
        evidence_ty,
        frame,
    } = contract;
    let location = ctx.op(worker_op).location;

    let terminal = TerminalFunctions::define(
        ctx,
        location,
        frame,
        source_result,
        Symbol::new(ROOT_DONE_K_SYMBOL),
        Symbol::new(ROOT_UNHANDLED_SYMBOL),
    )?;
    let built = terminal.build_frame(ctx, location);
    for &op in &built.ops {
        ctx.push_op(entry, op);
    }
    let evidence = build_initial_evidence(ctx, entry, location, evidence_ty);
    let worker_call = func::Call::operands([evidence, built.frame])
        .callee(worker)
        .results([])
        .build(ctx, location);
    ctx.push_op(entry, worker_call.op_ref());
    let mut completed = None;
    for op in terminal.read_answer(ctx, location, built.cell) {
        ctx.push_op(entry, op);
        completed = Some(ctx.op_result(op, 0));
    }

    for function in terminal.functions {
        ctx.push_op(module_block, function);
    }
    Ok(completed.expect("reading the cell produces the answer"))
}

/// Replace every `effect.delimit` with its completion cell, terminal frame,
/// the call of its body, and the read of the answer, and every
/// `effect.initial_evidence` with the target's initial evidence.
///
/// Runs after physicalization, like the root bridge: the body and the frame
/// members have empty results, so the body is entered by an ordinary call
/// that returns when its proper tail chain ends.
pub fn compose_value_delimiters(ctx: &mut IrContext, module: Module) -> Result<(), TargetAbiError> {
    let Some(module_block) = module.first_block(ctx) else {
        return Ok(());
    };
    let operations = collect_ops(ctx, module.op());
    let evidence_ty = ability::evidence_adt_type_ref(ctx);
    let initials: Vec<_> = operations
        .iter()
        .filter_map(|&op| effect::InitialEvidence::from_op(ctx, op).ok())
        .collect();
    for initial in initials {
        let op = initial.op_ref();
        let location = ctx.op(op).location;
        let block = ctx
            .op(op)
            .parent_block
            .ok_or_else(|| TargetAbiError::new("initial evidence: operation is detached"))?;
        let scratch = ctx.create_block(BlockData {
            location,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let evidence = build_initial_evidence(ctx, scratch, location, evidence_ty);
        for built in ctx.block(scratch).ops.clone() {
            ctx.detach_op(built);
            ctx.insert_op_before(block, op, built);
        }
        ctx.replace_all_uses(initial.result(ctx), evidence);
        ctx.detach_op(op);
        ctx.remove_op(op);
    }
    let delimiters: Vec<_> = operations
        .into_iter()
        .filter_map(|op| effect::Delimit::from_op(ctx, op).ok())
        .collect();
    let mut terminals: Vec<(TypeRef, TerminalFunctions)> = Vec::new();
    for delimit in delimiters {
        let op = delimit.op_ref();
        let location = ctx.op(op).location;
        let body = delimit.body(ctx);
        let answer = delimit.result_ty(ctx);
        let malformed = || TargetAbiError::new("value delimiter: body is not an exact Cps closure");
        let body_ty = crate::closure_lower::physical_closure_type_for_callee(ctx, body)
            .filter(|ty| get_physical_closure_convention(ctx, *ty) == Some(CallingConvention::Cps))
            .ok_or_else(malformed)?;
        let signature = cps_closure_function_type(ctx, body_ty).ok_or_else(malformed)?;
        let callable = func::FuncSig::from_type_ref(ctx, signature).ok_or_else(malformed)?;
        let &[body_evidence, frame_ty] = callable.inputs(ctx) else {
            return Err(malformed());
        };
        if body_evidence != evidence_ty || !callable.results(ctx).is_empty() {
            return Err(malformed());
        }
        let index = match terminals.iter().position(|(frame, _)| *frame == frame_ty) {
            Some(index) => index,
            None => {
                let frame = validate_root_continuation_frame(
                    ctx,
                    frame_ty,
                    answer,
                    evidence_ty,
                    &[],
                    ContractPhase::Physical,
                )?;
                let index = terminals.len();
                let functions = TerminalFunctions::define(
                    ctx,
                    location,
                    frame,
                    answer,
                    Symbol::new(&format!("{DELIMIT_DONE_PREFIX}_{index}")),
                    Symbol::new(&format!("{DELIMIT_UNHANDLED_PREFIX}_{index}")),
                )?;
                terminals.push((frame_ty, functions));
                index
            }
        };
        let terminal = &terminals[index].1;
        if terminal.answer != answer {
            return Err(TargetAbiError::new(
                "value delimiter: answer differs from the answer of its body's frame",
            ));
        }
        let block = ctx
            .op(op)
            .parent_block
            .ok_or_else(|| TargetAbiError::new("value delimiter: operation is detached"))?;
        let built = terminal.build_frame(ctx, location);
        for &new in &built.ops {
            ctx.insert_op_before(block, op, new);
        }
        let evidence = delimit.evidence(ctx);
        let call = func::CallIndirect::operands(body, [evidence, built.frame])
            .signature(signature)
            .build(ctx, location);
        call.set_calling_convention(ctx, CallingConvention::Cps);
        ctx.insert_op_before(block, op, call.op_ref());
        let mut completed = None;
        for new in terminal.read_answer(ctx, location, built.cell) {
            ctx.insert_op_before(block, op, new);
            completed = Some(ctx.op_result(new, 0));
        }
        let completed = completed.expect("reading the cell produces the answer");
        ctx.replace_all_uses(delimit.result(ctx), completed);
        ctx.detach_op(op);
        ctx.remove_op(op);
    }
    for (_, terminal) in terminals {
        for function in terminal.functions {
            ctx.push_op(module_block, function);
        }
    }
    Ok(())
}

/// The terminal `Done<R>` and `Dispatch<R>` of a completion cell: `Done`
/// stores the answer in the cell, and `Dispatch` is never reached.
struct TerminalFunctions {
    frame: RootFrameContract,
    answer: TypeRef,
    cell_ty: TypeRef,
    done: Symbol,
    dispatch: Symbol,
    /// The definitions of `done` and `dispatch`, not yet in the module.
    functions: [OpRef; 2],
}

/// A completion cell and the frame that completes into it.
struct TerminalFrame {
    ops: Vec<OpRef>,
    cell: ValueRef,
    frame: ValueRef,
}

impl TerminalFunctions {
    /// A `core.nil` answer is stored as it is. Any other answer is stored
    /// erased, because the cell needs a value before the answer exists.
    fn slot_type(ctx: &mut IrContext, answer: TypeRef) -> TypeRef {
        if answer == core::Nil::params().intern(ctx).as_type_ref() {
            answer
        } else {
            tribute_rt::Anyref::params().intern(ctx).as_type_ref()
        }
    }

    fn define(
        ctx: &mut IrContext,
        location: Location,
        frame: RootFrameContract,
        answer: TypeRef,
        done: Symbol,
        dispatch: Symbol,
    ) -> Result<Self, TargetAbiError> {
        let slot = Self::slot_type(ctx, answer);
        let cell_ty = completion_cell_type(ctx, slot);
        let contract = physical_parameter_attrs(ctx, CallingConvention::Cps);
        // Each function's environment is its own closure; `Done` captures the
        // cell and `Dispatch` captures nothing.
        let done_ty = func::func_sig_with_param_attrs(
            ctx,
            [(frame.done, contract.clone()), (answer, contract)],
            [],
            AttributeMap::new(),
        )
        .with_call_conv(ctx, func::CallConv::Tail)
        .as_type_ref();
        let environment_name = bind_name(ctx, "__env");
        let argument_name = bind_name(ctx, "__arg");
        let answer_name = bind_name(ctx, "__answer");
        let done_entry = ctx.create_block(BlockData {
            location,
            args: vec![
                BlockArgData {
                    ty: frame.done,
                    attrs: environment_name.clone(),
                },
                BlockArgData {
                    ty: answer,
                    attrs: answer_name,
                },
            ],
            ops: smallvec![],
            parent_region: None,
        });
        let done_args = ctx.block_args(done_entry).to_vec();
        let cell = closure::Unpack::operands(done_args[0])
            .results([cell_ty])
            .build(ctx, location);
        ctx.push_op(done_entry, cell.op_ref());
        let cell = cell.results(ctx)[0];
        let stored = if slot == answer {
            done_args[1]
        } else {
            let erased = tribute_rt::convert_value(ctx, location, done_args[1], slot);
            ctx.push_op(done_entry, erased);
            ctx.op_result(erased, 0)
        };
        let store = adt::StructSet::operands(cell, stored)
            .r#type(cell_ty)
            .field(0)
            .build(ctx, location);
        ctx.push_op(done_entry, store.op_ref());
        let done_return = func::Return::operands([]).build(ctx, location);
        ctx.push_op(done_entry, done_return.op_ref());
        let done_region = ctx.create_region(RegionData {
            location,
            blocks: smallvec![done_entry],
            parent_op: None,
        });
        let done_function = func::Func::operands()
            .sym_name(ctx.intern_symbol_text(&done))
            .r#type(done_ty)
            .regions(done_region)
            .build(ctx, location);
        done_function.set_calling_convention(ctx, CallingConvention::Cps);

        let dispatch_ty = dispatch_entry_function_type(ctx, frame.dispatch)?;
        let dispatch_entry = ctx.create_block(BlockData {
            location,
            args: func::FuncSig::from_type_ref(ctx, dispatch_ty)
                .expect("validated dispatch entry contract")
                .inputs(ctx)
                .iter()
                .copied()
                .enumerate()
                .map(|(index, ty)| BlockArgData {
                    ty,
                    attrs: if index == 1 {
                        environment_name.clone()
                    } else {
                        argument_name.clone()
                    },
                })
                .collect(),
            ops: smallvec![],
            parent_region: None,
        });
        let dispatch_unreachable = func::Unreachable::operands().build(ctx, location);
        ctx.push_op(dispatch_entry, dispatch_unreachable.op_ref());
        let dispatch_region = ctx.create_region(RegionData {
            location,
            blocks: smallvec![dispatch_entry],
            parent_op: None,
        });
        let dispatch_function = func::Func::operands()
            .sym_name(ctx.intern_symbol_text(&dispatch))
            .r#type(dispatch_ty)
            .regions(dispatch_region)
            .build(ctx, location);
        dispatch_function.set_calling_convention(ctx, CallingConvention::Cps);

        Ok(Self {
            frame,
            answer,
            cell_ty,
            done,
            dispatch,
            functions: [done_function.op_ref(), dispatch_function.op_ref()],
        })
    }

    /// Build a fresh cell and the frame whose `Done` writes it.
    fn build_frame(&self, ctx: &mut IrContext, location: Location) -> TerminalFrame {
        let mut ops = Vec::new();
        let anyref_ty = tribute_rt::Anyref::params().intern(ctx).as_type_ref();
        let initial = if Self::slot_type(ctx, self.answer) != anyref_ty {
            core::NilValue::operands().build(ctx, location).op_ref()
        } else {
            adt::RefNull::operands()
                .r#type(anyref_ty)
                .results(anyref_ty)
                .build(ctx, location)
                .op_ref()
        };
        ops.push(initial);
        let cell_new = adt::StructNew::operands([ctx.op_result(initial, 0)])
            .r#type(self.cell_ty)
            .results(self.cell_ty)
            .build(ctx, location);
        ops.push(cell_new.op_ref());
        let mut member = |ctx: &mut IrContext, symbol: &Symbol, captures: &[ValueRef], ty| {
            let closure = closure::New::operands(captures.to_vec())
                .func_ref(SymbolPath::new([symbol.clone()]))
                .results(ty)
                .build(ctx, location);
            ops.push(closure.op_ref());
            closure.result(ctx)
        };
        let done = member(ctx, &self.done, &[cell_new.result(ctx)], self.frame.done);
        let dispatch = member(ctx, &self.dispatch, &[], self.frame.dispatch);
        let frame = adt::StructNew::operands([done, dispatch])
            .r#type(self.frame.layout)
            .results(self.frame.reference)
            .build(ctx, location);
        ops.push(frame.op_ref());
        TerminalFrame {
            ops,
            cell: cell_new.result(ctx),
            frame: frame.result(ctx),
        }
    }

    /// The operations that read the answer out of `cell`; the last one
    /// produces it.
    fn read_answer(&self, ctx: &mut IrContext, location: Location, cell: ValueRef) -> Vec<OpRef> {
        let slot = Self::slot_type(ctx, self.answer);
        let stored = adt::StructGet::operands(cell)
            .r#type(self.cell_ty)
            .field(0)
            .results(slot)
            .build(ctx, location);
        let mut ops = vec![stored.op_ref()];
        if slot != self.answer {
            ops.push(tribute_rt::convert_value(
                ctx,
                location,
                stored.result(ctx),
                self.answer,
            ));
        }
        ops
    }
}

/// Build the target's initial evidence: an empty evidence array.
fn build_initial_evidence(
    ctx: &mut IrContext,
    block: BlockRef,
    location: Location,
    evidence_ty: TypeRef,
) -> ValueRef {
    let empty = adt::ArrayOf::operands([])
        .r#type(evidence_ty)
        .build(ctx, location);
    ctx.push_op(block, empty.op_ref());
    empty.result(ctx)
}

fn completion_cell_type(ctx: &mut IrContext, value_ty: TypeRef) -> TypeRef {
    let name = if value_ty == core::Nil::params().intern(ctx).as_type_ref() {
        ROOT_COMPLETION_CELL_NAME
    } else {
        ERASED_COMPLETION_CELL_NAME
    };
    adt::Struct::from_fields(
        ctx,
        name,
        [(ROOT_COMPLETION_CELL_VALUE_FIELD, value_ty)],
        AttributeMap::new(),
    )
    .as_type_ref()
}

/// Recover semantic R only from authenticated callable and nominal frame metadata.
pub(crate) fn dispatch_answer_type(
    ctx: &mut IrContext,
    evidence: ValueRef,
    dispatch: ValueRef,
    resume: ValueRef,
) -> Result<TypeRef, TargetAbiError> {
    let evidence_type = ability::evidence_adt_type_ref(ctx);
    if ctx.value_ty(evidence) != evidence_type {
        return Err(TargetAbiError::new(
            "CPS dispatch evidence differs from canonical Evidence type",
        ));
    }
    let dispatch = crate::closure_lower::physical_closure_type_for_callee(ctx, dispatch)
        .ok_or_else(|| {
            TargetAbiError::new("CPS dispatch lacks authenticated callable provenance")
        })?;
    let resume = crate::closure_lower::physical_closure_type_for_callee(ctx, resume)
        .ok_or_else(|| TargetAbiError::new("CPS resume lacks authenticated callable provenance"))?;
    let resume_signature = cps_closure_function_type(ctx, resume)
        .and_then(|ty| func::FuncSig::from_type_ref(ctx, ty))
        .ok_or_else(|| TargetAbiError::new("CPS resume lacks exact callable signature"))?;
    let frame = *resume_signature
        .inputs(ctx)
        .get(1)
        .ok_or_else(|| TargetAbiError::new("CPS resume lacks frame input"))?;
    let answer = continuation_frame::result_type(ctx, frame)
        .ok_or_else(|| TargetAbiError::new("CPS resume frame lacks answer type"))?;
    let results = resume_signature.results(ctx);
    if !(results.is_empty()
        || (results.len() == 1
            && is_parameterless_type(ctx, results[0], trunk_ir::dialect::core::Never::NAME)))
    {
        return Err(TargetAbiError::new(
            "CPS resume must have logical never or physical empty results",
        ));
    }
    let contract = validate_root_continuation_frame(
        ctx,
        frame,
        answer,
        ctx.value_ty(evidence),
        results,
        ContractPhase::Logical,
    )?;
    if contract.dispatch != dispatch {
        return Err(TargetAbiError::new(
            "CPS dispatch differs from exact nominal frame Dispatch",
        ));
    }
    let signature = dispatch_callable_function_type(ctx, dispatch)?;
    let signature = func::FuncSig::from_type_ref(ctx, signature).unwrap();
    if signature.inputs(ctx).get(1) != Some(&resume) {
        return Err(TargetAbiError::new(
            "CPS dispatch resume differs from exact required resume",
        ));
    }
    Ok(answer)
}

/// Validate before closure storage erasure or physicalization changes any surface.
pub(crate) fn validate_dispatch_contracts(
    ctx: &mut IrContext,
    module: Module,
) -> Result<(), TargetAbiError> {
    for op in collect_ops(ctx, module.op()) {
        if !effect::DispatchCps::matches(ctx, op) {
            continue;
        }
        let operands = ctx.op_operands(op).to_vec();
        let [evidence, dispatch, resume, payload] = operands.as_slice() else {
            return Err(TargetAbiError::new("CPS dispatch requires four operands"));
        };
        let packed_payload = tribute_rt::Anyref::params().intern(ctx).as_type_ref();
        if ctx.value_ty(*payload) != packed_payload {
            return Err(TargetAbiError::new(
                "CPS dispatch packed payload must have exact tribute_rt.anyref type",
            ));
        }
        let answer = ctx
            .op_attrs(op)
            .get_type("answer_type")
            .ok_or_else(|| TargetAbiError::new("CPS dispatch requires answer_type: Type"))?;
        if !ctx.op_result_types(op).is_empty()
            || dispatch_answer_type(ctx, *evidence, *dispatch, *resume)? != answer
        {
            return Err(TargetAbiError::new(
                "CPS dispatch answer_type differs from resume frame R",
            ));
        }
    }
    Ok(())
}

fn validate_root_continuation_frame(
    ctx: &IrContext,
    frame: TypeRef,
    source_result: TypeRef,
    evidence: TypeRef,
    physical_results: &[TypeRef],
    phase: ContractPhase,
) -> Result<RootFrameContract, TargetAbiError> {
    let reference_data = ctx.get_type(frame);
    if reference_data.name != tribute_ir::dialect::adt::Typeref::NAME {
        return Err(TargetAbiError::new(
            "target root bridge: worker frame must be an exact nominal adt.typeref",
        ));
    }
    // Physicalization consumes the frame answer provenance after checking it
    // here; the physical frame's Done input then carries the source result.
    let expected_provenance = (phase == ContractPhase::Logical).then_some(source_result);
    if continuation_frame::result_type(ctx, frame) != expected_provenance {
        return Err(TargetAbiError::new(
            "target root bridge: worker frame result provenance differs from root source result",
        ));
    }
    let name = ctx.type_attrs(frame).get_str("name").ok_or_else(|| {
        TargetAbiError::new("target root bridge: worker frame lacks nominal layout identity")
    })?;
    let layout = ctx.type_alias_by_text(name).ok_or_else(|| {
        TargetAbiError::new("target root bridge: worker frame must have an exact nominal layout")
    })?;
    let layout_struct = adt::Struct::from_type_ref(ctx, layout)
        .filter(|layout_struct| layout_struct.name(ctx) == name)
        .filter(|_| continuation_frame::result_type(ctx, layout) == expected_provenance)
        .ok_or_else(|| {
            TargetAbiError::new("target root bridge: worker frame layout provenance is malformed")
        })?;
    let fields: Vec<_> = layout_struct.fields(ctx).collect();
    let [(done_name, done), (dispatch_name, dispatch)] = fields.as_slice() else {
        return Err(TargetAbiError::new(
            "target root bridge: worker frame layout must contain done then dispatch",
        ));
    };
    if *done_name != "done" || *dispatch_name != "dispatch" {
        return Err(TargetAbiError::new(
            "target root bridge: worker frame field roles must be exact Done then Dispatch",
        ));
    }
    validate_root_done_type(ctx, *done, source_result, physical_results)?;
    validate_root_dispatch_type(ctx, *dispatch, frame, evidence, physical_results)?;
    Ok(RootFrameContract {
        reference: frame,
        layout,
        done: *done,
        dispatch: *dispatch,
    })
}

fn validate_root_done_type(
    ctx: &IrContext,
    done: TypeRef,
    source_result: TypeRef,
    physical_results: &[TypeRef],
) -> Result<(), TargetAbiError> {
    if get_physical_closure_convention(ctx, done) != Some(CallingConvention::Cps)
        || get_physical_closure_environment_index(ctx, done) != Some(0)
    {
        return Err(TargetAbiError::new(
            "target root bridge: frame Done must carry exact Cps closure provenance",
        ));
    }
    let callable = cps_closure_function_type(ctx, done).ok_or_else(|| {
        TargetAbiError::new("target root bridge: frame Done is not an exact Cps closure")
    })?;
    let callable = func::FuncSig::from_type_ref(ctx, callable).ok_or_else(|| {
        TargetAbiError::new("target root bridge: frame Done callable is not func.func_sig")
    })?;
    if callable.results(ctx) != physical_results || callable.inputs(ctx) != [source_result] {
        return Err(TargetAbiError::new(
            "target root bridge: frame Done must accept the exact source result and return empty",
        ));
    }
    Ok(())
}

fn validate_root_dispatch_type(
    ctx: &IrContext,
    dispatch: TypeRef,
    frame: TypeRef,
    evidence: TypeRef,
    physical_results: &[TypeRef],
) -> Result<(), TargetAbiError> {
    if get_physical_closure_convention(ctx, dispatch) != Some(CallingConvention::Cps)
        || get_physical_closure_environment_index(ctx, dispatch) != Some(1)
    {
        return Err(TargetAbiError::new(
            "target root bridge: frame Dispatch must carry exact Cps closure provenance",
        ));
    }
    let callable = dispatch_callable_function_type(ctx, dispatch)?;
    let callable =
        func::FuncSig::from_type_ref(ctx, callable).expect("validated dispatch callable");
    let params = callable.inputs(ctx);
    let [actual_evidence, resume, prompt, ability, operation, payload] = params else {
        return Err(TargetAbiError::new(
            "target root bridge: frame Dispatch must have the exact terminal dispatch ABI",
        ));
    };
    if callable.results(ctx) != physical_results
        || *actual_evidence != evidence
        || !is_parameterless_type(ctx, *prompt, trunk_ir::dialect::core::I32::NAME)
        || !is_parameterless_type(ctx, *ability, trunk_ir::dialect::core::I32::NAME)
        || !is_parameterless_type(ctx, *operation, trunk_ir::dialect::core::I32::NAME)
        || !is_parameterless_type(ctx, *payload, tribute_ir::dialect::tribute_rt::Anyref::NAME)
    {
        return Err(TargetAbiError::new(
            "target root bridge: frame Dispatch operands differ from the exact terminal ABI",
        ));
    }
    if get_physical_closure_convention(ctx, *resume) != Some(CallingConvention::Cps)
        || get_physical_closure_environment_index(ctx, *resume) != Some(0)
    {
        return Err(TargetAbiError::new(
            "target root bridge: frame Dispatch resume must carry exact Cps closure provenance",
        ));
    }
    let resume = cps_closure_function_type(ctx, *resume).ok_or_else(|| {
        TargetAbiError::new("target root bridge: frame Dispatch resume is not an exact Cps closure")
    })?;
    let resume = func::FuncSig::from_type_ref(ctx, resume).ok_or_else(|| {
        TargetAbiError::new(
            "target root bridge: frame Dispatch resume callable is not func.func_sig",
        )
    })?;
    if resume.results(ctx) != physical_results
        || resume.inputs(ctx).len() != 3
        || resume.inputs(ctx)[0] != evidence
        || resume.inputs(ctx)[1] != frame
        || !is_parameterless_type(
            ctx,
            resume.inputs(ctx)[2],
            tribute_ir::dialect::tribute_rt::Anyref::NAME,
        )
    {
        return Err(TargetAbiError::new(
            "target root bridge: frame Dispatch resume differs from the exact frame ABI",
        ));
    }
    Ok(())
}

fn dispatch_callable_function_type(
    ctx: &IrContext,
    dispatch: TypeRef,
) -> Result<TypeRef, TargetAbiError> {
    cps_closure_function_type(ctx, dispatch).ok_or_else(|| {
        TargetAbiError::new("target root bridge: frame Dispatch is not an exact Cps closure")
    })
}

fn dispatch_entry_function_type(
    ctx: &mut IrContext,
    dispatch: TypeRef,
) -> Result<TypeRef, TargetAbiError> {
    let callable_ty = dispatch_callable_function_type(ctx, dispatch)?;
    let callable = func::FuncSig::from_type_ref(ctx, callable_ty).ok_or_else(|| {
        TargetAbiError::new("target root bridge: frame Dispatch callable is not func.func_sig")
    })?;
    let environment_attrs = physical_parameter_attrs(ctx, CallingConvention::Cps);
    Ok(callable
        .rebuild(ctx, |inputs, _| {
            inputs.insert(1, (dispatch, environment_attrs));
        })
        .as_type_ref())
}

fn is_parameterless_type(ctx: &IrContext, ty: TypeRef, name: trunk_ir::TypeName) -> bool {
    let data = ctx.get_type(ty);
    data.name == name && data.params.is_empty() && data.attrs.is_empty()
}

fn bind_name(ctx: &mut IrContext, name: &str) -> AttributeMap {
    let name = ctx.string_attr(name);
    AttributeMap::from_entries(ctx, [("bind_name", name)])
}

fn rewrite_symbol_refs(ctx: &mut IrContext, op: OpRef, old: &SymbolPath, new: &SymbolPath) {
    if core::Module::from_op(ctx, op).is_ok() {
        return;
    }
    for key in ["callee", "func_ref"] {
        if ctx.op_attrs(op).get_symbol_ref(key) == Some(old) {
            ctx.op_attrs_mut(op).insert(key, new.clone());
        }
    }
    let regions = ctx.op_regions(op).collect::<trunk_ir::RegionList>();
    for region in regions {
        let blocks = ctx.region(region).blocks.clone();
        for block in blocks {
            let nested_ops = ctx.block(block).ops.clone();
            for nested in nested_ops {
                rewrite_symbol_refs(ctx, nested, old, new);
            }
        }
    }
}

fn exact_convention(
    ctx: &IrContext,
    op: OpRef,
) -> Result<Option<CallingConvention>, TargetAbiError> {
    DynConventionBearing::exact(ctx, op)
        .map_err(|error| TargetAbiError::new(format!("target ABI: {error}")))
}

fn collect_functions(
    ctx: &IrContext,
    ops: &[OpRef],
    never: TypeRef,
) -> Result<HashMap<SymbolPath, FunctionIdentity>, TargetAbiError> {
    let mut functions = HashMap::default();
    for &op in ops {
        let Ok(function) = func::Func::from_op(ctx, op) else {
            continue;
        };
        let Some(convention) = exact_convention(ctx, op)? else {
            continue;
        };
        let signature = function.r#type(ctx);
        let callable = func::FuncSig::from_type_ref(ctx, signature).ok_or_else(|| {
            TargetAbiError::new("target ABI: tagged function must have a func.func_sig signature")
        })?;
        if convention == CallingConvention::Cps && callable.results(ctx) != [never] {
            return Err(TargetAbiError::new(format!(
                "target ABI: Cps function `{}` must have logical core.never result",
                function.sym_name(ctx)
            )));
        }
        let key = defined_function_name(ctx, op)?;
        let identity = FunctionIdentity {
            signature,
            convention,
            environment_index: environment_index(ctx, op, callable.inputs(ctx))?,
        };
        if functions.insert(key, identity).is_some() {
            return Err(TargetAbiError::new(
                "target ABI: duplicate tagged function symbol",
            ));
        }
    }
    Ok(functions)
}

fn validate_transfers(
    ctx: &IrContext,
    ops: &[OpRef],
    functions: &HashMap<SymbolPath, FunctionIdentity>,
    never: TypeRef,
) -> Result<(), TargetAbiError> {
    for &op in ops {
        if func::Call::matches(ctx, op) || func::TailCall::matches(ctx, op) {
            let convention = exact_convention(ctx, op)?;
            let callee = ctx.op_attrs(op).get_symbol_ref("callee").ok_or_else(|| {
                TargetAbiError::new("target ABI: direct transfer lacks callee metadata")
            })?;
            let Some(convention) = convention else {
                if function_for_symbol_optional(callee, functions)
                    .is_some_and(|identity| identity.convention == CallingConvention::Cps)
                {
                    return Err(TargetAbiError::new(
                        "target ABI: transfer to Cps callee lacks convention metadata",
                    ));
                }
                continue;
            };
            let identity = function_for_symbol(callee, functions)?;
            if identity.convention != convention {
                return Err(TargetAbiError::new(
                    "target ABI: direct transfer convention differs from callee",
                ));
            }
            let callable = func::FuncSig::from_type_ref(ctx, identity.signature).unwrap();
            if !operands_match(ctx, ctx.op_operands(op), callable.inputs(ctx)) {
                return Err(TargetAbiError::new(
                    "target ABI: direct transfer operands differ from callee signature",
                ));
            }
            if func::Call::matches(ctx, op) {
                if convention == CallingConvention::Cps {
                    return Err(TargetAbiError::new(
                        "target ABI: Cps direct transfer must use func.tail_call",
                    ));
                }
                if ctx.op_result_types(op) != callable.results(ctx) {
                    return Err(TargetAbiError::new(
                        "target ABI: direct call result differs from callee signature",
                    ));
                }
            } else if !is_valid_tail_transfer(ctx, op, convention, callable, never)? {
                return Err(TargetAbiError::new(
                    "target ABI: direct tail call must be a Cps core.never transfer \
                     or return the result of a value-returning caller",
                ));
            }
            continue;
        }

        let Some(call) = DynIndirectCallLike::get(ctx, op) else {
            continue;
        };
        let signature = call.exact_signature(ctx);
        let convention = exact_convention(ctx, op)?;
        // An invalid signature attribute is rejected below, not skipped.
        let has_signature = ctx.op_attrs(op).contains_key("signature");
        if !has_signature && convention.is_none() {
            continue;
        }
        let convention = convention.ok_or_else(|| {
            TargetAbiError::new("target ABI: indirect signature has no convention metadata")
        })?;
        // The interface yields only valid `func.func_sig` contracts.
        let callable = signature
            .and_then(|signature| func::FuncSig::from_type_ref(ctx, signature))
            .ok_or_else(|| {
                TargetAbiError::new("target ABI: indirect transfer lacks exact callable signature")
            })?;
        let signature = callable.as_type_ref();
        let callee = call
            .indirect_callee(ctx)
            .ok_or_else(|| TargetAbiError::new("target ABI: indirect transfer lacks callee"))?;
        let callee_type = ctx.value_ty(callee);
        if let Some(closure_type) =
            crate::closure_lower::physical_closure_type_for_callee(ctx, callee)
        {
            let closure_signature =
                tribute_ir::dialect::closure::Closure::from_type_ref(ctx, closure_type)
                    .unwrap()
                    .func_type(ctx);
            if closure_signature != signature
                || get_physical_closure_convention(ctx, closure_type) != Some(convention)
            {
                return Err(TargetAbiError::new(
                    "target ABI: indirect signature differs from exact closure contract",
                ));
            }
        } else if func::FuncSig::from_type_ref(ctx, callee_type).is_some()
            && callee_type != signature
        {
            return Err(TargetAbiError::new(
                "target ABI: indirect signature differs from typed callee",
            ));
        }
        let args = call.indirect_arguments(ctx).ok_or_else(|| {
            TargetAbiError::new("target ABI: indirect transfer has malformed operands")
        })?;
        if !operands_match(ctx, args, callable.inputs(ctx)) {
            return Err(TargetAbiError::new(
                "target ABI: indirect transfer operands differ from exact callable signature",
            ));
        }
        if func::CallIndirect::matches(ctx, op) {
            if convention == CallingConvention::Cps {
                return Err(TargetAbiError::new(
                    "target ABI: Cps indirect transfer must use func.tail_call_indirect",
                ));
            }
            if ctx.op_result_types(op) != callable.results(ctx) {
                return Err(TargetAbiError::new(
                    "target ABI: indirect call result differs from exact callable signature",
                ));
            }
        } else if !is_valid_tail_transfer(ctx, op, convention, callable, never)? {
            return Err(TargetAbiError::new(
                "target ABI: indirect tail call must be a Cps core.never transfer \
                 or return the result of a value-returning caller",
            ));
        }
    }
    Ok(())
}

fn is_transfer(ctx: &IrContext, op: OpRef) -> bool {
    func::Call::matches(ctx, op)
        || func::TailCall::matches(ctx, op)
        || func::CallIndirect::matches(ctx, op)
        || func::TailCallIndirect::matches(ctx, op)
}

fn operands_match(ctx: &IrContext, operands: &[ValueRef], params: &[TypeRef]) -> bool {
    operands.len() == params.len()
        && operands
            .iter()
            .zip(params)
            .all(|(operand, expected)| ctx.value_ty(*operand) == *expected)
}

/// A tail transfer is a CPS `core.never` transfer, or a source `become`
/// between value-returning callables whose callee result is the caller's.
fn is_valid_tail_transfer(
    ctx: &IrContext,
    op: OpRef,
    convention: CallingConvention,
    callable: func::FuncSig,
    never: TypeRef,
) -> Result<bool, TargetAbiError> {
    if convention == CallingConvention::Cps {
        return Ok(
            callable.single_result(ctx) == Some(never) && is_cps_never_caller(ctx, op, never)?
        );
    }
    let caller = enclosing_function(ctx, op)?;
    let caller_callable =
        func::FuncSig::from_type_ref(ctx, caller.r#type(ctx)).ok_or_else(|| {
            TargetAbiError::new("target ABI: enclosing function is not func.func_sig")
        })?;
    Ok(
        exact_convention(ctx, caller.op_ref())? != Some(CallingConvention::Cps)
            && caller_callable.results(ctx) == callable.results(ctx),
    )
}

fn enclosing_function(ctx: &IrContext, op: OpRef) -> Result<func::Func, TargetAbiError> {
    let mut current = Some(op);
    while let Some(candidate) = current {
        if let Ok(function) = func::Func::from_op(ctx, candidate) {
            return Ok(function);
        }
        current = parent_op(ctx, candidate);
    }
    Err(TargetAbiError::new(
        "target ABI: tail transfer has no enclosing function",
    ))
}

fn is_cps_never_caller(ctx: &IrContext, op: OpRef, never: TypeRef) -> Result<bool, TargetAbiError> {
    let mut current = Some(op);
    while let Some(candidate) = current {
        if let Ok(function) = func::Func::from_op(ctx, candidate) {
            let callable =
                func::FuncSig::from_type_ref(ctx, function.r#type(ctx)).ok_or_else(|| {
                    TargetAbiError::new("target ABI: enclosing function is not func.func_sig")
                })?;
            return Ok(
                exact_convention(ctx, candidate)? == Some(CallingConvention::Cps)
                    && callable.single_result(ctx) == Some(never),
            );
        }
        current = parent_op(ctx, candidate);
    }
    Err(TargetAbiError::new(
        "target ABI: tail transfer has no enclosing function",
    ))
}

/// The tagged function named by a root-qualified reference.
fn function_for_symbol(
    symbol: &SymbolPath,
    functions: &HashMap<SymbolPath, FunctionIdentity>,
) -> Result<FunctionIdentity, TargetAbiError> {
    function_for_symbol_optional(symbol, functions)
        .ok_or_else(|| TargetAbiError::new(format!("target ABI: unknown callable `{symbol}`")))
}

fn function_for_symbol_optional(
    symbol: &SymbolPath,
    functions: &HashMap<SymbolPath, FunctionIdentity>,
) -> Option<FunctionIdentity> {
    functions.get(symbol).copied()
}

/// The root-qualified name a definition is referenced by.
fn defined_function_name(ctx: &IrContext, op: OpRef) -> Result<SymbolPath, TargetAbiError> {
    qualified_name(ctx, op)
        .ok_or_else(|| TargetAbiError::new("target ABI: function definition has no symbol"))
}

fn validate_constant(
    ctx: &mut IrContext,
    constant: func::Constant,
    identity: FunctionIdentity,
    never: TypeRef,
) -> Result<(), TargetAbiError> {
    let target = func::FuncSig::from_type_ref(ctx, identity.signature).unwrap();
    if identity.convention == CallingConvention::Cps && target.single_result(ctx) != Some(never) {
        return Err(TargetAbiError::new(
            "target ABI: Cps function reference must have logical core.never result",
        ));
    }
    if identity
        .environment_index
        .is_some_and(|index| index >= target.inputs(ctx).len())
    {
        return Err(TargetAbiError::new(
            "target ABI: closure environment index is outside target signature",
        ));
    }
    let expected = target
        .rebuild(ctx, |inputs, _| {
            if let Some(index) = identity.environment_index {
                inputs.remove(index);
            }
        })
        .as_type_ref();
    if ctx.op_result_types(constant.op_ref()) != [expected] {
        return Err(TargetAbiError::new(
            "target ABI: function reference differs from target signature",
        ));
    }
    Ok(())
}

fn environment_index(
    ctx: &IrContext,
    function: OpRef,
    params: &[TypeRef],
) -> Result<Option<usize>, TargetAbiError> {
    let attributes = &ctx.op(function).attributes;
    let present = attributes
        .view(ctx)
        .get(CLOSURE_ENVIRONMENT_INDEX_ATTR)
        .is_some();
    let declared = attributes
        .view(ctx)
        .get_u32(CLOSURE_ENVIRONMENT_INDEX_ATTR)
        .ok()
        .flatten()
        .map(|index| index as usize);
    if present && declared.is_none() {
        return Err(TargetAbiError::new(
            "target ABI: malformed closure environment index metadata",
        ));
    }
    if let Some(index) = declared {
        validate_environment_slot(ctx, params, index)?;
    }

    let Some(region) = ctx.op_region(function, 0) else {
        return Ok(declared);
    };
    let Some(&entry) = ctx.region(region).blocks.first() else {
        return declared.map_or(Ok(None), |_| {
            Err(TargetAbiError::new(
                "target ABI: closure environment provenance has no entry block",
            ))
        });
    };
    let arguments = &ctx.block(entry).args;
    if arguments.len() != params.len() {
        return Err(TargetAbiError::new(
            "target ABI: function signature and entry block arity differ",
        ));
    }
    let indices: Vec<_> = arguments
        .iter()
        .enumerate()
        .filter_map(|(index, argument)| {
            (argument.attrs.view(ctx).get_str("bind_name") == Some("__env")).then_some(index)
        })
        .collect();
    match indices.as_slice() {
        [] if declared.is_some() => Err(TargetAbiError::new(
            "target ABI: closure environment provenance has no matching `__env` parameter",
        )),
        [] => Ok(None),
        [index] => {
            validate_environment_slot(ctx, params, *index)?;
            if declared.is_some_and(|declared| declared != *index) {
                return Err(TargetAbiError::new(
                    "target ABI: closure environment index differs from `__env` parameter",
                ));
            }
            Ok(Some(*index))
        }
        _ => Err(TargetAbiError::new(
            "target ABI: function has multiple `__env` parameters",
        )),
    }
}

/// A code function's environment is its own closure, so the slot must hold
/// a convention-proven closure type.
fn validate_environment_slot(
    ctx: &IrContext,
    params: &[TypeRef],
    index: usize,
) -> Result<(), TargetAbiError> {
    let Some(&environment) = params.get(index) else {
        return Err(TargetAbiError::new(
            "target ABI: closure environment index is outside function signature",
        ));
    };
    if get_physical_closure_convention(ctx, environment).is_none() {
        return Err(TargetAbiError::new(
            "target ABI: closure environment must be a convention-proven closure",
        ));
    }
    Ok(())
}

fn parent_op(ctx: &IrContext, op: OpRef) -> Option<OpRef> {
    ctx.op(op).parent_block.and_then(|block| {
        ctx.block(block)
            .parent_region
            .and_then(|region| ctx.region(region).parent_op)
    })
}

struct PhysicalTypeConverter<'a> {
    ctx: &'a mut IrContext,
    never: TypeRef,
    embedded: HashMap<TypeRef, TypeRef>,
    callable: HashMap<(TypeRef, CallingConvention, bool), TypeRef>,
}

impl<'a> PhysicalTypeConverter<'a> {
    fn new(ctx: &'a mut IrContext, never: TypeRef) -> Self {
        Self {
            ctx,
            never,
            embedded: HashMap::default(),
            callable: HashMap::default(),
        }
    }

    /// Physicalize a module-internal callable: it uses the `tail` machine
    /// convention and consumes its parameters, so any internal callable can
    /// be a proper tail transfer's target.
    fn convert_callable(
        &mut self,
        ty: TypeRef,
        convention: CallingConvention,
    ) -> Result<TypeRef, TargetAbiError> {
        self.convert_callable_with(ty, convention, false)
    }

    /// Physicalize a callable with a platform `abi`, which keeps the
    /// platform convention and parameter contract.
    fn convert_platform_callable(
        &mut self,
        ty: TypeRef,
        convention: CallingConvention,
    ) -> Result<TypeRef, TargetAbiError> {
        if convention == CallingConvention::Cps {
            return Err(TargetAbiError::new(
                "target ABI: a callable with a platform abi cannot be Cps",
            ));
        }
        self.convert_callable_with(ty, convention, true)
    }

    fn convert_callable_with(
        &mut self,
        ty: TypeRef,
        convention: CallingConvention,
        platform: bool,
    ) -> Result<TypeRef, TargetAbiError> {
        if let Some(&converted) = self.callable.get(&(ty, convention, platform)) {
            return Ok(converted);
        }
        let callable = func::FuncSig::from_type_ref(self.ctx, ty).ok_or_else(|| {
            TargetAbiError::new("target ABI: proven callable is not a valid func.func_sig")
        })?;
        if convention == CallingConvention::Cps && callable.results(self.ctx) != [self.never] {
            return Err(TargetAbiError::new(
                "target ABI: Cps callable must have logical core.never result",
            ));
        }
        let inputs = callable
            .inputs_with_attrs(self.ctx)
            .map(|(ty, attrs)| (ty, attrs.clone()))
            .collect();
        let mut inputs = self.convert_params(inputs)?;
        if !platform {
            let contract = physical_parameter_attrs(self.ctx, convention);
            for (_, attrs) in &mut inputs {
                attrs.view_mut(self.ctx).extend(contract.clone());
            }
        }
        // A physical Cps callable has no result, so the logical result's
        // parameter attributes are dropped with it.
        let results = if convention == CallingConvention::Cps {
            vec![]
        } else {
            let results = callable
                .results_with_attrs(self.ctx)
                .map(|(ty, attrs)| (ty, attrs.clone()))
                .collect();
            self.convert_params(results)?
        };
        let mut attrs = self.convert_func_attributes(callable)?;
        if attrs.view(self.ctx).contains_key(func::CALL_CONV_ATTR) {
            return Err(TargetAbiError::new(
                "target ABI: logical callable already carries a machine call_conv",
            ));
        }
        if !platform {
            func::CallConv::Tail.set_in(self.ctx, &mut attrs);
        }
        let converted =
            func::func_sig_with_param_attrs(self.ctx, inputs, results, attrs).as_type_ref();
        self.callable.insert((ty, convention, platform), converted);
        Ok(converted)
    }

    fn convert_embedded(&mut self, ty: TypeRef) -> Result<TypeRef, TargetAbiError> {
        if let Some(&converted) = self.embedded.get(&ty) {
            return Ok(converted);
        }
        let data = self.ctx.get_type(ty).clone();
        if tribute_ir::dialect::closure::Closure::matches(self.ctx, ty) {
            let [function] = data.params.as_slice() else {
                return Err(TargetAbiError::new(
                    "target ABI: closure type must contain one callable",
                ));
            };
            let convention = get_physical_closure_convention(self.ctx, ty).ok_or_else(|| {
                TargetAbiError::new("target ABI: closure callable has no exact convention metadata")
            })?;
            let mut converted = data.clone();
            converted.params[0] = self.convert_callable(*function, convention)?;
            self.convert_type_attributes(&mut converted)?;
            let converted = self.intern_if_changed(ty, converted);
            self.embedded.insert(ty, converted);
            return Ok(converted);
        }
        if func::FuncSig::matches(self.ctx, ty) {
            let callable = func::FuncSig::from_type_ref(self.ctx, ty).ok_or_else(|| {
                TargetAbiError::new("target ABI: malformed nested func.func_sig type")
            })?;
            let source_results = callable.results(self.ctx).to_vec();
            if source_results.contains(&self.never) {
                return Err(TargetAbiError::new(
                    "target ABI: untagged nested func.func_sig<(...)->core.never>",
                ));
            }
            let source_inputs = callable.inputs(self.ctx).to_vec();
            let inputs = source_inputs
                .into_iter()
                .map(|input| self.convert_embedded(input))
                .collect::<Result<Vec<_>, _>>()?;
            let results = source_results
                .into_iter()
                .map(|result| self.convert_embedded(result))
                .collect::<Result<Vec<_>, _>>()?;
            let attrs = self.convert_func_attributes(callable)?;
            let converted =
                func::func_sig_with_attrs(self.ctx, inputs, results, attrs).as_type_ref();
            self.embedded.insert(ty, converted);
            return Ok(converted);
        }
        let mut converted = data;
        for parameter in &mut converted.params {
            *parameter = self.convert_embedded(*parameter)?;
        }
        self.convert_type_attributes(&mut converted)?;
        // Frame answer provenance is read only by the logical dispatch and
        // root contract checks, which run before conversion. Physical frames
        // are ordinary nominal layouts.
        if continuation_frame::result_type(self.ctx, ty).is_some() {
            converted
                .attrs
                .view_mut(self.ctx)
                .remove(continuation_frame::RESULT_ATTR);
        }
        let converted = self.intern_if_changed(ty, converted);
        self.embedded.insert(ty, converted);
        Ok(converted)
    }

    fn convert_func_attributes(
        &mut self,
        callable: func::FuncSig,
    ) -> Result<AttributeMap, TargetAbiError> {
        let attributes = callable
            .non_reserved_attrs(self.ctx)
            .map(|(name, value)| (name, value.clone()))
            .collect::<Vec<_>>();
        let attributes = attributes
            .into_iter()
            .map(|(name, value)| Ok((name, self.convert_attribute(value)?)))
            .collect::<Result<Vec<_>, TargetAbiError>>()?;
        Ok(AttributeMap::from_refs(self.ctx, attributes))
    }

    fn convert_type_attributes(&mut self, data: &mut TypeData) -> Result<(), TargetAbiError> {
        let attributes: Vec<_> = data
            .attrs
            .iter()
            .map(|(name, value)| (name, value.clone()))
            .collect();
        for (name, value) in attributes {
            let value = self.convert_attribute(value)?;
            data.attrs.view_mut(self.ctx).insert(name, value);
        }
        Ok(())
    }

    fn convert_attribute(&mut self, attribute: Attribute) -> Result<Attribute, TargetAbiError> {
        attribute.try_map_types(&mut |ty| self.convert_embedded(ty))
    }

    fn convert_params(
        &mut self,
        params: Vec<(TypeRef, AttributeMap)>,
    ) -> Result<Vec<(TypeRef, AttributeMap)>, TargetAbiError> {
        params
            .into_iter()
            .map(|(ty, attrs)| {
                let attrs = attrs
                    .into_iter()
                    .map(|(name, value)| Ok((name, self.convert_attribute(value)?)))
                    .collect::<Result<Vec<_>, TargetAbiError>>()?;
                let attrs = AttributeMap::from_refs(self.ctx, attrs);
                Ok((self.convert_embedded(ty)?, attrs))
            })
            .collect()
    }

    fn intern_if_changed(&mut self, original: TypeRef, data: TypeData) -> TypeRef {
        if data == *self.ctx.get_type(original) {
            original
        } else {
            self.ctx.intern_type(data)
        }
    }
}

fn collect_ops(ctx: &IrContext, root: OpRef) -> Vec<OpRef> {
    let mut operations = Vec::new();
    let _ = walk_op::<()>(ctx, root, &mut |op| {
        operations.push(op);
        ControlFlow::Continue(WalkAction::Advance)
    });
    operations
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::assert_unchanged_on_error;
    use tribute_ir::calling_convention::CALLING_CONVENTION_ATTR;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;
    use trunk_ir::types::TypeDataBuilder;

    fn function(ctx: &IrContext, module: Module, name: &str) -> func::Func {
        module
            .ops(ctx)
            .iter()
            .copied()
            .find_map(|op| {
                let function = func::Func::from_op(ctx, op).ok()?;
                (function.sym_name(ctx) == name).then_some(function)
            })
            .unwrap()
    }

    fn is_worker_call(ctx: &IrContext, op: OpRef) -> bool {
        func::Call::from_op(ctx, op).is_ok_and(|call| call.callee(ctx) == [ROOT_MAIN_SYMBOL])
    }

    fn dispatch_fixture(answer_name: &str, frame_name: &str) -> (IrContext, Module, OpRef) {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            &format!(
                r#"core.module @test {{
            !Answer = core.{answer_name}
            !Evidence = adt.array<adt.typeref<{{name = "std::runtime::evidence::Marker"}}>>
            !Frame = adt.typeref<{{name = "{frame_name}", tribute.cps_continuation_frame_result = !Answer}}>
            !Done = closure.closure<func.func_sig<(!Answer) -> core.never>, {{tribute.calling_convention = "cps", tribute.closure_environment_index = 0}}>
            !Resume = closure.closure<func.func_sig<(!Evidence, !Frame, tribute_rt.anyref) -> core.never>, {{tribute.calling_convention = "cps", tribute.closure_environment_index = 0}}>
            !Dispatch = closure.closure<func.func_sig<(!Evidence, !Resume, core.i32, core.i32, core.i32, tribute_rt.anyref) -> core.never>, {{tribute.calling_convention = "cps", tribute.closure_environment_index = 1}}>
            !{frame_name} = adt.struct<{frame_name}(done: !Done, dispatch: !Dispatch), {{tribute.cps_continuation_frame_result = !Answer}}>
            func.func @run(%ev: !Evidence, %dispatch: !Dispatch, %resume: !Resume, %payload: tribute_rt.anyref) -> core.never attributes {{tribute.calling_convention = "cps"}} {{
                effect.dispatch_cps %ev, %dispatch, %resume, %payload {{ability = @State, op_name = "get", answer_type = !Answer}}
            }}
            func.func @"std::runtime::evidence::lookup"(%ev: !Evidence, %id: core.i32) -> core.i32 {{
                func.return %id
            }}
        }}"#
            ),
        );
        let dispatch = collect_ops(&ctx, module.op())
            .into_iter()
            .find(|&op| effect::DispatchCps::matches(&ctx, op))
            .unwrap();
        (ctx, module, dispatch)
    }

    #[test]
    fn semantic_dispatch_reaches_the_canonical_wasm_target_signature() {
        let (mut ctx, module, _) = dispatch_fixture("i32", "frame");
        lower_cps_signatures_to_physical(&mut ctx, module).unwrap();
        crate::closure_lower::lower_prepared_closures(&mut ctx, module).unwrap();
        let runtime = crate::evidence_lowering::prepare_evidence_runtime(&ctx, module).unwrap();
        for op in module.ops_snapshot(&ctx) {
            if let Ok(function) = func::Func::from_op(&ctx, op) {
                crate::evidence_lowering::lower_evidence_func(&mut ctx, function, &runtime)
                    .unwrap();
            }
        }
        crate::closure_lower::finalize_closure_storage_layout(&mut ctx, module);
        let result = crate::wasm::lower::lower_to_wasm(&mut ctx, module, &mut Default::default());
        let printed = print_module(&ctx, module.op());
        assert!(result.is_ok(), "{result:?}\n{printed}");
        assert!(!printed.contains("effect.dispatch_cps"), "{printed}");
    }

    #[test]
    fn dispatch_answer_contract_survives_atomic_physicalization() {
        for answer in ["i32", "i64", "nil"] {
            let (mut ctx, module, dispatch) = dispatch_fixture(answer, "frame");
            let semantic_answer = ctx.op_attrs(dispatch).get_type("answer_type").unwrap();
            validate_dispatch_contracts(&mut ctx, module).unwrap();
            lower_cps_signatures_to_physical(&mut ctx, module).unwrap();
            assert_eq!(
                ctx.op_attrs(dispatch).get_type("answer_type"),
                Some(semantic_answer)
            );
            let printed = print_module(&ctx, module.op());
            assert!(
                !printed.contains(continuation_frame::RESULT_ATTR),
                "physicalization must consume frame answer provenance:\n{printed}"
            );
            assert!(
                func::FuncSig::from_type_ref(&ctx, function(&ctx, module, "run").r#type(&ctx))
                    .unwrap()
                    .results(&ctx)
                    .is_empty()
            );
        }
    }

    #[test]
    fn malformed_dispatch_is_rejected_without_mutating_any_surface() {
        for mutation in 0..9 {
            let (mut ctx, module, dispatch) = dispatch_fixture("i32", "frame");
            match mutation {
                0 => {
                    ctx.op_attrs_mut(dispatch).remove("answer_type");
                }
                1 => {
                    ctx.op_attrs_mut(dispatch)
                        .insert("answer_type", Attribute::Int(0));
                }
                2 => {
                    let wrong = core::Nil::params().intern(&mut ctx).as_type_ref();
                    ctx.op_attrs_mut(dispatch)
                        .insert("answer_type", Attribute::Type(wrong));
                }
                3..=6 => {
                    let index = if mutation == 3 { 1 } else { 2 };
                    let value = ctx.op_operands(dispatch)[index];
                    let mut ty = ctx.get_type(ctx.value_ty(value)).clone();
                    if mutation == 3 || mutation == 4 {
                        ty.attrs
                            .view_mut(&mut ctx)
                            .insert(CLOSURE_ENVIRONMENT_INDEX_ATTR, Attribute::Int(9));
                    } else if mutation == 5 {
                        ty.attrs
                            .view_mut(&mut ctx)
                            .insert(CALLING_CONVENTION_ATTR, Attribute::Int(0));
                    } else {
                        let signature = func::FuncSig::from_type_ref(&ctx, ty.params[0]).unwrap();
                        let mut inputs = signature.inputs(&ctx).to_vec();
                        let answer = ctx.op_attrs(dispatch).get_type("answer_type").unwrap();
                        inputs[1] = tribute_ir::continuation_frame::ref_type(
                            &mut ctx,
                            "other_nominal_frame",
                            answer,
                        );
                        let results = signature.results(&ctx).to_vec();
                        ty.params[0] = func::func_sig(&mut ctx, inputs, results).as_type_ref();
                    }
                    let ty = ctx.intern_type(ty);
                    let entry = ctx.op(dispatch).parent_block.unwrap();
                    ctx.set_block_arg_type(entry, index as u32, ty);
                }
                7 | 8 => {
                    let ty = if mutation == 7 {
                        TypeDataBuilder::new("core.i32").intern(&mut ctx)
                    } else {
                        core::Ptr::params().intern(&mut ctx).as_type_ref()
                    };
                    let entry = ctx.op(dispatch).parent_block.unwrap();
                    ctx.set_block_arg_type(entry, 3, ty);
                }
                _ => unreachable!(),
            }
            let before = print_module(&ctx, module.op());
            let aliases = ctx.type_aliases().to_vec();
            let ops = collect_ops(&ctx, module.op());
            let result_types: Vec<_> = ops
                .iter()
                .map(|&op| ctx.op_result_types(op).to_vec())
                .collect();
            let error = lower_cps_signatures_to_physical(&mut ctx, module)
                .expect_err(&format!("mutation {mutation}"));
            if mutation >= 7 {
                assert!(error.to_string().contains("packed payload"), "{error}");
            }
            assert_eq!(print_module(&ctx, module.op()), before);
            assert_eq!(ctx.type_aliases(), aliases);
            assert_eq!(collect_ops(&ctx, module.op()), ops);
            for (op, expected) in ops.into_iter().zip(result_types) {
                assert_eq!(ctx.op_result_types(op), expected);
            }
        }
    }

    #[test]
    fn empty_direct_results_survive_mixed_cps_physicalization() {
        let mut ctx = IrContext::new();
        let evidence = ability::evidence_adt_type_ref(&mut ctx);
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
            !Evidence = adt.array<adt.typeref<{name = "std::runtime::evidence::Marker"}>>
            !direct_closure = closure.closure<func.func_sig<() -> ()>, {tribute.calling_convention = "direct"}>
            !evidence_closure = closure.closure<func.func_sig<(!Evidence) -> ()>, {tribute.calling_convention = "evidence_direct"}>
            func.func @direct() attributes {tribute.calling_convention = "direct"} { func.return }
            func.func @evidence(%ev: !Evidence) attributes {tribute.calling_convention = "evidence_direct"} { func.return }
            func.func @unit() -> core.nil attributes {tribute.calling_convention = "direct"} {
                %nil = core.nil_value : core.nil
                func.return %nil
            }
            func.func @caller(%ev: !Evidence) attributes {tribute.calling_convention = "evidence_direct"} {
                func.call {callee = @direct, tribute.calling_convention = "direct"}
                func.call %ev {callee = @evidence, tribute.calling_convention = "evidence_direct"}
                %direct = func.constant {func_ref = @direct} : func.func_sig<() -> ()>
                %evidence = func.constant {func_ref = @evidence} : func.func_sig<(!Evidence) -> ()>
                func.call_indirect %direct {signature = func.func_sig<() -> ()>, tribute.calling_convention = "direct"}
                func.call_indirect %evidence, %ev {signature = func.func_sig<(!Evidence) -> ()>, tribute.calling_convention = "evidence_direct"}
                %nil = func.call {callee = @unit, tribute.calling_convention = "direct"} : core.nil
                func.return
            }
            func.func @cps() -> core.never attributes {tribute.calling_convention = "cps"} { func.unreachable }
        }"#,
        );
        assert_eq!(ctx.type_alias_by_text("Evidence"), Some(evidence));
        let cps = function(&ctx, module, "cps");
        let unchanged: Vec<_> = collect_ops(&ctx, module.op())
            .into_iter()
            .filter(|&op| op != cps.op_ref())
            .map(|op| {
                // Physicalization consumes the convention of every transfer
                // it validates against a non-closure callee.
                // Every internal signature gains the tail machine
                // convention; only its results matter here.
                let mut attributes = ctx.op(op).attributes.clone();
                if is_transfer(&ctx, op) {
                    attributes
                        .view_mut(&mut ctx)
                        .remove(CALLING_CONVENTION_ATTR);
                }
                attributes.view_mut(&mut ctx).remove("type");
                attributes.view_mut(&mut ctx).remove("signature");
                (op, attributes, ctx.op_result_types(op).to_vec())
            })
            .collect();
        lower_cps_signatures_to_physical(&mut ctx, module).unwrap();
        for (op, before, results) in unchanged {
            let mut after = ctx.op(op).attributes.clone();
            after.view_mut(&mut ctx).remove("type");
            after.view_mut(&mut ctx).remove("signature");
            assert_eq!(after, before);
            assert_eq!(ctx.op_result_types(op).len(), results.len());
        }
        for name in ["direct", "evidence", "unit"] {
            let function = function(&ctx, module, name);
            let callable = func::FuncSig::from_type_ref(&ctx, function.r#type(&ctx)).unwrap();
            assert_eq!(callable.results(&ctx).len(), usize::from(name == "unit"));
        }
        assert!(
            func::FuncSig::from_type_ref(&ctx, cps.r#type(&ctx))
                .unwrap()
                .results(&ctx)
                .is_empty()
        );
        for op in collect_ops(&ctx, module.op()) {
            if func::CallIndirect::matches(&ctx, op) {
                assert!(ctx.op_result_types(op).is_empty());
                let signature = DynIndirectCallLike::get(&ctx, op)
                    .and_then(|call| call.exact_signature(&ctx))
                    .unwrap();
                assert!(
                    func::FuncSig::from_type_ref(&ctx, signature)
                        .unwrap()
                        .results(&ctx)
                        .is_empty()
                );
            }
        }
    }

    #[test]
    fn physicalizes_dispatch_aware_exact_cps_contracts_only() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !cps = closure.closure<func.func_sig<(core.i32, core.i32, core.i32, core.i32) -> core.never>, {tribute.calling_convention = "cps"}>
  func.func @direct() -> core.never attributes {tribute.calling_convention = "direct"} { func.unreachable }
  func.func @evidence() -> core.never attributes {tribute.calling_convention = "evidence_direct"} { func.unreachable }
  func.func @cps() -> core.never attributes {tribute.calling_convention = "cps"} { func.unreachable }
  func.func @run(%callee: core.i32, %evidence: core.i32, %env: tribute_rt.anyref, %done: core.i32, %dispatch: core.i32, %value: core.i32) -> core.never attributes {tribute.calling_convention = "cps"} {
    func.tail_call_indirect %callee, %evidence, %env, %done, %dispatch, %value {signature = func.func_sig<(core.i32, tribute_rt.anyref, core.i32, core.i32, core.i32) -> core.never>, tribute.calling_convention = "cps"}
  }
}"#,
        );

        lower_cps_signatures_to_physical(&mut ctx, module).unwrap();

        let never = core::Never::params().intern(&mut ctx).as_type_ref();
        for (name, expected) in [
            ("direct", Some(never)),
            ("evidence", Some(never)),
            ("cps", None),
            ("run", None),
        ] {
            let signature = function(&ctx, module, name).r#type(&ctx);
            assert_eq!(
                func::FuncSig::from_type_ref(&ctx, signature)
                    .unwrap()
                    .single_result(&ctx),
                expected
            );
        }
        let printed = print_module(&ctx, module.op());
        assert!(
            printed.contains(
                "signature = func.func_sig<(core.i32 {tribute.ownership = \"consumed\"}, tribute_rt.anyref {tribute.ownership = \"consumed\"}, core.i32 {tribute.ownership = \"consumed\"}, core.i32 {tribute.ownership = \"consumed\"}, core.i32 {tribute.ownership = \"consumed\"}) -> (), {call_conv = \"tail\"}>"
            ),
            "{printed}"
        );
        assert!(
            printed.contains(
                "closure.closure<func.func_sig<(core.i32 {tribute.ownership = \"consumed\"}, core.i32 {tribute.ownership = \"consumed\"}, core.i32 {tribute.ownership = \"consumed\"}, core.i32 {tribute.ownership = \"consumed\"}) -> (), {call_conv = \"tail\"}>"
            ),
            "{printed}"
        );
    }

    #[test]
    fn unrelated_type_metadata_never_becomes_an_indirect_signature() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @run(%callee: core.i32, %value: core.i32) -> core.never attributes {tribute.calling_convention = "cps"} {
    func.tail_call_indirect %callee, %value {signature = func.func_sig<(core.i32) -> core.never>, tribute.calling_convention = "cps"}
  }
}"#,
        );
        let run = function(&ctx, module, "run");
        let body = run.body_if_present(&ctx).unwrap();
        let indirect = ctx.block(ctx.region(body).blocks[0]).ops[0];
        let signature = DynIndirectCallLike::get(&ctx, indirect)
            .and_then(|call| call.exact_signature(&ctx))
            .unwrap();
        ctx.op_attrs_mut(indirect)
            .insert("unrelated_callable_metadata", Attribute::Type(signature));
        let error = assert_unchanged_on_error(&mut ctx, module, lower_cps_signatures_to_physical);

        assert!(
            error
                .to_string()
                .contains("untagged nested func.func_sig<(...)->core.never>"),
            "{error}"
        );
    }

    fn compose_promoted_root() -> (IrContext, Module) {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @main(%evidence: core.i32, %frame: core.i32) -> core.never attributes {tribute.calling_convention = "cps"} {
    func.unreachable
  }
}"#,
        );
        let main = function(&ctx, module, "main");
        let nil = core::Nil::params().intern(&mut ctx).as_type_ref();
        let never = core::Never::params().intern(&mut ctx).as_type_ref();
        let evidence = ability::evidence_adt_type_ref(&mut ctx);
        let done_callable = func::func_sig(&mut ctx, [nil], [never]).as_type_ref();
        let done = tribute_ir::calling_convention::physical_closure_type_with_environment_index(
            &mut ctx,
            done_callable,
            CallingConvention::Cps,
            0,
        );
        let frame_name = "__tribute_continuation_frame_root_nil";
        let frame = tribute_ir::continuation_frame::ref_type(&mut ctx, frame_name, nil);
        let anyref = tribute_rt::Anyref::params().intern(&mut ctx).as_type_ref();
        let i32_ty = TypeDataBuilder::new("core.i32").intern(&mut ctx);
        let dispatch = tribute_ir::calling_convention::cps_dispatch_type(
            &mut ctx, evidence, frame, anyref, i32_ty,
        );
        let layout =
            tribute_ir::continuation_frame::layout_type(&mut ctx, frame_name, nil, done, dispatch);
        ctx.register_type_alias(Symbol::new(frame_name), layout);
        let worker = func::func_sig(&mut ctx, [evidence, frame], [never]).as_type_ref();
        ctx.op_attrs_mut(main.op_ref())
            .insert("type", Attribute::Type(worker));
        let entry = ctx.region(main.body(&ctx)).blocks[0];
        ctx.set_block_arg_type(entry, 0, evidence);
        ctx.set_block_arg_type(entry, 1, frame);

        lower_cps_signatures_to_physical(&mut ctx, module).unwrap();
        compose_root_entry_bridge(&mut ctx, module).unwrap();
        (ctx, module)
    }

    #[test]
    fn promoted_root_uses_typed_completion_and_ordinary_call() {
        let (ctx, module) = compose_promoted_root();
        let wrapper = function(&ctx, module, "main");
        let worker = function(&ctx, module, ROOT_MAIN_SYMBOL);
        let done_k = function(&ctx, module, ROOT_DONE_K_SYMBOL);
        let dispatch = function(&ctx, module, ROOT_UNHANDLED_SYMBOL);

        assert_eq!(
            DynConventionBearing::of(&ctx, wrapper.op_ref()),
            Some(CallingConvention::Direct)
        );
        for function in [worker, done_k, dispatch] {
            assert_eq!(
                DynConventionBearing::of(&ctx, function.op_ref()),
                Some(CallingConvention::Cps)
            );
            assert!(
                func::FuncSig::from_type_ref(&ctx, function.r#type(&ctx))
                    .unwrap()
                    .results(&ctx)
                    .is_empty()
            );
        }
        for function in [worker, done_k, dispatch] {
            assert!(
                !ctx.op_attrs(function.op_ref())
                    .contains_key(CLOSURE_ENVIRONMENT_INDEX_ATTR)
            );
        }

        let wrapper_ops = collect_ops(&ctx, wrapper.op_ref());
        let call = wrapper_ops
            .iter()
            .copied()
            .find(|op| is_worker_call(&ctx, *op))
            .expect("wrapper must make exactly one ordinary worker call");
        assert!(func::Call::from_op(&ctx, call).is_ok());
        assert_eq!(
            ctx.op_attrs(call).get_symbol_ref("callee"),
            Some(&SymbolPath::new([ROOT_MAIN_SYMBOL]))
        );
        let worker_callable = func::FuncSig::from_type_ref(&ctx, worker.r#type(&ctx)).unwrap();
        let [worker_evidence, worker_frame] = worker_callable.inputs(&ctx) else {
            panic!("root worker must have Evidence and ContinuationFrame parameters");
        };
        assert_eq!(ctx.value_ty(ctx.op_operands(call)[0]), *worker_evidence);
        assert_eq!(ctx.value_ty(ctx.op_operands(call)[1]), *worker_frame);
        assert!(ctx.get_type(*worker_frame).name == tribute_ir::dialect::adt::Typeref::NAME);
        let frame_name = ctx.type_attrs(*worker_frame).get_str("name").unwrap();
        let frame_layout = ctx
            .type_alias_by_text(frame_name)
            .expect("worker frame must retain its exact nominal layout");
        let fields: Vec<_> = adt::Struct::from_type_ref(&ctx, frame_layout)
            .expect("frame layout must be a valid adt.struct")
            .fields(&ctx)
            .collect();
        let [(done_name, done_ty), (dispatch_name, dispatch_ty)] = fields.as_slice() else {
            panic!("frame must have distinct Done and Dispatch fields");
        };
        assert_eq!(*done_name, Symbol::new("done"));
        assert_eq!(*dispatch_name, Symbol::new("dispatch"));
        assert_ne!(*worker_frame, *done_ty);
        assert_ne!(*done_ty, *dispatch_ty);
        assert!(wrapper_ops.iter().any(|op| {
            adt::StructNew::from_op(&ctx, *op).is_ok()
                && ctx.op_result_types(*op) == [*worker_frame]
                && ctx
                    .op_operands(*op)
                    .iter()
                    .map(|value| ctx.value_ty(*value))
                    .collect::<Vec<_>>()
                    == vec![*done_ty, *dispatch_ty]
        }));
        assert!(
            wrapper_ops
                .iter()
                .all(|op| func::TailCall::from_op(&ctx, *op).is_err())
        );
        assert!(wrapper_ops.iter().any(|op| {
            adt::StructNew::from_op(&ctx, *op).is_ok()
                && ctx.op_result_types(*op).first().is_some_and(|ty| {
                    ctx.type_attrs(*ty).get_str("name") == Some(ROOT_COMPLETION_CELL_NAME)
                })
        }));
        assert!(
            wrapper_ops
                .iter()
                .any(|op| adt::ArrayOf::from_op(&ctx, *op).is_ok())
        );
        assert!(
            wrapper_ops
                .iter()
                .any(|op| adt::StructGet::from_op(&ctx, *op).is_ok())
        );
        assert!(
            collect_ops(&ctx, done_k.op_ref())
                .iter()
                .any(|op| adt::StructSet::from_op(&ctx, *op).is_ok())
        );
        let printed = print_module(&ctx, module.op());
        assert!(!printed.contains("__tribute_cps_control"), "{printed}");
        assert!(!printed.contains("Step"), "{printed}");
        assert!(!printed.contains("trampoline"), "{printed}");
    }

    #[test]
    fn promoted_root_has_a_single_parameterless_entry() {
        let (ctx, module) = compose_promoted_root();
        let wrapper = function(&ctx, module, "main");
        let signature = func::FuncSig::from_type_ref(&ctx, wrapper.r#type(&ctx)).unwrap();
        assert!(signature.inputs(&ctx).is_empty());
        assert_eq!(
            DynConventionBearing::of(&ctx, wrapper.op_ref()),
            Some(CallingConvention::Direct)
        );
        let ops = collect_ops(&ctx, wrapper.op_ref());
        let evidence = ops
            .iter()
            .copied()
            .find_map(|op| adt::ArrayOf::from_op(&ctx, op).ok())
            .expect("wrapper must build initial evidence");
        let call = ops
            .into_iter()
            .find(|op| is_worker_call(&ctx, *op))
            .expect("wrapper must call CPS worker");
        assert_eq!(ctx.op_operands(call)[0], evidence.result(&ctx));
        let functions: Vec<_> = module
            .ops(&ctx)
            .iter()
            .copied()
            .filter_map(|op| func::Func::from_op(&ctx, op).ok())
            .map(|function| function.sym_name(&ctx).to_string())
            .collect();
        assert_eq!(
            functions,
            [
                ROOT_MAIN_SYMBOL,
                ROOT_DONE_K_SYMBOL,
                ROOT_UNHANDLED_SYMBOL,
                "main"
            ]
        );
    }

    /// The root `main` takes no hidden parameters and passes fresh initial
    /// evidence to the EvidenceDirect worker.
    fn assert_parameterless_evidence_entry(ctx: &IrContext, module: Module) {
        let entry_main = function(ctx, module, "main");
        let signature = func::FuncSig::from_type_ref(ctx, entry_main.r#type(ctx)).unwrap();
        assert!(signature.inputs(ctx).is_empty());
        assert_eq!(
            DynConventionBearing::of(ctx, entry_main.op_ref()),
            Some(CallingConvention::Direct)
        );
        let ops = collect_ops(ctx, entry_main.op_ref());
        let empty = ops
            .iter()
            .copied()
            .find_map(|op| adt::ArrayOf::from_op(ctx, op).ok())
            .expect("entry bridge must build initial evidence");
        let call = ops
            .iter()
            .copied()
            .find_map(|op| func::Call::from_op(ctx, op).ok())
            .expect("entry bridge must call the root worker");
        assert_eq!(*call.callee(ctx), SymbolPath::new([ROOT_MAIN_SYMBOL]));
        assert_eq!(ctx.op_operands(call.op_ref()), [empty.result(ctx)]);
        // The bridge reads the worker's convention and does not record it on
        // the call it builds.
        assert_eq!(DynConventionBearing::of(ctx, call.op_ref()), None);
    }

    const EVIDENCE_DIRECT_MAIN: &str = r#"core.module @test {
  !Evidence = adt.array<adt.typeref<{name = "std::runtime::evidence::Marker"}>>
  func.func @main(%evidence: !Evidence) -> core.nil attributes {tribute.calling_convention = "evidence_direct"} {
    %nil = core.nil_value : core.nil
    func.return %nil
  }
  func.func @caller(%evidence: !Evidence) -> core.nil attributes {tribute.calling_convention = "evidence_direct"} {
    %result = func.call %evidence {callee = @main, tribute.calling_convention = "evidence_direct"} : core.nil
    func.return %result
  }
}"#;

    #[test]
    fn evidence_direct_main_is_bridged_to_parameterless_entry() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, EVIDENCE_DIRECT_MAIN);
        compose_root_entry_bridge(&mut ctx, module).unwrap();

        assert_parameterless_evidence_entry(&ctx, module);
        let caller = function(&ctx, module, "caller");
        let call = collect_ops(&ctx, caller.op_ref())
            .into_iter()
            .find_map(|op| func::Call::from_op(&ctx, op).ok())
            .unwrap();
        assert_eq!(*call.callee(&ctx), SymbolPath::new([ROOT_MAIN_SYMBOL]));
    }

    #[test]
    fn direct_main_is_wrapped_without_inputs() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @main() -> core.nil attributes {tribute.calling_convention = "direct"} {
    %nil = core.nil_value : core.nil
    func.return %nil
  }
  func.func @caller() -> core.nil attributes {tribute.calling_convention = "direct"} {
    %result = func.call {callee = @main, tribute.calling_convention = "direct"} : core.nil
    func.return %result
  }
}"#,
        );
        compose_root_entry_bridge(&mut ctx, module).unwrap();

        let wrapper = function(&ctx, module, "main");
        let ops = collect_ops(&ctx, wrapper.op_ref());
        let calls: Vec<_> = ops
            .iter()
            .copied()
            .filter_map(|op| func::Call::from_op(&ctx, op).ok())
            .collect();
        let [call] = calls.as_slice() else {
            panic!("Direct wrapper must make exactly one call");
        };
        assert_eq!(*call.callee(&ctx), SymbolPath::new([ROOT_MAIN_SYMBOL]));
        assert!(ctx.op_operands(call.op_ref()).is_empty());
        assert!(
            !ops.iter()
                .any(|op| adt::ArrayOf::from_op(&ctx, *op).is_ok())
        );
        let caller = function(&ctx, module, "caller");
        let caller_call = collect_ops(&ctx, caller.op_ref())
            .into_iter()
            .find_map(|op| func::Call::from_op(&ctx, op).ok())
            .unwrap();
        assert_eq!(
            *caller_call.callee(&ctx),
            SymbolPath::new([ROOT_MAIN_SYMBOL])
        );
    }

    #[test]
    fn root_bridge_rejects_non_nil_main() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @main() -> core.i32 attributes {tribute.calling_convention = "direct"} {
    %zero = arith.const {value = 0} : core.i32
    func.return %zero
  }
}"#,
        );
        let error = compose_root_entry_bridge(&mut ctx, module).unwrap_err();
        assert!(error.to_string().contains("must return core.nil"));
    }

    #[test]
    fn root_bridge_rejects_reserved_symbol_collision() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            &EVIDENCE_DIRECT_MAIN.replace("@caller", "@__tribute_main"),
        );
        let error = compose_root_entry_bridge(&mut ctx, module).unwrap_err();
        assert!(error.to_string().contains("reserved root symbol collision"));
    }

    #[test]
    fn root_bridge_creates_no_lambda_for_target_phase() {
        let (ctx, module) = compose_promoted_root();
        let printed = print_module(&ctx, module.op());

        assert!(!printed.contains("closure.lambda"), "{printed}");
        assert!(printed.contains("adt.struct_new"), "{printed}");
    }

    #[test]
    fn malformed_root_frame_contract_fails_before_bridge_mutation() {
        for (frame_result, malformed_layout, expected) in [
            (
                false,
                false,
                "result provenance differs from root source result",
            ),
            (true, false, "must have an exact nominal layout"),
            (true, true, "layout provenance is malformed"),
        ] {
            let mut ctx = IrContext::new();
            let module = parse_test_module(
                &mut ctx,
                r#"core.module @test {
  func.func @main(%evidence: core.i32, %frame: core.i32) -> core.never attributes {tribute.calling_convention = "cps"} {
    func.unreachable
  }
}"#,
            );
            let main = function(&ctx, module, "main");
            let nil = core::Nil::params().intern(&mut ctx).as_type_ref();
            let never = core::Never::params().intern(&mut ctx).as_type_ref();
            let evidence = ability::evidence_adt_type_ref(&mut ctx);
            let frame_name = "__tribute_malformed_root_frame";
            let i32_ty = TypeDataBuilder::new("core.i32").intern(&mut ctx);
            let frame = tribute_ir::continuation_frame::ref_type(
                &mut ctx,
                frame_name,
                if frame_result { nil } else { i32_ty },
            );
            if malformed_layout {
                let wrong = adt::Struct::from_fields::<String>(
                    &mut ctx,
                    frame_name,
                    [],
                    AttributeMap::new(),
                )
                .as_type_ref();
                ctx.register_type_alias(Symbol::new(frame_name), wrong);
            }
            let worker = func::func_sig(&mut ctx, [evidence, frame], [never]).as_type_ref();
            ctx.op_attrs_mut(main.op_ref())
                .insert("type", Attribute::Type(worker));
            let entry = ctx.region(main.body(&ctx)).blocks[0];
            ctx.set_block_arg_type(entry, 0, evidence);
            ctx.set_block_arg_type(entry, 1, frame);

            let before = print_module(&ctx, module.op());
            let aliases = ctx.type_aliases().to_vec();
            let error = lower_cps_signatures_to_physical(&mut ctx, module).unwrap_err();

            assert!(error.to_string().contains(expected), "{error}");
            assert_eq!(print_module(&ctx, module.op()), before);
            assert_eq!(ctx.type_aliases(), aliases);
        }
    }

    #[test]
    fn parameterized_dispatch_tag_fails_before_bridge_mutation() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @main(%evidence: core.i32, %frame: core.i32) -> core.never attributes {tribute.calling_convention = "cps"} {
    func.unreachable
  }
}"#,
        );
        let main = function(&ctx, module, "main");
        let nil = core::Nil::params().intern(&mut ctx).as_type_ref();
        let never = core::Never::params().intern(&mut ctx).as_type_ref();
        let evidence = ability::evidence_adt_type_ref(&mut ctx);
        let frame_name = "__tribute_parameterized_dispatch_tag";
        let frame = tribute_ir::continuation_frame::ref_type(&mut ctx, frame_name, nil);
        let done = tribute_ir::calling_convention::cps_done_type(&mut ctx, nil);
        let anyref = tribute_rt::Anyref::params().intern(&mut ctx).as_type_ref();
        let parameterized_i32 = TypeDataBuilder::new("core.i32").param(nil).intern(&mut ctx);
        let dispatch = tribute_ir::calling_convention::cps_dispatch_type(
            &mut ctx,
            evidence,
            frame,
            anyref,
            parameterized_i32,
        );
        let layout =
            tribute_ir::continuation_frame::layout_type(&mut ctx, frame_name, nil, done, dispatch);
        ctx.register_type_alias(Symbol::new(frame_name), layout);
        let worker = func::func_sig(&mut ctx, [evidence, frame], [never]).as_type_ref();
        ctx.op_attrs_mut(main.op_ref())
            .insert("type", Attribute::Type(worker));
        let entry = ctx.region(main.body(&ctx)).blocks[0];
        ctx.set_block_arg_type(entry, 0, evidence);
        ctx.set_block_arg_type(entry, 1, frame);

        let error = assert_unchanged_on_error(&mut ctx, module, lower_cps_signatures_to_physical);

        assert!(
            error
                .to_string()
                .contains("frame Dispatch operands differ from the exact terminal ABI"),
            "{error}"
        );
    }

    #[test]
    fn malformed_transfers_and_ambiguous_nested_never_leave_ir_unchanged() {
        for (input, expected) in [
            (
                r#"core.module @test {
  func.func @run(%callee: core.i32) -> core.never attributes {tribute.calling_convention = "cps"} {
    func.tail_call_indirect %callee {tribute.calling_convention = "cps"}
  }
}"#,
                "lacks exact callable signature",
            ),
            (
                r#"core.module @test {
  func.func @pure(%callee: func.func_sig<(core.i32, core.i32) -> core.i32>, %left: core.i32, %right: core.i32) -> core.i32 attributes {tribute.calling_convention = "direct"} {
    %result = func.call_indirect %callee, %left, %right {tribute.calling_convention = "direct"} : core.i32
    func.return %result
  }
}"#,
                "lacks exact callable signature",
            ),
            (
                r#"core.module @test {
  func.func @run(%callee: core.i32) -> core.never attributes {tribute.calling_convention = "cps"} {
    func.tail_call_indirect %callee {signature = core.i32, tribute.calling_convention = "cps"}
  }
}"#,
                "lacks exact callable signature",
            ),
            (
                r#"core.module @test {
  func.func @run(%callee: core.i32) -> core.never attributes {tribute.calling_convention = "cps"} {
    func.tail_call_indirect %callee {signature = core.i32}
  }
}"#,
                "indirect signature has no convention metadata",
            ),
            (
                r#"core.module @test {
  func.func @callee(%value: core.i32) -> core.never attributes {tribute.calling_convention = "cps"} { func.unreachable }
  func.func @run(%value: core.i1) -> core.never attributes {tribute.calling_convention = "cps"} {
    func.tail_call %value {callee = @callee, tribute.calling_convention = "cps"}
  }
}"#,
                "operands differ",
            ),
            (
                r#"core.module @test {
  !ambiguous = closure.closure<func.func_sig<() -> core.never>>
}"#,
                "no exact convention metadata",
            ),
        ] {
            let mut ctx = IrContext::new();
            let module = parse_test_module(&mut ctx, input);
            let error =
                assert_unchanged_on_error(&mut ctx, module, lower_cps_signatures_to_physical);

            assert!(error.to_string().contains(expected), "{error}");
        }
    }

    #[test]
    fn closure_transfer_convention_is_left_for_closure_lowering() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !callback = closure.closure<func.func_sig<() -> core.i32>, {tribute.calling_convention = "direct"}>
  func.func @run(%callback: !callback) -> core.i32 attributes {tribute.calling_convention = "direct"} {
    %result = func.call_indirect %callback {signature = func.func_sig<() -> core.i32>, tribute.calling_convention = "direct"} : core.i32
    func.return %result
  }
}"#,
        );

        lower_cps_signatures_to_physical(&mut ctx, module).unwrap();

        let run = function(&ctx, module, "run");
        assert_eq!(
            DynConventionBearing::of(&ctx, run.op_ref()),
            Some(CallingConvention::Direct)
        );
        let call = collect_ops(&ctx, run.op_ref())
            .into_iter()
            .find(|&op| func::CallIndirect::matches(&ctx, op))
            .unwrap();
        assert_eq!(
            DynConventionBearing::of(&ctx, call),
            Some(CallingConvention::Direct)
        );
    }

    #[test]
    fn raw_closure_storage_never_satisfies_a_semantic_direct_transfer() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !semantic = closure.closure<func.func_sig<() -> core.i32>, {tribute.calling_convention = "direct"}>
  !_closure = adt.struct<_closure(func_ptr: core.i32, env: tribute_rt.anyref)>
  func.func @factory(%callback: !semantic) -> core.i32 attributes {tribute.calling_convention = "direct"} {
    func.unreachable
  }
  func.func @run(%raw: !_closure) -> core.i32 attributes {tribute.calling_convention = "direct"} {
    %result = func.call %raw {callee = @factory, tribute.calling_convention = "direct"} : core.i32
    func.return %result
  }
}"#,
        );
        let error = assert_unchanged_on_error(&mut ctx, module, lower_cps_signatures_to_physical);

        assert!(error.to_string().contains("operands differ"), "{error}");
    }

    #[test]
    fn bodied_and_bodyless_closure_targets_share_environment_provenance() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !Env = closure.closure<func.func_sig<(core.i32, core.i32) -> core.never>, {tribute.calling_convention = "cps", tribute.closure_environment_index = 1}>
  func.func @external(%evidence: core.i32, %environment: !Env, %done: core.i32) -> core.never attributes {tribute.calling_convention = "cps", tribute.closure_environment_index = 1}
  func.func @defined(%evidence: core.i32, %__env: !Env, %done: core.i32) -> core.never attributes {tribute.calling_convention = "cps", tribute.closure_environment_index = 1} {
    func.unreachable
  }
  func.func @holder() -> core.i32 {
    %external = func.constant {func_ref = @external} : func.func_sig<(core.i32, core.i32) -> core.never>
    %defined = func.constant {func_ref = @defined} : func.func_sig<(core.i32, core.i32) -> core.never>
    func.unreachable
  }
}"#,
        );

        lower_cps_signatures_to_physical(&mut ctx, module).unwrap();

        let printed = print_module(&ctx, module.op());
        assert!(
            printed.contains("func.func @external(%arg0: core.i32, %arg1: !Env, %arg2: core.i32)"),
            "{printed}"
        );
        assert!(
            printed.contains("func.func @defined")
                && printed.matches("func.constant").count() == 2
                && printed.contains("!t0 = func.func_sig<(core.i32 {tribute.ownership = \"consumed\"}, core.i32 {tribute.ownership = \"consumed\"}) -> (), {call_conv = \"tail\"}>")
                && printed.matches(": !t0").count() == 2,
            "{printed}"
        );
    }

    #[test]
    fn physicalizes_explicit_generated_cps_environment_slots() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !Zero = closure.closure<func.func_sig<() -> core.never>, {tribute.calling_convention = "cps", tribute.closure_environment_index = 0}>
  !One = closure.closure<func.func_sig<(core.i32) -> core.never>, {tribute.calling_convention = "cps", tribute.closure_environment_index = 0}>
  func.func @generated_zero(%__env: !Zero) -> core.never attributes {tribute.calling_convention = "cps", tribute.closure_environment_index = 0} {
    func.unreachable
  }
  func.func @generated_one(%__env: !One, %value: core.i32) -> core.never attributes {tribute.calling_convention = "cps", tribute.closure_environment_index = 0} {
    func.unreachable
  }
  func.func @holder() -> core.i32 {
    %zero = func.constant {func_ref = @generated_zero} : func.func_sig<() -> core.never>
    %one = func.constant {func_ref = @generated_one} : func.func_sig<(core.i32) -> core.never>
    func.unreachable
  }
}"#,
        );

        lower_cps_signatures_to_physical(&mut ctx, module).unwrap();

        for (name, parameter_count) in [("generated_zero", 1), ("generated_one", 2)] {
            let function = function(&ctx, module, name);
            let signature = func::FuncSig::from_type_ref(&ctx, function.r#type(&ctx)).unwrap();
            assert!(signature.results(&ctx).is_empty());
            assert_eq!(signature.inputs(&ctx).len(), parameter_count);
            let environment = signature.inputs(&ctx)[0];
            assert!(get_physical_closure_convention(&ctx, environment).is_some());
        }
        let printed = print_module(&ctx, module.op());
        assert!(
            printed.contains(": func.func_sig<() -> (), {call_conv = \"tail\"}>"),
            "{printed}"
        );
        assert!(
            printed.contains(
                ": func.func_sig<(core.i32 {tribute.ownership = \"consumed\"}) -> (), {call_conv = \"tail\"}>"
            ),
            "{printed}"
        );
    }

    #[test]
    fn malformed_or_missing_environment_provenance_fails_before_mutation() {
        for (function, expected) in [
            (
                "func.func @external(%evidence: core.i32, %environment: tribute_rt.anyref, %done: core.i32) -> core.never attributes {tribute.calling_convention = \"cps\"}",
                "function reference differs",
            ),
            (
                "func.func @external(%evidence: core.i32, %environment: tribute_rt.anyref, %done: core.i32) -> core.never attributes {tribute.calling_convention = \"cps\", tribute.closure_environment_index = 3}",
                "outside function signature",
            ),
            (
                "func.func @external(%evidence: core.i32, %environment: core.i32, %done: core.i32) -> core.never attributes {tribute.calling_convention = \"cps\", tribute.closure_environment_index = 1}",
                "convention-proven closure",
            ),
            (
                "func.func @external(%evidence: core.i32, %environment: !Env, %done: core.i32) -> core.never attributes {tribute.calling_convention = \"cps\", tribute.closure_environment_index = 1} { func.unreachable }",
                "no matching `__env`",
            ),
            (
                "func.func @external(%environment: !Env, %__env: !Env, %done: core.i32) -> core.never attributes {tribute.calling_convention = \"cps\", tribute.closure_environment_index = 0} { func.unreachable }",
                "differs from `__env` parameter",
            ),
        ] {
            let mut ctx = IrContext::new();
            let module = parse_test_module(
                &mut ctx,
                &format!(
                    r#"core.module @test {{
  !Env = closure.closure<func.func_sig<(core.i32, core.i32) -> core.never>, {{tribute.calling_convention = "cps", tribute.closure_environment_index = 1}}>
  {function}
  func.func @holder() -> core.i32 {{
    %function = func.constant {{func_ref = @external}} : func.func_sig<(core.i32, core.i32) -> core.never>
    func.unreachable
  }}
}}"#
                ),
            );
            let error =
                assert_unchanged_on_error(&mut ctx, module, lower_cps_signatures_to_physical);

            assert!(error.to_string().contains(expected), "{error}");
        }
    }

    #[test]
    fn duplicate_environment_provenance_fails_before_mutation() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !Env2 = closure.closure<func.func_sig<() -> core.never>, {tribute.calling_convention = "cps", tribute.closure_environment_index = 0}>
  !Env = closure.closure<func.func_sig<(!Env2) -> core.never>, {tribute.calling_convention = "cps", tribute.closure_environment_index = 0}>
  func.func @external(%__env: !Env, %environment: !Env2) -> core.never attributes {tribute.calling_convention = "cps", tribute.closure_environment_index = 0} {
    func.unreachable
  }
  func.func @holder() -> core.i32 {
    %function = func.constant {func_ref = @external} : func.func_sig<(!Env2) -> core.never>
    func.unreachable
  }
}"#,
        );
        let external = function(&ctx, module, "external");
        let entry = ctx.region(external.body(&ctx)).blocks[0];
        let environment_name = ctx.string_attr("__env");
        ctx.block_arg_attrs_mut(entry, 1)
            .insert("bind_name", environment_name);
        let error = assert_unchanged_on_error(&mut ctx, module, lower_cps_signatures_to_physical);

        assert!(
            error.to_string().contains("multiple `__env` parameters"),
            "{error}"
        );
    }

    #[test]
    fn malformed_zero_parameter_func_sig_fails_without_panicking() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, "core.module @test {}");
        let malformed = trunk_ir::types::TypeDataBuilder::new("func.func_sig").intern(&mut ctx);
        ctx.op_attrs_mut(module.op())
            .insert("malformed_callable", Attribute::Type(malformed));
        let error = assert_unchanged_on_error(&mut ctx, module, lower_cps_signatures_to_physical);

        assert!(
            error.to_string().contains("malformed nested func.func_sig"),
            "{error}"
        );
    }

    #[test]
    fn already_physical_cps_function_fails_before_target_abi_mutation() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @broken() -> core.never attributes {tribute.calling_convention = "cps"} { func.unreachable }
}"#,
        );
        let broken = function(&ctx, module, "broken");
        let resultless = func::func_sig(&mut ctx, [], []).as_type_ref();
        ctx.op_attrs_mut(broken.op_ref())
            .insert("type", Attribute::Type(resultless));
        let error = assert_unchanged_on_error(&mut ctx, module, lower_cps_signatures_to_physical);

        assert!(
            error
                .to_string()
                .contains("must have logical core.never result"),
            "{error}"
        );
    }

    #[test]
    fn raw_constant_is_unchanged_and_malformed_tagged_constant_fails_closed() {
        let mut ctx = IrContext::new();
        let raw = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @external(%value: core.i32) -> core.i32
  func.func @holder() -> core.i32 {
    %function = func.constant {func_ref = @external} : func.func_sig<(core.i32) -> core.i32>
    func.unreachable
  }
}"#,
        );
        let before = print_module(&ctx, raw.op());

        lower_cps_signatures_to_physical(&mut ctx, raw).unwrap();

        assert_eq!(print_module(&ctx, raw.op()), before);

        let mut ctx = IrContext::new();
        let malformed = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @external(%value: core.i32) -> core.never attributes {tribute.calling_convention = "cps"}
  func.func @holder() -> core.i32 {
    %function = func.constant {func_ref = @external} : func.func_sig<(core.i1) -> core.never>
    func.unreachable
  }
}"#,
        );
        let error =
            assert_unchanged_on_error(&mut ctx, malformed, lower_cps_signatures_to_physical);

        assert!(error.to_string().contains("function reference differs"));
    }

    #[test]
    fn malformed_calling_convention_is_rejected_before_mutation() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @broken() -> core.never attributes {tribute.calling_convention = 9} { func.unreachable }
}"#,
        );
        let error = assert_unchanged_on_error(&mut ctx, module, lower_cps_signatures_to_physical);

        assert!(error.to_string().contains("malformed calling-convention"));
    }

    #[test]
    fn convention_violations_are_rejected_before_mutation() {
        for (body, expected) in [
            (
                r#"%c = arith.const {value = 0, tribute.calling_convention = "direct"} : core.i32"#,
                "arith.const: calling-convention metadata on an operation that cannot carry it",
            ),
            (
                r#"%c = closure.new {func_ref = @callee, tribute.calling_convention = "cps"} : !direct"#,
                "closure.new: calling convention differs from the one its callable type records",
            ),
        ] {
            let mut ctx = IrContext::new();
            let module = parse_test_module(
                &mut ctx,
                &format!(
                    r#"core.module @test {{
  !fn = func.func_sig<(core.i32) -> core.i32>
  !direct = closure.closure<!fn, {{tribute.calling_convention = "direct", tribute.closure_environment_index = 0}}>
  func.func @callee(%x: core.i32) -> core.i32 attributes {{tribute.calling_convention = "direct"}} {{
    func.return %x
  }}
  func.func @main() -> core.nil attributes {{tribute.calling_convention = "direct"}} {{
    {body}
    %n = core.nil_value : core.nil
    func.return %n
  }}
}}"#
                ),
            );
            let error =
                assert_unchanged_on_error(&mut ctx, module, lower_cps_signatures_to_physical);
            assert!(error.to_string().contains(expected), "{error}");
        }
    }
}
