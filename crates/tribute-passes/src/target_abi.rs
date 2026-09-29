//! Shared target-neutral physicalization of convention-proven CPS signatures.
//!
//! The pass consumes exact callable metadata, validates the whole transfer surface, and
//! only then maps logical CPS `core.never` results to the shared empty-result
//! list used by target backends.

use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::ops::ControlFlow;

use tribute_core::calling_convention::{
    CLOSURE_ENVIRONMENT_INDEX_ATTR, CPS_CONTINUATION_FRAME_RESULT_ATTR, cps_closure_function_type,
    cps_continuation_frame_result_type, get_physical_closure_environment_index,
};
use tribute_core::{
    CALLING_CONVENTION_ATTR, CallingConvention, get_calling_convention,
    get_physical_closure_convention,
};
use tribute_ir::dialect::{ability, effect, tribute_rt};
use trunk_ir::Symbol;
use trunk_ir::context::{BlockArgData, BlockData, IrContext, RegionData};
use trunk_ir::dialect::{adt, arith, core, func};
use trunk_ir::op_interface::IndirectCallLikeOps;
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::refs::{BlockRef, OpRef, TypeRef, ValueRef};
use trunk_ir::rewrite::Module;
use trunk_ir::smallvec::smallvec;
use trunk_ir::symbol_table::qualified_name;
use trunk_ir::types::{Attribute, AttributeMap, Location, TypeData, TypeDataBuilder};
use trunk_ir::walk::{WalkAction, walk_op};

const ROOT_EXPORT_CONVENTION_ATTR: &str = "tribute.root_export_convention";
const ROOT_SOURCE_RESULT_ATTR: &str = "tribute.root_source_result";
const ROOT_MAIN_SYMBOL: &str = "__tribute_main";
const ROOT_DONE_K_SYMBOL: &str = "__tribute_done_k";
const ROOT_UNHANDLED_SYMBOL: &str = "__tribute_unhandled";
const ROOT_COMPLETION_CELL_NAME: &str = "__tribute_completion_cell";
const ROOT_COMPLETION_CELL_VALUE_FIELD: &str = "value";

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
    let ops = collect_ops(ctx, module.op());
    let never = core::never(ctx).as_type_ref();
    let anyref = tribute_rt::anyref(ctx).as_type_ref();
    let functions = collect_functions(ctx, &ops, never, anyref)?;
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
            let converted = match convention {
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
                .op(op)
                .attributes
                .contains_key(CLOSURE_ENVIRONMENT_INDEX_ATTR)
            {
                consumed_attributes.push((op, Symbol::new(CLOSURE_ENVIRONMENT_INDEX_ATTR)));
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
        let indirect_signature = IndirectCallLikeOps::exact_signature(converter.ctx, op);
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
            func::remove_indirect_call_signature(&mut converted_attributes);
            indirect_signatures.push((op, signature));
        }
        let op_attributes: Vec<_> = converted_attributes
            .iter()
            .map(|(name, value)| (*name, value.clone()))
            .collect();
        for (name, value) in op_attributes {
            if func::Func::matches(converter.ctx, op) && name == Symbol::new("type") {
                continue;
            }
            let converted = converter.convert_attribute(value)?;
            converted_attributes.insert(name, converted);
        }
        for (name, value) in converted_attributes {
            if original_attributes.get(name) != Some(&value) {
                attributes.push((op, name, value));
            }
        }

        let regions = converter.ctx.op(op).regions.to_vec();
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
                        converted_attrs.insert(*name, converter.convert_attribute(value.clone())?);
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
        ctx.op_mut(op)
            .attributes
            .insert(Symbol::new("type"), Attribute::Type(ty));
    }
    for (op, index, ty) in result_types {
        ctx.set_op_result_type(op, index, ty);
    }
    for (op, name, value) in attributes {
        ctx.op_mut(op).attributes.insert(name, value);
    }
    for (op, name) in consumed_attributes {
        ctx.op_mut(op).attributes.remove(name);
    }
    for (op, signature) in indirect_signatures {
        assert!(IndirectCallLikeOps::set_exact_signature(ctx, op, signature));
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
        ContractPhase::Logical => vec![core::never(ctx).as_type_ref()],
        ContractPhase::Physical => vec![],
    };
    let expected_results = expected_results.as_slice();
    let Some(module_block) = module.first_block(ctx) else {
        return Ok(None);
    };
    let top_level_ops = ctx.block(module_block).ops.to_vec();
    let roots: Vec<_> = top_level_ops
        .iter()
        .copied()
        .filter(|&op| {
            func::Func::from_op(ctx, op)
                .is_ok_and(|function| function.sym_name(ctx) == Symbol::new("main"))
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

    let export_convention = root_export_convention(ctx, worker_op)?;
    let source_result = root_source_result(ctx, worker_op)?;
    if export_convention.is_some() != source_result.is_some() {
        return Err(TargetAbiError::new(
            "target root bridge: preserved export convention and source result must be paired",
        ));
    }
    let Some(export_convention) = export_convention else {
        return Ok(None);
    };
    let source_result = source_result.expect("paired root metadata checked above");
    if matches!(export_convention, CallingConvention::Cps) {
        return Err(TargetAbiError::new(
            "target root bridge: root export convention must be Direct or EvidenceDirect",
        ));
    }
    if get_calling_convention(ctx, worker_op) != Some(CallingConvention::Cps) {
        return Err(TargetAbiError::new(
            "target root bridge: preserved export metadata requires a Cps root worker",
        ));
    }

    let nil_ty = core::nil(ctx).as_type_ref();
    if source_result != nil_ty {
        return Err(TargetAbiError::new(
            "target root bridge: the current root source result must be core.nil",
        ));
    }
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
    if ctx.op(worker_op).regions.is_empty() {
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
    let top_level_ops = ctx.block(module_block).ops.to_vec();
    let main = Symbol::new("main");
    let root_main = Symbol::new(ROOT_MAIN_SYMBOL);
    let mut roots = top_level_ops.iter().copied().filter(|&op| {
        func::Func::from_op(ctx, op).is_ok_and(|function| function.sym_name(ctx) == main)
    });
    let Some(worker_op) = roots.next() else {
        return Ok(());
    };
    if top_level_ops.iter().any(|&op| {
        func::Func::from_op(ctx, op).is_ok_and(|function| function.sym_name(ctx) == root_main)
    }) {
        return Err(TargetAbiError::new(
            "target root bridge: reserved root symbol collision",
        ));
    }
    if ctx.op(worker_op).regions.is_empty() {
        return Err(TargetAbiError::new(
            "target root bridge: root `main` must be a definition",
        ));
    }
    let worker = func::Func::from_op(ctx, worker_op).expect("filtered root function");
    let signature = func::FuncSig::from_type_ref(ctx, worker.r#type(ctx))
        .ok_or_else(|| TargetAbiError::new("target root bridge: main is not func.func_sig"))?;
    let inputs = signature.inputs(ctx).to_vec();
    let results = signature.results(ctx).to_vec();
    let nil_ty = core::nil(ctx).as_type_ref();
    let evidence_ty = ability::evidence_adt_type_ref(ctx);
    let convention = get_calling_convention(ctx, worker_op);
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
    ctx.op_mut(worker_op)
        .attributes
        .insert(Symbol::new("sym_name"), Attribute::Symbol(root_main));
    for &op in &top_level_ops {
        rewrite_symbol_refs(ctx, op, main, root_main);
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
            if let Some(convention) = convention {
                set_root_convention(ctx, call.op_ref(), convention);
            }
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
        .sym_name(main)
        .r#type(wrapper_ty)
        .regions(body)
        .build(ctx, location);
    set_root_convention(ctx, wrapper.op_ref(), CallingConvention::Direct);
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
    worker: Symbol,
) -> Result<ValueRef, TargetAbiError> {
    let RootEntryContract {
        worker_op,
        source_result,
        evidence_ty,
        frame,
    } = contract;
    let root_done_k = Symbol::new(ROOT_DONE_K_SYMBOL);
    let root_dispatch = Symbol::new(ROOT_UNHANDLED_SYMBOL);
    let location = ctx.op(worker_op).location;
    remove_root_contract(ctx, worker_op);

    let cell_ty = root_completion_cell_type(ctx, source_result);
    let anyref_ty = tribute_rt::anyref(ctx).as_type_ref();
    let done_function_ty = func::func_sig(ctx, [anyref_ty, source_result], [])
        .with_call_conv(ctx, func::CallConv::Tail)
        .as_type_ref();
    let done_entry = ctx.create_block(BlockData {
        location,
        args: vec![
            BlockArgData {
                ty: anyref_ty,
                attrs: bind_name("__env"),
            },
            BlockArgData {
                ty: source_result,
                attrs: bind_name("__answer"),
            },
        ],
        ops: smallvec![],
        parent_region: None,
    });
    let done_args = ctx.block_args(done_entry).to_vec();
    let cell = adt::RefCast::operands(done_args[0])
        .r#type(cell_ty)
        .results(cell_ty)
        .build(ctx, location);
    ctx.push_op(done_entry, cell.op_ref());
    let store = adt::StructSet::operands(cell.result(ctx), done_args[1])
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
        .sym_name(root_done_k)
        .r#type(done_function_ty)
        .regions(done_region)
        .build(ctx, location);
    set_root_convention(ctx, done_function.op_ref(), CallingConvention::Cps);

    let dispatch_function_ty = dispatch_entry_function_type(ctx, frame.dispatch, anyref_ty)?;
    let dispatch_entry = ctx.create_block(BlockData {
        location,
        args: func::FuncSig::from_type_ref(ctx, dispatch_function_ty)
            .expect("validated dispatch entry contract")
            .inputs(ctx)
            .iter()
            .copied()
            .enumerate()
            .map(|(index, ty)| BlockArgData {
                ty,
                attrs: bind_name(if index == 1 { "__env" } else { "__arg" }),
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
        .sym_name(root_dispatch)
        .r#type(dispatch_function_ty)
        .regions(dispatch_region)
        .build(ctx, location);
    set_root_convention(ctx, dispatch_function.op_ref(), CallingConvention::Cps);

    let initial = arith::Const::operands()
        .value(Attribute::Unit)
        .results(source_result)
        .build(ctx, location);
    ctx.push_op(entry, initial.op_ref());
    let cell_new = adt::StructNew::operands([initial.result(ctx)])
        .r#type(cell_ty)
        .results(cell_ty)
        .build(ctx, location);
    ctx.push_op(entry, cell_new.op_ref());
    let erased_cell = core::UnrealizedConversionCast::operands(cell_new.result(ctx))
        .results(anyref_ty)
        .build(ctx, location);
    ctx.push_op(entry, erased_cell.op_ref());
    let done_constant = func::Constant::operands()
        .func_ref(root_done_k)
        .results(done_function_ty)
        .build(ctx, location);
    ctx.push_op(entry, done_constant.op_ref());
    let closure_struct_ty = crate::closure_lower::closure_struct_type_ref(ctx);
    let done_closure =
        adt::StructNew::operands([done_constant.result(ctx), erased_cell.result(ctx)])
            .r#type(closure_struct_ty)
            .results(closure_struct_ty)
            .build(ctx, location);
    ctx.push_op(entry, done_closure.op_ref());
    let typed_done = core::UnrealizedConversionCast::operands(done_closure.result(ctx))
        .results(frame.done)
        .build(ctx, location);
    ctx.push_op(entry, typed_done.op_ref());

    let dispatch_constant = func::Constant::operands()
        .func_ref(root_dispatch)
        .results(dispatch_function_ty)
        .build(ctx, location);
    ctx.push_op(entry, dispatch_constant.op_ref());
    let dispatch_closure =
        adt::StructNew::operands([dispatch_constant.result(ctx), erased_cell.result(ctx)])
            .r#type(closure_struct_ty)
            .results(closure_struct_ty)
            .build(ctx, location);
    ctx.push_op(entry, dispatch_closure.op_ref());
    let typed_dispatch = core::UnrealizedConversionCast::operands(dispatch_closure.result(ctx))
        .results(frame.dispatch)
        .build(ctx, location);
    ctx.push_op(entry, typed_dispatch.op_ref());
    let frame_value =
        adt::StructNew::operands([typed_done.result(ctx), typed_dispatch.result(ctx)])
            .r#type(frame.layout)
            .results(frame.reference)
            .build(ctx, location);
    ctx.push_op(entry, frame_value.op_ref());

    let evidence = build_initial_evidence(ctx, entry, location, evidence_ty);
    let worker_call = func::Call::operands([evidence, frame_value.result(ctx)])
        .callee(worker)
        .results([])
        .build(ctx, location);
    set_root_convention(ctx, worker_call.op_ref(), CallingConvention::Cps);
    ctx.push_op(entry, worker_call.op_ref());
    let completed = adt::StructGet::operands(cell_new.result(ctx))
        .r#type(cell_ty)
        .field(0)
        .results(source_result)
        .build(ctx, location);
    ctx.push_op(entry, completed.op_ref());

    ctx.push_op(module_block, done_function.op_ref());
    ctx.push_op(module_block, dispatch_function.op_ref());
    Ok(completed.result(ctx))
}

/// Build the target's initial evidence: an empty evidence array.
fn build_initial_evidence(
    ctx: &mut IrContext,
    block: BlockRef,
    location: Location,
    evidence_ty: TypeRef,
) -> ValueRef {
    let i32_ty = ctx.intern_type(TypeDataBuilder::new("core", "i32").build());
    let zero = arith::Const::operands()
        .value(Attribute::Int(0))
        .results(i32_ty)
        .build(ctx, location);
    ctx.push_op(block, zero.op_ref());
    let empty = adt::ArrayNew::operands([zero.result(ctx)])
        .r#type(evidence_ty)
        .results(evidence_ty)
        .build(ctx, location);
    ctx.push_op(block, empty.op_ref());
    empty.result(ctx)
}

fn root_completion_cell_type(ctx: &mut IrContext, value_ty: TypeRef) -> TypeRef {
    ctx.intern_type(TypeData {
        dialect: Symbol::new("adt"),
        name: Symbol::new("struct"),
        params: smallvec![value_ty],
        attrs: [
            (
                Symbol::new("name"),
                Attribute::Symbol(Symbol::new(ROOT_COMPLETION_CELL_NAME)),
            ),
            (
                Symbol::new("fields"),
                Attribute::List(vec![Attribute::List(vec![
                    Attribute::Symbol(Symbol::new(ROOT_COMPLETION_CELL_VALUE_FIELD)),
                    Attribute::Type(value_ty),
                ])]),
            ),
        ]
        .into_iter()
        .collect(),
    })
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
    let answer = cps_continuation_frame_result_type(ctx, frame)
        .ok_or_else(|| TargetAbiError::new("CPS resume frame lacks answer type"))?;
    let results = resume_signature.results(ctx);
    if !(results.is_empty()
        || (results.len() == 1
            && is_parameterless_dialect_type(
                ctx,
                results[0],
                Symbol::new("core"),
                Symbol::new("never"),
            )))
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
        let packed_payload = tribute_rt::anyref(ctx).as_type_ref();
        if ctx.value_ty(*payload) != packed_payload {
            return Err(TargetAbiError::new(
                "CPS dispatch packed payload must have exact tribute_rt.anyref type",
            ));
        }
        let answer = ctx
            .op(op)
            .attributes
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
    if reference_data.dialect != Symbol::new("adt") || reference_data.name != Symbol::new("typeref")
    {
        return Err(TargetAbiError::new(
            "target root bridge: worker frame must be an exact nominal adt.typeref",
        ));
    }
    // Physicalization consumes the frame answer provenance after checking it
    // here; the physical frame's Done input then carries the source result.
    let expected_provenance = (phase == ContractPhase::Logical).then_some(source_result);
    if cps_continuation_frame_result_type(ctx, frame) != expected_provenance {
        return Err(TargetAbiError::new(
            "target root bridge: worker frame result provenance differs from root source result",
        ));
    }
    let name = reference_data.attrs.get_symbol("name").ok_or_else(|| {
        TargetAbiError::new("target root bridge: worker frame lacks nominal layout identity")
    })?;
    let layout = ctx.type_alias_by_name(name).ok_or_else(|| {
        TargetAbiError::new("target root bridge: worker frame must have an exact nominal layout")
    })?;
    let layout_data = ctx.get_type(layout);
    if layout_data.dialect != Symbol::new("adt")
        || layout_data.name != Symbol::new("struct")
        || layout_data.attrs.get_symbol("name") != Some(name)
        || cps_continuation_frame_result_type(ctx, layout) != expected_provenance
    {
        return Err(TargetAbiError::new(
            "target root bridge: worker frame layout provenance is malformed",
        ));
    }
    let fields = layout_data.attrs.get("fields").ok_or_else(|| {
        TargetAbiError::new("target root bridge: worker frame layout lacks fields")
    })?;
    let Attribute::List(fields) = fields else {
        return Err(TargetAbiError::new(
            "target root bridge: worker frame layout fields are malformed",
        ));
    };
    let [Attribute::List(done_field), Attribute::List(dispatch_field)] = fields.as_slice() else {
        return Err(TargetAbiError::new(
            "target root bridge: worker frame layout must contain done then dispatch",
        ));
    };
    let [Attribute::Symbol(done_name), Attribute::Type(done)] = done_field.as_slice() else {
        return Err(TargetAbiError::new(
            "target root bridge: worker frame done field is malformed",
        ));
    };
    let [Attribute::Symbol(dispatch_name), Attribute::Type(dispatch)] = dispatch_field.as_slice()
    else {
        return Err(TargetAbiError::new(
            "target root bridge: worker frame dispatch field is malformed",
        ));
    };
    if *done_name != Symbol::new("done") || *dispatch_name != Symbol::new("dispatch") {
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
        || !is_parameterless_dialect_type(ctx, *prompt, Symbol::new("core"), Symbol::new("i32"))
        || !is_parameterless_dialect_type(ctx, *ability, Symbol::new("core"), Symbol::new("i32"))
        || !is_parameterless_dialect_type(ctx, *operation, Symbol::new("core"), Symbol::new("i32"))
        || !is_parameterless_dialect_type(
            ctx,
            *payload,
            Symbol::new("tribute_rt"),
            Symbol::new("anyref"),
        )
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
        || !is_parameterless_dialect_type(
            ctx,
            resume.inputs(ctx)[2],
            Symbol::new("tribute_rt"),
            Symbol::new("anyref"),
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
    anyref: TypeRef,
) -> Result<TypeRef, TargetAbiError> {
    let callable_ty = dispatch_callable_function_type(ctx, dispatch)?;
    let callable = func::FuncSig::from_type_ref(ctx, callable_ty).ok_or_else(|| {
        TargetAbiError::new("target root bridge: frame Dispatch callable is not func.func_sig")
    })?;
    Ok(callable
        .rebuild(ctx, |inputs, _| {
            inputs.insert(1, (anyref, AttributeMap::new()));
        })
        .as_type_ref())
}

fn is_parameterless_dialect_type(
    ctx: &IrContext,
    ty: TypeRef,
    dialect: Symbol,
    name: Symbol,
) -> bool {
    ctx.types().is_dialect(ty, dialect, name)
        && ctx.get_type(ty).params.is_empty()
        && ctx.get_type(ty).attrs.is_empty()
}

fn root_export_convention(
    ctx: &IrContext,
    op: OpRef,
) -> Result<Option<CallingConvention>, TargetAbiError> {
    let Some(attribute) = ctx.op(op).attributes.get(ROOT_EXPORT_CONVENTION_ATTR) else {
        return Ok(None);
    };
    let Attribute::Int(code) = attribute else {
        return Err(TargetAbiError::new(
            "target root bridge: root export convention metadata is malformed",
        ));
    };
    let code = u8::try_from(*code).map_err(|_| {
        TargetAbiError::new("target root bridge: root export convention metadata is malformed")
    })?;
    CallingConvention::try_from(code).map(Some).map_err(|_| {
        TargetAbiError::new("target root bridge: root export convention metadata is malformed")
    })
}

fn root_source_result(ctx: &IrContext, op: OpRef) -> Result<Option<TypeRef>, TargetAbiError> {
    let Some(attribute) = ctx.op(op).attributes.get(ROOT_SOURCE_RESULT_ATTR) else {
        return Ok(None);
    };
    let Attribute::Type(result) = attribute else {
        return Err(TargetAbiError::new(
            "target root bridge: root source result metadata is malformed",
        ));
    };
    Ok(Some(*result))
}

fn set_root_convention(ctx: &mut IrContext, op: OpRef, convention: CallingConvention) {
    ctx.op_mut(op).attributes.insert(
        Symbol::new(CALLING_CONVENTION_ATTR),
        Attribute::Int(convention as i128),
    );
}

fn bind_name(name: &str) -> AttributeMap {
    [(
        Symbol::new("bind_name"),
        Attribute::Symbol(Symbol::from_dynamic(name)),
    )]
    .into_iter()
    .collect()
}

fn remove_root_contract(ctx: &mut IrContext, op: OpRef) {
    ctx.op_mut(op)
        .attributes
        .remove(ROOT_EXPORT_CONVENTION_ATTR);
    ctx.op_mut(op).attributes.remove(ROOT_SOURCE_RESULT_ATTR);
}

fn rewrite_symbol_refs(ctx: &mut IrContext, op: OpRef, old: Symbol, new: Symbol) {
    if core::Module::from_op(ctx, op).is_ok() {
        return;
    }
    for key in [Symbol::new("callee"), Symbol::new("func_ref")] {
        if ctx.op(op).attributes.get_symbol(key) == Some(old) {
            ctx.op_mut(op)
                .attributes
                .insert(key, Attribute::Symbol(new));
        }
    }
    let regions = ctx.op(op).regions.clone();
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
    let present = ctx
        .op(op)
        .attributes
        .get(Symbol::new(CALLING_CONVENTION_ATTR))
        .is_some();
    let convention = get_calling_convention(ctx, op);
    if present && convention.is_none() {
        return Err(TargetAbiError::new(
            "target ABI: malformed calling-convention metadata",
        ));
    }
    Ok(convention)
}

fn collect_functions(
    ctx: &IrContext,
    ops: &[OpRef],
    never: TypeRef,
    anyref: TypeRef,
) -> Result<HashMap<Symbol, FunctionIdentity>, TargetAbiError> {
    let mut functions = HashMap::new();
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
            environment_index: environment_index(ctx, op, callable.inputs(ctx), anyref)?,
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
    functions: &HashMap<Symbol, FunctionIdentity>,
    never: TypeRef,
) -> Result<(), TargetAbiError> {
    for &op in ops {
        if func::Call::matches(ctx, op) || func::TailCall::matches(ctx, op) {
            let convention = exact_convention(ctx, op)?;
            let callee = ctx.op(op).attributes.get_symbol("callee").ok_or_else(|| {
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
            } else if convention != CallingConvention::Cps
                || callable.single_result(ctx) != Some(never)
                || !is_cps_never_caller(ctx, op, never)?
            {
                return Err(TargetAbiError::new(
                    "target ABI: direct tail call must be a Cps core.never transfer",
                ));
            }
            continue;
        }

        if !func::CallIndirect::matches(ctx, op) && !func::TailCallIndirect::matches(ctx, op) {
            continue;
        }
        let signature = IndirectCallLikeOps::exact_signature(ctx, op);
        let convention = exact_convention(ctx, op)?;
        // An invalid signature attribute is rejected below, not skipped.
        let has_signature = ctx.op(op).attributes.contains_key(Symbol::new("signature"));
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
        let callee = IndirectCallLikeOps::callee(ctx, op)
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
        let args = IndirectCallLikeOps::arguments(ctx, op).ok_or_else(|| {
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
        } else if convention != CallingConvention::Cps
            || callable.single_result(ctx) != Some(never)
            || !is_cps_never_caller(ctx, op, never)?
        {
            return Err(TargetAbiError::new(
                "target ABI: indirect tail call must be a Cps core.never transfer",
            ));
        }
    }
    Ok(())
}

fn operands_match(ctx: &IrContext, operands: &[ValueRef], params: &[TypeRef]) -> bool {
    operands.len() == params.len()
        && operands
            .iter()
            .zip(params)
            .all(|(operand, expected)| ctx.value_ty(*operand) == *expected)
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
    symbol: Symbol,
    functions: &HashMap<Symbol, FunctionIdentity>,
) -> Result<FunctionIdentity, TargetAbiError> {
    function_for_symbol_optional(symbol, functions)
        .ok_or_else(|| TargetAbiError::new(format!("target ABI: unknown callable `{symbol}`")))
}

fn function_for_symbol_optional(
    symbol: Symbol,
    functions: &HashMap<Symbol, FunctionIdentity>,
) -> Option<FunctionIdentity> {
    functions.get(&symbol).copied()
}

/// The root-qualified name a definition is referenced by.
fn defined_function_name(ctx: &IrContext, op: OpRef) -> Result<Symbol, TargetAbiError> {
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
    anyref: TypeRef,
) -> Result<Option<usize>, TargetAbiError> {
    let attributes = &ctx.op(function).attributes;
    let present = attributes
        .get(Symbol::new(CLOSURE_ENVIRONMENT_INDEX_ATTR))
        .is_some();
    let declared = attributes
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
        validate_environment_slot(params, anyref, index)?;
    }

    let Some(&region) = ctx.op(function).regions.first() else {
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
            (argument.attrs.get_symbol("bind_name") == Some(Symbol::new("__env"))).then_some(index)
        })
        .collect();
    match indices.as_slice() {
        [] if declared.is_some() => Err(TargetAbiError::new(
            "target ABI: closure environment provenance has no matching `__env` parameter",
        )),
        [] => Ok(None),
        [index] => {
            validate_environment_slot(params, anyref, *index)?;
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

fn validate_environment_slot(
    params: &[TypeRef],
    anyref: TypeRef,
    index: usize,
) -> Result<(), TargetAbiError> {
    if index >= params.len() {
        return Err(TargetAbiError::new(
            "target ABI: closure environment index is outside function signature",
        ));
    }
    if params.get(index) != Some(&anyref) {
        return Err(TargetAbiError::new(
            "target ABI: closure environment must have exact tribute_rt.anyref type",
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
    callable: HashMap<(TypeRef, CallingConvention), TypeRef>,
}

impl<'a> PhysicalTypeConverter<'a> {
    fn new(ctx: &'a mut IrContext, never: TypeRef) -> Self {
        Self {
            ctx,
            never,
            embedded: HashMap::new(),
            callable: HashMap::new(),
        }
    }

    fn convert_callable(
        &mut self,
        ty: TypeRef,
        convention: CallingConvention,
    ) -> Result<TypeRef, TargetAbiError> {
        if let Some(&converted) = self.callable.get(&(ty, convention)) {
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
        let inputs = self.convert_params(inputs)?;
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
        if attrs.contains_key(func::CALL_CONV_ATTR) {
            return Err(TargetAbiError::new(
                "target ABI: logical callable already carries a machine call_conv",
            ));
        }
        if convention == CallingConvention::Cps {
            func::CallConv::Tail.set_in(&mut attrs);
        }
        let converted =
            func::func_sig_with_param_attrs(self.ctx, inputs, results, attrs).as_type_ref();
        self.callable.insert((ty, convention), converted);
        Ok(converted)
    }

    fn convert_embedded(&mut self, ty: TypeRef) -> Result<TypeRef, TargetAbiError> {
        if let Some(&converted) = self.embedded.get(&ty) {
            return Ok(converted);
        }
        let data = self.ctx.get_type(ty).clone();
        if data.dialect == Symbol::new("closure") && data.name == Symbol::new("closure") {
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
        if data.dialect == func::DIALECT_NAME() && data.name == func::FUNC_SIG() {
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
        let mut converted = data.clone();
        for parameter in &mut converted.params {
            *parameter = self.convert_embedded(*parameter)?;
        }
        self.convert_type_attributes(&mut converted)?;
        // Frame answer provenance is read only by the logical dispatch and
        // root contract checks, which run before conversion. Physical frames
        // are ordinary nominal layouts.
        if cps_continuation_frame_result_type(self.ctx, ty).is_some() {
            converted.attrs.remove(CPS_CONTINUATION_FRAME_RESULT_ATTR);
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
            .map(|(name, value)| (*name, value.clone()))
            .collect::<Vec<_>>();
        attributes
            .into_iter()
            .map(|(name, value)| Ok((name, self.convert_attribute(value)?)))
            .collect()
    }

    fn convert_type_attributes(&mut self, data: &mut TypeData) -> Result<(), TargetAbiError> {
        let attributes: Vec<_> = data
            .attrs
            .iter()
            .map(|(name, value)| (*name, value.clone()))
            .collect();
        for (name, value) in attributes {
            data.attrs.insert(name, self.convert_attribute(value)?);
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
                    .collect::<Result<_, TargetAbiError>>()?;
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
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;

    fn function(ctx: &IrContext, module: Module, name: &str) -> func::Func {
        module
            .ops(ctx)
            .into_iter()
            .find_map(|op| {
                let function = func::Func::from_op(ctx, op).ok()?;
                (function.sym_name(ctx) == Symbol::from_dynamic(name)).then_some(function)
            })
            .unwrap()
    }

    fn is_worker_call(ctx: &IrContext, op: OpRef) -> bool {
        func::Call::from_op(ctx, op)
            .is_ok_and(|call| call.callee(ctx) == Symbol::new(ROOT_MAIN_SYMBOL))
    }

    fn dispatch_fixture(answer_name: &str, frame_name: &str) -> (IrContext, Module, OpRef) {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            &format!(
                r#"core.module @test {{
            !Answer = core.{answer_name}
            !Evidence = core.array(adt.struct() {{name = @_Marker, fields = [[@ability_id, core.i32], [@prompt_tag, core.i32], [@tr_dispatch_fn, core.ptr], [@handler_dispatch, core.ptr]]}})
            !Frame = adt.typeref() {{name = @{frame_name}, tribute.cps_continuation_frame_result = !Answer}}
            !Done = closure.closure(func.func_sig<(!Answer) -> core.never>) {{tribute.calling_convention = 2, tribute.closure_environment_index = 0}}
            !Resume = closure.closure(func.func_sig<(!Evidence, !Frame, tribute_rt.anyref) -> core.never>) {{tribute.calling_convention = 2, tribute.closure_environment_index = 0}}
            !Dispatch = closure.closure(func.func_sig<(!Evidence, !Resume, core.i32, core.i32, core.i32, tribute_rt.anyref) -> core.never>) {{tribute.calling_convention = 2, tribute.closure_environment_index = 1}}
            !{frame_name} = adt.struct() {{name = @{frame_name}, tribute.cps_continuation_frame_result = !Answer, fields = [[@done, !Done], [@dispatch, !Dispatch]]}}
            func.func @run(%ev: !Evidence, %dispatch: !Dispatch, %resume: !Resume, %payload: tribute_rt.anyref) -> core.never attributes {{tribute.calling_convention = 2}} {{
                effect.dispatch_cps %ev, %dispatch, %resume, %payload {{ability_ref = core.ability_ref() {{name = @State}}, op_name = @get, answer_type = !Answer}}
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
            let semantic_answer = ctx.op(dispatch).attributes.get_type("answer_type").unwrap();
            validate_dispatch_contracts(&mut ctx, module).unwrap();
            lower_cps_signatures_to_physical(&mut ctx, module).unwrap();
            assert_eq!(
                ctx.op(dispatch).attributes.get_type("answer_type"),
                Some(semantic_answer)
            );
            let printed = print_module(&ctx, module.op());
            assert!(
                !printed.contains(CPS_CONTINUATION_FRAME_RESULT_ATTR),
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
                    ctx.op_mut(dispatch).attributes.remove("answer_type");
                }
                1 => {
                    ctx.op_mut(dispatch)
                        .attributes
                        .insert(Symbol::new("answer_type"), Attribute::Int(0));
                }
                2 => {
                    let wrong = core::nil(&mut ctx).as_type_ref();
                    ctx.op_mut(dispatch)
                        .attributes
                        .insert(Symbol::new("answer_type"), Attribute::Type(wrong));
                }
                3..=6 => {
                    let index = if mutation == 3 { 1 } else { 2 };
                    let value = ctx.op_operands(dispatch)[index];
                    let mut ty = ctx.get_type(ctx.value_ty(value)).clone();
                    if mutation == 3 || mutation == 4 {
                        ty.attrs.insert(
                            Symbol::new(CLOSURE_ENVIRONMENT_INDEX_ATTR),
                            Attribute::Int(9),
                        );
                    } else if mutation == 5 {
                        ty.attrs
                            .insert(Symbol::new(CALLING_CONVENTION_ATTR), Attribute::Int(0));
                    } else {
                        let signature = func::FuncSig::from_type_ref(&ctx, ty.params[0]).unwrap();
                        let mut inputs = signature.inputs(&ctx).to_vec();
                        let answer = ctx.op(dispatch).attributes.get_type("answer_type").unwrap();
                        inputs[1] =
                            tribute_core::calling_convention::cps_continuation_frame_ref_type(
                                &mut ctx,
                                Symbol::new("other_nominal_frame"),
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
                        ctx.intern_type(
                            TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build(),
                        )
                    } else {
                        core::ptr(&mut ctx).as_type_ref()
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
            !Evidence = core.array(adt.struct() {name = @_Marker, fields = [[@ability_id, core.i32], [@prompt_tag, core.i32], [@tr_dispatch_fn, core.ptr], [@handler_dispatch, core.ptr]]})
            !direct_closure = closure.closure(func.func_sig<() -> ()>) {tribute.calling_convention = 0}
            !evidence_closure = closure.closure(func.func_sig<(!Evidence) -> ()>) {tribute.calling_convention = 1}
            func.func @direct() attributes {tribute.calling_convention = 0} { func.return }
            func.func @evidence(%ev: !Evidence) attributes {tribute.calling_convention = 1} { func.return }
            func.func @unit() -> core.nil attributes {tribute.calling_convention = 0} {
                %nil = arith.const {value = unit} : core.nil
                func.return %nil
            }
            func.func @caller(%ev: !Evidence) attributes {tribute.calling_convention = 1} {
                func.call {callee = @direct, tribute.calling_convention = 0}
                func.call %ev {callee = @evidence, tribute.calling_convention = 1}
                %direct = func.constant {func_ref = @direct} : func.func_sig<() -> ()>
                %evidence = func.constant {func_ref = @evidence} : func.func_sig<(!Evidence) -> ()>
                func.call_indirect %direct {signature = func.func_sig<() -> ()>, tribute.calling_convention = 0}
                func.call_indirect %evidence, %ev {signature = func.func_sig<(!Evidence) -> ()>, tribute.calling_convention = 1}
                %nil = func.call {callee = @unit, tribute.calling_convention = 0} : core.nil
                func.return
            }
            func.func @cps() -> core.never attributes {tribute.calling_convention = 2} { func.unreachable }
        }"#,
        );
        assert_eq!(
            ctx.type_alias_by_name(Symbol::new("Evidence")),
            Some(evidence)
        );
        let cps = function(&ctx, module, "cps");
        let unchanged: Vec<_> = collect_ops(&ctx, module.op())
            .into_iter()
            .filter(|&op| op != cps.op_ref())
            .map(|op| {
                (
                    op,
                    ctx.op(op).attributes.clone(),
                    ctx.op_result_types(op).to_vec(),
                )
            })
            .collect();
        let aliases = ctx.type_aliases().to_vec();
        lower_cps_signatures_to_physical(&mut ctx, module).unwrap();
        for (op, before, results) in unchanged {
            assert_eq!(ctx.op(op).attributes, before);
            assert_eq!(ctx.op_result_types(op), results);
        }
        assert_eq!(ctx.type_aliases(), aliases);
        assert!(
            func::FuncSig::from_type_ref(&ctx, cps.r#type(&ctx))
                .unwrap()
                .results(&ctx)
                .is_empty()
        );
        for op in collect_ops(&ctx, module.op()) {
            if func::CallIndirect::matches(&ctx, op) {
                assert!(ctx.op_result_types(op).is_empty());
                let signature = IndirectCallLikeOps::exact_signature(&ctx, op).unwrap();
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
  !cps = closure.closure(func.func_sig<(core.i32, core.i32, core.i32, core.i32) -> core.never>) {tribute.calling_convention = 2}
  func.func @direct() -> core.never attributes {tribute.calling_convention = 0} { func.unreachable }
  func.func @evidence() -> core.never attributes {tribute.calling_convention = 1} { func.unreachable }
  func.func @cps() -> core.never attributes {tribute.calling_convention = 2} { func.unreachable }
  func.func @run(%callee: core.i32, %evidence: core.i32, %env: tribute_rt.anyref, %done: core.i32, %dispatch: core.i32, %value: core.i32) -> core.never attributes {tribute.calling_convention = 2} {
    func.tail_call_indirect %callee, %evidence, %env, %done, %dispatch, %value {signature = func.func_sig<(core.i32, tribute_rt.anyref, core.i32, core.i32, core.i32) -> core.never>, tribute.calling_convention = 2}
  }
}"#,
        );

        lower_cps_signatures_to_physical(&mut ctx, module).unwrap();

        let never = core::never(&mut ctx).as_type_ref();
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
                "signature = func.func_sig<(core.i32, tribute_rt.anyref, core.i32, core.i32, core.i32) -> ()> {call_conv = @tail}"
            ),
            "{printed}"
        );
        assert!(
            printed.contains(
                "closure.closure(func.func_sig<(core.i32, core.i32, core.i32, core.i32) -> ()> {call_conv = @tail})"
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
  func.func @run(%callee: core.i32, %value: core.i32) -> core.never attributes {tribute.calling_convention = 2} {
    func.tail_call_indirect %callee, %value {signature = func.func_sig<(core.i32) -> core.never>, tribute.calling_convention = 2}
  }
}"#,
        );
        let run = function(&ctx, module, "run");
        let body = run.body_if_present(&ctx).unwrap();
        let indirect = ctx.block(ctx.region(body).blocks[0]).ops[0];
        let signature = IndirectCallLikeOps::exact_signature(&ctx, indirect).unwrap();
        ctx.op_mut(indirect).attributes.insert(
            Symbol::new("unrelated_callable_metadata"),
            Attribute::Type(signature),
        );
        let before = print_module(&ctx, module.op());

        let error = lower_cps_signatures_to_physical(&mut ctx, module).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("untagged nested func.func_sig<(...)->core.never>"),
            "{error}"
        );
        assert_eq!(print_module(&ctx, module.op()), before);
    }

    fn compose_promoted_root(export: CallingConvention) -> (IrContext, Module) {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @main(%evidence: core.i32, %frame: core.i32) -> core.never attributes {tribute.calling_convention = 2} {
    func.unreachable
  }
}"#,
        );
        let main = function(&ctx, module, "main");
        let nil = core::nil(&mut ctx).as_type_ref();
        let never = core::never(&mut ctx).as_type_ref();
        let evidence = ability::evidence_adt_type_ref(&mut ctx);
        let done_callable = func::func_sig(&mut ctx, [nil], [never]).as_type_ref();
        let done = tribute_core::calling_convention::physical_closure_type_with_environment_index(
            &mut ctx,
            done_callable,
            CallingConvention::Cps,
            0,
        );
        let frame_name = Symbol::new("__tribute_continuation_frame_root_nil");
        let frame = tribute_core::calling_convention::cps_continuation_frame_ref_type(
            &mut ctx, frame_name, nil,
        );
        let anyref = tribute_rt::anyref(&mut ctx).as_type_ref();
        let i32_ty = ctx.intern_type(TypeDataBuilder::new("core", "i32").build());
        let dispatch = tribute_core::calling_convention::cps_dispatch_type(
            &mut ctx, evidence, frame, anyref, i32_ty,
        );
        let layout = tribute_core::calling_convention::cps_continuation_frame_layout_type(
            &mut ctx, frame_name, nil, done, dispatch,
        );
        ctx.register_type_alias(frame_name, layout);
        let worker = func::func_sig(&mut ctx, [evidence, frame], [never]).as_type_ref();
        ctx.op_mut(main.op_ref())
            .attributes
            .insert(Symbol::new("type"), Attribute::Type(worker));
        let entry = ctx.region(main.body(&ctx)).blocks[0];
        ctx.set_block_arg_type(entry, 0, evidence);
        ctx.set_block_arg_type(entry, 1, frame);
        ctx.op_mut(main.op_ref()).attributes.insert(
            Symbol::new(ROOT_EXPORT_CONVENTION_ATTR),
            Attribute::Int(export as i128),
        );
        ctx.op_mut(main.op_ref())
            .attributes
            .insert(Symbol::new(ROOT_SOURCE_RESULT_ATTR), Attribute::Type(nil));

        lower_cps_signatures_to_physical(&mut ctx, module).unwrap();
        compose_root_entry_bridge(&mut ctx, module).unwrap();
        (ctx, module)
    }

    #[test]
    fn promoted_direct_root_uses_typed_completion_and_ordinary_call() {
        let (ctx, module) = compose_promoted_root(CallingConvention::Direct);
        let wrapper = function(&ctx, module, "main");
        let worker = function(&ctx, module, ROOT_MAIN_SYMBOL);
        let done_k = function(&ctx, module, ROOT_DONE_K_SYMBOL);
        let dispatch = function(&ctx, module, ROOT_UNHANDLED_SYMBOL);

        assert_eq!(
            get_calling_convention(&ctx, wrapper.op_ref()),
            Some(CallingConvention::Direct)
        );
        for function in [worker, done_k, dispatch] {
            assert_eq!(
                get_calling_convention(&ctx, function.op_ref()),
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
                !ctx.op(function.op_ref())
                    .attributes
                    .contains_key(CLOSURE_ENVIRONMENT_INDEX_ATTR)
            );
        }
        assert!(
            !ctx.op(worker.op_ref())
                .attributes
                .contains_key(ROOT_EXPORT_CONVENTION_ATTR)
                && !ctx
                    .op(worker.op_ref())
                    .attributes
                    .contains_key(ROOT_SOURCE_RESULT_ATTR)
        );

        let wrapper_ops = collect_ops(&ctx, wrapper.op_ref());
        let call = wrapper_ops
            .iter()
            .copied()
            .find(|op| is_worker_call(&ctx, *op))
            .expect("wrapper must make exactly one ordinary worker call");
        assert!(func::Call::from_op(&ctx, call).is_ok());
        assert_eq!(
            ctx.op(call).attributes.get_symbol("callee"),
            Some(Symbol::new(ROOT_MAIN_SYMBOL))
        );
        let worker_callable = func::FuncSig::from_type_ref(&ctx, worker.r#type(&ctx)).unwrap();
        let [worker_evidence, worker_frame] = worker_callable.inputs(&ctx) else {
            panic!("root worker must have Evidence and ContinuationFrame parameters");
        };
        assert_eq!(ctx.value_ty(ctx.op_operands(call)[0]), *worker_evidence);
        assert_eq!(ctx.value_ty(ctx.op_operands(call)[1]), *worker_frame);
        assert_eq!(ctx.get_type(*worker_frame).dialect, Symbol::new("adt"));
        assert_eq!(ctx.get_type(*worker_frame).name, Symbol::new("typeref"));
        let frame_name = ctx
            .get_type(*worker_frame)
            .attrs
            .get_symbol("name")
            .unwrap();
        let frame_layout = ctx
            .type_alias_by_name(frame_name)
            .expect("worker frame must retain its exact nominal layout");
        let fields = ctx.get_type(frame_layout).attrs.get("fields");
        let Attribute::List(fields) = fields.expect("frame fields") else {
            panic!("frame fields must be a list");
        };
        let [Attribute::List(done_field), Attribute::List(dispatch_field)] = fields.as_slice()
        else {
            panic!("frame must have distinct Done and Dispatch fields");
        };
        let [Attribute::Symbol(done_name), Attribute::Type(done_ty)] = done_field.as_slice() else {
            panic!("Done field must retain its exact type");
        };
        let [
            Attribute::Symbol(dispatch_name),
            Attribute::Type(dispatch_ty),
        ] = dispatch_field.as_slice()
        else {
            panic!("Dispatch field must retain its exact type");
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
                    ctx.get_type(*ty).attrs.get_symbol("name")
                        == Some(Symbol::new(ROOT_COMPLETION_CELL_NAME))
                })
        }));
        assert!(
            wrapper_ops
                .iter()
                .any(|op| adt::ArrayNew::from_op(&ctx, *op).is_ok())
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
    fn promoted_evidence_root_has_a_single_parameterless_entry() {
        let (ctx, module) = compose_promoted_root(CallingConvention::EvidenceDirect);
        let wrapper = function(&ctx, module, "main");
        let signature = func::FuncSig::from_type_ref(&ctx, wrapper.r#type(&ctx)).unwrap();
        assert!(signature.inputs(&ctx).is_empty());
        assert_eq!(
            get_calling_convention(&ctx, wrapper.op_ref()),
            Some(CallingConvention::Direct)
        );
        let ops = collect_ops(&ctx, wrapper.op_ref());
        let evidence = ops
            .iter()
            .copied()
            .find_map(|op| adt::ArrayNew::from_op(&ctx, op).ok())
            .expect("wrapper must build initial evidence");
        let call = ops
            .into_iter()
            .find(|op| is_worker_call(&ctx, *op))
            .expect("wrapper must call CPS worker");
        assert_eq!(ctx.op_operands(call)[0], evidence.result(&ctx));
        let functions: Vec<_> = module
            .ops(&ctx)
            .into_iter()
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
            get_calling_convention(ctx, entry_main.op_ref()),
            Some(CallingConvention::Direct)
        );
        let ops = collect_ops(ctx, entry_main.op_ref());
        let empty = ops
            .iter()
            .copied()
            .find_map(|op| adt::ArrayNew::from_op(ctx, op).ok())
            .expect("entry bridge must build initial evidence");
        let call = ops
            .iter()
            .copied()
            .find_map(|op| func::Call::from_op(ctx, op).ok())
            .expect("entry bridge must call the root worker");
        assert_eq!(call.callee(ctx), Symbol::new(ROOT_MAIN_SYMBOL));
        assert_eq!(ctx.op_operands(call.op_ref()), [empty.result(ctx)]);
        assert_eq!(
            get_calling_convention(ctx, call.op_ref()),
            Some(CallingConvention::EvidenceDirect)
        );
    }

    const EVIDENCE_DIRECT_MAIN: &str = r#"core.module @test {
  !Evidence = core.array(adt.struct() {name = @_Marker, fields = [[@ability_id, core.i32], [@prompt_tag, core.i32], [@tr_dispatch_fn, core.ptr], [@handler_dispatch, core.ptr]]})
  func.func @main(%evidence: !Evidence) -> core.nil attributes {tribute.calling_convention = 1} {
    %nil = arith.const {value = unit} : core.nil
    func.return %nil
  }
  func.func @caller(%evidence: !Evidence) -> core.nil attributes {tribute.calling_convention = 1} {
    %result = func.call %evidence {callee = @main, tribute.calling_convention = 1} : core.nil
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
        assert_eq!(call.callee(&ctx), Symbol::new(ROOT_MAIN_SYMBOL));
    }

    #[test]
    fn direct_main_is_wrapped_without_inputs() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @main() -> core.nil attributes {tribute.calling_convention = 0} {
    %nil = arith.const {value = unit} : core.nil
    func.return %nil
  }
  func.func @caller() -> core.nil attributes {tribute.calling_convention = 0} {
    %result = func.call {callee = @main, tribute.calling_convention = 0} : core.nil
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
        assert_eq!(call.callee(&ctx), Symbol::new(ROOT_MAIN_SYMBOL));
        assert!(ctx.op_operands(call.op_ref()).is_empty());
        assert!(
            !ops.iter()
                .any(|op| adt::ArrayNew::from_op(&ctx, *op).is_ok())
        );
        let caller = function(&ctx, module, "caller");
        let caller_call = collect_ops(&ctx, caller.op_ref())
            .into_iter()
            .find_map(|op| func::Call::from_op(&ctx, op).ok())
            .unwrap();
        assert_eq!(caller_call.callee(&ctx), Symbol::new(ROOT_MAIN_SYMBOL));
    }

    #[test]
    fn root_bridge_rejects_non_nil_main() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @main() -> core.i32 attributes {tribute.calling_convention = 0} {
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
        let (ctx, module) = compose_promoted_root(CallingConvention::Direct);
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
  func.func @main(%evidence: core.i32, %frame: core.i32) -> core.never attributes {tribute.calling_convention = 2} {
    func.unreachable
  }
}"#,
            );
            let main = function(&ctx, module, "main");
            let nil = core::nil(&mut ctx).as_type_ref();
            let never = core::never(&mut ctx).as_type_ref();
            let evidence = ability::evidence_adt_type_ref(&mut ctx);
            let frame_name = Symbol::new("__tribute_malformed_root_frame");
            let i32_ty = ctx.intern_type(TypeDataBuilder::new("core", "i32").build());
            let frame = tribute_core::calling_convention::cps_continuation_frame_ref_type(
                &mut ctx,
                frame_name,
                if frame_result { nil } else { i32_ty },
            );
            if malformed_layout {
                let wrong = ctx.intern_type(
                    TypeDataBuilder::new(Symbol::new("adt"), Symbol::new("struct"))
                        .attr("name", Attribute::Symbol(frame_name))
                        .attr("fields", Attribute::List(vec![]))
                        .build(),
                );
                ctx.register_type_alias(frame_name, wrong);
            }
            let worker = func::func_sig(&mut ctx, [evidence, frame], [never]).as_type_ref();
            ctx.op_mut(main.op_ref())
                .attributes
                .insert(Symbol::new("type"), Attribute::Type(worker));
            let entry = ctx.region(main.body(&ctx)).blocks[0];
            ctx.set_block_arg_type(entry, 0, evidence);
            ctx.set_block_arg_type(entry, 1, frame);
            ctx.op_mut(main.op_ref()).attributes.insert(
                Symbol::new(ROOT_EXPORT_CONVENTION_ATTR),
                Attribute::Int(CallingConvention::Direct as i128),
            );
            ctx.op_mut(main.op_ref())
                .attributes
                .insert(Symbol::new(ROOT_SOURCE_RESULT_ATTR), Attribute::Type(nil));

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
  func.func @main(%evidence: core.i32, %frame: core.i32) -> core.never attributes {tribute.calling_convention = 2} {
    func.unreachable
  }
}"#,
        );
        let main = function(&ctx, module, "main");
        let nil = core::nil(&mut ctx).as_type_ref();
        let never = core::never(&mut ctx).as_type_ref();
        let evidence = ability::evidence_adt_type_ref(&mut ctx);
        let frame_name = Symbol::new("__tribute_parameterized_dispatch_tag");
        let frame = tribute_core::calling_convention::cps_continuation_frame_ref_type(
            &mut ctx, frame_name, nil,
        );
        let done = tribute_core::calling_convention::cps_done_type(&mut ctx, nil);
        let anyref = tribute_rt::anyref(&mut ctx).as_type_ref();
        let parameterized_i32 = ctx.intern_type(
            TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32"))
                .param(nil)
                .build(),
        );
        let dispatch = tribute_core::calling_convention::cps_dispatch_type(
            &mut ctx,
            evidence,
            frame,
            anyref,
            parameterized_i32,
        );
        let layout = tribute_core::calling_convention::cps_continuation_frame_layout_type(
            &mut ctx, frame_name, nil, done, dispatch,
        );
        ctx.register_type_alias(frame_name, layout);
        let worker = func::func_sig(&mut ctx, [evidence, frame], [never]).as_type_ref();
        ctx.op_mut(main.op_ref())
            .attributes
            .insert(Symbol::new("type"), Attribute::Type(worker));
        let entry = ctx.region(main.body(&ctx)).blocks[0];
        ctx.set_block_arg_type(entry, 0, evidence);
        ctx.set_block_arg_type(entry, 1, frame);
        ctx.op_mut(main.op_ref()).attributes.insert(
            Symbol::new(ROOT_EXPORT_CONVENTION_ATTR),
            Attribute::Int(CallingConvention::Direct as i128),
        );
        ctx.op_mut(main.op_ref())
            .attributes
            .insert(Symbol::new(ROOT_SOURCE_RESULT_ATTR), Attribute::Type(nil));

        let before = print_module(&ctx, module.op());
        let error = lower_cps_signatures_to_physical(&mut ctx, module).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("frame Dispatch operands differ from the exact terminal ABI"),
            "{error}"
        );
        assert_eq!(print_module(&ctx, module.op()), before);
    }

    #[test]
    fn malformed_transfers_and_ambiguous_nested_never_leave_ir_unchanged() {
        for (input, expected) in [
            (
                r#"core.module @test {
  func.func @run(%callee: core.i32) -> core.never attributes {tribute.calling_convention = 2} {
    func.tail_call_indirect %callee {tribute.calling_convention = 2}
  }
}"#,
                "lacks exact callable signature",
            ),
            (
                r#"core.module @test {
  func.func @pure(%callee: func.func_sig<(core.i32, core.i32) -> core.i32>, %left: core.i32, %right: core.i32) -> core.i32 attributes {tribute.calling_convention = 0} {
    %result = func.call_indirect %callee, %left, %right {tribute.calling_convention = 0} : core.i32
    func.return %result
  }
}"#,
                "lacks exact callable signature",
            ),
            (
                r#"core.module @test {
  func.func @run(%callee: core.i32) -> core.never attributes {tribute.calling_convention = 2} {
    func.tail_call_indirect %callee {signature = core.i32, tribute.calling_convention = 2}
  }
}"#,
                "lacks exact callable signature",
            ),
            (
                r#"core.module @test {
  func.func @run(%callee: core.i32) -> core.never attributes {tribute.calling_convention = 2} {
    func.tail_call_indirect %callee {signature = core.i32}
  }
}"#,
                "indirect signature has no convention metadata",
            ),
            (
                r#"core.module @test {
  func.func @callee(%value: core.i32) -> core.never attributes {tribute.calling_convention = 2} { func.unreachable }
  func.func @run(%value: core.bool) -> core.never attributes {tribute.calling_convention = 2} {
    func.tail_call %value {callee = @callee, tribute.calling_convention = 2}
  }
}"#,
                "operands differ",
            ),
            (
                r#"core.module @test {
  !ambiguous = closure.closure(func.func_sig<() -> core.never>)
}"#,
                "no exact convention metadata",
            ),
        ] {
            let mut ctx = IrContext::new();
            let module = parse_test_module(&mut ctx, input);
            let before = print_module(&ctx, module.op());

            let error = lower_cps_signatures_to_physical(&mut ctx, module).unwrap_err();

            assert!(error.to_string().contains(expected), "{error}");
            assert_eq!(print_module(&ctx, module.op()), before);
        }
    }

    #[test]
    fn raw_closure_storage_never_satisfies_a_semantic_direct_transfer() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !semantic = closure.closure(func.func_sig<() -> core.i32>) {tribute.calling_convention = 0}
  !_closure = adt.struct(core.i32, tribute_rt.anyref) {name = @_closure}
  func.func @factory(%callback: !semantic) -> core.i32 attributes {tribute.calling_convention = 0} {
    func.unreachable
  }
  func.func @run(%raw: !_closure) -> core.i32 attributes {tribute.calling_convention = 0} {
    %result = func.call %raw {callee = @factory, tribute.calling_convention = 0} : core.i32
    func.return %result
  }
}"#,
        );
        let before = print_module(&ctx, module.op());

        let error = lower_cps_signatures_to_physical(&mut ctx, module).unwrap_err();

        assert!(error.to_string().contains("operands differ"), "{error}");
        assert_eq!(print_module(&ctx, module.op()), before);
    }

    #[test]
    fn bodied_and_bodyless_closure_targets_share_environment_provenance() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @external(%evidence: core.i32, %environment: tribute_rt.anyref, %done: core.i32) -> core.never attributes {tribute.calling_convention = 2, tribute.closure_environment_index = 1}
  func.func @defined(%evidence: core.i32, %__env: tribute_rt.anyref, %done: core.i32) -> core.never attributes {tribute.calling_convention = 2, tribute.closure_environment_index = 1} {
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
            printed.contains(
                "func.func @external(%arg0: core.i32, %arg1: tribute_rt.anyref, %arg2: core.i32)"
            ),
            "{printed}"
        );
        assert!(
            printed.contains("func.func @defined")
                && printed.matches("func.constant").count() == 2
                && printed.contains("!t0 = func.func_sig<(core.i32, core.i32) -> ()>")
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
  func.func @generated_zero(%__env: tribute_rt.anyref) -> core.never attributes {tribute.calling_convention = 2, tribute.closure_environment_index = 0} {
    func.unreachable
  }
  func.func @generated_one(%__env: tribute_rt.anyref, %value: core.i32) -> core.never attributes {tribute.calling_convention = 2, tribute.closure_environment_index = 0} {
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

        let anyref = tribute_rt::anyref(&mut ctx).as_type_ref();
        for (name, parameter_count) in [("generated_zero", 1), ("generated_one", 2)] {
            let function = function(&ctx, module, name);
            let signature = func::FuncSig::from_type_ref(&ctx, function.r#type(&ctx)).unwrap();
            assert!(signature.results(&ctx).is_empty());
            assert_eq!(signature.inputs(&ctx).len(), parameter_count);
            assert_eq!(signature.inputs(&ctx)[0], anyref);
        }
        let printed = print_module(&ctx, module.op());
        assert!(printed.contains(": func.func_sig<() -> ()>"), "{printed}");
        assert!(
            printed.contains(": func.func_sig<(core.i32) -> ()>"),
            "{printed}"
        );
    }

    #[test]
    fn malformed_or_missing_environment_provenance_fails_before_mutation() {
        for (function, expected) in [
            (
                "func.func @external(%evidence: core.i32, %environment: tribute_rt.anyref, %done: core.i32) -> core.never attributes {tribute.calling_convention = 2}",
                "function reference differs",
            ),
            (
                "func.func @external(%evidence: core.i32, %environment: tribute_rt.anyref, %done: core.i32) -> core.never attributes {tribute.calling_convention = 2, tribute.closure_environment_index = 3}",
                "outside function signature",
            ),
            (
                "func.func @external(%evidence: core.i32, %environment: core.i32, %done: core.i32) -> core.never attributes {tribute.calling_convention = 2, tribute.closure_environment_index = 1}",
                "exact tribute_rt.anyref",
            ),
            (
                "func.func @external(%evidence: core.i32, %environment: tribute_rt.anyref, %done: core.i32) -> core.never attributes {tribute.calling_convention = 2, tribute.closure_environment_index = 1} { func.unreachable }",
                "no matching `__env`",
            ),
            (
                "func.func @external(%environment: tribute_rt.anyref, %__env: tribute_rt.anyref, %done: core.i32) -> core.never attributes {tribute.calling_convention = 2, tribute.closure_environment_index = 0} { func.unreachable }",
                "differs from `__env` parameter",
            ),
        ] {
            let mut ctx = IrContext::new();
            let module = parse_test_module(
                &mut ctx,
                &format!(
                    r#"core.module @test {{
  {function}
  func.func @holder() -> core.i32 {{
    %function = func.constant {{func_ref = @external}} : func.func_sig<(core.i32, core.i32) -> core.never>
    func.unreachable
  }}
}}"#
                ),
            );
            let before = print_module(&ctx, module.op());

            let error = lower_cps_signatures_to_physical(&mut ctx, module).unwrap_err();

            assert!(error.to_string().contains(expected), "{error}");
            assert_eq!(print_module(&ctx, module.op()), before);
        }
    }

    #[test]
    fn duplicate_environment_provenance_fails_before_mutation() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @external(%__env: tribute_rt.anyref, %environment: tribute_rt.anyref) -> core.never attributes {tribute.calling_convention = 2, tribute.closure_environment_index = 0} {
    func.unreachable
  }
  func.func @holder() -> core.i32 {
    %function = func.constant {func_ref = @external} : func.func_sig<(tribute_rt.anyref) -> core.never>
    func.unreachable
  }
}"#,
        );
        let external = function(&ctx, module, "external");
        let entry = ctx.region(external.body(&ctx)).blocks[0];
        ctx.block_mut(entry).args[1].attrs.insert(
            Symbol::new("bind_name"),
            Attribute::Symbol(Symbol::new("__env")),
        );
        let before = print_module(&ctx, module.op());

        let error = lower_cps_signatures_to_physical(&mut ctx, module).unwrap_err();

        assert!(
            error.to_string().contains("multiple `__env` parameters"),
            "{error}"
        );
        assert_eq!(print_module(&ctx, module.op()), before);
    }

    #[test]
    fn malformed_zero_parameter_func_sig_fails_without_panicking() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, "core.module @test {}");
        let malformed = ctx.intern_type(
            trunk_ir::types::TypeDataBuilder::new(Symbol::new("func"), Symbol::new("func_sig"))
                .build(),
        );
        ctx.op_mut(module.op()).attributes.insert(
            Symbol::new("malformed_callable"),
            Attribute::Type(malformed),
        );
        let before = print_module(&ctx, module.op());

        let error = lower_cps_signatures_to_physical(&mut ctx, module).unwrap_err();

        assert!(
            error.to_string().contains("malformed nested func.func_sig"),
            "{error}"
        );
        assert_eq!(print_module(&ctx, module.op()), before);
    }

    #[test]
    fn already_physical_cps_function_fails_before_target_abi_mutation() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @broken() -> core.never attributes {tribute.calling_convention = 2} { func.unreachable }
}"#,
        );
        let broken = function(&ctx, module, "broken");
        let resultless = func::func_sig(&mut ctx, [], []).as_type_ref();
        ctx.op_mut(broken.op_ref())
            .attributes
            .insert(Symbol::new("type"), Attribute::Type(resultless));
        let before = print_module(&ctx, module.op());

        let error = lower_cps_signatures_to_physical(&mut ctx, module).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("must have logical core.never result"),
            "{error}"
        );
        assert_eq!(print_module(&ctx, module.op()), before);
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
  func.func @external(%value: core.i32) -> core.never attributes {tribute.calling_convention = 2}
  func.func @holder() -> core.i32 {
    %function = func.constant {func_ref = @external} : func.func_sig<(core.bool) -> core.never>
    func.unreachable
  }
}"#,
        );
        let before = print_module(&ctx, malformed.op());

        let error = lower_cps_signatures_to_physical(&mut ctx, malformed).unwrap_err();

        assert!(error.to_string().contains("function reference differs"));
        assert_eq!(print_module(&ctx, malformed.op()), before);
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
        let before = print_module(&ctx, module.op());

        let error = lower_cps_signatures_to_physical(&mut ctx, module).unwrap_err();

        assert!(error.to_string().contains("malformed calling-convention"));
        assert_eq!(print_module(&ctx, module.op()), before);
    }
}
