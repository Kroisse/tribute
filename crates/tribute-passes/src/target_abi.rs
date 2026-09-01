//! Shared target-neutral physicalization of convention-proven CPS signatures.
//!
//! The pass is intentionally not wired into the production pipeline here. It
//! consumes exact callable metadata, validates the whole transfer surface, and
//! only then maps logical CPS `core.never` results to the shared empty-result
//! marker used by target backends.

use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::ops::ControlFlow;

use tribute_core::calling_convention::CLOSURE_ENVIRONMENT_INDEX_ATTR;
use tribute_core::{
    CALLING_CONVENTION_ATTR, CallingConvention, INDIRECT_CALL_SIGNATURE_ATTR,
    get_calling_convention, get_indirect_call_signature, get_physical_closure_convention,
};
use tribute_ir::dialect::{ability, closure, tribute_rt};
use trunk_ir::Symbol;
use trunk_ir::context::{BlockArgData, BlockData, IrContext, RegionData};
use trunk_ir::dialect::{adt, arith, core, func};
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::refs::{OpRef, TypeRef, ValueRef};
use trunk_ir::rewrite::Module;
use trunk_ir::smallvec::smallvec;
use trunk_ir::types::{Attribute, AttributeMap, TypeData, TypeDataBuilder};
use trunk_ir::walk::{WalkAction, walk_op};

const ROOT_EXPORT_CONVENTION_ATTR: &str = "tribute.root_export_convention";
const ROOT_SOURCE_RESULT_ATTR: &str = "tribute.root_source_result";
const CPS_MAIN_SYMBOL: &str = "__tribute_cps_main";
const ROOT_DONE_K_SYMBOL: &str = "__tribute_root_done_k";
const ROOT_COMPLETION_CELL_NAME: &str = "__tribute_root_completion_cell";
const ROOT_COMPLETION_CELL_VALUE_FIELD: &str = "value";
const ROOT_CPS_CALL_ATTR: &str = "tribute.root_cps_call";

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
    let nil = core::nil(ctx).as_type_ref();
    let anyref = tribute_rt::anyref(ctx).as_type_ref();
    let functions = collect_functions(ctx, &ops, never, anyref)?;
    validate_transfers(ctx, &ops, &functions, never)?;

    let aliases = ctx.type_aliases().to_vec();
    let mut converter = PhysicalTypeConverter::new(ctx, never, nil);
    let mut alias_updates = Vec::new();
    let mut function_types = Vec::new();
    let mut result_types = Vec::new();
    let mut attributes = Vec::new();
    let mut block_args = Vec::new();

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
            if convention.is_some() {
                let identity = function_for_symbol(
                    converter.ctx,
                    op,
                    function.sym_name(converter.ctx),
                    &functions,
                )?;
                if let Some(index) = identity.environment_index
                    && converter
                        .ctx
                        .op(op)
                        .attributes
                        .get(Symbol::new(CLOSURE_ENVIRONMENT_INDEX_ATTR))
                        .is_none()
                {
                    attributes.push((
                        op,
                        Symbol::new(CLOSURE_ENVIRONMENT_INDEX_ATTR),
                        Attribute::Int(index as i128),
                    ));
                }
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
                if let Some(identity) = function_for_symbol_optional(
                    converter.ctx,
                    op,
                    constant.func_ref(converter.ctx),
                    &functions,
                )? {
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

        let op_attributes: Vec<_> = converter
            .ctx
            .op(op)
            .attributes
            .iter()
            .map(|(name, value)| (*name, value.clone()))
            .collect();
        for (name, value) in op_attributes {
            if func::Func::matches(converter.ctx, op) && name == Symbol::new("type") {
                continue;
            }
            let converted = if name == Symbol::new(INDIRECT_CALL_SIGNATURE_ATTR) {
                let Attribute::Type(signature) = value else {
                    return Err(TargetAbiError::new(
                        "target ABI: indirect callable signature must be a type attribute",
                    ));
                };
                let convention = exact_convention(converter.ctx, op)?.ok_or_else(|| {
                    TargetAbiError::new(
                        "target ABI: indirect callable signature has no convention metadata",
                    )
                })?;
                Attribute::Type(converter.convert_callable(signature, convention)?)
            } else {
                converter.convert_attribute(value.clone())?
            };
            if converted != value {
                attributes.push((op, name, converted));
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
    for (block, index, ty) in block_args {
        ctx.set_block_arg_type(block, index, ty);
    }
    Ok(())
}

/// Whether this module carries the exact root metadata owned by the logical
/// CPS route. Until that route is enabled, legacy modules must not enter the
/// physicalization boundary: their compatibility calling-convention markers
/// are not whole-program CPS ABI provenance.
pub fn has_root_entry_contract(ctx: &IrContext, module: Module) -> bool {
    let Some(block) = module.first_block(ctx) else {
        return false;
    };
    ctx.block(block).ops.iter().copied().any(|op| {
        func::Func::from_op(ctx, op).is_ok_and(|function| {
            function.sym_name(ctx) == Symbol::new("main")
                && (ctx
                    .op(op)
                    .attributes
                    .contains_key(ROOT_EXPORT_CONVENTION_ATTR)
                    || ctx.op(op).attributes.contains_key(ROOT_SOURCE_RESULT_ATTR))
        })
    })
}

/// Construct the target-independent export delimiter for a root worker that
/// was promoted from Direct or EvidenceDirect to Cps.
///
/// This runs after physical signature lowering: the worker and `done_k` use
/// the target's empty `core.nil` result, while the wrapper keeps the exact
/// source ABI and performs one ordinary call. Legacy modules carry no root
/// metadata, so they are left unchanged.
pub fn compose_root_entry_bridge(
    ctx: &mut IrContext,
    module: Module,
) -> Result<(), TargetAbiError> {
    let Some(module_block) = module.first_block(ctx) else {
        return Ok(());
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
        return Ok(());
    };

    let export_convention = root_export_convention(ctx, worker_op)?;
    let source_result = root_source_result(ctx, worker_op)?;
    if export_convention.is_some() != source_result.is_some() {
        return Err(TargetAbiError::new(
            "target root bridge: preserved export convention and source result must be paired",
        ));
    }
    let Some(export_convention) = export_convention else {
        return Ok(());
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
    let worker_callable = core::Func::from_type_ref(ctx, worker.r#type(ctx))
        .ok_or_else(|| TargetAbiError::new("target root bridge: main is not core.func"))?;
    let evidence_ty = ability::evidence_adt_type_ref(ctx);
    let worker_params = worker_callable.params(ctx);
    if worker_callable.r#return(ctx) != nil_ty
        || worker_params.len() != 2
        || worker_params[0] != evidence_ty
    {
        return Err(TargetAbiError::new(
            "target root bridge: Cps root must have exact physical evidence and done_k ABI",
        ));
    }
    let done_k_ty = worker_params[1];
    validate_root_done_k_type(ctx, done_k_ty, source_result, nil_ty)?;
    if ctx.op(worker_op).regions.is_empty() {
        return Err(TargetAbiError::new(
            "target root bridge: root worker must be a definition",
        ));
    }

    let cps_main = Symbol::new(CPS_MAIN_SYMBOL);
    let root_done_k = Symbol::new(ROOT_DONE_K_SYMBOL);
    for &op in &top_level_ops {
        let Ok(function) = func::Func::from_op(ctx, op) else {
            continue;
        };
        if matches!(function.sym_name(ctx), name if name == cps_main || name == root_done_k) {
            return Err(TargetAbiError::new(
                "target root bridge: reserved root symbol collision",
            ));
        }
    }

    let location = ctx.op(worker_op).location;
    ctx.op_mut(worker_op)
        .attributes
        .insert(Symbol::new("sym_name"), Attribute::Symbol(cps_main));
    remove_root_contract(ctx, worker_op);
    for &op in &top_level_ops {
        rewrite_symbol_refs(ctx, op, Symbol::new("main"), cps_main);
    }

    let cell_ty = root_completion_cell_type(ctx, source_result);
    let anyref_ty = tribute_rt::anyref(ctx).as_type_ref();
    let done_callable_ty = core::func(ctx, nil_ty, [source_result]).as_type_ref();
    let done_function_ty = core::func(ctx, nil_ty, [anyref_ty, source_result]).as_type_ref();
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
    let cell = adt::ref_cast(ctx, location, done_args[0], cell_ty, cell_ty);
    ctx.push_op(done_entry, cell.op_ref());
    let store = adt::struct_set(ctx, location, cell.result(ctx), done_args[1], cell_ty, 0);
    ctx.push_op(done_entry, store.op_ref());
    let done_nil = arith::r#const(ctx, location, nil_ty, Attribute::Unit);
    ctx.push_op(done_entry, done_nil.op_ref());
    let done_return = func::r#return(ctx, location, [done_nil.result(ctx)]);
    ctx.push_op(done_entry, done_return.op_ref());
    let done_region = ctx.create_region(RegionData {
        location,
        blocks: smallvec![done_entry],
        parent_op: None,
    });
    let done_function = func::func(ctx, location, root_done_k, done_function_ty, done_region);
    set_root_convention(ctx, done_function.op_ref(), CallingConvention::Cps);
    ctx.op_mut(done_function.op_ref()).attributes.insert(
        Symbol::new(CLOSURE_ENVIRONMENT_INDEX_ATTR),
        Attribute::Int(0),
    );

    let wrapper_params = if export_convention == CallingConvention::EvidenceDirect {
        vec![evidence_ty]
    } else {
        vec![]
    };
    let wrapper_entry = ctx.create_block(BlockData {
        location,
        args: wrapper_params
            .iter()
            .copied()
            .map(|ty| BlockArgData {
                ty,
                attrs: bind_name("__evidence"),
            })
            .collect(),
        ops: smallvec![],
        parent_region: None,
    });
    let initial = arith::r#const(ctx, location, source_result, Attribute::Unit);
    ctx.push_op(wrapper_entry, initial.op_ref());
    let cell_new = adt::struct_new(ctx, location, [initial.result(ctx)], cell_ty, cell_ty);
    ctx.push_op(wrapper_entry, cell_new.op_ref());
    let erased_cell =
        core::unrealized_conversion_cast(ctx, location, cell_new.result(ctx), anyref_ty);
    ctx.push_op(wrapper_entry, erased_cell.op_ref());
    let done_constant = func::constant(ctx, location, done_callable_ty, root_done_k);
    ctx.push_op(wrapper_entry, done_constant.op_ref());
    let closure_struct_ty = crate::closure_lower::closure_struct_type_ref(ctx);
    let done_closure = adt::struct_new(
        ctx,
        location,
        [done_constant.result(ctx), erased_cell.result(ctx)],
        closure_struct_ty,
        closure_struct_ty,
    );
    ctx.push_op(wrapper_entry, done_closure.op_ref());
    let typed_done =
        core::unrealized_conversion_cast(ctx, location, done_closure.result(ctx), done_k_ty);
    ctx.push_op(wrapper_entry, typed_done.op_ref());

    let evidence = if export_convention == CallingConvention::EvidenceDirect {
        ctx.block_args(wrapper_entry)[0]
    } else {
        let i32_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());
        let zero = arith::r#const(ctx, location, i32_ty, Attribute::Int(0));
        ctx.push_op(wrapper_entry, zero.op_ref());
        let empty = adt::array_new(ctx, location, [zero.result(ctx)], evidence_ty, evidence_ty);
        ctx.push_op(wrapper_entry, empty.op_ref());
        empty.result(ctx)
    };
    let worker_call = func::call(
        ctx,
        location,
        [evidence, typed_done.result(ctx)],
        nil_ty,
        cps_main,
    );
    set_root_convention(ctx, worker_call.op_ref(), CallingConvention::Cps);
    ctx.op_mut(worker_call.op_ref())
        .attributes
        .insert(Symbol::new(ROOT_CPS_CALL_ATTR), Attribute::Bool(true));
    ctx.push_op(wrapper_entry, worker_call.op_ref());
    let completed = adt::struct_get(
        ctx,
        location,
        cell_new.result(ctx),
        source_result,
        cell_ty,
        0,
    );
    ctx.push_op(wrapper_entry, completed.op_ref());
    let wrapper_return = func::r#return(ctx, location, [completed.result(ctx)]);
    ctx.push_op(wrapper_entry, wrapper_return.op_ref());
    let wrapper_region = ctx.create_region(RegionData {
        location,
        blocks: smallvec![wrapper_entry],
        parent_op: None,
    });
    let wrapper_ty = core::func(ctx, source_result, wrapper_params.iter().copied()).as_type_ref();
    let wrapper = func::func(
        ctx,
        location,
        Symbol::new("main"),
        wrapper_ty,
        wrapper_region,
    );
    set_root_convention(ctx, wrapper.op_ref(), export_convention);

    ctx.push_op(module_block, done_function.op_ref());
    ctx.push_op(module_block, wrapper.op_ref());
    Ok(())
}

fn root_completion_cell_type(ctx: &mut IrContext, value_ty: TypeRef) -> TypeRef {
    ctx.types.intern(TypeData {
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

fn validate_root_done_k_type(
    ctx: &IrContext,
    done_k_ty: TypeRef,
    source_result: TypeRef,
    physical_result: TypeRef,
) -> Result<(), TargetAbiError> {
    if get_physical_closure_convention(ctx, done_k_ty) != Some(CallingConvention::Cps) {
        return Err(TargetAbiError::new(
            "target root bridge: done_k must carry exact Cps closure provenance",
        ));
    }
    let done = closure::Closure::from_type_ref(ctx, done_k_ty)
        .ok_or_else(|| TargetAbiError::new("target root bridge: done_k is not a closure"))?;
    let callable = core::Func::from_type_ref(ctx, done.func_type(ctx))
        .ok_or_else(|| TargetAbiError::new("target root bridge: done_k is not core.func"))?;
    if callable.r#return(ctx) != physical_result || callable.params(ctx) != [source_result] {
        return Err(TargetAbiError::new(
            "target root bridge: done_k must accept the exact source result and return empty",
        ));
    }
    Ok(())
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
) -> Result<HashMap<(OpRef, Symbol), FunctionIdentity>, TargetAbiError> {
    let mut functions = HashMap::new();
    for &op in ops {
        let Ok(function) = func::Func::from_op(ctx, op) else {
            continue;
        };
        let Some(convention) = exact_convention(ctx, op)? else {
            continue;
        };
        let signature = function.r#type(ctx);
        let callable = core::Func::from_type_ref(ctx, signature).ok_or_else(|| {
            TargetAbiError::new("target ABI: tagged function must have a core.func signature")
        })?;
        if convention == CallingConvention::Cps && callable.r#return(ctx) != never {
            return Err(TargetAbiError::new(format!(
                "target ABI: Cps function `{}` must have logical core.never result",
                function.sym_name(ctx)
            )));
        }
        let key = (symbol_scope(ctx, op)?, function.sym_name(ctx));
        let identity = FunctionIdentity {
            signature,
            convention,
            environment_index: environment_index(ctx, op, callable.params(ctx), anyref)?,
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
    functions: &HashMap<(OpRef, Symbol), FunctionIdentity>,
    never: TypeRef,
) -> Result<(), TargetAbiError> {
    for &op in ops {
        if func::Call::matches(ctx, op) || func::TailCall::matches(ctx, op) {
            let Some(convention) = exact_convention(ctx, op)? else {
                continue;
            };
            let callee = ctx.op(op).attributes.get_symbol("callee").ok_or_else(|| {
                TargetAbiError::new("target ABI: direct transfer lacks callee metadata")
            })?;
            let identity = function_for_symbol(ctx, op, callee, functions)?;
            if identity.convention != convention {
                return Err(TargetAbiError::new(
                    "target ABI: direct transfer convention differs from callee",
                ));
            }
            let callable = core::Func::from_type_ref(ctx, identity.signature).unwrap();
            if !operands_match(ctx, ctx.op_operands(op), callable.params(ctx)) {
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
                if ctx.op_result_types(op) != [callable.r#return(ctx)] {
                    return Err(TargetAbiError::new(
                        "target ABI: direct call result differs from callee signature",
                    ));
                }
            } else if convention != CallingConvention::Cps
                || callable.r#return(ctx) != never
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
        let signature = get_indirect_call_signature(ctx, op);
        let convention = exact_convention(ctx, op)?;
        if signature.is_none() && convention.is_none() {
            continue;
        }
        let convention = convention.ok_or_else(|| {
            TargetAbiError::new("target ABI: indirect signature has no convention metadata")
        })?;
        let signature = signature.ok_or_else(|| {
            TargetAbiError::new("target ABI: indirect transfer lacks exact callable signature")
        })?;
        let callable = core::Func::from_type_ref(ctx, signature).ok_or_else(|| {
            TargetAbiError::new("target ABI: indirect callable signature is not core.func")
        })?;
        let args = ctx.op_operands(op).get(1..).unwrap_or_default();
        if !operands_match(ctx, args, callable.params(ctx)) {
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
            if ctx.op_result_types(op) != [callable.r#return(ctx)] {
                return Err(TargetAbiError::new(
                    "target ABI: indirect call result differs from exact callable signature",
                ));
            }
        } else if convention != CallingConvention::Cps
            || callable.r#return(ctx) != never
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
                core::Func::from_type_ref(ctx, function.r#type(ctx)).ok_or_else(|| {
                    TargetAbiError::new("target ABI: enclosing function is not core.func")
                })?;
            return Ok(
                exact_convention(ctx, candidate)? == Some(CallingConvention::Cps)
                    && callable.r#return(ctx) == never,
            );
        }
        current = parent_op(ctx, candidate);
    }
    Err(TargetAbiError::new(
        "target ABI: tail transfer has no enclosing function",
    ))
}

fn function_for_symbol(
    ctx: &IrContext,
    op: OpRef,
    symbol: Symbol,
    functions: &HashMap<(OpRef, Symbol), FunctionIdentity>,
) -> Result<FunctionIdentity, TargetAbiError> {
    function_for_symbol_optional(ctx, op, symbol, functions)?
        .ok_or_else(|| TargetAbiError::new(format!("target ABI: unknown callable `{symbol}`")))
}

fn function_for_symbol_optional(
    ctx: &IrContext,
    op: OpRef,
    symbol: Symbol,
    functions: &HashMap<(OpRef, Symbol), FunctionIdentity>,
) -> Result<Option<FunctionIdentity>, TargetAbiError> {
    Ok(functions.get(&(symbol_scope(ctx, op)?, symbol)).copied())
}

fn validate_constant(
    ctx: &mut IrContext,
    constant: func::Constant,
    identity: FunctionIdentity,
    never: TypeRef,
) -> Result<(), TargetAbiError> {
    let target = core::Func::from_type_ref(ctx, identity.signature).unwrap();
    if identity.convention == CallingConvention::Cps && target.r#return(ctx) != never {
        return Err(TargetAbiError::new(
            "target ABI: Cps function reference must have logical core.never result",
        ));
    }
    let mut params = target.params(ctx).to_vec();
    if let Some(index) = identity.environment_index {
        if index >= params.len() {
            return Err(TargetAbiError::new(
                "target ABI: closure environment index is outside target signature",
            ));
        }
        params.remove(index);
    }
    let expected = core::func(ctx, target.r#return(ctx), params).as_type_ref();
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

fn symbol_scope(ctx: &IrContext, op: OpRef) -> Result<OpRef, TargetAbiError> {
    let mut current = Some(op);
    while let Some(candidate) = current {
        if core::Module::matches(ctx, candidate) {
            return Ok(candidate);
        }
        current = parent_op(ctx, candidate);
    }
    Err(TargetAbiError::new(
        "target ABI: operation has no enclosing module",
    ))
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
    nil: TypeRef,
    embedded: HashMap<TypeRef, TypeRef>,
    callable: HashMap<(TypeRef, CallingConvention), TypeRef>,
}

impl<'a> PhysicalTypeConverter<'a> {
    fn new(ctx: &'a mut IrContext, never: TypeRef, nil: TypeRef) -> Self {
        Self {
            ctx,
            never,
            nil,
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
        let mut data = self.ctx.types.get(ty).clone();
        if data.dialect != Symbol::new("core")
            || data.name != Symbol::new("func")
            || data.params.is_empty()
        {
            return Err(TargetAbiError::new(
                "target ABI: proven callable is not core.func",
            ));
        }
        if convention == CallingConvention::Cps && data.params[0] != self.never {
            return Err(TargetAbiError::new(
                "target ABI: Cps callable must have logical core.never result",
            ));
        }
        data.params[0] = if convention == CallingConvention::Cps {
            self.nil
        } else {
            self.convert_embedded(data.params[0])?
        };
        for parameter in &mut data.params[1..] {
            *parameter = self.convert_embedded(*parameter)?;
        }
        self.convert_type_attributes(&mut data)?;
        let converted = self.intern_if_changed(ty, data);
        self.callable.insert((ty, convention), converted);
        Ok(converted)
    }

    fn convert_embedded(&mut self, ty: TypeRef) -> Result<TypeRef, TargetAbiError> {
        if let Some(&converted) = self.embedded.get(&ty) {
            return Ok(converted);
        }
        let data = self.ctx.types.get(ty).clone();
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
        if data.dialect == Symbol::new("core") && data.name == Symbol::new("func") {
            if data.params.is_empty() {
                return Err(TargetAbiError::new(
                    "target ABI: core.func type has no result parameter",
                ));
            }
            let callable = core::Func::from_type_ref(self.ctx, ty).unwrap();
            if callable.r#return(self.ctx) == self.never {
                return Err(TargetAbiError::new(
                    "target ABI: untagged nested core.func<core.never, ...>",
                ));
            }
        }
        let mut converted = data.clone();
        for parameter in &mut converted.params {
            *parameter = self.convert_embedded(*parameter)?;
        }
        self.convert_type_attributes(&mut converted)?;
        let converted = self.intern_if_changed(ty, converted);
        self.embedded.insert(ty, converted);
        Ok(converted)
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
        match attribute {
            Attribute::Type(ty) => Ok(Attribute::Type(self.convert_embedded(ty)?)),
            Attribute::List(values) => Ok(Attribute::List(
                values
                    .into_iter()
                    .map(|value| self.convert_attribute(value))
                    .collect::<Result<_, _>>()?,
            )),
            other => Ok(other),
        }
    }

    fn intern_if_changed(&mut self, original: TypeRef, data: TypeData) -> TypeRef {
        if data == *self.ctx.types.get(original) {
            original
        } else {
            self.ctx.types.intern(data)
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

    #[test]
    fn physicalizes_dispatch_aware_exact_cps_contracts_only() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !cps = closure.closure(core.func(core.never, core.i32, core.i32, core.i32, core.i32)) {tribute.calling_convention = 2}
  func.func @direct() -> core.never attributes {tribute.calling_convention = 0} { func.unreachable }
  func.func @evidence() -> core.never attributes {tribute.calling_convention = 1} { func.unreachable }
  func.func @cps() -> core.never attributes {tribute.calling_convention = 2} { func.unreachable }
  func.func @run(%callee: core.i32, %evidence: core.i32, %env: tribute_rt.anyref, %done: core.i32, %dispatch: core.i32, %value: core.i32) -> core.never attributes {tribute.calling_convention = 2} {
    func.tail_call_indirect %callee, %evidence, %env, %done, %dispatch, %value {func.indirect_call_signature = core.func(core.never, core.i32, tribute_rt.anyref, core.i32, core.i32, core.i32), tribute.calling_convention = 2}
  }
}"#,
        );

        lower_cps_signatures_to_physical(&mut ctx, module).unwrap();

        let never = core::never(&mut ctx).as_type_ref();
        let nil = core::nil(&mut ctx).as_type_ref();
        for (name, expected) in [
            ("direct", never),
            ("evidence", never),
            ("cps", nil),
            ("run", nil),
        ] {
            let signature = function(&ctx, module, name).r#type(&ctx);
            assert_eq!(
                core::Func::from_type_ref(&ctx, signature)
                    .unwrap()
                    .r#return(&ctx),
                expected
            );
        }
        let printed = print_module(&ctx, module.op());
        assert!(
            printed.contains("func.indirect_call_signature = core.func(core.nil"),
            "{printed}"
        );
        assert!(
            printed.contains("closure.closure(core.func(core.nil"),
            "{printed}"
        );
    }

    fn compose_promoted_root(export: CallingConvention) -> (IrContext, Module) {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @main(%evidence: core.i32, %done: core.i32) -> core.never attributes {tribute.calling_convention = 2} {
    func.unreachable
  }
}"#,
        );
        let main = function(&ctx, module, "main");
        let nil = core::nil(&mut ctx).as_type_ref();
        let never = core::never(&mut ctx).as_type_ref();
        let evidence = ability::evidence_adt_type_ref(&mut ctx);
        let done_callable = core::func(&mut ctx, never, [nil]).as_type_ref();
        let done = tribute_core::calling_convention::physical_closure_type_with_environment_index(
            &mut ctx,
            done_callable,
            CallingConvention::Cps,
            0,
        );
        let worker = core::func(&mut ctx, never, [evidence, done]).as_type_ref();
        ctx.op_mut(main.op_ref())
            .attributes
            .insert(Symbol::new("type"), Attribute::Type(worker));
        let entry = ctx.region(main.body(&ctx)).blocks[0];
        ctx.set_block_arg_type(entry, 0, evidence);
        ctx.set_block_arg_type(entry, 1, done);
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
        let (mut ctx, module) = compose_promoted_root(CallingConvention::Direct);
        let nil = core::nil(&mut ctx).as_type_ref();
        let wrapper = function(&ctx, module, "main");
        let worker = function(&ctx, module, CPS_MAIN_SYMBOL);
        let done_k = function(&ctx, module, ROOT_DONE_K_SYMBOL);

        assert_eq!(
            get_calling_convention(&ctx, wrapper.op_ref()),
            Some(CallingConvention::Direct)
        );
        for function in [worker, done_k] {
            assert_eq!(
                get_calling_convention(&ctx, function.op_ref()),
                Some(CallingConvention::Cps)
            );
            assert_eq!(
                core::Func::from_type_ref(&ctx, function.r#type(&ctx))
                    .unwrap()
                    .r#return(&ctx),
                nil
            );
        }
        assert_eq!(
            ctx.op(done_k.op_ref())
                .attributes
                .get_u32(CLOSURE_ENVIRONMENT_INDEX_ATTR),
            Ok(Some(0))
        );
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
            .find(|op| ctx.op(*op).attributes.get_bool(ROOT_CPS_CALL_ATTR) == Some(true))
            .expect("wrapper must make exactly one marked ordinary worker call");
        assert!(func::Call::from_op(&ctx, call).is_ok());
        assert_eq!(
            ctx.op(call).attributes.get_symbol("callee"),
            Some(Symbol::new(CPS_MAIN_SYMBOL))
        );
        assert!(
            wrapper_ops
                .iter()
                .all(|op| func::TailCall::from_op(&ctx, *op).is_err())
        );
        assert!(wrapper_ops.iter().any(|op| {
            adt::StructNew::from_op(&ctx, *op).is_ok()
                && ctx.op_result_types(*op).first().is_some_and(|ty| {
                    ctx.types.get(*ty).attrs.get_symbol("name")
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
    }

    #[test]
    fn promoted_evidence_root_forwards_its_exact_evidence_argument() {
        let (ctx, module) = compose_promoted_root(CallingConvention::EvidenceDirect);
        let wrapper = function(&ctx, module, "main");
        assert_eq!(
            get_calling_convention(&ctx, wrapper.op_ref()),
            Some(CallingConvention::EvidenceDirect)
        );
        let entry = ctx.region(wrapper.body(&ctx)).blocks[0];
        let evidence = ctx.block_args(entry)[0];
        let call = collect_ops(&ctx, wrapper.op_ref())
            .into_iter()
            .find(|op| ctx.op(*op).attributes.get_bool(ROOT_CPS_CALL_ATTR) == Some(true))
            .expect("wrapper must call CPS worker");
        assert_eq!(ctx.op_operands(call)[0], evidence);
        assert!(
            !collect_ops(&ctx, wrapper.op_ref())
                .iter()
                .any(|op| adt::ArrayNew::from_op(&ctx, *op).is_ok()),
            "EvidenceDirect wrapper must forward its exact evidence instead of synthesizing one"
        );
    }

    #[test]
    fn root_bridge_creates_no_lambda_for_target_phase() {
        let (ctx, module) = compose_promoted_root(CallingConvention::Direct);
        let printed = print_module(&ctx, module.op());

        assert!(!printed.contains("closure.lambda"), "{printed}");
        assert!(printed.contains("adt.struct_new"), "{printed}");
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
  func.func @callee(%value: core.i32) -> core.never attributes {tribute.calling_convention = 2} { func.unreachable }
  func.func @run(%value: core.bool) -> core.never attributes {tribute.calling_convention = 2} {
    func.tail_call %value {callee = @callee, tribute.calling_convention = 2}
  }
}"#,
                "operands differ",
            ),
            (
                r#"core.module @test {
  !ambiguous = closure.closure(core.func(core.never))
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
  !semantic = closure.closure(core.func(core.i32)) {tribute.calling_convention = 0}
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
    %external = func.constant {func_ref = @external} : core.func(core.never, core.i32, core.i32)
    %defined = func.constant {func_ref = @defined} : core.func(core.never, core.i32, core.i32)
    func.unreachable
  }
}"#,
        );

        lower_cps_signatures_to_physical(&mut ctx, module).unwrap();

        let printed = print_module(&ctx, module.op());
        assert!(
            printed.contains(
                "func.func @external(%arg0: core.i32, %arg1: tribute_rt.anyref, %arg2: core.i32) -> core.nil"
            ),
            "{printed}"
        );
        assert!(
            printed.contains("func.func @defined")
                && printed.matches("func.constant").count() == 2
                && printed
                    .matches("core.func(core.nil, core.i32, core.i32)")
                    .count()
                    == 2,
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
    %zero = func.constant {func_ref = @generated_zero} : core.func(core.never)
    %one = func.constant {func_ref = @generated_one} : core.func(core.never, core.i32)
    func.unreachable
  }
}"#,
        );

        lower_cps_signatures_to_physical(&mut ctx, module).unwrap();

        let anyref = tribute_rt::anyref(&mut ctx).as_type_ref();
        let nil = core::nil(&mut ctx).as_type_ref();
        for (name, parameter_count) in [("generated_zero", 1), ("generated_one", 2)] {
            let function = function(&ctx, module, name);
            let signature = core::Func::from_type_ref(&ctx, function.r#type(&ctx)).unwrap();
            assert_eq!(signature.r#return(&ctx), nil);
            assert_eq!(signature.params(&ctx).len(), parameter_count);
            assert_eq!(signature.params(&ctx)[0], anyref);
        }
        let printed = print_module(&ctx, module.op());
        assert!(printed.contains(": core.func(core.nil)"), "{printed}");
        assert!(
            printed.contains(": core.func(core.nil, core.i32)"),
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
    %function = func.constant {{func_ref = @external}} : core.func(core.never, core.i32, core.i32)
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
    %function = func.constant {func_ref = @external} : core.func(core.never, tribute_rt.anyref)
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
    fn malformed_zero_parameter_core_func_fails_without_panicking() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, "core.module @test {}");
        let malformed = ctx.types.intern(
            trunk_ir::types::TypeDataBuilder::new(Symbol::new("core"), Symbol::new("func")).build(),
        );
        ctx.op_mut(module.op()).attributes.insert(
            Symbol::new("malformed_callable"),
            Attribute::Type(malformed),
        );
        let before = print_module(&ctx, module.op());

        let error = lower_cps_signatures_to_physical(&mut ctx, module).unwrap_err();

        assert!(
            error.to_string().contains("has no result parameter"),
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
    %function = func.constant {func_ref = @external} : core.func(core.i32, core.i32)
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
    %function = func.constant {func_ref = @external} : core.func(core.never, core.bool)
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
