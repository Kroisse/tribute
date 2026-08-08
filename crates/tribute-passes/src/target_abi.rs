//! Target-owned physical CPS signatures and root entry bridge.
//!
//! Shared legalization deliberately keeps `core.never` as the logical bottom
//! result of CPS callables. Native and Wasm call this module after shared
//! closure/effect lowering to select the physical empty-result ABI and, when
//! the exact root export was promoted to CPS, delimit it with an ordinary
//! Direct/EvidenceDirect wrapper.

use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt;
use std::ops::ControlFlow;

use tribute_core::{
    CALLING_CONVENTION_ATTR, CallingConvention, INDIRECT_CALL_SIGNATURE_ATTR,
    ROOT_EXPORT_CONVENTION_ATTR, ROOT_SOURCE_RESULT_ATTR, get_calling_convention,
    get_indirect_call_signature, get_physical_closure_convention, get_root_export_convention,
    get_root_source_result, set_calling_convention,
};
use tribute_ir::dialect::{ability, closure, tribute_rt};
use trunk_ir::Symbol;
use trunk_ir::context::{BlockArgData, BlockData, IrContext, RegionData};
use trunk_ir::dialect::{adt, arith, core, func, scf};
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::refs::{BlockRef, OpRef, TypeRef};
use trunk_ir::rewrite::Module;
use trunk_ir::smallvec::smallvec;
use trunk_ir::types::{Attribute, AttributeMap, TypeData};
use trunk_ir::walk::{WalkAction, walk_op};

pub const ROOT_CPS_CALL_ATTR: &str = "tribute.root_cps_call";
pub const ROOT_DONE_K_ATTR: &str = "tribute.root_done_k";
pub const ROOT_CPS_WORKER_ATTR: &str = "tribute.root_cps_worker";
pub const ROOT_WRAPPER_ATTR: &str = "tribute.root_wrapper";

const CPS_MAIN_SYMBOL: &str = "__tribute_cps_main";
const ROOT_DONE_K_SYMBOL: &str = "__tribute_root_done_k";
const ROOT_REJECT_DISPATCH_SYMBOL: &str = "__tribute_root_reject_dispatch";
const ROOT_COMPLETION_CELL_NAME: &str = "__tribute_root_completion_cell";
const ROOT_COMPLETION_CELL_VALUE_FIELD: &str = "value";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TargetAbiError {
    message: String,
}

impl TargetAbiError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for TargetAbiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl Error for TargetAbiError {}

/// Build the root completion cell using the canonical `adt.struct` field
/// metadata consumed by native and Wasm layout lowering.
pub(crate) fn root_completion_cell_type(ctx: &mut IrContext, value_ty: TypeRef) -> TypeRef {
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

/// Rewrite only logical CPS callable result slots from `core.never` to the
/// target's physical empty-result marker (`core.nil`).
///
/// Selection is occurrence-specific: function signatures use their operation
/// convention, physical closure types use their explicit outer type
/// convention, and `func.constant` uses its exact target symbol. An untagged
/// nested `core.func<core.never, ...>` is rejected instead of inferred.
pub fn lower_cps_signatures_to_physical(
    ctx: &mut IrContext,
    module: Module,
) -> Result<(), TargetAbiError> {
    let ops = collect_ops(ctx, module.op());
    let never_ty = core::never(ctx).as_type_ref();
    let nil_ty = core::nil(ctx).as_type_ref();
    let anyref_ty = tribute_rt::anyref(ctx).as_type_ref();

    let mut functions = HashMap::new();
    let mut function_conventions = HashMap::new();
    for &op in &ops {
        let Ok(function) = func::Func::from_op(ctx, op) else {
            continue;
        };
        let Some(convention) = get_calling_convention(ctx, op) else {
            continue;
        };
        let signature = function.r#type(ctx);
        let Some(callable) = core::Func::from_type_ref(ctx, signature) else {
            return Err(TargetAbiError::new(format!(
                "target signature lowering: `{}` does not have a core.func signature",
                function.sym_name(ctx)
            )));
        };
        let environment_index = exact_environment_index(ctx, op, callable.params(ctx), anyref_ty)?;
        let identity = FunctionIdentity {
            signature,
            convention,
            environment_index,
        };
        function_conventions.insert(op, convention);
        let scope = symbol_scope(ctx, op)?;
        if functions
            .insert((scope, function.sym_name(ctx)), identity)
            .is_some()
        {
            return Err(TargetAbiError::new(format!(
                "target signature lowering: duplicate function symbol `{}` in one module scope",
                function.sym_name(ctx)
            )));
        }
    }

    // `tribute_control_to_cps` deliberately uses an unused `core.never`
    // result to mark an SCF region whose branches end in proper-tail
    // transfers. It is not a value at the target ABI. Validate that exact
    // structural role before planning its physical empty result, so malformed
    // source Never and unproven control flow cannot be erased by shape.
    let structured_never_results =
        validate_structured_never_results(ctx, &ops, never_ty, &function_conventions)?;

    let blocks = collect_blocks(ctx, &ops);
    let aliases = ctx.type_aliases().to_vec();
    let mut converter = PhysicalTypeConverter::new(ctx, never_ty, nil_ty);
    let mut plan = PhysicalizationPlan::default();

    for (name, ty) in aliases {
        let converted = converter.convert_nested(ty)?;
        if converted != ty {
            plan.type_aliases.push((name, converted));
        }
    }

    for &op in &ops {
        let function = func::Func::from_op(converter.ctx, op).ok();
        if let Some(function) = function {
            let signature = function.r#type(converter.ctx);
            let converted = match get_calling_convention(converter.ctx, op) {
                Some(convention) => converter.convert_callable(signature, convention)?,
                None => converter.convert_unproven_callable(signature)?,
            };
            if converted != signature {
                plan.op_attributes
                    .push((op, Symbol::new("type"), Attribute::Type(converted)));
            }
        }

        let result_types = converter.ctx.op_result_types(op).to_vec();
        for (index, ty) in result_types.into_iter().enumerate() {
            let converted = if structured_never_results.contains(&(op, index as u32)) {
                nil_ty
            } else if let Ok(constant) = func::Constant::from_op(converter.ctx, op) {
                let target = constant.func_ref(converter.ctx);
                let scope = symbol_scope(converter.ctx, op)?;
                let identity = functions.get(&(scope, target)).ok_or_else(|| {
                    TargetAbiError::new(format!(
                        "target signature lowering: func.constant references unknown or \
                         convention-less target `{target}` in its module scope"
                    ))
                })?;
                validate_constant_signature(converter.ctx, target, ty, *identity, never_ty)?;
                converter.convert_callable(ty, identity.convention)?
            } else {
                converter.convert_nested(ty)?
            };
            if converted != ty {
                plan.op_results.push((op, index as u32, converted));
            }
        }

        let attributes: Vec<_> = converter
            .ctx
            .op(op)
            .attributes
            .iter()
            .map(|(name, value)| (*name, value.clone()))
            .collect();
        for (name, value) in attributes {
            if function.is_some() && name == Symbol::new("type") {
                continue;
            }
            let converted = if name == Symbol::new(INDIRECT_CALL_SIGNATURE_ATTR) {
                let Attribute::Type(signature) = &value else {
                    return Err(TargetAbiError::new(
                        "target signature lowering: indirect callable signature must be a type attribute",
                    ));
                };
                let convention = get_calling_convention(converter.ctx, op).ok_or_else(|| {
                    TargetAbiError::new(
                        "target signature lowering: indirect callable signature has no valid calling convention",
                    )
                })?;
                Attribute::Type(converter.convert_callable(*signature, convention)?)
            } else {
                converter.convert_attribute(value.clone())?
            };
            if converted != value {
                plan.op_attributes.push((op, name, converted));
            }
        }
    }

    for block in blocks {
        let args = converter.ctx.block(block).args.clone();
        for (index, arg) in args.into_iter().enumerate() {
            let converted = converter.convert_nested(arg.ty)?;
            if converted != arg.ty {
                plan.block_arg_types.push((block, index as u32, converted));
            }
            for (name, value) in arg.attrs {
                let converted = converter.convert_attribute(value.clone())?;
                if converted != value {
                    plan.block_arg_attributes
                        .push((block, index, name, converted));
                }
            }
        }
    }

    drop(converter);
    for (name, ty) in plan.type_aliases {
        ctx.register_type_alias(name, ty);
    }
    for (op, index, ty) in plan.op_results {
        ctx.set_op_result_type(op, index, ty);
    }
    for (op, name, value) in plan.op_attributes {
        ctx.op_mut(op).attributes.insert(name, value);
    }
    for (block, index, ty) in plan.block_arg_types {
        ctx.set_block_arg_type(block, index, ty);
    }
    for (block, index, name, value) in plan.block_arg_attributes {
        ctx.block_mut(block).args[index].attrs.insert(name, value);
    }

    debug_assert!(collect_ops(ctx, module.op()).into_iter().all(|op| {
        let Ok(function) = func::Func::from_op(ctx, op) else {
            return true;
        };
        get_calling_convention(ctx, op) != Some(CallingConvention::Cps)
            || core::Func::from_type_ref(ctx, function.r#type(ctx))
                .is_some_and(|signature| signature.r#return(ctx) == nil_ty)
    }));
    Ok(())
}

/// Project native-only zero-sized `core.nil` parameters out of callable ABIs.
///
/// Wasm deliberately represents `core.nil` as a nullable reference, so this
/// pass is intentionally invoked only by the native pipeline.  It plans every
/// affected declaration, definition, direct reference, and transfer before
/// changing IR; a malformed exact signature therefore leaves the module
/// untouched.
pub fn lower_native_nil_abi(ctx: &mut IrContext, module: Module) -> Result<(), TargetAbiError> {
    let nil_ty = core::nil(ctx).as_type_ref();
    let anyref_ty = tribute_rt::anyref(ctx).as_type_ref();
    let ops = collect_ops(ctx, module.op());
    let mut functions = HashMap::new();
    let mut plan = NativeNilAbiPlan::default();

    for &op in &ops {
        let Ok(function) = func::Func::from_op(ctx, op) else {
            continue;
        };
        let signature = function.r#type(ctx);
        let callable = core::Func::from_type_ref(ctx, signature).ok_or_else(|| {
            TargetAbiError::new(format!(
                "native Nil ABI: function `{}` does not have a core.func signature",
                function.sym_name(ctx)
            ))
        })?;
        let params = callable.params(ctx).to_vec();
        let nil_params: Vec<_> = params
            .iter()
            .enumerate()
            .filter_map(|(index, &ty)| (ty == nil_ty).then_some(index))
            .collect();
        let identity = NativeNilFunctionIdentity {
            signature,
            environment_index: exact_environment_index(ctx, op, &params, anyref_ty)?,
            nil_params: nil_params.clone(),
        };
        let scope = symbol_scope(ctx, op)?;
        if functions
            .insert((scope, function.sym_name(ctx)), identity)
            .is_some()
        {
            return Err(TargetAbiError::new(format!(
                "native Nil ABI: duplicate function symbol `{}` in one module scope",
                function.sym_name(ctx)
            )));
        }

        if nil_params.is_empty() {
            continue;
        }
        if get_calling_convention(ctx, op).is_none() {
            return Err(TargetAbiError::new(format!(
                "native Nil ABI: function `{}` with core.nil parameter has no valid calling convention",
                function.sym_name(ctx)
            )));
        }
        if let Some(&region) = ctx.op(op).regions.first() {
            let Some(&entry) = ctx.region(region).blocks.first() else {
                return Err(TargetAbiError::new(format!(
                    "native Nil ABI: function `{}` definition has no entry block",
                    function.sym_name(ctx)
                )));
            };
            let args = ctx.block_args(entry);
            if args.len() != params.len()
                || args
                    .iter()
                    .zip(&params)
                    .any(|(&arg, &ty)| ctx.value_ty(arg) != ty)
            {
                return Err(TargetAbiError::new(format!(
                    "native Nil ABI: function `{}` entry block does not match its exact signature",
                    function.sym_name(ctx)
                )));
            }
            plan.entry_nil_args.push((entry, nil_params));
        }
    }

    for &op in &ops {
        if let Ok(call) = func::Call::from_op(ctx, op) {
            let target = call.callee(ctx);
            let scope = symbol_scope(ctx, op)?;
            let identity = functions.get(&(scope, target)).ok_or_else(|| {
                TargetAbiError::new(format!(
                    "native Nil ABI: func.call references unknown target `{target}`"
                ))
            })?;
            if !identity.nil_params.is_empty() {
                validate_direct_nil_transfer(ctx, op, identity, 0, target)?;
                plan.transfer_nil_operands
                    .push((op, identity.nil_params.clone()));
            }
        } else if let Ok(tail) = func::TailCall::from_op(ctx, op) {
            let target = tail.callee(ctx);
            let scope = symbol_scope(ctx, op)?;
            let identity = functions.get(&(scope, target)).ok_or_else(|| {
                TargetAbiError::new(format!(
                    "native Nil ABI: func.tail_call references unknown target `{target}`"
                ))
            })?;
            if !identity.nil_params.is_empty() {
                validate_direct_nil_transfer(ctx, op, identity, 0, target)?;
                plan.transfer_nil_operands
                    .push((op, identity.nil_params.clone()));
            }
        } else if func::CallIndirect::from_op(ctx, op).is_ok()
            || func::TailCallIndirect::from_op(ctx, op).is_ok()
        {
            let operands = ctx.op_operands(op);
            let Some((_callee, args)) = operands.split_first() else {
                return Err(TargetAbiError::new(
                    "native Nil ABI: indirect transfer has no callee operand",
                ));
            };
            if args.iter().any(|&arg| ctx.value_ty(arg) == nil_ty) {
                let nil_args = validate_indirect_nil_transfer(ctx, op, nil_ty)?;
                if !nil_args.is_empty() {
                    plan.transfer_nil_operands.push((op, nil_args));
                }
            }
        } else if let Ok(constant) = func::Constant::from_op(ctx, op) {
            let target = constant.func_ref(ctx);
            let scope = symbol_scope(ctx, op)?;
            let identity = functions.get(&(scope, target)).ok_or_else(|| {
                TargetAbiError::new(format!(
                    "native Nil ABI: func.constant references unknown target `{target}`"
                ))
            })?;
            if !identity.nil_params.is_empty() {
                let actual = ctx.value_ty(constant.result(ctx));
                validate_native_constant_signature(ctx, target, actual, identity.clone())?;
            }
        } else if func::Return::from_op(ctx, op).is_ok() {
            let operands = ctx.op_operands(op);
            if operands.is_empty() {
                continue;
            }
            if !operands
                .iter()
                .any(|&operand| ctx.value_ty(operand) == nil_ty)
            {
                continue;
            }
            if operands.len() != 1 || ctx.value_ty(operands[0]) != nil_ty {
                return Err(TargetAbiError::new(
                    "native Nil ABI: core.nil return operand must be the sole return value",
                ));
            }
            let function = enclosing_function(ctx, op).ok_or_else(|| {
                TargetAbiError::new(
                    "native Nil ABI: func.return with core.nil operand is outside a function",
                )
            })?;
            let function = func::Func::from_op(ctx, function).map_err(|_| {
                TargetAbiError::new(
                    "native Nil ABI: func.return with core.nil operand has a malformed owner",
                )
            })?;
            let signature = core::Func::from_type_ref(ctx, function.r#type(ctx)).ok_or_else(|| {
                TargetAbiError::new(
                    "native Nil ABI: func.return with core.nil operand has a malformed signature",
                )
            })?;
            if signature.r#return(ctx) != nil_ty {
                return Err(TargetAbiError::new(
                    "native Nil ABI: core.nil return operand does not match its function result",
                ));
            }
            plan.nil_return_operands.push(op);
        }
    }

    // Final native IR must not retain Nil block arguments.  The only legal
    // occurrences at this boundary are entry parameters whose exact callable
    // signature is being projected above.
    let entry_blocks: HashSet<_> = plan
        .entry_nil_args
        .iter()
        .map(|(block, _)| *block)
        .collect();
    for block in collect_blocks(ctx, &ops) {
        if entry_blocks.contains(&block) {
            continue;
        }
        if ctx
            .block_args(block)
            .iter()
            .any(|&arg| ctx.value_ty(arg) == nil_ty)
        {
            return Err(TargetAbiError::new(
                "native Nil ABI: residual core.nil block argument is not an exact callable parameter",
            ));
        }
    }

    let aliases = ctx.type_aliases().to_vec();
    let mut converter = NativeNilTypeConverter::new(ctx, nil_ty);
    for (name, ty) in aliases {
        let converted = converter.convert_type(ty)?;
        if converted != ty {
            plan.type_aliases.push((name, converted));
        }
    }
    for &op in &ops {
        let results = converter.ctx.op_result_types(op).to_vec();
        for (index, ty) in results.into_iter().enumerate() {
            let converted = converter.convert_type(ty)?;
            if converted != ty {
                plan.op_results.push((op, index as u32, converted));
            }
        }
        let attributes: Vec<_> = converter
            .ctx
            .op(op)
            .attributes
            .iter()
            .map(|(name, value)| (*name, value.clone()))
            .collect();
        for (name, value) in attributes {
            let converted = converter.convert_attribute(value.clone())?;
            if converted != value {
                plan.op_attributes.push((op, name, converted));
            }
        }
    }
    for block in collect_blocks(converter.ctx, &ops) {
        let args = converter.ctx.block(block).args.clone();
        for (index, arg) in args.into_iter().enumerate() {
            let converted = converter.convert_type(arg.ty)?;
            if converted != arg.ty {
                plan.block_arg_types.push((block, index as u32, converted));
            }
            for (name, value) in arg.attrs {
                let converted = converter.convert_attribute(value.clone())?;
                if converted != value {
                    plan.block_arg_attributes
                        .push((block, index, name, converted));
                }
            }
        }
    }
    drop(converter);

    for (name, ty) in plan.type_aliases {
        ctx.register_type_alias(name, ty);
    }
    for (op, index, ty) in plan.op_results {
        ctx.set_op_result_type(op, index, ty);
    }
    for (op, name, value) in plan.op_attributes {
        ctx.op_mut(op).attributes.insert(name, value);
    }
    for (block, index, ty) in plan.block_arg_types {
        ctx.set_block_arg_type(block, index, ty);
    }
    for (block, index, name, value) in plan.block_arg_attributes {
        ctx.block_mut(block).args[index].attrs.insert(name, value);
    }
    for (op, mut indices) in plan.transfer_nil_operands {
        indices.sort_unstable();
        indices.dedup();
        for index in indices.into_iter().rev() {
            ctx.remove_op_operand(op, index as u32);
        }
    }
    for op in plan.nil_return_operands {
        ctx.remove_op_operand(op, 0);
    }
    for (block, mut indices) in plan.entry_nil_args {
        indices.sort_unstable();
        indices.dedup();
        for index in indices.into_iter().rev() {
            let value = ctx.block_args(block)[index];
            let location = ctx.block(block).location;
            let unit = arith::r#const(ctx, location, nil_ty, Attribute::Unit);
            if let Some(&first) = ctx.block(block).ops.first() {
                ctx.insert_op_before(block, first, unit.op_ref());
            } else {
                ctx.push_op(block, unit.op_ref());
            }
            ctx.replace_all_uses(value, unit.result(ctx));
            debug_assert!(ctx.uses(value).is_empty());
            ctx.remove_block_arg(block, index as u32);
        }
    }
    Ok(())
}

#[derive(Clone)]
struct NativeNilFunctionIdentity {
    signature: TypeRef,
    environment_index: Option<usize>,
    nil_params: Vec<usize>,
}

#[derive(Default)]
struct NativeNilAbiPlan {
    type_aliases: Vec<(Symbol, TypeRef)>,
    op_results: Vec<(OpRef, u32, TypeRef)>,
    op_attributes: Vec<(OpRef, Symbol, Attribute)>,
    block_arg_types: Vec<(BlockRef, u32, TypeRef)>,
    block_arg_attributes: Vec<(BlockRef, usize, Symbol, Attribute)>,
    entry_nil_args: Vec<(BlockRef, Vec<usize>)>,
    transfer_nil_operands: Vec<(OpRef, Vec<usize>)>,
    nil_return_operands: Vec<OpRef>,
}

fn validate_direct_nil_transfer(
    ctx: &IrContext,
    op: OpRef,
    identity: &NativeNilFunctionIdentity,
    operand_offset: usize,
    target: Symbol,
) -> Result<(), TargetAbiError> {
    let callable = core::Func::from_type_ref(ctx, identity.signature)
        .expect("native Nil identity was prevalidated as core.func");
    let operands = ctx.op_operands(op);
    let expected = callable.params(ctx);
    if operands.len() != expected.len() + operand_offset
        || operands[operand_offset..]
            .iter()
            .zip(expected)
            .any(|(&operand, &ty)| ctx.value_ty(operand) != ty)
    {
        return Err(TargetAbiError::new(format!(
            "native Nil ABI: transfer to `{target}` does not match its exact pre-projection signature"
        )));
    }
    Ok(())
}

/// Validate an indirect transfer against the exact callable type carried by
/// its callee value, then return only the argument operand indices that the
/// native zero-size ABI may project. The callee remains at operand zero.
fn validate_indirect_nil_transfer(
    ctx: &IrContext,
    op: OpRef,
    nil_ty: TypeRef,
) -> Result<Vec<usize>, TargetAbiError> {
    if get_calling_convention(ctx, op).is_none() {
        return Err(TargetAbiError::new(
            "native Nil ABI: indirect transfer with core.nil argument has no valid calling convention",
        ));
    }
    let operands = ctx.op_operands(op);
    let Some((&callee, args)) = operands.split_first() else {
        return Err(TargetAbiError::new(
            "native Nil ABI: indirect transfer has no callee operand",
        ));
    };
    let callable_ty = get_indirect_call_signature(ctx, op).or_else(|| {
        let callee_ty = ctx.value_ty(callee);
        core::Func::from_type_ref(ctx, callee_ty)
            .map(|_| callee_ty)
            .or_else(|| closure::Closure::from_type_ref(ctx, callee_ty).map(|closure| closure.func_type(ctx)))
    }).ok_or_else(|| {
        TargetAbiError::new(
            "native Nil ABI: indirect transfer with core.nil argument has no exact callable signature provenance",
        )
    })?;
    let callable = core::Func::from_type_ref(ctx, callable_ty).ok_or_else(|| {
        TargetAbiError::new(
            "native Nil ABI: indirect transfer closure does not contain an exact core.func type",
        )
    })?;
    let expected = callable.params(ctx);
    if args.len() != expected.len()
        || args
            .iter()
            .zip(expected)
            .any(|(&arg, &ty)| !matches_indirect_callable_parameter(ctx, ty, ctx.value_ty(arg)))
    {
        return Err(TargetAbiError::new(
            "native Nil ABI: indirect transfer does not match its exact pre-projection callable signature",
        ));
    }
    Ok(expected
        .iter()
        .enumerate()
        .filter_map(|(index, &ty)| (ty == nil_ty).then_some(index + 1))
        .collect())
}

/// Closure lowering changes abstract closure values into the exact canonical
/// `_closure { table_idx, env }` runtime pair. This is a representation
/// projection, not a signature inference: the preserved callable type remains
/// authoritative and every other type must match identically.
fn matches_indirect_callable_parameter(
    ctx: &IrContext,
    expected: TypeRef,
    actual: TypeRef,
) -> bool {
    expected == actual
        || (closure::Closure::matches(ctx, expected)
            && crate::closure_lower::is_closure_struct_type_ref(ctx, actual))
}

fn validate_native_constant_signature(
    ctx: &mut IrContext,
    target: Symbol,
    actual: TypeRef,
    identity: NativeNilFunctionIdentity,
) -> Result<(), TargetAbiError> {
    let definition = core::Func::from_type_ref(ctx, identity.signature)
        .expect("native Nil identity was prevalidated as core.func");
    let mut expected_params = definition.params(ctx).to_vec();
    if let Some(index) = identity.environment_index {
        expected_params.remove(index);
    }
    let expected = core::func(
        ctx,
        definition.r#return(ctx),
        expected_params.iter().copied(),
    )
    .as_type_ref();
    if actual != expected {
        return Err(TargetAbiError::new(format!(
            "native Nil ABI: func.constant @{target} does not match the exact environment-less signature of its target"
        )));
    }
    Ok(())
}

struct NativeNilTypeConverter<'a> {
    ctx: &'a mut IrContext,
    nil_ty: TypeRef,
    cache: HashMap<TypeRef, TypeRef>,
}

impl<'a> NativeNilTypeConverter<'a> {
    fn new(ctx: &'a mut IrContext, nil_ty: TypeRef) -> Self {
        Self {
            ctx,
            nil_ty,
            cache: HashMap::new(),
        }
    }

    fn convert_type(&mut self, ty: TypeRef) -> Result<TypeRef, TargetAbiError> {
        if let Some(&converted) = self.cache.get(&ty) {
            return Ok(converted);
        }
        let data = self.ctx.types.get(ty).clone();
        let mut converted = data.clone();
        if data.dialect == Symbol::new("core") && data.name == Symbol::new("func") {
            if data.params.is_empty() {
                return Err(TargetAbiError::new(
                    "native Nil ABI: malformed core.func type",
                ));
            }
            converted.params = smallvec![self.convert_type(data.params[0])?];
            for &param in &data.params[1..] {
                if param != self.nil_ty {
                    converted.params.push(self.convert_type(param)?);
                }
            }
        } else {
            for param in &mut converted.params {
                *param = self.convert_type(*param)?;
            }
        }
        self.convert_type_attributes(&mut converted)?;
        let result = if converted == data {
            ty
        } else {
            self.ctx.types.intern(converted)
        };
        self.cache.insert(ty, result);
        Ok(result)
    }

    fn convert_type_attributes(&mut self, data: &mut TypeData) -> Result<(), TargetAbiError> {
        let attrs: Vec<_> = data
            .attrs
            .iter()
            .map(|(name, value)| (*name, value.clone()))
            .collect();
        for (name, value) in attrs {
            data.attrs.insert(name, self.convert_attribute(value)?);
        }
        Ok(())
    }

    fn convert_attribute(&mut self, attribute: Attribute) -> Result<Attribute, TargetAbiError> {
        match attribute {
            Attribute::Type(ty) => Ok(Attribute::Type(self.convert_type(ty)?)),
            Attribute::List(values) => Ok(Attribute::List(
                values
                    .into_iter()
                    .map(|value| self.convert_attribute(value))
                    .collect::<Result<_, _>>()?,
            )),
            other => Ok(other),
        }
    }
}

/// Construct the target root delimiter after physical CPS signature lowering.
///
/// Direct roots remain ordinary. A promoted root worker is renamed and called
/// once by a Direct/EvidenceDirect wrapper. Its terminal continuation stores
/// the source result in a typed completion cell before returning through the
/// target's empty result ABI.
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
    let worker_convention = checked_calling_convention(ctx, worker_op)?;
    let export_convention = checked_root_export_convention(ctx, worker_op)?;
    let source_result = checked_root_source_result(ctx, worker_op)?;
    if export_convention.is_some() != source_result.is_some() {
        return Err(TargetAbiError::new(
            "target root bridge: preserved root export convention and source result must either \
             both be present or both be absent",
        ));
    }
    let Some(worker_convention) = worker_convention else {
        if export_convention.is_none() && source_result.is_none() {
            return Ok(());
        }
        return Err(TargetAbiError::new(
            "target root bridge: root main has no calling convention",
        ));
    };
    if worker_convention != CallingConvention::Cps {
        let Some(export_convention) = export_convention else {
            return Ok(());
        };
        let Some(_source_result) = source_result else {
            return Err(TargetAbiError::new(
                "target root bridge: missing preserved root source result type",
            ));
        };
        if worker_convention != export_convention {
            return Err(TargetAbiError::new(
                "target root bridge: unpromoted root convention does not match its preserved \
                 export convention",
            ));
        }
        remove_root_contract(ctx, worker_op);
        return Ok(());
    }

    let export_convention = export_convention.ok_or_else(|| {
        TargetAbiError::new(
            "target root bridge: missing preserved root export convention for promoted Cps main",
        )
    })?;
    if export_convention == CallingConvention::Cps {
        return Err(TargetAbiError::new(
            "target root bridge: the preserved root export convention must be Direct or \
             EvidenceDirect",
        ));
    }
    let source_result = source_result.ok_or_else(|| {
        TargetAbiError::new(
            "target root bridge: missing preserved root source result type for promoted Cps main",
        )
    })?;

    let nil_ty = core::nil(ctx).as_type_ref();
    if source_result != nil_ty {
        return Err(TargetAbiError::new(
            "target root bridge: current source main contract must return core.nil",
        ));
    }
    let evidence_ty = ability::evidence_adt_type_ref(ctx);
    let worker = func::Func::from_op(ctx, worker_op)
        .map_err(|_| TargetAbiError::new("target root bridge: root main is not func.func"))?;
    let worker_signature = core::Func::from_type_ref(ctx, worker.r#type(ctx))
        .ok_or_else(|| TargetAbiError::new("target root bridge: main is not core.func"))?;
    if worker_signature.r#return(ctx) != nil_ty {
        return Err(TargetAbiError::new(
            "target root bridge: Cps root must have a physically empty result",
        ));
    }
    let worker_params = worker_signature.params(ctx);
    if worker_params.len() != 3 || worker_params[0] != evidence_ty {
        return Err(TargetAbiError::new(
            "target root bridge: source main must have no parameters beyond physical evidence and \
             done_k and dispatch",
        ));
    }
    let done_k_ty = worker_params[1];
    validate_root_done_k_type(ctx, done_k_ty, source_result, nil_ty)?;
    let dispatch_ty = worker_params[2];
    let dispatch_function = validate_root_dispatch_type(ctx, dispatch_ty, nil_ty)?;
    if ctx.op(worker_op).regions.is_empty() {
        return Err(TargetAbiError::new(
            "target root bridge: root main must be a definition",
        ));
    }

    let cps_main = Symbol::new(CPS_MAIN_SYMBOL);
    let root_done_k = Symbol::new(ROOT_DONE_K_SYMBOL);
    let root_reject_dispatch = Symbol::new(ROOT_REJECT_DISPATCH_SYMBOL);
    for &op in &top_level_ops {
        let Ok(function) = func::Func::from_op(ctx, op) else {
            continue;
        };
        let name = function.sym_name(ctx);
        if name == cps_main || name == root_done_k || name == root_reject_dispatch {
            return Err(TargetAbiError::new(format!(
                "target root bridge: reserved symbol collision at `{name}`"
            )));
        }
    }

    let location = ctx.op(worker_op).location;
    ctx.op_mut(worker_op)
        .attributes
        .insert(Symbol::new("sym_name"), Attribute::Symbol(cps_main));
    ctx.op_mut(worker_op)
        .attributes
        .insert(Symbol::new(ROOT_CPS_WORKER_ATTR), Attribute::Bool(true));
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
    set_calling_convention(ctx, done_function.op_ref(), CallingConvention::Cps);
    ctx.op_mut(done_function.op_ref())
        .attributes
        .insert(Symbol::new(ROOT_DONE_K_ATTR), Attribute::Bool(true));

    let dispatch_params = core::Func::from_type_ref(ctx, dispatch_function)
        .expect("validated root dispatch function remains core.func")
        .params(ctx)
        .to_vec();
    let mut reject_params = Vec::with_capacity(dispatch_params.len() + 1);
    reject_params.push(anyref_ty);
    reject_params.extend(dispatch_params.iter().copied());
    let reject_function_ty = core::func(ctx, nil_ty, reject_params.clone()).as_type_ref();
    let reject_entry = ctx.create_block(BlockData {
        location,
        args: reject_params
            .iter()
            .copied()
            .enumerate()
            .map(|(index, ty)| BlockArgData {
                ty,
                attrs: if index == 0 {
                    bind_name("__env")
                } else {
                    Default::default()
                },
            })
            .collect(),
        ops: smallvec![],
        parent_region: None,
    });
    let reject = func::unreachable(ctx, location);
    ctx.push_op(reject_entry, reject.op_ref());
    let reject_region = ctx.create_region(RegionData {
        location,
        blocks: smallvec![reject_entry],
        parent_op: None,
    });
    let reject_function = func::func(
        ctx,
        location,
        root_reject_dispatch,
        reject_function_ty,
        reject_region,
    );
    set_calling_convention(ctx, reject_function.op_ref(), CallingConvention::Cps);

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
    let reject_constant = func::constant(ctx, location, reject_function_ty, root_reject_dispatch);
    ctx.push_op(wrapper_entry, reject_constant.op_ref());
    let reject_closure = adt::struct_new(
        ctx,
        location,
        [reject_constant.result(ctx), erased_cell.result(ctx)],
        closure_struct_ty,
        closure_struct_ty,
    );
    ctx.push_op(wrapper_entry, reject_closure.op_ref());
    let typed_dispatch =
        core::unrealized_conversion_cast(ctx, location, reject_closure.result(ctx), dispatch_ty);
    ctx.push_op(wrapper_entry, typed_dispatch.op_ref());

    let evidence = if export_convention == CallingConvention::EvidenceDirect {
        ctx.block_args(wrapper_entry)[0]
    } else {
        let i32_ty = ctx.types.intern(TypeData {
            dialect: Symbol::new("core"),
            name: Symbol::new("i32"),
            params: smallvec![],
            attrs: AttributeMap::new(),
        });
        let zero = arith::r#const(ctx, location, i32_ty, Attribute::Int(0));
        ctx.push_op(wrapper_entry, zero.op_ref());
        let empty = adt::array_new(ctx, location, [zero.result(ctx)], evidence_ty, evidence_ty);
        ctx.push_op(wrapper_entry, empty.op_ref());
        empty.result(ctx)
    };
    let worker_call = func::call(
        ctx,
        location,
        [evidence, typed_done.result(ctx), typed_dispatch.result(ctx)],
        nil_ty,
        cps_main,
    );
    set_calling_convention(ctx, worker_call.op_ref(), CallingConvention::Cps);
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
    set_calling_convention(ctx, wrapper.op_ref(), export_convention);
    ctx.op_mut(wrapper.op_ref())
        .attributes
        .insert(Symbol::new(ROOT_WRAPPER_ATTR), Attribute::Bool(true));

    ctx.push_op(module_block, done_function.op_ref());
    ctx.push_op(module_block, reject_function.op_ref());
    ctx.push_op(module_block, wrapper.op_ref());
    Ok(())
}

fn exact_environment_index(
    ctx: &IrContext,
    function: OpRef,
    signature_params: &[TypeRef],
    anyref_ty: TypeRef,
) -> Result<Option<usize>, TargetAbiError> {
    let Some(&region) = ctx.op(function).regions.first() else {
        return Ok(None);
    };
    let Some(&entry) = ctx.region(region).blocks.first() else {
        return Ok(None);
    };
    if ctx.block(entry).args.len() != signature_params.len() {
        return Err(TargetAbiError::new(
            "target CPS signature lowering: function signature and entry block arity differ",
        ));
    }
    let indices: Vec<_> = ctx
        .block(entry)
        .args
        .iter()
        .enumerate()
        .filter_map(|(index, arg)| {
            (arg.attrs.get_symbol("bind_name") == Some(Symbol::new("__env"))).then_some(index)
        })
        .collect();
    match indices.as_slice() {
        [] => Ok(None),
        [index] => {
            if signature_params[*index] != anyref_ty {
                return Err(TargetAbiError::new(
                    "target CPS signature lowering: `__env` must have exact tribute_rt.anyref type",
                ));
            }
            Ok(Some(*index))
        }
        _ => Err(TargetAbiError::new(
            "target CPS signature lowering: function has multiple `__env` parameters",
        )),
    }
}

/// Find `scf.if : core.never` markers that have the one CPS meaning which can
/// be physicalized: every branch transfers control in tail position and the
/// marker result is unused. Any other operation result of `core.never` reaches
/// this boundary without enough provenance for an ABI rewrite, so reject it.
fn validate_structured_never_results(
    ctx: &IrContext,
    ops: &[OpRef],
    never_ty: TypeRef,
    function_conventions: &HashMap<OpRef, CallingConvention>,
) -> Result<HashSet<(OpRef, u32)>, TargetAbiError> {
    let mut results = HashSet::new();
    for &op in ops {
        for (index, &ty) in ctx.op_result_types(op).iter().enumerate() {
            if ty != never_ty {
                continue;
            }
            if !scf::If::matches(ctx, op) || index != 0 || ctx.op_results(op).len() != 1 {
                return Err(TargetAbiError::new(
                    "target signature lowering: residual core.never operation result is not a \
                     proven structured CPS tail marker",
                ));
            }
            let function = enclosing_function(ctx, op).ok_or_else(|| {
                TargetAbiError::new(
                    "target signature lowering: structured core.never marker has no owning \
                     callable",
                )
            })?;
            if function_conventions.get(&function) != Some(&CallingConvention::Cps) {
                return Err(TargetAbiError::new(
                    "target signature lowering: structured core.never marker is not owned by a \
                     proven Cps callable",
                ));
            }
            let result = ctx.op_result(op, 0);
            if ctx.has_uses(result) {
                return Err(TargetAbiError::new(
                    "target signature lowering: structured core.never marker result must be \
                     unused",
                ));
            }
            if !structured_if_ends_in_proper_tail(ctx, op, never_ty) {
                return Err(TargetAbiError::new(format!(
                    "target signature lowering: structured core.never marker must end every \
                         branch in a proper-tail transfer or typed unreachable (branch endings: {})",
                    structured_if_branch_endings(ctx, op)
                )));
            }
            results.insert((op, index as u32));
        }
    }
    Ok(results)
}

fn structured_if_branch_endings(ctx: &IrContext, op: OpRef) -> String {
    let Ok(if_op) = scf::If::from_op(ctx, op) else {
        return "<not scf.if>".to_owned();
    };
    [if_op.then_region(ctx), if_op.else_region(ctx)]
        .into_iter()
        .map(|region| {
            let Some(block) = ctx.region(region).blocks.first().copied() else {
                return "<empty region>".to_owned();
            };
            let ops = &ctx.block(block).ops;
            let Some(op) = ops.last().copied() else {
                return "<empty block>".to_owned();
            };
            let data = ctx.op(op);
            let all_ops = ops
                .iter()
                .map(|&op| {
                    let data = ctx.op(op);
                    format!("{}.{}", data.dialect, data.name)
                })
                .collect::<Vec<_>>()
                .join(" -> ");
            format!("{}.{} [{}]", data.dialect, data.name, all_ops)
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn enclosing_function(ctx: &IrContext, op: OpRef) -> Option<OpRef> {
    let mut current = Some(op);
    while let Some(candidate) = current {
        if func::Func::from_op(ctx, candidate).is_ok() {
            return Some(candidate);
        }
        if candidate != op && core::Module::from_op(ctx, candidate).is_ok() {
            return None;
        }
        current = ctx.op(candidate).parent_block.and_then(|block| {
            let region = ctx.block(block).parent_region?;
            ctx.region(region).parent_op
        });
    }
    None
}

fn structured_if_ends_in_proper_tail(ctx: &IrContext, op: OpRef, never_ty: TypeRef) -> bool {
    let Ok(if_op) = scf::If::from_op(ctx, op) else {
        return false;
    };
    [if_op.then_region(ctx), if_op.else_region(ctx)]
        .into_iter()
        .all(|region| region_ends_in_proper_tail(ctx, region, never_ty))
}

fn region_ends_in_proper_tail(
    ctx: &IrContext,
    region: trunk_ir::refs::RegionRef,
    never_ty: TypeRef,
) -> bool {
    let [block] = ctx.region(region).blocks.as_slice() else {
        return false;
    };
    let Some(&terminator) = ctx.block(*block).ops.last() else {
        return false;
    };
    let data = ctx.op(terminator);
    if data.dialect == Symbol::new("func")
        && (data.name == Symbol::new("tail_call") || data.name == Symbol::new("tail_call_indirect"))
    {
        return get_calling_convention(ctx, terminator) == Some(CallingConvention::Cps);
    }
    if data.dialect == Symbol::new("effect") && data.name == Symbol::new("dispatch_cps") {
        return ctx.op_result_types(terminator).is_empty();
    }
    // Handler dispatches may have a statically unreachable reject arm. It is
    // a genuine non-returning exit, unlike an empty/fallthrough block, and
    // carries no value through the physical control-flow edge.
    if data.dialect == Symbol::new("func") && data.name == Symbol::new("unreachable") {
        return true;
    }
    scf::If::matches(ctx, terminator)
        && ctx.op_result_types(terminator) == [never_ty]
        && !ctx.has_uses(ctx.op_result(terminator, 0))
        && structured_if_ends_in_proper_tail(ctx, terminator, never_ty)
        || (scf::Switch::matches(ctx, terminator)
            && structured_switch_ends_in_proper_tail(ctx, terminator, never_ty))
}

/// A resultless `scf.switch` may terminate a proven `core.never` branch only
/// when it is a complete dispatch whose every case/default region transfers
/// control. This keeps structured source control explicit without admitting a
/// fallthrough or an empty arm as a physical CPS exit.
fn structured_switch_ends_in_proper_tail(ctx: &IrContext, op: OpRef, never_ty: TypeRef) -> bool {
    let Ok(switch) = scf::Switch::from_op(ctx, op) else {
        return false;
    };
    let [body] = ctx.op(switch.op_ref()).regions.as_slice() else {
        return false;
    };
    let [block] = ctx.region(*body).blocks.as_slice() else {
        return false;
    };
    let arms = ctx.block(*block).ops.as_slice();
    !arms.is_empty()
        && arms.iter().all(|&arm| {
            let data = ctx.op(arm);
            let is_arm = scf::Case::matches(ctx, arm) || scf::Default::matches(ctx, arm);
            let [region] = data.regions.as_slice() else {
                return false;
            };
            is_arm && region_ends_in_proper_tail(ctx, *region, never_ty)
        })
        && arms.iter().any(|&arm| scf::Default::matches(ctx, arm))
}

fn collect_ops(ctx: &IrContext, root: OpRef) -> Vec<OpRef> {
    let mut ops = Vec::new();
    let _ = walk_op::<()>(ctx, root, &mut |op| {
        ops.push(op);
        ControlFlow::Continue(WalkAction::Advance)
    });
    ops
}

fn collect_blocks(ctx: &IrContext, ops: &[OpRef]) -> Vec<BlockRef> {
    let mut seen = HashSet::new();
    let mut blocks = Vec::new();
    for &op in ops {
        for &region in &ctx.op(op).regions {
            for &block in &ctx.region(region).blocks {
                if seen.insert(block) {
                    blocks.push(block);
                }
            }
        }
    }
    blocks
}

/// Return the nearest owning `core.module` for an operation's symbol lookup.
///
/// `func.constant` names are module-local. The physicalizer traverses nested
/// modules to rewrite their types too, so a bare symbol alone cannot identify
/// a callable without conflating a nested local declaration with a root one.
fn symbol_scope(ctx: &IrContext, op: OpRef) -> Result<OpRef, TargetAbiError> {
    let mut current = Some(op);
    while let Some(candidate) = current {
        if core::Module::from_op(ctx, candidate).is_ok() {
            return Ok(candidate);
        }
        current = ctx.op(candidate).parent_block.and_then(|block| {
            let region = ctx.block(block).parent_region?;
            ctx.region(region).parent_op
        });
    }
    Err(TargetAbiError::new(
        "target signature lowering: operation has no owning core.module symbol scope",
    ))
}

#[derive(Clone, Copy)]
struct FunctionIdentity {
    signature: TypeRef,
    convention: CallingConvention,
    environment_index: Option<usize>,
}

#[derive(Default)]
struct PhysicalizationPlan {
    type_aliases: Vec<(Symbol, TypeRef)>,
    op_results: Vec<(OpRef, u32, TypeRef)>,
    op_attributes: Vec<(OpRef, Symbol, Attribute)>,
    block_arg_types: Vec<(BlockRef, u32, TypeRef)>,
    block_arg_attributes: Vec<(BlockRef, usize, Symbol, Attribute)>,
}

fn validate_constant_signature(
    ctx: &mut IrContext,
    target: Symbol,
    actual: TypeRef,
    identity: FunctionIdentity,
    never_ty: TypeRef,
) -> Result<(), TargetAbiError> {
    let definition = core::Func::from_type_ref(ctx, identity.signature)
        .expect("function identity was prevalidated as core.func");
    let mut expected_params = definition.params(ctx).to_vec();
    if let Some(index) = identity.environment_index {
        expected_params.remove(index);
    }
    let expected_result = definition.r#return(ctx);
    if identity.convention == CallingConvention::Cps && expected_result != never_ty {
        return Err(TargetAbiError::new(format!(
            "target signature lowering: Cps target `{target}` must have logical core.never result"
        )));
    }
    let expected = core::func(ctx, expected_result, expected_params.iter().copied()).as_type_ref();
    if actual != expected {
        return Err(TargetAbiError::new(format!(
            "target signature lowering: func.constant @{} does not match the exact \
             environment-less signature of its target",
            target
        )));
    }
    Ok(())
}

struct PhysicalTypeConverter<'a> {
    ctx: &'a mut IrContext,
    never_ty: TypeRef,
    nil_ty: TypeRef,
    nested_cache: HashMap<TypeRef, TypeRef>,
    callable_cache: HashMap<(TypeRef, CallingConvention), TypeRef>,
}

impl<'a> PhysicalTypeConverter<'a> {
    fn new(ctx: &'a mut IrContext, never_ty: TypeRef, nil_ty: TypeRef) -> Self {
        Self {
            ctx,
            never_ty,
            nil_ty,
            nested_cache: HashMap::new(),
            callable_cache: HashMap::new(),
        }
    }

    fn convert_unproven_callable(&mut self, ty: TypeRef) -> Result<TypeRef, TargetAbiError> {
        let callable = core::Func::from_type_ref(self.ctx, ty).ok_or_else(|| {
            TargetAbiError::new("target signature lowering: function type is not core.func")
        })?;
        if callable.r#return(self.ctx) == self.never_ty {
            return Err(TargetAbiError::new(
                "target signature lowering: core.never function has no explicit convention",
            ));
        }
        self.convert_callable(ty, CallingConvention::Direct)
    }

    fn convert_callable(
        &mut self,
        ty: TypeRef,
        convention: CallingConvention,
    ) -> Result<TypeRef, TargetAbiError> {
        if let Some(&converted) = self.callable_cache.get(&(ty, convention)) {
            return Ok(converted);
        }
        let mut data = self.ctx.types.get(ty).clone();
        if data.dialect != Symbol::new("core")
            || data.name != Symbol::new("func")
            || data.params.is_empty()
        {
            return Err(TargetAbiError::new(
                "target signature lowering: proven callable is not a valid core.func",
            ));
        }
        if convention == CallingConvention::Cps && data.params[0] != self.never_ty {
            return Err(TargetAbiError::new(
                "target signature lowering: Cps callable must have logical core.never result",
            ));
        }

        let old_result = data.params[0];
        data.params[0] = if convention == CallingConvention::Cps {
            self.nil_ty
        } else {
            self.convert_nested(old_result)?
        };
        for index in 1..data.params.len() {
            data.params[index] = self.convert_nested(data.params[index])?;
        }
        self.convert_type_attributes(&mut data)?;

        let converted = if data == *self.ctx.types.get(ty) {
            ty
        } else {
            self.ctx.types.intern(data)
        };
        self.callable_cache.insert((ty, convention), converted);
        Ok(converted)
    }

    fn convert_nested(&mut self, ty: TypeRef) -> Result<TypeRef, TargetAbiError> {
        if let Some(&converted) = self.nested_cache.get(&ty) {
            return Ok(converted);
        }
        let data = self.ctx.types.get(ty).clone();
        if data.dialect == Symbol::new("closure") && data.name == Symbol::new("closure") {
            if data.params.len() != 1 {
                return Err(TargetAbiError::new(
                    "target signature lowering: physical closure must contain one core.func",
                ));
            }
            let inner = data.params[0];
            let convention = get_physical_closure_convention(self.ctx, ty);
            if convention.is_none()
                && data
                    .attrs
                    .contains_key(tribute_core::CALLING_CONVENTION_ATTR)
            {
                return Err(TargetAbiError::new(
                    "target signature lowering: physical closure has an invalid convention tag",
                ));
            }
            let converted_inner = if let Some(convention) = convention {
                self.convert_callable(inner, convention)?
            } else {
                let callable = core::Func::from_type_ref(self.ctx, inner).ok_or_else(|| {
                    TargetAbiError::new(
                        "target signature lowering: closure does not contain core.func",
                    )
                })?;
                if callable.r#return(self.ctx) == self.never_ty {
                    return Err(TargetAbiError::new(
                        "target signature lowering: untagged nested core.func<core.never, ...>",
                    ));
                }
                self.convert_callable(inner, CallingConvention::Direct)?
            };
            let mut converted_data = data.clone();
            converted_data.params[0] = converted_inner;
            self.convert_type_attributes(&mut converted_data)?;
            let converted = if converted_data == data {
                ty
            } else {
                self.ctx.types.intern(converted_data)
            };
            self.nested_cache.insert(ty, converted);
            return Ok(converted);
        }

        if data.dialect == Symbol::new("core") && data.name == Symbol::new("func") {
            let callable = core::Func::from_type_ref(self.ctx, ty)
                .expect("core.func type was matched structurally");
            if callable.r#return(self.ctx) == self.never_ty {
                return Err(TargetAbiError::new(
                    "target signature lowering: untagged nested core.func<core.never, ...>",
                ));
            }
            let converted = self.convert_callable(ty, CallingConvention::Direct)?;
            self.nested_cache.insert(ty, converted);
            return Ok(converted);
        }

        let mut converted_data = data.clone();
        for param in &mut converted_data.params {
            *param = self.convert_nested(*param)?;
        }
        self.convert_type_attributes(&mut converted_data)?;
        let converted = if converted_data == data {
            ty
        } else {
            self.ctx.types.intern(converted_data)
        };
        self.nested_cache.insert(ty, converted);
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
            Attribute::Type(ty) => Ok(Attribute::Type(self.convert_nested(ty)?)),
            Attribute::List(values) => Ok(Attribute::List(
                values
                    .into_iter()
                    .map(|value| self.convert_attribute(value))
                    .collect::<Result<_, _>>()?,
            )),
            other => Ok(other),
        }
    }
}

fn validate_root_done_k_type(
    ctx: &IrContext,
    done_k_ty: TypeRef,
    source_result: TypeRef,
    physical_result: TypeRef,
) -> Result<(), TargetAbiError> {
    if get_physical_closure_convention(ctx, done_k_ty) != Some(CallingConvention::Cps) {
        return Err(TargetAbiError::new(
            "target root bridge: done_k closure must carry exact Cps type provenance",
        ));
    }
    let done_closure = closure::Closure::from_type_ref(ctx, done_k_ty).ok_or_else(|| {
        TargetAbiError::new("target root bridge: done_k parameter is not closure.closure")
    })?;
    let done_callable =
        core::Func::from_type_ref(ctx, done_closure.func_type(ctx)).ok_or_else(|| {
            TargetAbiError::new("target root bridge: done_k closure does not contain core.func")
        })?;
    if done_callable.r#return(ctx) != physical_result
        || done_callable.params(ctx) != [source_result]
    {
        return Err(TargetAbiError::new(
            "target root bridge: done_k must accept the exact source result and return physically \
             empty",
        ));
    }
    Ok(())
}

/// Validate the final root reject-dispatch contract without inferring it from
/// argument shape. The root owns the only rejecting dispatcher; every other
/// operation receives a captured, result-indexed dispatch closure.
fn validate_root_dispatch_type(
    ctx: &IrContext,
    dispatch_ty: TypeRef,
    physical_result: TypeRef,
) -> Result<TypeRef, TargetAbiError> {
    if get_physical_closure_convention(ctx, dispatch_ty) != Some(CallingConvention::Cps) {
        return Err(TargetAbiError::new(
            "target root bridge: dispatch closure must carry exact Cps type provenance",
        ));
    }
    let closure = closure::Closure::from_type_ref(ctx, dispatch_ty).ok_or_else(|| {
        TargetAbiError::new("target root bridge: dispatch parameter is not closure.closure")
    })?;
    let function_ty = closure.func_type(ctx);
    let callable = core::Func::from_type_ref(ctx, function_ty).ok_or_else(|| {
        TargetAbiError::new("target root bridge: dispatch closure does not contain core.func")
    })?;
    let params = callable.params(ctx);
    let i32_ty = ctx
        .types
        .iter()
        .find_map(|(ty, data)| {
            (data.dialect == Symbol::new("core") && data.name == Symbol::new("i32")).then_some(ty)
        })
        .ok_or_else(|| TargetAbiError::new("target root bridge: core.i32 type is unavailable"))?;
    let payload_ty = ctx.types.get(params[5]);
    if callable.r#return(ctx) != physical_result
        || params.len() != 6
        || !ability::is_evidence_type_ref(ctx, params[0])
        || get_physical_closure_convention(ctx, params[1]) != Some(CallingConvention::Cps)
        || params[2..5] != [i32_ty, i32_ty, i32_ty]
        || payload_ty.dialect != Symbol::new("tribute_rt")
        || payload_ty.name != Symbol::new("anyref")
    {
        return Err(TargetAbiError::new(
            "target root bridge: dispatch must be Dispatch<R> with exact physical Cps result",
        ));
    }
    Ok(function_ty)
}

fn checked_calling_convention(
    ctx: &IrContext,
    op: OpRef,
) -> Result<Option<CallingConvention>, TargetAbiError> {
    if !ctx.op(op).attributes.contains_key(CALLING_CONVENTION_ATTR) {
        return Ok(None);
    }
    get_calling_convention(ctx, op)
        .ok_or_else(|| {
            TargetAbiError::new(
                "target root bridge: root main has an invalid tribute.calling_convention attribute",
            )
        })
        .map(Some)
}

fn checked_root_export_convention(
    ctx: &IrContext,
    op: OpRef,
) -> Result<Option<CallingConvention>, TargetAbiError> {
    if !ctx
        .op(op)
        .attributes
        .contains_key(ROOT_EXPORT_CONVENTION_ATTR)
    {
        return Ok(None);
    }
    let convention = get_root_export_convention(ctx, op).ok_or_else(|| {
        TargetAbiError::new(
            "target root bridge: preserved root export convention attribute is malformed",
        )
    })?;
    Ok(Some(convention))
}

fn checked_root_source_result(
    ctx: &IrContext,
    op: OpRef,
) -> Result<Option<TypeRef>, TargetAbiError> {
    if !ctx.op(op).attributes.contains_key(ROOT_SOURCE_RESULT_ATTR) {
        return Ok(None);
    }
    get_root_source_result(ctx, op)
        .ok_or_else(|| {
            TargetAbiError::new(
                "target root bridge: preserved root source result attribute is malformed",
            )
        })
        .map(Some)
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
    let regions = ctx.op(op).regions.to_vec();
    for region in regions {
        let blocks = ctx.region(region).blocks.to_vec();
        for block in blocks {
            let ops = ctx.block(block).ops.to_vec();
            for nested in ops {
                rewrite_symbol_refs(ctx, nested, old, new);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;

    fn function_by_name(ctx: &IrContext, module: Module, name: &str) -> func::Func {
        let name = Symbol::from_dynamic(name);
        module
            .ops(ctx)
            .into_iter()
            .find_map(|op| {
                let function = func::Func::from_op(ctx, op).ok()?;
                (function.sym_name(ctx) == name).then_some(function)
            })
            .unwrap_or_else(|| panic!("missing function `{name}`"))
    }

    #[test]
    fn physicalizes_only_proven_cps_occurrences_with_identical_never_shapes() {
        let input = r#"core.module @test {
  !direct_closure = closure.closure(core.func(core.never, core.i32)) {tribute.calling_convention = 0}
  !cps_closure = closure.closure(core.func(core.never, core.i32)) {tribute.calling_convention = 2}

  func.func @holder(%direct: !direct_closure, %cps: !cps_closure) -> core.nil attributes {tribute.calling_convention = 0} {
    func.unreachable
  }

  func.func @direct() -> core.never attributes {tribute.calling_convention = 0} {
    func.unreachable
  }

  func.func @cps() -> core.never attributes {tribute.calling_convention = 2} {
    func.unreachable
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);
        let direct_before = function_by_name(&ctx, module, "direct").r#type(&ctx);
        let cps_before = function_by_name(&ctx, module, "cps").r#type(&ctx);
        assert_eq!(
            direct_before, cps_before,
            "fixture must share one structurally interned core.func type"
        );

        lower_cps_signatures_to_physical(&mut ctx, module).unwrap();

        let never = core::never(&mut ctx).as_type_ref();
        let nil = core::nil(&mut ctx).as_type_ref();
        let direct = function_by_name(&ctx, module, "direct");
        let cps = function_by_name(&ctx, module, "cps");
        assert_eq!(
            core::Func::from_type_ref(&ctx, direct.r#type(&ctx))
                .unwrap()
                .r#return(&ctx),
            never
        );
        assert_eq!(
            core::Func::from_type_ref(&ctx, cps.r#type(&ctx))
                .unwrap()
                .r#return(&ctx),
            nil
        );

        let direct_alias = ctx
            .type_alias_by_name(Symbol::new("direct_closure"))
            .unwrap();
        let cps_alias = ctx.type_alias_by_name(Symbol::new("cps_closure")).unwrap();
        let direct_inner = closure::Closure::from_type_ref(&ctx, direct_alias)
            .unwrap()
            .func_type(&ctx);
        let cps_inner = closure::Closure::from_type_ref(&ctx, cps_alias)
            .unwrap()
            .func_type(&ctx);
        assert_eq!(
            core::Func::from_type_ref(&ctx, direct_inner)
                .unwrap()
                .r#return(&ctx),
            never
        );
        assert_eq!(
            core::Func::from_type_ref(&ctx, cps_inner)
                .unwrap()
                .r#return(&ctx),
            nil
        );
    }

    #[test]
    fn untagged_nested_never_callable_fails_without_mutation() {
        let input = r#"core.module @test {
  !untagged = closure.closure(core.func(core.never, core.i32))

  func.func @holder(%callee: !untagged) -> core.nil attributes {tribute.calling_convention = 0} {
    func.unreachable
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);
        let before = print_module(&ctx, module.op());

        let error = lower_cps_signatures_to_physical(&mut ctx, module).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("untagged nested core.func<core.never"),
            "{error}"
        );
        assert_eq!(print_module(&ctx, module.op()), before);
    }

    #[test]
    fn physicalizes_proper_tail_structured_never_for_native_cfg_and_wasm_if() {
        let input = r#"core.module @test {
  func.func @done() -> core.never attributes {tribute.calling_convention = 2} {
    func.unreachable
  }
  func.func @run(%condition: core.i1) -> core.never attributes {tribute.calling_convention = 2} {
    %never = scf.if %condition : core.never {
      func.tail_call {callee = @done, tribute.calling_convention = 2}
    } {
      func.tail_call {callee = @done, tribute.calling_convention = 2}
    }
  }
}"#;

        let mut native_ctx = IrContext::new();
        let native_module = parse_test_module(&mut native_ctx, input);
        lower_cps_signatures_to_physical(&mut native_ctx, native_module).unwrap();
        let nil = core::nil(&mut native_ctx).as_type_ref();
        let never = core::never(&mut native_ctx).as_type_ref();
        let structured = collect_ops(&native_ctx, native_module.op())
            .into_iter()
            .find(|&op| scf::If::matches(&native_ctx, op))
            .expect("structured tail marker");
        assert_eq!(native_ctx.op_result_types(structured), [nil]);

        trunk_ir::transforms::scf_to_cf::lower_scf_to_cf(&mut native_ctx, native_module);
        let native_ir = print_module(&native_ctx, native_module.op());
        assert!(!native_ir.contains("scf.if"), "{native_ir}");
        assert!(!native_ir.contains("core.never"), "{native_ir}");
        for block in collect_blocks(&native_ctx, &collect_ops(&native_ctx, native_module.op())) {
            for arg in &native_ctx.block(block).args {
                assert_ne!(arg.ty, never, "native CFG retained a core.never block arg");
            }
            for &op in &native_ctx.block(block).ops {
                let data = native_ctx.op(op);
                if data.dialect == Symbol::new("cf")
                    && (data.name == Symbol::new("br") || data.name == Symbol::new("cond_br"))
                {
                    assert!(
                        native_ctx
                            .op_operands(op)
                            .iter()
                            .all(|&operand| native_ctx.value_ty(operand) != never),
                        "native CFG edge retained a core.never operand: {native_ir}"
                    );
                }
            }
        }

        let mut wasm_ctx = IrContext::new();
        let wasm_module = parse_test_module(&mut wasm_ctx, input);
        lower_cps_signatures_to_physical(&mut wasm_ctx, wasm_module).unwrap();
        let wasm_tc = crate::wasm::type_converter::wasm_type_converter(&mut wasm_ctx);
        trunk_ir_wasm_backend::passes::scf_to_wasm::lower(&mut wasm_ctx, wasm_module, wasm_tc);
        let wasm_ir = print_module(&wasm_ctx, wasm_module.op());
        assert!(wasm_ir.contains("wasm.if"), "{wasm_ir}");
        assert!(!wasm_ir.contains("core.never"), "{wasm_ir}");
        let wasm_if = collect_ops(&wasm_ctx, wasm_module.op())
            .into_iter()
            .find(|&op| {
                let data = wasm_ctx.op(op);
                data.dialect == Symbol::new("wasm") && data.name == Symbol::new("if")
            })
            .expect("physical wasm if");
        assert_eq!(wasm_ctx.op_result_types(wasm_if), [nil]);
    }

    #[test]
    fn malformed_structured_never_marker_fails_without_mutation() {
        let input = r#"core.module @test {
  func.func @run(%condition: core.i1) -> core.never attributes {tribute.calling_convention = 2} {
    %never = scf.if %condition : core.never {
      %unit = arith.const {value = unit} : core.nil
    } {
      %unit = arith.const {value = unit} : core.nil
    }
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);
        let before = print_module(&ctx, module.op());

        let error = lower_cps_signatures_to_physical(&mut ctx, module).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("must end every branch in a proper-tail transfer or typed unreachable"),
            "{error}"
        );
        assert_eq!(print_module(&ctx, module.op()), before);
    }

    #[test]
    fn native_nil_abi_projects_interleaved_direct_indirect_and_constant_signatures() {
        let input = r#"core.module @test {
  !cps = closure.closure(core.func(core.nil, core.i32, core.nil, core.i64)) {tribute.calling_convention = 2}

  func.func @callee(%head: core.i32, %unit: core.nil, %tail: core.i64) -> core.nil attributes {tribute.calling_convention = 2} {
    func.unreachable
  }
  func.func @direct(%head: core.i32, %unit: core.nil, %tail: core.i64) -> core.nil attributes {tribute.calling_convention = 2} {
    func.tail_call %head, %unit, %tail {callee = @callee, tribute.calling_convention = 2}
  }
  func.func @indirect(%callee: !cps, %head: core.i32, %unit: core.nil, %tail: core.i64) -> core.nil attributes {tribute.calling_convention = 2} {
    func.tail_call_indirect %callee, %head, %unit, %tail {tribute.calling_convention = 2}
  }
  func.func @constant_user() -> core.nil attributes {tribute.calling_convention = 2} {
    %reference = func.constant {func_ref = @callee} : core.func(core.nil, core.i32, core.nil, core.i64)
    func.unreachable
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);
        let nil = core::nil(&mut ctx).as_type_ref();
        let i32_ty = ctx.types.intern(TypeData {
            dialect: Symbol::new("core"),
            name: Symbol::new("i32"),
            params: smallvec![],
            attrs: AttributeMap::new(),
        });
        let i64_ty = ctx.types.intern(TypeData {
            dialect: Symbol::new("core"),
            name: Symbol::new("i64"),
            params: smallvec![],
            attrs: AttributeMap::new(),
        });

        lower_native_nil_abi(&mut ctx, module).unwrap();

        for name in ["callee", "direct"] {
            let function = function_by_name(&ctx, module, name);
            let callable = core::Func::from_type_ref(&ctx, function.r#type(&ctx)).unwrap();
            assert_eq!(callable.params(&ctx), [i32_ty, i64_ty]);
            let entry = ctx.region(function.body(&ctx)).blocks[0];
            assert_eq!(
                ctx.block_args(entry)
                    .iter()
                    .map(|&arg| ctx.value_ty(arg))
                    .collect::<Vec<_>>(),
                vec![i32_ty, i64_ty]
            );
        }

        let indirect = function_by_name(&ctx, module, "indirect");
        let indirect_entry = ctx.region(indirect.body(&ctx)).blocks[0];
        let projected_callee_ty = ctx.value_ty(ctx.block_args(indirect_entry)[0]);
        let projected_inner = closure::Closure::from_type_ref(&ctx, projected_callee_ty)
            .unwrap()
            .func_type(&ctx);
        assert_eq!(
            core::Func::from_type_ref(&ctx, projected_inner)
                .unwrap()
                .params(&ctx),
            [i32_ty, i64_ty]
        );
        assert_eq!(
            ctx.block_args(indirect_entry)
                .iter()
                .map(|&arg| ctx.value_ty(arg))
                .collect::<Vec<_>>(),
            vec![projected_callee_ty, i32_ty, i64_ty]
        );
        let indirect_tail = collect_ops(&ctx, indirect.op_ref())
            .into_iter()
            .find(|&op| func::TailCallIndirect::from_op(&ctx, op).is_ok())
            .unwrap();
        assert_eq!(
            ctx.op_operands(indirect_tail)
                .iter()
                .map(|&value| ctx.value_ty(value))
                .collect::<Vec<_>>(),
            vec![projected_callee_ty, i32_ty, i64_ty]
        );

        let direct = function_by_name(&ctx, module, "direct");
        let direct_tail = collect_ops(&ctx, direct.op_ref())
            .into_iter()
            .find(|&op| func::TailCall::from_op(&ctx, op).is_ok())
            .unwrap();
        assert_eq!(
            ctx.op_operands(direct_tail)
                .iter()
                .map(|&value| ctx.value_ty(value))
                .collect::<Vec<_>>(),
            vec![i32_ty, i64_ty]
        );

        let constant_user = function_by_name(&ctx, module, "constant_user");
        let constant = collect_ops(&ctx, constant_user.op_ref())
            .into_iter()
            .find(|&op| func::Constant::from_op(&ctx, op).is_ok())
            .unwrap();
        let constant_callable =
            core::Func::from_type_ref(&ctx, ctx.op_result_types(constant)[0]).unwrap();
        assert_eq!(constant_callable.params(&ctx), [i32_ty, i64_ty]);
        assert!(
            collect_blocks(&ctx, &collect_ops(&ctx, module.op()))
                .into_iter()
                .flat_map(|block| ctx.block_args(block))
                .all(|&arg| ctx.value_ty(arg) != nil)
        );
    }

    #[test]
    fn native_nil_abi_rejects_mismatched_direct_transfer_without_mutation() {
        let input = r#"core.module @test {
  func.func @callee(%value: core.i32, %unit: core.nil) -> core.nil attributes {tribute.calling_convention = 2} {
    func.unreachable
  }
  func.func @caller(%value: core.i32) -> core.nil attributes {tribute.calling_convention = 2} {
    func.tail_call %value {callee = @callee, tribute.calling_convention = 2}
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);
        let before = print_module(&ctx, module.op());

        let error = lower_native_nil_abi(&mut ctx, module).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("does not match its exact pre-projection signature")
        );
        assert_eq!(print_module(&ctx, module.op()), before);
    }

    #[test]
    fn native_nil_abi_rejects_mismatched_nil_return_without_mutation() {
        let input = r#"core.module @test {
  func.func @bad() -> core.i32 attributes {tribute.calling_convention = 0} {
    %unit = arith.const {value = unit} : core.nil
    func.return %unit
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);
        let before = print_module(&ctx, module.op());

        let error = lower_native_nil_abi(&mut ctx, module).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("does not match its function result")
        );
        assert_eq!(print_module(&ctx, module.op()), before);
    }

    #[test]
    fn native_nil_abi_rejects_multi_value_nil_return_without_mutation() {
        let input = r#"core.module @test {
  func.func @bad() -> core.nil attributes {tribute.calling_convention = 0} {
    %unit = arith.const {value = unit} : core.nil
    %value = arith.const {value = 1} : core.i32
    func.return %unit, %value
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);
        let before = print_module(&ctx, module.op());

        let error = lower_native_nil_abi(&mut ctx, module).unwrap_err();

        assert!(error.to_string().contains("must be the sole return value"));
        assert_eq!(print_module(&ctx, module.op()), before);
    }

    #[test]
    fn native_nil_abi_rejects_mismatched_indirect_callable_without_mutation() {
        let input = r#"core.module @test {
  !expects_i32 = closure.closure(core.func(core.nil, core.i32)) {tribute.calling_convention = 2}

  func.func @caller(%callee: !expects_i32, %unit: core.nil) -> core.nil attributes {tribute.calling_convention = 2} {
    func.tail_call_indirect %callee, %unit {tribute.calling_convention = 2}
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);
        let before = print_module(&ctx, module.op());

        let error = lower_native_nil_abi(&mut ctx, module).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("does not match its exact pre-projection callable signature")
        );
        assert_eq!(print_module(&ctx, module.op()), before);
    }

    #[test]
    fn native_nil_abi_uses_provenance_after_closure_lowering_erases_callee_type() {
        let input = r#"core.module @test {
  !done = closure.closure(core.func(core.never, core.nil)) {tribute.calling_convention = 2}

  func.func @generated(%done: !done, %answer: core.nil) -> core.never attributes {tribute.calling_convention = 2} {
    func.tail_call_indirect %done, %answer {tribute.calling_convention = 2}
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);
        let generated = function_by_name(&ctx, module, "generated");

        crate::closure_lower::lower_closures_in_func(&mut ctx, generated);
        let raw_tail = collect_ops(&ctx, generated.op_ref())
            .into_iter()
            .find(|&op| func::TailCallIndirect::from_op(&ctx, op).is_ok())
            .unwrap();
        let raw_callee = ctx.op_operands(raw_tail)[0];
        assert_eq!(
            ctx.types.get(ctx.value_ty(raw_callee)).name,
            Symbol::new("i32")
        );
        let raw_signature = get_indirect_call_signature(&ctx, raw_tail)
            .expect("raw pointer transfer must retain its exact callable provenance");
        let raw_callable = core::Func::from_type_ref(&ctx, raw_signature).unwrap();
        let raw_params = raw_callable.params(&ctx).to_vec();
        let nil = core::nil(&mut ctx).as_type_ref();
        let anyref = tribute_rt::anyref(&mut ctx).as_type_ref();
        assert_eq!(raw_params, [anyref, nil]);

        lower_cps_signatures_to_physical(&mut ctx, module).unwrap();
        lower_native_nil_abi(&mut ctx, module).unwrap();

        let tail = collect_ops(&ctx, generated.op_ref())
            .into_iter()
            .find(|&op| func::TailCallIndirect::from_op(&ctx, op).is_ok())
            .unwrap();
        assert_eq!(
            ctx.op_operands(tail).len(),
            2,
            "raw callee plus environment only"
        );
        let signature = get_indirect_call_signature(&ctx, tail)
            .expect("native lowering must preserve projected exact provenance for func-to-clif");
        let callable = core::Func::from_type_ref(&ctx, signature).unwrap();
        assert_eq!(callable.params(&ctx), [anyref].as_slice());
    }

    #[test]
    fn cps_root_bridge_uses_typed_completion_and_an_ordinary_wrapper_call() {
        let input = r#"core.module @test {
  tribute_control.func @main() -> core.nil convention(cps) {
    %unit = arith.const {value = unit} : core.nil
    tribute_control.return %unit
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);
        crate::tribute_control_to_cps::tribute_control_to_cps(&mut ctx, module, &[]).unwrap();
        let logical_main = function_by_name(&ctx, module, "main");
        let nil = core::nil(&mut ctx).as_type_ref();
        tribute_core::set_root_export_convention(
            &mut ctx,
            logical_main.op_ref(),
            CallingConvention::Direct,
        );
        tribute_core::set_root_source_result(&mut ctx, logical_main.op_ref(), nil);

        lower_cps_signatures_to_physical(&mut ctx, module).unwrap();
        compose_root_entry_bridge(&mut ctx, module).unwrap();

        let wrapper = function_by_name(&ctx, module, "main");
        let worker = function_by_name(&ctx, module, CPS_MAIN_SYMBOL);
        let done_k = function_by_name(&ctx, module, ROOT_DONE_K_SYMBOL);
        assert_eq!(
            get_calling_convention(&ctx, wrapper.op_ref()),
            Some(CallingConvention::Direct)
        );
        assert_eq!(
            get_calling_convention(&ctx, worker.op_ref()),
            Some(CallingConvention::Cps)
        );
        assert_eq!(
            get_calling_convention(&ctx, done_k.op_ref()),
            Some(CallingConvention::Cps)
        );
        for function in [worker, done_k] {
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
                .get_bool(ROOT_DONE_K_ATTR),
            Some(true)
        );
        assert_eq!(
            ctx.op(worker.op_ref())
                .attributes
                .get_bool(ROOT_CPS_WORKER_ATTR),
            Some(true)
        );
        assert_eq!(
            ctx.op(wrapper.op_ref())
                .attributes
                .get_bool(ROOT_WRAPPER_ATTR),
            Some(true)
        );

        let wrapper_ops = collect_ops(&ctx, wrapper.op_ref());
        let root_calls: Vec<_> = wrapper_ops
            .iter()
            .copied()
            .filter(|&op| {
                func::Call::from_op(&ctx, op).is_ok()
                    && ctx.op(op).attributes.get_bool(ROOT_CPS_CALL_ATTR) == Some(true)
            })
            .collect();
        assert_eq!(root_calls.len(), 1);
        assert_eq!(
            ctx.op(root_calls[0]).attributes.get_symbol("callee"),
            Some(Symbol::new(CPS_MAIN_SYMBOL))
        );
        assert_eq!(ctx.op_result_types(root_calls[0]), [nil]);
        assert!(
            wrapper_ops
                .iter()
                .all(|&op| func::TailCall::from_op(&ctx, op).is_err())
        );

        let cell = wrapper_ops
            .iter()
            .find_map(|&op| {
                let new = adt::StructNew::from_op(&ctx, op).ok()?;
                let ty = new.r#type(&ctx);
                (ctx.types.get(ty).attrs.get_symbol("name")
                    == Some(Symbol::new(ROOT_COMPLETION_CELL_NAME)))
                .then_some(ty)
            })
            .expect("root bridge must construct its completion cell");
        assert_eq!(ctx.types.get(cell).params.as_slice(), [nil]);
        assert_eq!(
            trunk_ir::adt_layout::get_struct_fields(&ctx, cell),
            Some(vec![(Symbol::new(ROOT_COMPLETION_CELL_VALUE_FIELD), nil)])
        );
        let (native_types, _) = crate::native::type_converter::native_type_converter(&mut ctx);
        let layout = trunk_ir::adt_layout::compute_struct_layout(&ctx, cell, &native_types)
            .expect("root completion cell must have a native struct layout");
        assert_eq!(layout.field_offsets, vec![0]);
        assert_eq!(layout.total_size, 0);
        assert_eq!(layout.alignment, 1);
        let root_call_index = wrapper_ops
            .iter()
            .position(|&op| op == root_calls[0])
            .expect("root call must belong to the wrapper");
        let completion_read_index = wrapper_ops
            .iter()
            .position(|&op| adt::StructGet::from_op(&ctx, op).is_ok())
            .expect("shared root wrapper must read its typed completion cell");
        assert!(
            root_call_index < completion_read_index,
            "the shared wrapper must read completion only after the worker call"
        );
        let wrapper_return = wrapper_ops
            .iter()
            .copied()
            .find(|&op| func::Return::from_op(&ctx, op).is_ok())
            .expect("root wrapper must return its completion value");
        assert!(
            ctx.op_operands(wrapper_return).len() == 1,
            "the shared/Wasm wrapper must preserve the typed Nil completion value"
        );
        let done_ops = collect_ops(&ctx, done_k.op_ref());
        assert!(
            done_ops.iter().any(|&op| {
                adt::StructSet::from_op(&ctx, op).is_ok()
                    && ctx.op_operands(op).get(1).copied()
                        == Some(ctx.block_args(ctx.region(done_k.body(&ctx)).blocks[0])[1])
            }),
            "root done_k must store its exact answer into the completion cell"
        );
        let printed = print_module(&ctx, module.op());
        assert!(!printed.contains("__tribute_cps_control"), "{printed}");
        assert!(!printed.contains("tribute_rt.unbox_int"), "{printed}");

        lower_native_nil_abi(&mut ctx, module).unwrap();
        let wrapper = function_by_name(&ctx, module, "main");
        let native_wrapper_return = collect_ops(&ctx, wrapper.op_ref())
            .into_iter()
            .find(|&op| func::Return::from_op(&ctx, op).is_ok())
            .expect("native wrapper must retain its return terminator");
        assert!(
            ctx.op_operands(native_wrapper_return).is_empty(),
            "native Nil projection must remove only the return payload"
        );
        let native_completion_read = collect_ops(&ctx, wrapper.op_ref())
            .into_iter()
            .find(|&op| adt::StructGet::from_op(&ctx, op).is_ok())
            .expect("native lowering retains the completion read until ADT lowering");
        assert!(
            !ctx.has_uses(ctx.op_results(native_completion_read)[0]),
            "native Nil projection must leave the zero-sized completion read unused"
        );
        let done_k = function_by_name(&ctx, module, ROOT_DONE_K_SYMBOL);
        let done_signature = core::Func::from_type_ref(&ctx, done_k.r#type(&ctx)).unwrap();
        assert_eq!(done_signature.params(&ctx).len(), 1);
        let done_entry = ctx.region(done_k.body(&ctx)).blocks[0];
        assert_eq!(ctx.block_args(done_entry).len(), 1);
        assert_eq!(
            ctx.value_ty(ctx.block_args(done_entry)[0]),
            tribute_rt::anyref(&mut ctx).as_type_ref()
        );
        assert!(
            collect_blocks(&ctx, &collect_ops(&ctx, module.op()))
                .into_iter()
                .flat_map(|block| ctx.block_args(block))
                .all(|&arg| ctx.value_ty(arg) != nil)
        );
    }

    #[test]
    fn promoted_evidence_direct_root_forwards_its_exact_evidence_argument() {
        let input = r#"core.module @test {
  tribute_control.func @main() -> core.nil convention(cps) {
    %unit = arith.const {value = unit} : core.nil
    tribute_control.return %unit
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);
        crate::tribute_control_to_cps::tribute_control_to_cps(&mut ctx, module, &[]).unwrap();
        let logical_main = function_by_name(&ctx, module, "main");
        let nil = core::nil(&mut ctx).as_type_ref();
        tribute_core::set_root_export_convention(
            &mut ctx,
            logical_main.op_ref(),
            CallingConvention::EvidenceDirect,
        );
        tribute_core::set_root_source_result(&mut ctx, logical_main.op_ref(), nil);

        lower_cps_signatures_to_physical(&mut ctx, module).unwrap();
        compose_root_entry_bridge(&mut ctx, module).unwrap();

        let wrapper = function_by_name(&ctx, module, "main");
        let wrapper_entry = ctx.region(wrapper.body(&ctx)).blocks[0];
        let evidence = ctx.block_args(wrapper_entry)[0];
        let wrapper_ops = collect_ops(&ctx, wrapper.op_ref());
        let root_call = wrapper_ops
            .iter()
            .copied()
            .find(|&op| ctx.op(op).attributes.get_bool(ROOT_CPS_CALL_ATTR) == Some(true))
            .expect("promoted root wrapper must call its CPS worker");
        assert_eq!(ctx.op_operands(root_call)[0], evidence);
        let evidence_ty = ability::evidence_adt_type_ref(&mut ctx);
        assert_eq!(ctx.value_ty(evidence), evidence_ty);
        assert!(
            wrapper_ops
                .iter()
                .all(|&op| adt::ArrayNew::from_op(&ctx, op).is_err()),
            "EvidenceDirect wrapper must not synthesize empty evidence"
        );
    }

    #[test]
    fn unpromoted_roots_only_drop_the_temporary_export_contract() {
        for (source_convention, export_convention) in [
            ("direct", CallingConvention::Direct),
            ("evidence_direct", CallingConvention::EvidenceDirect),
        ] {
            let input = format!(
                r#"core.module @test {{
  tribute_control.func @main() -> core.nil convention({source_convention}) {{
    %unit = arith.const {{value = unit}} : core.nil
    tribute_control.return %unit
  }}
}}"#
            );
            let mut ctx = IrContext::new();
            let module = parse_test_module(&mut ctx, &input);
            crate::tribute_control_to_cps::tribute_control_to_cps(&mut ctx, module, &[]).unwrap();
            let main = function_by_name(&ctx, module, "main");
            let nil = core::nil(&mut ctx).as_type_ref();
            tribute_core::set_root_export_convention(&mut ctx, main.op_ref(), export_convention);
            tribute_core::set_root_source_result(&mut ctx, main.op_ref(), nil);
            let signature_before = main.r#type(&ctx);
            let body_before = main.body(&ctx);
            let body_ops_before = collect_ops(&ctx, main.op_ref());

            lower_cps_signatures_to_physical(&mut ctx, module).unwrap();
            compose_root_entry_bridge(&mut ctx, module).unwrap();

            let main = function_by_name(&ctx, module, "main");
            assert_eq!(main.r#type(&ctx), signature_before);
            assert_eq!(
                get_calling_convention(&ctx, main.op_ref()),
                Some(export_convention)
            );
            assert_eq!(main.body(&ctx), body_before);
            assert_eq!(collect_ops(&ctx, main.op_ref()), body_ops_before);
            assert!(
                !ctx.op(main.op_ref())
                    .attributes
                    .contains_key(ROOT_EXPORT_CONVENTION_ATTR)
            );
            assert!(
                !ctx.op(main.op_ref())
                    .attributes
                    .contains_key(ROOT_SOURCE_RESULT_ATTR)
            );
            assert_eq!(
                module
                    .ops(&ctx)
                    .into_iter()
                    .filter(|&op| func::Func::from_op(&ctx, op).is_ok())
                    .count(),
                1
            );
        }
    }

    #[test]
    fn incomplete_unpromoted_root_contract_is_rejected_without_mutation() {
        enum IncompleteContract {
            MissingExportConvention,
            MissingSourceResult,
        }

        for incomplete in [
            IncompleteContract::MissingExportConvention,
            IncompleteContract::MissingSourceResult,
        ] {
            let input = r#"core.module @test {
  tribute_control.func @main() -> core.nil convention(direct) {
    %unit = arith.const {value = unit} : core.nil
    tribute_control.return %unit
  }
}"#;
            let mut ctx = IrContext::new();
            let module = parse_test_module(&mut ctx, input);
            crate::tribute_control_to_cps::tribute_control_to_cps(&mut ctx, module, &[]).unwrap();
            let main = function_by_name(&ctx, module, "main");
            let nil = core::nil(&mut ctx).as_type_ref();
            match incomplete {
                IncompleteContract::MissingExportConvention => {
                    tribute_core::set_root_source_result(&mut ctx, main.op_ref(), nil);
                }
                IncompleteContract::MissingSourceResult => {
                    tribute_core::set_root_export_convention(
                        &mut ctx,
                        main.op_ref(),
                        CallingConvention::Direct,
                    );
                }
            }
            lower_cps_signatures_to_physical(&mut ctx, module).unwrap();
            let before = print_module(&ctx, module.op());

            let error = compose_root_entry_bridge(&mut ctx, module).unwrap_err();

            assert!(
                error
                    .to_string()
                    .contains("must either both be present or both be absent"),
                "{error}"
            );
            assert_eq!(print_module(&ctx, module.op()), before);
        }
    }

    #[test]
    fn physicalization_and_root_bridge_keep_nested_main_scope_local() {
        let input = r#"core.module @test {
  tribute_control.func @main() -> core.nil convention(cps) {
    %unit = arith.const {value = unit} : core.nil
    tribute_control.return %unit
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);
        crate::tribute_control_to_cps::tribute_control_to_cps(&mut ctx, module, &[]).unwrap();
        let main = function_by_name(&ctx, module, "main");
        let nil = core::nil(&mut ctx).as_type_ref();
        tribute_core::set_root_export_convention(
            &mut ctx,
            main.op_ref(),
            CallingConvention::Direct,
        );
        tribute_core::set_root_source_result(&mut ctx, main.op_ref(), nil);

        // Add this before physicalization: a recursive traversal must not
        // conflate this local `@main` with the immediate root declaration.
        let location = ctx.op(module.op()).location;
        let nested_entry = ctx.create_block(BlockData {
            location,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let nested_function_type = core::func(&mut ctx, nil, []).as_type_ref();
        let local_constant = func::constant(
            &mut ctx,
            location,
            nested_function_type,
            Symbol::new("main"),
        );
        ctx.push_op(nested_entry, local_constant.op_ref());
        let local_call = func::call(&mut ctx, location, [], nil, Symbol::new("main"));
        let local_result = local_call.result(&ctx);
        ctx.push_op(nested_entry, local_call.op_ref());
        let local_return = func::r#return(&mut ctx, location, [local_result]);
        ctx.push_op(nested_entry, local_return.op_ref());
        let nested_function_region = ctx.create_region(RegionData {
            location,
            blocks: smallvec![nested_entry],
            parent_op: None,
        });
        let nested_function = func::func(
            &mut ctx,
            location,
            Symbol::new("main"),
            nested_function_type,
            nested_function_region,
        );
        set_calling_convention(
            &mut ctx,
            nested_function.op_ref(),
            CallingConvention::Direct,
        );
        let nested_module_block = ctx.create_block(BlockData {
            location,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        ctx.push_op(nested_module_block, nested_function.op_ref());
        let nested_module_region = ctx.create_region(RegionData {
            location,
            blocks: smallvec![nested_module_block],
            parent_op: None,
        });
        let nested_module = core::module(
            &mut ctx,
            location,
            Symbol::new("nested"),
            nested_module_region,
        );
        ctx.push_op(module.first_block(&ctx).unwrap(), nested_module.op_ref());

        lower_cps_signatures_to_physical(&mut ctx, module).unwrap();
        compose_root_entry_bridge(&mut ctx, module).unwrap();

        assert_eq!(
            function_by_name(&ctx, module, CPS_MAIN_SYMBOL).sym_name(&ctx),
            Symbol::new(CPS_MAIN_SYMBOL)
        );
        assert_eq!(
            function_by_name(&ctx, module, "main").sym_name(&ctx),
            Symbol::new("main")
        );

        let nested = module
            .ops(&ctx)
            .into_iter()
            .find(|&op| core::Module::from_op(&ctx, op).is_ok())
            .expect("nested module");
        let nested_call = collect_ops(&ctx, nested)
            .into_iter()
            .find(|&op| func::Call::from_op(&ctx, op).is_ok())
            .expect("nested local main call");
        let nested_constant = collect_ops(&ctx, nested)
            .into_iter()
            .find_map(|op| func::Constant::from_op(&ctx, op).ok())
            .expect("nested local main constant");
        assert_eq!(
            ctx.op(nested_call).attributes.get_symbol("callee"),
            Some(Symbol::new("main"))
        );
        assert_eq!(nested_constant.func_ref(&ctx), Symbol::new("main"));
        assert_eq!(
            ctx.op_result_types(nested_constant.op_ref()),
            [nested_function_type]
        );
        assert!(collect_ops(&ctx, nested).into_iter().any(|op| {
            func::Func::from_op(&ctx, op)
                .is_ok_and(|function| function.sym_name(&ctx) == Symbol::new("main"))
        }));
    }

    #[test]
    fn promoted_cps_root_requires_complete_well_formed_export_contract_without_mutation() {
        enum ContractCase {
            Missing,
            Malformed,
            Inconsistent,
        }

        for case in [
            ContractCase::Missing,
            ContractCase::Malformed,
            ContractCase::Inconsistent,
        ] {
            let input = r#"core.module @test {
  tribute_control.func @main() -> core.nil convention(cps) {
    %unit = arith.const {value = unit} : core.nil
    tribute_control.return %unit
  }
}"#;
            let mut ctx = IrContext::new();
            let module = parse_test_module(&mut ctx, input);
            crate::tribute_control_to_cps::tribute_control_to_cps(&mut ctx, module, &[]).unwrap();
            let main = function_by_name(&ctx, module, "main");
            let nil = core::nil(&mut ctx).as_type_ref();
            match case {
                ContractCase::Missing => {}
                ContractCase::Malformed => {
                    ctx.op_mut(main.op_ref())
                        .attributes
                        .insert(Symbol::new(ROOT_EXPORT_CONVENTION_ATTR), Attribute::Int(99));
                    tribute_core::set_root_source_result(&mut ctx, main.op_ref(), nil);
                }
                ContractCase::Inconsistent => {
                    tribute_core::set_root_export_convention(
                        &mut ctx,
                        main.op_ref(),
                        CallingConvention::Direct,
                    );
                    let i32 = ctx.types.intern(TypeData {
                        dialect: Symbol::new("core"),
                        name: Symbol::new("i32"),
                        params: smallvec![],
                        attrs: AttributeMap::new(),
                    });
                    tribute_core::set_root_source_result(&mut ctx, main.op_ref(), i32);
                }
            }
            lower_cps_signatures_to_physical(&mut ctx, module).unwrap();
            let before = print_module(&ctx, module.op());

            let error = compose_root_entry_bridge(&mut ctx, module).unwrap_err();

            assert!(error.to_string().contains("target root bridge"), "{error}");
            assert_eq!(print_module(&ctx, module.op()), before);
        }
    }
}
