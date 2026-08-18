use std::collections::{HashMap, HashSet};
use std::fmt;

use tribute_core::calling_convention::get_physical_closure_environment_index;
use tribute_core::{
    CallingConvention, get_calling_convention, get_indirect_call_signature,
    get_physical_closure_convention,
};
use tribute_ir::dialect::closure;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::{adt, arith, clif, core, func, mem};
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::rewrite::{Module, TypeConverter};
use trunk_ir::transforms::call_graph::{build_call_graph, recursive_functions};
use trunk_ir::{Attribute, OpRef, RegionRef, Symbol, TypeRef, ValueRef};

pub const PARAMETER_OWNERSHIP_ATTR: &str = "tribute.rc.parameter_ownership_v1";
pub const PARAMETER_ENTRY_OWNERSHIP_ATTR: &str = "tribute.rc.parameter_entry_ownership_v1";
pub const CALL_ARGUMENT_OWNERSHIP_ATTR: &str = "tribute.rc.call_argument_ownership_v1";
pub const OWNERSHIP_CONTRACT_ID_ATTR: &str = "tribute.rc.ownership_contract_id_v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParameterOwnership {
    Borrowed,
    Owned,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RcOwnership {
    Plain,
    Borrowed,
    Retained,
    Consumed,
    Acquire,
    Transfer,
}

impl RcOwnership {
    fn as_attribute(self) -> Attribute {
        Attribute::Symbol(Symbol::new(match self {
            Self::Plain => "plain",
            Self::Borrowed => "borrowed",
            Self::Retained => "retained",
            Self::Consumed => "consumed",
            Self::Acquire => "acquire",
            Self::Transfer => "transfer",
        }))
    }

    fn list_attribute(entries: &[Self]) -> Attribute {
        Attribute::List(entries.iter().copied().map(Self::as_attribute).collect())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwnershipContractError(&'static str);

impl fmt::Display for OwnershipContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "native RC ownership contract: {}", self.0)
    }
}

impl std::error::Error for OwnershipContractError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrustedCallContract {
    pub actions: Vec<RcOwnership>,
    pub is_tail: bool,
    pub is_indirect: bool,
    direct_callee: Option<Symbol>,
    indirect_signature: Option<TypeRef>,
    indirect_physical_closures: Option<Vec<Option<TypeRef>>>,
}

impl ParameterOwnership {
    fn as_attribute(self) -> Attribute {
        Attribute::Symbol(Symbol::new(match self {
            Self::Borrowed => "borrowed",
            Self::Owned => "owned",
        }))
    }
}

pub struct TrustedOwnershipSummaries {
    summaries: HashMap<Symbol, Vec<ParameterOwnership>>,
    entry_contracts: HashMap<Symbol, Vec<RcOwnership>>,
    entry_physical_closures: HashMap<Symbol, Vec<Option<TypeRef>>>,
    call_contracts: HashMap<i128, TrustedCallContract>,
}

pub struct ValidatedOwnershipContracts {
    pub summaries: HashMap<Symbol, Vec<ParameterOwnership>>,
    pub entry_contracts: HashMap<Symbol, Vec<RcOwnership>>,
    pub call_contracts: HashMap<OpRef, TrustedCallContract>,
}

pub fn compute_and_attach(
    ctx: &mut IrContext,
    module: Module,
    type_converter: &TypeConverter,
) -> Result<TrustedOwnershipSummaries, OwnershipContractError> {
    let Some(module_block) = module.first_block(ctx) else {
        return Ok(TrustedOwnershipSummaries {
            summaries: HashMap::new(),
            entry_contracts: HashMap::new(),
            entry_physical_closures: HashMap::new(),
            call_contracts: HashMap::new(),
        });
    };
    let module_ops = ctx.block(module_block).ops.to_vec();
    let mut definitions: HashMap<Symbol, Vec<OpRef>> = HashMap::new();
    for &op in &module_ops {
        if let Ok(function) = func::Func::from_op(ctx, op) {
            definitions
                .entry(function.sym_name(ctx))
                .or_default()
                .push(op);
        }
    }

    let unique_functions: HashMap<Symbol, OpRef> = definitions
        .iter()
        .filter_map(|(&symbol, ops)| (ops.len() == 1).then_some((symbol, ops[0])))
        .collect();
    let recursive = recursive_functions(&build_call_graph(ctx, module));
    let mut ineligible = recursive;
    for (&symbol, &op) in &unique_functions {
        if ctx.op(op).attributes.contains_key("abi") {
            ineligible.insert(symbol);
        }
    }
    collect_escaping_function_symbols(ctx, &module_ops, &unique_functions, &mut ineligible);

    let mut summaries = HashMap::new();
    for (&symbol, &op) in &unique_functions {
        let function = func::Func::from_op(ctx, op).expect("collected func.func");
        let parameters = entry_parameters(ctx, function.body(ctx));
        let initial: Vec<ParameterOwnership> = parameters
            .iter()
            .map(|&parameter| {
                if !ineligible.contains(&symbol)
                    && lowers_to_rc_managed(ctx, parameter, type_converter)
                {
                    ParameterOwnership::Borrowed
                } else {
                    ParameterOwnership::Owned
                }
            })
            .collect();
        summaries.insert(symbol, initial);
    }

    loop {
        let mut changed = false;
        for (&symbol, &op) in &unique_functions {
            if ineligible.contains(&symbol) {
                continue;
            }
            let function = func::Func::from_op(ctx, op).expect("collected func.func");
            let body = function.body(ctx);
            let parameters = entry_parameters(ctx, body);
            for (index, &parameter) in parameters.iter().enumerate() {
                if summaries[&symbol][index] == ParameterOwnership::Borrowed
                    && !value_is_borrowed(ctx, body, parameter, &summaries, &mut HashSet::new())
                {
                    summaries.get_mut(&symbol).expect("summary exists")[index] =
                        ParameterOwnership::Owned;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }

    let mut entry_contracts = HashMap::new();
    let mut entry_physical_closures = HashMap::new();
    for (&symbol, &op) in &unique_functions {
        let function = func::Func::from_op(ctx, op).expect("collected func.func");
        let consuming_cps = is_defined_physical_cps_function(ctx, op);
        let parameters = entry_parameters(ctx, function.body(ctx));
        let entries = parameters
            .iter()
            .copied()
            .enumerate()
            .map(|(index, parameter)| {
                if !lowers_to_rc_managed(ctx, parameter, type_converter) {
                    RcOwnership::Plain
                } else if consuming_cps {
                    RcOwnership::Consumed
                } else if summaries[&symbol][index] == ParameterOwnership::Borrowed {
                    RcOwnership::Borrowed
                } else {
                    RcOwnership::Retained
                }
            })
            .collect();
        let physical_closures = parameters
            .iter()
            .map(|&parameter| {
                let ty = ctx.value_ty(parameter);
                type_is_proven_physical_closure(ctx, ty).then_some(ty)
            })
            .collect();
        entry_contracts.insert(symbol, entries);
        entry_physical_closures.insert(symbol, physical_closures);
    }

    let mut pending_calls = Vec::new();
    let mut next_contract_id = 1i128;
    for &op in unique_functions.values() {
        let function = func::Func::from_op(ctx, op).expect("collected func.func");
        collect_call_contracts(
            ctx,
            function.body(ctx),
            type_converter,
            &entry_contracts,
            &mut next_contract_id,
            &mut pending_calls,
        )?;
    }

    strip_contract_metadata(ctx, module.op());

    for (&symbol, &op) in &unique_functions {
        let entries = summaries[&symbol]
            .iter()
            .copied()
            .map(ParameterOwnership::as_attribute)
            .collect();
        ctx.op_mut(op).attributes.insert(
            Symbol::new(PARAMETER_OWNERSHIP_ATTR),
            Attribute::List(entries),
        );
        ctx.op_mut(op).attributes.insert(
            Symbol::new(PARAMETER_ENTRY_OWNERSHIP_ATTR),
            RcOwnership::list_attribute(&entry_contracts[&symbol]),
        );
    }

    let mut call_contracts = HashMap::new();
    for (op, id, contract) in pending_calls {
        ctx.op_mut(op)
            .attributes
            .insert(Symbol::new(OWNERSHIP_CONTRACT_ID_ATTR), Attribute::Int(id));
        ctx.op_mut(op).attributes.insert(
            Symbol::new(CALL_ARGUMENT_OWNERSHIP_ATTR),
            RcOwnership::list_attribute(&contract.actions),
        );
        call_contracts.insert(id, contract);
    }

    Ok(TrustedOwnershipSummaries {
        summaries,
        entry_contracts,
        entry_physical_closures,
        call_contracts,
    })
}

fn strip_contract_metadata(ctx: &mut IrContext, op: OpRef) {
    let regions = ctx.op(op).regions.to_vec();
    let attributes = &mut ctx.op_mut(op).attributes;
    attributes.remove(PARAMETER_OWNERSHIP_ATTR);
    attributes.remove(PARAMETER_ENTRY_OWNERSHIP_ATTR);
    attributes.remove(CALL_ARGUMENT_OWNERSHIP_ATTR);
    attributes.remove(OWNERSHIP_CONTRACT_ID_ATTR);
    for region in regions {
        let blocks = ctx.region(region).blocks.clone();
        for block in blocks {
            let ops = ctx.block(block).ops.clone();
            for nested in ops {
                strip_contract_metadata(ctx, nested);
            }
        }
    }
}

fn is_defined_physical_cps_function(ctx: &IrContext, op: OpRef) -> bool {
    if get_calling_convention(ctx, op) != Some(CallingConvention::Cps)
        || ctx.op(op).attributes.contains_key("abi")
        || ctx
            .op(op)
            .regions
            .first()
            .is_none_or(|region| ctx.region(*region).blocks.is_empty())
    {
        return false;
    }
    let Some(signature) = ctx.op(op).attributes.get_type("type") else {
        return false;
    };
    let Some(callable) = core::Func::from_type_ref(ctx, signature) else {
        return false;
    };
    let result = ctx.types.get(callable.r#return(ctx));
    result.dialect == Symbol::new("core") && result.name == Symbol::new("nil")
}

fn type_lowers_to_anyref(ctx: &mut IrContext, ty: TypeRef, type_converter: &TypeConverter) -> bool {
    let converted = type_converter.convert_type_or_identity(ctx, ty);
    let data = ctx.types.get(converted);
    data.dialect == Symbol::new("tribute_rt") && data.name == Symbol::new("anyref")
}

fn type_is_proven_physical_closure(ctx: &IrContext, ty: TypeRef) -> bool {
    let Some(environment_index) = get_physical_closure_environment_index(ctx, ty) else {
        return false;
    };
    if get_physical_closure_convention(ctx, ty).is_none() {
        return false;
    }
    let Some(closure) = closure::Closure::from_type_ref(ctx, ty) else {
        return false;
    };
    let Some(callable) = core::Func::from_type_ref(ctx, closure.func_type(ctx)) else {
        return false;
    };
    environment_index <= callable.params(ctx).len()
}

fn type_lowers_to_rc_managed(
    ctx: &mut IrContext,
    ty: TypeRef,
    type_converter: &TypeConverter,
) -> bool {
    type_lowers_to_anyref(ctx, ty, type_converter) || type_is_proven_physical_closure(ctx, ty)
}

fn lowered_indirect_signature(
    ctx: &mut IrContext,
    signature: TypeRef,
    type_converter: &TypeConverter,
) -> Result<TypeRef, OwnershipContractError> {
    let callable = core::Func::from_type_ref(ctx, signature).ok_or(OwnershipContractError(
        "indirect proper-tail signature is not core.func",
    ))?;
    let return_type = type_converter.convert_type_or_identity(ctx, callable.r#return(ctx));
    let parameter_types: Vec<_> = callable
        .params(ctx)
        .iter()
        .map(|&parameter| type_converter.convert_type_or_identity(ctx, parameter))
        .collect();
    Ok(core::func(ctx, return_type, parameter_types.iter().copied()).as_type_ref())
}

fn indirect_physical_closures(
    ctx: &IrContext,
    signature: TypeRef,
) -> Result<Vec<Option<TypeRef>>, OwnershipContractError> {
    let callable = core::Func::from_type_ref(ctx, signature).ok_or(OwnershipContractError(
        "indirect proper-tail signature is not core.func",
    ))?;
    Ok(callable
        .params(ctx)
        .iter()
        .map(|&parameter| type_is_proven_physical_closure(ctx, parameter).then_some(parameter))
        .collect())
}

fn collect_call_contracts(
    ctx: &mut IrContext,
    region: RegionRef,
    type_converter: &TypeConverter,
    entries: &HashMap<Symbol, Vec<RcOwnership>>,
    next_id: &mut i128,
    pending: &mut Vec<(OpRef, i128, TrustedCallContract)>,
) -> Result<(), OwnershipContractError> {
    let blocks = ctx.region(region).blocks.clone();
    for block in blocks {
        let ops = ctx.block(block).ops.clone();
        for op in ops {
            let direct = func::Call::matches(ctx, op) || func::TailCall::matches(ctx, op);
            let indirect =
                func::CallIndirect::matches(ctx, op) || func::TailCallIndirect::matches(ctx, op);
            let is_tail =
                func::TailCall::matches(ctx, op) || func::TailCallIndirect::matches(ctx, op);
            if is_tail && get_calling_convention(ctx, op) != Some(CallingConvention::Cps) {
                let arguments = ctx.op_operands(op)[usize::from(indirect)..].to_vec();
                if arguments
                    .into_iter()
                    .any(|argument| lowers_to_rc_managed(ctx, argument, type_converter))
                {
                    return Err(OwnershipContractError(
                        "RC proper-tail call lacks exact CPS provenance",
                    ));
                }
                continue;
            }
            if direct || indirect {
                let actions = if direct {
                    let callee = ctx.op(op).attributes.get_symbol("callee").ok_or(
                        OwnershipContractError("direct call lacks exact callee symbol"),
                    )?;
                    let Some(parameter_entries) = entries.get(&callee) else {
                        if is_tail {
                            return Err(OwnershipContractError(
                                "proper-tail direct callee has no trusted local entry contract",
                            ));
                        }
                        continue;
                    };
                    let operands = ctx.op_operands(op);
                    if operands.len() != parameter_entries.len() {
                        return Err(OwnershipContractError(
                            "direct call arity differs from callee entry contract",
                        ));
                    }
                    parameter_entries
                        .iter()
                        .copied()
                        .map(|entry| action_for_entry(entry, is_tail))
                        .collect::<Result<Vec<_>, _>>()?
                } else {
                    if !is_tail {
                        continue;
                    }
                    if get_calling_convention(ctx, op) != Some(CallingConvention::Cps) {
                        return Err(OwnershipContractError(
                            "indirect proper-tail call lacks exact CPS provenance",
                        ));
                    }
                    let signature =
                        get_indirect_call_signature(ctx, op).ok_or(OwnershipContractError(
                            "indirect proper-tail call lacks exact callable signature",
                        ))?;
                    let callable = core::Func::from_type_ref(ctx, signature).ok_or(
                        OwnershipContractError("indirect proper-tail signature is not core.func"),
                    )?;
                    let result = ctx.types.get(callable.r#return(ctx));
                    if result.dialect != Symbol::new("core") || result.name != Symbol::new("nil") {
                        return Err(OwnershipContractError(
                            "indirect proper-tail signature is not physically empty",
                        ));
                    }
                    let args = ctx.op_operands(op).get(1..).unwrap_or_default();
                    if args.len() != callable.params(ctx).len() {
                        return Err(OwnershipContractError(
                            "indirect proper-tail arity differs from exact signature",
                        ));
                    }
                    let parameters = callable.params(ctx).to_vec();
                    parameters
                        .iter()
                        .map(|&ty| {
                            if type_lowers_to_rc_managed(ctx, ty, type_converter) {
                                RcOwnership::Transfer
                            } else {
                                RcOwnership::Plain
                            }
                        })
                        .collect()
                };
                let id = *next_id;
                *next_id += 1;
                pending.push((
                    op,
                    id,
                    TrustedCallContract {
                        actions,
                        is_tail,
                        is_indirect: indirect,
                        direct_callee: direct.then(|| {
                            ctx.op(op)
                                .attributes
                                .get_symbol("callee")
                                .expect("validated direct callee")
                        }),
                        indirect_signature: if indirect {
                            get_indirect_call_signature(ctx, op)
                                .map(|signature| {
                                    lowered_indirect_signature(ctx, signature, type_converter)
                                })
                                .transpose()?
                        } else {
                            None
                        },
                        indirect_physical_closures: if indirect {
                            get_indirect_call_signature(ctx, op)
                                .map(|signature| indirect_physical_closures(ctx, signature))
                                .transpose()?
                        } else {
                            None
                        },
                    },
                ));
            }
            let nested_regions = ctx.op(op).regions.clone();
            for nested in nested_regions {
                collect_call_contracts(ctx, nested, type_converter, entries, next_id, pending)?;
            }
        }
    }
    Ok(())
}

fn action_for_entry(
    entry: RcOwnership,
    is_tail: bool,
) -> Result<RcOwnership, OwnershipContractError> {
    if is_tail {
        return match entry {
            RcOwnership::Plain => Ok(RcOwnership::Plain),
            RcOwnership::Consumed => Ok(RcOwnership::Transfer),
            RcOwnership::Borrowed | RcOwnership::Retained => Err(OwnershipContractError(
                "proper-tail RC parameter is not consumed by its callee",
            )),
            RcOwnership::Acquire | RcOwnership::Transfer => unreachable!("entry mode"),
        };
    }
    Ok(match entry {
        RcOwnership::Plain => RcOwnership::Plain,
        RcOwnership::Borrowed => RcOwnership::Borrowed,
        RcOwnership::Retained => RcOwnership::Retained,
        RcOwnership::Consumed => RcOwnership::Acquire,
        RcOwnership::Acquire | RcOwnership::Transfer => unreachable!("entry mode"),
    })
}

impl TrustedOwnershipSummaries {
    pub fn validated_for_clif(
        &self,
        ctx: &IrContext,
        module_ops: &[OpRef],
    ) -> Result<ValidatedOwnershipContracts, OwnershipContractError> {
        let mut definitions: HashMap<Symbol, Vec<OpRef>> = HashMap::new();
        for &op in module_ops {
            if let Ok(function) = clif::Func::from_op(ctx, op) {
                definitions
                    .entry(function.sym_name(ctx))
                    .or_default()
                    .push(op);
            }
        }

        let summaries = self
            .summaries
            .iter()
            .filter_map(|(&symbol, expected)| {
                let ops = definitions.get(&symbol)?;
                if ops.len() != 1 {
                    return None;
                }
                let function = clif::Func::from_op(ctx, ops[0]).ok()?;
                let parameters = entry_parameters(ctx, function.body(ctx));
                if parameters.len() != expected.len() {
                    return None;
                }
                if expected
                    .iter()
                    .zip(&parameters)
                    .any(|(ownership, &parameter)| {
                        *ownership == ParameterOwnership::Borrowed
                            && !is_anyref_value(ctx, parameter)
                    })
                {
                    return None;
                }
                let Attribute::List(entries) =
                    ctx.op(ops[0]).attributes.get(PARAMETER_OWNERSHIP_ATTR)?
                else {
                    return None;
                };
                let actual: Option<Vec<_>> = entries
                    .iter()
                    .map(|entry| match entry {
                        Attribute::Symbol(value) if *value == Symbol::new("borrowed") => {
                            Some(ParameterOwnership::Borrowed)
                        }
                        Attribute::Symbol(value) if *value == Symbol::new("owned") => {
                            Some(ParameterOwnership::Owned)
                        }
                        _ => None,
                    })
                    .collect();
                let actual = actual?;
                if actual != *expected {
                    return None;
                }
                Some((symbol, actual))
            })
            .collect();

        let mut entry_contracts = HashMap::new();
        for (&symbol, expected) in &self.entry_contracts {
            let physical_closures =
                self.entry_physical_closures
                    .get(&symbol)
                    .ok_or(OwnershipContractError(
                        "trusted physical entry provenance disappeared during func_to_clif",
                    ))?;
            let ops = definitions.get(&symbol).ok_or(OwnershipContractError(
                "trusted function disappeared during func_to_clif",
            ))?;
            if ops.len() != 1 {
                return Err(OwnershipContractError(
                    "trusted function is not unique after func_to_clif",
                ));
            }
            let function = clif::Func::from_op(ctx, ops[0]).map_err(|_| {
                OwnershipContractError("trusted function did not lower to clif.func")
            })?;
            let parameters = entry_parameters(ctx, function.body(ctx));
            if parameters.len() != expected.len() || physical_closures.len() != expected.len() {
                return Err(OwnershipContractError(
                    "parameter entry contract arity changed during func_to_clif",
                ));
            }
            if ctx
                .op(ops[0])
                .attributes
                .get(PARAMETER_ENTRY_OWNERSHIP_ATTR)
                != Some(&RcOwnership::list_attribute(expected))
                || (expected.iter().any(|entry| *entry != RcOwnership::Plain)
                    && (expected.contains(&RcOwnership::Consumed)
                        != is_defined_physical_cps_function(ctx, ops[0])))
                || expected.iter().zip(parameters).zip(physical_closures).any(
                    |((entry, parameter), physical_closure)| match entry {
                        RcOwnership::Plain => is_anyref_value(ctx, parameter),
                        _ if is_anyref_value(ctx, parameter) => false,
                        _ => {
                            physical_closure.is_none()
                                || !is_core_ptr_type(ctx, ctx.value_ty(parameter))
                        }
                    },
                )
            {
                return Err(OwnershipContractError(
                    "parameter entry contract changed during func_to_clif",
                ));
            }
            entry_contracts.insert(symbol, expected.clone());
        }

        let mut call_contracts = HashMap::new();
        let mut seen_ids = HashSet::new();
        for &function_op in module_ops {
            let Ok(function) = clif::Func::from_op(ctx, function_op) else {
                continue;
            };
            collect_validated_call_contracts(
                ctx,
                function.body(ctx),
                &self.call_contracts,
                &entry_contracts,
                &mut seen_ids,
                &mut call_contracts,
            )?;
        }
        if seen_ids.len() != self.call_contracts.len() {
            return Err(OwnershipContractError(
                "trusted call contract disappeared during func_to_clif",
            ));
        }

        Ok(ValidatedOwnershipContracts {
            summaries,
            entry_contracts,
            call_contracts,
        })
    }

    #[cfg(test)]
    pub(crate) fn attach_locally_borrowed_for_tests(ctx: &mut IrContext, module: Module) -> Self {
        let Some(module_block) = module.first_block(ctx) else {
            return Self {
                summaries: HashMap::new(),
                entry_contracts: HashMap::new(),
                entry_physical_closures: HashMap::new(),
                call_contracts: HashMap::new(),
            };
        };
        let mut summaries = HashMap::new();
        let mut entry_contracts = HashMap::new();
        let mut entry_physical_closures = HashMap::new();
        let op_count = ctx.block(module_block).ops.len();
        for index in 0..op_count {
            let op = ctx.block(module_block).ops[index];
            let Ok(function) = clif::Func::from_op(ctx, op) else {
                continue;
            };
            let symbol = function.sym_name(ctx);
            let summary: Vec<_> = entry_parameters(ctx, function.body(ctx))
                .into_iter()
                .map(|parameter| {
                    if is_anyref_value(ctx, parameter) {
                        ParameterOwnership::Borrowed
                    } else {
                        ParameterOwnership::Owned
                    }
                })
                .collect();
            let entries = summary
                .iter()
                .copied()
                .map(ParameterOwnership::as_attribute)
                .collect();
            ctx.op_mut(op).attributes.insert(
                Symbol::new(PARAMETER_OWNERSHIP_ATTR),
                Attribute::List(entries),
            );
            let entry_contract: Vec<_> = entry_parameters(ctx, function.body(ctx))
                .into_iter()
                .map(|parameter| {
                    if is_anyref_value(ctx, parameter) {
                        RcOwnership::Borrowed
                    } else {
                        RcOwnership::Plain
                    }
                })
                .collect();
            ctx.op_mut(op).attributes.insert(
                Symbol::new(PARAMETER_ENTRY_OWNERSHIP_ATTR),
                RcOwnership::list_attribute(&entry_contract),
            );
            entry_contracts.insert(symbol, entry_contract);
            entry_physical_closures.insert(symbol, vec![None; summary.len()]);
            summaries.insert(symbol, summary);
        }
        Self {
            summaries,
            entry_contracts,
            entry_physical_closures,
            call_contracts: HashMap::new(),
        }
    }
}

fn collect_validated_call_contracts(
    ctx: &IrContext,
    region: RegionRef,
    trusted: &HashMap<i128, TrustedCallContract>,
    entry_contracts: &HashMap<Symbol, Vec<RcOwnership>>,
    seen_ids: &mut HashSet<i128>,
    validated: &mut HashMap<OpRef, TrustedCallContract>,
) -> Result<(), OwnershipContractError> {
    for &block in &ctx.region(region).blocks {
        for (index, &op) in ctx.block(block).ops.iter().enumerate() {
            let is_tail =
                clif::ReturnCall::matches(ctx, op) || clif::ReturnCallIndirect::matches(ctx, op);
            let is_indirect =
                clif::CallIndirect::matches(ctx, op) || clif::ReturnCallIndirect::matches(ctx, op);
            if is_tail && index + 1 != ctx.block(block).ops.len() {
                return Err(OwnershipContractError("proper-tail call is not final"));
            }
            let id = match ctx.op(op).attributes.get(OWNERSHIP_CONTRACT_ID_ATTR) {
                None => None,
                Some(Attribute::Int(id)) => Some(*id),
                Some(_) => {
                    return Err(OwnershipContractError(
                        "call ownership contract identity is malformed",
                    ));
                }
            };
            if let Some(id) = id {
                if !seen_ids.insert(id) {
                    return Err(OwnershipContractError(
                        "call ownership contract identity is duplicated",
                    ));
                }
                let expected = trusted.get(&id).ok_or(OwnershipContractError(
                    "call ownership contract identity is untrusted",
                ))?;
                let args = ctx
                    .op_operands(op)
                    .get(usize::from(is_indirect)..)
                    .unwrap_or_default();
                let argument_count = ctx
                    .op_operands(op)
                    .len()
                    .saturating_sub(usize::from(is_indirect));
                let expected_call_family = match (expected.is_tail, expected.is_indirect) {
                    (false, false) => clif::Call::matches(ctx, op),
                    (false, true) => clif::CallIndirect::matches(ctx, op),
                    (true, false) => clif::ReturnCall::matches(ctx, op),
                    (true, true) => clif::ReturnCallIndirect::matches(ctx, op),
                };
                // Direct operands may be ABI-equivalent `core.ptr` values after conversion;
                // the already-validated callee entry contract is the exact parameter type proof.
                let actions_match_types = if !expected.is_tail && !expected.is_indirect {
                    expected
                        .direct_callee
                        .and_then(|callee| entry_contracts.get(&callee))
                        .is_some_and(|entries| {
                            entries.len() == expected.actions.len()
                                && entries
                                    .iter()
                                    .zip(&expected.actions)
                                    .all(|(entry, action)| match action {
                                        RcOwnership::Plain => *entry == RcOwnership::Plain,
                                        RcOwnership::Borrowed
                                        | RcOwnership::Retained
                                        | RcOwnership::Acquire => *entry != RcOwnership::Plain,
                                        RcOwnership::Consumed | RcOwnership::Transfer => false,
                                    })
                        })
                } else if expected.is_tail && expected.is_indirect {
                    expected
                        .indirect_signature
                        .and_then(|signature| core::Func::from_type_ref(ctx, signature))
                        .zip(expected.indirect_physical_closures.as_deref())
                        .is_some_and(|(callable, physical_closures)| {
                            let parameters = callable.params(ctx);
                            parameters.len() == expected.actions.len()
                                && physical_closures.len() == expected.actions.len()
                                && parameters
                                    .iter()
                                    .zip(&expected.actions)
                                    .zip(physical_closures)
                                    .all(|((&parameter, action), physical_closure)| match action {
                                        RcOwnership::Plain => {
                                            !is_anyref_type(ctx, parameter)
                                                && physical_closure.is_none()
                                        }
                                        RcOwnership::Transfer => {
                                            is_anyref_type(ctx, parameter)
                                                || (physical_closure.is_some()
                                                    && is_core_ptr_type(ctx, parameter))
                                        }
                                        _ => false,
                                    })
                        })
                } else {
                    args.iter()
                        .zip(&expected.actions)
                        .all(|(&argument, action)| match (expected.is_tail, action) {
                            (true, RcOwnership::Plain) => !is_anyref_value(ctx, argument),
                            (true, RcOwnership::Transfer) => is_anyref_value(ctx, argument),
                            (false, RcOwnership::Plain) => !is_anyref_value(ctx, argument),
                            (
                                false,
                                RcOwnership::Borrowed
                                | RcOwnership::Retained
                                | RcOwnership::Acquire,
                            ) => is_anyref_value(ctx, argument),
                            _ => false,
                        })
                };
                if !expected_call_family
                    || ctx.op(op).attributes.get(CALL_ARGUMENT_OWNERSHIP_ATTR)
                        != Some(&RcOwnership::list_attribute(&expected.actions))
                    || is_tail != expected.is_tail
                    || is_indirect != expected.is_indirect
                    || expected.direct_callee != ctx.op(op).attributes.get_symbol("callee")
                    || expected.indirect_signature
                        != (is_indirect
                            .then(|| ctx.op(op).attributes.get_type("sig"))
                            .flatten())
                    || expected.actions.len() != argument_count
                    || !actions_match_types
                {
                    return Err(OwnershipContractError(
                        "call ownership contract changed during func_to_clif",
                    ));
                }
                validated.insert(op, expected.clone());
            } else if ctx
                .op(op)
                .attributes
                .contains_key(CALL_ARGUMENT_OWNERSHIP_ATTR)
            {
                return Err(OwnershipContractError(
                    "call ownership actions have no trusted identity",
                ));
            } else if is_tail
                && ctx
                    .op_operands(op)
                    .get(usize::from(is_indirect)..)
                    .unwrap_or_default()
                    .iter()
                    .any(|&argument| is_anyref_value(ctx, argument))
            {
                return Err(OwnershipContractError("proper-tail transfer is untrusted"));
            }
            for &nested in &ctx.op(op).regions {
                collect_validated_call_contracts(
                    ctx,
                    nested,
                    trusted,
                    entry_contracts,
                    seen_ids,
                    validated,
                )?;
            }
        }
    }
    Ok(())
}

fn entry_parameters(ctx: &IrContext, body: RegionRef) -> Vec<ValueRef> {
    ctx.region(body)
        .blocks
        .first()
        .map(|&entry| ctx.block_args(entry).to_vec())
        .unwrap_or_default()
}

fn is_anyref_value(ctx: &IrContext, value: ValueRef) -> bool {
    is_anyref_type(ctx, ctx.value_ty(value))
}

fn is_anyref_type(ctx: &IrContext, ty: TypeRef) -> bool {
    let ty = ctx.types.get(ty);
    ty.dialect == Symbol::new("tribute_rt") && ty.name == Symbol::new("anyref")
}

fn is_core_ptr_type(ctx: &IrContext, ty: TypeRef) -> bool {
    let ty = ctx.types.get(ty);
    ty.dialect == Symbol::new("core") && ty.name == Symbol::new("ptr")
}

fn lowers_to_rc_managed(
    ctx: &mut IrContext,
    value: ValueRef,
    type_converter: &TypeConverter,
) -> bool {
    type_lowers_to_rc_managed(ctx, ctx.value_ty(value), type_converter)
}

fn collect_escaping_function_symbols(
    ctx: &IrContext,
    module_ops: &[OpRef],
    functions: &HashMap<Symbol, OpRef>,
    ineligible: &mut HashSet<Symbol>,
) {
    for &op in module_ops {
        let Ok(function) = func::Func::from_op(ctx, op) else {
            continue;
        };
        collect_escaping_function_symbols_in_region(ctx, function.body(ctx), functions, ineligible);
    }
}

fn collect_escaping_function_symbols_in_region(
    ctx: &IrContext,
    region: RegionRef,
    functions: &HashMap<Symbol, OpRef>,
    ineligible: &mut HashSet<Symbol>,
) {
    for &block in &ctx.region(region).blocks {
        for &op in &ctx.block(block).ops {
            let symbol = if let Ok(constant) = func::Constant::from_op(ctx, op) {
                Some(constant.func_ref(ctx))
            } else if let Ok(tail_call) = func::TailCall::from_op(ctx, op) {
                Some(tail_call.callee(ctx))
            } else {
                None
            };
            if let Some(symbol) = symbol
                && functions.contains_key(&symbol)
            {
                ineligible.insert(symbol);
            }

            if let Ok(nested_function) = func::Func::from_op(ctx, op) {
                collect_escaping_function_symbols_in_region(
                    ctx,
                    nested_function.body(ctx),
                    functions,
                    ineligible,
                );
            } else {
                for &nested in &ctx.op(op).regions {
                    collect_escaping_function_symbols_in_region(ctx, nested, functions, ineligible);
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BorrowedUseKind {
    LoadAddress,
    StoreAddress { address_operand: usize },
    Comparison,
    TransparentAlias,
    DirectCall,
    Escaping,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BorrowedUse {
    Safe,
    TransparentAlias(ValueRef),
    DirectCall,
    Escaping,
}

pub(super) fn classify_borrowed_use(
    ctx: &IrContext,
    body: RegionRef,
    op: OpRef,
    operand_index: usize,
    kind: BorrowedUseKind,
) -> BorrowedUse {
    let Some(parent_block) = ctx.op(op).parent_block else {
        return BorrowedUse::Escaping;
    };
    if ctx.block(parent_block).parent_region != Some(body) {
        return BorrowedUse::Escaping;
    }

    match kind {
        BorrowedUseKind::LoadAddress if operand_index == 0 => BorrowedUse::Safe,
        BorrowedUseKind::StoreAddress { address_operand } if operand_index == address_operand => {
            BorrowedUse::Safe
        }
        BorrowedUseKind::Comparison => BorrowedUse::Safe,
        BorrowedUseKind::TransparentAlias
            if operand_index == 0
                && ctx.op_operands(op).len() == 1
                && ctx.op_results(op).len() == 1 =>
        {
            BorrowedUse::TransparentAlias(ctx.op_results(op)[0])
        }
        BorrowedUseKind::DirectCall => BorrowedUse::DirectCall,
        _ => BorrowedUse::Escaping,
    }
}

fn value_is_borrowed(
    ctx: &IrContext,
    body: RegionRef,
    value: ValueRef,
    summaries: &HashMap<Symbol, Vec<ParameterOwnership>>,
    visited: &mut HashSet<ValueRef>,
) -> bool {
    if !visited.insert(value) {
        return true;
    }
    ctx.uses(value).iter().all(|use_| {
        let op = use_.user;
        let operand_index = use_.operand_index as usize;
        match classify_borrowed_use(ctx, body, op, operand_index, borrowed_use_kind(ctx, op)) {
            BorrowedUse::Safe => true,
            BorrowedUse::TransparentAlias(alias) => {
                value_is_borrowed(ctx, body, alias, summaries, visited)
            }
            BorrowedUse::DirectCall => {
                let call = func::Call::from_op(ctx, op).expect("classified func.call");
                summaries
                    .get(&call.callee(ctx))
                    .and_then(|summary| summary.get(operand_index))
                    == Some(&ParameterOwnership::Borrowed)
            }
            BorrowedUse::Escaping => false,
        }
    })
}

fn borrowed_use_kind(ctx: &IrContext, op: OpRef) -> BorrowedUseKind {
    if func::Call::matches(ctx, op) {
        return BorrowedUseKind::DirectCall;
    }
    if mem::Load::matches(ctx, op) || clif::Load::matches(ctx, op) {
        return BorrowedUseKind::LoadAddress;
    }
    if mem::Store::matches(ctx, op) {
        return BorrowedUseKind::StoreAddress { address_operand: 0 };
    }
    if clif::Store::matches(ctx, op) {
        return BorrowedUseKind::StoreAddress { address_operand: 1 };
    }
    if arith::Cmpi::matches(ctx, op) || clif::Icmp::matches(ctx, op) {
        return BorrowedUseKind::Comparison;
    }
    if core::UnrealizedConversionCast::matches(ctx, op)
        || adt::VariantCast::matches(ctx, op)
        || adt::RefCast::matches(ctx, op)
    {
        return BorrowedUseKind::TransparentAlias;
    }
    if adt::StructGet::matches(ctx, op)
        || adt::VariantIs::matches(ctx, op)
        || adt::VariantGet::matches(ctx, op)
        || adt::ArrayGet::matches(ctx, op)
        || adt::ArrayLen::matches(ctx, op)
        || adt::RefIsNull::matches(ctx, op)
    {
        return BorrowedUseKind::LoadAddress;
    }
    BorrowedUseKind::Escaping
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native::rc_insertion::{BorrowedParameterPolicy, insert_rc_with_trusted_summaries};
    use crate::native::type_converter::native_type_converter;
    use insta::assert_snapshot;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;
    use trunk_ir_cranelift_backend::passes::func_to_clif;

    fn summarize(ir: &str) -> String {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, ir);
        let type_converter = TypeConverter::new();
        compute_and_attach(&mut ctx, module, &type_converter).expect("ownership contracts");
        if let Some(block) = module.first_block(&ctx) {
            let ops = ctx.block(block).ops.clone();
            for op in ops {
                ctx.op_mut(op)
                    .attributes
                    .remove(PARAMETER_ENTRY_OWNERSHIP_ATTR);
            }
        }
        print_module(&ctx, module.op())
            .lines()
            .filter(|line| line.contains(PARAMETER_OWNERSHIP_ATTR))
            .map(str::trim)
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn direct_chain_reaches_borrowed_fixed_point() {
        let summaries = summarize(
            r#"core.module @test {
  func.func @leaf(%0: tribute_rt.anyref) -> core.i32 {
    %1 = mem.load %0 {offset = 0} : core.i32
    func.return %1
  }
  func.func @middle(%0: tribute_rt.anyref) -> core.i32 {
    %1 = func.call %0 {callee = @leaf} : core.i32
    func.return %1
  }
  func.func @root(%0: tribute_rt.anyref) -> core.i32 {
    %1 = func.call %0 {callee = @middle} : core.i32
    func.return %1
  }
}"#,
        );
        assert_snapshot!("trusted_direct_chain_summaries", summaries);
    }

    #[test]
    fn nested_function_references_make_targets_ineligible() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @constant_target(%0: tribute_rt.anyref) -> core.i32 {
    %1 = mem.load %0 {offset = 0} : core.i32
    func.return %1
  }
  func.func @tail_target(%0: tribute_rt.anyref) -> core.nil {
    func.return
  }
  func.func @outer() -> core.nil {
    func.func @nested() -> core.nil {
      %0 = func.constant {func_ref = @constant_target} : core.ptr
      func.tail_call {callee = @tail_target}
    }
    func.return
  }
}"#,
        );
        let summaries = compute_and_attach(&mut ctx, module, &TypeConverter::new())
            .expect("ownership contracts");

        assert_eq!(
            summaries.summaries.get(&Symbol::new("constant_target")),
            Some(&vec![ParameterOwnership::Owned])
        );
        assert_eq!(
            summaries.summaries.get(&Symbol::new("tail_target")),
            Some(&vec![ParameterOwnership::Owned])
        );
    }

    #[test]
    fn cycles_unknown_indirect_and_external_are_owned() {
        let summaries = summarize(
            r#"core.module @test {
  func.func @left(%0: tribute_rt.anyref) -> core.nil {
    %1 = func.call %0 {callee = @right} : core.nil
    func.return
  }
  func.func @right(%0: tribute_rt.anyref) -> core.nil {
    %1 = func.call %0 {callee = @left} : core.nil
    func.return
  }
  func.func @unknown(%0: tribute_rt.anyref) -> core.nil {
    %1 = func.call %0 {callee = @missing} : core.nil
    func.return
  }
  func.func @indirect(%0: tribute_rt.anyref, %1: core.ptr) -> core.nil {
    %2 = func.call_indirect %1, %0 : core.nil
    func.return
  }
  func.func @external(%0: tribute_rt.anyref) -> core.nil attributes {abi = "C"} {
    func.unreachable
  }
}"#,
        );
        assert_snapshot!("untrusted_call_summaries", summaries);
    }

    fn lower_func_ops_to_clif(ctx: &mut IrContext, region: RegionRef) {
        let blocks = ctx.region(region).blocks.to_vec();
        for block in blocks {
            let ops = ctx.block(block).ops.to_vec();
            for op in ops {
                let regions = ctx.op(op).regions.to_vec();
                if func::Func::matches(ctx, op)
                    || func::Call::matches(ctx, op)
                    || func::Return::matches(ctx, op)
                {
                    ctx.op_mut(op).dialect = Symbol::new("clif");
                }
                for nested in regions {
                    lower_func_ops_to_clif(ctx, nested);
                }
            }
        }
    }

    fn lowered_ordinary_call_contract()
    -> (IrContext, Module, TrustedOwnershipSummaries, OpRef, i128) {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @target(%0: tribute_rt.anyref, %1: core.i32) -> core.i32 {
    %2 = clif.load %0 {offset = 0} : core.i32
    func.return %1
  }
  func.func @caller(%0: tribute_rt.anyref, %1: core.i32) -> core.i32 {
    %2 = func.call %0, %1 {callee = @target} : core.i32
    func.return %2
  }
}"#,
        );
        let (type_converter, _) = native_type_converter(&mut ctx);
        let trusted =
            compute_and_attach(&mut ctx, module, &type_converter).expect("ordinary call contract");
        func_to_clif::lower(&mut ctx, module, type_converter).expect("func_to_clif");
        let module_block = module.first_block(&ctx).expect("module body");
        let caller =
            clif::Func::from_op(&ctx, ctx.block(module_block).ops[1]).expect("lowered caller");
        let call = ctx
            .region(caller.body(&ctx))
            .blocks
            .iter()
            .flat_map(|&block| ctx.block(block).ops.iter().copied())
            .find(|&op| clif::Call::matches(&ctx, op))
            .expect("ordinary direct call");
        let id = match ctx.op(call).attributes.get(OWNERSHIP_CONTRACT_ID_ATTR) {
            Some(Attribute::Int(id)) => *id,
            _ => panic!("trusted contract identity"),
        };
        (ctx, module, trusted, call, id)
    }

    fn assert_rc_rejects_unchanged(
        ctx: &mut IrContext,
        module: Module,
        trusted: &TrustedOwnershipSummaries,
    ) {
        let before = print_module(ctx, module.op());
        assert!(
            insert_rc_with_trusted_summaries(
                ctx,
                module,
                BorrowedParameterPolicy::ElideProvenBorrowed,
                trusted,
            )
            .is_err()
        );
        assert_eq!(print_module(ctx, module.op()), before);
    }

    #[test]
    fn ordinary_direct_contract_on_non_call_fails_before_mutation() {
        let (mut ctx, module, trusted, call, _) = lowered_ordinary_call_contract();
        ctx.op_mut(call).name = Symbol::new("load");

        assert_rc_rejects_unchanged(&mut ctx, module, &trusted);
    }

    fn lowered_indirect_tail_contract_with(
        ir: &str,
    ) -> (IrContext, Module, TrustedOwnershipSummaries, OpRef, i128) {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, ir);
        let (type_converter, _) = native_type_converter(&mut ctx);
        let trusted =
            compute_and_attach(&mut ctx, module, &type_converter).expect("indirect tail contract");
        func_to_clif::lower(&mut ctx, module, type_converter).expect("func_to_clif");
        let module_block = module.first_block(&ctx).expect("module body");
        let caller =
            clif::Func::from_op(&ctx, ctx.block(module_block).ops[0]).expect("lowered caller");
        let call = ctx
            .region(caller.body(&ctx))
            .blocks
            .iter()
            .flat_map(|&block| ctx.block(block).ops.iter().copied())
            .find(|&op| clif::ReturnCallIndirect::matches(&ctx, op))
            .expect("indirect tail call");
        let id = match ctx.op(call).attributes.get(OWNERSHIP_CONTRACT_ID_ATTR) {
            Some(Attribute::Int(id)) => *id,
            _ => panic!("trusted contract identity"),
        };
        (ctx, module, trusted, call, id)
    }

    fn lowered_indirect_tail_contract()
    -> (IrContext, Module, TrustedOwnershipSummaries, OpRef, i128) {
        lowered_indirect_tail_contract_with(
            r#"core.module @test {
  func.func @caller(%0: core.ptr, %1: tribute_rt.intref) -> core.nil attributes {tribute.calling_convention = 2} {
    func.tail_call_indirect %0, %1 {func.indirect_call_signature = core.func(core.nil, tribute_rt.intref), tribute.calling_convention = 2}
  }
}"#,
        )
    }

    #[test]
    fn indirect_tail_contract_uses_the_converted_callable_signature() {
        let (mut ctx, module, trusted, call, _) = lowered_indirect_tail_contract();
        let signature = ctx
            .op(call)
            .attributes
            .get_type("sig")
            .expect("lowered indirect signature");
        let callable = core::Func::from_type_ref(&ctx, signature).expect("core.func signature");
        assert!(callable.params(&ctx).iter().all(|&parameter| {
            let ty = ctx.types.get(parameter);
            ty.dialect == Symbol::new("core") && ty.name == Symbol::new("ptr")
        }));

        insert_rc_with_trusted_summaries(
            &mut ctx,
            module,
            BorrowedParameterPolicy::ElideProvenBorrowed,
            &trusted,
        )
        .expect("converted indirect tail contract");
    }

    #[test]
    fn indirect_tail_transfer_accepts_an_adapted_ptr_for_an_anyref_signature() {
        let (mut ctx, module, trusted, call, _) = lowered_indirect_tail_contract_with(
            r#"core.module @test {
  func.func @caller(%0: core.ptr, %1: core.ptr) -> core.nil attributes {tribute.calling_convention = 2} {
    %2 = core.unrealized_conversion_cast %1 : core.ptr
    func.tail_call_indirect %0, %2 {func.indirect_call_signature = core.func(core.nil, tribute_rt.anyref), tribute.calling_convention = 2}
  }
}"#,
        );
        let signature = ctx
            .op(call)
            .attributes
            .get_type("sig")
            .expect("lowered indirect signature");
        let callable = core::Func::from_type_ref(&ctx, signature).expect("core.func signature");
        assert!(is_anyref_type(&ctx, callable.params(&ctx)[0]));
        assert!(!is_anyref_value(&ctx, ctx.op_operands(call)[1]));

        insert_rc_with_trusted_summaries(
            &mut ctx,
            module,
            BorrowedParameterPolicy::ElideProvenBorrowed,
            &trusted,
        )
        .expect("adapted indirect tail transfer");
    }

    #[test]
    fn indirect_tail_transfer_accepts_a_proven_closure_ptr_signature() {
        let (mut ctx, module, trusted, call, id) = lowered_indirect_tail_contract_with(
            r#"core.module @test {
  !handler = closure.closure(core.func(core.nil)) {tribute.calling_convention = 2, tribute.closure_environment_index = 0}
  func.func @caller(%0: core.ptr, %1: !handler) -> core.nil attributes {tribute.calling_convention = 2} {
    func.tail_call_indirect %0, %1 {func.indirect_call_signature = core.func(core.nil, !handler), tribute.calling_convention = 2}
  }
}"#,
        );
        let contract = trusted.call_contracts.get(&id).expect("trusted call");
        assert_eq!(contract.actions, [RcOwnership::Transfer]);
        let signature = ctx
            .op(call)
            .attributes
            .get_type("sig")
            .expect("lowered indirect signature");
        let callable = core::Func::from_type_ref(&ctx, signature).expect("core.func signature");
        assert!(is_core_ptr_type(&ctx, callable.params(&ctx)[0]));

        insert_rc_with_trusted_summaries(
            &mut ctx,
            module,
            BorrowedParameterPolicy::ElideProvenBorrowed,
            &trusted,
        )
        .expect("proven closure indirect tail transfer");
    }

    #[test]
    fn indirect_tail_closure_transfer_without_provenance_fails_closed() {
        let (mut ctx, module, mut trusted, _call, id) = lowered_indirect_tail_contract_with(
            r#"core.module @test {
  !handler = closure.closure(core.func(core.nil)) {tribute.calling_convention = 2, tribute.closure_environment_index = 0}
  func.func @caller(%0: core.ptr, %1: !handler) -> core.nil attributes {tribute.calling_convention = 2} {
    func.tail_call_indirect %0, %1 {func.indirect_call_signature = core.func(core.nil, !handler), tribute.calling_convention = 2}
  }
}"#,
        );
        trusted
            .call_contracts
            .get_mut(&id)
            .expect("trusted call")
            .indirect_physical_closures = None;

        assert_rc_rejects_unchanged(&mut ctx, module, &trusted);
    }

    #[test]
    fn indirect_tail_plain_signature_cannot_masquerade_as_transfer() {
        let (mut ctx, module, mut trusted, call, id) = lowered_indirect_tail_contract();
        let actions = {
            let contract = trusted.call_contracts.get_mut(&id).expect("trusted call");
            contract.actions[0] = RcOwnership::Transfer;
            contract.actions.clone()
        };
        ctx.op_mut(call).attributes.insert(
            Symbol::new(CALL_ARGUMENT_OWNERSHIP_ATTR),
            RcOwnership::list_attribute(&actions),
        );

        assert_rc_rejects_unchanged(&mut ctx, module, &trusted);
    }

    #[test]
    fn indirect_tail_signature_mutation_fails_before_rc_mutation() {
        let (mut ctx, module, trusted, call, _) = lowered_indirect_tail_contract();
        let nil = core::nil(&mut ctx).as_type_ref();
        let malformed = core::func(&mut ctx, nil, [nil]).as_type_ref();
        ctx.op_mut(call)
            .attributes
            .insert(Symbol::new("sig"), Attribute::Type(malformed));

        assert_rc_rejects_unchanged(&mut ctx, module, &trusted);
    }

    #[test]
    fn ordinary_direct_action_type_mismatch_fails_before_mutation() {
        for (argument_index, action) in [
            (0, RcOwnership::Plain),
            (1, RcOwnership::Borrowed),
            (1, RcOwnership::Retained),
            (1, RcOwnership::Acquire),
            (0, RcOwnership::Consumed),
            (0, RcOwnership::Transfer),
        ] {
            let (mut ctx, module, mut trusted, call, id) = lowered_ordinary_call_contract();
            let contract = trusted.call_contracts.get_mut(&id).expect("trusted call");
            contract.actions[argument_index] = action;
            ctx.op_mut(call).attributes.insert(
                Symbol::new(CALL_ARGUMENT_OWNERSHIP_ATTR),
                RcOwnership::list_attribute(&contract.actions),
            );

            assert_rc_rejects_unchanged(&mut ctx, module, &trusted);
        }
    }

    fn rc_after_metadata_mutation(mutate: impl FnOnce(&mut IrContext, OpRef)) -> String {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @leaf(%0: tribute_rt.anyref) -> core.i32 {
    %1 = clif.load %0 {offset = 0} : core.i32
    func.return %1
  }
  func.func @caller(%0: tribute_rt.anyref) -> core.i32 {
    %1 = func.call %0 {callee = @leaf} : core.i32
    func.return %1
  }
}"#,
        );
        let type_converter = TypeConverter::new();
        let trusted =
            compute_and_attach(&mut ctx, module, &type_converter).expect("ownership contracts");
        let module_block = module.first_block(&ctx).expect("module body");
        let caller = ctx.block(module_block).ops[1];
        mutate(&mut ctx, caller);
        let body = module.body(&ctx).expect("module region");
        lower_func_ops_to_clif(&mut ctx, body);
        insert_rc_with_trusted_summaries(
            &mut ctx,
            module,
            BorrowedParameterPolicy::ElideProvenBorrowed,
            &trusted,
        )
        .expect("RC insertion");
        print_module(&ctx, module.op())
            .lines()
            .filter(|line| line.contains("tribute_rt.retain"))
            .map(str::trim)
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn missing_metadata_fails_closed() {
        let retains = rc_after_metadata_mutation(|ctx, caller| {
            ctx.op_mut(caller)
                .attributes
                .remove(PARAMETER_OWNERSHIP_ATTR);
        });
        assert_eq!(retains.matches("tribute_rt.retain").count(), 1, "{retains}");
    }

    #[test]
    fn inconsistent_metadata_fails_closed() {
        let retains = rc_after_metadata_mutation(|ctx, caller| {
            ctx.op_mut(caller).attributes.insert(
                Symbol::new(PARAMETER_OWNERSHIP_ATTR),
                Attribute::List(vec![Attribute::Symbol(Symbol::new("owned"))]),
            );
        });
        assert_eq!(retains.matches("tribute_rt.retain").count(), 1, "{retains}");
    }

    #[test]
    fn cps_entry_and_direct_indirect_edge_contracts_are_positional() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @target(%0: tribute_rt.anyref, %1: core.i32) -> core.nil attributes {tribute.calling_convention = 2} {
    func.unreachable
  }
  func.func @direct(%0: tribute_rt.anyref, %1: core.i32) -> core.nil attributes {tribute.calling_convention = 2} {
    func.tail_call %0, %1 {callee = @target, tribute.calling_convention = 2}
  }
  func.func @indirect(%0: core.ptr, %1: tribute_rt.anyref, %2: core.i32) -> core.nil attributes {tribute.calling_convention = 2} {
    func.tail_call_indirect %0, %1, %2 {func.indirect_call_signature = core.func(core.nil, tribute_rt.anyref, core.i32), tribute.calling_convention = 2}
  }
}"#,
        );
        let trusted = compute_and_attach(&mut ctx, module, &TypeConverter::new())
            .expect("exact CPS contracts");

        assert_eq!(
            trusted.entry_contracts[&Symbol::new("target")],
            [RcOwnership::Consumed, RcOwnership::Plain]
        );
        assert_eq!(trusted.call_contracts.len(), 2);
        assert!(trusted.call_contracts.values().all(|contract| {
            contract.actions == [RcOwnership::Transfer, RcOwnership::Plain] && contract.is_tail
        }));
        assert_eq!(
            trusted.summaries[&Symbol::new("target")],
            [ParameterOwnership::Owned, ParameterOwnership::Owned]
        );
    }

    #[test]
    fn missing_direct_or_indirect_provenance_fails_before_mutation() {
        for ir in [
            r#"core.module @test {
  func.func @caller(%0: tribute_rt.anyref) -> core.nil attributes {tribute.calling_convention = 2} {
    func.tail_call %0 {callee = @missing, tribute.calling_convention = 2}
  }
}"#,
            r#"core.module @test {
  func.func @caller(%0: core.ptr, %1: tribute_rt.anyref) -> core.nil attributes {tribute.calling_convention = 2} {
    func.tail_call_indirect %0, %1 {tribute.calling_convention = 2}
  }
}"#,
            r#"core.module @test {
  func.func @target(%0: tribute_rt.anyref) -> core.nil {
    func.unreachable
  }
  func.func @caller(%0: tribute_rt.anyref) -> core.nil {
    func.tail_call %0 {callee = @target}
  }
}"#,
        ] {
            let mut ctx = IrContext::new();
            let module = parse_test_module(&mut ctx, ir);
            let before = print_module(&ctx, module.op());
            assert!(compute_and_attach(&mut ctx, module, &TypeConverter::new()).is_err());
            assert_eq!(print_module(&ctx, module.op()), before);
        }
    }
}
