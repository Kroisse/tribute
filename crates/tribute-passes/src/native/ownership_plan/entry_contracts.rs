//! Cached native parameter-entry contracts.

use std::sync::OnceLock;

use trunk_ir::analysis::{Analysis, AnalysisContext, AnalysisError};
use trunk_ir::transforms::call_graph::{CallGraph, recursive_functions};

use super::*;
use crate::target_abi::{CONSUMED, OWNERSHIP_ATTR};

type Contracts = HashMap<SymbolPath, Vec<EntryOwnership>>;

/// Parameter-entry contracts of every `func.func` in a module.
///
/// The target is a `core.module` op. The analysis proves which managed
/// parameters of defined functions are only borrowed; whether a planner
/// relies on that proof is a policy it applies by selecting a view, so the
/// policy never takes part in the cached computation.
pub struct NativeEntryContracts {
    proven: Contracts,
    /// Functions with a body. Only their borrowed entries are proven rather
    /// than declared.
    defined: HashSet<SymbolPath>,
    retained: OnceLock<Contracts>,
}

impl NativeEntryContracts {
    /// Select the contracts for the borrowed-parameter policy without
    /// recomputing the proof.
    pub fn view(&self, elide_proven_borrowed_parameters: bool) -> &Contracts {
        if elide_proven_borrowed_parameters {
            return &self.proven;
        }
        self.retained.get_or_init(|| {
            let mut contracts = self.proven.clone();
            for (symbol, entries) in &mut contracts {
                if !self.defined.contains(symbol) {
                    continue;
                }
                for entry in entries {
                    if *entry == EntryOwnership::Borrowed {
                        *entry = EntryOwnership::Retained;
                    }
                }
            }
            contracts
        })
    }
}

impl Analysis for NativeEntryContracts {
    fn compute(ctx: &mut AnalysisContext<'_>, target: OpRef) -> Result<Self, AnalysisError> {
        let module = ctx.get::<NativeOwnershipModuleFacts>(target)?;
        let call_graph = ctx.get::<CallGraph>(target)?;
        compute_entry_contracts(
            ctx.ir(),
            &call_graph,
            module.definitions(),
            module.managed_layouts(),
        )
        .map_err(|error| AnalysisError::new::<Self>(target, error))
    }
}

fn compute_entry_contracts(
    ctx: &IrContext,
    call_graph: &CallGraph,
    definitions: &HashMap<SymbolPath, OpRef>,
    managed_layouts: &HashSet<TypeRef>,
) -> Result<NativeEntryContracts, OwnershipPlanError> {
    let recursive = recursive_functions(call_graph);
    let mut summaries = HashMap::default();
    let mut defined = HashSet::default();
    for (symbol, &op) in definitions {
        let symbol = symbol.clone();
        let entry = match ownership_callable_body(ctx, op)? {
            CallableBody::Declaration => {
                let signature = validate_bodyless_signature(ctx, op, managed_layouts)?;
                let entries = bodyless_c_entry_contract(ctx, signature, op, managed_layouts)
                    .unwrap_or_else(|| vec![EntryOwnership::Plain; signature.inputs(ctx).len()]);
                summaries.insert(symbol, entries);
                continue;
            }
            CallableBody::Definition { entry, .. } => entry,
        };
        let ineligible = recursive.contains(&symbol) || ctx.op(op).attributes.contains_key("abi");
        let signature = ctx
            .op(op)
            .attributes
            .get_type("type")
            .and_then(|ty| func::FuncSig::from_type_ref(ctx, ty))
            .ok_or_else(|| OwnershipPlanError::new("function definition lacks exact signature"))?;
        let consumed = consumed_inputs(ctx, signature)?;
        if consumed.len() != ctx.block_args(entry).len() {
            return Err(OwnershipPlanError::new(
                "function entry arity differs from its exact signature",
            ));
        }
        defined.insert(symbol.clone());
        summaries.insert(
            symbol,
            ctx.block_args(entry)
                .iter()
                .zip(consumed)
                .map(|(&parameter, consumed)| {
                    if !is_managed_value(ctx, parameter, managed_layouts) {
                        EntryOwnership::Plain
                    } else if consumed {
                        EntryOwnership::Consumed
                    } else if ineligible {
                        EntryOwnership::Retained
                    } else {
                        EntryOwnership::Borrowed
                    }
                })
                .collect::<Vec<_>>(),
        );
    }

    loop {
        let mut changed = false;
        for (symbol, &op) in definitions {
            let CallableBody::Definition {
                region: body,
                entry,
            } = ownership_callable_body(ctx, op)?
            else {
                continue;
            };
            for (index, &parameter) in ctx.block_args(entry).iter().enumerate() {
                if summaries[symbol][index] == EntryOwnership::Borrowed
                    && !value_is_borrowed(ctx, body, parameter, &summaries, &mut HashSet::default())
                {
                    summaries.get_mut(symbol).unwrap()[index] = EntryOwnership::Retained;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    Ok(NativeEntryContracts {
        proven: summaries,
        defined,
        retained: OnceLock::new(),
    })
}

/// Native `extern "C"` declarations are the one trusted bodyless managed
/// boundary. Their logical managed arguments are borrowed for the call; a
/// logical managed result is a fresh owned value, as for every call result.
fn bodyless_c_entry_contract(
    ctx: &IrContext,
    signature: func::FuncSig,
    op: OpRef,
    managed_layouts: &HashSet<TypeRef>,
) -> Option<Vec<EntryOwnership>> {
    if ctx.op(op).attributes.get_str(ctx, "abi") != Some("C") {
        return None;
    }
    Some(
        signature
            .inputs(ctx)
            .iter()
            .map(|&ty| {
                if is_typed_managed_reference(ctx, ty, managed_layouts) {
                    EntryOwnership::Borrowed
                } else {
                    EntryOwnership::Plain
                }
            })
            .collect(),
    )
}

/// Which inputs of `signature` carry the `consumed` entry contract that the
/// representation/ABI boundary records in the exact physical signature.
///
/// The marker is inert on unmanaged inputs; callers combine it with the typed
/// managed-reference contract. Any other ownership value is rejected.
pub(super) fn consumed_inputs(
    ctx: &IrContext,
    signature: func::FuncSig,
) -> Result<Vec<bool>, OwnershipPlanError> {
    signature
        .input_attrs(ctx)
        .map(|attrs| match attrs.get(OWNERSHIP_ATTR) {
            None => Ok(false),
            Some(Attribute::String(mode)) if ctx.str(*mode) == CONSUMED => Ok(true),
            Some(_) => Err(OwnershipPlanError::new(format!(
                "unknown {OWNERSHIP_ATTR} parameter contract"
            ))),
        })
        .collect()
}

fn value_is_borrowed(
    ctx: &IrContext,
    body: RegionRef,
    value: ValueRef,
    summaries: &Contracts,
    visiting: &mut HashSet<ValueRef>,
) -> bool {
    if !visiting.insert(value) {
        return true;
    }
    ctx.uses(value).iter().all(|use_| {
        let op = use_.user;
        if ctx
            .op(op)
            .parent_block
            .is_none_or(|block| ctx.block(block).parent_region != Some(body))
        {
            return false;
        }
        let index = use_.operand_index as usize;
        if (adt::StructGet::matches(ctx, op)
            || adt::VariantGet::matches(ctx, op)
            || adt::VariantIs::matches(ctx, op)
            || adt::RefIsNull::matches(ctx, op))
            && index == 0
        {
            return true;
        }
        if (adt::RefCast::matches(ctx, op) || core::UnrealizedConversionCast::matches(ctx, op))
            && index == 0
            && ctx.op_results(op).len() == 1
        {
            return value_is_borrowed(ctx, body, ctx.op_result(op, 0), summaries, visiting);
        }
        if let Ok(call) = func::Call::from_op(ctx, op) {
            return summaries
                .get(call.callee(ctx))
                .and_then(|entries| entries.get(index))
                == Some(&EntryOwnership::Borrowed);
        }
        false
    })
}

fn validate_bodyless_signature(
    ctx: &IrContext,
    op: OpRef,
    managed_layouts: &HashSet<TypeRef>,
) -> Result<func::FuncSig, OwnershipPlanError> {
    let signature = ctx
        .op(op)
        .attributes
        .get_type("type")
        .and_then(|ty| func::FuncSig::from_type_ref(ctx, ty))
        .ok_or_else(|| OwnershipPlanError::new("bodyless function lacks exact signature"))?;
    if ctx.op(op).attributes.get_str(ctx, "abi") == Some("C") {
        return Ok(signature);
    }
    if signature
        .inputs(ctx)
        .iter()
        .chain(signature.results(ctx))
        .any(|&ty| is_typed_managed_reference(ctx, ty, managed_layouts))
    {
        return Err(OwnershipPlanError::new(
            "bodyless native declaration exposes a managed reference",
        ));
    }
    Ok(signature)
}
