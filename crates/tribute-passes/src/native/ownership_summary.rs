use std::collections::{HashMap, HashSet};

use trunk_ir::context::IrContext;
use trunk_ir::dialect::{adt, arith, clif, core, func, mem};
use trunk_ir::ops::DialectOp;
use trunk_ir::rewrite::{Module, TypeConverter};
use trunk_ir::transforms::call_graph::{build_call_graph, recursive_functions};
use trunk_ir::{Attribute, OpRef, RegionRef, Symbol, ValueRef};

pub const PARAMETER_OWNERSHIP_ATTR: &str = "tribute.rc.parameter_ownership_v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParameterOwnership {
    Borrowed,
    Owned,
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
}

pub fn compute_and_attach(
    ctx: &mut IrContext,
    module: Module,
    type_converter: &TypeConverter,
) -> TrustedOwnershipSummaries {
    let Some(module_block) = module.first_block(ctx) else {
        return TrustedOwnershipSummaries {
            summaries: HashMap::new(),
        };
    };
    let module_ops = ctx.block(module_block).ops.to_vec();
    let mut definitions: HashMap<Symbol, Vec<OpRef>> = HashMap::new();
    for &op in &module_ops {
        ctx.op_mut(op).attributes.remove(PARAMETER_OWNERSHIP_ATTR);
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
                if !ineligible.contains(&symbol) && lowers_to_anyref(ctx, parameter, type_converter)
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
    }

    TrustedOwnershipSummaries { summaries }
}

impl TrustedOwnershipSummaries {
    pub fn validated_for_clif(
        &self,
        ctx: &IrContext,
        module_ops: &[OpRef],
    ) -> HashMap<Symbol, Vec<ParameterOwnership>> {
        let mut definitions: HashMap<Symbol, Vec<OpRef>> = HashMap::new();
        for &op in module_ops {
            if let Ok(function) = clif::Func::from_op(ctx, op) {
                definitions
                    .entry(function.sym_name(ctx))
                    .or_default()
                    .push(op);
            }
        }

        self.summaries
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
            .collect()
    }

    #[cfg(test)]
    pub(crate) fn attach_locally_borrowed_for_tests(ctx: &mut IrContext, module: Module) -> Self {
        let Some(module_block) = module.first_block(ctx) else {
            return Self {
                summaries: HashMap::new(),
            };
        };
        let mut summaries = HashMap::new();
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
            summaries.insert(symbol, summary);
        }
        Self { summaries }
    }
}

fn entry_parameters(ctx: &IrContext, body: RegionRef) -> Vec<ValueRef> {
    ctx.region(body)
        .blocks
        .first()
        .map(|&entry| ctx.block_args(entry).to_vec())
        .unwrap_or_default()
}

fn is_anyref_value(ctx: &IrContext, value: ValueRef) -> bool {
    let ty = ctx.types.get(ctx.value_ty(value));
    ty.dialect == Symbol::new("tribute_rt") && ty.name == Symbol::new("anyref")
}

fn lowers_to_anyref(ctx: &mut IrContext, value: ValueRef, type_converter: &TypeConverter) -> bool {
    let original = ctx.value_ty(value);
    let converted = type_converter.convert_type_or_identity(ctx, original);
    let ty = ctx.types.get(converted);
    ty.dialect == Symbol::new("tribute_rt") && ty.name == Symbol::new("anyref")
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
        visit_region_ops(ctx, function.body(ctx), &mut |nested| {
            let symbol = if let Ok(constant) = func::Constant::from_op(ctx, nested) {
                Some(constant.func_ref(ctx))
            } else if let Ok(tail_call) = func::TailCall::from_op(ctx, nested) {
                Some(tail_call.callee(ctx))
            } else {
                None
            };
            if let Some(symbol) = symbol
                && functions.contains_key(&symbol)
            {
                ineligible.insert(symbol);
            }
        });
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
        let Some(parent_block) = ctx.op(op).parent_block else {
            return false;
        };
        if ctx.block(parent_block).parent_region != Some(body) {
            return false;
        }
        let operand_index = use_.operand_index as usize;
        if let Ok(call) = func::Call::from_op(ctx, op) {
            return summaries
                .get(&call.callee(ctx))
                .and_then(|summary| summary.get(operand_index))
                == Some(&ParameterOwnership::Borrowed);
        }
        if is_transparent_alias(ctx, op, operand_index) {
            return ctx.op_results(op).len() == 1
                && value_is_borrowed(ctx, body, ctx.op_results(op)[0], summaries, visited);
        }
        is_local_borrowed_use(ctx, op, operand_index)
    })
}

fn is_transparent_alias(ctx: &IrContext, op: OpRef, operand_index: usize) -> bool {
    operand_index == 0
        && (core::UnrealizedConversionCast::matches(ctx, op)
            || adt::VariantCast::matches(ctx, op)
            || adt::RefCast::matches(ctx, op))
}

fn is_local_borrowed_use(ctx: &IrContext, op: OpRef, operand_index: usize) -> bool {
    if mem::Load::matches(ctx, op) || clif::Load::matches(ctx, op) {
        return operand_index == 0;
    }
    if mem::Store::matches(ctx, op) {
        return operand_index == 0;
    }
    if clif::Store::matches(ctx, op) {
        return operand_index == 1;
    }
    if arith::Cmpi::matches(ctx, op) || clif::Icmp::matches(ctx, op) {
        return true;
    }
    operand_index == 0
        && (adt::StructGet::matches(ctx, op)
            || adt::VariantIs::matches(ctx, op)
            || adt::VariantGet::matches(ctx, op)
            || adt::ArrayGet::matches(ctx, op)
            || adt::ArrayLen::matches(ctx, op)
            || adt::RefIsNull::matches(ctx, op))
}

fn visit_region_ops(ctx: &IrContext, region: RegionRef, visitor: &mut impl FnMut(OpRef)) {
    for &block in &ctx.region(region).blocks {
        for &op in &ctx.block(block).ops {
            visitor(op);
            if !func::Func::matches(ctx, op) {
                for &nested in &ctx.op(op).regions {
                    visit_region_ops(ctx, nested, visitor);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native::rc_insertion::{BorrowedParameterPolicy, insert_rc_with_trusted_summaries};
    use insta::assert_snapshot;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;

    fn summarize(ir: &str) -> String {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, ir);
        let type_converter = TypeConverter::new();
        compute_and_attach(&mut ctx, module, &type_converter);
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
        let trusted = compute_and_attach(&mut ctx, module, &type_converter);
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
        );
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
}
