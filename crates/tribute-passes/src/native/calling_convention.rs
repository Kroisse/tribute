//! Project Tribute callable conventions onto the generic Cranelift ABI.
//!
//! The generic Cranelift backend consumes only `clif.calling_convention`.
//! This boundary is the last place that understands the proven Tribute CPS
//! convention, so it validates that every proper-tail transfer has an exact
//! CPS target before attaching generic backend metadata.

use std::collections::HashMap;
use std::ops::ControlFlow;

use tribute_core::{CALLING_CONVENTION_ATTR, CallingConvention, get_calling_convention};
use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::{clif, core, func};
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::OpRef;
use trunk_ir::rewrite::Module;
use trunk_ir::types::Attribute;
use trunk_ir::walk::{WalkAction, walk_op};

use crate::target_abi::TargetAbiError;

#[derive(Clone, Copy)]
struct FunctionConvention {
    convention: CallingConvention,
    imported: bool,
}

/// Validate Tribute's final callable conventions and attach generic Cranelift
/// ABI metadata without exposing Tribute-specific attributes to the backend.
///
/// Direct calls use the callee declaration's Cranelift signature.  Indirect
/// calls carry their signature convention explicitly because no declaration is
/// available once a closure has become a function pointer.
pub fn project_to_clif(ctx: &mut IrContext, module: Module) -> Result<(), TargetAbiError> {
    let ops = collect_ops(ctx, module.op());
    let mut functions = HashMap::new();
    let mut function_ops = Vec::new();
    let mut indirect_ops = Vec::new();
    let mut direct_tail_ops = Vec::new();

    for &op in &ops {
        let Ok(function) = func::Func::from_op(ctx, op) else {
            continue;
        };
        let imported = ctx.op(op).attributes.contains_key("abi");
        let convention = checked_convention(ctx, op)?.unwrap_or(CallingConvention::Direct);
        if imported && convention == CallingConvention::Cps {
            return Err(TargetAbiError::new(format!(
                "native calling convention: imported function `{}` cannot use the CPS Tail ABI",
                function.sym_name(ctx)
            )));
        }
        let scope = symbol_scope(ctx, op)?;
        if functions
            .insert(
                (scope, function.sym_name(ctx)),
                FunctionConvention {
                    convention,
                    imported,
                },
            )
            .is_some()
        {
            return Err(TargetAbiError::new(format!(
                "native calling convention: duplicate function symbol `{}` in one module scope",
                function.sym_name(ctx)
            )));
        }
        function_ops.push((op, convention));
    }

    for &op in &ops {
        if func::TailCall::from_op(ctx, op).is_ok() {
            let convention = require_cps_transfer(ctx, op, "direct proper-tail transfer")?;
            debug_assert_eq!(convention, CallingConvention::Cps);
            let callee = ctx.op(op).attributes.get_symbol("callee").ok_or_else(|| {
                TargetAbiError::new(
                    "native calling convention: direct proper-tail transfer has no callee",
                )
            })?;
            let scope = symbol_scope(ctx, op)?;
            let target = functions.get(&(scope, callee)).ok_or_else(|| {
                TargetAbiError::new(format!(
                    "native calling convention: direct proper-tail transfer targets unknown `{callee}`"
                ))
            })?;
            if target.imported || target.convention != CallingConvention::Cps {
                return Err(TargetAbiError::new(format!(
                    "native calling convention: direct proper-tail transfer target `{callee}` is not a proven CPS worker"
                )));
            }
            direct_tail_ops.push(op);
            continue;
        }

        if func::TailCallIndirect::from_op(ctx, op).is_ok() {
            require_cps_transfer(ctx, op, "indirect proper-tail transfer")?;
            if !ctx.op_result_types(op).is_empty() {
                return Err(TargetAbiError::new(
                    "native calling convention: indirect proper-tail transfer must have no results",
                ));
            }
            indirect_ops.push((op, CallingConvention::Cps));
            continue;
        }

        if func::CallIndirect::from_op(ctx, op).is_ok() {
            let convention = checked_convention(ctx, op)?.unwrap_or(CallingConvention::Direct);
            if convention == CallingConvention::Cps {
                return Err(TargetAbiError::new(
                    "native calling convention: CPS indirect call must be a proper-tail transfer",
                ));
            }
            indirect_ops.push((op, convention));
            continue;
        }

        if func::Call::from_op(ctx, op).is_ok() {
            // A direct call's FuncRef refers to the separately declared callee
            // signature.  Still reject malformed source metadata here rather
            // than allowing it to disappear during generic lowering.
            let _ = checked_convention(ctx, op)?;
        }
    }

    // Apply only after every function and transfer was validated.
    for (op, convention) in function_ops {
        set_clif_convention(ctx, op, convention);
    }
    for op in direct_tail_ops {
        set_clif_convention(ctx, op, CallingConvention::Cps);
    }
    for (op, convention) in indirect_ops {
        set_clif_convention(ctx, op, convention);
    }
    Ok(())
}

fn require_cps_transfer(
    ctx: &IrContext,
    op: OpRef,
    transfer: &str,
) -> Result<CallingConvention, TargetAbiError> {
    let convention = checked_convention(ctx, op)?.ok_or_else(|| {
        TargetAbiError::new(format!(
            "native calling convention: {transfer} lacks Tribute convention provenance"
        ))
    })?;
    if convention != CallingConvention::Cps {
        return Err(TargetAbiError::new(format!(
            "native calling convention: {transfer} must use the proven CPS convention"
        )));
    }
    Ok(convention)
}

fn checked_convention(
    ctx: &IrContext,
    op: OpRef,
) -> Result<Option<CallingConvention>, TargetAbiError> {
    if !ctx.op(op).attributes.contains_key(CALLING_CONVENTION_ATTR) {
        return Ok(None);
    }
    get_calling_convention(ctx, op).ok_or_else(|| {
        TargetAbiError::new(
            "native calling convention: operation has an invalid tribute.calling_convention attribute",
        )
    }).map(Some)
}

fn set_clif_convention(ctx: &mut IrContext, op: OpRef, convention: CallingConvention) {
    let value = match convention {
        CallingConvention::Cps => clif::CALLING_CONVENTION_TAIL,
        CallingConvention::Direct | CallingConvention::EvidenceDirect => {
            clif::CALLING_CONVENTION_PLATFORM
        }
    };
    ctx.op_mut(op).attributes.insert(
        Symbol::new(clif::CALLING_CONVENTION_ATTR),
        Attribute::Symbol(Symbol::new(value)),
    );
}

fn collect_ops(ctx: &IrContext, root: OpRef) -> Vec<OpRef> {
    let mut ops = Vec::new();
    let _ = walk_op::<()>(ctx, root, &mut |op| {
        ops.push(op);
        ControlFlow::Continue(WalkAction::Advance)
    });
    ops
}

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
        "native calling convention: operation has no owning core.module symbol scope",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;

    fn clif_convention(ctx: &IrContext, op: OpRef) -> Option<Symbol> {
        ctx.op(op)
            .attributes
            .get_symbol(clif::CALLING_CONVENTION_ATTR)
    }

    fn function(ctx: &IrContext, module: Module, name: &str) -> OpRef {
        let symbol = Symbol::from_dynamic(name);
        module
            .ops(ctx)
            .into_iter()
            .find(|&op| {
                func::Func::from_op(ctx, op).is_ok_and(|function| function.sym_name(ctx) == symbol)
            })
            .unwrap_or_else(|| panic!("missing function `{name}`"))
    }

    #[test]
    fn projects_tail_only_for_proven_cps_workers_and_transfers() {
        let input = r#"core.module @test {
  !cps = closure.closure(core.func(core.nil, core.i32)) {tribute.calling_convention = 2}
  func.func @direct(%value: core.i32) -> core.nil attributes {tribute.calling_convention = 0} {
    %result = func.call_indirect %value, %value {tribute.calling_convention = 0} : core.i32
    func.return
  }
  func.func @worker(%value: core.i32) -> core.nil attributes {tribute.calling_convention = 2} {
    func.tail_call %value {callee = @done, tribute.calling_convention = 2}
  }
  func.func @done(%value: core.i32) -> core.nil attributes {tribute.calling_convention = 2} {
    func.return
  }
  func.func @indirect(%callee: !cps, %value: core.i32) -> core.nil attributes {tribute.calling_convention = 2} {
    func.tail_call_indirect %callee, %value {tribute.calling_convention = 2}
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);
        project_to_clif(&mut ctx, module).unwrap();
        let output = print_module(&ctx, module.op());
        assert_eq!(
            clif_convention(&ctx, function(&ctx, module, "direct")),
            Some(Symbol::new(clif::CALLING_CONVENTION_PLATFORM))
        );
        assert_eq!(
            clif_convention(&ctx, function(&ctx, module, "worker")),
            Some(Symbol::new(clif::CALLING_CONVENTION_TAIL))
        );
        let tail_ops: Vec<_> = collect_ops(&ctx, module.op())
            .into_iter()
            .filter(|&op| {
                func::TailCall::from_op(&ctx, op).is_ok()
                    || func::TailCallIndirect::from_op(&ctx, op).is_ok()
            })
            .collect();
        assert_eq!(tail_ops.len(), 2);
        assert!(tail_ops.iter().all(|&op| {
            clif_convention(&ctx, op) == Some(Symbol::new(clif::CALLING_CONVENTION_TAIL))
        }));
        assert!(output.contains("func.call_indirect %0, %0 {"));
        assert!(output.contains("clif.calling_convention = @platform"));
    }

    #[test]
    fn rejects_unproven_tail_transfer_without_mutating() {
        let input = r#"core.module @test {
  func.func @worker() -> core.nil attributes {tribute.calling_convention = 2} {
    func.tail_call {callee = @worker}
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);
        let before = print_module(&ctx, module.op());
        let error = project_to_clif(&mut ctx, module).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("lacks Tribute convention provenance")
        );
        assert_eq!(print_module(&ctx, module.op()), before);
    }

    #[test]
    fn rejects_malformed_convention_without_mutating() {
        let input = r#"core.module @test {
  func.func @worker() -> core.nil attributes {tribute.calling_convention = 99} {
    func.return
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);
        let before = print_module(&ctx, module.op());
        let error = project_to_clif(&mut ctx, module).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("invalid tribute.calling_convention")
        );
        assert_eq!(print_module(&ctx, module.op()), before);
    }

    #[test]
    fn preserves_root_markers_while_projecting_worker_and_wrapper_abis() {
        let input = r#"core.module @test {
  func.func @__tribute_cps_main() -> core.nil attributes {tribute.calling_convention = 2, tribute.root_cps_worker = true} {
    func.tail_call {callee = @__tribute_root_done_k, tribute.calling_convention = 2}
  }
  func.func @__tribute_root_done_k() -> core.nil attributes {tribute.calling_convention = 2, tribute.root_done_k = true} {
    func.return
  }
  func.func @main() -> core.nil attributes {tribute.calling_convention = 0, tribute.root_wrapper = true} {
    func.call {callee = @__tribute_cps_main, tribute.calling_convention = 2, tribute.root_cps_call = true} : core.nil
    func.return
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);
        project_to_clif(&mut ctx, module).unwrap();

        let worker = function(&ctx, module, "__tribute_cps_main");
        let done = function(&ctx, module, "__tribute_root_done_k");
        let wrapper = function(&ctx, module, "main");
        assert_eq!(
            clif_convention(&ctx, worker),
            Some(Symbol::new(clif::CALLING_CONVENTION_TAIL))
        );
        assert_eq!(
            clif_convention(&ctx, done),
            Some(Symbol::new(clif::CALLING_CONVENTION_TAIL))
        );
        assert_eq!(
            clif_convention(&ctx, wrapper),
            Some(Symbol::new(clif::CALLING_CONVENTION_PLATFORM))
        );
        let root_call = collect_ops(&ctx, wrapper)
            .into_iter()
            .find(|&op| ctx.op(op).attributes.contains_key("tribute.root_cps_call"))
            .expect("root wrapper must retain its marked ordinary worker call");
        assert!(clif_convention(&ctx, root_call).is_none());
    }
}
