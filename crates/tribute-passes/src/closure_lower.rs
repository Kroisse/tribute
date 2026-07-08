//! Lower closure operations in indirect calls.
//!
//! This pass transforms `func.call_indirect` operations when the callee
//! is a closure:
//!
//! Before:
//! ```text
//! %closure = closure.new @lifted_func, %env
//! %result = func.call_indirect %closure, %args...
//! ```
//!
//! After:
//! ```text
//! %closure = closure.new @lifted_func, %env
//! %funcref = closure.func %closure
//! %env = closure.env %closure
//! %result = func.call_indirect %funcref, %env, %args...
//! ```
//!
//! Uses `RewritePattern` + `PatternApplicator` for declarative transformation.

use tribute_ir::dialect::closure;
use tribute_ir::dialect::tribute_rt;
use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::adt;
use trunk_ir::dialect::core;
use trunk_ir::dialect::func;
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::pass::{Pass, PassRunResult};
use trunk_ir::refs::{OpRef, TypeRef, ValueRef};
use trunk_ir::rewrite::{
    ConversionTarget, Module, PatternApplicator, PatternRewriter, RewritePattern, TypeConverter,
};
use trunk_ir::types::{Attribute, TypeDataBuilder};

/// Create the unified closure struct type in arena: `{ table_idx: i32, env: anyref }`.
pub fn closure_struct_type_ref(ctx: &mut IrContext) -> TypeRef {
    let i32_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());
    let anyref_ty = tribute_rt::anyref(ctx).as_type_ref();
    ctx.types.intern(
        TypeDataBuilder::new(Symbol::new("adt"), Symbol::new("struct"))
            .param(i32_ty)
            .param(anyref_ty)
            .attr("name", Attribute::Symbol(Symbol::new("_closure")))
            .build(),
    )
}

/// Check if a TypeRef is an adt.struct with name "_closure".
fn is_closure_struct_type_ref(ctx: &IrContext, ty: TypeRef) -> bool {
    let data = ctx.types.get(ty);
    if data.dialect != Symbol::new("adt") || data.name != Symbol::new("struct") {
        return false;
    }
    matches!(
        data.attrs.get(&Symbol::new("name")),
        Some(Attribute::Symbol(s)) if *s == Symbol::new("_closure")
    )
}

// ============================================================================
// Rewrite Patterns
// ============================================================================

/// Update function signatures to convert `core.func` params to `closure.closure`.
struct UpdateFuncSignatureArena;

impl RewritePattern for UpdateFuncSignatureArena {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(func_op) = func::Func::from_op(ctx, op) else {
            return false;
        };

        let func_ty = func_op.r#type(ctx);
        let func_data = ctx.types.get(func_ty);
        if func_data.dialect != Symbol::new("core") || func_data.name != Symbol::new("func") {
            return false;
        }

        // params[0] = return, params[1..] = param types
        if func_data.params.is_empty() {
            return false;
        }

        // Clone data we need before mutating ctx.types
        let params: Vec<TypeRef> = func_data.params.to_vec();

        let mut needs_update = false;
        let mut new_params = Vec::with_capacity(params.len());
        new_params.push(params[0]); // return type

        for &param_ty in &params[1..] {
            if core::Func::matches(ctx, param_ty) {
                // Convert core.func to closure.closure wrapping the func type
                let closure_ty = closure::closure(ctx, param_ty).as_type_ref();
                new_params.push(closure_ty);
                needs_update = true;
            } else {
                new_params.push(param_ty);
            }
        }

        if !needs_update {
            return false;
        }

        // Build new func type
        let return_ty = new_params[0];
        let new_func_ty = core::func(ctx, return_ty, new_params[1..].iter().copied()).as_type_ref();

        // Rebuild the function with new type
        let func_name = func_op.sym_name(ctx);
        let body = func_op.body(ctx);
        let loc = ctx.op(op).location;
        if let Some(entry) = ctx.region(body).blocks.first().copied() {
            let arg_count = ctx.block_args(entry).len();
            for (idx, &new_ty) in new_params[1..].iter().enumerate().take(arg_count) {
                ctx.set_block_arg_type(entry, idx as u32, new_ty);
            }
        }
        ctx.detach_region(body);
        let new_op = func::func(ctx, loc, func_name, new_func_ty, body).op_ref();
        rewriter.replace_op(new_op);
        true
    }

    fn name(&self) -> &'static str {
        "UpdateFuncSignatureArena"
    }
}

/// Lower `closure.new` to `func.constant` + `adt.struct_new`.
struct LowerClosureNewArena;

impl RewritePattern for LowerClosureNewArena {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(closure_new) = closure::New::from_op(ctx, op) else {
            return false;
        };

        let loc = ctx.op(op).location;
        let func_ref = closure_new.func_ref(ctx);
        let env = closure_new.env(ctx);

        // Extract function type from closure.closure result type
        let result_ty = ctx.op_result_types(op)[0];
        let func_ty = closure::Closure::from_type_ref(ctx, result_ty)
            .map(|c| c.func_type(ctx))
            .expect("closure.new result type must contain a valid func type (from func.constant)");

        // Generate: %funcref = func.constant @func_ref : func_type
        let constant_op = func::constant(ctx, loc, func_ty, func_ref);
        let funcref = ctx.op_result(constant_op.op_ref(), 0);

        // Generate: %closure = adt.struct_new(%funcref, %env) : closure_struct_type
        let struct_ty = closure_struct_type_ref(ctx);
        let struct_new_op = adt::struct_new(ctx, loc, vec![funcref, env], struct_ty, struct_ty);

        rewriter.insert_op(constant_op.op_ref());
        rewriter.replace_op(struct_new_op.op_ref());
        true
    }

    fn name(&self) -> &'static str {
        "LowerClosureNewArena"
    }
}

/// Lower `func.call_indirect` on closure values.
struct LowerClosureCallArena {
    evidence_from_param: Option<ValueRef>,
}

impl RewritePattern for LowerClosureCallArena {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if func::CallIndirect::from_op(ctx, op).is_err() {
            return false;
        }

        let operands = ctx.op_operands(op);
        if operands.is_empty() {
            return false;
        }
        let callee = operands[0];
        let callee_ty = ctx.value_ty(callee);

        // Determine if callee is a closure
        let callee_is_closure = if closure::Closure::matches(ctx, callee_ty) {
            true
        } else if is_closure_struct_type_ref(ctx, callee_ty) {
            // Already lowered closure struct
            true
        } else if core::Func::matches(ctx, callee_ty) {
            // core.func in func.call_indirect is always a closure value.
            // Direct function calls use func.call; call_indirect operates on
            // closure values (block args, env captures, etc.).
            true
        } else {
            // Fallback: check if result of closure.new
            if let trunk_ir::refs::ValueDef::OpResult(def_op, _) = ctx.value_def(callee) {
                closure::New::from_op(ctx, def_op).is_ok()
            } else {
                false
            }
        };

        if !callee_is_closure {
            return false;
        }

        let loc = ctx.op(op).location;
        let args: Vec<ValueRef> = operands[1..].to_vec();
        let caller_result_ty = ctx.op_result_types(op)[0];

        let i32_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());
        let anyref_ty = tribute_rt::anyref(ctx).as_type_ref();

        // Determine actual return type from the closure's func type.
        // Effectful lambdas may return anyref even if the caller's
        // declared type says otherwise (e.g., Nat).
        let callee_return_ty =
            extract_return_type_from_callee(ctx, callee_ty).unwrap_or(caller_result_ty);

        // Generate: %table_idx = closure.func %closure
        let table_idx_op = closure::func(ctx, loc, callee, i32_ty);
        let table_idx = ctx.op_result(table_idx_op.op_ref(), 0);

        // Generate: %env = closure.env %closure
        let env_op = closure::env(ctx, loc, callee, anyref_ty);
        let env = ctx.op_result(env_op.op_ref(), 0);

        let evidence = if let Some(evidence) = self.evidence_from_param {
            evidence
        } else {
            let evidence_ty = tribute_ir::dialect::ability::evidence_adt_type_ref(ctx);
            let null_op = adt::ref_null(ctx, loc, evidence_ty, evidence_ty);
            rewriter.insert_op(null_op.op_ref());
            null_op.result(ctx)
        };

        // Generate: %result = func.call_indirect %table_idx, [%evidence, %env, %args...]
        let mut new_args = vec![evidence, env];
        new_args.extend(args);
        let new_call = func::call_indirect(ctx, loc, table_idx, new_args, callee_return_ty);

        rewriter.insert_op(table_idx_op.op_ref());
        rewriter.insert_op(env_op.op_ref());

        // If the closure's return type differs from the caller's expected type,
        // insert a cast so downstream code sees the expected type.
        if callee_return_ty != caller_result_ty {
            let cast =
                core::unrealized_conversion_cast(ctx, loc, new_call.result(ctx), caller_result_ty);
            rewriter.insert_op(new_call.op_ref());
            rewriter.replace_op(cast.op_ref());
        } else {
            rewriter.replace_op(new_call.op_ref());
        }
        true
    }

    fn name(&self) -> &'static str {
        "LowerClosureCallArena"
    }
}

/// Lower `closure.func` to `adt.struct_get` field 0.
struct LowerClosureFuncArena;

impl RewritePattern for LowerClosureFuncArena {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if closure::Func::from_op(ctx, op).is_err() {
            return false;
        }

        let loc = ctx.op(op).location;
        let closure_value = ctx.op_operands(op)[0];
        let i32_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());
        let struct_ty = closure_struct_type_ref(ctx);

        let get_op = adt::struct_get(ctx, loc, closure_value, i32_ty, struct_ty, 0);
        rewriter.replace_op(get_op.op_ref());
        true
    }

    fn name(&self) -> &'static str {
        "LowerClosureFuncArena"
    }
}

/// Lower `closure.env` to `adt.struct_get` field 1.
struct LowerClosureEnvArena;

impl RewritePattern for LowerClosureEnvArena {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if closure::Env::from_op(ctx, op).is_err() {
            return false;
        }

        let loc = ctx.op(op).location;
        let closure_value = ctx.op_operands(op)[0];
        let result_ty = ctx.op_result_types(op)[0];
        let struct_ty = closure_struct_type_ref(ctx);

        let get_op = adt::struct_get(ctx, loc, closure_value, result_ty, struct_ty, 1);
        rewriter.replace_op(get_op.op_ref());
        true
    }

    fn name(&self) -> &'static str {
        "LowerClosureEnvArena"
    }
}

/// Extract the return type from a callee type (closure.closure or core.func).
fn extract_return_type_from_callee(ctx: &IrContext, callee_ty: TypeRef) -> Option<TypeRef> {
    let data = ctx.types.get(callee_ty);
    // closure.closure<core.func<Return, Params...>>
    if data.dialect == Symbol::new("closure") && data.name == Symbol::new("closure") {
        let func_ty = *data.params.first()?;
        let func_data = ctx.types.get(func_ty);
        if func_data.dialect == Symbol::new("core") && func_data.name == Symbol::new("func") {
            return func_data.params.first().copied();
        }
    }
    // core.func<Return, Params...>
    if data.dialect == Symbol::new("core") && data.name == Symbol::new("func") {
        return data.params.first().copied();
    }
    None
}

fn evidence_param_for_func(ctx: &IrContext, func_op: func::Func) -> Option<ValueRef> {
    let func_ty = func_op.r#type(ctx);
    if !crate::evidence::has_evidence_first_param(ctx, func_ty) {
        return None;
    }

    let body = func_op.body(ctx);
    let entry = ctx.region(body).blocks.first().copied()?;
    let evidence = *ctx.block_args(entry).first()?;
    tribute_ir::dialect::ability::is_evidence_type_ref(ctx, ctx.value_ty(evidence))
        .then_some(evidence)
}

/// Lower closures using arena IR.
///
/// This compatibility entry point prepares module-level function signatures,
/// then lowers each function body independently.
pub(crate) fn lower_closures(ctx: &mut IrContext, module: Module) {
    prepare_closure_lowering(ctx, module);

    for op in module.ops(ctx) {
        let Ok(func_op) = func::Func::from_op(ctx, op) else {
            continue;
        };
        lower_closures_in_func(ctx, func_op);
    }
}

/// Prepare module-level closure lowering state.
///
/// This updates function signatures (`core.func` params → `closure.closure`) and
/// remains module-scoped because function signatures are interprocedural
/// contracts.
pub(crate) fn prepare_closure_lowering(ctx: &mut IrContext, module: Module) {
    let applicator =
        PatternApplicator::new(TypeConverter::new()).add_pattern(UpdateFuncSignatureArena);
    applicator.apply_partial(ctx, module);
}

/// Lower closure operations in one function body.
///
/// Closure calls receive the enclosing function's evidence parameter directly
/// while the call is rewritten. Pure functions synthesize null evidence at the
/// call site, avoiding a module-wide span-based post-processing pass.
pub(crate) fn lower_closures_in_func(ctx: &mut IrContext, func_op: func::Func) {
    let evidence_from_param = evidence_param_for_func(ctx, func_op);
    let applicator = PatternApplicator::new(TypeConverter::new())
        .with_target(
            ConversionTarget::new()
                .legal_op("func", "func")
                .recursive_legal_op("func", "func"),
        )
        .add_pattern(LowerClosureCallArena {
            evidence_from_param,
        })
        .add_pattern(LowerClosureNewArena)
        .add_pattern(LowerClosureFuncArena)
        .add_pattern(LowerClosureEnvArena);
    applicator.apply_partial(ctx, func_op);
}

/// PassManager-friendly wrapper for [`lower_closures`].
pub struct LowerClosures;

impl Pass for LowerClosures {
    type Target = core::Module;

    fn name(&self) -> &'static str {
        "lower-closures"
    }

    fn run(&mut self, ctx: &mut IrContext, target: core::Module) -> PassRunResult {
        lower_closures(ctx, target.into());
        Ok(())
    }
}

/// PassManager-friendly module preparation for closure lowering.
pub struct PrepareClosureLowering;

impl Pass for PrepareClosureLowering {
    type Target = core::Module;

    fn name(&self) -> &'static str {
        "prepare-closure-lowering"
    }

    fn run(&mut self, ctx: &mut IrContext, target: core::Module) -> PassRunResult {
        prepare_closure_lowering(ctx, target.into());
        Ok(())
    }
}

/// PassManager-friendly function-local closure lowering pass.
pub struct LowerClosuresInFunc;

impl Pass for LowerClosuresInFunc {
    type Target = func::Func;

    fn name(&self) -> &'static str {
        "lower-closures-in-func"
    }

    fn run(&mut self, ctx: &mut IrContext, target: func::Func) -> PassRunResult {
        lower_closures_in_func(ctx, target);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ops::ControlFlow;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;
    use trunk_ir::walk::{WalkAction, walk_op};

    fn evidence_type_str() -> &'static str {
        "core.array(adt.struct() {fields = [[@ability_id, core.i32], [@prompt_tag, core.i32], [@tr_dispatch_fn, core.ptr], [@handler_dispatch, core.ptr]], name = @_Marker})"
    }

    fn closure_test_module(ctx: &mut IrContext) -> Module {
        let ev_ty = evidence_type_str();
        parse_test_module(
            ctx,
            &format!(
                r#"core.module @test {{
  !closure = closure.closure(core.func(tribute_rt.anyref, tribute_rt.anyref))

  func.func @callee(%ev: {ev_ty}, %env: tribute_rt.anyref, %arg: tribute_rt.anyref) -> tribute_rt.anyref {{
      func.return %arg
  }}

  func.func @selected(%ev: {ev_ty}, %payload: tribute_rt.anyref) -> tribute_rt.anyref {{
      %env = adt.ref_null {{type = tribute_rt.anyref}} : tribute_rt.anyref
      %closure = closure.new %env {{func_ref = @callee}} : !closure
      %result = func.call_indirect %closure, %payload : tribute_rt.anyref
      func.return %result
  }}

  func.func @untouched(%ev: {ev_ty}, %payload: tribute_rt.anyref) -> tribute_rt.anyref {{
      %env = adt.ref_null {{type = tribute_rt.anyref}} : tribute_rt.anyref
      %closure = closure.new %env {{func_ref = @callee}} : !closure
      %result = func.call_indirect %closure, %payload : tribute_rt.anyref
      func.return %result
  }}
}}"#
            ),
        )
    }

    fn func_by_name(ctx: &IrContext, module: Module, name: &'static str) -> func::Func {
        let name = Symbol::new(name);
        module
            .ops(ctx)
            .into_iter()
            .filter_map(|op| func::Func::from_op(ctx, op).ok())
            .find(|func_op| func_op.sym_name(ctx) == name)
            .expect("test function should exist")
    }

    fn func_by_name_recursive(ctx: &IrContext, module: Module, name: &'static str) -> func::Func {
        let name = Symbol::new(name);
        let mut found = None;
        let _ = walk_op::<()>(ctx, module.op(), &mut |op| {
            if let Ok(func_op) = func::Func::from_op(ctx, op)
                && func_op.sym_name(ctx) == name
            {
                found = Some(func_op);
                return ControlFlow::Break(());
            }
            ControlFlow::Continue(WalkAction::Advance)
        });
        found.expect("test function should exist")
    }

    fn call_indirect_operands_in_func(ctx: &IrContext, func_op: func::Func) -> Vec<Vec<ValueRef>> {
        let mut calls = Vec::new();
        for &block in &ctx.region(func_op.body(ctx)).blocks {
            for &op in &ctx.block(block).ops {
                if func::CallIndirect::from_op(ctx, op).is_ok() {
                    calls.push(ctx.op_operands(op).to_vec());
                }
            }
        }
        calls
    }

    fn entry_evidence_arg(ctx: &IrContext, func_op: func::Func) -> ValueRef {
        let entry = ctx.region(func_op.body(ctx)).blocks[0];
        ctx.block_args(entry)[0]
    }

    fn nested_closure_test_module(ctx: &mut IrContext) -> Module {
        let ev_ty = evidence_type_str();
        parse_test_module(
            ctx,
            &format!(
                r#"core.module @test {{
  !closure = closure.closure(core.func(tribute_rt.anyref, tribute_rt.anyref))

  func.func @callee(%ev: {ev_ty}, %env: tribute_rt.anyref, %arg: tribute_rt.anyref) -> tribute_rt.anyref {{
      func.return %arg
  }}

  func.func @outer(%outer_ev: {ev_ty}, %payload: tribute_rt.anyref) -> tribute_rt.anyref {{
      func.func @inner(%inner_ev: {ev_ty}, %inner_payload: tribute_rt.anyref) -> tribute_rt.anyref {{
          %env = adt.ref_null {{type = tribute_rt.anyref}} : tribute_rt.anyref
          %closure = closure.new %env {{func_ref = @callee}} : !closure
          %result = func.call_indirect %closure, %inner_payload : tribute_rt.anyref
          func.return %result
      }}
      func.return %payload
  }}
}}"#
            ),
        )
    }

    #[test]
    fn prepare_pass_adapter_updates_function_signatures() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @apply(%f: core.func(tribute_rt.anyref, tribute_rt.anyref), %arg: tribute_rt.anyref) -> tribute_rt.anyref {
      %result = func.call_indirect %f, %arg : tribute_rt.anyref
      func.return %result
  }
}"#,
        );

        let mut pass = PrepareClosureLowering;
        let core_module = core::Module::from_op(&ctx, module.op()).unwrap();
        pass.run(&mut ctx, core_module).unwrap();

        let ir = print_module(&ctx, module.op());
        assert!(
            ir.contains("closure.closure(core.func(tribute_rt.anyref, tribute_rt.anyref))"),
            "function-typed params should be prepared as closure params:\n{ir}"
        );
    }

    #[test]
    fn function_pass_rewrites_only_selected_function_and_uses_evidence_param() {
        let mut ctx = IrContext::new();
        let module = closure_test_module(&mut ctx);
        let selected = func_by_name(&ctx, module, "selected");

        let mut pass = LowerClosuresInFunc;
        pass.run(&mut ctx, selected).unwrap();

        let selected_calls = call_indirect_operands_in_func(&ctx, selected);
        assert_eq!(selected_calls.len(), 1);
        let selected_operands = &selected_calls[0];
        assert!(
            selected_operands.len() >= 3,
            "lowered closure call should have table index, evidence, and env operands"
        );
        assert_eq!(
            selected_operands[1],
            entry_evidence_arg(&ctx, selected),
            "lowered closure call should pass the enclosing function's evidence argument immediately after table index"
        );

        let untouched = func_by_name(&ctx, module, "untouched");
        let untouched_ir = print_module(&ctx, untouched.op_ref());
        assert!(
            untouched_ir.contains("closure.new") && untouched_ir.contains("func.call_indirect"),
            "function-local pass should not rewrite other functions:\n{untouched_ir}"
        );
    }

    #[test]
    fn module_entrypoint_still_prepares_and_lowers_all_functions() {
        let mut ctx = IrContext::new();
        let module = closure_test_module(&mut ctx);

        lower_closures(&mut ctx, module);

        let ir = print_module(&ctx, module.op());
        assert!(
            !ir.contains("closure.new"),
            "module entrypoint should lower closure.new:\n{ir}"
        );
        assert!(
            !ir.contains("closure.func") && !ir.contains("closure.env"),
            "module entrypoint should lower closure accessors:\n{ir}"
        );

        for name in ["selected", "untouched"] {
            let func_op = func_by_name(&ctx, module, name);
            let calls = call_indirect_operands_in_func(&ctx, func_op);
            assert_eq!(
                calls.len(),
                1,
                "{name} should have one lowered indirect call"
            );
            assert_eq!(
                calls[0][1],
                entry_evidence_arg(&ctx, func_op),
                "{name} should pass the enclosing function's evidence argument immediately after table index"
            );
        }
    }

    #[test]
    fn function_pass_leaves_nested_func_for_own_evidence_processing() {
        let mut ctx = IrContext::new();
        let module = nested_closure_test_module(&mut ctx);
        let outer = func_by_name_recursive(&ctx, module, "outer");

        lower_closures_in_func(&mut ctx, outer);

        let inner = func_by_name_recursive(&ctx, module, "inner");
        let inner_after_outer = print_module(&ctx, inner.op_ref());
        assert!(
            inner_after_outer.contains("closure.new"),
            "outer function pass should not lower nested function body:\n{inner_after_outer}"
        );

        lower_closures_in_func(&mut ctx, inner);

        let inner_calls = call_indirect_operands_in_func(&ctx, inner);
        assert_eq!(inner_calls.len(), 1);
        assert_eq!(
            inner_calls[0][1],
            entry_evidence_arg(&ctx, inner),
            "nested function pass should use the nested function's own evidence argument"
        );
    }
}
