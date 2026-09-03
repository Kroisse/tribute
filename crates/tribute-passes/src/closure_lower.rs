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

use std::collections::{HashMap, HashSet};
use std::ops::ControlFlow;

use tribute_core::calling_convention::{
    CLOSURE_CALLABLE_TYPE_ATTR, get_physical_closure_environment_index,
};
use tribute_core::{
    CallableAbi, CallingConvention, get_calling_convention, get_closure_callable_type,
    get_physical_closure_convention, set_closure_callable_type,
};
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
use trunk_ir::walk::{WalkAction, walk_op, walk_region};

/// Create the unified closure struct type in arena: `{ table_idx: i32, env: anyref }`.
pub fn closure_struct_type_ref(ctx: &mut IrContext) -> TypeRef {
    let i32_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());
    let anyref_ty = tribute_rt::anyref(ctx).as_type_ref();
    ctx.types.intern(
        TypeDataBuilder::new(Symbol::new("adt"), Symbol::new("struct"))
            .attr("name", Attribute::Symbol(Symbol::new("_closure")))
            .attr(
                "fields",
                Attribute::List(vec![
                    Attribute::List(vec![
                        Attribute::Symbol(Symbol::new("func_ptr")),
                        Attribute::Type(i32_ty),
                    ]),
                    Attribute::List(vec![
                        Attribute::Symbol(Symbol::new("env")),
                        Attribute::Type(anyref_ty),
                    ]),
                ]),
            )
            .build(),
    )
}

/// Check if a TypeRef is an adt.struct with name "_closure".
pub(crate) fn is_closure_struct_type_ref(ctx: &IrContext, ty: TypeRef) -> bool {
    let data = ctx.types.get(ty);
    if data.dialect != Symbol::new("adt") || data.name != Symbol::new("struct") {
        return false;
    }
    data.attrs.get_symbol("name") == Some(Symbol::new("_closure"))
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
        if get_physical_closure_convention(ctx, result_ty).is_some() {
            set_closure_callable_type(ctx, struct_new_op.op_ref(), result_ty);
        }

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
    legacy_evidence: Option<ValueRef>,
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
            !matches!(
                ctx.value_def(callee),
                trunk_ir::refs::ValueDef::OpResult(defining_op, _)
                    if func::Constant::from_op(ctx, defining_op).is_ok()
            )
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
        let exact_contract = if physical_closure_type_for_callee(ctx, callee).is_some() {
            let Some(convention) = get_calling_convention(ctx, op) else {
                return false;
            };
            let Some(contract) = exact_physical_call_contract(
                ctx,
                callee,
                convention,
                &args,
                caller_result_ty,
                anyref_ty,
            ) else {
                return false;
            };
            Some(contract)
        } else {
            None
        };

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

        let new_args = if let Some(contract) = &exact_contract {
            let mut args = args;
            for &(index, expected) in &contract.argument_casts {
                let cast = core::unrealized_conversion_cast(ctx, loc, args[index], expected);
                rewriter.insert_op(cast.op_ref());
                args[index] = cast.result(ctx);
            }
            args.insert(contract.environment_index, env);
            args
        } else if let Some(convention) = get_calling_convention(ctx, op) {
            let hidden =
                usize::from(convention.needs_evidence()) + usize::from(convention.needs_done_k());
            let Some(source_args) = args.get(hidden..) else {
                return false;
            };
            let abi = CallableAbi::new(convention, source_args.iter().copied(), env);
            abi.interpose_environment(&args, env)
        } else {
            // Compatibility for hand-written legacy IR without metadata.
            let evidence = if let Some(evidence) = self.legacy_evidence {
                evidence
            } else {
                let evidence_ty = tribute_ir::dialect::ability::evidence_adt_type_ref(ctx);
                let null_op = adt::ref_null(ctx, loc, evidence_ty, evidence_ty);
                rewriter.insert_op(null_op.op_ref());
                null_op.result(ctx)
            };
            let mut legacy = vec![evidence, env];
            legacy.extend_from_slice(&args);
            legacy
        };
        let signature = exact_contract
            .as_ref()
            .map(|contract| contract.signature)
            .unwrap_or_else(|| {
                core::func(
                    ctx,
                    callee_return_ty,
                    new_args
                        .iter()
                        .map(|&value| ctx.value_ty(value))
                        .collect::<Vec<_>>(),
                )
                .as_type_ref()
            });
        let new_call = func::call_indirect(
            ctx,
            loc,
            table_idx,
            new_args,
            callee_return_ty,
            Some(signature),
        );
        if exact_contract.is_some() {
            copy_indirect_call_attributes(ctx, op, new_call.op_ref());
        }

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

/// Lower a convention-proven closure-valued proper tail transfer. Untagged
/// legacy closure values remain on their existing route.
struct LowerClosureTailCallArena;

impl RewritePattern for LowerClosureTailCallArena {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if func::TailCallIndirect::from_op(ctx, op).is_err() {
            return false;
        }
        let operands = ctx.op_operands(op).to_vec();
        let Some((&callee, args)) = operands.split_first() else {
            return false;
        };
        let Some(convention) = get_calling_convention(ctx, op) else {
            return false;
        };
        if convention != CallingConvention::Cps
            || physical_closure_type_for_callee(ctx, callee).is_none()
        {
            return false;
        }

        let location = ctx.op(op).location;
        let i32_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());
        let anyref_ty = tribute_rt::anyref(ctx).as_type_ref();
        let never = core::never(ctx).as_type_ref();
        let Some(contract) =
            exact_physical_call_contract(ctx, callee, convention, args, never, anyref_ty)
        else {
            return false;
        };
        let func_ref = closure::func(ctx, location, callee, i32_ty);
        let environment = closure::env(ctx, location, callee, anyref_ty);
        let mut args = args.to_vec();
        for (index, expected) in contract.argument_casts {
            let cast = core::unrealized_conversion_cast(ctx, location, args[index], expected);
            rewriter.insert_op(cast.op_ref());
            args[index] = cast.result(ctx);
        }
        args.insert(contract.environment_index, environment.result(ctx));
        let tail = func::tail_call_indirect(
            ctx,
            location,
            func_ref.result(ctx),
            args,
            Some(contract.signature),
        );
        copy_indirect_call_attributes(ctx, op, tail.op_ref());

        rewriter.insert_op(func_ref.op_ref());
        rewriter.insert_op(environment.op_ref());
        rewriter.replace_op(tail.op_ref());
        true
    }

    fn name(&self) -> &'static str {
        "LowerClosureTailCallArena"
    }
}

/// Copy source metadata without replacing the physical indirect-call contract.
fn copy_indirect_call_attributes(ctx: &mut IrContext, source: OpRef, destination: OpRef) {
    let mut attributes = ctx.op(source).attributes.clone();
    attributes.remove(func::INDIRECT_CALL_SIGNATURE_ATTR);
    ctx.op_mut(destination).attributes.extend(attributes);
}

fn physical_closure_type_for_callee(ctx: &IrContext, callee: ValueRef) -> Option<TypeRef> {
    let ty = ctx.value_ty(callee);
    if get_physical_closure_convention(ctx, ty).is_some() {
        return Some(ty);
    }
    let trunk_ir::refs::ValueDef::OpResult(pack, _) = ctx.value_def(callee) else {
        return None;
    };
    let closure_ty = get_closure_callable_type(ctx, pack)?;
    (get_physical_closure_convention(ctx, closure_ty).is_some()
        && is_lowered_closure_pack(ctx, pack, closure_ty))
    .then_some(closure_ty)
}

fn is_lowered_closure_pack(ctx: &IrContext, op: OpRef, closure_ty: TypeRef) -> bool {
    if adt::StructNew::from_op(ctx, op).is_err() {
        return false;
    }
    let [result] = ctx.op_results(op) else {
        return false;
    };
    if !is_closure_struct_type_ref(ctx, ctx.value_ty(*result)) {
        return false;
    }
    let [function, _environment] = ctx.op_operands(op) else {
        return false;
    };
    let trunk_ir::refs::ValueDef::OpResult(constant, _) = ctx.value_def(*function) else {
        return false;
    };
    if func::Constant::from_op(ctx, constant).is_err() {
        return false;
    }
    let Some(closure) = closure::Closure::from_type_ref(ctx, closure_ty) else {
        return false;
    };
    ctx.value_ty(*function) == closure.func_type(ctx)
}

struct PhysicalCallContract {
    environment_index: usize,
    signature: TypeRef,
    argument_casts: Vec<(usize, TypeRef)>,
}

fn exact_physical_call_contract(
    ctx: &mut IrContext,
    callee: ValueRef,
    convention: CallingConvention,
    args: &[ValueRef],
    result: TypeRef,
    environment: TypeRef,
) -> Option<PhysicalCallContract> {
    let closure_ty = physical_closure_type_for_callee(ctx, callee)?;
    (get_physical_closure_convention(ctx, closure_ty) == Some(convention)).then_some(())?;
    let function = closure::Closure::from_type_ref(ctx, closure_ty)?.func_type(ctx);
    let callable = core::Func::from_type_ref(ctx, function)?;
    if callable.r#return(ctx) != result || callable.params(ctx).len() != args.len() {
        return None;
    }
    let mut casts = Vec::new();
    for (index, (argument, expected)) in args.iter().zip(callable.params(ctx)).enumerate() {
        let actual = ctx.value_ty(*argument);
        if actual != *expected {
            if !is_closure_struct_type_ref(ctx, actual) {
                return None;
            }
            if closure::Closure::matches(ctx, *expected) {
                if physical_closure_type_for_callee(ctx, *argument) != Some(*expected) {
                    return None;
                }
            } else if *expected != environment {
                return None;
            }
            casts.push((index, *expected));
        }
    }
    let environment_index = get_physical_closure_environment_index(ctx, closure_ty)?;
    if environment_index > args.len() {
        return None;
    }
    let mut params = callable.params(ctx).to_vec();
    params.insert(environment_index, environment);
    Some(PhysicalCallContract {
        environment_index,
        signature: core::func(ctx, result, params).as_type_ref(),
        argument_casts: casts,
    })
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
    ctx.block_args(entry).first().copied()
}

fn tagged_closure_transfers_are_legal(ctx: &mut IrContext, func_op: func::Func) -> bool {
    let Some(&body) = ctx.op(func_op.op_ref()).regions.first() else {
        return true;
    };
    let mut transfers = Vec::new();
    let _ = walk_region::<()>(ctx, body, &mut |op| {
        if func::Func::matches(ctx, op) {
            return ControlFlow::Continue(WalkAction::Skip);
        }
        if func::CallIndirect::matches(ctx, op) || func::TailCallIndirect::matches(ctx, op) {
            transfers.push(op);
        }
        ControlFlow::Continue(WalkAction::Advance)
    });
    if transfers.is_empty() {
        return true;
    }

    let anyref = tribute_rt::anyref(ctx).as_type_ref();
    let never = core::never(ctx).as_type_ref();
    transfers.into_iter().all(|op| {
        let operands = ctx.op_operands(op).to_vec();
        let Some((&callee, args)) = operands.split_first() else {
            return false;
        };
        if physical_closure_type_for_callee(ctx, callee).is_none() {
            return true;
        }
        let Some(convention) = get_calling_convention(ctx, op) else {
            return false;
        };
        if func::CallIndirect::matches(ctx, op) {
            let Some(&result) = ctx.op_result_types(op).first() else {
                return false;
            };
            exact_physical_call_contract(ctx, callee, convention, args, result, anyref).is_some()
        } else {
            convention == CallingConvention::Cps
                && exact_physical_call_contract(ctx, callee, convention, args, never, anyref)
                    .is_some()
        }
    })
}

/// Lower closures using arena IR.
///
/// This compatibility entry point prepares module-level function signatures,
/// then lowers every function body in the module tree.
pub(crate) fn lower_closures(ctx: &mut IrContext, module: Module) {
    prepare_closure_lowering(ctx, module);

    lower_prepared_closures(ctx, module);
}

/// Lower every already-prepared function body in a module to closure storage.
///
/// The worklist is refreshed after each function so functions introduced by an
/// earlier transformation are lowered once as well. Processing each function
/// operation at most once keeps this traversal bounded without relying on
/// function names or target-specific pipeline ordering.
pub fn lower_prepared_closures(ctx: &mut IrContext, module: Module) {
    let mut lowered = HashSet::new();
    let mut worklist = Vec::new();

    loop {
        let _ = walk_op::<()>(ctx, module.op(), &mut |op| {
            if let Ok(func_op) = func::Func::from_op(ctx, op)
                && lowered.insert(op)
            {
                worklist.push(func_op);
            }
            ControlFlow::Continue(WalkAction::Advance)
        });

        let Some(func_op) = worklist.pop() else {
            break;
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
/// Closure calls already carry convention-specific hidden operands. This pass
/// only interposes the physical closure environment.
pub(crate) fn lower_closures_in_func(ctx: &mut IrContext, func_op: func::Func) {
    if ctx.op(func_op.op_ref()).regions.is_empty() {
        return;
    }
    if !tagged_closure_transfers_are_legal(ctx, func_op) {
        return;
    }
    let legacy_evidence = evidence_param_for_func(ctx, func_op);
    let applicator = PatternApplicator::new(TypeConverter::new())
        .with_target(
            ConversionTarget::new()
                .legal_op("func", "func")
                .recursive_legal_op("func", "func"),
        )
        .add_pattern(LowerClosureCallArena { legacy_evidence })
        .add_pattern(LowerClosureTailCallArena)
        .add_pattern(LowerClosureNewArena)
        .add_pattern(LowerClosureFuncArena)
        .add_pattern(LowerClosureEnvArena);
    applicator.apply_partial(ctx, func_op);
}

/// Select the canonical `_closure` storage type after all semantic consumers
/// have validated the exact convention-proven callable type.
///
/// The plan covers aliases, nested type parameters, type-bearing operation
/// attributes, results, and block arguments before applying any update. The
/// pack provenance is consumed by exact closure lowering, then removed: it is
/// not a type-equivalence rule or final physical-IR contract.
pub fn finalize_closure_storage_layout(ctx: &mut IrContext, module: Module) {
    let closure_struct = closure_struct_type_ref(ctx);
    let ops = collect_ops(ctx, module.op());
    let aliases = ctx.type_aliases().to_vec();
    let mut physicalizer = ClosureTypePhysicalizer::new(ctx, closure_struct);
    let mut alias_updates = Vec::new();
    let mut attribute_removals = Vec::new();
    let mut attribute_updates = Vec::new();
    let mut result_updates = Vec::new();
    let mut block_arg_updates = Vec::new();

    for (name, ty) in aliases {
        let converted = physicalizer.convert_type(ty);
        if converted != ty {
            alias_updates.push((name, converted));
        }
    }
    for op in ops {
        for (name, value) in physicalizer.ctx.op(op).attributes.clone() {
            if name == Symbol::new(CLOSURE_CALLABLE_TYPE_ATTR) {
                attribute_removals.push(op);
                continue;
            }
            let converted = physicalizer.convert_attribute(value.clone());
            if converted != value {
                attribute_updates.push((op, name, converted));
            }
        }
        for (index, ty) in physicalizer
            .ctx
            .op_result_types(op)
            .to_vec()
            .into_iter()
            .enumerate()
        {
            let converted = physicalizer.convert_type(ty);
            if converted != ty {
                result_updates.push((op, index as u32, converted));
            }
        }
        for region in physicalizer.ctx.op(op).regions.clone() {
            for block in physicalizer.ctx.region(region).blocks.clone() {
                let args = physicalizer
                    .ctx
                    .block(block)
                    .args
                    .iter()
                    .map(|argument| argument.ty)
                    .collect::<Vec<_>>();
                for (index, ty) in args.into_iter().enumerate() {
                    let converted = physicalizer.convert_type(ty);
                    if converted != ty {
                        block_arg_updates.push((block, index as u32, converted));
                    }
                }
            }
        }
    }

    drop(physicalizer);
    for (name, ty) in alias_updates {
        ctx.register_type_alias(name, ty);
    }
    for op in attribute_removals {
        ctx.op_mut(op)
            .attributes
            .remove(Symbol::new(CLOSURE_CALLABLE_TYPE_ATTR));
    }
    for (op, name, value) in attribute_updates {
        ctx.op_mut(op).attributes.insert(name, value);
    }
    for (op, index, ty) in result_updates {
        ctx.set_op_result_type(op, index, ty);
    }
    for (block, index, ty) in block_arg_updates {
        ctx.set_block_arg_type(block, index, ty);
    }
}

struct ClosureTypePhysicalizer<'a> {
    ctx: &'a mut IrContext,
    closure_struct: TypeRef,
    cache: HashMap<TypeRef, TypeRef>,
    visiting: HashSet<TypeRef>,
}

impl<'a> ClosureTypePhysicalizer<'a> {
    fn new(ctx: &'a mut IrContext, closure_struct: TypeRef) -> Self {
        Self {
            ctx,
            closure_struct,
            cache: HashMap::new(),
            visiting: HashSet::new(),
        }
    }

    fn convert_type(&mut self, ty: TypeRef) -> TypeRef {
        if closure::Closure::matches(self.ctx, ty) {
            return self.closure_struct;
        }
        if let Some(&converted) = self.cache.get(&ty) {
            return converted;
        }
        if !self.visiting.insert(ty) {
            return ty;
        }

        let data = self.ctx.types.get(ty).clone();
        let mut converted = data.clone();
        for parameter in &mut converted.params {
            *parameter = self.convert_type(*parameter);
        }
        let attributes: Vec<_> = data
            .attrs
            .iter()
            .map(|(name, value)| (*name, self.convert_attribute(value.clone())))
            .collect();
        converted.attrs.clear();
        converted.attrs.extend(attributes);
        let converted = if converted == data {
            ty
        } else {
            self.ctx.types.intern(converted)
        };
        self.visiting.remove(&ty);
        self.cache.insert(ty, converted);
        converted
    }

    fn convert_attribute(&mut self, attribute: Attribute) -> Attribute {
        match attribute {
            Attribute::Type(ty) => Attribute::Type(self.convert_type(ty)),
            Attribute::List(values) => Attribute::List(
                values
                    .into_iter()
                    .map(|value| self.convert_attribute(value))
                    .collect(),
            ),
            value => value,
        }
    }
}

fn collect_ops(ctx: &IrContext, root: OpRef) -> Vec<OpRef> {
    let mut ops = Vec::new();
    let _ = walk_op::<()>(ctx, root, &mut |op| {
        ops.push(op);
        ControlFlow::Continue(WalkAction::Advance)
    });
    ops
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

/// PassManager-friendly closure lowering for a module prepared earlier in the
/// pipeline.
pub struct LowerPreparedClosures;

impl Pass for LowerPreparedClosures {
    type Target = core::Module;

    fn name(&self) -> &'static str {
        "lower-prepared-closures"
    }

    fn run(&mut self, ctx: &mut IrContext, target: core::Module) -> PassRunResult {
        lower_prepared_closures(ctx, target.into());
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

    fn assert_module_is_structurally_valid(ctx: &IrContext, module: Module) {
        let use_chains = trunk_ir::validation::validate_use_chains(ctx, module);
        assert!(
            use_chains.is_ok(),
            "invalid use chains after lowering:\n{use_chains}"
        );

        let operation_verifiers = trunk_ir::validation::validate_operation_verifiers(ctx, module);
        assert!(
            operation_verifiers.is_ok(),
            "invalid operations after lowering:\n{operation_verifiers}"
        );
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
        assert_eq!(
            ir.matches("signature").count(),
            2,
            "each lowered indirect call must retain an exact signature:\n{ir}"
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
    fn module_entrypoint_lowers_nested_function_closures() {
        let mut ctx = IrContext::new();
        let module = nested_closure_test_module(&mut ctx);

        lower_closures(&mut ctx, module);
        assert_module_is_structurally_valid(&ctx, module);

        let inner = func_by_name_recursive(&ctx, module, "inner");
        let inner_ir = print_module(&ctx, inner.op_ref());
        assert!(
            !inner_ir.contains("closure.new"),
            "module lowering must revisit nested functions:\n{inner_ir}"
        );
        assert!(
            !inner_ir.contains("closure.func") && !inner_ir.contains("closure.env"),
            "nested closure accessors must be fully lowered:\n{inner_ir}"
        );
    }

    #[test]
    fn prepared_module_worklist_lowers_nested_function_closures() {
        let mut ctx = IrContext::new();
        let module = nested_closure_test_module(&mut ctx);

        prepare_closure_lowering(&mut ctx, module);
        let core_module = core::Module::from_op(&ctx, module.op()).unwrap();
        let mut pass = LowerPreparedClosures;
        pass.run(&mut ctx, core_module).unwrap();
        assert_module_is_structurally_valid(&ctx, module);

        let inner = func_by_name_recursive(&ctx, module, "inner");
        let inner_ir = print_module(&ctx, inner.op_ref());
        assert!(
            !inner_ir.contains("closure.new"),
            "prepared module worklist must lower nested functions:\n{inner_ir}"
        );
    }

    #[test]
    fn storage_finalization_rewrites_every_type_surface_but_not_pack_provenance() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !closure = closure.closure(core.func(core.i32, core.i32)) {tribute.calling_convention = 0}
  !nested = core.tuple(!closure)
  func.func @run(%callback: !closure) -> !nested {
    %environment = adt.ref_null {type = tribute_rt.anyref} : tribute_rt.anyref
    %created = closure.new %environment {func_ref = @callback} : !closure
    %packed = core.tuple_pack %created : !nested
    func.return %packed
  }
}"#,
        );

        let run = func_by_name(&ctx, module, "run");
        let logical = ctx
            .type_aliases()
            .iter()
            .find_map(|(name, ty)| (*name == Symbol::new("closure")).then_some(*ty))
            .expect("test module must define the logical closure alias");
        let pack = ctx
            .block(ctx.region(run.body(&ctx)).blocks[0])
            .ops
            .iter()
            .copied()
            .find(|op| closure::New::from_op(&ctx, *op).is_ok())
            .expect("test module must create a closure");
        ctx.op_mut(pack).attributes.insert(
            Symbol::new(CLOSURE_CALLABLE_TYPE_ATTR),
            Attribute::Type(logical),
        );

        finalize_closure_storage_layout(&mut ctx, module);

        let physical = closure_struct_type_ref(&mut ctx);
        let signature = core::Func::from_type_ref(&ctx, run.r#type(&ctx)).unwrap();
        assert_eq!(signature.params(&ctx), [physical]);
        assert_eq!(
            ctx.value_ty(ctx.block_args(ctx.region(run.body(&ctx)).blocks[0])[0]),
            physical
        );
        assert_eq!(ctx.op_result_types(pack), [physical]);
        assert!(
            !ctx.op(pack)
                .attributes
                .contains_key(CLOSURE_CALLABLE_TYPE_ATTR),
            "final physical IR must not retain logical closure provenance"
        );
        let printed = print_module(&ctx, module.op());
        assert!(
            !printed.contains(CLOSURE_CALLABLE_TYPE_ATTR),
            "final physical IR must not retain logical closure provenance:\n{printed}"
        );
        assert!(printed.contains("!closure = adt.struct"), "{printed}");
        assert!(
            printed.contains("!nested = core.tuple(!closure)"),
            "{printed}"
        );
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

    #[test]
    fn tagged_dispatch_aware_tail_retains_exact_physical_signature() {
        let mut ctx = IrContext::new();
        let ev = evidence_type_str();
        let module = parse_test_module(
            &mut ctx,
            &format!(
                r#"core.module @test {{
  !cps = closure.closure(core.func(core.never, {ev}, core.i32, core.i32, core.i32)) {{tribute.calling_convention = 2, tribute.closure_environment_index = 1}}
  func.func @run(%callee: !cps, %evidence: {ev}, %done: core.i32, %dispatch: core.i32, %value: core.i32) -> core.never attributes {{tribute.calling_convention = 2}} {{
    func.tail_call_indirect %callee, %evidence, %done, %dispatch, %value {{tribute.calling_convention = 2, signature = core.func(core.never, {ev}, core.i32, core.i32, core.i32)}}
  }}
}}"#
            ),
        );
        let run = func_by_name(&ctx, module, "run");
        let entry_args = ctx.block_args(ctx.region(run.body(&ctx)).blocks[0]);
        let evidence = entry_args[1];
        let done = entry_args[2];
        let dispatch = entry_args[3];
        let value = entry_args[4];

        lower_closures_in_func(&mut ctx, run);

        let tail = ctx
            .block(ctx.region(run.body(&ctx)).blocks[0])
            .ops
            .iter()
            .copied()
            .find(|op| func::TailCallIndirect::from_op(&ctx, *op).is_ok())
            .unwrap();
        let signature = func::indirect_call_signature(&ctx, tail).unwrap();
        let callable = core::Func::from_type_ref(&ctx, signature).unwrap();
        assert_eq!(ctx.op_operands(tail).len(), 6);
        assert_eq!(ctx.op_operands(tail)[1], evidence);
        assert_eq!(
            ctx.value_ty(ctx.op_operands(tail)[2]),
            tribute_rt::anyref(&mut ctx).as_type_ref()
        );
        assert_eq!(ctx.op_operands(tail)[3], done);
        assert_eq!(ctx.op_operands(tail)[4], dispatch);
        assert_eq!(ctx.op_operands(tail)[5], value);
        assert_eq!(
            callable.params(&ctx),
            ctx.op_operands(tail)[1..]
                .iter()
                .map(|&operand| ctx.value_ty(operand))
                .collect::<Vec<_>>(),
            "the physical signature must include the inserted environment operand"
        );
    }

    #[test]
    fn metadata_less_tagged_closure_transfer_leaves_function_unchanged() {
        let mut ctx = IrContext::new();
        let evidence = evidence_type_str();
        let module = parse_test_module(
            &mut ctx,
            &format!(
                r#"core.module @test {{
  !cps = closure.closure(core.func(core.never, {evidence}, core.i32)) {{tribute.calling_convention = 2}}
  func.func @callee(%evidence: {evidence}, %env: tribute_rt.anyref, %done: core.i32) -> core.never attributes {{tribute.calling_convention = 2}} {{
    func.unreachable
  }}
  func.func @run(%evidence: {evidence}, %done: core.i32) -> core.never attributes {{tribute.calling_convention = 2}} {{
    %environment = adt.ref_null {{type = tribute_rt.anyref}} : tribute_rt.anyref
    %callee = closure.new %environment {{func_ref = @callee}} : !cps
    func.tail_call_indirect %callee, %evidence, %done
  }}
}}"#
            ),
        );
        let run = func_by_name(&ctx, module, "run");
        let before = print_module(&ctx, module.op());

        lower_closures_in_func(&mut ctx, run);

        assert_eq!(print_module(&ctx, module.op()), before);
    }

    #[test]
    fn bodyless_declaration_remains_unchanged() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @external(%value: core.i32) -> core.i32
}"#,
        );
        let external = func_by_name(&ctx, module, "external");
        let before = print_module(&ctx, module.op());

        lower_closures_in_func(&mut ctx, external);

        assert_eq!(print_module(&ctx, module.op()), before);
    }

    #[test]
    fn raw_function_constant_indirect_call_remains_unchanged() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @external(%value: core.i32) -> core.i32
  func.func @caller(%value: core.i32) -> core.i32 {
    %callee = func.constant {func_ref = @external} : core.func(core.i32, core.i32)
    %result = func.call_indirect %callee, %value : core.i32
    func.return %result
  }
}"#,
        );
        let caller = func_by_name(&ctx, module, "caller");
        let before = print_module(&ctx, module.op());

        lower_closures_in_func(&mut ctx, caller);

        assert_eq!(print_module(&ctx, module.op()), before);
    }

    #[test]
    fn differently_typed_tagged_closure_pack_fails_closed() {
        let mut ctx = IrContext::new();
        let evidence = evidence_type_str();
        let module = parse_test_module(
            &mut ctx,
            &format!(
                r#"core.module @test {{
  !_closure = adt.struct(core.i32, tribute_rt.anyref) {{name = @_closure}}
  !expected = closure.closure(core.func(core.never, {evidence}, core.i32, core.i32)) {{tribute.calling_convention = 2}}
  !actual = closure.closure(core.func(core.never, {evidence}, core.i32, core.bool)) {{tribute.calling_convention = 2}}
  !outer = closure.closure(core.func(core.never, {evidence}, core.i32, !expected)) {{tribute.calling_convention = 2}}

  func.func @actual_fn(%evidence: {evidence}, %done: core.i32, %value: core.bool) -> core.never attributes {{tribute.calling_convention = 2}} {{
    func.unreachable
  }}
  func.func @run(%callee: !outer, %evidence: {evidence}, %done: core.i32) -> core.never attributes {{tribute.calling_convention = 2}} {{
    %function = func.constant {{func_ref = @actual_fn}} : core.func(core.never, {evidence}, core.i32, core.bool)
    %environment = adt.ref_null {{type = tribute_rt.anyref}} : tribute_rt.anyref
    %argument = adt.struct_new %function, %environment {{type = !_closure, tribute.closure_callable_type = !actual}} : !_closure
    func.tail_call_indirect %callee, %evidence, %done, %argument {{tribute.calling_convention = 2}}
  }}
}}"#
            ),
        );
        let run = func_by_name(&ctx, module, "run");
        let before = print_module(&ctx, module.op());

        lower_closures_in_func(&mut ctx, run);

        assert_eq!(print_module(&ctx, module.op()), before);
    }
}
