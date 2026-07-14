//! Lambda/closure lowering.
//!
//! Emits `closure.lambda` ops for source-level lambda expressions.
//! The downstream `lower_closure_lambda` pass handles extraction into
//! top-level functions and closure conversion.

use std::collections::HashSet;

use tribute_core::{CallableAbi, set_calling_convention};
use trunk_ir::Symbol;
use trunk_ir::context::{BlockArgData, BlockData, IrContext, RegionData};
use trunk_ir::dialect::{adt, core, func};
use trunk_ir::refs::{TypeRef, ValueRef};
use trunk_ir::types::{Attribute, Location};

use crate::ast::TypeKind;

use tribute_ir::dialect::{ability, closure};

use crate::ast::{Expr, ExprKind, LocalId, Param, ResolvedRef, Stmt, TypedRef};

use super::super::context::{CaptureInfo, IrLoweringCtx};
use super::IrBuilder;
use crate::ast::CallingConvention;

/// Collect all local variable references in an expression.
pub(super) fn collect_free_vars<'db>(expr: &Expr<TypedRef<'db>>, free_vars: &mut HashSet<LocalId>) {
    match &*expr.kind {
        ExprKind::Var(typed_ref) => {
            if let ResolvedRef::Local { id, .. } = &typed_ref.resolved {
                free_vars.insert(*id);
            }
        }
        ExprKind::IntLit(_)
        | ExprKind::NatLit(_)
        | ExprKind::FloatLit(_)
        | ExprKind::BoolLit(_)
        | ExprKind::StringLit(_)
        | ExprKind::BytesLit(_)
        | ExprKind::RuneLit(_)
        | ExprKind::Nil
        | ExprKind::Error => {}
        ExprKind::BinOp { lhs, rhs, .. } => {
            collect_free_vars(lhs, free_vars);
            collect_free_vars(rhs, free_vars);
        }
        ExprKind::Call { callee, args } => {
            collect_free_vars(callee, free_vars);
            for arg in args {
                collect_free_vars(arg, free_vars);
            }
        }
        ExprKind::Cons { args, .. } => {
            for arg in args {
                collect_free_vars(arg, free_vars);
            }
        }
        ExprKind::MethodCall { receiver, args, .. } => {
            collect_free_vars(receiver, free_vars);
            for arg in args {
                collect_free_vars(arg, free_vars);
            }
        }
        ExprKind::Block { stmts, value } => {
            for stmt in stmts {
                match stmt {
                    Stmt::Let { value, .. } => collect_free_vars(value, free_vars),
                    Stmt::Expr { expr, .. } => collect_free_vars(expr, free_vars),
                }
            }
            collect_free_vars(value, free_vars);
        }
        ExprKind::Record { fields, .. } => {
            for (_, value) in fields {
                collect_free_vars(value, free_vars);
            }
        }
        ExprKind::Tuple(elements) => {
            for elem in elements {
                collect_free_vars(elem, free_vars);
            }
        }
        ExprKind::Case { scrutinee, arms } => {
            collect_free_vars(scrutinee, free_vars);
            for arm in arms {
                if let Some(guard) = &arm.guard {
                    collect_free_vars(guard, free_vars);
                }
                collect_free_vars(&arm.body, free_vars);
            }
        }
        ExprKind::Lambda { body, .. } => {
            collect_free_vars(body, free_vars);
        }
        ExprKind::Handle { body, handlers } => {
            collect_free_vars(body, free_vars);
            for handler in handlers {
                collect_free_vars(&handler.body, free_vars);
            }
        }
        ExprKind::List(elements) => {
            for elem in elements {
                collect_free_vars(elem, free_vars);
            }
        }
        ExprKind::Resume { arg, local_id } => {
            collect_free_vars(arg, free_vars);
            if let Some(id) = local_id {
                free_vars.insert(*id);
            }
        }
    }
}

/// Collect all local variable references in a block (statements + value expression).
pub(super) fn collect_free_vars_in_block<'db>(
    stmts: &[Stmt<TypedRef<'db>>],
    value: &Expr<TypedRef<'db>>,
    free_vars: &mut HashSet<LocalId>,
) {
    for stmt in stmts {
        match stmt {
            Stmt::Let { value: v, .. } => collect_free_vars(v, free_vars),
            Stmt::Expr { expr, .. } => collect_free_vars(expr, free_vars),
        }
    }
    collect_free_vars(value, free_vars);
}

/// Analyze captures for a CPS continuation (remaining stmts + value).
///
/// Finds all variables from the current scope that are referenced in the
/// remaining computation. `excluded_ids` contains IDs that should not be
/// captured (e.g., the continuation parameter).
pub(super) fn analyze_continuation_captures<'db>(
    ctx: &IrLoweringCtx<'db>,
    ir: &mut IrContext,
    remaining_stmts: &[Stmt<TypedRef<'db>>],
    value: &Expr<TypedRef<'db>>,
    excluded_ids: &HashSet<LocalId>,
) -> Vec<CaptureInfo> {
    let mut free_vars = HashSet::new();
    collect_free_vars_in_block(remaining_stmts, value, &mut free_vars);

    let mut captures = Vec::new();
    for (local_id, name, value) in ctx.all_bindings() {
        if free_vars.contains(&local_id) && !excluded_ids.contains(&local_id) {
            captures.push(CaptureInfo {
                name,
                local_id,
                ty: ir.value_ty(value),
                value,
            });
        }
    }
    captures
}

/// Analyze captures for a lambda expression.
fn analyze_captures<'db>(
    ctx: &IrLoweringCtx<'db>,
    ir: &mut IrContext,
    params: &[Param],
    body: &Expr<TypedRef<'db>>,
) -> Vec<CaptureInfo> {
    let mut free_vars = HashSet::new();
    collect_free_vars(body, &mut free_vars);

    let param_ids: HashSet<LocalId> = params.iter().filter_map(|p| p.local_id).collect();

    let mut captures = Vec::new();
    for (local_id, name, value) in ctx.all_bindings() {
        if free_vars.contains(&local_id) && !param_ids.contains(&local_id) {
            captures.push(CaptureInfo {
                name,
                local_id,
                ty: ir.value_ty(value),
                value,
            });
        }
    }

    captures
}

/// Lower a lambda expression to a `closure.lambda` op.
///
/// Emits a high-level `closure.lambda` with captured values and a body region.
/// The downstream `lower_closure_lambda` pass extracts the body into a
/// top-level `func.func` and replaces this op with `closure.new`.
pub(super) fn lower_lambda<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    location: Location,
    params: &[Param],
    body: &Expr<TypedRef<'db>>,
    param_ir_types: &[TypeRef],
    result_ir_ty: TypeRef,
    convention: CallingConvention,
) -> Option<ValueRef> {
    let any_ty = builder.ctx.anyref_type(builder.ir);
    let evidence_ty = ability::evidence_adt_type_ref(builder.ir);
    let abi = CallableAbi::new(convention, param_ir_types.iter().copied(), result_ir_ty);

    // Step 1: Analyze captures
    let captures = analyze_captures(builder.ctx, builder.ir, params, body);

    // Step 2: Build lambda body region.
    // Entry block has only the lambda's formal parameters — evidence/env are
    // added later by the lower_closure_lambda pass.
    let mut block_args = Vec::new();
    for (i, param) in params.iter().enumerate() {
        let ty = param_ir_types.get(i).copied().unwrap_or(any_ty);
        let mut arg = BlockArgData {
            ty,
            attrs: Default::default(),
        };
        arg.attrs
            .insert(Symbol::new("bind_name"), Attribute::Symbol(param.name));
        block_args.push(arg);
    }

    let block_args = if convention.needs_evidence() {
        let mut evidence_arg = BlockArgData {
            ty: evidence_ty,
            attrs: Default::default(),
        };
        evidence_arg.attrs.insert(
            Symbol::new("bind_name"),
            Attribute::Symbol(Symbol::new("__evidence")),
        );
        let mut args = vec![evidence_arg];
        if convention.needs_done_k() {
            let mut done_k_arg = BlockArgData {
                ty: any_ty,
                attrs: Default::default(),
            };
            done_k_arg.attrs.insert(
                Symbol::new("bind_name"),
                Attribute::Symbol(Symbol::new("__done_k")),
            );
            args.push(done_k_arg);
        }
        args.append(&mut block_args);
        args
    } else {
        block_args
    };

    let entry_block = builder.ir.create_block(BlockData {
        location,
        args: block_args,
        ops: Default::default(),
        parent_region: None,
    });

    // Enter scope for lambda body
    {
        let mut scope = builder.ctx.scope();

        let param_offset = abi.source_param_offset();
        for (i, param) in params.iter().enumerate() {
            if let Some(local_id) = param.local_id {
                let arg_val = builder.ir.block_arg(entry_block, (i + param_offset) as u32);
                scope.bind(local_id, param.name, arg_val);
            }
        }

        // No need to rebind captures — they are already in scope from the parent.
        // The closure.lambda body is NOT isolated from above, so parent-scope
        // ValueRefs are valid inside the body region.

        if convention.needs_done_k() {
            let prev_done_k = scope.done_k;
            let prev_evidence = scope.evidence;
            let evidence_val = builder.ir.block_arg(entry_block, 0);
            let done_k_val = builder.ir.block_arg(entry_block, 1);
            scope.evidence = Some(evidence_val);
            scope.done_k = Some(done_k_val);

            let mut inner_builder = IrBuilder::new(&mut scope, builder.ir, entry_block);
            match super::expr::lower_block_cps_for_expr(&mut inner_builder, body.clone()) {
                Some((result, true)) => {
                    // CPS result: effectful call happened; add func.return with the result.
                    // The callee already handled done_k via its continuation chain.
                    let anyref_ty = inner_builder.ctx.anyref_type(inner_builder.ir);
                    let result = inner_builder.cast_if_needed(location, result, anyref_ty);
                    let ret_op = func::r#return(inner_builder.ir, location, [result]);
                    inner_builder
                        .ir
                        .push_op(inner_builder.block, ret_op.op_ref());
                }
                Some((result, false)) => {
                    // Pure result: call done_k(result)
                    let anyref_ty = inner_builder.ctx.anyref_type(inner_builder.ir);
                    let result = inner_builder.cast_if_needed(location, result, anyref_ty);
                    super::emit_done_k_call(&mut inner_builder, location, done_k_val, result);
                }
                None => {
                    let nil = inner_builder.emit_nil(location);
                    super::emit_done_k_call(&mut inner_builder, location, done_k_val, nil);
                }
            }
            scope.done_k = prev_done_k;
            scope.evidence = prev_evidence;
        } else {
            let prev_evidence = scope.evidence;
            if convention.needs_evidence() {
                scope.evidence = Some(builder.ir.block_arg(entry_block, 0));
            }
            let mut inner_builder = IrBuilder::new(&mut scope, builder.ir, entry_block);
            let result = super::expr::lower_expr(&mut inner_builder, body.clone())
                .unwrap_or_else(|| inner_builder.emit_nil(location));
            let result = inner_builder.cast_if_needed(location, result, result_ir_ty);
            let ret_op = func::r#return(inner_builder.ir, location, [result]);
            inner_builder
                .ir
                .push_op(inner_builder.block, ret_op.op_ref());
            scope.evidence = prev_evidence;
        }
    }

    let body_region = builder.ir.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![entry_block],
        parent_op: None,
    });

    // Step 3: Emit closure.lambda
    let capture_values: Vec<ValueRef> = captures.iter().map(|c| c.value).collect();

    let func_param_types = abi.lowered_params(evidence_ty, any_ty);
    let func_result_ty = abi.lowered_result(any_ty);
    let closure_func_ty = builder
        .ctx
        .func_type(builder.ir, &func_param_types, func_result_ty);
    let closure_ty = builder.ctx.closure_type(builder.ir, closure_func_ty);

    let lambda_op = closure::lambda(
        builder.ir,
        location,
        capture_values,
        closure_ty,
        body_region,
    );
    set_calling_convention(builder.ir, lambda_op.op_ref(), convention);
    builder.ir.push_op(builder.block, lambda_op.op_ref());
    Some(lambda_op.result(builder.ir))
}

/// Wrap a named function reference as a CPS closure value.
///
/// Generates a thin wrapper function with `(evidence, env, done_k, params...) -> anyref`
/// signature. For pure originals: calls the function, then calls done_k(result).
/// For effectful originals: forwards done_k to the function.
/// Then emits `closure.new @wrapper, nil`.
pub(super) fn wrap_func_as_closure(
    builder: &mut IrBuilder<'_, '_>,
    location: Location,
    func_name: Symbol,
    param_ir_types: &[TypeRef],
    result_ir_ty: TypeRef,
    target_convention: Option<CallingConvention>,
) -> ValueRef {
    let any_ty = builder.ctx.anyref_type(builder.ir);
    let evidence_ty = ability::evidence_adt_type_ref(builder.ir);
    let source_convention = builder
        .ctx
        .function_calling_convention(func_name)
        .unwrap_or(CallingConvention::Direct);
    let convention = target_convention.unwrap_or(source_convention);
    assert!(
        source_convention <= convention,
        "cannot adapt {source_convention:?} function to weaker {convention:?} convention"
    );
    let abi = CallableAbi::new(convention, param_ir_types.iter().copied(), result_ir_ty);
    let source_abi = CallableAbi::new(
        source_convention,
        param_ir_types.iter().copied(),
        result_ir_ty,
    );
    let logical_param_types = abi.lowered_params(evidence_ty, any_ty);
    let physical_param_types = abi.interpose_environment(&logical_param_types, any_ty);
    let lowered_result_ty = abi.lowered_result(any_ty);

    // Generate unique wrapper name
    let wrapper_name = builder.ctx.gen_lambda_name();

    let env_index = usize::from(convention.needs_evidence());
    let done_k_index = convention.needs_done_k().then_some(env_index + 1);
    let source_offset = abi.source_param_offset() + 1;
    let mut block_args = Vec::with_capacity(physical_param_types.len());
    for (i, &ty) in physical_param_types.iter().enumerate() {
        let name = if convention.needs_evidence() && i == 0 {
            Symbol::new("__evidence")
        } else if i == env_index {
            Symbol::new("__env")
        } else if Some(i) == done_k_index {
            Symbol::new("__done_k")
        } else {
            Symbol::from_dynamic(&format!("__arg_{}", i - source_offset))
        };
        let mut arg = BlockArgData {
            ty,
            attrs: Default::default(),
        };
        arg.attrs
            .insert(Symbol::new("bind_name"), Attribute::Symbol(name));
        block_args.push(arg);
    }

    let total_params = block_args.len();

    let entry_block = builder.ir.create_block(BlockData {
        location,
        args: block_args,
        ops: Default::default(),
        parent_region: None,
    });
    let arg_values: Vec<ValueRef> = (0..total_params)
        .map(|i| builder.ir.block_arg(entry_block, i as u32))
        .collect();

    let user_params = &arg_values[source_offset..];

    // Coerce arguments to match callee's declared parameter types
    let mut call_args: Vec<ValueRef> = user_params.to_vec();
    if let Some(scheme) = builder.ctx.lookup_function_type(func_name) {
        let body = scheme.body(builder.ctx.db);
        if let TypeKind::Func { params, .. } = body.kind(builder.ctx.db) {
            for (i, param_ty) in params.iter().enumerate() {
                if i < call_args.len() {
                    let target_ty = builder.ctx.convert_type(builder.ir, *param_ty);
                    let value_ty = builder.ir.value_ty(call_args[i]);
                    if value_ty != target_ty {
                        let cast = core::unrealized_conversion_cast(
                            builder.ir,
                            location,
                            call_args[i],
                            target_ty,
                        );
                        builder.ir.push_op(entry_block, cast.op_ref());
                        call_args[i] = cast.result(builder.ir);
                    }
                }
            }
        }
    }

    let mut lowered_call_args =
        Vec::with_capacity(source_abi.source_param_offset() + call_args.len());
    if source_convention.needs_evidence() {
        lowered_call_args.push(arg_values[0]);
    }
    if source_convention.needs_done_k()
        && let Some(done_k_index) = done_k_index
    {
        lowered_call_args.push(arg_values[done_k_index]);
    }
    lowered_call_args.append(&mut call_args);
    let source_result_ty = source_abi.lowered_result(any_ty);
    let call_op = func::call(
        builder.ir,
        location,
        lowered_call_args,
        source_result_ty,
        func_name,
    );
    set_calling_convention(builder.ir, call_op.op_ref(), source_convention);
    builder.ir.push_op(entry_block, call_op.op_ref());
    if convention.needs_done_k() && !source_convention.needs_done_k() {
        let call_result = call_op.result(builder.ir);
        let mut inner_builder = IrBuilder::new(builder.ctx, builder.ir, entry_block);
        super::emit_done_k_call(
            &mut inner_builder,
            location,
            arg_values[done_k_index.expect("Cps adapter has done_k")],
            call_result,
        );
    } else {
        let result = if source_result_ty == lowered_result_ty {
            call_op.result(builder.ir)
        } else {
            let cast = core::unrealized_conversion_cast(
                builder.ir,
                location,
                call_op.result(builder.ir),
                lowered_result_ty,
            );
            builder.ir.push_op(entry_block, cast.op_ref());
            cast.result(builder.ir)
        };
        let ret_op = func::r#return(builder.ir, location, [result]);
        builder.ir.push_op(entry_block, ret_op.op_ref());
    }

    // Create region and wrapper func op
    let body_region = builder.ir.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![entry_block],
        parent_op: None,
    });

    let wrapper_func_ty =
        builder
            .ctx
            .func_type(builder.ir, &physical_param_types, lowered_result_ty);

    let func_op = func::func(
        builder.ir,
        location,
        wrapper_name,
        wrapper_func_ty,
        body_region,
    );
    set_calling_convention(builder.ir, func_op.op_ref(), convention);

    let module_block = builder
        .ctx
        .module_block()
        .expect("module block should be set");
    builder.ir.push_op(module_block, func_op.op_ref());

    // Emit closure.new @wrapper, null_env at the call site
    let null_op = adt::ref_null(builder.ir, location, any_ty, any_ty);
    builder.ir.push_op(builder.block, null_op.op_ref());
    let null_env = null_op.result(builder.ir);
    let closure_func_ty =
        builder
            .ctx
            .func_type(builder.ir, &logical_param_types, lowered_result_ty);
    let closure_ty = builder.ctx.closure_type(builder.ir, closure_func_ty);
    let closure_op = closure::new(builder.ir, location, null_env, closure_ty, wrapper_name);
    set_calling_convention(builder.ir, closure_op.op_ref(), convention);
    builder.ir.push_op(builder.block, closure_op.op_ref());
    closure_op.result(builder.ir)
}
