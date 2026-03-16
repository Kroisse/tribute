//! Lambda/closure lowering.
//!
//! Emits `closure.lambda` ops for source-level lambda expressions.
//! The downstream `lower_closure_lambda` pass handles extraction into
//! top-level functions and closure conversion.

use std::collections::HashSet;

use trunk_ir::Symbol;
use trunk_ir::context::{BlockArgData, BlockData, IrContext, RegionData};
use trunk_ir::dialect::{adt, core, func};
use trunk_ir::refs::{TypeRef, ValueRef};
use trunk_ir::types::{Attribute, Location};

use crate::ast::TypeKind;

use tribute_ir::dialect::{ability as arena_ability, closure as arena_closure};

use crate::ast::{Expr, ExprKind, LocalId, Param, ResolvedRef, Stmt, TypedRef};

use super::super::context::{CaptureInfo, IrLoweringCtx};
use super::IrBuilder;

/// Collect all local variable references in an expression.
fn collect_free_vars<'db>(expr: &Expr<TypedRef<'db>>, free_vars: &mut HashSet<LocalId>) {
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
        ExprKind::Resume { arg, local_id } => {
            collect_free_vars(arg, free_vars);
            // The continuation value must be captured by lambdas
            if let Some(id) = local_id {
                free_vars.insert(*id);
            }
        }
        ExprKind::List(elements) => {
            for elem in elements {
                collect_free_vars(elem, free_vars);
            }
        }
    }
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
    effect: Option<TypeRef>,
    param_ir_types: &[TypeRef],
    result_ir_ty: TypeRef,
) -> Option<ValueRef> {
    // Step 1: Analyze captures
    let captures = analyze_captures(builder.ctx, builder.ir, params, body);

    // Step 2: Build lambda body region.
    // Entry block has only the lambda's formal parameters — evidence/env are
    // added later by the lower_closure_lambda pass.
    let any_ty = builder.ctx.anyref_type(builder.ir);
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

    let entry_block = builder.ir.create_block(BlockData {
        location,
        args: block_args,
        ops: Default::default(),
        parent_region: None,
    });

    // Enter scope for lambda body
    builder.ctx.enter_scope();

    // Bind lambda parameters
    for (i, param) in params.iter().enumerate() {
        if let Some(local_id) = param.local_id {
            let arg_val = builder.ir.block_arg(entry_block, i as u32);
            builder.ctx.bind(local_id, param.name, arg_val);
        }
    }

    // No need to rebind captures — they are already in scope from the parent.
    // The closure.lambda body is NOT isolated from above, so parent-scope
    // ValueRefs are valid inside the body region.

    // Lower the lambda body
    {
        let mut inner_builder = IrBuilder::new(builder.ctx, builder.ir, entry_block);
        if let Some(result) = super::expr::lower_expr(&mut inner_builder, body.clone()) {
            let result = inner_builder.cast_if_needed(location, result, result_ir_ty);
            let ret_op = func::r#return(inner_builder.ir, location, [result]);
            inner_builder
                .ir
                .push_op(inner_builder.block, ret_op.op_ref());
        } else {
            let nil = inner_builder.emit_nil(location);
            let ret_op = func::r#return(inner_builder.ir, location, [nil]);
            inner_builder
                .ir
                .push_op(inner_builder.block, ret_op.op_ref());
        }
    }

    builder.ctx.exit_scope();

    let body_region = builder.ir.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![entry_block],
        parent_op: None,
    });

    // Step 3: Emit closure.lambda
    let capture_values: Vec<ValueRef> = captures.iter().map(|c| c.value).collect();
    let closure_func_ty =
        builder
            .ctx
            .func_type_with_effect(builder.ir, param_ir_types, result_ir_ty, effect);
    let closure_ty = builder.ctx.closure_type(builder.ir, closure_func_ty);

    let lambda_op = arena_closure::lambda(
        builder.ir,
        location,
        capture_values,
        closure_ty,
        body_region,
    );
    builder.ir.push_op(builder.block, lambda_op.op_ref());
    Some(lambda_op.result(builder.ir))
}

/// Wrap a named function reference as a closure value.
///
/// Generates a thin wrapper function with `(evidence, env, params...) -> result`
/// signature that forwards the call to the original function (ignoring env),
/// then emits `closure.new @wrapper, nil`.
pub(super) fn wrap_func_as_closure(
    builder: &mut IrBuilder<'_, '_>,
    location: Location,
    func_name: Symbol,
    param_ir_types: &[TypeRef],
    result_ir_ty: TypeRef,
) -> ValueRef {
    let any_ty = builder.ctx.anyref_type(builder.ir);
    let evidence_ty = arena_ability::evidence_adt_type_ref(builder.ir);

    // Generate unique wrapper name
    let wrapper_name = builder.ctx.gen_lambda_name();

    // Block args: [evidence, env, param1, param2, ...]
    let mut block_args = Vec::new();
    {
        let mut arg = BlockArgData {
            ty: evidence_ty,
            attrs: Default::default(),
        };
        arg.attrs.insert(
            Symbol::new("bind_name"),
            Attribute::Symbol(Symbol::new("__evidence")),
        );
        block_args.push(arg);
    }
    {
        let mut arg = BlockArgData {
            ty: any_ty,
            attrs: Default::default(),
        };
        arg.attrs.insert(
            Symbol::new("bind_name"),
            Attribute::Symbol(Symbol::new("__env")),
        );
        block_args.push(arg);
    }
    for (i, &ty) in param_ir_types.iter().enumerate() {
        let mut arg = BlockArgData {
            ty,
            attrs: Default::default(),
        };
        arg.attrs.insert(
            Symbol::new("bind_name"),
            Attribute::Symbol(Symbol::from_dynamic(&format!("__arg_{}", i))),
        );
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

    // Forward call: func.call @func_name(params...) — skip evidence and env
    // Coerce arguments to match callee's declared parameter types (handles
    // polymorphic callees where wrapper params are concrete but callee expects `any`).
    let mut call_args: Vec<ValueRef> = arg_values[2..].to_vec();
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
    let call_op = func::call(builder.ir, location, call_args, result_ir_ty, func_name);
    builder.ir.push_op(entry_block, call_op.op_ref());
    let call_result = call_op.result(builder.ir);

    let ret_op = func::r#return(builder.ir, location, [call_result]);
    builder.ir.push_op(entry_block, ret_op.op_ref());

    // Create region and wrapper func op
    let body_region = builder.ir.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![entry_block],
        parent_op: None,
    });

    let mut all_param_types = vec![evidence_ty, any_ty];
    all_param_types.extend_from_slice(param_ir_types);
    let wrapper_func_ty =
        builder
            .ctx
            .func_type_with_effect(builder.ir, &all_param_types, result_ir_ty, None);

    let func_op = func::func(
        builder.ir,
        location,
        wrapper_name,
        wrapper_func_ty,
        body_region,
    );

    let module_block = builder
        .ctx
        .module_block()
        .expect("module block should be set");
    builder.ir.push_op(module_block, func_op.op_ref());

    // Emit closure.new @wrapper, null_env at the call site
    let any_ty = builder.ctx.anyref_type(builder.ir);
    let null_op = adt::ref_null(builder.ir, location, any_ty, any_ty);
    builder.ir.push_op(builder.block, null_op.op_ref());
    let null_env = null_op.result(builder.ir);
    let closure_func_ty = builder
        .ctx
        .func_type(builder.ir, param_ir_types, result_ir_ty);
    let closure_ty = builder.ctx.closure_type(builder.ir, closure_func_ty);
    let closure_op = arena_closure::new(builder.ir, location, null_env, closure_ty, wrapper_name);
    builder.ir.push_op(builder.block, closure_op.op_ref());
    closure_op.result(builder.ir)
}
