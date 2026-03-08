//! Lambda/closure lowering.
//!
//! Performs closure conversion: captures analysis, lifted function generation,
//! and closure.new emission.

use std::collections::HashSet;

use trunk_ir::Symbol;
use trunk_ir::context::{BlockArgData, BlockData, IrContext, RegionData};
use trunk_ir::dialect::{adt, func};
use trunk_ir::refs::{TypeRef, ValueRef};
use trunk_ir::types::{Attribute, Location};

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

    let any_ty = ctx.any_type(ir);

    let mut captures = Vec::new();
    for (local_id, name, value) in ctx.all_bindings() {
        if free_vars.contains(&local_id) && !param_ids.contains(&local_id) {
            captures.push(CaptureInfo {
                name,
                local_id,
                ty: any_ty,
                value,
            });
        }
    }

    captures
}

/// Lower a lambda expression to a closure.
///
/// Performs closure conversion:
/// 1. Analyze captures
/// 2. Generate a lifted function with `(evidence, env, params...)` signature
/// 3. Push the lifted function to the module block
/// 4. Create an env struct with captured values
/// 5. Emit `closure.new` at the call site
pub(super) fn lower_lambda<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    location: Location,
    params: &[Param],
    body: &Expr<TypedRef<'db>>,
    effect: Option<TypeRef>,
) -> Option<ValueRef> {
    // Step 1: Analyze captures
    let captures = analyze_captures(builder.ctx, builder.ir, params, body);

    // Step 2: Generate unique name for the lifted function
    let lifted_name = builder.ctx.gen_lambda_name();

    // Step 3: Build the lifted function
    let any_ty = builder.ctx.any_type(builder.ir);
    let evidence_ty = arena_ability::evidence_adt_type_ref(builder.ir);

    // Build env struct type if captures exist
    let concrete_env_ty = if captures.is_empty() {
        None
    } else {
        let fields: Vec<(Symbol, TypeRef)> = captures
            .iter()
            .enumerate()
            .map(|(i, cap)| (Symbol::from_dynamic(&format!("_{}", i)), cap.ty))
            .collect();
        let env_name = Symbol::from_dynamic(&format!("{}::env", lifted_name));
        Some(builder.ctx.adt_struct_type(builder.ir, env_name, &fields))
    };

    // Param args: [evidence, env, param1, param2, ...]
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
    for param in params {
        let mut arg = BlockArgData {
            ty: any_ty,
            attrs: Default::default(),
        };
        arg.attrs
            .insert(Symbol::new("bind_name"), Attribute::Symbol(param.name));
        block_args.push(arg);
    }

    let total_params = block_args.len();

    // Create entry block for the lifted function
    let entry_block = builder.ir.create_block(BlockData {
        location,
        args: block_args,
        ops: Default::default(),
        parent_region: None,
    });
    let arg_values: Vec<ValueRef> = (0..total_params)
        .map(|i| builder.ir.block_arg(entry_block, i as u32))
        .collect();

    // arg_values[0] = evidence, arg_values[1] = env, arg_values[2..] = params
    let raw_env_value = arg_values[1];

    // Cast env from anyref to concrete struct type if captures exist
    let env_value = if let Some(env_struct_ty) = concrete_env_ty {
        let cast_op = adt::ref_cast(
            builder.ir,
            location,
            raw_env_value,
            env_struct_ty,
            env_struct_ty,
        );
        builder.ir.push_op(entry_block, cast_op.op_ref());
        Some(cast_op.result(builder.ir))
    } else {
        None
    };

    // Enter scope for lambda body
    builder.ctx.enter_scope();

    // Bind lambda parameters (skip evidence and env at indices 0 and 1)
    for (i, param) in params.iter().enumerate() {
        if let Some(local_id) = param.local_id {
            let arg_val = arg_values[i + 2];
            builder.ctx.bind(local_id, param.name, arg_val);
        }
    }

    // Extract captured values from env and bind them
    if let (Some(env_val), Some(env_struct_ty)) = (env_value, concrete_env_ty) {
        for (i, cap) in captures.iter().enumerate() {
            let extracted = adt::struct_get(
                builder.ir,
                location,
                env_val,
                cap.ty,
                env_struct_ty,
                i as u32,
            );
            builder.ir.push_op(entry_block, extracted.op_ref());
            let extracted_val = extracted.result(builder.ir);
            builder.ctx.bind(cap.local_id, cap.name, extracted_val);
        }
    }

    // Lower the lambda body
    {
        let mut inner_builder = IrBuilder::new(builder.ctx, builder.ir, entry_block);
        if let Some(result) = super::expr::lower_expr(&mut inner_builder, body.clone()) {
            let result = inner_builder.cast_if_needed(location, result, any_ty);
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

    // Create region and func op
    let body_region = builder.ir.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![entry_block],
        parent_op: None,
    });

    // Build function type: (evidence, env, params...) -> any
    let mut all_param_types = vec![evidence_ty, any_ty];
    for _ in params {
        all_param_types.push(any_ty);
    }
    let func_ty = builder
        .ctx
        .func_type_with_effect(builder.ir, &all_param_types, any_ty, effect);

    let func_op = func::func(builder.ir, location, lifted_name, func_ty, body_region);

    // Push lifted function to module block (in-place, no lifted_functions vec)
    let module_block = builder
        .ctx
        .module_block()
        .expect("module block should be set");
    builder.ir.push_op(module_block, func_op.op_ref());

    // Step 4: Create env struct with captured values at the call site
    let capture_values: Vec<ValueRef> = captures.iter().map(|c| c.value).collect();
    let closure_env = if captures.is_empty() {
        builder.emit_nil(location)
    } else {
        let struct_ty = concrete_env_ty.unwrap();
        let struct_op = adt::struct_new(builder.ir, location, capture_values, struct_ty, struct_ty);
        builder.ir.push_op(builder.block, struct_op.op_ref());
        struct_op.result(builder.ir)
    };

    // Step 5: Create closure.new
    let closure_func_ty = {
        let param_tys: Vec<TypeRef> = (0..params.len()).map(|_| any_ty).collect();
        builder.ctx.func_type(builder.ir, &param_tys, any_ty)
    };
    let closure_ty = builder.ctx.closure_type(builder.ir, closure_func_ty);

    let closure_op = arena_closure::new(builder.ir, location, closure_env, closure_ty, lifted_name);
    builder.ir.push_op(builder.block, closure_op.op_ref());
    Some(closure_op.result(builder.ir))
}
