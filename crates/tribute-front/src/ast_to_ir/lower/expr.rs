//! Expression lowering.
//!
//! Transforms AST expressions to arena TrunkIR operations.

use std::collections::{HashMap, HashSet};

use salsa::Accumulator;
use tribute_core::diagnostic::{CompilationPhase, Diagnostic, DiagnosticSeverity};
use trunk_ir::Symbol;
use trunk_ir::dialect::{adt, arith, cont, core, func};
use trunk_ir::refs::{TypeRef, ValueRef};
use trunk_ir::types::{Attribute, Location};

use crate::ast::{BinOpKind, Expr, ExprKind, PatternKind, ResolvedRef, Stmt, TypeKind, TypedRef};

use tribute_ir::dialect::{ability, closure};

use super::case::bind_pattern_fields;
use super::{
    IrBuilder, extract_ctor_id, extract_type_name, get_or_create_tuple_type, qualified_type_name,
    resolve_enum_type_attr,
};

/// Lower an expression to arena TrunkIR.
pub(super) fn lower_expr<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    expr: Expr<TypedRef<'db>>,
) -> Option<ValueRef> {
    let expr_node_id = expr.id;
    let location = builder.location(expr_node_id);

    match *expr.kind {
        ExprKind::NatLit(n) => {
            let value = super::validate_nat_i31(builder.db(), location, n)?;
            let i32_ty = builder.ctx.i32_type(builder.ir);
            let op = arith::r#const(builder.ir, location, i32_ty, Attribute::Int(value as i128));
            builder.ir.push_op(builder.block, op.op_ref());
            let result = op.result(builder.ir);

            Some(result)
        }

        ExprKind::IntLit(n) => {
            let value = super::validate_int_i31(builder.db(), location, n)?;
            let i32_ty = builder.ctx.i32_type(builder.ir);
            let op = arith::r#const(builder.ir, location, i32_ty, Attribute::Int(value as i128));
            builder.ir.push_op(builder.block, op.op_ref());
            let result = op.result(builder.ir);

            Some(result)
        }

        ExprKind::RuneLit(c) => {
            let i32_ty = builder.ctx.i32_type(builder.ir);
            let op = arith::r#const(
                builder.ir,
                location,
                i32_ty,
                Attribute::Int(c as i32 as i128),
            );
            builder.ir.push_op(builder.block, op.op_ref());
            let result = op.result(builder.ir);

            Some(result)
        }

        ExprKind::FloatLit(f) => {
            let f64_ty = builder.ctx.f64_type(builder.ir);
            let op = arith::r#const(
                builder.ir,
                location,
                f64_ty,
                Attribute::FloatBits(f.value().to_bits()),
            );
            builder.ir.push_op(builder.block, op.op_ref());
            let result = op.result(builder.ir);

            Some(result)
        }

        ExprKind::BoolLit(b) => {
            let bool_ty = builder.ctx.bool_type(builder.ir);
            let op = arith::r#const(builder.ir, location, bool_ty, Attribute::Bool(b));
            builder.ir.push_op(builder.block, op.op_ref());
            let result = op.result(builder.ir);

            Some(result)
        }

        ExprKind::StringLit(ref s) => {
            let string_ty = builder.ctx.string_type(builder.ir);
            let op = adt::string_const(builder.ir, location, string_ty, s.clone());
            builder.ir.push_op(builder.block, op.op_ref());
            let result = op.result(builder.ir);

            Some(result)
        }

        ExprKind::BytesLit(ref _bytes) => builder.emit_unsupported(location, "bytes literal"),

        ExprKind::Nil => Some(builder.emit_nil(location)),

        ExprKind::Var(ref typed_ref) => match &typed_ref.resolved {
            ResolvedRef::Local { id, .. } => builder.ctx.lookup(*id),
            ResolvedRef::Function { id } => {
                let db = builder.db();
                let func_name = Symbol::from_dynamic(&id.qualified_name(db).to_string());
                // Extract param/result types from the function type
                let (param_ir_types, result_ir_ty) = match typed_ref.ty.kind(db) {
                    TypeKind::Func { params, result, .. } => {
                        let p: Vec<_> = params
                            .iter()
                            .map(|t| builder.ctx.convert_type(builder.ir, *t))
                            .collect();
                        let r = builder.ctx.convert_type(builder.ir, *result);
                        (p, r)
                    }
                    _ => {
                        let any = builder.ctx.anyref_type(builder.ir);
                        (vec![], any)
                    }
                };
                let result = super::lambda::wrap_func_as_closure(
                    builder,
                    location,
                    func_name,
                    &param_ir_types,
                    result_ir_ty,
                );
                Some(result)
            }
            ResolvedRef::Constructor { variant, .. } => {
                match typed_ref.ty.kind(builder.db()) {
                    TypeKind::Func { params, result, .. } => {
                        // Constructor with args used as a first-class function value
                        let p: Vec<_> = params
                            .iter()
                            .map(|t| builder.ctx.convert_type(builder.ir, *t))
                            .collect();
                        let r = builder.ctx.convert_type(builder.ir, *result);
                        let result =
                            super::lambda::wrap_func_as_closure(builder, location, *variant, &p, r);
                        Some(result)
                    }
                    _ => {
                        // Zero-argument constructor
                        let result_ty = builder.ctx.convert_type(builder.ir, typed_ref.ty);
                        let type_attr =
                            resolve_enum_type_attr(builder.ctx, builder.ir, typed_ref.ty);
                        let op = adt::variant_new(
                            builder.ir,
                            location,
                            std::iter::empty(),
                            result_ty,
                            type_attr,
                            *variant,
                        );
                        builder.ir.push_op(builder.block, op.op_ref());
                        let result = op.result(builder.ir);

                        Some(result)
                    }
                }
            }
            ResolvedRef::Builtin(_)
            | ResolvedRef::Module { .. }
            | ResolvedRef::TypeDef { .. }
            | ResolvedRef::Ability { .. } => None,
            ResolvedRef::AbilityOp { ability, op } => {
                Diagnostic {
                    message: format!(
                        "ability operation `{}::{}` cannot be used as a value; it must be called directly",
                        ability.qualified_name(builder.db()), op
                    ),
                    span: location.span,
                    severity: DiagnosticSeverity::Error,
                    phase: CompilationPhase::Lowering,
                }
                .accumulate(builder.db());
                None
            }
        },

        ExprKind::BinOp { op, lhs, rhs } => {
            let is_float = builder
                .ctx
                .get_node_type(expr_node_id)
                .map(|ty| matches!(ty.kind(builder.db()), TypeKind::Float))
                .unwrap_or(false);
            let lhs_val = lower_expr(builder, lhs)?;
            let rhs_val = lower_expr(builder, rhs)?;
            lower_binop(builder, op, lhs_val, rhs_val, is_float, location)
        }

        ExprKind::Block { stmts, value } => lower_block(builder, stmts, value),

        ExprKind::Call { callee, args } => {
            let mut arg_values = builder.collect_args(args)?;

            match *callee.kind {
                ExprKind::Var(ref typed_ref) => match &typed_ref.resolved {
                    ResolvedRef::Function { id } => {
                        let callee_name =
                            Symbol::from_dynamic(&id.qualified_name(builder.db()).to_string());

                        let func_scheme = builder.ctx.lookup_function_type(callee_name);

                        // Insert casts for arguments if we have type scheme information
                        if let Some(scheme) = func_scheme {
                            let body = scheme.body(builder.db());
                            if let TypeKind::Func { params, .. } = body.kind(builder.db()) {
                                for (i, param_ty) in params.iter().enumerate() {
                                    if i < arg_values.len() {
                                        let target_ty =
                                            builder.ctx.convert_type(builder.ir, *param_ty);
                                        arg_values[i] = builder.cast_if_needed(
                                            location,
                                            arg_values[i],
                                            target_ty,
                                        );
                                    }
                                }
                            }
                        }

                        let result_ty = builder.call_result_type(&typed_ref.ty);
                        let op =
                            func::call(builder.ir, location, arg_values, result_ty, callee_name);
                        builder.ir.push_op(builder.block, op.op_ref());
                        let result = op.result(builder.ir);

                        Some(result)
                    }
                    ResolvedRef::Local { id, .. } => {
                        let callee_val = builder.ctx.lookup(*id)?;

                        // Check if callee is a continuation type
                        if let TypeKind::Continuation { result, .. } =
                            typed_ref.ty.kind(builder.db())
                        {
                            assert_eq!(
                                arg_values.len(),
                                1,
                                "ICE: continuation resume expects exactly 1 argument, got {}",
                                arg_values.len()
                            );
                            let resume_value = arg_values[0];

                            if builder.ctx.cps_handler_mode {
                                // CPS: tail-call continuation closure.
                                // The continuation expects anyref and returns anyref.
                                // Uses tail_call_indirect so the handler frame is freed
                                // and the return value goes directly to the handle frame.
                                let anyref_ty = builder.ctx.anyref_type(builder.ir);
                                let closure_func_ty = builder.ctx.func_type_with_effect(
                                    builder.ir,
                                    &[anyref_ty],
                                    anyref_ty,
                                    None,
                                );
                                let closure_ty =
                                    builder.ctx.closure_type(builder.ir, closure_func_ty);
                                let callee_closure =
                                    builder.cast_if_needed(location, callee_val, closure_ty);
                                let resume_anyref =
                                    builder.cast_if_needed(location, resume_value, anyref_ty);
                                let op = func::call_indirect(
                                    builder.ir,
                                    location,
                                    callee_closure,
                                    vec![resume_anyref],
                                    anyref_ty,
                                );
                                builder.ir.push_op(builder.block, op.op_ref());
                                let result = op.result(builder.ir);
                                Some(result)
                            } else {
                                let result_ty = builder.ctx.convert_type(builder.ir, *result);
                                let op = cont::resume(
                                    builder.ir,
                                    location,
                                    callee_val,
                                    resume_value,
                                    result_ty,
                                );
                                builder.ir.push_op(builder.block, op.op_ref());
                                let result = op.result(builder.ir);
                                Some(result)
                            }
                        } else {
                            // Regular call_indirect for closures
                            if let TypeKind::Func { params, .. } = typed_ref.ty.kind(builder.db()) {
                                for (i, param_ty) in params.iter().enumerate() {
                                    if i < arg_values.len() {
                                        let target_ty =
                                            builder.ctx.convert_type(builder.ir, *param_ty);
                                        arg_values[i] = builder.cast_if_needed(
                                            location,
                                            arg_values[i],
                                            target_ty,
                                        );
                                    }
                                }
                            }

                            let result_ty = builder.call_result_type(&typed_ref.ty);
                            let op = func::call_indirect(
                                builder.ir, location, callee_val, arg_values, result_ty,
                            );
                            builder.ir.push_op(builder.block, op.op_ref());
                            let result = op.result(builder.ir);

                            Some(result)
                        }
                    }
                    ResolvedRef::Constructor { variant, .. } => {
                        let result_ty = builder.call_result_type(&typed_ref.ty);
                        let type_attr =
                            resolve_enum_type_attr(builder.ctx, builder.ir, typed_ref.ty);
                        let op = adt::variant_new(
                            builder.ir, location, arg_values, result_ty, type_attr, *variant,
                        );
                        builder.ir.push_op(builder.block, op.op_ref());
                        let result = op.result(builder.ir);

                        Some(result)
                    }
                    ResolvedRef::AbilityOp { ability, op } => {
                        let qualified_name = ability.qualified_name(builder.db()).to_string();
                        let ability_name = Symbol::from_dynamic(&qualified_name);
                        let result_ty = builder
                            .ctx
                            .get_node_type(expr_node_id)
                            .map(|t| builder.ctx.convert_type(builder.ir, *t))
                            .unwrap_or_else(|| builder.call_result_type(&typed_ref.ty));
                        super::handle::lower_ability_op_call(
                            builder,
                            location,
                            ability_name,
                            *op,
                            arg_values,
                            result_ty,
                        )
                    }
                    _ => builder.emit_unsupported(location, "builtin/module call"),
                },
                _ => {
                    // General expression callee -> indirect call
                    let callee_val = lower_expr(builder, callee)?;
                    let result_ty = builder
                        .ctx
                        .get_node_type(expr_node_id)
                        .map(|t| builder.ctx.convert_type(builder.ir, *t))
                        .unwrap_or_else(|| builder.ctx.anyref_type(builder.ir));
                    let op = func::call_indirect(
                        builder.ir, location, callee_val, arg_values, result_ty,
                    );
                    builder.ir.push_op(builder.block, op.op_ref());
                    let result = op.result(builder.ir);

                    Some(result)
                }
            }
        }

        ExprKind::Cons { ctor, args } => {
            let arg_values = builder.collect_args(args)?;

            match &ctor.resolved {
                ResolvedRef::Constructor { variant, .. } => {
                    let result_ty = builder.call_result_type(&ctor.ty);
                    let type_attr = resolve_enum_type_attr(builder.ctx, builder.ir, ctor.ty);
                    let op = adt::variant_new(
                        builder.ir, location, arg_values, result_ty, type_attr, *variant,
                    );
                    builder.ir.push_op(builder.block, op.op_ref());
                    let result = op.result(builder.ir);

                    Some(result)
                }
                _ => builder.emit_unsupported(location, "non-constructor in Cons"),
            }
        }

        ExprKind::Tuple(elements) => {
            let values: Vec<_> = elements
                .iter()
                .map(|elem| lower_expr(builder, elem.clone()))
                .collect::<Option<Vec<_>>>()?;
            let any_ty = builder.ctx.anyref_type(builder.ir);
            let (result_ty, type_attr) =
                match get_or_create_tuple_type(builder.ctx, builder.ir, expr_node_id) {
                    Some((name, struct_ty)) => {
                        let rt = builder.ctx.adt_typeref(builder.ir, name);
                        (rt, struct_ty)
                    }
                    None => (any_ty, any_ty),
                };
            let op = adt::struct_new(builder.ir, location, values, result_ty, type_attr);
            builder.ir.push_op(builder.block, op.op_ref());
            let result = op.result(builder.ir);

            Some(result)
        }

        ExprKind::Record {
            type_name,
            fields,
            spread,
        } => {
            let db = builder.db();

            let spread_val = match &spread {
                Some(spread_expr) => Some(lower_expr(builder, spread_expr.clone())?),
                None => None,
            };

            let struct_name = extract_type_name(db, &type_name.resolved);
            let ctor_id = extract_ctor_id(&type_name.resolved);
            let struct_ty = builder.ctx.adt_typeref(builder.ir, struct_name);

            let field_order = builder
                .ctx
                .get_struct_field_order(ctor_id)
                .unwrap_or_else(|| {
                    panic!(
                        "ICE: struct `{}` field order not registered before IR lowering",
                        struct_name
                    )
                });
            let field_order = field_order.clone();

            let valid_fields: HashSet<Symbol> = field_order.iter().copied().collect();

            let mut field_map: HashMap<Symbol, ValueRef> = HashMap::new();
            for (name, expr) in fields {
                if !valid_fields.contains(&name) {
                    Diagnostic {
                        message: format!("unknown field `{}` for struct `{}`", name, struct_name),
                        span: location.span,
                        severity: DiagnosticSeverity::Error,
                        phase: CompilationPhase::Lowering,
                    }
                    .accumulate(db);
                    continue;
                }

                if field_map.contains_key(&name) {
                    Diagnostic {
                        message: format!("duplicate field `{}`", name),
                        span: location.span,
                        severity: DiagnosticSeverity::Error,
                        phase: CompilationPhase::Lowering,
                    }
                    .accumulate(db);
                    continue;
                }

                let val = lower_expr(builder, expr.clone())?;
                field_map.insert(name, val);
            }

            let qualified = qualified_type_name(db, &ctor_id);
            let type_attr = match builder.ctx.get_type(qualified) {
                Some(ty) => ty,
                None => builder.ctx.anyref_type(builder.ir),
            };
            let any_ty = builder.ctx.anyref_type(builder.ir);

            let mut ordered_values: Vec<ValueRef> = Vec::with_capacity(field_order.len());
            for (i, field_name) in field_order.iter().enumerate() {
                if let Some(val) = field_map.get(field_name) {
                    ordered_values.push(*val);
                } else if let Some(base) = spread_val {
                    let get_op =
                        adt::struct_get(builder.ir, location, base, any_ty, type_attr, i as u32);
                    builder.ir.push_op(builder.block, get_op.op_ref());
                    ordered_values.push(get_op.result(builder.ir));
                } else {
                    Diagnostic {
                        message: format!("missing field: {}", field_name),
                        span: location.span,
                        severity: DiagnosticSeverity::Error,
                        phase: CompilationPhase::Lowering,
                    }
                    .accumulate(db);
                    return Some(builder.emit_nil(location));
                }
            }

            let op = adt::struct_new(builder.ir, location, ordered_values, struct_ty, type_attr);
            builder.ir.push_op(builder.block, op.op_ref());
            let result = op.result(builder.ir);

            Some(result)
        }

        ExprKind::MethodCall { .. } => {
            unreachable!("MethodCall should be desugared before IR lowering")
        }

        ExprKind::Case { scrutinee, arms } => {
            let scrutinee_val = lower_expr(builder, scrutinee)?;
            let any_ty = builder.ctx.anyref_type(builder.ir);
            let mut result_ty = builder
                .ctx
                .get_node_type(expr_node_id)
                .copied()
                .map(|ty| builder.ctx.convert_type(builder.ir, ty))
                .unwrap_or(any_ty);

            if result_ty == any_ty
                && let Some(first_arm) = arms.first()
                && let Some(arm_ty) = builder.ctx.get_node_type(first_arm.body.id).copied()
            {
                let converted = builder.ctx.convert_type(builder.ir, arm_ty);
                if converted != any_ty {
                    result_ty = converted;
                }
            }

            let location = builder.location(expr_node_id);
            super::case::lower_case_chain(builder, location, scrutinee_val, result_ty, &arms, false)
        }

        ExprKind::Lambda { params, body } => {
            let db = builder.ctx.db;
            let node_ty = builder.ctx.get_node_type(expr_node_id).copied();
            debug_assert!(
                node_ty.is_some(),
                "lambda node type should be populated by typeck"
            );
            let (effect_row, param_ir_types, result_ir_ty) = match node_ty.map(|t| (t, t.kind(db)))
            {
                Some((
                    _,
                    TypeKind::Func {
                        params: p,
                        result,
                        effect,
                    },
                )) => {
                    let pir: Vec<_> = p
                        .iter()
                        .map(|t| builder.ctx.convert_type(builder.ir, *t))
                        .collect();
                    let eff = if effect.is_pure(db) {
                        None
                    } else {
                        Some(*effect)
                    };
                    // Effectful lambdas are invoked through handler prompts,
                    // which use polymorphic (boxed) types. Force the return
                    // type to `tribute_rt.anyref` so the closure boxes its
                    // result before returning, matching the `call_indirect`
                    // in `__prompt_body_N`.
                    //
                    // Only apply this when the lambda has concrete abilities
                    // (not just a tail variable from polymorphic inference).
                    let has_concrete_abilities = !effect.effects(db).is_empty();
                    let rir = if has_concrete_abilities {
                        builder.ctx.anyref_type(builder.ir)
                    } else {
                        builder.ctx.convert_type(builder.ir, *result)
                    };
                    (eff, pir, rir)
                }
                _ => {
                    let any = builder.ctx.anyref_type(builder.ir);
                    (None, vec![any; params.len()], any)
                }
            };
            let effect_ty = effect_row.map(|row| builder.ctx.convert_effect_row(builder.ir, row));
            super::lambda::lower_lambda(
                builder,
                location,
                &params,
                &body,
                effect_ty,
                &param_ir_types,
                result_ir_ty,
            )
        }

        ExprKind::Handle { body, handlers } => {
            let handle_val = super::handle::lower_handle(builder, location, &body, &handlers)?;
            let node_ty = builder.ctx.get_node_type(expr_node_id).copied();
            if let Some(ty) = node_ty {
                let ir_ty = builder.ctx.convert_type(builder.ir, ty);
                let any_ty = builder.ctx.anyref_type(builder.ir);
                if ir_ty != any_ty {
                    let cast_op =
                        core::unrealized_conversion_cast(builder.ir, location, handle_val, ir_ty);
                    builder.ir.push_op(builder.block, cast_op.op_ref());
                    return Some(cast_op.result(builder.ir));
                }
            }
            Some(handle_val)
        }

        ExprKind::Resume { arg, local_id } => {
            // In CPS mode, `resume(value)` is lowered as a call to the
            // continuation closure bound by the `op` handler arm.
            let arg_val = lower_expr(builder, arg)?;
            let Some(lid) = local_id else {
                return builder.emit_unsupported(location, "resume without local_id");
            };
            let Some(k_val) = builder.ctx.lookup(lid) else {
                return builder.emit_unsupported(location, "resume: continuation not bound");
            };
            let anyref_ty = builder.ctx.anyref_type(builder.ir);
            let call_op =
                func::call_indirect(builder.ir, location, k_val, vec![arg_val], anyref_ty);
            builder.ir.push_op(builder.block, call_op.op_ref());
            Some(call_op.result(builder.ir))
        }

        ExprKind::List(_) => builder.emit_unsupported(location, "list expression"),

        ExprKind::Error => Some(builder.emit_nil(location)),
    }
}

/// Lower a binary operation.
fn lower_binop<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    op: BinOpKind,
    lhs: ValueRef,
    rhs: ValueRef,
    is_float: bool,
    location: Location,
) -> Option<ValueRef> {
    let bool_ty = builder.ctx.bool_type(builder.ir);
    let result_ty = if is_float {
        builder.ctx.f64_type(builder.ir)
    } else {
        builder.ctx.i32_type(builder.ir)
    };

    macro_rules! emit_binop {
        ($op_fn:path, $ty:expr) => {{
            let op = $op_fn(builder.ir, location, lhs, rhs, $ty);
            builder.ir.push_op(builder.block, op.op_ref());
            op.result(builder.ir)
        }};
    }

    let result = match op {
        BinOpKind::Add => emit_binop!(arith::add, result_ty),
        BinOpKind::Sub => emit_binop!(arith::sub, result_ty),
        BinOpKind::Mul => emit_binop!(arith::mul, result_ty),
        BinOpKind::Div => emit_binop!(arith::div, result_ty),
        BinOpKind::Mod => emit_binop!(arith::rem, result_ty),
        BinOpKind::Eq => emit_binop!(arith::cmp_eq, bool_ty),
        BinOpKind::Ne => emit_binop!(arith::cmp_ne, bool_ty),
        BinOpKind::Lt => emit_binop!(arith::cmp_lt, bool_ty),
        BinOpKind::Le => emit_binop!(arith::cmp_le, bool_ty),
        BinOpKind::Gt => emit_binop!(arith::cmp_gt, bool_ty),
        BinOpKind::Ge => emit_binop!(arith::cmp_ge, bool_ty),
        BinOpKind::And => emit_binop!(arith::and, bool_ty),
        BinOpKind::Or => emit_binop!(arith::or, bool_ty),
        BinOpKind::Concat => {
            Diagnostic {
                message: "string concatenation not yet supported in IR lowering".to_string(),
                span: location.span,
                severity: DiagnosticSeverity::Warning,
                phase: CompilationPhase::Lowering,
            }
            .accumulate(builder.db());
            builder.emit_nil(location)
        }
    };

    Some(result)
}

/// CPS-lower an expression (used by handle body lowering).
///
/// Like `lower_block_cps` but works on a single expression, wrapping it in
/// a trivial block if needed.
pub(super) fn lower_block_cps_for_expr<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    expr: Expr<TypedRef<'db>>,
) -> Option<(ValueRef, bool)> {
    match *expr.kind {
        ExprKind::Block { stmts, value } => lower_block_cps(builder, stmts, value),
        _ => {
            // Non-block expression: check if it's a direct ability op call
            if let Some(result) = super::expr::try_lower_value_ability_op(builder, &expr) {
                return Some((result, true));
            }
            // Reject nested ability-op subexpressions that would bypass CPS.
            // Full expression-level CPS lifting is not yet implemented.
            if contains_nested_ability_op(&expr) {
                let location = builder.location(expr.id);
                Diagnostic {
                    message:
                        "ability operation in nested expression position is not yet supported; \
                              extract to a let binding"
                            .to_string(),
                    span: location.span,
                    severity: DiagnosticSeverity::Error,
                    phase: CompilationPhase::Lowering,
                }
                .accumulate(builder.db());
            }
            let result = lower_expr(builder, expr)?;
            Some((result, false))
        }
    }
}

/// Lower a block expression (let bindings + value).
///
/// Detects direct ability op calls in statements and applies CPS transformation:
/// the remaining computation becomes a `closure.lambda` continuation, and the
/// ability op is emitted as `ability.perform` with the continuation.
fn lower_block<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    stmts: Vec<Stmt<TypedRef<'db>>>,
    value: Expr<TypedRef<'db>>,
) -> Option<ValueRef> {
    let (result, _is_cps) = lower_block_cps(builder, stmts, value)?;
    Some(result)
}

/// Lower a block with CPS transformation.
///
/// Returns `(result_value, is_cps)`:
/// - `(value, false)` if no ability ops were found — result is a pure value
/// - `(value, true)` if the block ended with `ability.perform` — result will
///   become `YieldResult::Shift` after downstream lowering
fn lower_block_cps<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    stmts: Vec<Stmt<TypedRef<'db>>>,
    value: Expr<TypedRef<'db>>,
) -> Option<(ValueRef, bool)> {
    builder.ctx.enter_scope();

    let mut stmts_iter = stmts.into_iter().peekable();

    while let Some(stmt) = stmts_iter.peek() {
        if is_direct_ability_op_stmt(stmt) {
            let stmt = stmts_iter.next().unwrap();
            let remaining: Vec<_> = stmts_iter.collect();
            let result = lower_cps_ability_op(builder, stmt, remaining, value)?;
            builder.ctx.exit_scope();
            return Some((result, true));
        }

        let stmt = stmts_iter.next().unwrap();
        lower_single_stmt(builder, stmt);
    }

    // Check if the value expression is a direct ability op call
    if let Some(result) = try_lower_value_ability_op(builder, &value) {
        builder.ctx.exit_scope();
        return Some((result, true));
    }

    let result = lower_expr(builder, value)?;
    builder.ctx.exit_scope();
    Some((result, false))
}

/// Try to CPS-transform a value expression that is a direct ability op call.
///
/// When the block's value expression is `State::get()`, creates a trivial
/// continuation that wraps the result in `YieldResult::Done`.
fn try_lower_value_ability_op<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    value: &Expr<TypedRef<'db>>,
) -> Option<ValueRef> {
    let ExprKind::Call { callee, .. } = &*value.kind else {
        return None;
    };
    let ExprKind::Var(typed_ref) = &*callee.kind else {
        return None;
    };
    let ResolvedRef::AbilityOp { ability, op } = &typed_ref.resolved else {
        return None;
    };

    let call_expr_id = value.id;
    let location = builder.location(call_expr_id);

    // Lower ability op arguments
    let ExprKind::Call { args, .. } = *value.clone().kind else {
        unreachable!();
    };
    let mut arg_values = builder.collect_args(args)?;

    // Pack multiple arguments into a tuple if needed
    if arg_values.len() > 1 {
        let any_ty = builder.ctx.anyref_type(builder.ir);
        let tuple_op = adt::struct_new(builder.ir, location, arg_values, any_ty, any_ty);
        builder.ir.push_op(builder.block, tuple_op.op_ref());
        arg_values = vec![tuple_op.result(builder.ir)];
    }

    let anyref_ty = builder.ctx.anyref_type(builder.ir);

    // Build trivial identity continuation: fn(result) { result }
    let entry_block = builder.ir.create_block(trunk_ir::context::BlockData {
        location,
        args: vec![trunk_ir::context::BlockArgData {
            ty: anyref_ty,
            attrs: Default::default(),
        }],
        ops: Default::default(),
        parent_region: None,
    });

    let param_val = builder.ir.block_arg(entry_block, 0);
    let ret = func::r#return(builder.ir, location, [param_val]);
    builder.ir.push_op(entry_block, ret.op_ref());

    let body_region = builder.ir.create_region(trunk_ir::context::RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![entry_block],
        parent_op: None,
    });

    // Closure type: fn(anyref) -> anyref
    let closure_func_ty =
        builder
            .ctx
            .func_type_with_effect(builder.ir, &[anyref_ty], anyref_ty, None);
    let closure_ty = builder.ctx.closure_type(builder.ir, closure_func_ty);

    // No captures needed for identity continuation
    let lambda_op = closure::lambda(
        builder.ir,
        location,
        Vec::<ValueRef>::new(),
        closure_ty,
        body_region,
    );
    builder.ir.push_op(builder.block, lambda_op.op_ref());
    let continuation = lambda_op.result(builder.ir);

    // Emit ability.perform
    let ability_name = Symbol::from_dynamic(&ability.qualified_name(builder.db()).to_string());
    let ability_ref = builder.ctx.ability_ref_type(builder.ir, ability_name, &[]);

    let perform_op = ability::perform(
        builder.ir,
        location,
        continuation,
        arg_values,
        anyref_ty,
        ability_ref,
        *op,
    );
    builder.ir.push_op(builder.block, perform_op.op_ref());

    Some(perform_op.result(builder.ir))
}

/// Check if an expression contains a nested ability op call (not at the top level).
///
/// Returns true if any subexpression (argument, operand, etc.) is a call to
/// an ability operation. Direct top-level ability op calls are handled by
/// `try_lower_value_ability_op`; this catches the cases that bypass CPS.
fn contains_nested_ability_op<'db>(expr: &Expr<TypedRef<'db>>) -> bool {
    match &*expr.kind {
        ExprKind::Call { callee, args } => {
            // Check args (not callee — a direct call is fine, handled elsewhere)
            let callee_is_ability_op = matches!(
                &*callee.kind,
                ExprKind::Var(tr) if matches!(&tr.resolved, ResolvedRef::AbilityOp { .. })
            );
            if !callee_is_ability_op {
                // Non-ability-op call: check if any arg contains an ability op
                args.iter().any(contains_ability_op)
            } else {
                // Direct ability op call at top level — not "nested"
                false
            }
        }
        ExprKind::BinOp { lhs, rhs, .. } => contains_ability_op(lhs) || contains_ability_op(rhs),
        ExprKind::Tuple(elems) => elems.iter().any(contains_ability_op),
        _ => false,
    }
}

/// Check if an expression IS or CONTAINS an ability op call.
fn contains_ability_op<'db>(expr: &Expr<TypedRef<'db>>) -> bool {
    match &*expr.kind {
        ExprKind::Call { callee, args } => {
            if let ExprKind::Var(tr) = &*callee.kind
                && matches!(&tr.resolved, ResolvedRef::AbilityOp { .. })
            {
                return true;
            }
            contains_ability_op(callee) || args.iter().any(contains_ability_op)
        }
        ExprKind::BinOp { lhs, rhs, .. } => contains_ability_op(lhs) || contains_ability_op(rhs),
        ExprKind::Block { stmts, value } => {
            stmts.iter().any(|s| match s {
                Stmt::Let { value, .. } => contains_ability_op(value),
                Stmt::Expr { expr, .. } => contains_ability_op(expr),
            }) || contains_ability_op(value)
        }
        ExprKind::Tuple(elems) => elems.iter().any(contains_ability_op),
        ExprKind::Case { scrutinee, arms } => {
            contains_ability_op(scrutinee) || arms.iter().any(|a| contains_ability_op(&a.body))
        }
        _ => false,
    }
}

/// Check if a statement contains a direct ability op call.
fn is_direct_ability_op_stmt<'db>(stmt: &Stmt<TypedRef<'db>>) -> bool {
    let call_expr = match stmt {
        Stmt::Let { value, .. } => value,
        Stmt::Expr { expr, .. } => expr,
    };
    let ExprKind::Call { callee, .. } = &*call_expr.kind else {
        return false;
    };
    let ExprKind::Var(tr) = &*callee.kind else {
        return false;
    };
    matches!(&tr.resolved, ResolvedRef::AbilityOp { .. })
}

/// Lower a single non-CPS statement (let binding or expression statement).
fn lower_single_stmt<'db>(builder: &mut IrBuilder<'_, 'db>, stmt: Stmt<TypedRef<'db>>) {
    match stmt {
        Stmt::Let {
            id: _,
            pattern,
            ty: _,
            value,
        } => {
            if let Some(val) = lower_expr(builder, value) {
                bind_stmt_pattern(builder, &pattern, val);
            }
        }
        Stmt::Expr { id: _, expr } => {
            let _ = lower_expr(builder, expr);
        }
    }
}

/// Bind a value to a statement's let-pattern.
fn bind_stmt_pattern<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    pattern: &crate::ast::Pattern<TypedRef<'db>>,
    val: ValueRef,
) {
    match &*pattern.kind {
        PatternKind::Bind {
            name,
            local_id: Some(local_id),
        } => {
            builder.ctx.bind(*local_id, *name, val);
        }
        PatternKind::Wildcard => {}
        PatternKind::Tuple(_) => {
            let location = builder.ctx.location(pattern.id);
            bind_pattern_fields(
                builder.ctx,
                builder.ir,
                builder.block,
                location,
                val,
                pattern,
            );
        }
        _ => {
            let location = builder.ctx.location(pattern.id);
            Diagnostic {
                message: "pattern destructuring not yet supported in IR lowering".to_string(),
                span: location.span,
                severity: DiagnosticSeverity::Warning,
                phase: CompilationPhase::Lowering,
            }
            .accumulate(builder.db());
        }
    }
}

/// CPS-transform a statement containing a direct ability op call.
///
/// Emits `closure.lambda` (continuation for remaining computation) +
/// `ability.perform` (the ability op with explicit continuation).
fn lower_cps_ability_op<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    stmt: Stmt<TypedRef<'db>>,
    remaining_stmts: Vec<Stmt<TypedRef<'db>>>,
    value: Expr<TypedRef<'db>>,
) -> Option<ValueRef> {
    // Decompose the statement into pattern + call expression
    let (pattern, call_expr) = match stmt {
        Stmt::Let { pattern, value, .. } => (Some(pattern), value),
        Stmt::Expr { expr, .. } => (None, expr),
    };

    let call_expr_id = call_expr.id;
    let location = builder.location(call_expr_id);

    let ExprKind::Call { callee, args } = *call_expr.kind else {
        unreachable!("ICE: lower_cps_ability_op called with non-call expression");
    };
    let ExprKind::Var(typed_ref) = *callee.kind else {
        unreachable!("ICE: lower_cps_ability_op called with non-var callee");
    };
    let ResolvedRef::AbilityOp { ability, op } = &typed_ref.resolved else {
        unreachable!("ICE: lower_cps_ability_op called with non-ability-op callee");
    };

    // Lower ability op arguments
    let mut arg_values = builder.collect_args(args)?;

    // Pack multiple arguments into a tuple if needed (matching lower_ability_op_call)
    if arg_values.len() > 1 {
        let any_ty = builder.ctx.anyref_type(builder.ir);
        let tuple_op = adt::struct_new(builder.ir, location, arg_values, any_ty, any_ty);
        builder.ir.push_op(builder.block, tuple_op.op_ref());
        arg_values = vec![tuple_op.result(builder.ir)];
    }

    // Determine the logical result type of the ability op (for casting inside continuation)
    let logical_result_ty = builder
        .ctx
        .get_node_type(call_expr_id)
        .map(|t| builder.ctx.convert_type(builder.ir, *t))
        .unwrap_or_else(|| builder.call_result_type(&typed_ref.ty));

    // Continuation parameter type is always anyref — the effect handling
    // runtime passes boxed values through handler dispatch, and types like
    // core.nil are not representable in backends (Cranelift, WASM).
    let anyref_ty = builder.ctx.anyref_type(builder.ir);

    // Build continuation closure for remaining computation
    let continuation = build_cps_continuation(
        builder,
        location,
        pattern.as_ref(),
        anyref_ty,
        logical_result_ty,
        remaining_stmts,
        value,
    )?;

    // Emit ability.perform
    let ability_name = Symbol::from_dynamic(&ability.qualified_name(builder.db()).to_string());
    let ability_ref = builder.ctx.ability_ref_type(builder.ir, ability_name, &[]);
    let anyref_ty = builder.ctx.anyref_type(builder.ir);

    let perform_op = ability::perform(
        builder.ir,
        location,
        continuation,
        arg_values,
        anyref_ty,
        ability_ref,
        *op,
    );
    builder.ir.push_op(builder.block, perform_op.op_ref());

    Some(perform_op.result(builder.ir))
}

/// Build a CPS continuation closure for the remaining computation after an
/// ability op call.
///
/// The continuation is a `closure.lambda` whose body parameter is the ability
/// op's result, bound to the given pattern. The body contains the remaining
/// statements and value expression. Pure results are wrapped in
/// `YieldResult::Done`; CPS results (from nested ability.perform) are
/// returned directly.
fn build_cps_continuation<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    location: Location,
    pattern: Option<&crate::ast::Pattern<TypedRef<'db>>>,
    param_type: TypeRef,
    logical_type: TypeRef,
    remaining_stmts: Vec<Stmt<TypedRef<'db>>>,
    value: Expr<TypedRef<'db>>,
) -> Option<ValueRef> {
    let anyref_ty = builder.ctx.anyref_type(builder.ir);

    // Analyze captures: variables from current scope used in remaining computation
    let excluded_ids = HashSet::new();
    let captures = super::lambda::analyze_continuation_captures(
        builder.ctx,
        builder.ir,
        &remaining_stmts,
        &value,
        &excluded_ids,
    );

    // Build body region with one parameter: the ability op result (anyref)
    let entry_block = builder.ir.create_block(trunk_ir::context::BlockData {
        location,
        args: vec![trunk_ir::context::BlockArgData {
            ty: param_type,
            attrs: Default::default(),
        }],
        ops: Default::default(),
        parent_region: None,
    });

    builder.ctx.enter_scope();

    // Cast anyref parameter to the logical result type and bind to the pattern
    let param_val = builder.ir.block_arg(entry_block, 0);
    let typed_param = if param_type != logical_type {
        let cast_op =
            core::unrealized_conversion_cast(builder.ir, location, param_val, logical_type);
        builder.ir.push_op(entry_block, cast_op.op_ref());
        cast_op.result(builder.ir)
    } else {
        param_val
    };
    if let Some(pattern) = pattern {
        let mut inner_builder = IrBuilder::new(builder.ctx, builder.ir, entry_block);
        bind_stmt_pattern(&mut inner_builder, pattern, typed_param);
    }

    // Lower remaining computation inside the continuation body
    let (body_result, is_cps) = {
        let mut inner_builder = IrBuilder::new(builder.ctx, builder.ir, entry_block);
        lower_block_cps(&mut inner_builder, remaining_stmts, value)?
    };

    // Emit return: pass through result directly (no YieldResult wrapping)
    {
        let mut inner_builder = IrBuilder::new(builder.ctx, builder.ir, entry_block);
        if !is_cps {
            // Pure result → return as anyref
            let result_anyref = inner_builder.cast_if_needed(location, body_result, anyref_ty);
            let ret = func::r#return(inner_builder.ir, location, [result_anyref]);
            inner_builder.ir.push_op(inner_builder.block, ret.op_ref());
        } else {
            // CPS result (from nested ability.perform) → no explicit return.
            // lower_ability_perform will add tail_call when lowering the
            // ability.perform op.
        }
    }

    builder.ctx.exit_scope();

    let body_region = builder.ir.create_region(trunk_ir::context::RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![entry_block],
        parent_op: None,
    });

    // Closure type: fn(param_type) -> anyref
    let closure_func_ty =
        builder
            .ctx
            .func_type_with_effect(builder.ir, &[param_type], anyref_ty, None);
    let closure_ty = builder.ctx.closure_type(builder.ir, closure_func_ty);

    // Emit closure.lambda
    let capture_values: Vec<ValueRef> = captures.iter().map(|c| c.value).collect();
    let lambda_op = closure::lambda(
        builder.ir,
        location,
        capture_values,
        closure_ty,
        body_region,
    );
    builder.ir.push_op(builder.block, lambda_op.op_ref());
    Some(lambda_op.result(builder.ir))
}
