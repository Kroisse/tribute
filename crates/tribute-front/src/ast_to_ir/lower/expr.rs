//! Expression lowering.
//!
//! Transforms AST expressions to arena TrunkIR operations.

use std::collections::{HashMap, HashSet};

use salsa::Accumulator;
use tribute_core::diagnostic::{CompilationPhase, Diagnostic, DiagnosticSeverity};
use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::{adt, arith, core, func};
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
            let string_ty = builder.ctx.anyref_type(builder.ir);
            let op = adt::string_const(builder.ir, location, string_ty, s.clone());
            builder.ir.push_op(builder.block, op.op_ref());
            let result = op.result(builder.ir);

            Some(result)
        }

        ExprKind::BytesLit(ref bytes) => {
            let bytes_ty = builder.ctx.bytes_type(builder.ir);
            let op = adt::bytes_const(builder.ir, location, bytes_ty, bytes.clone().into());
            builder.ir.push_op(builder.block, op.op_ref());
            let result = op.result(builder.ir);

            Some(result)
        }

        ExprKind::Nil => Some(builder.emit_nil(location)),

        ExprKind::Var(ref typed_ref) => match &typed_ref.resolved {
            ResolvedRef::Local { id, .. } => builder.ctx.lookup(*id),
            ResolvedRef::Function { id } => {
                let db = builder.db();
                let func_name = id.qualified(db);
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
            ResolvedRef::AbilityOp { ability, op, .. } => {
                Diagnostic::new(
                    format!(
                        "ability operation `{}::{}` cannot be used as a value; it must be called directly",
                        ability.qualified(builder.db()), op
                    ),
                    location.span,
                    DiagnosticSeverity::Error,
                    CompilationPhase::Lowering,
                )
                .accumulate(builder.db());
                None
            }
        },

        ExprKind::BinOp { op, lhs, rhs } => {
            // Check operand type (lhs), not result type, since comparisons
            // return Bool but need to know if operands are float.
            let is_float = builder
                .ctx
                .get_node_type(lhs.id)
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
                        let callee_name = id.qualified(builder.db());

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
                                let op = ability::resume(
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
                    ResolvedRef::AbilityOp { ability, op, kind } => {
                        let qualified_name = ability.qualified(builder.db()).to_string();
                        let ability_name = Symbol::from_dynamic(&qualified_name);
                        let result_ty = builder
                            .ctx
                            .get_node_type(expr_node_id)
                            .map(|t| builder.ctx.convert_type(builder.ir, *t))
                            .unwrap_or_else(|| builder.call_result_type(&typed_ref.ty));
                        use crate::ast::OpDeclKind;
                        match kind {
                            OpDeclKind::Fn => super::handle::lower_ability_fn_call(
                                builder,
                                location,
                                ability_name,
                                *op,
                                arg_values,
                                result_ty,
                            ),
                            OpDeclKind::Op => super::handle::lower_ability_op_call(
                                builder,
                                location,
                                ability_name,
                                *op,
                                arg_values,
                                result_ty,
                            ),
                        }
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
                    Diagnostic::new(
                        format!("unknown field `{}` for struct `{}`", name, struct_name),
                        location.span,
                        DiagnosticSeverity::Error,
                        CompilationPhase::Lowering,
                    )
                    .accumulate(db);
                    continue;
                }

                if field_map.contains_key(&name) {
                    Diagnostic::new(
                        format!("duplicate field `{}`", name),
                        location.span,
                        DiagnosticSeverity::Error,
                        CompilationPhase::Lowering,
                    )
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
                    Diagnostic::new(
                        format!("missing field: {}", field_name),
                        location.span,
                        DiagnosticSeverity::Error,
                        CompilationPhase::Lowering,
                    )
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
                    // Effectful lambdas with concrete abilities use anyref as
                    // their return type. This ensures the CPS handler chain
                    // (which passes boxed values) has consistent types.
                    // Call sites using these closures via func.call_indirect
                    // will get anyref and must cast to the expected type.
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

            // Cast arg to anyref if needed
            let arg_cast = if builder.ir.value_ty(arg_val) != anyref_ty {
                let cast =
                    core::unrealized_conversion_cast(builder.ir, location, arg_val, anyref_ty);
                builder.ir.push_op(builder.block, cast.op_ref());
                cast.result(builder.ir)
            } else {
                arg_val
            };

            // Cast k_val from anyref to closure type so closure_lower can
            // properly decompose it (extract fn_ptr + env and add evidence).
            let closure_func_ty =
                builder
                    .ctx
                    .func_type_with_effect(builder.ir, &[anyref_ty], anyref_ty, None);
            let closure_ty = builder.ctx.closure_type(builder.ir, closure_func_ty);
            let k_cast = core::unrealized_conversion_cast(builder.ir, location, k_val, closure_ty);
            builder.ir.push_op(builder.block, k_cast.op_ref());

            let call_op = func::call_indirect(
                builder.ir,
                location,
                k_cast.result(builder.ir),
                vec![arg_cast],
                anyref_ty,
            );
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

    macro_rules! emit_cmpi {
        ($predicate:expr) => {{
            let op = arith::cmpi(
                builder.ir,
                location,
                lhs,
                rhs,
                bool_ty,
                Symbol::new($predicate),
            );
            builder.ir.push_op(builder.block, op.op_ref());
            op.result(builder.ir)
        }};
    }

    macro_rules! emit_cmpf {
        ($predicate:expr) => {{
            let op = arith::cmpf(
                builder.ir,
                location,
                lhs,
                rhs,
                bool_ty,
                Symbol::new($predicate),
            );
            builder.ir.push_op(builder.block, op.op_ref());
            op.result(builder.ir)
        }};
    }

    let result = match op {
        BinOpKind::Add if is_float => emit_binop!(arith::addf, result_ty),
        BinOpKind::Add => emit_binop!(arith::addi, result_ty),
        BinOpKind::Sub if is_float => emit_binop!(arith::subf, result_ty),
        BinOpKind::Sub => emit_binop!(arith::subi, result_ty),
        BinOpKind::Mul if is_float => emit_binop!(arith::mulf, result_ty),
        BinOpKind::Mul => emit_binop!(arith::muli, result_ty),
        BinOpKind::Div if is_float => emit_binop!(arith::divf, result_ty),
        BinOpKind::Div => emit_binop!(arith::divsi, result_ty),
        BinOpKind::Mod => emit_binop!(arith::remsi, result_ty),
        BinOpKind::Eq if is_float => emit_cmpf!("oeq"),
        BinOpKind::Eq => emit_cmpi!("eq"),
        BinOpKind::Ne if is_float => emit_cmpf!("one"),
        BinOpKind::Ne => emit_cmpi!("ne"),
        BinOpKind::Lt if is_float => emit_cmpf!("olt"),
        BinOpKind::Lt => emit_cmpi!("slt"),
        BinOpKind::Le if is_float => emit_cmpf!("ole"),
        BinOpKind::Le => emit_cmpi!("sle"),
        BinOpKind::Gt if is_float => emit_cmpf!("ogt"),
        BinOpKind::Gt => emit_cmpi!("sgt"),
        BinOpKind::Ge if is_float => emit_cmpf!("oge"),
        BinOpKind::Ge => emit_cmpi!("sge"),
        BinOpKind::And => emit_binop!(arith::and, bool_ty),
        BinOpKind::Or => emit_binop!(arith::or, bool_ty),
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
                Diagnostic::new(
                    "ability operation in nested expression position is not yet supported; \
                          extract to a let binding",
                    location.span,
                    DiagnosticSeverity::Error,
                    CompilationPhase::Lowering,
                )
                .accumulate(builder.db());
                return None;
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
    let mut scope = builder.ctx.scope();
    // Reborrow builder fields through the scope guard.
    let builder = &mut IrBuilder::new(&mut scope, builder.ir, builder.block);

    let mut stmts_iter = stmts.into_iter().peekable();

    while let Some(stmt) = stmts_iter.peek() {
        if is_direct_ability_op_stmt(stmt) {
            let stmt = stmts_iter.next().unwrap();
            let remaining: Vec<_> = stmts_iter.collect();
            return lower_cps_ability_op(builder, stmt, remaining, value).map(|r| (r, true));
        }

        let stmt = stmts_iter.next().unwrap();
        lower_single_stmt(builder, stmt);
    }

    // Check if the value expression is a direct ability op call
    if let Some(result) = try_lower_value_ability_op(builder, &value) {
        return Some((result, true));
    }

    lower_expr(builder, value).map(|r| (r, false))
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
    let ResolvedRef::AbilityOp { ability, op, kind } = &typed_ref.resolved else {
        return None;
    };

    // fn operations use direct call (no CPS), so they are not handled here.
    if *kind == crate::ast::OpDeclKind::Fn {
        return None;
    }

    let call_expr_id = value.id;
    let location = builder.location(call_expr_id);

    // Lower ability op arguments
    let ExprKind::Call { args, .. } = *value.clone().kind else {
        unreachable!();
    };
    let mut arg_values = builder.collect_args(args)?;

    // Pack multiple arguments into a tuple if needed
    arg_values = pack_ability_args(builder, location, arg_values);

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
    let ability_name = ability.qualified(builder.db());
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

/// Check if an expression contains a nested CPS ability op call (not at the top level).
///
/// Returns true if any subexpression (argument, operand, etc.) is a call to
/// an `op` ability operation (which requires CPS). `fn` operations are excluded
/// because they use direct calls without CPS.
fn contains_nested_ability_op<'db>(expr: &Expr<TypedRef<'db>>) -> bool {
    match &*expr.kind {
        ExprKind::Call { callee, args } => {
            // Check args (not callee — a direct call is fine, handled elsewhere)
            let callee_is_cps_ability_op = matches!(
                &*callee.kind,
                ExprKind::Var(tr) if matches!(
                    &tr.resolved,
                    ResolvedRef::AbilityOp { kind: crate::ast::OpDeclKind::Op, .. }
                )
            );
            if !callee_is_cps_ability_op {
                // Non-CPS-ability-op call: check callee (e.g. foo(bar)(emit()))
                // and args for nested ability ops
                contains_ability_op(callee) || args.iter().any(contains_ability_op)
            } else {
                // Direct CPS ability op call at top level — not "nested"
                false
            }
        }
        ExprKind::BinOp { lhs, rhs, .. } => contains_ability_op(lhs) || contains_ability_op(rhs),
        ExprKind::Tuple(elems) => elems.iter().any(contains_ability_op),
        ExprKind::Block { stmts, value } => {
            stmts.iter().any(|s| match s {
                Stmt::Let { value, .. } => contains_ability_op(value),
                Stmt::Expr { expr, .. } => contains_ability_op(expr),
            }) || contains_ability_op(value)
        }
        ExprKind::Case { scrutinee, arms } => {
            contains_ability_op(scrutinee) || arms.iter().any(|a| contains_ability_op(&a.body))
        }
        ExprKind::Record { fields, .. } => fields.iter().any(|(_, v)| contains_ability_op(v)),
        _ => false,
    }
}

/// Check if an expression IS or CONTAINS an `op` ability op call (CPS-requiring).
///
/// `fn` operations use direct calls and do not trigger CPS transformation.
fn contains_ability_op<'db>(expr: &Expr<TypedRef<'db>>) -> bool {
    match &*expr.kind {
        ExprKind::Call { callee, args } => {
            if let ExprKind::Var(tr) = &*callee.kind
                && matches!(
                    &tr.resolved,
                    ResolvedRef::AbilityOp {
                        kind: crate::ast::OpDeclKind::Op,
                        ..
                    }
                )
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

/// Check if a statement contains a direct ability op call that requires CPS.
///
/// `fn` operations are excluded because they use direct calls (no CPS).
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
    matches!(
        &tr.resolved,
        ResolvedRef::AbilityOp {
            kind: crate::ast::OpDeclKind::Op,
            ..
        }
    )
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
            Diagnostic::new(
                "pattern destructuring not yet supported in IR lowering",
                location.span,
                DiagnosticSeverity::Warning,
                CompilationPhase::Lowering,
            )
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
    let ResolvedRef::AbilityOp { ability, op, .. } = &typed_ref.resolved else {
        unreachable!("ICE: lower_cps_ability_op called with non-ability-op callee");
    };

    // Lower ability op arguments
    let mut arg_values = builder.collect_args(args)?;

    // Pack multiple arguments into a tuple if needed (matching lower_ability_op_call)
    arg_values = pack_ability_args(builder, location, arg_values);

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
    let ability_name = ability.qualified(builder.db());
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

    {
        let mut scope = builder.ctx.scope();

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
            let mut inner_builder = IrBuilder::new(&mut scope, builder.ir, entry_block);
            bind_stmt_pattern(&mut inner_builder, pattern, typed_param);
        }

        // Lower remaining computation inside the continuation body
        let result = {
            let mut inner_builder = IrBuilder::new(&mut scope, builder.ir, entry_block);
            lower_block_cps(&mut inner_builder, remaining_stmts, value)
        };
        let Some((body_result, is_cps)) = result else {
            return None; // scope drops automatically
        };

        // Emit return: pass through result directly (no YieldResult wrapping)
        let mut inner_builder = IrBuilder::new(&mut scope, builder.ir, entry_block);
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

/// Pack multiple ability op arguments into a single anyref tuple if needed.
fn pack_ability_args(
    builder: &mut IrBuilder<'_, '_>,
    location: Location,
    arg_values: Vec<ValueRef>,
) -> Vec<ValueRef> {
    if arg_values.len() > 1 {
        let any_ty = builder.ctx.anyref_type(builder.ir);
        let tuple_ty = ability_args_tuple_type(builder.ir, arg_values.len());
        let tuple_op = adt::struct_new(builder.ir, location, arg_values, any_ty, tuple_ty);
        builder.ir.push_op(builder.block, tuple_op.op_ref());
        vec![tuple_op.result(builder.ir)]
    } else {
        arg_values
    }
}

/// Create an `adt.struct` type for packing multiple ability operation arguments.
///
/// The struct has N fields named `_0`, `_1`, ..., all typed as `anyref`.
/// Because the type is interned, calling this with the same `num_fields`
/// always returns the same `TypeRef`.
pub(super) fn ability_args_tuple_type(ir: &mut IrContext, num_fields: usize) -> TypeRef {
    use trunk_ir::types::TypeDataBuilder;

    let anyref_ty = tribute_ir::dialect::tribute_rt::anyref(ir).as_type_ref();
    let fields_attr: Vec<Attribute> = (0..num_fields)
        .map(|i| {
            Attribute::List(vec![
                Attribute::Symbol(Symbol::from_dynamic(&format!("_{i}"))),
                Attribute::Type(anyref_ty),
            ])
        })
        .collect();

    ir.types.intern(
        TypeDataBuilder::new(Symbol::new("adt"), Symbol::new("struct"))
            .attr(
                "name",
                Attribute::Symbol(Symbol::new("__ability_args_tuple")),
            )
            .attr("fields", Attribute::List(fields_attr))
            .build(),
    )
}
