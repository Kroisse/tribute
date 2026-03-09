//! Expression lowering.
//!
//! Transforms AST expressions to arena TrunkIR operations.

use std::collections::{HashMap, HashSet};

use salsa::Accumulator;
use tribute_core::diagnostic::{CompilationPhase, Diagnostic, DiagnosticSeverity};
use trunk_ir::Symbol;
use trunk_ir::dialect::{adt, arith, cont, core, func};
use trunk_ir::refs::ValueRef;
use trunk_ir::types::{Attribute, Location};

use crate::ast::{BinOpKind, Expr, ExprKind, ResolvedRef, Stmt, TypeKind, TypedRef};

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
            let op = arith::r#const(
                builder.ir,
                location,
                i32_ty,
                Attribute::IntBits(value as u64),
            );
            builder.ir.push_op(builder.block, op.op_ref());
            let result = op.result(builder.ir);

            Some(result)
        }

        ExprKind::IntLit(n) => {
            let value = super::validate_int_i31(builder.db(), location, n)?;
            let i32_ty = builder.ctx.i32_type(builder.ir);
            let op = arith::r#const(
                builder.ir,
                location,
                i32_ty,
                Attribute::IntBits(value as u64),
            );
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
                Attribute::IntBits(c as i32 as u64),
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
                let func_name = Symbol::from_dynamic(&id.qualified_name(builder.db()).to_string());
                let func_ty = builder.ctx.convert_type(builder.ir, typed_ref.ty);
                let op = func::constant(builder.ir, location, func_ty, func_name);
                builder.ir.push_op(builder.block, op.op_ref());
                let result = op.result(builder.ir);

                Some(result)
            }
            ResolvedRef::Constructor { variant, .. } => {
                match typed_ref.ty.kind(builder.db()) {
                    TypeKind::Func { .. } => {
                        // Constructor with args used as a first-class function value
                        let func_ty = builder.ctx.convert_type(builder.ir, typed_ref.ty);
                        let op = func::constant(builder.ir, location, func_ty, *variant);
                        builder.ir.push_op(builder.block, op.op_ref());
                        let result = op.result(builder.ir);

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
                    let result_ty = builder.ctx.any_type(builder.ir);
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
            let any_ty = builder.ctx.any_type(builder.ir);
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
                None => builder.ctx.any_type(builder.ir),
            };
            let any_ty = builder.ctx.any_type(builder.ir);

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
            let any_ty = builder.ctx.any_type(builder.ir);
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
            let effect_row = node_ty.and_then(|ty| {
                if let TypeKind::Func { effect, .. } = ty.kind(db) {
                    if !effect.is_pure(db) {
                        Some(*effect)
                    } else {
                        None
                    }
                } else {
                    None
                }
            });
            let effect_ty = effect_row.map(|row| builder.ctx.convert_effect_row(builder.ir, row));
            super::lambda::lower_lambda(builder, location, &params, &body, effect_ty)
        }

        ExprKind::Handle { body, handlers } => {
            let handle_val = super::handle::lower_handle(builder, location, &body, &handlers)?;
            let node_ty = builder.ctx.get_node_type(expr_node_id).copied();
            if let Some(ty) = node_ty {
                let ir_ty = builder.ctx.convert_type(builder.ir, ty);
                let any_ty = builder.ctx.any_type(builder.ir);
                if ir_ty != any_ty {
                    let cast_op =
                        core::unrealized_conversion_cast(builder.ir, location, handle_val, ir_ty);
                    builder.ir.push_op(builder.block, cast_op.op_ref());
                    return Some(cast_op.result(builder.ir));
                }
            }
            Some(handle_val)
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

/// Lower a block expression (let bindings + value).
fn lower_block<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    stmts: Vec<Stmt<TypedRef<'db>>>,
    value: Expr<TypedRef<'db>>,
) -> Option<ValueRef> {
    builder.ctx.enter_scope();

    for stmt in stmts {
        match stmt {
            Stmt::Let {
                id: _,
                pattern,
                ty: _,
                value,
            } => {
                if let Some(val) = lower_expr(builder, value) {
                    match &*pattern.kind {
                        crate::ast::PatternKind::Bind {
                            name,
                            local_id: Some(local_id),
                        } => {
                            builder.ctx.bind(*local_id, *name, val);
                        }
                        crate::ast::PatternKind::Wildcard => {
                            // Value computed for side effects only
                        }
                        crate::ast::PatternKind::Tuple(_) => {
                            let location = builder.ctx.location(pattern.id);
                            bind_pattern_fields(
                                builder.ctx,
                                builder.ir,
                                builder.block,
                                location,
                                val,
                                &pattern,
                            );
                        }
                        _ => {
                            let location = builder.ctx.location(pattern.id);
                            Diagnostic {
                                message: "pattern destructuring not yet supported in IR lowering"
                                    .to_string(),
                                span: location.span,
                                severity: DiagnosticSeverity::Warning,
                                phase: CompilationPhase::Lowering,
                            }
                            .accumulate(builder.db());
                        }
                    }
                }
            }
            Stmt::Expr { id: _, expr } => {
                let _ = lower_expr(builder, expr);
            }
        }
    }

    let result = lower_expr(builder, value);
    builder.ctx.exit_scope();
    result
}
