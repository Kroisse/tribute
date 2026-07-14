//! Expression lowering.
//!
//! Transforms AST expressions to arena TrunkIR operations.

use std::collections::{HashMap, HashSet};

use salsa::Accumulator;
use tribute_core::diagnostic::{CompilationPhase, Diagnostic, DiagnosticSeverity};
use tribute_core::set_calling_convention;
use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::{adt, arith, core, func, scf};
use trunk_ir::refs::{TypeRef, ValueRef};
use trunk_ir::types::{Attribute, Location};
use trunk_ir::{BlockData, RegionData};

use crate::ast::{
    BinOpKind, Expr, ExprKind, Pattern, PatternKind, ResolvedRef, Stmt, TypeKind, TypedRef,
};

use tribute_ir::dialect::{ability, closure};

use super::case::bind_pattern_fields;
use super::{
    IrBuilder, extract_ctor_id, extract_type_name, get_or_create_tuple_type, qualified_type_name,
    resolve_enum_type_attr,
};
use crate::ast::CallingConvention;

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
                    None,
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
                        let result = super::lambda::wrap_func_as_closure(
                            builder, location, *variant, &p, r, None,
                        );
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
            ResolvedRef::Module { .. }
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
            // Short-circuit evaluation:
            //   a && b → scf.if(a, then={yield b}, else={yield false})
            //   a || b → scf.if(a, then={yield true}, else={yield b})
            let lhs_val = lower_expr(builder, lhs)?;
            let bool_ty = builder.ctx.bool_type(builder.ir);

            let (then_region, else_region) = match op {
                BinOpKind::And => {
                    let then_region =
                        build_short_circuit_rhs_region(builder, location, bool_ty, rhs);
                    let else_region =
                        build_short_circuit_const_region(builder.ir, location, bool_ty, false);
                    (then_region, else_region)
                }
                BinOpKind::Or => {
                    let then_region =
                        build_short_circuit_const_region(builder.ir, location, bool_ty, true);
                    let else_region =
                        build_short_circuit_rhs_region(builder, location, bool_ty, rhs);
                    (then_region, else_region)
                }
            };

            let if_op = scf::r#if(
                builder.ir,
                location,
                lhs_val,
                bool_ty,
                then_region,
                else_region,
            );
            builder.ir.push_op(builder.block, if_op.op_ref());
            Some(if_op.result(builder.ir))
        }

        ExprKind::Block { stmts, value } => lower_block(builder, stmts, value),

        ExprKind::Call { callee, args } => {
            let arg_exprs = args;
            let mut arg_values = builder.collect_args(arg_exprs.clone())?;

            match *callee.kind {
                ExprKind::Var(ref typed_ref) => match &typed_ref.resolved {
                    ResolvedRef::Function { id } => {
                        let callee_name = id.qualified(builder.db());

                        // Insert casts for arguments if we have type scheme information
                        adapt_named_function_args(
                            builder,
                            location,
                            callee_name,
                            &arg_exprs,
                            &mut arg_values,
                        );
                        cast_args_from_signature(builder, location, callee_name, &mut arg_values);

                        let result_ty = builder.call_result_type(&typed_ref.ty);
                        let convention = builder
                            .ctx
                            .function_calling_convention(callee_name)
                            .unwrap_or_else(|| {
                                builder
                                    .ctx
                                    .calling_convention_for_type(typed_ref.ty)
                                    .unwrap_or(CallingConvention::Direct)
                            });
                        let call_result_ty = match convention {
                            CallingConvention::Direct => result_ty,
                            CallingConvention::EvidenceDirect => {
                                let evidence = super::get_or_create_evidence(builder, location);
                                arg_values.insert(0, evidence);
                                result_ty
                            }
                            CallingConvention::Cps => {
                                let evidence = super::get_or_create_evidence(builder, location);
                                let done_k = builder.ctx.done_k.unwrap_or_else(|| {
                                    super::create_identity_done_k(builder, location)
                                });
                                arg_values.insert(0, done_k);
                                arg_values.insert(0, evidence);
                                builder.ctx.anyref_type(builder.ir)
                            }
                        };
                        let op = func::call(
                            builder.ir,
                            location,
                            arg_values,
                            call_result_ty,
                            callee_name,
                        );
                        set_calling_convention(builder.ir, op.op_ref(), convention);
                        builder.ir.push_op(builder.block, op.op_ref());
                        let result = op.result(builder.ir);
                        let result = builder.cast_if_needed(location, result, result_ty);

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
                                // Continuation closures use internal convention: fn(result) -> anyref
                                let anyref_ty = builder.ctx.anyref_type(builder.ir);
                                let closure_func_ty =
                                    builder.ctx.func_type(builder.ir, &[anyref_ty], anyref_ty);
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
                                set_calling_convention(
                                    builder.ir,
                                    op.op_ref(),
                                    CallingConvention::Direct,
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

                            let convention = calling_convention_for_type(builder.ctx, typed_ref.ty);
                            let expected_ty = builder.call_result_type(&typed_ref.ty);
                            let call_result_ty = if convention.needs_done_k() {
                                builder.ctx.anyref_type(builder.ir)
                            } else {
                                expected_ty
                            };
                            let mut hidden_args = Vec::new();
                            if convention.needs_evidence() {
                                hidden_args.push(super::get_or_create_evidence(builder, location));
                            }
                            if convention.needs_done_k() {
                                let done_k = builder.ctx.done_k.unwrap_or_else(|| {
                                    super::create_identity_done_k(builder, location)
                                });
                                hidden_args.push(done_k);
                            }
                            hidden_args.append(&mut arg_values);
                            let closure_param_types: Vec<_> = hidden_args
                                .iter()
                                .map(|v| builder.ir.value_ty(*v))
                                .collect();
                            let closure_func_ty = builder.ctx.func_type(
                                builder.ir,
                                &closure_param_types,
                                call_result_ty,
                            );
                            let closure_ty = builder.ctx.closure_type(builder.ir, closure_func_ty);
                            let callee = builder.cast_if_needed(location, callee_val, closure_ty);
                            let op = func::call_indirect(
                                builder.ir,
                                location,
                                callee,
                                hidden_args,
                                call_result_ty,
                            );
                            set_calling_convention(builder.ir, op.op_ref(), convention);
                            builder.ir.push_op(builder.block, op.op_ref());
                            let result = op.result(builder.ir);
                            let result = builder.cast_if_needed(location, result, expected_ty);
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
                    let callee_node_id = callee.id;
                    let callee_val = lower_expr(builder, callee)?;
                    let convention = builder
                        .ctx
                        .get_node_type(callee_node_id)
                        .copied()
                        .map(|ty| calling_convention_for_type(builder.ctx, ty))
                        .unwrap_or(CallingConvention::Cps);
                    let expected_ty = builder
                        .ctx
                        .get_node_type(expr_node_id)
                        .map(|t| builder.ctx.convert_type(builder.ir, *t))
                        .unwrap_or_else(|| builder.ctx.anyref_type(builder.ir));
                    let call_result_ty = if convention.needs_done_k() {
                        builder.ctx.anyref_type(builder.ir)
                    } else {
                        expected_ty
                    };
                    let mut hidden_args = Vec::new();
                    if convention.needs_evidence() {
                        hidden_args.push(super::get_or_create_evidence(builder, location));
                    }
                    if convention.needs_done_k() {
                        hidden_args.push(
                            builder.ctx.done_k.unwrap_or_else(|| {
                                super::create_identity_done_k(builder, location)
                            }),
                        );
                    }
                    hidden_args.append(&mut arg_values);
                    let closure_param_types: Vec<_> = hidden_args
                        .iter()
                        .map(|v| builder.ir.value_ty(*v))
                        .collect();
                    let closure_func_ty =
                        builder
                            .ctx
                            .func_type(builder.ir, &closure_param_types, call_result_ty);
                    let closure_ty = builder.ctx.closure_type(builder.ir, closure_func_ty);
                    let callee = builder.cast_if_needed(location, callee_val, closure_ty);
                    let op = func::call_indirect(
                        builder.ir,
                        location,
                        callee,
                        hidden_args,
                        call_result_ty,
                    );
                    set_calling_convention(builder.ir, op.op_ref(), convention);
                    builder.ir.push_op(builder.block, op.op_ref());
                    let result = op.result(builder.ir);
                    let result = builder.cast_if_needed(location, result, expected_ty);

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
            let (param_ir_types, result_ir_ty, convention) = match node_ty.map(|t| (t, t.kind(db)))
            {
                Some((
                    func_ty,
                    TypeKind::Func {
                        params: p, result, ..
                    },
                )) => {
                    let pir: Vec<_> = p
                        .iter()
                        .map(|t| builder.ctx.convert_type(builder.ir, *t))
                        .collect();
                    let convention = builder
                        .ctx
                        .calling_convention_for_type(func_ty)
                        .expect("lambda node type must be a function");
                    let rir = if convention == CallingConvention::Cps {
                        builder.ctx.anyref_type(builder.ir)
                    } else {
                        builder.ctx.convert_type(builder.ir, *result)
                    };
                    (pir, rir, convention)
                }
                _ => {
                    let any = builder.ctx.anyref_type(builder.ir);
                    (vec![any; params.len()], any, CallingConvention::Cps)
                }
            };
            super::lambda::lower_lambda(
                builder,
                location,
                &params,
                &body,
                &param_ir_types,
                result_ir_ty,
                convention,
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
            let closure_func_ty = builder.ctx.func_type(builder.ir, &[anyref_ty], anyref_ty);
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

fn evaluated_expr_any<'db>(
    expr: &Expr<TypedRef<'db>>,
    predicate: &mut impl FnMut(&Expr<TypedRef<'db>>) -> bool,
) -> bool {
    if predicate(expr) {
        return true;
    }

    match &*expr.kind {
        ExprKind::Call { callee, args } => {
            evaluated_expr_any(callee, predicate)
                || args.iter().any(|arg| evaluated_expr_any(arg, predicate))
        }
        ExprKind::Cons { args, .. } | ExprKind::Tuple(args) | ExprKind::List(args) => {
            args.iter().any(|arg| evaluated_expr_any(arg, predicate))
        }
        ExprKind::Record { fields, spread, .. } => {
            spread
                .as_ref()
                .is_some_and(|expr| evaluated_expr_any(expr, predicate))
                || fields
                    .iter()
                    .any(|(_, expr)| evaluated_expr_any(expr, predicate))
        }
        ExprKind::Block { stmts, value } => {
            stmts.iter().any(|stmt| match stmt {
                Stmt::Let { value, .. } => evaluated_expr_any(value, predicate),
                Stmt::Expr { expr, .. } => evaluated_expr_any(expr, predicate),
            }) || evaluated_expr_any(value, predicate)
        }
        ExprKind::BinOp { lhs, rhs, .. } => {
            evaluated_expr_any(lhs, predicate) || evaluated_expr_any(rhs, predicate)
        }
        ExprKind::Case { scrutinee, arms } => {
            evaluated_expr_any(scrutinee, predicate)
                || arms.iter().any(|arm| {
                    arm.guard
                        .as_ref()
                        .is_some_and(|guard| evaluated_expr_any(guard, predicate))
                        || evaluated_expr_any(&arm.body, predicate)
                })
        }
        ExprKind::Resume { arg, .. } => evaluated_expr_any(arg, predicate),
        ExprKind::MethodCall { receiver, args, .. } => {
            evaluated_expr_any(receiver, predicate)
                || args.iter().any(|arg| evaluated_expr_any(arg, predicate))
        }
        // These introduce independently lowered evaluation domains.
        ExprKind::Lambda { .. } | ExprKind::Handle { .. } => false,
        _ => false,
    }
}

/// Check whether evaluating an expression can execute a call that needs CPS.
///
/// This does not descend into lambdas or handlers, which establish their own
/// lowering domains.
pub(super) fn contains_cps_call_in_evaluation<'db>(
    ctx: &super::super::context::IrLoweringCtx<'db>,
    expr: &Expr<TypedRef<'db>>,
) -> bool {
    evaluated_expr_any(expr, &mut |expr| is_cps_call_expr(ctx, expr))
}

/// Lower a function body expression with CPS transformation for effectful functions.
///
/// Like `lower_block_cps_for_expr`, but also handles top-level effectful function calls
/// (not just ability ops) as CPS points.
pub(super) fn lower_block_cps_for_body<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    expr: Expr<TypedRef<'db>>,
) -> Option<(ValueRef, bool)> {
    // Effectful call detection is already integrated into lower_block_cps.
    lower_block_cps_for_expr(builder, expr)
}

/// CPS-lower an expression, using an empty statement list for non-block forms.
///
/// This keeps direct-call handling and nested-call lifting in the same block
/// CPS state machine instead of maintaining a second expression-only path.
pub(super) fn lower_block_cps_for_expr<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    expr: Expr<TypedRef<'db>>,
) -> Option<(ValueRef, bool)> {
    match *expr.kind {
        ExprKind::Block { stmts, value } => lower_block_cps(builder, stmts, value),
        _ => lower_block_cps(builder, vec![], expr),
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
        if can_lower_cps_call_in_current_context(builder.ctx)
            && let Some((lifted_stmt, rebuilt_stmt)) =
                lift_nested_cps_call_from_stmt(builder.ctx, stmt.clone())
        {
            let mut remaining = vec![rebuilt_stmt];
            remaining.extend(stmts_iter.skip(1));
            return if is_direct_ability_op_stmt(&lifted_stmt) {
                lower_cps_ability_op(builder, lifted_stmt, remaining, value).map(|r| (r, true))
            } else {
                lower_cps_call(builder, lifted_stmt, remaining, value).map(|r| (r, true))
            };
        }

        if is_direct_ability_op_stmt(stmt) {
            let stmt = stmts_iter.next().unwrap();
            let remaining: Vec<_> = stmts_iter.collect();
            return lower_cps_ability_op(builder, stmt, remaining, value).map(|r| (r, true));
        }

        // CPS-transform calls whose selected convention is Cps when done_k is set.
        // Skip in handler arm bodies (cps_handler_mode) — need scf.yield terminator.
        if can_lower_cps_call_in_current_context(builder.ctx) && is_cps_call_stmt(builder.ctx, stmt)
        {
            let stmt = stmts_iter.next().unwrap();
            let remaining: Vec<_> = stmts_iter.collect();
            return lower_cps_call(builder, stmt, remaining, value).map(|r| (r, true));
        }

        let stmt = stmts_iter.next().unwrap();
        lower_single_stmt(builder, stmt);
    }

    // Check if the value expression is a direct ability op call
    if let Some(result) = try_lower_value_ability_op(builder, &value) {
        return Some((result, true));
    }

    // Check if the value expression is a named Cps function call with done_k.
    // If so, treat it as a CPS call: done_k is passed to the callee, which will
    // call it internally. The caller must NOT call done_k again (is_cps = true).
    // Skip in handler arm context (cps_handler_mode) — result flows to scf.yield.
    if can_lower_cps_call_in_current_context(builder.ctx)
        && let ExprKind::Call { callee: c, .. } = &*value.kind
        && let ExprKind::Var(tr) = &*c.kind
        && !matches!(&tr.resolved, ResolvedRef::AbilityOp { .. })
        && callee_requires_cps_by_definition(builder.ctx, tr)
        && let Some(result) = try_lower_value_effectful_call(builder, value.clone())
    {
        return Some((result, true));
    }
    // Local closure calls only participate in the CPS chain when their row
    // requires the CPS convention.
    if can_lower_cps_call_in_current_context(builder.ctx)
        && let ExprKind::Call { callee: c, .. } = &*value.kind
        && let ExprKind::Var(tr) = &*c.kind
        && matches!(&tr.resolved, ResolvedRef::Local { .. })
        && !matches!(&tr.ty.kind(builder.db()), TypeKind::Continuation { .. })
        && calling_convention_for_type(builder.ctx, tr.ty) == CallingConvention::Cps
    {
        let result = lower_expr(builder, value)?;
        return Some((result, true));
    }
    if can_lower_cps_call_in_current_context(builder.ctx) && is_cps_call_expr(builder.ctx, &value) {
        let result = lower_expr(builder, value)?;
        return Some((result, true));
    }

    if can_lower_cps_call_in_current_context(builder.ctx)
        && let Some((lifted_stmt, rebuilt_value)) =
            lift_nested_cps_call(builder.ctx, value.clone(), false)
    {
        return if is_direct_ability_op_stmt(&lifted_stmt) {
            lower_cps_ability_op(builder, lifted_stmt, vec![], rebuilt_value).map(|r| (r, true))
        } else {
            lower_cps_call(builder, lifted_stmt, vec![], rebuilt_value).map(|r| (r, true))
        };
    }

    lower_expr(builder, value).map(|r| (r, false))
}

fn can_lower_cps_call_in_current_context(ctx: &super::super::context::IrLoweringCtx<'_>) -> bool {
    ctx.done_k.is_some() && !ctx.cps_handler_mode
}

fn calling_convention_for_type<'db>(
    ctx: &super::super::context::IrLoweringCtx<'db>,
    ty: crate::ast::Type<'db>,
) -> CallingConvention {
    ctx.calling_convention_for_type(ty)
        .unwrap_or(CallingConvention::Cps)
}

fn is_cps_call_expr<'db>(
    ctx: &super::super::context::IrLoweringCtx<'db>,
    expr: &Expr<TypedRef<'db>>,
) -> bool {
    let ExprKind::Call { callee, .. } = &*expr.kind else {
        return false;
    };
    let ExprKind::Var(tr) = &*callee.kind else {
        return ctx
            .get_node_type(callee.id)
            .copied()
            .is_none_or(|ty| calling_convention_for_type(ctx, ty) == CallingConvention::Cps);
    };
    match &tr.resolved {
        ResolvedRef::AbilityOp {
            kind: crate::ast::OpDeclKind::Op,
            ..
        } => true,
        ResolvedRef::AbilityOp { .. } => false,
        ResolvedRef::Local { .. } => {
            !matches!(tr.ty.kind(ctx.db), TypeKind::Continuation { .. })
                && calling_convention_for_type(ctx, tr.ty) == CallingConvention::Cps
        }
        _ => callee_requires_cps_by_definition(ctx, tr),
    }
}

fn make_lifted_call<'db>(
    ctx: &mut super::super::context::IrLoweringCtx<'db>,
    call: Expr<TypedRef<'db>>,
) -> Option<(Stmt<TypedRef<'db>>, Expr<TypedRef<'db>>)> {
    let ty = *ctx.get_node_type(call.id)?;
    let local_id = ctx.next_local_id();
    let name = Symbol::new("__cps_tmp");
    let pattern = Pattern::new(
        call.id,
        PatternKind::Bind {
            name,
            local_id: Some(local_id),
        },
    );
    let replacement = Expr::new(
        call.id,
        ExprKind::Var(TypedRef::new(ResolvedRef::local(local_id, name), ty)),
    );
    let stmt = Stmt::Let {
        id: call.id,
        pattern,
        ty: None,
        value: call,
    };
    Some((stmt, replacement))
}

fn lift_nested_cps_call_from_stmt<'db>(
    ctx: &mut super::super::context::IrLoweringCtx<'db>,
    stmt: Stmt<TypedRef<'db>>,
) -> Option<(Stmt<TypedRef<'db>>, Stmt<TypedRef<'db>>)> {
    match stmt {
        Stmt::Let {
            id,
            pattern,
            ty,
            value,
        } => {
            let (lifted, value) = lift_nested_cps_call(ctx, value, false)?;
            Some((
                lifted,
                Stmt::Let {
                    id,
                    pattern,
                    ty,
                    value,
                },
            ))
        }
        Stmt::Expr { id, expr } => {
            let (lifted, expr) = lift_nested_cps_call(ctx, expr, false)?;
            Some((lifted, Stmt::Expr { id, expr }))
        }
    }
}

fn lift_nested_cps_call<'db>(
    ctx: &mut super::super::context::IrLoweringCtx<'db>,
    expr: Expr<TypedRef<'db>>,
    include_self: bool,
) -> Option<(Stmt<TypedRef<'db>>, Expr<TypedRef<'db>>)> {
    let id = expr.id;
    let original = expr.clone();
    let rebuilt = match *expr.kind {
        ExprKind::Call { callee, mut args } => {
            if let Some((lifted, callee)) = lift_nested_cps_call(ctx, callee.clone(), true) {
                return Some((lifted, Expr::new(id, ExprKind::Call { callee, args })));
            }
            for index in 0..args.len() {
                if let Some((lifted, arg)) = lift_nested_cps_call(ctx, args[index].clone(), true) {
                    args[index] = arg;
                    return Some((lifted, Expr::new(id, ExprKind::Call { callee, args })));
                }
            }
            Expr::new(id, ExprKind::Call { callee, args })
        }
        ExprKind::Cons { ctor, mut args } => {
            for index in 0..args.len() {
                if let Some((lifted, arg)) = lift_nested_cps_call(ctx, args[index].clone(), true) {
                    args[index] = arg;
                    return Some((lifted, Expr::new(id, ExprKind::Cons { ctor, args })));
                }
            }
            Expr::new(id, ExprKind::Cons { ctor, args })
        }
        ExprKind::Tuple(mut elements) => {
            for index in 0..elements.len() {
                if let Some((lifted, element)) =
                    lift_nested_cps_call(ctx, elements[index].clone(), true)
                {
                    elements[index] = element;
                    return Some((lifted, Expr::new(id, ExprKind::Tuple(elements))));
                }
            }
            Expr::new(id, ExprKind::Tuple(elements))
        }
        ExprKind::Record {
            type_name,
            mut fields,
            spread,
        } => {
            let mut spread = spread;
            if let Some(spread_expr) = spread.clone()
                && let Some((lifted, replacement)) = lift_nested_cps_call(ctx, spread_expr, true)
            {
                spread = Some(replacement);
                return Some((
                    lifted,
                    Expr::new(
                        id,
                        ExprKind::Record {
                            type_name,
                            fields,
                            spread,
                        },
                    ),
                ));
            }
            for index in 0..fields.len() {
                if let Some((lifted, field)) =
                    lift_nested_cps_call(ctx, fields[index].1.clone(), true)
                {
                    fields[index].1 = field;
                    return Some((
                        lifted,
                        Expr::new(
                            id,
                            ExprKind::Record {
                                type_name,
                                fields,
                                spread,
                            },
                        ),
                    ));
                }
            }
            Expr::new(
                id,
                ExprKind::Record {
                    type_name,
                    fields,
                    spread,
                },
            )
        }
        ExprKind::BinOp { op, lhs, rhs } => {
            if let Some((lifted, lhs)) = lift_nested_cps_call(ctx, lhs.clone(), true) {
                return Some((lifted, Expr::new(id, ExprKind::BinOp { op, lhs, rhs })));
            }
            Expr::new(id, ExprKind::BinOp { op, lhs, rhs })
        }
        ExprKind::Case { scrutinee, arms } => {
            if let Some((lifted, scrutinee)) = lift_nested_cps_call(ctx, scrutinee.clone(), true) {
                return Some((lifted, Expr::new(id, ExprKind::Case { scrutinee, arms })));
            }
            Expr::new(id, ExprKind::Case { scrutinee, arms })
        }
        _ => original.clone(),
    };

    if include_self && is_cps_call_expr(ctx, &rebuilt) {
        make_lifted_call(ctx, rebuilt)
    } else {
        None
    }
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

    // Build continuation for the value expression.
    // If done_k is set (inside an effectful function), use done_k directly
    // since there's nothing after this ability op call.
    // Otherwise, build a trivial identity continuation: fn(done_k, result) { return result }
    let continuation = if let Some(done_k) = builder.ctx.done_k {
        done_k
    } else {
        super::create_identity_done_k(builder, location)
    };

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

/// Try to lower a value expression that is an effectful function call.
///
/// When done_k is set (inside an effectful function), the effectful call receives
/// done_k as its last argument, making it a CPS tail call.
fn try_lower_value_effectful_call<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    expr: Expr<TypedRef<'db>>,
) -> Option<ValueRef> {
    let ExprKind::Call { callee, args } = *expr.kind else {
        return None;
    };
    let ExprKind::Var(typed_ref) = *callee.kind else {
        return None;
    };
    let callee_name = match &typed_ref.resolved {
        ResolvedRef::Function { id } => id.qualified(builder.db()),
        _ => return None,
    };

    let location = builder.location(expr.id);
    let arg_exprs = args;
    let mut arg_values = builder.collect_args(arg_exprs.clone())?;

    // Insert casts for arguments using type scheme information
    adapt_named_function_args(builder, location, callee_name, &arg_exprs, &mut arg_values);
    cast_args_from_signature(builder, location, callee_name, &mut arg_values);

    let anyref_ty = builder.ctx.anyref_type(builder.ir);

    // Get evidence and done_k — the caller's evidence and continuation
    let evidence = super::get_or_create_evidence(builder, location);
    let done_k = builder.ctx.done_k?;
    let mut cps_args = vec![evidence, done_k];
    cps_args.append(&mut arg_values);

    let call_op = func::call(builder.ir, location, cps_args, anyref_ty, callee_name);
    set_calling_convention(builder.ir, call_op.op_ref(), CallingConvention::Cps);
    builder.ir.push_op(builder.block, call_op.op_ref());
    let call_result = call_op.result(builder.ir);

    Some(call_result)
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

/// Check if a statement contains a call that needs CPS transformation.
///
/// This includes:
/// - Named functions whose definition selects Cps
/// - Local and computed closures whose function type selects Cps
///
/// Ability ops are excluded (handled separately by is_direct_ability_op_stmt).
fn is_cps_call_stmt<'db>(
    ctx: &super::super::context::IrLoweringCtx<'db>,
    stmt: &Stmt<TypedRef<'db>>,
) -> bool {
    let call_expr = match stmt {
        Stmt::Let { value, .. } => value,
        Stmt::Expr { expr, .. } => expr,
    };
    is_cps_call_expr(ctx, call_expr) && !is_direct_ability_op_stmt(stmt)
}

/// Check whether a named callee's definition selects Cps.
fn callee_requires_cps_by_definition<'db>(
    ctx: &super::super::context::IrLoweringCtx<'db>,
    tr: &TypedRef<'db>,
) -> bool {
    let callee_name = match &tr.resolved {
        ResolvedRef::Function { id } => id.qualified(ctx.db),
        _ => {
            // For locals/closures, use the call-site function type.
            return ctx.calling_convention_for_type(tr.ty) == Some(CallingConvention::Cps);
        }
    };
    // Look up the function's TypeScheme (definition type)
    ctx.function_calling_convention(callee_name) == Some(CallingConvention::Cps)
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

/// CPS-transform a statement containing a Cps function call.
///
/// Similar to `lower_cps_ability_op`, but for regular function calls to effectful
/// functions. Builds a continuation for the remaining computation and passes it
/// as the first argument (done_k) to the callee function or closure.
fn lower_cps_call<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    stmt: Stmt<TypedRef<'db>>,
    remaining_stmts: Vec<Stmt<TypedRef<'db>>>,
    value: Expr<TypedRef<'db>>,
) -> Option<ValueRef> {
    // Decompose statement into pattern + call expression
    let (pattern, call_expr) = match stmt {
        Stmt::Let { pattern, value, .. } => (Some(pattern), value),
        Stmt::Expr { expr, .. } => (None, expr),
    };

    let call_expr_id = call_expr.id;
    let location = builder.location(call_expr_id);

    let ExprKind::Call { callee, args } = *call_expr.kind else {
        unreachable!("ICE: lower_cps_call called with non-call expression");
    };
    let callee_id = callee.id;
    let callee_kind = *callee.kind;

    // Lower arguments
    let arg_exprs = args;
    let mut arg_values = builder.collect_args(arg_exprs.clone())?;

    // Determine the logical result type of the function call
    let logical_result_ty = builder
        .ctx
        .get_node_type(call_expr_id)
        .map(|t| builder.ctx.convert_type(builder.ir, *t))
        .unwrap_or_else(|| match &callee_kind {
            ExprKind::Var(typed_ref) => builder.call_result_type(&typed_ref.ty),
            _ => builder.ctx.anyref_type(builder.ir),
        });

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

    match callee_kind {
        ExprKind::Var(typed_ref) => match &typed_ref.resolved {
            ResolvedRef::Function { id } => {
                let callee_name = id.qualified(builder.db());

                // Insert casts for arguments using type scheme information
                adapt_named_function_args(
                    builder,
                    location,
                    callee_name,
                    &arg_exprs,
                    &mut arg_values,
                );
                cast_args_from_signature(builder, location, callee_name, &mut arg_values);

                // Call the Cps function with evidence + continuation as hidden args.
                let evidence = super::get_or_create_evidence(builder, location);
                let mut cps_args = vec![evidence, continuation];
                cps_args.append(&mut arg_values);

                let call_op = func::call(builder.ir, location, cps_args, anyref_ty, callee_name);
                set_calling_convention(builder.ir, call_op.op_ref(), CallingConvention::Cps);
                builder.ir.push_op(builder.block, call_op.op_ref());
                Some(call_op.result(builder.ir))
            }
            ResolvedRef::Local { id, .. } => {
                let callee_val = builder.ctx.lookup(*id)?;

                // Insert casts for arguments
                if let TypeKind::Func { params, .. } = typed_ref.ty.kind(builder.db()) {
                    for (i, param_ty) in params.iter().enumerate() {
                        if i < arg_values.len() {
                            let target_ty = builder.ctx.convert_type(builder.ir, *param_ty);
                            arg_values[i] =
                                builder.cast_if_needed(location, arg_values[i], target_ty);
                        }
                    }
                }

                // Cast callee to CPS closure type so closure_lower extracts
                // the correct return type (anyref, not the source-level type).
                let evidence = super::get_or_create_evidence(builder, location);
                let mut cps_param_types = vec![builder.ir.value_ty(evidence), anyref_ty];
                cps_param_types.extend(arg_values.iter().map(|v| builder.ir.value_ty(*v)));
                let cps_func_ty = builder
                    .ctx
                    .func_type(builder.ir, &cps_param_types, anyref_ty);
                let cps_closure_ty = builder.ctx.closure_type(builder.ir, cps_func_ty);
                let callee_cps = builder.cast_if_needed(location, callee_val, cps_closure_ty);

                // Call closure with continuation as first arg (done_k)
                let mut cps_args = vec![evidence, continuation];
                cps_args.append(&mut arg_values);

                let call_op =
                    func::call_indirect(builder.ir, location, callee_cps, cps_args, anyref_ty);
                set_calling_convention(builder.ir, call_op.op_ref(), CallingConvention::Cps);
                builder.ir.push_op(builder.block, call_op.op_ref());
                Some(call_op.result(builder.ir))
            }
            _ => unreachable!("ICE: lower_cps_call with unexpected callee kind"),
        },
        callee_kind => {
            let callee = Expr::new(callee_id, callee_kind);
            let callee_val = lower_expr(builder, callee)?;
            let evidence = super::get_or_create_evidence(builder, location);
            let mut cps_param_types = vec![builder.ir.value_ty(evidence), anyref_ty];
            cps_param_types.extend(arg_values.iter().map(|v| builder.ir.value_ty(*v)));
            let cps_func_ty = builder
                .ctx
                .func_type(builder.ir, &cps_param_types, anyref_ty);
            let cps_closure_ty = builder.ctx.closure_type(builder.ir, cps_func_ty);
            let callee_cps = builder.cast_if_needed(location, callee_val, cps_closure_ty);

            let mut cps_args = vec![evidence, continuation];
            cps_args.append(&mut arg_values);
            let call_op =
                func::call_indirect(builder.ir, location, callee_cps, cps_args, anyref_ty);
            set_calling_convention(builder.ir, call_op.op_ref(), CallingConvention::Cps);
            builder.ir.push_op(builder.block, call_op.op_ref());
            Some(call_op.result(builder.ir))
        }
    }
}

/// Build a CPS continuation closure for the remaining computation after an
/// ability op call.
///
/// The continuation is a `closure.lambda` whose body parameter is the ability
/// op's result, bound to the given pattern. The body contains the remaining
/// statements and value expression. Pure results are wrapped in
/// `YieldResult::Done`; CPS results (from nested ability.perform) are
/// returned directly.
/// Add an internal context value (evidence or done_k) to the capture list
/// if present and not already captured.
/// Cast call arguments to match the callee's declared parameter types.
///
/// Looks up the callee's TypeScheme and inserts `unrealized_conversion_cast`
/// for any argument whose IR type doesn't match the declared parameter type.
fn adapt_named_function_args<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    location: Location,
    callee_name: Symbol,
    arg_exprs: &[Expr<TypedRef<'db>>],
    arg_values: &mut [ValueRef],
) {
    let Some(scheme) = builder.ctx.lookup_function_type(callee_name) else {
        return;
    };
    let TypeKind::Func {
        params: expected_params,
        ..
    } = scheme.body(builder.ctx.db).kind(builder.ctx.db)
    else {
        return;
    };
    let expected_params = expected_params.clone();

    for (i, (arg_expr, expected_ty)) in arg_exprs.iter().zip(expected_params.iter()).enumerate() {
        let ExprKind::Var(typed_ref) = &*arg_expr.kind else {
            continue;
        };
        let ResolvedRef::Function { id } = &typed_ref.resolved else {
            continue;
        };
        let TypeKind::Func { params, result, .. } = typed_ref.ty.kind(builder.ctx.db) else {
            continue;
        };
        if !matches!(expected_ty.kind(builder.ctx.db), TypeKind::Func { .. }) {
            continue;
        }
        let func_name = id.qualified(builder.ctx.db);
        let source_convention = builder
            .ctx
            .function_calling_convention(func_name)
            .unwrap_or_default();
        let target_convention = builder
            .ctx
            .calling_convention_for_type(*expected_ty)
            .expect("expected parameter type must be a function");
        if source_convention == target_convention {
            continue;
        }
        let param_ir_types: Vec<_> = params
            .iter()
            .map(|ty| builder.ctx.convert_type(builder.ir, *ty))
            .collect();
        let result_ir_ty = builder.ctx.convert_type(builder.ir, *result);
        arg_values[i] = super::lambda::wrap_func_as_closure(
            builder,
            location,
            func_name,
            &param_ir_types,
            result_ir_ty,
            Some(target_convention),
        );
    }
}

fn cast_args_from_signature(
    builder: &mut IrBuilder<'_, '_>,
    location: Location,
    callee_name: Symbol,
    arg_values: &mut [ValueRef],
) {
    if let Some(sig) = super::FuncSignature::lookup(builder.ctx, builder.ir, callee_name) {
        for (i, target_ty) in sig.param_types.iter().enumerate() {
            if i < arg_values.len() {
                arg_values[i] = builder.cast_if_needed(location, arg_values[i], *target_ty);
            }
        }
    }
}

fn capture_ctx_value(
    captures: &mut Vec<super::super::context::CaptureInfo>,
    builder: &IrBuilder<'_, '_>,
    name: Symbol,
    value: Option<ValueRef>,
) {
    if let Some(val) = value
        && !captures.iter().any(|c| c.value == val)
    {
        captures.push(super::super::context::CaptureInfo {
            name,
            local_id: crate::ast::LocalId::UNRESOLVED,
            ty: builder.ir.value_ty(val),
            value: val,
        });
    }
}

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
    let mut captures = super::lambda::analyze_continuation_captures(
        builder.ctx,
        builder.ir,
        &remaining_stmts,
        &value,
        &excluded_ids,
    );

    // Inside effectful functions, capture done_k so it's available
    // at the end of the continuation chain after lambda lifting.
    // Evidence is NOT captured here — it's provided by the lifted function's
    // own evidence parameter (added by lower_closure_lambda).
    capture_ctx_value(
        &mut captures,
        builder,
        Symbol::new("__done_k"),
        builder.ctx.done_k,
    );

    // Build body region with one parameter: the ability op result (anyref).
    // Continuation closures are internal mechanism closures, not user lambdas,
    // so they do NOT follow user-lambda CPS convention.
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
            let result_anyref = inner_builder.cast_if_needed(location, body_result, anyref_ty);
            if let Some(done_k_val) = inner_builder.ctx.done_k {
                // Inside an effectful function: call done_k(result) instead of func.return
                super::emit_done_k_call(&mut inner_builder, location, done_k_val, result_anyref);
            } else {
                // Pure function: return as anyref
                let ret = func::r#return(inner_builder.ir, location, [result_anyref]);
                inner_builder.ir.push_op(inner_builder.block, ret.op_ref());
            }
        } else {
            // CPS result: add func.return as block terminator.
            // For ability.perform: lower_ability_perform removes dead code after it.
            let result_anyref = inner_builder.cast_if_needed(location, body_result, anyref_ty);
            let ret = func::r#return(inner_builder.ir, location, [result_anyref]);
            inner_builder.ir.push_op(inner_builder.block, ret.op_ref());
        }
    }

    let body_region = builder.ir.create_region(trunk_ir::context::RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![entry_block],
        parent_op: None,
    });

    // Closure type: fn(param_type) -> anyref
    let closure_func_ty = builder.ctx.func_type(builder.ir, &[param_type], anyref_ty);
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
pub(super) fn pack_ability_args(
    builder: &mut IrBuilder<'_, '_>,
    location: Location,
    arg_values: Vec<ValueRef>,
) -> Vec<ValueRef> {
    if arg_values.len() > 1 {
        let any_ty = builder.ctx.anyref_type(builder.ir);
        // Cast each arg to anyref (inserts box_int etc. via unrealized_conversion_cast)
        let boxed_args: Vec<ValueRef> = arg_values
            .into_iter()
            .map(|v| builder.cast_if_needed(location, v, any_ty))
            .collect();
        let tuple_ty = ability_args_tuple_type(builder.ir, boxed_args.len());
        let tuple_op = adt::struct_new(builder.ir, location, boxed_args, any_ty, tuple_ty);
        builder.ir.push_op(builder.block, tuple_op.op_ref());
        vec![tuple_op.result(builder.ir)]
    } else {
        arg_values
    }
}

/// Build a region that evaluates an expression and yields its result.
/// Used for short-circuit evaluation of boolean operators.
fn build_short_circuit_rhs_region<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    location: Location,
    bool_ty: TypeRef,
    rhs: Expr<TypedRef<'db>>,
) -> trunk_ir::refs::RegionRef {
    let block = builder.ir.create_block(BlockData {
        location,
        args: vec![],
        ops: Default::default(),
        parent_region: None,
    });

    let rhs_val = {
        let mut inner = IrBuilder::new(builder.ctx, builder.ir, block);
        let outer_done_k = inner.ctx.done_k;
        if outer_done_k.is_some() && contains_cps_call_in_evaluation(inner.ctx, &rhs) {
            let identity_done_k = super::create_identity_done_k(&mut inner, location);
            inner.ctx.done_k = Some(identity_done_k);
        }
        let result = lower_block_cps_for_expr(&mut inner, rhs).map(|(value, _)| value);
        inner.ctx.done_k = outer_done_k;
        result
    };

    let yield_val = match rhs_val {
        Some(v) => {
            let mut builder = IrBuilder::new(builder.ctx, builder.ir, block);
            builder.cast_if_needed(location, v, bool_ty)
        }
        None => {
            let op = arith::r#const(builder.ir, location, bool_ty, Attribute::Bool(false));
            builder.ir.push_op(block, op.op_ref());
            op.result(builder.ir)
        }
    };
    let yield_op = scf::r#yield(builder.ir, location, [yield_val]);
    builder.ir.push_op(block, yield_op.op_ref());

    builder.ir.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![block],
        parent_op: None,
    })
}

/// Build a region that yields a boolean constant.
/// Used for short-circuit evaluation of boolean operators.
fn build_short_circuit_const_region(
    ir: &mut IrContext,
    location: Location,
    bool_ty: TypeRef,
    value: bool,
) -> trunk_ir::refs::RegionRef {
    let block = ir.create_block(BlockData {
        location,
        args: vec![],
        ops: Default::default(),
        parent_region: None,
    });

    let const_op = arith::r#const(ir, location, bool_ty, Attribute::Bool(value));
    ir.push_op(block, const_op.op_ref());
    let yield_op = scf::r#yield(ir, location, [const_op.result(ir)]);
    ir.push_op(block, yield_op.op_ref());

    ir.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![block],
        parent_op: None,
    })
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
