//! Expression lowering.
//!
//! Transforms AST expressions to arena TrunkIR operations.

use std::collections::{HashMap, HashSet};

use salsa::Accumulator;
use tribute_core::diagnostic::{CompilationPhase, Diagnostic, DiagnosticSeverity};
use tribute_core::set_calling_convention;
use trunk_ir::Symbol;
use trunk_ir::adt_layout::get_enum_variants;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::{adt, arith, core, func, scf};
use trunk_ir::refs::{TypeRef, ValueRef};
use trunk_ir::types::{Attribute, Location};
use trunk_ir::{BlockData, RegionData};

use crate::ast::{BinOpKind, Expr, ExprKind, PatternKind, ResolvedRef, Stmt, TypeKind, TypedRef};

use tribute_ir::dialect::{ability, closure, list};

use super::case::bind_pattern_fields;
use super::control::{
    ContinuationRef, ControlDomain, ControlResultRef, continuation_abi, control_from_abi,
    emit_control_return, invoke_continuation,
};
use super::{
    IrBuilder, extract_ctor_id, extract_type_name, get_or_create_tuple_type, qualified_type_name,
    resolve_enum_type_attr_for_constructor,
};
use crate::ast::CallingConvention;

/// Coerce constructor arguments to the representation recorded in the enum
/// layout. Generic enum fields are erased to `anyref`, so primitive payloads
/// must cross an explicit conversion boundary before `adt.variant_new`.
fn cast_variant_args<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    location: Location,
    args: Vec<ValueRef>,
    enum_ty: TypeRef,
    variant: Symbol,
) -> Vec<ValueRef> {
    let field_types = get_enum_variants(builder.ir, enum_ty)
        .and_then(|variants| {
            variants
                .into_iter()
                .find_map(|(tag, fields)| (tag == variant).then_some(fields))
        })
        .expect("resolved constructor must exist in enum metadata");

    assert_eq!(
        args.len(),
        field_types.len(),
        "type checking must enforce constructor arity"
    );

    args.into_iter()
        .zip(field_types)
        .map(|(arg, field_ty)| builder.cast_if_needed(location, arg, field_ty))
        .collect()
}

/// Lower a source value after the computation entry point has established
/// that evaluating it cannot transfer control.
pub(super) fn lower_value<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    expr: Expr<TypedRef<'db>>,
) -> Option<ValueRef> {
    assert_eq!(
        evaluation_control_class(builder.ctx, &expr),
        EvaluationControlClass::Direct,
        "ICE: lower_value received an expression that must be lowered through lower_comp"
    );
    lower_value_impl(builder, expr, false)
}

pub(super) fn lower_value_normalized<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    expr: Expr<TypedRef<'db>>,
) -> Option<ValueRef> {
    assert_eq!(
        evaluation_control_class(builder.ctx, &expr),
        EvaluationControlClass::Direct,
        "ICE: normalized value lowering received a CPS expression"
    );
    lower_value_impl(builder, expr, true)
}

/// Raw direct-value emission. `nested_regions_normalized` distinguishes a raw
/// Direct/EvidenceDirect entry from a value owned by the CPS normalizer.
fn lower_value_impl<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    expr: Expr<TypedRef<'db>>,
    nested_regions_normalized: bool,
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
                        let type_attr = resolve_enum_type_attr_for_constructor(
                            builder.ctx,
                            builder.ir,
                            &typed_ref.resolved,
                            typed_ref.ty,
                        );
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
            let lhs_val = lower_value_impl(builder, lhs, nested_regions_normalized)?;
            let bool_ty = builder.ctx.bool_type(builder.ir);

            let (then_region, else_region) = match op {
                BinOpKind::And => {
                    let then_region = build_short_circuit_rhs_region(
                        builder,
                        location,
                        bool_ty,
                        rhs,
                        nested_regions_normalized,
                    );
                    let else_region =
                        build_short_circuit_const_region(builder.ir, location, bool_ty, false);
                    (then_region, else_region)
                }
                BinOpKind::Or => {
                    let then_region =
                        build_short_circuit_const_region(builder.ir, location, bool_ty, true);
                    let else_region = build_short_circuit_rhs_region(
                        builder,
                        location,
                        bool_ty,
                        rhs,
                        nested_regions_normalized,
                    );
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

        ExprKind::Block { stmts, value } => {
            lower_block(builder, stmts, value, nested_regions_normalized)
        }

        ExprKind::Call { callee, args } => {
            let arg_exprs = args;
            let mut arg_values =
                lower_value_args(builder, arg_exprs.clone(), nested_regions_normalized)?;

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
                        let type_attr = resolve_enum_type_attr_for_constructor(
                            builder.ctx,
                            builder.ir,
                            &typed_ref.resolved,
                            typed_ref.ty,
                        );
                        let arg_values =
                            cast_variant_args(builder, location, arg_values, type_attr, *variant);
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
                            OpDeclKind::Op => unreachable!(
                                "ICE: general ability operations must be lowered through lower_comp"
                            ),
                        }
                    }
                    _ => builder.emit_unsupported(location, "builtin/module call"),
                },
                _ => {
                    // General expression callee -> indirect call
                    let callee_node_id = callee.id;
                    let callee_val = lower_value_impl(builder, callee, nested_regions_normalized)?;
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
            let arg_values = lower_value_args(builder, args, nested_regions_normalized)?;

            match &ctor.resolved {
                ResolvedRef::Constructor { variant, .. } => {
                    let result_ty = builder.call_result_type(&ctor.ty);
                    let type_attr = resolve_enum_type_attr_for_constructor(
                        builder.ctx,
                        builder.ir,
                        &ctor.resolved,
                        ctor.ty,
                    );
                    let arg_values =
                        cast_variant_args(builder, location, arg_values, type_attr, *variant);
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
                .map(|elem| lower_value_impl(builder, elem.clone(), nested_regions_normalized))
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
                Some(spread_expr) => Some(lower_value_impl(
                    builder,
                    spread_expr.clone(),
                    nested_regions_normalized,
                )?),
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

                let val = lower_value_impl(builder, expr.clone(), nested_regions_normalized)?;
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
            let scrutinee_val = lower_value_impl(builder, scrutinee, nested_regions_normalized)?;
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
            super::case::lower_case_chain(
                builder,
                location,
                scrutinee_val,
                result_ty,
                &arms,
                false,
                nested_regions_normalized,
            )
        }

        ExprKind::Lambda { params, body } => {
            let db = builder.ctx.db;
            let node_ty = builder.ctx.get_node_type(expr_node_id).copied();
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
                tribute_core::CallableAbi::new(
                    convention,
                    param_ir_types.iter().copied(),
                    result_ir_ty,
                ),
                nested_regions_normalized,
            )
        }

        ExprKind::Handle { body, handlers } => {
            // A source or ambient parent closes the private carrier at this
            // delimiter. `HandleAnswer` composition uses lower_comp instead.
            if !nested_regions_normalized {
                let normalized = super::super::normalize::normalize_for_cps(
                    builder.ctx,
                    Expr::new(expr_node_id, ExprKind::Handle { body, handlers }),
                );
                return lower_value_impl(builder, normalized, true);
            }
            super::handle::lower_handle_source(builder, location, expr_node_id, &body, &handlers)
        }

        ExprKind::Resume { .. } => {
            unreachable!("resume must be lowered through the current computation continuation")
        }

        ExprKind::List(elements) => {
            // Evaluate source elements before constructing the persistent
            // sequence so the reverse construction fold cannot reorder or
            // duplicate effects.
            let values = lower_value_args(builder, elements, nested_regions_normalized)?;
            let list_ast_ty = builder.ctx.get_node_type(expr_node_id).copied();
            let element_ast_ty = list_ast_ty.and_then(|ty| match ty.kind(builder.db()) {
                TypeKind::Named { id, args, .. }
                    if id.is_builtin_list(builder.db()) && args.len() == 1 =>
                {
                    Some(args[0])
                }
                _ => None,
            });
            let element_ty = element_ast_ty
                .map(|ty| builder.ctx.convert_type(builder.ir, ty))
                .unwrap_or_else(|| builder.ctx.anyref_type(builder.ir));
            let list_ty = builder.ctx.anyref_type(builder.ir);

            let empty = list::empty(builder.ir, location, list_ty, element_ty);
            builder.ir.push_op(builder.block, empty.op_ref());
            let mut result = empty.result(builder.ir);
            for value in values.into_iter().rev() {
                let prepend =
                    list::prepend(builder.ir, location, value, result, list_ty, element_ty);
                builder.ir.push_op(builder.block, prepend.op_ref());
                result = prepend.result(builder.ir);
            }
            Some(result)
        }

        ExprKind::Error => Some(builder.emit_nil(location)),
    }
}

/// Lower strict arguments without changing the caller's normalization
/// ownership. Raw Direct/EvidenceDirect entries use raw value lowering;
/// normalized computation regions keep their nested lambda/handle boundaries
/// on the normalized entry path.
fn lower_value_args<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    args: impl IntoIterator<Item = Expr<TypedRef<'db>>>,
    nested_regions_normalized: bool,
) -> Option<Vec<ValueRef>> {
    args.into_iter()
        .map(|arg| {
            if nested_regions_normalized {
                lower_value_normalized(builder, arg)
            } else {
                lower_value(builder, arg)
            }
        })
        .collect()
}

/// Lowering-only classification for the current evaluation domain.
///
/// This deliberately consumes the typed AST rather than introducing a second
/// type-checker effect table: selected calling conventions and resolved callee
/// identities already determine whether a source evaluation can transfer
/// control.  It does not inspect lambda bodies, whose effects are latent until
/// invocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum EvaluationControlClass {
    Direct,
    Cps,
}

impl EvaluationControlClass {
    fn join(self, other: Self) -> Self {
        if self == Self::Cps || other == Self::Cps {
            Self::Cps
        } else {
            Self::Direct
        }
    }
}

pub(super) fn evaluation_control_class<'db>(
    ctx: &super::super::context::IrLoweringCtx<'db>,
    expr: &Expr<TypedRef<'db>>,
) -> EvaluationControlClass {
    evaluation_control_class_with_handle_propagation(ctx, expr, false)
}

/// Classify a source-logical worker before shared CPS legalization.
///
/// Unlike the temporary legacy physical route, the logical boundary does not
/// own a private handler carrier. A handled computation therefore remains a
/// CPS evaluation until the shared legalizer constructs its continuations.
pub(super) fn logical_evaluation_control_class<'db>(
    ctx: &super::super::context::IrLoweringCtx<'db>,
    expr: &Expr<TypedRef<'db>>,
) -> EvaluationControlClass {
    evaluation_control_class_with_handle_propagation(ctx, expr, true)
}

/// Classify evaluation with an explicit nested-handle propagation policy.
///
/// Source convention selection treats handles as source delimiters. Ambient
/// and `HandleAnswer` CPS domains instead retain the private carrier so a
/// foreign Escape reaches its dynamic owner unchanged.
fn evaluation_control_class_with_handle_propagation<'db>(
    ctx: &super::super::context::IrLoweringCtx<'db>,
    expr: &Expr<TypedRef<'db>>,
    propagates_handles: bool,
) -> EvaluationControlClass {
    let children = |children: &[Expr<TypedRef<'db>>]| {
        children
            .iter()
            .fold(EvaluationControlClass::Direct, |class, child| {
                class.join(evaluation_control_class_with_handle_propagation(
                    ctx,
                    child,
                    propagates_handles,
                ))
            })
    };

    match &*expr.kind {
        ExprKind::Handle { .. } if propagates_handles => EvaluationControlClass::Cps,
        ExprKind::Lambda { .. } => EvaluationControlClass::Direct,
        // A source delimiter is Direct by default. The existing HandleAnswer
        // domain selects carrier propagation structurally below.
        ExprKind::Handle { .. } => EvaluationControlClass::Direct,
        ExprKind::Resume { .. } => EvaluationControlClass::Cps,
        ExprKind::Call { callee, args } => {
            let call = if is_cps_call_expr(ctx, expr) {
                EvaluationControlClass::Cps
            } else {
                EvaluationControlClass::Direct
            };
            call.join(evaluation_control_class_with_handle_propagation(
                ctx,
                callee,
                propagates_handles,
            ))
            .join(children(args))
        }
        ExprKind::Cons { args, .. } | ExprKind::Tuple(args) | ExprKind::List(args) => {
            children(args)
        }
        ExprKind::Record { fields, spread, .. } => {
            let spread = spread
                .as_ref()
                .map_or(EvaluationControlClass::Direct, |spread| {
                    evaluation_control_class_with_handle_propagation(
                        ctx,
                        spread,
                        propagates_handles,
                    )
                });
            fields.iter().fold(spread, |class, (_, field)| {
                class.join(evaluation_control_class_with_handle_propagation(
                    ctx,
                    field,
                    propagates_handles,
                ))
            })
        }
        ExprKind::Block { stmts, value } => {
            let statements = stmts
                .iter()
                .fold(EvaluationControlClass::Direct, |class, stmt| {
                    let expr = match stmt {
                        Stmt::Let { value, .. } => value,
                        Stmt::Expr { expr, .. } => expr,
                    };
                    class.join(evaluation_control_class_with_handle_propagation(
                        ctx,
                        expr,
                        propagates_handles,
                    ))
                });
            statements.join(evaluation_control_class_with_handle_propagation(
                ctx,
                value,
                propagates_handles,
            ))
        }
        ExprKind::BinOp { lhs, rhs, .. } => {
            evaluation_control_class_with_handle_propagation(ctx, lhs, propagates_handles).join(
                evaluation_control_class_with_handle_propagation(ctx, rhs, propagates_handles),
            )
        }
        ExprKind::Case { scrutinee, arms } => arms.iter().fold(
            evaluation_control_class_with_handle_propagation(ctx, scrutinee, propagates_handles),
            |class, arm| {
                let guard = arm
                    .guard
                    .as_ref()
                    .map_or(EvaluationControlClass::Direct, |guard| {
                        evaluation_control_class_with_handle_propagation(
                            ctx,
                            guard,
                            propagates_handles,
                        )
                    });
                class
                    .join(guard)
                    .join(evaluation_control_class_with_handle_propagation(
                        ctx,
                        &arm.body,
                        propagates_handles,
                    ))
            },
        ),
        ExprKind::MethodCall { receiver, args, .. } => {
            evaluation_control_class_with_handle_propagation(ctx, receiver, propagates_handles)
                .join(children(args))
        }
        ExprKind::Var(_)
        | ExprKind::NatLit(_)
        | ExprKind::IntLit(_)
        | ExprKind::FloatLit(_)
        | ExprKind::StringLit(_)
        | ExprKind::BytesLit(_)
        | ExprKind::BoolLit(_)
        | ExprKind::Nil
        | ExprKind::RuneLit(_)
        | ExprKind::Error => EvaluationControlClass::Direct,
    }
}

/// Classify evaluation under a typed CPS answer domain, preserving #817's
/// Direct/EvidenceDirect/Cps ordering.
pub(super) fn evaluation_control_class_in<'db, D: ControlDomain>(
    ctx: &super::super::context::IrLoweringCtx<'db>,
    expr: &Expr<TypedRef<'db>>,
) -> EvaluationControlClass {
    evaluation_control_class_with_handle_propagation(ctx, expr, D::PROPAGATES_HANDLES)
}

/// Lower an unnormalized computation entry exactly once.
pub(super) fn lower_comp<'db, D: ControlDomain>(
    builder: &mut IrBuilder<'_, 'db>,
    expr: Expr<TypedRef<'db>>,
    continuation: ContinuationRef<D>,
) -> Option<ControlResultRef<D>> {
    let expr = super::super::normalize::normalize_for_cps(builder.ctx, expr);
    lower_comp_normalized(builder, expr, continuation)
}

/// Lower an expression already owned by the normalizer traversal.
pub(super) fn lower_comp_normalized<'db, D: ControlDomain>(
    builder: &mut IrBuilder<'_, 'db>,
    expr: Expr<TypedRef<'db>>,
    continuation: ContinuationRef<D>,
) -> Option<ControlResultRef<D>> {
    if evaluation_control_class_in::<D>(builder.ctx, &expr) == EvaluationControlClass::Direct {
        let location = builder.location(expr.id);
        let value = lower_value_normalized(builder, expr)?;
        return Some(invoke_continuation(builder, location, continuation, value));
    }

    let location = builder.location(expr.id);
    match *expr.kind {
        ExprKind::Block { stmts, value } => {
            lower_normalized_block(builder, stmts, value, continuation)
        }
        ExprKind::Call { callee, args } => {
            lower_cps_call_expr(builder, location, expr.id, callee, args, continuation)
        }
        ExprKind::Case { scrutinee, arms } => {
            super::case::lower_case_comp(builder, location, scrutinee, arms, continuation)
        }
        ExprKind::BinOp { op, lhs, rhs } => {
            lower_short_circuit_comp(builder, location, op, lhs, rhs, continuation)
        }
        ExprKind::Resume { arg, local_id } => super::handle::lower_resume_comp(
            builder,
            location,
            expr.id,
            arg,
            local_id,
            continuation,
        ),
        ExprKind::Handle { body, handlers } => {
            super::handle::lower_handle_comp(builder, location, &body, &handlers, continuation)
        }
        _ => panic!("ICE: non-normal CPS expression reached computation lowering"),
    }
}

/// Consume one normalized block. Direct statements execute inline; the first
/// CPS RHS receives a generated continuation for the owned remainder.
pub(super) fn lower_normalized_block<'db, D: ControlDomain>(
    builder: &mut IrBuilder<'_, 'db>,
    stmts: Vec<Stmt<TypedRef<'db>>>,
    value: Expr<TypedRef<'db>>,
    continuation: ContinuationRef<D>,
) -> Option<ControlResultRef<D>> {
    let mut scope = builder.ctx.scope();
    let builder = &mut IrBuilder::new(&mut scope, builder.ir, builder.block);
    let mut stmts = stmts.into_iter();

    while let Some(stmt) = stmts.next() {
        let rhs = match &stmt {
            Stmt::Let { value, .. } => value,
            Stmt::Expr { expr, .. } => expr,
        };
        if evaluation_control_class_in::<D>(builder.ctx, rhs) == EvaluationControlClass::Direct {
            lower_single_stmt(builder, stmt);
            continue;
        }

        let remaining_stmts = stmts.collect();
        let (receiver, rhs) = match stmt {
            Stmt::Let { pattern, value, .. } => (ContinuationReceiver::Bind(pattern), value),
            Stmt::Expr { expr, .. } => (ContinuationReceiver::Discard, expr),
        };
        let logical_ty = expression_ir_type(builder, &rhs);
        let anyref_ty = builder.ctx.anyref_type(builder.ir);
        let next = build_cps_continuation(
            builder,
            builder.location(rhs.id),
            anyref_ty,
            logical_ty,
            ContinuationBody {
                receiver,
                remaining_stmts,
                final_value: value,
                outer_k: continuation,
            },
        )?;
        return lower_comp_normalized(builder, rhs, next);
    }

    lower_comp_normalized(builder, value, continuation)
}

/// Lower a direct-only block expression without entering computation lowering.
fn lower_block<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    stmts: Vec<Stmt<TypedRef<'db>>>,
    value: Expr<TypedRef<'db>>,
    nested_regions_normalized: bool,
) -> Option<ValueRef> {
    let mut scope = builder.ctx.scope();
    let builder = &mut IrBuilder::new(&mut scope, builder.ir, builder.block);
    for stmt in stmts {
        lower_single_stmt_with_mode(builder, stmt, nested_regions_normalized);
    }
    lower_value_impl(builder, value, nested_regions_normalized)
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
    lower_single_stmt_with_mode(builder, stmt, true);
}

fn lower_single_stmt_with_mode<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    stmt: Stmt<TypedRef<'db>>,
    nested_regions_normalized: bool,
) {
    match stmt {
        Stmt::Let {
            id: _,
            pattern,
            ty: _,
            value,
        } => {
            if let Some(val) = lower_value_impl(builder, value, nested_regions_normalized) {
                bind_stmt_pattern(builder, &pattern, val);
            }
        }
        Stmt::Expr { id: _, expr } => {
            let _ = lower_value_impl(builder, expr, nested_regions_normalized);
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

fn lower_cps_call_expr<'db, D: ControlDomain>(
    builder: &mut IrBuilder<'_, 'db>,
    location: Location,
    _call_expr_id: crate::ast::NodeId,
    callee: Expr<TypedRef<'db>>,
    args: Vec<Expr<TypedRef<'db>>>,
    continuation: ContinuationRef<D>,
) -> Option<ControlResultRef<D>> {
    let callee_id = callee.id;
    let callee_kind = *callee.kind;
    let arg_exprs = args;
    let mut arg_values = arg_exprs
        .iter()
        .cloned()
        .map(|arg| lower_value_normalized(builder, arg))
        .collect::<Option<Vec<_>>>()?;

    let anyref_ty = builder.ctx.anyref_type(builder.ir);
    let continuation = continuation_abi(continuation);

    match callee_kind {
        ExprKind::Var(typed_ref) => match &typed_ref.resolved {
            ResolvedRef::AbilityOp {
                ability,
                op,
                kind: crate::ast::OpDeclKind::Op,
            } => {
                arg_values = pack_ability_args(builder, location, arg_values);
                let ability_name = ability.qualified(builder.db());
                let ability_ref = builder.ctx.ability_ref_type(builder.ir, ability_name, &[]);
                let perform = ability::legacy_perform(
                    builder.ir,
                    location,
                    continuation,
                    arg_values,
                    anyref_ty,
                    ability_ref,
                    *op,
                );
                builder.ir.push_op(builder.block, perform.op_ref());
                Some(control_from_abi(perform.result(builder.ir)))
            }
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
                Some(control_from_abi(call_op.result(builder.ir)))
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
                Some(control_from_abi(call_op.result(builder.ir)))
            }
            _ => unreachable!("ICE: lower_cps_call with unexpected callee kind"),
        },
        callee_kind => {
            let callee = Expr::new(callee_id, callee_kind);
            let callee_val = lower_value_normalized(builder, callee)?;
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
            Some(control_from_abi(call_op.result(builder.ir)))
        }
    }
}

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

/// Add an internal context value (evidence or done_k) to the capture list if
/// present and not already captured.
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

enum ContinuationReceiver<'db> {
    Bind(crate::ast::Pattern<TypedRef<'db>>),
    Discard,
}

struct ContinuationBody<'db, D: ControlDomain> {
    receiver: ContinuationReceiver<'db>,
    remaining_stmts: Vec<Stmt<TypedRef<'db>>>,
    final_value: Expr<TypedRef<'db>>,
    outer_k: ContinuationRef<D>,
}

/// Build a CPS continuation closure for the remaining computation after a CPS
/// producer.
///
/// The continuation's parameter is the CPS producer's source result, bound
/// or discarded by `receiver`. Its body lowers the remaining computation and
/// returns the opaque compatibility control answer through the `anyref` ABI.
fn build_cps_continuation<'db, D: ControlDomain>(
    builder: &mut IrBuilder<'_, 'db>,
    location: Location,
    param_type: TypeRef,
    logical_type: TypeRef,
    body: ContinuationBody<'db, D>,
) -> Option<ContinuationRef<D>> {
    let anyref_ty = builder.ctx.anyref_type(builder.ir);

    // Analyze captures: variables from current scope used in remaining computation
    let excluded_ids = HashSet::new();
    let mut captures = super::lambda::analyze_continuation_captures(
        builder.ctx,
        builder.ir,
        &body.remaining_stmts,
        &body.final_value,
        &excluded_ids,
    );
    capture_ctx_value(
        &mut captures,
        builder,
        Symbol::new("__outer_k"),
        Some(continuation_abi(body.outer_k)),
    );
    // The dynamic owner is lowering metadata rather than a source local, so
    // AST free-variable analysis cannot see it. A continuation formed inside
    // a general handler may compare it after closure extraction.
    capture_ctx_value(
        &mut captures,
        builder,
        Symbol::new("__handler_owner"),
        builder.ctx.handler_owner_tag(),
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
        if let ContinuationReceiver::Bind(pattern) = &body.receiver {
            let mut inner_builder = IrBuilder::new(&mut scope, builder.ir, entry_block);
            bind_stmt_pattern(&mut inner_builder, pattern, typed_param);
        }

        let result = {
            let mut inner_builder = IrBuilder::new(&mut scope, builder.ir, entry_block);
            lower_normalized_block(
                &mut inner_builder,
                body.remaining_stmts,
                body.final_value,
                body.outer_k,
            )
        }?;
        {
            let mut inner_builder = IrBuilder::new(&mut scope, builder.ir, entry_block);
            emit_control_return(&mut inner_builder, location, result);
        };
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
    Some(super::control::continuation_from_abi(
        lambda_op.result(builder.ir),
    ))
}

fn expression_ir_type(builder: &mut IrBuilder<'_, '_>, expr: &Expr<TypedRef<'_>>) -> TypeRef {
    builder
        .ctx
        .get_node_type(expr.id)
        .copied()
        .map(|ty| builder.ctx.convert_type(builder.ir, ty))
        .unwrap_or_else(|| builder.ctx.anyref_type(builder.ir))
}

fn lower_short_circuit_comp<'db, D: ControlDomain>(
    builder: &mut IrBuilder<'_, 'db>,
    location: Location,
    op: BinOpKind,
    lhs: Expr<TypedRef<'db>>,
    rhs: Expr<TypedRef<'db>>,
    continuation: ContinuationRef<D>,
) -> Option<ControlResultRef<D>> {
    let lhs = lower_value_normalized(builder, lhs)?;
    let anyref_ty = builder.ctx.anyref_type(builder.ir);

    let const_value = match op {
        BinOpKind::And => false,
        BinOpKind::Or => true,
    };
    let const_block = builder.ir.create_block(BlockData {
        location,
        args: vec![],
        ops: Default::default(),
        parent_region: None,
    });
    let const_result = {
        let mut inner = IrBuilder::new(builder.ctx, builder.ir, const_block);
        let bool_ty = inner.ctx.bool_type(inner.ir);
        let constant = arith::r#const(inner.ir, location, bool_ty, Attribute::Bool(const_value));
        inner.ir.push_op(inner.block, constant.op_ref());
        let value = constant.result(inner.ir);
        invoke_continuation(&mut inner, location, continuation, value)
    };
    {
        let mut inner = IrBuilder::new(builder.ctx, builder.ir, const_block);
        super::control::emit_control_yield(&mut inner, location, const_result);
    }
    let const_region = builder.ir.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![const_block],
        parent_op: None,
    });

    let rhs_block = builder.ir.create_block(BlockData {
        location,
        args: vec![],
        ops: Default::default(),
        parent_region: None,
    });
    let rhs_result = {
        let mut inner = IrBuilder::new(builder.ctx, builder.ir, rhs_block);
        lower_comp_normalized(&mut inner, rhs, continuation)?
    };
    {
        let mut inner = IrBuilder::new(builder.ctx, builder.ir, rhs_block);
        super::control::emit_control_yield(&mut inner, location, rhs_result);
    }
    let rhs_region = builder.ir.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![rhs_block],
        parent_op: None,
    });

    let (then_region, else_region) = match op {
        BinOpKind::And => (rhs_region, const_region),
        BinOpKind::Or => (const_region, rhs_region),
    };
    let if_op = scf::r#if(
        builder.ir,
        location,
        lhs,
        anyref_ty,
        then_region,
        else_region,
    );
    builder.ir.push_op(builder.block, if_op.op_ref());
    Some(control_from_abi(if_op.result(builder.ir)))
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
    nested_regions_normalized: bool,
) -> trunk_ir::refs::RegionRef {
    let block = builder.ir.create_block(BlockData {
        location,
        args: vec![],
        ops: Default::default(),
        parent_region: None,
    });

    let rhs_val = {
        let mut inner = IrBuilder::new(builder.ctx, builder.ir, block);
        if nested_regions_normalized {
            lower_value_normalized(&mut inner, rhs)
        } else {
            lower_value(&mut inner, rhs)
        }
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
