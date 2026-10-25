//! Values and patterns: expressions, constructors, aggregates, case chains,
//! and bindings.
use super::*;

pub(super) fn lower_expr<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    expr: Expr<TypedRef<'db>>,
    declarations: &mut Declarations<'db>,
) -> Option<ValueRef> {
    let location = builder.location(expr.id);
    match *expr.kind {
        ExprKind::NatLit(value) => {
            let ty = builder.ctx.i32_type(builder.ir);
            let value = arith::Const::operands()
                .value(Attribute::Int(value as i128))
                .results(ty)
                .build(builder.ir, location);
            builder.ir.push_op(builder.block, value.op_ref());
            Some(value.result(builder.ir))
        }
        ExprKind::IntLit(value) => {
            let ty = builder.ctx.i32_type(builder.ir);
            let value = arith::Const::operands()
                .value(Attribute::Int(value as i128))
                .results(ty)
                .build(builder.ir, location);
            builder.ir.push_op(builder.block, value.op_ref());
            Some(value.result(builder.ir))
        }
        ExprKind::BoolLit(value) => {
            let ty = builder.ctx.bool_type(builder.ir);
            let value = arith::Const::operands()
                .value(Attribute::Bool(value))
                .results(ty)
                .build(builder.ir, location);
            builder.ir.push_op(builder.block, value.op_ref());
            Some(value.result(builder.ir))
        }
        ExprKind::FloatLit(value) => {
            let ty = builder.ctx.f64_type(builder.ir);
            let value = arith::Const::operands()
                .value(Attribute::FloatBits(value.value().to_bits()))
                .results(ty)
                .build(builder.ir, location);
            builder.ir.push_op(builder.block, value.op_ref());
            Some(value.result(builder.ir))
        }
        ExprKind::Nil => Some(builder.emit_nil(location)),
        ExprKind::RuneLit(value) => {
            let ty = builder.ctx.i32_type(builder.ir);
            let value = arith::Const::operands()
                .value(Attribute::Int(value as i32 as i128))
                .results(ty)
                .build(builder.ir, location);
            builder.ir.push_op(builder.block, value.op_ref());
            Some(value.result(builder.ir))
        }
        ExprKind::BytesLit(value) => {
            let ty = builder.ctx.bytes_type(builder.ir);
            let value = adt::BytesConst::operands()
                .value(value.into())
                .results(ty)
                .build(builder.ir, location);
            builder.ir.push_op(builder.block, value.op_ref());
            Some(value.result(builder.ir))
        }
        ExprKind::StringLit(value) => {
            let ty = builder.ctx.anyref_type(builder.ir);
            let value = builder.ir.intern_str(&value);
            let value = adt::StringConst::operands()
                .value(value)
                .results(ty)
                .build(builder.ir, location);
            builder.ir.push_op(builder.block, value.op_ref());
            Some(value.result(builder.ir))
        }
        ExprKind::Var(reference) => match reference.resolved {
            ResolvedRef::Local { id, .. } => Some(
                lower_local_reference(builder, expr.id, id, declarations)
                    .unwrap_or_else(|| panic!("missing logical binding for local {id:?}")),
            ),
            ResolvedRef::Function { id } => {
                let name = id.path(builder.ctx.db);
                Some(lower_function_ref(
                    builder,
                    location,
                    name.clone(),
                    RowArguments::Instance(expr.id),
                    declarations,
                    None,
                ))
            }
            ResolvedRef::AbilityOp { .. } => {
                // Typechecking has already diagnosed this non-call operation
                // reference.  Preserve error recovery without inventing a
                // perform declaration from malformed source.
                Some(builder.emit_nil(location))
            }
            _ => panic!("unsupported resolved reference at source-logical boundary"),
        },
        ExprKind::Block { stmts, value } => {
            let mut scope = builder.ctx.scope();
            let mut inner = IrBuilder::new(&mut scope, builder.ir, builder.block);
            for statement in stmts {
                lower_statement(&mut inner, statement, declarations);
            }
            lower_expr(&mut inner, value, declarations)
        }
        ExprKind::BinOp {
            op: binop,
            lhs,
            rhs,
        } => {
            let lhs = lower_expr(builder, lhs, declarations)?;
            let bool_ty = builder.ctx.bool_type(builder.ir);
            let then_block = builder.ir.create_block(BlockData {
                location,
                args: vec![],
                ops: Default::default(),
                parent_region: None,
            });
            let then_value = {
                let mut inner = IrBuilder::new(builder.ctx, builder.ir, then_block);
                match binop {
                    crate::ast::BinOpKind::And => {
                        lower_expr(&mut inner, rhs.clone(), declarations)?
                    }
                    crate::ast::BinOpKind::Or => emit_bool(&mut inner, location, true),
                }
            };
            let then_yield = scf::Yield::operands([then_value]).build(builder.ir, location);
            builder.ir.push_op(then_block, then_yield.op_ref());
            let then_region = builder.ir.create_region(RegionData {
                location,
                blocks: trunk_ir::smallvec::smallvec![then_block],
                parent_op: None,
            });
            let else_block = builder.ir.create_block(BlockData {
                location,
                args: vec![],
                ops: Default::default(),
                parent_region: None,
            });
            let else_value = {
                let mut inner = IrBuilder::new(builder.ctx, builder.ir, else_block);
                match binop {
                    crate::ast::BinOpKind::And => emit_bool(&mut inner, location, false),
                    crate::ast::BinOpKind::Or => lower_expr(&mut inner, rhs, declarations)?,
                }
            };
            let else_yield = scf::Yield::operands([else_value]).build(builder.ir, location);
            builder.ir.push_op(else_block, else_yield.op_ref());
            let else_region = builder.ir.create_region(RegionData {
                location,
                blocks: trunk_ir::smallvec::smallvec![else_block],
                parent_op: None,
            });
            let branch = scf::If::operands(lhs)
                .results(bool_ty)
                .regions(then_region, else_region)
                .build(builder.ir, location);
            builder.ir.push_op(builder.block, branch.op_ref());
            Some(branch.result(builder.ir))
        }
        ExprKind::Call { callee, args, .. } => {
            let result_ty = builder
                .ctx
                .get_node_type(expr.id)
                .copied()
                .map(|ty| builder.ctx.convert_logical_type(builder.ir, ty))
                .unwrap_or_else(|| panic!("missing typechecked result for call"));
            lower_call(
                builder,
                location,
                expr.id,
                result_ty,
                callee,
                args,
                declarations,
                CallPosition::Value,
            )
        }
        ExprKind::Become { call } => lower_become(builder, location, call, declarations),
        ExprKind::Lambda { params, body } => {
            let signature = declarations
                .lambda_signatures
                .get(&expr.id)
                .cloned()
                .unwrap_or_else(|| panic!("missing solved logical lambda signature"));
            lower_lambda(
                builder,
                location,
                signature,
                params,
                body,
                declarations,
                None,
            )
        }
        ExprKind::Handle { body, handlers } => {
            let result_ty = builder
                .ctx
                .get_node_type(expr.id)
                .copied()
                .map(|ty| builder.ctx.convert_logical_type(builder.ir, ty))
                .unwrap_or_else(|| panic!("missing typechecked result for handle"));
            lower_handle(
                builder,
                location,
                expr.id,
                result_ty,
                body,
                handlers,
                declarations,
            )
        }
        ExprKind::Cons { ctor, args } => {
            lower_constructor(builder, location, expr.id, ctor, args, declarations)
        }
        ExprKind::Tuple(elements) => {
            lower_tuple(builder, location, expr.id, elements, declarations)
        }
        ExprKind::List(elements) => lower_list(builder, location, expr.id, elements, declarations),
        ExprKind::Record {
            type_name,
            fields,
            spread,
        } => lower_record(
            builder,
            location,
            expr.id,
            type_name,
            fields,
            spread,
            declarations,
        ),
        ExprKind::Case { scrutinee, arms } => {
            let scrutinee = lower_expr(builder, scrutinee, declarations)?;
            let result_ty = expr_type_for_id(builder, expr.id);
            let exhaustive = declarations.exhaustive_cases.contains(&expr.id);
            // The unguarded arms of an exhaustive case match every value, so
            // the arms after the last of them never run; typechecking reports
            // them as unreachable. Dropping them leaves no path on which every
            // arm fails.
            let arms = match arms.iter().rposition(|arm| arm.guard.is_none()) {
                Some(last) if exhaustive => &arms[..=last],
                _ => &arms[..],
            };
            lower_case_chain(
                builder,
                location,
                scrutinee,
                result_ty,
                arms,
                exhaustive,
                declarations,
            )
        }
        ExprKind::Resume { arg, local_id } => {
            let token = builder.ctx.lookup_resume(local_id?)?;
            let (input_ty, _) =
                tribute_control::resume_token_parts(builder.ir, builder.ir.value_ty(token))
                    .expect("typechecked resume local must lower to a resume token");
            let value = lower_expr(builder, arg, declarations)?;
            let value = builder.cast_if_needed(location, value, input_ty);
            let resume =
                tribute_control::Resume::operands(token, value).build(builder.ir, location);
            attach_evidence_plan(
                builder.ctx,
                builder.ir,
                resume.op_ref(),
                expr.id,
                declarations,
            );
            builder.ir.push_op(builder.block, resume.op_ref());
            Some(resume.result(builder.ir))
        }
        ExprKind::Error => Some(builder.emit_nil(location)),
        _ => panic!("unsupported source expression at source-logical boundary"),
    }
}

pub(super) fn lower_constructor<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    location: Location,
    id: crate::ast::NodeId,
    ctor: TypedRef<'db>,
    args: Vec<Expr<TypedRef<'db>>>,
    declarations: &mut Declarations<'db>,
) -> Option<ValueRef> {
    let ResolvedRef::Constructor { variant, .. } = ctor.resolved.clone() else {
        panic!("non-constructor in source logical constructor expression");
    };
    let values = args
        .into_iter()
        .map(|arg| lower_expr(builder, arg, declarations))
        .collect::<Option<Vec<_>>>()?;
    let result_ty = expr_type_for_id(builder, id);
    let type_attr = super::super::resolve_enum_type_attr_for_constructor(
        builder.ctx,
        builder.ir,
        &ctor.resolved,
        ctor.ty,
    );
    let values =
        super::super::expr::cast_variant_args(builder, location, values, type_attr, &variant);
    let tag = builder.ir.intern_symbol_text(&variant);
    let variant = adt::VariantNew::operands(values)
        .r#type(type_attr)
        .tag(tag)
        .results(result_ty)
        .build(builder.ir, location);
    builder.ir.push_op(builder.block, variant.op_ref());
    Some(variant.result(builder.ir))
}

pub(super) fn lower_tuple<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    location: Location,
    id: crate::ast::NodeId,
    elements: Vec<Expr<TypedRef<'db>>>,
    declarations: &mut Declarations<'db>,
) -> Option<ValueRef> {
    let values = elements
        .into_iter()
        .map(|element| lower_expr(builder, element, declarations))
        .collect::<Option<Vec<_>>>()?;
    let (_, type_attr) =
        super::super::get_or_create_logical_tuple_type(builder.ctx, builder.ir, id)
            .unwrap_or_else(|| panic!("missing typechecked tuple layout"));
    let result_ty = expr_type_for_id(builder, id);
    let tuple = adt::StructNew::operands(values)
        .r#type(type_attr)
        .results(result_ty)
        .build(builder.ir, location);
    builder.ir.push_op(builder.block, tuple.op_ref());
    Some(tuple.result(builder.ir))
}

pub(super) fn lower_list<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    location: Location,
    id: crate::ast::NodeId,
    elements: Vec<Expr<TypedRef<'db>>>,
    declarations: &mut Declarations<'db>,
) -> Option<ValueRef> {
    let values = elements
        .into_iter()
        .map(|element| lower_expr(builder, element, declarations))
        .collect::<Option<Vec<_>>>()?;
    let list_source = builder
        .ctx
        .get_node_type(id)
        .copied()
        .unwrap_or_else(|| panic!("missing typechecked list type"));
    let TypeKind::Named {
        id: list_id, args, ..
    } = list_source.kind(builder.db())
    else {
        panic!("source list expression did not have List type");
    };
    if !list_id.is_builtin_list(builder.db()) || args.len() != 1 {
        panic!("source list expression did not have unary built-in List type");
    }
    let element_ty = builder.ctx.convert_logical_type(builder.ir, args[0]);
    let list_ty = expr_type_for_id(builder, id);
    let empty = list::Empty::operands()
        .element_type(element_ty)
        .results(list_ty)
        .build(builder.ir, location);
    builder.ir.push_op(builder.block, empty.op_ref());
    let mut value = empty.result(builder.ir);
    for element in values.into_iter().rev() {
        let prepend = list::Prepend::operands(element, value)
            .element_type(element_ty)
            .build(builder.ir, location);
        builder.ir.push_op(builder.block, prepend.op_ref());
        value = prepend.result(builder.ir);
    }
    Some(value)
}

pub(super) fn lower_record<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    location: Location,
    id: crate::ast::NodeId,
    type_name: TypedRef<'db>,
    fields: Vec<FieldInit<TypedRef<'db>>>,
    spread: Option<Expr<TypedRef<'db>>>,
    declarations: &mut Declarations<'db>,
) -> Option<ValueRef> {
    if let ResolvedRef::Constructor { variant, .. } = type_name.resolved.clone() {
        let layout = super::super::resolve_enum_type_attr_for_constructor(
            builder.ctx,
            builder.ir,
            &type_name.resolved,
            type_name.ty,
        );
        if tribute_ir::dialect::adt::layout::get_enum_variants(builder.ir, layout).is_some() {
            return lower_variant_record(
                builder,
                location,
                id,
                layout,
                &variant,
                fields,
                declarations,
            );
        }
    }
    let ctor = super::super::extract_ctor_id(&type_name.resolved);
    let struct_name = super::super::extract_type_name(builder.db(), &type_name.resolved);
    let field_order = builder
        .ctx
        .get_struct_field_order(ctor)
        .cloned()
        .unwrap_or_else(|| panic!("prescan did not register struct field order {struct_name}"));
    let layout = builder
        .ctx
        .get_type(&super::super::qualified_type_name(builder.db(), &ctor))
        .unwrap_or_else(|| panic!("prescan did not register struct layout {struct_name}"));
    // A record spread is evaluated before every explicit field.  The values
    // are nevertheless assembled in declaration layout order below, so these
    // two ordering concerns must stay separate.
    let spread = match spread {
        Some(spread) => Some(lower_expr(builder, spread, declarations)?),
        None => None,
    };
    // Lower explicit fields in source order, then place their already-evaluated
    // values in declaration layout order.
    let mut values = HashMap::default();
    for FieldInit {
        name, value: field, ..
    } in fields
    {
        if !field_order.contains(&name) || values.contains_key(&name) {
            panic!("typechecked record has an invalid field layout");
        }
        values.insert(name, lower_expr(builder, field, declarations)?);
    }
    let mut ordered = Vec::with_capacity(field_order.len());
    for (index, name) in field_order.iter().enumerate() {
        if let Some(value) = values.get(name) {
            ordered.push(*value);
        } else {
            let base =
                spread.unwrap_or_else(|| panic!("typechecked record is missing field {name}"));
            // The layout owns concrete field types.  `struct_get` needs a
            // result type, obtained from the matching getter expression type
            // only after normal typechecking; use layout metadata directly.
            let field_types =
                tribute_ir::dialect::adt::layout::get_struct_fields(builder.ir, layout)
                    .unwrap_or_else(|| panic!("prescanned struct layout is malformed"));
            let get = adt::StructGet::operands(base)
                .r#type(layout)
                .field(index as u32)
                .results(field_types[index].1)
                .build(builder.ir, location);
            builder.ir.push_op(builder.block, get.op_ref());
            ordered.push(get.result(builder.ir));
        }
    }
    let result_ty = expr_type_for_id(builder, id);
    let record = adt::StructNew::operands(ordered)
        .r#type(layout)
        .results(result_ty)
        .build(builder.ir, location);
    builder.ir.push_op(builder.block, record.op_ref());
    Some(record.result(builder.ir))
}

/// Lower a record literal that constructs a named-field enum variant. Fields
/// are evaluated in source order and assembled in declaration order; type
/// checking rejects spreads, so every field is written.
pub(super) fn lower_variant_record<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    location: Location,
    id: crate::ast::NodeId,
    layout: TypeRef,
    variant: &Symbol,
    fields: Vec<FieldInit<TypedRef<'db>>>,
    declarations: &mut Declarations<'db>,
) -> Option<ValueRef> {
    let field_order = builder
        .ctx
        .variant_field_names(layout, variant.clone())
        .unwrap_or_else(|| panic!("prescan did not register field names of variant {variant}"));
    let mut values = HashMap::default();
    for FieldInit {
        name, value: field, ..
    } in fields
    {
        if !field_order.contains(&name) || values.contains_key(&name) {
            panic!("typechecked variant record has an invalid field layout");
        }
        values.insert(name, lower_expr(builder, field, declarations)?);
    }
    let ordered = field_order
        .iter()
        .map(|name| {
            *values
                .get(name)
                .unwrap_or_else(|| panic!("typechecked variant record is missing field {name}"))
        })
        .collect();
    let ordered =
        super::super::expr::cast_variant_args(builder, location, ordered, layout, variant);
    let result_ty = expr_type_for_id(builder, id);
    let tag = builder.ir.intern_symbol_text(variant);
    let value = adt::VariantNew::operands(ordered)
        .r#type(layout)
        .tag(tag)
        .results(result_ty)
        .build(builder.ir, location);
    builder.ir.push_op(builder.block, value.op_ref());
    Some(value.result(builder.ir))
}

/// Lower source pattern selection without introducing a control-dialect
/// conditional.  Pattern tests and guards are ordinary structured values, so
/// the logical boundary keeps them in `scf.if`.
pub(super) fn lower_case_chain<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    location: Location,
    scrutinee: ValueRef,
    result_ty: TypeRef,
    arms: &[Arm<TypedRef<'db>>],
    exhaustive: bool,
    declarations: &mut Declarations<'db>,
) -> Option<ValueRef> {
    match arms {
        // Typechecking rejects a case it does not prove exhaustive, and an
        // exhaustive chain ends with an unguarded arm lowered without a test.
        [] => panic!("a case whose arms can all fail reached lowering"),
        [last] if exhaustive && last.guard.is_none() => {
            let mut scope = builder.ctx.scope();
            super::super::case::bind_logical_pattern_fields(
                &mut scope,
                builder.ir,
                builder.block,
                location,
                scrutinee,
                &last.pattern,
            );
            lower_expr(
                &mut IrBuilder::new(&mut scope, builder.ir, builder.block),
                last.body.clone(),
                declarations,
            )
        }
        [first, rest @ ..] => {
            let condition = super::super::case::emit_logical_pattern_check(
                builder,
                location,
                scrutinee,
                &first.pattern,
            )?;
            let then_region = build_case_arm_region(
                builder.ctx,
                builder.ir,
                CaseArmRequest {
                    location,
                    scrutinee,
                    arm: first,
                    rest,
                    result_ty,
                    exhaustive,
                },
                declarations,
            )?;
            let else_region = build_case_else_region(
                builder.ctx,
                builder.ir,
                location,
                scrutinee,
                rest,
                result_ty,
                exhaustive,
                declarations,
            )?;
            let branch = scf::If::operands(condition)
                .results(result_ty)
                .regions(then_region, else_region)
                .build(builder.ir, location);
            builder.ir.push_op(builder.block, branch.op_ref());
            Some(branch.result(builder.ir))
        }
    }
}

pub(super) struct CaseArmRequest<'a, 'db> {
    pub(super) location: Location,
    pub(super) scrutinee: ValueRef,
    pub(super) arm: &'a Arm<TypedRef<'db>>,
    pub(super) rest: &'a [Arm<TypedRef<'db>>],
    pub(super) result_ty: TypeRef,
    pub(super) exhaustive: bool,
}

pub(super) fn build_case_arm_region<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    ir: &mut IrContext,
    request: CaseArmRequest<'_, 'db>,
    declarations: &mut Declarations<'db>,
) -> Option<trunk_ir::refs::RegionRef> {
    let CaseArmRequest {
        location,
        scrutinee,
        arm,
        rest,
        result_ty,
        exhaustive,
    } = request;
    let block = ir.create_block(BlockData {
        location,
        args: vec![],
        ops: Default::default(),
        parent_region: None,
    });
    let value = {
        let mut scope = ctx.scope();
        super::super::case::bind_logical_pattern_fields(
            &mut scope,
            ir,
            block,
            location,
            scrutinee,
            &arm.pattern,
        );
        let mut nested = IrBuilder::new(&mut scope, ir, block);
        if let Some(guard) = &arm.guard {
            (|| {
                let condition = lower_expr(&mut nested, guard.clone(), declarations)?;
                let then_region = build_case_body_region(
                    nested.ctx,
                    nested.ir,
                    location,
                    arm.body.clone(),
                    result_ty,
                    declarations,
                )?;
                let else_region = build_case_else_region(
                    nested.ctx,
                    nested.ir,
                    location,
                    scrutinee,
                    rest,
                    result_ty,
                    exhaustive,
                    declarations,
                )?;
                let branch = scf::If::operands(condition)
                    .results(result_ty)
                    .regions(then_region, else_region)
                    .build(nested.ir, location);
                nested.ir.push_op(nested.block, branch.op_ref());
                Some(branch.result(nested.ir))
            })()
        } else {
            lower_expr(&mut nested, arm.body.clone(), declarations)
        }
    };
    finish_case_region(ctx, ir, location, block, value, result_ty)
}

/// End a structured case region with its value, or leave it ended by a
/// source `become`.
pub(super) fn finish_case_region(
    ctx: &mut IrLoweringCtx<'_>,
    ir: &mut IrContext,
    location: Location,
    block: BlockRef,
    value: Option<ValueRef>,
    result_ty: TypeRef,
) -> Option<trunk_ir::refs::RegionRef> {
    if value.is_some() || !ends_with_tail_call(ir, block) {
        let value = IrBuilder::new(ctx, ir, block).cast_if_needed(location, value?, result_ty);
        let yield_op = scf::Yield::operands([value]).build(ir, location);
        ir.push_op(block, yield_op.op_ref());
    }
    Some(ir.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![block],
        parent_op: None,
    }))
}

pub(super) fn build_case_body_region<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    ir: &mut IrContext,
    location: Location,
    body: Expr<TypedRef<'db>>,
    result_ty: TypeRef,
    declarations: &mut Declarations<'db>,
) -> Option<trunk_ir::refs::RegionRef> {
    let block = ir.create_block(BlockData {
        location,
        args: vec![],
        ops: Default::default(),
        parent_region: None,
    });
    let value = lower_expr(&mut IrBuilder::new(ctx, ir, block), body, declarations);
    finish_case_region(ctx, ir, location, block, value, result_ty)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_case_else_region<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    ir: &mut IrContext,
    location: Location,
    scrutinee: ValueRef,
    arms: &[Arm<TypedRef<'db>>],
    result_ty: TypeRef,
    exhaustive: bool,
    declarations: &mut Declarations<'db>,
) -> Option<trunk_ir::refs::RegionRef> {
    let block = ir.create_block(BlockData {
        location,
        args: vec![],
        ops: Default::default(),
        parent_region: None,
    });
    let value = lower_case_chain(
        &mut IrBuilder::new(ctx, ir, block),
        location,
        scrutinee,
        result_ty,
        arms,
        exhaustive,
        declarations,
    );
    finish_case_region(ctx, ir, location, block, value, result_ty)
}

pub(super) fn emit_bool(
    builder: &mut IrBuilder<'_, '_>,
    location: Location,
    value: bool,
) -> ValueRef {
    let ty = builder.ctx.bool_type(builder.ir);
    let op = arith::Const::operands()
        .value(Attribute::Bool(value))
        .results(ty)
        .build(builder.ir, location);
    builder.ir.push_op(builder.block, op.op_ref());
    op.result(builder.ir)
}

/// Declare on the lambda `binding` binds the local binders its uses
/// instantiate.
pub(super) fn quantify_local_rows<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    binding: crate::ast::NodeId,
    lambda: ValueRef,
    declarations: &Declarations<'db>,
) {
    let Some(use_) = declarations
        .local_uses
        .iter()
        .find(|use_| use_.binding == binding && !use_.rows.is_empty())
    else {
        return;
    };
    let trunk_ir::refs::ValueDef::OpResult(op, _) = builder.ir.value_def(lambda) else {
        return;
    };
    let binders: Vec<_> = use_
        .rows
        .iter()
        .map(|(variable, _)| {
            let tail = builder
                .ctx
                .ability_row(AbilityRow::open(builder.ctx.db, *variable))
                .tail
                .expect("an open row has a tail");
            Attribute::Int(i128::from(tail))
        })
        .collect();
    builder.ir.op_attrs_mut(op).insert(
        tribute_control::QUANTIFIED_ROWS_ATTR,
        Attribute::List(binders),
    );
}

/// The value the local reference `node` reads: a lambda that quantifies
/// rows is instantiated with the rows this use passes.
pub(super) fn lower_local_reference<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    node: crate::ast::NodeId,
    local: crate::ast::LocalId,
    declarations: &Declarations<'db>,
) -> Option<ValueRef> {
    let value = local_callables::lookup(builder.ctx, node, declarations)
        .or_else(|| builder.ctx.lookup(local))?;
    let trunk_ir::refs::ValueDef::OpResult(lambda, _) = builder.ir.value_def(value) else {
        return Some(value);
    };
    let quantified = tribute_control::quantified_rows(builder.ir, lambda)
        .expect("a lowered lambda has well-formed quantified rows");
    if quantified.is_empty() || !tribute_control::Lambda::matches(builder.ir, lambda) {
        return Some(value);
    }
    let use_ = declarations
        .local_uses
        .iter()
        .find(|use_| use_.site == node)
        .expect("typeck records an instance for every use of a binding or for none");
    let rows: Vec<_> = use_
        .rows
        .iter()
        .map(|(_, row)| builder.ctx.ability_row(*row))
        .collect();
    let ty = tribute_control::substitute_rows(
        builder.ir,
        builder.ir.value_ty(value),
        tribute_control::RowArgs::Binders(&quantified, &rows),
    )
    .expect("instantiation substitutes only the lambda's binders");
    let location = builder.location(node);
    let instantiate = tribute_control::Instantiate::operands(value)
        .row_args(rows)
        .results(ty)
        .build(builder.ir, location);
    builder.ir.push_op(builder.block, instantiate.op_ref());
    Some(instantiate.result(builder.ir))
}

pub(super) fn lower_statement<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    statement: Stmt<TypedRef<'db>>,
    declarations: &mut Declarations<'db>,
) {
    match statement {
        Stmt::Let { pattern, value, .. } => {
            let lowered = local_callables::materialize(builder, &pattern, &value, declarations)
                .or_else(|| {
                    let lambda = matches!(&*value.kind, ExprKind::Lambda { .. });
                    let lowered = lower_expr(builder, value, declarations)?;
                    if lambda {
                        quantify_local_rows(builder, pattern.id, lowered, declarations);
                    }
                    Some(lowered)
                });
            if let Some(value) = lowered {
                bind_pattern(builder, &pattern, value);
            }
        }
        Stmt::Expr { expr, .. } => {
            let _ = lower_expr(builder, expr, declarations);
        }
    }
}

pub(super) fn bind_pattern<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    pattern: &Pattern<TypedRef<'db>>,
    value: ValueRef,
) {
    match &*pattern.kind {
        PatternKind::Bind {
            name,
            local_id: Some(id),
        } => builder.ctx.bind(*id, name.clone(), value),
        PatternKind::Wildcard => {}
        _ => super::super::case::bind_logical_pattern_fields(
            builder.ctx,
            builder.ir,
            builder.block,
            builder.location(pattern.id),
            value,
            pattern,
        ),
    }
}
