//! Case expression and pattern matching lowering.
//!
//! Lowers case expressions to a chain of `scf.if` operations, with
//! pattern checks generating boolean conditions and pattern bindings
//! extracted inside the matched region.

use tribute_ir::dialect::list;
use trunk_ir::Symbol;
use trunk_ir::adt_layout::get_enum_variants;
use trunk_ir::context::{BlockData, IrContext, RegionData};
use trunk_ir::dialect::{adt, arith, scf};
use trunk_ir::refs::{BlockRef, TypeRef, ValueRef};
use trunk_ir::types::{Attribute, Location};

use crate::ast::{LiteralPattern, Pattern, PatternKind, ResolvedRef, TypedRef};

use super::super::context::IrLoweringCtx;
use super::IrBuilder;

pub(super) fn emit_logical_pattern_check<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    location: Location,
    scrutinee: ValueRef,
    pattern: &Pattern<TypedRef<'db>>,
) -> Option<ValueRef> {
    let bool_ty = builder.ctx.bool_type(builder.ir);
    match &*pattern.kind {
        PatternKind::Wildcard | PatternKind::Bind { .. } | PatternKind::Error => {
            let op = arith::r#const(builder.ir, location, bool_ty, Attribute::Bool(true));
            builder.ir.push_op(builder.block, op.op_ref());
            Some(op.result(builder.ir))
        }
        PatternKind::Literal(literal) => emit_literal_check(builder, location, scrutinee, literal),
        PatternKind::Tuple(elements) => {
            let struct_ty =
                super::get_or_create_logical_tuple_type(builder.ctx, builder.ir, pattern.id)
                    .unwrap_or_else(|| panic!("missing typechecked logical tuple layout"))
                    .1;
            let mut conditions = Vec::with_capacity(elements.len());
            for (index, element) in elements.iter().enumerate() {
                let element_ty = builder
                    .ctx
                    .get_node_type(element.id)
                    .copied()
                    .map(|ty| builder.ctx.convert_logical_type(builder.ir, ty))
                    .unwrap_or_else(|| panic!("missing typechecked logical tuple pattern field"));
                let get = adt::struct_get(
                    builder.ir,
                    location,
                    scrutinee,
                    element_ty,
                    struct_ty,
                    index as u32,
                );
                builder.ir.push_op(builder.block, get.op_ref());
                conditions.push(emit_logical_pattern_check(
                    builder,
                    location,
                    get.result(builder.ir),
                    element,
                )?);
            }
            let Some((first, rest)) = conditions.split_first() else {
                let op = arith::r#const(builder.ir, location, bool_ty, Attribute::Bool(true));
                builder.ir.push_op(builder.block, op.op_ref());
                return Some(op.result(builder.ir));
            };
            let mut result = *first;
            for condition in rest {
                let and = arith::and(builder.ir, location, result, *condition, bool_ty);
                builder.ir.push_op(builder.block, and.op_ref());
                result = and.result(builder.ir);
            }
            Some(result)
        }
        PatternKind::List(elements) => {
            emit_logical_list_pattern_check(builder, location, scrutinee, pattern, elements, true)
        }
        PatternKind::ListRest { head, .. } => {
            emit_logical_list_pattern_check(builder, location, scrutinee, pattern, head, false)
        }
        PatternKind::As { pattern, .. } => {
            emit_logical_pattern_check(builder, location, scrutinee, pattern)
        }
        PatternKind::Variant { ctor, fields } => {
            emit_logical_variant_pattern_check(builder, location, scrutinee, ctor, fields)
        }
        _ => panic!("unsupported logical pattern at source-logical boundary"),
    }
}

fn logical_variant_layout<'db>(
    ctx: &IrLoweringCtx<'db>,
    ir: &mut IrContext,
    ctor: &TypedRef<'db>,
) -> (Symbol, TypeRef) {
    let ResolvedRef::Constructor { variant, .. } = ctor.resolved else {
        panic!("non-constructor in logical variant pattern");
    };
    let layout = super::resolve_enum_type_attr_for_constructor(ctx, ir, &ctor.resolved, ctor.ty);
    (variant, layout)
}

fn emit_logical_variant_pattern_check<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    location: Location,
    scrutinee: ValueRef,
    ctor: &TypedRef<'db>,
    fields: &[Pattern<TypedRef<'db>>],
) -> Option<ValueRef> {
    let bool_ty = builder.ctx.bool_type(builder.ir);
    let (variant, enum_ty) = logical_variant_layout(builder.ctx, builder.ir, ctor);
    let tag = adt::variant_is(builder.ir, location, scrutinee, bool_ty, enum_ty, variant);
    builder.ir.push_op(builder.block, tag.op_ref());
    if fields.is_empty() {
        return Some(tag.result(builder.ir));
    }
    let then_block = builder.ir.create_block(BlockData {
        location,
        args: vec![],
        ops: Default::default(),
        parent_region: None,
    });
    let then_value = {
        let mut nested = IrBuilder::new(builder.ctx, builder.ir, then_block);
        let cast = adt::variant_cast(nested.ir, location, scrutinee, enum_ty, enum_ty, variant);
        nested.ir.push_op(nested.block, cast.op_ref());
        let variant_fields = get_enum_variants(nested.ir, enum_ty)
            .expect("logical enum layout must contain variants")
            .into_iter()
            .find_map(|(tag, fields)| (tag == variant).then_some(fields))
            .expect("resolved logical enum variant must exist");
        let mut conditions = Vec::with_capacity(fields.len());
        for (index, field) in fields.iter().enumerate() {
            let field_ty = *variant_fields
                .get(index)
                .expect("type checking must reject out-of-range logical variant fields");
            let get = adt::variant_get(
                nested.ir,
                location,
                cast.result(nested.ir),
                field_ty,
                enum_ty,
                variant,
                index as u32,
            );
            nested.ir.push_op(nested.block, get.op_ref());
            let value = get.result(nested.ir);
            conditions.push(emit_logical_pattern_check(
                &mut nested,
                location,
                value,
                field,
            )?);
        }
        let mut combined = conditions[0];
        for condition in conditions.into_iter().skip(1) {
            let and = arith::and(nested.ir, location, combined, condition, bool_ty);
            nested.ir.push_op(nested.block, and.op_ref());
            combined = and.result(nested.ir);
        }
        combined
    };
    let yield_op = scf::r#yield(builder.ir, location, [then_value]);
    builder.ir.push_op(then_block, yield_op.op_ref());
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
    let false_value = arith::r#const(builder.ir, location, bool_ty, Attribute::Bool(false));
    builder.ir.push_op(else_block, false_value.op_ref());
    let yield_op = scf::r#yield(builder.ir, location, [false_value.result(builder.ir)]);
    builder.ir.push_op(else_block, yield_op.op_ref());
    let else_region = builder.ir.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![else_block],
        parent_op: None,
    });
    let checked = scf::r#if(
        builder.ir,
        location,
        tag.result(builder.ir),
        bool_ty,
        then_region,
        else_region,
    );
    builder.ir.push_op(builder.block, checked.op_ref());
    Some(checked.result(builder.ir))
}

fn logical_list_types<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    ir: &mut IrContext,
    pattern: &Pattern<TypedRef<'db>>,
) -> (TypeRef, TypeRef) {
    let list_source = ctx
        .get_node_type(pattern.id)
        .copied()
        .unwrap_or_else(|| panic!("missing typechecked logical list pattern type"));
    let crate::ast::TypeKind::Named { id, args, .. } = list_source.kind(ctx.db()) else {
        panic!("logical list pattern has non-list source type");
    };
    if !id.is_builtin_list(ctx.db()) || args.len() != 1 {
        panic!("logical list pattern has malformed List type");
    }
    (
        ctx.convert_logical_type(ir, list_source),
        ctx.convert_logical_type(ir, args[0]),
    )
}

fn emit_logical_list_pattern_check<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    location: Location,
    scrutinee: ValueRef,
    whole_pattern: &Pattern<TypedRef<'db>>,
    elements: &[Pattern<TypedRef<'db>>],
    exact: bool,
) -> Option<ValueRef> {
    let (list_ty, element_ty) = logical_list_types(builder.ctx, builder.ir, whole_pattern);
    emit_logical_list_pattern_suffix(
        builder, location, scrutinee, elements, exact, list_ty, element_ty,
    )
}

#[allow(clippy::too_many_arguments)]
fn emit_logical_list_pattern_suffix<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    location: Location,
    current: ValueRef,
    elements: &[Pattern<TypedRef<'db>>],
    exact: bool,
    list_ty: TypeRef,
    element_ty: TypeRef,
) -> Option<ValueRef> {
    let bool_ty = builder.ctx.bool_type(builder.ir);
    let Some((element, rest)) = elements.split_first() else {
        let terminal = if exact {
            list::is_empty(builder.ir, location, current, bool_ty, element_ty).op_ref()
        } else {
            arith::r#const(builder.ir, location, bool_ty, Attribute::Bool(true)).op_ref()
        };
        builder.ir.push_op(builder.block, terminal);
        return Some(builder.ir.op_result(terminal, 0));
    };
    let empty = list::is_empty(builder.ir, location, current, bool_ty, element_ty);
    builder.ir.push_op(builder.block, empty.op_ref());
    let true_value = arith::r#const(builder.ir, location, bool_ty, Attribute::Bool(true));
    builder.ir.push_op(builder.block, true_value.op_ref());
    let non_empty = arith::xor(
        builder.ir,
        location,
        empty.result(builder.ir),
        true_value.result(builder.ir),
        bool_ty,
    );
    builder.ir.push_op(builder.block, non_empty.op_ref());
    let then_block = builder.ir.create_block(BlockData {
        location,
        args: vec![],
        ops: Default::default(),
        parent_region: None,
    });
    let then_value = {
        let mut nested = IrBuilder::new(builder.ctx, builder.ir, then_block);
        let head = list::head(nested.ir, location, current, element_ty, element_ty);
        nested.ir.push_op(nested.block, head.op_ref());
        let head_value = head.result(nested.ir);
        let condition = emit_logical_pattern_check(&mut nested, location, head_value, element)?;
        let match_block = nested.ir.create_block(BlockData {
            location,
            args: vec![],
            ops: Default::default(),
            parent_region: None,
        });
        let suffix = {
            let mut matched = IrBuilder::new(nested.ctx, nested.ir, match_block);
            let tail = list::tail(matched.ir, location, current, list_ty, element_ty);
            matched.ir.push_op(matched.block, tail.op_ref());
            let tail_value = tail.result(matched.ir);
            emit_logical_list_pattern_suffix(
                &mut matched,
                location,
                tail_value,
                rest,
                exact,
                list_ty,
                element_ty,
            )?
        };
        let yield_op = scf::r#yield(nested.ir, location, [suffix]);
        nested.ir.push_op(match_block, yield_op.op_ref());
        let match_region = nested.ir.create_region(RegionData {
            location,
            blocks: trunk_ir::smallvec::smallvec![match_block],
            parent_op: None,
        });
        let mismatch_block = nested.ir.create_block(BlockData {
            location,
            args: vec![],
            ops: Default::default(),
            parent_region: None,
        });
        let false_value = arith::r#const(nested.ir, location, bool_ty, Attribute::Bool(false));
        nested.ir.push_op(mismatch_block, false_value.op_ref());
        let yield_op = scf::r#yield(nested.ir, location, [false_value.result(nested.ir)]);
        nested.ir.push_op(mismatch_block, yield_op.op_ref());
        let mismatch_region = nested.ir.create_region(RegionData {
            location,
            blocks: trunk_ir::smallvec::smallvec![mismatch_block],
            parent_op: None,
        });
        let guarded = scf::r#if(
            nested.ir,
            location,
            condition,
            bool_ty,
            match_region,
            mismatch_region,
        );
        nested.ir.push_op(nested.block, guarded.op_ref());
        guarded.result(nested.ir)
    };
    let yield_op = scf::r#yield(builder.ir, location, [then_value]);
    builder.ir.push_op(then_block, yield_op.op_ref());
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
    let false_value = arith::r#const(builder.ir, location, bool_ty, Attribute::Bool(false));
    builder.ir.push_op(else_block, false_value.op_ref());
    let yield_op = scf::r#yield(builder.ir, location, [false_value.result(builder.ir)]);
    builder.ir.push_op(else_block, yield_op.op_ref());
    let else_region = builder.ir.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![else_block],
        parent_op: None,
    });
    let guarded = scf::r#if(
        builder.ir,
        location,
        non_empty.result(builder.ir),
        bool_ty,
        then_region,
        else_region,
    );
    builder.ir.push_op(builder.block, guarded.op_ref());
    Some(guarded.result(builder.ir))
}

/// Emit a literal equality check.
fn emit_literal_check<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    location: Location,
    scrutinee: ValueRef,
    lit: &LiteralPattern,
) -> Option<ValueRef> {
    let bool_ty = builder.ctx.bool_type(builder.ir);
    let i32_ty = builder.ctx.i32_type(builder.ir);

    match lit {
        LiteralPattern::Nat(n) => {
            let value = super::validate_nat_i31(builder.db(), location, *n)?;
            let const_op =
                arith::r#const(builder.ir, location, i32_ty, Attribute::Int(value as i128));
            builder.ir.push_op(builder.block, const_op.op_ref());
            let const_val = const_op.result(builder.ir);
            let cmp_op = arith::cmpi(
                builder.ir,
                location,
                scrutinee,
                const_val,
                bool_ty,
                Symbol::new("eq"),
            );
            builder.ir.push_op(builder.block, cmp_op.op_ref());
            Some(cmp_op.result(builder.ir))
        }
        LiteralPattern::Int(n) => {
            let value = super::validate_int_i31(builder.db(), location, *n)?;
            let const_op =
                arith::r#const(builder.ir, location, i32_ty, Attribute::Int(value as i128));
            builder.ir.push_op(builder.block, const_op.op_ref());
            let const_val = const_op.result(builder.ir);
            let cmp_op = arith::cmpi(
                builder.ir,
                location,
                scrutinee,
                const_val,
                bool_ty,
                Symbol::new("eq"),
            );
            builder.ir.push_op(builder.block, cmp_op.op_ref());
            Some(cmp_op.result(builder.ir))
        }
        LiteralPattern::Bool(b) => {
            let const_op = arith::r#const(builder.ir, location, bool_ty, Attribute::Bool(*b));
            builder.ir.push_op(builder.block, const_op.op_ref());
            let const_val = const_op.result(builder.ir);
            let cmp_op = arith::cmpi(
                builder.ir,
                location,
                scrutinee,
                const_val,
                bool_ty,
                Symbol::new("eq"),
            );
            builder.ir.push_op(builder.block, cmp_op.op_ref());
            Some(cmp_op.result(builder.ir))
        }
        _ => {
            unreachable!("unsupported literal pattern in IR lowering: {:?}", lit)
        }
    }
}

/// Logical counterpart to [`bind_pattern_fields`].  Tuple and list extraction
/// must preserve recursively logical element types; using physical type erasure
/// would create `func.func_sig` fields before the shared CPS boundary.
pub(super) fn bind_logical_pattern_fields<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    ir: &mut IrContext,
    block: BlockRef,
    location: Location,
    scrutinee: ValueRef,
    pattern: &Pattern<TypedRef<'db>>,
) {
    match &*pattern.kind {
        PatternKind::Bind {
            name,
            local_id: Some(id),
        } => ctx.bind(*id, *name, scrutinee),
        PatternKind::Wildcard | PatternKind::Literal(_) | PatternKind::Error => {}
        PatternKind::Bind { local_id: None, .. } => {}
        PatternKind::Tuple(elements) => {
            let struct_ty = super::get_or_create_logical_tuple_type(ctx, ir, pattern.id)
                .unwrap_or_else(|| panic!("missing typechecked logical tuple layout"))
                .1;
            for (index, element) in elements.iter().enumerate() {
                let element_ty = ctx
                    .get_node_type(element.id)
                    .copied()
                    .map(|ty| ctx.convert_logical_type(ir, ty))
                    .unwrap_or_else(|| panic!("missing logical tuple pattern field type"));
                let get =
                    adt::struct_get(ir, location, scrutinee, element_ty, struct_ty, index as u32);
                ir.push_op(block, get.op_ref());
                bind_logical_pattern_fields(ctx, ir, block, location, get.result(ir), element);
            }
        }
        PatternKind::List(elements) => bind_logical_list_pattern_fields(
            ctx, ir, block, location, scrutinee, pattern, elements, None,
        ),
        PatternKind::ListRest {
            head,
            rest,
            rest_local_id,
        } => bind_logical_list_pattern_fields(
            ctx,
            ir,
            block,
            location,
            scrutinee,
            pattern,
            head,
            rest.zip(*rest_local_id),
        ),
        PatternKind::As {
            pattern,
            name,
            local_id,
        } => {
            if let Some(id) = local_id {
                ctx.bind(*id, *name, scrutinee);
            }
            bind_logical_pattern_fields(ctx, ir, block, location, scrutinee, pattern);
        }
        PatternKind::Variant { ctor, fields } => {
            let (variant, enum_ty) = logical_variant_layout(ctx, ir, ctor);
            let cast = adt::variant_cast(ir, location, scrutinee, enum_ty, enum_ty, variant);
            ir.push_op(block, cast.op_ref());
            let variant_fields = get_enum_variants(ir, enum_ty)
                .expect("logical enum layout must contain variants")
                .into_iter()
                .find_map(|(tag, fields)| (tag == variant).then_some(fields))
                .expect("resolved logical enum variant must exist");
            for (index, field) in fields.iter().enumerate() {
                let field_ty = *variant_fields
                    .get(index)
                    .expect("type checking must reject out-of-range logical variant fields");
                let get = adt::variant_get(
                    ir,
                    location,
                    cast.result(ir),
                    field_ty,
                    enum_ty,
                    variant,
                    index as u32,
                );
                ir.push_op(block, get.op_ref());
                bind_logical_pattern_fields(ctx, ir, block, location, get.result(ir), field);
            }
        }
        _ => panic!("unsupported logical pattern binding at source-logical boundary"),
    }
}

#[allow(clippy::too_many_arguments)]
fn bind_logical_list_pattern_fields<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    ir: &mut IrContext,
    block: BlockRef,
    location: Location,
    scrutinee: ValueRef,
    whole_pattern: &Pattern<TypedRef<'db>>,
    elements: &[Pattern<TypedRef<'db>>],
    rest: Option<(Symbol, crate::ast::LocalId)>,
) {
    let (list_ty, element_ty) = logical_list_types(ctx, ir, whole_pattern);
    let mut current = scrutinee;
    for element in elements {
        let head = list::head(ir, location, current, element_ty, element_ty);
        ir.push_op(block, head.op_ref());
        bind_logical_pattern_fields(ctx, ir, block, location, head.result(ir), element);
        let tail = list::tail(ir, location, current, list_ty, element_ty);
        ir.push_op(block, tail.op_ref());
        current = tail.result(ir);
    }
    if let Some((name, id)) = rest {
        ctx.bind(id, name, current);
    }
}
