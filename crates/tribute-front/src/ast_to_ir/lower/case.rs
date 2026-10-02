//! Case expression and pattern matching lowering.
//!
//! Lowers case expressions to a chain of `scf.if` operations, with
//! pattern checks generating boolean conditions and pattern bindings
//! extracted inside the matched region.

use salsa::Accumulator;
use tribute_core::diagnostic::{CompilationPhase, Diagnostic, DiagnosticSeverity};
use tribute_ir::dialect::list;
use trunk_ir::Symbol;
use trunk_ir::adt_layout::{get_enum_variants, get_struct_fields};
use trunk_ir::context::{BlockData, IrContext, RegionData};
use trunk_ir::dialect::{adt, arith, scf};
use trunk_ir::refs::{BlockRef, TypeRef, ValueRef};
use trunk_ir::types::{Attribute, Location, StringRef};

use crate::ast::{LiteralPattern, NodeId, Pattern, PatternKind, ResolvedRef, TypedRef};

use super::super::context::IrLoweringCtx;
use super::IrBuilder;

/// The logical tuple layout of a tuple pattern and its field types.
///
/// Field types come from the tuple layout, not from the element patterns: a
/// constructor pattern's node type is the constructor's callable type, not
/// the field value it matches.
fn logical_tuple_pattern_layout<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    ir: &mut IrContext,
    pattern: NodeId,
) -> (TypeRef, Vec<TypeRef>) {
    let struct_ty = super::get_or_create_logical_tuple_type(ctx, ir, pattern)
        .unwrap_or_else(|| panic!("missing typechecked logical tuple layout"))
        .1;
    let field_tys = get_struct_fields(ir, struct_ty)
        .expect("logical tuple layout must be an adt.struct")
        .into_iter()
        .map(|(_, ty)| ty)
        .collect();
    (struct_ty, field_tys)
}

pub(super) fn emit_logical_pattern_check<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    location: Location,
    scrutinee: ValueRef,
    pattern: &Pattern<TypedRef<'db>>,
) -> Option<ValueRef> {
    let bool_ty = builder.ctx.bool_type(builder.ir);
    match &*pattern.kind {
        PatternKind::Wildcard | PatternKind::Bind { .. } | PatternKind::Error => {
            let op = arith::Const::operands()
                .value(Attribute::Bool(true))
                .results(bool_ty)
                .build(builder.ir, location);
            builder.ir.push_op(builder.block, op.op_ref());
            Some(op.result(builder.ir))
        }
        PatternKind::Literal(literal) => emit_literal_check(builder, location, scrutinee, literal),
        PatternKind::Tuple(elements) => {
            let (struct_ty, field_tys) =
                logical_tuple_pattern_layout(builder.ctx, builder.ir, pattern.id);
            let mut conditions = Vec::with_capacity(elements.len());
            for ((index, element), element_ty) in elements.iter().enumerate().zip(field_tys) {
                let get = adt::StructGet::operands(scrutinee)
                    .r#type(struct_ty)
                    .field(index as u32)
                    .results(element_ty)
                    .build(builder.ir, location);
                builder.ir.push_op(builder.block, get.op_ref());
                conditions.push(emit_logical_pattern_check(
                    builder,
                    location,
                    get.result(builder.ir),
                    element,
                )?);
            }
            Some(combine_conditions(builder, location, conditions))
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
        PatternKind::Variant { .. } | PatternKind::Record { .. } => {
            let (layout, fields) = logical_constructor_pattern(builder.ctx, builder.ir, pattern);
            match layout {
                ConstructorLayout::Struct {
                    ty,
                    fields: field_tys,
                } => {
                    let mut conditions = Vec::with_capacity(fields.len());
                    for (index, field) in fields {
                        let get = adt::StructGet::operands(scrutinee)
                            .r#type(ty)
                            .field(index as u32)
                            .results(field_tys[index])
                            .build(builder.ir, location);
                        builder.ir.push_op(builder.block, get.op_ref());
                        conditions.push(emit_logical_pattern_check(
                            builder,
                            location,
                            get.result(builder.ir),
                            field,
                        )?);
                    }
                    Some(combine_conditions(builder, location, conditions))
                }
                ConstructorLayout::Variant(layout) => emit_logical_variant_pattern_check(
                    builder, location, scrutinee, &layout, &fields,
                ),
            }
        }
    }
}

/// Conjunction of pattern conditions; `true` when there are none.
fn combine_conditions(
    builder: &mut IrBuilder<'_, '_>,
    location: Location,
    conditions: Vec<ValueRef>,
) -> ValueRef {
    let Some((first, rest)) = conditions.split_first() else {
        let bool_ty = builder.ctx.bool_type(builder.ir);
        let op = arith::Const::operands()
            .value(Attribute::Bool(true))
            .results(bool_ty)
            .build(builder.ir, location);
        builder.ir.push_op(builder.block, op.op_ref());
        return op.result(builder.ir);
    };
    let mut result = *first;
    for condition in rest {
        let and = arith::And::operands(result, *condition).build(builder.ir, location);
        builder.ir.push_op(builder.block, and.op_ref());
        result = and.result(builder.ir);
    }
    result
}

/// The logical layout a constructor pattern destructures.
enum ConstructorLayout {
    /// A struct: its fields are read directly.
    Struct { ty: TypeRef, fields: Vec<TypeRef> },
    /// An enum variant: the tag is tested before its fields are read.
    Variant(VariantLayout),
}

/// The enum layout of one variant: the enum type, the variant's tag, and its
/// field types.
struct VariantLayout {
    ty: TypeRef,
    tag: StringRef,
    fields: Vec<TypeRef>,
}

/// The layout of a constructor pattern and its sub-patterns paired with the
/// field index each one matches. Brace-form fields are matched by name.
fn logical_constructor_pattern<'p, 'db>(
    ctx: &IrLoweringCtx<'db>,
    ir: &mut IrContext,
    pattern: &'p Pattern<TypedRef<'db>>,
) -> (ConstructorLayout, Vec<(usize, &'p Pattern<TypedRef<'db>>)>) {
    let ctor = match &*pattern.kind {
        PatternKind::Variant { ctor, .. }
        | PatternKind::Record {
            type_name: ctor, ..
        } => ctor,
        _ => panic!("unsupported logical constructor pattern at source-logical boundary"),
    };
    let ResolvedRef::Constructor { variant, .. } = ctor.resolved else {
        panic!("non-constructor in logical constructor pattern");
    };
    let ty = super::resolve_enum_type_attr_for_constructor(ctx, ir, &ctor.resolved, ctor.ty);
    let (layout, names) = match get_struct_fields(ir, ty) {
        Some(fields) => {
            let (names, fields) = fields
                .into_iter()
                .map(|(name, ty)| (Symbol::from_dynamic(ir.str(name)), ty))
                .unzip();
            (ConstructorLayout::Struct { ty, fields }, names)
        }
        None => {
            let fields = get_enum_variants(ir, ty)
                .expect("logical constructor layout must be a struct or an enum")
                .into_iter()
                .find_map(|(tag, fields)| (variant == ir.str(tag)).then_some(fields))
                .expect("resolved logical enum variant must exist");
            let names = match &*pattern.kind {
                PatternKind::Record { .. } => ctx
                    .variant_field_names(ty, variant)
                    .expect("named-field variant must have registered field names"),
                _ => Vec::new(),
            };
            (
                ConstructorLayout::Variant(VariantLayout {
                    ty,
                    tag: ir.intern_symbol_text(variant),
                    fields,
                }),
                names,
            )
        }
    };
    let field_count = match &layout {
        ConstructorLayout::Struct { fields, .. }
        | ConstructorLayout::Variant(VariantLayout { fields, .. }) => fields.len(),
    };
    let indexed: Vec<_> = match &*pattern.kind {
        PatternKind::Variant { fields, .. } => fields.iter().enumerate().collect(),
        PatternKind::Record { fields, .. } => fields
            .iter()
            .map(|field| {
                let index = names
                    .iter()
                    .position(|name| *name == field.name)
                    .expect("type checking must reject unknown record pattern fields");
                let pattern = field
                    .pattern
                    .as_ref()
                    .expect("name resolution must expand record pattern shorthand");
                (index, pattern)
            })
            .collect(),
        _ => unreachable!(),
    };
    assert!(
        indexed.iter().all(|(index, _)| *index < field_count),
        "type checking must reject out-of-range logical constructor fields"
    );
    (layout, indexed)
}

fn emit_logical_variant_pattern_check<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    location: Location,
    scrutinee: ValueRef,
    layout: &VariantLayout,
    fields: &[(usize, &Pattern<TypedRef<'db>>)],
) -> Option<ValueRef> {
    let &VariantLayout {
        ty: enum_ty,
        tag: variant,
        fields: ref variant_fields,
    } = layout;
    let bool_ty = builder.ctx.bool_type(builder.ir);
    let tag = adt::VariantIs::operands(scrutinee)
        .r#type(enum_ty)
        .tag(variant)
        .results(bool_ty)
        .build(builder.ir, location);
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
        let cast = adt::VariantCast::operands(scrutinee)
            .r#type(enum_ty)
            .tag(variant)
            .results(enum_ty)
            .build(nested.ir, location);
        nested.ir.push_op(nested.block, cast.op_ref());
        let mut conditions = Vec::with_capacity(fields.len());
        for &(index, field) in fields {
            let field_ty = variant_fields[index];
            let get = adt::VariantGet::operands(cast.result(nested.ir))
                .r#type(enum_ty)
                .tag(variant)
                .field(index as u32)
                .results(field_ty)
                .build(nested.ir, location);
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
            let and = arith::And::operands(combined, condition).build(nested.ir, location);
            nested.ir.push_op(nested.block, and.op_ref());
            combined = and.result(nested.ir);
        }
        combined
    };
    let yield_op = scf::Yield::operands([then_value]).build(builder.ir, location);
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
    let false_value = arith::Const::operands()
        .value(Attribute::Bool(false))
        .results(bool_ty)
        .build(builder.ir, location);
    builder.ir.push_op(else_block, false_value.op_ref());
    let yield_op =
        scf::Yield::operands([false_value.result(builder.ir)]).build(builder.ir, location);
    builder.ir.push_op(else_block, yield_op.op_ref());
    let else_region = builder.ir.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![else_block],
        parent_op: None,
    });
    let checked = scf::If::operands(tag.result(builder.ir))
        .results(bool_ty)
        .regions(then_region, else_region)
        .build(builder.ir, location);
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
            list::IsEmpty::operands(current)
                .element_type(element_ty)
                .results(bool_ty)
                .build(builder.ir, location)
                .op_ref()
        } else {
            arith::Const::operands()
                .value(Attribute::Bool(true))
                .results(bool_ty)
                .build(builder.ir, location)
                .op_ref()
        };
        builder.ir.push_op(builder.block, terminal);
        return Some(builder.ir.op_result(terminal, 0));
    };
    let empty = list::IsEmpty::operands(current)
        .element_type(element_ty)
        .results(bool_ty)
        .build(builder.ir, location);
    builder.ir.push_op(builder.block, empty.op_ref());
    let true_value = arith::Const::operands()
        .value(Attribute::Bool(true))
        .results(bool_ty)
        .build(builder.ir, location);
    builder.ir.push_op(builder.block, true_value.op_ref());
    let non_empty = arith::Xor::operands(empty.result(builder.ir), true_value.result(builder.ir))
        .build(builder.ir, location);
    builder.ir.push_op(builder.block, non_empty.op_ref());
    let then_block = builder.ir.create_block(BlockData {
        location,
        args: vec![],
        ops: Default::default(),
        parent_region: None,
    });
    let then_value = {
        let mut nested = IrBuilder::new(builder.ctx, builder.ir, then_block);
        let head = list::Head::operands(current)
            .element_type(element_ty)
            .results(element_ty)
            .build(nested.ir, location);
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
            let tail = list::Tail::operands(current)
                .element_type(element_ty)
                .results(list_ty)
                .build(matched.ir, location);
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
        let yield_op = scf::Yield::operands([suffix]).build(nested.ir, location);
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
        let false_value = arith::Const::operands()
            .value(Attribute::Bool(false))
            .results(bool_ty)
            .build(nested.ir, location);
        nested.ir.push_op(mismatch_block, false_value.op_ref());
        let yield_op =
            scf::Yield::operands([false_value.result(nested.ir)]).build(nested.ir, location);
        nested.ir.push_op(mismatch_block, yield_op.op_ref());
        let mismatch_region = nested.ir.create_region(RegionData {
            location,
            blocks: trunk_ir::smallvec::smallvec![mismatch_block],
            parent_op: None,
        });
        let guarded = scf::If::operands(condition)
            .results(bool_ty)
            .regions(match_region, mismatch_region)
            .build(nested.ir, location);
        nested.ir.push_op(nested.block, guarded.op_ref());
        guarded.result(nested.ir)
    };
    let yield_op = scf::Yield::operands([then_value]).build(builder.ir, location);
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
    let false_value = arith::Const::operands()
        .value(Attribute::Bool(false))
        .results(bool_ty)
        .build(builder.ir, location);
    builder.ir.push_op(else_block, false_value.op_ref());
    let yield_op =
        scf::Yield::operands([false_value.result(builder.ir)]).build(builder.ir, location);
    builder.ir.push_op(else_block, yield_op.op_ref());
    let else_region = builder.ir.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![else_block],
        parent_op: None,
    });
    let guarded = scf::If::operands(non_empty.result(builder.ir))
        .results(bool_ty)
        .regions(then_region, else_region)
        .build(builder.ir, location);
    builder.ir.push_op(builder.block, guarded.op_ref());
    Some(guarded.result(builder.ir))
}

/// Emit a literal equality check. A literal pattern matches the values its
/// type's `==` finds equal to the literal.
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
            let literal = emit_const(builder, location, Attribute::Int(value as i128), i32_ty);
            Some(emit_cmpi_eq(builder, location, scrutinee, literal))
        }
        LiteralPattern::Int(n) => {
            let value = super::validate_int_i31(builder.db(), location, *n)?;
            let literal = emit_const(builder, location, Attribute::Int(value as i128), i32_ty);
            Some(emit_cmpi_eq(builder, location, scrutinee, literal))
        }
        LiteralPattern::Rune(c) => {
            let literal = emit_const(builder, location, Attribute::Int(*c as i32 as i128), i32_ty);
            Some(emit_cmpi_eq(builder, location, scrutinee, literal))
        }
        LiteralPattern::Bool(b) => {
            let literal = emit_const(builder, location, Attribute::Bool(*b), bool_ty);
            Some(emit_cmpi_eq(builder, location, scrutinee, literal))
        }
        LiteralPattern::Float(value) => {
            let f64_ty = builder.ctx.f64_type(builder.ir);
            let literal = emit_const(
                builder,
                location,
                Attribute::FloatBits(value.value().to_bits()),
                f64_ty,
            );
            let cmp_op = arith::Cmpf::operands(scrutinee, literal)
                .predicate("oeq")
                .build(builder.ir, location);
            builder.ir.push_op(builder.block, cmp_op.op_ref());
            Some(cmp_op.result(builder.ir))
        }
        LiteralPattern::String(text) => {
            let equality = builder.ctx.literal_equalities().string;
            let anyref_ty = builder.ctx.anyref_type(builder.ir);
            let text = builder.ir.intern_str(text);
            let literal = adt::StringConst::operands()
                .value(text)
                .results(anyref_ty)
                .build(builder.ir, location);
            builder.ir.push_op(builder.block, literal.op_ref());
            let literal = literal.result(builder.ir);
            emit_equality_call(builder, location, equality, "String", scrutinee, literal)
        }
        LiteralPattern::Bytes(bytes) => {
            let equality = builder.ctx.literal_equalities().bytes;
            let bytes_ty = builder.ctx.bytes_type(builder.ir);
            let literal = adt::BytesConst::operands()
                .value(bytes.clone().into())
                .results(bytes_ty)
                .build(builder.ir, location);
            builder.ir.push_op(builder.block, literal.op_ref());
            let literal = literal.result(builder.ir);
            emit_equality_call(builder, location, equality, "Bytes", scrutinee, literal)
        }
        // `Nil` is the only value of its type.
        LiteralPattern::Nil => Some(emit_const(
            builder,
            location,
            Attribute::Bool(true),
            bool_ty,
        )),
    }
}

fn emit_const(
    builder: &mut IrBuilder<'_, '_>,
    location: Location,
    value: Attribute,
    ty: TypeRef,
) -> ValueRef {
    let const_op = arith::Const::operands()
        .value(value)
        .results(ty)
        .build(builder.ir, location);
    builder.ir.push_op(builder.block, const_op.op_ref());
    const_op.result(builder.ir)
}

fn emit_cmpi_eq(
    builder: &mut IrBuilder<'_, '_>,
    location: Location,
    scrutinee: ValueRef,
    literal: ValueRef,
) -> ValueRef {
    let cmp_op = arith::Cmpi::operands(scrutinee, literal)
        .predicate("eq")
        .build(builder.ir, location);
    builder.ir.push_op(builder.block, cmp_op.op_ref());
    cmp_op.result(builder.ir)
}

/// Compare `scrutinee` with `literal` through the prelude's `==` for
/// `type_name`, which a program without the prelude lacks.
fn emit_equality_call(
    builder: &mut IrBuilder<'_, '_>,
    location: Location,
    equality: Option<Symbol>,
    type_name: &str,
    scrutinee: ValueRef,
    literal: ValueRef,
) -> Option<ValueRef> {
    let Some(equality) = equality else {
        Diagnostic::new(
            format!("{type_name} literal patterns need the prelude's `{type_name}::==`"),
            location.span,
            DiagnosticSeverity::Error,
            CompilationPhase::Lowering,
        )
        .accumulate(builder.db());
        return None;
    };
    let equal =
        super::logical::emit_named_call(builder, location, equality, vec![scrutinee, literal]);
    let bool_ty = builder.ctx.bool_type(builder.ir);
    Some(builder.cast_if_needed(location, equal, bool_ty))
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
            let (struct_ty, field_tys) = logical_tuple_pattern_layout(ctx, ir, pattern.id);
            for ((index, element), element_ty) in elements.iter().enumerate().zip(field_tys) {
                let get = adt::StructGet::operands(scrutinee)
                    .r#type(struct_ty)
                    .field(index as u32)
                    .results(element_ty)
                    .build(ir, location);
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
        PatternKind::Variant { .. } | PatternKind::Record { .. } => {
            let (layout, fields) = logical_constructor_pattern(ctx, ir, pattern);
            match layout {
                ConstructorLayout::Struct {
                    ty,
                    fields: field_tys,
                } => {
                    for (index, field) in fields {
                        let get = adt::StructGet::operands(scrutinee)
                            .r#type(ty)
                            .field(index as u32)
                            .results(field_tys[index])
                            .build(ir, location);
                        ir.push_op(block, get.op_ref());
                        bind_logical_pattern_fields(
                            ctx,
                            ir,
                            block,
                            location,
                            get.result(ir),
                            field,
                        );
                    }
                }
                ConstructorLayout::Variant(VariantLayout {
                    ty,
                    tag,
                    fields: field_tys,
                }) => {
                    let cast = adt::VariantCast::operands(scrutinee)
                        .r#type(ty)
                        .tag(tag)
                        .results(ty)
                        .build(ir, location);
                    ir.push_op(block, cast.op_ref());
                    for (index, field) in fields {
                        let get = adt::VariantGet::operands(cast.result(ir))
                            .r#type(ty)
                            .tag(tag)
                            .field(index as u32)
                            .results(field_tys[index])
                            .build(ir, location);
                        ir.push_op(block, get.op_ref());
                        bind_logical_pattern_fields(
                            ctx,
                            ir,
                            block,
                            location,
                            get.result(ir),
                            field,
                        );
                    }
                }
            }
        }
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
        let head = list::Head::operands(current)
            .element_type(element_ty)
            .results(element_ty)
            .build(ir, location);
        ir.push_op(block, head.op_ref());
        bind_logical_pattern_fields(ctx, ir, block, location, head.result(ir), element);
        let tail = list::Tail::operands(current)
            .element_type(element_ty)
            .results(list_ty)
            .build(ir, location);
        ir.push_op(block, tail.op_ref());
        current = tail.result(ir);
    }
    if let Some((name, id)) = rest {
        ctx.bind(id, name, current);
    }
}
