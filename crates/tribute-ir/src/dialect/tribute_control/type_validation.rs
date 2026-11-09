//! Whole-IR validation of nominal types and managed reference boundaries.

use super::*;

pub(super) fn declaration_map<'a>(
    ctx: &IrContext,
    declarations: &'a [OperationDeclaration],
    errors: &mut Vec<ValidationError>,
) -> HashMap<(&'a SymbolPath, StringRef), &'a OperationDeclaration> {
    let mut map = HashMap::default();
    for declaration in declarations {
        let key = (&declaration.ability, declaration.op_name);
        if map.insert(key, declaration).is_some() {
            push_type_error(
                errors,
                format!(
                    "duplicate operation declaration for {}::{}",
                    declaration.ability,
                    ctx.str(declaration.op_name)
                ),
            );
        }
    }
    map
}

fn is_adt_typeref(ctx: &IrContext, ty: TypeRef) -> bool {
    let data = ctx.get_type(ty);
    data.name == crate::dialect::adt::Typeref::NAME
}

pub(super) fn contains_adt_typeref(
    ctx: &IrContext,
    ty: TypeRef,
    visiting: &mut HashSet<TypeRef>,
) -> bool {
    if !visiting.insert(ty) {
        return false;
    }
    let data = ctx.get_type(ty);
    let contains = is_adt_typeref(ctx, ty)
        || data
            .params
            .iter()
            .copied()
            .any(|param| contains_adt_typeref(ctx, param, visiting))
        || data
            .attrs
            .values()
            .any(|attribute| attribute_contains_adt_typeref(ctx, attribute, visiting));
    visiting.remove(&ty);
    contains
}

fn attribute_contains_adt_typeref(
    ctx: &IrContext,
    attribute: &Attribute,
    visiting: &mut HashSet<TypeRef>,
) -> bool {
    let mut contains = false;
    attribute.visit_types(&mut |ty| {
        contains = contains || contains_adt_typeref(ctx, ty, visiting);
    });
    contains
}

pub(super) fn nominal_identity(ctx: &IrContext, ty: TypeRef) -> Option<StringRef> {
    adt::nominal_name_ref(ctx, ty)
}

pub(super) fn canonical_nominal_layouts(
    ctx: &IrContext,
    reachable_types: &HashSet<TypeRef>,
    errors: &mut Vec<ValidationError>,
) -> HashMap<StringRef, TypeRef> {
    let mut layouts = HashMap::default();
    let mut referenced_names = HashSet::default();
    let mut sorted_reachable_types = reachable_types.iter().copied().collect::<Vec<_>>();
    sorted_reachable_types.sort_unstable();
    for ty in sorted_reachable_types {
        if is_adt_typeref(ctx, ty) {
            if let Some(name) = ctx.type_attrs(ty).get_string_ref("name") {
                referenced_names.insert(name);
            }
            continue;
        }
        if !adt::Struct::matches(ctx, ty) && !adt::Enum::matches(ctx, ty) {
            continue;
        }
        let Some(name) = ctx.type_attrs(ty).get_string_ref("name") else {
            continue;
        };
        if let Some(previous) = layouts.get(&name).copied() {
            if previous != ty {
                push_type_error(
                    errors,
                    format!(
                        "nominal layout {:?} is declared more than once ({previous} and {ty})",
                        ctx.str(name)
                    ),
                );
            }
        } else {
            layouts.insert(name, ty);
        }
    }
    for name in referenced_names {
        if layouts.contains_key(&name) {
            continue;
        }
        let candidates = ctx
            .types()
            .iter()
            .filter_map(|(ty, data)| {
                ((adt::Struct::matches(ctx, ty) || adt::Enum::matches(ctx, ty))
                    && data.attrs.view(ctx).get_string_ref("name") == Some(name))
                .then_some(ty)
            })
            .collect::<Vec<_>>();
        if let [layout] = candidates.as_slice() {
            layouts.insert(name, *layout);
        } else if let [first, second, ..] = candidates.as_slice() {
            push_type_error(
                errors,
                format!(
                    "nominal layout {:?} is declared more than once ({first} and {second})",
                    ctx.str(name)
                ),
            );
        }
    }
    layouts
}

fn is_core_ptr(ctx: &IrContext, ty: TypeRef) -> bool {
    let data = ctx.get_type(ty);
    data.name.is(ctx, "core.ptr")
}

fn collect_attribute_types(ctx: &IrContext, attribute: &Attribute, types: &mut HashSet<TypeRef>) {
    attribute.visit_types(&mut |ty| collect_reachable_type(ctx, ty, types));
}

fn collect_reachable_type(ctx: &IrContext, ty: TypeRef, types: &mut HashSet<TypeRef>) {
    if !types.insert(ty) {
        return;
    }
    let mut children = Vec::new();
    trunk_ir::walk::visit_type_children(ctx, ty, &mut |child| children.push(child));
    for child in children {
        collect_reachable_type(ctx, child, types);
    }
}

/// Collect semantic types reachable from a module operation and its nested IR.
///
/// Context-wide type aliases are printer/parser conveniences, not IR roots:
/// an alias whose type is never used by an operation, operand, result, block
/// argument, or type-bearing attribute must not affect validation of this
/// module. A reachable `adt.typeref` may resolve one unique layout alias; an
/// ambiguous nominal name remains invalid.
/// Whether class specialization has run on `module`: `Some(false)` when it
/// holds a `tribute_control` form of a `tribute_cps` operation and no
/// `tribute_cps` operation, `None` when it holds both, and `Some(true)`
/// otherwise, which includes a module without callables.
pub fn specialized(ctx: &IrContext, module: Module) -> Option<bool> {
    let mut control = false;
    let mut cps = false;
    let _ = walk_op::<()>(ctx, module.op(), &mut |op| {
        let name = ctx.op(op).name;
        if name.dialect(ctx) == tribute_cps::DIALECT_NAME {
            cps = true;
        } else if Func::matches(ctx, op)
            || Lambda::matches(ctx, op)
            || FuncRef::matches(ctx, op)
            || CallIndirect::matches(ctx, op)
            || TailCallIndirect::matches(ctx, op)
        {
            control = true;
        }
        ControlFlow::Continue(WalkAction::Advance)
    });
    match (control, cps) {
        (true, true) => None,
        (control, _) => Some(!control),
    }
}

pub(super) fn collect_reachable_ir_types(ctx: &IrContext, module: OpRef) -> HashSet<TypeRef> {
    fn collect_op_types(ctx: &IrContext, op: OpRef, types: &mut HashSet<TypeRef>) {
        for value in ctx.op_operands(op).iter().chain(ctx.op_results(op).iter()) {
            collect_reachable_type(ctx, ctx.value_ty(*value), types);
        }
        for attribute in ctx.op(op).attributes.values() {
            collect_attribute_types(ctx, attribute, types);
        }
        for region in ctx.op_regions(op) {
            for block in ctx.region(region).blocks.iter().copied() {
                for argument in ctx.block_args(block) {
                    collect_reachable_type(ctx, ctx.value_ty(*argument), types);
                }
            }
        }
    }

    let mut types = HashSet::default();
    collect_op_types(ctx, module, &mut types);
    for region in ctx.op_regions(module) {
        walk_region_ops(ctx, region, &mut |op| collect_op_types(ctx, op, &mut types));
    }
    types
}

fn raw_pointer_cast_origin(
    ctx: &IrContext,
    value: ValueRef,
    visiting: &mut HashSet<ValueRef>,
) -> bool {
    if !visiting.insert(value) {
        return false;
    }
    if is_core_ptr(ctx, ctx.value_ty(value)) {
        return true;
    }
    let ValueDef::OpResult(producer, _) = ctx.value_def(value) else {
        return false;
    };
    let transparent_cast = core::UnrealizedConversionCast::from_op(ctx, producer).is_ok()
        || adt::RefCast::from_op(ctx, producer).is_ok();
    transparent_cast
        && ctx
            .op_operands(producer)
            .first()
            .copied()
            .is_some_and(|operand| raw_pointer_cast_origin(ctx, operand, visiting))
}

pub(super) fn validate_managed_reference_boundaries(
    ctx: &IrContext,
    body: RegionRef,
    reachable_types: &HashSet<TypeRef>,
    nominal_layouts: &HashMap<StringRef, TypeRef>,
    errors: &mut Vec<ValidationError>,
) {
    for ty in reachable_types.iter().copied() {
        if !is_adt_typeref(ctx, ty) {
            continue;
        }
        let data = ctx.get_type(ty);
        let Some(name) = ctx.type_attrs(ty).get_string_ref("name") else {
            push_type_error(
                errors,
                format!("{ty}: adt.typeref requires nominal name metadata"),
            );
            continue;
        };
        if !data.params.is_empty() {
            push_type_error(
                errors,
                format!("{ty}: adt.typeref cannot have type parameters"),
            );
        }
        let Some(&layout) = nominal_layouts.get(&name) else {
            push_type_error(
                errors,
                format!(
                    "{ty}: adt.typeref {:?} has no verified nominal declaration",
                    ctx.str(name)
                ),
            );
            continue;
        };
        let binders = adt::Struct::from_type_ref(ctx, layout)
            .and_then(|layout| layout.row_params(ctx))
            .or_else(|| adt::Enum::from_type_ref(ctx, layout).and_then(|l| l.row_params(ctx)))
            .unwrap_or(0);
        let args =
            adt::Typeref::from_type_ref(ctx, ty).map_or(0, |typeref| typeref.rows(ctx).len());
        if usize::try_from(binders).ok() != Some(args) {
            push_type_error(
                errors,
                format!(
                    "{ty}: adt.typeref {:?} passes {args} row arguments to a layout with {binders} row binders",
                    ctx.str(name)
                ),
            );
        }
    }

    walk_region_ops(ctx, body, &mut |op| {
        let data = ctx.op(op);
        if data.name.is(ctx, "adt.ref_null") {
            let ([], [result], Some(declared)) = (
                ctx.op_operands(op),
                ctx.op_result_types(op),
                ctx.op_attrs(op).get_type("type"),
            ) else {
                push_op_error(ctx, op, errors, "adt.ref_null requires one typed result");
                return;
            };
            if *result != declared || !is_adt_typeref(ctx, *result) {
                push_op_error(
                    ctx,
                    op,
                    errors,
                    "adt.ref_null result and type attribute must be the same managed adt.typeref",
                );
            }
        }
        if data.name.is(ctx, "adt.ref_cast") {
            let ([operand], [result], Some(declared)) = (
                ctx.op_operands(op),
                ctx.op_result_types(op),
                ctx.op_attrs(op).get_type("type"),
            ) else {
                push_op_error(
                    ctx,
                    op,
                    errors,
                    "adt.ref_cast requires one typed operand and result",
                );
                return;
            };
            let operand_type = ctx.value_ty(*operand);
            let source = nominal_identity(ctx, operand_type);
            let target = nominal_identity(ctx, *result);
            if declared != *result
                || !is_adt_typeref(ctx, operand_type)
                || !is_adt_typeref(ctx, *result)
                || source.is_none()
                || source != target
            {
                push_op_error(
                    ctx,
                    op,
                    errors,
                    "adt.ref_cast requires compatible managed nominal reference types",
                );
            }
        }
        for result in ctx.op_results(op) {
            if is_adt_typeref(ctx, ctx.value_ty(*result))
                && raw_pointer_cast_origin(ctx, *result, &mut HashSet::default())
            {
                push_op_error(
                    ctx,
                    op,
                    errors,
                    "core.ptr cast chain cannot produce a managed adt.typeref",
                );
            }
        }
    });
}
