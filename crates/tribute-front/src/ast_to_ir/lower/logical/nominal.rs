//! Nominal layouts and the logical types of signatures and expressions.
use super::*;

/// Register nominal layouts used exclusively by source-logical lowering.
///
/// Constructor schemes are the authoritative semantic field types, so records,
/// constructors, accessors, and pattern extraction share the same recursive
/// `tribute_control` callable layouts.
pub(super) fn prescan_logical_nominal_layouts<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    ir: &mut IrContext,
    entries: &[DeclEntry<TypedRef<'db>>],
    well_known_types: &mut super::super::decl::WellKnownTypePrescan,
    runtime_types: &mut Vec<(SymbolPath, TypeRef)>,
    constructors: &HashMap<crate::ast::CtorId<'db>, crate::ast::TypeScheme<'db>>,
    specialized_enum_variants: &SortedMap<crate::ast::NodeId, crate::ast::TypeScheme<'db>>,
) {
    for entry in entries {
        let prefix = &entry.module;
        match &entry.decl {
            Decl::Struct(structure) => {
                let qualified = prefix.child(structure.name.clone());
                let ctor = CtorId::new(ctx.db, qualified);
                let field_types =
                    constructor_fields(ctx, constructors, ctor, structure.fields.len());
                let fields = structure
                    .fields
                    .iter()
                    .zip(field_types)
                    .map(|(field, ty)| {
                        (
                            field.name.clone().unwrap_or_else(|| Symbol::new("_")),
                            ctx.convert_logical_type(ir, ty),
                        )
                    })
                    .collect::<Vec<_>>();
                let name = super::super::qualified_type_name(ctx.db, &ctor);
                let layout = ctx.adt_struct_type(ir, &name, &fields);
                ctx.register_type(name.clone(), layout);
                ir.register_type_alias(name, layout);
                if let Some(path) = ctx.runtime_type_path(structure.id) {
                    runtime_types.push((path.clone(), layout));
                }
            }
            Decl::Enum(enumeration) => {
                let qualified = prefix.child(enumeration.name.clone());
                let enum_ctor = CtorId::new(ctx.db, qualified.clone());
                let variants = enumeration
                    .variants
                    .iter()
                    .map(|variant| {
                        let variant_ctor =
                            CtorId::new(ctx.db, qualified.child(variant.name.clone()));
                        let fields = if enumeration.id.variant().is_some() {
                            let scheme = *specialized_enum_variants.get(&variant.id)
                                .expect("missing specialized enum variant schema");
                            let result = match scheme.body(ctx.db).kind(ctx.db) {
                                TypeKind::Func { result, .. } => *result,
                                _ => scheme.body(ctx.db),
                            };
                            assert!(matches!(result.kind(ctx.db), TypeKind::Named { id, args, .. }
                                if args.is_empty() && *id.path(ctx.db) == qualified
                                    && id.origin(ctx.db) == crate::ast::TypeOrigin::Source { disambiguator: enumeration.disambiguator }),
                                "specialized enum variant schema has wrong owner");
                            constructor_schema_fields(ctx, scheme, variant.fields.len())
                        } else {
                            constructor_fields(ctx, constructors, variant_ctor, variant.fields.len())
                        }
                        .into_iter()
                        .map(|ty| ctx.convert_logical_type(ir, ty))
                        .collect();
                        (variant.name.clone(), fields)
                    })
                    .collect::<Vec<_>>();
                let name = super::super::qualified_type_name(ctx.db, &enum_ctor);
                let location = ctx.location(enumeration.id);
                let definition = crate::ast::TypeDefId::declared(
                    ctx.db,
                    qualified.clone(),
                    enumeration.disambiguator,
                );
                let layout = if well_known_types.is_string(definition) {
                    ctx.adt_enum_type_with_definition(ir, &name, &variants, location)
                } else {
                    ctx.adt_enum_type(ir, &name, &variants)
                };
                ctx.register_type(name.clone(), layout);
                ir.register_type_alias(name, layout);
                for variant in &enumeration.variants {
                    if let Some(names) = variant
                        .fields
                        .iter()
                        .map(|field| field.name.clone())
                        .collect()
                    {
                        ctx.register_variant_field_names(layout, variant.name.clone(), names);
                    }
                }
                if well_known_types.is_string(definition) {
                    well_known_types.record_string(layout);
                }
            }
            _ => {}
        }
    }
}

/// First phase of logical nominal prescan. All identities must be known before
/// converting any field: recursive and forward references are logical nominal
/// values even while their concrete layout is still being assembled.
pub(super) fn collect_logical_nominal_identities<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    entries: &[DeclEntry<TypedRef<'db>>],
) {
    for entry in entries {
        let prefix = &entry.module;
        match &entry.decl {
            Decl::Struct(structure) => {
                let qualified = prefix.child(structure.name.clone());
                let ctor = CtorId::new(ctx.db, qualified);
                ctx.declare_logical_nominal(super::super::qualified_type_name(ctx.db, &ctor));
                ctx.register_struct_fields(
                    ctor,
                    structure
                        .fields
                        .iter()
                        .map(|field| field.name.clone().unwrap_or_else(|| Symbol::new("_")))
                        .collect(),
                );
            }
            Decl::Enum(enumeration) => {
                let qualified = prefix.child(enumeration.name.clone());
                let ctor = CtorId::new(ctx.db, qualified);
                ctx.declare_logical_nominal(super::super::qualified_type_name(ctx.db, &ctor));
            }
            _ => {}
        }
    }
}

pub(super) fn constructor_fields<'db>(
    ctx: &IrLoweringCtx<'db>,
    constructors: &HashMap<crate::ast::CtorId<'db>, crate::ast::TypeScheme<'db>>,
    constructor: CtorId<'db>,
    field_count: usize,
) -> Vec<crate::ast::Type<'db>> {
    let scheme = constructors
        .get(&constructor)
        .unwrap_or_else(|| panic!("missing typechecked constructor schema for {constructor:?}"));
    constructor_schema_fields(ctx, *scheme, field_count)
}

pub(super) fn constructor_schema_fields<'db>(
    ctx: &IrLoweringCtx<'db>,
    scheme: crate::ast::TypeScheme<'db>,
    field_count: usize,
) -> Vec<crate::ast::Type<'db>> {
    match scheme.body(ctx.db()).kind(ctx.db()) {
        TypeKind::Func { params, .. } => {
            assert_eq!(
                params.len(),
                field_count,
                "constructor field schema mismatch"
            );
            params.clone()
        }
        _ if field_count == 0 => vec![],
        _ => panic!("non-nullary constructor is missing callable field schema"),
    }
}

pub(super) fn function_signature<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    ir: &mut IrContext,
    function: &FuncDecl<TypedRef<'db>>,
) -> FuncSignature {
    let qualified = ctx.qualify_name(&function.name);
    // Nested declarations are exported under their qualified identity; a
    // short-name lookup can silently select an unrelated root declaration.
    FuncSignature::lookup_logical(ctx, ir, &qualified)
        .or_else(|| {
            (qualified != [&function.name])
                .then(|| {
                    FuncSignature::lookup_logical(ctx, ir, std::slice::from_ref(&function.name))
                })
                .flatten()
        })
        .unwrap_or_else(|| {
            panic!(
                "missing typechecked signature for function {}",
                function.name
            )
        })
}

pub(super) fn expr_type(builder: &mut IrBuilder<'_, '_>, expr: &Expr<TypedRef<'_>>) -> TypeRef {
    builder
        .ctx
        .get_node_type(expr.id)
        .copied()
        .map(|ty| builder.ctx.convert_logical_type(builder.ir, ty))
        .unwrap_or_else(|| panic!("missing typechecked expression type"))
}

pub(super) fn expr_type_for_id(builder: &mut IrBuilder<'_, '_>, id: crate::ast::NodeId) -> TypeRef {
    builder
        .ctx
        .get_node_type(id)
        .copied()
        .map(|ty| builder.ctx.convert_logical_type(builder.ir, ty))
        .unwrap_or_else(|| panic!("missing typechecked expression type"))
}
