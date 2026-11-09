//! Nominal layouts and the logical types of signatures and expressions.

use rustc_hash::FxHashMap as HashMap;
use tribute_ir::dialect::adt;
use trunk_ir::context::IrContext;
use trunk_ir::refs::{OpRef, TypeRef};
use trunk_ir::{Symbol, SymbolPath};

use crate::SortedMap;
use crate::ast::{CtorId, Decl, DeclEntry, Expr, FuncDecl, TypeKind, TypedRef};
use crate::ast_to_ir::context::IrLoweringCtx;
use crate::ast_to_ir::lower::{FuncSignature, IrBuilder};

/// Register nominal layouts used exclusively by source-logical lowering.
///
/// Constructor schemes are the authoritative semantic field types, so records,
/// constructors, accessors, and pattern extraction share the same recursive
/// `tribute_control` callable layouts. Returns those schemes in declaration
/// order.
pub(super) fn prescan_logical_nominal_layouts<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    ir: &mut IrContext,
    entries: &[DeclEntry<TypedRef<'db>>],
    well_known_types: &mut super::super::decl::WellKnownTypePrescan,
    runtime_types: &mut Vec<(SymbolPath, TypeRef)>,
    constructors: &HashMap<crate::ast::CtorId<'db>, crate::ast::TypeScheme<'db>>,
    specialized_enum_variants: &SortedMap<crate::ast::NodeId, crate::ast::TypeScheme<'db>>,
) -> Vec<crate::ast::TypeScheme<'db>> {
    let mut layout_schemes = Vec::new();
    for entry in entries {
        let prefix = &entry.module;
        match &entry.decl {
            Decl::Struct(structure) => {
                let qualified = prefix.child(structure.name.clone());
                let ctor = CtorId::new(ctx.db, qualified);
                let scheme = constructor_scheme(constructors, ctor);
                layout_schemes.push(scheme);
                let row_params = layout_row_params(ctx, [scheme], structure.row_params.len());
                let outer = ctx.enter_layout(scheme.effect_params(ctx.db));
                let field_types = constructor_schema_fields(ctx, scheme, structure.fields.len());
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
                ctx.leave_definition(outer);
                let name = super::super::qualified_type_name(ctx.db, &ctor);
                let field_types: Vec<_> = fields.iter().map(|&(_, ty)| ty).collect();
                let layout = ctx.adt_struct_type(ir, &name, &field_types, row_params);
                let field_names: Vec<_> = fields
                    .iter()
                    .map(|(field, _)| ir.intern_symbol_text(field))
                    .collect();
                let decl = adt::StructDecl::operands()
                    .r#type(layout)
                    .fields(field_names)
                    .build(ir, ctx.location(structure.id));
                declare(ctx, ir, decl.op_ref());
                ctx.register_type(name.clone(), layout);
                ir.register_type_alias(name, layout);
                if let Some(path) = ctx.runtime_type_path(structure.id) {
                    runtime_types.push((path.clone(), layout));
                }
            }
            Decl::Enum(enumeration) => {
                let qualified = prefix.child(enumeration.name.clone());
                let enum_ctor = CtorId::new(ctx.db, qualified.clone());
                let schemes: Vec<_> = enumeration
                    .variants
                    .iter()
                    .map(|variant| {
                        if enumeration.id.variant().is_some() {
                            let scheme = *specialized_enum_variants.get(&variant.id)
                                .expect("missing specialized enum variant schema");
                            let result = match scheme.body(ctx.db).kind(ctx.db) {
                                TypeKind::Func { result, .. } => *result,
                                _ => *scheme.body(ctx.db),
                            };
                            assert!(matches!(result.kind(ctx.db), TypeKind::Named { id, args, .. }
                                if args.is_empty() && *id.path(ctx.db) == qualified
                                    && *id.origin(ctx.db) == crate::ast::TypeOrigin::Source { disambiguator: enumeration.disambiguator }),
                                "specialized enum variant schema has wrong owner");
                            scheme
                        } else {
                            let variant_ctor =
                                CtorId::new(ctx.db, qualified.child(variant.name.clone()));
                            constructor_scheme(constructors, variant_ctor)
                        }
                    })
                    .collect();
                let row_params =
                    layout_row_params(ctx, schemes.iter().copied(), enumeration.row_params.len());
                layout_schemes.extend(schemes.iter().copied());
                let binders = schemes
                    .first()
                    .map(|scheme| scheme.effect_params(ctx.db).clone())
                    .unwrap_or_default();
                let outer = ctx.enter_layout(&binders);
                let variants = enumeration
                    .variants
                    .iter()
                    .zip(&schemes)
                    .map(|(variant, &scheme)| {
                        let fields: Vec<TypeRef> =
                            constructor_schema_fields(ctx, scheme, variant.fields.len())
                                .into_iter()
                                .map(|ty| ctx.convert_logical_type(ir, ty))
                                .collect();
                        (variant.name.clone(), fields)
                    })
                    .collect::<Vec<_>>();
                ctx.leave_definition(outer);
                let name = super::super::qualified_type_name(ctx.db, &enum_ctor);
                let location = ctx.location(enumeration.id);
                let definition = crate::ast::TypeDefId::declared(
                    ctx.db,
                    qualified.clone(),
                    enumeration.disambiguator,
                );
                let variant_fields: Vec<_> =
                    variants.iter().map(|(_, fields)| fields.clone()).collect();
                let layout = if well_known_types.is_string(definition) {
                    ctx.adt_enum_type_with_definition(
                        ir,
                        &name,
                        &variant_fields,
                        row_params,
                        location,
                    )
                } else {
                    ctx.adt_enum_type(ir, &name, &variant_fields, row_params)
                };
                for (position, (variant, _)) in variants.iter().enumerate() {
                    let position =
                        u32::try_from(position).expect("an enum has fewer than 2^32 variants");
                    ctx.register_variant_position(layout, variant.clone(), position);
                }
                let variant_names: Vec<_> = variants
                    .iter()
                    .map(|(variant, _)| ir.intern_symbol_text(variant))
                    .collect();
                let decl = adt::EnumDecl::operands()
                    .r#type(layout)
                    .variants(variant_names)
                    .build(ir, location);
                declare(ctx, ir, decl.op_ref());
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
    layout_schemes
}

/// Add the nominal declaration `decl` to the module body.
fn declare(ctx: &IrLoweringCtx<'_>, ir: &mut IrContext, decl: OpRef) {
    let block = ctx
        .module_block()
        .expect("the module block exists before the nominal prescan");
    ir.push_op(block, decl);
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

fn constructor_scheme<'db>(
    constructors: &HashMap<crate::ast::CtorId<'db>, crate::ast::TypeScheme<'db>>,
    constructor: CtorId<'db>,
) -> crate::ast::TypeScheme<'db> {
    *constructors
        .get(&constructor)
        .unwrap_or_else(|| panic!("missing typechecked constructor schema for {constructor:?}"))
}

/// The number of row binders of a layout whose declaration has `declared`
/// row parameters. Every constructor scheme quantifies them in the same
/// order, which the layout's binders follow.
fn layout_row_params<'db>(
    ctx: &IrLoweringCtx<'db>,
    schemes: impl IntoIterator<Item = crate::ast::TypeScheme<'db>>,
    declared: usize,
) -> u32 {
    let mut schemes = schemes.into_iter();
    if let Some(first) = schemes.next() {
        let binders = first.effect_params(ctx.db);
        assert_eq!(
            binders.len(),
            declared,
            "a constructor scheme quantifies its declaration's row parameters"
        );
        assert!(
            schemes.all(|scheme| scheme.effect_params(ctx.db) == binders),
            "the constructors of one layout quantify the same row parameters"
        );
    }
    u32::try_from(declared).expect("a declaration has fewer than 2^32 row parameters")
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
