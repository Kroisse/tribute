//! Resolve unresolved type references to concrete ADT types.
//!
//! This pass converts `tribute.type(name=X)` references to their actual ADT types
//! (`adt.enum` or `adt.struct`) by looking up type definitions in the module.
//!
//! ## Why this pass is needed
//!
//! During lowering, some type references remain as `tribute.type(name=X)` even after
//! the resolve pass. This can happen when:
//! - Types are referenced before they are defined
//! - Nested type references in struct fields or enum variants
//! - Cross-module type references
//!
//! This pass ensures all type references are resolved before emit, so emit doesn't
//! need to handle tribute-specific types.
//!
//! ## Type Resolution
//!
//! | Source Type                  | Target Type                    |
//! |------------------------------|--------------------------------|
//! | `tribute.type(name="Foo")`   | `adt.struct(name="Foo", ...)`  |
//! | `tribute.type(name="Bar")`   | `adt.enum(name="Bar", ...)`    |

use std::collections::HashMap;

use tracing::debug;
use tribute_ir::dialect::tribute;
use trunk_ir::dialect::core::Module;
use trunk_ir::dialect::func;
use trunk_ir::rewrite::{
    ConversionTarget, OpAdaptor, PatternApplicator, RewritePattern, RewriteResult, TypeConverter,
};
use trunk_ir::{Attribute, Block, BlockArg, DialectOp, IdVec, Operation, Region, Symbol, Type};

/// Resolve all `tribute.type` references to their actual ADT types.
///
/// This pass should run after resolve and before emit.
pub fn lower<'db>(db: &'db dyn salsa::Database, module: Module<'db>) -> Module<'db> {
    // First, collect all ADT type definitions from the module
    let type_defs = collect_type_definitions(db, &module);

    if type_defs.is_empty() {
        return module;
    }

    debug!(
        "resolve_type_references: collected {} type definitions",
        type_defs.len()
    );

    // Resolve types inside type_defs themselves (for recursive types)
    // This ensures that when we replace `tribute.type { name: "Expr" }` with the
    // adt.enum type, the variants inside that enum also have resolved types.
    let resolved_type_defs = resolve_type_defs(db, &type_defs);

    // Apply patterns to resolve type references
    let applicator = PatternApplicator::new(TypeConverter::new())
        .add_pattern(ResolveFuncTypePattern {
            type_defs: resolved_type_defs.clone(),
        })
        .add_pattern(ResolveEnumDefTypesPattern {
            type_defs: resolved_type_defs.clone(),
        })
        .add_pattern(ResolveOperationTypesPattern {
            type_defs: resolved_type_defs,
        });

    let target = ConversionTarget::new();
    applicator.apply_partial(db, module, target).module
}

/// Resolve types inside type_defs themselves.
///
/// For recursive types like `enum Expr { Add(Expr, Expr) }`, the `type_defs`
/// initially contains the enum type with `tribute.type { name: "Expr" }` in its
/// variants. This function resolves those inner type references.
fn resolve_type_defs<'db>(
    db: &'db dyn salsa::Database,
    type_defs: &TypeDefs<'db>,
) -> TypeDefs<'db> {
    let mut resolved = TypeDefs::new();

    for (&name, &ty) in type_defs.iter() {
        // For enum types, resolve the types inside the variants attribute
        if trunk_ir::dialect::adt::is_enum_type(db, ty) {
            let resolved_ty = resolve_enum_type_shallow(db, ty, type_defs);
            resolved.insert(name, resolved_ty);
        } else {
            resolved.insert(name, ty);
        }
    }

    resolved
}

/// Resolve types inside an enum type's variants, but only one level deep.
///
/// This resolves `tribute.type` references to their ADT types, but doesn't
/// recurse into the resolved ADT types (to avoid infinite recursion with
/// recursive types).
fn resolve_enum_type_shallow<'db>(
    db: &'db dyn salsa::Database,
    ty: Type<'db>,
    type_defs: &TypeDefs<'db>,
) -> Type<'db> {
    // Get the variants attribute
    let Some(Attribute::List(variants)) = ty.get_attr(db, Symbol::new("variants")) else {
        return ty;
    };

    let mut changed = false;
    let new_variants: Vec<Attribute> = variants
        .iter()
        .map(|variant_attr| {
            let Attribute::List(pair) = variant_attr else {
                return variant_attr.clone();
            };
            if pair.len() < 2 {
                return variant_attr.clone();
            }

            // pair[0] is the variant name (Symbol), pair[1] is the field types (List)
            let Attribute::List(field_attrs) = &pair[1] else {
                return variant_attr.clone();
            };

            // Resolve each field type (shallow - only tribute.type → type_defs/primitive lookup)
            let new_field_attrs: Vec<Attribute> = field_attrs
                .iter()
                .map(|attr| {
                    if let Attribute::Type(field_ty) = attr {
                        // Only resolve tribute.type references, not recursively
                        if tribute::is_unresolved_type(db, *field_ty)
                            && let Some(Attribute::Symbol(name_sym)) =
                                field_ty.get_attr(db, Symbol::new("name"))
                        {
                            // Check user-defined types first
                            if let Some(&adt_ty) = type_defs.get(name_sym) {
                                changed = true;
                                return Attribute::Type(adt_ty);
                            }
                            // Check primitive types
                            if let Some(primitive_ty) = resolve_primitive_type(db, name_sym) {
                                changed = true;
                                return Attribute::Type(primitive_ty);
                            }
                        }
                        attr.clone()
                    } else {
                        attr.clone()
                    }
                })
                .collect();

            if new_field_attrs
                .iter()
                .zip(field_attrs.iter())
                .any(|(a, b)| a != b)
            {
                Attribute::List(vec![pair[0].clone(), Attribute::List(new_field_attrs)])
            } else {
                variant_attr.clone()
            }
        })
        .collect();

    if !changed {
        return ty;
    }

    // Rebuild the type with resolved variants
    let mut new_attrs = ty.attrs(db).clone();
    new_attrs.insert(Symbol::new("variants"), Attribute::List(new_variants));

    let new_params: IdVec<Type<'db>> = ty.params(db).iter().copied().collect();
    Type::new(db, ty.dialect(db), ty.name(db), new_params, new_attrs)
}

/// Resolve primitive type names to their runtime types.
fn resolve_primitive_type<'db>(db: &'db dyn salsa::Database, name: &Symbol) -> Option<Type<'db>> {
    use tribute_ir::dialect::tribute_rt;
    use trunk_ir::dialect::core;

    let name_str = name.to_string();
    match &*name_str {
        "Int" => Some(tribute_rt::int_type(db)),
        "Bool" => Some(tribute_rt::bool_type(db)),
        "Float" => Some(tribute_rt::float_type(db)),
        "Nat" => Some(tribute_rt::nat_type(db)),
        "Bytes" => Some(*core::Bytes::new(db)),
        "Nil" => Some(*core::Nil::new(db)),
        _ => None,
    }
}

/// Collected type definitions: name -> ADT type
type TypeDefs<'db> = HashMap<Symbol, Type<'db>>;

/// Collect all ADT type definitions from the module.
fn collect_type_definitions<'db>(
    db: &'db dyn salsa::Database,
    module: &Module<'db>,
) -> TypeDefs<'db> {
    let mut type_defs = HashMap::new();

    fn collect_from_region<'db>(
        db: &'db dyn salsa::Database,
        region: &Region<'db>,
        type_defs: &mut TypeDefs<'db>,
    ) {
        for block in region.blocks(db).iter() {
            for op in block.operations(db).iter() {
                // Collect struct definitions
                if let Ok(struct_def) = tribute::StructDef::from_operation(db, *op) {
                    let name = struct_def.sym_name(db);
                    // The struct type is in the operation's result types
                    if let Some(&struct_ty) = op.results(db).first() {
                        debug!("resolve_type_references: found struct {}", name);
                        type_defs.insert(name, struct_ty);
                    }
                }

                // Collect enum definitions
                if let Ok(enum_def) = tribute::EnumDef::from_operation(db, *op) {
                    let name = enum_def.sym_name(db);
                    // The enum type is in the operation's result types
                    if let Some(&enum_ty) = op.results(db).first() {
                        debug!("resolve_type_references: found enum {}", name);
                        type_defs.insert(name, enum_ty);
                    }
                }

                // Recursively process nested regions
                for nested in op.regions(db).iter() {
                    collect_from_region(db, nested, type_defs);
                }
            }
        }
    }

    collect_from_region(db, &module.body(db), &mut type_defs);
    type_defs
}

/// Resolve a type, replacing `tribute.type(name=X)` with the actual ADT type.
fn resolve_type<'db>(
    db: &'db dyn salsa::Database,
    ty: Type<'db>,
    type_defs: &TypeDefs<'db>,
) -> Type<'db> {
    // Check if this is a tribute.type reference
    if tribute::is_unresolved_type(db, ty)
        && let Some(Attribute::Symbol(name_sym)) = ty.get_attr(db, Symbol::new("name"))
        && let Some(adt_type) = type_defs.get(name_sym)
    {
        debug!(
            "resolve_type_references: resolving tribute.type(name={}) to adt type",
            name_sym
        );
        return *adt_type;
    }

    // Check if this is an adt.enum type with unresolved variants
    // Use shallow resolution to avoid infinite recursion with recursive types
    if trunk_ir::dialect::adt::is_enum_type(db, ty) {
        let resolved = resolve_enum_type_shallow(db, ty, type_defs);
        if resolved != ty {
            return resolved;
        }
    }

    // Recursively resolve type parameters
    let params = ty.params(db);
    if params.is_empty() {
        return ty;
    }

    let new_params: IdVec<Type<'db>> = params
        .iter()
        .map(|&t| resolve_type(db, t, type_defs))
        .collect();

    if new_params.iter().zip(params.iter()).all(|(a, b)| a == b) {
        return ty;
    }

    Type::new(
        db,
        ty.dialect(db),
        ty.name(db),
        new_params,
        ty.attrs(db).clone(),
    )
}

/// Pattern to resolve types in func.func signatures.
struct ResolveFuncTypePattern<'db> {
    type_defs: TypeDefs<'db>,
}

impl<'db> RewritePattern<'db> for ResolveFuncTypePattern<'db> {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(func_op) = func::Func::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let func_type = func_op.r#type(db);
        let resolved_type = resolve_type(db, func_type, &self.type_defs);

        if resolved_type == func_type {
            return RewriteResult::Unchanged;
        }

        debug!(
            "resolve_type_references: resolved func.func {} signature types",
            func_op.sym_name(db)
        );

        let new_op = op
            .modify(db)
            .attr("type", Attribute::Type(resolved_type))
            .build();
        RewriteResult::Replace(new_op)
    }
}

/// Pattern to resolve types in enum definitions.
///
/// Enum definitions store field types in the `variants` attribute. This pattern
/// walks through each variant's field types and resolves any `tribute.type` references
/// to their actual ADT types.
struct ResolveEnumDefTypesPattern<'db> {
    type_defs: TypeDefs<'db>,
}

impl<'db> RewritePattern<'db> for ResolveEnumDefTypesPattern<'db> {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(enum_def) = tribute::EnumDef::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        // Get the enum's result type (which contains the variants attribute)
        let Some(&enum_ty) = op.results(db).first() else {
            return RewriteResult::Unchanged;
        };

        // Resolve types in the enum type itself (including variants attribute)
        let resolved_enum_ty = resolve_enum_type(db, enum_ty, &self.type_defs);

        if resolved_enum_ty == enum_ty {
            return RewriteResult::Unchanged;
        }

        debug!(
            "resolve_type_references: resolved enum_def {} field types",
            enum_def.sym_name(db)
        );

        // Rebuild the operation with the resolved enum type
        let new_op = op
            .modify(db)
            .results(IdVec::from(vec![resolved_enum_ty]))
            .build();
        RewriteResult::Replace(new_op)
    }
}

/// Resolve types inside an enum type, including variant field types.
fn resolve_enum_type<'db>(
    db: &'db dyn salsa::Database,
    ty: Type<'db>,
    type_defs: &TypeDefs<'db>,
) -> Type<'db> {
    // Get the variants attribute
    let Some(Attribute::List(variants)) = ty.get_attr(db, Symbol::new("variants")) else {
        return ty;
    };

    let mut changed = false;
    let new_variants: Vec<Attribute> = variants
        .iter()
        .map(|variant_attr| {
            let Attribute::List(pair) = variant_attr else {
                return variant_attr.clone();
            };
            if pair.len() < 2 {
                return variant_attr.clone();
            }

            // pair[0] is the variant name (Symbol), pair[1] is the field types (List)
            let Attribute::List(field_attrs) = &pair[1] else {
                return variant_attr.clone();
            };

            // Resolve each field type
            let new_field_attrs: Vec<Attribute> = field_attrs
                .iter()
                .map(|attr| {
                    if let Attribute::Type(field_ty) = attr {
                        let resolved = resolve_type(db, *field_ty, type_defs);
                        if resolved != *field_ty {
                            changed = true;
                            Attribute::Type(resolved)
                        } else {
                            attr.clone()
                        }
                    } else {
                        attr.clone()
                    }
                })
                .collect();

            if new_field_attrs
                .iter()
                .zip(field_attrs.iter())
                .any(|(a, b)| a != b)
            {
                Attribute::List(vec![pair[0].clone(), Attribute::List(new_field_attrs)])
            } else {
                variant_attr.clone()
            }
        })
        .collect();

    if !changed {
        return ty;
    }

    // Rebuild the type with resolved variants
    let mut new_attrs = ty.attrs(db).clone();
    new_attrs.insert(Symbol::new("variants"), Attribute::List(new_variants));

    // Collect params into IdVec
    let new_params: IdVec<Type<'db>> = ty.params(db).iter().copied().collect();
    Type::new(db, ty.dialect(db), ty.name(db), new_params, new_attrs)
}

/// Pattern to resolve types in operation results and regions.
struct ResolveOperationTypesPattern<'db> {
    type_defs: TypeDefs<'db>,
}

impl<'db> RewritePattern<'db> for ResolveOperationTypesPattern<'db> {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let results = op.results(db);
        let mut results_changed = false;

        // Resolve result types
        let new_results: IdVec<Type<'db>> = results
            .iter()
            .map(|&ty| {
                let resolved = resolve_type(db, ty, &self.type_defs);
                if resolved != ty {
                    results_changed = true;
                }
                resolved
            })
            .collect();

        // Also resolve types in nested regions (block arguments)
        let regions = op.regions(db);
        let mut regions_changed = false;
        let new_regions: IdVec<Region<'db>> = regions
            .iter()
            .map(|region| {
                let blocks = region.blocks(db);
                let location = region.location(db);
                let mut any_block_changed = false;

                let new_blocks: IdVec<Block<'db>> = blocks
                    .iter()
                    .map(|block| {
                        let args = block.args(db);
                        let mut args_changed = false;

                        let new_args: IdVec<BlockArg<'db>> = args
                            .iter()
                            .map(|arg| {
                                let resolved = resolve_type(db, arg.ty(db), &self.type_defs);
                                if resolved != arg.ty(db) {
                                    args_changed = true;
                                    BlockArg::new(db, resolved, arg.attrs(db).clone())
                                } else {
                                    *arg
                                }
                            })
                            .collect();

                        if args_changed {
                            any_block_changed = true;
                            Block::new(
                                db,
                                block.id(db),
                                block.location(db),
                                new_args,
                                block.operations(db).clone(),
                            )
                        } else {
                            *block
                        }
                    })
                    .collect();

                if any_block_changed {
                    regions_changed = true;
                    Region::new(db, location, new_blocks)
                } else {
                    *region
                }
            })
            .collect();

        if !results_changed && !regions_changed {
            return RewriteResult::Unchanged;
        }

        let mut builder = op.modify(db);
        if results_changed {
            builder = builder.results(new_results);
        }
        if regions_changed {
            builder = builder.regions(new_regions);
        }

        RewriteResult::Replace(builder.build())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tribute_ir::dialect::tribute;
    use trunk_ir::DialectType;
    use trunk_ir::dialect::{adt, core};

    #[test]
    fn test_resolve_type_preserves_non_tribute_types() {
        let db = salsa::DatabaseImpl::new();
        let type_defs = HashMap::new();

        let i32_ty = core::I32::new(&db).as_type();
        let resolved = resolve_type(&db, i32_ty, &type_defs);
        assert_eq!(resolved, i32_ty);
    }

    #[test]
    fn test_resolve_enum_variant_field_types() {
        let db = salsa::DatabaseImpl::new();

        // Create a tribute.type reference for "Bytes"
        let bytes_typeref =
            tribute::unresolved_type(&db, Symbol::new("Bytes"), trunk_ir::IdVec::new());

        // Create an enum with a variant that has a tribute.type field
        // enum String { Leaf(Bytes) }
        let string_enum_ty = adt::enum_type(
            &db,
            Symbol::new("String"),
            vec![(Symbol::new("Leaf"), vec![bytes_typeref])],
        );

        // Build type_defs with Bytes -> core.bytes
        let mut type_defs = HashMap::new();
        let bytes_ty = *core::Bytes::new(&db);
        type_defs.insert(Symbol::new("Bytes"), bytes_ty);

        // Resolve the enum type
        let resolved = resolve_enum_type(&db, string_enum_ty, &type_defs);

        // Check that the Leaf variant's field type is now core.bytes
        let Some(Attribute::List(variants)) = resolved.get_attr(&db, Symbol::new("variants"))
        else {
            panic!("Expected variants attribute");
        };

        let Attribute::List(leaf_pair) = &variants[0] else {
            panic!("Expected variant pair");
        };

        let Attribute::List(leaf_fields) = &leaf_pair[1] else {
            panic!("Expected field list");
        };

        let Attribute::Type(field_ty) = &leaf_fields[0] else {
            panic!("Expected field type");
        };

        assert_eq!(
            *field_ty, bytes_ty,
            "Enum variant field type should be resolved from tribute.type to core.bytes"
        );
    }

    #[test]
    fn test_resolve_enum_variant_with_primitive_types() {
        let db = salsa::DatabaseImpl::new();

        // Create tribute.type references for primitive types
        let int_typeref = tribute::unresolved_type(&db, Symbol::new("Int"), trunk_ir::IdVec::new());
        let bytes_typeref =
            tribute::unresolved_type(&db, Symbol::new("Bytes"), trunk_ir::IdVec::new());

        // Create an enum: enum String { Leaf(Bytes), Branch(String, String, Int, Int) }
        let string_typeref =
            tribute::unresolved_type(&db, Symbol::new("String"), trunk_ir::IdVec::new());
        let string_enum_ty = adt::enum_type(
            &db,
            Symbol::new("String"),
            vec![
                (Symbol::new("Leaf"), vec![bytes_typeref]),
                (
                    Symbol::new("Branch"),
                    vec![string_typeref, string_typeref, int_typeref, int_typeref],
                ),
            ],
        );

        // Build type_defs - note: String maps to itself (recursive type)
        let mut type_defs = HashMap::new();
        type_defs.insert(Symbol::new("String"), string_enum_ty);

        // Resolve the enum type (shallow to avoid infinite recursion)
        let resolved = resolve_enum_type_shallow(&db, string_enum_ty, &type_defs);

        // Check Leaf variant
        let Some(Attribute::List(variants)) = resolved.get_attr(&db, Symbol::new("variants"))
        else {
            panic!("Expected variants attribute");
        };

        // Check Leaf(Bytes) - Bytes should be resolved to core.bytes
        let Attribute::List(leaf_pair) = &variants[0] else {
            panic!("Expected variant pair");
        };
        let Attribute::List(leaf_fields) = &leaf_pair[1] else {
            panic!("Expected field list");
        };
        let Attribute::Type(leaf_field_ty) = &leaf_fields[0] else {
            panic!("Expected field type");
        };
        let bytes_ty = *core::Bytes::new(&db);
        assert_eq!(
            *leaf_field_ty, bytes_ty,
            "Leaf variant field should be resolved to core.bytes"
        );

        // Check Branch variant - String references should remain (shallow resolution)
        let Attribute::List(branch_pair) = &variants[1] else {
            panic!("Expected variant pair");
        };
        let Attribute::List(branch_fields) = &branch_pair[1] else {
            panic!("Expected field list");
        };

        // Int fields should be resolved to tribute_rt.int
        let Attribute::Type(int_field_ty) = &branch_fields[2] else {
            panic!("Expected field type");
        };
        let int_ty = tribute_ir::dialect::tribute_rt::int_type(&db);
        assert_eq!(
            *int_field_ty, int_ty,
            "Int fields should be resolved to tribute_rt.int"
        );
    }
}
