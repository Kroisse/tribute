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

    // Apply patterns to resolve type references
    let applicator = PatternApplicator::new(TypeConverter::new())
        .add_pattern(ResolveFuncTypePattern {
            type_defs: type_defs.clone(),
        })
        .add_pattern(ResolveOperationTypesPattern { type_defs });

    let target = ConversionTarget::new();
    applicator.apply_partial(db, module, target).module
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
    use trunk_ir::DialectType;
    use trunk_ir::dialect::core;

    #[test]
    fn test_resolve_type_preserves_non_tribute_types() {
        let db = salsa::DatabaseImpl::new();
        let type_defs = HashMap::new();

        let i32_ty = core::I32::new(&db).as_type();
        let resolved = resolve_type(&db, i32_ty, &type_defs);
        assert_eq!(resolved, i32_ty);
    }
}
