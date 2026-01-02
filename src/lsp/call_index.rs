//! Call index for signature help functionality.
//!
//! Provides utilities for looking up function definitions by name.

use trunk_ir::dialect::core::Module;
use trunk_ir::{Attribute, Region, Symbol, Type};

/// Index for signature help lookups.
pub struct CallIndex;

impl CallIndex {
    /// Find a function definition's type by name.
    pub fn find_function_type<'db>(
        db: &'db dyn salsa::Database,
        module: &Module<'db>,
        name: Symbol,
    ) -> Option<Type<'db>> {
        Self::find_function_in_region(db, &module.body(db), name)
    }

    fn find_function_in_region<'db>(
        db: &'db dyn salsa::Database,
        region: &Region<'db>,
        name: Symbol,
    ) -> Option<Type<'db>> {
        use trunk_ir::DialectOp;
        use trunk_ir::dialect::func;

        for block in region.blocks(db).iter() {
            for op in block.operations(db).iter() {
                if let Ok(func_op) = func::Func::from_operation(db, *op)
                    && func_op.sym_name(db) == name
                {
                    return Some(func_op.ty(db));
                }

                // Recurse into nested regions
                for region in op.regions(db).iter() {
                    if let Some(ty) = Self::find_function_in_region(db, region, name) {
                        return Some(ty);
                    }
                }
            }
        }

        None
    }
}

/// Get parameter names from a function definition.
pub fn get_param_names<'db>(
    db: &'db dyn salsa::Database,
    module: &Module<'db>,
    name: Symbol,
) -> Vec<Option<Symbol>> {
    use trunk_ir::DialectOp;
    use trunk_ir::dialect::func;

    fn find_in_region<'db>(
        db: &'db dyn salsa::Database,
        region: &Region<'db>,
        name: Symbol,
    ) -> Option<Vec<Option<Symbol>>> {
        for block in region.blocks(db).iter() {
            for op in block.operations(db).iter() {
                if let Ok(func_op) = func::Func::from_operation(db, *op)
                    && func_op.sym_name(db) == name
                {
                    // Get parameter names from entry block arguments
                    let body = func_op.body(db);
                    let blocks = body.blocks(db);
                    if let Some(entry_block) = blocks.first() {
                        let names: Vec<Option<Symbol>> = entry_block
                            .args(db)
                            .iter()
                            .map(|arg| {
                                arg.get_attr(db, Symbol::new("bind_name")).and_then(|attr| {
                                    match attr {
                                        Attribute::Symbol(sym) => Some(*sym),
                                        _ => None,
                                    }
                                })
                            })
                            .collect();
                        return Some(names);
                    }
                }

                // Recurse into nested regions
                for region in op.regions(db).iter() {
                    if let Some(names) = find_in_region(db, region, name) {
                        return Some(names);
                    }
                }
            }
        }
        None
    }

    find_in_region(db, &module.body(db), name).unwrap_or_default()
}
