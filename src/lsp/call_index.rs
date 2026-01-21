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

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use tree_sitter::Parser;
    use tribute::SourceCst;
    use tribute::compile_for_lsp;

    fn make_source(path: &str, text: &str) -> SourceCst {
        salsa::with_attached_database(|db| {
            let mut parser = Parser::new();
            parser
                .set_language(&tree_sitter_tribute::LANGUAGE.into())
                .expect("Failed to set language");
            let tree = parser.parse(text, None).expect("tree");
            SourceCst::from_path(db, path, text.into(), Some(tree))
        })
        .expect("attached db")
    }

    #[salsa_test]
    fn test_find_function_type(db: &salsa::DatabaseImpl) {
        let source_text = "fn add(x: Int, y: Int) -> Int { x + y }";
        let source = make_source("test.trb", source_text);

        let module = compile_for_lsp(db, source);

        // Should find the function type for "add"
        let func_ty = CallIndex::find_function_type(db, &module, Symbol::new("add"));
        assert!(func_ty.is_some(), "Should find function 'add'");

        // Should not find a non-existent function
        let missing = CallIndex::find_function_type(db, &module, Symbol::new("missing"));
        assert!(missing.is_none(), "Should not find non-existent function");
    }

    #[salsa_test]
    fn test_get_param_names(db: &salsa::DatabaseImpl) {
        let source_text = "fn greet(name: Text, count: Int) { }";
        let source = make_source("test.trb", source_text);

        let module = compile_for_lsp(db, source);

        let names = get_param_names(db, &module, Symbol::new("greet"));
        assert_eq!(names.len(), 2, "Should have 2 parameters");

        // Parameter names should be present
        assert!(
            names
                .iter()
                .any(|n| n.map(|s| s.to_string()) == Some("name".to_string())),
            "Should have parameter 'name'"
        );
        assert!(
            names
                .iter()
                .any(|n| n.map(|s| s.to_string()) == Some("count".to_string())),
            "Should have parameter 'count'"
        );
    }

    #[salsa_test]
    fn test_get_param_names_no_params(db: &salsa::DatabaseImpl) {
        let source_text = "fn hello() { }";
        let source = make_source("test.trb", source_text);

        let module = compile_for_lsp(db, source);

        let names = get_param_names(db, &module, Symbol::new("hello"));
        assert!(names.is_empty(), "Should have no parameters");
    }

    #[salsa_test]
    fn test_get_param_names_missing_function(db: &salsa::DatabaseImpl) {
        let source_text = "fn foo() { }";
        let source = make_source("test.trb", source_text);

        let module = compile_for_lsp(db, source);

        let names = get_param_names(db, &module, Symbol::new("missing"));
        assert!(names.is_empty(), "Should return empty for missing function");
    }
}
