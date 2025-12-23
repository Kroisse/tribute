//! Strip source dialect operations before wasm lowering.
//!
//! This pass removes `src.var` operations that are markers for IDE hover
//! (marked with `resolved_local` attribute). These operations have unused
//! results and are safe to remove.

use trunk_ir::dialect::core::Module;
use trunk_ir::dialect::src;
use trunk_ir::rewrite::{PatternApplicator, RewritePattern, RewriteResult};
use trunk_ir::{DialectOp, Operation, Symbol};

/// Strip source dialect operations that are metadata markers.
pub fn lower<'db>(db: &'db dyn salsa::Database, module: Module<'db>) -> Module<'db> {
    PatternApplicator::new()
        .add_pattern(ResolvedLocalPattern)
        .apply(db, module)
        .module
}

/// Pattern to remove src.var with resolved_local attribute.
/// These are IDE hover markers with unused results.
struct ResolvedLocalPattern;

impl RewritePattern for ResolvedLocalPattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
    ) -> RewriteResult<'db> {
        // Check if this is src.var
        let Ok(_src_var) = src::Var::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        // Check if it has resolved_local attribute (hover marker)
        let attrs = op.attributes(db);
        if attrs.get(&Symbol::new("resolved_local")).is_some() {
            // This is a hover marker - its result should be unused
            // Safe to erase with no replacement
            return RewriteResult::Erase {
                replacement_values: vec![],
            };
        }

        // Non-hover src.var - shouldn't happen but leave unchanged
        #[cfg(debug_assertions)]
        eprintln!(
            "WARNING: Non-hover src.var in wasm lowering: {:?}",
            op.name(db)
        );
        RewriteResult::Unchanged
    }
}
