//! Constant inlining pass for Tribute.
//!
//! This pass inlines constant values at their use sites:
//! - Finds `tribute.var` operations marked with `resolved_const=true`
//! - Replaces them with `arith.const` operations containing the inlined value
//!
//! ## Example
//!
//! ```tribute
//! const MAX_SIZE = 1024;
//!
//! fn example() {
//!     let x = MAX_SIZE;
//! }
//! ```
//!
//! After name resolution, `MAX_SIZE` reference is marked as `tribute.var` with:
//! - `resolved_const = true`
//! - `value = 1024`
//!
//! After this pass, it becomes:
//! - `arith.const(1024)` with type `i64`
//!
//! Uses `RewritePattern` + `PatternApplicator` for declarative transformation.

use trunk_ir::dialect::arith;
use trunk_ir::dialect::core::Module;
use trunk_ir::rewrite::{
    ConversionTarget, OpAdaptor, PatternApplicator, RewritePattern, RewriteResult, TypeConverter,
};
use trunk_ir::{Attribute, DialectOp, Operation};

// =============================================================================
// Attribute Keys
// =============================================================================

trunk_ir::symbols! {
    ATTR_RESOLVED_CONST => "resolved_const",
    ATTR_VALUE => "value",
}

// =============================================================================
// Inline Const Pattern
// =============================================================================

/// Pattern to inline constant references.
///
/// Matches `tribute.var` operations with `resolved_const=true` attribute
/// and replaces them with `arith.const` operations containing the inlined value.
struct InlineConstPattern;

impl<'db> RewritePattern<'db> for InlineConstPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        // Check if this is a resolved const reference
        let attrs = op.attributes(db);
        let is_resolved_const = matches!(
            attrs.get(&ATTR_RESOLVED_CONST()),
            Some(Attribute::Bool(true)) | Some(Attribute::IntBits(1))
        );

        if !is_resolved_const {
            return RewriteResult::Unchanged;
        }

        // Get the value attribute
        let Some(value_attr) = attrs.get(&ATTR_VALUE()) else {
            return RewriteResult::Unchanged;
        };

        // Get the result type
        let Some(result_ty) = op.results(db).first().copied() else {
            return RewriteResult::Unchanged;
        };

        let location = op.location(db);

        // Create arith.const with the inlined value
        let const_op = arith::r#const(db, location, result_ty, value_attr.clone());

        RewriteResult::Replace(const_op.as_operation())
    }
}

// =============================================================================
// Pipeline Integration
// =============================================================================

/// Inline constants in a module.
///
/// Uses `PatternApplicator` for declarative transformation.
/// The tracked version is in pipeline.rs (stage_const_inline).
pub fn inline_module<'db>(db: &'db dyn salsa::Database, module: &Module<'db>) -> Module<'db> {
    let applicator = PatternApplicator::new(TypeConverter::new()).add_pattern(InlineConstPattern);
    let target = ConversionTarget::new();

    applicator.apply_partial(db, *module, target).module
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;

    // Import the resolve module tests to reuse their test data
    use crate::resolve::tests::resolve_const_reference_module;

    #[salsa::tracked]
    fn inline_after_resolve(db: &dyn salsa::Database) -> Module<'_> {
        // Get the resolved module (with marked const references)
        let resolved = resolve_const_reference_module(db);
        // Run const inlining
        inline_module(db, &resolved)
    }

    #[salsa_test]
    fn test_const_inlining(db: &salsa::DatabaseImpl) {
        use trunk_ir::{Attribute, Symbol};

        let inlined = inline_after_resolve(db);

        // Collect all operations
        let mut ops = Vec::new();
        for block in inlined.body(db).blocks(db).iter() {
            for op in block.operations(db).iter() {
                ops.push(*op);
                // Also collect from nested regions (function bodies)
                for region in op.regions(db).iter() {
                    for nested_block in region.blocks(db).iter() {
                        for nested_op in nested_block.operations(db).iter() {
                            ops.push(*nested_op);
                        }
                    }
                }
            }
        }

        // Should have no more tribute.var with resolved_const
        let resolved_consts: Vec<_> = ops
            .iter()
            .filter(|op| {
                op.dialect(db) == Symbol::new("tribute")
                    && op.name(db) == Symbol::new("var")
                    && matches!(
                        op.attributes(db).get(&Symbol::new("resolved_const")),
                        Some(Attribute::Bool(true))
                    )
            })
            .collect();

        assert!(
            resolved_consts.is_empty(),
            "resolved const references should be inlined"
        );

        // Should have arith.const with value 1024
        let const_ops: Vec<_> = ops
            .iter()
            .filter(|op| {
                op.dialect(db) == Symbol::new("arith") && op.name(db) == Symbol::new("const")
            })
            .collect();

        assert!(!const_ops.is_empty(), "should have arith.const operations");

        // Verify the const has the right value
        let has_correct_value = const_ops.iter().any(|op| {
            matches!(
                op.attributes(db).get(&Symbol::new("value")),
                Some(Attribute::IntBits(1024))
            )
        });

        assert!(has_correct_value, "arith.const should have value 1024");
    }
}
