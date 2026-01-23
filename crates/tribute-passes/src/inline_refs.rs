//! Inline `tribute.ref` operations.
//!
//! This pass inlines `tribute.ref` operations by replacing their results
//! with their operands. `tribute.ref` is a pass-through operation that
//! preserves source location for LSP hover functionality.
//!
//! ## Pipeline Position
//!
//! This pass runs after TDNR and before code generation:
//! ```text
//! stage_tdnr → inline_refs → lambda_lift → ...
//! ```
//!
//! LSP uses `compile_for_lsp` which stops at TDNR, preserving `tribute.ref`
//! for hover information.

use tribute_ir::dialect::tribute;
use trunk_ir::dialect::core::Module;
use trunk_ir::rewrite::{
    ConversionError, ConversionTarget, OpAdaptor, PatternApplicator, RewritePattern, RewriteResult,
    TypeConverter,
};
use trunk_ir::{DialectOp, Operation};

/// Pattern to inline `tribute.ref` operations.
struct InlineTributeRefPattern;

impl<'db> RewritePattern<'db> for InlineTributeRefPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        // Match: tribute.ref
        if tribute::Ref::from_operation(db, *op).is_err() {
            return RewriteResult::Unchanged;
        }

        let operands = adaptor.operands();
        if operands.is_empty() {
            return RewriteResult::Unchanged;
        }

        // Inline: replace the result with the operand
        RewriteResult::erase(vec![operands[0]])
    }
}

/// Inline all `tribute.ref` operations in a module.
pub fn inline_refs<'db>(
    db: &'db dyn salsa::Database,
    module: Module<'db>,
) -> Result<Module<'db>, ConversionError> {
    let applicator =
        PatternApplicator::new(TypeConverter::new()).add_pattern(InlineTributeRefPattern);
    let target = ConversionTarget::new().illegal_op(tribute::DIALECT_NAME(), tribute::REF());
    Ok(applicator.apply(db, module, target)?.module)
}
