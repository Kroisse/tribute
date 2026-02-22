//! Lower cf dialect operations to clif dialect.
//!
//! This pass converts CFG-based control flow operations to Cranelift equivalents:
//! - `cf.br` -> `clif.jump`
//! - `cf.cond_br` -> `clif.brif`

use trunk_ir::dialect::cf;
use trunk_ir::dialect::core::Module;
use trunk_ir::rewrite::{
    ConversionError, ConversionTarget, PatternApplicator, PatternRewriter, RewritePattern,
    TypeConverter,
};
use trunk_ir::{DialectOp, Operation};

/// Lower cf dialect to clif dialect.
///
/// Returns an error if any `cf.*` operations remain after conversion.
///
/// The `type_converter` parameter allows language-specific backends to provide
/// their own type conversion rules.
pub fn lower<'db>(
    db: &'db dyn salsa::Database,
    module: Module<'db>,
    type_converter: TypeConverter,
) -> Result<Module<'db>, ConversionError> {
    let target = ConversionTarget::new()
        .legal_dialect("clif")
        .illegal_dialect("cf");

    Ok(PatternApplicator::new(type_converter)
        .add_pattern(CfBrPattern)
        .add_pattern(CfCondBrPattern)
        .apply(db, module, target)?
        .module)
}

/// Pattern for `cf.br` -> `clif.jump`
///
/// Both operations have the same structure: variadic args + single successor block.
/// We just change the dialect/name.
struct CfBrPattern;

impl<'db> RewritePattern<'db> for CfBrPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        rewriter: &mut PatternRewriter<'db, '_>,
    ) -> bool {
        let Ok(_br_op) = cf::Br::from_operation(db, *op) else {
            return false;
        };

        let new_op = op.modify(db).dialect_str("clif").name_str("jump").build();
        rewriter.replace_op(new_op);
        true
    }
}

/// Pattern for `cf.cond_br` -> `clif.brif`
///
/// Both operations have the same structure: cond operand + two successor blocks.
/// We just change the dialect/name.
struct CfCondBrPattern;

impl<'db> RewritePattern<'db> for CfCondBrPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        rewriter: &mut PatternRewriter<'db, '_>,
    ) -> bool {
        let Ok(_cond_br_op) = cf::CondBr::from_operation(db, *op) else {
            return false;
        };

        let new_op = op.modify(db).dialect_str("clif").name_str("brif").build();
        rewriter.replace_op(new_op);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use insta::assert_snapshot;
    use salsa_test_macros::salsa_test;
    use trunk_ir::dialect::{clif, core};
    use trunk_ir::{
        Attribute, Block, BlockBuilder, BlockId, DialectType, Location, PathId, Region, Span, idvec,
    };

    fn test_location(db: &dyn salsa::Database) -> Location<'_> {
        let path = PathId::new(db, "file:///test.trb".to_owned());
        Location::new(path, Span::new(0, 0))
    }

    fn test_converter() -> TypeConverter {
        TypeConverter::new()
    }

    /// Format module operations for snapshot testing
    fn format_module_ops(db: &dyn salsa::Database, module: &Module<'_>) -> String {
        let body = module.body(db);
        let ops = &body.blocks(db)[0].operations(db);
        ops.iter()
            .map(|op| format_op(db, op, 0))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn format_op(db: &dyn salsa::Database, op: &Operation<'_>, indent: usize) -> String {
        let prefix = "  ".repeat(indent);
        let name = op.full_name(db);
        let operands = op.operands(db);
        let successors = op.successors(db);

        let mut parts = vec![name];

        if !operands.is_empty() {
            parts.push(format!("operands={}", operands.len()));
        }

        if !successors.is_empty() {
            parts.push(format!("successors={}", successors.len()));
        }

        format!("{}{}", prefix, parts.join(" "))
    }

    #[salsa::tracked]
    fn make_cf_br_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();

        let target = BlockBuilder::new(db, location).arg(i32_ty).build();
        let dummy_const = trunk_ir::dialect::arith::Const::i32(db, location, 42);
        let br_op = cf::br(db, location, [dummy_const.result(db)], target);

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![dummy_const.as_operation(), br_op.as_operation()],
        );
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa::tracked]
    fn make_cf_cond_br_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let i1_ty = core::I1::new(db).as_type();

        let cond_val =
            trunk_ir::dialect::arith::r#const(db, location, i1_ty, Attribute::Bool(true));
        let then_block = BlockBuilder::new(db, location).build();
        let else_block = BlockBuilder::new(db, location).build();
        let cond_br_op = cf::cond_br(db, location, cond_val.result(db), then_block, else_block);

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![cond_val.as_operation(), cond_br_op.as_operation()],
        );
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa::tracked]
    fn format_lowered_module<'db>(db: &'db dyn salsa::Database, module: Module<'db>) -> String {
        let lowered = lower(db, module, test_converter()).expect("conversion should succeed");
        format_module_ops(db, &lowered)
    }

    #[salsa_test]
    fn test_cf_br_to_clif_jump(db: &salsa::DatabaseImpl) {
        let module = make_cf_br_module(db);
        let formatted = format_lowered_module(db, module);

        // cf.br should become clif.jump, arith.const should remain unchanged
        assert_snapshot!(formatted, @r"
        arith.const
        clif.jump operands=1 successors=1
        ");
    }

    #[salsa_test]
    fn test_cf_cond_br_to_clif_brif(db: &salsa::DatabaseImpl) {
        let module = make_cf_cond_br_module(db);
        let formatted = format_lowered_module(db, module);

        // cf.cond_br should become clif.brif, arith.const should remain unchanged
        assert_snapshot!(formatted, @r"
        arith.const
        clif.brif operands=1 successors=2
        ");
    }

    #[salsa::tracked]
    fn lower_and_check_br_successors(db: &dyn salsa::Database, module: Module<'_>) -> bool {
        let lowered = lower(db, module, test_converter()).expect("conversion should succeed");
        let body = lowered.body(db);
        let ops = body.blocks(db)[0].operations(db);
        let jump_op = ops
            .iter()
            .find(|op| op.name(db) == trunk_ir::Symbol::new("jump"))
            .unwrap();
        let jump = clif::Jump::from_operation(db, *jump_op).unwrap();
        // Verify successor is accessible
        let _dest = jump.dest(db);
        true
    }

    #[salsa_test]
    fn test_cf_br_preserves_successors(db: &salsa::DatabaseImpl) {
        let module = make_cf_br_module(db);
        assert!(lower_and_check_br_successors(db, module));
    }

    #[salsa::tracked]
    fn lower_and_check_cond_br_successors(db: &dyn salsa::Database, module: Module<'_>) -> bool {
        let lowered = lower(db, module, test_converter()).expect("conversion should succeed");
        let body = lowered.body(db);
        let ops = body.blocks(db)[0].operations(db);
        let brif_op = ops
            .iter()
            .find(|op| op.name(db) == trunk_ir::Symbol::new("brif"))
            .unwrap();
        let brif = clif::Brif::from_operation(db, *brif_op).unwrap();
        // Verify both successors are accessible
        let _then = brif.then_dest(db);
        let _else_ = brif.else_dest(db);
        true
    }

    #[salsa_test]
    fn test_cf_cond_br_preserves_successors(db: &salsa::DatabaseImpl) {
        let module = make_cf_cond_br_module(db);
        assert!(lower_and_check_cond_br_successors(db, module));
    }
}
