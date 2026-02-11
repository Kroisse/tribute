//! Lower func dialect operations to clif dialect.
//!
//! This pass converts function-level operations to Cranelift equivalents:
//! - `func.func` -> `clif.func`
//! - `func.call` -> `clif.call`
//! - `func.call_indirect` -> `clif.call_indirect`
//! - `func.return` -> `clif.return`
//! - `func.tail_call` -> `clif.return_call`
//! - `func.unreachable` -> `clif.trap`
//! - `func.constant` -> `clif.symbol_addr`

use trunk_ir::dialect::core::Module;
use trunk_ir::dialect::{clif, func};
use trunk_ir::rewrite::{
    ConversionError, ConversionTarget, OpAdaptor, PatternApplicator, RewritePattern, RewriteResult,
    TypeConverter,
};
use trunk_ir::{Attribute, DialectOp, Operation, Symbol};

/// Lower func dialect to clif dialect.
///
/// Returns an error if any `func.*` operations remain after conversion.
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
        .illegal_dialect("func");

    Ok(PatternApplicator::new(type_converter)
        .add_pattern(FuncFuncPattern)
        .add_pattern(FuncCallPattern)
        .add_pattern(FuncCallIndirectPattern)
        .add_pattern(FuncReturnPattern)
        .add_pattern(FuncTailCallPattern)
        .add_pattern(FuncUnreachablePattern)
        .add_pattern(FuncConstantPattern)
        .apply(db, module, target)?
        .module)
}

/// Pattern for `func.func` -> `clif.func`
struct FuncFuncPattern;

impl<'db> RewritePattern<'db> for FuncFuncPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(_func_op) = func::Func::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let new_op = op.modify(db).dialect_str("clif").name_str("func").build();
        RewriteResult::Replace(new_op)
    }
}

/// Pattern for `func.call` -> `clif.call`
struct FuncCallPattern;

impl<'db> RewritePattern<'db> for FuncCallPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(call_op) = func::Call::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let new_op = op
            .modify(db)
            .dialect_str("clif")
            .name_str("call")
            .attr("callee", Attribute::Symbol(call_op.callee(db)))
            .build();

        RewriteResult::Replace(new_op)
    }
}

/// Pattern for `func.call_indirect` -> `clif.call_indirect`
struct FuncCallIndirectPattern;

impl<'db> RewritePattern<'db> for FuncCallIndirectPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(_call_indirect) = func::CallIndirect::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let new_op = op
            .modify(db)
            .dialect_str("clif")
            .name_str("call_indirect")
            .build();

        RewriteResult::Replace(new_op)
    }
}

/// Pattern for `func.return` -> `clif.return`
struct FuncReturnPattern;

impl<'db> RewritePattern<'db> for FuncReturnPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(_return_op) = func::Return::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let new_op = op.modify(db).dialect_str("clif").name_str("return").build();

        RewriteResult::Replace(new_op)
    }
}

/// Pattern for `func.tail_call` -> `clif.return_call`
struct FuncTailCallPattern;

impl<'db> RewritePattern<'db> for FuncTailCallPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(tail_call_op) = func::TailCall::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let new_op = op
            .modify(db)
            .dialect_str("clif")
            .name_str("return_call")
            .attr("callee", Attribute::Symbol(tail_call_op.callee(db)))
            .build();

        RewriteResult::Replace(new_op)
    }
}

/// Pattern for `func.unreachable` -> `clif.trap`
struct FuncUnreachablePattern;

impl<'db> RewritePattern<'db> for FuncUnreachablePattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(_unreachable_op) = func::Unreachable::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let new_op = clif::trap(db, op.location(db), Symbol::new("unreachable"));
        RewriteResult::Replace(new_op.as_operation())
    }
}

/// Pattern for `func.constant` -> `clif.symbol_addr`
///
/// In the Cranelift backend, function references are symbol addresses
/// rather than table indices (unlike WASM).
struct FuncConstantPattern;

impl<'db> RewritePattern<'db> for FuncConstantPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(const_op) = func::Constant::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let result_ty = op.results(db)[0];
        let new_op = clif::symbol_addr(db, op.location(db), result_ty, const_op.func_ref(db));
        RewriteResult::Replace(new_op.as_operation())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use insta::assert_snapshot;
    use salsa_test_macros::salsa_test;
    use trunk_ir::dialect::{arith, core};
    use trunk_ir::{Attribute, Block, BlockId, DialectType, Location, PathId, Region, Span, idvec};

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
        let results = op.results(db);
        let attrs = op.attributes(db);

        let mut parts = vec![name];

        for (key, attr) in attrs.iter() {
            if (*key == "callee"
                || *key == "sym_name"
                || *key == "code"
                || *key == "sym"
                || *key == "func_ref")
                && let Attribute::Symbol(s) = attr
            {
                parts.push(format!("{}={}", key, s));
            }
        }

        if !operands.is_empty() {
            parts.push(format!("operands={}", operands.len()));
        }

        if !results.is_empty() {
            let result_types: Vec<_> = results.iter().map(|t| t.name(db).to_string()).collect();
            parts.push(format!("-> {}", result_types.join(", ")));
        }

        let mut result = format!("{}{}", prefix, parts.join(" "));

        // Recurse into regions
        for region in op.regions(db).iter() {
            for block in region.blocks(db).iter() {
                for nested_op in block.operations(db).iter() {
                    result.push('\n');
                    result.push_str(&format_op(db, nested_op, indent + 1));
                }
            }
        }

        result
    }

    #[salsa::tracked]
    fn make_func_call_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();

        let func_call = func::call(db, location, vec![], i32_ty, Symbol::new("foo"));

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![func_call.as_operation()],
        );
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa::tracked]
    fn make_func_func_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let nil_ty = core::Nil::new(db).as_type();
        let func_ty = core::Func::new(db, idvec![], nil_ty).as_type();

        let func_return = func::r#return(db, location, vec![]);

        let body_block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![func_return.as_operation()],
        );
        let body_region = Region::new(db, location, idvec![body_block]);

        let func_func =
            func::func(db, location, Symbol::new("test_fn"), func_ty, body_region).as_operation();

        let block = Block::new(db, BlockId::fresh(), location, idvec![], idvec![func_func]);
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa::tracked]
    fn format_lowered_module<'db>(db: &'db dyn salsa::Database, module: Module<'db>) -> String {
        let lowered = lower(db, module, test_converter()).expect("conversion should succeed");
        format_module_ops(db, &lowered)
    }

    #[salsa_test]
    fn test_func_call_to_clif(db: &salsa::DatabaseImpl) {
        let module = make_func_call_module(db);
        let formatted = format_lowered_module(db, module);

        assert_snapshot!(formatted, @"clif.call callee=foo -> i32");
    }

    #[salsa_test]
    fn test_func_func_to_clif(db: &salsa::DatabaseImpl) {
        let module = make_func_func_module(db);
        let formatted = format_lowered_module(db, module);

        assert_snapshot!(formatted);
    }

    #[salsa::tracked]
    fn make_func_constant_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let func_ty = core::Func::new(db, idvec![], core::Nil::new(db).as_type()).as_type();

        let func_constant = func::constant(db, location, func_ty, Symbol::new("test_func"));

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![func_constant.as_operation()],
        );
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa_test]
    fn test_func_constant_to_clif(db: &salsa::DatabaseImpl) {
        let module = make_func_constant_module(db);
        let formatted = format_lowered_module(db, module);

        assert_snapshot!(formatted, @"clif.symbol_addr sym=test_func -> func");
    }

    #[salsa::tracked]
    fn make_func_unreachable_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);

        let unreachable_op = func::unreachable(db, location);

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![unreachable_op.as_operation()],
        );
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa_test]
    fn test_func_unreachable_to_clif(db: &salsa::DatabaseImpl) {
        let module = make_func_unreachable_module(db);
        let formatted = format_lowered_module(db, module);

        assert_snapshot!(formatted, @"clif.trap code=unreachable");
    }

    #[salsa::tracked]
    fn make_call_indirect_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();

        let callee_op = arith::r#const(db, location, i32_ty, Attribute::IntBits(0));
        let callee_val = callee_op.result(db);

        let arg_op = arith::r#const(db, location, i32_ty, Attribute::IntBits(42));
        let arg_val = arg_op.result(db);

        let call_indirect = func::call_indirect(db, location, callee_val, vec![arg_val], i32_ty);

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![
                callee_op.as_operation(),
                arg_op.as_operation(),
                call_indirect.as_operation()
            ],
        );
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa_test]
    fn test_call_indirect_to_clif(db: &salsa::DatabaseImpl) {
        let module = make_call_indirect_module(db);
        let formatted = format_lowered_module(db, module);

        // func.call_indirect should become clif.call_indirect
        // arith.const ops should remain unchanged (different dialect)
        assert_snapshot!(formatted);
    }

    #[salsa::tracked]
    fn make_func_tail_call_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);

        let tail_call = func::tail_call(db, location, vec![], Symbol::new("target_fn"));

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![tail_call.as_operation()],
        );
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa_test]
    fn test_func_tail_call_to_clif(db: &salsa::DatabaseImpl) {
        let module = make_func_tail_call_module(db);
        let formatted = format_lowered_module(db, module);

        assert_snapshot!(formatted, @"clif.return_call callee=target_fn");
    }
}
