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
use trunk_ir::dialect::{adt, clif, core, func};
use trunk_ir::rewrite::{
    ConversionError, ConversionTarget, PatternApplicator, PatternRewriter, RewritePattern,
    TypeConverter,
};
use trunk_ir::{Attribute, DialectOp, DialectType, Operation, Symbol, Type};

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
    // Phase 1: Adapt closure structs for native backend.
    // This runs separately because adt.* ops are "legal" in the func->clif target
    // and would be skipped by the main pattern applicator.
    let module = adapt_closure_structs(db, module);

    // Phase 2: Lower func dialect to clif dialect.
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

/// Adapt closure struct operations for the native backend.
///
/// Runs as a pre-processing step with no legality constraints so that
/// `adt.*` operations on `_closure` structs are visited by the pattern.
fn adapt_closure_structs<'db>(db: &'db dyn salsa::Database, module: Module<'db>) -> Module<'db> {
    let target = ConversionTarget::new();
    PatternApplicator::new(TypeConverter::new())
        .add_pattern(ClosureStructAdaptPattern)
        .apply_partial(db, module, target)
        .module
}

/// Pattern for `func.func` -> `clif.func`
///
/// Converts the function type attribute's parameter and return types using
/// the type converter, ensuring high-level types (e.g., `core.array`) are
/// mapped to their native representations before Cranelift emission.
struct FuncFuncPattern;

impl<'db> RewritePattern<'db> for FuncFuncPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        rewriter: &mut PatternRewriter<'db, '_>,
    ) -> bool {
        let Ok(func_op) = func::Func::from_operation(db, *op) else {
            return false;
        };

        let type_converter = rewriter.type_converter();
        let func_type_attr = func_op.r#type(db);

        // Convert parameter and return types in the function signature
        let mut builder = op.modify(db).dialect_str("clif").name_str("func");
        if let Some(func_ty) = core::Func::from_type(db, func_type_attr) {
            let new_params = type_converter.convert_types(db, &func_ty.params(db));
            let new_ret = type_converter
                .convert_type(db, func_ty.result(db))
                .unwrap_or(func_ty.result(db));
            let new_func_ty = core::Func::new(db, new_params, new_ret).as_type();
            builder = builder.attr("type", Attribute::Type(new_func_ty));
        }

        rewriter.replace_op(builder.build());
        true
    }
}

/// Pattern for `func.call` -> `clif.call`
struct FuncCallPattern;

impl<'db> RewritePattern<'db> for FuncCallPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        rewriter: &mut PatternRewriter<'db, '_>,
    ) -> bool {
        let Ok(call_op) = func::Call::from_operation(db, *op) else {
            return false;
        };

        let new_op = op
            .modify(db)
            .dialect_str("clif")
            .name_str("call")
            .attr("callee", Attribute::Symbol(call_op.callee(db)))
            .build();

        rewriter.replace_op(new_op);
        true
    }
}

/// Pattern for `func.call_indirect` -> `clif.call_indirect`
///
/// Constructs the required `sig` attribute by collecting operand/result types.
/// `func.call_indirect` has operands: [callee, args...] and one result.
/// The `sig` is a `core.func` type with param types from args and the result type.
struct FuncCallIndirectPattern;

impl<'db> RewritePattern<'db> for FuncCallIndirectPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        rewriter: &mut PatternRewriter<'db, '_>,
    ) -> bool {
        let Ok(_call_indirect) = func::CallIndirect::from_operation(db, *op) else {
            return false;
        };

        // Collect argument types (skip operand 0 which is the callee).
        // Bail out if any type is unavailable so the ConversionTarget can report the unconverted op.
        let mut param_types = Vec::new();
        for i in 1..rewriter.num_operands() {
            let Some(ty) = rewriter.operand_type(i) else {
                return false;
            };
            param_types.push(ty);
        }

        let Some(result_ty) = rewriter.result_type(db, op, 0) else {
            return false;
        };
        let sig_ty = core::Func::new(db, param_types.into(), result_ty).as_type();

        let new_op = op
            .modify(db)
            .dialect_str("clif")
            .name_str("call_indirect")
            .attr("sig", Attribute::Type(sig_ty))
            .build();

        rewriter.replace_op(new_op);
        true
    }
}

/// Pattern for `func.return` -> `clif.return`
struct FuncReturnPattern;

impl<'db> RewritePattern<'db> for FuncReturnPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        rewriter: &mut PatternRewriter<'db, '_>,
    ) -> bool {
        let Ok(_return_op) = func::Return::from_operation(db, *op) else {
            return false;
        };

        let new_op = op.modify(db).dialect_str("clif").name_str("return").build();

        rewriter.replace_op(new_op);
        true
    }
}

/// Pattern for `func.tail_call` -> `clif.return_call`
struct FuncTailCallPattern;

impl<'db> RewritePattern<'db> for FuncTailCallPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        rewriter: &mut PatternRewriter<'db, '_>,
    ) -> bool {
        let Ok(tail_call_op) = func::TailCall::from_operation(db, *op) else {
            return false;
        };

        let new_op = op
            .modify(db)
            .dialect_str("clif")
            .name_str("return_call")
            .attr("callee", Attribute::Symbol(tail_call_op.callee(db)))
            .build();

        rewriter.replace_op(new_op);
        true
    }
}

/// Pattern for `func.unreachable` -> `clif.trap`
struct FuncUnreachablePattern;

impl<'db> RewritePattern<'db> for FuncUnreachablePattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        rewriter: &mut PatternRewriter<'db, '_>,
    ) -> bool {
        let Ok(_unreachable_op) = func::Unreachable::from_operation(db, *op) else {
            return false;
        };

        let new_op = clif::trap(db, op.location(db), Symbol::new("unreachable"));
        rewriter.replace_op(new_op.as_operation());
        true
    }
}

/// Pattern for `func.constant` -> `clif.symbol_addr`
///
/// In the Cranelift backend, function references are symbol addresses (pointers)
/// rather than table indices (unlike WASM). The result type is `core.ptr`.
struct FuncConstantPattern;

impl<'db> RewritePattern<'db> for FuncConstantPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        rewriter: &mut PatternRewriter<'db, '_>,
    ) -> bool {
        let Ok(const_op) = func::Constant::from_operation(db, *op) else {
            return false;
        };

        let ptr_ty = core::Ptr::new(db).as_type();
        let new_op = clif::symbol_addr(db, op.location(db), ptr_ty, const_op.func_ref(db));
        rewriter.replace_op(new_op.as_operation());
        true
    }
}

// =============================================================================
// Closure struct adaptation for native backend
// =============================================================================

const CLOSURE_STRUCT_NAME: &str = "_closure";

/// Check if a type is the closure struct type (name == "_closure").
fn is_closure_struct(db: &dyn salsa::Database, ty: Type<'_>) -> bool {
    adt::get_type_name(db, ty)
        .map(|name| name == Symbol::new(CLOSURE_STRUCT_NAME))
        .unwrap_or(false)
}

/// Create the native closure struct type: `{ func_ptr: i64, env: ptr }`.
///
/// In the native backend, both fields are 64-bit:
/// - field 0: function pointer (from `clif.symbol_addr`) — typed `i64` rather
///   than `ptr` so the RC insertion pass does NOT retain/release it (function
///   pointers are code addresses, not heap-allocated objects)
/// - field 1: environment pointer (boxed captures)
fn native_closure_struct_type(db: &dyn salsa::Database) -> Type<'_> {
    let i64_ty = core::I64::new(db).as_type();
    let ptr_ty = core::Ptr::new(db).as_type();
    adt::struct_type(
        db,
        Symbol::new(CLOSURE_STRUCT_NAME),
        vec![
            (Symbol::new("func_ptr"), i64_ty),
            (Symbol::new("env"), ptr_ty),
        ],
    )
}

/// Adapt closure struct operations for the native backend.
///
/// The shared `closure_lower` pass produces `_closure { i32, anyref }` structs
/// (designed for WASM). For native codegen, we need `_closure { i64, ptr }`:
/// - `adt.struct_new` on `_closure`: change type attribute to `{ i64, ptr }`
/// - `adt.struct_get` on `_closure` field 0: change result type to `i64`
///   (NOT `ptr`, so the RC pass doesn't retain/release function pointers)
/// - `adt.struct_get` on `_closure` field 1: change result type to `ptr`
struct ClosureStructAdaptPattern;

impl<'db> RewritePattern<'db> for ClosureStructAdaptPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        rewriter: &mut PatternRewriter<'db, '_>,
    ) -> bool {
        let native_ty = native_closure_struct_type(db);

        // Handle adt.struct_new on _closure
        if let Ok(struct_new) = adt::StructNew::from_operation(db, *op) {
            let ty = struct_new.r#type(db);
            if !is_closure_struct(db, ty) {
                return false;
            }
            let new_op = op
                .modify(db)
                .attr("type", Attribute::Type(native_ty))
                .results(vec![native_ty].into())
                .build();
            rewriter.replace_op(new_op);
            return true;
        }

        // Handle adt.struct_get on _closure
        if let Ok(struct_get) = adt::StructGet::from_operation(db, *op) {
            let ty = struct_get.r#type(db);
            if !is_closure_struct(db, ty) {
                return false;
            }
            let field_idx = struct_get.field(db);
            let mut builder = op.modify(db).attr("type", Attribute::Type(native_ty));
            if field_idx == 0 {
                // func_ptr: i64 (not ptr — avoids RC tracking of function pointers)
                let i64_ty = core::I64::new(db).as_type();
                builder = builder.results(vec![i64_ty].into());
            } else if field_idx == 1 {
                // env: ptr (heap-allocated captures — needs RC tracking)
                let ptr_ty = core::Ptr::new(db).as_type();
                builder = builder.results(vec![ptr_ty].into());
            }
            let new_op = builder.build();
            rewriter.replace_op(new_op);
            return true;
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use insta::assert_snapshot;
    use salsa_test_macros::salsa_test;
    use trunk_ir::dialect::{arith, core, wasm};
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
            match attr {
                Attribute::Symbol(s)
                    if *key == "callee"
                        || *key == "sym_name"
                        || *key == "code"
                        || *key == "sym"
                        || *key == "func_ref" =>
                {
                    parts.push(format!("{}={}", key, s));
                }
                Attribute::Type(ty) if *key == "sig" || *key == "type" => {
                    parts.push(format!("{}={}", key, ty.name(db)));
                }
                Attribute::IntBits(n) if *key == "field" => {
                    parts.push(format!("field={}", n));
                }
                _ => {}
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

        assert_snapshot!(formatted, @"clif.symbol_addr sym=test_func -> ptr");
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

    // =========================================================================
    // Closure struct adaptation tests
    // =========================================================================

    /// Create a WASM-style closure struct type: `{ table_idx: i32, env: anyref }`
    fn wasm_closure_struct_type(db: &dyn salsa::Database) -> trunk_ir::Type<'_> {
        let i32_ty = core::I32::new(db).as_type();
        // Use core.ptr as a stand-in for anyref in tests
        let anyref_ty = core::Ptr::new(db).as_type();
        adt::struct_type(
            db,
            Symbol::new("_closure"),
            vec![
                (Symbol::new("table_idx"), i32_ty),
                (Symbol::new("env"), anyref_ty),
            ],
        )
    }

    #[salsa::tracked]
    fn make_closure_struct_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();
        let ptr_ty = core::Ptr::new(db).as_type();
        let closure_ty = wasm_closure_struct_type(db);

        // Simulate closure creation: func.constant + adt.struct_new
        let func_const = func::constant(db, location, i32_ty, Symbol::new("lifted_fn"));
        let func_ptr_val = func_const.result(db);

        let env_op = arith::r#const(db, location, ptr_ty, Attribute::IntBits(0));
        let env_val = env_op.result(db);

        let struct_new = adt::struct_new(
            db,
            location,
            vec![func_ptr_val, env_val],
            closure_ty,
            closure_ty,
        );
        let closure_val = struct_new.result(db);

        // Extract func ptr (field 0) and env (field 1)
        let get_func = adt::struct_get(db, location, closure_val, i32_ty, closure_ty, 0);
        let func_ptr = get_func.result(db);

        let get_env = adt::struct_get(db, location, closure_val, ptr_ty, closure_ty, 1);
        let env = get_env.result(db);

        // Indirect call through the closure
        let call_result_ty = i32_ty;
        let call_indirect = func::call_indirect(db, location, func_ptr, vec![env], call_result_ty);

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![
                func_const.as_operation(),
                env_op.as_operation(),
                struct_new.as_operation(),
                get_func.as_operation(),
                get_env.as_operation(),
                call_indirect.as_operation()
            ],
        );
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa_test]
    fn test_closure_struct_adaptation(db: &salsa::DatabaseImpl) {
        let module = make_closure_struct_module(db);
        let formatted = format_lowered_module(db, module);

        // func.constant -> clif.symbol_addr with ptr result
        // adt.struct_new on _closure -> type adapted to { ptr, ptr }
        // adt.struct_get field 0 -> result type becomes ptr
        // func.call_indirect -> clif.call_indirect with sig attribute
        assert_snapshot!(formatted);
    }

    // Note: The previous test `test_call_indirect_unchanged_on_missing_types` was
    // removed during the OpAdaptor→PatternRewriter migration. It tested internal
    // behavior by manually constructing an OpAdaptor with None operand types.
    // With the new PatternRewriter API (whose constructor is pub(crate) to trunk_ir),
    // this white-box test cannot be replicated from outside the crate. The behavior
    // is still upheld by the pattern implementation (bail on missing types).

    /// Test closure struct adaptation with actual `wasm.anyref` env type.
    ///
    /// Verifies that `adt.struct_get` on field 1 (env) produces `ptr` result
    /// even when the original type is `wasm.anyref`.
    #[salsa::tracked]
    fn make_closure_struct_anyref_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();
        let anyref_ty = wasm::Anyref::new(db).as_type();

        // WASM-style closure struct: { table_idx: i32, env: anyref }
        let closure_ty = adt::struct_type(
            db,
            Symbol::new("_closure"),
            vec![
                (Symbol::new("table_idx"), i32_ty),
                (Symbol::new("env"), anyref_ty),
            ],
        );

        let func_const = func::constant(db, location, i32_ty, Symbol::new("lifted_fn"));
        let func_ptr_val = func_const.result(db);

        let env_op = arith::r#const(db, location, anyref_ty, Attribute::IntBits(0));
        let env_val = env_op.result(db);

        let struct_new = adt::struct_new(
            db,
            location,
            vec![func_ptr_val, env_val],
            closure_ty,
            closure_ty,
        );
        let closure_val = struct_new.result(db);

        // Extract func ptr (field 0) and env (field 1)
        let get_func = adt::struct_get(db, location, closure_val, i32_ty, closure_ty, 0);
        let func_ptr = get_func.result(db);

        let get_env = adt::struct_get(db, location, closure_val, anyref_ty, closure_ty, 1);
        let env = get_env.result(db);

        let call_indirect = func::call_indirect(db, location, func_ptr, vec![env], i32_ty);

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![
                func_const.as_operation(),
                env_op.as_operation(),
                struct_new.as_operation(),
                get_func.as_operation(),
                get_env.as_operation(),
                call_indirect.as_operation()
            ],
        );
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa_test]
    fn test_closure_struct_anyref_adaptation(db: &salsa::DatabaseImpl) {
        let module = make_closure_struct_anyref_module(db);
        let formatted = format_lowered_module(db, module);

        // Both struct_get field 0 and field 1 should produce ptr results
        assert_snapshot!(formatted);
    }
}
