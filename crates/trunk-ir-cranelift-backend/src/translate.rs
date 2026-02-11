//! Cranelift native object file translation.
//!
//! This module provides functions for validating and emitting native object files
//! from TrunkIR modules that have already been lowered to the clif dialect.

use std::collections::HashMap;

use cranelift_codegen::ir::{self as cl_ir, UserFuncName};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_codegen::{Context, isa};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_module::{Linkage, Module as CraneliftModule, default_libcall_names};
use cranelift_object::{ObjectBuilder, ObjectModule};
use target_lexicon::Triple;
use trunk_ir::dialect::{clif, core};
use trunk_ir::{DialectOp, DialectType, Symbol};

use crate::function::{FunctionTranslator, translate_signature};
use crate::{CompilationError, CompilationResult, validate_clif_ir};

/// Emit a native object file from a lowered TrunkIR module.
///
/// This function assumes the module has already been lowered to clif dialect
/// and all type conversions have been resolved. It:
/// 1. Validates the IR (checks for non-clif ops)
/// 2. Emits native code via Cranelift
///
/// For Tribute-specific compilation (including lowering from high-level IR),
/// use the orchestration in the main crate's pipeline.
#[salsa::tracked]
pub fn emit_module_to_native<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
) -> CompilationResult<Vec<u8>> {
    validate_clif_ir(db, module)?;
    emit_module_impl(db, module)
}

/// Internal implementation of module emission.
fn emit_module_impl<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
) -> CompilationResult<Vec<u8>> {
    // 1. ISA setup — use host triple
    let triple = Triple::host();
    let mut flag_builder = settings::builder();
    flag_builder
        .set("is_pic", "true")
        .map_err(|e| CompilationError::codegen(format!("{e}")))?;
    let isa_builder = isa::lookup(triple).map_err(|e| CompilationError::codegen(format!("{e}")))?;
    let isa = isa_builder
        .finish(settings::Flags::new(flag_builder))
        .map_err(|e| CompilationError::codegen(format!("{e}")))?;
    let call_conv = isa.default_call_conv();

    // 2. ObjectModule creation
    let module_name = module.name(db).to_string();
    let obj_builder = ObjectBuilder::new(isa, module_name, default_libcall_names())
        .map_err(|e| CompilationError::codegen(format!("{e}")))?;
    let mut obj_module = ObjectModule::new(obj_builder);

    // 3. First pass — declare all functions
    let mut func_ids: HashMap<Symbol, cranelift_module::FuncId> = HashMap::new();
    let body = module.body(db);
    for block in body.blocks(db).iter() {
        for op in block.operations(db).iter() {
            if let Ok(func_op) = clif::Func::from_operation(db, *op) {
                let name_sym = func_op.sym_name(db);
                let func_type_attr = func_op.r#type(db);
                let func_ty = core::Func::from_type(db, func_type_attr).ok_or_else(|| {
                    CompilationError::type_error("expected core.func type on clif.func")
                })?;

                let sig = translate_signature(db, func_ty, call_conv)?;

                let name_str = name_sym.to_string();
                let linkage = if name_str == "main" {
                    Linkage::Export
                } else {
                    Linkage::Local
                };

                let func_id = obj_module
                    .declare_function(&name_str, linkage, &sig)
                    .map_err(|e| CompilationError::codegen(format!("{e}")))?;
                func_ids.insert(name_sym, func_id);
            }
        }
    }

    // 4. Second pass — define functions
    let mut fb_ctx = FunctionBuilderContext::new();

    for block in body.blocks(db).iter() {
        for op in block.operations(db).iter() {
            if let Ok(func_op) = clif::Func::from_operation(db, *op) {
                let name_sym = func_op.sym_name(db);
                let func_type_attr = func_op.r#type(db);
                let func_ty = core::Func::from_type(db, func_type_attr).ok_or_else(|| {
                    CompilationError::type_error("expected core.func type on clif.func")
                })?;

                let sig = translate_signature(db, func_ty, call_conv)?;
                let func_id = func_ids[&name_sym];

                let mut cl_func = cl_ir::Function::with_name_signature(
                    UserFuncName::user(0, func_id.as_u32()),
                    sig,
                );

                // Declare all known functions as FuncRefs within this function
                let mut func_refs: HashMap<Symbol, cl_ir::FuncRef> = HashMap::new();
                for (&sym, &fid) in &func_ids {
                    let fref = obj_module.declare_func_in_func(fid, &mut cl_func);
                    func_refs.insert(sym, fref);
                }

                // Build the function body
                {
                    let builder = FunctionBuilder::new(&mut cl_func, &mut fb_ctx);
                    let mut translator = FunctionTranslator::new(db, builder, &func_refs);

                    let entry_block = translator.builder.create_block();
                    translator
                        .builder
                        .append_block_params_for_function_params(entry_block);
                    translator.builder.switch_to_block(entry_block);

                    // Map TrunkIR block arguments (function params) to Cranelift block params
                    let func_body = func_op.body(db);
                    if let Some(ir_entry_block) = func_body.blocks(db).first() {
                        let cl_params: Vec<_> =
                            translator.builder.block_params(entry_block).to_vec();
                        for (i, &cl_param) in cl_params.iter().enumerate() {
                            let ir_arg = ir_entry_block.arg(db, i);
                            translator.values.insert(ir_arg, cl_param);
                        }

                        // Translate each operation in the entry block
                        for ir_op in ir_entry_block.operations(db).iter() {
                            translator.translate_op(ir_op)?;
                        }
                    }

                    translator.builder.seal_all_blocks();
                    translator.builder.finalize();
                }

                // Compile the function via Cranelift
                let mut ctx = Context::for_function(cl_func);
                obj_module
                    .define_function(func_id, &mut ctx)
                    .map_err(|e| CompilationError::codegen(format!("{e}")))?;
            }
        }
    }

    // 5. Emit object file
    let product = obj_module.finish();
    let bytes = product
        .emit()
        .map_err(|e| CompilationError::codegen(format!("{e}")))?;

    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use trunk_ir::{
        Block, BlockArg, BlockId, DialectType, IdVec, Location, PathId, Region, Span, idvec,
    };

    fn test_location(db: &dyn salsa::Database) -> Location<'_> {
        let path = PathId::new(db, "file:///test.trb".to_owned());
        Location::new(path, Span::new(0, 0))
    }

    /// Helper: build a `core.module` containing the given `clif.func` operations.
    fn build_module<'db>(
        db: &'db dyn salsa::Database,
        location: Location<'db>,
        func_ops: Vec<trunk_ir::Operation<'db>>,
    ) -> core::Module<'db> {
        let ops: IdVec<trunk_ir::Operation<'db>> = func_ops.into_iter().collect();
        let module_block = Block::new(db, BlockId::fresh(), location, idvec![], ops);
        let module_body = Region::new(db, location, idvec![module_block]);
        core::Module::create(db, location, "test".into(), module_body)
    }

    // --- iconst + return ---

    #[salsa::tracked]
    fn make_iconst_return_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let location = test_location(db);
        let i64_ty = core::I64::new(db).as_type();

        // fn main() -> I64 { iconst 42; return }
        let func_ty = core::Func::new(db, IdVec::new(), i64_ty).as_type();
        let iconst_op = clif::iconst(db, location, i64_ty, 42);
        let ret_op = clif::r#return(db, location, [iconst_op.result(db)]);

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![iconst_op.as_operation(), ret_op.as_operation()],
        );
        let body = Region::new(db, location, idvec![block]);
        let func_op = clif::func(db, location, Symbol::new("main"), func_ty, body);

        build_module(db, location, vec![func_op.as_operation()])
    }

    #[salsa_test]
    fn test_emit_iconst_return(db: &salsa::DatabaseImpl) {
        let module = make_iconst_return_module(db);
        let result = emit_module_to_native(db, module);
        assert!(result.is_ok(), "emit failed: {:?}", result.err());
        let bytes = result.unwrap();
        assert!(!bytes.is_empty(), "object file should not be empty");
    }

    // --- arithmetic: iconst + iadd + return ---

    #[salsa::tracked]
    fn make_arithmetic_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let location = test_location(db);
        let i64_ty = core::I64::new(db).as_type();

        // fn main() -> I64 { let a = 10; let b = 20; return a + b; }
        let func_ty = core::Func::new(db, IdVec::new(), i64_ty).as_type();

        let a = clif::iconst(db, location, i64_ty, 10);
        let b = clif::iconst(db, location, i64_ty, 20);
        // Constructor order: operands, result_ty (no attributes)
        let sum = clif::iadd(db, location, a.result(db), b.result(db), i64_ty);
        let ret = clif::r#return(db, location, [sum.result(db)]);

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![
                a.as_operation(),
                b.as_operation(),
                sum.as_operation(),
                ret.as_operation()
            ],
        );
        let body = Region::new(db, location, idvec![block]);
        let func_op = clif::func(db, location, Symbol::new("main"), func_ty, body);

        build_module(db, location, vec![func_op.as_operation()])
    }

    #[salsa_test]
    fn test_emit_arithmetic(db: &salsa::DatabaseImpl) {
        let module = make_arithmetic_module(db);
        let result = emit_module_to_native(db, module);
        assert!(result.is_ok(), "emit failed: {:?}", result.err());
        assert!(!result.unwrap().is_empty());
    }

    // --- function call ---

    #[salsa::tracked]
    fn make_function_call_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let location = test_location(db);
        let i64_ty = core::I64::new(db).as_type();

        // fn add(a: I64, b: I64) -> I64 { return a + b; }
        let add_func_ty = core::Func::new(db, IdVec::from(vec![i64_ty, i64_ty]), i64_ty).as_type();

        let add_block_id = BlockId::fresh();
        let arg_a_ty = BlockArg::of_type(db, i64_ty);
        let arg_b_ty = BlockArg::of_type(db, i64_ty);
        // Create block with args first to get Value references
        let add_block = Block::new(
            db,
            add_block_id,
            location,
            idvec![arg_a_ty, arg_b_ty],
            idvec![],
        );
        let arg_a = add_block.arg(db, 0);
        let arg_b = add_block.arg(db, 1);
        // Constructor order: operands (lhs, rhs), result_ty
        let sum = clif::iadd(db, location, arg_a, arg_b, i64_ty);
        let ret = clif::r#return(db, location, [sum.result(db)]);
        // Rebuild block with operations (same block id preserves arg Values)
        let add_block = Block::new(
            db,
            add_block_id,
            location,
            idvec![arg_a_ty, arg_b_ty],
            idvec![sum.as_operation(), ret.as_operation()],
        );
        let add_body = Region::new(db, location, idvec![add_block]);
        let add_func =
            clif::func(db, location, Symbol::new("add"), add_func_ty, add_body).as_operation();

        // fn main() -> I64 { return add(10, 20); }
        let main_func_ty = core::Func::new(db, IdVec::new(), i64_ty).as_type();
        let c10 = clif::iconst(db, location, i64_ty, 10);
        let c20 = clif::iconst(db, location, i64_ty, 20);
        // Constructor order: variadic operands, result_ty, callee attribute
        let call = clif::call(
            db,
            location,
            [c10.result(db), c20.result(db)],
            i64_ty,
            Symbol::new("add"),
        );
        let main_ret = clif::r#return(db, location, [call.result(db)]);
        let main_block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![
                c10.as_operation(),
                c20.as_operation(),
                call.as_operation(),
                main_ret.as_operation()
            ],
        );
        let main_body = Region::new(db, location, idvec![main_block]);
        let main_func =
            clif::func(db, location, Symbol::new("main"), main_func_ty, main_body).as_operation();

        build_module(db, location, vec![add_func, main_func])
    }

    #[salsa_test]
    fn test_emit_function_call(db: &salsa::DatabaseImpl) {
        let module = make_function_call_module(db);
        let result = emit_module_to_native(db, module);
        assert!(result.is_ok(), "emit failed: {:?}", result.err());
        assert!(!result.unwrap().is_empty());
    }

    // --- validation: reject non-clif ops ---

    #[salsa::tracked]
    fn make_invalid_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let location = test_location(db);
        let i64_ty = core::I64::new(db).as_type();

        let const_op =
            trunk_ir::dialect::func::constant(db, location, i64_ty, Symbol::new("some_func"))
                .as_operation();

        let block = Block::new(db, BlockId::fresh(), location, idvec![], idvec![const_op]);
        let region = Region::new(db, location, idvec![block]);
        core::Module::create(db, location, "test".into(), region)
    }

    #[salsa_test]
    fn test_emit_rejects_non_clif_module(db: &salsa::DatabaseImpl) {
        let module = make_invalid_module(db);
        let result = emit_module_to_native(db, module);
        assert!(result.is_err());
    }
}
