//! Cranelift native object file translation.
//!
//! This module provides functions for validating and emitting native object files
//! from TrunkIR modules that have already been lowered to the clif dialect.

use std::collections::HashMap;

use cranelift_codegen::ir::types as cl_types;
use cranelift_codegen::ir::{self as cl_ir, InstBuilder, UserFuncName};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_codegen::{Context, isa};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_module::{
    DataDescription, FuncId, Linkage, Module as CraneliftModule, default_libcall_names,
};
use cranelift_object::{ObjectBuilder, ObjectModule};
use target_lexicon::Triple;
use trunk_ir::dialect::{clif, core};
use trunk_ir::{DialectOp, DialectType, Region, Symbol};

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

    // 3. First pass — declare all functions (including nested core.module)
    let mut func_ids: HashMap<Symbol, cranelift_module::FuncId> = HashMap::new();
    let body = module.body(db);
    let all_func_ops = collect_clif_funcs(db, body);
    for func_op in &all_func_ops {
        let name_sym = func_op.sym_name(db);
        let func_type_attr = func_op.r#type(db);
        let func_ty = core::Func::from_type(db, func_type_attr)
            .ok_or_else(|| CompilationError::type_error("expected core.func type on clif.func"))?;

        let sig = translate_signature(db, func_ty, call_conv)?;

        // Determine linkage based on function attributes:
        // - "main" is exported (entry point)
        // - Functions with an "abi" attribute (e.g., extern "C") are imports
        // - Everything else is local
        let has_abi = func_op
            .as_operation()
            .attributes(db)
            .get(&Symbol::new("abi"))
            .is_some();
        let linkage = if name_sym == "main" {
            Linkage::Export
        } else if has_abi {
            Linkage::Import
        } else {
            Linkage::Local
        };

        // Mangle local function names for linker compatibility.
        // Export/Import names are kept as-is (ABI boundary).
        let linker_name = if linkage == Linkage::Local {
            name_sym.with_str(mangle_native_name)
        } else {
            name_sym.to_string()
        };

        let func_id = obj_module
            .declare_function(&linker_name, linkage, &sig)
            .map_err(|e| CompilationError::codegen(format!("{e}")))?;
        func_ids.insert(name_sym, func_id);
    }

    // 3b. Declare runtime functions (imported, provided by linker)
    declare_runtime_functions(db, &mut obj_module, &mut func_ids, call_conv)?;

    // 3c. Collect release functions and declare RTTI infrastructure
    let rtti_info = collect_and_declare_rtti(&mut obj_module, &mut func_ids, call_conv)?;

    // 4. Second pass — define functions (skip extern/imported functions)
    let mut fb_ctx = FunctionBuilderContext::new();

    for func_op in &all_func_ops {
        // Skip extern functions (abi attribute = imported, no body to compile)
        let has_abi = func_op
            .as_operation()
            .attributes(db)
            .get(&Symbol::new("abi"))
            .is_some();
        if has_abi {
            continue;
        }

        let name_sym = func_op.sym_name(db);
        let func_type_attr = func_op.r#type(db);
        let func_ty = core::Func::from_type(db, func_type_attr)
            .ok_or_else(|| CompilationError::type_error("expected core.func type on clif.func"))?;

        let sig = translate_signature(db, func_ty, call_conv)?;
        let func_id = func_ids[&name_sym];

        let mut cl_func =
            cl_ir::Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

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

            let func_body = func_op.body(db);
            let ir_blocks = func_body.blocks(db);

            // Phase 1: Create all Cranelift blocks upfront
            let entry_block = translator.builder.create_block();
            translator
                .builder
                .append_block_params_for_function_params(entry_block);

            if let Some(ir_entry_block) = ir_blocks.first() {
                translator
                    .block_map
                    .insert(ir_entry_block.id(db), entry_block);
            }

            // Create Cranelift blocks for non-entry IR blocks
            for ir_block in ir_blocks.iter().skip(1) {
                let cl_block = translator.builder.create_block();
                // Add block parameters matching TrunkIR block args
                for arg in ir_block.args(db).iter() {
                    let cl_ty = crate::function::translate_type(db, arg.ty(db))?;
                    translator.builder.append_block_param(cl_block, cl_ty);
                }
                translator.block_map.insert(ir_block.id(db), cl_block);
            }

            // Phase 2: Translate operations in each block
            for ir_block in ir_blocks.iter() {
                let cl_block = translator.lookup_block(*ir_block)?;
                translator.builder.switch_to_block(cl_block);

                // Map block arguments to Cranelift block params
                let cl_params: Vec<_> = translator.builder.block_params(cl_block).to_vec();
                let ir_arg_count = ir_block.args(db).len();
                if ir_arg_count != cl_params.len() {
                    return Err(crate::CompilationError::codegen(format!(
                        "block arg count mismatch: TrunkIR block has {} args but Cranelift block has {} params",
                        ir_arg_count,
                        cl_params.len(),
                    )));
                }
                for (i, &cl_param) in cl_params.iter().enumerate() {
                    let ir_arg = ir_block.arg(db, i);
                    translator.values.insert(ir_arg, cl_param);
                }

                // Translate each operation in this block
                for ir_op in ir_block.operations(db).iter() {
                    translator.translate_op(ir_op).map_err(|e| {
                        CompilationError::codegen(format!(
                            "{e} (in {}.{} of function {})",
                            ir_op.dialect(db),
                            ir_op.name(db),
                            name_sym,
                        ))
                    })?;
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

    // 4b. Define RTTI table and __tribute_deep_release
    define_rtti_infrastructure(&mut obj_module, &func_ids, &rtti_info, call_conv)?;

    // 5. Emit object file
    let product = obj_module.finish();
    let bytes = product
        .emit()
        .map_err(|e| CompilationError::codegen(format!("{e}")))?;

    Ok(bytes)
}

/// Mangle a TrunkIR symbol name for native linking.
///
/// Produces a fixed-length hash format inspired by Rust's name mangling:
/// `_trb_<last_component>_h<hash>`. The last path component is preserved
/// for debuggability, and a 16-hex-char hash of the full name ensures
/// uniqueness with predictable length.
fn mangle_native_name(name: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    name.hash(&mut hasher);
    let hash = hasher.finish();

    let last = name.rsplit("::").next().unwrap_or(name);
    format!("_trb_{}_h{:016x}", last, hash)
}

/// Recursively collect all `clif.func` operations from a region,
/// including those nested inside `core.module` operations.
fn collect_clif_funcs<'db>(
    db: &'db dyn salsa::Database,
    region: Region<'db>,
) -> Vec<clif::Func<'db>> {
    let mut funcs = Vec::new();
    collect_clif_funcs_from_region(db, region, &mut funcs);
    funcs
}

fn collect_clif_funcs_from_region<'db>(
    db: &'db dyn salsa::Database,
    region: Region<'db>,
    funcs: &mut Vec<clif::Func<'db>>,
) {
    for block in region.blocks(db).iter() {
        for op in block.operations(db).iter() {
            if let Ok(func_op) = clif::Func::from_operation(db, *op) {
                funcs.push(func_op);
            } else if core::Module::matches(db, *op) {
                // Recurse into nested core.module operations
                for nested_region in op.regions(db).iter() {
                    collect_clif_funcs_from_region(db, *nested_region, funcs);
                }
            }
        }
    }
}

/// Declare runtime functions that are imported (provided by the linker).
///
/// Currently declares:
/// - `__tribute_alloc(i64) -> ptr`: heap allocation (malloc wrapper)
/// - `__tribute_dealloc(ptr, i64)`: heap deallocation (free wrapper)
///
/// These use `Import` linkage so they can be overridden at link time
/// with custom allocator implementations.
fn declare_runtime_functions(
    _db: &dyn salsa::Database,
    obj_module: &mut ObjectModule,
    func_ids: &mut HashMap<Symbol, cranelift_module::FuncId>,
    call_conv: isa::CallConv,
) -> CompilationResult<()> {
    let ptr_ty = obj_module.target_config().pointer_type();
    let i64_ty = cranelift_codegen::ir::types::I64;

    // __tribute_alloc(size: i64) -> ptr
    let mut alloc_sig = cl_ir::Signature::new(call_conv);
    alloc_sig.params.push(cl_ir::AbiParam::new(i64_ty));
    alloc_sig.returns.push(cl_ir::AbiParam::new(ptr_ty));

    let alloc_sym = Symbol::new("__tribute_alloc");
    if let std::collections::hash_map::Entry::Vacant(e) = func_ids.entry(alloc_sym) {
        let func_id = obj_module
            .declare_function("__tribute_alloc", Linkage::Import, &alloc_sig)
            .map_err(|e| CompilationError::codegen(format!("{e}")))?;
        e.insert(func_id);
    }

    // __tribute_dealloc(ptr: ptr, size: i64)
    let mut dealloc_sig = cl_ir::Signature::new(call_conv);
    dealloc_sig.params.push(cl_ir::AbiParam::new(ptr_ty)); // ptr
    dealloc_sig.params.push(cl_ir::AbiParam::new(i64_ty)); // size
    // no return value

    let dealloc_sym = Symbol::new("__tribute_dealloc");
    if let std::collections::hash_map::Entry::Vacant(e) = func_ids.entry(dealloc_sym) {
        let func_id = obj_module
            .declare_function("__tribute_dealloc", Linkage::Import, &dealloc_sig)
            .map_err(|e| CompilationError::codegen(format!("{e}")))?;
        e.insert(func_id);
    }

    Ok(())
}

/// Information collected about RTTI release functions for code emission.
struct RttiEmitInfo {
    /// FuncId for `__tribute_deep_release` (always present).
    deep_release_func_id: FuncId,
    /// DataId for the RTTI function pointer table (None if no release functions exist).
    rtti_table_data_id: Option<cranelift_module::DataId>,
    /// Mapping from rtti_idx to the corresponding release function's FuncId.
    release_functions: HashMap<u32, FuncId>,
    /// Maximum rtti_idx seen (determines table size). 0 if no release functions.
    max_rtti_idx: u32,
}

/// Prefix for per-type release functions generated by the RTTI pass.
const RELEASE_FN_PREFIX: &str = "__tribute_release_";

/// Collect release functions from func_ids and declare RTTI infrastructure.
///
/// Scans `func_ids` for functions matching `__tribute_release_<N>` pattern,
/// then declares `__tribute_deep_release` and the RTTI table.
fn collect_and_declare_rtti(
    obj_module: &mut ObjectModule,
    func_ids: &mut HashMap<Symbol, FuncId>,
    call_conv: cranelift_codegen::isa::CallConv,
) -> CompilationResult<RttiEmitInfo> {
    let ptr_ty = obj_module.target_config().pointer_type();

    // Scan func_ids for __tribute_release_<N> functions
    let mut release_functions: HashMap<u32, FuncId> = HashMap::new();
    let mut max_rtti_idx: u32 = 0;

    for (sym, &func_id) in func_ids.iter() {
        let name = sym.to_string();
        if let Some(suffix) = name.strip_prefix(RELEASE_FN_PREFIX)
            && let Ok(idx) = suffix.parse::<u32>()
        {
            release_functions.insert(idx, func_id);
            max_rtti_idx = max_rtti_idx.max(idx);
        }
    }

    // Declare __tribute_deep_release(ptr, size) -> void
    let mut deep_release_sig = cl_ir::Signature::new(call_conv);
    deep_release_sig.params.push(cl_ir::AbiParam::new(ptr_ty));
    deep_release_sig
        .params
        .push(cl_ir::AbiParam::new(cl_types::I64));
    // no return value

    let deep_release_sym = Symbol::new("__tribute_deep_release");
    let deep_release_func_id = obj_module
        .declare_function("__tribute_deep_release", Linkage::Export, &deep_release_sig)
        .map_err(|e| CompilationError::codegen(format!("{e}")))?;
    func_ids.insert(deep_release_sym, deep_release_func_id);

    // Declare RTTI table if we have release functions
    let rtti_table_data_id = if !release_functions.is_empty() {
        let data_id = obj_module
            .declare_data("__tribute_rtti_table", Linkage::Local, false, false)
            .map_err(|e| CompilationError::codegen(format!("{e}")))?;
        Some(data_id)
    } else {
        None
    };

    Ok(RttiEmitInfo {
        deep_release_func_id,
        rtti_table_data_id,
        release_functions,
        max_rtti_idx,
    })
}

/// Define the RTTI function pointer table and the `__tribute_deep_release` dispatch function.
fn define_rtti_infrastructure(
    obj_module: &mut ObjectModule,
    func_ids: &HashMap<Symbol, FuncId>,
    rtti_info: &RttiEmitInfo,
    call_conv: cranelift_codegen::isa::CallConv,
) -> CompilationResult<()> {
    let ptr_ty = obj_module.target_config().pointer_type();
    let ptr_size = ptr_ty.bytes();

    // Define RTTI table if it exists
    if let Some(data_id) = rtti_info.rtti_table_data_id {
        let table_size = (rtti_info.max_rtti_idx + 1) as usize;
        let mut data_desc = DataDescription::new();
        // Use define() with zero-filled buffer instead of define_zeroinit() to ensure
        // the table is placed in a data section (not BSS), since we add relocations
        // for function pointers below. BSS + relocations causes linker crashes on macOS.
        let table_bytes = vec![0u8; table_size * ptr_size as usize];
        data_desc.define(table_bytes.into_boxed_slice());
        data_desc.set_align(ptr_size as u64);

        for (&rtti_idx, &release_func_id) in &rtti_info.release_functions {
            let fref = obj_module.declare_func_in_data(release_func_id, &mut data_desc);
            data_desc.write_function_addr(rtti_idx * ptr_size, fref);
        }

        obj_module
            .define_data(data_id, &data_desc)
            .map_err(|e| CompilationError::codegen(format!("{e}")))?;
    }

    // Define __tribute_deep_release(ptr, size)
    let mut deep_release_sig = cl_ir::Signature::new(call_conv);
    deep_release_sig.params.push(cl_ir::AbiParam::new(ptr_ty));
    deep_release_sig
        .params
        .push(cl_ir::AbiParam::new(cl_types::I64));

    let mut cl_func = cl_ir::Function::with_name_signature(
        UserFuncName::user(0, rtti_info.deep_release_func_id.as_u32()),
        deep_release_sig,
    );

    // Look up __tribute_dealloc FuncId
    let dealloc_sym = Symbol::new("__tribute_dealloc");
    let dealloc_func_id = func_ids[&dealloc_sym];
    let dealloc_fref = obj_module.declare_func_in_func(dealloc_func_id, &mut cl_func);

    let mut fb_ctx = FunctionBuilderContext::new();
    {
        let mut builder = FunctionBuilder::new(&mut cl_func, &mut fb_ctx);

        if rtti_info.release_functions.is_empty() {
            // No release functions: always do shallow dealloc
            build_shallow_only_deep_release(&mut builder, ptr_ty, dealloc_fref);
        } else {
            // Has release functions: dispatch via RTTI table
            let rtti_table_data_id = rtti_info.rtti_table_data_id.unwrap();
            let rtti_table_gv = obj_module.declare_data_in_func(rtti_table_data_id, builder.func);

            // Declare release function signature for call_indirect
            let mut release_sig = cl_ir::Signature::new(call_conv);
            release_sig.params.push(cl_ir::AbiParam::new(ptr_ty));
            let release_sig_ref = builder.import_signature(release_sig);

            build_dispatching_deep_release(
                &mut builder,
                ptr_ty,
                dealloc_fref,
                rtti_table_gv,
                release_sig_ref,
            );
        }

        builder.seal_all_blocks();
        builder.finalize();
    }

    let mut ctx = Context::for_function(cl_func);
    obj_module
        .define_function(rtti_info.deep_release_func_id, &mut ctx)
        .map_err(|e| CompilationError::codegen(format!("{e}")))?;

    Ok(())
}

/// Build `__tribute_deep_release` body when there are no release functions.
///
/// Simply computes raw_ptr and calls `__tribute_dealloc(raw_ptr, alloc_size)`.
fn build_shallow_only_deep_release(
    builder: &mut FunctionBuilder,
    ptr_ty: cl_types::Type,
    dealloc_fref: cl_ir::FuncRef,
) {
    let entry = builder.create_block();
    builder.append_block_params_for_function_params(entry);
    builder.switch_to_block(entry);

    let payload_ptr = builder.block_params(entry)[0];
    let alloc_size = builder.block_params(entry)[1];

    // raw_ptr = payload_ptr - 8
    let neg8 = builder.ins().iconst(ptr_ty, -8);
    let raw_ptr = builder.ins().iadd(payload_ptr, neg8);

    // __tribute_dealloc(raw_ptr, alloc_size)
    builder.ins().call(dealloc_fref, &[raw_ptr, alloc_size]);

    builder.ins().return_(&[]);
}

/// Build `__tribute_deep_release` body with RTTI table dispatch.
///
/// ```text
/// entry(payload_ptr, alloc_size):
///   raw_ptr = payload_ptr - 8
///   rtti_idx = load i32 from raw_ptr+4
///   entry_offset = rtti_idx * ptr_size
///   release_fn = load ptr from rtti_table[entry_offset]
///   if release_fn == null: goto shallow
///   else: goto deep
///
/// shallow:
///   __tribute_dealloc(raw_ptr, alloc_size)
///   return
///
/// deep:
///   call_indirect release_fn(payload_ptr)
///   return
/// ```
fn build_dispatching_deep_release(
    builder: &mut FunctionBuilder,
    ptr_ty: cl_types::Type,
    dealloc_fref: cl_ir::FuncRef,
    rtti_table_gv: cl_ir::GlobalValue,
    release_sig_ref: cl_ir::SigRef,
) {
    let entry = builder.create_block();
    let shallow_block = builder.create_block();
    let deep_block = builder.create_block();

    builder.append_block_params_for_function_params(entry);
    builder.switch_to_block(entry);

    let payload_ptr = builder.block_params(entry)[0];
    let alloc_size = builder.block_params(entry)[1];

    // raw_ptr = payload_ptr - 8
    let neg8 = builder.ins().iconst(ptr_ty, -8);
    let raw_ptr = builder.ins().iadd(payload_ptr, neg8);

    // rtti_idx = load i32 from raw_ptr + 4
    let rtti_idx = builder
        .ins()
        .load(cl_types::I32, cl_ir::MemFlags::trusted(), raw_ptr, 4);

    // entry_offset = uextend(rtti_idx) * ptr_size
    let rtti_idx_64 = builder.ins().uextend(ptr_ty, rtti_idx);
    let ptr_size_val = builder.ins().iconst(ptr_ty, ptr_ty.bytes() as i64);
    let entry_offset = builder.ins().imul(rtti_idx_64, ptr_size_val);

    // table_base = global_value for rtti_table
    let table_base = builder.ins().global_value(ptr_ty, rtti_table_gv);

    // entry_addr = table_base + entry_offset
    let entry_addr = builder.ins().iadd(table_base, entry_offset);

    // release_fn = load ptr from entry_addr
    let release_fn = builder
        .ins()
        .load(ptr_ty, cl_ir::MemFlags::trusted(), entry_addr, 0);

    // if release_fn == null: shallow, else: deep
    let null = builder.ins().iconst(ptr_ty, 0);
    let is_null = builder
        .ins()
        .icmp(cl_ir::condcodes::IntCC::Equal, release_fn, null);
    builder
        .ins()
        .brif(is_null, shallow_block, &[], deep_block, &[]);

    // shallow_block: __tribute_dealloc(raw_ptr, alloc_size); return
    builder.switch_to_block(shallow_block);
    builder.ins().call(dealloc_fref, &[raw_ptr, alloc_size]);
    builder.ins().return_(&[]);

    // deep_block: call_indirect release_fn(payload_ptr); return
    builder.switch_to_block(deep_block);
    builder
        .ins()
        .call_indirect(release_sig_ref, release_fn, &[payload_ptr]);
    builder.ins().return_(&[]);
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

    // --- multi-block: brif + jump ---

    #[salsa::tracked]
    fn make_brif_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let location = test_location(db);
        let i64_ty = core::I64::new(db).as_type();
        let i8_ty = core::I8::new(db).as_type();

        // fn main(cond: I8) -> I64 {
        //   entry:
        //     brif cond, then_block, else_block
        //   then_block:
        //     return 1
        //   else_block:
        //     return 0
        // }
        let func_ty = core::Func::new(db, IdVec::from(vec![i8_ty]), i64_ty).as_type();

        let entry_block_id = BlockId::fresh();
        let then_block_id = BlockId::fresh();
        let else_block_id = BlockId::fresh();

        // Create then_block: return 1
        let c1 = clif::iconst(db, location, i64_ty, 1);
        let ret1 = clif::r#return(db, location, [c1.result(db)]);
        let then_block = Block::new(
            db,
            then_block_id,
            location,
            idvec![],
            idvec![c1.as_operation(), ret1.as_operation()],
        );

        // Create else_block: return 0
        let c0 = clif::iconst(db, location, i64_ty, 0);
        let ret0 = clif::r#return(db, location, [c0.result(db)]);
        let else_block = Block::new(
            db,
            else_block_id,
            location,
            idvec![],
            idvec![c0.as_operation(), ret0.as_operation()],
        );

        // Create entry block with cond param and brif
        let entry_cond_arg = BlockArg::of_type(db, i8_ty);
        let entry_block_tmp = Block::new(
            db,
            entry_block_id,
            location,
            idvec![entry_cond_arg],
            idvec![],
        );
        let cond_val = entry_block_tmp.arg(db, 0);
        let brif_op = clif::brif(db, location, cond_val, then_block, else_block);
        let entry_block = Block::new(
            db,
            entry_block_id,
            location,
            idvec![entry_cond_arg],
            idvec![brif_op.as_operation()],
        );

        let body = Region::new(db, location, idvec![entry_block, then_block, else_block]);
        let func_op = clif::func(db, location, Symbol::new("main"), func_ty, body);

        build_module(db, location, vec![func_op.as_operation()])
    }

    #[salsa_test]
    fn test_emit_brif_multi_block(db: &salsa::DatabaseImpl) {
        let module = make_brif_module(db);
        let result = emit_module_to_native(db, module);
        assert!(result.is_ok(), "emit failed: {:?}", result.err());
        let bytes = result.unwrap();
        assert!(!bytes.is_empty(), "object file should not be empty");
    }

    #[salsa::tracked]
    fn make_jump_with_args_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let location = test_location(db);
        let i64_ty = core::I64::new(db).as_type();

        // fn main() -> I64 {
        //   entry:
        //     %c = iconst 42
        //     jump merge_block(%c)
        //   merge_block(%val: I64):
        //     return %val
        // }
        let func_ty = core::Func::new(db, IdVec::new(), i64_ty).as_type();

        let entry_block_id = BlockId::fresh();
        let merge_block_id = BlockId::fresh();

        // Create merge_block with arg: return %val
        let merge_arg = BlockArg::of_type(db, i64_ty);
        let merge_block_tmp = Block::new(db, merge_block_id, location, idvec![merge_arg], idvec![]);
        let merge_val = merge_block_tmp.arg(db, 0);
        let ret = clif::r#return(db, location, [merge_val]);
        let merge_block = Block::new(
            db,
            merge_block_id,
            location,
            idvec![merge_arg],
            idvec![ret.as_operation()],
        );

        // Create entry block: iconst 42, jump merge_block(42)
        let c42 = clif::iconst(db, location, i64_ty, 42);
        let jump_op = clif::jump(db, location, [c42.result(db)], merge_block);
        let entry_block = Block::new(
            db,
            entry_block_id,
            location,
            idvec![],
            idvec![c42.as_operation(), jump_op.as_operation()],
        );

        let body = Region::new(db, location, idvec![entry_block, merge_block]);
        let func_op = clif::func(db, location, Symbol::new("main"), func_ty, body);

        build_module(db, location, vec![func_op.as_operation()])
    }

    #[salsa_test]
    fn test_emit_jump_with_block_args(db: &salsa::DatabaseImpl) {
        let module = make_jump_with_args_module(db);
        let result = emit_module_to_native(db, module);
        assert!(result.is_ok(), "emit failed: {:?}", result.err());
        let bytes = result.unwrap();
        assert!(!bytes.is_empty(), "object file should not be empty");
    }

    // --- validation: jump arg count mismatch ---

    #[salsa::tracked]
    fn make_jump_too_many_args_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let location = test_location(db);
        let i64_ty = core::I64::new(db).as_type();

        // fn main() -> I64 {
        //   entry:
        //     %c = iconst 42
        //     jump dest(%c)       ← 1 arg
        //   dest:                  ← 0 params  (mismatch!)
        //     return iconst 0
        // }
        let func_ty = core::Func::new(db, IdVec::new(), i64_ty).as_type();

        let entry_block_id = BlockId::fresh();
        let dest_block_id = BlockId::fresh();

        // dest block: no params, just return 0
        let c0 = clif::iconst(db, location, i64_ty, 0);
        let ret0 = clif::r#return(db, location, [c0.result(db)]);
        let dest_block = Block::new(
            db,
            dest_block_id,
            location,
            idvec![],
            idvec![c0.as_operation(), ret0.as_operation()],
        );

        // entry block: jump to dest with 1 arg (dest expects 0)
        let c42 = clif::iconst(db, location, i64_ty, 42);
        let jump_op = clif::jump(db, location, [c42.result(db)], dest_block);
        let entry_block = Block::new(
            db,
            entry_block_id,
            location,
            idvec![],
            idvec![c42.as_operation(), jump_op.as_operation()],
        );

        let body = Region::new(db, location, idvec![entry_block, dest_block]);
        let func_op = clif::func(db, location, Symbol::new("main"), func_ty, body);

        build_module(db, location, vec![func_op.as_operation()])
    }

    #[salsa_test]
    fn test_emit_rejects_jump_arg_count_mismatch(db: &salsa::DatabaseImpl) {
        let module = make_jump_too_many_args_module(db);
        let result = emit_module_to_native(db, module);
        assert!(result.is_err(), "should fail due to arg count mismatch");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("argument count"),
            "error should mention argument count mismatch, got: {err_msg}"
        );
    }

    #[salsa::tracked]
    fn make_jump_too_few_args_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let location = test_location(db);
        let i64_ty = core::I64::new(db).as_type();

        // fn main() -> I64 {
        //   entry:
        //     jump dest()          ← 0 args
        //   dest(%val: I64):       ← 1 param  (mismatch!)
        //     return %val
        // }
        let func_ty = core::Func::new(db, IdVec::new(), i64_ty).as_type();

        let entry_block_id = BlockId::fresh();
        let dest_block_id = BlockId::fresh();

        // dest block: 1 param
        let dest_arg = BlockArg::of_type(db, i64_ty);
        let dest_block_tmp = Block::new(db, dest_block_id, location, idvec![dest_arg], idvec![]);
        let dest_val = dest_block_tmp.arg(db, 0);
        let ret = clif::r#return(db, location, [dest_val]);
        let dest_block = Block::new(
            db,
            dest_block_id,
            location,
            idvec![dest_arg],
            idvec![ret.as_operation()],
        );

        // entry block: jump with 0 args (dest expects 1)
        let jump_op = clif::jump(db, location, [], dest_block);
        let entry_block = Block::new(
            db,
            entry_block_id,
            location,
            idvec![],
            idvec![jump_op.as_operation()],
        );

        let body = Region::new(db, location, idvec![entry_block, dest_block]);
        let func_op = clif::func(db, location, Symbol::new("main"), func_ty, body);

        build_module(db, location, vec![func_op.as_operation()])
    }

    #[salsa_test]
    fn test_emit_rejects_jump_too_few_args(db: &salsa::DatabaseImpl) {
        let module = make_jump_too_few_args_module(db);
        let result = emit_module_to_native(db, module);
        assert!(result.is_err(), "should fail due to arg count mismatch");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("argument count"),
            "error should mention argument count mismatch, got: {err_msg}"
        );
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
