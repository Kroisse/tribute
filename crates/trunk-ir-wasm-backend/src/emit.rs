//! WebAssembly binary emission from wasm dialect operations.
//!
//! This module converts lowered wasm dialect TrunkIR operations to
//! a WebAssembly binary using the `wasm_encoder` crate.

mod call_indirect_collection;
mod definitions;
mod gc_types_collection;
mod handlers;
mod helpers;
mod value_emission;

use call_indirect_collection::*;
use definitions::*;
use gc_types_collection::*;
use handlers::{
    handle_array_copy, handle_array_get, handle_array_get_s, handle_array_get_u, handle_array_new,
    handle_array_new_default, handle_array_set, handle_block, handle_br, handle_br_if,
    handle_bytes_from_data, handle_call, handle_call_indirect, handle_f32_const, handle_f32_load,
    handle_f32_store, handle_f64_const, handle_f64_load, handle_f64_store, handle_global_get,
    handle_global_set, handle_i32_const, handle_i32_load, handle_i32_load8_s, handle_i32_load8_u,
    handle_i32_load16_s, handle_i32_load16_u, handle_i32_store, handle_i32_store8,
    handle_i32_store16, handle_i64_const, handle_i64_load, handle_i64_load8_s, handle_i64_load8_u,
    handle_i64_load16_s, handle_i64_load16_u, handle_i64_load32_s, handle_i64_load32_u,
    handle_i64_store, handle_i64_store8, handle_i64_store16, handle_i64_store32, handle_if,
    handle_local_get, handle_local_set, handle_local_tee, handle_loop, handle_memory_grow,
    handle_memory_size, handle_ref_cast, handle_ref_func, handle_ref_null, handle_ref_test,
    handle_return_call, handle_struct_get, handle_struct_new, handle_struct_set,
};
use helpers::*;
use value_emission::*;

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

use tracing::debug;

use trunk_ir::dialect::{core, wasm};
use trunk_ir::{
    Attribute, Attrs, BlockId, DialectOp, DialectType, Operation, Region, Symbol, Type, Value,
};
use wasm_encoder::{
    AbstractHeapType, ArrayType, CodeSection, CompositeInnerType, CompositeType, ConstExpr,
    DataCountSection, DataSection, ElementSection, Elements, EntityType, ExportKind, ExportSection,
    Function, FunctionSection, GlobalSection, GlobalType, HeapType, ImportSection, Instruction,
    MemorySection, MemoryType, Module, RefType, StructType, SubType, TableSection, TableType,
    TypeSection, ValType,
};

use crate::errors;
use crate::gc_types::{ATTR_FIELD_IDX, ATTR_TYPE, GcTypeDef};
#[cfg(test)]
use crate::gc_types::{BYTES_ARRAY_IDX, BYTES_STRUCT_IDX, FIRST_USER_TYPE_IDX};
use crate::{CompilationError, CompilationResult};

trunk_ir::symbols! {
    ATTR_SYM_NAME => "sym_name",
    ATTR_FIELD => "field",
    ATTR_HEAP_TYPE => "heap_type",
    ATTR_TARGET_TYPE => "target_type",
    ATTR_CALLEE => "callee",
    ATTR_VALUE => "value",
    ATTR_INDEX => "index",
    ATTR_TARGET => "target",
    ATTR_MODULE => "module",
    ATTR_NAME => "name",
    ATTR_FUNC => "func",
    ATTR_FUNC_NAME => "func_name",
    ATTR_MIN => "min",
    ATTR_MAX => "max",
    ATTR_SHARED => "shared",
    ATTR_MEMORY64 => "memory64",
    ATTR_OFFSET => "offset",
    ATTR_REFTYPE => "reftype",
    ATTR_TABLE => "table",
    ATTR_PASSIVE => "passive",
    ATTR_DATA_IDX => "data_idx",
    ATTR_LEN => "len",
    ATTR_ALIGN => "align",
    ATTR_MEMORY => "memory",
}

/// Simple wasm operations that follow the pattern:
/// emit operands → single instruction → set result local
static SIMPLE_OPS: LazyLock<HashMap<Symbol, Instruction<'static>>> = LazyLock::new(|| {
    [
        // i32 arithmetic
        ("i32_add", Instruction::I32Add),
        ("i32_sub", Instruction::I32Sub),
        ("i32_mul", Instruction::I32Mul),
        ("i32_div_s", Instruction::I32DivS),
        ("i32_div_u", Instruction::I32DivU),
        ("i32_rem_s", Instruction::I32RemS),
        ("i32_rem_u", Instruction::I32RemU),
        // i32 comparison
        ("i32_eq", Instruction::I32Eq),
        ("i32_ne", Instruction::I32Ne),
        ("i32_lt_s", Instruction::I32LtS),
        ("i32_lt_u", Instruction::I32LtU),
        ("i32_le_s", Instruction::I32LeS),
        ("i32_le_u", Instruction::I32LeU),
        ("i32_gt_s", Instruction::I32GtS),
        ("i32_gt_u", Instruction::I32GtU),
        ("i32_ge_s", Instruction::I32GeS),
        ("i32_ge_u", Instruction::I32GeU),
        // i32 bitwise
        ("i32_and", Instruction::I32And),
        ("i32_or", Instruction::I32Or),
        ("i32_xor", Instruction::I32Xor),
        ("i32_shl", Instruction::I32Shl),
        ("i32_shr_s", Instruction::I32ShrS),
        ("i32_shr_u", Instruction::I32ShrU),
        // i64 arithmetic
        ("i64_add", Instruction::I64Add),
        ("i64_sub", Instruction::I64Sub),
        ("i64_mul", Instruction::I64Mul),
        ("i64_div_s", Instruction::I64DivS),
        ("i64_div_u", Instruction::I64DivU),
        ("i64_rem_s", Instruction::I64RemS),
        ("i64_rem_u", Instruction::I64RemU),
        // i64 comparison
        ("i64_eq", Instruction::I64Eq),
        ("i64_ne", Instruction::I64Ne),
        ("i64_lt_s", Instruction::I64LtS),
        ("i64_lt_u", Instruction::I64LtU),
        ("i64_le_s", Instruction::I64LeS),
        ("i64_le_u", Instruction::I64LeU),
        ("i64_gt_s", Instruction::I64GtS),
        ("i64_gt_u", Instruction::I64GtU),
        ("i64_ge_s", Instruction::I64GeS),
        ("i64_ge_u", Instruction::I64GeU),
        // i64 bitwise
        ("i64_and", Instruction::I64And),
        ("i64_or", Instruction::I64Or),
        ("i64_xor", Instruction::I64Xor),
        ("i64_shl", Instruction::I64Shl),
        ("i64_shr_s", Instruction::I64ShrS),
        ("i64_shr_u", Instruction::I64ShrU),
        // f32 arithmetic
        ("f32_add", Instruction::F32Add),
        ("f32_sub", Instruction::F32Sub),
        ("f32_mul", Instruction::F32Mul),
        ("f32_div", Instruction::F32Div),
        ("f32_neg", Instruction::F32Neg),
        // f32 comparison
        ("f32_eq", Instruction::F32Eq),
        ("f32_ne", Instruction::F32Ne),
        ("f32_lt", Instruction::F32Lt),
        ("f32_le", Instruction::F32Le),
        ("f32_gt", Instruction::F32Gt),
        ("f32_ge", Instruction::F32Ge),
        // f64 arithmetic
        ("f64_add", Instruction::F64Add),
        ("f64_sub", Instruction::F64Sub),
        ("f64_mul", Instruction::F64Mul),
        ("f64_div", Instruction::F64Div),
        ("f64_neg", Instruction::F64Neg),
        // f64 comparison
        ("f64_eq", Instruction::F64Eq),
        ("f64_ne", Instruction::F64Ne),
        ("f64_lt", Instruction::F64Lt),
        ("f64_le", Instruction::F64Le),
        ("f64_gt", Instruction::F64Gt),
        ("f64_ge", Instruction::F64Ge),
        // Integer conversions
        ("i32_wrap_i64", Instruction::I32WrapI64),
        ("i64_extend_i32_s", Instruction::I64ExtendI32S),
        ("i64_extend_i32_u", Instruction::I64ExtendI32U),
        // Float to int conversions
        ("i32_trunc_f32_s", Instruction::I32TruncF32S),
        ("i32_trunc_f32_u", Instruction::I32TruncF32U),
        ("i32_trunc_f64_s", Instruction::I32TruncF64S),
        ("i32_trunc_f64_u", Instruction::I32TruncF64U),
        ("i64_trunc_f32_s", Instruction::I64TruncF32S),
        ("i64_trunc_f32_u", Instruction::I64TruncF32U),
        ("i64_trunc_f64_s", Instruction::I64TruncF64S),
        ("i64_trunc_f64_u", Instruction::I64TruncF64U),
        // Int to float conversions
        ("f32_convert_i32_s", Instruction::F32ConvertI32S),
        ("f32_convert_i32_u", Instruction::F32ConvertI32U),
        ("f32_convert_i64_s", Instruction::F32ConvertI64S),
        ("f32_convert_i64_u", Instruction::F32ConvertI64U),
        ("f64_convert_i32_s", Instruction::F64ConvertI32S),
        ("f64_convert_i32_u", Instruction::F64ConvertI32U),
        ("f64_convert_i64_s", Instruction::F64ConvertI64S),
        ("f64_convert_i64_u", Instruction::F64ConvertI64U),
        // Float to float conversions
        ("f32_demote_f64", Instruction::F32DemoteF64),
        ("f64_promote_f32", Instruction::F64PromoteF32),
        // Reinterpretations
        ("i32_reinterpret_f32", Instruction::I32ReinterpretF32),
        ("i64_reinterpret_f64", Instruction::I64ReinterpretF64),
        ("f32_reinterpret_i32", Instruction::F32ReinterpretI32),
        ("f64_reinterpret_i64", Instruction::F64ReinterpretI64),
        // Misc
        ("drop", Instruction::Drop),
        ("return", Instruction::Return),
        ("unreachable", Instruction::Unreachable),
        ("ref_is_null", Instruction::RefIsNull),
        ("array_len", Instruction::ArrayLen),
        // i31ref (WasmGC fixnum)
        ("ref_i31", Instruction::RefI31),
        ("i31_get_s", Instruction::I31GetS),
        ("i31_get_u", Instruction::I31GetU),
    ]
    .into_iter()
    .map(|(k, v)| (Symbol::new(k), v))
    .collect()
});

#[derive(Default)]
struct ModuleInfo<'db> {
    imports: Vec<ImportFuncDef<'db>>,
    funcs: Vec<FunctionDef<'db>>,
    exports: Vec<ExportDef>,
    memory: Option<MemoryDef>,
    data: Vec<DataDef>,
    tables: Vec<TableDef>,
    elements: Vec<ElementDef>,
    globals: Vec<GlobalDef>,
    gc_types: Vec<GcTypeDef>,
    type_idx_by_type: HashMap<Type<'db>, u32>,
    /// Placeholder struct type_idx lookup (for wasm.structref types)
    placeholder_struct_type_idx: HashMap<(Type<'db>, usize), u32>,
    /// Function type lookup map for boxing/unboxing at call sites.
    func_types: HashMap<Symbol, core::Func<'db>>,
    /// Function index lookup map (import index or func index).
    func_indices: HashMap<Symbol, u32>,
    /// Block argument types map for resolving block argument types.
    block_arg_types: HashMap<(BlockId, usize), Type<'db>>,
    /// Functions referenced via ref.func that need declarative elem segment.
    ref_funcs: HashSet<Symbol>,
    /// Additional function types from call_indirect that need to be added to type section.
    /// Stored as (type_idx, core::Func) pairs.
    call_indirect_types: Vec<(u32, core::Func<'db>)>,
}

/// Context for emitting a single function's code.
struct FunctionEmitContext<'db> {
    /// Maps values to their local indices.
    value_locals: HashMap<Value<'db>, u32>,
    /// Effective types for values (after unification).
    effective_types: HashMap<Value<'db>, Type<'db>>,
    /// The function's expected return type (from function signature).
    /// Used to determine block types for polymorphic operations.
    func_return_type: Option<Type<'db>>,
}

pub fn emit_wasm<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
) -> CompilationResult<Vec<u8>> {
    debug!("emit_wasm: collecting module info...");
    let module_info = match collect_module_info(db, module) {
        Ok(info) => {
            debug!("emit_wasm: module info collected successfully");
            info
        }
        Err(e) => {
            debug!("emit_wasm: collect_module_info failed: {:?}", e);
            return Err(e);
        }
    };

    let mut type_section = TypeSection::new();
    let mut import_section = ImportSection::new();
    let mut function_section = FunctionSection::new();
    let mut table_section = TableSection::new();
    let mut memory_section = MemorySection::new();
    let mut global_section = GlobalSection::new();
    let mut export_section = ExportSection::new();
    let mut element_section = ElementSection::new();
    let mut code_section = CodeSection::new();
    let mut data_section = DataSection::new();

    let gc_type_count = module_info.gc_types.len() as u32;
    let mut next_type_index = gc_type_count;

    // All GC types must be in a single rec group for nominal typing.
    // This makes structurally identical types distinct for ref.test.
    let gc_subtypes: Vec<SubType> = module_info
        .gc_types
        .iter()
        .map(|gc_type| match gc_type {
            GcTypeDef::Struct(fields) => SubType {
                is_final: true,
                supertype_idx: None,
                composite_type: CompositeType {
                    shared: false,
                    inner: CompositeInnerType::Struct(StructType {
                        fields: fields.clone().into_boxed_slice(),
                    }),
                    descriptor: None,
                    describes: None,
                },
            },
            GcTypeDef::Array(field) => SubType {
                is_final: true,
                supertype_idx: None,
                composite_type: CompositeType {
                    shared: false,
                    inner: CompositeInnerType::Array(ArrayType(*field)),
                    descriptor: None,
                    describes: None,
                },
            },
        })
        .collect();
    if !gc_subtypes.is_empty() {
        type_section.ty().rec(gc_subtypes);
    }

    for import_def in module_info.imports.iter() {
        let params = import_def
            .ty
            .params(db)
            .iter()
            .map(|ty| type_to_valtype(db, *ty, &module_info.type_idx_by_type))
            .collect::<CompilationResult<Vec<_>>>()?;
        let results = result_types(db, import_def.ty.result(db), &module_info.type_idx_by_type)?;
        type_section.ty().function(params, results);
        let type_index = next_type_index;
        next_type_index += 1;
        import_section.import(
            &import_def.module.to_string(),
            &import_def.name.to_string(),
            EntityType::Function(type_index),
        );
    }

    for func_def in module_info.funcs.iter() {
        debug!("Processing function type for: {:?}", func_def.name);
        let params = func_def
            .ty
            .params(db)
            .iter()
            .map(|ty| type_to_valtype(db, *ty, &module_info.type_idx_by_type))
            .collect::<CompilationResult<Vec<_>>>()?;

        // Check if the function return type needs to be adjusted.
        //
        // Handler lambdas may be typed to return funcref but the computation lambda
        // actually returns i32. Detect this case and adjust the return type.
        //
        // Key distinction:
        // - Handler arm lambdas (k(value)): done path calls call_indirect → returns funcref
        // - Computation lambda: done path executes body → returns i32
        //
        // NOTE: Functions that perform abilities (can yield) also have type issues,
        // but fixing those at emit time breaks callers. A proper fix is needed in
        // the lowering pass (cont_to_wasm) to update function types and call sites.
        let declared_result = func_def.ty.result(db);
        debug!(
            "  checking return type adjustment for {}: declared={}.{}",
            func_def.name,
            declared_result.dialect(db),
            declared_result.name(db)
        );

        let effective_result = if let Some(body_region) = func_def.op.regions(db).first() {
            // Handler lambdas with funcref return type
            if core::Func::from_type(db, declared_result).is_some()
                || wasm::Funcref::from_type(db, declared_result).is_some()
            {
                debug!("  checking funcref function for handler dispatch...");
                if should_adjust_handler_return_to_i32(db, body_region) {
                    debug!(
                        "  adjusting return type from funcref to i32 for computation lambda: {}",
                        func_def.name
                    );
                    core::I32::new(db).as_type()
                } else {
                    debug!(
                        "  keeping funcref return type for handler arm lambda: {}",
                        func_def.name
                    );
                    declared_result
                }
            } else {
                declared_result
            }
        } else {
            declared_result
        };

        let results = match result_types(db, effective_result, &module_info.type_idx_by_type) {
            Ok(r) => {
                debug!("  results: {:?}", r);
                r
            }
            Err(e) => {
                debug!("Function results conversion failed: {:?}", e);
                return Err(e);
            }
        };
        type_section.ty().function(params, results);
        let type_index = next_type_index;
        next_type_index += 1;
        function_section.function(type_index);
        debug!("  type_index: {}", type_index);
    }

    // Emit call_indirect function types (these were collected during module info gathering)
    for (type_idx, func_ty) in &module_info.call_indirect_types {
        debug!(
            "Emitting call_indirect type idx={}: params={:?}, result={}.{}",
            type_idx,
            func_ty
                .params(db)
                .iter()
                .map(|t| format!("{}.{}", t.dialect(db), t.name(db)))
                .collect::<Vec<_>>(),
            func_ty.result(db).dialect(db),
            func_ty.result(db).name(db)
        );
        let params = func_ty
            .params(db)
            .iter()
            .map(|ty| type_to_valtype(db, *ty, &module_info.type_idx_by_type))
            .collect::<CompilationResult<Vec<_>>>()?;
        let results = result_types(db, func_ty.result(db), &module_info.type_idx_by_type)?;
        type_section.ty().function(params, results);
        // Verify the type index matches what we expect
        assert_eq!(
            *type_idx, next_type_index,
            "call_indirect type index mismatch: expected {}, got {}",
            next_type_index, type_idx
        );
        next_type_index += 1;
    }

    if let Some(memory) = &module_info.memory {
        memory_section.memory(MemoryType {
            minimum: memory.min as u64,
            maximum: memory.max.map(|value| value as u64),
            memory64: memory.memory64,
            shared: memory.shared,
            page_size_log2: None,
        });
    }

    // Generate table section
    for table_def in &module_info.tables {
        table_section.table(TableType {
            element_type: table_def.reftype,
            minimum: table_def.min as u64,
            maximum: table_def.max.map(|v| v as u64),
            table64: false,
            shared: false,
        });
    }

    // Generate global section
    for global_def in &module_info.globals {
        let init_expr = match global_def.valtype {
            ValType::I32 => ConstExpr::i32_const(global_def.init as i32),
            ValType::I64 => ConstExpr::i64_const(global_def.init),
            ValType::F32 => ConstExpr::f32_const(f32::from_bits(global_def.init as u32).into()),
            ValType::F64 => ConstExpr::f64_const(f64::from_bits(global_def.init as u64).into()),
            ValType::Ref(ref_type) => ConstExpr::ref_null(ref_type.heap_type),
            unsupported => {
                return Err(CompilationError::type_error(format!(
                    "unsupported global value type: {:?}",
                    unsupported
                )));
            }
        };
        global_section.global(
            GlobalType {
                val_type: global_def.valtype,
                mutable: global_def.mutable,
                shared: false,
            },
            &init_expr,
        );
    }

    debug!("Processing {} exports...", module_info.exports.len());
    for export in module_info.exports.iter() {
        debug!("  export: {:?} -> {:?}", export.name, export.target);
        match &export.target {
            ExportTarget::Func(sym) => {
                let Some(index) = module_info.func_indices.get(sym) else {
                    debug!("  function not found: {:?}", sym);
                    return Err(CompilationError::function_not_found(&sym.to_string()));
                };
                export_section.export(export.name.as_str(), export.kind, *index);
            }
            ExportTarget::Memory(index) => {
                export_section.export(export.name.as_str(), export.kind, *index);
            }
        }
    }
    if module_info.exports.is_empty() {
        for func_def in module_info.funcs.iter() {
            let Some(index) = module_info.func_indices.get(&func_def.name) else {
                continue;
            };
            // Export with qualified name to avoid collisions (e.g., std::List::map vs std::Option::map)
            let name = func_def.name.to_string();
            export_section.export(&name, ExportKind::Func, *index);
        }
    }

    for data in module_info.data.iter() {
        if data.passive {
            data_section.passive(data.bytes.iter().copied());
        } else {
            let offset = ConstExpr::i32_const(data.offset);
            data_section.active(0, &offset, data.bytes.iter().copied());
        }
    }

    // Generate element section (active element segments)
    for elem_def in &module_info.elements {
        let func_idxs: Vec<u32> = elem_def
            .funcs
            .iter()
            .filter_map(|name| module_info.func_indices.get(name).copied())
            .collect();
        if !func_idxs.is_empty() {
            let offset = ConstExpr::i32_const(elem_def.offset);
            element_section.active(
                Some(elem_def.table),
                &offset,
                Elements::Functions(Cow::Owned(func_idxs)),
            );
        }
    }

    // Generate declarative element segment for functions referenced via ref.func
    // This is required by WebAssembly to declare that these functions can be referenced
    if !module_info.ref_funcs.is_empty() {
        let func_idxs: Vec<u32> = module_info
            .ref_funcs
            .iter()
            .filter_map(|name| module_info.func_indices.get(name).copied())
            .collect();
        if !func_idxs.is_empty() {
            element_section.declared(Elements::Functions(Cow::Owned(func_idxs)));
        }
    }

    debug!(
        "emit_wasm: emitting {} functions...",
        module_info.funcs.len()
    );
    for (i, func_def) in module_info.funcs.iter().enumerate() {
        debug!("emit_wasm: emitting function {}: {:?}", i, func_def.name);
        match emit_function(db, func_def, &module_info) {
            Ok(function) => {
                code_section.function(&function);
            }
            Err(e) => {
                debug!("emit_wasm: emit_function failed: {:?}", e);
                return Err(e);
            }
        }
    }

    let mut module_bytes = Module::new();
    // Section order per WebAssembly spec:
    // Type, Import, Function, Table, Memory, Global, Export, Start, Element, Code, Data
    module_bytes.section(&type_section);
    if !module_info.imports.is_empty() {
        module_bytes.section(&import_section);
    }
    if !module_info.funcs.is_empty() {
        module_bytes.section(&function_section);
    }
    if !module_info.tables.is_empty() {
        module_bytes.section(&table_section);
    }
    if module_info.memory.is_some() {
        module_bytes.section(&memory_section);
    }
    if !module_info.globals.is_empty() {
        module_bytes.section(&global_section);
    }
    module_bytes.section(&export_section);
    if !module_info.elements.is_empty() || !module_info.ref_funcs.is_empty() {
        module_bytes.section(&element_section);
    }
    // Data count section is required when using array.new_data or memory.init
    if !module_info.data.is_empty() {
        module_bytes.section(&DataCountSection {
            count: module_info.data.len() as u32,
        });
    }
    if !module_info.funcs.is_empty() {
        module_bytes.section(&code_section);
    }
    if !module_info.data.is_empty() {
        module_bytes.section(&data_section);
    }

    Ok(module_bytes.finish())
}

/// Recursively collect wasm operations from a region, including nested core.module operations.
/// WebAssembly doesn't support nested modules, so we flatten all functions from namespaced
/// modules into a single list.
fn collect_wasm_ops_from_region<'db>(
    db: &'db dyn salsa::Database,
    region: &Region<'db>,
    info: &mut ModuleInfo<'db>,
) -> CompilationResult<()> {
    let core_dialect = Symbol::new("core");
    let module_name = Symbol::new("module");

    for block in region.blocks(db).iter() {
        for op in block.operations(db).iter() {
            let dialect = op.dialect(db);
            let name = op.name(db);

            // Recurse into nested core.module operations
            if dialect == core_dialect && name == module_name {
                for nested_region in op.regions(db).iter() {
                    collect_wasm_ops_from_region(db, nested_region, info)?;
                }
                continue;
            }

            // Collect wasm operations using typed wrappers
            if let Ok(func_op) = wasm::Func::from_operation(db, *op) {
                if let Ok(func_def) = extract_function_def(db, func_op) {
                    debug!("Including function: {}", func_def.name);
                    info.funcs.push(func_def);
                }
            } else if let Ok(import_op) = wasm::ImportFunc::from_operation(db, *op) {
                info.imports.push(extract_import_def(db, import_op)?);
            } else if let Ok(export_op) = wasm::ExportFunc::from_operation(db, *op) {
                info.exports.push(extract_export_func(db, export_op)?);
            } else if let Ok(export_mem_op) = wasm::ExportMemory::from_operation(db, *op) {
                info.exports.push(extract_export_memory(db, export_mem_op)?);
            } else if let Ok(memory_op) = wasm::Memory::from_operation(db, *op) {
                info.memory = Some(extract_memory_def(db, memory_op)?);
            } else if let Ok(data_op) = wasm::Data::from_operation(db, *op) {
                info.data.push(extract_data_def(db, data_op)?);
            } else if let Ok(table_op) = wasm::Table::from_operation(db, *op) {
                info.tables.push(extract_table_def(db, table_op)?);
            } else if let Ok(elem_op) = wasm::Elem::from_operation(db, *op) {
                info.elements.push(extract_element_def(db, elem_op)?);
            } else if let Ok(global_op) = wasm::Global::from_operation(db, *op) {
                info.globals.push(extract_global_def(db, global_op)?);
            }
        }
    }

    Ok(())
}

/// Collect block argument types from the module.
/// Returns a map from (BlockId, arg_index) to Type.
fn collect_block_arg_types<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
) -> HashMap<(BlockId, usize), Type<'db>> {
    let mut map = HashMap::new();

    fn collect_from_region<'db>(
        db: &'db dyn salsa::Database,
        region: &Region<'db>,
        map: &mut HashMap<(BlockId, usize), Type<'db>>,
    ) {
        for block in region.blocks(db).iter() {
            let block_id = block.id(db);
            for (idx, arg) in block.args(db).iter().enumerate() {
                map.insert((block_id, idx), arg.ty(db));
            }
            // Recursively collect from nested regions in operations
            for op in block.operations(db).iter() {
                for nested_region in op.regions(db).iter() {
                    collect_from_region(db, nested_region, map);
                }
            }
        }
    }

    collect_from_region(db, &module.body(db), &mut map);
    map
}

fn collect_module_info<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
) -> CompilationResult<ModuleInfo<'db>> {
    // Collect block argument types for resolving block argument types
    let mut info = ModuleInfo {
        block_arg_types: collect_block_arg_types(db, module),
        ..Default::default()
    };

    // Recursively collect wasm operations from the module and any nested core.module operations.
    collect_wasm_ops_from_region(db, &module.body(db), &mut info)?;

    // Collect GC types (structs, arrays)
    let (gc_types, mut type_idx_by_type, placeholder_struct_type_idx) =
        collect_gc_types(db, module, &info.block_arg_types)?;
    info.gc_types = gc_types;

    // Collect function types from call_indirect operations
    // Pass the counts so call_indirect types get indices after GC types and function definitions
    let gc_type_count = info.gc_types.len();
    let func_type_count = info.imports.len() + info.funcs.len();
    let call_indirect_types = collect_call_indirect_types(
        db,
        module,
        &mut type_idx_by_type,
        &info.block_arg_types,
        gc_type_count,
        func_type_count,
    )?;

    info.type_idx_by_type = type_idx_by_type;
    info.call_indirect_types = call_indirect_types;
    info.placeholder_struct_type_idx = placeholder_struct_type_idx;

    // Build function type lookup map for boxing/unboxing.
    // Use the qualified name already stored in func/import definitions.
    for func in &info.funcs {
        info.func_types.insert(func.name, func.ty);
    }
    for import in &info.imports {
        info.func_types.insert(import.sym, import.ty);
    }

    // Build function index map (import index or func index).
    for (index, import_def) in info.imports.iter().enumerate() {
        info.func_indices.insert(import_def.sym, index as u32);
    }
    let import_count = info.imports.len() as u32;
    for (index, func_def) in info.funcs.iter().enumerate() {
        info.func_indices
            .insert(func_def.name, import_count + index as u32);
    }

    // Collect functions referenced via ref.func for declarative elem segment
    info.ref_funcs = collect_ref_funcs(db, module);

    // Auto-create a funcref table if call_indirect is used but no table is defined.
    // WebAssembly requires a table for call_indirect to reference.
    if info.tables.is_empty() && has_call_indirect(db, module) {
        debug!("Auto-generating funcref table for call_indirect");
        info.tables.push(TableDef {
            reftype: RefType::FUNCREF,
            min: 0,
            max: None,
        });
    }

    Ok(info)
}

// GcTypesResult and collect_gc_types moved to gc_types_collection module
fn emit_function<'db>(
    db: &'db dyn salsa::Database,
    func_def: &FunctionDef<'db>,
    module_info: &ModuleInfo<'db>,
) -> CompilationResult<Function> {
    debug!("=== emit_function: {:?} ===", func_def.name);
    let region = func_def
        .op
        .regions(db)
        .first()
        .ok_or_else(|| CompilationError::invalid_module("wasm.func missing body region"))?;
    let blocks = region.blocks(db);
    let block = blocks
        .first()
        .ok_or_else(|| CompilationError::invalid_module("wasm.func has no entry block"))?;

    let params = func_def.ty.params(db);
    if params.len() != block.args(db).len() {
        return Err(CompilationError::invalid_module(
            "function parameter count does not match entry block args",
        ));
    }

    // Get the function's expected return type for polymorphic block type inference
    let func_return_type = Some(func_def.ty.result(db));
    let mut ctx = FunctionEmitContext {
        value_locals: HashMap::new(),
        effective_types: HashMap::new(),
        func_return_type,
    };
    let mut locals: Vec<ValType> = Vec::new();

    for (index, arg) in block.args(db).iter().enumerate() {
        ctx.value_locals.insert(block.arg(db, index), index as u32);
        // Block args have their types from the block definition
        ctx.effective_types.insert(block.arg(db, index), arg.ty(db));
    }

    let param_count = params.len() as u32;

    assign_locals_in_region(db, region, param_count, &mut locals, &mut ctx, module_info)?;

    let mut function = Function::new(compress_locals(&locals));

    emit_region_ops(db, region, &ctx, module_info, &mut function)?;

    function.instruction(&Instruction::End);

    Ok(function)
}

fn assign_locals_in_region<'db>(
    db: &'db dyn salsa::Database,
    region: &Region<'db>,
    param_count: u32,
    locals: &mut Vec<ValType>,
    ctx: &mut FunctionEmitContext<'db>,
    module_info: &ModuleInfo<'db>,
) -> CompilationResult<()> {
    for block in region.blocks(db).iter() {
        // Allocate locals for block arguments (for nested regions)
        // In WASM, these represent values that flow through control structures
        for (index, arg) in block.args(db).iter().enumerate() {
            let block_arg = block.arg(db, index);
            // Skip if already assigned (e.g., function entry block args)
            if ctx.value_locals.contains_key(&block_arg) {
                continue;
            }
            let arg_ty = arg.ty(db);
            let val_type = type_to_valtype(db, arg_ty, &module_info.type_idx_by_type)?;
            let local_index = param_count + locals.len() as u32;
            ctx.value_locals.insert(block_arg, local_index);
            ctx.effective_types.insert(block_arg, arg_ty);
            locals.push(val_type);
            tracing::debug!(
                "Allocated local {} for block arg {:?} type {}.{}",
                local_index,
                block.id(db),
                arg_ty.dialect(db),
                arg_ty.name(db)
            );
        }

        for op in block.operations(db).iter() {
            // IMPORTANT: Process nested regions FIRST so that their effective types
            // are available when we compute the effective type for this operation.
            // This is critical for wasm.if which needs to know the branch result types.
            for nested in op.regions(db).iter() {
                assign_locals_in_region(db, nested, param_count, locals, ctx, module_info)?;
            }

            let result_types = op.results(db);
            if result_types.len() > 1 {
                return Err(CompilationError::unsupported_feature("multi-result ops"));
            }
            // Note: We no longer filter out nil types here because they may be
            // used as arguments (e.g., empty closure environments passed as ref.null none).
            // The local allocation handles nil types specially below.
            if let Some(result_ty) = result_types.first().copied() {
                // Use the IR result type directly. Type variables are resolved at AST level
                // before IR generation, so types should already be concrete.
                let effective_ty = result_ty;

                // For wasm.ref_cast and wasm.struct_new with placeholder structref type,
                // use the concrete type from the placeholder map for the local variable type.
                // This ensures struct.get operations can access the correct type.
                let is_ref_cast = wasm::RefCast::matches(db, *op);
                let is_struct_new = wasm::StructNew::matches(db, *op);
                let val_type = if is_ref_cast || is_struct_new {
                    let attrs = op.attributes(db);

                    // For ref_cast: check target_type attr; for struct_new: check type attr or result type
                    let placeholder_ty = if is_ref_cast {
                        attrs.get(&ATTR_TARGET_TYPE()).and_then(|attr| {
                            if let Attribute::Type(ty) = attr {
                                if wasm::Structref::from_type(db, *ty).is_some() {
                                    Some(*ty)
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        })
                    } else if is_struct_new {
                        // Check type attribute or result type for structref
                        attrs
                            .get(&ATTR_TYPE())
                            .and_then(|attr| {
                                if let Attribute::Type(ty) = attr {
                                    if wasm::Structref::from_type(db, *ty).is_some() {
                                        Some(*ty)
                                    } else {
                                        None
                                    }
                                } else {
                                    None
                                }
                            })
                            .or_else(|| {
                                op.results(db).first().copied().and_then(|ty| {
                                    if wasm::Structref::from_type(db, ty).is_some() {
                                        Some(ty)
                                    } else {
                                        None
                                    }
                                })
                            })
                    } else {
                        None
                    };

                    if let Some(placeholder_type) = placeholder_ty {
                        // Get field_count: from attribute for ref_cast, from operands for struct_new
                        let field_count = if is_struct_new {
                            Some(op.operands(db).len())
                        } else {
                            attrs.get(&Symbol::new("field_count")).and_then(|attr| {
                                if let Attribute::IntBits(fc) = attr {
                                    Some(*fc as usize)
                                } else {
                                    None
                                }
                            })
                        };

                        if let Some(fc) = field_count {
                            if let Some(&type_idx) = module_info
                                .placeholder_struct_type_idx
                                .get(&(placeholder_type, fc))
                            {
                                debug!(
                                    "{} local: using concrete type_idx={} for placeholder (field_count={})",
                                    if is_ref_cast {
                                        "ref_cast"
                                    } else {
                                        "struct_new"
                                    },
                                    type_idx,
                                    fc
                                );
                                ValType::Ref(RefType {
                                    nullable: true,
                                    heap_type: HeapType::Concrete(type_idx),
                                })
                            } else {
                                type_to_valtype(db, effective_ty, &module_info.type_idx_by_type)?
                            }
                        } else {
                            type_to_valtype(db, effective_ty, &module_info.type_idx_by_type)?
                        }
                    } else {
                        type_to_valtype(db, effective_ty, &module_info.type_idx_by_type)?
                    }
                } else {
                    match type_to_valtype(db, effective_ty, &module_info.type_idx_by_type) {
                        Ok(vt) => vt,
                        Err(e) => {
                            debug!(
                                "type_to_valtype failed for op {}.{}: {:?}",
                                op.dialect(db),
                                op.name(db),
                                e
                            );
                            return Err(e);
                        }
                    }
                };
                let local_index = param_count + locals.len() as u32;
                let result_value = op.result(db, 0);
                ctx.value_locals.insert(result_value, local_index);
                // Record the effective type for boxing/unboxing decisions
                ctx.effective_types.insert(result_value, effective_ty);
                locals.push(val_type);
            }
        }
    }
    Ok(())
}

fn emit_region_ops<'db>(
    db: &'db dyn salsa::Database,
    region: &Region<'db>,
    ctx: &FunctionEmitContext<'db>,
    module_info: &ModuleInfo<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let blocks = region.blocks(db);
    if blocks.len() != 1 {
        return Err(CompilationError::unsupported_feature("multi-block regions"));
    }
    let block = &blocks[0];
    for op in block.operations(db).iter() {
        // Skip wasm.yield - it's handled by region_result_value + emit_value_get,
        // not emitted as a real Wasm instruction
        if wasm::Yield::from_operation(db, *op).is_ok() {
            continue;
        }
        emit_op(db, op, ctx, module_info, function)?;
    }
    Ok(())
}

fn region_result_value<'db>(
    db: &'db dyn salsa::Database,
    region: &Region<'db>,
) -> Option<Value<'db>> {
    let blocks = region.blocks(db);
    let block = blocks.last()?;
    let op = block.operations(db).last()?;

    // Check for wasm.yield - its operand is the region's result value.
    // This handles cases where the result is defined outside the region
    // (e.g., handler dispatch done body where result is the scrutinee).
    if let Ok(yield_op) = wasm::Yield::from_operation(db, *op) {
        return Some(yield_op.value(db));
    }

    // Fallback: the last operation's first result
    if op.results(db).is_empty() {
        None
    } else {
        Some(op.result(db, 0))
    }
}

fn emit_op<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
    ctx: &FunctionEmitContext<'db>,
    module_info: &ModuleInfo<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let wasm_dialect = Symbol::new("wasm");
    if op.dialect(db) != wasm_dialect {
        return Err(CompilationError::unsupported_feature(
            "non-wasm op in wasm backend",
        ));
    }

    let name = op.name(db);
    let operands = op.operands(db);

    debug!("emit_op: {}.{}", op.dialect(db), name);

    // Handle wasm.nop - it's a placeholder for nil constants
    // For primitive types, no WASM instruction is emitted.
    // For reference types (func, anyref, etc.), emit ref.null so the value can be used.
    if wasm::Nop::matches(db, *op) {
        // Check if the result type is a reference type
        if let Some(result_ty) = op.results(db).first() {
            debug!(
                "wasm.nop: result_ty={}.{}",
                result_ty.dialect(db),
                result_ty.name(db)
            );
            if core::Func::from_type(db, *result_ty).is_some()
                || wasm::Funcref::from_type(db, *result_ty).is_some()
            {
                debug!("wasm.nop: emitting ref.null func");
                // Emit ref.null func for function reference types
                function.instruction(&Instruction::RefNull(HeapType::Abstract {
                    shared: false,
                    ty: AbstractHeapType::Func,
                }));
                set_result_local(db, op, ctx, function)?;
            } else if wasm::Anyref::from_type(db, *result_ty).is_some()
                || wasm::Structref::from_type(db, *result_ty).is_some()
            {
                debug!("wasm.nop: emitting ref.null any");
                // Emit ref.null any for other reference types
                function.instruction(&Instruction::RefNull(HeapType::Abstract {
                    shared: false,
                    ty: AbstractHeapType::Any,
                }));
                set_result_local(db, op, ctx, function)?;
            } else if core::Nil::from_type(db, *result_ty).is_some() {
                // Nil type - emit ref.null none for use as empty environment
                debug!("wasm.nop: emitting ref.null none for nil type");
                function.instruction(&Instruction::RefNull(HeapType::Abstract {
                    shared: false,
                    ty: AbstractHeapType::None,
                }));
                set_result_local(db, op, ctx, function)?;
            } else {
                debug!("wasm.nop: skipping (unknown primitive type)");
            }
            // For primitive types (nil, i32, etc.), skip - no runtime value
        }
        return Ok(());
    }

    // Fast path: simple operations (emit operands → instruction → set result)
    if let Some(instr) = SIMPLE_OPS.get(&name) {
        emit_operands(db, operands, ctx, &module_info.block_arg_types, function)?;
        function.instruction(instr);
        set_result_local(db, op, ctx, function)?;
        return Ok(());
    }

    // Special cases: const, control flow, calls, locals, GC ops
    if let Ok(const_op) = wasm::I32Const::from_operation(db, *op) {
        handle_i32_const(db, const_op, ctx, function)
    } else if let Ok(const_op) = wasm::I64Const::from_operation(db, *op) {
        handle_i64_const(db, const_op, ctx, function)
    } else if let Ok(const_op) = wasm::F32Const::from_operation(db, *op) {
        handle_f32_const(db, const_op, ctx, function)
    } else if let Ok(const_op) = wasm::F64Const::from_operation(db, *op) {
        handle_f64_const(db, const_op, ctx, function)
    } else if wasm::If::matches(db, *op) {
        handle_if(db, op, ctx, module_info, function)
    } else if wasm::Block::matches(db, *op) {
        handle_block(db, op, ctx, module_info, function)
    } else if wasm::Loop::matches(db, *op) {
        handle_loop(db, op, ctx, module_info, function)
    } else if let Ok(br_op) = wasm::Br::from_operation(db, *op) {
        handle_br(db, br_op, function)
    } else if let Ok(br_if_op) = wasm::BrIf::from_operation(db, *op) {
        handle_br_if(db, br_if_op, ctx, module_info, function)
    } else if let Ok(call_op) = wasm::Call::from_operation(db, *op) {
        handle_call(db, call_op, ctx, module_info, function)
    } else if wasm::CallIndirect::matches(db, *op) {
        handle_call_indirect(db, op, ctx, module_info, function)
    } else if let Ok(return_call_op) = wasm::ReturnCall::from_operation(db, *op) {
        handle_return_call(db, return_call_op, ctx, module_info, function)
    } else if let Ok(local_op) = wasm::LocalGet::from_operation(db, *op) {
        handle_local_get(db, local_op, ctx, function)
    } else if let Ok(local_op) = wasm::LocalSet::from_operation(db, *op) {
        handle_local_set(db, local_op, ctx, module_info, function)
    } else if let Ok(local_op) = wasm::LocalTee::from_operation(db, *op) {
        handle_local_tee(db, local_op, ctx, module_info, function)
    } else if let Ok(global_op) = wasm::GlobalGet::from_operation(db, *op) {
        handle_global_get(db, global_op, ctx, module_info, function)
    } else if let Ok(global_op) = wasm::GlobalSet::from_operation(db, *op) {
        handle_global_set(db, global_op, ctx, module_info, function)
    } else if wasm::StructNew::matches(db, *op) {
        handle_struct_new(db, op, ctx, module_info, function)
    } else if wasm::StructGet::matches(db, *op) {
        handle_struct_get(db, op, ctx, module_info, function)
    } else if let Ok(struct_set_op) = wasm::StructSet::from_operation(db, *op) {
        handle_struct_set(db, struct_set_op, ctx, module_info, function)
    } else if let Ok(array_new_op) = wasm::ArrayNew::from_operation(db, *op) {
        handle_array_new(db, array_new_op, ctx, module_info, function)
    } else if wasm::ArrayNewDefault::matches(db, *op) {
        handle_array_new_default(db, op, ctx, module_info, function)
    } else if let Ok(array_get_op) = wasm::ArrayGet::from_operation(db, *op) {
        handle_array_get(db, array_get_op, ctx, module_info, function)
    } else if let Ok(array_get_s_op) = wasm::ArrayGetS::from_operation(db, *op) {
        handle_array_get_s(db, array_get_s_op, ctx, module_info, function)
    } else if let Ok(array_get_u_op) = wasm::ArrayGetU::from_operation(db, *op) {
        handle_array_get_u(db, array_get_u_op, ctx, module_info, function)
    } else if let Ok(array_set_op) = wasm::ArraySet::from_operation(db, *op) {
        handle_array_set(db, array_set_op, ctx, module_info, function)
    } else if let Ok(array_copy_op) = wasm::ArrayCopy::from_operation(db, *op) {
        handle_array_copy(db, array_copy_op, ctx, module_info, function)
    } else if wasm::RefNull::matches(db, *op) {
        handle_ref_null(db, op, ctx, module_info, function)
    } else if let Ok(ref_func_op) = wasm::RefFunc::from_operation(db, *op) {
        handle_ref_func(db, ref_func_op, ctx, module_info, function)
    } else if wasm::RefCast::matches(db, *op) {
        handle_ref_cast(db, op, ctx, module_info, function)
    } else if wasm::RefTest::matches(db, *op) {
        handle_ref_test(db, op, ctx, module_info, function)
    } else if let Ok(bytes_op) = wasm::BytesFromData::from_operation(db, *op) {
        handle_bytes_from_data(db, bytes_op, ctx, function)

    // === Linear Memory Management ===
    } else if let Ok(mem_size_op) = wasm::MemorySize::from_operation(db, *op) {
        handle_memory_size(db, mem_size_op, ctx, function)
    } else if let Ok(mem_grow_op) = wasm::MemoryGrow::from_operation(db, *op) {
        handle_memory_grow(db, mem_grow_op, ctx, module_info, function)

    // === Full-Width Loads ===
    } else if let Ok(load_op) = wasm::I32Load::from_operation(db, *op) {
        handle_i32_load(db, load_op, ctx, module_info, function)
    } else if let Ok(load_op) = wasm::I64Load::from_operation(db, *op) {
        handle_i64_load(db, load_op, ctx, module_info, function)
    } else if let Ok(load_op) = wasm::F32Load::from_operation(db, *op) {
        handle_f32_load(db, load_op, ctx, module_info, function)
    } else if let Ok(load_op) = wasm::F64Load::from_operation(db, *op) {
        handle_f64_load(db, load_op, ctx, module_info, function)

    // === Partial-Width Loads (i32) ===
    } else if let Ok(load_op) = wasm::I32Load8S::from_operation(db, *op) {
        handle_i32_load8_s(db, load_op, ctx, module_info, function)
    } else if let Ok(load_op) = wasm::I32Load8U::from_operation(db, *op) {
        handle_i32_load8_u(db, load_op, ctx, module_info, function)
    } else if let Ok(load_op) = wasm::I32Load16S::from_operation(db, *op) {
        handle_i32_load16_s(db, load_op, ctx, module_info, function)
    } else if let Ok(load_op) = wasm::I32Load16U::from_operation(db, *op) {
        handle_i32_load16_u(db, load_op, ctx, module_info, function)

    // === Partial-Width Loads (i64) ===
    } else if let Ok(load_op) = wasm::I64Load8S::from_operation(db, *op) {
        handle_i64_load8_s(db, load_op, ctx, module_info, function)
    } else if let Ok(load_op) = wasm::I64Load8U::from_operation(db, *op) {
        handle_i64_load8_u(db, load_op, ctx, module_info, function)
    } else if let Ok(load_op) = wasm::I64Load16S::from_operation(db, *op) {
        handle_i64_load16_s(db, load_op, ctx, module_info, function)
    } else if let Ok(load_op) = wasm::I64Load16U::from_operation(db, *op) {
        handle_i64_load16_u(db, load_op, ctx, module_info, function)
    } else if let Ok(load_op) = wasm::I64Load32S::from_operation(db, *op) {
        handle_i64_load32_s(db, load_op, ctx, module_info, function)
    } else if let Ok(load_op) = wasm::I64Load32U::from_operation(db, *op) {
        handle_i64_load32_u(db, load_op, ctx, module_info, function)

    // === Full-Width Stores ===
    } else if let Ok(store_op) = wasm::I32Store::from_operation(db, *op) {
        handle_i32_store(db, store_op, ctx, module_info, function)
    } else if let Ok(store_op) = wasm::I64Store::from_operation(db, *op) {
        handle_i64_store(db, store_op, ctx, module_info, function)
    } else if let Ok(store_op) = wasm::F32Store::from_operation(db, *op) {
        handle_f32_store(db, store_op, ctx, module_info, function)
    } else if let Ok(store_op) = wasm::F64Store::from_operation(db, *op) {
        handle_f64_store(db, store_op, ctx, module_info, function)

    // === Partial-Width Stores ===
    } else if let Ok(store_op) = wasm::I32Store8::from_operation(db, *op) {
        handle_i32_store8(db, store_op, ctx, module_info, function)
    } else if let Ok(store_op) = wasm::I32Store16::from_operation(db, *op) {
        handle_i32_store16(db, store_op, ctx, module_info, function)
    } else if let Ok(store_op) = wasm::I64Store8::from_operation(db, *op) {
        handle_i64_store8(db, store_op, ctx, module_info, function)
    } else if let Ok(store_op) = wasm::I64Store16::from_operation(db, *op) {
        handle_i64_store16(db, store_op, ctx, module_info, function)
    } else if let Ok(store_op) = wasm::I64Store32::from_operation(db, *op) {
        handle_i64_store32(db, store_op, ctx, module_info, function)
    } else {
        tracing::error!(
            "unsupported wasm op: {} (dialect={}, operands={}, results={}, attrs={:?})",
            name,
            op.dialect(db),
            op.operands(db).len(),
            op.results(db).len(),
            op.attributes(db).keys().collect::<Vec<_>>()
        );
        Err(CompilationError::unsupported_feature_msg(format!(
            "wasm op not supported: {}",
            name
        )))
    }
}

fn set_result_local<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
    ctx: &FunctionEmitContext<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let results = op.results(db);
    if results.is_empty() {
        return Ok(());
    }
    // Note: We no longer skip nil types here because they may be used as
    // arguments (e.g., empty closure environments). Nil types now have local
    // mappings and produce ref.null none values.
    let local = ctx
        .value_locals
        .get(&op.result(db, 0))
        .ok_or_else(|| CompilationError::invalid_module("result missing local mapping"))?;
    function.instruction(&Instruction::LocalSet(*local));
    Ok(())
}

fn resolve_callee(path: Symbol, module_info: &ModuleInfo) -> CompilationResult<u32> {
    module_info
        .func_indices
        .get(&path)
        .copied()
        .ok_or_else(|| CompilationError::function_not_found(&path.to_string()))
}

/// Detect if a function body's handler dispatch should return i32 instead of funcref.
///
/// Handler dispatch generates wasm.if operations:
/// - then branch: suspend path (calls continuation, returns funcref)
/// - else branch: done path (returns actual computation result)
///
/// For handler lambdas that call continuations (k(value)), the done path also calls
/// the continuation via call_indirect, so it legitimately returns funcref.
/// For the computation lambda, the done path returns the actual result (i32).
///
/// Returns true if the function should return i32 (computation lambda case).
fn should_adjust_handler_return_to_i32<'db>(
    db: &'db dyn salsa::Database,
    region: &Region<'db>,
) -> bool {
    // Walk the region looking for wasm.if operations
    for block in region.blocks(db).iter() {
        for op in block.operations(db).iter() {
            // Check if this is a wasm.if (handler dispatch generates these)
            if wasm::If::matches(db, *op) {
                // Get the else region (done path) - it's region index 1
                let regions = op.regions(db);
                if let Some(else_region) = regions.get(1) {
                    // Check if the else branch contains call_indirect (continuation call)
                    // If it does, this lambda legitimately returns funcref
                    // If it doesn't, this is the computation lambda returning i32
                    let has_call_indirect = region_contains_call_indirect(db, else_region);
                    debug!(
                        "should_adjust_handler_return_to_i32: wasm.if else branch has_call_indirect={}",
                        has_call_indirect
                    );
                    if !has_call_indirect {
                        // This is the computation lambda - its done path returns i32
                        return true;
                    }
                }
                return false;
            }
            // Also check nested regions (wasm.block, etc.)
            for nested_region in op.regions(db).iter() {
                if should_adjust_handler_return_to_i32(db, nested_region) {
                    return true;
                }
            }
        }
    }
    false
}

/// Check if a region contains any wasm.call_indirect operations.
fn region_contains_call_indirect<'db>(db: &'db dyn salsa::Database, region: &Region<'db>) -> bool {
    for block in region.blocks(db).iter() {
        for op in block.operations(db).iter() {
            if wasm::CallIndirect::matches(db, *op) {
                return true;
            }
            // Check nested regions
            for nested_region in op.regions(db).iter() {
                if region_contains_call_indirect(db, nested_region) {
                    return true;
                }
            }
        }
    }
    false
}

fn compress_locals(locals: &[ValType]) -> Vec<(u32, ValType)> {
    let mut compressed = Vec::new();
    let mut iter = locals.iter();
    let Some(mut current) = iter.next().copied() else {
        return compressed;
    };
    let mut count: u32 = 1;
    for val in iter {
        if *val == current {
            count += 1;
        } else {
            compressed.push((count, current));
            current = *val;
            count = 1;
        }
    }
    compressed.push((count, current));
    compressed
}

fn attr_u32<'db>(attrs: &Attrs<'db>, key: Symbol) -> CompilationResult<u32> {
    match attrs.get(&key) {
        Some(Attribute::IntBits(bits)) => Ok(*bits as u32),
        _ => Err(CompilationError::from(
            errors::CompilationErrorKind::MissingAttribute("u32"),
        )),
    }
}

/// Get field index from attributes, trying both `field_idx` and `field` attribute names.
fn attr_field_idx<'db>(attrs: &Attrs<'db>) -> CompilationResult<u32> {
    attr_u32(attrs, ATTR_FIELD_IDX()).or_else(|_| attr_u32(attrs, ATTR_FIELD()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use trunk_ir::{Block, BlockBuilder, BlockId, Location, PathId, Region, Span, idvec};

    fn test_location(db: &dyn salsa::Database) -> Location<'_> {
        let path = PathId::new(db, "file:///test.trb".to_owned());
        Location::new(path, Span::new(0, 0))
    }

    /// Helper to extract the GcTypeDef kind for testing.
    fn gc_type_kind(gc_type: &GcTypeDef) -> &'static str {
        match gc_type {
            GcTypeDef::Struct(_) => "struct",
            GcTypeDef::Array(_) => "array",
        }
    }

    // ========================================
    // Test: struct_new collects field types
    // ========================================

    #[salsa::tracked]
    fn make_struct_new_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();
        let i64_ty = core::I64::new(db).as_type();

        // Create two field values with i32_const
        let field0 = wasm::i32_const(db, location, i32_ty, 42).as_operation();

        let field1 = wasm::i64_const(db, location, i64_ty, 100).as_operation();

        // Create struct_new with two fields
        let struct_ty = core::I32::new(db).as_type(); // placeholder type
        let struct_new = wasm::struct_new(
            db,
            location,
            vec![field0.result(db, 0), field1.result(db, 0)],
            struct_ty,
            FIRST_USER_TYPE_IDX,
        )
        .as_operation();

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![field0, field1, struct_new],
        );
        let region = Region::new(db, location, idvec![block]);
        core::Module::create(db, location, "test".into(), region)
    }

    #[salsa_test]
    fn test_struct_new_collects_field_types(db: &salsa::DatabaseImpl) {
        let module = make_struct_new_module(db);
        let (gc_types, type_map, _) =
            collect_gc_types(db, module, &HashMap::new()).expect("collect_gc_types failed");

        // Should have 8 GC types: 7 built-in + 1 user struct
        assert_eq!(gc_types.len(), 8);
        // Index 0 is BoxedF64
        assert_eq!(gc_type_kind(&gc_types[0]), "struct");
        // Index 1 is BytesArray
        assert_eq!(gc_type_kind(&gc_types[1]), "array");
        // Index 2 is BytesStruct
        assert_eq!(gc_type_kind(&gc_types[2]), "struct");
        // Index 3 is Step
        assert_eq!(gc_type_kind(&gc_types[3]), "struct");
        // Index 4 is ClosureStruct
        assert_eq!(gc_type_kind(&gc_types[4]), "struct");
        // Index 5 is Marker
        assert_eq!(gc_type_kind(&gc_types[5]), "struct");
        // Index 6 is Evidence
        assert_eq!(gc_type_kind(&gc_types[6]), "array");
        // Index 7 is the user struct
        assert_eq!(gc_type_kind(&gc_types[7]), "struct");

        // Type should be in the map
        let i32_ty = core::I32::new(db).as_type();
        assert!(type_map.contains_key(&i32_ty));
    }

    // ========================================
    // Test: array_new collects element type
    // ========================================

    #[salsa::tracked]
    fn make_array_new_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();
        let f64_ty = core::F64::new(db).as_type();

        // Create size value
        let size = wasm::i32_const(db, location, i32_ty, 10).as_operation();

        // Create init value (f64)
        let init = wasm::f64_const(db, location, f64_ty, 0.0).as_operation();

        // Create array_new
        let array_new = wasm::array_new(
            db,
            location,
            size.result(db, 0),
            init.result(db, 0),
            i32_ty, // placeholder result type
            FIRST_USER_TYPE_IDX,
        )
        .as_operation();

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![size, init, array_new],
        );
        let region = Region::new(db, location, idvec![block]);
        core::Module::create(db, location, "test".into(), region)
    }

    #[salsa_test]
    fn test_array_new_collects_element_type(db: &salsa::DatabaseImpl) {
        let module = make_array_new_module(db);
        let (gc_types, _type_map, _) =
            collect_gc_types(db, module, &HashMap::new()).expect("collect_gc_types failed");

        // Should have 8 GC types: 7 built-in + 1 user array
        assert_eq!(gc_types.len(), 8);
        // Index 0 is BoxedF64 (struct)
        assert_eq!(gc_type_kind(&gc_types[0]), "struct");
        // Index 1 is BytesArray (array)
        assert_eq!(gc_type_kind(&gc_types[1]), "array");
        // Index 2 is BytesStruct (struct)
        assert_eq!(gc_type_kind(&gc_types[2]), "struct");
        // Index 3 is Step (struct)
        assert_eq!(gc_type_kind(&gc_types[3]), "struct");
        // Index 4 is ClosureStruct (struct)
        assert_eq!(gc_type_kind(&gc_types[4]), "struct");
        // Index 5 is Marker (struct)
        assert_eq!(gc_type_kind(&gc_types[5]), "struct");
        // Index 6 is Evidence (array)
        assert_eq!(gc_type_kind(&gc_types[6]), "array");
        // Index 7 is the user array
        assert_eq!(gc_type_kind(&gc_types[7]), "array");
    }

    // ========================================
    // Test: type index deduplication
    // ========================================

    #[salsa::tracked]
    fn make_dedup_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();

        // Create two struct_new operations with same type_idx
        let field = wasm::i32_const(db, location, i32_ty, 1).as_operation();

        let struct_new1 = wasm::struct_new(
            db,
            location,
            vec![field.result(db, 0)],
            i32_ty,
            FIRST_USER_TYPE_IDX,
        )
        .as_operation();

        let field2 = wasm::i32_const(db, location, i32_ty, 2).as_operation();

        let struct_new2 = wasm::struct_new(
            db,
            location,
            vec![field2.result(db, 0)],
            i32_ty,
            FIRST_USER_TYPE_IDX, // same type_idx
        )
        .as_operation();

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![field, struct_new1, field2, struct_new2],
        );
        let region = Region::new(db, location, idvec![block]);
        core::Module::create(db, location, "test".into(), region)
    }

    #[salsa_test]
    fn test_type_index_deduplication(db: &salsa::DatabaseImpl) {
        let module = make_dedup_module(db);
        let (gc_types, _type_map, _) =
            collect_gc_types(db, module, &HashMap::new()).expect("collect_gc_types failed");

        // Should have 8 GC types: 7 built-in + 1 user struct (same type_idx used twice)
        assert_eq!(gc_types.len(), 8);
        // Index 0 is BoxedF64
        assert_eq!(gc_type_kind(&gc_types[0]), "struct");
        // Index 1 is BytesArray
        assert_eq!(gc_type_kind(&gc_types[1]), "array");
        // Index 2 is BytesStruct
        assert_eq!(gc_type_kind(&gc_types[2]), "struct");
        // Index 3 is Step
        assert_eq!(gc_type_kind(&gc_types[3]), "struct");
        // Index 4 is ClosureStruct
        assert_eq!(gc_type_kind(&gc_types[4]), "struct");
        // Index 5 is Marker
        assert_eq!(gc_type_kind(&gc_types[5]), "struct");
        // Index 6 is Evidence
        assert_eq!(gc_type_kind(&gc_types[6]), "array");
        // Index 7 is the deduplicated user struct
        assert_eq!(gc_type_kind(&gc_types[7]), "struct");
    }

    // ========================================
    // Test: field count mismatch error
    // ========================================

    #[salsa::tracked]
    fn make_field_count_mismatch_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();

        // Create struct_new with 1 field
        let field = wasm::i32_const(db, location, i32_ty, 1).as_operation();

        let struct_new1 = wasm::struct_new(
            db,
            location,
            vec![field.result(db, 0)],
            i32_ty,
            FIRST_USER_TYPE_IDX,
        )
        .as_operation();

        // Create another struct_new with 2 fields (same type_idx)
        let field2a = wasm::i32_const(db, location, i32_ty, 2).as_operation();

        let field2b = wasm::i32_const(db, location, i32_ty, 3).as_operation();

        let struct_new2 = wasm::struct_new(
            db,
            location,
            vec![field2a.result(db, 0), field2b.result(db, 0)],
            i32_ty,
            FIRST_USER_TYPE_IDX, // same type_idx, different field count
        )
        .as_operation();

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![field, struct_new1, field2a, field2b, struct_new2],
        );
        let region = Region::new(db, location, idvec![block]);
        core::Module::create(db, location, "test".into(), region)
    }

    #[salsa_test]
    fn test_field_count_mismatch_error(db: &salsa::DatabaseImpl) {
        let module = make_field_count_mismatch_module(db);
        let result = collect_gc_types(db, module, &HashMap::new());

        // Should return an error due to field count mismatch
        let err = result.expect_err("expected error");
        assert!(err.to_string().contains("field count mismatch"));
    }

    // ========================================
    // Test: nested operations in function body
    // ========================================

    #[salsa::tracked]
    fn make_func_with_struct_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();
        let nil_ty = core::Nil::new(db).as_type();
        let func_ty = core::Func::new(db, idvec![], nil_ty).as_type();

        // Create struct_new inside function body
        let field = wasm::i32_const(db, location, i32_ty, 42).as_operation();

        let struct_new = wasm::struct_new(
            db,
            location,
            vec![field.result(db, 0)],
            i32_ty,
            FIRST_USER_TYPE_IDX,
        )
        .as_operation();

        let func_return = wasm::r#return(db, location, vec![]).as_operation();

        let body_block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![field, struct_new, func_return],
        );
        let body_region = Region::new(db, location, idvec![body_block]);

        // Create wasm.func using typed helper
        let wasm_func =
            wasm::func(db, location, Symbol::new("test_fn"), func_ty, body_region).as_operation();

        let block = Block::new(db, BlockId::fresh(), location, idvec![], idvec![wasm_func]);
        let region = Region::new(db, location, idvec![block]);
        core::Module::create(db, location, "test".into(), region)
    }

    #[salsa_test]
    fn test_nested_operations_in_function_body(db: &salsa::DatabaseImpl) {
        let module = make_func_with_struct_module(db);
        let (gc_types, _type_map, _) =
            collect_gc_types(db, module, &HashMap::new()).expect("collect_gc_types failed");

        // Should find the struct type from inside the function body
        // (8 types: 7 built-in + 1 user struct)
        assert_eq!(gc_types.len(), 8);
        // Index 0 is BoxedF64
        assert_eq!(gc_type_kind(&gc_types[0]), "struct");
        // Index 1 is BytesArray
        assert_eq!(gc_type_kind(&gc_types[1]), "array");
        // Index 2 is BytesStruct
        assert_eq!(gc_type_kind(&gc_types[2]), "struct");
        // Index 3 is Step
        assert_eq!(gc_type_kind(&gc_types[3]), "struct");
        // Index 4 is ClosureStruct
        assert_eq!(gc_type_kind(&gc_types[4]), "struct");
        // Index 5 is Marker
        assert_eq!(gc_type_kind(&gc_types[5]), "struct");
        // Index 6 is Evidence
        assert_eq!(gc_type_kind(&gc_types[6]), "array");
        // Index 7 is the user struct from inside the function body
        assert_eq!(gc_type_kind(&gc_types[7]), "struct");
    }

    // ========================================
    // Test: builtin type indices are skipped
    // ========================================

    /// Test that array_get with BYTES_ARRAY_IDX (1) doesn't panic.
    /// This tests the fix for the "attempt to subtract with overflow" bug
    /// that occurred when operations referenced builtin type indices.
    #[salsa::tracked]
    fn make_array_get_builtin_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();

        // Use BlockBuilder to create block with argument as placeholder for array reference
        let mut block_builder = BlockBuilder::new(db, location).arg(i32_ty);
        let array_ref_arg = block_builder.block_arg(db, 0);

        // Create index value
        let index = wasm::i32_const(db, location, i32_ty, 0);
        block_builder.op(index);

        // Create array_get_u with BYTES_ARRAY_IDX (builtin type)
        let array_get = wasm::array_get_u(
            db,
            location,
            array_ref_arg,
            index.result(db),
            i32_ty,
            BYTES_ARRAY_IDX,
        );
        block_builder.op(array_get);

        let block = block_builder.build();
        let region = Region::new(db, location, idvec![block]);
        core::Module::create(db, location, "test".into(), region)
    }

    #[salsa_test]
    fn test_array_get_with_builtin_type_idx_does_not_panic(db: &salsa::DatabaseImpl) {
        let module = make_array_get_builtin_module(db);
        let result = collect_gc_types(db, module, &HashMap::new());

        // Should complete without panic
        assert!(
            result.is_ok(),
            "collect_gc_types should not panic for builtin type indices"
        );

        let (gc_types, _type_map, _) = result.expect("collect_gc_types failed");

        // Should only have the 7 built-in types (no additional user types allocated)
        assert_eq!(gc_types.len(), 7);
        assert_eq!(gc_type_kind(&gc_types[0]), "struct"); // BoxedF64
        assert_eq!(gc_type_kind(&gc_types[1]), "array"); // BytesArray
        assert_eq!(gc_type_kind(&gc_types[2]), "struct"); // BytesStruct
        assert_eq!(gc_type_kind(&gc_types[3]), "struct"); // Step
        assert_eq!(gc_type_kind(&gc_types[4]), "struct"); // ClosureStruct
        assert_eq!(gc_type_kind(&gc_types[5]), "struct"); // Marker
        assert_eq!(gc_type_kind(&gc_types[6]), "array"); // Evidence
    }

    /// Test that struct_get with BYTES_STRUCT_IDX (2) doesn't panic.
    #[salsa::tracked]
    fn make_struct_get_builtin_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();

        // Use BlockBuilder to create block with argument as placeholder for struct reference
        let mut block_builder = BlockBuilder::new(db, location).arg(i32_ty);
        let struct_ref_arg = block_builder.block_arg(db, 0);

        // Create struct_get with BYTES_STRUCT_IDX (builtin type)
        let struct_get =
            wasm::struct_get(db, location, struct_ref_arg, i32_ty, BYTES_STRUCT_IDX, 0);
        block_builder.op(struct_get);

        let block = block_builder.build();
        let region = Region::new(db, location, idvec![block]);
        core::Module::create(db, location, "test".into(), region)
    }

    #[salsa_test]
    fn test_struct_get_with_builtin_type_idx_does_not_panic(db: &salsa::DatabaseImpl) {
        let module = make_struct_get_builtin_module(db);
        let result = collect_gc_types(db, module, &HashMap::new());

        // Should complete without panic
        assert!(
            result.is_ok(),
            "collect_gc_types should not panic for builtin type indices"
        );

        let (gc_types, _type_map, _) = result.expect("collect_gc_types failed");

        // Should only have the 7 built-in types (no additional user types allocated)
        assert_eq!(gc_types.len(), 7);
        assert_eq!(gc_type_kind(&gc_types[0]), "struct"); // BoxedF64
        assert_eq!(gc_type_kind(&gc_types[1]), "array"); // BytesArray
        assert_eq!(gc_type_kind(&gc_types[2]), "struct"); // BytesStruct
        assert_eq!(gc_type_kind(&gc_types[3]), "struct"); // Step
        assert_eq!(gc_type_kind(&gc_types[4]), "struct"); // ClosureStruct
        assert_eq!(gc_type_kind(&gc_types[5]), "struct"); // Marker
        assert_eq!(gc_type_kind(&gc_types[6]), "array"); // Evidence
    }

    /// Test that array_set with BYTES_ARRAY_IDX (1) doesn't panic.
    #[salsa::tracked]
    fn make_array_set_builtin_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();

        // Use BlockBuilder to create block with argument as placeholder for array reference
        let mut block_builder = BlockBuilder::new(db, location).arg(i32_ty);
        let array_ref_arg = block_builder.block_arg(db, 0);

        // Create index value
        let index = wasm::i32_const(db, location, i32_ty, 0);
        block_builder.op(index);

        // Create value to set
        let value = wasm::i32_const(db, location, i32_ty, 42);
        block_builder.op(value);

        // Create array_set with BYTES_ARRAY_IDX (builtin type)
        let array_set = wasm::array_set(
            db,
            location,
            array_ref_arg,
            index.result(db),
            value.result(db),
            BYTES_ARRAY_IDX,
        );
        block_builder.op(array_set);

        let block = block_builder.build();
        let region = Region::new(db, location, idvec![block]);
        core::Module::create(db, location, "test".into(), region)
    }

    #[salsa_test]
    fn test_array_set_with_builtin_type_idx_does_not_panic(db: &salsa::DatabaseImpl) {
        let module = make_array_set_builtin_module(db);
        let result = collect_gc_types(db, module, &HashMap::new());

        // Should complete without panic
        assert!(
            result.is_ok(),
            "collect_gc_types should not panic for builtin type indices"
        );

        let (gc_types, _type_map, _) = result.expect("collect_gc_types failed");

        // Should only have the 7 built-in types
        assert_eq!(gc_types.len(), 7);
    }

    // ========================================
    // Test: memory load/store operations
    // ========================================

    #[salsa::tracked]
    fn make_memory_ops_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();
        let nil_ty = core::Nil::new(db).as_type();
        let func_ty = core::Func::new(db, idvec![], nil_ty).as_type();

        // Create address operand (i32)
        let addr = wasm::i32_const(db, location, i32_ty, 0).as_operation();

        // Create value to store (i32)
        let value = wasm::i32_const(db, location, i32_ty, 42).as_operation();

        // i32_store with offset and align attributes
        let store_op = wasm::i32_store(
            db,
            location,
            addr.result(db, 0),
            value.result(db, 0),
            4,
            2,
            0,
        )
        .as_operation();

        // i32_load from same address
        let load_op =
            wasm::i32_load(db, location, addr.result(db, 0), i32_ty, 4, 2, 0).as_operation();

        // Return statement
        let func_return = wasm::r#return(db, location, vec![]).as_operation();

        let body_block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![addr, value, store_op, load_op, func_return],
        );
        let body_region = Region::new(db, location, idvec![body_block]);

        // Function definition
        let wasm_func =
            wasm::func(db, location, Symbol::new("test"), func_ty, body_region).as_operation();

        // Memory definition (required for load/store)
        let memory_op = wasm::memory(db, location, 1, 1, false, false).as_operation();

        let module_block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![memory_op, wasm_func],
        );
        let module_region = Region::new(db, location, idvec![module_block]);
        core::Module::create(db, location, "test".into(), module_region)
    }

    #[salsa_test]
    fn test_memory_load_store_emit(db: &salsa::DatabaseImpl) {
        let module = make_memory_ops_module(db);
        let result = emit_wasm(db, module);
        assert!(
            result.is_ok(),
            "Memory load/store should compile: {:?}",
            result.err()
        );

        let bytes = result.unwrap();
        // Check WASM magic number
        assert_eq!(&bytes[0..4], b"\x00asm", "Should have wasm magic number");
    }

    // ========================================
    // Test: memory_size and memory_grow
    // ========================================

    #[salsa::tracked]
    fn make_memory_grow_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();
        let nil_ty = core::Nil::new(db).as_type();
        let func_ty = core::Func::new(db, idvec![], nil_ty).as_type();

        // memory_size
        let size_op = wasm::memory_size(db, location, i32_ty, 0).as_operation();

        // delta for memory_grow
        let delta = wasm::i32_const(db, location, i32_ty, 1).as_operation();

        // memory_grow
        let grow_op =
            wasm::memory_grow(db, location, delta.result(db, 0), i32_ty, 0).as_operation();

        // Return statement
        let func_return = wasm::r#return(db, location, vec![]).as_operation();

        let body_block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![size_op, delta, grow_op, func_return],
        );
        let body_region = Region::new(db, location, idvec![body_block]);

        let wasm_func =
            wasm::func(db, location, Symbol::new("test"), func_ty, body_region).as_operation();

        let memory_op = wasm::memory(db, location, 1, 2, false, false).as_operation();

        let module_block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![memory_op, wasm_func],
        );
        let module_region = Region::new(db, location, idvec![module_block]);
        core::Module::create(db, location, "test".into(), module_region)
    }

    #[salsa_test]
    fn test_memory_grow_emit(db: &salsa::DatabaseImpl) {
        let module = make_memory_grow_module(db);
        let result = emit_wasm(db, module);
        assert!(
            result.is_ok(),
            "Memory grow should compile: {:?}",
            result.err()
        );

        let bytes = result.unwrap();
        assert_eq!(&bytes[0..4], b"\x00asm", "Should have wasm magic number");
    }

    // ========================================
    // Test: ref.null with anyref type (ref_handlers)
    // ========================================

    #[salsa::tracked]
    fn make_ref_null_anyref_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let location = test_location(db);
        let nil_ty = core::Nil::new(db).as_type();
        let anyref_ty = wasm::Anyref::new(db).as_type();
        let func_ty = core::Func::new(db, idvec![], nil_ty).as_type();

        // Create ref.null anyref (heap_type must be a wasm type)
        let ref_null = wasm::ref_null(db, location, anyref_ty, anyref_ty, None).as_operation();

        // Return statement
        let func_return = wasm::r#return(db, location, vec![]).as_operation();

        let body_block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![ref_null, func_return],
        );
        let body_region = Region::new(db, location, idvec![body_block]);

        let wasm_func = wasm::func(
            db,
            location,
            Symbol::new("test_ref_null"),
            func_ty,
            body_region,
        )
        .as_operation();

        let module_block = Block::new(db, BlockId::fresh(), location, idvec![], idvec![wasm_func]);
        let module_region = Region::new(db, location, idvec![module_block]);
        core::Module::create(db, location, "test".into(), module_region)
    }

    #[salsa_test]
    fn test_ref_null_anyref_type_emit(db: &salsa::DatabaseImpl) {
        let module = make_ref_null_anyref_module(db);
        let result = emit_wasm(db, module);
        assert!(
            result.is_ok(),
            "ref.null anyref type should compile: {:?}",
            result.err()
        );

        let bytes = result.unwrap();
        assert_eq!(&bytes[0..4], b"\x00asm", "Should have wasm magic number");
    }

    // ========================================
    // Test: struct_get operation (struct_handlers)
    // ========================================

    #[salsa::tracked]
    fn make_struct_get_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();
        let nil_ty = core::Nil::new(db).as_type();
        let func_ty = core::Func::new(db, idvec![], nil_ty).as_type();

        // Create a function that creates a struct and gets a field
        // Field value
        let field = wasm::i32_const(db, location, i32_ty, 42).as_operation();

        // Create struct
        let struct_new = wasm::struct_new(
            db,
            location,
            vec![field.result(db, 0)],
            i32_ty,
            FIRST_USER_TYPE_IDX,
        )
        .as_operation();

        // Get field from struct
        let struct_get = wasm::struct_get(
            db,
            location,
            struct_new.result(db, 0),
            i32_ty,
            FIRST_USER_TYPE_IDX,
            0,
        )
        .as_operation();

        // Return statement
        let func_return = wasm::r#return(db, location, vec![]).as_operation();

        let body_block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![field, struct_new, struct_get, func_return],
        );
        let body_region = Region::new(db, location, idvec![body_block]);

        let wasm_func = wasm::func(
            db,
            location,
            Symbol::new("test_struct_get"),
            func_ty,
            body_region,
        )
        .as_operation();

        let module_block = Block::new(db, BlockId::fresh(), location, idvec![], idvec![wasm_func]);
        let module_region = Region::new(db, location, idvec![module_block]);
        core::Module::create(db, location, "test".into(), module_region)
    }

    #[salsa_test]
    fn test_struct_get_emit(db: &salsa::DatabaseImpl) {
        let module = make_struct_get_module(db);
        let result = emit_wasm(db, module);
        assert!(
            result.is_ok(),
            "struct.get should compile: {:?}",
            result.err()
        );

        let bytes = result.unwrap();
        assert_eq!(&bytes[0..4], b"\x00asm", "Should have wasm magic number");
    }

    // ========================================
    // Test: anyref operand in struct (value_emission)
    // ========================================

    #[salsa::tracked]
    fn make_anyref_operand_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let location = test_location(db);
        let nil_ty = core::Nil::new(db).as_type();
        let anyref_ty = wasm::Anyref::new(db).as_type();
        let func_ty = core::Func::new(db, idvec![], nil_ty).as_type();

        // Create ref.null anyref - tests that reference operands work properly
        let anyref_val = wasm::ref_null(db, location, anyref_ty, anyref_ty, None).as_operation();

        // Create a struct with i32 and anyref fields (closure-like pattern)
        let i32_ty = core::I32::new(db).as_type();
        let i32_val = wasm::i32_const(db, location, i32_ty, 1).as_operation();

        // Two-field struct: (i32, anyref)
        let struct_new = wasm::struct_new(
            db,
            location,
            vec![i32_val.result(db, 0), anyref_val.result(db, 0)],
            i32_ty,
            FIRST_USER_TYPE_IDX,
        )
        .as_operation();

        // Return statement
        let func_return = wasm::r#return(db, location, vec![]).as_operation();

        let body_block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![anyref_val, i32_val, struct_new, func_return],
        );
        let body_region = Region::new(db, location, idvec![body_block]);

        let wasm_func = wasm::func(
            db,
            location,
            Symbol::new("test_anyref_operand"),
            func_ty,
            body_region,
        )
        .as_operation();

        let module_block = Block::new(db, BlockId::fresh(), location, idvec![], idvec![wasm_func]);
        let module_region = Region::new(db, location, idvec![module_block]);
        core::Module::create(db, location, "test".into(), module_region)
    }

    #[salsa_test]
    fn test_anyref_operand_in_struct(db: &salsa::DatabaseImpl) {
        let module = make_anyref_operand_module(db);
        let result = emit_wasm(db, module);
        assert!(
            result.is_ok(),
            "anyref operand should compile: {:?}",
            result.err()
        );

        let bytes = result.unwrap();
        assert_eq!(&bytes[0..4], b"\x00asm", "Should have wasm magic number");
    }

    // ========================================
    // Test: ref.cast with concrete type (ref_handlers)
    // ========================================

    #[salsa::tracked]
    fn make_ref_cast_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();
        let nil_ty = core::Nil::new(db).as_type();
        let func_ty = core::Func::new(db, idvec![], nil_ty).as_type();

        // Create a struct
        let field = wasm::i32_const(db, location, i32_ty, 42).as_operation();

        let struct_new = wasm::struct_new(
            db,
            location,
            vec![field.result(db, 0)],
            i32_ty,
            FIRST_USER_TYPE_IDX,
        )
        .as_operation();

        // Cast the struct to a concrete type (target_type, result_type, type_idx)
        let ref_cast = wasm::ref_cast(
            db,
            location,
            struct_new.result(db, 0),
            i32_ty,
            i32_ty,
            Some(FIRST_USER_TYPE_IDX),
        )
        .as_operation();

        // Return statement
        let func_return = wasm::r#return(db, location, vec![]).as_operation();

        let body_block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![field, struct_new, ref_cast, func_return],
        );
        let body_region = Region::new(db, location, idvec![body_block]);

        let wasm_func = wasm::func(
            db,
            location,
            Symbol::new("test_ref_cast"),
            func_ty,
            body_region,
        )
        .as_operation();

        let module_block = Block::new(db, BlockId::fresh(), location, idvec![], idvec![wasm_func]);
        let module_region = Region::new(db, location, idvec![module_block]);
        core::Module::create(db, location, "test".into(), module_region)
    }

    #[salsa_test]
    fn test_ref_cast_concrete_type_emit(db: &salsa::DatabaseImpl) {
        let module = make_ref_cast_module(db);
        let result = emit_wasm(db, module);
        assert!(
            result.is_ok(),
            "ref.cast with concrete type should compile: {:?}",
            result.err()
        );

        let bytes = result.unwrap();
        assert_eq!(&bytes[0..4], b"\x00asm", "Should have wasm magic number");
    }
}
