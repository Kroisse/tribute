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
    handle_array_new_default, handle_array_set, handle_f32_const, handle_f32_load,
    handle_f32_store, handle_f64_const, handle_f64_load, handle_f64_store, handle_i32_const,
    handle_i32_load, handle_i32_load8_s, handle_i32_load8_u, handle_i32_load16_s,
    handle_i32_load16_u, handle_i32_store, handle_i32_store8, handle_i32_store16, handle_i64_const,
    handle_i64_load, handle_i64_load8_s, handle_i64_load8_u, handle_i64_load16_s,
    handle_i64_load16_u, handle_i64_load32_s, handle_i64_load32_u, handle_i64_store,
    handle_i64_store8, handle_i64_store16, handle_i64_store32, handle_memory_grow,
    handle_memory_size, handle_ref_cast, handle_ref_func, handle_ref_null, handle_ref_test,
};
use helpers::*;
use value_emission::*;

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

use tracing::debug;

use tribute_ir::ModulePathExt;
use tribute_ir::dialect::{adt, tribute, tribute_rt};
#[cfg(test)]
use trunk_ir::IdVec;
use trunk_ir::dialect::{core, wasm};
use trunk_ir::{
    Attribute, Attrs, BlockId, DialectOp, DialectType, Operation, Region, Symbol, Type, Value,
    ValueDef,
};
use wasm_encoder::{
    AbstractHeapType, ArrayType, BlockType, CodeSection, CompositeInnerType, CompositeType,
    ConstExpr, DataCountSection, DataSection, ElementSection, Elements, EntityType, ExportKind,
    ExportSection, Function, FunctionSection, GlobalSection, GlobalType, HeapType, ImportSection,
    Instruction, MemorySection, MemoryType, Module, RefType, StorageType, StructType, SubType,
    TableSection, TableType, TypeSection, ValType,
};

use crate::errors;
#[cfg(test)]
use crate::gc_types::FIRST_USER_TYPE_IDX;
use crate::gc_types::{
    ATTR_FIELD_IDX, ATTR_TYPE, ATTR_TYPE_IDX, BYTES_ARRAY_IDX, BYTES_STRUCT_IDX,
    CLOSURE_STRUCT_IDX, GcTypeDef,
};
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
    /// Used to determine block types when IR type is type_var.
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
            // Export with simple name (last segment of qualified name)
            let name = func_def.name.last_segment().to_string();
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
    // Pass the count of function definitions so call_indirect types get indices after them
    let func_type_count = info.imports.len() + info.funcs.len();
    let call_indirect_types = collect_call_indirect_types(
        db,
        module,
        &mut type_idx_by_type,
        &info.block_arg_types,
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

    // Get the function's expected return type for use when IR type is type_var
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
        for op in block.operations(db).iter() {
            // Skip tribute.var operations - kept for LSP support, no runtime effect
            if op.dialect(db) == tribute::DIALECT_NAME() && op.name(db) == tribute::VAR() {
                continue;
            }

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
                // For generic function calls, infer the concrete return type from operands.
                // This ensures the local is typed correctly for unboxed values.
                let mut effective_ty = infer_call_result_type(
                    db,
                    op,
                    result_ty,
                    &module_info.func_types,
                    &module_info.block_arg_types,
                    ctx.func_return_type,
                );

                // For struct_get operations, the local type must match the actual struct
                // field type, not the IR result type. This is critical because:
                // 1. The IR may have a generic type.var that maps to anyref
                // 2. Handler lambdas may have captured types like tribute.type<Int>
                //    that convert to anyref but whose actual field is i32
                // 3. The WASM struct.get instruction returns the actual field type
                //
                // We check if the result type would convert to anyref (indicating
                // potential mismatch) and look up the actual field type from the
                // struct definition or GC type table.
                let effective_ty_is_ref =
                    type_to_valtype(db, effective_ty, &module_info.type_idx_by_type)
                        .ok()
                        .map(|vt| matches!(vt, ValType::Ref(_)))
                        .unwrap_or(false);
                if wasm::StructGet::matches(db, *op)
                    && (tribute::is_type_var(db, effective_ty) || effective_ty_is_ref)
                {
                    // Try to get struct type from attribute first, fall back to operand type
                    let struct_ty_opt = op
                        .attributes(db)
                        .get(&ATTR_TYPE())
                        .and_then(|attr| {
                            if let Attribute::Type(ty) = attr {
                                Some(*ty)
                            } else {
                                None
                            }
                        })
                        .or_else(|| {
                            // Infer from first operand (the struct reference)
                            op.operands(db)
                                .first()
                                .and_then(|v| value_type(db, *v, &module_info.block_arg_types))
                        });

                    if let Some(struct_ty) = struct_ty_opt {
                        debug!(
                            "struct_get struct type: {}.{}",
                            struct_ty.dialect(db),
                            struct_ty.name(db)
                        );
                        // For variant types, look up the actual field type from adt.enum type.
                        // The base_enum attribute contains the adt.enum type with variant info.
                        if adt::is_variant_instance_type(db, struct_ty)
                            && let (Some(base_enum_ty), Some(variant_tag)) = (
                                adt::get_base_enum(db, struct_ty),
                                adt::get_variant_tag(db, struct_ty),
                            )
                            && let Ok(field_idx) = attr_field_idx(op.attributes(db))
                            && let Some(variants) = adt::get_enum_variants(db, base_enum_ty)
                            && let Some((_, field_types)) =
                                variants.iter().find(|(tag, _)| *tag == variant_tag)
                            && let Some(field_ty) = field_types.get(field_idx as usize)
                        {
                            // Extract the type name from tribute.type's "name" attribute
                            // or directly from the type name for other types.
                            let type_name = if field_ty.dialect(db) == Symbol::new("tribute")
                                && field_ty.name(db) == Symbol::new("type")
                            {
                                // tribute.type stores name in "name" attribute
                                field_ty
                                    .get_attr(db, Symbol::new("name"))
                                    .and_then(|a| {
                                        if let Attribute::Symbol(s) = a {
                                            Some(*s)
                                        } else {
                                            None
                                        }
                                    })
                                    .unwrap_or_else(|| field_ty.name(db))
                            } else {
                                field_ty.name(db)
                            };
                            debug!(
                                "  -> variant {:?} field {} resolved type name: {}",
                                variant_tag, field_idx, type_name
                            );
                            // Convert unresolved src.* types to ty.* types.
                            // The types from adt.enum are unresolved (src.Int),
                            // but we need resolved types for local allocation.
                            if type_name == Symbol::new("Int") {
                                effective_ty = tribute_rt::int_type(db);
                            } else if type_name == Symbol::new("Nat") {
                                effective_ty = tribute_rt::nat_type(db);
                            } else if type_name == Symbol::new("Float") {
                                effective_ty = tribute_rt::float_type(db);
                            } else if type_name == Symbol::new("Bool") {
                                effective_ty = tribute_rt::bool_type(db);
                            }
                            // For other types (like Expr), keep effective_ty
                            // as type.var which maps to anyref
                        }

                        // For adt.struct types, look up the actual field type from the struct definition.
                        // This handles closure environment access where captured values have concrete types.
                        if adt::is_struct_type(db, struct_ty)
                            && let Ok(field_idx) = attr_field_idx(op.attributes(db))
                            && let Some(fields) = adt::get_struct_fields(db, struct_ty)
                            && let Some((_, field_ty)) = fields.get(field_idx as usize)
                        {
                            let type_name = if field_ty.dialect(db) == Symbol::new("tribute")
                                && field_ty.name(db) == Symbol::new("type")
                            {
                                field_ty
                                    .get_attr(db, Symbol::new("name"))
                                    .and_then(|a| {
                                        if let Attribute::Symbol(s) = a {
                                            Some(*s)
                                        } else {
                                            None
                                        }
                                    })
                                    .unwrap_or_else(|| field_ty.name(db))
                            } else {
                                field_ty.name(db)
                            };
                            debug!(
                                "  -> adt.struct field {} resolved type name: {}",
                                field_idx, type_name
                            );
                            // Convert to resolved types for local allocation.
                            if type_name == Symbol::new("Int") {
                                effective_ty = tribute_rt::int_type(db);
                            } else if type_name == Symbol::new("Nat") {
                                effective_ty = tribute_rt::nat_type(db);
                            } else if type_name == Symbol::new("Float") {
                                effective_ty = tribute_rt::float_type(db);
                            } else if type_name == Symbol::new("Bool") {
                                effective_ty = tribute_rt::bool_type(db);
                            }
                        }

                        // For adt.struct types with generic field types (type_var), fall back to
                        // the GC type table to determine the actual field type at WASM level.
                        // This handles handler lambdas where captured types are generic.
                        if adt::is_struct_type(db, struct_ty)
                            && let Ok(field_idx) = attr_field_idx(op.attributes(db))
                            && tribute::is_type_var(db, effective_ty)
                        {
                            // Look up the struct in the placeholder map using field count
                            if let Some(fields) = adt::get_struct_fields(db, struct_ty) {
                                let field_count = fields.len();
                                let key = (*wasm::Structref::new(db), field_count);
                                if let Some(&type_idx) =
                                    module_info.placeholder_struct_type_idx.get(&key)
                                    && let Some(GcTypeDef::Struct(gc_fields)) =
                                        module_info.gc_types.get(type_idx as usize)
                                    && let Some(field) = gc_fields.get(field_idx as usize)
                                {
                                    if matches!(field.element_type, StorageType::Val(ValType::I32))
                                    {
                                        debug!(
                                            "  -> adt.struct GC fallback: field {} is i32",
                                            field_idx
                                        );
                                        effective_ty = tribute_rt::int_type(db);
                                    } else if matches!(
                                        field.element_type,
                                        StorageType::Val(ValType::I64)
                                    ) {
                                        debug!(
                                            "  -> adt.struct GC fallback: field {} is i64",
                                            field_idx
                                        );
                                        effective_ty = tribute_rt::int_type(db);
                                    } else if matches!(
                                        field.element_type,
                                        StorageType::Val(ValType::F64)
                                    ) {
                                        debug!(
                                            "  -> adt.struct GC fallback: field {} is f64",
                                            field_idx
                                        );
                                        effective_ty = tribute_rt::float_type(db);
                                    }
                                }
                            }
                        }

                        // For placeholder structref types with field_count, look up the GC type
                        // to determine the actual field type. This handles cases where the struct
                        // type is already lowered to a WASM structref placeholder.
                        if wasm::Structref::from_type(db, struct_ty).is_some()
                            && let Ok(field_idx) = attr_field_idx(op.attributes(db))
                            && let Some(Attribute::IntBits(fc)) =
                                op.attributes(db).get(&Symbol::new("field_count"))
                        {
                            let key = (struct_ty, *fc as usize);
                            if let Some(&type_idx) =
                                module_info.placeholder_struct_type_idx.get(&key)
                                && let Some(GcTypeDef::Struct(fields)) =
                                    module_info.gc_types.get(type_idx as usize)
                                && let Some(field) = fields.get(field_idx as usize)
                            {
                                // Map struct field types to tribute_rt types for local allocation.
                                // - i32: Int/Nat (31-bit, current representation)
                                // - i64: Int/Nat (legacy 64-bit representation, kept for compatibility)
                                // - f64: Float
                                if matches!(field.element_type, StorageType::Val(ValType::I32)) {
                                    debug!(
                                        "  -> structref placeholder field {} is i32, using Int type",
                                        field_idx
                                    );
                                    effective_ty = tribute_rt::int_type(db);
                                } else if matches!(
                                    field.element_type,
                                    StorageType::Val(ValType::I64)
                                ) {
                                    // Legacy path: older code may have used i64 for Int/Nat.
                                    // Current tribute_rt.int/nat use 31-bit (i32) representation.
                                    debug!(
                                        "  -> structref placeholder field {} is i64 (legacy), using Int type",
                                        field_idx
                                    );
                                    effective_ty = tribute_rt::int_type(db);
                                } else if matches!(
                                    field.element_type,
                                    StorageType::Val(ValType::F64)
                                ) {
                                    debug!(
                                        "  -> structref placeholder field {} is f64, using Float type",
                                        field_idx
                                    );
                                    effective_ty = tribute_rt::float_type(db);
                                }
                            }
                        }
                    } else {
                        debug!(
                            "struct_get has no type info, result_ty: {}.{}",
                            effective_ty.dialect(db),
                            effective_ty.name(db)
                        );
                    }
                }

                // For wasm.if with type.var result, infer the effective type from the
                // then branch's result value. This ensures the local type matches the
                // actual value produced by the branches.
                if wasm::If::matches(db, *op) && tribute::is_type_var(db, effective_ty) {
                    if let Some(eff_ty) = infer_region_effective_type(db, op, ctx) {
                        debug!(
                            "wasm.if local: using then branch effective type {}.{} instead of IR type {}.{}",
                            eff_ty.dialect(db),
                            eff_ty.name(db),
                            effective_ty.dialect(db),
                            effective_ty.name(db)
                        );
                        effective_ty = eff_ty;
                    } else if let Some(ret_ty) = ctx.func_return_type
                        && !is_polymorphic_type(db, ret_ty)
                    {
                        debug!(
                            "wasm.if local: using function return type {}.{} instead of IR type {}.{}",
                            ret_ty.dialect(db),
                            ret_ty.name(db),
                            effective_ty.dialect(db),
                            effective_ty.name(db)
                        );
                        effective_ty = ret_ty;
                    }
                }

                // For wasm.block with polymorphic result type, infer the effective type from
                // the body region's result value or fall back to Step if function returns it.
                if wasm::Block::matches(db, *op) && is_polymorphic_type(db, effective_ty) {
                    if let Some(eff_ty) = infer_region_effective_type(db, op, ctx) {
                        debug!(
                            "wasm.block local: using body effective type {}.{} instead of IR type {}.{}",
                            eff_ty.dialect(db),
                            eff_ty.name(db),
                            effective_ty.dialect(db),
                            effective_ty.name(db)
                        );
                        effective_ty = eff_ty;
                    } else {
                        let upgraded =
                            upgrade_polymorphic_to_step(db, effective_ty, ctx.func_return_type);
                        if upgraded != effective_ty {
                            debug!(
                                "wasm.block local: using Step instead of polymorphic type {}.{}",
                                effective_ty.dialect(db),
                                effective_ty.name(db)
                            );
                            effective_ty = upgraded;
                        }
                    }
                }

                // For wasm.call_indirect with polymorphic result type in functions that return
                // Step, use Step as the local type. This ensures proper type
                // matching when storing the result of closure/continuation calls.
                if wasm::CallIndirect::matches(db, *op) && is_polymorphic_type(db, effective_ty) {
                    let upgraded =
                        upgrade_polymorphic_to_step(db, effective_ty, ctx.func_return_type);
                    if upgraded != effective_ty {
                        debug!(
                            "wasm.call_indirect local: using Step instead of polymorphic type {}.{}",
                            effective_ty.dialect(db),
                            effective_ty.name(db)
                        );
                        effective_ty = upgraded;
                    }
                }

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
        // Skip tribute.var operations - kept for LSP support, no runtime effect
        if op.dialect(db) == tribute::DIALECT_NAME() && op.name(db) == tribute::VAR() {
            continue;
        }
        // Skip wasm.yield - it's handled by region_result_value + emit_value_get,
        // not emitted as a real Wasm instruction
        if wasm::Yield::from_operation(db, *op).is_ok() {
            continue;
        }
        emit_op(db, op, ctx, module_info, function)?;
    }
    Ok(())
}

/// Check if a type is polymorphic (type_var or anyref).
/// These types need special handling for control flow result types.
fn is_polymorphic_type(db: &dyn salsa::Database, ty: Type<'_>) -> bool {
    tribute::is_type_var(db, ty) || wasm::Anyref::from_type(db, ty).is_some()
}

/// Try to infer a concrete effective type from a control flow operation's first region.
/// Returns the inferred type if it's more concrete than type_var/anyref, None otherwise.
fn infer_region_effective_type<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
    ctx: &FunctionEmitContext<'db>,
) -> Option<Type<'db>> {
    let region = op.regions(db).first()?;
    let result_value = region_result_value(db, region)?;

    // First try effective_types (populated during function setup)
    #[allow(clippy::collapsible_if)]
    if let Some(&ty) = ctx.effective_types.get(&result_value) {
        if !is_polymorphic_type(db, ty) {
            return Some(ty);
        }
    }

    // If not in effective_types, try to get the type from the value's definition
    // This handles remapped operations in resume functions
    #[allow(clippy::collapsible_if)]
    if let ValueDef::OpResult(def_op) = result_value.def(db) {
        if let Some(result_ty) = def_op.results(db).get(result_value.index(db)).copied() {
            if !is_polymorphic_type(db, result_ty) {
                return Some(result_ty);
            }
        }
    }

    None
}

/// Check if a type is the Step marker type.
/// Upgrade a polymorphic type to Step if the function returns Step.
/// Used for wasm.block/wasm.loop result types.
fn upgrade_polymorphic_to_step<'db>(
    db: &'db dyn salsa::Database,
    ty: Type<'db>,
    func_return_type: Option<Type<'db>>,
) -> Type<'db> {
    if !is_polymorphic_type(db, ty) {
        return ty;
    }

    if let Some(ret_ty) = func_return_type
        && is_step_type(db, ret_ty)
    {
        return crate::gc_types::step_marker_type(db);
    }
    ty
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

fn emit_value_get<'db>(
    value: Value<'db>,
    ctx: &FunctionEmitContext<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let index = ctx
        .value_locals
        .get(&value)
        .ok_or_else(|| CompilationError::invalid_module("value missing local mapping"))?;
    function.instruction(&Instruction::LocalGet(*index));
    Ok(())
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

    // Helper to get type_idx from attributes or inferred type.
    // Priority: type_idx attr > type attr > inferred_type (from result/operand)
    let get_type_idx_from_attrs = |attrs: &std::collections::BTreeMap<Symbol, Attribute<'db>>,
                                   inferred_type: Option<Type<'db>>|
     -> Option<u32> {
        // First try type_idx attribute
        if let Some(Attribute::IntBits(idx)) = attrs.get(&ATTR_TYPE_IDX()) {
            return Some(*idx as u32);
        }
        // Fall back to type attribute (legacy, will be removed)
        if let Some(Attribute::Type(ty)) = attrs.get(&ATTR_TYPE()) {
            // Special case: _closure struct type uses builtin CLOSURE_STRUCT_IDX
            if is_closure_struct_type(db, *ty) {
                return Some(CLOSURE_STRUCT_IDX);
            }
            return module_info.type_idx_by_type.get(ty).copied();
        }
        // Fall back to inferred type
        if let Some(ty) = inferred_type {
            // Special case: _closure struct type uses builtin CLOSURE_STRUCT_IDX
            if is_closure_struct_type(db, ty) {
                return Some(CLOSURE_STRUCT_IDX);
            }
            return module_info.type_idx_by_type.get(&ty).copied();
        }
        None
    };

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
        return handle_i32_const(db, const_op, ctx, function);
    } else if let Ok(const_op) = wasm::I64Const::from_operation(db, *op) {
        return handle_i64_const(db, const_op, ctx, function);
    } else if let Ok(const_op) = wasm::F32Const::from_operation(db, *op) {
        return handle_f32_const(db, const_op, ctx, function);
    } else if let Ok(const_op) = wasm::F64Const::from_operation(db, *op) {
        return handle_f64_const(db, const_op, ctx, function);
    } else if wasm::If::matches(db, *op) {
        let result_ty = op.results(db).first().copied();

        // First try to infer effective type from branches
        let branch_eff_ty = infer_region_effective_type(db, op, ctx);

        // Check if we can actually get a result value from the then region
        let then_region_result = op
            .regions(db)
            .first()
            .and_then(|r| region_result_value(db, r));
        let then_has_result_value = then_region_result.is_some();

        // Check if function returns Step - if so, if blocks should also produce Step
        let func_returns_step = ctx
            .func_return_type
            .map(|ty| is_step_type(db, ty))
            .unwrap_or(false);

        debug!(
            "wasm.if: then_has_result_value={}, branch_eff_ty={:?}, func_returns_step={}, result_ty={:?}",
            then_has_result_value,
            branch_eff_ty.map(|ty| format!("{}.{}", ty.dialect(db), ty.name(db))),
            func_returns_step,
            result_ty.map(|ty| format!("{}.{}", ty.dialect(db), ty.name(db)))
        );

        // Determine if we should use a result type
        // Only set has_result if we can actually find a result value
        let has_result = if !then_has_result_value {
            false
        } else if let Some(eff_ty) = branch_eff_ty {
            !is_nil_type(db, eff_ty)
        } else if func_returns_step && result_ty.is_some() {
            // Function returns Step, so if with result should produce Step
            true
        } else {
            matches!(result_ty, Some(ty) if !is_nil_type(db, ty))
        };

        // For wasm.if with results, we need to determine the actual block type.
        // If the IR result type is polymorphic or nil but the effective result type
        // from the then/else branches is concrete, we must use the effective type.
        let effective_ty = if has_result {
            // Try branch effective type first (handles Step in resume functions)
            if let Some(eff_ty) = branch_eff_ty {
                if !is_nil_type(db, eff_ty) {
                    debug!(
                        "wasm.if: using then branch effective type {}.{}",
                        eff_ty.dialect(db),
                        eff_ty.name(db)
                    );
                    Some(eff_ty)
                } else {
                    result_ty
                }
            } else if func_returns_step {
                // Function returns Step, use Step as the block type
                debug!("wasm.if: using Step type because function returns Step");
                Some(crate::gc_types::step_marker_type(db))
            } else if let Some(ir_ty) = result_ty {
                if tribute::is_type_var(db, ir_ty) {
                    // Fallback to function return type for polymorphic IR types
                    if let Some(ret_ty) = ctx.func_return_type {
                        if !is_polymorphic_type(db, ret_ty) {
                            debug!(
                                "wasm.if: using function return type {}.{} instead of type_var",
                                ret_ty.dialect(db),
                                ret_ty.name(db)
                            );
                            Some(ret_ty)
                        } else {
                            Some(ir_ty)
                        }
                    } else {
                        Some(ir_ty)
                    }
                } else {
                    Some(ir_ty)
                }
            } else {
                None
            }
        } else {
            None
        };

        let block_type = if has_result {
            let eff_ty = effective_ty.expect("effective_ty should be Some when has_result is true");
            // IMPORTANT: Check core.func BEFORE type_idx_by_type lookup.
            // core.func types should always use funcref block type, not concrete struct types.
            if core::Func::from_type(db, eff_ty).is_some() {
                debug!(
                    "wasm.if block_type: using funcref for core.func type {}.{}",
                    eff_ty.dialect(db),
                    eff_ty.name(db)
                );
                BlockType::Result(ValType::Ref(RefType::FUNCREF))
            } else if let Some(&type_idx) = module_info.type_idx_by_type.get(&eff_ty) {
                // ADT types - use concrete GC type reference
                debug!(
                    "wasm.if block_type: using concrete type_idx={} for {}.{}",
                    type_idx,
                    eff_ty.dialect(db),
                    eff_ty.name(db)
                );
                BlockType::Result(ValType::Ref(RefType {
                    nullable: true,
                    heap_type: HeapType::Concrete(type_idx),
                }))
            } else {
                debug!(
                    "wasm.if block_type: no type_idx for {}.{}, using type_to_valtype",
                    eff_ty.dialect(db),
                    eff_ty.name(db)
                );
                BlockType::Result(type_to_valtype(db, eff_ty, &module_info.type_idx_by_type)?)
            }
        } else {
            BlockType::Empty
        };
        if operands.len() != 1 {
            return Err(CompilationError::invalid_module(
                "wasm.if expects a single condition operand",
            ));
        }
        emit_operands(db, operands, ctx, &module_info.block_arg_types, function)?;
        function.instruction(&Instruction::If(block_type));
        let regions = op.regions(db);
        let then_region = regions
            .first()
            .ok_or_else(|| CompilationError::invalid_module("wasm.if missing then region"))?;
        let then_result = if has_result {
            Some(region_result_value(db, then_region).ok_or_else(|| {
                CompilationError::invalid_module("wasm.if then region missing result value")
            })?)
        } else {
            None
        };
        emit_region_ops(db, then_region, ctx, module_info, function)?;
        if let Some(value) = then_result {
            emit_value_get(value, ctx, function)?;
            // If the value's effective type is anyref/type_var but the block expects
            // a specific type, cast the result.
            if let (Some(eff_ty), Some(value_ty)) = (effective_ty, ctx.effective_types.get(&value))
                && (tribute::is_type_var(db, *value_ty)
                    || wasm::Anyref::from_type(db, *value_ty).is_some())
            {
                if core::Func::from_type(db, eff_ty).is_some() {
                    // core.func types need cast to funcref (abstract type)
                    debug!("wasm.if then: casting anyref branch result to funcref");
                    function.instruction(&Instruction::RefCastNullable(HeapType::Abstract {
                        shared: false,
                        ty: AbstractHeapType::Func,
                    }));
                } else if let Some(&type_idx) = module_info.type_idx_by_type.get(&eff_ty) {
                    // ADT types need cast to concrete struct type
                    debug!(
                        "wasm.if then: casting anyref branch result to (ref null {})",
                        type_idx
                    );
                    function
                        .instruction(&Instruction::RefCastNullable(HeapType::Concrete(type_idx)));
                }
            }
        }
        if let Some(else_region) = regions.get(1) {
            let else_result = if has_result {
                Some(region_result_value(db, else_region).ok_or_else(|| {
                    CompilationError::invalid_module("wasm.if else region missing result value")
                })?)
            } else {
                None
            };
            function.instruction(&Instruction::Else);
            emit_region_ops(db, else_region, ctx, module_info, function)?;
            if let Some(value) = else_result {
                emit_value_get(value, ctx, function)?;
                // Cast else branch result if needed (same logic as then branch)
                if let (Some(eff_ty), Some(value_ty)) =
                    (effective_ty, ctx.effective_types.get(&value))
                    && (tribute::is_type_var(db, *value_ty)
                        || wasm::Anyref::from_type(db, *value_ty).is_some())
                {
                    if core::Func::from_type(db, eff_ty).is_some() {
                        // core.func types need cast to funcref (abstract type)
                        debug!("wasm.if else: casting anyref branch result to funcref");
                        function.instruction(&Instruction::RefCastNullable(HeapType::Abstract {
                            shared: false,
                            ty: AbstractHeapType::Func,
                        }));
                    } else if let Some(&type_idx) = module_info.type_idx_by_type.get(&eff_ty) {
                        // ADT types need cast to concrete struct type
                        debug!(
                            "wasm.if else: casting anyref branch result to (ref null {})",
                            type_idx
                        );
                        function.instruction(&Instruction::RefCastNullable(HeapType::Concrete(
                            type_idx,
                        )));
                    }
                }
            }
        } else if has_result {
            return Err(CompilationError::invalid_module(
                "wasm.if with result requires else region",
            ));
        }
        function.instruction(&Instruction::End);
        if has_result {
            set_result_local(db, op, ctx, function)?;
        }
    } else if wasm::Block::matches(db, *op) {
        // Upgrade polymorphic block result type to Step if function returns Step
        let result_ty = op
            .results(db)
            .first()
            .map(|ty| upgrade_polymorphic_to_step(db, *ty, ctx.func_return_type));
        let has_result = matches!(result_ty, Some(ty) if !is_nil_type(db, ty));
        let block_type = if has_result {
            BlockType::Result(type_to_valtype(
                db,
                result_ty.expect("block result type"),
                &module_info.type_idx_by_type,
            )?)
        } else {
            BlockType::Empty
        };
        function.instruction(&Instruction::Block(block_type));
        let region = op
            .regions(db)
            .first()
            .ok_or_else(|| CompilationError::invalid_module("wasm.block missing body region"))?;
        emit_region_ops(db, region, ctx, module_info, function)?;
        if has_result {
            let value = region_result_value(db, region).ok_or_else(|| {
                CompilationError::invalid_module("wasm.block body missing result value")
            })?;
            emit_value_get(value, ctx, function)?;
        }
        function.instruction(&Instruction::End);
        if has_result {
            set_result_local(db, op, ctx, function)?;
        }
    } else if wasm::Loop::matches(db, *op) {
        // Upgrade polymorphic loop result type to Step if function returns Step
        let result_ty = op
            .results(db)
            .first()
            .map(|ty| upgrade_polymorphic_to_step(db, *ty, ctx.func_return_type));
        let has_result = matches!(result_ty, Some(ty) if !is_nil_type(db, ty));
        let block_type = if has_result {
            BlockType::Result(type_to_valtype(
                db,
                result_ty.expect("loop result type"),
                &module_info.type_idx_by_type,
            )?)
        } else {
            BlockType::Empty
        };
        function.instruction(&Instruction::Loop(block_type));
        let region = op
            .regions(db)
            .first()
            .ok_or_else(|| CompilationError::invalid_module("wasm.loop missing body region"))?;
        emit_region_ops(db, region, ctx, module_info, function)?;
        if has_result {
            let value = region_result_value(db, region).ok_or_else(|| {
                CompilationError::invalid_module("wasm.loop body missing result value")
            })?;
            emit_value_get(value, ctx, function)?;
        }
        function.instruction(&Instruction::End);
        if has_result {
            set_result_local(db, op, ctx, function)?;
        }
    } else if let Ok(br_op) = wasm::Br::from_operation(db, *op) {
        let depth = br_op.target(db);
        function.instruction(&Instruction::Br(depth));
    } else if let Ok(br_if_op) = wasm::BrIf::from_operation(db, *op) {
        if operands.len() != 1 {
            return Err(CompilationError::invalid_module(
                "wasm.br_if expects a single condition operand",
            ));
        }
        emit_operands(db, operands, ctx, &module_info.block_arg_types, function)?;
        let depth = br_if_op.target(db);
        function.instruction(&Instruction::BrIf(depth));
    } else if let Ok(call_op) = wasm::Call::from_operation(db, *op) {
        let callee = call_op.callee(db);
        let target = resolve_callee(callee, module_info)?;

        // Check if we need boxing for generic function calls
        if let Some(callee_ty) = module_info.func_types.get(&callee) {
            let param_types = callee_ty.params(db);
            emit_operands_with_boxing(db, operands, &param_types, ctx, module_info, function)?;
        } else {
            emit_operands(db, operands, ctx, &module_info.block_arg_types, function)?;
        }

        function.instruction(&Instruction::Call(target));

        // Check if we need unboxing for the return value
        if let Some(callee_ty) = module_info.func_types.get(&callee) {
            let return_ty = callee_ty.result(db);
            // If callee returns anyref (type.var), we need to unbox to the expected concrete type.
            // Since type inference doesn't propagate instantiated types to the IR,
            // we infer the result type from the first operand's type (works for identity-like functions).
            if tribute::is_type_var(db, return_ty)
                && !module_info.type_idx_by_type.contains_key(&return_ty)
            {
                // Try to infer concrete type from first operand
                if let Some(operand_ty) = operands
                    .first()
                    .and_then(|v| value_type(db, *v, &module_info.block_arg_types))
                    .filter(|ty| !tribute::is_type_var(db, *ty))
                {
                    emit_unboxing(db, operand_ty, function)?;
                }
            }
        }

        set_result_local(db, op, ctx, function)?;
    } else if wasm::CallIndirect::matches(db, *op) {
        // wasm.call_indirect: indirect function call
        // Operands: [arg1, arg2, ..., argN, funcref]
        // The funcref is the last operand (on top of stack in WebAssembly)
        //
        // If the last operand is a reference type (funcref/anyref), we use call_ref.
        // If it's an i32 (table index), we use the traditional call_indirect.

        if operands.is_empty() {
            return Err(CompilationError::invalid_module(
                "wasm.call_indirect requires at least a funcref operand",
            ));
        }

        // In func.call_indirect IR, the callee (funcref) is the FIRST operand, followed by args.
        // But WebAssembly expects: [args..., funcref/table_idx] with the call target last on stack.
        // Check the first operand to determine if we have a funcref (use call_ref) or i32 (use call_indirect).
        let first_operand = operands.first().copied().unwrap();
        let first_operand_ty = value_type(db, first_operand, &module_info.block_arg_types);
        debug!(
            "call_indirect: first_operand_ty={:?}",
            first_operand_ty.map(|ty| {
                ty.dialect(db)
                    .with_str(|d| ty.name(db).with_str(|n| format!("{}.{}", d, n)))
            })
        );
        // Debug: trace the value definition
        match first_operand.def(db) {
            ValueDef::OpResult(def_op) => {
                debug!(
                    "call_indirect: first_operand defined by {}.{}, results={:?}",
                    def_op.dialect(db),
                    def_op.name(db),
                    def_op
                        .results(db)
                        .iter()
                        .map(|t| {
                            t.dialect(db)
                                .with_str(|d| t.name(db).with_str(|n| format!("{}.{}", d, n)))
                        })
                        .collect::<Vec<_>>()
                );
            }
            ValueDef::BlockArg(block_id) => {
                debug!(
                    "call_indirect: first_operand is block arg from block {:?} idx {}",
                    block_id,
                    first_operand.index(db)
                );
            }
        }
        let is_ref_type = first_operand_ty.is_some_and(|ty| {
            let is_funcref = wasm::Funcref::from_type(db, ty).is_some();
            let is_anyref = wasm::Anyref::from_type(db, ty).is_some();
            let is_core_func = core::Func::from_type(db, ty).is_some();
            // Check if this is a closure struct (adt.struct with name "_closure")
            // Closure structs contain (funcref, anyref) and are used for call_indirect
            let is_closure_struct = is_closure_struct_type(db, ty);
            debug!(
                "call_indirect: is_funcref={}, is_anyref={}, is_core_func={}, is_closure_struct={}",
                is_funcref, is_anyref, is_core_func, is_closure_struct
            );
            is_funcref || is_anyref || is_core_func || is_closure_struct
        });
        debug!("call_indirect: is_ref_type={}", is_ref_type);

        // Build parameter types (all operands except first which is funcref)
        // Normalize IR types to wasm types - primitive IR types that might be boxed
        // (in polymorphic handlers) should use anyref.
        let anyref_ty = wasm::Anyref::new(db).as_type();
        let normalize_param_type = |ty: Type<'db>| -> Type<'db> {
            if tribute_rt::is_int(db, ty)
                || tribute_rt::is_nat(db, ty)
                || tribute_rt::is_bool(db, ty)
                || tribute_rt::is_float(db, ty)
                || tribute::is_type_var(db, ty)
                || core::Nil::from_type(db, ty).is_some()
            {
                anyref_ty
            } else {
                ty
            }
        };
        let param_types: Vec<Type<'db>> = operands
            .iter()
            .skip(1)
            .filter_map(|v| value_type(db, *v, &module_info.block_arg_types))
            .map(normalize_param_type)
            .collect();

        // Get result type - use enclosing function's return type if it's funcref
        // and the call_indirect has anyref result. This is needed because
        // WebAssembly GC has separate type hierarchies for anyref and funcref,
        // so we can't cast between them.
        let mut result_ty = op.results(db).first().copied().ok_or_else(|| {
            CompilationError::invalid_module("wasm.call_indirect must have a result type")
        })?;

        // If result type is anyref/type_var but enclosing function returns funcref or Step,
        // upgrade the result type accordingly. This is needed because WebAssembly GC has separate
        // type hierarchies, and effectful functions return Step for yield bubbling.
        let funcref_ty = wasm::Funcref::new(db).as_type();
        if let Some(func_ret_ty) = ctx.func_return_type {
            let is_anyref_result = wasm::Anyref::from_type(db, result_ty).is_some();
            let is_type_var_result = result_ty.dialect(db) == Symbol::new("tribute")
                && result_ty.name(db) == Symbol::new("type_var");
            let is_polymorphic_result = is_anyref_result || is_type_var_result;
            let func_returns_funcref = wasm::Funcref::from_type(db, func_ret_ty).is_some()
                || core::Func::from_type(db, func_ret_ty).is_some();
            // Check for Step type (trampoline-based effect system)
            let func_returns_step = is_step_type(db, func_ret_ty);
            if is_polymorphic_result && func_returns_funcref {
                debug!(
                    "call_indirect emit: upgrading polymorphic result to funcref for enclosing function"
                );
                result_ty = funcref_ty;
            } else if is_polymorphic_result && func_returns_step {
                debug!(
                    "call_indirect emit: upgrading polymorphic result to Step for enclosing function"
                );
                result_ty = crate::gc_types::step_marker_type(db);
            }
        }

        // Construct function type
        let func_type =
            core::Func::new(db, param_types.clone().into_iter().collect(), result_ty).as_type();

        debug!(
            "call_indirect emit: looking up func_type with result={}.{}",
            result_ty.dialect(db),
            result_ty.name(db)
        );

        // Get or compute type_idx
        let type_idx = match attr_u32(op.attributes(db), Symbol::new("type_idx")) {
            Ok(idx) => {
                debug!("call_indirect emit: using type_idx from attribute: {}", idx);
                idx
            }
            Err(_) => {
                // Look up type index
                let idx = module_info
                    .type_idx_by_type
                    .get(&func_type)
                    .copied()
                    .ok_or_else(|| {
                        debug!(
                            "call_indirect emit: func_type not found in type_idx_by_type! func_type={:?}",
                            func_type
                        );
                        CompilationError::invalid_module(
                            "wasm.call_indirect function type not registered in type section",
                        )
                    })?;
                debug!("call_indirect emit: looked up type_idx: {}", idx);
                idx
            }
        };

        if is_ref_type {
            // Use call_ref for typed function references
            // IR operand order: [funcref, arg1, arg2, ...]
            // WebAssembly stack order: [arg1, arg2, ..., funcref]
            // So we emit args first (operands[1..]), then funcref (operands[0])

            // Emit arguments (all operands except first funcref)
            for (i, operand) in operands.iter().skip(1).enumerate() {
                debug!(
                    "call_indirect: emitting arg {} of type {:?}",
                    i,
                    value_type(db, *operand, &module_info.block_arg_types).map(|t| t
                        .dialect(db)
                        .with_str(|d| t.name(db).with_str(|n| format!("{}.{}", d, n))))
                );
                match operand.def(db) {
                    ValueDef::OpResult(def_op) => {
                        debug!(
                            "  arg {} defined by {}.{}",
                            i,
                            def_op.dialect(db),
                            def_op.name(db)
                        );
                    }
                    ValueDef::BlockArg(block_id) => {
                        debug!("  arg {} is block arg from {:?}", i, block_id);
                    }
                }
                emit_value(db, *operand, ctx, function)?;
            }

            // Emit the funcref (first operand)
            emit_value(db, first_operand, ctx, function)?;

            // Cast anyref/closure struct to typed function reference if needed
            // Closure struct (adt.struct with name "_closure") contains funcref in field 0.
            // When we extract the funcref via struct_get, the IR type may still be adt.struct,
            // but the actual wasm value is funcref. Cast to the concrete function type.
            if let Some(ty) = first_operand_ty
                && (wasm::Anyref::from_type(db, ty).is_some()
                    || core::Func::from_type(db, ty).is_some()
                    || is_closure_struct_type(db, ty))
            {
                // Cast to (ref null func_type)
                function.instruction(&Instruction::RefCastNullable(HeapType::Concrete(type_idx)));
            }

            // Emit call_ref with the function type index
            function.instruction(&Instruction::CallRef(type_idx));

            // The call_ref returns the declared result type of the called function.
            // But the local where we store the result may have a different (concrete) type.
            // We need to cast the result to match the local's type.
            //
            // Note: This is a workaround for unresolved type variables (tribute.type_var)
            // in the IR. The proper fix would be to resolve types earlier in the pipeline.
        } else {
            // Traditional call_indirect with i32 table index
            let table = attr_u32(op.attributes(db), Symbol::new("table")).unwrap_or(0);

            // Emit all operands (arguments first, then table index)
            emit_operands(db, operands, ctx, &module_info.block_arg_types, function)?;

            function.instruction(&Instruction::CallIndirect {
                type_index: type_idx,
                table_index: table,
            });
        }

        set_result_local(db, op, ctx, function)?;
    } else if let Ok(return_call_op) = wasm::ReturnCall::from_operation(db, *op) {
        let callee = return_call_op.callee(db);
        let target = resolve_callee(callee, module_info)?;

        // Check if we need boxing for generic function calls
        if let Some(callee_ty) = module_info.func_types.get(&callee) {
            let param_types = callee_ty.params(db);
            emit_operands_with_boxing(db, operands, &param_types, ctx, module_info, function)?;
        } else {
            emit_operands(db, operands, ctx, &module_info.block_arg_types, function)?;
        }

        // Note: Return unboxing is not needed for tail calls since
        // the caller's return type should match the callee's.
        function.instruction(&Instruction::ReturnCall(target));
    } else if let Ok(local_op) = wasm::LocalGet::from_operation(db, *op) {
        let index = local_op.index(db);
        function.instruction(&Instruction::LocalGet(index));
        set_result_local(db, op, ctx, function)?;
    } else if let Ok(local_op) = wasm::LocalSet::from_operation(db, *op) {
        let index = local_op.index(db);
        emit_operands(db, operands, ctx, &module_info.block_arg_types, function)?;
        function.instruction(&Instruction::LocalSet(index));
    } else if let Ok(local_op) = wasm::LocalTee::from_operation(db, *op) {
        let index = local_op.index(db);
        emit_operands(db, operands, ctx, &module_info.block_arg_types, function)?;
        function.instruction(&Instruction::LocalTee(index));
        set_result_local(db, op, ctx, function)?;
    } else if let Ok(global_op) = wasm::GlobalGet::from_operation(db, *op) {
        let index = global_op.index(db);
        function.instruction(&Instruction::GlobalGet(index));
        set_result_local(db, op, ctx, function)?;
    } else if let Ok(global_op) = wasm::GlobalSet::from_operation(db, *op) {
        let index = global_op.index(db);
        emit_operands(db, operands, ctx, &module_info.block_arg_types, function)?;
        function.instruction(&Instruction::GlobalSet(index));
    } else if wasm::StructNew::matches(db, *op) {
        // struct_new needs all field values on the stack, including nil types.
        // emit_operands handles nil types by emitting ref.null none.
        emit_operands(db, operands, ctx, &module_info.block_arg_types, function)?;
        let attrs = op.attributes(db);
        let field_count = operands.len();
        let result_type = op.results(db).first().copied();

        // Check if this uses a placeholder type (wasm.structref)
        // Priority: explicit type attr > placeholder result type > type_idx attr > inferred result type
        let type_idx = if let Some(Attribute::Type(ty)) = attrs.get(&ATTR_TYPE()) {
            if wasm::Structref::from_type(db, *ty).is_some() {
                // Use placeholder map for wasm.structref
                // All (type, field_count) pairs are registered by collect_gc_types upfront
                module_info
                    .placeholder_struct_type_idx
                    .get(&(*ty, field_count))
                    .copied()
            } else if is_closure_struct_type(db, *ty) {
                // Special case: _closure struct type uses builtin CLOSURE_STRUCT_IDX
                Some(CLOSURE_STRUCT_IDX)
            } else {
                // Regular type
                module_info.type_idx_by_type.get(ty).copied()
            }
        } else if let Some(ty) = result_type
            && wasm::Structref::from_type(db, ty).is_some()
        {
            // Result type is a placeholder (wasm.structref) - use placeholder map
            // This takes precedence over explicit type_idx=0 for placeholder types
            module_info
                .placeholder_struct_type_idx
                .get(&(ty, field_count))
                .copied()
        } else if let Some(Attribute::IntBits(idx)) = attrs.get(&ATTR_TYPE_IDX()) {
            Some(*idx as u32)
        } else if let Some(ty) = result_type {
            // Special case: _closure struct type uses builtin CLOSURE_STRUCT_IDX
            if is_closure_struct_type(db, ty) {
                Some(CLOSURE_STRUCT_IDX)
            } else {
                // Infer type from result type (non-placeholder)
                module_info.type_idx_by_type.get(&ty).copied()
            }
        } else {
            None
        }
        .ok_or_else(|| CompilationError::missing_attribute("type or type_idx"))?;

        function.instruction(&Instruction::StructNew(type_idx));
        set_result_local(db, op, ctx, function)?;
    } else if wasm::StructGet::matches(db, *op) {
        emit_operands(db, operands, ctx, &module_info.block_arg_types, function)?;
        let attrs = op.attributes(db);

        // Check if operand is anyref and needs casting to struct type
        // This happens when a closure is captured (stored as anyref) and later used
        let operand_is_anyref = operands
            .first()
            .and_then(|op_val| {
                let ty = value_type(db, *op_val, &module_info.block_arg_types)?;
                if wasm::Anyref::from_type(db, ty).is_some() {
                    Some(true)
                } else {
                    None
                }
            })
            .unwrap_or(false);

        // CRITICAL: For struct_get, the type_idx MUST match the operand's actual type.
        // We need to trace through ref.cast operations to find the actual type,
        // because the IR type might be different from the wasm type after casting.
        let operand = operands.first().copied();
        let type_idx = if let Some(op_val) = operand {
            // Check if the operand was defined by a ref.cast - if so, use its target type
            if let ValueDef::OpResult(def_op) = op_val.def(db) {
                debug!(
                    "struct_get: operand defined by {}.{} at index {}",
                    def_op.dialect(db),
                    def_op.name(db),
                    op_val.index(db)
                );
                if wasm::RefCast::matches(db, def_op) {
                    // Get the ref.cast's target_type and field_count
                    let def_attrs = def_op.attributes(db);
                    if let Some(Attribute::Type(target_ty)) = def_attrs.get(&ATTR_TARGET_TYPE()) {
                        // For placeholder types like wasm.structref, we MUST use field_count
                        // to distinguish between different concrete types with same abstract type.
                        let is_placeholder = wasm::Structref::from_type(db, *target_ty).is_some();

                        if is_placeholder {
                            // Use placeholder lookup with field_count
                            let field_count = if let Some(Attribute::IntBits(fc)) =
                                def_attrs.get(&Symbol::new("field_count"))
                            {
                                debug!(
                                    "struct_get: ref_cast (placeholder) has field_count={}",
                                    *fc
                                );
                                *fc as usize
                            } else {
                                debug!("struct_get: ref_cast (placeholder) has NO field_count!");
                                // Last resort - use struct_get's type attr
                                if let Some(Attribute::Type(ty)) = attrs.get(&ATTR_TYPE()) {
                                    adt::get_struct_fields(db, *ty)
                                        .map(|f| f.len())
                                        .unwrap_or(0)
                                } else {
                                    0
                                }
                            };
                            debug!(
                                "struct_get: looking up placeholder ({}.{}, field_count={})",
                                target_ty.dialect(db),
                                target_ty.name(db),
                                field_count
                            );
                            let result = module_info
                                .placeholder_struct_type_idx
                                .get(&(*target_ty, field_count))
                                .copied();
                            debug!("struct_get: placeholder lookup result = {:?}", result);
                            result
                        } else if let Some(&idx) = module_info.type_idx_by_type.get(target_ty) {
                            // Non-placeholder concrete type - use direct lookup
                            debug!(
                                "struct_get: using ref_cast direct type_idx={} for {}.{}",
                                idx,
                                target_ty.dialect(db),
                                target_ty.name(db)
                            );
                            Some(idx)
                        } else {
                            // Non-placeholder but not found - try placeholder lookup as fallback
                            let field_count = if let Some(Attribute::IntBits(fc)) =
                                def_attrs.get(&Symbol::new("field_count"))
                            {
                                *fc as usize
                            } else if let Some(Attribute::Type(ty)) = attrs.get(&ATTR_TYPE()) {
                                adt::get_struct_fields(db, *ty)
                                    .map(|f| f.len())
                                    .unwrap_or(0)
                            } else {
                                0
                            };
                            module_info
                                .placeholder_struct_type_idx
                                .get(&(*target_ty, field_count))
                                .copied()
                        }
                    } else {
                        // No target_type attr on ref_cast, fall back
                        debug!("struct_get: ref_cast has NO target_type attribute!");
                        let inferred_type = value_type(db, op_val, &module_info.block_arg_types);
                        get_type_idx_from_attrs(attrs, inferred_type)
                    }
                } else {
                    // Not a ref_cast, use normal lookup
                    let inferred_type = value_type(db, op_val, &module_info.block_arg_types);
                    if let Some(inferred) = inferred_type {
                        // Special case: _closure struct type uses builtin CLOSURE_STRUCT_IDX
                        if is_closure_struct_type(db, inferred) {
                            debug!("struct_get: using CLOSURE_STRUCT_IDX for _closure type");
                            Some(CLOSURE_STRUCT_IDX)
                        } else if let Some(&idx) = module_info.type_idx_by_type.get(&inferred) {
                            debug!(
                                "struct_get: using operand's type_idx={} for {}.{}",
                                idx,
                                inferred.dialect(db),
                                inferred.name(db)
                            );
                            Some(idx)
                        } else {
                            get_type_idx_from_attrs(attrs, Some(inferred))
                        }
                    } else {
                        get_type_idx_from_attrs(attrs, inferred_type)
                    }
                }
            } else {
                // Block arg - use normal lookup
                let inferred_type = value_type(db, op_val, &module_info.block_arg_types);
                if let Some(inferred) = inferred_type {
                    // Special case: _closure struct type uses builtin CLOSURE_STRUCT_IDX
                    if is_closure_struct_type(db, inferred) {
                        debug!("struct_get: using CLOSURE_STRUCT_IDX for block arg _closure type");
                        Some(CLOSURE_STRUCT_IDX)
                    } else if let Some(&idx) = module_info.type_idx_by_type.get(&inferred) {
                        debug!(
                            "struct_get: using block arg type_idx={} for {}.{}",
                            idx,
                            inferred.dialect(db),
                            inferred.name(db)
                        );
                        Some(idx)
                    } else {
                        get_type_idx_from_attrs(attrs, Some(inferred))
                    }
                } else {
                    get_type_idx_from_attrs(attrs, inferred_type)
                }
            }
        } else {
            debug!("struct_get: no operand, using fallback");
            get_type_idx_from_attrs(attrs, None)
        }
        .ok_or_else(|| CompilationError::missing_attribute("type or type_idx"))?;

        let field_idx = attr_field_idx(attrs)?;
        debug!(
            "struct_get: emitting StructGet with type_idx={}, field_idx={}, operand_is_anyref={}",
            type_idx, field_idx, operand_is_anyref
        );

        // If operand was anyref (from closure capture), cast it to the struct type first
        if operand_is_anyref {
            debug!("struct_get: casting anyref to struct type_idx={}", type_idx);
            function.instruction(&Instruction::RefCastNullable(HeapType::Concrete(type_idx)));
        }

        function.instruction(&Instruction::StructGet {
            struct_type_index: type_idx,
            field_index: field_idx,
        });

        // Check if boxing is needed: result local expects anyref but struct field is i64
        // This happens when extracting values from structs where the IR uses generic/wrapper
        // types but the actual struct field contains a primitive (i64).
        let needs_boxing = if !op.results(db).is_empty() {
            let result_value = op.result(db, 0);

            // Check if the result local would be anyref by examining the effective type
            let local_type = ctx
                .effective_types
                .get(&result_value)
                .copied()
                .or_else(|| op.results(db).first().copied());

            let expects_anyref = local_type
                .map(|ty| wasm::Anyref::from_type(db, ty).is_some() || tribute::is_type_var(db, ty))
                .unwrap_or(false);

            // Check if the struct field is i64
            let field_is_i64 = module_info
                .gc_types
                .get(type_idx as usize)
                .map(|gc_type| {
                    if let GcTypeDef::Struct(fields) = gc_type {
                        fields
                            .get(field_idx as usize)
                            .map(|field| {
                                matches!(field.element_type, StorageType::Val(ValType::I64))
                            })
                            .unwrap_or(false)
                    } else {
                        false
                    }
                })
                .unwrap_or(false);

            debug!(
                "struct_get boxing check: expects_anyref={}, field_is_i64={}, type_idx={}, field_idx={}",
                expects_anyref, field_is_i64, type_idx, field_idx
            );

            expects_anyref && field_is_i64
        } else {
            false
        };

        if needs_boxing {
            debug!("struct_get: boxing i64 to i31ref for anyref local");
            function.instruction(&Instruction::I32WrapI64);
            function.instruction(&Instruction::RefI31);
        }

        set_result_local(db, op, ctx, function)?;
    } else if let Ok(struct_set_op) = wasm::StructSet::from_operation(db, *op) {
        emit_operands(db, operands, ctx, &module_info.block_arg_types, function)?;
        // Infer type from operand[0] (the struct ref)
        let inferred_type = operands
            .first()
            .and_then(|v| value_type(db, *v, &module_info.block_arg_types));
        let type_idx = get_type_idx_from_attrs(op.attributes(db), inferred_type)
            .ok_or_else(|| CompilationError::missing_attribute("type or type_idx"))?;
        let field_idx = struct_set_op.field_idx(db);
        function.instruction(&Instruction::StructSet {
            struct_type_index: type_idx,
            field_index: field_idx,
        });
    } else if let Ok(array_new_op) = wasm::ArrayNew::from_operation(db, *op) {
        return handle_array_new(db, array_new_op, ctx, module_info, function);
    } else if let Ok(array_new_default_op) = wasm::ArrayNewDefault::from_operation(db, *op) {
        return handle_array_new_default(db, array_new_default_op, ctx, module_info, function);
    } else if let Ok(array_get_op) = wasm::ArrayGet::from_operation(db, *op) {
        return handle_array_get(db, array_get_op, ctx, module_info, function);
    } else if let Ok(array_get_s_op) = wasm::ArrayGetS::from_operation(db, *op) {
        return handle_array_get_s(db, array_get_s_op, ctx, module_info, function);
    } else if let Ok(array_get_u_op) = wasm::ArrayGetU::from_operation(db, *op) {
        return handle_array_get_u(db, array_get_u_op, ctx, module_info, function);
    } else if let Ok(array_set_op) = wasm::ArraySet::from_operation(db, *op) {
        return handle_array_set(db, array_set_op, ctx, module_info, function);
    } else if let Ok(array_copy_op) = wasm::ArrayCopy::from_operation(db, *op) {
        return handle_array_copy(db, array_copy_op, ctx, module_info, function);
    } else if wasm::RefNull::matches(db, *op) {
        return handle_ref_null(db, op, ctx, module_info, function);
    } else if let Ok(ref_func_op) = wasm::RefFunc::from_operation(db, *op) {
        return handle_ref_func(db, ref_func_op, ctx, module_info, function);
    } else if wasm::RefCast::matches(db, *op) {
        return handle_ref_cast(db, op, ctx, module_info, function);
    } else if wasm::RefTest::matches(db, *op) {
        return handle_ref_test(db, op, ctx, module_info, function);
    } else if let Ok(bytes_op) = wasm::BytesFromData::from_operation(db, *op) {
        // Compound operation: create Bytes struct from passive data segment
        // Stack operations:
        //   i32.const <offset>    ; offset within data segment
        //   i32.const <len>       ; number of bytes to copy
        //   array.new_data $bytes_array <data_idx>
        //   i32.const 0           ; offset field (we use the whole array)
        //   i32.const <len>       ; len field
        //   struct.new $bytes_struct
        let data_idx = bytes_op.data_idx(db);
        let offset = bytes_op.offset(db);
        let len = bytes_op.len(db);

        // Push offset and length for array.new_data
        function.instruction(&Instruction::I32Const(offset as i32));
        function.instruction(&Instruction::I32Const(len as i32));
        function.instruction(&Instruction::ArrayNewData {
            array_type_index: BYTES_ARRAY_IDX,
            array_data_index: data_idx,
        });

        // Push struct fields: offset (0) and len
        function.instruction(&Instruction::I32Const(0));
        function.instruction(&Instruction::I32Const(len as i32));
        function.instruction(&Instruction::StructNew(BYTES_STRUCT_IDX));

        set_result_local(db, op, ctx, function)?;

    // === Linear Memory Management ===
    } else if let Ok(mem_size_op) = wasm::MemorySize::from_operation(db, *op) {
        return handle_memory_size(db, mem_size_op, ctx, function);
    } else if let Ok(mem_grow_op) = wasm::MemoryGrow::from_operation(db, *op) {
        return handle_memory_grow(db, mem_grow_op, ctx, module_info, function);

    // === Full-Width Loads ===
    } else if let Ok(load_op) = wasm::I32Load::from_operation(db, *op) {
        return handle_i32_load(db, load_op, ctx, module_info, function);
    } else if let Ok(load_op) = wasm::I64Load::from_operation(db, *op) {
        return handle_i64_load(db, load_op, ctx, module_info, function);
    } else if let Ok(load_op) = wasm::F32Load::from_operation(db, *op) {
        return handle_f32_load(db, load_op, ctx, module_info, function);
    } else if let Ok(load_op) = wasm::F64Load::from_operation(db, *op) {
        return handle_f64_load(db, load_op, ctx, module_info, function);

    // === Partial-Width Loads (i32) ===
    } else if let Ok(load_op) = wasm::I32Load8S::from_operation(db, *op) {
        return handle_i32_load8_s(db, load_op, ctx, module_info, function);
    } else if let Ok(load_op) = wasm::I32Load8U::from_operation(db, *op) {
        return handle_i32_load8_u(db, load_op, ctx, module_info, function);
    } else if let Ok(load_op) = wasm::I32Load16S::from_operation(db, *op) {
        return handle_i32_load16_s(db, load_op, ctx, module_info, function);
    } else if let Ok(load_op) = wasm::I32Load16U::from_operation(db, *op) {
        return handle_i32_load16_u(db, load_op, ctx, module_info, function);

    // === Partial-Width Loads (i64) ===
    } else if let Ok(load_op) = wasm::I64Load8S::from_operation(db, *op) {
        return handle_i64_load8_s(db, load_op, ctx, module_info, function);
    } else if let Ok(load_op) = wasm::I64Load8U::from_operation(db, *op) {
        return handle_i64_load8_u(db, load_op, ctx, module_info, function);
    } else if let Ok(load_op) = wasm::I64Load16S::from_operation(db, *op) {
        return handle_i64_load16_s(db, load_op, ctx, module_info, function);
    } else if let Ok(load_op) = wasm::I64Load16U::from_operation(db, *op) {
        return handle_i64_load16_u(db, load_op, ctx, module_info, function);
    } else if let Ok(load_op) = wasm::I64Load32S::from_operation(db, *op) {
        return handle_i64_load32_s(db, load_op, ctx, module_info, function);
    } else if let Ok(load_op) = wasm::I64Load32U::from_operation(db, *op) {
        return handle_i64_load32_u(db, load_op, ctx, module_info, function);

    // === Full-Width Stores ===
    } else if let Ok(store_op) = wasm::I32Store::from_operation(db, *op) {
        return handle_i32_store(db, store_op, ctx, module_info, function);
    } else if let Ok(store_op) = wasm::I64Store::from_operation(db, *op) {
        return handle_i64_store(db, store_op, ctx, module_info, function);
    } else if let Ok(store_op) = wasm::F32Store::from_operation(db, *op) {
        return handle_f32_store(db, store_op, ctx, module_info, function);
    } else if let Ok(store_op) = wasm::F64Store::from_operation(db, *op) {
        return handle_f64_store(db, store_op, ctx, module_info, function);

    // === Partial-Width Stores ===
    } else if let Ok(store_op) = wasm::I32Store8::from_operation(db, *op) {
        return handle_i32_store8(db, store_op, ctx, module_info, function);
    } else if let Ok(store_op) = wasm::I32Store16::from_operation(db, *op) {
        return handle_i32_store16(db, store_op, ctx, module_info, function);
    } else if let Ok(store_op) = wasm::I64Store8::from_operation(db, *op) {
        return handle_i64_store8(db, store_op, ctx, module_info, function);
    } else if let Ok(store_op) = wasm::I64Store16::from_operation(db, *op) {
        return handle_i64_store16(db, store_op, ctx, module_info, function);
    } else if let Ok(store_op) = wasm::I64Store32::from_operation(db, *op) {
        return handle_i64_store32(db, store_op, ctx, module_info, function);
    } else {
        tracing::error!("unsupported wasm op: {}", name);
        return Err(CompilationError::unsupported_feature_msg(format!(
            "wasm op not supported: {}",
            name
        )));
    }

    Ok(())
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

        // Should have 6 GC types: 5 built-in (BoxedF64, BytesArray, BytesStruct, Step, ClosureStruct) + 1 user struct
        assert_eq!(gc_types.len(), 6);
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
        // Index 5 is the user struct
        assert_eq!(gc_type_kind(&gc_types[5]), "struct");

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

        // Should have 6 GC types: 5 built-in (BoxedF64, BytesArray, BytesStruct, Step, ClosureStruct) + 1 user array
        assert_eq!(gc_types.len(), 6);
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
        // Index 5 is the user array
        assert_eq!(gc_type_kind(&gc_types[5]), "array");
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

        // Should have 6 GC types: 5 built-in + 1 user struct (same type_idx used twice)
        assert_eq!(gc_types.len(), 6);
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
        // Index 5 is the deduplicated user struct
        assert_eq!(gc_type_kind(&gc_types[5]), "struct");
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
        // (6 types: 5 built-in + 1 user struct)
        assert_eq!(gc_types.len(), 6);
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
        // Index 5 is the user struct from inside the function body
        assert_eq!(gc_type_kind(&gc_types[5]), "struct");
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

        // Should only have the 5 built-in types (no additional user types allocated)
        assert_eq!(gc_types.len(), 5);
        assert_eq!(gc_type_kind(&gc_types[0]), "struct"); // BoxedF64
        assert_eq!(gc_type_kind(&gc_types[1]), "array"); // BytesArray
        assert_eq!(gc_type_kind(&gc_types[2]), "struct"); // BytesStruct
        assert_eq!(gc_type_kind(&gc_types[3]), "struct"); // Step
        assert_eq!(gc_type_kind(&gc_types[4]), "struct"); // ClosureStruct
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

        // Should only have the 5 built-in types (no additional user types allocated)
        assert_eq!(gc_types.len(), 5);
        assert_eq!(gc_type_kind(&gc_types[0]), "struct"); // BoxedF64
        assert_eq!(gc_type_kind(&gc_types[1]), "array"); // BytesArray
        assert_eq!(gc_type_kind(&gc_types[2]), "struct"); // BytesStruct
        assert_eq!(gc_type_kind(&gc_types[3]), "struct"); // Step
        assert_eq!(gc_type_kind(&gc_types[4]), "struct"); // ClosureStruct
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

        // Should only have the 5 built-in types
        assert_eq!(gc_types.len(), 5);
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
    // Test: struct_new with result type placeholder (no type attr)
    // ========================================

    /// Test that struct_new with result type wasm.structref (but without explicit type attr)
    /// is detected as a placeholder type and gets a unique type index.
    /// This simulates what happens in cont_to_wasm.rs when creating state structs.
    #[salsa::tracked]
    fn make_struct_new_result_type_placeholder_module(
        db: &dyn salsa::Database,
    ) -> core::Module<'_> {
        let location = test_location(db);
        let i64_ty = core::I64::new(db).as_type();
        let structref_ty = wasm::Structref::new(db).as_type();

        // Create struct_new using the dialect helper function (like cont_to_wasm.rs does)
        // This sets type_idx as an attribute but NOT the "type" attribute
        let field_value = wasm::i64_const(db, location, i64_ty, 42).as_operation();

        let field_val = field_value.result(db, 0);

        // Use wasm::struct_new helper function - this sets type_idx=0 but result type is structref
        let fields: IdVec<Value> = idvec![field_val];
        let struct_new_op = wasm::struct_new(db, location, fields, structref_ty, 0);

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![field_value, struct_new_op.as_operation()],
        );
        let region = Region::new(db, location, idvec![block]);
        core::Module::create(db, location, "test".into(), region)
    }

    #[salsa_test]
    fn test_struct_new_result_type_placeholder(db: &salsa::DatabaseImpl) {
        let module = make_struct_new_result_type_placeholder_module(db);
        let (gc_types, _type_map, placeholder_map) =
            collect_gc_types(db, module, &HashMap::new()).expect("collect_gc_types failed");

        // The structref_ty placeholder with 1 field should be registered
        let structref_ty = wasm::Structref::new(db).as_type();
        let key = (structref_ty, 1usize);
        assert!(
            placeholder_map.contains_key(&key),
            "placeholder map should contain (structref_ty, 1) key"
        );

        // Should have 6 types: 5 built-in + 1 user placeholder struct
        assert_eq!(gc_types.len(), 6, "gc_types should have 6 types");

        // The user struct (index 5) should have 1 field of type I64
        let user_struct = &gc_types[5];
        if let GcTypeDef::Struct(fields) = user_struct {
            assert_eq!(fields.len(), 1, "struct should have 1 field");
        } else {
            panic!("expected struct type at index 5");
        }
    }
}
