//! WebAssembly binary emission from wasm dialect operations.
//!
//! This module converts lowered wasm dialect TrunkIR operations to
//! a WebAssembly binary using the `wasm_encoder` crate.

mod call_indirect_collection;
mod definitions;
mod gc_types_collection;
mod handlers;
pub(crate) mod helpers;
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

use trunk_ir::Symbol;
use trunk_ir::arena::ArenaModule;
use trunk_ir::arena::IrContext;
use trunk_ir::arena::dialect::wasm as arena_wasm;
use trunk_ir::arena::ops::ArenaDialectOp;
use trunk_ir::arena::refs::{OpRef, RegionRef, TypeRef, ValueRef};
use trunk_ir::arena::types::Attribute as ArenaAttribute;
use wasm_encoder::{
    AbstractHeapType, ArrayType, CodeSection, CompositeInnerType, CompositeType, ConstExpr,
    DataCountSection, DataSection, ElementSection, Elements, EntityType, ExportKind, ExportSection,
    Function, FunctionSection, GlobalSection, GlobalType, HeapType, ImportSection, Instruction,
    MemorySection, MemoryType, Module, RefType, StructType, SubType, TableSection, TableType,
    TypeSection, ValType,
};

use crate::gc_types::GcTypeDef;
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
struct ModuleInfo {
    imports: Vec<ImportFuncDef>,
    funcs: Vec<FunctionDef>,
    exports: Vec<ExportDef>,
    memory: Option<MemoryDef>,
    data: Vec<DataDef>,
    tables: Vec<TableDef>,
    elements: Vec<ElementDef>,
    globals: Vec<GlobalDef>,
    gc_types: Vec<GcTypeDef>,
    type_idx_by_type: HashMap<TypeRef, u32>,
    /// Placeholder struct type_idx lookup (for wasm.structref types)
    placeholder_struct_type_idx: HashMap<(TypeRef, usize), u32>,
    /// Function type lookup map (core.func TypeRef).
    func_types: HashMap<Symbol, TypeRef>,
    /// Function index lookup map (import index or func index).
    func_indices: HashMap<Symbol, u32>,
    /// Functions referenced via ref.func that need declarative elem segment.
    ref_funcs: HashSet<Symbol>,
    /// Additional function types from call_indirect that need to be added to type section.
    /// Stored as (type_idx, core.func TypeRef) pairs.
    call_indirect_types: Vec<(u32, TypeRef)>,
    /// Pre-interned common types for use in handlers.
    common_types: CommonTypes,
}

/// Pre-interned common types to avoid needing `&mut IrContext` during emission.
#[derive(Default)]
struct CommonTypes {
    anyref: Option<TypeRef>,
    funcref: Option<TypeRef>,
    step: Option<TypeRef>,
}

/// Context for emitting a single function's code.
struct FunctionEmitContext {
    /// Maps values to their local indices.
    value_locals: HashMap<ValueRef, u32>,
    /// Effective types for values (after unification).
    effective_types: HashMap<ValueRef, TypeRef>,
    /// The function's expected return type (from function signature).
    func_return_type: Option<TypeRef>,
}

pub fn emit_wasm(ctx: &mut IrContext, module: ArenaModule) -> CompilationResult<Vec<u8>> {
    debug!("emit_wasm: collecting module info...");
    let module_info = match collect_module_info(ctx, module) {
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
        let (params_refs, result_ref) = func_type_parts(ctx, import_def.func_type)
            .ok_or_else(|| CompilationError::type_error("import func type is not core.func"))?;
        let params = params_refs
            .iter()
            .map(|ty| type_to_valtype(ctx, *ty, &module_info.type_idx_by_type))
            .collect::<CompilationResult<Vec<_>>>()?;
        let results = result_types(ctx, result_ref, &module_info.type_idx_by_type)?;
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
        let (params_refs, declared_result) = func_type_parts(ctx, func_def.func_type)
            .ok_or_else(|| CompilationError::type_error("func type is not core.func"))?;
        let params = params_refs
            .iter()
            .map(|ty| type_to_valtype(ctx, *ty, &module_info.type_idx_by_type))
            .collect::<CompilationResult<Vec<_>>>()?;

        let declared_result_data = ctx.types.get(declared_result);
        debug!(
            "  checking return type adjustment for {}: declared={}.{}",
            func_def.name, declared_result_data.dialect, declared_result_data.name
        );

        let effective_result = {
            let regions = &ctx.op(func_def.op).regions;
            if let Some(&body_region) = regions.first() {
                if is_type(ctx, declared_result, "core", "func")
                    || is_type(ctx, declared_result, "wasm", "funcref")
                {
                    debug!("  checking funcref function for handler dispatch...");
                    if should_adjust_handler_return_to_i32(ctx, body_region) {
                        debug!(
                            "  adjusting return type from funcref to i32 for computation lambda: {}",
                            func_def.name
                        );
                        intern_simple_type(ctx, "core", "i32")
                    } else {
                        declared_result
                    }
                } else {
                    declared_result
                }
            } else {
                declared_result
            }
        };

        let results = match result_types(ctx, effective_result, &module_info.type_idx_by_type) {
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

    // Emit call_indirect function types
    for (type_idx, func_ty) in &module_info.call_indirect_types {
        let (params_refs, result_ref) = func_type_parts(ctx, *func_ty)
            .ok_or_else(|| CompilationError::type_error("call_indirect type is not core.func"))?;
        debug!(
            "Emitting call_indirect type idx={}: params={:?}, result={}.{}",
            type_idx,
            params_refs
                .iter()
                .map(|t| {
                    let d = ctx.types.get(*t);
                    format!("{}.{}", d.dialect, d.name)
                })
                .collect::<Vec<_>>(),
            ctx.types.get(result_ref).dialect,
            ctx.types.get(result_ref).name
        );
        let params = params_refs
            .iter()
            .map(|ty| type_to_valtype(ctx, *ty, &module_info.type_idx_by_type))
            .collect::<CompilationResult<Vec<_>>>()?;
        let results = result_types(ctx, result_ref, &module_info.type_idx_by_type)?;
        type_section.ty().function(params, results);
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

    for table_def in &module_info.tables {
        table_section.table(TableType {
            element_type: table_def.reftype,
            minimum: table_def.min as u64,
            maximum: table_def.max.map(|v| v as u64),
            table64: false,
            shared: false,
        });
    }

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
        match emit_function(ctx, func_def, &module_info) {
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

/// Recursively collect wasm operations from a region.
fn collect_wasm_ops_from_region(
    ctx: &IrContext,
    region: RegionRef,
    info: &mut ModuleInfo,
) -> CompilationResult<()> {
    let core_dialect = Symbol::new("core");
    let module_name = Symbol::new("module");

    for &block_ref in &ctx.region(region).blocks {
        for &op in &ctx.block(block_ref).ops {
            let op_data = ctx.op(op);
            let dialect = op_data.dialect;
            let name = op_data.name;

            if dialect == core_dialect && name == module_name {
                for &nested_region in &op_data.regions {
                    collect_wasm_ops_from_region(ctx, nested_region, info)?;
                }
                continue;
            }

            if let Ok(func_op) = arena_wasm::Func::from_op(ctx, op) {
                if let Ok(func_def) = extract_function_def(ctx, func_op) {
                    debug!("Including function: {}", func_def.name);
                    info.funcs.push(func_def);
                }
            } else if let Ok(import_op) = arena_wasm::ImportFunc::from_op(ctx, op) {
                info.imports.push(extract_import_def(ctx, import_op)?);
            } else if let Ok(export_op) = arena_wasm::ExportFunc::from_op(ctx, op) {
                info.exports.push(extract_export_func(ctx, export_op)?);
            } else if let Ok(export_mem_op) = arena_wasm::ExportMemory::from_op(ctx, op) {
                info.exports
                    .push(extract_export_memory(ctx, export_mem_op)?);
            } else if let Ok(memory_op) = arena_wasm::Memory::from_op(ctx, op) {
                info.memory = Some(extract_memory_def(ctx, memory_op)?);
            } else if let Ok(data_op) = arena_wasm::Data::from_op(ctx, op) {
                info.data.push(extract_data_def(ctx, data_op)?);
            } else if let Ok(table_op) = arena_wasm::Table::from_op(ctx, op) {
                info.tables.push(extract_table_def(ctx, table_op)?);
            } else if let Ok(elem_op) = arena_wasm::Elem::from_op(ctx, op) {
                info.elements.push(extract_element_def(ctx, elem_op)?);
            } else if let Ok(global_op) = arena_wasm::Global::from_op(ctx, op) {
                info.globals.push(extract_global_def(ctx, global_op)?);
            }
        }
    }

    Ok(())
}

fn collect_module_info(ctx: &mut IrContext, module: ArenaModule) -> CompilationResult<ModuleInfo> {
    let mut info = ModuleInfo::default();

    let body = module
        .body(ctx)
        .ok_or_else(|| CompilationError::invalid_module("module has no body region"))?;

    collect_wasm_ops_from_region(ctx, body, &mut info)?;

    // Collect GC types
    let (gc_types, mut type_idx_by_type, placeholder_struct_type_idx) =
        collect_gc_types(ctx, module)?;
    info.gc_types = gc_types;

    // Collect function types from call_indirect operations
    let gc_type_count = info.gc_types.len();
    let func_type_count = info.imports.len() + info.funcs.len();

    // Register function definition types in type_idx_by_type
    for (i, import_def) in info.imports.iter().enumerate() {
        type_idx_by_type
            .entry(import_def.func_type)
            .or_insert(gc_type_count as u32 + i as u32);
    }
    for (i, func_def) in info.funcs.iter().enumerate() {
        type_idx_by_type
            .entry(func_def.func_type)
            .or_insert(gc_type_count as u32 + info.imports.len() as u32 + i as u32);
    }

    let call_indirect_types = collect_call_indirect_types(
        ctx,
        module,
        &mut type_idx_by_type,
        gc_type_count,
        func_type_count,
    )?;

    info.type_idx_by_type = type_idx_by_type;
    info.call_indirect_types = call_indirect_types;
    info.placeholder_struct_type_idx = placeholder_struct_type_idx;

    // Build function type lookup map
    for func in &info.funcs {
        info.func_types.insert(func.name, func.func_type);
    }
    for import in &info.imports {
        info.func_types.insert(import.sym, import.func_type);
    }

    // Build function index map
    for (index, import_def) in info.imports.iter().enumerate() {
        info.func_indices.insert(import_def.sym, index as u32);
    }
    let import_count = info.imports.len() as u32;
    for (index, func_def) in info.funcs.iter().enumerate() {
        info.func_indices
            .insert(func_def.name, import_count + index as u32);
    }

    // Collect functions referenced via ref.func
    info.ref_funcs = collect_ref_funcs(ctx, module);

    // Auto-create a funcref table if call_indirect is used but no table is defined
    if info.tables.is_empty() && has_call_indirect(ctx, module) {
        debug!("Auto-generating funcref table for call_indirect");
        info.tables.push(TableDef {
            reftype: RefType::FUNCREF,
            min: 0,
            max: None,
        });
    }

    // Pre-intern common types so handlers don't need &mut IrContext
    info.common_types = CommonTypes {
        anyref: Some(intern_simple_type(ctx, "wasm", "anyref")),
        funcref: Some(intern_simple_type(ctx, "wasm", "funcref")),
        step: Some(intern_named_adt_struct(ctx, "_Step")),
    };

    Ok(info)
}

fn emit_function(
    ctx: &IrContext,
    func_def: &FunctionDef,
    module_info: &ModuleInfo,
) -> CompilationResult<Function> {
    debug!("=== emit_function: {:?} ===", func_def.name);
    let regions = &ctx.op(func_def.op).regions;
    let &region = regions
        .first()
        .ok_or_else(|| CompilationError::invalid_module("wasm.func missing body region"))?;
    let blocks = &ctx.region(region).blocks;
    let &block = blocks
        .first()
        .ok_or_else(|| CompilationError::invalid_module("wasm.func has no entry block"))?;

    let (params_refs, result_ref) = func_type_parts(ctx, func_def.func_type)
        .ok_or_else(|| CompilationError::type_error("func type is not core.func"))?;
    let block_args = ctx.block_args(block);
    if params_refs.len() != block_args.len() {
        return Err(CompilationError::invalid_module(
            "function parameter count does not match entry block args",
        ));
    }

    let func_return_type = Some(result_ref);
    let mut emit_ctx = FunctionEmitContext {
        value_locals: HashMap::new(),
        effective_types: HashMap::new(),
        func_return_type,
    };
    let mut locals: Vec<ValType> = Vec::new();

    for (index, &arg) in block_args.iter().enumerate() {
        emit_ctx.value_locals.insert(arg, index as u32);
        let arg_ty = ctx.value_ty(arg);
        emit_ctx.effective_types.insert(arg, arg_ty);
    }

    let param_count = params_refs.len() as u32;

    assign_locals_in_region(
        ctx,
        region,
        param_count,
        &mut locals,
        &mut emit_ctx,
        module_info,
    )?;

    let mut function = Function::new(compress_locals(&locals));

    emit_region_ops(ctx, region, &emit_ctx, module_info, &mut function)?;

    function.instruction(&Instruction::End);

    Ok(function)
}

fn assign_locals_in_region(
    ctx: &IrContext,
    region: RegionRef,
    param_count: u32,
    locals: &mut Vec<ValType>,
    emit_ctx: &mut FunctionEmitContext,
    module_info: &ModuleInfo,
) -> CompilationResult<()> {
    for &block_ref in &ctx.region(region).blocks {
        let block_args = ctx.block_args(block_ref);
        for &block_arg in block_args.iter() {
            if emit_ctx.value_locals.contains_key(&block_arg) {
                continue;
            }
            let arg_ty = ctx.value_ty(block_arg);
            let val_type = type_to_valtype(ctx, arg_ty, &module_info.type_idx_by_type)?;
            let local_index = param_count + locals.len() as u32;
            emit_ctx.value_locals.insert(block_arg, local_index);
            emit_ctx.effective_types.insert(block_arg, arg_ty);
            locals.push(val_type);
            let ty_data = ctx.types.get(arg_ty);
            tracing::debug!(
                "Allocated local {} for block arg type {}.{}",
                local_index,
                ty_data.dialect,
                ty_data.name
            );
        }

        for &op in &ctx.block(block_ref).ops {
            // Process nested regions FIRST
            let nested_regions = ctx.op(op).regions.clone();
            for &nested in &nested_regions {
                assign_locals_in_region(ctx, nested, param_count, locals, emit_ctx, module_info)?;
            }

            let result_types = ctx.op_result_types(op);
            if result_types.len() > 1 {
                return Err(CompilationError::unsupported_feature("multi-result ops"));
            }
            if let Some(&result_ty) = result_types.first() {
                let effective_ty = result_ty;

                let is_ref_cast = arena_wasm::RefCast::matches(ctx, op);
                let is_struct_new = arena_wasm::StructNew::matches(ctx, op);
                let val_type = if is_ref_cast || is_struct_new {
                    let attrs = &ctx.op(op).attributes;

                    let placeholder_ty = if is_ref_cast {
                        attrs.get(&ATTR_TARGET_TYPE()).and_then(|attr| {
                            if let ArenaAttribute::Type(ty) = attr {
                                if is_type(ctx, *ty, "wasm", "structref") {
                                    Some(*ty)
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        })
                    } else if is_struct_new {
                        attrs
                            .get(&Symbol::new("type"))
                            .and_then(|attr| {
                                if let ArenaAttribute::Type(ty) = attr {
                                    if is_type(ctx, *ty, "wasm", "structref") {
                                        Some(*ty)
                                    } else {
                                        None
                                    }
                                } else {
                                    None
                                }
                            })
                            .or_else(|| {
                                ctx.op_result_types(op).first().copied().and_then(|ty| {
                                    if is_type(ctx, ty, "wasm", "structref") {
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
                        let field_count = if is_struct_new {
                            Some(ctx.op_operands(op).len())
                        } else {
                            attrs.get(&Symbol::new("field_count")).and_then(|attr| {
                                if let ArenaAttribute::IntBits(fc) = attr {
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
                                type_to_valtype(ctx, effective_ty, &module_info.type_idx_by_type)?
                            }
                        } else {
                            type_to_valtype(ctx, effective_ty, &module_info.type_idx_by_type)?
                        }
                    } else {
                        type_to_valtype(ctx, effective_ty, &module_info.type_idx_by_type)?
                    }
                } else {
                    match type_to_valtype(ctx, effective_ty, &module_info.type_idx_by_type) {
                        Ok(vt) => vt,
                        Err(e) => {
                            let op_data = ctx.op(op);
                            debug!(
                                "type_to_valtype failed for op {}.{}: {:?}",
                                op_data.dialect, op_data.name, e
                            );
                            return Err(e);
                        }
                    }
                };
                let local_index = param_count + locals.len() as u32;
                let result_value = ctx.op_result(op, 0);
                emit_ctx.value_locals.insert(result_value, local_index);
                emit_ctx.effective_types.insert(result_value, effective_ty);
                locals.push(val_type);
            }
        }
    }
    Ok(())
}

/// Describes what kind of construct a nesting level represents.
#[derive(Clone, Debug)]
enum NestingKind {
    Block,
    Loop { arg_locals: Vec<u32> },
    If,
}

fn emit_region_ops(
    ctx: &IrContext,
    region: RegionRef,
    emit_ctx: &FunctionEmitContext,
    module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    emit_region_ops_nested(ctx, region, emit_ctx, module_info, function, &[])
}

fn emit_region_ops_nested(
    ctx: &IrContext,
    region: RegionRef,
    emit_ctx: &FunctionEmitContext,
    module_info: &ModuleInfo,
    function: &mut Function,
    nesting: &[NestingKind],
) -> CompilationResult<()> {
    let blocks = &ctx.region(region).blocks;
    if blocks.len() != 1 {
        return Err(CompilationError::unsupported_feature("multi-block regions"));
    }
    let block_ref = blocks[0];
    let ops = ctx.block(block_ref).ops.clone();
    let mut iter = ops.iter().peekable();
    while let Some(&op) = iter.next() {
        if let Ok(yield_op) = arena_wasm::Yield::from_op(ctx, op) {
            let followed_by_br = iter
                .peek()
                .is_some_and(|&&next| arena_wasm::Br::from_op(ctx, next).is_ok());
            if followed_by_br {
                let br_op = arena_wasm::Br::from_op(ctx, **iter.peek().unwrap()).unwrap();
                let depth = br_op.target(ctx) as usize;
                let value = yield_op.value(ctx);
                let index = *emit_ctx.value_locals.get(&value).ok_or_else(|| {
                    CompilationError::invalid_module("wasm.yield value missing local")
                })?;

                if depth < nesting.len()
                    && let NestingKind::Loop { ref arg_locals } = nesting[nesting.len() - 1 - depth]
                {
                    for &local in arg_locals.iter() {
                        function.instruction(&Instruction::LocalGet(index));
                        function.instruction(&Instruction::LocalSet(local));
                    }
                    continue;
                }

                function.instruction(&Instruction::LocalGet(index));
            }
            continue;
        }
        emit_op_nested(ctx, op, emit_ctx, module_info, function, nesting)?;
    }
    Ok(())
}

fn region_result_value(ctx: &IrContext, region: RegionRef) -> Option<ValueRef> {
    let blocks = &ctx.region(region).blocks;
    let &block_ref = blocks.last()?;
    let ops = &ctx.block(block_ref).ops;
    let &op = ops.last()?;

    if let Ok(yield_op) = arena_wasm::Yield::from_op(ctx, op) {
        return Some(yield_op.value(ctx));
    }

    let result_types = ctx.op_result_types(op);
    if result_types.is_empty() {
        None
    } else {
        Some(ctx.op_result(op, 0))
    }
}

fn emit_op_nested(
    ctx: &IrContext,
    op: OpRef,
    emit_ctx: &FunctionEmitContext,
    module_info: &ModuleInfo,
    function: &mut Function,
    nesting: &[NestingKind],
) -> CompilationResult<()> {
    let op_data = ctx.op(op);
    let wasm_dialect = Symbol::new("wasm");
    if op_data.dialect != wasm_dialect {
        return Err(CompilationError::unsupported_feature(
            "non-wasm op in wasm backend",
        ));
    }

    let name = op_data.name;
    let operands = ctx.op_operands(op);

    debug!("emit_op: wasm.{}", name);

    // Handle wasm.nop
    if arena_wasm::Nop::matches(ctx, op) {
        if let Some(&result_ty) = ctx.op_result_types(op).first() {
            let ty_data = ctx.types.get(result_ty);
            debug!("wasm.nop: result_ty={}.{}", ty_data.dialect, ty_data.name);
            if is_type(ctx, result_ty, "core", "func") || is_type(ctx, result_ty, "wasm", "funcref")
            {
                debug!("wasm.nop: emitting ref.null func");
                function.instruction(&Instruction::RefNull(HeapType::Abstract {
                    shared: false,
                    ty: AbstractHeapType::Func,
                }));
                set_result_local(ctx, op, emit_ctx, function)?;
            } else if is_type(ctx, result_ty, "wasm", "anyref")
                || is_type(ctx, result_ty, "wasm", "structref")
            {
                debug!("wasm.nop: emitting ref.null any");
                function.instruction(&Instruction::RefNull(HeapType::Abstract {
                    shared: false,
                    ty: AbstractHeapType::Any,
                }));
                set_result_local(ctx, op, emit_ctx, function)?;
            } else if is_nil_type(ctx, result_ty) {
                debug!("wasm.nop: emitting ref.null none for nil type");
                function.instruction(&Instruction::RefNull(HeapType::Abstract {
                    shared: false,
                    ty: AbstractHeapType::None,
                }));
                set_result_local(ctx, op, emit_ctx, function)?;
            } else {
                debug!("wasm.nop: skipping (unknown primitive type)");
            }
        }
        return Ok(());
    }

    // Fast path: simple operations
    if let Some(instr) = SIMPLE_OPS.get(&name) {
        emit_operands(ctx, operands, emit_ctx, function)?;
        function.instruction(instr);
        set_result_local(ctx, op, emit_ctx, function)?;
        return Ok(());
    }

    // Special cases
    if let Ok(const_op) = arena_wasm::I32Const::from_op(ctx, op) {
        handle_i32_const(ctx, const_op, emit_ctx, function)
    } else if let Ok(const_op) = arena_wasm::I64Const::from_op(ctx, op) {
        handle_i64_const(ctx, const_op, emit_ctx, function)
    } else if let Ok(const_op) = arena_wasm::F32Const::from_op(ctx, op) {
        handle_f32_const(ctx, const_op, emit_ctx, function)
    } else if let Ok(const_op) = arena_wasm::F64Const::from_op(ctx, op) {
        handle_f64_const(ctx, const_op, emit_ctx, function)
    } else if arena_wasm::If::matches(ctx, op) {
        handle_if(ctx, op, emit_ctx, module_info, function, nesting)
    } else if arena_wasm::Block::matches(ctx, op) {
        handle_block(ctx, op, emit_ctx, module_info, function, nesting)
    } else if arena_wasm::Loop::matches(ctx, op) {
        handle_loop(ctx, op, emit_ctx, module_info, function, nesting)
    } else if let Ok(br_op) = arena_wasm::Br::from_op(ctx, op) {
        handle_br(ctx, br_op, function)
    } else if let Ok(br_if_op) = arena_wasm::BrIf::from_op(ctx, op) {
        handle_br_if(ctx, br_if_op, emit_ctx, module_info, function)
    } else if let Ok(call_op) = arena_wasm::Call::from_op(ctx, op) {
        handle_call(ctx, call_op, emit_ctx, module_info, function)
    } else if arena_wasm::CallIndirect::matches(ctx, op) {
        handle_call_indirect(ctx, op, emit_ctx, module_info, function)
    } else if let Ok(return_call_op) = arena_wasm::ReturnCall::from_op(ctx, op) {
        handle_return_call(ctx, return_call_op, emit_ctx, module_info, function)
    } else if let Ok(local_op) = arena_wasm::LocalGet::from_op(ctx, op) {
        handle_local_get(ctx, local_op, emit_ctx, function)
    } else if let Ok(local_op) = arena_wasm::LocalSet::from_op(ctx, op) {
        handle_local_set(ctx, local_op, emit_ctx, function)
    } else if let Ok(local_op) = arena_wasm::LocalTee::from_op(ctx, op) {
        handle_local_tee(ctx, local_op, emit_ctx, function)
    } else if let Ok(global_op) = arena_wasm::GlobalGet::from_op(ctx, op) {
        handle_global_get(ctx, global_op, emit_ctx, module_info, function)
    } else if let Ok(global_op) = arena_wasm::GlobalSet::from_op(ctx, op) {
        handle_global_set(ctx, global_op, emit_ctx, function)
    } else if arena_wasm::StructNew::matches(ctx, op) {
        handle_struct_new(ctx, op, emit_ctx, module_info, function)
    } else if arena_wasm::StructGet::matches(ctx, op) {
        handle_struct_get(ctx, op, emit_ctx, module_info, function)
    } else if let Ok(struct_set_op) = arena_wasm::StructSet::from_op(ctx, op) {
        handle_struct_set(ctx, struct_set_op, emit_ctx, module_info, function)
    } else if let Ok(array_new_op) = arena_wasm::ArrayNew::from_op(ctx, op) {
        handle_array_new(ctx, array_new_op, emit_ctx, module_info, function)
    } else if arena_wasm::ArrayNewDefault::matches(ctx, op) {
        handle_array_new_default(ctx, op, emit_ctx, module_info, function)
    } else if let Ok(array_get_op) = arena_wasm::ArrayGet::from_op(ctx, op) {
        handle_array_get(ctx, array_get_op, emit_ctx, module_info, function)
    } else if let Ok(array_get_s_op) = arena_wasm::ArrayGetS::from_op(ctx, op) {
        handle_array_get_s(ctx, array_get_s_op, emit_ctx, module_info, function)
    } else if let Ok(array_get_u_op) = arena_wasm::ArrayGetU::from_op(ctx, op) {
        handle_array_get_u(ctx, array_get_u_op, emit_ctx, module_info, function)
    } else if let Ok(array_set_op) = arena_wasm::ArraySet::from_op(ctx, op) {
        handle_array_set(ctx, array_set_op, emit_ctx, module_info, function)
    } else if let Ok(array_copy_op) = arena_wasm::ArrayCopy::from_op(ctx, op) {
        handle_array_copy(ctx, array_copy_op, emit_ctx, module_info, function)
    } else if arena_wasm::RefNull::matches(ctx, op) {
        handle_ref_null(ctx, op, emit_ctx, module_info, function)
    } else if let Ok(ref_func_op) = arena_wasm::RefFunc::from_op(ctx, op) {
        handle_ref_func(ctx, ref_func_op, emit_ctx, module_info, function)
    } else if arena_wasm::RefCast::matches(ctx, op) {
        handle_ref_cast(ctx, op, emit_ctx, module_info, function)
    } else if arena_wasm::RefTest::matches(ctx, op) {
        handle_ref_test(ctx, op, emit_ctx, module_info, function)
    } else if let Ok(bytes_op) = arena_wasm::BytesFromData::from_op(ctx, op) {
        handle_bytes_from_data(ctx, bytes_op, emit_ctx, function)
    } else if let Ok(mem_size_op) = arena_wasm::MemorySize::from_op(ctx, op) {
        handle_memory_size(ctx, mem_size_op, emit_ctx, function)
    } else if let Ok(mem_grow_op) = arena_wasm::MemoryGrow::from_op(ctx, op) {
        handle_memory_grow(ctx, mem_grow_op, emit_ctx, module_info, function)
    } else if let Ok(load_op) = arena_wasm::I32Load::from_op(ctx, op) {
        handle_i32_load(ctx, load_op, emit_ctx, module_info, function)
    } else if let Ok(load_op) = arena_wasm::I64Load::from_op(ctx, op) {
        handle_i64_load(ctx, load_op, emit_ctx, module_info, function)
    } else if let Ok(load_op) = arena_wasm::F32Load::from_op(ctx, op) {
        handle_f32_load(ctx, load_op, emit_ctx, module_info, function)
    } else if let Ok(load_op) = arena_wasm::F64Load::from_op(ctx, op) {
        handle_f64_load(ctx, load_op, emit_ctx, module_info, function)
    } else if let Ok(load_op) = arena_wasm::I32Load8S::from_op(ctx, op) {
        handle_i32_load8_s(ctx, load_op, emit_ctx, module_info, function)
    } else if let Ok(load_op) = arena_wasm::I32Load8U::from_op(ctx, op) {
        handle_i32_load8_u(ctx, load_op, emit_ctx, module_info, function)
    } else if let Ok(load_op) = arena_wasm::I32Load16S::from_op(ctx, op) {
        handle_i32_load16_s(ctx, load_op, emit_ctx, module_info, function)
    } else if let Ok(load_op) = arena_wasm::I32Load16U::from_op(ctx, op) {
        handle_i32_load16_u(ctx, load_op, emit_ctx, module_info, function)
    } else if let Ok(load_op) = arena_wasm::I64Load8S::from_op(ctx, op) {
        handle_i64_load8_s(ctx, load_op, emit_ctx, module_info, function)
    } else if let Ok(load_op) = arena_wasm::I64Load8U::from_op(ctx, op) {
        handle_i64_load8_u(ctx, load_op, emit_ctx, module_info, function)
    } else if let Ok(load_op) = arena_wasm::I64Load16S::from_op(ctx, op) {
        handle_i64_load16_s(ctx, load_op, emit_ctx, module_info, function)
    } else if let Ok(load_op) = arena_wasm::I64Load16U::from_op(ctx, op) {
        handle_i64_load16_u(ctx, load_op, emit_ctx, module_info, function)
    } else if let Ok(load_op) = arena_wasm::I64Load32S::from_op(ctx, op) {
        handle_i64_load32_s(ctx, load_op, emit_ctx, module_info, function)
    } else if let Ok(load_op) = arena_wasm::I64Load32U::from_op(ctx, op) {
        handle_i64_load32_u(ctx, load_op, emit_ctx, module_info, function)
    } else if let Ok(store_op) = arena_wasm::I32Store::from_op(ctx, op) {
        handle_i32_store(ctx, store_op, emit_ctx, module_info, function)
    } else if let Ok(store_op) = arena_wasm::I64Store::from_op(ctx, op) {
        handle_i64_store(ctx, store_op, emit_ctx, module_info, function)
    } else if let Ok(store_op) = arena_wasm::F32Store::from_op(ctx, op) {
        handle_f32_store(ctx, store_op, emit_ctx, module_info, function)
    } else if let Ok(store_op) = arena_wasm::F64Store::from_op(ctx, op) {
        handle_f64_store(ctx, store_op, emit_ctx, module_info, function)
    } else if let Ok(store_op) = arena_wasm::I32Store8::from_op(ctx, op) {
        handle_i32_store8(ctx, store_op, emit_ctx, module_info, function)
    } else if let Ok(store_op) = arena_wasm::I32Store16::from_op(ctx, op) {
        handle_i32_store16(ctx, store_op, emit_ctx, module_info, function)
    } else if let Ok(store_op) = arena_wasm::I64Store8::from_op(ctx, op) {
        handle_i64_store8(ctx, store_op, emit_ctx, module_info, function)
    } else if let Ok(store_op) = arena_wasm::I64Store16::from_op(ctx, op) {
        handle_i64_store16(ctx, store_op, emit_ctx, module_info, function)
    } else if let Ok(store_op) = arena_wasm::I64Store32::from_op(ctx, op) {
        handle_i64_store32(ctx, store_op, emit_ctx, module_info, function)
    } else {
        let op_data = ctx.op(op);
        tracing::error!(
            "unsupported wasm op: {} (dialect={}, operands={}, results={}, attrs={:?})",
            name,
            op_data.dialect,
            ctx.op_operands(op).len(),
            ctx.op_result_types(op).len(),
            op_data.attributes.keys().collect::<Vec<_>>()
        );
        Err(CompilationError::unsupported_feature_msg(format!(
            "wasm op not supported: {}",
            name
        )))
    }
}

fn set_result_local(
    ctx: &IrContext,
    op: OpRef,
    emit_ctx: &FunctionEmitContext,
    function: &mut Function,
) -> CompilationResult<()> {
    let results = ctx.op_result_types(op);
    if results.is_empty() {
        return Ok(());
    }
    let local = emit_ctx
        .value_locals
        .get(&ctx.op_result(op, 0))
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

fn should_adjust_handler_return_to_i32(ctx: &IrContext, region: RegionRef) -> bool {
    for &block_ref in &ctx.region(region).blocks {
        for &op in &ctx.block(block_ref).ops {
            if arena_wasm::If::matches(ctx, op) {
                let regions = &ctx.op(op).regions;
                if let Some(&else_region) = regions.get(1) {
                    let has_call_indirect = region_contains_call_indirect(ctx, else_region);
                    debug!(
                        "should_adjust_handler_return_to_i32: wasm.if else branch has_call_indirect={}",
                        has_call_indirect
                    );
                    if !has_call_indirect {
                        return true;
                    }
                }
                return false;
            }
            for &nested_region in &ctx.op(op).regions {
                if should_adjust_handler_return_to_i32(ctx, nested_region) {
                    return true;
                }
            }
        }
    }
    false
}

fn region_contains_call_indirect(ctx: &IrContext, region: RegionRef) -> bool {
    for &block_ref in &ctx.region(region).blocks {
        for &op in &ctx.block(block_ref).ops {
            if arena_wasm::CallIndirect::matches(ctx, op) {
                return true;
            }
            for &nested_region in &ctx.op(op).regions {
                if region_contains_call_indirect(ctx, nested_region) {
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

/// Intern a simple type with no params or attrs.
fn intern_simple_type(ctx: &mut IrContext, dialect: &'static str, name: &'static str) -> TypeRef {
    ctx.types.intern(trunk_ir::arena::types::TypeData {
        dialect: Symbol::new(dialect),
        name: Symbol::new(name),
        params: Default::default(),
        attrs: Default::default(),
    })
}

/// Intern a named adt.struct type (e.g., _Step, _Continuation).
fn intern_named_adt_struct(ctx: &mut IrContext, name: &'static str) -> TypeRef {
    let mut attrs = std::collections::BTreeMap::new();
    attrs.insert(
        Symbol::new("name"),
        ArenaAttribute::Symbol(Symbol::new(name)),
    );
    ctx.types.intern(trunk_ir::arena::types::TypeData {
        dialect: Symbol::new("adt"),
        name: Symbol::new("struct"),
        params: Default::default(),
        attrs,
    })
}
