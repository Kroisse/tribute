//! WebAssembly binary emission from wasm dialect operations.
//!
//! This module converts lowered wasm dialect TrunkIR operations to
//! a WebAssembly binary using the `wasm_encoder` crate.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::LazyLock;

use tracing::debug;

use tribute_ir::ModulePathExt;
use tribute_ir::dialect::{ability, adt, closure, tribute};
use trunk_ir::dialect::{core, func, wasm};
use trunk_ir::{
    Attribute, Attrs, DialectOp, DialectType, IdVec, Operation, Region, Symbol, Type, Value,
    ValueDef,
};
use wasm_encoder::{
    AbstractHeapType, ArrayType, BlockType, CodeSection, CompositeInnerType, CompositeType,
    ConstExpr, DataCountSection, DataSection, ElementSection, Elements, EntityType, ExportKind,
    ExportSection, FieldType, Function, FunctionSection, GlobalSection, GlobalType, HeapType,
    ImportSection, Instruction, MemArg, MemorySection, MemoryType, Module, RefType, StorageType,
    StructType, SubType, TableSection, TableType, TypeSection, ValType,
};

use crate::errors;
use crate::{CompilationError, CompilationResult};

/// Type index for BoxedF64 (Float wrapper for polymorphic contexts).
/// This is always index 0 in the GC type section.
const BOXED_F64_IDX: u32 = 0;

/// Type index for BytesArray (array i8) - backing storage for Bytes.
/// This is always index 1 in the GC type section.
const BYTES_ARRAY_IDX: u32 = 1;

/// Type index for BytesStruct (struct { data: ref BytesArray, offset: i32, len: i32 }).
/// This is always index 2 in the GC type section.
const BYTES_STRUCT_IDX: u32 = 2;

/// First type index available for user-defined types.
const FIRST_USER_TYPE_IDX: u32 = 3;

trunk_ir::symbols! {
    ATTR_SYM_NAME => "sym_name",
    ATTR_TYPE => "type",
    ATTR_TYPE_IDX => "type_idx",
    ATTR_FIELD_IDX => "field_idx",
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
    ATTR_BYTES => "bytes",
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

struct FunctionDef<'db> {
    name: Symbol,
    ty: core::Func<'db>,
    op: Operation<'db>,
}

struct ImportFuncDef<'db> {
    sym: Symbol,
    module: Symbol,
    name: Symbol,
    ty: core::Func<'db>,
}

struct ExportDef {
    name: String,
    kind: ExportKind,
    target: ExportTarget,
}

#[derive(Debug)]
enum ExportTarget {
    Func(Symbol),
    Memory(u32),
}

struct MemoryDef {
    min: u32,
    max: Option<u32>,
    shared: bool,
    memory64: bool,
}

struct DataDef {
    offset: i32,
    bytes: Vec<u8>,
    passive: bool,
}

struct TableDef {
    reftype: RefType,
    min: u32,
    max: Option<u32>,
}

struct ElementDef {
    table: u32,
    offset: i32,
    funcs: Vec<Symbol>,
}

struct GlobalDef {
    valtype: ValType,
    mutable: bool,
    init: i64,
}

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
}

/// Context for emitting a single function's code.
struct FunctionEmitContext<'db> {
    /// Maps values to their local indices.
    value_locals: HashMap<Value<'db>, u32>,
    /// Effective types for values (after unification).
    effective_types: HashMap<Value<'db>, Type<'db>>,
}

enum GcTypeDef {
    Struct(Vec<FieldType>),
    Array(FieldType),
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
        let results = match result_types(db, func_def.ty.result(db), &module_info.type_idx_by_type)
        {
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
    if !module_info.elements.is_empty() {
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
    let wasm_dialect = Symbol::new("wasm");
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

            // Collect wasm operations
            if dialect == wasm_dialect {
                match name {
                    n if n == Symbol::new("func") => {
                        if let Ok(func_def) = extract_function_def(db, op) {
                            debug!("Including function: {}", func_def.name);
                            info.funcs.push(func_def);
                        }
                    }
                    n if n == Symbol::new("import_func") => {
                        info.imports.push(extract_import_def(db, op)?);
                    }
                    n if n == Symbol::new("export_func") => {
                        info.exports.push(extract_export_func(db, op)?);
                    }
                    n if n == Symbol::new("export_memory") => {
                        info.exports.push(extract_export_memory(db, op)?);
                    }
                    n if n == Symbol::new("memory") => {
                        info.memory = Some(extract_memory_def(db, op)?);
                    }
                    n if n == Symbol::new("data") => {
                        info.data.push(extract_data_def(db, op)?);
                    }
                    n if n == Symbol::new("table") => {
                        info.tables.push(extract_table_def(db, op)?);
                    }
                    n if n == Symbol::new("elem") => {
                        info.elements.push(extract_element_def(db, op)?);
                    }
                    n if n == Symbol::new("global") => {
                        info.globals.push(extract_global_def(db, op)?);
                    }
                    _ => {}
                }
            }
        }
    }

    Ok(())
}
fn collect_module_info<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
) -> CompilationResult<ModuleInfo<'db>> {
    let mut info = ModuleInfo::default();

    // Recursively collect wasm operations from the module and any nested core.module operations.
    collect_wasm_ops_from_region(db, &module.body(db), &mut info)?;

    // Collect GC types (structs, arrays)
    let (gc_types, mut type_idx_by_type, placeholder_struct_type_idx) =
        collect_gc_types(db, module)?;
    info.gc_types = gc_types;

    // Collect function types from call_indirect operations
    collect_call_indirect_types(db, module, &mut type_idx_by_type)?;

    info.type_idx_by_type = type_idx_by_type;
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

    Ok(info)
}

/// Collect GC types (structs, arrays) and return:
/// - Vec<GcTypeDef>: The GC type definitions (BoxedF64 is always at index 0)
/// - HashMap<Type, u32>: Mapping from Type to type index
type GcTypesResult<'db> = (
    Vec<GcTypeDef>,
    HashMap<Type<'db>, u32>,
    HashMap<(Type<'db>, usize), u32>,
);

fn collect_gc_types<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
) -> CompilationResult<GcTypesResult<'db>> {
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum GcKind {
        Struct,
        Array,
        Unknown,
    }

    struct GcTypeBuilder<'db> {
        kind: GcKind,
        fields: Vec<Option<Type<'db>>>,
        array_elem: Option<Type<'db>>,
        field_count: Option<usize>,
    }

    impl<'db> GcTypeBuilder<'db> {
        fn new() -> Self {
            Self {
                kind: GcKind::Unknown,
                fields: Vec::new(),
                array_elem: None,
                field_count: None,
            }
        }
    }

    let wasm_dialect = Symbol::new("wasm");
    let mut builders: Vec<GcTypeBuilder<'db>> = Vec::new();
    let mut type_idx_by_type: HashMap<Type<'db>, u32> = HashMap::new();
    // For placeholder types like wasm.structref, use (type, field_count) as key
    // to handle multiple structs with same placeholder type but different field counts
    let mut placeholder_struct_type_idx: HashMap<(Type<'db>, usize), u32> = HashMap::new();

    // Start at FIRST_USER_TYPE_IDX since indices 0-2 are reserved for built-in types:
    // 0: BoxedF64, 1: BytesArray, 2: BytesStruct
    let mut next_type_idx: u32 = FIRST_USER_TYPE_IDX;

    /// Returns true if this is a built-in type (0-2) that shouldn't use a builder
    fn is_builtin_type(idx: u32) -> bool {
        idx < FIRST_USER_TYPE_IDX
    }

    /// Get or create a builder for a user-defined type.
    /// Returns None for built-in types (0-2) which are predefined.
    fn try_get_builder<'db, 'a>(
        builders: &'a mut Vec<GcTypeBuilder<'db>>,
        idx: u32,
    ) -> Option<&'a mut GcTypeBuilder<'db>> {
        // Skip built-in types (0-2) as they are predefined
        if is_builtin_type(idx) {
            return None;
        }
        // Subtract FIRST_USER_TYPE_IDX because indices 0-2 are reserved for built-in types
        // User type indices start at FIRST_USER_TYPE_IDX
        let adjusted_idx = (idx - FIRST_USER_TYPE_IDX) as usize;
        if builders.len() <= adjusted_idx {
            builders.resize_with(adjusted_idx + 1, GcTypeBuilder::new);
        }
        Some(&mut builders[adjusted_idx])
    }

    fn register_type<'db>(type_idx_by_type: &mut HashMap<Type<'db>, u32>, idx: u32, ty: Type<'db>) {
        type_idx_by_type.entry(ty).or_insert(idx);
    }

    fn record_struct_field<'db>(
        type_idx: u32,
        builder: &mut GcTypeBuilder<'db>,
        field_idx: u32,
        ty: Type<'db>,
    ) -> CompilationResult<()> {
        if matches!(builder.field_count, Some(count) if field_idx as usize >= count) {
            let count = builder.field_count.expect("count checked by matches");
            return Err(CompilationError::type_error(format!(
                "struct type index {type_idx} field index {field_idx} out of bounds (fields: {count})",
            )));
        }
        let idx = field_idx as usize;
        if builder.fields.len() <= idx {
            builder.fields.resize_with(idx + 1, || None);
        }
        if let Some(existing) = builder.fields[idx] {
            if existing != ty {
                return Err(CompilationError::type_error(format!(
                    "struct type index {type_idx} field {field_idx} type mismatch",
                )));
            }
        } else {
            builder.fields[idx] = Some(ty);
        }
        Ok(())
    }

    fn record_array_elem<'db>(
        type_idx: u32,
        builder: &mut GcTypeBuilder<'db>,
        ty: Type<'db>,
    ) -> CompilationResult<()> {
        if let Some(existing) = builder.array_elem {
            if existing != ty {
                return Err(CompilationError::type_error(format!(
                    "array type index {type_idx} element type mismatch",
                )));
            }
        } else {
            builder.array_elem = Some(ty);
        }
        Ok(())
    }

    // Helper to get type_idx from attributes, supporting both type_idx and type attributes
    let get_type_idx = |attrs: &std::collections::BTreeMap<Symbol, Attribute<'db>>,
                        type_idx_by_type: &mut HashMap<Type<'db>, u32>,
                        next_type_idx: &mut u32|
     -> Option<u32> {
        // First try type_idx attribute
        if let Some(Attribute::IntBits(idx)) = attrs.get(&ATTR_TYPE_IDX()) {
            return Some(*idx as u32);
        }
        // Fall back to type attribute
        if let Some(Attribute::Type(ty)) = attrs.get(&ATTR_TYPE()) {
            if let Some(&idx) = type_idx_by_type.get(ty) {
                return Some(idx);
            }
            // Allocate new type_idx
            let idx = *next_type_idx;
            *next_type_idx += 1;
            type_idx_by_type.insert(*ty, idx);
            return Some(idx);
        }
        None
    };

    let mut visit_op = |op: &Operation<'db>| -> CompilationResult<()> {
        if op.dialect(db) != wasm_dialect {
            return Ok(());
        }
        let name = op.name(db);
        if name == Symbol::new("struct_new") {
            let attrs = op.attributes(db);
            let field_count = op.operands(db).len();

            // Check if this uses a placeholder type (wasm.structref) that allows
            // multiple structs with same type but different field counts
            let is_placeholder_type = attrs
                .get(&ATTR_TYPE())
                .map(|attr| {
                    if let Attribute::Type(ty) = attr {
                        wasm::Structref::from_type(db, *ty).is_some()
                    } else {
                        false
                    }
                })
                .unwrap_or(false);

            let type_idx = if is_placeholder_type {
                // For placeholder types, use (type, field_count) as key to allow
                // different field counts with same placeholder type
                let ty = match attrs.get(&ATTR_TYPE()) {
                    Some(Attribute::Type(ty)) => *ty,
                    _ => unreachable!("checked above"),
                };
                let key = (ty, field_count);
                if let Some(&idx) = placeholder_struct_type_idx.get(&key) {
                    idx
                } else {
                    let idx = next_type_idx;
                    next_type_idx += 1;
                    placeholder_struct_type_idx.insert(key, idx);
                    // Note: Placeholder types are NOT inserted into type_idx_by_type
                    // They are only stored in placeholder_struct_type_idx to avoid
                    // confusion and ensure proper lookup via (type, field_count) key
                    debug!(
                        "GC: struct_new allocated type_idx={} for placeholder (field_count={})",
                        idx, field_count
                    );
                    idx
                }
            } else {
                // For regular types, use standard type_idx lookup
                let Some(idx) = get_type_idx(attrs, &mut type_idx_by_type, &mut next_type_idx)
                else {
                    return Ok(());
                };
                idx
            };

            if let Some(builder) = try_get_builder(&mut builders, type_idx) {
                builder.kind = GcKind::Struct;

                // For placeholder types, we allow different field counts via different type_idx
                // For explicit type_idx, check for mismatch (error case)
                if !is_placeholder_type
                    && matches!(builder.field_count, Some(existing_count) if existing_count != field_count)
                {
                    let existing_count = builder.field_count.expect("count checked by matches");
                    return Err(CompilationError::type_error(format!(
                        "struct type index {type_idx} field count mismatch ({existing_count} vs {field_count})",
                    )));
                }

                builder.field_count = Some(field_count);
                if builder.fields.len() < field_count {
                    builder.fields.resize_with(field_count, || None);
                }

                if let Some(result_ty) = op.results(db).first().copied() {
                    register_type(&mut type_idx_by_type, type_idx, result_ty);
                }
                for (field_idx, value) in op.operands(db).iter().enumerate() {
                    if let Some(ty) = value_type(db, *value) {
                        record_struct_field(type_idx, builder, field_idx as u32, ty)?;
                    }
                }
            }
        } else if name == Symbol::new("struct_get") {
            let attrs = op.attributes(db);
            let Some(type_idx) = get_type_idx(attrs, &mut type_idx_by_type, &mut next_type_idx)
            else {
                return Ok(());
            };
            let field_idx = attr_field_idx(attrs)?;
            if let Some(builder) = try_get_builder(&mut builders, type_idx) {
                builder.kind = GcKind::Struct;
                if matches!(builder.field_count, Some(count) if field_idx as usize >= count) {
                    let count = builder.field_count.expect("count checked by matches");
                    return Err(CompilationError::type_error(format!(
                        "struct type index {type_idx} field index {field_idx} out of bounds (fields: {count})",
                    )));
                }
                if let Some(ty) = op
                    .operands(db)
                    .first()
                    .copied()
                    .and_then(|value| value_type(db, value))
                {
                    register_type(&mut type_idx_by_type, type_idx, ty);
                }
                // Only record field type if result type is concrete (not a type variable).
                // Type variables map to ANYREF and would conflict with concrete types.
                if let Some(result_ty) = op.results(db).first().copied()
                    && !tribute::is_type_var(db, result_ty)
                {
                    record_struct_field(type_idx, builder, field_idx, result_ty)?;
                }
            }
        } else if name == Symbol::new("struct_set") {
            let attrs = op.attributes(db);
            let Some(type_idx) = get_type_idx(attrs, &mut type_idx_by_type, &mut next_type_idx)
            else {
                return Ok(());
            };
            let field_idx = attr_field_idx(attrs)?;
            if let Some(builder) = try_get_builder(&mut builders, type_idx) {
                builder.kind = GcKind::Struct;
                if matches!(builder.field_count, Some(count) if field_idx as usize >= count) {
                    let count = builder.field_count.expect("count checked by matches");
                    return Err(CompilationError::type_error(format!(
                        "struct type index {type_idx} field index {field_idx} out of bounds (fields: {count})",
                    )));
                }
                if let Some(ty) = op
                    .operands(db)
                    .first()
                    .copied()
                    .and_then(|value| value_type(db, value))
                {
                    register_type(&mut type_idx_by_type, type_idx, ty);
                }
                if let Some(ty) = op
                    .operands(db)
                    .get(1)
                    .copied()
                    .and_then(|value| value_type(db, value))
                {
                    record_struct_field(type_idx, builder, field_idx, ty)?;
                }
            }
        } else if name == Symbol::new("array_new") || name == Symbol::new("array_new_default") {
            let attrs = op.attributes(db);
            let Some(type_idx) = get_type_idx(attrs, &mut type_idx_by_type, &mut next_type_idx)
            else {
                return Ok(());
            };
            if let Some(builder) = try_get_builder(&mut builders, type_idx) {
                builder.kind = GcKind::Array;
                if let Some(result_ty) = op.results(db).first().copied() {
                    register_type(&mut type_idx_by_type, type_idx, result_ty);
                }
                if let Some(ty) = op
                    .operands(db)
                    .get(1)
                    .copied()
                    .and_then(|value| value_type(db, value))
                {
                    record_array_elem(type_idx, builder, ty)?;
                }
            }
        } else if name == Symbol::new("array_get")
            || name == Symbol::new("array_get_s")
            || name == Symbol::new("array_get_u")
        {
            let attrs = op.attributes(db);
            let Some(type_idx) = get_type_idx(attrs, &mut type_idx_by_type, &mut next_type_idx)
            else {
                return Ok(());
            };
            if let Some(builder) = try_get_builder(&mut builders, type_idx) {
                builder.kind = GcKind::Array;
                if let Some(ty) = op
                    .operands(db)
                    .first()
                    .copied()
                    .and_then(|value| value_type(db, value))
                {
                    register_type(&mut type_idx_by_type, type_idx, ty);
                }
                // Only record element type if result type is concrete (not a type variable).
                if let Some(result_ty) = op.results(db).first().copied()
                    && !tribute::is_type_var(db, result_ty)
                {
                    record_array_elem(type_idx, builder, result_ty)?;
                }
            }
        } else if name == Symbol::new("array_set") {
            let attrs = op.attributes(db);
            let Some(type_idx) = get_type_idx(attrs, &mut type_idx_by_type, &mut next_type_idx)
            else {
                return Ok(());
            };
            if let Some(builder) = try_get_builder(&mut builders, type_idx) {
                builder.kind = GcKind::Array;
                if let Some(ty) = op
                    .operands(db)
                    .first()
                    .copied()
                    .and_then(|value| value_type(db, value))
                {
                    register_type(&mut type_idx_by_type, type_idx, ty);
                }
                if let Some(ty) = op
                    .operands(db)
                    .get(2)
                    .copied()
                    .and_then(|value| value_type(db, value))
                {
                    record_array_elem(type_idx, builder, ty)?;
                }
            }
        } else if name == Symbol::new("array_copy") {
            // array_copy has dst_type_idx: u32 and src_type_idx: u32 attributes
            let attrs = op.attributes(db);
            if let Some(&Attribute::IntBits(dst_idx)) = attrs.get(&Symbol::new("dst_type_idx")) {
                let dst_type_idx = dst_idx as u32;
                if let Some(builder) = try_get_builder(&mut builders, dst_type_idx) {
                    builder.kind = GcKind::Array;
                }
            }
            if let Some(&Attribute::IntBits(src_idx)) = attrs.get(&Symbol::new("src_type_idx")) {
                let src_type_idx = src_idx as u32;
                if let Some(builder) = try_get_builder(&mut builders, src_type_idx) {
                    builder.kind = GcKind::Array;
                }
            }
        } else if name == Symbol::new("ref_null")
            || name == Symbol::new("ref_cast")
            || name == Symbol::new("ref_test")
        {
            let attrs = op.attributes(db);
            // Try specific attribute names first, then fall back to generic "type" attribute
            let type_idx = if name == Symbol::new("ref_null") {
                attr_u32(attrs, ATTR_HEAP_TYPE())
                    .ok()
                    .or_else(|| get_type_idx(attrs, &mut type_idx_by_type, &mut next_type_idx))
            } else {
                attr_u32(attrs, ATTR_TARGET_TYPE())
                    .ok()
                    .or_else(|| get_type_idx(attrs, &mut type_idx_by_type, &mut next_type_idx))
            };
            let Some(type_idx) = type_idx else {
                return Ok(());
            };
            if let Some(result_ty) = op.results(db).first().copied() {
                register_type(&mut type_idx_by_type, type_idx, result_ty);
            }
            if let Some(builder) = try_get_builder(&mut builders, type_idx)
                && builder.kind == GcKind::Unknown
            {
                builder.kind = GcKind::Struct;
            }
        }
        Ok(())
    };

    // Recursively visit operations, including nested core.module operations.
    /// Recursively visit all operations in a region, including nested regions.
    fn visit_all_ops_recursive<'db>(
        db: &'db dyn salsa::Database,
        region: &Region<'db>,
        visit_op: &mut impl FnMut(&Operation<'db>) -> CompilationResult<()>,
    ) -> CompilationResult<()> {
        for block in region.blocks(db).iter() {
            for op in block.operations(db).iter() {
                visit_op(op)?;
                // Recursively visit nested regions (for wasm.if, wasm.block, etc.)
                for nested_region in op.regions(db).iter() {
                    visit_all_ops_recursive(db, nested_region, visit_op)?;
                }
            }
        }
        Ok(())
    }

    fn visit_region<'db>(
        db: &'db dyn salsa::Database,
        region: &Region<'db>,
        visit_op: &mut impl FnMut(&Operation<'db>) -> CompilationResult<()>,
    ) -> CompilationResult<()> {
        let wasm_dialect = Symbol::new("wasm");
        let core_dialect = Symbol::new("core");
        let module_name = Symbol::new("module");
        let func_name = Symbol::new("func");

        for block in region.blocks(db).iter() {
            for op in block.operations(db).iter() {
                let dialect = op.dialect(db);
                let name = op.name(db);

                // Recurse into nested core.module operations
                if dialect == core_dialect && name == module_name {
                    for nested_region in op.regions(db).iter() {
                        visit_region(db, nested_region, visit_op)?;
                    }
                    continue;
                }

                // Visit wasm.func body (recursively including nested regions)
                if dialect == wasm_dialect && name == func_name {
                    if let Some(func_region) = op.regions(db).first() {
                        visit_all_ops_recursive(db, func_region, visit_op)?;
                    }
                } else {
                    visit_op(op)?;
                    // Also visit nested regions for control flow ops (wasm.if, wasm.block, etc.)
                    for nested_region in op.regions(db).iter() {
                        visit_all_ops_recursive(db, nested_region, visit_op)?;
                    }
                }
            }
        }
        Ok(())
    }

    visit_region(db, &module.body(db), &mut visit_op)?;

    // Check if a type is a variant instance type (created by adt_to_wasm lowering).
    // Uses the is_variant attribute instead of name-based heuristics.
    let is_variant_type = |ty: Type<'db>| -> bool { adt::is_variant_instance_type(db, ty) };

    let to_field_type = |ty: Type<'db>, type_idx_by_type: &HashMap<_, _>| -> FieldType {
        debug!(
            "GC: to_field_type called with {}.{}",
            ty.dialect(db),
            ty.name(db)
        );
        let element_type = if core::I32::from_type(db, ty).is_some() {
            debug!("GC: to_field_type -> I32");
            StorageType::Val(ValType::I32)
        } else if core::I64::from_type(db, ty).is_some()
            || tribute::Int::from_type(db, ty).is_some()
            || tribute::Nat::from_type(db, ty).is_some()
        {
            // Int/Nat (arbitrary precision) is lowered to i64 for Phase 1
            // TODO: Implement i31ref/BigInt hybrid for WasmGC
            debug!("GC: to_field_type -> I64");
            StorageType::Val(ValType::I64)
        } else if core::F32::from_type(db, ty).is_some() {
            StorageType::Val(ValType::F32)
        } else if core::F64::from_type(db, ty).is_some() {
            StorageType::Val(ValType::F64)
        } else if is_variant_type(ty) {
            // For variant types (e.g., type.var$Num), use anyref to enable polymorphism.
            // Without proper WasmGC subtyping hierarchy, concrete struct refs
            // are not subtypes of anyref. Using anyref for all variant fields
            // allows values of any variant type to be stored uniformly.
            debug!(
                "GC: to_field_type {}.{} -> ANYREF (variant type needs polymorphism)",
                ty.dialect(db),
                ty.name(db)
            );
            StorageType::Val(ValType::Ref(RefType::ANYREF))
        } else if let Some(type_idx) = type_idx_by_type.get(&ty).copied() {
            // Non-variant struct types use concrete refs
            debug!(
                "GC: to_field_type {}.{} -> Concrete({})",
                ty.dialect(db),
                ty.name(db),
                type_idx
            );
            StorageType::Val(ValType::Ref(RefType {
                nullable: true,
                heap_type: HeapType::Concrete(type_idx),
            }))
        } else {
            debug!(
                "GC: to_field_type {}.{} -> ANYREF (not in type_idx_by_type)",
                ty.dialect(db),
                ty.name(db)
            );
            StorageType::Val(ValType::Ref(RefType::ANYREF))
        };
        FieldType {
            element_type,
            mutable: false,
        }
    };

    let mut result = Vec::new();
    for builder in builders {
        match builder.kind {
            GcKind::Array => {
                let elem = builder
                    .array_elem
                    .map(|ty| to_field_type(ty, &type_idx_by_type))
                    .unwrap_or(FieldType {
                        element_type: StorageType::Val(ValType::I32),
                        mutable: false,
                    });
                result.push(GcTypeDef::Array(elem));
            }
            GcKind::Struct | GcKind::Unknown => {
                let fields = builder
                    .fields
                    .into_iter()
                    .map(|ty| {
                        ty.map(|ty| to_field_type(ty, &type_idx_by_type))
                            .unwrap_or(FieldType {
                                element_type: StorageType::Val(ValType::I32),
                                mutable: false,
                            })
                    })
                    .collect::<Vec<_>>();
                result.push(GcTypeDef::Struct(fields));
            }
        }
    }

    // Insert built-in types at reserved indices (in reverse order since we insert at 0)
    // Index 2: BytesStruct (struct { data: ref BytesArray, offset: i32, len: i32 })
    let bytes_struct_type = GcTypeDef::Struct(vec![
        FieldType {
            element_type: StorageType::Val(ValType::Ref(RefType {
                nullable: false,
                heap_type: HeapType::Concrete(BYTES_ARRAY_IDX),
            })),
            mutable: false,
        },
        FieldType {
            element_type: StorageType::Val(ValType::I32),
            mutable: false,
        },
        FieldType {
            element_type: StorageType::Val(ValType::I32),
            mutable: false,
        },
    ]);
    result.insert(0, bytes_struct_type);

    // Index 1: BytesArray (array i8)
    // NOTE: mutable: true is required for array.copy operation in Bytes::concat
    let bytes_array_type = GcTypeDef::Array(FieldType {
        element_type: StorageType::I8,
        mutable: true,
    });
    result.insert(0, bytes_array_type);

    // Index 0: BoxedF64 (struct with single f64 field for Float boxing)
    let boxed_f64_type = GcTypeDef::Struct(vec![FieldType {
        element_type: StorageType::Val(ValType::F64),
        mutable: false,
    }]);
    result.insert(0, boxed_f64_type);

    Ok((result, type_idx_by_type, placeholder_struct_type_idx))
}

/// Collect function types from wasm.call_indirect operations.
///
/// This ensures that all function types used in call_indirect are registered
/// in the type section before emission.
fn collect_call_indirect_types<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
    type_idx_by_type: &mut HashMap<Type<'db>, u32>,
) -> CompilationResult<()> {
    fn collect_from_region<'db>(
        db: &'db dyn salsa::Database,
        region: &trunk_ir::Region<'db>,
        type_idx_by_type: &mut HashMap<Type<'db>, u32>,
        next_type_idx: &mut u32,
    ) -> CompilationResult<()> {
        for block in region.blocks(db).iter() {
            for op in block.operations(db).iter() {
                // Recursively process nested regions
                for nested in op.regions(db).iter() {
                    collect_from_region(db, nested, type_idx_by_type, next_type_idx)?;
                }

                // Check if this is a call_indirect
                if op.dialect(db) == Symbol::new("wasm")
                    && op.name(db) == Symbol::new("call_indirect")
                {
                    // Build function type from operands and results
                    let operands = op.operands(db);

                    if operands.is_empty() {
                        continue; // Skip invalid call_indirect
                    }

                    // Parameters: all operands except last (funcref)
                    let param_types: IdVec<Type<'db>> = operands
                        .iter()
                        .take(operands.len() - 1)
                        .filter_map(|v| value_type(db, *v))
                        .collect();

                    // Result type
                    let result_ty = match op.results(db).first().copied() {
                        Some(ty) => ty,
                        None => continue, // Skip if no result
                    };

                    // Create function type
                    let func_type = core::Func::new(db, param_types, result_ty).as_type();

                    // Register if not already registered
                    type_idx_by_type.entry(func_type).or_insert_with(|| {
                        let idx = *next_type_idx;
                        *next_type_idx += 1;
                        idx
                    });
                }
            }
        }
        Ok(())
    }

    // Start with the next available type index (after GC types)
    let mut next_type_idx = type_idx_by_type
        .values()
        .max()
        .map(|&idx| idx + 1)
        .unwrap_or(0);

    collect_from_region(db, &module.body(db), type_idx_by_type, &mut next_type_idx)?;

    Ok(())
}

fn extract_function_def<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
) -> CompilationResult<FunctionDef<'db>> {
    let attrs = op.attributes(db);
    let name_attr = attrs.get(&ATTR_SYM_NAME()).ok_or_else(|| {
        CompilationError::from(errors::CompilationErrorKind::MissingAttribute("sym_name"))
    })?;
    let ty_attr = attrs.get(&ATTR_TYPE()).ok_or_else(|| {
        CompilationError::from(errors::CompilationErrorKind::MissingAttribute("type"))
    })?;

    let name = match name_attr {
        Attribute::Symbol(sym) => *sym,
        _ => {
            return Err(CompilationError::from(
                errors::CompilationErrorKind::InvalidAttribute("sym_name"),
            ));
        }
    };
    let ty = match ty_attr {
        Attribute::Type(ty) => *ty,
        _ => {
            return Err(CompilationError::from(
                errors::CompilationErrorKind::InvalidAttribute("type"),
            ));
        }
    };

    let func_ty = core::Func::from_type(db, ty)
        .ok_or_else(|| CompilationError::type_error("wasm.func requires core.func type"))?;

    Ok(FunctionDef {
        name,
        ty: func_ty,
        op: *op,
    })
}

fn extract_import_def<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
) -> CompilationResult<ImportFuncDef<'db>> {
    let attrs = op.attributes(db);
    let module = attr_symbol(attrs, ATTR_MODULE())?;
    let name = attr_symbol(attrs, ATTR_NAME())?;
    let sym = attr_symbol_ref_attr(attrs, ATTR_SYM_NAME())?;
    let ty = attr_type(attrs, ATTR_TYPE())?;

    let func_ty = core::Func::from_type(db, ty)
        .ok_or_else(|| CompilationError::type_error("wasm.import_func requires core.func type"))?;

    Ok(ImportFuncDef {
        sym,
        module,
        name,
        ty: func_ty,
    })
}

fn extract_export_func<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
) -> CompilationResult<ExportDef> {
    let attrs = op.attributes(db);
    let name = attr_string(attrs, ATTR_NAME())?;
    let func = attr_symbol_ref_attr(attrs, ATTR_FUNC())?;
    Ok(ExportDef {
        name,
        kind: ExportKind::Func,
        target: ExportTarget::Func(func),
    })
}

fn extract_export_memory<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
) -> CompilationResult<ExportDef> {
    let attrs = op.attributes(db);
    let name = attr_string(attrs, ATTR_NAME())?;
    let index = attr_u32(attrs, ATTR_INDEX())?;
    Ok(ExportDef {
        name,
        kind: ExportKind::Memory,
        target: ExportTarget::Memory(index),
    })
}

fn extract_memory_def<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
) -> CompilationResult<MemoryDef> {
    let attrs = op.attributes(db);
    let min = attr_u32(attrs, ATTR_MIN())?;
    let max = match attrs.get(&ATTR_MAX()) {
        Some(Attribute::IntBits(bits)) => Some(*bits as u32),
        _ => None,
    };
    let shared = match attrs.get(&ATTR_SHARED()) {
        Some(Attribute::Bool(value)) => *value,
        _ => false,
    };
    let memory64 = match attrs.get(&ATTR_MEMORY64()) {
        Some(Attribute::Bool(value)) => *value,
        _ => false,
    };
    Ok(MemoryDef {
        min,
        max,
        shared,
        memory64,
    })
}

fn extract_data_def<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
) -> CompilationResult<DataDef> {
    let attrs = op.attributes(db);
    let passive = matches!(attrs.get(&ATTR_PASSIVE()), Some(Attribute::Bool(true)));
    let offset = if passive {
        0 // Passive segments don't have an offset
    } else {
        attr_i32_attr(attrs, ATTR_OFFSET())?
    };
    let bytes = match attrs.get(&ATTR_BYTES()) {
        Some(Attribute::Bytes(value)) => value.clone(),
        _ => {
            return Err(CompilationError::from(
                errors::CompilationErrorKind::InvalidAttribute("bytes"),
            ));
        }
    };
    Ok(DataDef {
        offset,
        bytes,
        passive,
    })
}

fn extract_table_def<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
) -> CompilationResult<TableDef> {
    let table_op = wasm::Table::from_operation(db, *op)
        .map_err(|_| CompilationError::invalid_operation("wasm.table"))?;
    let reftype_sym = table_op.reftype(db);
    let reftype = reftype_sym.with_str(|s| match s {
        "funcref" => Ok(RefType::FUNCREF),
        "externref" => Ok(RefType::EXTERNREF),
        other => Err(CompilationError::from(
            errors::CompilationErrorKind::InvalidAttribute(Box::leak(
                format!("reftype: {}", other).into_boxed_str(),
            )),
        )),
    })?;
    let min = table_op.min(db);
    let max = table_op.max(db);
    Ok(TableDef { reftype, min, max })
}

fn extract_element_def<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
) -> CompilationResult<ElementDef> {
    let elem_op = wasm::Elem::from_operation(db, *op)
        .map_err(|_| CompilationError::invalid_operation("wasm.elem"))?;
    let table = elem_op.table(db).unwrap_or(0);
    let offset = elem_op.offset(db).unwrap_or(0);

    // Collect function references from the funcs region
    let funcs_region = elem_op.funcs(db);
    let mut funcs = Vec::new();
    for block in funcs_region.blocks(db).iter() {
        for inner_op in block.operations(db).iter() {
            // Look for func.constant operations
            if let Ok(const_op) = func::Constant::from_operation(db, *inner_op) {
                funcs.push(const_op.func_ref(db));
            }
        }
    }

    Ok(ElementDef {
        table,
        offset,
        funcs,
    })
}

fn extract_global_def<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
) -> CompilationResult<GlobalDef> {
    let global_op = wasm::Global::from_operation(db, *op)
        .map_err(|_| CompilationError::invalid_operation("wasm.global"))?;
    let valtype_sym = global_op.valtype(db);
    let valtype = valtype_sym.with_str(|s| match s {
        "i32" => Ok(ValType::I32),
        "i64" => Ok(ValType::I64),
        "f32" => Ok(ValType::F32),
        "f64" => Ok(ValType::F64),
        "funcref" => Ok(ValType::Ref(RefType::FUNCREF)),
        "externref" => Ok(ValType::Ref(RefType::EXTERNREF)),
        "anyref" => Ok(ValType::Ref(RefType::ANYREF)),
        other => Err(CompilationError::from(
            errors::CompilationErrorKind::InvalidAttribute(Box::leak(
                format!("valtype: {}", other).into_boxed_str(),
            )),
        )),
    })?;
    let mutable = global_op.mutable(db);
    let init = global_op.init(db);
    Ok(GlobalDef {
        valtype,
        mutable,
        init,
    })
}

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

    let mut ctx = FunctionEmitContext {
        value_locals: HashMap::new(),
        effective_types: HashMap::new(),
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
            if let Some(result_ty) = result_types
                .first()
                .copied()
                .filter(|ty| !is_nil_type(db, *ty))
            {
                // For generic function calls, infer the concrete return type from operands.
                // This ensures the local is typed correctly for unboxed values.
                let mut effective_ty =
                    infer_call_result_type(db, op, result_ty, &module_info.func_types);

                // For struct_get on variant types with type.var result, the local type
                // must match the struct field type, not the IR result type.
                // The IR may have a generic type.var, but the WASM struct.get returns
                // the actual field type (e.g., I64 for Num variant's Int field).
                // We check if the result type is type.var and the struct type is a variant.
                // Use variant tag to determine field type semantics.
                if op.dialect(db) == Symbol::new("wasm")
                    && op.name(db) == Symbol::new("struct_get")
                    && tribute::is_type_var(db, effective_ty)
                {
                    if let Some(Attribute::Type(struct_ty)) = op.attributes(db).get(&ATTR_TYPE()) {
                        debug!(
                            "struct_get type attr: {}.{}",
                            struct_ty.dialect(db),
                            struct_ty.name(db)
                        );
                        // For variant types, look up the actual field type from adt.enum type.
                        // The base_enum attribute contains the adt.enum type with variant info.
                        if adt::is_variant_instance_type(db, *struct_ty)
                            && let (Some(base_enum_ty), Some(variant_tag)) = (
                                adt::get_base_enum(db, *struct_ty),
                                adt::get_variant_tag(db, *struct_ty),
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
                                effective_ty = tribute::Int::new(db).as_type();
                            } else if type_name == Symbol::new("Nat") {
                                effective_ty = tribute::Nat::new(db).as_type();
                            } else if type_name == Symbol::new("Float") {
                                effective_ty = core::F64::new(db).as_type();
                            } else if type_name == Symbol::new("Bool") {
                                effective_ty = core::I1::new(db).as_type();
                            }
                            // For other types (like Expr), keep effective_ty
                            // as type.var which maps to anyref
                        }
                    } else {
                        debug!(
                            "struct_get has no type attr, result_ty: {}.{}",
                            effective_ty.dialect(db),
                            effective_ty.name(db)
                        );
                    }
                }

                // For wasm.if with type.var result, infer the effective type from the
                // then branch's result value. This ensures the local type matches the
                // actual value produced by the branches.
                // Try to get effective type from the then region's result value
                if op.dialect(db) == Symbol::new("wasm")
                    && op.name(db) == Symbol::new("if")
                    && tribute::is_type_var(db, effective_ty)
                    && let Some(then_region) = op.regions(db).first()
                    && let Some(then_result) = region_result_value(db, then_region)
                    && let Some(eff_ty) = ctx.effective_types.get(&then_result)
                    && !tribute::is_type_var(db, *eff_ty)
                {
                    debug!(
                        "wasm.if local: using then branch effective type {}.{} instead of IR type {}.{}",
                        eff_ty.dialect(db),
                        eff_ty.name(db),
                        effective_ty.dialect(db),
                        effective_ty.name(db)
                    );
                    effective_ty = *eff_ty;
                }

                let val_type =
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

    // Helper to get type_idx from attributes, supporting both type_idx and type attributes
    let get_type_idx_from_attrs =
        |attrs: &std::collections::BTreeMap<Symbol, Attribute<'db>>| -> Option<u32> {
            // First try type_idx attribute
            if let Some(Attribute::IntBits(idx)) = attrs.get(&ATTR_TYPE_IDX()) {
                return Some(*idx as u32);
            }
            // Fall back to type attribute
            if let Some(Attribute::Type(ty)) = attrs.get(&ATTR_TYPE()) {
                return module_info.type_idx_by_type.get(ty).copied();
            }
            None
        };

    let name = op.name(db);
    let operands = op.operands(db);

    debug!("emit_op: {}.{}", op.dialect(db), name);

    // Skip wasm.nop - it's a placeholder for nil constants
    // No WASM instruction is emitted, and nil values have no local mapping
    if name == Symbol::new("nop") {
        return Ok(());
    }

    // Fast path: simple operations (emit operands → instruction → set result)
    if let Some(instr) = SIMPLE_OPS.get(&name) {
        emit_operands(db, operands, ctx, function)?;
        function.instruction(instr);
        set_result_local(db, op, ctx, function)?;
        return Ok(());
    }

    // Special cases: const, control flow, calls, locals, GC ops
    if name == Symbol::new("i32_const") {
        let value = attr_i32(db, op, ATTR_VALUE())?;
        function.instruction(&Instruction::I32Const(value));
        set_result_local(db, op, ctx, function)?;
    } else if name == Symbol::new("i64_const") {
        let value = attr_i64(db, op, ATTR_VALUE())?;
        function.instruction(&Instruction::I64Const(value));
        set_result_local(db, op, ctx, function)?;
    } else if name == Symbol::new("f32_const") {
        let value = attr_f32(db, op, ATTR_VALUE())?;
        function.instruction(&Instruction::F32Const(value.into()));
        set_result_local(db, op, ctx, function)?;
    } else if name == Symbol::new("f64_const") {
        let value = attr_f64(db, op, ATTR_VALUE())?;
        function.instruction(&Instruction::F64Const(value.into()));
        set_result_local(db, op, ctx, function)?;
    } else if name == Symbol::new("if") {
        let result_ty = op.results(db).first().copied();
        let has_result = matches!(result_ty, Some(ty) if !is_nil_type(db, ty));

        // For wasm.if with results, we need to determine the actual block type.
        // If the IR result type is type.var (anyref) but the effective result type
        // from the then/else branches is concrete (like I64), we must use the
        // effective type for the block type.
        let block_type = if has_result {
            let ir_ty = result_ty.expect("if result type");
            // Check if the branch result values have an effective type computed
            // If the IR type is a type variable and we have a different effective type,
            // use the effective type for the block
            let effective_ty = if tribute::is_type_var(db, ir_ty) {
                // Try to get effective type from the then region's result value
                let regions = op.regions(db);
                let then_result_ty = regions
                    .first()
                    .and_then(|r| region_result_value(db, r))
                    .and_then(|v| ctx.effective_types.get(&v).copied());

                if let Some(eff_ty) = then_result_ty {
                    if !tribute::is_type_var(db, eff_ty) {
                        debug!(
                            "wasm.if: using then branch effective type {}.{} instead of IR type {}.{}",
                            eff_ty.dialect(db),
                            eff_ty.name(db),
                            ir_ty.dialect(db),
                            ir_ty.name(db)
                        );
                        eff_ty
                    } else {
                        ir_ty
                    }
                } else {
                    ir_ty
                }
            } else {
                ir_ty
            };

            BlockType::Result(type_to_valtype(
                db,
                effective_ty,
                &module_info.type_idx_by_type,
            )?)
        } else {
            BlockType::Empty
        };
        if operands.len() != 1 {
            return Err(CompilationError::invalid_module(
                "wasm.if expects a single condition operand",
            ));
        }
        emit_operands(db, operands, ctx, function)?;
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
    } else if name == Symbol::new("block") {
        let result_ty = op.results(db).first().copied();
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
    } else if name == Symbol::new("loop") {
        let result_ty = op.results(db).first().copied();
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
    } else if name == Symbol::new("br") {
        let depth = attr_u32(op.attributes(db), ATTR_TARGET())?;
        function.instruction(&Instruction::Br(depth));
    } else if name == Symbol::new("br_if") {
        if operands.len() != 1 {
            return Err(CompilationError::invalid_module(
                "wasm.br_if expects a single condition operand",
            ));
        }
        emit_operands(db, operands, ctx, function)?;
        let depth = attr_u32(op.attributes(db), ATTR_TARGET())?;
        function.instruction(&Instruction::BrIf(depth));
    } else if name == Symbol::new("call") {
        let callee = attr_symbol_ref(db, op, ATTR_CALLEE())?;
        let target = resolve_callee(callee, module_info)?;

        // Check if we need boxing for generic function calls
        if let Some(callee_ty) = module_info.func_types.get(&callee) {
            let param_types = callee_ty.params(db);
            emit_operands_with_boxing(db, operands, &param_types, ctx, module_info, function)?;
        } else {
            emit_operands(db, operands, ctx, function)?;
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
                    .and_then(|v| value_type(db, *v))
                    .filter(|ty| !tribute::is_type_var(db, *ty))
                {
                    emit_unboxing(db, operand_ty, function)?;
                }
            }
        }

        set_result_local(db, op, ctx, function)?;
    } else if name == Symbol::new("call_indirect") {
        // wasm.call_indirect: indirect function call through a funcref
        // Operands: [arg1, arg2, ..., argN, funcref]
        // The funcref is the last operand (on top of stack in WebAssembly)

        // Get or compute type_idx
        let type_idx = match attr_u32(op.attributes(db), Symbol::new("type_idx")) {
            Ok(idx) => idx,
            Err(_) => {
                // type_idx not provided (from func_to_wasm) - compute from signature
                // Signature: (param1, param2, ..., paramN) -> result
                // where params are all operands except the last (funcref)

                if operands.is_empty() {
                    return Err(CompilationError::invalid_module(
                        "wasm.call_indirect requires at least a funcref operand",
                    ));
                }

                // Build parameter types (all operands except last)
                let param_types: Vec<Type<'db>> = operands
                    .iter()
                    .take(operands.len() - 1)
                    .filter_map(|v| value_type(db, *v))
                    .collect();

                // Get result type
                let result_ty = op.results(db).first().copied().ok_or_else(|| {
                    CompilationError::invalid_module("wasm.call_indirect must have a result type")
                })?;

                // Construct function type
                let func_type =
                    core::Func::new(db, param_types.into_iter().collect(), result_ty).as_type();

                // Look up type index
                module_info
                    .type_idx_by_type
                    .get(&func_type)
                    .copied()
                    .ok_or_else(|| {
                        CompilationError::invalid_module(
                            "wasm.call_indirect function type not registered in type section",
                        )
                    })?
            }
        };
        let table = attr_u32(op.attributes(db), Symbol::new("table")).unwrap_or(0);

        // Emit all operands (arguments first, then funcref)
        emit_operands(db, operands, ctx, function)?;

        // Emit call_indirect instruction
        function.instruction(&Instruction::CallIndirect {
            type_index: type_idx,
            table_index: table,
        });

        set_result_local(db, op, ctx, function)?;
    } else if name == Symbol::new("return_call") {
        let callee = attr_symbol_ref(db, op, ATTR_CALLEE())?;
        let target = resolve_callee(callee, module_info)?;

        // Check if we need boxing for generic function calls
        if let Some(callee_ty) = module_info.func_types.get(&callee) {
            let param_types = callee_ty.params(db);
            emit_operands_with_boxing(db, operands, &param_types, ctx, module_info, function)?;
        } else {
            emit_operands(db, operands, ctx, function)?;
        }

        // Note: Return unboxing is not needed for tail calls since
        // the caller's return type should match the callee's.
        function.instruction(&Instruction::ReturnCall(target));
    } else if name == Symbol::new("local_get") {
        let index = attr_index(db, op)?;
        function.instruction(&Instruction::LocalGet(index));
        set_result_local(db, op, ctx, function)?;
    } else if name == Symbol::new("local_set") {
        let index = attr_index(db, op)?;
        emit_operands(db, operands, ctx, function)?;
        function.instruction(&Instruction::LocalSet(index));
    } else if name == Symbol::new("local_tee") {
        let index = attr_index(db, op)?;
        emit_operands(db, operands, ctx, function)?;
        function.instruction(&Instruction::LocalTee(index));
        set_result_local(db, op, ctx, function)?;
    } else if name == Symbol::new("global_get") {
        let index = attr_index(db, op)?;
        function.instruction(&Instruction::GlobalGet(index));
        set_result_local(db, op, ctx, function)?;
    } else if name == Symbol::new("global_set") {
        let index = attr_index(db, op)?;
        emit_operands(db, operands, ctx, function)?;
        function.instruction(&Instruction::GlobalSet(index));
    } else if name == Symbol::new("struct_new") {
        emit_operands(db, operands, ctx, function)?;
        let attrs = op.attributes(db);
        let field_count = operands.len();

        // Check if this uses a placeholder type (wasm.structref)
        let type_idx = if let Some(Attribute::Type(ty)) = attrs.get(&ATTR_TYPE()) {
            if wasm::Structref::from_type(db, *ty).is_some() {
                // Use placeholder map for wasm.structref
                // All (type, field_count) pairs are registered by collect_gc_types upfront
                module_info
                    .placeholder_struct_type_idx
                    .get(&(*ty, field_count))
                    .copied()
            } else {
                // Regular type
                module_info.type_idx_by_type.get(ty).copied()
            }
        } else if let Some(Attribute::IntBits(idx)) = attrs.get(&ATTR_TYPE_IDX()) {
            Some(*idx as u32)
        } else {
            None
        }
        .ok_or_else(|| CompilationError::missing_attribute("type or type_idx"))?;

        function.instruction(&Instruction::StructNew(type_idx));
        set_result_local(db, op, ctx, function)?;
    } else if name == Symbol::new("struct_get") {
        emit_operands(db, operands, ctx, function)?;
        let attrs = op.attributes(db);
        let type_idx = get_type_idx_from_attrs(attrs)
            .ok_or_else(|| CompilationError::missing_attribute("type or type_idx"))?;
        let field_idx = attr_field_idx(attrs)?;
        function.instruction(&Instruction::StructGet {
            struct_type_index: type_idx,
            field_index: field_idx,
        });
        set_result_local(db, op, ctx, function)?;
    } else if name == Symbol::new("struct_set") {
        emit_operands(db, operands, ctx, function)?;
        let attrs = op.attributes(db);
        let type_idx = get_type_idx_from_attrs(attrs)
            .ok_or_else(|| CompilationError::missing_attribute("type or type_idx"))?;
        let field_idx = attr_field_idx(attrs)?;
        function.instruction(&Instruction::StructSet {
            struct_type_index: type_idx,
            field_index: field_idx,
        });
    } else if name == Symbol::new("array_new") {
        emit_operands(db, operands, ctx, function)?;
        let attrs = op.attributes(db);
        let type_idx = get_type_idx_from_attrs(attrs)
            .ok_or_else(|| CompilationError::missing_attribute("type or type_idx"))?;
        function.instruction(&Instruction::ArrayNew(type_idx));
        set_result_local(db, op, ctx, function)?;
    } else if name == Symbol::new("array_new_default") {
        emit_operands(db, operands, ctx, function)?;
        let attrs = op.attributes(db);
        let type_idx = get_type_idx_from_attrs(attrs)
            .ok_or_else(|| CompilationError::missing_attribute("type or type_idx"))?;
        function.instruction(&Instruction::ArrayNewDefault(type_idx));
        set_result_local(db, op, ctx, function)?;
    } else if name == Symbol::new("array_get") {
        emit_operands(db, operands, ctx, function)?;
        let attrs = op.attributes(db);
        let type_idx = get_type_idx_from_attrs(attrs)
            .ok_or_else(|| CompilationError::missing_attribute("type or type_idx"))?;
        function.instruction(&Instruction::ArrayGet(type_idx));
        set_result_local(db, op, ctx, function)?;
    } else if name == Symbol::new("array_get_s") {
        emit_operands(db, operands, ctx, function)?;
        let attrs = op.attributes(db);
        let type_idx = get_type_idx_from_attrs(attrs)
            .ok_or_else(|| CompilationError::missing_attribute("type or type_idx"))?;
        function.instruction(&Instruction::ArrayGetS(type_idx));
        set_result_local(db, op, ctx, function)?;
    } else if name == Symbol::new("array_get_u") {
        emit_operands(db, operands, ctx, function)?;
        let attrs = op.attributes(db);
        let type_idx = get_type_idx_from_attrs(attrs)
            .ok_or_else(|| CompilationError::missing_attribute("type or type_idx"))?;
        function.instruction(&Instruction::ArrayGetU(type_idx));
        set_result_local(db, op, ctx, function)?;
    } else if name == Symbol::new("array_set") {
        emit_operands(db, operands, ctx, function)?;
        let attrs = op.attributes(db);
        let type_idx = get_type_idx_from_attrs(attrs)
            .ok_or_else(|| CompilationError::missing_attribute("type or type_idx"))?;
        function.instruction(&Instruction::ArraySet(type_idx));
    } else if name == Symbol::new("array_copy") {
        emit_operands(db, operands, ctx, function)?;
        let attrs = op.attributes(db);
        let dst_type_idx = attrs
            .get(&Symbol::new("dst_type_idx"))
            .and_then(|a| match a {
                Attribute::IntBits(v) => Some(*v as u32),
                _ => None,
            })
            .ok_or_else(|| CompilationError::missing_attribute("dst_type_idx"))?;
        let src_type_idx = attrs
            .get(&Symbol::new("src_type_idx"))
            .and_then(|a| match a {
                Attribute::IntBits(v) => Some(*v as u32),
                _ => None,
            })
            .ok_or_else(|| CompilationError::missing_attribute("src_type_idx"))?;
        function.instruction(&Instruction::ArrayCopy {
            array_type_index_dst: dst_type_idx,
            array_type_index_src: src_type_idx,
        });
    } else if name == Symbol::new("ref_null") {
        let attrs = op.attributes(db);
        let heap_type = attr_heap_type(attrs, ATTR_HEAP_TYPE())
            .ok()
            .or_else(|| get_type_idx_from_attrs(attrs).map(HeapType::Concrete))
            .ok_or_else(|| CompilationError::missing_attribute("heap_type or type"))?;
        function.instruction(&Instruction::RefNull(heap_type));
        set_result_local(db, op, ctx, function)?;
    } else if name == Symbol::new("ref_func") {
        // wasm.ref_func: create a funcref from function name
        let func_name = attr_symbol_ref(db, op, ATTR_FUNC_NAME())?;
        let func_idx = resolve_callee(func_name, module_info)?;
        function.instruction(&Instruction::RefFunc(func_idx));
        set_result_local(db, op, ctx, function)?;
    } else if name == Symbol::new("ref_cast") {
        emit_operands(db, operands, ctx, function)?;
        let attrs = op.attributes(db);
        let heap_type = attr_heap_type(attrs, ATTR_TARGET_TYPE())
            .ok()
            .or_else(|| get_type_idx_from_attrs(attrs).map(HeapType::Concrete))
            .ok_or_else(|| CompilationError::missing_attribute("target_type or type"))?;
        function.instruction(&Instruction::RefCastNullable(heap_type));
        set_result_local(db, op, ctx, function)?;
    } else if name == Symbol::new("ref_test") {
        emit_operands(db, operands, ctx, function)?;
        let attrs = op.attributes(db);
        let heap_type = attr_heap_type(attrs, ATTR_TARGET_TYPE())
            .ok()
            .or_else(|| get_type_idx_from_attrs(attrs).map(HeapType::Concrete))
            .ok_or_else(|| CompilationError::missing_attribute("target_type or type"))?;
        function.instruction(&Instruction::RefTestNullable(heap_type));
        set_result_local(db, op, ctx, function)?;
    } else if name == Symbol::new("bytes_from_data") {
        // Compound operation: create Bytes struct from passive data segment
        // Stack operations:
        //   i32.const <offset>    ; offset within data segment
        //   i32.const <len>       ; number of bytes to copy
        //   array.new_data $bytes_array <data_idx>
        //   i32.const 0           ; offset field (we use the whole array)
        //   i32.const <len>       ; len field
        //   struct.new $bytes_struct
        let attrs = op.attributes(db);
        let data_idx = attr_u32(attrs, ATTR_DATA_IDX())?;
        let offset = attr_u32(attrs, ATTR_OFFSET())?;
        let len = attr_u32(attrs, ATTR_LEN())?;

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
    } else if name == Symbol::new("memory_size") {
        let memory = extract_memory_index(db, op);
        function.instruction(&Instruction::MemorySize(memory));
        set_result_local(db, op, ctx, function)?;
    } else if name == Symbol::new("memory_grow") {
        emit_operands(db, operands, ctx, function)?;
        let memory = extract_memory_index(db, op);
        function.instruction(&Instruction::MemoryGrow(memory));
        set_result_local(db, op, ctx, function)?;

    // === Full-Width Loads ===
    } else if name == Symbol::new("i32_load") {
        emit_operands(db, operands, ctx, function)?;
        let memarg = extract_memarg(db, op, 2); // natural align: 4 bytes = log2(4) = 2
        function.instruction(&Instruction::I32Load(memarg));
        set_result_local(db, op, ctx, function)?;
    } else if name == Symbol::new("i64_load") {
        emit_operands(db, operands, ctx, function)?;
        let memarg = extract_memarg(db, op, 3); // natural align: 8 bytes = log2(8) = 3
        function.instruction(&Instruction::I64Load(memarg));
        set_result_local(db, op, ctx, function)?;
    } else if name == Symbol::new("f32_load") {
        emit_operands(db, operands, ctx, function)?;
        let memarg = extract_memarg(db, op, 2);
        function.instruction(&Instruction::F32Load(memarg));
        set_result_local(db, op, ctx, function)?;
    } else if name == Symbol::new("f64_load") {
        emit_operands(db, operands, ctx, function)?;
        let memarg = extract_memarg(db, op, 3);
        function.instruction(&Instruction::F64Load(memarg));
        set_result_local(db, op, ctx, function)?;

    // === Partial-Width Loads (i32) ===
    } else if name == Symbol::new("i32_load8_s") {
        emit_operands(db, operands, ctx, function)?;
        let memarg = extract_memarg(db, op, 0); // natural align: 1 byte = log2(1) = 0
        function.instruction(&Instruction::I32Load8S(memarg));
        set_result_local(db, op, ctx, function)?;
    } else if name == Symbol::new("i32_load8_u") {
        emit_operands(db, operands, ctx, function)?;
        let memarg = extract_memarg(db, op, 0);
        function.instruction(&Instruction::I32Load8U(memarg));
        set_result_local(db, op, ctx, function)?;
    } else if name == Symbol::new("i32_load16_s") {
        emit_operands(db, operands, ctx, function)?;
        let memarg = extract_memarg(db, op, 1); // natural align: 2 bytes = log2(2) = 1
        function.instruction(&Instruction::I32Load16S(memarg));
        set_result_local(db, op, ctx, function)?;
    } else if name == Symbol::new("i32_load16_u") {
        emit_operands(db, operands, ctx, function)?;
        let memarg = extract_memarg(db, op, 1);
        function.instruction(&Instruction::I32Load16U(memarg));
        set_result_local(db, op, ctx, function)?;

    // === Partial-Width Loads (i64) ===
    } else if name == Symbol::new("i64_load8_s") {
        emit_operands(db, operands, ctx, function)?;
        let memarg = extract_memarg(db, op, 0);
        function.instruction(&Instruction::I64Load8S(memarg));
        set_result_local(db, op, ctx, function)?;
    } else if name == Symbol::new("i64_load8_u") {
        emit_operands(db, operands, ctx, function)?;
        let memarg = extract_memarg(db, op, 0);
        function.instruction(&Instruction::I64Load8U(memarg));
        set_result_local(db, op, ctx, function)?;
    } else if name == Symbol::new("i64_load16_s") {
        emit_operands(db, operands, ctx, function)?;
        let memarg = extract_memarg(db, op, 1);
        function.instruction(&Instruction::I64Load16S(memarg));
        set_result_local(db, op, ctx, function)?;
    } else if name == Symbol::new("i64_load16_u") {
        emit_operands(db, operands, ctx, function)?;
        let memarg = extract_memarg(db, op, 1);
        function.instruction(&Instruction::I64Load16U(memarg));
        set_result_local(db, op, ctx, function)?;
    } else if name == Symbol::new("i64_load32_s") {
        emit_operands(db, operands, ctx, function)?;
        let memarg = extract_memarg(db, op, 2);
        function.instruction(&Instruction::I64Load32S(memarg));
        set_result_local(db, op, ctx, function)?;
    } else if name == Symbol::new("i64_load32_u") {
        emit_operands(db, operands, ctx, function)?;
        let memarg = extract_memarg(db, op, 2);
        function.instruction(&Instruction::I64Load32U(memarg));
        set_result_local(db, op, ctx, function)?;

    // === Full-Width Stores ===
    } else if name == Symbol::new("i32_store") {
        emit_operands(db, operands, ctx, function)?;
        let memarg = extract_memarg(db, op, 2);
        function.instruction(&Instruction::I32Store(memarg));
        // No set_result_local - stores don't return a value
    } else if name == Symbol::new("i64_store") {
        emit_operands(db, operands, ctx, function)?;
        let memarg = extract_memarg(db, op, 3);
        function.instruction(&Instruction::I64Store(memarg));
    } else if name == Symbol::new("f32_store") {
        emit_operands(db, operands, ctx, function)?;
        let memarg = extract_memarg(db, op, 2);
        function.instruction(&Instruction::F32Store(memarg));
    } else if name == Symbol::new("f64_store") {
        emit_operands(db, operands, ctx, function)?;
        let memarg = extract_memarg(db, op, 3);
        function.instruction(&Instruction::F64Store(memarg));

    // === Partial-Width Stores ===
    } else if name == Symbol::new("i32_store8") {
        emit_operands(db, operands, ctx, function)?;
        let memarg = extract_memarg(db, op, 0);
        function.instruction(&Instruction::I32Store8(memarg));
    } else if name == Symbol::new("i32_store16") {
        emit_operands(db, operands, ctx, function)?;
        let memarg = extract_memarg(db, op, 1);
        function.instruction(&Instruction::I32Store16(memarg));
    } else if name == Symbol::new("i64_store8") {
        emit_operands(db, operands, ctx, function)?;
        let memarg = extract_memarg(db, op, 0);
        function.instruction(&Instruction::I64Store8(memarg));
    } else if name == Symbol::new("i64_store16") {
        emit_operands(db, operands, ctx, function)?;
        let memarg = extract_memarg(db, op, 1);
        function.instruction(&Instruction::I64Store16(memarg));
    } else if name == Symbol::new("i64_store32") {
        emit_operands(db, operands, ctx, function)?;
        let memarg = extract_memarg(db, op, 2);
        function.instruction(&Instruction::I64Store32(memarg));
    } else {
        tracing::error!("unsupported wasm op: {}", name);
        return Err(CompilationError::unsupported_feature(
            "wasm op not supported",
        ));
    }

    Ok(())
}

fn emit_operands<'db>(
    db: &'db dyn salsa::Database,
    operands: &IdVec<Value<'db>>,
    ctx: &FunctionEmitContext<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    for value in operands.iter() {
        // Skip nil type values - they have no runtime representation
        if let Some(ty) = value_type(db, *value)
            && is_nil_type(db, ty)
        {
            debug!(
                "  emit_operands: skipping nil type value {:?}",
                value.def(db)
            );
            continue;
        }

        // Try direct lookup first
        if let Some(index) = ctx.value_locals.get(value) {
            function.instruction(&Instruction::LocalGet(*index));
            continue;
        }

        // Handle stale block argument references (issue #43)
        // The resolver creates operands that reference OLD block arguments, but value_locals
        // only contains NEW block arguments. For block args, we can use the index directly
        // since parameters are always locals 0, 1, 2, etc.
        if let ValueDef::BlockArg(_block_id) = value.def(db) {
            let index = value.index(db) as u32;
            function.instruction(&Instruction::LocalGet(index));
            continue;
        }

        // If operand not found and not a block arg, this is an ERROR - stale value reference!
        if let ValueDef::OpResult(stale_op) = value.def(db) {
            // For tribute.var, try to find what it references by looking at its name attribute
            if stale_op.dialect(db) == tribute::DIALECT_NAME()
                && stale_op.name(db) == tribute::VAR()
            {
                if let Some(Attribute::Symbol(var_name)) =
                    stale_op.attributes(db).get(&Symbol::new("name"))
                {
                    tracing::error!(
                        "emit_operands: stale SSA value: tribute.var '{}' index={} (var references should have been resolved)",
                        var_name,
                        value.index(db)
                    );
                } else {
                    tracing::error!(
                        "emit_operands: stale SSA value: tribute.var (no name) index={}",
                        value.index(db)
                    );
                }
            } else {
                tracing::error!(
                    "emit_operands: stale SSA value: {}.{} index={}",
                    stale_op.dialect(db),
                    stale_op.name(db),
                    value.index(db)
                );
            }
            return Err(CompilationError::invalid_module(
                "stale SSA value in wasm backend (missing local mapping)",
            ));
        }
    }
    Ok(())
}

/// Emit operands with boxing when calling generic functions.
/// If a parameter expects anyref (type.var) but the operand is a concrete type (Int, Float),
/// we need to box the value.
fn emit_operands_with_boxing<'db>(
    db: &'db dyn salsa::Database,
    operands: &IdVec<Value<'db>>,
    param_types: &IdVec<Type<'db>>,
    ctx: &FunctionEmitContext<'db>,
    module_info: &ModuleInfo<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let mut param_iter = param_types.iter();

    for value in operands.iter() {
        // Get the corresponding parameter type first (must stay synchronized with operands)
        let param_ty = param_iter.next();

        // Skip nil type values - they have no runtime representation
        if let Some(ty) = value_type(db, *value)
            && is_nil_type(db, ty)
        {
            continue;
        }

        // Emit the value (local.get)
        emit_value(db, *value, ctx, function)?;

        // Check if boxing is needed
        // If parameter expects anyref (type.var) AND doesn't have a concrete type index, box the operand
        // Types with a type index (like struct types) are already reference types and don't need boxing
        // Use effective_types to get the actual computed type, falling back to IR type
        if param_ty.is_some_and(|ty| {
            tribute::is_type_var(db, *ty) && !module_info.type_idx_by_type.contains_key(ty)
        }) {
            // Use effective type if available (computed during local allocation),
            // otherwise fall back to IR result type
            let operand_ty = ctx
                .effective_types
                .get(value)
                .copied()
                .or_else(|| value_type(db, *value));
            if let Some(operand_ty) = operand_ty {
                debug!(
                    "emit_operands_with_boxing: param expects anyref, operand effective_ty={}.{}",
                    operand_ty.dialect(db),
                    operand_ty.name(db)
                );
                emit_boxing(db, operand_ty, function)?;
            }
        }
    }
    Ok(())
}

/// Emit a single value (local.get or block arg fallback).
fn emit_value<'db>(
    db: &'db dyn salsa::Database,
    value: Value<'db>,
    ctx: &FunctionEmitContext<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    // Try direct lookup first
    if let Some(index) = ctx.value_locals.get(&value) {
        function.instruction(&Instruction::LocalGet(*index));
        return Ok(());
    }

    // Handle stale block argument references
    if let ValueDef::BlockArg(_block_id) = value.def(db) {
        let index = value.index(db) as u32;
        function.instruction(&Instruction::LocalGet(index));
        return Ok(());
    }

    // If operand not found and not a block arg, this is an error
    if let ValueDef::OpResult(stale_op) = value.def(db) {
        tracing::error!(
            "stale SSA value: {}.{} index={}",
            stale_op.dialect(db),
            stale_op.name(db),
            value.index(db)
        );
    }
    Err(CompilationError::invalid_module(
        "stale SSA value in wasm backend (missing local mapping)",
    ))
}

/// Emit boxing instructions to convert a concrete type to anyref.
/// - Int (i64) → i31ref: truncate to i32 and use ref.i31
/// - Float (f64) → BoxedF64 struct: wrap in a struct with single f64 field
fn emit_boxing<'db>(
    db: &'db dyn salsa::Database,
    ty: Type<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    debug!("emit_boxing: type={}.{}", ty.dialect(db), ty.name(db));
    if tribute::Int::from_type(db, ty).is_some() || tribute::Nat::from_type(db, ty).is_some() {
        debug!("  -> boxing Int/Nat to i31ref");
        // Int/Nat (i64) → i31ref
        // Truncate i64 to i32, then convert to i31ref
        // Note: This only works correctly for values that fit in 31 bits!
        // Phase 1 limitation: values outside -2^30..2^30-1 will be incorrect
        function.instruction(&Instruction::I32WrapI64);
        function.instruction(&Instruction::RefI31);
        Ok(())
    } else if core::F64::from_type(db, ty).is_some() {
        // Float (f64) → BoxedF64 struct
        // Create a struct with the f64 value
        function.instruction(&Instruction::StructNew(BOXED_F64_IDX));
        Ok(())
    } else {
        // For reference types (structs, etc.), no boxing needed - they're already subtypes of anyref
        // Just leave the value as-is on the stack
        Ok(())
    }
}

/// Emit unboxing instructions to convert anyref to a concrete type.
/// - i31ref → Int (i64): extract i32 and extend to i64
/// - BoxedF64 → Float (f64): cast and extract f64 field
fn emit_unboxing<'db>(
    db: &'db dyn salsa::Database,
    ty: Type<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    if tribute::Int::from_type(db, ty).is_some() {
        // anyref → i31ref → Int (i64)
        // Cast anyref to i31ref, extract i32, then sign-extend to i64
        function.instruction(&Instruction::RefCastNullable(HeapType::I31));
        function.instruction(&Instruction::I31GetS);
        function.instruction(&Instruction::I64ExtendI32S);
        Ok(())
    } else if tribute::Nat::from_type(db, ty).is_some() {
        // anyref → i31ref → Nat (i64)
        // Cast anyref to i31ref, extract u32, then zero-extend to i64
        function.instruction(&Instruction::RefCastNullable(HeapType::I31));
        function.instruction(&Instruction::I31GetU);
        function.instruction(&Instruction::I64ExtendI32U);
        Ok(())
    } else if core::F64::from_type(db, ty).is_some() {
        // anyref → BoxedF64 → Float (f64)
        // Cast to BoxedF64 struct, then extract f64 field
        function.instruction(&Instruction::RefCastNullable(HeapType::Concrete(
            BOXED_F64_IDX,
        )));
        function.instruction(&Instruction::StructGet {
            struct_type_index: BOXED_F64_IDX,
            field_index: 0,
        });
        Ok(())
    } else {
        // For reference types, assume no unboxing needed
        Ok(())
    }
}

fn value_type<'db>(db: &'db dyn salsa::Database, value: Value<'db>) -> Option<Type<'db>> {
    match value.def(db) {
        ValueDef::OpResult(op) => op.results(db).get(value.index(db)).copied(),
        // BlockArg type lookup requires block context; callers handle None
        ValueDef::BlockArg(_block_id) => None,
    }
}

/// Infer the actual result type for a call operation.
/// For generic function calls where the IR result type is `type.var`,
/// we infer the concrete type from the operand types.
fn infer_call_result_type<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
    result_ty: Type<'db>,
    func_types: &HashMap<Symbol, core::Func<'db>>,
) -> Type<'db> {
    // Only handle wasm.call operations
    if op.dialect(db) != Symbol::new("wasm") || op.name(db) != Symbol::new("call") {
        return result_ty;
    }

    // Get the callee
    let callee = match attr_symbol_ref(db, op, ATTR_CALLEE()) {
        Ok(c) => c,
        Err(_) => return result_ty,
    };

    // Look up the callee's function type
    let callee_ty = match func_types.get(&callee) {
        Some(ty) => ty,
        None => return result_ty,
    };

    // Check if the callee returns type.var (generic)
    let return_ty = callee_ty.result(db);
    if !tribute::is_type_var(db, return_ty) {
        // Callee returns a concrete type, use it
        return return_ty;
    }

    // Infer concrete type from first operand (works for identity-like functions)
    if let Some(operand_ty) = op
        .operands(db)
        .first()
        .and_then(|v| value_type(db, *v))
        .filter(|ty| !tribute::is_type_var(db, *ty))
    {
        return operand_ty;
    }

    result_ty
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
    // Skip nil types - they don't have local mappings and don't need local.set
    if is_nil_type(db, results[0]) {
        return Ok(());
    }
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

fn type_to_valtype<'db>(
    db: &'db dyn salsa::Database,
    ty: Type<'db>,
    type_idx_by_type: &HashMap<Type<'db>, u32>,
) -> CompilationResult<ValType> {
    if core::I32::from_type(db, ty).is_some() || core::I1::from_type(db, ty).is_some() {
        // core.i1 (Bool) is represented as i32 in WebAssembly
        Ok(ValType::I32)
    } else if core::I64::from_type(db, ty).is_some()
        || tribute::Int::from_type(db, ty).is_some()
        || tribute::Nat::from_type(db, ty).is_some()
    {
        // Int/Nat (arbitrary precision) is lowered to i64 for Phase 1
        // TODO: Implement i31ref/BigInt hybrid for WasmGC
        Ok(ValType::I64)
    } else if core::F32::from_type(db, ty).is_some() {
        Ok(ValType::F32)
    } else if core::F64::from_type(db, ty).is_some() {
        Ok(ValType::F64)
    } else if core::Bytes::from_type(db, ty).is_some() {
        // Bytes uses WasmGC struct representation
        Ok(ValType::Ref(RefType {
            nullable: true,
            heap_type: HeapType::Concrete(BYTES_STRUCT_IDX),
        }))
    } else if core::String::from_type(db, ty).is_some()
        || (ty.dialect(db) == core::DIALECT_NAME() && ty.name(db) == Symbol::new("ptr"))
    {
        // String and ptr still use linear memory (i32 pointer)
        Ok(ValType::I32)
    } else if ty.dialect(db) == wasm::DIALECT_NAME() {
        // WASM dialect types (e.g., wasm.structref for continuation frames)
        // IMPORTANT: Must check BEFORE type_idx_by_type.get() to avoid returning
        // incorrect concrete type_idx for placeholder types like wasm.structref
        let name = ty.name(db);
        if name == Symbol::new("structref") {
            Ok(ValType::Ref(RefType {
                nullable: true,
                heap_type: HeapType::Abstract {
                    shared: false,
                    ty: AbstractHeapType::Struct,
                },
            }))
        } else if name == Symbol::new("funcref") {
            Ok(ValType::Ref(RefType::FUNCREF))
        } else if name == Symbol::new("anyref") {
            Ok(ValType::Ref(RefType::ANYREF))
        } else if name == Symbol::new("i31ref") {
            Ok(ValType::Ref(RefType {
                nullable: false,
                heap_type: HeapType::Abstract {
                    shared: false,
                    ty: AbstractHeapType::I31,
                },
            }))
        } else {
            Err(CompilationError::type_error(format!(
                "unsupported wasm type: wasm.{}",
                name
            )))
        }
    } else if let Some(&type_idx) = type_idx_by_type.get(&ty) {
        // ADT types (structs, variants) - use concrete GC type reference
        // Check this BEFORE tribute::is_type_var to handle struct types with type_idx
        Ok(ValType::Ref(RefType {
            nullable: true,
            heap_type: HeapType::Concrete(type_idx),
        }))
    } else if tribute::is_type_var(db, ty) {
        // Generic type variables use anyref (uniform representation)
        // Values must be boxed when passed to generic functions
        Ok(ValType::Ref(RefType::ANYREF))
    } else if ty.dialect(db) == adt::DIALECT_NAME() {
        // ADT base types (e.g., adt.Expr) without specific variant type_idx
        // These represent "any variant of this enum" and use anyref
        Ok(ValType::Ref(RefType::ANYREF))
    } else if core::Func::from_type(db, ty).is_some() {
        // Function types as values use anyref (closures are passed as structs)
        // In Tribute, function-typed parameters receive closures at runtime
        Ok(ValType::Ref(RefType::ANYREF))
    } else if closure::Closure::from_type(db, ty).is_some() {
        // Closure types - use anyref for unregistered closures
        // (registered closures are handled earlier by the generic type_idx_by_type lookup)
        Ok(ValType::Ref(RefType::ANYREF))
    } else if ability::EvidencePtr::from_type(db, ty).is_some() {
        // Evidence pointer for ability system - use anyref as runtime handle
        Ok(ValType::Ref(RefType::ANYREF))
    } else {
        Err(CompilationError::type_error(format!(
            "unsupported wasm value type: {}.{}",
            ty.dialect(db),
            ty.name(db)
        )))
    }
}

fn result_types<'db>(
    db: &'db dyn salsa::Database,
    ty: Type<'db>,
    type_idx_by_type: &HashMap<Type<'db>, u32>,
) -> CompilationResult<Vec<ValType>> {
    if is_nil_type(db, ty) {
        Ok(Vec::new())
    } else {
        Ok(vec![type_to_valtype(db, ty, type_idx_by_type)?])
    }
}

fn is_nil_type<'db>(db: &'db dyn salsa::Database, ty: Type<'db>) -> bool {
    ty.dialect(db) == core::DIALECT_NAME() && ty.name(db) == Symbol::new("nil")
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

fn attr_i32<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
    key: Symbol,
) -> CompilationResult<i32> {
    match op.attributes(db).get(&key) {
        Some(Attribute::IntBits(bits)) => {
            let value = *bits as u32;
            Ok(i32::from_ne_bytes(value.to_ne_bytes()))
        }
        _ => Err(CompilationError::from(
            errors::CompilationErrorKind::InvalidAttribute("value"),
        )),
    }
}

fn attr_i64<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
    key: Symbol,
) -> CompilationResult<i64> {
    match op.attributes(db).get(&key) {
        Some(Attribute::IntBits(bits)) => Ok(i64::from_ne_bytes(bits.to_ne_bytes())),
        _ => Err(CompilationError::from(
            errors::CompilationErrorKind::InvalidAttribute("value"),
        )),
    }
}

fn attr_f32<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
    key: Symbol,
) -> CompilationResult<f32> {
    match op.attributes(db).get(&key) {
        Some(Attribute::FloatBits(bits)) => Ok(f32::from_bits(*bits as u32)),
        _ => Err(CompilationError::from(
            errors::CompilationErrorKind::InvalidAttribute("value"),
        )),
    }
}

fn attr_f64<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
    key: Symbol,
) -> CompilationResult<f64> {
    match op.attributes(db).get(&key) {
        Some(Attribute::FloatBits(bits)) => Ok(f64::from_bits(*bits)),
        _ => Err(CompilationError::from(
            errors::CompilationErrorKind::InvalidAttribute("value"),
        )),
    }
}

fn attr_index<'db>(db: &'db dyn salsa::Database, op: &Operation<'db>) -> CompilationResult<u32> {
    match op.attributes(db).get(&ATTR_INDEX()) {
        Some(Attribute::IntBits(bits)) => Ok(*bits as u32),
        _ => Err(CompilationError::from(
            errors::CompilationErrorKind::InvalidAttribute("index"),
        )),
    }
}

fn attr_symbol_ref<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
    key: Symbol,
) -> CompilationResult<Symbol> {
    match op.attributes(db).get(&key) {
        Some(Attribute::Symbol(sym)) => Ok(*sym),
        _ => Err(CompilationError::from(
            errors::CompilationErrorKind::InvalidAttribute("callee"),
        )),
    }
}

fn attr_symbol_ref_attr<'db>(attrs: &'db Attrs<'db>, key: Symbol) -> CompilationResult<Symbol> {
    match attrs.get(&key) {
        Some(Attribute::Symbol(sym)) => Ok(*sym),
        _ => Err(CompilationError::from(
            errors::CompilationErrorKind::MissingAttribute("symbol_ref"),
        )),
    }
}

fn attr_string<'db>(attrs: &Attrs<'db>, key: Symbol) -> CompilationResult<String> {
    match attrs.get(&key) {
        Some(Attribute::String(value)) => Ok(value.clone()),
        _ => Err(CompilationError::from(
            errors::CompilationErrorKind::MissingAttribute("string"),
        )),
    }
}

fn attr_symbol<'db>(attrs: &Attrs<'db>, key: Symbol) -> CompilationResult<Symbol> {
    match attrs.get(&key) {
        Some(Attribute::Symbol(value)) => Ok(*value),
        _ => Err(CompilationError::from(
            errors::CompilationErrorKind::MissingAttribute("symbol"),
        )),
    }
}

fn attr_type<'db>(attrs: &Attrs<'db>, key: Symbol) -> CompilationResult<Type<'db>> {
    match attrs.get(&key) {
        Some(Attribute::Type(value)) => Ok(*value),
        _ => Err(CompilationError::from(
            errors::CompilationErrorKind::MissingAttribute("type"),
        )),
    }
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

fn attr_heap_type<'db>(attrs: &Attrs<'db>, key: Symbol) -> CompilationResult<HeapType> {
    match attrs.get(&key) {
        Some(Attribute::IntBits(bits)) => Ok(HeapType::Concrete(*bits as u32)),
        Some(Attribute::Symbol(sym)) => {
            // Handle abstract heap types specified by name
            sym.with_str(|name| match name {
                "any" => Ok(HeapType::Abstract {
                    shared: false,
                    ty: AbstractHeapType::Any,
                }),
                "func" => Ok(HeapType::Abstract {
                    shared: false,
                    ty: AbstractHeapType::Func,
                }),
                "extern" => Ok(HeapType::Abstract {
                    shared: false,
                    ty: AbstractHeapType::Extern,
                }),
                "none" => Ok(HeapType::Abstract {
                    shared: false,
                    ty: AbstractHeapType::None,
                }),
                "struct" => Ok(HeapType::Abstract {
                    shared: false,
                    ty: AbstractHeapType::Struct,
                }),
                "array" => Ok(HeapType::Abstract {
                    shared: false,
                    ty: AbstractHeapType::Array,
                }),
                "i31" => Ok(HeapType::Abstract {
                    shared: false,
                    ty: AbstractHeapType::I31,
                }),
                _ => Err(CompilationError::from(
                    errors::CompilationErrorKind::MissingAttribute("unknown abstract heap type"),
                )),
            })
        }
        _ => Err(CompilationError::from(
            errors::CompilationErrorKind::MissingAttribute("heap_type"),
        )),
    }
}

fn attr_i32_attr<'db>(attrs: &Attrs<'db>, key: Symbol) -> CompilationResult<i32> {
    match attrs.get(&key) {
        Some(Attribute::IntBits(bits)) => {
            let value = *bits as u32;
            Ok(i32::from_ne_bytes(value.to_ne_bytes()))
        }
        _ => Err(CompilationError::from(
            errors::CompilationErrorKind::MissingAttribute("i32"),
        )),
    }
}

/// Extract MemArg from operation attributes for linear memory load/store operations.
/// Defaults: offset=0, align=natural alignment (log2), memory=0
fn extract_memarg<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
    natural_align: u32, // log2 of natural alignment (0=1, 1=2, 2=4, 3=8)
) -> MemArg {
    let attrs = op.attributes(db);

    let offset = match attrs.get(&ATTR_OFFSET()) {
        Some(Attribute::IntBits(v)) => *v,
        _ => 0,
    };

    let align = match attrs.get(&ATTR_ALIGN()) {
        Some(Attribute::IntBits(v)) => *v as u32,
        _ => natural_align, // Use natural alignment if not specified
    };

    let memory_index = match attrs.get(&ATTR_MEMORY()) {
        Some(Attribute::IntBits(v)) => *v as u32,
        _ => 0, // Default to memory 0
    };

    MemArg {
        offset,
        align,
        memory_index,
    }
}

/// Extract memory index from operation attributes for memory management operations.
/// Defaults to memory index 0.
fn extract_memory_index<'db>(db: &'db dyn salsa::Database, op: &Operation<'db>) -> u32 {
    match op.attributes(db).get(&ATTR_MEMORY()) {
        Some(Attribute::IntBits(v)) => *v as u32,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use trunk_ir::{Block, BlockId, Location, PathId, Region, Span, idvec};

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
        let field0 = Operation::of_name(db, location, "wasm.i32_const")
            .attr("value", Attribute::IntBits(42))
            .results(idvec![i32_ty])
            .build();

        let field1 = Operation::of_name(db, location, "wasm.i64_const")
            .attr("value", Attribute::IntBits(100))
            .results(idvec![i64_ty])
            .build();

        // Create struct_new with two fields
        let struct_ty = core::I32::new(db).as_type(); // placeholder type
        let struct_new = Operation::of_name(db, location, "wasm.struct_new")
            .operands(idvec![field0.result(db, 0), field1.result(db, 0)])
            .results(idvec![struct_ty])
            .attr("type_idx", Attribute::IntBits(FIRST_USER_TYPE_IDX as u64))
            .build();

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
            collect_gc_types(db, module).expect("collect_gc_types failed");

        // Should have 4 GC types: 3 built-in (BoxedF64, BytesArray, BytesStruct) + 1 user struct
        assert_eq!(gc_types.len(), 4);
        // Index 0 is BoxedF64
        assert_eq!(gc_type_kind(&gc_types[0]), "struct");
        // Index 1 is BytesArray
        assert_eq!(gc_type_kind(&gc_types[1]), "array");
        // Index 2 is BytesStruct
        assert_eq!(gc_type_kind(&gc_types[2]), "struct");
        // Index 3 is the user struct
        assert_eq!(gc_type_kind(&gc_types[3]), "struct");

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
        let size = Operation::of_name(db, location, "wasm.i32_const")
            .attr("value", Attribute::IntBits(10))
            .results(idvec![i32_ty])
            .build();

        // Create init value (f64)
        let init = Operation::of_name(db, location, "wasm.f64_const")
            .attr("value", Attribute::FloatBits(0.0_f64.to_bits()))
            .results(idvec![f64_ty])
            .build();

        // Create array_new
        let array_new = Operation::of_name(db, location, "wasm.array_new")
            .operands(idvec![size.result(db, 0), init.result(db, 0)])
            .results(idvec![i32_ty]) // placeholder result type
            .attr("type_idx", Attribute::IntBits(FIRST_USER_TYPE_IDX as u64))
            .build();

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
            collect_gc_types(db, module).expect("collect_gc_types failed");

        // Should have 4 GC types: 3 built-in (BoxedF64, BytesArray, BytesStruct) + 1 user array
        assert_eq!(gc_types.len(), 4);
        // Index 0 is BoxedF64 (struct)
        assert_eq!(gc_type_kind(&gc_types[0]), "struct");
        // Index 1 is BytesArray (array)
        assert_eq!(gc_type_kind(&gc_types[1]), "array");
        // Index 2 is BytesStruct (struct)
        assert_eq!(gc_type_kind(&gc_types[2]), "struct");
        // Index 3 is the user array
        assert_eq!(gc_type_kind(&gc_types[3]), "array");
    }

    // ========================================
    // Test: type index deduplication
    // ========================================

    #[salsa::tracked]
    fn make_dedup_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();

        // Create two struct_new operations with same type_idx
        let field = Operation::of_name(db, location, "wasm.i32_const")
            .attr("value", Attribute::IntBits(1))
            .results(idvec![i32_ty])
            .build();

        let struct_new1 = Operation::of_name(db, location, "wasm.struct_new")
            .operands(idvec![field.result(db, 0)])
            .results(idvec![i32_ty])
            .attr("type_idx", Attribute::IntBits(FIRST_USER_TYPE_IDX as u64))
            .build();

        let field2 = Operation::of_name(db, location, "wasm.i32_const")
            .attr("value", Attribute::IntBits(2))
            .results(idvec![i32_ty])
            .build();

        let struct_new2 = Operation::of_name(db, location, "wasm.struct_new")
            .operands(idvec![field2.result(db, 0)])
            .results(idvec![i32_ty])
            .attr("type_idx", Attribute::IntBits(FIRST_USER_TYPE_IDX as u64)) // same type_idx
            .build();

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
            collect_gc_types(db, module).expect("collect_gc_types failed");

        // Should have 4 GC types: 3 built-in + 1 user struct (same type_idx used twice)
        assert_eq!(gc_types.len(), 4);
        // Index 0 is BoxedF64
        assert_eq!(gc_type_kind(&gc_types[0]), "struct");
        // Index 1 is BytesArray
        assert_eq!(gc_type_kind(&gc_types[1]), "array");
        // Index 2 is BytesStruct
        assert_eq!(gc_type_kind(&gc_types[2]), "struct");
        // Index 3 is the deduplicated user struct
        assert_eq!(gc_type_kind(&gc_types[3]), "struct");
    }

    // ========================================
    // Test: field count mismatch error
    // ========================================

    #[salsa::tracked]
    fn make_field_count_mismatch_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();

        // Create struct_new with 1 field
        let field = Operation::of_name(db, location, "wasm.i32_const")
            .attr("value", Attribute::IntBits(1))
            .results(idvec![i32_ty])
            .build();

        let struct_new1 = Operation::of_name(db, location, "wasm.struct_new")
            .operands(idvec![field.result(db, 0)])
            .results(idvec![i32_ty])
            .attr("type_idx", Attribute::IntBits(FIRST_USER_TYPE_IDX as u64))
            .build();

        // Create another struct_new with 2 fields (same type_idx)
        let field2a = Operation::of_name(db, location, "wasm.i32_const")
            .attr("value", Attribute::IntBits(2))
            .results(idvec![i32_ty])
            .build();

        let field2b = Operation::of_name(db, location, "wasm.i32_const")
            .attr("value", Attribute::IntBits(3))
            .results(idvec![i32_ty])
            .build();

        let struct_new2 = Operation::of_name(db, location, "wasm.struct_new")
            .operands(idvec![field2a.result(db, 0), field2b.result(db, 0)])
            .results(idvec![i32_ty])
            .attr("type_idx", Attribute::IntBits(FIRST_USER_TYPE_IDX as u64)) // same type_idx, different field count
            .build();

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
        let result = collect_gc_types(db, module);

        // Should return an error due to field count mismatch
        assert!(result.is_err());
        let err = result.err().expect("expected error");
        assert!(err.to_string().contains("field count mismatch"));
    }

    // ========================================
    // Test: placeholder struct types (wasm.structref)
    // ========================================

    /// Create a module with multiple struct_new operations using wasm.structref
    /// placeholder type but with different field counts.
    #[salsa::tracked]
    fn make_placeholder_struct_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();
        let structref_ty = wasm::Structref::new(db).as_type();

        // Create struct_new with 1 field using wasm.structref type attribute
        let field1 = Operation::of_name(db, location, "wasm.i32_const")
            .attr("value", Attribute::IntBits(1))
            .results(idvec![i32_ty])
            .build();

        let struct_new1 = Operation::of_name(db, location, "wasm.struct_new")
            .operands(idvec![field1.result(db, 0)])
            .results(idvec![structref_ty])
            .attr("type", Attribute::Type(structref_ty))
            .build();

        // Create struct_new with 2 fields using same wasm.structref type
        let field2a = Operation::of_name(db, location, "wasm.i32_const")
            .attr("value", Attribute::IntBits(2))
            .results(idvec![i32_ty])
            .build();

        let field2b = Operation::of_name(db, location, "wasm.i32_const")
            .attr("value", Attribute::IntBits(3))
            .results(idvec![i32_ty])
            .build();

        let struct_new2 = Operation::of_name(db, location, "wasm.struct_new")
            .operands(idvec![field2a.result(db, 0), field2b.result(db, 0)])
            .results(idvec![structref_ty])
            .attr("type", Attribute::Type(structref_ty))
            .build();

        // Create struct_new with 0 fields (empty struct)
        let struct_new3 = Operation::of_name(db, location, "wasm.struct_new")
            .operands(idvec![])
            .results(idvec![structref_ty])
            .attr("type", Attribute::Type(structref_ty))
            .build();

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![
                field1,
                struct_new1,
                field2a,
                field2b,
                struct_new2,
                struct_new3
            ],
        );
        let region = Region::new(db, location, idvec![block]);
        core::Module::create(db, location, "test".into(), region)
    }

    #[salsa_test]
    fn test_placeholder_struct_types(db: &salsa::DatabaseImpl) {
        let module = make_placeholder_struct_module(db);
        let (gc_types, _type_map, placeholder_map) =
            collect_gc_types(db, module).expect("collect_gc_types failed");

        // Should have 6 GC types: 3 built-in + 3 user structs (one per field count)
        assert_eq!(gc_types.len(), 6);

        // Verify placeholder_map has entries for each (type, field_count) pair
        let structref_ty = wasm::Structref::new(db).as_type();
        assert!(
            placeholder_map.contains_key(&(structref_ty, 0)),
            "placeholder_map should have entry for (structref, 0)"
        );
        assert!(
            placeholder_map.contains_key(&(structref_ty, 1)),
            "placeholder_map should have entry for (structref, 1)"
        );
        assert!(
            placeholder_map.contains_key(&(structref_ty, 2)),
            "placeholder_map should have entry for (structref, 2)"
        );

        // Verify each entry has a distinct type_idx
        let idx_0 = placeholder_map[&(structref_ty, 0)];
        let idx_1 = placeholder_map[&(structref_ty, 1)];
        let idx_2 = placeholder_map[&(structref_ty, 2)];
        assert_ne!(idx_0, idx_1, "type_idx for 0 and 1 fields should differ");
        assert_ne!(idx_1, idx_2, "type_idx for 1 and 2 fields should differ");
        assert_ne!(idx_0, idx_2, "type_idx for 0 and 2 fields should differ");

        // Verify GC type definitions have correct field counts
        // User types start at index FIRST_USER_TYPE_IDX (3)
        let field_counts: Vec<usize> = gc_types[FIRST_USER_TYPE_IDX as usize..]
            .iter()
            .map(|gc_type| match gc_type {
                GcTypeDef::Struct(fields) => fields.len(),
                _ => panic!("expected struct type"),
            })
            .collect();

        // Should have structs with 0, 1, and 2 fields (order may vary)
        assert!(field_counts.contains(&0), "should have 0-field struct");
        assert!(field_counts.contains(&1), "should have 1-field struct");
        assert!(field_counts.contains(&2), "should have 2-field struct");
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
        let field = Operation::of_name(db, location, "wasm.i32_const")
            .attr("value", Attribute::IntBits(42))
            .results(idvec![i32_ty])
            .build();

        let struct_new = Operation::of_name(db, location, "wasm.struct_new")
            .operands(idvec![field.result(db, 0)])
            .results(idvec![i32_ty])
            .attr("type_idx", Attribute::IntBits(FIRST_USER_TYPE_IDX as u64))
            .build();

        let func_return = Operation::of_name(db, location, "wasm.return").build();

        let body_block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![field, struct_new, func_return],
        );
        let body_region = Region::new(db, location, idvec![body_block]);

        // Create wasm.func
        let wasm_func = Operation::of_name(db, location, "wasm.func")
            .attr("sym_name", Attribute::Symbol(Symbol::new("test_fn")))
            .attr("type", Attribute::Type(func_ty))
            .region(body_region)
            .build();

        let block = Block::new(db, BlockId::fresh(), location, idvec![], idvec![wasm_func]);
        let region = Region::new(db, location, idvec![block]);
        core::Module::create(db, location, "test".into(), region)
    }

    #[salsa_test]
    fn test_nested_operations_in_function_body(db: &salsa::DatabaseImpl) {
        let module = make_func_with_struct_module(db);
        let (gc_types, _type_map, _) =
            collect_gc_types(db, module).expect("collect_gc_types failed");

        // Should find the struct type from inside the function body
        // (4 types: 3 built-in + 1 user struct)
        assert_eq!(gc_types.len(), 4);
        // Index 0 is BoxedF64
        assert_eq!(gc_type_kind(&gc_types[0]), "struct");
        // Index 1 is BytesArray
        assert_eq!(gc_type_kind(&gc_types[1]), "array");
        // Index 2 is BytesStruct
        assert_eq!(gc_type_kind(&gc_types[2]), "struct");
        // Index 3 is the user struct from inside the function body
        assert_eq!(gc_type_kind(&gc_types[3]), "struct");
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

        // Create a ref.null for the bytes array type
        let bytes_array_ref = Operation::of_name(db, location, "wasm.ref_null")
            .attr("heap_type", Attribute::IntBits(BYTES_ARRAY_IDX as u64))
            .results(idvec![i32_ty]) // placeholder type
            .build();

        // Create index value
        let index = Operation::of_name(db, location, "wasm.i32_const")
            .attr("value", Attribute::IntBits(0))
            .results(idvec![i32_ty])
            .build();

        // Create array_get_u with BYTES_ARRAY_IDX (builtin type)
        let array_get = Operation::of_name(db, location, "wasm.array_get_u")
            .operands(idvec![bytes_array_ref.result(db, 0), index.result(db, 0)])
            .results(idvec![i32_ty])
            .attr("type_idx", Attribute::IntBits(BYTES_ARRAY_IDX as u64))
            .build();

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![bytes_array_ref, index, array_get],
        );
        let region = Region::new(db, location, idvec![block]);
        core::Module::create(db, location, "test".into(), region)
    }

    #[salsa_test]
    fn test_array_get_with_builtin_type_idx_does_not_panic(db: &salsa::DatabaseImpl) {
        let module = make_array_get_builtin_module(db);
        let result = collect_gc_types(db, module);

        // Should complete without panic
        assert!(
            result.is_ok(),
            "collect_gc_types should not panic for builtin type indices"
        );

        let (gc_types, _type_map, _) = result.expect("collect_gc_types failed");

        // Should only have the 3 built-in types (no additional user types allocated)
        assert_eq!(gc_types.len(), 3);
        assert_eq!(gc_type_kind(&gc_types[0]), "struct"); // BoxedF64
        assert_eq!(gc_type_kind(&gc_types[1]), "array"); // BytesArray
        assert_eq!(gc_type_kind(&gc_types[2]), "struct"); // BytesStruct
    }

    /// Test that struct_get with BYTES_STRUCT_IDX (2) doesn't panic.
    #[salsa::tracked]
    fn make_struct_get_builtin_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();

        // Create a ref.null for the bytes struct type
        let bytes_struct_ref = Operation::of_name(db, location, "wasm.ref_null")
            .attr("heap_type", Attribute::IntBits(BYTES_STRUCT_IDX as u64))
            .results(idvec![i32_ty]) // placeholder type
            .build();

        // Create struct_get with BYTES_STRUCT_IDX (builtin type)
        let struct_get = Operation::of_name(db, location, "wasm.struct_get")
            .operands(idvec![bytes_struct_ref.result(db, 0)])
            .results(idvec![i32_ty])
            .attr("type_idx", Attribute::IntBits(BYTES_STRUCT_IDX as u64))
            .attr("field_idx", Attribute::IntBits(0))
            .build();

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![bytes_struct_ref, struct_get],
        );
        let region = Region::new(db, location, idvec![block]);
        core::Module::create(db, location, "test".into(), region)
    }

    #[salsa_test]
    fn test_struct_get_with_builtin_type_idx_does_not_panic(db: &salsa::DatabaseImpl) {
        let module = make_struct_get_builtin_module(db);
        let result = collect_gc_types(db, module);

        // Should complete without panic
        assert!(
            result.is_ok(),
            "collect_gc_types should not panic for builtin type indices"
        );

        let (gc_types, _type_map, _) = result.expect("collect_gc_types failed");

        // Should only have the 3 built-in types (no additional user types allocated)
        assert_eq!(gc_types.len(), 3);
        assert_eq!(gc_type_kind(&gc_types[0]), "struct"); // BoxedF64
        assert_eq!(gc_type_kind(&gc_types[1]), "array"); // BytesArray
        assert_eq!(gc_type_kind(&gc_types[2]), "struct"); // BytesStruct
    }

    /// Test that array_set with BYTES_ARRAY_IDX (1) doesn't panic.
    #[salsa::tracked]
    fn make_array_set_builtin_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();

        // Create a ref.null for the bytes array type
        let bytes_array_ref = Operation::of_name(db, location, "wasm.ref_null")
            .attr("heap_type", Attribute::IntBits(BYTES_ARRAY_IDX as u64))
            .results(idvec![i32_ty])
            .build();

        // Create index value
        let index = Operation::of_name(db, location, "wasm.i32_const")
            .attr("value", Attribute::IntBits(0))
            .results(idvec![i32_ty])
            .build();

        // Create value to set
        let value = Operation::of_name(db, location, "wasm.i32_const")
            .attr("value", Attribute::IntBits(42))
            .results(idvec![i32_ty])
            .build();

        // Create array_set with BYTES_ARRAY_IDX (builtin type)
        let array_set = Operation::of_name(db, location, "wasm.array_set")
            .operands(idvec![
                bytes_array_ref.result(db, 0),
                index.result(db, 0),
                value.result(db, 0)
            ])
            .attr("type_idx", Attribute::IntBits(BYTES_ARRAY_IDX as u64))
            .build();

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![bytes_array_ref, index, value, array_set],
        );
        let region = Region::new(db, location, idvec![block]);
        core::Module::create(db, location, "test".into(), region)
    }

    #[salsa_test]
    fn test_array_set_with_builtin_type_idx_does_not_panic(db: &salsa::DatabaseImpl) {
        let module = make_array_set_builtin_module(db);
        let result = collect_gc_types(db, module);

        // Should complete without panic
        assert!(
            result.is_ok(),
            "collect_gc_types should not panic for builtin type indices"
        );

        let (gc_types, _type_map, _) = result.expect("collect_gc_types failed");

        // Should only have the 3 built-in types
        assert_eq!(gc_types.len(), 3);
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
        let addr = Operation::of_name(db, location, "wasm.i32_const")
            .attr("value", Attribute::IntBits(0))
            .results(idvec![i32_ty])
            .build();

        // Create value to store (i32)
        let value = Operation::of_name(db, location, "wasm.i32_const")
            .attr("value", Attribute::IntBits(42))
            .results(idvec![i32_ty])
            .build();

        // i32_store with offset and align attributes
        let store_op = Operation::of_name(db, location, "wasm.i32_store")
            .operands(idvec![addr.result(db, 0), value.result(db, 0)])
            .attr("offset", Attribute::IntBits(4))
            .attr("align", Attribute::IntBits(2))
            .attr("memory", Attribute::IntBits(0))
            .build();

        // i32_load from same address
        let load_op = Operation::of_name(db, location, "wasm.i32_load")
            .operands(idvec![addr.result(db, 0)])
            .results(idvec![i32_ty])
            .attr("offset", Attribute::IntBits(4))
            .attr("align", Attribute::IntBits(2))
            .attr("memory", Attribute::IntBits(0))
            .build();

        // Return statement
        let func_return = Operation::of_name(db, location, "wasm.return")
            .operands(idvec![])
            .build();

        let body_block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![addr, value, store_op, load_op, func_return],
        );
        let body_region = Region::new(db, location, idvec![body_block]);

        // Function definition
        let wasm_func = Operation::of_name(db, location, "wasm.func")
            .attr("sym_name", Attribute::Symbol(Symbol::new("test")))
            .attr("type", Attribute::Type(func_ty))
            .regions(idvec![body_region])
            .build();

        // Memory definition (required for load/store)
        let memory_op = Operation::of_name(db, location, "wasm.memory")
            .attr("min", Attribute::IntBits(1))
            .attr("max", Attribute::IntBits(1))
            .attr("shared", Attribute::Bool(false))
            .attr("memory64", Attribute::Bool(false))
            .build();

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
        let size_op = Operation::of_name(db, location, "wasm.memory_size")
            .results(idvec![i32_ty])
            .attr("memory", Attribute::IntBits(0))
            .build();

        // delta for memory_grow
        let delta = Operation::of_name(db, location, "wasm.i32_const")
            .attr("value", Attribute::IntBits(1))
            .results(idvec![i32_ty])
            .build();

        // memory_grow
        let grow_op = Operation::of_name(db, location, "wasm.memory_grow")
            .operands(idvec![delta.result(db, 0)])
            .results(idvec![i32_ty])
            .attr("memory", Attribute::IntBits(0))
            .build();

        // Return statement
        let func_return = Operation::of_name(db, location, "wasm.return")
            .operands(idvec![])
            .build();

        let body_block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![size_op, delta, grow_op, func_return],
        );
        let body_region = Region::new(db, location, idvec![body_block]);

        let wasm_func = Operation::of_name(db, location, "wasm.func")
            .attr("sym_name", Attribute::Symbol(Symbol::new("test")))
            .attr("type", Attribute::Type(func_ty))
            .regions(idvec![body_region])
            .build();

        let memory_op = Operation::of_name(db, location, "wasm.memory")
            .attr("min", Attribute::IntBits(1))
            .attr("max", Attribute::IntBits(2))
            .attr("shared", Attribute::Bool(false))
            .attr("memory64", Attribute::Bool(false))
            .build();

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
}
