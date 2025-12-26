//! WebAssembly binary emission from wasm dialect operations.
//!
//! This module converts lowered wasm dialect TrunkIR operations to
//! a WebAssembly binary using the `wasm_encoder` crate.

use std::collections::HashMap;
use std::sync::LazyLock;

use tracing::debug;

use trunk_ir::dialect::{core, ty};
use trunk_ir::{
    Attribute, Attrs, DialectType, IdVec, Operation, QualifiedName, Region, Symbol, Type, Value,
    ValueDef,
};
use wasm_encoder::{
    BlockType, CodeSection, ConstExpr, DataSection, EntityType, ExportKind, ExportSection,
    FieldType, Function, FunctionSection, HeapType, ImportSection, Instruction, MemorySection,
    MemoryType, Module, RefType, StorageType, TypeSection, ValType,
};

use crate::errors;
use crate::{CompilationError, CompilationResult};

trunk_ir::symbols! {
    ATTR_SYM_NAME => "sym_name",
    ATTR_TYPE => "type",
    ATTR_TYPE_IDX => "type_idx",
    ATTR_FIELD_IDX => "field_idx",
    ATTR_HEAP_TYPE => "heap_type",
    ATTR_TARGET_TYPE => "target_type",
    ATTR_CALLEE => "callee",
    ATTR_VALUE => "value",
    ATTR_INDEX => "index",
    ATTR_TARGET => "target",
    ATTR_MODULE => "module",
    ATTR_NAME => "name",
    ATTR_FUNC => "func",
    ATTR_MIN => "min",
    ATTR_MAX => "max",
    ATTR_SHARED => "shared",
    ATTR_MEMORY64 => "memory64",
    ATTR_OFFSET => "offset",
    ATTR_BYTES => "bytes",
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
    name: QualifiedName,
    ty: core::Func<'db>,
    op: Operation<'db>,
}

struct ImportFuncDef<'db> {
    sym: QualifiedName,
    module: String,
    name: String,
    ty: core::Func<'db>,
}

struct ExportDef {
    name: String,
    kind: ExportKind,
    target: ExportTarget,
}

#[derive(Debug)]
enum ExportTarget {
    Func(QualifiedName),
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
}

struct ModuleInfo<'db> {
    imports: Vec<ImportFuncDef<'db>>,
    funcs: Vec<FunctionDef<'db>>,
    exports: Vec<ExportDef>,
    memory: Option<MemoryDef>,
    data: Vec<DataDef>,
    gc_types: Vec<GcTypeDef>,
    type_idx_by_type: HashMap<Type<'db>, u32>,
    /// Function type lookup map for boxing/unboxing at call sites.
    func_types: HashMap<QualifiedName, core::Func<'db>>,
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
    let mut memory_section = MemorySection::new();
    let mut export_section = ExportSection::new();
    let mut code_section = CodeSection::new();
    let mut data_section = DataSection::new();

    let mut func_indices: HashMap<QualifiedName, u32> = HashMap::new();
    let gc_type_count = module_info.gc_types.len() as u32;
    let mut next_type_index = gc_type_count;

    for (index, import_def) in module_info.imports.iter().enumerate() {
        func_indices.insert(import_def.sym.clone(), index as u32);
    }
    let import_count = module_info.imports.len() as u32;
    for (index, func_def) in module_info.funcs.iter().enumerate() {
        func_indices.insert(func_def.name.clone(), import_count + index as u32);
    }

    for gc_type in &module_info.gc_types {
        match gc_type {
            GcTypeDef::Struct(fields) => {
                type_section.ty().struct_(fields.clone());
            }
            GcTypeDef::Array(field) => {
                type_section.ty().array(&field.element_type, field.mutable);
            }
        }
    }

    for import_def in module_info.imports.iter() {
        let params = import_def
            .ty
            .params(db)
            .iter()
            .map(|ty| type_to_valtype(db, *ty))
            .collect::<CompilationResult<Vec<_>>>()?;
        let results = result_types(db, import_def.ty.result(db))?;
        type_section.ty().function(params, results);
        let type_index = next_type_index;
        next_type_index += 1;
        import_section.import(
            import_def.module.as_str(),
            import_def.name.as_str(),
            EntityType::Function(type_index),
        );
    }

    for func_def in module_info.funcs.iter() {
        debug!("Processing function type for: {:?}", func_def.name);
        let params = func_def
            .ty
            .params(db)
            .iter()
            .map(|ty| type_to_valtype(db, *ty))
            .collect::<CompilationResult<Vec<_>>>();
        let params = match params {
            Ok(p) => {
                debug!("  params: {:?}", p);
                p
            }
            Err(e) => {
                debug!("Function params conversion failed: {:?}", e);
                return Err(e);
            }
        };
        let results = match result_types(db, func_def.ty.result(db)) {
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

    debug!("Processing {} exports...", module_info.exports.len());
    for export in module_info.exports.iter() {
        debug!("  export: {:?} -> {:?}", export.name, export.target);
        match &export.target {
            ExportTarget::Func(sym) => {
                let Some(index) = func_indices.get(sym) else {
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
            let Some(index) = func_indices.get(&func_def.name) else {
                continue;
            };
            // Export with simple name (last segment of qualified name)
            let name = func_def.name.name().to_string();
            export_section.export(&name, ExportKind::Func, *index);
        }
    }

    for data in module_info.data.iter() {
        let offset = ConstExpr::i32_const(data.offset);
        data_section.active(0, &offset, data.bytes.iter().copied());
    }

    debug!(
        "emit_wasm: emitting {} functions...",
        module_info.funcs.len()
    );
    for (i, func_def) in module_info.funcs.iter().enumerate() {
        debug!("emit_wasm: emitting function {}: {:?}", i, func_def.name);
        match emit_function(
            db,
            func_def,
            &func_indices,
            &module_info.type_idx_by_type,
            &module_info.func_types,
        ) {
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
    if module_info.memory.is_some() {
        module_bytes.section(&memory_section);
    }
    module_bytes.section(&export_section);
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
    funcs: &mut Vec<FunctionDef<'db>>,
    imports: &mut Vec<ImportFuncDef<'db>>,
    exports: &mut Vec<ExportDef>,
    memory: &mut Option<MemoryDef>,
    data: &mut Vec<DataDef>,
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
                    collect_wasm_ops_from_region(
                        db,
                        nested_region,
                        funcs,
                        imports,
                        exports,
                        memory,
                        data,
                    )?;
                }
                continue;
            }

            // Collect wasm operations
            if dialect == wasm_dialect {
                match name {
                    n if n == Symbol::new("func") => {
                        if let Ok(func_def) = extract_function_def(db, op) {
                            debug!("Including function: {}", func_def.name);
                            funcs.push(func_def);
                        }
                    }
                    n if n == Symbol::new("import_func") => {
                        imports.push(extract_import_def(db, op)?);
                    }
                    n if n == Symbol::new("export_func") => {
                        exports.push(extract_export_func(db, op)?);
                    }
                    n if n == Symbol::new("export_memory") => {
                        exports.push(extract_export_memory(db, op)?);
                    }
                    n if n == Symbol::new("memory") => {
                        *memory = Some(extract_memory_def(db, op)?);
                    }
                    n if n == Symbol::new("data") => {
                        data.push(extract_data_def(db, op)?);
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
    let mut funcs = Vec::new();
    let mut imports = Vec::new();
    let mut exports = Vec::new();
    let mut memory = None;
    let mut data = Vec::new();

    // Recursively collect wasm operations from the module and any nested core.module operations.
    collect_wasm_ops_from_region(
        db,
        &module.body(db),
        &mut funcs,
        &mut imports,
        &mut exports,
        &mut memory,
        &mut data,
    )?;

    let (gc_types, type_idx_by_type) = collect_gc_types(db, module)?;

    // Build function type lookup map for boxing/unboxing.
    // Use the qualified name already stored in func/import definitions.
    let mut func_types = HashMap::new();
    for func in &funcs {
        func_types.insert(func.name.clone(), func.ty);
    }
    for import in &imports {
        func_types.insert(import.sym.clone(), import.ty);
    }

    Ok(ModuleInfo {
        imports,
        funcs,
        exports,
        memory,
        data,
        gc_types,
        type_idx_by_type,
        func_types,
    })
}

fn collect_gc_types<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
) -> CompilationResult<(Vec<GcTypeDef>, HashMap<Type<'db>, u32>)> {
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
    let mut next_type_idx: u32 = 0;

    fn ensure_builder<'db, 'a>(
        builders: &'a mut Vec<GcTypeBuilder<'db>>,
        idx: u32,
    ) -> &'a mut GcTypeBuilder<'db> {
        if builders.len() <= idx as usize {
            builders.resize_with(idx as usize + 1, GcTypeBuilder::new);
        }
        &mut builders[idx as usize]
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
            let Some(type_idx) = get_type_idx(attrs, &mut type_idx_by_type, &mut next_type_idx)
            else {
                return Ok(());
            };
            let builder = ensure_builder(&mut builders, type_idx);
            builder.kind = GcKind::Struct;
            let field_count = op.operands(db).len();
            if matches!(builder.field_count, Some(existing_count) if existing_count != field_count)
            {
                let existing_count = builder.field_count.expect("count checked by matches");
                return Err(CompilationError::type_error(format!(
                    "struct type index {type_idx} field count mismatch ({existing_count} vs {field_count})",
                )));
            } else {
                builder.field_count = Some(field_count);
                if builder.fields.len() < field_count {
                    builder.fields.resize_with(field_count, || None);
                }
            }
            if let Some(result_ty) = op.results(db).first().copied() {
                register_type(&mut type_idx_by_type, type_idx, result_ty);
            }
            for (field_idx, value) in op.operands(db).iter().enumerate() {
                if let Some(ty) = value_type(db, *value) {
                    record_struct_field(type_idx, builder, field_idx as u32, ty)?;
                }
            }
        } else if name == Symbol::new("struct_get") {
            let attrs = op.attributes(db);
            let Some(type_idx) = get_type_idx(attrs, &mut type_idx_by_type, &mut next_type_idx)
            else {
                return Ok(());
            };
            let field_idx = attr_u32(attrs, ATTR_FIELD_IDX())?;
            let builder = ensure_builder(&mut builders, type_idx);
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
            if let Some(result_ty) = op.results(db).first().copied() {
                record_struct_field(type_idx, builder, field_idx, result_ty)?;
            }
        } else if name == Symbol::new("struct_set") {
            let attrs = op.attributes(db);
            let Some(type_idx) = get_type_idx(attrs, &mut type_idx_by_type, &mut next_type_idx)
            else {
                return Ok(());
            };
            let field_idx = attr_u32(attrs, ATTR_FIELD_IDX())?;
            let builder = ensure_builder(&mut builders, type_idx);
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
        } else if name == Symbol::new("array_new") || name == Symbol::new("array_new_default") {
            let attrs = op.attributes(db);
            let Some(type_idx) = get_type_idx(attrs, &mut type_idx_by_type, &mut next_type_idx)
            else {
                return Ok(());
            };
            let builder = ensure_builder(&mut builders, type_idx);
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
        } else if name == Symbol::new("array_get") {
            let attrs = op.attributes(db);
            let Some(type_idx) = get_type_idx(attrs, &mut type_idx_by_type, &mut next_type_idx)
            else {
                return Ok(());
            };
            let builder = ensure_builder(&mut builders, type_idx);
            builder.kind = GcKind::Array;
            if let Some(ty) = op
                .operands(db)
                .first()
                .copied()
                .and_then(|value| value_type(db, value))
            {
                register_type(&mut type_idx_by_type, type_idx, ty);
            }
            if let Some(result_ty) = op.results(db).first().copied() {
                record_array_elem(type_idx, builder, result_ty)?;
            }
        } else if name == Symbol::new("array_set") {
            let attrs = op.attributes(db);
            let Some(type_idx) = get_type_idx(attrs, &mut type_idx_by_type, &mut next_type_idx)
            else {
                return Ok(());
            };
            let builder = ensure_builder(&mut builders, type_idx);
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
            let builder = ensure_builder(&mut builders, type_idx);
            if builder.kind == GcKind::Unknown {
                builder.kind = GcKind::Struct;
            }
        }
        Ok(())
    };

    // Recursively visit operations, including nested core.module operations.
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

                // Visit wasm.func body
                if dialect == wasm_dialect && name == func_name {
                    if let Some(func_region) = op.regions(db).first() {
                        for block in func_region.blocks(db).iter() {
                            for inner_op in block.operations(db).iter() {
                                visit_op(inner_op)?;
                            }
                        }
                    }
                } else {
                    visit_op(op)?;
                }
            }
        }
        Ok(())
    }

    visit_region(db, &module.body(db), &mut visit_op)?;

    let to_field_type = |ty: Type<'db>, type_idx_by_type: &HashMap<_, _>| -> FieldType {
        let element_type = if core::I32::from_type(db, ty).is_some() {
            StorageType::Val(ValType::I32)
        } else if core::I64::from_type(db, ty).is_some()
            || ty::Int::from_type(db, ty).is_some()
            || ty::Nat::from_type(db, ty).is_some()
        {
            // Int/Nat (arbitrary precision) is lowered to i64 for Phase 1
            // TODO: Implement i31ref/BigInt hybrid for WasmGC
            StorageType::Val(ValType::I64)
        } else if core::F32::from_type(db, ty).is_some() {
            StorageType::Val(ValType::F32)
        } else if core::F64::from_type(db, ty).is_some() {
            StorageType::Val(ValType::F64)
        } else if let Some(type_idx) = type_idx_by_type.get(&ty).copied() {
            StorageType::Val(ValType::Ref(RefType {
                nullable: true,
                heap_type: HeapType::Concrete(type_idx),
            }))
        } else {
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

    Ok((result, type_idx_by_type))
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
        Attribute::QualifiedName(qn) => qn.clone(),
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
    let module = attr_string(attrs, ATTR_MODULE())?;
    let name = attr_string(attrs, ATTR_NAME())?;
    let sym = attr_symbol(attrs, ATTR_SYM_NAME())?;
    let ty = attr_type(attrs, ATTR_TYPE())?;

    let func_ty = core::Func::from_type(db, ty)
        .ok_or_else(|| CompilationError::type_error("wasm.import_func requires core.func type"))?;

    Ok(ImportFuncDef {
        sym: sym.into(), // Convert Symbol to QualifiedName
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
        target: ExportTarget::Func(func.clone()),
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
    let offset = attr_i32_attr(attrs, ATTR_OFFSET())?;
    let bytes = match attrs.get(&ATTR_BYTES()) {
        Some(Attribute::Bytes(value)) => value.clone(),
        _ => {
            return Err(CompilationError::from(
                errors::CompilationErrorKind::InvalidAttribute("bytes"),
            ));
        }
    };
    Ok(DataDef { offset, bytes })
}

fn emit_function<'db>(
    db: &'db dyn salsa::Database,
    func_def: &FunctionDef<'db>,
    func_indices: &HashMap<QualifiedName, u32>,
    type_idx_by_type: &HashMap<Type<'db>, u32>,
    func_types: &HashMap<QualifiedName, core::Func<'db>>,
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

    let mut value_locals: HashMap<Value<'db>, u32> = HashMap::new();
    let mut locals: Vec<ValType> = Vec::new();

    for (index, _) in block.args(db).iter().enumerate() {
        value_locals.insert(block.arg(db, index), index as u32);
    }

    let param_count = params.len() as u32;

    assign_locals_in_region(db, region, param_count, &mut locals, &mut value_locals)?;

    let mut function = Function::new(compress_locals(&locals));

    emit_region_ops(
        db,
        region,
        &value_locals,
        func_indices,
        type_idx_by_type,
        func_types,
        &mut function,
    )?;

    function.instruction(&Instruction::End);

    Ok(function)
}

fn assign_locals_in_region<'db>(
    db: &'db dyn salsa::Database,
    region: &Region<'db>,
    param_count: u32,
    locals: &mut Vec<ValType>,
    value_locals: &mut HashMap<Value<'db>, u32>,
) -> CompilationResult<()> {
    for block in region.blocks(db).iter() {
        for op in block.operations(db).iter() {
            let result_types = op.results(db);
            if result_types.len() > 1 {
                return Err(CompilationError::unsupported_feature("multi-result ops"));
            }
            if let Some(result_ty) = result_types
                .first()
                .copied()
                .filter(|ty| !is_nil_type(db, *ty))
            {
                let val_type = match type_to_valtype(db, result_ty) {
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
                value_locals.insert(op.result(db, 0), local_index);
                locals.push(val_type);
            }
            for nested in op.regions(db).iter() {
                assign_locals_in_region(db, nested, param_count, locals, value_locals)?;
            }
        }
    }
    Ok(())
}

fn emit_region_ops<'db>(
    db: &'db dyn salsa::Database,
    region: &Region<'db>,
    value_locals: &HashMap<Value<'db>, u32>,
    func_indices: &HashMap<QualifiedName, u32>,
    type_idx_by_type: &HashMap<Type<'db>, u32>,
    func_types: &HashMap<QualifiedName, core::Func<'db>>,
    function: &mut Function,
) -> CompilationResult<()> {
    let blocks = region.blocks(db);
    if blocks.len() != 1 {
        return Err(CompilationError::unsupported_feature("multi-block regions"));
    }
    let block = &blocks[0];
    for op in block.operations(db).iter() {
        emit_op(
            db,
            op,
            value_locals,
            func_indices,
            type_idx_by_type,
            func_types,
            function,
        )?;
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
    if op.results(db).is_empty() {
        None
    } else {
        Some(op.result(db, 0))
    }
}

fn emit_value_get<'db>(
    value: Value<'db>,
    value_locals: &HashMap<Value<'db>, u32>,
    function: &mut Function,
) -> CompilationResult<()> {
    let index = value_locals
        .get(&value)
        .ok_or_else(|| CompilationError::invalid_module("value missing local mapping"))?;
    function.instruction(&Instruction::LocalGet(*index));
    Ok(())
}

fn emit_op<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
    value_locals: &HashMap<Value<'db>, u32>,
    func_indices: &HashMap<QualifiedName, u32>,
    type_idx_by_type: &HashMap<Type<'db>, u32>,
    func_types: &HashMap<QualifiedName, core::Func<'db>>,
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
                return type_idx_by_type.get(ty).copied();
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
        emit_operands(db, operands, value_locals, function)?;
        function.instruction(instr);
        set_result_local(db, op, value_locals, function)?;
        return Ok(());
    }

    // Special cases: const, control flow, calls, locals, GC ops
    if name == Symbol::new("i32_const") {
        let value = attr_i32(db, op, ATTR_VALUE())?;
        function.instruction(&Instruction::I32Const(value));
        set_result_local(db, op, value_locals, function)?;
    } else if name == Symbol::new("i64_const") {
        let value = attr_i64(db, op, ATTR_VALUE())?;
        function.instruction(&Instruction::I64Const(value));
        set_result_local(db, op, value_locals, function)?;
    } else if name == Symbol::new("f32_const") {
        let value = attr_f32(db, op, ATTR_VALUE())?;
        function.instruction(&Instruction::F32Const(value.into()));
        set_result_local(db, op, value_locals, function)?;
    } else if name == Symbol::new("f64_const") {
        let value = attr_f64(db, op, ATTR_VALUE())?;
        function.instruction(&Instruction::F64Const(value.into()));
        set_result_local(db, op, value_locals, function)?;
    } else if name == Symbol::new("if") {
        let result_ty = op.results(db).first().copied();
        let has_result = matches!(result_ty, Some(ty) if !is_nil_type(db, ty));
        let block_type = if has_result {
            BlockType::Result(type_to_valtype(db, result_ty.expect("if result type"))?)
        } else {
            BlockType::Empty
        };
        if operands.len() != 1 {
            return Err(CompilationError::invalid_module(
                "wasm.if expects a single condition operand",
            ));
        }
        emit_operands(db, operands, value_locals, function)?;
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
        emit_region_ops(
            db,
            then_region,
            value_locals,
            func_indices,
            type_idx_by_type,
            func_types,
            function,
        )?;
        if let Some(value) = then_result {
            emit_value_get(value, value_locals, function)?;
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
            emit_region_ops(
                db,
                else_region,
                value_locals,
                func_indices,
                type_idx_by_type,
                func_types,
                function,
            )?;
            if let Some(value) = else_result {
                emit_value_get(value, value_locals, function)?;
            }
        } else if has_result {
            return Err(CompilationError::invalid_module(
                "wasm.if with result requires else region",
            ));
        }
        function.instruction(&Instruction::End);
        if has_result {
            set_result_local(db, op, value_locals, function)?;
        }
    } else if name == Symbol::new("block") {
        let result_ty = op.results(db).first().copied();
        let has_result = matches!(result_ty, Some(ty) if !is_nil_type(db, ty));
        let block_type = if has_result {
            BlockType::Result(type_to_valtype(db, result_ty.expect("block result type"))?)
        } else {
            BlockType::Empty
        };
        function.instruction(&Instruction::Block(block_type));
        let region = op
            .regions(db)
            .first()
            .ok_or_else(|| CompilationError::invalid_module("wasm.block missing body region"))?;
        emit_region_ops(
            db,
            region,
            value_locals,
            func_indices,
            type_idx_by_type,
            func_types,
            function,
        )?;
        if has_result {
            let value = region_result_value(db, region).ok_or_else(|| {
                CompilationError::invalid_module("wasm.block body missing result value")
            })?;
            emit_value_get(value, value_locals, function)?;
        }
        function.instruction(&Instruction::End);
        if has_result {
            set_result_local(db, op, value_locals, function)?;
        }
    } else if name == Symbol::new("loop") {
        let result_ty = op.results(db).first().copied();
        let has_result = matches!(result_ty, Some(ty) if !is_nil_type(db, ty));
        let block_type = if has_result {
            BlockType::Result(type_to_valtype(db, result_ty.expect("loop result type"))?)
        } else {
            BlockType::Empty
        };
        function.instruction(&Instruction::Loop(block_type));
        let region = op
            .regions(db)
            .first()
            .ok_or_else(|| CompilationError::invalid_module("wasm.loop missing body region"))?;
        emit_region_ops(
            db,
            region,
            value_locals,
            func_indices,
            type_idx_by_type,
            func_types,
            function,
        )?;
        if has_result {
            let value = region_result_value(db, region).ok_or_else(|| {
                CompilationError::invalid_module("wasm.loop body missing result value")
            })?;
            emit_value_get(value, value_locals, function)?;
        }
        function.instruction(&Instruction::End);
        if has_result {
            set_result_local(db, op, value_locals, function)?;
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
        emit_operands(db, operands, value_locals, function)?;
        let depth = attr_u32(op.attributes(db), ATTR_TARGET())?;
        function.instruction(&Instruction::BrIf(depth));
    } else if name == Symbol::new("call") {
        let callee = attr_symbol_ref(db, op, ATTR_CALLEE())?;
        let target = resolve_callee(callee, func_indices)?;

        // Check if we need boxing for generic function calls
        if let Some(callee_ty) = func_types.get(callee) {
            let param_types = callee_ty.params(db);
            emit_operands_with_boxing(db, operands, &param_types, value_locals, function)?;
        } else {
            emit_operands(db, operands, value_locals, function)?;
        }

        function.instruction(&Instruction::Call(target));

        // Check if we need unboxing for the return value
        if let Some(callee_ty) = func_types.get(callee) {
            let return_ty = callee_ty.result(db);
            // If callee returns anyref (type.var) but we expect concrete type, unbox
            if ty::is_var(db, return_ty)
                && let Some(result_ty) = op.results(db).first()
            {
                emit_unboxing(db, *result_ty, function)?;
            }
        }

        set_result_local(db, op, value_locals, function)?;
    } else if name == Symbol::new("return_call") {
        let callee = attr_symbol_ref(db, op, ATTR_CALLEE())?;
        let target = resolve_callee(callee, func_indices)?;

        // Check if we need boxing for generic function calls
        if let Some(callee_ty) = func_types.get(callee) {
            let param_types = callee_ty.params(db);
            emit_operands_with_boxing(db, operands, &param_types, value_locals, function)?;
        } else {
            emit_operands(db, operands, value_locals, function)?;
        }

        // Note: Return unboxing is not needed for tail calls since
        // the caller's return type should match the callee's.
        function.instruction(&Instruction::ReturnCall(target));
    } else if name == Symbol::new("local_get") {
        let index = attr_local_index(db, op)?;
        function.instruction(&Instruction::LocalGet(index));
        set_result_local(db, op, value_locals, function)?;
    } else if name == Symbol::new("local_set") {
        let index = attr_local_index(db, op)?;
        emit_operands(db, operands, value_locals, function)?;
        function.instruction(&Instruction::LocalSet(index));
    } else if name == Symbol::new("local_tee") {
        let index = attr_local_index(db, op)?;
        emit_operands(db, operands, value_locals, function)?;
        function.instruction(&Instruction::LocalTee(index));
        set_result_local(db, op, value_locals, function)?;
    } else if name == Symbol::new("struct_new") {
        emit_operands(db, operands, value_locals, function)?;
        let attrs = op.attributes(db);
        let type_idx = get_type_idx_from_attrs(attrs)
            .ok_or_else(|| CompilationError::missing_attribute("type or type_idx"))?;
        function.instruction(&Instruction::StructNew(type_idx));
        set_result_local(db, op, value_locals, function)?;
    } else if name == Symbol::new("struct_get") {
        emit_operands(db, operands, value_locals, function)?;
        let attrs = op.attributes(db);
        let type_idx = get_type_idx_from_attrs(attrs)
            .ok_or_else(|| CompilationError::missing_attribute("type or type_idx"))?;
        let field_idx = attr_u32(attrs, ATTR_FIELD_IDX())?;
        function.instruction(&Instruction::StructGet {
            struct_type_index: type_idx,
            field_index: field_idx,
        });
        set_result_local(db, op, value_locals, function)?;
    } else if name == Symbol::new("struct_set") {
        emit_operands(db, operands, value_locals, function)?;
        let attrs = op.attributes(db);
        let type_idx = get_type_idx_from_attrs(attrs)
            .ok_or_else(|| CompilationError::missing_attribute("type or type_idx"))?;
        let field_idx = attr_u32(attrs, ATTR_FIELD_IDX())?;
        function.instruction(&Instruction::StructSet {
            struct_type_index: type_idx,
            field_index: field_idx,
        });
    } else if name == Symbol::new("array_new") {
        emit_operands(db, operands, value_locals, function)?;
        let attrs = op.attributes(db);
        let type_idx = get_type_idx_from_attrs(attrs)
            .ok_or_else(|| CompilationError::missing_attribute("type or type_idx"))?;
        function.instruction(&Instruction::ArrayNew(type_idx));
        set_result_local(db, op, value_locals, function)?;
    } else if name == Symbol::new("array_new_default") {
        emit_operands(db, operands, value_locals, function)?;
        let attrs = op.attributes(db);
        let type_idx = get_type_idx_from_attrs(attrs)
            .ok_or_else(|| CompilationError::missing_attribute("type or type_idx"))?;
        function.instruction(&Instruction::ArrayNewDefault(type_idx));
        set_result_local(db, op, value_locals, function)?;
    } else if name == Symbol::new("array_get") {
        emit_operands(db, operands, value_locals, function)?;
        let attrs = op.attributes(db);
        let type_idx = get_type_idx_from_attrs(attrs)
            .ok_or_else(|| CompilationError::missing_attribute("type or type_idx"))?;
        function.instruction(&Instruction::ArrayGet(type_idx));
        set_result_local(db, op, value_locals, function)?;
    } else if name == Symbol::new("array_set") {
        emit_operands(db, operands, value_locals, function)?;
        let attrs = op.attributes(db);
        let type_idx = get_type_idx_from_attrs(attrs)
            .ok_or_else(|| CompilationError::missing_attribute("type or type_idx"))?;
        function.instruction(&Instruction::ArraySet(type_idx));
    } else if name == Symbol::new("ref_null") {
        let attrs = op.attributes(db);
        let heap_type = attr_heap_type(attrs, ATTR_HEAP_TYPE())
            .ok()
            .or_else(|| get_type_idx_from_attrs(attrs).map(HeapType::Concrete))
            .ok_or_else(|| CompilationError::missing_attribute("heap_type or type"))?;
        function.instruction(&Instruction::RefNull(heap_type));
        set_result_local(db, op, value_locals, function)?;
    } else if name == Symbol::new("ref_cast") {
        emit_operands(db, operands, value_locals, function)?;
        let attrs = op.attributes(db);
        let heap_type = attr_heap_type(attrs, ATTR_TARGET_TYPE())
            .ok()
            .or_else(|| get_type_idx_from_attrs(attrs).map(HeapType::Concrete))
            .ok_or_else(|| CompilationError::missing_attribute("target_type or type"))?;
        function.instruction(&Instruction::RefCastNullable(heap_type));
        set_result_local(db, op, value_locals, function)?;
    } else if name == Symbol::new("ref_test") {
        emit_operands(db, operands, value_locals, function)?;
        let attrs = op.attributes(db);
        let heap_type = attr_heap_type(attrs, ATTR_TARGET_TYPE())
            .ok()
            .or_else(|| get_type_idx_from_attrs(attrs).map(HeapType::Concrete))
            .ok_or_else(|| CompilationError::missing_attribute("target_type or type"))?;
        function.instruction(&Instruction::RefTestNullable(heap_type));
        set_result_local(db, op, value_locals, function)?;
    } else {
        return Err(CompilationError::unsupported_feature(
            "wasm op not supported",
        ));
    }

    Ok(())
}

fn emit_operands<'db>(
    db: &'db dyn salsa::Database,
    operands: &IdVec<Value<'db>>,
    value_locals: &HashMap<Value<'db>, u32>,
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
        if let Some(index) = value_locals.get(value) {
            debug!(
                "  emit_operands: found value {:?} -> local {}",
                value.def(db),
                index
            );
            function.instruction(&Instruction::LocalGet(*index));
            continue;
        }

        // Handle stale block argument references (issue #43)
        // The resolver creates operands that reference OLD block arguments, but value_locals
        // only contains NEW block arguments. For block args, we can use the index directly
        // since parameters are always locals 0, 1, 2, etc.
        if let ValueDef::BlockArg(block_id) = value.def(db) {
            let index = value.index(db) as u32;
            debug!(
                "  emit_operands: BlockArg fallback {:?} -> local {}",
                block_id, index
            );
            function.instruction(&Instruction::LocalGet(index));
            continue;
        }

        // If operand not found and not a block arg, this is an ERROR - stale value reference!
        if let ValueDef::OpResult(stale_op) = value.def(db) {
            debug!(
                "  emit_operands: STALE OpResult! op={}.{}; value_locals has {} entries",
                stale_op.dialect(db),
                stale_op.name(db),
                value_locals.len(),
            );
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
    value_locals: &HashMap<Value<'db>, u32>,
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
        emit_value(db, *value, value_locals, function)?;

        // Check if boxing is needed
        // If parameter expects anyref (type.var), box the operand
        if let Some(&param_ty) = param_ty
            && ty::is_var(db, param_ty)
            && let Some(operand_ty) = value_type(db, *value)
        {
            emit_boxing(db, operand_ty, function)?;
        }
    }
    Ok(())
}

/// Emit a single value (local.get or block arg fallback).
fn emit_value<'db>(
    db: &'db dyn salsa::Database,
    value: Value<'db>,
    value_locals: &HashMap<Value<'db>, u32>,
    function: &mut Function,
) -> CompilationResult<()> {
    // Try direct lookup first
    if let Some(index) = value_locals.get(&value) {
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
    Err(CompilationError::invalid_module(
        "stale SSA value in wasm backend (missing local mapping)",
    ))
}

/// Emit boxing instructions to convert a concrete type to anyref.
/// - Int (i64) → i31ref: truncate to i32 and use ref.i31
/// - Float (f64) → BoxedF64 struct (TODO: requires BoxedF64 type to be defined)
fn emit_boxing<'db>(
    db: &'db dyn salsa::Database,
    ty: Type<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    if ty::Int::from_type(db, ty).is_some() || ty::Nat::from_type(db, ty).is_some() {
        // Int/Nat (i64) → i31ref
        // Truncate i64 to i32, then convert to i31ref
        // Note: This only works correctly for values that fit in 31 bits!
        // Phase 1 limitation: values outside -2^30..2^30-1 will be incorrect
        function.instruction(&Instruction::I32WrapI64);
        function.instruction(&Instruction::RefI31);
        Ok(())
    } else if core::F64::from_type(db, ty).is_some() {
        // Float (f64) → BoxedF64 struct
        // TODO: This requires defining BoxedF64 type and using struct.new
        // For now, return an error
        Err(CompilationError::unsupported_feature(
            "Float boxing not yet implemented",
        ))
    } else {
        // For reference types (structs, etc.), no boxing needed - they're already subtypes of anyref
        // Just leave the value as-is on the stack
        Ok(())
    }
}

/// Emit unboxing instructions to convert anyref to a concrete type.
/// - i31ref → Int (i64): extract i32 and extend to i64
fn emit_unboxing<'db>(
    db: &'db dyn salsa::Database,
    ty: Type<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    if ty::Int::from_type(db, ty).is_some() {
        // anyref (i31ref) → Int (i64)
        // Extract i32 from i31ref, then sign-extend to i64
        function.instruction(&Instruction::I31GetS);
        function.instruction(&Instruction::I64ExtendI32S);
        Ok(())
    } else if ty::Nat::from_type(db, ty).is_some() {
        // anyref (i31ref) → Nat (i64)
        // Extract u32 from i31ref, then zero-extend to i64
        function.instruction(&Instruction::I31GetU);
        function.instruction(&Instruction::I64ExtendI32U);
        Ok(())
    } else if core::F64::from_type(db, ty).is_some() {
        // anyref (BoxedF64) → Float (f64)
        // TODO: This requires BoxedF64 type and struct.get
        Err(CompilationError::unsupported_feature(
            "Float unboxing not yet implemented",
        ))
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

fn set_result_local<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
    value_locals: &HashMap<Value<'db>, u32>,
    function: &mut Function,
) -> CompilationResult<()> {
    if op.results(db).is_empty() {
        return Ok(());
    }
    let local = value_locals
        .get(&op.result(db, 0))
        .ok_or_else(|| CompilationError::invalid_module("result missing local mapping"))?;
    function.instruction(&Instruction::LocalSet(*local));
    Ok(())
}

fn resolve_callee(
    path: &QualifiedName,
    func_indices: &HashMap<QualifiedName, u32>,
) -> CompilationResult<u32> {
    func_indices
        .get(path)
        .copied()
        .ok_or_else(|| CompilationError::function_not_found(&path.to_string()))
}

fn type_to_valtype<'db>(db: &'db dyn salsa::Database, ty: Type<'db>) -> CompilationResult<ValType> {
    if core::I32::from_type(db, ty).is_some() || core::I1::from_type(db, ty).is_some() {
        // core.i1 (Bool) is represented as i32 in WebAssembly
        Ok(ValType::I32)
    } else if core::I64::from_type(db, ty).is_some()
        || ty::Int::from_type(db, ty).is_some()
        || ty::Nat::from_type(db, ty).is_some()
    {
        // Int/Nat (arbitrary precision) is lowered to i64 for Phase 1
        // TODO: Implement i31ref/BigInt hybrid for WasmGC
        Ok(ValType::I64)
    } else if core::F32::from_type(db, ty).is_some() {
        Ok(ValType::F32)
    } else if core::F64::from_type(db, ty).is_some() {
        Ok(ValType::F64)
    } else if core::String::from_type(db, ty).is_some()
        || core::Bytes::from_type(db, ty).is_some()
        || (ty.dialect(db) == core::DIALECT_NAME() && ty.name(db) == Symbol::new("ptr"))
    {
        Ok(ValType::I32)
    } else if ty::is_var(db, ty) {
        // Generic type variables use anyref (uniform representation)
        // Values must be boxed when passed to generic functions
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
) -> CompilationResult<Vec<ValType>> {
    if is_nil_type(db, ty) {
        Ok(Vec::new())
    } else {
        Ok(vec![type_to_valtype(db, ty)?])
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

fn attr_local_index<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
) -> CompilationResult<u32> {
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
) -> CompilationResult<&'db QualifiedName> {
    match op.attributes(db).get(&key) {
        Some(Attribute::QualifiedName(path)) => Ok(path),
        _ => Err(CompilationError::from(
            errors::CompilationErrorKind::InvalidAttribute("callee"),
        )),
    }
}

fn attr_symbol_ref_attr<'db>(
    attrs: &'db Attrs<'db>,
    key: Symbol,
) -> CompilationResult<&'db QualifiedName> {
    match attrs.get(&key) {
        Some(Attribute::QualifiedName(path)) => Ok(path),
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

fn attr_heap_type<'db>(attrs: &Attrs<'db>, key: Symbol) -> CompilationResult<HeapType> {
    match attrs.get(&key) {
        Some(Attribute::IntBits(bits)) => Ok(HeapType::Concrete(*bits as u32)),
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
            .attr("type_idx", Attribute::IntBits(0))
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
        let (gc_types, type_map) = collect_gc_types(db, module).expect("collect_gc_types failed");

        // Should have one GC type
        assert_eq!(gc_types.len(), 1);
        assert_eq!(gc_type_kind(&gc_types[0]), "struct");

        // Type 0 should be in the map
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
            .attr("type_idx", Attribute::IntBits(0))
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
        let (gc_types, _type_map) = collect_gc_types(db, module).expect("collect_gc_types failed");

        // Should have one GC type (array)
        assert_eq!(gc_types.len(), 1);
        assert_eq!(gc_type_kind(&gc_types[0]), "array");
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
            .attr("type_idx", Attribute::IntBits(0))
            .build();

        let field2 = Operation::of_name(db, location, "wasm.i32_const")
            .attr("value", Attribute::IntBits(2))
            .results(idvec![i32_ty])
            .build();

        let struct_new2 = Operation::of_name(db, location, "wasm.struct_new")
            .operands(idvec![field2.result(db, 0)])
            .results(idvec![i32_ty])
            .attr("type_idx", Attribute::IntBits(0)) // same type_idx
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
        let (gc_types, _type_map) = collect_gc_types(db, module).expect("collect_gc_types failed");

        // Should have only one GC type (same type_idx used twice)
        assert_eq!(gc_types.len(), 1);
        assert_eq!(gc_type_kind(&gc_types[0]), "struct");
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
            .attr("type_idx", Attribute::IntBits(0))
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
            .attr("type_idx", Attribute::IntBits(0)) // same type_idx, different field count
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
            .attr("type_idx", Attribute::IntBits(0))
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
        let test_name = QualifiedName::simple(Symbol::new("test_fn"));
        let wasm_func = Operation::of_name(db, location, "wasm.func")
            .attr("sym_name", Attribute::QualifiedName(test_name))
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
        let (gc_types, _type_map) = collect_gc_types(db, module).expect("collect_gc_types failed");

        // Should find the struct type from inside the function body
        assert_eq!(gc_types.len(), 1);
        assert_eq!(gc_type_kind(&gc_types[0]), "struct");
    }
}
