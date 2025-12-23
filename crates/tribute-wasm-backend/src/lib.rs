//! Wasm backend that emits WebAssembly binaries from wasm.* TrunkIR operations.

use std::collections::HashMap;

use trunk_ir::dialect::core;
use trunk_ir::{
    Attribute, Attrs, DialectType, IdVec, Operation, QualifiedName, Region, Symbol, Type, Value,
    ValueDef,
};
use wasm_encoder::{
    BlockType, CodeSection, ConstExpr, DataSection, EntityType, ExportKind, ExportSection,
    FieldType, Function, FunctionSection, HeapType, ImportSection, Instruction, MemorySection,
    MemoryType, Module, RefType, StorageType, TypeSection, ValType,
};

mod errors;
pub mod lower_wasm;
pub mod translate;

pub use errors::{CompilationError, CompilationResult};
pub use translate::{WasmBinary, compile_to_wasm};

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

struct FunctionDef<'db> {
    name: Symbol,
    ty: core::Func<'db>,
    op: Operation<'db>,
}

struct ImportFuncDef<'db> {
    sym: Symbol,
    module: String,
    name: String,
    ty: core::Func<'db>,
}

struct ExportDef {
    name: String,
    kind: ExportKind,
    target: ExportTarget,
}

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
}

struct ModuleInfo<'db> {
    imports: Vec<ImportFuncDef<'db>>,
    funcs: Vec<FunctionDef<'db>>,
    exports: Vec<ExportDef>,
    memory: Option<MemoryDef>,
    data: Vec<DataDef>,
    gc_types: Vec<GcTypeDef>,
}

enum GcTypeDef {
    Struct(Vec<FieldType>),
    Array(FieldType),
}

pub fn emit_wasm<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
) -> CompilationResult<Vec<u8>> {
    let module_info = collect_module_info(db, module)?;

    let mut type_section = TypeSection::new();
    let mut import_section = ImportSection::new();
    let mut function_section = FunctionSection::new();
    let mut memory_section = MemorySection::new();
    let mut export_section = ExportSection::new();
    let mut code_section = CodeSection::new();
    let mut data_section = DataSection::new();

    let mut func_indices = HashMap::new();
    let gc_type_count = module_info.gc_types.len() as u32;
    let mut next_type_index = gc_type_count;

    for (index, import_def) in module_info.imports.iter().enumerate() {
        func_indices.insert(import_def.sym, index as u32);
    }
    let import_count = module_info.imports.len() as u32;
    for (index, func_def) in module_info.funcs.iter().enumerate() {
        func_indices.insert(func_def.name, import_count + index as u32);
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
        let params = func_def
            .ty
            .params(db)
            .iter()
            .map(|ty| type_to_valtype(db, *ty))
            .collect::<CompilationResult<Vec<_>>>()?;
        let results = result_types(db, func_def.ty.result(db))?;
        type_section.ty().function(params, results);
        let type_index = next_type_index;
        next_type_index += 1;
        function_section.function(type_index);
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

    for export in module_info.exports.iter() {
        match export.target {
            ExportTarget::Func(sym) => {
                let Some(index) = func_indices.get(&sym) else {
                    return Err(CompilationError::function_not_found(&sym.to_string()));
                };
                export_section.export(export.name.as_str(), export.kind, *index);
            }
            ExportTarget::Memory(index) => {
                export_section.export(export.name.as_str(), export.kind, index);
            }
        }
    }
    if module_info.exports.is_empty() {
        for func_def in module_info.funcs.iter() {
            let Some(index) = func_indices.get(&func_def.name) else {
                continue;
            };
            func_def.name.with_str(|name| {
                export_section.export(name, ExportKind::Func, *index);
            });
        }
    }

    for data in module_info.data.iter() {
        let offset = ConstExpr::i32_const(data.offset);
        data_section.active(0, &offset, data.bytes.iter().copied());
    }

    for func_def in &module_info.funcs {
        let function = emit_function(db, func_def, &func_indices)?;
        code_section.function(&function);
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

fn collect_module_info<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
) -> CompilationResult<ModuleInfo<'db>> {
    let mut funcs = Vec::new();
    let mut imports = Vec::new();
    let mut exports = Vec::new();
    let mut memory = None;
    let mut data = Vec::new();
    let func_dialect = Symbol::new("func");
    let func_name = Symbol::new("func");
    let wasm_dialect = Symbol::new("wasm");

    let body = module.body(db);
    for block in body.blocks(db).iter() {
        for op in block.operations(db).iter() {
            if op.dialect(db) == func_dialect && op.name(db) == func_name {
                funcs.push(extract_function_def(db, op)?);
            } else if op.dialect(db) == wasm_dialect {
                match op.name(db) {
                    name if name == Symbol::new("import_func") => {
                        imports.push(extract_import_def(db, op)?);
                    }
                    name if name == Symbol::new("export_func") => {
                        exports.push(extract_export_func(db, op)?);
                    }
                    name if name == Symbol::new("export_memory") => {
                        exports.push(extract_export_memory(db, op)?);
                    }
                    name if name == Symbol::new("memory") => {
                        memory = Some(extract_memory_def(db, op)?);
                    }
                    name if name == Symbol::new("data") => {
                        data.push(extract_data_def(db, op)?);
                    }
                    _ => {}
                }
            }
        }
    }

    Ok(ModuleInfo {
        imports,
        funcs,
        exports,
        memory,
        data,
        gc_types: collect_gc_types(db, module)?,
    })
}

fn collect_gc_types<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
) -> CompilationResult<Vec<GcTypeDef>> {
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

    let mut visit_op = |op: &Operation<'db>| -> CompilationResult<()> {
        if op.dialect(db) != wasm_dialect {
            return Ok(());
        }
        let name = op.name(db);
        if name == Symbol::new("struct_new") {
            let type_idx = attr_u32(op.attributes(db), ATTR_TYPE_IDX())?;
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
            let type_idx = attr_u32(op.attributes(db), ATTR_TYPE_IDX())?;
            let field_idx = attr_u32(op.attributes(db), ATTR_FIELD_IDX())?;
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
            let type_idx = attr_u32(op.attributes(db), ATTR_TYPE_IDX())?;
            let field_idx = attr_u32(op.attributes(db), ATTR_FIELD_IDX())?;
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
            let type_idx = attr_u32(op.attributes(db), ATTR_TYPE_IDX())?;
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
            let type_idx = attr_u32(op.attributes(db), ATTR_TYPE_IDX())?;
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
            let type_idx = attr_u32(op.attributes(db), ATTR_TYPE_IDX())?;
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
            let type_idx = if name == Symbol::new("ref_null") {
                attr_u32(op.attributes(db), ATTR_HEAP_TYPE())?
            } else {
                attr_u32(op.attributes(db), ATTR_TARGET_TYPE())?
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

    let body = module.body(db);
    for block in body.blocks(db).iter() {
        for op in block.operations(db).iter() {
            if op.dialect(db) == Symbol::new("func") && op.name(db) == Symbol::new("func") {
                if let Some(region) = op.regions(db).first() {
                    for block in region.blocks(db).iter() {
                        for op in block.operations(db).iter() {
                            visit_op(op)?;
                        }
                    }
                }
            } else {
                visit_op(op)?;
            }
        }
    }

    let to_field_type = |ty: Type<'db>, type_idx_by_type: &HashMap<_, _>| -> FieldType {
        let element_type = if core::I32::from_type(db, ty).is_some() {
            StorageType::Val(ValType::I32)
        } else if core::I64::from_type(db, ty).is_some() {
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

    Ok(result)
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
        .ok_or_else(|| CompilationError::type_error("func.func requires core.func type"))?;

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
    let sym = func.name();
    Ok(ExportDef {
        name,
        kind: ExportKind::Func,
        target: ExportTarget::Func(sym),
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
    func_indices: &HashMap<Symbol, u32>,
) -> CompilationResult<Function> {
    let region = func_def
        .op
        .regions(db)
        .first()
        .ok_or_else(|| CompilationError::invalid_module("func.func missing body region"))?;
    let blocks = region.blocks(db);
    let block = blocks
        .first()
        .ok_or_else(|| CompilationError::invalid_module("func.func has no entry block"))?;

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

    emit_region_ops(db, region, &value_locals, func_indices, &mut function)?;

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
                let val_type = type_to_valtype(db, result_ty)?;
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
    func_indices: &HashMap<Symbol, u32>,
    function: &mut Function,
) -> CompilationResult<()> {
    let blocks = region.blocks(db);
    if blocks.len() != 1 {
        return Err(CompilationError::unsupported_feature("multi-block regions"));
    }
    let block = &blocks[0];
    for op in block.operations(db).iter() {
        emit_op(db, op, value_locals, func_indices, function)?;
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
    func_indices: &HashMap<Symbol, u32>,
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
    } else if name == Symbol::new("i32_add") {
        emit_operands(operands, value_locals, function)?;
        function.instruction(&Instruction::I32Add);
        set_result_local(db, op, value_locals, function)?;
    } else if name == Symbol::new("i32_sub") {
        emit_operands(operands, value_locals, function)?;
        function.instruction(&Instruction::I32Sub);
        set_result_local(db, op, value_locals, function)?;
    } else if name == Symbol::new("i32_mul") {
        emit_operands(operands, value_locals, function)?;
        function.instruction(&Instruction::I32Mul);
        set_result_local(db, op, value_locals, function)?;
    } else if name == Symbol::new("i32_eq") {
        emit_operands(operands, value_locals, function)?;
        function.instruction(&Instruction::I32Eq);
        set_result_local(db, op, value_locals, function)?;
    } else if name == Symbol::new("i32_ne") {
        emit_operands(operands, value_locals, function)?;
        function.instruction(&Instruction::I32Ne);
        set_result_local(db, op, value_locals, function)?;
    } else if name == Symbol::new("i32_lt_s") {
        emit_operands(operands, value_locals, function)?;
        function.instruction(&Instruction::I32LtS);
        set_result_local(db, op, value_locals, function)?;
    } else if name == Symbol::new("i32_lt_u") {
        emit_operands(operands, value_locals, function)?;
        function.instruction(&Instruction::I32LtU);
        set_result_local(db, op, value_locals, function)?;
    } else if name == Symbol::new("i32_le_s") {
        emit_operands(operands, value_locals, function)?;
        function.instruction(&Instruction::I32LeS);
        set_result_local(db, op, value_locals, function)?;
    } else if name == Symbol::new("i32_le_u") {
        emit_operands(operands, value_locals, function)?;
        function.instruction(&Instruction::I32LeU);
        set_result_local(db, op, value_locals, function)?;
    } else if name == Symbol::new("i32_gt_s") {
        emit_operands(operands, value_locals, function)?;
        function.instruction(&Instruction::I32GtS);
        set_result_local(db, op, value_locals, function)?;
    } else if name == Symbol::new("i32_gt_u") {
        emit_operands(operands, value_locals, function)?;
        function.instruction(&Instruction::I32GtU);
        set_result_local(db, op, value_locals, function)?;
    } else if name == Symbol::new("i32_ge_s") {
        emit_operands(operands, value_locals, function)?;
        function.instruction(&Instruction::I32GeS);
        set_result_local(db, op, value_locals, function)?;
    } else if name == Symbol::new("i32_ge_u") {
        emit_operands(operands, value_locals, function)?;
        function.instruction(&Instruction::I32GeU);
        set_result_local(db, op, value_locals, function)?;
    } else if name == Symbol::new("i32_div_s") {
        emit_operands(operands, value_locals, function)?;
        function.instruction(&Instruction::I32DivS);
        set_result_local(db, op, value_locals, function)?;
    } else if name == Symbol::new("i32_div_u") {
        emit_operands(operands, value_locals, function)?;
        function.instruction(&Instruction::I32DivU);
        set_result_local(db, op, value_locals, function)?;
    } else if name == Symbol::new("i32_rem_s") {
        emit_operands(operands, value_locals, function)?;
        function.instruction(&Instruction::I32RemS);
        set_result_local(db, op, value_locals, function)?;
    } else if name == Symbol::new("i32_rem_u") {
        emit_operands(operands, value_locals, function)?;
        function.instruction(&Instruction::I32RemU);
        set_result_local(db, op, value_locals, function)?;
    } else if name == Symbol::new("i64_add") {
        emit_operands(operands, value_locals, function)?;
        function.instruction(&Instruction::I64Add);
        set_result_local(db, op, value_locals, function)?;
    } else if name == Symbol::new("i64_sub") {
        emit_operands(operands, value_locals, function)?;
        function.instruction(&Instruction::I64Sub);
        set_result_local(db, op, value_locals, function)?;
    } else if name == Symbol::new("i64_mul") {
        emit_operands(operands, value_locals, function)?;
        function.instruction(&Instruction::I64Mul);
        set_result_local(db, op, value_locals, function)?;
    } else if name == Symbol::new("f32_add") {
        emit_operands(operands, value_locals, function)?;
        function.instruction(&Instruction::F32Add);
        set_result_local(db, op, value_locals, function)?;
    } else if name == Symbol::new("f32_sub") {
        emit_operands(operands, value_locals, function)?;
        function.instruction(&Instruction::F32Sub);
        set_result_local(db, op, value_locals, function)?;
    } else if name == Symbol::new("f32_mul") {
        emit_operands(operands, value_locals, function)?;
        function.instruction(&Instruction::F32Mul);
        set_result_local(db, op, value_locals, function)?;
    } else if name == Symbol::new("f32_div") {
        emit_operands(operands, value_locals, function)?;
        function.instruction(&Instruction::F32Div);
        set_result_local(db, op, value_locals, function)?;
    } else if name == Symbol::new("f64_add") {
        emit_operands(operands, value_locals, function)?;
        function.instruction(&Instruction::F64Add);
        set_result_local(db, op, value_locals, function)?;
    } else if name == Symbol::new("f64_sub") {
        emit_operands(operands, value_locals, function)?;
        function.instruction(&Instruction::F64Sub);
        set_result_local(db, op, value_locals, function)?;
    } else if name == Symbol::new("f64_mul") {
        emit_operands(operands, value_locals, function)?;
        function.instruction(&Instruction::F64Mul);
        set_result_local(db, op, value_locals, function)?;
    } else if name == Symbol::new("f64_div") {
        emit_operands(operands, value_locals, function)?;
        function.instruction(&Instruction::F64Div);
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
        emit_operands(operands, value_locals, function)?;
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
        emit_region_ops(db, then_region, value_locals, func_indices, function)?;
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
            emit_region_ops(db, else_region, value_locals, func_indices, function)?;
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
        emit_region_ops(db, region, value_locals, func_indices, function)?;
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
        emit_region_ops(db, region, value_locals, func_indices, function)?;
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
        emit_operands(operands, value_locals, function)?;
        let depth = attr_u32(op.attributes(db), ATTR_TARGET())?;
        function.instruction(&Instruction::BrIf(depth));
    } else if name == Symbol::new("drop") {
        emit_operands(operands, value_locals, function)?;
        function.instruction(&Instruction::Drop);
    } else if name == Symbol::new("call") {
        emit_operands(operands, value_locals, function)?;
        let callee = attr_symbol_ref(db, op, ATTR_CALLEE())?;
        let target = resolve_callee(callee, func_indices)?;
        function.instruction(&Instruction::Call(target));
        set_result_local(db, op, value_locals, function)?;
    } else if name == Symbol::new("return_call") {
        emit_operands(operands, value_locals, function)?;
        let callee = attr_symbol_ref(db, op, ATTR_CALLEE())?;
        let target = resolve_callee(callee, func_indices)?;
        function.instruction(&Instruction::ReturnCall(target));
    } else if name == Symbol::new("return") {
        emit_operands(operands, value_locals, function)?;
        function.instruction(&Instruction::Return);
    } else if name == Symbol::new("local_get") {
        let index = attr_local_index(db, op)?;
        function.instruction(&Instruction::LocalGet(index));
        set_result_local(db, op, value_locals, function)?;
    } else if name == Symbol::new("local_set") {
        let index = attr_local_index(db, op)?;
        emit_operands(operands, value_locals, function)?;
        function.instruction(&Instruction::LocalSet(index));
    } else if name == Symbol::new("local_tee") {
        let index = attr_local_index(db, op)?;
        emit_operands(operands, value_locals, function)?;
        function.instruction(&Instruction::LocalTee(index));
        set_result_local(db, op, value_locals, function)?;
    } else if name == Symbol::new("struct_new") {
        emit_operands(operands, value_locals, function)?;
        let type_idx = attr_u32(op.attributes(db), ATTR_TYPE_IDX())?;
        function.instruction(&Instruction::StructNew(type_idx));
        set_result_local(db, op, value_locals, function)?;
    } else if name == Symbol::new("struct_get") {
        emit_operands(operands, value_locals, function)?;
        let type_idx = attr_u32(op.attributes(db), ATTR_TYPE_IDX())?;
        let field_idx = attr_u32(op.attributes(db), ATTR_FIELD_IDX())?;
        function.instruction(&Instruction::StructGet {
            struct_type_index: type_idx,
            field_index: field_idx,
        });
        set_result_local(db, op, value_locals, function)?;
    } else if name == Symbol::new("struct_set") {
        emit_operands(operands, value_locals, function)?;
        let type_idx = attr_u32(op.attributes(db), ATTR_TYPE_IDX())?;
        let field_idx = attr_u32(op.attributes(db), ATTR_FIELD_IDX())?;
        function.instruction(&Instruction::StructSet {
            struct_type_index: type_idx,
            field_index: field_idx,
        });
    } else if name == Symbol::new("array_new") {
        emit_operands(operands, value_locals, function)?;
        let type_idx = attr_u32(op.attributes(db), ATTR_TYPE_IDX())?;
        function.instruction(&Instruction::ArrayNew(type_idx));
        set_result_local(db, op, value_locals, function)?;
    } else if name == Symbol::new("array_new_default") {
        emit_operands(operands, value_locals, function)?;
        let type_idx = attr_u32(op.attributes(db), ATTR_TYPE_IDX())?;
        function.instruction(&Instruction::ArrayNewDefault(type_idx));
        set_result_local(db, op, value_locals, function)?;
    } else if name == Symbol::new("array_get") {
        emit_operands(operands, value_locals, function)?;
        let type_idx = attr_u32(op.attributes(db), ATTR_TYPE_IDX())?;
        function.instruction(&Instruction::ArrayGet(type_idx));
        set_result_local(db, op, value_locals, function)?;
    } else if name == Symbol::new("array_set") {
        emit_operands(operands, value_locals, function)?;
        let type_idx = attr_u32(op.attributes(db), ATTR_TYPE_IDX())?;
        function.instruction(&Instruction::ArraySet(type_idx));
    } else if name == Symbol::new("array_len") {
        emit_operands(operands, value_locals, function)?;
        function.instruction(&Instruction::ArrayLen);
        set_result_local(db, op, value_locals, function)?;
    } else if name == Symbol::new("ref_null") {
        let heap_type = attr_heap_type(op.attributes(db), ATTR_HEAP_TYPE())?;
        function.instruction(&Instruction::RefNull(heap_type));
        set_result_local(db, op, value_locals, function)?;
    } else if name == Symbol::new("ref_is_null") {
        emit_operands(operands, value_locals, function)?;
        function.instruction(&Instruction::RefIsNull);
        set_result_local(db, op, value_locals, function)?;
    } else if name == Symbol::new("ref_cast") {
        emit_operands(operands, value_locals, function)?;
        let heap_type = attr_heap_type(op.attributes(db), ATTR_TARGET_TYPE())?;
        function.instruction(&Instruction::RefCastNullable(heap_type));
        set_result_local(db, op, value_locals, function)?;
    } else if name == Symbol::new("ref_test") {
        emit_operands(operands, value_locals, function)?;
        let heap_type = attr_heap_type(op.attributes(db), ATTR_TARGET_TYPE())?;
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
    operands: &IdVec<Value<'db>>,
    value_locals: &HashMap<Value<'db>, u32>,
    function: &mut Function,
) -> CompilationResult<()> {
    for value in operands.iter() {
        let index = value_locals
            .get(value)
            .ok_or_else(|| CompilationError::invalid_module("operand missing local mapping"))?;
        function.instruction(&Instruction::LocalGet(*index));
    }
    Ok(())
}

fn value_type<'db>(db: &'db dyn salsa::Database, value: Value<'db>) -> Option<Type<'db>> {
    match value.def(db) {
        ValueDef::OpResult(op) => op.results(db).get(value.index(db)).copied(),
        ValueDef::BlockArg(block) => block.args(db).get(value.index(db)).copied(),
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

fn resolve_callee(path: &QualifiedName, func_indices: &HashMap<Symbol, u32>) -> CompilationResult<u32> {
    let name = path.name();
    func_indices
        .get(&name)
        .copied()
        .ok_or_else(|| CompilationError::function_not_found(&name.to_string()))
}

fn type_to_valtype<'db>(db: &'db dyn salsa::Database, ty: Type<'db>) -> CompilationResult<ValType> {
    if core::I32::from_type(db, ty).is_some() {
        Ok(ValType::I32)
    } else if core::I64::from_type(db, ty).is_some() {
        Ok(ValType::I64)
    } else if core::F32::from_type(db, ty).is_some() {
        Ok(ValType::F32)
    } else if core::F64::from_type(db, ty).is_some() {
        Ok(ValType::F64)
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
    #[cfg(feature = "wasmtime-tests")]
    use std::process::Command;
    #[cfg(feature = "wasmtime-tests")]
    use tempfile::NamedTempFile;
    use trunk_ir::dialect::core;
    use trunk_ir::dialect::func;
    use trunk_ir::dialect::wasm;
    use trunk_ir::{
        Attribute, BlockBuilder, DialectType, Location, PathId, Region, Span, idvec,
    };

    #[salsa::tracked]
    fn build_basic_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let path = PathId::new(db, "file:///test.trb".to_owned());
        let location = Location::new(path, Span::new(0, 0));
        let i32_ty = core::I32::new(db).as_type();

        let mut block = BlockBuilder::new(db, location);
        let const_op = block.op(wasm::i32_const(
            db,
            location,
            i32_ty,
            Attribute::IntBits(42),
        ));
        block.op(wasm::r#return(db, location, vec![const_op.result(db)]));
        let block = block.build();
        let body = Region::new(db, location, idvec![block]);

        let func_ty = core::Func::new(db, idvec![], i32_ty).as_type();
        let func_op = func::func(db, location, Symbol::new("main"), func_ty, body);
        let mut top_builder = BlockBuilder::new(db, location);
        top_builder.op(func_op);
        let top_block = top_builder.build();
        let module_region = Region::new(db, location, idvec![top_block]);
        core::Module::create(db, location, Symbol::new("main"), module_region)
    }

    #[salsa::tracked]
    fn build_mul_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let path = PathId::new(db, "file:///mul.trb".to_owned());
        let location = Location::new(path, Span::new(0, 0));
        let i32_ty = core::I32::new(db).as_type();

        let mut block = BlockBuilder::new(db, location);
        let left = block.op(wasm::i32_const(db, location, i32_ty, Attribute::IntBits(6)));
        let right = block.op(wasm::i32_const(db, location, i32_ty, Attribute::IntBits(7)));
        let mul = block.op(wasm::i32_mul(
            db,
            location,
            left.result(db),
            right.result(db),
            i32_ty,
        ));
        block.op(wasm::r#return(db, location, vec![mul.result(db)]));
        let block = block.build();
        let body = Region::new(db, location, idvec![block]);

        let func_ty = core::Func::new(db, idvec![], i32_ty).as_type();
        let func_op = func::func(db, location, Symbol::new("main"), func_ty, body);

        let mut top_builder = BlockBuilder::new(db, location);
        top_builder.op(func_op);
        let top_block = top_builder.build();
        let module_region = Region::new(db, location, idvec![top_block]);
        core::Module::create(db, location, Symbol::new("main"), module_region)
    }

    #[salsa::tracked]
    fn build_call_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let path = PathId::new(db, "file:///call.trb".to_owned());
        let location = Location::new(path, Span::new(0, 0));
        let i32_ty = core::I32::new(db).as_type();

        let mut add_block_builder = BlockBuilder::new(db, location).args(idvec![i32_ty, i32_ty]);
        let add_lhs =
            add_block_builder.op(wasm::local_get(db, location, i32_ty, Attribute::IntBits(0)));
        let add_rhs =
            add_block_builder.op(wasm::local_get(db, location, i32_ty, Attribute::IntBits(1)));
        let add = add_block_builder.op(wasm::i32_add(
            db,
            location,
            add_lhs.result(db),
            add_rhs.result(db),
            i32_ty,
        ));
        add_block_builder.op(wasm::r#return(db, location, vec![add.result(db)]));
        let add_block = add_block_builder.build();
        let add_body = Region::new(db, location, idvec![add_block]);
        let add_ty = core::Func::new(db, idvec![i32_ty, i32_ty], i32_ty).as_type();
        let add_func = func::func(db, location, Symbol::new("add"), add_ty, add_body);

        let mut main_block_builder = BlockBuilder::new(db, location);
        let left = main_block_builder.op(wasm::i32_const(
            db,
            location,
            i32_ty,
            Attribute::IntBits(40),
        ));
        let right =
            main_block_builder.op(wasm::i32_const(db, location, i32_ty, Attribute::IntBits(2)));
        let callee = QualifiedName::simple(Symbol::new("add"));
        let call = main_block_builder.op(wasm::call(
            db,
            location,
            vec![left.result(db), right.result(db)],
            vec![i32_ty],
            Attribute::QualifiedName(callee),
        ));
        main_block_builder.op(wasm::r#return(db, location, call.result(db)));
        let main_block = main_block_builder.build();
        let main_body = Region::new(db, location, idvec![main_block]);
        let main_ty = core::Func::new(db, idvec![], i32_ty).as_type();
        let main_func = func::func(db, location, Symbol::new("main"), main_ty, main_body);

        let mut top_builder = BlockBuilder::new(db, location);
        top_builder.op(add_func);
        top_builder.op(main_func);
        let top_block = top_builder.build();
        let module_region = Region::new(db, location, idvec![top_block]);
        core::Module::create(db, location, Symbol::new("main"), module_region)
    }

    #[salsa::tracked]
    fn build_local_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let path = PathId::new(db, "file:///locals.trb".to_owned());
        let location = Location::new(path, Span::new(0, 0));
        let i32_ty = core::I32::new(db).as_type();

        let mut block = BlockBuilder::new(db, location);
        let c0 = block.op(wasm::i32_const(
            db,
            location,
            i32_ty,
            Attribute::IntBits(41),
        ));
        let c1 = block.op(wasm::i32_const(db, location, i32_ty, Attribute::IntBits(1)));
        let add = block.op(wasm::i32_add(
            db,
            location,
            c0.result(db),
            c1.result(db),
            i32_ty,
        ));
        block.op(wasm::local_set(
            db,
            location,
            add.result(db),
            Attribute::IntBits(0),
        ));
        let get_local = block.op(wasm::local_get(db, location, i32_ty, Attribute::IntBits(0)));
        block.op(wasm::r#return(db, location, vec![get_local.result(db)]));
        let block = block.build();
        let body = Region::new(db, location, idvec![block]);

        let func_ty = core::Func::new(db, idvec![], i32_ty).as_type();
        let func_op = func::func(db, location, Symbol::new("main"), func_ty, body);

        let mut top_builder = BlockBuilder::new(db, location);
        top_builder.op(func_op);
        let top_block = top_builder.build();
        let module_region = Region::new(db, location, idvec![top_block]);
        core::Module::create(db, location, Symbol::new("main"), module_region)
    }

    #[salsa::tracked]
    fn build_i64_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let path = PathId::new(db, "file:///i64.trb".to_owned());
        let location = Location::new(path, Span::new(0, 0));
        let i64_ty = core::I64::new(db).as_type();

        let mut block = BlockBuilder::new(db, location);
        let left = block.op(wasm::i64_const(
            db,
            location,
            i64_ty,
            Attribute::IntBits(40),
        ));
        let right = block.op(wasm::i64_const(db, location, i64_ty, Attribute::IntBits(2)));
        let add = block.op(wasm::i64_add(
            db,
            location,
            left.result(db),
            right.result(db),
            i64_ty,
        ));
        block.op(wasm::r#return(db, location, vec![add.result(db)]));
        let block = block.build();
        let body = Region::new(db, location, idvec![block]);

        let func_ty = core::Func::new(db, idvec![], i64_ty).as_type();
        let func_op = func::func(db, location, Symbol::new("main"), func_ty, body);

        let mut top_builder = BlockBuilder::new(db, location);
        top_builder.op(func_op);
        let top_block = top_builder.build();
        let module_region = Region::new(db, location, idvec![top_block]);
        core::Module::create(db, location, Symbol::new("main"), module_region)
    }

    #[salsa::tracked]
    fn build_i32_cmp_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let path = PathId::new(db, "file:///cmp.trb".to_owned());
        let location = Location::new(path, Span::new(0, 0));
        let i32_ty = core::I32::new(db).as_type();

        let mut block = BlockBuilder::new(db, location);
        let left = block.op(wasm::i32_const(
            db,
            location,
            i32_ty,
            Attribute::IntBits(40),
        ));
        let right = block.op(wasm::i32_const(
            db,
            location,
            i32_ty,
            Attribute::IntBits(40),
        ));
        let cmp = block.op(wasm::i32_eq(
            db,
            location,
            left.result(db),
            right.result(db),
            i32_ty,
        ));
        block.op(wasm::r#return(db, location, vec![cmp.result(db)]));
        let block = block.build();
        let body = Region::new(db, location, idvec![block]);

        let func_ty = core::Func::new(db, idvec![], i32_ty).as_type();
        let func_op = func::func(db, location, Symbol::new("main"), func_ty, body);

        let mut top_builder = BlockBuilder::new(db, location);
        top_builder.op(func_op);
        let top_block = top_builder.build();
        let module_region = Region::new(db, location, idvec![top_block]);
        core::Module::create(db, location, Symbol::new("main"), module_region)
    }

    #[salsa::tracked]
    fn build_if_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let path = PathId::new(db, "file:///if.trb".to_owned());
        let location = Location::new(path, Span::new(0, 0));
        let i32_ty = core::I32::new(db).as_type();

        let mut block = BlockBuilder::new(db, location);
        let cond = block.op(wasm::i32_const(db, location, i32_ty, Attribute::IntBits(1)));

        let mut then_block = BlockBuilder::new(db, location);
        then_block.op(wasm::i32_const(
            db,
            location,
            i32_ty,
            Attribute::IntBits(42),
        ));
        let then_region = Region::new(db, location, idvec![then_block.build()]);

        let mut else_block = BlockBuilder::new(db, location);
        else_block.op(wasm::i32_const(db, location, i32_ty, Attribute::IntBits(0)));
        let else_region = Region::new(db, location, idvec![else_block.build()]);

        let if_op = block.op(wasm::r#if(
            db,
            location,
            cond.result(db),
            i32_ty,
            then_region,
            else_region,
        ));
        block.op(wasm::r#return(db, location, vec![if_op.result(db)]));
        let block = block.build();
        let body = Region::new(db, location, idvec![block]);

        let func_ty = core::Func::new(db, idvec![], i32_ty).as_type();
        let func_op = func::func(db, location, Symbol::new("main"), func_ty, body);

        let mut top_builder = BlockBuilder::new(db, location);
        top_builder.op(func_op);
        let top_block = top_builder.build();
        let module_region = Region::new(db, location, idvec![top_block]);
        core::Module::create(db, location, Symbol::new("main"), module_region)
    }

    #[salsa::tracked]
    fn build_loop_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let path = PathId::new(db, "file:///loop.trb".to_owned());
        let location = Location::new(path, Span::new(0, 0));
        let i32_ty = core::I32::new(db).as_type();
        let nil_ty = core::Nil::new(db).as_type();

        let mut loop_body = BlockBuilder::new(db, location);
        let cond = loop_body.op(wasm::i32_const(db, location, i32_ty, Attribute::IntBits(0)));
        loop_body.op(wasm::br_if(
            db,
            location,
            cond.result(db),
            Attribute::IntBits(0),
        ));
        let loop_region = Region::new(db, location, idvec![loop_body.build()]);

        let mut block = BlockBuilder::new(db, location);
        block.op(wasm::r#loop(
            db,
            location,
            nil_ty,
            Attribute::Unit,
            loop_region,
        ));
        let result = block.op(wasm::i32_const(
            db,
            location,
            i32_ty,
            Attribute::IntBits(42),
        ));
        block.op(wasm::r#return(db, location, vec![result.result(db)]));
        let block = block.build();
        let body = Region::new(db, location, idvec![block]);

        let func_ty = core::Func::new(db, idvec![], i32_ty).as_type();
        let func_op = func::func(db, location, Symbol::new("main"), func_ty, body);

        let mut top_builder = BlockBuilder::new(db, location);
        top_builder.op(func_op);
        let top_block = top_builder.build();
        let module_region = Region::new(db, location, idvec![top_block]);
        core::Module::create(db, location, Symbol::new("main"), module_region)
    }

    #[salsa::tracked]
    fn build_block_result_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let path = PathId::new(db, "file:///block-result.trb".to_owned());
        let location = Location::new(path, Span::new(0, 0));
        let i32_ty = core::I32::new(db).as_type();

        let mut inner = BlockBuilder::new(db, location);
        inner.op(wasm::i32_const(
            db,
            location,
            i32_ty,
            Attribute::IntBits(42),
        ));
        let inner_region = Region::new(db, location, idvec![inner.build()]);

        let mut block = BlockBuilder::new(db, location);
        let block_op = block.op(wasm::block(
            db,
            location,
            i32_ty,
            Attribute::Unit,
            inner_region,
        ));
        block.op(wasm::r#return(db, location, vec![block_op.result(db)]));
        let block = block.build();
        let body = Region::new(db, location, idvec![block]);

        let func_ty = core::Func::new(db, idvec![], i32_ty).as_type();
        let func_op = func::func(db, location, Symbol::new("main"), func_ty, body);

        let mut top_builder = BlockBuilder::new(db, location);
        top_builder.op(func_op);
        let top_block = top_builder.build();
        let module_region = Region::new(db, location, idvec![top_block]);
        core::Module::create(db, location, Symbol::new("main"), module_region)
    }

    #[salsa::tracked]
    fn build_loop_result_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let path = PathId::new(db, "file:///loop-result.trb".to_owned());
        let location = Location::new(path, Span::new(0, 0));
        let i32_ty = core::I32::new(db).as_type();

        let mut loop_body = BlockBuilder::new(db, location);
        loop_body.op(wasm::i32_const(
            db,
            location,
            i32_ty,
            Attribute::IntBits(42),
        ));
        let loop_region = Region::new(db, location, idvec![loop_body.build()]);

        let mut block = BlockBuilder::new(db, location);
        let loop_op = block.op(wasm::r#loop(
            db,
            location,
            i32_ty,
            Attribute::Unit,
            loop_region,
        ));
        block.op(wasm::r#return(db, location, vec![loop_op.result(db)]));
        let block = block.build();
        let body = Region::new(db, location, idvec![block]);

        let func_ty = core::Func::new(db, idvec![], i32_ty).as_type();
        let func_op = func::func(db, location, Symbol::new("main"), func_ty, body);

        let mut top_builder = BlockBuilder::new(db, location);
        top_builder.op(func_op);
        let top_block = top_builder.build();
        let module_region = Region::new(db, location, idvec![top_block]);
        core::Module::create(db, location, Symbol::new("main"), module_region)
    }

    #[salsa::tracked]
    fn build_br_block_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let path = PathId::new(db, "file:///br-block.trb".to_owned());
        let location = Location::new(path, Span::new(0, 0));
        let i32_ty = core::I32::new(db).as_type();
        let nil_ty = core::Nil::new(db).as_type();

        let mut inner = BlockBuilder::new(db, location);
        inner.op(wasm::br(db, location, Attribute::IntBits(1)));
        inner.op(wasm::i32_const(db, location, i32_ty, Attribute::IntBits(0)));
        let inner_region = Region::new(db, location, idvec![inner.build()]);

        let mut outer = BlockBuilder::new(db, location);
        outer.op(wasm::block(
            db,
            location,
            nil_ty,
            Attribute::Unit,
            inner_region,
        ));
        let outer_region = Region::new(db, location, idvec![outer.build()]);

        let mut block = BlockBuilder::new(db, location);
        block.op(wasm::block(
            db,
            location,
            nil_ty,
            Attribute::Unit,
            outer_region,
        ));
        let result = block.op(wasm::i32_const(
            db,
            location,
            i32_ty,
            Attribute::IntBits(42),
        ));
        block.op(wasm::r#return(db, location, vec![result.result(db)]));
        let block = block.build();
        let body = Region::new(db, location, idvec![block]);

        let func_ty = core::Func::new(db, idvec![], i32_ty).as_type();
        let func_op = func::func(db, location, Symbol::new("main"), func_ty, body);

        let mut top_builder = BlockBuilder::new(db, location);
        top_builder.op(func_op);
        let top_block = top_builder.build();
        let module_region = Region::new(db, location, idvec![top_block]);
        core::Module::create(db, location, Symbol::new("main"), module_region)
    }

    #[salsa_test]
    fn emits_basic_wasm_module(db: &salsa::DatabaseImpl) {
        let module = build_basic_module(db);

        let bytes = emit_wasm(db, module).expect("emit wasm");
        assert!(bytes.len() >= 8, "wasm header should be present");
        assert_eq!(&bytes[0..4], b"\0asm");
        assert_eq!(&bytes[4..8], &[0x01, 0x00, 0x00, 0x00]);
    }

    #[cfg(feature = "wasmtime-tests")]
    #[salsa_test]
    fn runs_in_wasmtime(db: &salsa::DatabaseImpl) {
        let module = build_basic_module(db);
        assert_wasmtime_result(db, &module, "42");
    }

    #[cfg(feature = "wasmtime-tests")]
    #[salsa_test]
    fn runs_mul_in_wasmtime(db: &salsa::DatabaseImpl) {
        let module = build_mul_module(db);
        assert_wasmtime_result(db, &module, "42");
    }

    #[cfg(feature = "wasmtime-tests")]
    #[salsa_test]
    fn runs_call_in_wasmtime(db: &salsa::DatabaseImpl) {
        let module = build_call_module(db);
        assert_wasmtime_result(db, &module, "42");
    }

    #[cfg(feature = "wasmtime-tests")]
    #[salsa_test]
    fn runs_locals_in_wasmtime(db: &salsa::DatabaseImpl) {
        let module = build_local_module(db);
        assert_wasmtime_result(db, &module, "42");
    }

    #[cfg(feature = "wasmtime-tests")]
    #[salsa_test]
    fn runs_i64_in_wasmtime(db: &salsa::DatabaseImpl) {
        let module = build_i64_module(db);
        assert_wasmtime_result(db, &module, "42");
    }

    #[cfg(feature = "wasmtime-tests")]
    #[salsa_test]
    fn runs_cmp_in_wasmtime(db: &salsa::DatabaseImpl) {
        let module = build_i32_cmp_module(db);
        assert_wasmtime_result(db, &module, "1");
    }

    #[cfg(feature = "wasmtime-tests")]
    #[salsa_test]
    fn runs_if_in_wasmtime(db: &salsa::DatabaseImpl) {
        let module = build_if_module(db);
        assert_wasmtime_result(db, &module, "42");
    }

    #[cfg(feature = "wasmtime-tests")]
    #[salsa_test]
    fn runs_block_result_in_wasmtime(db: &salsa::DatabaseImpl) {
        let module = build_block_result_module(db);
        assert_wasmtime_result(db, &module, "42");
    }

    #[cfg(feature = "wasmtime-tests")]
    #[salsa_test]
    fn runs_loop_result_in_wasmtime(db: &salsa::DatabaseImpl) {
        let module = build_loop_result_module(db);
        assert_wasmtime_result(db, &module, "42");
    }

    #[cfg(feature = "wasmtime-tests")]
    #[salsa_test]
    fn runs_loop_in_wasmtime(db: &salsa::DatabaseImpl) {
        let module = build_loop_module(db);
        assert_wasmtime_result(db, &module, "42");
    }

    #[cfg(feature = "wasmtime-tests")]
    #[salsa_test]
    fn runs_br_block_in_wasmtime(db: &salsa::DatabaseImpl) {
        let module = build_br_block_module(db);
        assert_wasmtime_result(db, &module, "42");
    }

    #[cfg(feature = "wasmtime-tests")]
    fn assert_wasmtime_result(db: &salsa::DatabaseImpl, module: &core::Module<'_>, expected: &str) {
        let bytes = emit_wasm(db, module).expect("emit wasm");

        let mut temp = NamedTempFile::new().expect("tempfile");
        std::io::Write::write_all(&mut temp, &bytes).expect("write wasm");
        let path = temp.into_temp_path();

        let wasmtime = std::env::var("TRIBUTE_WASMTIME").unwrap_or_else(|_| "wasmtime".to_string());
        let output = Command::new(wasmtime)
            .arg("run")
            .arg("-C")
            .arg("cache=n")
            .arg("--invoke")
            .arg("main")
            .arg(path.as_os_str())
            .output()
            .expect("run wasmtime");

        assert!(
            output.status.success(),
            "wasmtime failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );

        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains(expected),
            "expected output to contain {expected}, got: {stdout}"
        );
    }

    #[cfg(feature = "wasmtime-tests")]
    #[salsa_test]
    fn runs_wasi_hello_world(db: &salsa::DatabaseImpl) {
        let module = build_wasi_hello_module(db);
        let bytes = emit_wasm(db, &module).expect("emit wasm");
        let mut temp = NamedTempFile::new().expect("tempfile");
        std::io::Write::write_all(&mut temp, &bytes).expect("write wasm");
        let path = temp.into_temp_path();

        let wasmtime = std::env::var("TRIBUTE_WASMTIME").unwrap_or_else(|_| "wasmtime".to_string());
        let output = Command::new(wasmtime)
            .arg("run")
            .arg("-C")
            .arg("cache=n")
            .arg(path.as_os_str())
            .output()
            .expect("run wasmtime");

        assert!(
            output.status.success(),
            "wasmtime failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );

        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains("hello"),
            "expected output to contain hello, got: {stdout}"
        );
    }

    #[cfg(feature = "wasmtime-tests")]
    #[salsa::tracked]
    fn build_wasi_hello_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let path = PathId::new(db, "file:///wasi.trb".to_owned());
        let location = Location::new(path, Span::new(0, 0));
        let i32_ty = core::I32::new(db).as_type();

        let message = b"hello\n";
        let mut iovec = Vec::new();
        iovec.extend_from_slice(&8u32.to_le_bytes());
        iovec.extend_from_slice(&(message.len() as u32).to_le_bytes());

        let import_ty =
            core::Func::new(db, idvec![i32_ty, i32_ty, i32_ty, i32_ty], i32_ty).as_type();
        let import_op = wasm::import_func(
            db,
            location,
            Attribute::String("wasi_snapshot_preview1".into()),
            Attribute::String("fd_write".into()),
            Attribute::Symbol(Symbol::new("fd_write")),
            Attribute::Type(import_ty),
        );

        let memory_op = wasm::memory(
            db,
            location,
            Attribute::IntBits(1),
            Attribute::Unit,
            Attribute::Bool(false),
            Attribute::Bool(false),
        );

        let data_iovec = wasm::data(db, location, Attribute::IntBits(0), Attribute::Bytes(iovec));
        let data_msg = wasm::data(
            db,
            location,
            Attribute::IntBits(8),
            Attribute::Bytes(message.to_vec()),
        );

        let export_memory = wasm::export_memory(
            db,
            location,
            Attribute::String("memory".into()),
            Attribute::IntBits(0),
        );

        let c_fd = wasm::i32_const(db, location, i32_ty, Attribute::IntBits(1));
        let c_iovec_ptr = wasm::i32_const(db, location, i32_ty, Attribute::IntBits(0));
        let c_iovec_len = wasm::i32_const(db, location, i32_ty, Attribute::IntBits(1));
        let c_nwritten = wasm::i32_const(db, location, i32_ty, Attribute::IntBits(16));
        let callee = QualifiedName::simple(Symbol::new("fd_write"));
        let call = wasm::call(
            db,
            location,
            vec![
                c_fd.result(db),
                c_iovec_ptr.result(db),
                c_iovec_len.result(db),
                c_nwritten.result(db),
            ],
            vec![i32_ty],
            Attribute::QualifiedName(callee),
        );
        let drop = wasm::drop(db, location, call.result(db)[0]);
        let ret = wasm::r#return(db, location, Vec::new());

        let mut start_block_builder = BlockBuilder::new(db, location);
        start_block_builder.op(c_fd);
        start_block_builder.op(c_iovec_ptr);
        start_block_builder.op(c_iovec_len);
        start_block_builder.op(c_nwritten);
        start_block_builder.op(call);
        start_block_builder.op(drop);
        start_block_builder.op(ret);
        let start_block = start_block_builder.build();
        let start_body = Region::new(db, location, idvec![start_block]);
        let start_ty = core::Func::new(db, idvec![], core::Nil::new(db).as_type()).as_type();
        let start_func = func::func(db, location, Symbol::new("_start"), start_ty, start_body);

        let export_start = wasm::export_func(
            db,
            location,
            Attribute::String("_start".into()),
            Attribute::QualifiedName(QualifiedName::simple(Symbol::new("_start"))),
        );

        let mut top_builder = BlockBuilder::new(db, location);
        top_builder.op(import_op);
        top_builder.op(memory_op);
        top_builder.op(data_iovec);
        top_builder.op(data_msg);
        top_builder.op(export_memory);
        top_builder.op(start_func);
        top_builder.op(export_start);
        let top_block = top_builder.build();
        let module_region = Region::new(db, location, idvec![top_block]);
        core::Module::create(db, location, Symbol::new("main"), module_region)
    }
}
