//! Wasm backend that emits WebAssembly binaries from wasm.* TrunkIR operations.

use std::collections::HashMap;

use trunk_ir::dialect::core;
use trunk_ir::{Attribute, Attrs, DialectType, IdVec, Operation, Symbol, SymbolVec, Type, Value};
use wasm_encoder::{
    CodeSection, ConstExpr, DataSection, EntityType, ExportKind, ExportSection, Function,
    FunctionSection, ImportSection, Instruction, MemorySection, MemoryType, Module, TypeSection,
    ValType,
};

mod errors;

pub use errors::{CompilationError, CompilationResult};

trunk_ir::symbols! {
    ATTR_SYM_NAME => "sym_name",
    ATTR_TYPE => "type",
    ATTR_CALLEE => "callee",
    ATTR_VALUE => "value",
    ATTR_INDEX => "index",
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
}

pub fn emit_wasm<'db>(
    db: &'db dyn salsa::Database,
    module: &core::Module<'db>,
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
    let mut type_indices = Vec::new();

    for (index, import_def) in module_info.imports.iter().enumerate() {
        func_indices.insert(import_def.sym, index as u32);
    }
    let import_count = module_info.imports.len() as u32;
    for (index, func_def) in module_info.funcs.iter().enumerate() {
        func_indices.insert(func_def.name, import_count + index as u32);
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
        let type_index = type_indices.len() as u32;
        type_indices.push(type_index);
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
        let type_index = type_indices.len() as u32;
        type_indices.push(type_index);
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
    module: &core::Module<'db>,
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
    })
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
    let sym = func
        .last()
        .ok_or_else(|| CompilationError::invalid_module("export func missing symbol"))?;
    Ok(ExportDef {
        name,
        kind: ExportKind::Func,
        target: ExportTarget::Func(*sym),
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
    if blocks.len() != 1 {
        return Err(CompilationError::unsupported_feature(
            "multi-block functions",
        ));
    }

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

    for op in block.operations(db).iter() {
        let result_types = op.results(db);
        if result_types.len() > 1 {
            return Err(CompilationError::unsupported_feature("multi-result ops"));
        }
        if let Some(result_ty) = result_types.first() {
            let val_type = type_to_valtype(db, *result_ty)?;
            let local_index = param_count + locals.len() as u32;
            value_locals.insert(op.result(db, 0), local_index);
            locals.push(val_type);
        }
    }

    let mut function = Function::new(compress_locals(&locals));

    for op in block.operations(db).iter() {
        emit_op(db, op, &value_locals, func_indices, &mut function)?;
    }

    function.instruction(&Instruction::End);

    Ok(function)
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

fn resolve_callee(path: &SymbolVec, func_indices: &HashMap<Symbol, u32>) -> CompilationResult<u32> {
    let name = path
        .last()
        .ok_or_else(|| CompilationError::invalid_module("callee path is empty"))?;
    func_indices
        .get(name)
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
    let is_nil = ty.dialect(db) == core::DIALECT_NAME() && ty.name(db) == Symbol::new("nil");
    if is_nil {
        Ok(Vec::new())
    } else {
        Ok(vec![type_to_valtype(db, ty)?])
    }
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
) -> CompilationResult<&'db SymbolVec> {
    match op.attributes(db).get(&key) {
        Some(Attribute::SymbolRef(path)) => Ok(path),
        _ => Err(CompilationError::from(
            errors::CompilationErrorKind::InvalidAttribute("callee"),
        )),
    }
}

fn attr_symbol_ref_attr<'db>(
    attrs: &'db Attrs<'db>,
    key: Symbol,
) -> CompilationResult<&'db SymbolVec> {
    match attrs.get(&key) {
        Some(Attribute::SymbolRef(path)) => Ok(path),
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
    use trunk_ir::smallvec::smallvec;
    use trunk_ir::{
        Attribute, Block, DialectType, Location, PathId, Region, Span, SymbolVec, idvec,
    };

    #[salsa::tracked]
    fn build_basic_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let path = PathId::new(db, "file:///test.trb".to_owned());
        let location = Location::new(path, Span::new(0, 0));
        let i32_ty = core::I32::new(db).as_type();

        let const_op = Operation::of_name(db, location, "wasm.i32_const")
            .attr("value", Attribute::IntBits(42))
            .result(i32_ty)
            .build();
        let ret_op = Operation::of_name(db, location, "wasm.return")
            .operand(const_op.result(db, 0))
            .build();

        let block = Block::new(db, location, idvec![], idvec![const_op, ret_op]);
        let body = Region::new(db, location, idvec![block]);

        let func_ty = core::Func::new(db, idvec![], i32_ty).as_type();
        let func_op = Operation::of_name(db, location, "func.func")
            .attr("sym_name", Attribute::Symbol(Symbol::new("main")))
            .attr("type", Attribute::Type(func_ty))
            .region(body)
            .build();

        let top_block = Block::new(db, location, idvec![], idvec![func_op]);
        let module_region = Region::new(db, location, idvec![top_block]);
        core::Module::create(db, location, Symbol::new("main"), module_region)
    }

    #[salsa::tracked]
    fn build_mul_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let path = PathId::new(db, "file:///mul.trb".to_owned());
        let location = Location::new(path, Span::new(0, 0));
        let i32_ty = core::I32::new(db).as_type();

        let left = Operation::of_name(db, location, "wasm.i32_const")
            .attr("value", Attribute::IntBits(6))
            .result(i32_ty)
            .build();
        let right = Operation::of_name(db, location, "wasm.i32_const")
            .attr("value", Attribute::IntBits(7))
            .result(i32_ty)
            .build();
        let mul = Operation::of_name(db, location, "wasm.i32_mul")
            .operand(left.result(db, 0))
            .operand(right.result(db, 0))
            .result(i32_ty)
            .build();
        let ret = Operation::of_name(db, location, "wasm.return")
            .operand(mul.result(db, 0))
            .build();

        let block = Block::new(db, location, idvec![], idvec![left, right, mul, ret]);
        let body = Region::new(db, location, idvec![block]);

        let func_ty = core::Func::new(db, idvec![], i32_ty).as_type();
        let func_op = Operation::of_name(db, location, "func.func")
            .attr("sym_name", Attribute::Symbol(Symbol::new("main")))
            .attr("type", Attribute::Type(func_ty))
            .region(body)
            .build();

        let top_block = Block::new(db, location, idvec![], idvec![func_op]);
        let module_region = Region::new(db, location, idvec![top_block]);
        core::Module::create(db, location, Symbol::new("main"), module_region)
    }

    #[salsa::tracked]
    fn build_call_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let path = PathId::new(db, "file:///call.trb".to_owned());
        let location = Location::new(path, Span::new(0, 0));
        let i32_ty = core::I32::new(db).as_type();

        let add_lhs = Operation::of_name(db, location, "wasm.local_get")
            .attr("index", Attribute::IntBits(0))
            .result(i32_ty)
            .build();
        let add_rhs = Operation::of_name(db, location, "wasm.local_get")
            .attr("index", Attribute::IntBits(1))
            .result(i32_ty)
            .build();
        let add = Operation::of_name(db, location, "wasm.i32_add")
            .operand(add_lhs.result(db, 0))
            .operand(add_rhs.result(db, 0))
            .result(i32_ty)
            .build();
        let add_ret = Operation::of_name(db, location, "wasm.return")
            .operand(add.result(db, 0))
            .build();
        let add_block = Block::new(
            db,
            location,
            idvec![i32_ty, i32_ty],
            idvec![add_lhs, add_rhs, add, add_ret],
        );
        let add_body = Region::new(db, location, idvec![add_block]);
        let add_ty = core::Func::new(db, idvec![i32_ty, i32_ty], i32_ty).as_type();
        let add_func = Operation::of_name(db, location, "func.func")
            .attr("sym_name", Attribute::Symbol(Symbol::new("add")))
            .attr("type", Attribute::Type(add_ty))
            .region(add_body)
            .build();

        let left = Operation::of_name(db, location, "wasm.i32_const")
            .attr("value", Attribute::IntBits(40))
            .result(i32_ty)
            .build();
        let right = Operation::of_name(db, location, "wasm.i32_const")
            .attr("value", Attribute::IntBits(2))
            .result(i32_ty)
            .build();
        let callee: SymbolVec = smallvec![Symbol::new("add")];
        let call = Operation::of_name(db, location, "wasm.call")
            .attr("callee", Attribute::SymbolRef(callee))
            .operand(left.result(db, 0))
            .operand(right.result(db, 0))
            .result(i32_ty)
            .build();
        let ret = Operation::of_name(db, location, "wasm.return")
            .operand(call.result(db, 0))
            .build();

        let main_block = Block::new(db, location, idvec![], idvec![left, right, call, ret]);
        let main_body = Region::new(db, location, idvec![main_block]);
        let main_ty = core::Func::new(db, idvec![], i32_ty).as_type();
        let main_func = Operation::of_name(db, location, "func.func")
            .attr("sym_name", Attribute::Symbol(Symbol::new("main")))
            .attr("type", Attribute::Type(main_ty))
            .region(main_body)
            .build();

        let top_block = Block::new(db, location, idvec![], idvec![add_func, main_func]);
        let module_region = Region::new(db, location, idvec![top_block]);
        core::Module::create(db, location, Symbol::new("main"), module_region)
    }

    #[salsa::tracked]
    fn build_local_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let path = PathId::new(db, "file:///locals.trb".to_owned());
        let location = Location::new(path, Span::new(0, 0));
        let i32_ty = core::I32::new(db).as_type();

        let c0 = Operation::of_name(db, location, "wasm.i32_const")
            .attr("value", Attribute::IntBits(41))
            .result(i32_ty)
            .build();
        let c1 = Operation::of_name(db, location, "wasm.i32_const")
            .attr("value", Attribute::IntBits(1))
            .result(i32_ty)
            .build();
        let add = Operation::of_name(db, location, "wasm.i32_add")
            .operand(c0.result(db, 0))
            .operand(c1.result(db, 0))
            .result(i32_ty)
            .build();
        let set_local = Operation::of_name(db, location, "wasm.local_set")
            .attr("index", Attribute::IntBits(0))
            .operand(add.result(db, 0))
            .build();
        let get_local = Operation::of_name(db, location, "wasm.local_get")
            .attr("index", Attribute::IntBits(0))
            .result(i32_ty)
            .build();
        let ret = Operation::of_name(db, location, "wasm.return")
            .operand(get_local.result(db, 0))
            .build();

        let block = Block::new(
            db,
            location,
            idvec![],
            idvec![c0, c1, add, set_local, get_local, ret],
        );
        let body = Region::new(db, location, idvec![block]);

        let func_ty = core::Func::new(db, idvec![], i32_ty).as_type();
        let func_op = Operation::of_name(db, location, "func.func")
            .attr("sym_name", Attribute::Symbol(Symbol::new("main")))
            .attr("type", Attribute::Type(func_ty))
            .region(body)
            .build();

        let top_block = Block::new(db, location, idvec![], idvec![func_op]);
        let module_region = Region::new(db, location, idvec![top_block]);
        core::Module::create(db, location, Symbol::new("main"), module_region)
    }

    #[salsa::tracked]
    fn build_i64_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let path = PathId::new(db, "file:///i64.trb".to_owned());
        let location = Location::new(path, Span::new(0, 0));
        let i64_ty = core::I64::new(db).as_type();

        let left = Operation::of_name(db, location, "wasm.i64_const")
            .attr("value", Attribute::IntBits(40))
            .result(i64_ty)
            .build();
        let right = Operation::of_name(db, location, "wasm.i64_const")
            .attr("value", Attribute::IntBits(2))
            .result(i64_ty)
            .build();
        let add = Operation::of_name(db, location, "wasm.i64_add")
            .operand(left.result(db, 0))
            .operand(right.result(db, 0))
            .result(i64_ty)
            .build();
        let ret = Operation::of_name(db, location, "wasm.return")
            .operand(add.result(db, 0))
            .build();

        let block = Block::new(db, location, idvec![], idvec![left, right, add, ret]);
        let body = Region::new(db, location, idvec![block]);

        let func_ty = core::Func::new(db, idvec![], i64_ty).as_type();
        let func_op = Operation::of_name(db, location, "func.func")
            .attr("sym_name", Attribute::Symbol(Symbol::new("main")))
            .attr("type", Attribute::Type(func_ty))
            .region(body)
            .build();

        let top_block = Block::new(db, location, idvec![], idvec![func_op]);
        let module_region = Region::new(db, location, idvec![top_block]);
        core::Module::create(db, location, Symbol::new("main"), module_region)
    }

    #[salsa::tracked]
    fn build_i32_cmp_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let path = PathId::new(db, "file:///cmp.trb".to_owned());
        let location = Location::new(path, Span::new(0, 0));
        let i32_ty = core::I32::new(db).as_type();

        let left = Operation::of_name(db, location, "wasm.i32_const")
            .attr("value", Attribute::IntBits(40))
            .result(i32_ty)
            .build();
        let right = Operation::of_name(db, location, "wasm.i32_const")
            .attr("value", Attribute::IntBits(40))
            .result(i32_ty)
            .build();
        let cmp = Operation::of_name(db, location, "wasm.i32_eq")
            .operand(left.result(db, 0))
            .operand(right.result(db, 0))
            .result(i32_ty)
            .build();
        let ret = Operation::of_name(db, location, "wasm.return")
            .operand(cmp.result(db, 0))
            .build();

        let block = Block::new(db, location, idvec![], idvec![left, right, cmp, ret]);
        let body = Region::new(db, location, idvec![block]);

        let func_ty = core::Func::new(db, idvec![], i32_ty).as_type();
        let func_op = Operation::of_name(db, location, "func.func")
            .attr("sym_name", Attribute::Symbol(Symbol::new("main")))
            .attr("type", Attribute::Type(func_ty))
            .region(body)
            .build();

        let top_block = Block::new(db, location, idvec![], idvec![func_op]);
        let module_region = Region::new(db, location, idvec![top_block]);
        core::Module::create(db, location, Symbol::new("main"), module_region)
    }

    #[salsa_test]
    fn emits_basic_wasm_module(db: &salsa::DatabaseImpl) {
        let module = build_basic_module(db);

        let bytes = emit_wasm(db, &module).expect("emit wasm");
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
        let import_op = Operation::of_name(db, location, "wasm.import_func")
            .attr("module", Attribute::String("wasi_snapshot_preview1".into()))
            .attr("name", Attribute::String("fd_write".into()))
            .attr("sym_name", Attribute::Symbol(Symbol::new("fd_write")))
            .attr("type", Attribute::Type(import_ty))
            .build();

        let memory_op = Operation::of_name(db, location, "wasm.memory")
            .attr("min", Attribute::IntBits(1))
            .build();

        let data_iovec = Operation::of_name(db, location, "wasm.data")
            .attr("offset", Attribute::IntBits(0))
            .attr("bytes", Attribute::Bytes(iovec))
            .build();
        let data_msg = Operation::of_name(db, location, "wasm.data")
            .attr("offset", Attribute::IntBits(8))
            .attr("bytes", Attribute::Bytes(message.to_vec()))
            .build();

        let export_memory = Operation::of_name(db, location, "wasm.export_memory")
            .attr("name", Attribute::String("memory".into()))
            .attr("index", Attribute::IntBits(0))
            .build();

        let c_fd = Operation::of_name(db, location, "wasm.i32_const")
            .attr("value", Attribute::IntBits(1))
            .result(i32_ty)
            .build();
        let c_iovec_ptr = Operation::of_name(db, location, "wasm.i32_const")
            .attr("value", Attribute::IntBits(0))
            .result(i32_ty)
            .build();
        let c_iovec_len = Operation::of_name(db, location, "wasm.i32_const")
            .attr("value", Attribute::IntBits(1))
            .result(i32_ty)
            .build();
        let c_nwritten = Operation::of_name(db, location, "wasm.i32_const")
            .attr("value", Attribute::IntBits(16))
            .result(i32_ty)
            .build();
        let callee: SymbolVec = smallvec![Symbol::new("fd_write")];
        let call = Operation::of_name(db, location, "wasm.call")
            .attr("callee", Attribute::SymbolRef(callee))
            .operand(c_fd.result(db, 0))
            .operand(c_iovec_ptr.result(db, 0))
            .operand(c_iovec_len.result(db, 0))
            .operand(c_nwritten.result(db, 0))
            .result(i32_ty)
            .build();
        let drop = Operation::of_name(db, location, "wasm.drop")
            .operand(call.result(db, 0))
            .build();
        let ret = Operation::of_name(db, location, "wasm.return").build();

        let start_block = Block::new(
            db,
            location,
            idvec![],
            idvec![c_fd, c_iovec_ptr, c_iovec_len, c_nwritten, call, drop, ret],
        );
        let start_body = Region::new(db, location, idvec![start_block]);
        let start_ty = core::Func::new(db, idvec![], core::Nil::new(db).as_type()).as_type();
        let start_func = Operation::of_name(db, location, "func.func")
            .attr("sym_name", Attribute::Symbol(Symbol::new("_start")))
            .attr("type", Attribute::Type(start_ty))
            .region(start_body)
            .build();

        let export_start = Operation::of_name(db, location, "wasm.export_func")
            .attr("name", Attribute::String("_start".into()))
            .attr(
                "func",
                Attribute::SymbolRef(smallvec![Symbol::new("_start")]),
            )
            .build();

        let top_block = Block::new(
            db,
            location,
            idvec![],
            idvec![
                import_op,
                memory_op,
                data_iovec,
                data_msg,
                export_memory,
                start_func,
                export_start
            ],
        );
        let module_region = Region::new(db, location, idvec![top_block]);
        core::Module::create(db, location, Symbol::new("main"), module_region)
    }
}
