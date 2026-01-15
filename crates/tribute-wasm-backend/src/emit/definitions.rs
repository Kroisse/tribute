//! Definition extraction from wasm dialect operations.
//!
//! This module extracts WebAssembly module definitions (functions, imports,
//! exports, memory, data, tables, elements, globals) from typed wasm dialect
//! operations.

use tracing::debug;

use trunk_ir::dialect::{core, func, wasm};
use trunk_ir::{Attribute, DialectOp, DialectType, Operation, Symbol};
use wasm_encoder::{ExportKind, RefType, ValType};

use crate::{CompilationError, CompilationResult};

trunk_ir::symbols! {
    ATTR_BYTES => "bytes",
}

// ============================================================================
// Definition structs
// ============================================================================

#[derive(Debug)]
pub(crate) struct FunctionDef<'db> {
    pub name: Symbol,
    pub ty: core::Func<'db>,
    pub op: Operation<'db>,
}

#[derive(Debug)]
pub(crate) struct ImportFuncDef<'db> {
    pub sym: Symbol,
    pub module: Symbol,
    pub name: Symbol,
    pub ty: core::Func<'db>,
}

#[derive(Debug)]
pub(crate) struct ExportDef {
    pub name: String,
    pub kind: ExportKind,
    pub target: ExportTarget,
}

#[derive(Debug)]
pub(crate) enum ExportTarget {
    Func(Symbol),
    Memory(u32),
}

#[derive(Debug)]
pub(crate) struct MemoryDef {
    pub min: u32,
    pub max: Option<u32>,
    pub shared: bool,
    pub memory64: bool,
}

#[derive(Debug)]
pub(crate) struct DataDef {
    pub offset: i32,
    pub bytes: Vec<u8>,
    pub passive: bool,
}

#[derive(Debug)]
pub(crate) struct TableDef {
    pub reftype: RefType,
    pub min: u32,
    pub max: Option<u32>,
}

#[derive(Debug)]
pub(crate) struct ElementDef {
    pub table: u32,
    pub offset: i32,
    pub funcs: Vec<Symbol>,
}

#[derive(Debug)]
pub(crate) struct GlobalDef {
    pub valtype: ValType,
    pub mutable: bool,
    pub init: i64,
}

// ============================================================================
// Extraction functions
// ============================================================================

pub(crate) fn extract_function_def<'db>(
    db: &'db dyn salsa::Database,
    func_op: wasm::Func<'db>,
) -> CompilationResult<FunctionDef<'db>> {
    let name = func_op.sym_name(db);
    let ty = func_op.r#type(db);

    let func_ty = core::Func::from_type(db, ty)
        .ok_or_else(|| CompilationError::type_error("wasm.func requires core.func type"))?;

    let result_ty = func_ty.result(db);
    debug!(
        "extract_function_def: {} fn_params={:?}, result={}.{}",
        name,
        func_ty
            .params(db)
            .iter()
            .map(|p| format!("{}.{}", p.dialect(db), p.name(db)))
            .collect::<Vec<_>>(),
        result_ty.dialect(db),
        result_ty.name(db),
    );

    Ok(FunctionDef {
        name,
        ty: func_ty,
        op: func_op.as_operation(),
    })
}

pub(crate) fn extract_import_def<'db>(
    db: &'db dyn salsa::Database,
    import_op: wasm::ImportFunc<'db>,
) -> CompilationResult<ImportFuncDef<'db>> {
    let module = import_op.module(db);
    let name = import_op.name(db);
    let sym = import_op.sym_name(db);
    let ty = import_op.r#type(db);

    let func_ty = core::Func::from_type(db, ty)
        .ok_or_else(|| CompilationError::type_error("wasm.import_func requires core.func type"))?;

    Ok(ImportFuncDef {
        sym,
        module,
        name,
        ty: func_ty,
    })
}

pub(crate) fn extract_export_func<'db>(
    db: &'db dyn salsa::Database,
    export_op: wasm::ExportFunc<'db>,
) -> CompilationResult<ExportDef> {
    let name = export_op.name(db);
    let func = export_op.func(db);
    Ok(ExportDef {
        name,
        kind: ExportKind::Func,
        target: ExportTarget::Func(func),
    })
}

pub(crate) fn extract_export_memory<'db>(
    db: &'db dyn salsa::Database,
    export_op: wasm::ExportMemory<'db>,
) -> CompilationResult<ExportDef> {
    let name = export_op.name(db);
    let index = export_op.index(db);
    Ok(ExportDef {
        name,
        kind: ExportKind::Memory,
        target: ExportTarget::Memory(index),
    })
}

pub(crate) fn extract_memory_def<'db>(
    db: &'db dyn salsa::Database,
    memory_op: wasm::Memory<'db>,
) -> CompilationResult<MemoryDef> {
    let min = memory_op.min(db);
    let max = memory_op.max(db);
    let shared = memory_op.shared(db);
    let memory64 = memory_op.memory64(db);
    Ok(MemoryDef {
        min,
        max: if max == 0 { None } else { Some(max) },
        shared,
        memory64,
    })
}

pub(crate) fn extract_data_def<'db>(
    db: &'db dyn salsa::Database,
    data_op: wasm::Data<'db>,
) -> CompilationResult<DataDef> {
    let passive = data_op.passive(db);
    let offset = if passive {
        0 // Passive segments don't have an offset
    } else {
        data_op.offset(db)
    };
    // bytes is typed as `any` in the dialect, so we access the raw attribute
    let attrs = data_op.as_operation().attributes(db);
    let bytes = match attrs.get(&ATTR_BYTES()) {
        Some(Attribute::Bytes(value)) => value.clone(),
        _ => {
            return Err(CompilationError::invalid_attribute(
                "missing or invalid 'bytes' attribute on wasm.data",
            ));
        }
    };
    Ok(DataDef {
        offset,
        bytes,
        passive,
    })
}

pub(crate) fn extract_table_def<'db>(
    db: &'db dyn salsa::Database,
    table_op: wasm::Table<'db>,
) -> CompilationResult<TableDef> {
    let reftype_sym = table_op.reftype(db);
    let reftype = reftype_sym.with_str(|s| match s {
        "funcref" => Ok(RefType::FUNCREF),
        "externref" => Ok(RefType::EXTERNREF),
        other => Err(CompilationError::invalid_attribute(format!(
            "reftype: {}",
            other
        ))),
    })?;
    let min = table_op.min(db);
    let max = table_op.max(db);
    Ok(TableDef { reftype, min, max })
}

pub(crate) fn extract_element_def<'db>(
    db: &'db dyn salsa::Database,
    elem_op: wasm::Elem<'db>,
) -> CompilationResult<ElementDef> {
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

pub(crate) fn extract_global_def<'db>(
    db: &'db dyn salsa::Database,
    global_op: wasm::Global<'db>,
) -> CompilationResult<GlobalDef> {
    let valtype_sym = global_op.valtype(db);
    let valtype = valtype_sym.with_str(|s| match s {
        "i32" => Ok(ValType::I32),
        "i64" => Ok(ValType::I64),
        "f32" => Ok(ValType::F32),
        "f64" => Ok(ValType::F64),
        "funcref" => Ok(ValType::Ref(RefType::FUNCREF)),
        "externref" => Ok(ValType::Ref(RefType::EXTERNREF)),
        "anyref" => Ok(ValType::Ref(RefType::ANYREF)),
        other => Err(CompilationError::invalid_attribute(format!(
            "valtype: {}",
            other
        ))),
    })?;
    let mutable = global_op.mutable(db);
    let init = global_op.init(db);
    Ok(GlobalDef {
        valtype,
        mutable,
        init,
    })
}
