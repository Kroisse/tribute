//! Definition extraction from wasm dialect operations.
//!
//! This module extracts WebAssembly module definitions (functions, imports,
//! exports, memory, data, tables, elements, globals) from typed wasm dialect
//! operations.

use tracing::debug;

use trunk_ir::IrContext;
use trunk_ir::Symbol;
use trunk_ir::dialect::func as arena_func;
use trunk_ir::dialect::wasm as wasm_dialect;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{OpRef, TypeRef};
use trunk_ir::types::Attribute;
use wasm_encoder::{ExportKind, RefType, ValType};

use crate::{CompilationError, CompilationResult};

// ============================================================================
// Definition structs
// ============================================================================

#[derive(Debug)]
pub(crate) struct FunctionDef {
    pub name: Symbol,
    /// core.func TypeRef - params are all but last, last is return type
    pub func_type: TypeRef,
    pub op: OpRef,
}

#[derive(Debug)]
pub(crate) struct ImportFuncDef {
    pub sym: Symbol,
    pub module: Symbol,
    pub name: Symbol,
    /// core.func TypeRef
    pub func_type: TypeRef,
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

pub(crate) fn extract_function_def(
    ctx: &IrContext,
    func_op: wasm_dialect::Func,
) -> CompilationResult<FunctionDef> {
    let name = func_op.sym_name(ctx);
    let ty = func_op.r#type(ctx);

    let ty_data = ctx.types.get(ty);
    if !(ty_data.dialect == Symbol::new("core") && ty_data.name == Symbol::new("func")) {
        return Err(CompilationError::type_error(
            "wasm.func requires core.func type",
        ));
    }

    if !ty_data.params.is_empty() {
        let result_ty = *ty_data.params.last().unwrap();
        let result_data = ctx.types.get(result_ty);
        debug!(
            "extract_function_def: {} fn_params={:?}, result={}.{}",
            name,
            ty_data.params[..ty_data.params.len() - 1]
                .iter()
                .map(|p| {
                    let d = ctx.types.get(*p);
                    format!("{}.{}", d.dialect, d.name)
                })
                .collect::<Vec<_>>(),
            result_data.dialect,
            result_data.name,
        );
    }

    Ok(FunctionDef {
        name,
        func_type: ty,
        op: func_op.op_ref(),
    })
}

pub(crate) fn extract_import_def(
    ctx: &IrContext,
    import_op: wasm_dialect::ImportFunc,
) -> CompilationResult<ImportFuncDef> {
    let module = import_op.module(ctx);
    let name = import_op.name(ctx);
    let sym = import_op.sym_name(ctx);
    let ty = import_op.r#type(ctx);

    let ty_data = ctx.types.get(ty);
    if !(ty_data.dialect == Symbol::new("core") && ty_data.name == Symbol::new("func")) {
        return Err(CompilationError::type_error(
            "wasm.import_func requires core.func type",
        ));
    }

    Ok(ImportFuncDef {
        sym,
        module,
        name,
        func_type: ty,
    })
}

pub(crate) fn extract_export_func(
    ctx: &IrContext,
    export_op: wasm_dialect::ExportFunc,
) -> CompilationResult<ExportDef> {
    let name = export_op.name(ctx);
    let func = export_op.func(ctx);
    Ok(ExportDef {
        name,
        kind: ExportKind::Func,
        target: ExportTarget::Func(func),
    })
}

pub(crate) fn extract_export_memory(
    ctx: &IrContext,
    export_op: wasm_dialect::ExportMemory,
) -> CompilationResult<ExportDef> {
    let name = export_op.name(ctx);
    let index = export_op.index(ctx);
    Ok(ExportDef {
        name,
        kind: ExportKind::Memory,
        target: ExportTarget::Memory(index),
    })
}

pub(crate) fn extract_memory_def(
    ctx: &IrContext,
    memory_op: wasm_dialect::Memory,
) -> CompilationResult<MemoryDef> {
    let min = memory_op.min(ctx);
    let max = memory_op.max(ctx);
    let shared = memory_op.shared(ctx);
    let memory64 = memory_op.memory64(ctx);
    Ok(MemoryDef {
        min,
        max: if max == 0 { None } else { Some(max) },
        shared,
        memory64,
    })
}

pub(crate) fn extract_data_def(
    ctx: &IrContext,
    data_op: wasm_dialect::Data,
) -> CompilationResult<DataDef> {
    let passive = data_op.passive(ctx);
    let offset = if passive { 0 } else { data_op.offset(ctx) };
    // bytes is typed as `any` in the dialect, so we access the raw attribute
    let op_data = ctx.op(data_op.op_ref());
    let bytes = match op_data.attributes.get(&Symbol::new("bytes")) {
        Some(Attribute::Bytes(value)) => value.to_vec(),
        _ => {
            return Err(CompilationError::invalid_attribute(
                "missing or invalid 'bytes' attribute on wasm.data",
            ));
        }
    };
    let offset = i32::try_from(offset)
        .map_err(|_| CompilationError::invalid_module("data segment offset exceeds i32::MAX"))?;
    Ok(DataDef {
        offset,
        bytes,
        passive,
    })
}

pub(crate) fn extract_table_def(
    ctx: &IrContext,
    table_op: wasm_dialect::Table,
) -> CompilationResult<TableDef> {
    let reftype_sym = table_op.reftype(ctx);
    let reftype = reftype_sym.with_str(|s| match s {
        "funcref" => Ok(RefType::FUNCREF),
        "externref" => Ok(RefType::EXTERNREF),
        other => Err(CompilationError::invalid_attribute(format!(
            "reftype: {}",
            other
        ))),
    })?;
    let min = table_op.min(ctx);
    let max = table_op.max(ctx);
    Ok(TableDef { reftype, min, max })
}

pub(crate) fn extract_element_def(
    ctx: &IrContext,
    elem_op: wasm_dialect::Elem,
) -> CompilationResult<ElementDef> {
    let table = elem_op.table(ctx).unwrap_or(0);
    let raw_offset = elem_op.offset(ctx).unwrap_or(0);
    let offset = i32::try_from(raw_offset)
        .map_err(|_| CompilationError::invalid_module("element segment offset exceeds i32::MAX"))?;

    // Collect function references from the funcs region
    let funcs_region = elem_op.funcs(ctx);
    let mut funcs = Vec::new();
    for &block_ref in &ctx.region(funcs_region).blocks {
        for &inner_op in &ctx.block(block_ref).ops {
            // Look for func.constant or wasm.ref_func operations
            if let Ok(const_op) = arena_func::Constant::from_op(ctx, inner_op) {
                funcs.push(const_op.func_ref(ctx));
            } else if let Ok(ref_func_op) = wasm_dialect::RefFunc::from_op(ctx, inner_op) {
                funcs.push(ref_func_op.func_name(ctx));
            }
        }
    }

    Ok(ElementDef {
        table,
        offset,
        funcs,
    })
}

pub(crate) fn extract_global_def(
    ctx: &IrContext,
    global_op: wasm_dialect::Global,
) -> CompilationResult<GlobalDef> {
    let valtype_sym = global_op.valtype(ctx);
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
    let mutable = global_op.mutable(ctx);
    let op_data = ctx.op(global_op.op_ref());
    let init = match op_data.attributes.get(&Symbol::new("init")) {
        Some(Attribute::Int(v)) => i64::try_from(*v).map_err(|_| {
            CompilationError::invalid_attribute(format!("global init value {} out of i64 range", v))
        })?,
        other => {
            debug!(
                "extract_global_def: missing or non-Int 'init' attribute (got {:?}), defaulting to 0",
                other
            );
            0
        }
    };
    Ok(GlobalDef {
        valtype,
        mutable,
        init,
    })
}
