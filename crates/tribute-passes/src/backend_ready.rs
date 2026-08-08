//! Tribute-specific final backend boundaries.
//!
//! Generic backend validators intentionally know nothing about Tribute's
//! logical dialects or CPS ABI. This verifier runs immediately before emission
//! and combines a full operation target with recursive type and convention
//! checks.

use std::collections::HashSet;
use std::ops::ControlFlow;

use tribute_core::{CALLING_CONVENTION_ATTR, CallingConvention, get_calling_convention};
use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::core;
use trunk_ir::ops::DialectType;
use trunk_ir::refs::{OpRef, TypeRef};
use trunk_ir::rewrite::Module;
use trunk_ir::types::Attribute;
use trunk_ir::walk::{WalkAction, walk_op};

use crate::target_abi::{
    ROOT_CPS_CALL_ATTR, ROOT_CPS_WORKER_ATTR, ROOT_DONE_K_ATTR, ROOT_WRAPPER_ATTR, TargetAbiError,
};

const CPS_MAIN_SYMBOL: &str = "__tribute_cps_main";
const ROOT_DONE_K_SYMBOL: &str = "__tribute_root_done_k";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TributeBackend {
    Native,
    Wasm,
}

impl TributeBackend {
    fn boundary(self) -> &'static str {
        match self {
            Self::Native => "tribute-backend-ready-native",
            Self::Wasm => "tribute-backend-ready-wasm",
        }
    }

    fn dialect(self) -> &'static str {
        match self {
            Self::Native => "clif",
            Self::Wasm => "wasm",
        }
    }

    fn function_name(self) -> &'static str {
        "func"
    }

    fn call_name(self) -> &'static str {
        "call"
    }

    fn return_name(self) -> &'static str {
        "return"
    }

    fn tail_names(self) -> [&'static str; 2] {
        ["return_call", "return_call_indirect"]
    }
}

/// Verify the final Tribute backend boundary without mutating the module.
pub fn verify_tribute_backend_ready(
    ctx: &IrContext,
    module: Module,
    backend: TributeBackend,
) -> Result<(), TargetAbiError> {
    let operation_result = match backend {
        TributeBackend::Native => trunk_ir_cranelift_backend::validate_clif_ir(ctx, module)
            .map_err(|error| error.to_string()),
        TributeBackend::Wasm => {
            trunk_ir_wasm_backend::validate_wasm_ir(ctx, module).map_err(|error| error.to_string())
        }
    };
    if let Err(error) = operation_result {
        return Err(TargetAbiError::new(format!(
            "{}: illegal or unknown operation(s): {error}",
            backend.boundary(),
        )));
    }

    let nil = ctx
        .types
        .iter()
        .find_map(|(ty, data)| {
            (data.dialect == Symbol::new("core") && data.name == Symbol::new("nil")).then_some(ty)
        })
        .ok_or_else(|| {
            TargetAbiError::new(format!(
                "{}: canonical core.nil type is not interned",
                backend.boundary()
            ))
        })?;
    let ops = collect_ops(ctx, module.op());
    let mut verifier = RecursiveTypeVerifier::new(ctx, backend);

    for (name, ty) in ctx.type_aliases() {
        if *name == Symbol::new("__tribute_cps_control") {
            return Err(TargetAbiError::new(format!(
                "{}: private CPS control carrier alias remains",
                backend.boundary()
            )));
        }
        verifier.verify_metadata(*ty, "type alias")?;
    }

    for &op in &ops {
        for &ty in ctx.op_result_types(op) {
            verifier
                .verify_result(ty, "operation result")
                .map_err(|error| {
                    TargetAbiError::new(format!(
                        "{error} (at operation {}.{})",
                        ctx.op(op).dialect,
                        ctx.op(op).name
                    ))
                })?;
        }
        for &operand in ctx.op_operands(op) {
            verifier
                .verify_value(ctx.value_ty(operand), "operation operand")
                .map_err(|error| {
                    TargetAbiError::new(format!(
                        "{error} (at operation {}.{})",
                        ctx.op(op).dialect,
                        ctx.op(op).name
                    ))
                })?;
        }
        for (key, attribute) in &ctx.op(op).attributes {
            verifier
                .verify_op_attribute(op, *key, attribute)
                .map_err(|error| {
                    TargetAbiError::new(format!(
                        "{error} (at operation {}.{})",
                        ctx.op(op).dialect,
                        ctx.op(op).name
                    ))
                })?;
        }
        for &region in &ctx.op(op).regions {
            for &block in &ctx.region(region).blocks {
                for arg in &ctx.block(block).args {
                    verifier.verify_value(arg.ty, "block argument")?;
                    for attribute in arg.attrs.values() {
                        verifier.verify_attribute(attribute, "block argument attribute")?;
                    }
                }
            }
        }
    }

    let mut root_workers = Vec::new();
    let mut root_wrappers = Vec::new();
    let mut root_done_functions = Vec::new();
    let mut root_calls = Vec::new();
    let mut reserved_worker_without_marker = false;
    let mut reserved_done_without_marker = false;
    let mut calls_to_reserved_worker = Vec::new();
    for &op in &ops {
        let data = ctx.op(op);
        let is_backend_function = data.dialect == Symbol::new(backend.dialect())
            && data.name == Symbol::new(backend.function_name());
        let convention = checked_calling_convention(ctx, op, backend)?;
        let is_root_done = exact_marker(ctx, op, ROOT_DONE_K_ATTR, backend)?;
        let is_root_worker = exact_marker(ctx, op, ROOT_CPS_WORKER_ATTR, backend)?;
        let is_root_wrapper = exact_marker(ctx, op, ROOT_WRAPPER_ATTR, backend)?;
        let is_root_call = exact_marker(ctx, op, ROOT_CPS_CALL_ATTR, backend)?;
        let symbol = data.attributes.get_symbol("sym_name");

        if is_backend_function && symbol == Some(Symbol::new(CPS_MAIN_SYMBOL)) && !is_root_worker {
            reserved_worker_without_marker = true;
        }
        if is_backend_function
            && symbol == Some(Symbol::new("main"))
            && convention == Some(CallingConvention::Cps)
        {
            return Err(TargetAbiError::new(format!(
                "{}: unwrapped Cps root main remains",
                backend.boundary()
            )));
        }
        if is_backend_function && symbol == Some(Symbol::new(ROOT_DONE_K_SYMBOL)) && !is_root_done {
            reserved_done_without_marker = true;
        }
        if reserved_done_without_marker {
            return Err(TargetAbiError::new(format!(
                "{}: reserved root done function is missing tribute.root_done_k",
                backend.boundary()
            )));
        }
        if data.dialect == Symbol::new(backend.dialect())
            && data.name == Symbol::new(backend.call_name())
            && data.attributes.get_symbol("callee") == Some(Symbol::new(CPS_MAIN_SYMBOL))
        {
            calls_to_reserved_worker.push(op);
        }

        if backend == TributeBackend::Wasm && is_backend_function && data.regions.is_empty() {
            return Err(TargetAbiError::new(format!(
                "{}: bodyless wasm.func `{}` must be lowered to wasm.import_func",
                backend.boundary(),
                data.attributes
                    .get_symbol("sym_name")
                    .map_or_else(|| "<unnamed>".to_owned(), |name| name.to_string())
            )));
        }

        if is_root_done
            && (!is_backend_function
                || convention != Some(CallingConvention::Cps)
                || data.attributes.get_symbol("sym_name") != Some(Symbol::new(ROOT_DONE_K_SYMBOL)))
        {
            return Err(TargetAbiError::new(format!(
                "{}: tribute.root_done_k is valid only on the exact Cps root done function",
                backend.boundary()
            )));
        }
        if is_root_worker
            && (!is_backend_function
                || convention != Some(CallingConvention::Cps)
                || symbol != Some(Symbol::new(CPS_MAIN_SYMBOL)))
        {
            return Err(TargetAbiError::new(format!(
                "{}: tribute.root_cps_worker is valid only on the exact Cps root worker",
                backend.boundary()
            )));
        }
        if is_root_wrapper
            && (!is_backend_function
                || !matches!(
                    convention,
                    Some(CallingConvention::Direct | CallingConvention::EvidenceDirect)
                ))
        {
            return Err(TargetAbiError::new(format!(
                "{}: tribute.root_wrapper is valid only on an ordinary root wrapper",
                backend.boundary()
            )));
        }

        if is_backend_function && convention == Some(CallingConvention::Cps) {
            let signature = data.attributes.get_type("type").ok_or_else(|| {
                TargetAbiError::new(format!(
                    "{}: Cps function has no signature",
                    backend.boundary()
                ))
            })?;
            let callable = core::Func::from_type_ref(ctx, signature).ok_or_else(|| {
                TargetAbiError::new(format!(
                    "{}: Cps function signature is not core.func",
                    backend.boundary()
                ))
            })?;
            if callable.r#return(ctx) != nil {
                return Err(TargetAbiError::new(format!(
                    "{}: Cps function result is not physically empty",
                    backend.boundary()
                )));
            }
            verify_cps_body(ctx, op, backend, is_root_done)?;
        }

        let is_tail = data.dialect == Symbol::new(backend.dialect())
            && backend
                .tail_names()
                .iter()
                .any(|name| data.name == Symbol::new(name));
        if is_tail {
            if !ctx.op_result_types(op).is_empty() {
                return Err(TargetAbiError::new(format!(
                    "{}: result-producing CPS transfer remains",
                    backend.boundary()
                )));
            }
            if convention != Some(CallingConvention::Cps) {
                return Err(TargetAbiError::new(format!(
                    "{}: proper-tail transfer lacks exact Cps convention",
                    backend.boundary()
                )));
            }
        }

        if is_root_call {
            if data.dialect != Symbol::new(backend.dialect())
                || data.name != Symbol::new(backend.call_name())
                || convention != Some(CallingConvention::Cps)
                || ctx.op_result_types(op) != [nil]
            {
                return Err(TargetAbiError::new(format!(
                    "{}: malformed tribute.root_cps_call exception",
                    backend.boundary()
                )));
            }
            root_calls.push(op);
        } else if convention == Some(CallingConvention::Cps) && !is_backend_function && !is_tail {
            return Err(TargetAbiError::new(format!(
                "{}: non-tail Cps operation is not the exact root wrapper call",
                backend.boundary()
            )));
        }

        if is_root_worker {
            root_workers.push(op);
        }
        if is_root_wrapper {
            root_wrappers.push(op);
        }
        if is_root_done {
            root_done_functions.push(op);
        }
    }

    let bridge_present = !root_workers.is_empty()
        || !root_wrappers.is_empty()
        || !root_done_functions.is_empty()
        || !root_calls.is_empty()
        || reserved_worker_without_marker
        || reserved_done_without_marker
        || !calls_to_reserved_worker.is_empty();
    if !bridge_present {
        return Ok(());
    }
    if reserved_worker_without_marker {
        return Err(TargetAbiError::new(format!(
            "{}: reserved Cps root worker is missing tribute.root_cps_worker",
            backend.boundary()
        )));
    }
    if reserved_done_without_marker {
        return Err(TargetAbiError::new(format!(
            "{}: reserved root done function is missing tribute.root_done_k",
            backend.boundary()
        )));
    }
    if root_workers.len() != 1 {
        return Err(TargetAbiError::new(format!(
            "{}: root bridge requires exactly one marked Cps worker",
            backend.boundary()
        )));
    }
    if root_wrappers.len() != 1 {
        return Err(TargetAbiError::new(format!(
            "{}: root bridge requires exactly one marked ordinary wrapper",
            backend.boundary()
        )));
    }
    if root_done_functions.len() != 1 {
        return Err(TargetAbiError::new(format!(
            "{}: root bridge requires exactly one marked done function",
            backend.boundary()
        )));
    }
    if root_calls.len() != 1 {
        return Err(TargetAbiError::new(format!(
            "{}: root bridge requires exactly one marked ordinary root CPS call",
            backend.boundary()
        )));
    }
    let root_worker = root_workers[0];
    let root_wrapper = root_wrappers[0];
    let root_call = root_calls[0];
    let worker_symbol = ctx
        .op(root_worker)
        .attributes
        .get_symbol("sym_name")
        .ok_or_else(|| {
            TargetAbiError::new(format!(
                "{}: marked Cps root worker has no symbol",
                backend.boundary()
            ))
        })?;
    if ctx.op(root_call).attributes.get_symbol("callee") != Some(worker_symbol) {
        return Err(TargetAbiError::new(format!(
            "{}: marked root CPS call does not target the marked Cps worker",
            backend.boundary()
        )));
    }
    if calls_to_reserved_worker.len() != 1 || calls_to_reserved_worker[0] != root_call {
        return Err(TargetAbiError::new(format!(
            "{}: root Cps worker must have exactly one marked ordinary wrapper call",
            backend.boundary()
        )));
    }
    let owner = enclosing_backend_function(ctx, root_call, backend).ok_or_else(|| {
        TargetAbiError::new(format!(
            "{}: root CPS call is not enclosed by a backend function",
            backend.boundary()
        ))
    })?;
    if owner != root_wrapper {
        return Err(TargetAbiError::new(format!(
            "{}: root CPS call is not owned by the marked ordinary wrapper",
            backend.boundary()
        )));
    }
    verify_root_wrapper_order(ctx, root_wrapper, root_call, backend)?;
    Ok(())
}

fn verify_cps_body(
    ctx: &IrContext,
    function: OpRef,
    backend: TributeBackend,
    root_done: bool,
) -> Result<(), TargetAbiError> {
    let nested = collect_ops(ctx, function);
    let ordinary_returns = nested
        .iter()
        .copied()
        .filter(|&op| {
            let data = ctx.op(op);
            data.dialect == Symbol::new(backend.dialect())
                && data.name == Symbol::new(backend.return_name())
        })
        .count();
    if root_done {
        if ordinary_returns != 1 {
            return Err(TargetAbiError::new(format!(
                "{}: root done function must contain exactly one terminal return",
                backend.boundary()
            )));
        }
    } else if ordinary_returns != 0 {
        return Err(TargetAbiError::new(format!(
            "{}: Cps body contains an ordinary return instead of a proper-tail transfer",
            backend.boundary()
        )));
    }
    Ok(())
}

fn checked_calling_convention(
    ctx: &IrContext,
    op: OpRef,
    backend: TributeBackend,
) -> Result<Option<CallingConvention>, TargetAbiError> {
    if !ctx.op(op).attributes.contains_key(CALLING_CONVENTION_ATTR) {
        return Ok(None);
    }
    get_calling_convention(ctx, op)
        .ok_or_else(|| {
            TargetAbiError::new(format!(
                "{}: operation has an invalid tribute.calling_convention attribute",
                backend.boundary()
            ))
        })
        .map(Some)
}

fn exact_marker(
    ctx: &IrContext,
    op: OpRef,
    marker: &str,
    backend: TributeBackend,
) -> Result<bool, TargetAbiError> {
    if !ctx.op(op).attributes.contains_key(marker) {
        return Ok(false);
    }
    if ctx.op(op).attributes.get_bool(marker) == Some(true) {
        return Ok(true);
    }
    Err(TargetAbiError::new(format!(
        "{}: `{marker}` marker must be exactly true",
        backend.boundary()
    )))
}

fn verify_root_wrapper_order(
    ctx: &IrContext,
    wrapper: OpRef,
    root_call: OpRef,
    backend: TributeBackend,
) -> Result<(), TargetAbiError> {
    let block = ctx.op(root_call).parent_block.ok_or_else(|| {
        TargetAbiError::new(format!(
            "{}: marked root CPS call has no containing block",
            backend.boundary()
        ))
    })?;
    let ops = &ctx.block(block).ops;
    let call_index = ops.iter().position(|&op| op == root_call).ok_or_else(|| {
        TargetAbiError::new(format!(
            "{}: marked root CPS call is not attached to its containing block",
            backend.boundary()
        ))
    })?;
    let return_after_call = ops.iter().skip(call_index + 1).any(|&op| {
        let data = ctx.op(op);
        data.dialect == Symbol::new(backend.dialect())
            && data.name == Symbol::new(backend.return_name())
    });
    if !return_after_call || enclosing_backend_function(ctx, root_call, backend) != Some(wrapper) {
        return Err(TargetAbiError::new(format!(
            "{}: root wrapper must resume after its ordinary CPS worker call",
            backend.boundary()
        )));
    }
    Ok(())
}

fn enclosing_backend_function(
    ctx: &IrContext,
    mut op: OpRef,
    backend: TributeBackend,
) -> Option<OpRef> {
    while let Some(block) = ctx.op(op).parent_block {
        let region = ctx.block(block).parent_region?;
        let parent = ctx.region(region).parent_op?;
        let data = ctx.op(parent);
        if data.dialect == Symbol::new(backend.dialect())
            && data.name == Symbol::new(backend.function_name())
        {
            return Some(parent);
        }
        op = parent;
    }
    None
}

fn collect_ops(ctx: &IrContext, root: OpRef) -> Vec<OpRef> {
    let mut ops = Vec::new();
    let _ = walk_op::<()>(ctx, root, &mut |op| {
        ops.push(op);
        ControlFlow::Continue(WalkAction::Advance)
    });
    ops
}

struct RecursiveTypeVerifier<'a> {
    ctx: &'a IrContext,
    backend: TributeBackend,
    visited: HashSet<(TypeRef, TypeSurface)>,
}

/// A final backend type can be valid as nominal/layout metadata while being
/// illegal in a runtime value slot. Keep the surfaces distinct so source
/// aggregates cannot leak into the native emitter as SSA values.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum TypeSurface {
    Value,
    Result,
    Metadata,
    Signature,
}

impl<'a> RecursiveTypeVerifier<'a> {
    fn new(ctx: &'a IrContext, backend: TributeBackend) -> Self {
        Self {
            ctx,
            backend,
            visited: HashSet::new(),
        }
    }

    fn verify_value(&mut self, ty: TypeRef, surface: &str) -> Result<(), TargetAbiError> {
        self.verify_on_surface(ty, TypeSurface::Value, surface)
    }

    fn verify_result(&mut self, ty: TypeRef, surface: &str) -> Result<(), TargetAbiError> {
        self.verify_on_surface(ty, TypeSurface::Result, surface)
    }

    fn verify_metadata(&mut self, ty: TypeRef, surface: &str) -> Result<(), TargetAbiError> {
        self.verify_on_surface(ty, TypeSurface::Metadata, surface)
    }

    fn verify_signature(&mut self, ty: TypeRef, surface: &str) -> Result<(), TargetAbiError> {
        self.verify_on_surface(ty, TypeSurface::Signature, surface)
    }

    fn verify_on_surface(
        &mut self,
        ty: TypeRef,
        type_surface: TypeSurface,
        surface: &str,
    ) -> Result<(), TargetAbiError> {
        if !self.visited.insert((ty, type_surface)) {
            return Ok(());
        }
        let data = self.ctx.types.get(ty);
        let dialect = data.dialect;
        let name = data.name;
        if !is_legal_final_type(self.ctx, self.backend, type_surface, ty)
            || data.attrs.get_symbol("name") == Some(Symbol::new("__tribute_cps_control"))
        {
            return Err(TargetAbiError::new(format!(
                "{}: illegal type `{dialect}.{name}` remains in {surface}",
                self.backend.boundary()
            )));
        }

        if type_surface == TypeSurface::Signature {
            if dialect != Symbol::new("core")
                || name != Symbol::new("func")
                || data.params.is_empty()
            {
                return Err(TargetAbiError::new(format!(
                    "{}: `{dialect}.{name}` is not a callable signature in {surface}",
                    self.backend.boundary()
                )));
            }
            self.verify_result(data.params[0], "function result")?;
            for &param in &data.params[1..] {
                self.verify_value(param, "function parameter")?;
            }
        } else if type_surface != TypeSurface::Result || !is_core_type(dialect, name, "nil") {
            for &param in &data.params {
                self.verify_metadata(param, "nested type parameter")?;
            }
        }
        for attribute in data.attrs.values() {
            self.verify_attribute(attribute, "type attribute")?;
        }
        Ok(())
    }

    fn verify_op_attribute(
        &mut self,
        op: OpRef,
        key: Symbol,
        attribute: &Attribute,
    ) -> Result<(), TargetAbiError> {
        let data = self.ctx.op(op);
        let is_function_signature = key == Symbol::new("type")
            && data.dialect == Symbol::new(self.backend.dialect())
            && data.name == Symbol::new(self.backend.function_name());
        let is_indirect_signature = key == Symbol::new("sig")
            && data.dialect == Symbol::new(self.backend.dialect())
            && (data.name == Symbol::new("call_indirect")
                || data.name == Symbol::new("return_call_indirect"));
        if is_function_signature || is_indirect_signature {
            let Attribute::Type(ty) = attribute else {
                return Err(TargetAbiError::new(format!(
                    "{}: callable signature attribute is not a type",
                    self.backend.boundary()
                )));
            };
            self.verify_signature(*ty, "callable signature")
        } else {
            self.verify_attribute(attribute, "operation attribute")
        }
    }

    fn verify_attribute(
        &mut self,
        attribute: &Attribute,
        surface: &str,
    ) -> Result<(), TargetAbiError> {
        match attribute {
            Attribute::Type(ty) => self.verify_metadata(*ty, surface),
            Attribute::List(values) => {
                for value in values {
                    self.verify_attribute(value, surface)?;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }
}

/// Target types are accepted positively and by use-site. The layout/alias
/// graph retains named aggregate descriptions for target lowering, while SSA
/// values are restricted to the selected emitter's value conversion.
fn is_legal_final_type(
    ctx: &IrContext,
    backend: TributeBackend,
    surface: TypeSurface,
    ty: TypeRef,
) -> bool {
    let data = ctx.types.get(ty);
    let dialect = data.dialect;
    let name = data.name;
    match surface {
        TypeSurface::Value => is_target_value_type(ctx, backend, ty),
        TypeSurface::Result => {
            is_core_type(dialect, name, "nil") || is_target_value_type(ctx, backend, ty)
        }
        TypeSurface::Signature => is_core_type(dialect, name, "func"),
        TypeSurface::Metadata => is_target_metadata_type(backend, dialect, name, data),
    }
}

fn is_target_value_type(ctx: &IrContext, backend: TributeBackend, ty: TypeRef) -> bool {
    let data = ctx.types.get(ty);
    let dialect = data.dialect;
    let name = data.name;
    match backend {
        // Exactly `translate_type`: aggregate and callable types remain
        // layout/signature metadata, never Cranelift SSA values.
        TributeBackend::Native => ["i1", "i8", "i16", "i32", "i64", "f32", "f64", "ptr"]
            .iter()
            .any(|candidate| is_core_type(dialect, name, candidate)),
        // Mirrors `type_to_valtype`; core.nil is the Wasm null-reference unit
        // representation, and only the concrete Bytes backing ref is a
        // supported core.ref value in the Wasm ABI.
        TributeBackend::Wasm => {
            let core_value = [
                "nil", "i1", "i32", "i64", "f32", "f64", "bytes", "ptr", "array", "func",
            ]
            .iter()
            .any(|candidate| is_core_type(dialect, name, candidate));
            let bytes_ref = dialect == Symbol::new("core")
                && name == Symbol::new("ref")
                && trunk_ir_wasm_backend::gc_types::concrete_wasm_ref_type(ctx, ty).is_some();
            let adt_value = dialect == Symbol::new("adt")
                && (["struct", "enum", "typeref", "variant_inst"]
                    .iter()
                    .any(|candidate| name == Symbol::new(candidate))
                    || data.attrs.get_bool("is_variant") == Some(true));
            let wasm_ref = dialect == Symbol::new("wasm")
                && ["anyref", "i31ref", "structref", "arrayref", "funcref"]
                    .iter()
                    .any(|candidate| name == Symbol::new(candidate));
            core_value || bytes_ref || adt_value || wasm_ref
        }
    }
}

fn is_target_metadata_type(
    backend: TributeBackend,
    dialect: Symbol,
    name: Symbol,
    data: &trunk_ir::types::TypeData,
) -> bool {
    let core = Symbol::new("core");
    let adt = Symbol::new("adt");
    let wasm = Symbol::new("wasm");
    if dialect == core {
        return [
            "func", "nil", "i1", "i8", "i16", "i32", "i64", "f32", "f64", "ptr", "bytes", "ref",
            "array",
        ]
        .iter()
        .any(|candidate| name == Symbol::new(candidate));
    }
    if dialect == adt {
        return ["struct", "enum", "typeref", "variant_inst"]
            .iter()
            .any(|candidate| name == Symbol::new(candidate))
            || data.attrs.get_bool("is_variant") == Some(true);
    }
    backend == TributeBackend::Wasm
        && dialect == wasm
        && ["anyref", "i31ref", "structref", "arrayref", "funcref"]
            .iter()
            .any(|candidate| name == Symbol::new(candidate))
}

fn is_core_type(dialect: Symbol, name: Symbol, expected: &str) -> bool {
    dialect == Symbol::new("core") && name == Symbol::from_dynamic(expected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;

    type RootMutation = fn(&mut IrContext, OpRef, OpRef, OpRef, OpRef);
    type RootBoundaryCase = (&'static str, RootMutation, &'static str);

    fn root_bridge_fixture(dialect: &str, wrapper: &str) -> String {
        format!(
            r#"core.module @test {{
  {dialect}.func @__tribute_cps_main() -> core.nil {{
    {dialect}.return_call {{callee = @__tribute_root_done_k}}
  }}
  {dialect}.func @__tribute_root_done_k() -> core.nil {{
    {dialect}.return
  }}
  {dialect}.func @{wrapper}() -> core.nil {{
    %call = {dialect}.call {{callee = @__tribute_cps_main}} : core.nil
    {dialect}.return
  }}
}}"#
        )
    }

    fn marked_root_bridge(
        dialect: &str,
        wrapper: &str,
    ) -> (IrContext, Module, OpRef, OpRef, OpRef, OpRef) {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, &root_bridge_fixture(dialect, wrapper));
        let mut worker = None;
        let mut done = None;
        let mut wrapper_op = None;
        for op in module.ops(&ctx) {
            let symbol = ctx.op(op).attributes.get_symbol("sym_name");
            if symbol == Some(Symbol::new(CPS_MAIN_SYMBOL)) {
                worker = Some(op);
            } else if symbol == Some(Symbol::new(ROOT_DONE_K_SYMBOL)) {
                done = Some(op);
            } else if symbol == Some(Symbol::from_dynamic(wrapper)) {
                wrapper_op = Some(op);
            }
        }
        let worker = worker.expect("root worker");
        let done = done.expect("root done");
        let wrapper_op = wrapper_op.expect("root wrapper");
        let root_call = collect_ops(&ctx, wrapper_op)
            .into_iter()
            .find(|&op| {
                ctx.op(op).attributes.get_symbol("callee") == Some(Symbol::new(CPS_MAIN_SYMBOL))
            })
            .expect("root call");

        for op in [worker, done] {
            ctx.op_mut(op).attributes.insert(
                Symbol::new(CALLING_CONVENTION_ATTR),
                Attribute::Int(CallingConvention::Cps as i128),
            );
        }
        ctx.op_mut(worker)
            .attributes
            .insert(Symbol::new(ROOT_CPS_WORKER_ATTR), Attribute::Bool(true));
        ctx.op_mut(done)
            .attributes
            .insert(Symbol::new(ROOT_DONE_K_ATTR), Attribute::Bool(true));
        ctx.op_mut(wrapper_op).attributes.insert(
            Symbol::new(CALLING_CONVENTION_ATTR),
            Attribute::Int(CallingConvention::Direct as i128),
        );
        ctx.op_mut(wrapper_op)
            .attributes
            .insert(Symbol::new(ROOT_WRAPPER_ATTR), Attribute::Bool(true));
        ctx.op_mut(root_call).attributes.insert(
            Symbol::new(CALLING_CONVENTION_ATTR),
            Attribute::Int(CallingConvention::Cps as i128),
        );
        ctx.op_mut(root_call)
            .attributes
            .insert(Symbol::new(ROOT_CPS_CALL_ATTR), Attribute::Bool(true));
        let worker_tail = collect_ops(&ctx, worker)
            .into_iter()
            .find(|&op| ctx.op(op).name == Symbol::new("return_call"))
            .expect("worker tail transfer");
        ctx.op_mut(worker_tail).attributes.insert(
            Symbol::new(CALLING_CONVENTION_ATTR),
            Attribute::Int(CallingConvention::Cps as i128),
        );
        (ctx, module, worker, done, wrapper_op, root_call)
    }

    #[test]
    fn backend_ready_accepts_marked_root_bridge_after_native_rename_and_for_wasm() {
        for (backend, dialect, wrapper) in [
            (TributeBackend::Native, "clif", "_tribute_main"),
            (TributeBackend::Wasm, "wasm", "main"),
        ] {
            let (ctx, module, ..) = marked_root_bridge(dialect, wrapper);
            verify_tribute_backend_ready(&ctx, module, backend).unwrap();
        }
    }

    #[test]
    fn backend_ready_rejects_invalid_convention_and_root_marker_gaps_without_mutation() {
        let cases: [RootBoundaryCase; 5] = [
            (
                "invalid convention",
                |ctx, _worker, _done, wrapper, _call| {
                    ctx.op_mut(wrapper)
                        .attributes
                        .insert(Symbol::new(CALLING_CONVENTION_ATTR), Attribute::Int(99));
                },
                "invalid tribute.calling_convention",
            ),
            (
                "unwrapped worker",
                |ctx, worker, _done, _wrapper, _call| {
                    ctx.op_mut(worker).attributes.remove(ROOT_CPS_WORKER_ATTR);
                },
                "missing tribute.root_cps_worker",
            ),
            (
                "unmarked root call",
                |ctx, _worker, _done, _wrapper, call| {
                    ctx.op_mut(call).attributes.remove(ROOT_CPS_CALL_ATTR);
                    ctx.op_mut(call).attributes.remove(CALLING_CONVENTION_ATTR);
                },
                "exactly one marked ordinary root CPS call",
            ),
            (
                "unmarked wrapper",
                |ctx, _worker, _done, wrapper, _call| {
                    ctx.op_mut(wrapper).attributes.remove(ROOT_WRAPPER_ATTR);
                },
                "exactly one marked ordinary wrapper",
            ),
            (
                "unmarked done_k",
                |ctx, _worker, done, _wrapper, _call| {
                    ctx.op_mut(done).attributes.remove(ROOT_DONE_K_ATTR);
                },
                "missing tribute.root_done_k",
            ),
        ];

        for (name, mutate, expected) in cases {
            let (mut ctx, module, worker, done, wrapper, call) =
                marked_root_bridge("clif", "_tribute_main");
            mutate(&mut ctx, worker, done, wrapper, call);
            let before = print_module(&ctx, module.op());
            let error =
                verify_tribute_backend_ready(&ctx, module, TributeBackend::Native).expect_err(name);
            assert!(error.to_string().contains(expected), "{name}: {error}");
            assert_eq!(print_module(&ctx, module.op()), before, "{name}");
        }
    }

    #[test]
    fn backend_ready_rejects_duplicate_marked_root_calls() {
        let (mut ctx, module, _worker, _done, _wrapper, root_call) =
            marked_root_bridge("wasm", "main");
        let mut mapping = trunk_ir::IrMapping::new();
        let duplicate = ctx.clone_op(root_call, &mut mapping);
        let block = ctx.op(root_call).parent_block.unwrap();
        ctx.push_op(block, duplicate);
        let error = verify_tribute_backend_ready(&ctx, module, TributeBackend::Wasm).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("exactly one marked ordinary root CPS call"),
            "{error}"
        );
    }

    #[test]
    fn backend_ready_rejects_an_unwrapped_cps_main() {
        let (mut ctx, module, worker, _done, _wrapper, _call) = marked_root_bridge("wasm", "main");
        ctx.op_mut(worker).attributes.insert(
            Symbol::new("sym_name"),
            Attribute::Symbol(Symbol::new("main")),
        );
        ctx.op_mut(worker).attributes.remove(ROOT_CPS_WORKER_ATTR);
        let before = print_module(&ctx, module.op());

        let error = verify_tribute_backend_ready(&ctx, module, TributeBackend::Wasm).unwrap_err();

        assert!(
            error.to_string().contains("unwrapped Cps root main"),
            "{error}"
        );
        assert_eq!(print_module(&ctx, module.op()), before);
    }

    #[test]
    fn backend_ready_rejects_unknown_top_level_nested_and_attribute_types() {
        let cases = [
            (
                "top-level",
                r#"!unknown = unknown.value()"#,
                "unknown.value",
            ),
            (
                "nested",
                r#"!unknown = core.array(unknown.value())"#,
                "unknown.value",
            ),
            (
                "attribute",
                r#"!unknown = adt.struct() {metadata = [unknown.value()], name = @Unknown}"#,
                "unknown.value",
            ),
        ];
        for (name, alias, expected) in cases {
            let mut ctx = IrContext::new();
            let module = parse_test_module(
                &mut ctx,
                &format!(
                    "core.module @test {{\n  {alias}\n  wasm.func @main() -> core.nil {{\n    wasm.return\n  }}\n}}"
                ),
            );
            let before = print_module(&ctx, module.op());
            let error = verify_tribute_backend_ready(&ctx, module, TributeBackend::Wasm)
                .expect_err(name)
                .to_string();
            assert!(error.contains(expected), "{name}: {error}");
            assert_eq!(print_module(&ctx, module.op()), before, "{name}");
        }
    }

    #[test]
    fn backend_ready_rejects_unknown_backend_operations() {
        for (backend, dialect) in [
            (TributeBackend::Native, "clif"),
            (TributeBackend::Wasm, "wasm"),
        ] {
            let mut ctx = IrContext::new();
            let module = parse_test_module(
                &mut ctx,
                &format!(
                    "core.module @test {{\n  {dialect}.func @main() -> core.nil {{\n    {dialect}.unknown\n    {dialect}.return\n  }}\n}}"
                ),
            );
            let error = verify_tribute_backend_ready(&ctx, module, backend)
                .expect_err(dialect)
                .to_string();
            assert!(error.contains(&format!("{dialect}.unknown")), "{error}");
        }
    }

    #[test]
    fn backend_ready_rejects_native_metadata_types_in_value_and_signature_slots() {
        let cases = [
            (
                "bytes parameter",
                r#"clif.func @main(%bytes: core.bytes) -> core.nil {
    clif.return
  }"#,
            ),
            (
                "callable parameter",
                r#"clif.func @main(%callback: core.func(core.nil, core.i32)) -> core.nil {
    clif.return
  }"#,
            ),
            (
                "aggregate result",
                r#"clif.func @main() -> adt.struct() {fields = [[@payload, core.i32]], name = @Box} {
    clif.return
  }"#,
            ),
        ];
        for (name, function) in cases {
            let mut ctx = IrContext::new();
            let module = parse_test_module(
                &mut ctx,
                &format!(
                    r#"core.module @test {{
  !layout = adt.struct() {{fields = [[@bytes, core.bytes]], name = @Layout}}
  clif.func @void() -> core.nil {{
    clif.return
  }}
  {function}
}}"#,
                ),
            );
            let before = print_module(&ctx, module.op());
            let error = verify_tribute_backend_ready(&ctx, module, TributeBackend::Native)
                .expect_err(name)
                .to_string();
            assert!(error.contains("illegal type"), "{name}: {error}");
            assert_eq!(print_module(&ctx, module.op()), before, "{name}");
        }
    }

    #[test]
    fn backend_ready_keeps_native_layout_metadata_out_of_value_legality() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !layout = adt.struct() {fields = [[@bytes, core.bytes]], name = @Layout}
  clif.func @main(%value: core.i32) -> core.nil {
    clif.return
  }
}"#,
        );
        verify_tribute_backend_ready(&ctx, module, TributeBackend::Native).unwrap();
    }

    #[test]
    fn backend_ready_derives_wasm_value_slots_from_emittable_valtypes() {
        let cases = [
            (
                "narrow integer parameter",
                r#"wasm.func @main(%byte: core.i8) -> core.nil {
    wasm.return
  }"#,
            ),
            (
                "narrow integer result",
                r#"wasm.func @main() -> core.i16 {
    wasm.return
  }"#,
            ),
        ];
        for (name, function) in cases {
            let mut ctx = IrContext::new();
            let module = parse_test_module(
                &mut ctx,
                &format!(
                    "core.module @test {{\n  wasm.func @void() -> core.nil {{\n    wasm.return\n  }}\n  {function}\n}}"
                ),
            );
            let before = print_module(&ctx, module.op());
            let error = verify_tribute_backend_ready(&ctx, module, TributeBackend::Wasm)
                .expect_err(name)
                .to_string();
            assert!(error.contains("illegal type"), "{name}: {error}");
            assert_eq!(print_module(&ctx, module.op()), before, "{name}");
        }
    }

    #[test]
    fn backend_ready_accepts_wasm_nil_value_representation() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  wasm.func @main(%unit: core.nil) -> core.nil {
    wasm.return
  }
}"#,
        );
        verify_tribute_backend_ready(&ctx, module, TributeBackend::Wasm).unwrap();
    }
}
