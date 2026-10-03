//! IR validation for wasm backend.
//!
//! This module validates that IR is ready for emission:
//! - All operations must be in the `wasm` dialect (error)
//!
//! Dialect validation errors prevent emission from proceeding.

use trunk_ir::IrContext;
use trunk_ir::Module;
use trunk_ir::Symbol;
use trunk_ir::dialect::wasm as wasm_dialect;
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::refs::{OpRef, RegionRef, TypeRef, ValueRef};
use trunk_ir::symbol_table::SymbolTable;

use crate::{CompilationError, CompilationResult};

/// Validation error details.
#[derive(Debug)]
pub struct ValidationError {
    pub message: String,
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

/// Validate that a module's IR is ready for wasm emission (arena version).
///
/// This function checks that all operations are in the `wasm` dialect
/// (except allowed exceptions like `core.module`).
///
/// Returns an error if validation fails, preventing emission.
pub fn validate_wasm_ir(ctx: &IrContext, module: Module) -> CompilationResult<()> {
    let mut errors: Vec<String> = Vec::new();

    let body = module
        .body(ctx)
        .ok_or_else(|| CompilationError::invalid_module("module has no body region"))?;
    // Direct callees resolve by root-qualified path over the whole module.
    let symbols = SymbolTable::collect(ctx, module);
    for (name, _) in symbols.duplicates() {
        errors.push(format!("symbol @{name} is defined more than once"));
    }
    validate_region(ctx, body, 0, &symbols, &mut errors);

    if errors.is_empty() {
        Ok(())
    } else {
        let message = format!(
            "IR validation failed with {} error(s):\n  - {}",
            errors.len(),
            errors.join("\n  - ")
        );
        Err(CompilationError::ir_validation(message))
    }
}

/// Validate a region recursively.
fn validate_region(
    ctx: &IrContext,
    region: RegionRef,
    depth: usize,
    symbols: &SymbolTable,
    errors: &mut Vec<String>,
) {
    for &block_ref in &ctx.region(region).blocks {
        for &op in &ctx.block(block_ref).ops {
            validate_operation(ctx, op, depth, symbols, errors);
        }
    }
}

/// Validate a single operation.
fn validate_operation(
    ctx: &IrContext,
    op: OpRef,
    depth: usize,
    symbols: &SymbolTable,
    errors: &mut Vec<String>,
) {
    let op_data = ctx.op(op);
    let dialect = op_data.dialect.clone();
    let name = op_data.name.clone();

    // Check dialect - must be wasm (with specific exceptions)
    if !is_allowed_dialect(ctx, op, depth) {
        errors.push(format!("Non-wasm operation found: {}.{}", dialect, name));
    }
    if wasm_dialect::CallIndirect::matches(ctx, op)
        && let Err(error) = crate::emit::helpers::exact_call_indirect_signature(ctx, op)
    {
        errors.push(error.to_string());
    }
    validate_return_call_indirect(ctx, op, errors);
    validate_direct_callable_contracts(ctx, op, symbols, errors);

    // Recursively validate nested regions
    for region in ctx.op_regions(op) {
        validate_region(ctx, region, depth + 1, symbols, errors);
    }
}

/// Find the nearest lexical `wasm.func` that owns `op`.
fn enclosing_wasm_func_signature(ctx: &IrContext, mut op: OpRef) -> Option<wasm_dialect::FuncSig> {
    loop {
        let region = ctx.block(ctx.op(op).parent_block?).parent_region?;
        let parent = ctx.region(region).parent_op?;
        if wasm_dialect::Func::matches(ctx, parent) {
            return ctx
                .op(parent)
                .attributes
                .get_type("type")
                .and_then(|ty| wasm_dialect::FuncSig::from_type_ref(ctx, ty));
        }
        op = parent;
    }
}

/// Resolve a direct target by its root-qualified path. A known malformed or
/// duplicated target returns `Some(None)` so it cannot be treated as an
/// undeclared runtime import.
fn resolve_wasm_callee(
    ctx: &IrContext,
    symbols: &SymbolTable,
    name: &Symbol,
) -> Option<Option<wasm_dialect::FuncSig>> {
    let found = *symbols.definitions_of(name).first()?;
    if symbols.resolve(name).is_none()
        || (!wasm_dialect::Func::matches(ctx, found)
            && !wasm_dialect::ImportFunc::matches(ctx, found))
    {
        return Some(None);
    }
    Some(
        ctx.op(found)
            .attributes
            .get_type("type")
            .and_then(|ty| wasm_dialect::FuncSig::from_type_ref(ctx, ty)),
    )
}

fn is_nil(ctx: &IrContext, ty: TypeRef) -> bool {
    let data = ctx.get_type(ty);
    data.dialect == Symbol::new("core") && data.name == Symbol::new("nil")
}

/// `core.nil` occupies a logical result slot but has no physical Wasm value.
/// The first list produces values consumed by the second, so assignability is
/// directional: a value may be widened but never narrowed.
fn result_list_matches(ctx: &IrContext, produced: &[TypeRef], received: &[TypeRef]) -> bool {
    let produced = produced
        .iter()
        .copied()
        .filter(|&result| !is_nil(ctx, result))
        .collect::<Vec<_>>();
    let received = received
        .iter()
        .copied()
        .filter(|&result| !is_nil(ctx, result))
        .collect::<Vec<_>>();
    produced.len() == received.len()
        && produced.iter().zip(received).all(|(&produced, received)| {
            is_wasm_physical_result_assignable(ctx, produced, received)
        })
}

/// `core.array` is emitted as the same abstract array reference as
/// `wasm.arrayref`; accepting that physical equivalence here does not permit
/// a reference downcast such as `wasm.anyref -> wasm.structref`. An array with
/// a runtime layout identifier is a concrete builtin reference, so an erased
/// `arrayref` does not satisfy it.
fn is_wasm_physical_result_assignable(
    ctx: &IrContext,
    produced: TypeRef,
    received: TypeRef,
) -> bool {
    let produced_data = ctx.get_type(produced);
    let received_data = ctx.get_type(received);
    (produced_data.dialect == Symbol::new("wasm")
        && produced_data.name == Symbol::new("arrayref")
        && received_data.dialect == Symbol::new("core")
        && received_data.name == Symbol::new("array")
        && crate::emit::helpers::builtin_layout_type_idx(ctx, received).is_none())
        || (produced_data.dialect == Symbol::new("core")
            && produced_data.name == Symbol::new("i32")
            && received_data.dialect == Symbol::new("core")
            && received_data.name == Symbol::new("i1"))
        || crate::assignability::is_wasm_physical_argument_assignable(ctx, produced, received)
}

fn check_value_types(
    ctx: &IrContext,
    op: OpRef,
    values: &[ValueRef],
    expected: &[TypeRef],
    role: &str,
    errors: &mut Vec<String>,
) {
    if values.len() != expected.len() {
        errors.push(format!(
            "wasm.{} {role} count mismatch: expected {}, found {}",
            ctx.op(op).name,
            expected.len(),
            values.len()
        ));
        return;
    }
    for (index, (&value, &ty)) in values.iter().zip(expected).enumerate() {
        if ctx.value_ty(value) != ty {
            errors.push(format!(
                "wasm.{} {role} #{index} type mismatch",
                ctx.op(op).name
            ));
        }
    }
}

/// Validate direct calls and returns against the target-owned Wasm signature.
/// This boundary is deliberately independent of binary validation: a malformed
/// IR program must not be emitted just because a later section happens to have
/// a compatible shape.
fn validate_direct_callable_contracts(
    ctx: &IrContext,
    op: OpRef,
    symbols: &SymbolTable,
    errors: &mut Vec<String>,
) {
    if wasm_dialect::Return::matches(ctx, op) {
        let Some(caller) = enclosing_wasm_func_signature(ctx, op) else {
            errors.push("wasm.return requires a valid enclosing wasm.func signature".into());
            return;
        };
        let expected = caller.results(ctx);
        let operands = ctx.op_operands(op);
        if !result_list_matches(
            ctx,
            &operands
                .iter()
                .map(|&value| ctx.value_ty(value))
                .collect::<Vec<_>>(),
            expected,
        ) {
            check_value_types(ctx, op, operands, expected, "return", errors);
        }
        return;
    }

    let is_tail = wasm_dialect::ReturnCall::matches(ctx, op);
    let is_direct = wasm_dialect::Call::matches(ctx, op) || is_tail;
    if !is_direct {
        return;
    }
    let Some(callee) = ctx.op(op).attributes.get_symbol_ref("callee") else {
        errors.push(format!(
            "wasm.{} requires a symbol callee attribute",
            ctx.op(op).name
        ));
        return;
    };
    let Some(resolved) = resolve_wasm_callee(ctx, symbols, callee) else {
        return;
    };
    let Some(signature) = resolved else {
        errors.push(format!(
            "wasm.{} requires a uniquely resolved valid wasm.func_sig",
            ctx.op(op).name
        ));
        return;
    };
    let operands = ctx.op_operands(op);
    let inputs = signature.inputs(ctx);
    if operands.len() != inputs.len() {
        errors.push(format!(
            "wasm.{} call argument count mismatch: expected {}, found {}",
            ctx.op(op).name,
            inputs.len(),
            operands.len()
        ));
    } else {
        for (index, (&operand, &input)) in operands.iter().zip(inputs).enumerate() {
            if !crate::assignability::is_wasm_physical_argument_assignable(
                ctx,
                ctx.value_ty(operand),
                input,
            ) {
                let actual = ctx.get_type(ctx.value_ty(operand));
                let expected = ctx.get_type(input);
                errors.push(format!(
                    "wasm.{} call argument #{index} type mismatch: found {}.{}, expected {}.{}",
                    ctx.op(op).name,
                    actual.dialect,
                    actual.name,
                    expected.dialect,
                    expected.name
                ));
            }
        }
    }
    if is_tail {
        let Some(caller) = enclosing_wasm_func_signature(ctx, op) else {
            errors.push("wasm.return_call requires a valid enclosing wasm.func signature".into());
            return;
        };
        if !tail_results_agree(ctx, caller.results(ctx), signature.results(ctx)) {
            errors.push("wasm.return_call tail caller/callee result lists differ".into());
        }
    } else if !result_list_matches(ctx, signature.results(ctx), ctx.op_result_types(op)) {
        let format_types = |types: &[TypeRef]| {
            types
                .iter()
                .map(|&ty| {
                    let data = ctx.get_type(ty);
                    format!("{}.{}", data.dialect, data.name)
                })
                .collect::<Vec<_>>()
                .join(", ")
        };
        errors.push(format!(
            "wasm.call result list mismatch: callee produces [{}], call declares [{}]",
            format_types(signature.results(ctx)),
            format_types(ctx.op_result_types(op))
        ));
    }
}

/// Validate the source-of-truth signature needed for a proper indirect tail
/// transfer before emission has a chance to construct a type section.
fn validate_return_call_indirect(ctx: &IrContext, op: OpRef, errors: &mut Vec<String>) {
    if !wasm_dialect::ReturnCallIndirect::matches(ctx, op) {
        return;
    }
    if let Err(error) = crate::emit::helpers::exact_return_call_indirect_signature(ctx, op) {
        errors.push(error.to_string());
        return;
    }
    let Some(caller) = enclosing_wasm_func_signature(ctx, op) else {
        errors.push(
            "wasm.return_call_indirect requires a valid enclosing wasm.func signature".into(),
        );
        return;
    };
    let Some(signature) = ctx
        .op(op)
        .attributes
        .get_type("signature")
        .and_then(|ty| wasm_dialect::FuncSig::from_type_ref(ctx, ty))
    else {
        // The exact helper above already produced the local diagnostic.
        return;
    };
    if !tail_results_agree(ctx, caller.results(ctx), signature.results(ctx)) {
        errors.push("wasm.return_call_indirect tail caller/callee result lists differ".into());
    }
}

/// A tail callee returns straight to the caller's caller, so their machine
/// result slots must agree. `core.nil` results occupy no Wasm slot.
fn tail_results_agree(ctx: &IrContext, caller: &[TypeRef], callee: &[TypeRef]) -> bool {
    let machine = |results: &[TypeRef]| {
        results
            .iter()
            .copied()
            .filter(|ty| !crate::emit::helpers::is_nil_type(ctx, *ty))
            .collect::<Vec<_>>()
    };
    machine(caller) == machine(callee)
}

/// Check if an operation's dialect is allowed in the emit phase.
fn is_allowed_dialect(ctx: &IrContext, op: OpRef, depth: usize) -> bool {
    let wasm_dialect = Symbol::new("wasm");
    let op_data = ctx.op(op);

    if op_data.dialect == wasm_dialect {
        return true;
    }

    // Allow core.module only at the top level (depth 0)
    if depth == 0 && op_data.dialect == Symbol::new("core") && op_data.name == Symbol::new("module")
    {
        return true;
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::parser::parse_test_module;

    fn assert_rejects_tail_signature(source: &str) {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, source);
        let error = validate_wasm_ir(&ctx, module).expect_err("invalid physical relation");
        assert!(
            error
                .to_string()
                .contains("operands do not match its exact signature"),
            "{error}"
        );
    }

    #[test]
    fn tail_result_agreement_ignores_zero_width_nil_results() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  wasm.func @target(%value: core.i32) {
    wasm.return
  }
  wasm.func @direct(%value: core.i32) -> core.nil {
    wasm.return_call %value {callee = @target}
  }
  wasm.func @indirect(%table_index: core.i32, %value: core.i32) -> core.nil {
    wasm.return_call_indirect %table_index, %value {signature = wasm.func_sig<(core.i32) -> ()>, table = 0, type_idx = 0}
  }
}"#,
        );
        validate_wasm_ir(&ctx, module).expect("nil results occupy no Wasm result slot");
    }

    #[test]
    fn rejects_invalid_return_call_indirect_exact_contracts() {
        let rejects = |source: &str, diagnostic: &str| {
            let mut ctx = IrContext::new();
            let module = parse_test_module(&mut ctx, source);
            let error = validate_wasm_ir(&ctx, module).unwrap_err();
            assert!(error.to_string().contains(diagnostic), "{error}");
        };

        rejects(
            r#"core.module @test {
  wasm.func @caller(%table_index: core.i32, %value: core.i32) -> core.nil {
    wasm.return_call_indirect %table_index, %value {table = 0, type_idx = 0}
  }
}"#,
            "lacks signature",
        );

        rejects(
            r#"core.module @test {
  wasm.func @caller(%table_index: core.i32, %value: core.i32) -> core.nil {
    wasm.return_call_indirect %table_index, %value {signature = core.i32, table = 0, type_idx = 0}
  }
}"#,
            "signature must be wasm.func_sig",
        );

        rejects(
            r#"core.module @test {
  wasm.func @caller(%table_index: core.i32, %value: core.i32) -> core.nil {
    wasm.return_call_indirect %table_index, %value {signature = wasm.func_sig<(core.i32) -> core.i32>, table = 0, type_idx = 0}
  }
}"#,
            "tail caller/callee result lists differ",
        );

        rejects(
            r#"core.module @test {
  wasm.func @caller() -> core.nil {
    wasm.return_call_indirect {signature = wasm.func_sig<() -> core.nil>, table = 0, type_idx = 0}
  }
}"#,
            "requires a table index operand",
        );

        rejects(
            r#"core.module @test {
  wasm.func @caller(%table_index: core.i64, %value: core.i32) -> core.nil {
    wasm.return_call_indirect %table_index, %value {signature = wasm.func_sig<(core.i32) -> core.nil>, table = 0, type_idx = 0}
  }
}"#,
            "first operand must be an i32 table index",
        );

        rejects(
            r#"core.module @test {
  wasm.func @caller(%table_index: core.i32, %value: core.i64) -> core.nil {
    wasm.return_call_indirect %table_index, %value {signature = wasm.func_sig<(core.i32) -> core.nil>, table = 0, type_idx = 0}
  }
}"#,
            "operands do not match its exact signature",
        );
    }

    #[test]
    fn accepts_physical_equivalence_and_gc_upcasts_in_return_call_indirect() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !Array = core.array<core.i32>
  !Struct = adt.struct<core.i32 {name = "value"}, {name = "Struct"}>
  wasm.func @typeref(%table_index: core.i32, %value: adt.typeref) -> core.nil {
    wasm.return_call_indirect %table_index, %value {signature = wasm.func_sig<(wasm.anyref) -> core.nil>, table = 0, type_idx = 0}
  }
  wasm.func @struct(%table_index: core.i32, %value: !Struct) -> core.nil {
    wasm.return_call_indirect %table_index, %value {signature = wasm.func_sig<(wasm.anyref) -> core.nil>, table = 0, type_idx = 0}
  }
  wasm.func @array(%table_index: core.i32, %value: !Array) -> core.nil {
    wasm.return_call_indirect %table_index, %value {signature = wasm.func_sig<(wasm.arrayref) -> core.nil>, table = 0, type_idx = 0}
  }
  wasm.func @i31(%table_index: core.i32, %value: wasm.i31ref) -> core.nil {
    wasm.return_call_indirect %table_index, %value {signature = wasm.func_sig<(wasm.anyref) -> core.nil>, table = 0, type_idx = 0}
  }
}"#,
        );

        validate_wasm_ir(&ctx, module).expect("physical-compatible tail arguments must validate");
    }

    #[test]
    fn rejects_downcasts_and_unrelated_reference_families_in_return_call_indirect() {
        for (value_ty, parameter_ty) in [
            ("wasm.anyref", "wasm.structref"),
            ("wasm.arrayref", "wasm.structref"),
            ("wasm.funcref", "wasm.anyref"),
        ] {
            assert_rejects_tail_signature(&format!(
                r#"core.module @test {{
  wasm.func @caller(%table_index: core.i32, %value: {value_ty}) -> core.nil {{
    wasm.return_call_indirect %table_index, %value {{signature = wasm.func_sig<({parameter_ty}) -> core.nil>, table = 0, type_idx = 0}}
  }}
}}"#
            ));
        }
    }

    #[test]
    fn rejects_unregistered_adt_struct_as_structref_tail_argument() {
        assert_rejects_tail_signature(
            r#"core.module @test {
  !Struct = adt.struct<core.i32 {name = "value"}, {name = "Struct"}>
  wasm.func @caller(%table_index: core.i32, %value: !Struct) -> core.nil {
    wasm.return_call_indirect %table_index, %value {signature = wasm.func_sig<(wasm.structref) -> core.nil>, table = 0, type_idx = 0}
  }
}"#,
        );
    }

    #[test]
    fn erased_arrayref_result_satisfies_plain_but_not_layout_arrays() {
        let mut ctx = IrContext::new();
        parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !Marker = adt.struct<{name = "_Marker", layout = "evidence_marker"}>
  !Evidence = core.array<!Marker, {layout = "evidence"}>
  !Plain = core.array<!Marker>
}"#,
        );
        let alias = |ctx: &IrContext, name: &str| {
            ctx.type_alias_by_name(&trunk_ir::Symbol::from_dynamic(name))
                .unwrap()
        };
        let evidence = alias(&ctx, "Evidence");
        let plain = alias(&ctx, "Plain");
        let arrayref =
            ctx.intern_type(trunk_ir::types::TypeDataBuilder::new("wasm", "arrayref").build());

        assert!(is_wasm_physical_result_assignable(&ctx, arrayref, plain));
        assert!(!is_wasm_physical_result_assignable(
            &ctx, arrayref, evidence
        ));
    }

    #[test]
    fn accepts_registered_concrete_gc_references_in_abstract_argument_slots() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !String = adt.enum<{name = "String"}>
  !Leaf = adt.enum<{base_enum = !String, is_variant = true, variant_tag = "Leaf"}>
  !Closure = adt.struct<{name = "_closure", layout = "closure"}>
  !Marker = adt.struct<{name = "_Marker", layout = "evidence_marker"}>
  !Evidence = core.array<!Marker, {layout = "evidence"}>
  !Data = core.array<core.i8, {layout = "bytes_data"}>
  !Bytes = adt.struct<!Data {name = "data"}, core.i32 {name = "offset"}, core.i32 {name = "len"}, {name = "_Bytes", layout = "bytes"}>
  wasm.func @byRef(%value: wasm.structref) -> core.nil { wasm.return }
  wasm.func @byArray(%value: wasm.arrayref) -> core.nil { wasm.return }
  wasm.func @caller(%leaf: !Leaf, %bytes: !Bytes, %typeref: adt.typeref, %closure: !Closure, %marker: !Marker, %evidence: !Evidence) -> core.nil {
    wasm.call %leaf {callee = @byRef}
    wasm.call %bytes {callee = @byRef}
    wasm.call %typeref {callee = @byRef}
    wasm.call %closure {callee = @byRef}
    wasm.call %marker {callee = @byRef}
    wasm.call %evidence {callee = @byArray}
    wasm.return
  }
}"#,
        );

        validate_wasm_ir(&ctx, module)
            .expect("registered concrete GC references widen to abstract argument slots");
    }

    #[test]
    fn accepts_registered_gc_references_in_anyref_slots() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !Marker = adt.struct<{name = "_Marker", layout = "evidence_marker"}>
  !Evidence = core.array<!Marker, {layout = "evidence"}>
  !Array = core.array<core.i32>
  !Data = core.array<core.i8, {layout = "bytes_data"}>
  !Bytes = adt.struct<!Data {name = "data"}, core.i32 {name = "offset"}, core.i32 {name = "len"}, {name = "_Bytes", layout = "bytes"}>
  wasm.func @byAny(%value: wasm.anyref) -> core.nil { wasm.return }
  wasm.func @caller(%bytes: !Bytes, %evidence: !Evidence, %array: !Array, %erased: adt.struct) -> core.nil {
    wasm.call %bytes {callee = @byAny}
    wasm.call %evidence {callee = @byAny}
    wasm.call %array {callee = @byAny}
    wasm.call %erased {callee = @byAny}
    wasm.return
  }
}"#,
        );
        validate_wasm_ir(&ctx, module).expect("registered GC references satisfy an anyref slot");

        for value_ty in ["wasm.funcref", "wasm.externref", "core.i64", "!TagOnly"] {
            let mut ctx = IrContext::new();
            let module = parse_test_module(
                &mut ctx,
                &format!(
                    r#"core.module @test {{
  !TagOnly = adt.enum<{{is_variant = true, variant_tag = "Leaf"}}>
  wasm.func @byAny(%value: wasm.anyref) -> core.nil {{ wasm.return }}
  wasm.func @caller(%value: {value_ty}) -> core.nil {{
    wasm.call %value {{callee = @byAny}}
    wasm.return
  }}
}}"#
                ),
            );
            let error = validate_wasm_ir(&ctx, module).expect_err(
                "non-internal references and unregistered variant spellings cannot satisfy anyref",
            );
            assert!(
                error.to_string().contains("call argument #0 type mismatch"),
                "{value_ty}: {error}"
            );
        }
    }

    #[test]
    fn rejects_unregistered_and_narrowing_gc_reference_arguments() {
        for (value_ty, parameter_ty) in [
            ("!Unregistered", "wasm.structref"),
            ("!TagOnly", "wasm.structref"),
            ("wasm.anyref", "wasm.structref"),
            ("wasm.arrayref", "wasm.structref"),
            ("wasm.funcref", "wasm.structref"),
            ("wasm.i31ref", "wasm.structref"),
            ("!Bytes", "wasm.arrayref"),
            ("wasm.structref", "wasm.arrayref"),
            ("!Leaf", "wasm.arrayref"),
        ] {
            let mut ctx = IrContext::new();
            let module = parse_test_module(
                &mut ctx,
                &format!(
                    r#"core.module @test {{
  !String = adt.enum<{{name = "String"}}>
  !Leaf = adt.enum<{{base_enum = !String, is_variant = true, variant_tag = "Leaf"}}>
  !TagOnly = adt.enum<{{is_variant = true, variant_tag = "Leaf"}}>
  !Unregistered = adt.struct<core.i32 {{name = "value"}}, {{name = "Unregistered"}}>
  !Data = core.array<core.i8, {{layout = "bytes_data"}}>
  !Bytes = adt.struct<!Data {{name = "data"}}, core.i32 {{name = "offset"}}, core.i32 {{name = "len"}}, {{name = "_Bytes", layout = "bytes"}}>
  wasm.func @callee(%value: {parameter_ty}) -> core.nil {{ wasm.return }}
  wasm.func @caller(%value: {value_ty}) -> core.nil {{
    wasm.call %value {{callee = @callee}}
    wasm.return
  }}
}}"#
                ),
            );

            let error = validate_wasm_ir(&ctx, module)
                .expect_err("unregistered or narrowing GC reference argument must be rejected");
            assert!(
                error.to_string().contains("call argument #0 type mismatch"),
                "{value_ty} -> {parameter_ty}: {error}"
            );
        }
    }

    #[test]
    fn widens_registered_concrete_gc_results_but_rejects_narrowing() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !String = adt.enum<{name = "String"}>
  !Leaf = adt.enum<{base_enum = !String, is_variant = true, variant_tag = "Leaf"}>
  wasm.func @produces(%leaf: !Leaf) -> !Leaf { wasm.return %leaf }
  wasm.func @caller(%leaf: !Leaf) -> core.nil {
    %value = wasm.call %leaf {callee = @produces} : wasm.structref
    wasm.return
  }
}"#,
        );
        validate_wasm_ir(&ctx, module)
            .expect("a registered concrete GC result widens to the declared abstract result");

        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !String = adt.enum<{name = "String"}>
  !Leaf = adt.enum<{base_enum = !String, is_variant = true, variant_tag = "Leaf"}>
  wasm.func @produces(%value: wasm.structref) -> wasm.structref { wasm.return %value }
  wasm.func @caller(%value: wasm.structref) -> core.nil {
    %leaf = wasm.call %value {callee = @produces} : !Leaf
    wasm.return
  }
}"#,
        );
        let error = validate_wasm_ir(&ctx, module)
            .expect_err("an abstract result cannot narrow to a concrete declared result");
        assert!(
            error.to_string().contains("wasm.call result list mismatch"),
            "{error}"
        );
    }

    #[test]
    fn validates_direct_call_and_return_result_contracts() {
        let rejects = |source: &str, diagnostic: &str| {
            let mut ctx = IrContext::new();
            let module = parse_test_module(&mut ctx, source);
            let error = validate_wasm_ir(&ctx, module).expect_err("mismatched direct contract");
            assert!(error.to_string().contains(diagnostic), "{error}");
        };

        rejects(
            r#"core.module @test {
  wasm.func @callee() -> core.i32 {
    %value = wasm.i32_const {value = 1} : core.i32
    wasm.return %value
  }
  wasm.func @caller() -> core.nil {
    %value = wasm.call {callee = @callee} : core.i64
    wasm.return
  }
}"#,
            "wasm.call result list mismatch",
        );
        rejects(
            r#"core.module @test {
  wasm.func @caller() -> core.i32 {
    %value = wasm.i64_const {value = 1} : core.i64
    wasm.return %value
  }
}"#,
            "wasm.return return #0 type mismatch",
        );

        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  wasm.func @unit() -> core.nil { wasm.return }
  wasm.func @caller() -> core.nil {
    wasm.call {callee = @unit}
    wasm.return
  }
}"#,
        );
        validate_wasm_ir(&ctx, module).expect("Unit direct call and return omit a physical result");
    }

    #[test]
    fn rejects_direct_result_narrowing_and_missing_callee_contracts() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  wasm.func @source() -> wasm.anyref {
    %value = wasm.nop : wasm.anyref
    wasm.return %value
  }
  wasm.func @caller() -> wasm.structref {
    %value = wasm.call {callee = @source} : wasm.structref
    wasm.return %value
  }
}"#,
        );
        let error = validate_wasm_ir(&ctx, module).expect_err("anyref cannot narrow to structref");
        assert!(
            error.to_string().contains("wasm.call result list mismatch"),
            "{error}"
        );
        let Err(error) = crate::emit_module_to_wasm(&mut ctx, module) else {
            panic!("the invalid direct result must not reach binary emission")
        };
        assert!(
            error.to_string().contains("wasm.call result list mismatch"),
            "{error}"
        );

        for callee in ["", " {callee = 0}"] {
            let mut ctx = IrContext::new();
            let module = parse_test_module(
                &mut ctx,
                &format!(
                    "core.module @test {{
  wasm.func @caller() -> core.nil {{
    wasm.call{callee}
    wasm.return
  }}
}}"
                ),
            );
            let error = validate_wasm_ir(&ctx, module).expect_err("callee must be a symbol");
            assert!(
                error
                    .to_string()
                    .contains("wasm.call requires a symbol callee attribute"),
                "{error}"
            );
        }
    }

    #[test]
    fn duplicate_qualified_definitions_are_reported() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @outer {
  core.module @inner {
    wasm.func @twice() { wasm.return }
    wasm.func @twice() { wasm.return }
  }
}"#,
        );
        let error = validate_wasm_ir(&ctx, module).expect_err("duplicate definition");
        assert!(
            error
                .to_string()
                .contains("symbol @inner::twice is defined more than once"),
            "{error}"
        );
    }

    #[test]
    fn resolves_tail_calls_by_root_qualified_path() {
        let source = |callee: &str| {
            format!(
                r#"core.module @outer {{
  wasm.func @target() -> core.i32 {{
    %value = wasm.i32_const {{value = 1}} : core.i32
    wasm.return %value
  }}
  core.module @inner {{
    wasm.func @target() -> core.nil {{ wasm.return }}
    wasm.func @caller() -> core.i32 {{ wasm.return_call {{callee = {callee}}} }}
  }}
}}"#
            )
        };

        // A bare name names the root definition, not the sibling in `inner`.
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, &source("@target"));
        validate_wasm_ir(&ctx, module).expect("root target has the caller's result");

        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, &source(r#"@"inner::target""#));
        let error = validate_wasm_ir(&ctx, module).expect_err("nested target has Unit result");
        assert!(
            error
                .to_string()
                .contains("wasm.return_call tail caller/callee result lists differ"),
            "{error}"
        );
    }

    #[test]
    fn rejects_return_call_indirect_with_a_mismatched_enclosing_result_list() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @outer {
  core.module @inner {
    wasm.func @caller(%index: core.i32) -> core.i32 {
      wasm.return_call_indirect %index {signature = wasm.func_sig<() -> core.nil>, table = 0, type_idx = 0}
    }
  }
}"#,
        );
        let error = validate_wasm_ir(&ctx, module).expect_err("tail result mismatch");
        assert!(
            error
                .to_string()
                .contains("wasm.return_call_indirect tail caller/callee result lists differ"),
            "{error}"
        );
    }
}
