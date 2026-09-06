//! IR validation for wasm backend.
//!
//! This module validates that IR is ready for emission:
//! - All operations must be in the `wasm` dialect (error)
//!
//! Dialect validation errors prevent emission from proceeding.

use trunk_ir::IrContext;
use trunk_ir::Module;
use trunk_ir::Symbol;
use trunk_ir::dialect::core;
use trunk_ir::dialect::wasm as wasm_dialect;
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::refs::{OpRef, RegionRef, TypeRef, ValueRef};

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
    validate_region(ctx, body, 0, &mut errors);

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
fn validate_region(ctx: &IrContext, region: RegionRef, depth: usize, errors: &mut Vec<String>) {
    for &block_ref in &ctx.region(region).blocks {
        for &op in &ctx.block(block_ref).ops {
            validate_operation(ctx, op, depth, errors);
        }
    }
}

/// Validate a single operation.
fn validate_operation(ctx: &IrContext, op: OpRef, depth: usize, errors: &mut Vec<String>) {
    let op_data = ctx.op(op);
    let dialect = op_data.dialect;
    let name = op_data.name;

    // Check dialect - must be wasm (with specific exceptions)
    if !is_allowed_dialect(ctx, op, depth) {
        errors.push(format!("Non-wasm operation found: {}.{}", dialect, name));
    }
    validate_return_call_indirect(ctx, op, errors);
    validate_direct_callable_contracts(ctx, op, errors);

    // Recursively validate nested regions
    for &region in &op_data.regions {
        validate_region(ctx, region, depth + 1, errors);
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

/// Resolve a direct target from the nearest enclosing module outwards. A known
/// malformed or ambiguous target returns `Some(None)` so it cannot be treated
/// as an undeclared runtime import.
fn resolve_wasm_callee(
    ctx: &IrContext,
    mut op: OpRef,
    name: Symbol,
) -> Option<Option<wasm_dialect::FuncSig>> {
    loop {
        let region = ctx.block(ctx.op(op).parent_block?).parent_region?;
        let parent = ctx.region(region).parent_op?;
        if core::Module::matches(ctx, parent) {
            let mut matches = ctx
                .region(region)
                .blocks
                .iter()
                .flat_map(|&block| ctx.block(block).ops.iter().copied())
                .filter(|&candidate| {
                    ctx.op(candidate).attributes.get_symbol("sym_name") == Some(name)
                });
            if let Some(found) = matches.next() {
                if matches.next().is_some()
                    || (!wasm_dialect::Func::matches(ctx, found)
                        && !wasm_dialect::ImportFunc::matches(ctx, found))
                {
                    return Some(None);
                }
                return Some(
                    ctx.op(found)
                        .attributes
                        .get_type("type")
                        .and_then(|ty| wasm_dialect::FuncSig::from_type_ref(ctx, ty)),
                );
            }
        }
        op = parent;
    }
}

fn is_nil(ctx: &IrContext, ty: TypeRef) -> bool {
    let data = ctx.types.get(ty);
    data.dialect == Symbol::new("core") && data.name == Symbol::new("nil")
}

/// `core.nil` occupies a logical result slot but has no physical Wasm value;
/// every non-nil result must remain in its original order and type.
fn result_list_matches(ctx: &IrContext, actual: &[TypeRef], expected: &[TypeRef]) -> bool {
    let expected = if actual.len() == expected.len() {
        expected.to_vec()
    } else {
        expected
            .iter()
            .copied()
            .filter(|&result| !is_nil(ctx, result))
            .collect()
    };
    actual.len() == expected.len()
        && actual.iter().zip(expected).all(|(&actual, expected)| {
            crate::emit::helpers::is_wasm_physical_argument_assignable(ctx, actual, expected)
        })
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
fn validate_direct_callable_contracts(ctx: &IrContext, op: OpRef, errors: &mut Vec<String>) {
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
    let Some(callee) = ctx.op(op).attributes.get_symbol("callee") else {
        return;
    };
    let Some(resolved) = resolve_wasm_callee(ctx, op, callee) else {
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
            if !crate::emit::helpers::is_wasm_physical_argument_assignable(
                ctx,
                ctx.value_ty(operand),
                input,
            ) {
                let actual = ctx.types.get(ctx.value_ty(operand));
                let expected = ctx.types.get(input);
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
        if caller.results(ctx) != signature.results(ctx) {
            errors.push("wasm.return_call tail caller/callee result lists differ".into());
        }
    } else if !result_list_matches(ctx, ctx.op_result_types(op), signature.results(ctx)) {
        errors.push("wasm.call result list mismatch".into());
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
    }
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
            "must have an empty result",
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
  !Array = core.array(core.i32)
  !Struct = adt.struct() {fields = [[@value, core.i32]], name = @Struct}
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
  !Struct = adt.struct() {fields = [[@value, core.i32]], name = @Struct}
  wasm.func @caller(%table_index: core.i32, %value: !Struct) -> core.nil {
    wasm.return_call_indirect %table_index, %value {signature = wasm.func_sig<(wasm.structref) -> core.nil>, table = 0, type_idx = 0}
  }
}"#,
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
    fn resolves_tail_call_against_the_nearest_module_owner() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @outer {
  wasm.func @target() -> core.i32 {
    %value = wasm.i32_const {value = 1} : core.i32
    wasm.return %value
  }
  core.module @inner {
    wasm.func @target() -> core.nil { wasm.return }
    wasm.func @caller() -> core.i32 { wasm.return_call {callee = @target} }
  }
}"#,
        );
        let error = validate_wasm_ir(&ctx, module).expect_err("nearest target has Unit result");
        assert!(
            error
                .to_string()
                .contains("wasm.return_call tail caller/callee result lists differ"),
            "{error}"
        );
    }
}
