//! IR validation for Cranelift backend.
//!
//! This module validates that IR is ready for emission:
//! - All operations must be explicitly legal for the native backend boundary.
//!
//! Dialect validation errors prevent emission from proceeding.

use std::collections::HashMap;

use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::clif;
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::printer::print_type;
use trunk_ir::refs::{OpRef, RegionRef, TypeRef, ValueRef};
use trunk_ir::rewrite::{ConversionTarget, Module};

use crate::{CompilationError, CompilationResult};

const NATIVE_BACKEND_READY_BOUNDARY: &str = "native-backend-ready";

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

/// Validate that a module's IR is ready for Cranelift emission.
///
/// This function checks that all operations are explicitly legal for the
/// native backend boundary.
///
/// Returns an error if validation fails, preventing emission.
pub fn validate_clif_ir(ctx: &IrContext, module: Module) -> CompilationResult<()> {
    let Some(body) = module.body(ctx) else {
        return Err(CompilationError::ir_validation("Module has no body region"));
    };

    let target = native_backend_ready_target();
    let failures = target.verify_full(ctx, body);

    let mut errors: Vec<String> = failures
        .into_iter()
        .map(|op| format!("{} in boundary {}", op, NATIVE_BACKEND_READY_BOUNDARY))
        .collect();
    errors.extend(validate_clif_contracts(ctx, module));

    if errors.is_empty() {
        return Ok(());
    }
    let message = format!(
        "IR validation failed for boundary {NATIVE_BACKEND_READY_BOUNDARY} with {} error(s):\n  - {}",
        errors.len(),
        errors.join("\n  - ")
    );
    Err(CompilationError::ir_validation(message))
}

/// Validate native callable contracts before instruction selection can erase
/// their TrunkIR type identity.
fn validate_clif_contracts(ctx: &IrContext, module: Module) -> Vec<String> {
    let Some(body) = module.body(ctx) else {
        return Vec::new();
    };
    let mut errors = Vec::new();
    let mut functions = HashMap::new();
    collect_clif_function_signatures(ctx, body, &mut functions, &mut errors);
    validate_clif_region(ctx, body, None, &functions, &mut errors);
    errors
}

fn collect_clif_function_signatures(
    ctx: &IrContext,
    region: RegionRef,
    functions: &mut HashMap<Symbol, clif::FuncSig>,
    errors: &mut Vec<String>,
) {
    for &block in &ctx.region(region).blocks {
        for &op in &ctx.block(block).ops {
            if clif::Func::matches(ctx, op) {
                let Some(name) = ctx.op(op).attributes.get_symbol("sym_name") else {
                    errors.push("clif.func requires a symbol name".into());
                    continue;
                };
                let Some(signature) = ctx
                    .op(op)
                    .attributes
                    .get_type("type")
                    .and_then(|ty| clif::FuncSig::from_type_ref(ctx, ty))
                else {
                    errors.push(format!(
                        "clif.func @{name} requires a valid clif.func_sig type"
                    ));
                    continue;
                };
                if functions.insert(name, signature).is_some() {
                    errors.push(format!(
                        "clif.func @{name} has a duplicate symbol definition"
                    ));
                }
            }
            for &nested in &ctx.op(op).regions {
                collect_clif_function_signatures(ctx, nested, functions, errors);
            }
        }
    }
}

fn runtime_types(ctx: &IrContext, types: &[TypeRef]) -> Vec<TypeRef> {
    types
        .iter()
        .copied()
        .filter(|ty| !crate::function::is_nil_type(ctx, *ty))
        .collect()
}

fn type_lists_match(expected: &[TypeRef], actual: &[TypeRef]) -> bool {
    expected.len() == actual.len()
        && expected
            .iter()
            .zip(actual)
            .all(|(&expected, &actual)| expected == actual)
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
            "clif.{} {role} count mismatch: expected {}, found {}",
            ctx.op(op).name,
            expected.len(),
            values.len()
        ));
        return;
    }
    for (index, (&value, &ty)) in values.iter().zip(expected).enumerate() {
        if ty != ctx.value_ty(value) {
            errors.push(format!(
                "clif.{} {role} #{index} type mismatch: expected {}, found {}",
                ctx.op(op).name,
                print_type(ctx, ty),
                print_type(ctx, ctx.value_ty(value))
            ));
        }
    }
}

fn check_result_types(
    ctx: &IrContext,
    op: OpRef,
    expected: &[TypeRef],
    role: &str,
    errors: &mut Vec<String>,
) {
    let actual = ctx.op_result_types(op);
    if !type_lists_match(expected, actual) {
        errors.push(format!(
            "clif.{} {role} mismatch: expected {:?}, found {:?}",
            ctx.op(op).name,
            expected,
            actual
        ));
    }
}

/// Calls and returns may retain their logical `core.nil` slots from
/// source-level `func` operations. The emitter treats those slots as
/// zero-width, so hand-written native IR may instead use the projected runtime
/// result list. Both forms must otherwise match the declared contract exactly.
fn check_call_result_types(
    ctx: &IrContext,
    op: OpRef,
    expected: &[TypeRef],
    role: &str,
    errors: &mut Vec<String>,
) {
    if type_lists_match(expected, ctx.op_result_types(op)) {
        return;
    }
    check_result_types(ctx, op, &runtime_types(ctx, expected), role, errors);
}

fn values_match_types(ctx: &IrContext, values: &[ValueRef], expected: &[TypeRef]) -> bool {
    values.len() == expected.len()
        && values
            .iter()
            .zip(expected)
            .all(|(&value, &ty)| ty == ctx.value_ty(value))
}

/// See [`check_call_result_types`]. A direct return may use the full logical
/// result list or its zero-width `core.nil` projection.
fn check_direct_return_types(
    ctx: &IrContext,
    op: OpRef,
    values: &[ValueRef],
    expected: &[TypeRef],
    errors: &mut Vec<String>,
) {
    if values_match_types(ctx, values, expected) {
        return;
    }
    check_value_types(
        ctx,
        op,
        values,
        &runtime_types(ctx, expected),
        "return",
        errors,
    );
}

fn signature_for_exact_indirect(
    ctx: &IrContext,
    op: OpRef,
    errors: &mut Vec<String>,
) -> Option<clif::FuncSig> {
    let signature = ctx.op(op).attributes.get_type("sig");
    let valid = signature.and_then(|ty| clif::FuncSig::from_type_ref(ctx, ty));
    if valid.is_none() {
        errors.push(format!(
            "clif.{} requires a valid exact clif.func_sig",
            ctx.op(op).name
        ));
    }
    valid
}

fn validate_clif_function(
    ctx: &IrContext,
    op: OpRef,
    errors: &mut Vec<String>,
) -> Option<clif::FuncSig> {
    clif::Func::from_op(ctx, op).ok()?;
    let Some(name) = ctx.op(op).attributes.get_symbol("sym_name") else {
        errors.push("clif.func requires a symbol name".into());
        return None;
    };
    let signature = ctx
        .op(op)
        .attributes
        .get_type("type")
        .and_then(|ty| clif::FuncSig::from_type_ref(ctx, ty));
    let Some(signature) = signature else {
        errors.push(format!(
            "clif.func @{name} requires a valid clif.func_sig type"
        ));
        return None;
    };
    if ctx.op(op).regions.len() > 1 {
        errors.push(format!("clif.func @{name} has more than one body region"));
    }
    if let Some(&body) = ctx.op(op).regions.first() {
        let Some(&entry) = ctx.region(body).blocks.first() else {
            errors.push(format!("clif.func @{name} body requires an entry block"));
            return Some(signature);
        };
        check_value_types(
            ctx,
            op,
            ctx.block_args(entry),
            signature.inputs(ctx),
            "entry argument",
            errors,
        );
    }
    Some(signature)
}

fn validate_clif_region(
    ctx: &IrContext,
    region: RegionRef,
    owner: Option<clif::FuncSig>,
    functions: &HashMap<Symbol, clif::FuncSig>,
    errors: &mut Vec<String>,
) {
    for &block in &ctx.region(region).blocks {
        for &op in &ctx.block(block).ops {
            if clif::Func::matches(ctx, op) {
                let function = validate_clif_function(ctx, op, errors);
                for &body in &ctx.op(op).regions {
                    validate_clif_region(ctx, body, function, functions, errors);
                }
                continue;
            }

            let operands = ctx.op_operands(op);
            if clif::Call::matches(ctx, op) {
                let Some(name) = ctx.op(op).attributes.get_symbol("callee") else {
                    errors.push("clif.call requires a symbol callee".into());
                    continue;
                };
                if let Some(signature) = functions.get(&name) {
                    check_value_types(
                        ctx,
                        op,
                        operands,
                        signature.inputs(ctx),
                        "call argument",
                        errors,
                    );
                    check_call_result_types(
                        ctx,
                        op,
                        signature.results(ctx),
                        "call result list",
                        errors,
                    );
                }
            } else if clif::CallIndirect::matches(ctx, op) {
                if operands.is_empty() {
                    errors.push("clif.call_indirect requires a callee operand".into());
                }
                if let Some(signature) = signature_for_exact_indirect(ctx, op, errors) {
                    let args = operands.get(1..).unwrap_or_default();
                    check_value_types(
                        ctx,
                        op,
                        args,
                        signature.inputs(ctx),
                        "call argument",
                        errors,
                    );
                    check_call_result_types(
                        ctx,
                        op,
                        signature.results(ctx),
                        "call result list",
                        errors,
                    );
                }
            } else if clif::Return::matches(ctx, op) {
                let Some(signature) = owner else {
                    errors.push("clif.return requires a nearest clif.func owner".into());
                    continue;
                };
                check_direct_return_types(ctx, op, operands, signature.results(ctx), errors);
            } else if clif::ReturnCall::matches(ctx, op) {
                let Some(caller) = owner else {
                    errors.push("clif.return_call requires a nearest clif.func owner".into());
                    continue;
                };
                let Some(callee) = ctx.op(op).attributes.get_symbol("callee") else {
                    errors.push("clif.return_call requires a symbol callee".into());
                    continue;
                };
                if let Some(signature) = functions.get(&callee) {
                    check_value_types(
                        ctx,
                        op,
                        operands,
                        signature.inputs(ctx),
                        "tail argument",
                        errors,
                    );
                    if runtime_types(ctx, caller.results(ctx))
                        != runtime_types(ctx, signature.results(ctx))
                    {
                        errors.push("clif.return_call caller/callee result lists differ".into());
                    }
                }
            } else if clif::ReturnCallIndirect::matches(ctx, op) {
                let Some(caller) = owner else {
                    errors.push(
                        "clif.return_call_indirect requires a nearest clif.func owner".into(),
                    );
                    continue;
                };
                if operands.is_empty() {
                    errors.push("clif.return_call_indirect requires a callee operand".into());
                }
                if let Some(signature) = signature_for_exact_indirect(ctx, op, errors) {
                    let args = operands.get(1..).unwrap_or_default();
                    check_value_types(
                        ctx,
                        op,
                        args,
                        signature.inputs(ctx),
                        "tail argument",
                        errors,
                    );
                    if runtime_types(ctx, caller.results(ctx))
                        != runtime_types(ctx, signature.results(ctx))
                    {
                        errors.push(
                            "clif.return_call_indirect caller/callee result lists differ".into(),
                        );
                    }
                }
            }

            for &nested in &ctx.op(op).regions {
                validate_clif_region(ctx, nested, owner, functions, errors);
            }
        }
    }
}

/// Conversion target for IR that is ready for Cranelift emission.
pub fn native_backend_ready_target() -> ConversionTarget {
    let mut target = ConversionTarget::new();
    target.add_legal_dialect("clif");
    target.add_legal_op("core", "module");
    target
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::OperationDataBuilder;
    use trunk_ir::context::{BlockData, IrContext, RegionData};
    use trunk_ir::location::Span;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::smallvec::smallvec;
    use trunk_ir::symbol::Symbol;
    use trunk_ir::types::{Attribute, Location};

    fn test_ctx() -> (IrContext, Location) {
        let mut ctx = IrContext::new();
        let path = ctx.paths.intern("test.trb".to_owned());
        let loc = Location::new(path, Span::new(0, 0));
        (ctx, loc)
    }

    fn make_module(ctx: &mut IrContext, loc: Location, dialect: &str, name: &str) -> Module {
        let op_data = OperationDataBuilder::new(
            loc,
            Symbol::from_dynamic(dialect),
            Symbol::from_dynamic(name),
        )
        .build(ctx);
        let op = ctx.create_op(op_data);

        let block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        ctx.push_op(block, op);
        let region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![block],
            parent_op: None,
        });
        let module_data =
            OperationDataBuilder::new(loc, Symbol::new("core"), Symbol::new("module"))
                .attr("sym_name", Attribute::Symbol(Symbol::new("test")))
                .region(region)
                .build(ctx);
        let module_op = ctx.create_op(module_data);
        Module::new(ctx, module_op).expect("test module should be valid")
    }

    fn validation_error(input: &str) -> String {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);
        validate_clif_ir(&ctx, module)
            .expect_err("native contract must fail closed")
            .to_string()
    }

    #[test]
    fn native_backend_ready_allows_clif_ops() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            "core.module @test { clif.func @f() { clif.return } }",
        );

        validate_clif_ir(&ctx, module).unwrap();
    }

    #[test]
    fn native_backend_ready_rejects_unknown_ops() {
        let (mut ctx, loc) = test_ctx();
        let module = make_module(&mut ctx, loc, "arith", "add");

        let err = validate_clif_ir(&ctx, module).unwrap_err().to_string();
        assert!(err.contains("native-backend-ready"));
        assert!(err.contains("arith.add"));
        assert!(err.contains("Unknown"));
    }

    #[test]
    fn native_boundary_rejects_malformed_target_signature_storage() {
        let error = validation_error(
            r#"core.module @test {
  clif.func {sym_name = @bad, type = clif.func_sig(core.i32) {num_inputs = 2, num_results = 1}}
}"#,
        );
        assert!(
            error.contains("@bad requires a valid clif.func_sig type"),
            "{error}"
        );
    }

    #[test]
    fn native_boundary_rejects_unused_direct_and_exact_indirect_result_mismatches() {
        let error = validation_error(
            r#"core.module @test {
  clif.func @direct_target(%value: core.i32) -> core.i32 { clif.return %value }
  clif.func @direct_caller() {
    %value = clif.iconst {value = 1} : core.i32
    %unused = clif.call %value {callee = @direct_target} : core.i64
    clif.return
  }
  clif.func @indirect_caller() {
    %callee = clif.iconst {value = 0} : core.ptr
    %first, %second = clif.call_indirect %callee {sig = clif.func_sig<() -> (core.i32, core.i64)>} : core.i64, core.i32
    clif.return
  }
}"#,
        );
        assert!(
            error.contains("clif.call call result list mismatch"),
            "{error}"
        );
        assert!(
            error.contains("clif.call_indirect call result list mismatch"),
            "{error}"
        );
    }

    #[test]
    fn native_boundary_accepts_direct_logical_nil_result_slots() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  clif.func @unit(%unit: core.nil) -> core.nil { clif.return %unit }
  clif.func @caller(%unit: core.nil) {
    %unused = clif.call %unit {callee = @unit} : core.nil
    clif.return
  }
}"#,
        );

        validate_clif_ir(&ctx, module).unwrap();
    }

    #[test]
    fn native_boundary_rejects_semantic_and_shape_matched_values_in_pointer_slots() {
        let error = validation_error(
            r#"core.module @test {
  !shaped = adt.struct() {fields = [[@field, core.i32]], name = @Impostor}
  clif.func @target(%value: core.ptr) -> core.ptr { clif.return %value }
  clif.func @semantic(%callee: core.ptr, %value: tribute_rt.anyref) -> core.ptr {
    %direct = clif.call %value {callee = @target} : tribute_rt.anyref
    %indirect = clif.call_indirect %callee, %direct {sig = clif.func_sig<(core.ptr) -> core.ptr>} : tribute_rt.anyref
    clif.return %indirect
  }
  clif.func @shaped(%value: !shaped) -> core.ptr {
    %direct = clif.call %value {callee = @target} : !shaped
    clif.return %direct
  }
}"#,
        );

        assert!(
            error.contains("clif.call call argument #0 type mismatch: expected core.ptr, found tribute_rt.anyref"),
            "{error}"
        );
        assert!(
            error.contains("clif.call_indirect call argument #0 type mismatch: expected core.ptr, found tribute_rt.anyref"),
            "{error}"
        );
        assert!(
            error.contains(
                "clif.call call argument #0 type mismatch: expected core.ptr, found adt.struct()"
            ),
            "{error}"
        );
    }

    #[test]
    fn native_boundary_rejects_argument_return_and_tail_contract_mismatches() {
        let error = validation_error(
            r#"core.module @test {
  clif.func @takes_i64(%value: core.i64) -> core.i32 {
    %result = clif.iconst {value = 1} : core.i32
    clif.return %result
  }
  clif.func @tail_target() -> core.i32 {
    %result = clif.iconst {value = 1} : core.i32
    clif.return %result
  }
  clif.func @bad_return() -> core.i32 { clif.return }
  clif.func @bad_argument() {
    %value = clif.iconst {value = 1} : core.i32
    %unused = clif.call %value {callee = @takes_i64} : core.i32
    clif.return
  }
  clif.func @bad_tail() { clif.return_call {callee = @tail_target} }
  clif.func @bad_indirect_tail(%callee: core.ptr, %value: core.i32) {
    clif.return_call_indirect %callee, %value {sig = clif.func_sig<(core.i64) -> core.i32>}
  }
}"#,
        );
        assert!(
            error.contains("clif.return return count mismatch"),
            "{error}"
        );
        assert!(
            error.contains("clif.call call argument #0 type mismatch"),
            "{error}"
        );
        assert!(
            error.contains("clif.return_call caller/callee result lists differ"),
            "{error}"
        );
        assert!(
            error.contains("clif.return_call_indirect tail argument #0 type mismatch"),
            "{error}"
        );
        assert!(
            error.contains("clif.return_call_indirect caller/callee result lists differ"),
            "{error}"
        );
    }
}
