//! IR validation for wasm backend.
//!
//! This module validates that IR is ready for emission:
//! - All operations must be in the `wasm` dialect (error)
//!
//! Dialect validation errors prevent emission from proceeding.

use trunk_ir::IrContext;
use trunk_ir::Module;
use trunk_ir::Symbol;
use trunk_ir::refs::OpRef;
use trunk_ir::rewrite::{ConversionTarget, LegalityDecision};

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
    let body = module
        .body(ctx)
        .ok_or_else(|| CompilationError::invalid_module("module has no body region"))?;
    let errors = wasm_backend_ready_target()
        .verify_full(ctx, body)
        .into_iter()
        .map(|failure| failure.to_string())
        .collect::<Vec<_>>();

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

/// Exact operation boundary accepted by the Wasm emitter.
///
/// TrunkIR has no dialect-operation registry.  Keep the inventory at the
/// backend boundary so a fabricated `wasm.*` operation cannot pass merely by
/// sharing the emitter dialect prefix.
pub fn wasm_backend_ready_target() -> ConversionTarget {
    ConversionTarget::new()
        .legal_op("core", "module")
        .dynamic_dialect("wasm", |ctx, op| {
            if is_emittable_wasm_operation(ctx, op) {
                LegalityDecision::Legal
            } else {
                LegalityDecision::Illegal
            }
        })
}

fn is_emittable_wasm_operation(ctx: &IrContext, op: OpRef) -> bool {
    if ctx.op(op).dialect != Symbol::new("wasm") {
        return false;
    }
    ctx.op(op).name.with_str(|name| {
        WASM_EMITTER_OPERATIONS
            .split_whitespace()
            .any(|supported| supported == name)
    })
}

const WASM_EMITTER_OPERATIONS: &str = r#"
block loop if br br_if return yield drop call call_indirect return_call
return_call_indirect unreachable nop func import_func export_func export_memory
memory data table elem global global_get global_set
i32_const i32_add i32_sub i32_mul i32_div_s i32_div_u i32_rem_s i32_rem_u
i32_eq i32_ne i32_lt_s i32_lt_u i32_le_s i32_le_u i32_gt_s i32_gt_u
i32_ge_s i32_ge_u i32_and i32_or i32_xor i32_shl i32_shr_s i32_shr_u
i64_const i64_add i64_sub i64_mul i64_div_s i64_div_u i64_rem_s i64_rem_u
i64_eq i64_ne i64_lt_s i64_lt_u i64_le_s i64_le_u i64_gt_s i64_gt_u
i64_ge_s i64_ge_u i64_and i64_or i64_xor i64_shl i64_shr_s i64_shr_u
f32_const f32_add f32_sub f32_mul f32_div f32_neg f32_eq f32_ne f32_lt f32_le
f32_gt f32_ge f64_const f64_add f64_sub f64_mul f64_div f64_neg f64_eq f64_ne
f64_lt f64_le f64_gt f64_ge local_get local_set local_tee struct_new struct_get
struct_set array_new array_new_default array_new_data bytes_from_data array_get
array_get_s array_get_u array_set array_len array_copy ref_null ref_func
ref_is_null ref_cast ref_test ref_i31 i31_get_s i31_get_u i32_wrap_i64
i64_extend_i32_s i64_extend_i32_u i32_trunc_f32_s i32_trunc_f32_u
i32_trunc_f64_s i32_trunc_f64_u i64_trunc_f32_s i64_trunc_f32_u
i64_trunc_f64_s i64_trunc_f64_u f32_convert_i32_s f32_convert_i32_u
f32_convert_i64_s f32_convert_i64_u f64_convert_i32_s f64_convert_i32_u
f64_convert_i64_s f64_convert_i64_u f32_demote_f64 f64_promote_f32
i32_reinterpret_f32 i64_reinterpret_f64 f32_reinterpret_i32 f64_reinterpret_i64
memory_size memory_grow i32_load i64_load f32_load f64_load i32_load8_s i32_load8_u
i32_load16_s i32_load16_u i64_load8_s i64_load8_u i64_load16_s i64_load16_u
i64_load32_s i64_load32_u i32_store i64_store f32_store f64_store i32_store8
i32_store16 i64_store8 i64_store16 i64_store32
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::parser::parse_test_module;

    #[test]
    fn rejects_unknown_wasm_operation() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  wasm.func @main() -> core.nil {
    wasm.unknown
    wasm.return
  }
}"#,
        );
        let error = validate_wasm_ir(&ctx, module).unwrap_err().to_string();
        assert!(error.contains("wasm.unknown"), "{error}");
    }
}
