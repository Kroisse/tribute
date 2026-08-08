//! IR validation for Cranelift backend.
//!
//! This module validates that IR is ready for emission:
//! - All operations must be explicitly legal for the native backend boundary.
//!
//! Dialect validation errors prevent emission from proceeding.

use trunk_ir::context::IrContext;
use trunk_ir::dialect::{clif, core};
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::rewrite::{ConversionTarget, LegalityDecision, Module};
use trunk_ir::walk::{WalkAction, walk_region};

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

    if failures.is_empty() {
        return validate_indirect_call_signatures(ctx, body)
            .map_err(CompilationError::ir_validation);
    }

    let errors: Vec<String> = failures
        .into_iter()
        .map(|op| format!("{} in boundary {}", op, NATIVE_BACKEND_READY_BOUNDARY))
        .collect();
    let message = format!(
        "IR validation failed for boundary {NATIVE_BACKEND_READY_BOUNDARY} with {} error(s):\n  - {}",
        errors.len(),
        errors.join("\n  - ")
    );
    Err(CompilationError::ir_validation(message))
}

/// Ensure that an indirect call's explicit signature describes the actual
/// SSA arguments that reach Cranelift. This prevents a pointer-sized closure
/// representation from silently replacing an `i32` source-data slot.
fn validate_indirect_call_signatures(
    ctx: &IrContext,
    body: trunk_ir::RegionRef,
) -> Result<(), String> {
    let mut error = None;
    let _ = walk_region::<()>(ctx, body, &mut |op| {
        if error.is_some() {
            return std::ops::ControlFlow::Break(());
        }
        let is_tail = clif::ReturnCallIndirect::matches(ctx, op);
        if !is_tail && !clif::CallIndirect::matches(ctx, op) {
            return std::ops::ControlFlow::Continue(WalkAction::Advance);
        }

        let Some(sig_ty) = ctx.op(op).attributes.get_type("sig") else {
            error = Some(format!(
                "indirect call {}.{} has no exact `sig` attribute",
                ctx.op(op).dialect,
                ctx.op(op).name
            ));
            return std::ops::ControlFlow::Break(());
        };
        let Some(signature) = core::Func::from_type_ref(ctx, sig_ty) else {
            error = Some("indirect call `sig` is not a core.func type".to_owned());
            return std::ops::ControlFlow::Break(());
        };
        let operands = ctx.op_operands(op);
        let params = signature.params(ctx);
        if operands.len() != params.len() + 1
            || operands[1..]
                .iter()
                .zip(params)
                .any(|(&operand, &expected)| ctx.value_ty(operand) != expected)
        {
            error = Some(format!(
                "indirect call {}.{} operands do not match its exact signature",
                ctx.op(op).dialect,
                ctx.op(op).name
            ));
            return std::ops::ControlFlow::Break(());
        }
        if is_tail && !is_core_nil(ctx, signature.r#return(ctx)) {
            error = Some("indirect proper-tail call must have a core.nil result".to_owned());
            return std::ops::ControlFlow::Break(());
        }
        std::ops::ControlFlow::Continue(WalkAction::Advance)
    });
    error.map_or(Ok(()), Err)
}

fn is_core_nil(ctx: &IrContext, ty: trunk_ir::TypeRef) -> bool {
    let data = ctx.types.get(ty);
    data.dialect == trunk_ir::Symbol::new("core") && data.name == trunk_ir::Symbol::new("nil")
}

/// Conversion target for IR that is ready for Cranelift emission.
pub fn native_backend_ready_target() -> ConversionTarget {
    ConversionTarget::new()
        .legal_op("core", "module")
        .dynamic_dialect("clif", |ctx, op| {
            if is_emittable_clif_operation(ctx, op) {
                LegalityDecision::Legal
            } else {
                LegalityDecision::Illegal
            }
        })
}

fn is_emittable_clif_operation(ctx: &IrContext, op: trunk_ir::OpRef) -> bool {
    if ctx.op(op).dialect != trunk_ir::Symbol::new("clif") {
        return false;
    }
    ctx.op(op).name.with_str(|name| {
        CLIF_EMITTER_OPERATIONS
            .split_whitespace()
            .any(|supported| supported == name)
    })
}

const CLIF_EMITTER_OPERATIONS: &str = r#"
func call call_indirect return iconst f32const f64const iadd isub imul sdiv udiv srem urem
ineg fadd fsub fmul fdiv fneg icmp fcmp band bor bxor ishl sshr ushr brif jump br_table
trap return_call return_call_indirect load store atomic_rmw stack_slot stack_addr symbol_addr
ireduce uextend sextend fpromote fdemote fcvt_to_sint fcvt_from_sint fcvt_to_uint fcvt_from_uint
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::OperationDataBuilder;
    use trunk_ir::context::{BlockData, IrContext, RegionData};
    use trunk_ir::location::Span;
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

    #[test]
    fn native_backend_ready_allows_clif_ops() {
        let (mut ctx, loc) = test_ctx();
        let module = make_module(&mut ctx, loc, "clif", "func");

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
    fn native_backend_ready_rejects_unknown_clif_operations() {
        let (mut ctx, loc) = test_ctx();
        let module = make_module(&mut ctx, loc, "clif", "unknown");

        let error = validate_clif_ir(&ctx, module).unwrap_err().to_string();

        assert!(error.contains("clif.unknown"), "{error}");
    }

    #[test]
    fn native_backend_ready_rejects_indirect_signature_type_mismatch() {
        let mut ctx = IrContext::new();
        let module = trunk_ir::parser::parse_test_module(
            &mut ctx,
            r#"core.module @test {
  clif.func @caller(%callee: core.ptr, %value: core.i32) -> core.nil {
    clif.return_call_indirect %callee, %value {sig = core.func(core.nil, core.i64)}
  }
}"#,
        );

        let error = validate_clif_ir(&ctx, module).unwrap_err().to_string();
        assert!(error.contains("operands do not match its exact signature"));
    }
}
