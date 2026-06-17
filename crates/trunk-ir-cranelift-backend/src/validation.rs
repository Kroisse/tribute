//! IR validation for Cranelift backend.
//!
//! This module validates that IR is ready for emission:
//! - All operations must be explicitly legal for the native backend boundary.
//!
//! Dialect validation errors prevent emission from proceeding.

use trunk_ir::context::IrContext;
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

    if failures.is_empty() {
        return Ok(());
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
}
