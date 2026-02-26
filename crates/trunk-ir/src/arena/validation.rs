//! Value integrity and use-chain validation for arena-based IR.
//!
//! Arena IR uses mutable `IrContext` with explicit use-chains. This module
//! provides two kinds of validation:
//!
//! 1. **Scope validation**: Checks that every operand in a function references
//!    a value defined within that function's region tree (block args + op results).
//!    This mirrors the Salsa-based `validation.rs`.
//!
//! 2. **Use-chain consistency**: Checks that the use-chain stored in `IrContext`
//!    exactly matches the actual operands of all operations. This is unique to
//!    arena IR.

use std::collections::HashSet;
use std::fmt;

use super::context::IrContext;
use super::refs::{BlockRef, OpRef, RegionRef, ValueDef, ValueRef};
use super::rewrite::ArenaModule;
use super::walk;

use crate::Symbol;

// ============================================================================
// Error types
// ============================================================================

/// Describes a stale or invalid value found during validation.
pub struct StaleValueError {
    /// Name of the function containing the stale reference.
    pub function_name: String,
    /// Full name of the consuming operation (e.g., "func.call").
    pub consumer_op: String,
    /// Index of the stale operand within the consuming operation.
    pub operand_index: usize,
    /// Human-readable description of the stale value.
    pub stale_value_description: String,
}

impl fmt::Display for StaleValueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "stale value in @{}: operand #{} of {} references {}",
            self.function_name, self.operand_index, self.consumer_op, self.stale_value_description,
        )
    }
}

impl fmt::Debug for StaleValueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// Describes a use-chain inconsistency.
pub struct UseChainError {
    pub message: String,
}

impl fmt::Display for UseChainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl fmt::Debug for UseChainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// Result of validation.
pub struct ValidationResult {
    pub stale_errors: Vec<StaleValueError>,
    pub use_chain_errors: Vec<UseChainError>,
}

impl ValidationResult {
    pub fn is_ok(&self) -> bool {
        self.stale_errors.is_empty() && self.use_chain_errors.is_empty()
    }
}

impl fmt::Display for ValidationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_ok() {
            return write!(f, "validation passed");
        }
        if !self.stale_errors.is_empty() {
            writeln!(f, "{} stale value(s) found:", self.stale_errors.len())?;
            for err in &self.stale_errors {
                writeln!(f, "  - {}", err)?;
            }
        }
        if !self.use_chain_errors.is_empty() {
            writeln!(
                f,
                "{} use-chain error(s) found:",
                self.use_chain_errors.len()
            )?;
            for err in &self.use_chain_errors {
                writeln!(f, "  - {}", err)?;
            }
        }
        Ok(())
    }
}

// ============================================================================
// Scope validation (value integrity)
// ============================================================================

/// Recursively collect all values defined in a region (block args + op results),
/// including those in nested sub-regions.
fn collect_defined_in_region(ctx: &IrContext, region: RegionRef, defined: &mut HashSet<ValueRef>) {
    for &block in &ctx.region(region).blocks {
        collect_defined_in_block(ctx, block, defined);
    }
}

fn collect_defined_in_block(ctx: &IrContext, block: BlockRef, defined: &mut HashSet<ValueRef>) {
    // Block arguments
    for &arg in ctx.block_args(block) {
        defined.insert(arg);
    }

    // Operation results + recurse into nested regions
    for &op in &ctx.block(block).ops {
        for &result in ctx.op_results(op) {
            defined.insert(result);
        }
        for &nested_region in &ctx.op(op).regions {
            collect_defined_in_region(ctx, nested_region, defined);
        }
    }
}

/// Describe a value for diagnostic purposes.
fn describe_value(ctx: &IrContext, v: ValueRef) -> String {
    match ctx.value_def(v) {
        ValueDef::OpResult(op, idx) => {
            let data = ctx.op(op);
            let full_name = format!("{}.{}", data.dialect, data.name);
            match data.attributes.get(&Symbol::new("sym_name")) {
                Some(super::types::Attribute::Symbol(s)) => {
                    format!("result #{} of {} (@{})", idx, full_name, s)
                }
                _ => format!("result #{} of {}", idx, full_name),
            }
        }
        ValueDef::BlockArg(block, idx) => {
            format!("block arg #{} of {:?}", idx, block)
        }
    }
}

/// Check that all operands in a region reference values in `defined_set`.
fn check_operands_in_region(
    ctx: &IrContext,
    region: RegionRef,
    defined_set: &HashSet<ValueRef>,
    function_name: &str,
    errors: &mut Vec<StaleValueError>,
) {
    for &block in &ctx.region(region).blocks {
        check_operands_in_block(ctx, block, defined_set, function_name, errors);
    }
}

fn check_operands_in_block(
    ctx: &IrContext,
    block: BlockRef,
    defined_set: &HashSet<ValueRef>,
    function_name: &str,
    errors: &mut Vec<StaleValueError>,
) {
    for &op in &ctx.block(block).ops {
        for (i, &operand) in ctx.op_operands(op).iter().enumerate() {
            if !defined_set.contains(&operand) {
                let data = ctx.op(op);
                errors.push(StaleValueError {
                    function_name: function_name.to_string(),
                    consumer_op: format!("{}.{}", data.dialect, data.name),
                    operand_index: i,
                    stale_value_description: describe_value(ctx, operand),
                });
            }
        }
        // Check nested regions
        for &nested_region in &ctx.op(op).regions {
            check_operands_in_region(ctx, nested_region, defined_set, function_name, errors);
        }
    }
}

/// Validate value integrity for all `func.func` operations in a module.
///
/// For each function, checks that every operand references a value defined
/// within that function's region tree.
pub fn validate_value_integrity(ctx: &IrContext, module: ArenaModule) -> ValidationResult {
    let mut errors = Vec::new();

    let body = match module.body(ctx) {
        Some(r) => r,
        None => {
            return ValidationResult {
                stale_errors: errors,
                use_chain_errors: vec![],
            };
        }
    };

    validate_functions_in_region(ctx, body, &mut errors);

    ValidationResult {
        stale_errors: errors,
        use_chain_errors: vec![],
    }
}

fn validate_functions_in_region(
    ctx: &IrContext,
    region: RegionRef,
    errors: &mut Vec<StaleValueError>,
) {
    let func_dialect = Symbol::new("func");
    let func_name_sym = Symbol::new("func");

    for &block in &ctx.region(region).blocks {
        for &op in &ctx.block(block).ops {
            let data = ctx.op(op);
            if data.dialect == func_dialect && data.name == func_name_sym {
                // This is a func.func
                let fn_name = data
                    .attributes
                    .get(&Symbol::new("sym_name"))
                    .and_then(|a| match a {
                        super::types::Attribute::Symbol(s) => Some(s.to_string()),
                        _ => None,
                    })
                    .unwrap_or_else(|| "<unnamed>".to_string());

                // Collect defined values from all function regions
                let mut defined = HashSet::new();
                for &func_region in &data.regions {
                    collect_defined_in_region(ctx, func_region, &mut defined);
                }
                // Check operands
                for &func_region in &data.regions {
                    check_operands_in_region(ctx, func_region, &defined, &fn_name, errors);
                }
            }

            // Recurse into nested regions (e.g., nested core.module)
            for &nested_region in &data.regions {
                validate_functions_in_region(ctx, nested_region, errors);
            }
        }
    }
}

// ============================================================================
// Use-chain consistency validation
// ============================================================================

/// Validate that the use-chain stored in `IrContext` matches the actual operands.
///
/// Checks two directions:
/// 1. For every operand of every op, there must be a corresponding entry in `uses(operand)`.
/// 2. For every use in the use-chain, the referenced op's operand must point back.
pub fn validate_use_chains(ctx: &IrContext, module: ArenaModule) -> ValidationResult {
    let mut errors = Vec::new();

    let body = match module.body(ctx) {
        Some(r) => r,
        None => {
            return ValidationResult {
                stale_errors: vec![],
                use_chain_errors: errors,
            };
        }
    };

    // Collect all (value, use) pairs from actual operands
    let mut actual_uses: HashSet<(ValueRef, OpRef, u32)> = HashSet::new();

    walk::walk_region::<std::convert::Infallible>(ctx, body, &mut |op| {
        for (idx, &operand) in ctx.op_operands(op).iter().enumerate() {
            actual_uses.insert((operand, op, idx as u32));
        }
        std::ops::ControlFlow::Continue(walk::WalkAction::Advance)
    });

    // Direction 1: actual operand → use-chain entry must exist
    for &(val, op, idx) in &actual_uses {
        let found = ctx
            .uses(val)
            .iter()
            .any(|u| u.user == op && u.operand_index == idx);
        if !found {
            let data = ctx.op(op);
            errors.push(UseChainError {
                message: format!(
                    "operand #{} of {}.{} ({:?}) uses {:?} but no use-chain entry exists",
                    idx, data.dialect, data.name, op, val,
                ),
            });
        }
    }

    // Direction 2: use-chain entry → actual operand must exist
    // Collect all values that have uses
    let mut checked_values: HashSet<ValueRef> = HashSet::new();
    for &(val, _, _) in &actual_uses {
        checked_values.insert(val);
    }

    // Also check block args and op results that might have stale use-chain entries
    walk::walk_region::<std::convert::Infallible>(ctx, body, &mut |op| {
        for &result in ctx.op_results(op) {
            checked_values.insert(result);
        }
        std::ops::ControlFlow::Continue(walk::WalkAction::Advance)
    });
    for &block in &ctx.region(body).blocks {
        collect_block_values(ctx, block, &mut checked_values);
    }

    for &val in &checked_values {
        for u in ctx.uses(val) {
            if !actual_uses.contains(&(val, u.user, u.operand_index)) {
                errors.push(UseChainError {
                    message: format!(
                        "use-chain entry for {:?} claims use by {:?} operand #{}, but no such operand exists",
                        val, u.user, u.operand_index,
                    ),
                });
            }
        }
    }

    ValidationResult {
        stale_errors: vec![],
        use_chain_errors: errors,
    }
}

fn collect_block_values(ctx: &IrContext, block: BlockRef, values: &mut HashSet<ValueRef>) {
    for &arg in ctx.block_args(block) {
        values.insert(arg);
    }
    for &op in &ctx.block(block).ops {
        for &region in &ctx.op(op).regions {
            for &inner_block in &ctx.region(region).blocks {
                collect_block_values(ctx, inner_block, values);
            }
        }
    }
}

/// Run both validations and combine results.
pub fn validate_all(ctx: &IrContext, module: ArenaModule) -> ValidationResult {
    let scope = validate_value_integrity(ctx, module);
    let uses = validate_use_chains(ctx, module);
    ValidationResult {
        stale_errors: scope.stale_errors,
        use_chain_errors: uses.use_chain_errors,
    }
}

/// Debug-only validation that panics on any error.
///
/// Only runs under `cfg!(debug_assertions)`. Useful for checkpoints after
/// IR transformation passes.
pub fn debug_assert_valid(ctx: &IrContext, module: ArenaModule, pass_name: &str) {
    if !cfg!(debug_assertions) {
        return;
    }
    let result = validate_all(ctx, module);
    if !result.is_ok() {
        panic!("Arena validation failed after `{}`:\n{}", pass_name, result,);
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Symbol;
    use crate::arena::context::OperationDataBuilder;
    use crate::arena::dialect::{arith, core, func};
    use crate::arena::types::{Attribute, Location};
    use crate::arena::{BlockArgData, BlockData, IrContext, RegionData, TypeDataBuilder};
    use crate::location::Span;
    use smallvec::smallvec;
    use std::collections::BTreeMap;

    fn test_location(ctx: &mut IrContext) -> Location {
        let path = ctx.paths.intern("test.trb".to_owned());
        Location::new(path, Span::new(0, 0))
    }

    fn make_i32_type(ctx: &mut IrContext) -> super::super::refs::TypeRef {
        ctx.types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build())
    }

    fn make_func_type(
        ctx: &mut IrContext,
        params: &[super::super::refs::TypeRef],
        ret: super::super::refs::TypeRef,
    ) -> super::super::refs::TypeRef {
        ctx.types.intern(
            TypeDataBuilder::new(Symbol::new("func"), Symbol::new("fn"))
                .param(ret)
                .params(params.iter().copied())
                .build(),
        )
    }

    /// Build a valid module: fn add() { 40 + 2 }
    fn build_valid_module(ctx: &mut IrContext) -> ArenaModule {
        let loc = test_location(ctx);
        let i32_ty = make_i32_type(ctx);

        let entry_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });

        let c0 = arith::r#const(ctx, loc, i32_ty, Attribute::IntBits(40));
        ctx.push_op(entry_block, c0.op_ref());
        let c0_val = c0.result(ctx);

        let c1 = arith::r#const(ctx, loc, i32_ty, Attribute::IntBits(2));
        ctx.push_op(entry_block, c1.op_ref());
        let c1_val = c1.result(ctx);

        let add_op = arith::add(ctx, loc, c0_val, c1_val, i32_ty);
        ctx.push_op(entry_block, add_op.op_ref());
        let sum = add_op.result(ctx);

        let ret = func::r#return(ctx, loc, [sum]);
        ctx.push_op(entry_block, ret.op_ref());

        let body_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry_block],
            parent_op: None,
        });

        let func_ty = make_func_type(ctx, &[], i32_ty);
        let func_op = func::func(ctx, loc, Symbol::new("add"), func_ty, body_region);

        let mod_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        ctx.push_op(mod_block, func_op.op_ref());

        let mod_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![mod_block],
            parent_op: None,
        });
        let module = core::module(ctx, loc, Symbol::new("test"), mod_region);

        ArenaModule::new(ctx, module.op_ref()).unwrap()
    }

    #[test]
    fn valid_module_passes() {
        let mut ctx = IrContext::new();
        let module = build_valid_module(&mut ctx);
        let result = validate_value_integrity(&ctx, module);
        assert!(result.is_ok(), "Valid module should pass: {}", result);
    }

    #[test]
    fn stale_op_result_detected() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = make_i32_type(&mut ctx);

        // func_a with a constant
        let entry_a = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let const_a = arith::r#const(&mut ctx, loc, i32_ty, Attribute::IntBits(99));
        ctx.push_op(entry_a, const_a.op_ref());
        let stale_value = const_a.result(&ctx);
        let ret_a = func::r#return(&mut ctx, loc, [stale_value]);
        ctx.push_op(entry_a, ret_a.op_ref());

        let body_a = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry_a],
            parent_op: None,
        });
        let func_ty = make_func_type(&mut ctx, &[], i32_ty);
        let func_a = func::func(&mut ctx, loc, Symbol::new("func_a"), func_ty, body_a);

        // func_b uses stale_value from func_a
        let entry_b = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let ret_b = func::r#return(&mut ctx, loc, [stale_value]);
        ctx.push_op(entry_b, ret_b.op_ref());

        let body_b = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry_b],
            parent_op: None,
        });
        let func_b = func::func(&mut ctx, loc, Symbol::new("func_b"), func_ty, body_b);

        // module
        let mod_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        ctx.push_op(mod_block, func_a.op_ref());
        ctx.push_op(mod_block, func_b.op_ref());
        let mod_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![mod_block],
            parent_op: None,
        });
        let module_op = core::module(&mut ctx, loc, Symbol::new("test"), mod_region);
        let module = ArenaModule::new(&ctx, module_op.op_ref()).unwrap();

        let result = validate_value_integrity(&ctx, module);
        assert!(!result.is_ok(), "Should detect stale op result");
        assert_eq!(result.stale_errors.len(), 1);
        assert_eq!(result.stale_errors[0].function_name, "func_b");
        assert!(
            result.stale_errors[0]
                .stale_value_description
                .contains("arith.const")
        );
    }

    #[test]
    fn stale_block_arg_detected() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = make_i32_type(&mut ctx);

        // func_a with a parameter
        let entry_a = ctx.create_block(BlockData {
            location: loc,
            args: vec![BlockArgData {
                ty: i32_ty,
                attrs: BTreeMap::new(),
            }],
            ops: smallvec![],
            parent_region: None,
        });
        let stale_block_arg = ctx.block_arg(entry_a, 0);
        let ret_a = func::r#return(&mut ctx, loc, [stale_block_arg]);
        ctx.push_op(entry_a, ret_a.op_ref());

        let body_a = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry_a],
            parent_op: None,
        });
        let func_ty = make_func_type(&mut ctx, &[i32_ty], i32_ty);
        let func_a = func::func(&mut ctx, loc, Symbol::new("func_a"), func_ty, body_a);

        // func_b uses the block arg from func_a
        let entry_b = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let ret_b = func::r#return(&mut ctx, loc, [stale_block_arg]);
        ctx.push_op(entry_b, ret_b.op_ref());

        let body_b = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry_b],
            parent_op: None,
        });
        let func_ty_b = make_func_type(&mut ctx, &[], i32_ty);
        let func_b = func::func(&mut ctx, loc, Symbol::new("func_b"), func_ty_b, body_b);

        let mod_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        ctx.push_op(mod_block, func_a.op_ref());
        ctx.push_op(mod_block, func_b.op_ref());
        let mod_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![mod_block],
            parent_op: None,
        });
        let module_op = core::module(&mut ctx, loc, Symbol::new("test"), mod_region);
        let module = ArenaModule::new(&ctx, module_op.op_ref()).unwrap();

        let result = validate_value_integrity(&ctx, module);
        assert!(!result.is_ok(), "Should detect stale block arg");
        assert_eq!(result.stale_errors.len(), 1);
        assert_eq!(result.stale_errors[0].function_name, "func_b");
        assert!(
            result.stale_errors[0]
                .stale_value_description
                .contains("block arg")
        );
    }

    #[test]
    fn nested_region_cross_ref_valid() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = make_i32_type(&mut ctx);

        // func with a param, scf.if that references param from outer scope
        let entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![BlockArgData {
                ty: i32_ty,
                attrs: BTreeMap::new(),
            }],
            ops: smallvec![],
            parent_region: None,
        });
        let param = ctx.block_arg(entry, 0);

        // then branch: yield param
        let then_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let yield_then = OperationDataBuilder::new(loc, Symbol::new("scf"), Symbol::new("yield"))
            .operand(param)
            .build(&mut ctx);
        let yield_then = ctx.create_op(yield_then);
        ctx.push_op(then_block, yield_then);
        let then_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![then_block],
            parent_op: None,
        });

        // else branch: const 1, add param+1, yield
        let else_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let c1 = arith::r#const(&mut ctx, loc, i32_ty, Attribute::IntBits(1));
        ctx.push_op(else_block, c1.op_ref());
        let c1_val = c1.result(&ctx);
        let sum = arith::add(&mut ctx, loc, param, c1_val, i32_ty);
        ctx.push_op(else_block, sum.op_ref());
        let sum_val = sum.result(&ctx);
        let yield_else = OperationDataBuilder::new(loc, Symbol::new("scf"), Symbol::new("yield"))
            .operand(sum_val)
            .build(&mut ctx);
        let yield_else = ctx.create_op(yield_else);
        ctx.push_op(else_block, yield_else);
        let else_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![else_block],
            parent_op: None,
        });

        // Create a bool condition
        let i1_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i1")).build());
        let cond = arith::r#const(&mut ctx, loc, i1_ty, Attribute::IntBits(1));
        ctx.push_op(entry, cond.op_ref());
        let cond_val = cond.result(&ctx);

        // scf.if
        let if_op = OperationDataBuilder::new(loc, Symbol::new("scf"), Symbol::new("if"))
            .operand(cond_val)
            .result(i32_ty)
            .region(then_region)
            .region(else_region)
            .build(&mut ctx);
        let if_op = ctx.create_op(if_op);
        ctx.push_op(entry, if_op);
        let if_result = ctx.op_result(if_op, 0);

        let ret = func::r#return(&mut ctx, loc, [if_result]);
        ctx.push_op(entry, ret.op_ref());

        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });
        let func_ty = make_func_type(&mut ctx, &[i32_ty], i32_ty);
        let func_op = func::func(&mut ctx, loc, Symbol::new("nested_fn"), func_ty, body);

        let mod_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        ctx.push_op(mod_block, func_op.op_ref());
        let mod_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![mod_block],
            parent_op: None,
        });
        let module_op = core::module(&mut ctx, loc, Symbol::new("test"), mod_region);
        let module = ArenaModule::new(&ctx, module_op.op_ref()).unwrap();

        let result = validate_value_integrity(&ctx, module);
        assert!(
            result.is_ok(),
            "Inner region referencing outer block arg should be valid: {}",
            result,
        );
    }

    #[test]
    fn cross_function_ref_invalid() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = make_i32_type(&mut ctx);

        // func_a
        let entry_a = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let const_a = arith::r#const(&mut ctx, loc, i32_ty, Attribute::IntBits(42));
        ctx.push_op(entry_a, const_a.op_ref());
        let value_from_a = const_a.result(&ctx);
        let ret_a = func::r#return(&mut ctx, loc, [value_from_a]);
        ctx.push_op(entry_a, ret_a.op_ref());
        let body_a = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry_a],
            parent_op: None,
        });
        let func_ty = make_func_type(&mut ctx, &[], i32_ty);
        let func_a = func::func(&mut ctx, loc, Symbol::new("func_a"), func_ty, body_a);

        // func_b uses value_from_a (stale!)
        let entry_b = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let local = arith::r#const(&mut ctx, loc, i32_ty, Attribute::IntBits(1));
        ctx.push_op(entry_b, local.op_ref());
        let local_val = local.result(&ctx);
        let add_op = arith::add(&mut ctx, loc, value_from_a, local_val, i32_ty);
        ctx.push_op(entry_b, add_op.op_ref());
        let ret_b = func::r#return(&mut ctx, loc, [value_from_a]);
        ctx.push_op(entry_b, ret_b.op_ref());
        let body_b = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry_b],
            parent_op: None,
        });
        let func_b = func::func(&mut ctx, loc, Symbol::new("func_b"), func_ty, body_b);

        let mod_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        ctx.push_op(mod_block, func_a.op_ref());
        ctx.push_op(mod_block, func_b.op_ref());
        let mod_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![mod_block],
            parent_op: None,
        });
        let module_op = core::module(&mut ctx, loc, Symbol::new("test"), mod_region);
        let module = ArenaModule::new(&ctx, module_op.op_ref()).unwrap();

        let result = validate_value_integrity(&ctx, module);
        assert!(
            !result.is_ok(),
            "Cross-function value ref should be invalid"
        );
        assert_eq!(result.stale_errors.len(), 2);
        for err in &result.stale_errors {
            assert_eq!(err.function_name, "func_b");
        }
    }

    #[test]
    fn use_chain_valid() {
        let mut ctx = IrContext::new();
        let module = build_valid_module(&mut ctx);
        let result = validate_use_chains(&ctx, module);
        assert!(result.is_ok(), "Use chains should be valid: {}", result);
    }

    #[test]
    fn rauw_preserves_use_chain_validity() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = make_i32_type(&mut ctx);

        let entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });

        let c0 = arith::r#const(&mut ctx, loc, i32_ty, Attribute::IntBits(40));
        ctx.push_op(entry, c0.op_ref());
        let c0_val = c0.result(&ctx);

        let c1 = arith::r#const(&mut ctx, loc, i32_ty, Attribute::IntBits(2));
        ctx.push_op(entry, c1.op_ref());
        let c1_val = c1.result(&ctx);

        // Use c0 in two places
        let add = arith::add(&mut ctx, loc, c0_val, c0_val, i32_ty);
        ctx.push_op(entry, add.op_ref());
        let add_val = add.result(&ctx);

        let ret = func::r#return(&mut ctx, loc, [add_val]);
        ctx.push_op(entry, ret.op_ref());

        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });
        let func_ty = make_func_type(&mut ctx, &[], i32_ty);
        let func_op = func::func(&mut ctx, loc, Symbol::new("f"), func_ty, body);

        let mod_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        ctx.push_op(mod_block, func_op.op_ref());
        let mod_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![mod_block],
            parent_op: None,
        });
        let module_op = core::module(&mut ctx, loc, Symbol::new("test"), mod_region);
        let module = ArenaModule::new(&ctx, module_op.op_ref()).unwrap();

        // Validate before RAUW
        let result = validate_all(&ctx, module);
        assert!(result.is_ok(), "Before RAUW: {}", result);

        // Replace c0 with c1
        ctx.replace_all_uses(c0_val, c1_val);

        // Validate after RAUW
        let result = validate_all(&ctx, module);
        assert!(result.is_ok(), "After RAUW: {}", result);

        // Verify c1 now has the uses
        assert!(!ctx.has_uses(c0_val));
        assert!(ctx.has_uses(c1_val));
    }
}
