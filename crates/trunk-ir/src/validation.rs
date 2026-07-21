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
//!
//! 3. **Operation verifiers**: Check local operation invariants that do not
//!    require whole-IR analysis or conversion-boundary state.

use std::collections::{HashMap, HashSet};
use std::fmt;

use derive_more::{Display, Error};

use super::context::IrContext;
use super::refs::{BlockRef, OpRef, RegionRef, ValueDef, ValueRef};
use super::rewrite::Module;
use super::walk;

use crate::Symbol;

// ============================================================================
// Error types
// ============================================================================

/// Describes an IR validation error.
#[derive(Display, Error)]
pub enum ValidationError {
    /// A stale or invalid value was found during scope validation.
    #[display(
        "stale value in @{function_name}: operand #{operand_index} of {consumer_op} references {stale_value_description}"
    )]
    StaleValue {
        /// Name of the function containing the stale reference.
        function_name: String,
        /// Full name of the consuming operation (e.g., "func.call").
        consumer_op: String,
        /// Index of the stale operand within the consuming operation.
        operand_index: usize,
        /// Human-readable description of the stale value.
        stale_value_description: String,
    },
    /// A use-chain inconsistency was found.
    #[display("{message}")]
    UseChain { message: String },
    /// An operation-level verifier error was found.
    #[display("{message}")]
    Operation { message: String },
}

impl fmt::Debug for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// Result of validation.
pub struct ValidationResult {
    pub errors: Vec<ValidationError>,
}

impl ValidationResult {
    pub fn is_ok(&self) -> bool {
        self.errors.is_empty()
    }
}

impl fmt::Display for ValidationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_ok() {
            return write!(f, "validation passed");
        }
        writeln!(f, "{} validation error(s) found:", self.errors.len())?;
        for err in &self.errors {
            writeln!(f, "  - {}", err)?;
        }
        Ok(())
    }
}

// ============================================================================
// Scope validation (value integrity)
// ============================================================================

/// Collect values defined directly in a region (block args + op results).
///
/// This is a **shallow** collection: it does NOT recurse into nested
/// sub-regions of operations. Values defined inside sub-regions are not
/// visible to the parent or sibling scopes.
fn collect_region_top_level(ctx: &IrContext, region: RegionRef, defined: &mut HashSet<ValueRef>) {
    for &block in &ctx.region(region).blocks {
        for &arg in ctx.block_args(block) {
            defined.insert(arg);
        }
        for &op in &ctx.block(block).ops {
            for &result in ctx.op_results(op) {
                defined.insert(result);
            }
        }
    }
}

/// Describe a value for diagnostic purposes.
fn describe_value(ctx: &IrContext, v: ValueRef) -> String {
    match ctx.value_def(v) {
        ValueDef::OpResult(op, idx) => {
            let data = ctx.op(op);
            let full_name = format!("{}.{}", data.dialect, data.name);
            match data.attributes.get("sym_name") {
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

/// Check that all operands in a region reference visible values.
///
/// `outer_visible` contains every value visible from ancestor scopes.
/// Values defined at this region level are added to the visible set before
/// checking operands, and the extended set is propagated into nested
/// sub-regions. Values defined inside sub-regions are **never** added back
/// to the outer set, enforcing directional visibility across region
/// boundaries.
fn check_operands_in_region(
    ctx: &IrContext,
    region: RegionRef,
    outer_visible: &HashSet<ValueRef>,
    function_name: &str,
    errors: &mut Vec<ValidationError>,
) {
    // Extend with values defined at this region level (shallow – no sub-regions).
    let mut visible = outer_visible.clone();
    collect_region_top_level(ctx, region, &mut visible);

    for &block in &ctx.region(region).blocks {
        for &op in &ctx.block(block).ops {
            for (i, &operand) in ctx.op_operands(op).iter().enumerate() {
                if !visible.contains(&operand) {
                    let data = ctx.op(op);
                    errors.push(ValidationError::StaleValue {
                        function_name: function_name.to_string(),
                        consumer_op: format!("{}.{}", data.dialect, data.name),
                        operand_index: i,
                        stale_value_description: describe_value(ctx, operand),
                    });
                }
            }
            // Propagate the extended visible set into nested regions.
            for &nested_region in &ctx.op(op).regions {
                check_operands_in_region(ctx, nested_region, &visible, function_name, errors);
            }
        }
    }
}

/// Validate value integrity for all `func.func` and `wasm.func` operations in a module.
///
/// For each function, checks that every operand references a value defined
/// within that function's region tree.
pub fn validate_value_integrity(ctx: &IrContext, module: Module) -> ValidationResult {
    let mut errors = Vec::new();

    let body = match module.body(ctx) {
        Some(r) => r,
        None => {
            return ValidationResult { errors };
        }
    };

    validate_functions_in_region(ctx, body, &mut errors);

    ValidationResult { errors }
}

fn validate_functions_in_region(
    ctx: &IrContext,
    region: RegionRef,
    errors: &mut Vec<ValidationError>,
) {
    let func_dialect = Symbol::new("func");
    let wasm_dialect = Symbol::new("wasm");
    let func_name_sym = Symbol::new("func");

    for &block in &ctx.region(region).blocks {
        for &op in &ctx.block(block).ops {
            let data = ctx.op(op);
            let is_function = (data.dialect == func_dialect || data.dialect == wasm_dialect)
                && data.name == func_name_sym;
            if is_function {
                // This is a func.func or wasm.func
                let fn_name = data
                    .attributes
                    .get_symbol("sym_name")
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "<unnamed>".to_string());

                // Check operands with visibility-based scoping.
                // No values from outside the function body are visible.
                for &func_region in &data.regions {
                    check_operands_in_region(ctx, func_region, &HashSet::new(), &fn_name, errors);
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
pub fn validate_use_chains(ctx: &IrContext, module: Module) -> ValidationResult {
    let mut errors = Vec::new();

    let body = match module.body(ctx) {
        Some(r) => r,
        None => {
            return ValidationResult { errors };
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
            errors.push(ValidationError::UseChain {
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
                errors.push(ValidationError::UseChain {
                    message: format!(
                        "use-chain entry for {:?} claims use by {:?} operand #{}, but no such operand exists",
                        val, u.user, u.operand_index,
                    ),
                });
            }
        }
    }

    ValidationResult { errors }
}

// ============================================================================
// Operation verifier validation
// ============================================================================

/// Validate local operation-level invariants.
///
/// Operation verifiers are for constraints that can be checked from one
/// operation and its immediate shape: required attributes, supported attribute
/// values, operand/result arity, region count, and terminator requirements.
/// They must not encode conversion-boundary legality or graph-wide invariants.
pub fn validate_operation_verifiers(ctx: &IrContext, module: Module) -> ValidationResult {
    let mut errors = Vec::new();

    let body = match module.body(ctx) {
        Some(r) => r,
        None => {
            return ValidationResult { errors };
        }
    };

    walk::walk_region::<std::convert::Infallible>(ctx, body, &mut |op| {
        validate_arith_cmpf_predicate(ctx, op, &mut errors);
        validate_scf_if_structure(ctx, op, &mut errors);
        std::ops::ControlFlow::Continue(walk::WalkAction::Advance)
    });

    ValidationResult { errors }
}

/// Validate operation-level semantic constraints that are independent of
/// value scope/use-chain integrity.
///
/// Deprecated compatibility alias for callers that have not migrated to
/// [`validate_operation_verifiers`].
#[deprecated(
    since = "0.1.0",
    note = "use validate_operation_verifiers for local operation invariant checks"
)]
pub fn validate_operation_semantics(ctx: &IrContext, module: Module) -> ValidationResult {
    validate_operation_verifiers(ctx, module)
}

fn validate_arith_cmpf_predicate(ctx: &IrContext, op: OpRef, errors: &mut Vec<ValidationError>) {
    let data = ctx.op(op);
    if data.dialect != Symbol::new("arith") || data.name != Symbol::new("cmpf") {
        return;
    }

    let Some(predicate) = data.attributes.get_symbol("predicate") else {
        errors.push(operation_verifier_error(
            ctx,
            op,
            "requires symbol predicate attribute",
        ));
        return;
    };

    if !is_allowed_cmpf_predicate(predicate) {
        errors.push(operation_verifier_error(
            ctx,
            op,
            format!(
                "has unsupported predicate '{}'; supported predicates are {}",
                predicate,
                supported_cmpf_predicates_text(),
            ),
        ));
    }
}

fn operation_verifier_error(
    ctx: &IrContext,
    op: OpRef,
    detail: impl Into<String>,
) -> ValidationError {
    let data = ctx.op(op);
    ValidationError::Operation {
        message: format!(
            "operation verifier failed for {}.{} ({}): {}",
            data.dialect,
            data.name,
            op,
            detail.into(),
        ),
    }
}

const SUPPORTED_CMPF_PREDICATES: [&str; 6] = ["oeq", "une", "olt", "ole", "ogt", "oge"];

fn supported_cmpf_predicates_text() -> String {
    SUPPORTED_CMPF_PREDICATES.join(", ")
}

fn is_allowed_cmpf_predicate(predicate: Symbol) -> bool {
    predicate == Symbol::new(SUPPORTED_CMPF_PREDICATES[0])
        || predicate == Symbol::new(SUPPORTED_CMPF_PREDICATES[1])
        || predicate == Symbol::new(SUPPORTED_CMPF_PREDICATES[2])
        || predicate == Symbol::new(SUPPORTED_CMPF_PREDICATES[3])
        || predicate == Symbol::new(SUPPORTED_CMPF_PREDICATES[4])
        || predicate == Symbol::new(SUPPORTED_CMPF_PREDICATES[5])
}

fn validate_scf_if_structure(ctx: &IrContext, op: OpRef, errors: &mut Vec<ValidationError>) {
    let data = ctx.op(op);
    if data.dialect != Symbol::new("scf") || data.name != Symbol::new("if") {
        return;
    }

    if ctx.op_operands(op).len() != 1 {
        errors.push(operation_verifier_error(
            ctx,
            op,
            format!(
                "expects 1 condition operand, found {}",
                ctx.op_operands(op).len()
            ),
        ));
    }

    if data.regions.len() != 2 {
        errors.push(operation_verifier_error(
            ctx,
            op,
            format!("expects 2 regions, found {}", data.regions.len()),
        ));
        return;
    }

    for (region_name, &region) in [
        ("then_region", &data.regions[0]),
        ("else_region", &data.regions[1]),
    ] {
        let blocks = &ctx.region(region).blocks;
        let [block] = blocks.as_slice() else {
            errors.push(operation_verifier_error(
                ctx,
                op,
                format!("{region_name} expects 1 block, found {}", blocks.len()),
            ));
            continue;
        };

        let Some((&yield_op, _body_ops)) = ctx.block(*block).ops.split_last() else {
            errors.push(operation_verifier_error(
                ctx,
                op,
                format!("{region_name} must terminate with scf.yield"),
            ));
            continue;
        };

        let yield_data = ctx.op(yield_op);
        if yield_data.dialect != Symbol::new("scf") || yield_data.name != Symbol::new("yield") {
            errors.push(operation_verifier_error(
                ctx,
                op,
                format!("{region_name} must terminate with scf.yield"),
            ));
            continue;
        }

        let yield_arity = ctx.op_operands(yield_op).len();
        let result_arity = ctx.op_results(op).len();
        if yield_arity != result_arity {
            errors.push(operation_verifier_error(
                ctx,
                op,
                format!(
                    "{region_name} yields {yield_arity} value(s), but scf.if has {result_arity} result(s)"
                ),
            ));
        }
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

// ============================================================================
// Call arity validation
// ============================================================================

/// Collect function signatures from module-level function definitions.
///
/// Builds a map from function symbol to expected parameter count by inspecting
/// `func.func`, `wasm.func`, and `clif.func` operations.
fn collect_function_signatures(ctx: &IrContext, module_body: RegionRef) -> HashMap<Symbol, usize> {
    let func_name_sym = Symbol::new("func");
    let func_dialect = Symbol::new("func");
    let wasm_dialect = Symbol::new("wasm");
    let clif_dialect = Symbol::new("clif");

    let core_dialect = Symbol::new("core");
    let core_func_name = Symbol::new("func");
    let sym_name_key = Symbol::new("sym_name");
    let type_key = Symbol::new("type");

    let mut signatures = HashMap::new();

    for &block in &ctx.region(module_body).blocks {
        for &op in &ctx.block(block).ops {
            let data = ctx.op(op);
            let is_function = (data.dialect == func_dialect
                || data.dialect == wasm_dialect
                || data.dialect == clif_dialect)
                && data.name == func_name_sym;
            if !is_function {
                continue;
            }

            let Some(sym_name) = data.attributes.get_symbol(sym_name_key) else {
                continue;
            };

            let Some(func_ty) = data.attributes.get_type(type_key) else {
                continue;
            };

            let ty_data = ctx.types.get(func_ty);
            if ty_data.dialect != core_dialect || ty_data.name != core_func_name {
                continue;
            }

            // core.func layout: params[0] = Return, params[1..] = Params
            let param_count = ty_data.params.len().saturating_sub(1);
            signatures.insert(sym_name, param_count);
        }
    }

    signatures
}

/// Walk all operations in a region tree and check call arity.
fn check_call_arity_in_region(
    ctx: &IrContext,
    region: RegionRef,
    signatures: &HashMap<Symbol, usize>,
    enclosing_fn: &str,
) {
    let func_dialect = Symbol::new("func");
    let call_name = Symbol::new("call");
    let tail_call_name = Symbol::new("tail_call");
    let callee_key = Symbol::new("callee");

    walk::walk_region::<std::convert::Infallible>(ctx, region, &mut |op| {
        let data = ctx.op(op);
        if data.dialect != func_dialect {
            return std::ops::ControlFlow::Continue(walk::WalkAction::Advance);
        }

        let is_call = data.name == call_name;
        let is_tail_call = data.name == tail_call_name;
        if !is_call && !is_tail_call {
            return std::ops::ControlFlow::Continue(walk::WalkAction::Advance);
        }

        let Some(callee_sym) = data.attributes.get_symbol(callee_key) else {
            return std::ops::ControlFlow::Continue(walk::WalkAction::Advance);
        };

        if let Some(&expected) = signatures.get(&callee_sym) {
            let actual = ctx.op_operands(op).len();
            if actual != expected {
                ctx.report_warning(
                    data.location.span,
                    format!(
                        "arity mismatch in '{}': call to '{}' has {} argument(s), expected {}",
                        enclosing_fn, callee_sym, actual, expected,
                    ),
                );
            }
        }

        std::ops::ControlFlow::Continue(walk::WalkAction::Advance)
    });
}

/// Validate that all `func.call` and `func.tail_call` operations have the
/// correct number of arguments matching the callee's function signature.
///
/// Arity mismatches are reported as warnings via `ctx.report_warning`.
pub fn validate_call_arity(ctx: &IrContext, module: Module) {
    let Some(body) = module.body(ctx) else {
        return;
    };

    let signatures = collect_function_signatures(ctx, body);

    // Walk each function definition and check call sites within
    let func_name_sym = Symbol::new("func");
    let func_dialect = Symbol::new("func");
    let wasm_dialect = Symbol::new("wasm");
    let clif_dialect = Symbol::new("clif");
    let sym_name_key = Symbol::new("sym_name");

    for &block in &ctx.region(body).blocks {
        for &op in &ctx.block(block).ops {
            let data = ctx.op(op);
            let is_function = (data.dialect == func_dialect
                || data.dialect == wasm_dialect
                || data.dialect == clif_dialect)
                && data.name == func_name_sym;
            if !is_function {
                continue;
            }

            let fn_name = data
                .attributes
                .get_symbol(sym_name_key)
                .map(|s| s.to_string())
                .unwrap_or_else(|| "<unnamed>".to_string());

            for &func_region in &data.regions {
                check_call_arity_in_region(ctx, func_region, &signatures, &fn_name);
            }
        }
    }
}

/// Run all validations and combine results.
///
/// Note: arity errors from `validate_call_arity` are intentionally excluded
/// because they are currently non-blocking warnings (see TODO(#582) in
/// pipeline). Use `validate_call_arity` directly when arity diagnostics
/// are needed.
pub fn validate_all(ctx: &IrContext, module: Module) -> ValidationResult {
    let scope = validate_value_integrity(ctx, module);
    let uses = validate_use_chains(ctx, module);
    let ops = validate_operation_verifiers(ctx, module);
    let mut errors = scope.errors;
    errors.extend(uses.errors);
    errors.extend(ops.errors);
    ValidationResult { errors }
}

/// Debug-only validation that panics on any error.
///
/// Only runs under `cfg!(debug_assertions)`. Useful for checkpoints after
/// IR transformation passes.
pub fn debug_assert_valid(ctx: &IrContext, module: Module, pass_name: &str) {
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
    use crate::context::OperationDataBuilder;
    use crate::dialect::{arith, core, func};
    use crate::location::Span;
    use crate::refs::{RegionRef, ValueRef};
    use crate::types::{Attribute, Location};
    use crate::{BlockArgData, BlockData, IrContext, RegionData, TypeDataBuilder};
    use smallvec::smallvec;
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
        crate::dialect::core::func(ctx, ret, params.iter().copied()).as_type_ref()
    }

    fn stale_value_errors(result: &ValidationResult) -> Vec<(&str, &str, &str)> {
        result
            .errors
            .iter()
            .filter_map(|error| {
                let ValidationError::StaleValue {
                    function_name,
                    consumer_op,
                    stale_value_description,
                    ..
                } = error
                else {
                    return None;
                };

                Some((
                    function_name.as_str(),
                    consumer_op.as_str(),
                    stale_value_description.as_str(),
                ))
            })
            .collect()
    }

    fn operation_error_messages(result: &ValidationResult) -> Vec<&str> {
        result
            .errors
            .iter()
            .filter_map(|error| {
                let ValidationError::Operation { message } = error else {
                    return None;
                };

                Some(message.as_str())
            })
            .collect()
    }

    fn single_block_yield_region(
        ctx: &mut IrContext,
        loc: Location,
        values: impl IntoIterator<Item = ValueRef>,
    ) -> RegionRef {
        let block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let mut yield_builder =
            OperationDataBuilder::new(loc, Symbol::new("scf"), Symbol::new("yield"));
        for value in values {
            yield_builder = yield_builder.operand(value);
        }
        let yield_data = yield_builder.build(ctx);
        let yield_op = ctx.create_op(yield_data);
        ctx.push_op(block, yield_op);
        ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![block],
            parent_op: None,
        })
    }

    fn wrap_if_in_module(
        ctx: &mut IrContext,
        loc: Location,
        i32_ty: super::super::refs::TypeRef,
        entry: super::super::refs::BlockRef,
        if_op: super::super::refs::OpRef,
    ) -> Module {
        ctx.push_op(entry, if_op);
        let zero = arith::r#const(ctx, loc, i32_ty, Attribute::Int(0));
        ctx.push_op(entry, zero.op_ref());
        let ret = func::r#return(ctx, loc, [zero.result(ctx)]);
        ctx.push_op(entry, ret.op_ref());

        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });
        let func_ty = make_func_type(ctx, &[], i32_ty);
        let func_op = func::func(ctx, loc, Symbol::new("bad_if"), func_ty, body);

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
        let module_op = core::module(ctx, loc, Symbol::new("test"), mod_region);
        Module::new(ctx, module_op.op_ref()).unwrap()
    }

    /// Build a valid module: fn add() { 40 + 2 }
    fn build_valid_module(ctx: &mut IrContext) -> Module {
        let loc = test_location(ctx);
        let i32_ty = make_i32_type(ctx);

        let entry_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });

        let c0 = arith::r#const(ctx, loc, i32_ty, Attribute::Int(40));
        ctx.push_op(entry_block, c0.op_ref());
        let c0_val = c0.result(ctx);

        let c1 = arith::r#const(ctx, loc, i32_ty, Attribute::Int(2));
        ctx.push_op(entry_block, c1.op_ref());
        let c1_val = c1.result(ctx);

        let add_op = arith::addi(ctx, loc, c0_val, c1_val, i32_ty);
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

        Module::new(ctx, module.op_ref()).unwrap()
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
        let const_a = arith::r#const(&mut ctx, loc, i32_ty, Attribute::Int(99));
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
        let module = Module::new(&ctx, module_op.op_ref()).unwrap();

        let result = validate_value_integrity(&ctx, module);
        assert!(!result.is_ok(), "Should detect stale op result");
        let stale_errors = stale_value_errors(&result);
        assert_eq!(stale_errors.len(), 1);
        assert_eq!(stale_errors[0].0, "func_b");
        assert!(stale_errors[0].2.contains("arith.const"));
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
                attrs: Default::default(),
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
        let module = Module::new(&ctx, module_op.op_ref()).unwrap();

        let result = validate_value_integrity(&ctx, module);
        assert!(!result.is_ok(), "Should detect stale block arg");
        let stale_errors = stale_value_errors(&result);
        assert_eq!(stale_errors.len(), 1);
        assert_eq!(stale_errors[0].0, "func_b");
        assert!(stale_errors[0].2.contains("block arg"));
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
                attrs: Default::default(),
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
        let c1 = arith::r#const(&mut ctx, loc, i32_ty, Attribute::Int(1));
        ctx.push_op(else_block, c1.op_ref());
        let c1_val = c1.result(&ctx);
        let sum = arith::addi(&mut ctx, loc, param, c1_val, i32_ty);
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
        let cond = arith::r#const(&mut ctx, loc, i1_ty, Attribute::Int(1));
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
        let module = Module::new(&ctx, module_op.op_ref()).unwrap();

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
        let const_a = arith::r#const(&mut ctx, loc, i32_ty, Attribute::Int(42));
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
        let local = arith::r#const(&mut ctx, loc, i32_ty, Attribute::Int(1));
        ctx.push_op(entry_b, local.op_ref());
        let local_val = local.result(&ctx);
        let add_op = arith::addi(&mut ctx, loc, value_from_a, local_val, i32_ty);
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
        let module = Module::new(&ctx, module_op.op_ref()).unwrap();

        let result = validate_value_integrity(&ctx, module);
        assert!(
            !result.is_ok(),
            "Cross-function value ref should be invalid"
        );
        let stale_errors = stale_value_errors(&result);
        assert_eq!(stale_errors.len(), 2);
        for (function_name, _, _) in stale_errors {
            assert_eq!(function_name, "func_b");
        }
    }

    #[test]
    fn wasm_func_stale_ref_detected() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = make_i32_type(&mut ctx);
        let wasm_func_ty = make_func_type(&mut ctx, &[], i32_ty);

        // func_a (func.func) with a constant
        let entry_a = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let const_a = arith::r#const(&mut ctx, loc, i32_ty, Attribute::Int(99));
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

        // func_b (wasm.func) uses stale_value from func_a
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
        // Build wasm.func manually
        let wasm_func_data =
            OperationDataBuilder::new(loc, Symbol::new("wasm"), Symbol::new("func"))
                .attr("sym_name", Attribute::Symbol(Symbol::new("func_b")))
                .attr("type", Attribute::Type(wasm_func_ty))
                .region(body_b)
                .build(&mut ctx);
        let wasm_func_op = ctx.create_op(wasm_func_data);

        // module
        let mod_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        ctx.push_op(mod_block, func_a.op_ref());
        ctx.push_op(mod_block, wasm_func_op);
        let mod_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![mod_block],
            parent_op: None,
        });
        let module_op = core::module(&mut ctx, loc, Symbol::new("test"), mod_region);
        let module = Module::new(&ctx, module_op.op_ref()).unwrap();

        let result = validate_value_integrity(&ctx, module);
        assert!(!result.is_ok(), "Should detect stale ref in wasm.func body");
        let stale_errors = stale_value_errors(&result);
        assert_eq!(stale_errors.len(), 1);
        assert_eq!(stale_errors[0].0, "func_b");
    }

    /// A value defined inside a nested region must not be visible in the outer
    /// scope. The old flat-set approach would silently accept such references;
    /// the new visibility-based checker must reject them.
    #[test]
    fn inner_value_not_visible_in_outer_scope() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = make_i32_type(&mut ctx);
        let i1_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i1")).build());

        let entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });

        // then region: defines %inner_val
        let then_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let inner_const = arith::r#const(&mut ctx, loc, i32_ty, Attribute::Int(42));
        ctx.push_op(then_block, inner_const.op_ref());
        let inner_val = inner_const.result(&ctx);
        let yield_then = OperationDataBuilder::new(loc, Symbol::new("scf"), Symbol::new("yield"))
            .operand(inner_val)
            .build(&mut ctx);
        let yield_then_op = ctx.create_op(yield_then);
        ctx.push_op(then_block, yield_then_op);
        let then_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![then_block],
            parent_op: None,
        });

        // else region: trivial
        let else_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let else_const = arith::r#const(&mut ctx, loc, i32_ty, Attribute::Int(0));
        ctx.push_op(else_block, else_const.op_ref());
        let else_val = else_const.result(&ctx);
        let yield_else = OperationDataBuilder::new(loc, Symbol::new("scf"), Symbol::new("yield"))
            .operand(else_val)
            .build(&mut ctx);
        let yield_else_op = ctx.create_op(yield_else);
        ctx.push_op(else_block, yield_else_op);
        let else_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![else_block],
            parent_op: None,
        });

        // Condition
        let cond_op = arith::r#const(&mut ctx, loc, i1_ty, Attribute::Int(1));
        ctx.push_op(entry, cond_op.op_ref());
        let cond = cond_op.result(&ctx);

        // scf.if
        let if_data = OperationDataBuilder::new(loc, Symbol::new("scf"), Symbol::new("if"))
            .operand(cond)
            .result(i32_ty)
            .region(then_region)
            .region(else_region)
            .build(&mut ctx);
        let if_op = ctx.create_op(if_data);
        ctx.push_op(entry, if_op);

        // BUG: outer block uses %inner_val which is defined only inside the then region
        let ret = func::r#return(&mut ctx, loc, [inner_val]);
        ctx.push_op(entry, ret.op_ref());

        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });
        let func_ty = make_func_type(&mut ctx, &[], i32_ty);
        let func_op = func::func(&mut ctx, loc, Symbol::new("bad_scope"), func_ty, body);

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
        let module = Module::new(&ctx, module_op.op_ref()).unwrap();

        let result = validate_value_integrity(&ctx, module);
        assert!(
            !result.is_ok(),
            "Value defined only in inner region must not be visible in outer scope"
        );
        let stale_errors = stale_value_errors(&result);
        assert_eq!(stale_errors.len(), 1);
        assert_eq!(stale_errors[0].0, "bad_scope");
        assert!(stale_errors[0].1.contains("return"));
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

        let c0 = arith::r#const(&mut ctx, loc, i32_ty, Attribute::Int(40));
        ctx.push_op(entry, c0.op_ref());
        let c0_val = c0.result(&ctx);

        let c1 = arith::r#const(&mut ctx, loc, i32_ty, Attribute::Int(2));
        ctx.push_op(entry, c1.op_ref());
        let c1_val = c1.result(&ctx);

        // Use c0 in two places
        let add = arith::addi(&mut ctx, loc, c0_val, c0_val, i32_ty);
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
        let module = Module::new(&ctx, module_op.op_ref()).unwrap();

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

    // ========================================================================
    // Call arity validation tests (textual IR)
    // ========================================================================

    #[test]
    fn call_arity_mismatch_too_few_args() {
        // add expects 2 params, caller passes 1
        let input = r#"core.module @test {
  func.func @add(%0: core.i32, %1: core.i32) -> core.i32 {
    %2 = arith.addi %0, %1 : core.i32
    func.return %2
  }
  func.func @main() -> core.i32 {
    %0 = arith.const {value = 1} : core.i32
    %1 = func.call %0 {callee = @add} : core.i32
    func.return %1
  }
}"#;
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(&mut ctx, input);

        validate_call_arity(&ctx, module);
        let diagnostics = ctx.diagnostics();
        assert_eq!(diagnostics.len(), 1, "Should detect arity mismatch");
        assert!(diagnostics[0].message.contains("main"));
        assert!(diagnostics[0].message.contains("add"));
        assert!(diagnostics[0].message.contains("1 argument(s), expected 2"));
    }

    #[test]
    fn call_arity_mismatch_too_many_args() {
        // add expects 1 param, caller passes 3
        let input = r#"core.module @test {
  func.func @add(%0: core.i32) -> core.i32 {
    func.return %0
  }
  func.func @main() -> core.i32 {
    %0 = arith.const {value = 1} : core.i32
    %1 = arith.const {value = 2} : core.i32
    %2 = arith.const {value = 3} : core.i32
    %3 = func.call %0, %1, %2 {callee = @add} : core.i32
    func.return %3
  }
}"#;
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(&mut ctx, input);

        validate_call_arity(&ctx, module);
        let diagnostics = ctx.diagnostics();
        assert_eq!(diagnostics.len(), 1, "Should detect too many args");
        assert!(diagnostics[0].message.contains("3 argument(s), expected 1"));
    }

    #[test]
    fn call_correct_arity_passes() {
        let input = r#"core.module @test {
  func.func @add(%0: core.i32, %1: core.i32) -> core.i32 {
    %2 = arith.addi %0, %1 : core.i32
    func.return %2
  }
  func.func @main() -> core.i32 {
    %0 = arith.const {value = 40} : core.i32
    %1 = arith.const {value = 2} : core.i32
    %2 = func.call %0, %1 {callee = @add} : core.i32
    func.return %2
  }
}"#;
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(&mut ctx, input);

        validate_call_arity(&ctx, module);
        assert!(!ctx.has_diagnostics(), "Correct arity should pass");
    }

    #[test]
    fn call_unknown_callee_skipped() {
        // extern_fn is NOT defined in this module — should be skipped
        let input = r#"core.module @test {
  func.func @main() -> core.i32 {
    %0 = arith.const {value = 1} : core.i32
    %1 = func.call %0 {callee = @extern_fn} : core.i32
    func.return %1
  }
}"#;
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(&mut ctx, input);

        validate_call_arity(&ctx, module);
        assert!(!ctx.has_diagnostics(), "Unknown callee should be skipped");
    }

    #[test]
    fn tail_call_arity_mismatch_detected() {
        // add expects 2 params, tail_call passes 1
        let input = r#"core.module @test {
  func.func @add(%0: core.i32, %1: core.i32) -> core.i32 {
    %2 = arith.addi %0, %1 : core.i32
    func.return %2
  }
  func.func @main() -> core.i32 {
    %0 = arith.const {value = 1} : core.i32
    func.tail_call %0 {callee = @add}
  }
}"#;
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(&mut ctx, input);

        validate_call_arity(&ctx, module);
        let diagnostics = ctx.diagnostics();
        assert_eq!(
            diagnostics.len(),
            1,
            "Should detect tail_call arity mismatch"
        );
        assert!(diagnostics[0].message.contains("add"));
        assert!(diagnostics[0].message.contains("1 argument(s), expected 2"));
    }

    #[test]
    fn zero_arg_function_called_with_args_detected() {
        let input = r#"core.module @test {
  func.func @unit() -> core.i32 {
    %0 = arith.const {value = 0} : core.i32
    func.return %0
  }
  func.func @main() -> core.i32 {
    %0 = arith.const {value = 1} : core.i32
    %1 = func.call %0 {callee = @unit} : core.i32
    func.return %1
  }
}"#;
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(&mut ctx, input);

        validate_call_arity(&ctx, module);
        let diagnostics = ctx.diagnostics();
        assert!(
            !diagnostics.is_empty(),
            "Should detect args to zero-param function"
        );
        assert!(diagnostics[0].message.contains("1 argument(s), expected 0"));
    }

    #[test]
    fn cmpf_subset_predicate_passes() {
        let input = r#"core.module @test {
  func.func @main(%0: core.f64, %1: core.f64) -> core.i1 {
    %2 = arith.cmpf %0, %1 {predicate = @une} : core.i1
    func.return %2
  }
}"#;
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(&mut ctx, input);

        let result = validate_operation_verifiers(&ctx, module);
        assert!(result.is_ok(), "{result}");
    }

    #[test]
    fn cmpf_unsupported_predicate_is_rejected() {
        let input = r#"core.module @test {
  func.func @main(%0: core.f64, %1: core.f64) -> core.i1 {
    %2 = arith.cmpf %0, %1 {predicate = @ueq} : core.i1
    func.return %2
  }
}"#;
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(&mut ctx, input);

        let result = validate_operation_verifiers(&ctx, module);
        let operation_errors = operation_error_messages(&result);
        assert_eq!(operation_errors.len(), 1);
        assert!(operation_errors[0].contains("operation verifier failed for arith.cmpf"));
        assert!(operation_errors[0].contains("unsupported predicate 'ueq'"));
    }

    #[test]
    fn cmpf_missing_predicate_is_rejected() {
        let input = r#"core.module @test {
  func.func @main(%0: core.f64, %1: core.f64) -> core.i1 {
    %2 = arith.cmpf %0, %1 : core.i1
    func.return %2
  }
}"#;
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(&mut ctx, input);

        let result = validate_operation_verifiers(&ctx, module);
        let operation_errors = operation_error_messages(&result);
        assert_eq!(operation_errors.len(), 1);
        assert!(operation_errors[0].contains("operation verifier failed for arith.cmpf"));
        assert!(operation_errors[0].contains("requires symbol predicate attribute"));
    }

    #[test]
    #[allow(deprecated)]
    fn deprecated_operation_semantics_alias_delegates_to_operation_verifiers() {
        let input = r#"core.module @test {
  func.func @main(%0: core.f64, %1: core.f64) -> core.i1 {
    %2 = arith.cmpf %0, %1 {predicate = @ueq} : core.i1
    func.return %2
  }
}"#;
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(&mut ctx, input);

        let result = validate_operation_semantics(&ctx, module);
        let operation_errors = operation_error_messages(&result);
        assert_eq!(operation_errors.len(), 1);
        assert!(operation_errors[0].contains("operation verifier failed for arith.cmpf"));
    }

    #[test]
    fn scf_if_yield_arity_mismatch_is_rejected() {
        let input = r#"core.module @test {
  func.func @main(%cond: core.i1, %x: core.i32) -> core.i32 {
    %r = scf.if %cond : core.i32 {
      scf.yield
    } {
      scf.yield %x
    }
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(&mut ctx, input);

        let result = validate_operation_verifiers(&ctx, module);
        let operation_errors = operation_error_messages(&result);
        assert_eq!(operation_errors.len(), 1);
        assert!(operation_errors[0].contains("operation verifier failed for scf.if"));
        assert!(operation_errors[0].contains("then_region yields 0 value(s)"));
        assert!(operation_errors[0].contains("scf.if has 1 result(s)"));
    }

    #[test]
    fn scf_if_missing_yield_terminator_is_rejected() {
        let input = r#"core.module @test {
  func.func @main(%cond: core.i1, %x: core.i32) -> core.i32 {
    %r = scf.if %cond : core.i32 {
      %zero = arith.const {value = 0} : core.i32
    } {
      scf.yield %x
    }
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(&mut ctx, input);

        let result = validate_operation_verifiers(&ctx, module);
        let operation_errors = operation_error_messages(&result);
        assert_eq!(operation_errors.len(), 1);
        assert!(operation_errors[0].contains("operation verifier failed for scf.if"));
        assert!(operation_errors[0].contains("then_region must terminate with scf.yield"));
    }

    #[test]
    fn scf_if_condition_operand_arity_is_rejected() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = make_i32_type(&mut ctx);
        let i1_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i1")).build());

        let entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let cond_a = arith::r#const(&mut ctx, loc, i1_ty, Attribute::Int(1));
        ctx.push_op(entry, cond_a.op_ref());
        let cond_b = arith::r#const(&mut ctx, loc, i1_ty, Attribute::Int(0));
        ctx.push_op(entry, cond_b.op_ref());

        let then_region = single_block_yield_region(&mut ctx, loc, []);
        let else_region = single_block_yield_region(&mut ctx, loc, []);
        let if_op = OperationDataBuilder::new(loc, Symbol::new("scf"), Symbol::new("if"))
            .operand(cond_a.result(&ctx))
            .operand(cond_b.result(&ctx))
            .region(then_region)
            .region(else_region)
            .build(&mut ctx);
        let if_op = ctx.create_op(if_op);
        let module = wrap_if_in_module(&mut ctx, loc, i32_ty, entry, if_op);

        let result = validate_operation_verifiers(&ctx, module);
        let operation_errors = operation_error_messages(&result);
        assert_eq!(operation_errors.len(), 1);
        assert!(operation_errors[0].contains("operation verifier failed for scf.if"));
        assert!(operation_errors[0].contains("expects 1 condition operand, found 2"));
    }

    #[test]
    fn scf_if_region_count_is_rejected() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = make_i32_type(&mut ctx);
        let i1_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i1")).build());

        let entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let cond = arith::r#const(&mut ctx, loc, i1_ty, Attribute::Int(1));
        ctx.push_op(entry, cond.op_ref());
        let then_region = single_block_yield_region(&mut ctx, loc, []);
        let if_op = OperationDataBuilder::new(loc, Symbol::new("scf"), Symbol::new("if"))
            .operand(cond.result(&ctx))
            .region(then_region)
            .build(&mut ctx);
        let if_op = ctx.create_op(if_op);
        let module = wrap_if_in_module(&mut ctx, loc, i32_ty, entry, if_op);

        let result = validate_operation_verifiers(&ctx, module);
        let operation_errors = operation_error_messages(&result);
        assert_eq!(operation_errors.len(), 1);
        assert!(operation_errors[0].contains("operation verifier failed for scf.if"));
        assert!(operation_errors[0].contains("expects 2 regions, found 1"));
    }

    #[test]
    fn scf_if_multiblock_region_is_rejected() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = make_i32_type(&mut ctx);
        let i1_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i1")).build());

        let entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let cond = arith::r#const(&mut ctx, loc, i1_ty, Attribute::Int(1));
        ctx.push_op(entry, cond.op_ref());

        let then_a = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let then_b = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let then_yield = OperationDataBuilder::new(loc, Symbol::new("scf"), Symbol::new("yield"))
            .build(&mut ctx);
        let then_yield = ctx.create_op(then_yield);
        ctx.push_op(then_a, then_yield);
        let then_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![then_a, then_b],
            parent_op: None,
        });
        let else_region = single_block_yield_region(&mut ctx, loc, []);

        let if_op = OperationDataBuilder::new(loc, Symbol::new("scf"), Symbol::new("if"))
            .operand(cond.result(&ctx))
            .region(then_region)
            .region(else_region)
            .build(&mut ctx);
        let if_op = ctx.create_op(if_op);
        let module = wrap_if_in_module(&mut ctx, loc, i32_ty, entry, if_op);

        let result = validate_operation_verifiers(&ctx, module);
        let operation_errors = operation_error_messages(&result);
        assert_eq!(operation_errors.len(), 1);
        assert!(operation_errors[0].contains("operation verifier failed for scf.if"));
        assert!(operation_errors[0].contains("then_region expects 1 block, found 2"));
    }

    #[test]
    fn validate_all_includes_cmpf_predicate_errors() {
        let input = r#"core.module @test {
  func.func @main(%0: core.f64, %1: core.f64) -> core.i1 {
    %2 = arith.cmpf %0, %1 {predicate = @one} : core.i1
    func.return %2
  }
}"#;
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(&mut ctx, input);

        let result = validate_all(&ctx, module);
        assert!(!result.is_ok());
        assert_eq!(operation_error_messages(&result).len(), 1);
    }
}
