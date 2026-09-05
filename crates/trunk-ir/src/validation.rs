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
use super::op_interface::{
    BranchOps, RegionBranchOps, RegionBranchPoint, RegionBranchTerminatorOps, RegionSuccessor,
    RegionValueTransfer,
};
use super::ops::DialectType;
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
            match data.attributes.get_symbol("sym_name") {
                Some(s) => {
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

    validate_core_func_types(ctx, &mut errors);

    let body = match module.body(ctx) {
        Some(r) => r,
        None => {
            return ValidationResult { errors };
        }
    };

    walk::walk_region::<std::convert::Infallible>(ctx, body, &mut |op| {
        let error_count = errors.len();
        validate_arith_cmpf_predicate(ctx, op, &mut errors);
        validate_scf_if_structure(ctx, op, &mut errors);
        validate_scf_loop_result_arity(ctx, op, &mut errors);
        validate_scf_switch_result_arity(ctx, op, &mut errors);
        validate_func_shapes(ctx, op, &mut errors);
        validate_func_indirect_call(ctx, op, &mut errors);
        if errors.len() == error_count {
            validate_branch_interface(ctx, op, &mut errors);
            validate_region_branch_interface(ctx, op, &mut errors);
            validate_region_branch_terminator_interface(ctx, op, &mut errors);
        }
        std::ops::ControlFlow::Continue(walk::WalkAction::Advance)
    });

    ValidationResult { errors }
}

fn validate_core_func_types(ctx: &IrContext, errors: &mut Vec<ValidationError>) {
    for (ty, data) in ctx.types.iter() {
        if data.dialect != crate::dialect::core::DIALECT_NAME()
            || data.name != crate::dialect::core::FUNC()
        {
            continue;
        }
        if let Err(error) = crate::dialect::core::Func::validate(ctx, ty) {
            errors.push(ValidationError::Operation {
                message: format!("type verifier failed for core.func ({ty}): {error}"),
            });
        }
    }
}

fn validate_func_indirect_call(ctx: &IrContext, op: OpRef, errors: &mut Vec<ValidationError>) {
    let data = ctx.op(op);
    if data.dialect != Symbol::new("func")
        || ![
            Symbol::new("call_indirect"),
            Symbol::new("tail_call_indirect"),
        ]
        .contains(&data.name)
    {
        return;
    }
    let tail = data.name == Symbol::new("tail_call_indirect");
    let Some((&callee, args)) = ctx.op_operands(op).split_first() else {
        errors.push(operation_verifier_error(
            ctx,
            op,
            "requires a callee operand",
        ));
        return;
    };
    let callee_ty = ctx.types.get(ctx.value_ty(callee));
    let func_ty = if callee_ty.dialect == Symbol::new("closure")
        && callee_ty.name == Symbol::new("closure")
    {
        let [func_ty] = callee_ty.params.as_slice() else {
            errors.push(operation_verifier_error(
                ctx,
                op,
                "callee closure type must contain exactly one function type",
            ));
            return;
        };
        *func_ty
    } else {
        ctx.value_ty(callee)
    };
    let typed = crate::dialect::core::Func::from_type_ref(ctx, func_ty);
    let exact = data
        .attributes
        .get_type("signature")
        .and_then(|ty| crate::dialect::core::Func::from_type_ref(ctx, ty));
    if data.attributes.contains_key("signature") && exact.is_none() {
        errors.push(operation_verifier_error(
            ctx,
            op,
            "invalid exact indirect signature",
        ));
        return;
    }
    if let (Some(typed), Some(exact)) = (typed, exact)
        && typed != exact
    {
        errors.push(operation_verifier_error(
            ctx,
            op,
            "exact indirect signature differs from typed callee",
        ));
    }
    let Some(func_ty) = exact.or(typed) else {
        if !tail {
            return;
        }
        errors.push(operation_verifier_error(
            ctx,
            op,
            "callee must have core.func or closure.closure<core.func> type",
        ));
        return;
    };
    if !tail && ctx.op_result_types(op) != func_ty.results(ctx) {
        errors.push(operation_verifier_error(
            ctx,
            op,
            "call result list mismatch",
        ));
    }
    check_value_types(ctx, op, args, func_ty.inputs(ctx), "call argument", errors);
}

fn validate_func_shapes(ctx: &IrContext, op: OpRef, errors: &mut Vec<ValidationError>) {
    use crate::dialect::{core, func};
    use crate::ops::DialectOp;
    if func::Func::matches(ctx, op) {
        let Some(signature) = ctx
            .op(op)
            .attributes
            .get_type("type")
            .and_then(|ty| core::Func::from_type_ref(ctx, ty))
        else {
            errors.push(operation_verifier_error(
                ctx,
                op,
                "requires valid core.func type",
            ));
            return;
        };
        if ctx.op(op).regions.len() > 1 {
            errors.push(operation_verifier_error(
                ctx,
                op,
                "expects at most one body",
            ));
        }
        if let Some(&region) = ctx.op(op).regions.first() {
            if let Some(&entry) = ctx.region(region).blocks.first() {
                check_value_types(
                    ctx,
                    op,
                    ctx.block_args(entry),
                    signature.inputs(ctx),
                    "entry argument",
                    errors,
                );
            } else {
                errors.push(operation_verifier_error(
                    ctx,
                    op,
                    "body requires an entry block",
                ));
            }
        }
    }
    if (func::Call::matches(ctx, op) || func::CallIndirect::matches(ctx, op))
        && ctx.op_results(op).len() > 1
    {
        errors.push(operation_verifier_error(
            ctx,
            op,
            "multiple call results are unsupported",
        ));
    }
    if (func::Call::matches(ctx, op) || func::TailCall::matches(ctx, op))
        && ctx.op(op).attributes.get_symbol("callee").is_none()
    {
        errors.push(operation_verifier_error(
            ctx,
            op,
            "requires symbol callee attribute",
        ));
    }
    if func::Return::matches(ctx, op)
        || func::TailCall::matches(ctx, op)
        || func::TailCallIndirect::matches(ctx, op)
    {
        if !ctx.op_results(op).is_empty() {
            errors.push(operation_verifier_error(ctx, op, "must be resultless"));
        }
        if ctx
            .op(op)
            .parent_block
            .is_none_or(|b| ctx.block(b).ops.last() != Some(&op))
        {
            errors.push(operation_verifier_error(
                ctx,
                op,
                "must terminate its block",
            ));
        }
    }
    if func::Return::matches(ctx, op) && ctx.op_operands(op).len() > 1 {
        errors.push(operation_verifier_error(
            ctx,
            op,
            "return count mismatch: at most one value is supported",
        ));
    }
}

/// Nearest dialect-registered owner, including a malformed owner boundary.
fn enclosing_func_signature(ctx: &IrContext, mut op: OpRef) -> Option<crate::dialect::core::Func> {
    loop {
        let block = ctx.op(op).parent_block?;
        let region = ctx.block(block).parent_region?;
        op = ctx.region(region).parent_op?;
        if let Some(signature) = crate::op_interface::CallableOwnerOps::signature(ctx, op) {
            return signature;
        }
        if crate::op_interface::IsolatedFromAboveOps::is_isolated(ctx, op) {
            return None;
        }
    }
}

fn typed_callee_signature(ctx: &IrContext, value: ValueRef) -> Option<crate::dialect::core::Func> {
    let ty = ctx.value_ty(value);
    let data = ctx.types.get(ty);
    let ty = if data.dialect == Symbol::new("closure") && data.name == Symbol::new("closure") {
        let [ty] = data.params.as_slice() else {
            return None;
        };
        *ty
    } else {
        ty
    };
    crate::dialect::core::Func::from_type_ref(ctx, ty)
}

fn check_value_types(
    ctx: &IrContext,
    op: OpRef,
    values: &[ValueRef],
    expected: &[crate::TypeRef],
    role: &str,
    errors: &mut Vec<ValidationError>,
) {
    if values.len() != expected.len() {
        errors.push(operation_verifier_error(
            ctx,
            op,
            format!(
                "{role} count mismatch: expected {}, found {}",
                expected.len(),
                values.len()
            ),
        ));
    } else {
        for (index, (&value, &ty)) in values.iter().zip(expected).enumerate() {
            if ctx.value_ty(value) != ty {
                errors.push(operation_verifier_error(
                    ctx,
                    op,
                    format!("{role} #{index} type mismatch"),
                ));
            }
        }
    }
}

/// Validate contextual shared function contracts; called by `validate_all`.
/// Local shapes and typed indirect operands/results are checked separately by
/// `validate_operation_verifiers`. An undeclared runtime symbol has no known
/// signature, but never exempts other fully typed contracts from validation.
pub fn validate_function_contracts(ctx: &IrContext, module: Module) -> ValidationResult {
    use crate::dialect::{core, func};
    use crate::ops::DialectOp;
    let mut errors = Vec::new();
    let Some(body) = module.body(ctx) else {
        return ValidationResult { errors };
    };
    // None is genuinely undeclared. A found but invalid/ambiguous declaration
    // is Some(None), so compatibility cannot hide malformed known contracts.
    fn resolve(ctx: &IrContext, mut op: OpRef, name: Symbol) -> Option<Option<core::Func>> {
        loop {
            let region = ctx.block(ctx.op(op).parent_block?).parent_region?;
            let parent = ctx.region(region).parent_op?;
            if core::Module::matches(ctx, parent) {
                let mut matches = ctx
                    .region(region)
                    .blocks
                    .iter()
                    .flat_map(|&b| ctx.block(b).ops.iter().copied())
                    .filter(|&candidate| {
                        ctx.op(candidate).attributes.get_symbol("sym_name") == Some(name)
                    });
                if let Some(found) = matches.next() {
                    if matches.next().is_some() || !func::Func::matches(ctx, found) {
                        return Some(None);
                    }
                    return Some(
                        ctx.op(found)
                            .attributes
                            .get_type("type")
                            .and_then(|ty| core::Func::from_type_ref(ctx, ty)),
                    );
                }
            }
            op = parent;
        }
    }
    walk::walk_region::<std::convert::Infallible>(ctx, body, &mut |op| {
        let mut verify = || {
            if func::Return::matches(ctx, op) {
                if let Some(caller) = enclosing_func_signature(ctx, op) {
                    check_value_types(
                        ctx,
                        op,
                        ctx.op_operands(op),
                        caller.results(ctx),
                        "return",
                        &mut errors,
                    );
                } else {
                    errors.push(operation_verifier_error(
                        ctx,
                        op,
                        "requires registered enclosing callable signature",
                    ));
                }
                return;
            }
            let tail = func::TailCall::matches(ctx, op) || func::TailCallIndirect::matches(ctx, op);
            let direct = func::Call::matches(ctx, op) || func::TailCall::matches(ctx, op);
            if !direct && !tail {
                return;
            }
            // Even an undeclared runtime target cannot erase the caller's
            // independently known ownership boundary.
            let caller = enclosing_func_signature(ctx, op);
            if tail && caller.is_none() {
                errors.push(operation_verifier_error(
                    ctx,
                    op,
                    "tail transfer requires registered enclosing callable signature",
                ));
                return;
            }
            let operands = ctx.op_operands(op);
            let (signature, args) = if direct {
                let Some(name) = ctx.op(op).attributes.get_symbol("callee") else {
                    return;
                };
                let Some(signature) = resolve(ctx, op, name) else {
                    return;
                };
                let Some(signature) = signature else {
                    errors.push(operation_verifier_error(
                        ctx,
                        op,
                        "requires uniquely resolved valid callable signature",
                    ));
                    return;
                };
                (signature, operands)
            } else {
                let Some((&callee, args)) = operands.split_first() else {
                    return;
                };
                let signature = ctx
                    .op(op)
                    .attributes
                    .get_type("signature")
                    .and_then(|ty| core::Func::from_type_ref(ctx, ty))
                    .or_else(|| typed_callee_signature(ctx, callee));
                let Some(signature) = signature else {
                    return;
                };
                (signature, args)
            };
            if direct {
                check_value_types(
                    ctx,
                    op,
                    args,
                    signature.inputs(ctx),
                    "call argument",
                    &mut errors,
                );
            }
            if tail {
                if caller.is_none_or(|caller| caller.results(ctx) != signature.results(ctx)) {
                    errors.push(operation_verifier_error(
                        ctx,
                        op,
                        "tail caller/callee result lists differ",
                    ));
                }
            } else if ctx.op_result_types(op) != signature.results(ctx) {
                errors.push(operation_verifier_error(
                    ctx,
                    op,
                    "call result list mismatch",
                ));
            }
        };
        verify();
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

fn validate_forwarding_types(
    ctx: &IrContext,
    op: OpRef,
    forwarded: &[ValueRef],
    inputs: &[ValueRef],
    edge_name: &str,
    errors: &mut Vec<ValidationError>,
) {
    if forwarded.len() != inputs.len() {
        errors.push(operation_verifier_error(
            ctx,
            op,
            format!(
                "{edge_name} forwards {} value(s), but its successor expects {} input(s)",
                forwarded.len(),
                inputs.len(),
            ),
        ));
        return;
    }
    for (index, (&value, &input)) in forwarded.iter().zip(inputs).enumerate() {
        if ctx.value_ty(value) != ctx.value_ty(input) {
            errors.push(operation_verifier_error(
                ctx,
                op,
                format!("{edge_name} value #{index} type does not match successor input type"),
            ));
        }
    }
}

fn forwarding_comes_from_operands(source: &[ValueRef], forwarded: &[ValueRef]) -> bool {
    let mut available = HashMap::<ValueRef, usize>::new();
    for &value in source {
        *available.entry(value).or_default() += 1;
    }
    for &value in forwarded {
        let Some(count) = available.get_mut(&value) else {
            return false;
        };
        if *count == 0 {
            return false;
        }
        *count -= 1;
    }
    true
}

fn validate_branch_interface(ctx: &IrContext, op: OpRef, errors: &mut Vec<ValidationError>) {
    let Some(interface) = BranchOps::get(ctx, op) else {
        return;
    };
    if !ctx.op_results(op).is_empty() {
        errors.push(operation_verifier_error(
            ctx,
            op,
            "Branch operation must be resultless",
        ));
    }
    let parent_block = ctx.op(op).parent_block;
    if parent_block.is_none_or(|block| ctx.block(block).ops.last().copied() != Some(op)) {
        errors.push(operation_verifier_error(
            ctx,
            op,
            "Branch operation must be the final operation in its block",
        ));
    }
    let successors = match interface.successors(ctx, op) {
        Ok(successors) => successors,
        Err(error) => {
            errors.push(operation_verifier_error(
                ctx,
                op,
                format!("Branch interface is incomplete: {error}"),
            ));
            return;
        }
    };
    let raw_successors = &ctx.op(op).successors;
    if successors.as_slice().len() != raw_successors.len() {
        errors.push(operation_verifier_error(
            ctx,
            op,
            format!(
                "Branch interface reports {} successor(s), but the operation stores {}",
                successors.as_slice().len(),
                raw_successors.len(),
            ),
        ));
        return;
    }
    let source_region = ctx
        .op(op)
        .parent_block
        .and_then(|block| ctx.block(block).parent_region);
    for (index, (edge, &raw_successor)) in
        successors.as_slice().iter().zip(raw_successors).enumerate()
    {
        if edge.block != raw_successor {
            errors.push(operation_verifier_error(
                ctx,
                op,
                format!("Branch interface successor #{index} does not match the stored successor"),
            ));
            continue;
        }
        if source_region.is_none() || ctx.block(edge.block).parent_region != source_region {
            errors.push(operation_verifier_error(
                ctx,
                op,
                format!("Branch successor #{index} leaves the source region"),
            ));
        }
        if !forwarding_comes_from_operands(ctx.op_operands(op), edge.forwarded.as_slice()) {
            errors.push(operation_verifier_error(
                ctx,
                op,
                format!("Branch successor #{index} reports a value that is not a branch operand"),
            ));
        }
        validate_forwarding_types(
            ctx,
            op,
            edge.forwarded.as_slice(),
            ctx.block_args(edge.block),
            &format!("Branch successor #{index}"),
            errors,
        );
    }
}

fn region_is_nested_under_op(ctx: &IrContext, region: RegionRef, owner: OpRef) -> bool {
    let mut current = Some(region);
    while let Some(region) = current {
        let Some(parent_op) = ctx.region(region).parent_op else {
            return false;
        };
        if parent_op == owner {
            return true;
        }
        current = ctx
            .op(parent_op)
            .parent_block
            .and_then(|block| ctx.block(block).parent_region);
    }
    false
}

fn validate_region_transfer(
    ctx: &IrContext,
    owner: OpRef,
    point: RegionBranchPoint,
    successor: RegionSuccessor,
    errors: &mut Vec<ValidationError>,
) {
    if let RegionSuccessor::Region(region) = successor
        && !region_is_nested_under_op(ctx, region, owner)
    {
        errors.push(operation_verifier_error(
            ctx,
            owner,
            format!("RegionBranch successor {region} is outside the operation's region tree"),
        ));
        return;
    }
    let transfer = match RegionBranchOps::value_transfer(ctx, owner, point, successor) {
        Ok(transfer) => transfer,
        Err(error) => {
            errors.push(operation_verifier_error(
                ctx,
                owner,
                format!("RegionBranch value mapping is incomplete: {error}"),
            ));
            return;
        }
    };
    let RegionValueTransfer {
        successor,
        forwarded,
        inputs,
    } = transfer;
    let source_operands = match point {
        RegionBranchPoint::Parent => ctx.op_operands(owner),
        RegionBranchPoint::Terminator(terminator) => ctx.op_operands(terminator),
    };
    if !forwarding_comes_from_operands(source_operands, forwarded.as_slice()) {
        errors.push(operation_verifier_error(
            ctx,
            owner,
            format!(
                "RegionBranch edge to {successor:?} reports a value that is not a source operand"
            ),
        ));
    }
    validate_forwarding_types(
        ctx,
        owner,
        forwarded.as_slice(),
        inputs.as_slice(),
        &format!("RegionBranch edge to {successor:?}"),
        errors,
    );
}

fn validate_region_branch_interface(ctx: &IrContext, op: OpRef, errors: &mut Vec<ValidationError>) {
    let Some(interface) = RegionBranchOps::get(ctx, op) else {
        return;
    };
    let successors = match interface.successors(ctx, op, RegionBranchPoint::Parent) {
        Ok(successors) => successors,
        Err(error) => {
            errors.push(operation_verifier_error(
                ctx,
                op,
                format!("RegionBranch entry mapping is incomplete: {error}"),
            ));
            return;
        }
    };
    let mut unique = HashSet::new();
    for &successor in successors.as_slice() {
        if !unique.insert(successor) {
            errors.push(operation_verifier_error(
                ctx,
                op,
                format!("RegionBranch entry reports duplicate successor {successor:?}"),
            ));
            continue;
        }
        validate_region_transfer(ctx, op, RegionBranchPoint::Parent, successor, errors);
        let RegionSuccessor::Region(region) = successor else {
            continue;
        };
        for &block in &ctx.region(region).blocks {
            let Some(_) = ctx.block(block).ops.last() else {
                errors.push(operation_verifier_error(
                    ctx,
                    op,
                    format!("RegionBranch successor region {region} contains an empty block"),
                ));
                continue;
            };
            // RegionBranch describes the semantic edges exposed by the owning
            // operation; it does not impose a terminator kind on every
            // successor region. Some existing resultless structured regions
            // use implicit fallthrough, while explicit terminators are checked
            // independently through RegionBranchTerminatorOps below.
        }
    }
}

fn validate_region_branch_terminator_interface(
    ctx: &IrContext,
    op: OpRef,
    errors: &mut Vec<ValidationError>,
) {
    if RegionBranchTerminatorOps::get(ctx, op).is_none() {
        return;
    }
    if !ctx.op_results(op).is_empty() {
        errors.push(operation_verifier_error(
            ctx,
            op,
            "RegionBranchTerminator must be resultless",
        ));
    }
    if !ctx.op(op).successors.is_empty() {
        errors.push(operation_verifier_error(
            ctx,
            op,
            "RegionBranchTerminator must not store raw block successors",
        ));
    }
    let Some(parent_block) = ctx.op(op).parent_block else {
        errors.push(operation_verifier_error(
            ctx,
            op,
            "RegionBranchTerminator is detached from a block",
        ));
        return;
    };
    if ctx.block(parent_block).ops.last().copied() != Some(op) {
        errors.push(operation_verifier_error(
            ctx,
            op,
            "RegionBranchTerminator must be the final operation in its block",
        ));
    }

    let point = RegionBranchPoint::Terminator(op);
    let mut region = ctx.block(parent_block).parent_region;
    while let Some(current_region) = region {
        let Some(owner) = ctx.region(current_region).parent_op else {
            break;
        };
        if let Some(interface) = RegionBranchOps::get(ctx, owner) {
            match interface.successors(ctx, owner, point) {
                Ok(successors) => {
                    if successors.as_slice().is_empty() {
                        errors.push(operation_verifier_error(
                            ctx,
                            op,
                            "owning RegionBranch reports no successor for this terminator",
                        ));
                        return;
                    }
                    let mut unique = HashSet::new();
                    for &successor in successors.as_slice() {
                        if !unique.insert(successor) {
                            errors.push(operation_verifier_error(
                                ctx,
                                op,
                                format!("terminator reports duplicate successor {successor:?}"),
                            ));
                            continue;
                        }
                        validate_region_transfer(ctx, owner, point, successor, errors);
                    }
                    return;
                }
                Err(error) if error.is_not_applicable() => {}
                Err(_) => return,
            }
        }
        region = ctx
            .op(owner)
            .parent_block
            .and_then(|block| ctx.block(block).parent_region);
    }
    errors.push(operation_verifier_error(
        ctx,
        op,
        "has no complete owning RegionBranch mapping",
    ));
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

/// Return whether `op` is a resultless transfer that may terminate a
/// structured `core.never` region without an `scf.yield`.
pub fn is_proper_tail_terminator(ctx: &IrContext, op: OpRef) -> bool {
    let data = ctx.op(op);
    (data.dialect == Symbol::new("func")
        && matches!(
            data.name.with_str(|name| name.to_owned()).as_str(),
            "tail_call" | "tail_call_indirect" | "unreachable"
        ))
        || (data.dialect == Symbol::new("ability")
            && matches!(
                data.name.with_str(|name| name.to_owned()).as_str(),
                "perform" | "handle_dispatch"
            ))
        || (data.dialect == Symbol::new("effect") && data.name == Symbol::new("dispatch_cps"))
        || (data.dialect == Symbol::new("scf")
            && matches!(
                data.name.with_str(|name| name.to_owned()).as_str(),
                "if" | "switch"
            ))
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
            let never_result = match ctx.op_result_types(op) {
                [ty] => {
                    let ty = ctx.types.get(*ty);
                    ty.dialect == Symbol::new("core") && ty.name == Symbol::new("never")
                }
                _ => false,
            };
            if never_result && is_proper_tail_terminator(ctx, yield_op) {
                continue;
            }
            errors.push(operation_verifier_error(
                ctx,
                op,
                format!("{region_name} must terminate with scf.yield"),
            ));
            continue;
        }
    }
}

fn validate_scf_loop_result_arity(ctx: &IrContext, op: OpRef, errors: &mut Vec<ValidationError>) {
    let data = ctx.op(op);
    if data.dialect == Symbol::new("scf")
        && data.name == Symbol::new("loop")
        && ctx.op_results(op).len() > 1
    {
        errors.push(operation_verifier_error(
            ctx,
            op,
            format!(
                "supports zero or one result, found {}",
                ctx.op_results(op).len()
            ),
        ));
    }
}

fn validate_scf_switch_result_arity(ctx: &IrContext, op: OpRef, errors: &mut Vec<ValidationError>) {
    let data = ctx.op(op);
    if data.dialect == Symbol::new("scf")
        && data.name == Symbol::new("switch")
        && !ctx.op_results(op).is_empty()
    {
        errors.push(operation_verifier_error(ctx, op, "must be resultless"));
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

            let Some(func_ty) = crate::dialect::core::Func::from_type_ref(ctx, func_ty) else {
                continue;
            };
            signatures.insert(sym_name, func_ty.inputs(ctx).len());
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
/// Local shapes and contextual contracts are separate checks. Known direct
/// callees, exact indirect signatures and registered return/tail owners are
/// checked here; undeclared runtime calls do not invent a signature.
pub fn validate_all(ctx: &IrContext, module: Module) -> ValidationResult {
    let scope = validate_value_integrity(ctx, module);
    let uses = validate_use_chains(ctx, module);
    let ops = validate_operation_verifiers(ctx, module);
    let mut errors = scope.errors;
    errors.extend(uses.errors);
    errors.extend(ops.errors);
    errors.extend(validate_function_contracts(ctx, module).errors);
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
        crate::dialect::core::func(ctx, params.iter().copied(), [ret]).as_type_ref()
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

    fn empty_module(ctx: &mut IrContext) -> Module {
        crate::parser::parse_test_module(ctx, "core.module @test {}")
    }

    #[test]
    fn malformed_core_func_counts_are_rejected_by_typed_and_whole_ir_validation() {
        let cases = [
            (
                "missing num_inputs",
                TypeDataBuilder::new(Symbol::new("core"), Symbol::new("func"))
                    .attr(core::NUM_RESULTS_ATTR, Attribute::Int(0)),
                "missing required `num_inputs`",
            ),
            (
                "missing num_results",
                TypeDataBuilder::new(Symbol::new("core"), Symbol::new("func"))
                    .attr(core::NUM_INPUTS_ATTR, Attribute::Int(0)),
                "missing required `num_results`",
            ),
            (
                "wrong num_inputs type",
                TypeDataBuilder::new(Symbol::new("core"), Symbol::new("func"))
                    .attr(core::NUM_INPUTS_ATTR, Attribute::String("zero".to_string()))
                    .attr(core::NUM_RESULTS_ATTR, Attribute::Int(0)),
                "`num_inputs` must be a u32",
            ),
            (
                "wrong num_results type",
                TypeDataBuilder::new(Symbol::new("core"), Symbol::new("func"))
                    .attr(core::NUM_INPUTS_ATTR, Attribute::Int(0))
                    .attr(
                        core::NUM_RESULTS_ATTR,
                        Attribute::String("zero".to_string()),
                    ),
                "`num_results` must be a u32",
            ),
            (
                "count sum mismatch",
                TypeDataBuilder::new(Symbol::new("core"), Symbol::new("func"))
                    .attr(core::NUM_INPUTS_ATTR, Attribute::Int(1))
                    .attr(core::NUM_RESULTS_ATTR, Attribute::Int(0)),
                "must equal params length",
            ),
            (
                "negative input",
                TypeDataBuilder::new(Symbol::new("core"), Symbol::new("func"))
                    .attr("num_inputs", Attribute::Int(-1))
                    .attr("num_results", Attribute::Int(0)),
                "`num_inputs` must be a u32",
            ),
            (
                "negative result",
                TypeDataBuilder::new(Symbol::new("core"), Symbol::new("func"))
                    .attr("num_results", Attribute::Int(-1))
                    .attr("num_inputs", Attribute::Int(0)),
                "`num_results` must be a u32",
            ),
            (
                "input outside u32",
                TypeDataBuilder::new(Symbol::new("core"), Symbol::new("func"))
                    .attr("num_inputs", Attribute::Int(u32::MAX as i128 + 1))
                    .attr("num_results", Attribute::Int(0)),
                "`num_inputs` must be a u32",
            ),
            (
                "result outside u32",
                TypeDataBuilder::new(Symbol::new("core"), Symbol::new("func"))
                    .attr("num_results", Attribute::Int(u32::MAX as i128 + 1))
                    .attr("num_inputs", Attribute::Int(0)),
                "`num_results` must be a u32",
            ),
            (
                "multiple results",
                TypeDataBuilder::new(Symbol::new("core"), Symbol::new("func"))
                    .attr(core::NUM_INPUTS_ATTR, Attribute::Int(0))
                    .attr(core::NUM_RESULTS_ATTR, Attribute::Int(2)),
                "supports at most one result",
            ),
        ];

        for (case, builder, expected) in cases {
            let mut ctx = IrContext::new();
            let module = empty_module(&mut ctx);
            let malformed = ctx.types.intern(builder.build());
            assert!(
                core::Func::from_type_ref(&ctx, malformed).is_none(),
                "{case} must fail typed validation"
            );
            let result = validate_operation_verifiers(&ctx, module);
            let messages = operation_error_messages(&result);
            assert!(
                messages.iter().any(|message| message.contains(expected)),
                "{case}: {result}"
            );
        }
    }

    fn operations_named(ctx: &IrContext, module: Module, dialect: &str, name: &str) -> Vec<OpRef> {
        let mut operations = Vec::new();
        let dialect = Symbol::from_dynamic(dialect);
        let name = Symbol::from_dynamic(name);
        let body = module.body(ctx).expect("test module must have a body");
        walk::walk_region::<std::convert::Infallible>(ctx, body, &mut |op| {
            let data = ctx.op(op);
            if data.dialect == dialect && data.name == name {
                operations.push(op);
            }
            std::ops::ControlFlow::Continue(walk::WalkAction::Advance)
        });
        operations
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
    fn tail_call_indirect_typed_cps_transfer_passes() {
        let input = r#"core.module @test {
  func.func @main(%k: closure.closure(core.func(core.never, core.i32)), %value: core.i32) -> core.never {
    func.tail_call_indirect %k, %value
  }
}"#;
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(&mut ctx, input);
        let result = validate_operation_verifiers(&ctx, module);
        assert!(result.is_ok(), "{result}");
    }

    #[test]
    fn tail_call_indirect_rejects_non_cps_result_and_bad_arguments() {
        let input = r#"core.module @test {
  func.func @main(%k: closure.closure(core.func(core.i32, core.i32)), %value: core.bool) -> core.never {
    func.tail_call_indirect %k, %value
  }
}"#;
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(&mut ctx, input);
        let result = validate_all(&ctx, module);
        let text = result.to_string();
        assert!(text.contains("caller/callee result lists differ"), "{text}");
        assert!(text.contains("argument #0 type"), "{text}");
    }

    #[test]
    fn tail_call_indirect_rejects_malformed_shapes_and_accepts_bare_function() {
        let input = r#"core.module @test {
  func.func @result(%k: closure.closure(core.func(core.never))) -> core.never {
    %bad = func.tail_call_indirect %k : core.i32
    func.unreachable
  }
  func.func @missing() -> core.never {
    func.tail_call_indirect
  }
  func.func @bad_closure(%k: closure.closure()) -> core.never {
    func.tail_call_indirect %k
  }
  func.func @not_callable(%value: core.i32) -> core.never {
    func.tail_call_indirect %value
  }
  func.func @direct_function(%k: core.func(core.never)) -> core.never {
    func.tail_call_indirect %k
  }
  func.func @arity(%k: closure.closure(core.func(core.never, core.i32))) -> core.never {
    func.tail_call_indirect %k
  }
}"#;
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(&mut ctx, input);
        let result = validate_operation_verifiers(&ctx, module);
        let text = result.to_string();
        assert!(text.contains("must be resultless"), "{text}");
        assert!(text.contains("requires a callee operand"), "{text}");
        assert!(
            text.contains("closure type must contain exactly one function type"),
            "{text}"
        );
        assert!(text.contains("callee must have core.func"), "{text}");
        assert!(
            text.contains("call argument count mismatch: expected 1, found 0"),
            "{text}"
        );
        assert!(text.contains("must terminate its block"), "{text}");
        assert_eq!(result.errors.len(), 6, "{text}");
    }

    #[test]
    fn scf_never_regions_accept_every_proper_tail_surface() {
        let input = r#"core.module @test {
  func.func @main(%cond: core.i1) -> core.never {
    %outer = scf.if %cond : core.never {
      ability.perform
    } {
      %nested = scf.if %cond : core.never {
        func.unreachable
      } {
        ability.handle_dispatch
      }
    }
  }
}"#;
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(&mut ctx, input);
        let result = validate_operation_verifiers(&ctx, module);
        assert!(result.is_ok(), "{result}");
    }

    #[test]
    fn resultless_scf_regions_still_require_yield() {
        let input = r#"core.module @test {
  func.func @main(%cond: core.i1) {
    scf.if %cond {
      func.unreachable
    } {
      func.unreachable
    }
    func.return
  }
}"#;
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(&mut ctx, input);
        let result = validate_operation_verifiers(&ctx, module);
        let text = result.to_string();
        assert!(
            text.contains("then_region must terminate with scf.yield"),
            "{text}"
        );
        assert!(
            text.contains("else_region must terminate with scf.yield"),
            "{text}"
        );
    }

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
        assert!(operation_errors[0].contains("forwards 0 value(s)"));
        assert!(operation_errors[0].contains("successor expects 1 input(s)"));
    }

    #[test]
    fn cf_branch_interface_reports_textual_value_forwarding() {
        let input = r#"core.module @test {
  func.func @main(%x: core.i32) -> core.i32 {
    ^entry:
      cf.br %x [^exit]
    ^exit(%forwarded: core.i32):
      func.return %forwarded
  }
}"#;
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(&mut ctx, input);

        let result = validate_operation_verifiers(&ctx, module);
        assert!(result.is_ok(), "{result}");
        let branches = operations_named(&ctx, module, "cf", "br");
        assert_eq!(branches.len(), 1);
        let branch = branches[0];
        let interface = BranchOps::get(&ctx, branch).expect("cf.br must implement Branch");
        let successors = interface.successors(&ctx, branch).unwrap();
        assert_eq!(successors.as_slice().len(), 1);
        let edge = &successors.as_slice()[0];
        assert_eq!(edge.forwarded.as_slice(), ctx.op_operands(branch));
        assert_eq!(ctx.block_args(edge.block).len(), 1);
    }

    #[test]
    fn cf_cond_branch_interface_reports_both_textual_successors() {
        let input = r#"core.module @test {
  func.func @main(%condition: core.i1, %value: core.i32) -> core.i32 {
    ^entry:
      cf.cond_br %condition [^left, ^right]
    ^left:
      cf.br %value [^exit]
    ^right:
      cf.br %value [^exit]
    ^exit(%forwarded: core.i32):
      func.return %forwarded
  }
}"#;
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(&mut ctx, input);

        let result = validate_operation_verifiers(&ctx, module);
        assert!(result.is_ok(), "{result}");
        let branches = operations_named(&ctx, module, "cf", "cond_br");
        assert_eq!(branches.len(), 1);
        let branch = branches[0];
        let successors = BranchOps::get(&ctx, branch)
            .unwrap()
            .successors(&ctx, branch)
            .unwrap();
        assert_eq!(successors.as_slice().len(), 2);
        assert!(
            successors
                .as_slice()
                .iter()
                .all(|edge| edge.forwarded.is_empty())
        );
    }

    #[test]
    fn cf_branch_interface_rejects_textual_cardinality_and_type_mismatches() {
        let input = r#"core.module @test {
  func.func @missing(%x: core.i32) -> core.i32 {
    ^entry:
      cf.br [^exit]
    ^exit(%forwarded: core.i32):
      func.return %forwarded
  }
  func.func @wrong_type(%flag: core.i1) -> core.i32 {
    ^entry:
      cf.br %flag [^exit]
    ^exit(%forwarded: core.i32):
      func.return %forwarded
  }
}"#;
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(&mut ctx, input);

        let result = validate_operation_verifiers(&ctx, module);
        let messages = operation_error_messages(&result);
        assert_eq!(messages.len(), 2, "{result}");
        assert!(messages.iter().any(|message| {
            message.contains("forwards 0 value(s)")
                && message.contains("successor expects 1 input(s)")
        }));
        assert!(
            messages
                .iter()
                .any(|message| message.contains("type does not match successor input type"))
        );
    }

    #[test]
    fn region_branch_interfaces_cover_nested_if_forwarding() {
        let input = r#"core.module @test {
  func.func @main(%cond: core.i1, %x: core.i32) -> core.i32 {
    %outer = scf.if %cond : core.i32 {
      %inner = scf.if %cond : core.i32 {
        scf.yield %x
      } {
        scf.yield %x
      }
      scf.yield %inner
    } {
      scf.yield %x
    }
    func.return %outer
  }
}"#;
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(&mut ctx, input);

        let result = validate_operation_verifiers(&ctx, module);
        assert!(result.is_ok(), "{result}");
        let ifs = operations_named(&ctx, module, "scf", "if");
        assert_eq!(ifs.len(), 2);
        for if_op in ifs {
            let interface = RegionBranchOps::get(&ctx, if_op).unwrap();
            let successors = interface
                .successors(&ctx, if_op, RegionBranchPoint::Parent)
                .unwrap();
            assert_eq!(successors.as_slice().len(), 2);
            assert!(
                successors
                    .as_slice()
                    .iter()
                    .all(|successor| matches!(successor, RegionSuccessor::Region(_)))
            );
        }
    }

    #[test]
    fn region_branch_interfaces_cover_loop_entry_backedge_and_exit() {
        let input = r#"core.module @test {
  func.func @main(%init: core.i32) -> core.i32 {
    %result = scf.loop %init : core.i32 {
      ^header(%iter: core.i32):
        scf.continue %iter
      ^exit_path(%value: core.i32):
        scf.break %value
    }
    func.return %result
  }
}"#;
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(&mut ctx, input);

        let result = validate_operation_verifiers(&ctx, module);
        assert!(result.is_ok(), "{result}");
        let loops = operations_named(&ctx, module, "scf", "loop");
        assert_eq!(loops.len(), 1);
        let loop_op = loops[0];
        let interface = RegionBranchOps::get(&ctx, loop_op).unwrap();
        let entry = interface
            .successors(&ctx, loop_op, RegionBranchPoint::Parent)
            .unwrap();
        assert_eq!(entry.as_slice().len(), 1);
        let RegionSuccessor::Region(body) = entry.as_slice()[0] else {
            unreachable!()
        };
        let continues = operations_named(&ctx, module, "scf", "continue");
        assert_eq!(continues.len(), 1);
        let continue_op = continues[0];
        let backedge = RegionBranchOps::value_transfer(
            &ctx,
            loop_op,
            RegionBranchPoint::Terminator(continue_op),
            RegionSuccessor::Region(body),
        )
        .unwrap();
        assert_eq!(backedge.forwarded.as_slice().len(), 1);
        assert_eq!(backedge.inputs.as_slice().len(), 1);

        let breaks = operations_named(&ctx, module, "scf", "break");
        assert_eq!(breaks.len(), 1);
        let break_op = breaks[0];
        let exit = RegionBranchOps::value_transfer(
            &ctx,
            loop_op,
            RegionBranchPoint::Terminator(break_op),
            RegionSuccessor::Parent,
        )
        .unwrap();
        assert_eq!(exit.forwarded.as_slice().len(), 1);
        assert_eq!(exit.inputs.as_slice().len(), 1);
    }

    #[test]
    fn region_branch_interfaces_cover_switch_case_and_default() {
        let input = r#"core.module @test {
  func.func @main(%discriminant: core.i32) -> core.nil {
    scf.switch %discriminant {
      scf.case {value = 0} {
        scf.yield
      }
      scf.default {
        scf.yield
      }
    }
    %unit = arith.const {value = 0} : core.nil
    func.return %unit
  }
}"#;
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(&mut ctx, input);

        let result = validate_operation_verifiers(&ctx, module);
        assert!(result.is_ok(), "{result}");
        let switches = operations_named(&ctx, module, "scf", "switch");
        assert_eq!(switches.len(), 1);
        let switch = switches[0];
        let interface = RegionBranchOps::get(&ctx, switch).unwrap();
        let successors = interface
            .successors(&ctx, switch, RegionBranchPoint::Parent)
            .unwrap();
        assert_eq!(successors.as_slice().len(), 2);
        assert!(
            successors
                .as_slice()
                .iter()
                .all(|successor| matches!(successor, RegionSuccessor::Region(_)))
        );
        for wrapper in operations_named(&ctx, module, "scf", "case")
            .into_iter()
            .chain(operations_named(&ctx, module, "scf", "default"))
        {
            assert!(RegionBranchOps::get(&ctx, wrapper).is_some());
        }
    }

    #[test]
    fn typed_region_models_fail_closed_for_invalid_semantic_queries() {
        let input = r#"core.module @test {
  func.func @main(%cond: core.i1, %discriminant: core.i32, %value: core.i32) -> core.i32 {
    %selected = scf.if %cond : core.i32 {
      scf.yield %value
    } {
      scf.yield %value
    }
    scf.switch %discriminant {
      scf.case {value = 0} {
        scf.yield
      }
      scf.default {
        scf.yield
      }
    }
    %looped = scf.loop %value : core.i32 {
      ^continue_path(%continue_value: core.i32):
        scf.continue %continue_value
      ^break_path(%break_value: core.i32):
        scf.break %break_value
    }
    scf.loop %value {
      ^resultless_break_path(%resultless_value: core.i32):
        scf.break %resultless_value
    }
    %zero = arith.const {value = 0} : core.i32
    func.return %selected
  }
  func.func @stray(%value: core.i32) {
    scf.break %value
  }
  func.func @stray_continue(%other_value: core.i32) {
    scf.continue %other_value
  }
}"#;
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(&mut ctx, input);
        let direct_owner = |op: OpRef| {
            ctx.op(op)
                .parent_block
                .and_then(|block| ctx.block(block).parent_region)
                .and_then(|region| ctx.region(region).parent_op)
        };

        let if_op = operations_named(&ctx, module, "scf", "if")[0];
        let switch = operations_named(&ctx, module, "scf", "switch")[0];
        let case = operations_named(&ctx, module, "scf", "case")[0];
        let default = operations_named(&ctx, module, "scf", "default")[0];
        let yields = operations_named(&ctx, module, "scf", "yield");
        let if_yield = *yields
            .iter()
            .find(|&&op| direct_owner(op) == Some(if_op))
            .unwrap();
        let case_yield = *yields
            .iter()
            .find(|&&op| direct_owner(op) == Some(case))
            .unwrap();

        let if_interface = RegionBranchOps::get(&ctx, if_op).unwrap();
        let error = if_interface
            .successors(&ctx, if_op, RegionBranchPoint::Terminator(case_yield))
            .unwrap_err();
        assert!(error.is_not_applicable());
        assert!(error.to_string().contains("not an scf.if region yield"));
        assert!(
            if_interface
                .entry_successor_operands(&ctx, if_op, RegionSuccessor::Parent)
                .is_err()
        );

        let switch_interface = RegionBranchOps::get(&ctx, switch).unwrap();
        assert_eq!(
            switch_interface
                .successors(&ctx, switch, RegionBranchPoint::Terminator(case_yield),)
                .unwrap()
                .as_slice(),
            &[RegionSuccessor::Parent]
        );
        assert!(
            switch_interface
                .successors(&ctx, switch, RegionBranchPoint::Terminator(if_yield),)
                .unwrap_err()
                .is_not_applicable()
        );
        assert!(
            switch_interface
                .entry_successor_operands(&ctx, switch, RegionSuccessor::Parent)
                .is_err()
        );

        for wrapper in [case, default] {
            let interface = RegionBranchOps::get(&ctx, wrapper).unwrap();
            assert!(
                interface
                    .successors(&ctx, wrapper, RegionBranchPoint::Terminator(if_yield),)
                    .unwrap_err()
                    .is_not_applicable()
            );
            assert!(
                interface
                    .entry_successor_operands(&ctx, wrapper, RegionSuccessor::Parent)
                    .is_err()
            );
        }

        let loops = operations_named(&ctx, module, "scf", "loop");
        let result_loop = *loops
            .iter()
            .find(|&&op| !ctx.op_results(op).is_empty())
            .unwrap();
        let result_loop_interface = RegionBranchOps::get(&ctx, result_loop).unwrap();
        assert!(
            result_loop_interface
                .successors(&ctx, result_loop, RegionBranchPoint::Terminator(if_yield),)
                .unwrap_err()
                .is_not_applicable()
        );
        assert!(
            result_loop_interface
                .entry_successor_operands(&ctx, result_loop, RegionSuccessor::Parent)
                .is_err()
        );
        let loop_entry = result_loop_interface
            .successors(&ctx, result_loop, RegionBranchPoint::Parent)
            .unwrap();
        assert_eq!(loop_entry.as_slice().len(), 1);
        let RegionSuccessor::Region(loop_body) = loop_entry.as_slice()[0] else {
            unreachable!()
        };

        let continues = operations_named(&ctx, module, "scf", "continue");
        let continue_op = *continues
            .iter()
            .find(|&&op| direct_owner(op) == Some(result_loop))
            .unwrap();
        let stray_continue = *continues
            .iter()
            .find(|&&op| {
                direct_owner(op).is_some_and(|owner| ctx.op(owner).dialect == Symbol::new("func"))
            })
            .unwrap();
        assert!(
            RegionBranchTerminatorOps::get(&ctx, continue_op)
                .unwrap()
                .successor_operands(&ctx, continue_op, RegionSuccessor::Parent)
                .is_err()
        );
        let unrelated_region = ctx.op(case).regions[0];
        assert!(
            RegionBranchTerminatorOps::get(&ctx, continue_op)
                .unwrap()
                .successor_operands(&ctx, continue_op, RegionSuccessor::Region(unrelated_region),)
                .is_err()
        );
        assert!(
            RegionBranchTerminatorOps::get(&ctx, stray_continue)
                .unwrap()
                .successor_operands(&ctx, stray_continue, RegionSuccessor::Region(loop_body),)
                .unwrap_err()
                .to_string()
                .contains("no enclosing scf.loop")
        );
        assert!(
            RegionBranchTerminatorOps::get(&ctx, case_yield)
                .unwrap()
                .successor_operands(&ctx, case_yield, RegionSuccessor::Region(loop_body))
                .is_err()
        );

        let breaks = operations_named(&ctx, module, "scf", "break");
        let result_break = *breaks
            .iter()
            .find(|&&op| direct_owner(op) == Some(result_loop))
            .unwrap();
        assert!(
            RegionBranchTerminatorOps::get(&ctx, result_break)
                .unwrap()
                .successor_operands(&ctx, result_break, RegionSuccessor::Region(loop_body))
                .is_err()
        );
        let resultless_break = *breaks
            .iter()
            .find(|&&op| {
                direct_owner(op).is_some_and(|owner| {
                    ctx.op(owner).dialect == Symbol::new("scf")
                        && ctx.op(owner).name == Symbol::new("loop")
                        && ctx.op_results(owner).is_empty()
                })
            })
            .unwrap();
        assert!(
            RegionBranchTerminatorOps::get(&ctx, resultless_break)
                .unwrap()
                .successor_operands(&ctx, resultless_break, RegionSuccessor::Parent)
                .unwrap()
                .is_empty()
        );
        let stray_break = *breaks
            .iter()
            .find(|&&op| {
                direct_owner(op).is_some_and(|owner| ctx.op(owner).dialect == Symbol::new("func"))
            })
            .unwrap();
        assert!(
            RegionBranchTerminatorOps::get(&ctx, stray_break)
                .unwrap()
                .successor_operands(&ctx, stray_break, RegionSuccessor::Parent)
                .unwrap_err()
                .to_string()
                .contains("no enclosing scf.loop")
        );

        let constant = operations_named(&ctx, module, "arith", "const")[0];
        assert!(
            RegionBranchOps::value_transfer(
                &ctx,
                constant,
                RegionBranchPoint::Parent,
                RegionSuccessor::Parent,
            )
            .unwrap_err()
            .to_string()
            .contains("no RegionBranch registration")
        );
        assert!(
            RegionBranchOps::value_transfer(
                &ctx,
                if_op,
                RegionBranchPoint::Terminator(constant),
                RegionSuccessor::Parent,
            )
            .unwrap_err()
            .to_string()
            .contains("no RegionBranchTerminator registration")
        );
    }

    #[test]
    fn typed_models_reject_malformed_wrappers_shapes_and_region_structure() {
        let input = r#"core.module @test {
  func.func @main(%cond: core.i1, %discriminant: core.i32, %value: core.i32) -> core.i32 {
    ^entry:
      cf.cond_br %cond [^left, ^right]
    ^left:
      cf.br %value [^merge]
      %after_branch = arith.const {value = 0} : core.i32
    ^right:
      cf.br %value [^merge]
    ^merge(%forwarded: core.i32):
      %selected = scf.if %cond : core.i32 {
        scf.yield %forwarded
      } {
        scf.yield %forwarded
      }
      scf.switch %discriminant {
        scf.case {value = 0} { scf.yield }
      }
      scf.switch %discriminant {
        scf.case {value = 1} {
          ^empty_case:
        }
      }
      scf.switch %discriminant {
        %unsupported = arith.const {value = 2} : core.i32
      }
      scf.switch %discriminant {
        scf.default { scf.yield }
        scf.default { scf.yield }
      }
      %looped = scf.loop %forwarded : core.i32 {
        ^loop_exit(%loop_value: core.i32):
          scf.break %loop_value
      }
      func.return %selected
  }
  func.func @stray(%stray_value: core.i32) {
    scf.break %stray_value
  }
}"#;
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(&mut ctx, input);

        let validation = validate_operation_verifiers(&ctx, module);
        let messages = operation_error_messages(&validation).join("\n");
        for expected in [
            "Branch operation must be the final operation in its block",
            "contains an empty block",
            "body contains unsupported operation",
            "contains multiple default regions",
            "has no complete owning RegionBranch mapping",
        ] {
            assert!(
                messages.contains(expected),
                "missing {expected}: {validation}"
            );
        }

        let branch = operations_named(&ctx, module, "cf", "br")[0];
        let cond_branch = operations_named(&ctx, module, "cf", "cond_br")[0];
        let if_op = operations_named(&ctx, module, "scf", "if")[0];
        let switches = operations_named(&ctx, module, "scf", "switch");
        let loop_op = operations_named(&ctx, module, "scf", "loop")[0];
        let case = operations_named(&ctx, module, "scf", "case")[0];
        let yield_op = operations_named(&ctx, module, "scf", "yield")[0];
        let break_op = operations_named(&ctx, module, "scf", "break")[0];
        let wrong_op = operations_named(&ctx, module, "func", "return")[0];

        assert!(
            BranchOps::get(&ctx, branch)
                .unwrap()
                .successors(&ctx, wrong_op)
                .unwrap_err()
                .to_string()
                .contains("malformed cf.br")
        );
        let region_interface = RegionBranchOps::get(&ctx, if_op).unwrap();
        assert!(
            region_interface
                .successors(&ctx, wrong_op, RegionBranchPoint::Parent)
                .is_err()
        );
        assert!(
            region_interface
                .entry_successor_operands(&ctx, wrong_op, RegionSuccessor::Parent)
                .is_err()
        );
        assert!(
            RegionBranchTerminatorOps::get(&ctx, yield_op)
                .unwrap()
                .successor_operands(&ctx, wrong_op, RegionSuccessor::Parent)
                .is_err()
        );

        ctx.op_mut(branch).successors.clear();
        assert!(
            BranchOps::get(&ctx, branch)
                .unwrap()
                .successors(&ctx, branch)
                .is_err()
        );
        ctx.op_mut(cond_branch).successors.pop();
        assert!(
            BranchOps::get(&ctx, cond_branch)
                .unwrap()
                .successors(&ctx, cond_branch)
                .is_err()
        );
        ctx.op_mut(loop_op).regions.clear();
        assert!(
            RegionBranchOps::get(&ctx, loop_op)
                .unwrap()
                .successors(&ctx, loop_op, RegionBranchPoint::Parent)
                .is_err()
        );
        ctx.remove_op_operand(switches[0], 0);
        assert!(
            RegionBranchOps::get(&ctx, switches[0])
                .unwrap()
                .successors(&ctx, switches[0], RegionBranchPoint::Parent)
                .is_err()
        );
        let second_switch_body = ctx.op(switches[1]).regions[0];
        ctx.region_mut(second_switch_body).blocks.clear();
        assert!(
            RegionBranchOps::get(&ctx, switches[1])
                .unwrap()
                .successors(&ctx, switches[1], RegionBranchPoint::Parent)
                .is_err()
        );
        ctx.op_mut(case).regions.clear();
        assert!(
            RegionBranchOps::get(&ctx, case)
                .unwrap()
                .successors(&ctx, case, RegionBranchPoint::Parent)
                .is_err()
        );
        ctx.remove_op_operand(break_op, 0);
        assert!(
            RegionBranchTerminatorOps::get(&ctx, break_op)
                .unwrap()
                .successor_operands(&ctx, break_op, RegionSuccessor::Parent)
                .is_err()
        );
    }

    #[test]
    fn region_branch_interfaces_reject_incomplete_loop_mappings() {
        let input = r#"core.module @test {
  func.func @main(%init: core.i32) -> core.i32 {
    %result = scf.loop %init : core.i32 {
      ^header(%iter: core.i32, %extra: core.i32):
        scf.continue
      ^exit_path(%value: core.i32):
        scf.break %value
    }
    func.return %result
  }
}"#;
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(&mut ctx, input);

        let result = validate_operation_verifiers(&ctx, module);
        let messages = operation_error_messages(&result);
        assert_eq!(messages.len(), 2, "{result}");
        assert!(messages.iter().any(|message| {
            message.contains("forwards 1 value(s)")
                && message.contains("successor expects 2 input(s)")
        }));
        assert!(messages.iter().any(|message| {
            message.contains("forwards 0 value(s)")
                && message.contains("successor expects 2 input(s)")
        }));
    }

    #[test]
    fn region_branch_interface_rejects_textual_forwarding_type_mismatch() {
        let input = r#"core.module @test {
  func.func @main(%cond: core.i1, %x: core.i32) -> core.i32 {
    %result = scf.if %cond : core.i32 {
      scf.yield %cond
    } {
      scf.yield %x
    }
    func.return %result
  }
}"#;
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(&mut ctx, input);

        let result = validate_operation_verifiers(&ctx, module);
        let messages = operation_error_messages(&result);
        assert_eq!(messages.len(), 1, "{result}");
        assert!(messages[0].contains("type does not match successor input type"));
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
    fn scf_loop_and_switch_result_arity_is_rejected() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = make_i32_type(&mut ctx);
        let entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });

        let loop_body = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let loop_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![loop_body],
            parent_op: None,
        });
        let loop_data = OperationDataBuilder::new(loc, Symbol::new("scf"), Symbol::new("loop"))
            .result(i32_ty)
            .result(i32_ty)
            .region(loop_region)
            .build(&mut ctx);
        let loop_op = ctx.create_op(loop_data);
        ctx.push_op(entry, loop_op);

        let discriminant = arith::r#const(&mut ctx, loc, i32_ty, Attribute::Int(0));
        ctx.push_op(entry, discriminant.op_ref());
        let switch_body = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let switch_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![switch_body],
            parent_op: None,
        });
        let switch_data = OperationDataBuilder::new(loc, Symbol::new("scf"), Symbol::new("switch"))
            .operand(discriminant.result(&ctx))
            .result(i32_ty)
            .region(switch_region)
            .build(&mut ctx);
        let switch_op = ctx.create_op(switch_data);
        ctx.push_op(entry, switch_op);

        let ret = func::r#return(&mut ctx, loc, std::iter::empty());
        ctx.push_op(entry, ret.op_ref());
        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });
        let func_ty = make_func_type(&mut ctx, &[], i32_ty);
        let func_op = func::func(&mut ctx, loc, Symbol::new("malformed"), func_ty, body);
        let module_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![func_op.op_ref()],
            parent_region: None,
        });
        let module_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![module_block],
            parent_op: None,
        });
        let module_op = core::module(&mut ctx, loc, Symbol::new("test"), module_region);
        let module = Module::new(&ctx, module_op.op_ref()).unwrap();

        let result = validate_operation_verifiers(&ctx, module);
        let messages = operation_error_messages(&result);
        assert_eq!(messages.len(), 2, "{result}");
        assert!(messages.iter().any(|message| message.contains("scf.loop")));
        assert!(
            messages
                .iter()
                .any(|message| message.contains("scf.switch"))
        );
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
