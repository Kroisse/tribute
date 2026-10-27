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

use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;
use std::fmt;

use cranelift_entity::EntitySet;
use derive_more::{Display, Error};

use super::analysis::AnalysisCache;
use super::context::IrContext;
use super::dialect::core;
use super::op_def::OpDef;
use super::op_flag::{IsolatedFromAbove, OpFlag};
use super::op_interface::{
    DynBranch, DynCallableExit, DynCallableOwner, DynRegionBranch, DynRegionBranchTerminator,
    RegionBranchPoint, RegionSuccessor, RegionValueTransfer,
};
use super::ops::{DialectOp, DialectType};
use super::pass::{Pass, PassRunResult};
use super::refs::{OpRef, RegionRef, ValueDef, ValueRef};
use super::rewrite::Module;
use super::symbol_table::{SymbolTable, link_name, qualified_name};
use super::walk;

use crate::{Symbol, SymbolPath};

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
        consumer_op: std::borrow::Cow<'static, str>,
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
    /// An operation's region list and its regions' parent links disagree.
    #[display("{message}")]
    RegionLink { message: String },
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
            let full_name = ctx.op(op).name.display(ctx);
            match ctx.op_attrs(op).get_str("sym_name") {
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
                    errors.push(ValidationError::StaleValue {
                        function_name: function_name.to_string(),
                        consumer_op: ctx.op(op).name.to_text(ctx),
                        operand_index: i,
                        stale_value_description: describe_value(ctx, operand),
                    });
                }
            }
            // Propagate the extended visible set into nested regions.
            for nested_region in ctx.op_regions(op) {
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
    for &block in &ctx.region(region).blocks {
        for &op in &ctx.block(block).ops {
            let is_function = crate::dialect::func::Func::matches(ctx, op)
                || crate::dialect::wasm::Func::matches(ctx, op);
            if is_function {
                // This is a func.func or wasm.func
                let fn_name = ctx.op_attrs(op).get_str("sym_name").unwrap_or("<unnamed>");

                // Check operands with visibility-based scoping.
                // No values from outside the function body are visible.
                for func_region in ctx.op_regions(op) {
                    check_operands_in_region(
                        ctx,
                        func_region,
                        &HashSet::default(),
                        fn_name,
                        errors,
                    );
                }
            }

            // Recurse into nested regions (e.g., nested core.module)
            for nested_region in ctx.op_regions(op) {
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
///
/// It also checks that every region in an operation's region list names that
/// operation as its parent; both are linked by the context's region methods.
pub fn validate_use_chains(ctx: &IrContext, module: Module) -> ValidationResult {
    let mut errors = Vec::new();

    let body = match module.body(ctx) {
        Some(r) => r,
        None => {
            return ValidationResult { errors };
        }
    };

    // Arena IDs are dense. Keep only module membership and the values to check;
    // actual operands already provide the reverse lookup for use-chain entries.
    let mut module_ops = EntitySet::<OpRef>::new();
    let mut checked_values = EntitySet::<ValueRef>::new();
    for &block in &ctx.region(body).blocks {
        for &arg in ctx.block_args(block) {
            checked_values.insert(arg);
        }
    }

    walk::walk_region::<std::convert::Infallible>(ctx, body, &mut |op| {
        module_ops.insert(op);
        for &result in ctx.op_results(op) {
            checked_values.insert(result);
        }
        for region in ctx.op_regions(op) {
            let parent = ctx.region(region).parent_op;
            if parent != Some(op) {
                errors.push(ValidationError::RegionLink {
                    message: format!(
                        "{} ({:?}) lists {:?}, whose parent is {:?}",
                        ctx.op(op).name.display(ctx),
                        op,
                        region,
                        parent,
                    ),
                });
            }
            for &block in &ctx.region(region).blocks {
                for &arg in ctx.block_args(block) {
                    checked_values.insert(arg);
                }
            }
        }
        // Direction 1: actual operand -> use-chain entry must exist.
        for (idx, &val) in ctx.op_operands(op).iter().enumerate() {
            checked_values.insert(val);
            if !ctx
                .uses(val)
                .iter()
                .any(|u| u.user == op && u.operand_index == idx as u32)
            {
                errors.push(ValidationError::UseChain {
                    message: format!(
                        "operand #{} of {} ({:?}) uses {:?} but no use-chain entry exists",
                        idx,
                        ctx.op(op).name.display(ctx),
                        op,
                        val,
                    ),
                });
            }
        }
        std::ops::ControlFlow::Continue(walk::WalkAction::Advance)
    });

    // Direction 2: every recorded use must be an actual operand in this module.
    // Membership must be checked even when a detached/foreign user still has a
    // matching operand. Unused results and nested block args are checked too.
    for val in checked_values.iter() {
        for u in ctx.uses(val) {
            if !module_ops.contains(u.user)
                || ctx.op_operands(u.user).get(u.operand_index as usize) != Some(&val)
            {
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

    validate_type_shapes(ctx, &mut errors);

    // The root module is not visited by the body walk, so check its own
    // schema (including a missing body region) first.
    validate_op_schema(ctx, module.op(), &mut errors);
    validate_symbol_refs(ctx, module.op(), &mut errors);

    let body = match module.body(ctx) {
        Some(r) => r,
        None => {
            return ValidationResult { errors };
        }
    };

    walk::walk_region::<std::convert::Infallible>(ctx, body, &mut |op| {
        validate_symbol_refs(ctx, op, &mut errors);
        if !validate_op_schema(ctx, op, &mut errors) {
            return std::ops::ControlFlow::Continue(walk::WalkAction::Advance);
        }
        let error_count = errors.len();
        validate_scf_if_structure(ctx, op, &mut errors);
        validate_func_shapes(ctx, op, &mut errors);
        if errors.len() == error_count {
            validate_branch_interface(ctx, op, &mut errors);
            validate_region_branch_interface(ctx, op, &mut errors);
            validate_region_branch_terminator_interface(ctx, op, &mut errors);
        }
        std::ops::ControlFlow::Continue(walk::WalkAction::Advance)
    });

    ValidationResult { errors }
}

/// Check `root` and every operation nested in it against its declarative
/// schema only.
///
/// Unlike [`validate_operation_verifiers`], this skips `#[verify]` hooks,
/// the hand-written operation checks, and interface verification. It is the
/// debug checkpoint run after each pipeline pass, where the IR must satisfy
/// the declared constraints again; a pass nested under an operation checks only that
/// operation's subtree.
pub fn validate_op_schemas(ctx: &IrContext, root: OpRef) -> ValidationResult {
    let mut errors = Vec::new();
    let _ = walk::walk_op::<std::convert::Infallible>(ctx, root, &mut |op| {
        if let Some(def) = OpDef::of(ctx, op) {
            report_schema_violations(ctx, op, def.schema.verify(ctx, op), &mut errors);
        }
        std::ops::ControlFlow::Continue(walk::WalkAction::Advance)
    });
    ValidationResult { errors }
}

/// Check an operation against its registered declarative schema, including
/// type constraints and its `#[verify]` hook.
///
/// Returns `false` when the operation violates its schema. Later
/// operation-local checks assume the declared shape, so callers skip them for
/// that operation.
fn validate_op_schema(ctx: &IrContext, op: OpRef, errors: &mut Vec<ValidationError>) -> bool {
    let Some(def) = OpDef::of(ctx, op) else {
        return true;
    };
    report_schema_violations(ctx, op, def.verify(ctx, op), errors)
}

/// A symbol reference names a definition, so it is never the root module.
fn validate_symbol_refs(ctx: &IrContext, op: OpRef, errors: &mut Vec<ValidationError>) {
    let mut root_refs = 0;
    ctx.op(op).attributes.visit_symbol_refs(&mut |reference| {
        root_refs += usize::from(reference.is_root());
    });
    if root_refs != 0 {
        errors.push(operation_verifier_error(
            ctx,
            op,
            "a symbol reference has no component",
        ));
    }
}

/// Record each violation as an operation error; returns whether there were
/// none.
fn report_schema_violations(
    ctx: &IrContext,
    op: OpRef,
    violations: Vec<impl std::fmt::Display>,
    errors: &mut Vec<ValidationError>,
) -> bool {
    for violation in &violations {
        errors.push(operation_verifier_error(ctx, op, violation.to_string()));
    }
    violations.is_empty()
}

fn validate_type_shapes(ctx: &IrContext, errors: &mut Vec<ValidationError>) {
    for (ty, data) in ctx.types().iter() {
        if let Err(error) = data.validate_param_attrs(ctx) {
            errors.push(ValidationError::Operation {
                message: format!(
                    "type verifier failed for {} ({ty}): {error}",
                    data.name.display(ctx)
                ),
            });
        }
        let Some(verifier) = data.name.def().and_then(|def| def.verifier) else {
            continue;
        };
        if let Err(error) = verifier(ctx, ty) {
            errors.push(ValidationError::Operation {
                message: format!(
                    "type verifier failed for {} ({ty}): {error}",
                    data.name.display(ctx)
                ),
            });
        }
    }
}

fn validate_func_shapes(ctx: &IrContext, op: OpRef, errors: &mut Vec<ValidationError>) {
    use crate::dialect::func;
    use crate::ops::DialectOp;
    if func::Func::matches(ctx, op) {
        let Some(signature) = ctx
            .op_attrs(op)
            .get_type("type")
            .and_then(|ty| func::FuncSig::from_type_ref(ctx, ty))
        else {
            errors.push(operation_verifier_error(
                ctx,
                op,
                "requires valid func.func_sig type",
            ));
            return;
        };
        if let Some(region) = ctx.op_region(op, 0) {
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
    if (func::Return::matches(ctx, op)
        || func::TailCall::matches(ctx, op)
        || func::TailCallIndirect::matches(ctx, op))
        && ctx
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
    if func::Return::matches(ctx, op) && ctx.op_operands(op).len() > 1 {
        errors.push(operation_verifier_error(
            ctx,
            op,
            "return count mismatch: at most one value is supported",
        ));
    }
}

/// Nearest dialect-registered owner, including a malformed owner boundary.
fn enclosing_func_signature(
    ctx: &IrContext,
    mut op: OpRef,
) -> Option<crate::dialect::func::FuncSig> {
    loop {
        let block = ctx.op(op).parent_block?;
        let region = ctx.block(block).parent_region?;
        op = ctx.region(region).parent_op?;
        if let Some(owner) = DynCallableOwner::get(ctx, op) {
            return owner.callable_signature(ctx);
        }
        if IsolatedFromAbove::is_set(ctx, op) {
            return None;
        }
    }
}

fn typed_callee_signature(
    ctx: &IrContext,
    value: ValueRef,
) -> Option<crate::dialect::func::FuncSig> {
    let ty = ctx.value_ty(value);
    let data = ctx.get_type(ty);
    let ty = if data.name.is(ctx, "closure.closure") {
        let [ty] = data.params.as_slice() else {
            return None;
        };
        *ty
    } else {
        ty
    };
    crate::dialect::func::FuncSig::from_type_ref(ctx, ty)
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
    function_contracts(ctx, module, &SymbolTable::collect(ctx, module))
}

fn function_contracts(ctx: &IrContext, module: Module, symbols: &SymbolTable) -> ValidationResult {
    use crate::dialect::func;
    use crate::ops::DialectOp;
    let mut errors = Vec::new();
    let Some(body) = module.body(ctx) else {
        return ValidationResult { errors };
    };
    // A reference is resolved in the module nearest to the operation that
    // holds it. None is genuinely undeclared. A found but invalid or
    // duplicated declaration is Some(None), so compatibility cannot hide
    // malformed known contracts.
    let resolve = |op: OpRef, name: &SymbolPath| -> Option<Option<func::FuncSig>> {
        let found = *symbols.definitions(ctx, op, name).first()?;
        if symbols.resolve(ctx, op, name).is_none() || !func::Func::matches(ctx, found) {
            return Some(None);
        }
        Some(
            ctx.op_attrs(found)
                .get_type("type")
                .and_then(|ty| func::FuncSig::from_type_ref(ctx, ty)),
        )
    };
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
                let Some(name) = ctx.op_attrs(op).get_symbol_ref("callee") else {
                    return;
                };
                let Some(signature) = resolve(op, name) else {
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
                    .op_attrs(op)
                    .get_type("signature")
                    .and_then(|ty| func::FuncSig::from_type_ref(ctx, ty))
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

fn operation_verifier_error(
    ctx: &IrContext,
    op: OpRef,
    detail: impl Into<String>,
) -> ValidationError {
    ValidationError::Operation {
        message: format!(
            "operation verifier failed for {} ({}): {}",
            ctx.op(op).name.display(ctx),
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
    let mut available = HashMap::<ValueRef, usize>::default();
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
    let Some(interface) = DynBranch::get(ctx, op) else {
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
    let successors = match interface.successors(ctx) {
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
    let raw_successors = ctx.op_successors(op);
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
    for (index, (edge, raw_successor)) in
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
    let transfer = match DynRegionBranch::value_transfer(ctx, owner, point, successor) {
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
    let Some(interface) = DynRegionBranch::get(ctx, op) else {
        return;
    };
    let successors = match interface.successors(ctx, RegionBranchPoint::Parent) {
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
    let mut unique = HashSet::default();
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
            // independently through DynRegionBranchTerminator below.
        }
    }
}

fn validate_region_branch_terminator_interface(
    ctx: &IrContext,
    op: OpRef,
    errors: &mut Vec<ValidationError>,
) {
    if DynRegionBranchTerminator::get(ctx, op).is_none() {
        return;
    }
    if !ctx.op_results(op).is_empty() {
        errors.push(operation_verifier_error(
            ctx,
            op,
            "RegionBranchTerminator must be resultless",
        ));
    }
    if ctx.op_has_successors(op) {
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
        if let Some(interface) = DynRegionBranch::get(ctx, owner) {
            match interface.successors(ctx, point) {
                Ok(successors) => {
                    if successors.as_slice().is_empty() {
                        errors.push(operation_verifier_error(
                            ctx,
                            op,
                            "owning RegionBranch reports no successor for this terminator",
                        ));
                        return;
                    }
                    let mut unique = HashSet::default();
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

/// Return whether `op` is a resultless transfer that may terminate a
/// structured `core.never` region without an `scf.yield`.
pub fn is_proper_tail_terminator(ctx: &IrContext, op: OpRef) -> bool {
    use crate::dialect::{func, scf};
    let name = ctx.op(op).name;
    func::TailCall::matches(ctx, op)
        || func::TailCallIndirect::matches(ctx, op)
        || func::Unreachable::matches(ctx, op)
        || (name.dialect(ctx) == "ability"
            && matches!(
                name.strip_dialect(ctx),
                "perform" | "handle_dispatch" | "exit" | "handle" | "abort"
            ))
        || name.is(ctx, "effect.dispatch_cps")
        || scf::If::matches(ctx, op)
        || scf::Switch::matches(ctx, op)
}

fn validate_scf_if_structure(ctx: &IrContext, op: OpRef, errors: &mut Vec<ValidationError>) {
    if !crate::dialect::scf::If::matches(ctx, op) {
        return;
    }

    // The schema guarantees one condition operand and two regions.
    for (region_name, region) in ["then_region", "else_region"]
        .into_iter()
        .zip(ctx.op_regions(op))
    {
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

        if !crate::dialect::scf::Yield::matches(ctx, yield_op) {
            let never_result = match ctx.op_result_types(op) {
                [ty] => crate::dialect::core::Never::matches(ctx, *ty),
                _ => false,
            };
            if never_result && is_proper_tail_terminator(ctx, yield_op) {
                continue;
            }
            // An arm may leave the callable instead of yielding; the result
            // then comes from the arms that yield.
            if !ctx.op_results(op).is_empty() && DynCallableExit::prove(ctx, yield_op).is_ok() {
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

// ============================================================================
// Call arity validation
// ============================================================================

/// Whether `op` is a `func`, `wasm` or `clif` function definition.
fn is_arity_checked_function(ctx: &IrContext, op: OpRef) -> bool {
    crate::dialect::func::Func::matches(ctx, op)
        || crate::dialect::wasm::Func::matches(ctx, op)
        || crate::dialect::clif::Func::matches(ctx, op)
}

/// The parameter count of the function definition `op`.
fn parameter_count(ctx: &IrContext, op: OpRef) -> Option<usize> {
    if !is_arity_checked_function(ctx, op) {
        return None;
    }
    let func_ty = ctx.op_attrs(op).get_type("type")?;
    let func_ty = crate::dialect::func::FuncSig::from_type_ref(ctx, func_ty)?;
    Some(func_ty.inputs(ctx).len())
}

/// Walk all operations in a region tree and check call arity.
fn check_call_arity_in_region(
    ctx: &IrContext,
    region: RegionRef,
    symbols: &SymbolTable,
    enclosing_fn: &SymbolPath,
) {
    walk::walk_region::<std::convert::Infallible>(ctx, region, &mut |op| {
        let is_call = crate::dialect::func::Call::matches(ctx, op);
        let is_tail_call = crate::dialect::func::TailCall::matches(ctx, op);
        if !is_call && !is_tail_call {
            return std::ops::ControlFlow::Continue(walk::WalkAction::Advance);
        }

        let Some(callee_sym) = ctx.op_attrs(op).get_symbol_ref("callee") else {
            return std::ops::ControlFlow::Continue(walk::WalkAction::Advance);
        };

        if let Some(expected) = symbols
            .resolve(ctx, op, callee_sym)
            .and_then(|callee| parameter_count(ctx, callee))
        {
            let actual = ctx.op_operands(op).len();
            if actual != expected {
                ctx.report_warning(
                    ctx.op(op).location.span,
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
fn validate_call_arity(ctx: &IrContext, module: Module) {
    let symbols = SymbolTable::collect(ctx, module);
    let _ = walk::walk_op::<std::convert::Infallible>(ctx, module.op(), &mut |op| {
        if !is_arity_checked_function(ctx, op) {
            return std::ops::ControlFlow::Continue(walk::WalkAction::Advance);
        }
        if let Some(name) = qualified_name(ctx, op) {
            for func_region in ctx.op_regions(op) {
                check_call_arity_in_region(ctx, func_region, &symbols, &name);
            }
        }
        std::ops::ControlFlow::Continue(walk::WalkAction::Skip)
    });
}

/// Check that every `func.call` and `func.tail_call` passes as many
/// arguments as its callee's signature takes. A mismatch is a warning
/// reported to the context's diagnostics; it does not fail the run.
pub struct CheckCallArity;

impl Pass for CheckCallArity {
    type Target = core::Module;

    fn name(&self) -> &'static str {
        "check-call-arity"
    }

    fn run(
        &mut self,
        ctx: &mut IrContext,
        target: core::Module,
        _analyses: &mut AnalysisCache,
    ) -> PassRunResult {
        validate_call_arity(ctx, target.into());
        Ok(())
    }
}

/// In debug builds, reject a module with [`validate_value_integrity`]
/// errors, labeled with the pipeline point `boundary`. It does nothing in
/// release builds.
pub struct CheckValueIntegrity {
    pub boundary: &'static str,
}

impl Pass for CheckValueIntegrity {
    type Target = core::Module;

    fn name(&self) -> &'static str {
        "check-value-integrity"
    }

    fn run(
        &mut self,
        ctx: &mut IrContext,
        target: core::Module,
        _analyses: &mut AnalysisCache,
    ) -> PassRunResult {
        if cfg!(debug_assertions) {
            let result = validate_value_integrity(ctx, target.into());
            if !result.is_ok() {
                return Err(format!("value integrity errors {}: {result}", self.boundary).into());
            }
        }
        Ok(())
    }
}

/// Run all validations and combine results.
///
/// Local shapes and contextual contracts are separate checks. Known direct
/// callees, exact indirect signatures and registered return/tail owners are
/// checked here; undeclared runtime calls do not invent a signature.
///
/// Shared analyses such as the [`SymbolTable`] are queried through
/// `analyses`.
/// Bodyless declarations that share a link name import one external symbol,
/// so their `type` signatures must agree. Returns one message per declaration
/// that disagrees with the first declaration of its link name.
pub fn link_name_conflicts(ctx: &IrContext, module: Module) -> Vec<String> {
    let mut declared: HashMap<Symbol, (SymbolPath, crate::refs::TypeRef)> = HashMap::default();
    let mut errors = Vec::new();
    let _ = walk::walk_op::<()>(ctx, module.op(), &mut |op| {
        if !ctx.op_has_regions(op)
            && let Some(ty) = ctx.op_attrs(op).get_type("type")
            && let Some(name) = link_name(ctx, op)
            && let Some(path) = qualified_name(ctx, op)
        {
            match declared.get(&name) {
                Some((first, first_ty)) if *first_ty != ty => errors.push(format!(
                    "declarations @{first} and @{path} link as `{name}` with different \
                     signatures {} and {}",
                    crate::printer::print_type(ctx, *first_ty),
                    crate::printer::print_type(ctx, ty),
                )),
                Some(_) => {}
                None => {
                    declared.insert(name, (path, ty));
                }
            }
        }
        std::ops::ControlFlow::Continue(walk::WalkAction::Advance)
    });
    errors
}

pub fn validate_all(
    ctx: &IrContext,
    module: Module,
    analyses: &mut AnalysisCache,
) -> ValidationResult {
    let scope = validate_value_integrity(ctx, module);
    let uses = validate_use_chains(ctx, module);
    let ops = validate_operation_verifiers(ctx, module);
    let mut errors = scope.errors;
    errors.extend(uses.errors);
    errors.extend(ops.errors);
    let symbols = analyses.require::<SymbolTable>(ctx, module.op());
    errors.extend(function_contracts(ctx, module, &symbols).errors);
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
    let result = validate_all(ctx, module, &mut Default::default());
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
    use crate::context::OperationDataBuilder;
    use crate::dialect::{arith, core, func};
    use crate::location::Span;
    use crate::refs::{RegionRef, ValueRef};
    use crate::types::{Attribute, Location};
    use crate::{BlockArgData, BlockData, IrContext, RegionData, TypeDataBuilder};
    use smallvec::smallvec;
    fn test_location(ctx: &mut IrContext) -> Location {
        let path = ctx.intern_path("test.trb");
        Location::new(path, Span::new(0, 0))
    }

    fn make_i32_type(ctx: &mut IrContext) -> super::super::refs::TypeRef {
        TypeDataBuilder::new("core.i32").intern(ctx)
    }

    fn make_func_type(
        ctx: &mut IrContext,
        params: &[super::super::refs::TypeRef],
        ret: super::super::refs::TypeRef,
    ) -> super::super::refs::TypeRef {
        crate::dialect::func::func_sig(ctx, params.iter().copied(), [ret]).as_type_ref()
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
                    consumer_op.as_ref(),
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

    #[test]
    fn root_module_schema_is_verified() {
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(
            &mut ctx,
            "core.module @m { func.func @f() { func.return } }",
        );
        ctx.op_attrs_mut(module.op()).remove("sym_name");
        let text = validate_all(&ctx, module, &mut Default::default()).to_string();
        assert!(
            text.contains("core.module") && text.contains("missing required attribute `sym_name`"),
            "{text}"
        );

        let loc = test_location(&mut ctx);
        let bodyless = OperationDataBuilder::new(loc, "core.module")
            .attr("sym_name", Attribute::SymbolRef(SymbolPath::new(["m"])))
            .build(&mut ctx);
        let bodyless = ctx.create_op(bodyless);
        let bodyless = Module::new(&ctx, bodyless).unwrap();
        let text = validate_operation_verifiers(&ctx, bodyless).to_string();
        assert!(text.contains("expected 1 region(s), found 0"), "{text}");
    }

    fn empty_module(ctx: &mut IrContext) -> Module {
        crate::parser::parse_test_module(ctx, "core.module @test {}")
    }

    #[test]
    fn malformed_func_sig_counts_are_rejected_by_typed_and_whole_ir_validation() {
        let cases = [
            (
                "missing num_inputs",
                TypeDataBuilder::new("func.func_sig")
                    .attr(func::NUM_RESULTS_ATTR, Attribute::Int(0)),
                "missing required `num_inputs`",
            ),
            (
                "missing num_results",
                TypeDataBuilder::new("func.func_sig")
                    .attr(func::NUM_INPUTS_ATTR, Attribute::Int(0)),
                "missing required `num_results`",
            ),
            (
                "wrong num_inputs type",
                TypeDataBuilder::new("func.func_sig")
                    .attr(func::NUM_INPUTS_ATTR, Attribute::Bool(false))
                    .attr(func::NUM_RESULTS_ATTR, Attribute::Int(0)),
                "`num_inputs` must be a u32",
            ),
            (
                "wrong num_results type",
                TypeDataBuilder::new("func.func_sig")
                    .attr(func::NUM_INPUTS_ATTR, Attribute::Int(0))
                    .attr(func::NUM_RESULTS_ATTR, Attribute::Bool(false)),
                "`num_results` must be a u32",
            ),
            (
                "count sum mismatch",
                TypeDataBuilder::new("func.func_sig")
                    .attr(func::NUM_INPUTS_ATTR, Attribute::Int(1))
                    .attr(func::NUM_RESULTS_ATTR, Attribute::Int(0)),
                "must equal params length",
            ),
            (
                "negative input",
                TypeDataBuilder::new("func.func_sig")
                    .attr("num_inputs", Attribute::Int(-1))
                    .attr("num_results", Attribute::Int(0)),
                "`num_inputs` must be a u32",
            ),
            (
                "negative result",
                TypeDataBuilder::new("func.func_sig")
                    .attr("num_results", Attribute::Int(-1))
                    .attr("num_inputs", Attribute::Int(0)),
                "`num_results` must be a u32",
            ),
            (
                "input outside u32",
                TypeDataBuilder::new("func.func_sig")
                    .attr("num_inputs", Attribute::Int(u32::MAX as i128 + 1))
                    .attr("num_results", Attribute::Int(0)),
                "`num_inputs` must be a u32",
            ),
            (
                "result outside u32",
                TypeDataBuilder::new("func.func_sig")
                    .attr("num_results", Attribute::Int(u32::MAX as i128 + 1))
                    .attr("num_inputs", Attribute::Int(0)),
                "`num_results` must be a u32",
            ),
            (
                "multiple results",
                TypeDataBuilder::new("func.func_sig")
                    .attr(func::NUM_INPUTS_ATTR, Attribute::Int(0))
                    .attr(func::NUM_RESULTS_ATTR, Attribute::Int(2)),
                "supports at most one result",
            ),
        ];

        for (case, builder, expected) in cases {
            let mut ctx = IrContext::new();
            let module = empty_module(&mut ctx);
            let malformed = builder.intern(&mut ctx);
            assert!(
                func::FuncSig::from_type_ref(&ctx, malformed).is_none(),
                "{case} must fail typed validation"
            );
            for result in [
                validate_operation_verifiers(&ctx, module),
                validate_all(&ctx, module, &mut Default::default()),
            ] {
                let messages = operation_error_messages(&result);
                assert!(
                    messages.iter().any(|message| message.contains(expected)),
                    "{case}: {result}"
                );
            }
        }
    }

    fn operations_named(ctx: &IrContext, module: Module, name: crate::OpName) -> Vec<OpRef> {
        let mut operations = Vec::new();
        let body = module.body(ctx).expect("test module must have a body");
        walk::walk_region::<std::convert::Infallible>(ctx, body, &mut |op| {
            let data = ctx.op(op);
            if data.name == name {
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
        let mut yield_builder = OperationDataBuilder::new(loc, "scf.yield");
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
        let zero = arith::Const::operands()
            .value(Attribute::Int(0))
            .results(i32_ty)
            .build(ctx, loc);
        ctx.push_op(entry, zero.op_ref());
        let ret = func::Return::operands([zero.result(ctx)]).build(ctx, loc);
        ctx.push_op(entry, ret.op_ref());

        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });
        let func_ty = make_func_type(ctx, &[], i32_ty);
        let func_op = func::Func::operands()
            .sym_name("bad_if")
            .r#type(func_ty)
            .regions(body)
            .build(ctx, loc);

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
        let module_op = core::Module::operands()
            .sym_name("test")
            .regions(mod_region)
            .build(ctx, loc);
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

        let c0 = arith::Const::operands()
            .value(Attribute::Int(40))
            .results(i32_ty)
            .build(ctx, loc);
        ctx.push_op(entry_block, c0.op_ref());
        let c0_val = c0.result(ctx);

        let c1 = arith::Const::operands()
            .value(Attribute::Int(2))
            .results(i32_ty)
            .build(ctx, loc);
        ctx.push_op(entry_block, c1.op_ref());
        let c1_val = c1.result(ctx);

        let add_op = arith::Addi::operands(c0_val, c1_val).build(ctx, loc);
        ctx.push_op(entry_block, add_op.op_ref());
        let sum = add_op.result(ctx);

        let ret = func::Return::operands([sum]).build(ctx, loc);
        ctx.push_op(entry_block, ret.op_ref());

        let body_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry_block],
            parent_op: None,
        });

        let func_ty = make_func_type(ctx, &[], i32_ty);
        let func_op = func::Func::operands()
            .sym_name("add")
            .r#type(func_ty)
            .regions(body_region)
            .build(ctx, loc);

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
        let module = core::Module::operands()
            .sym_name("test")
            .regions(mod_region)
            .build(ctx, loc);

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
        let const_a = arith::Const::operands()
            .value(Attribute::Int(99))
            .results(i32_ty)
            .build(&mut ctx, loc);
        ctx.push_op(entry_a, const_a.op_ref());
        let stale_value = const_a.result(&ctx);
        let ret_a = func::Return::operands([stale_value]).build(&mut ctx, loc);
        ctx.push_op(entry_a, ret_a.op_ref());

        let body_a = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry_a],
            parent_op: None,
        });
        let func_ty = make_func_type(&mut ctx, &[], i32_ty);
        let func_a = func::Func::operands()
            .sym_name("func_a")
            .r#type(func_ty)
            .regions(body_a)
            .build(&mut ctx, loc);

        // func_b uses stale_value from func_a
        let entry_b = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let ret_b = func::Return::operands([stale_value]).build(&mut ctx, loc);
        ctx.push_op(entry_b, ret_b.op_ref());

        let body_b = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry_b],
            parent_op: None,
        });
        let func_b = func::Func::operands()
            .sym_name("func_b")
            .r#type(func_ty)
            .regions(body_b)
            .build(&mut ctx, loc);

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
        let module_op = core::Module::operands()
            .sym_name("test")
            .regions(mod_region)
            .build(&mut ctx, loc);
        let module = Module::new(&ctx, module_op.op_ref()).unwrap();

        let result = validate_value_integrity(&ctx, module);
        assert!(!result.is_ok(), "Should detect stale op result");
        let stale_errors = stale_value_errors(&result);
        assert_eq!(stale_errors.len(), 1);
        assert_eq!(stale_errors[0].0, "func_b");
        assert!(stale_errors[0].2.contains("arith.const"));

        // In debug builds the check pass rejects the module.
        if cfg!(debug_assertions) {
            let error = CheckValueIntegrity {
                boundary: "at test",
            }
            .run(&mut ctx, module_op, &mut AnalysisCache::new())
            .expect_err("a stale value fails the value integrity check");
            assert!(
                error.to_string().contains("value integrity errors at test"),
                "{error}"
            );
        }
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
        let ret_a = func::Return::operands([stale_block_arg]).build(&mut ctx, loc);
        ctx.push_op(entry_a, ret_a.op_ref());

        let body_a = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry_a],
            parent_op: None,
        });
        let func_ty = make_func_type(&mut ctx, &[i32_ty], i32_ty);
        let func_a = func::Func::operands()
            .sym_name("func_a")
            .r#type(func_ty)
            .regions(body_a)
            .build(&mut ctx, loc);

        // func_b uses the block arg from func_a
        let entry_b = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let ret_b = func::Return::operands([stale_block_arg]).build(&mut ctx, loc);
        ctx.push_op(entry_b, ret_b.op_ref());

        let body_b = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry_b],
            parent_op: None,
        });
        let func_ty_b = make_func_type(&mut ctx, &[], i32_ty);
        let func_b = func::Func::operands()
            .sym_name("func_b")
            .r#type(func_ty_b)
            .regions(body_b)
            .build(&mut ctx, loc);

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
        let module_op = core::Module::operands()
            .sym_name("test")
            .regions(mod_region)
            .build(&mut ctx, loc);
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
        let yield_then = OperationDataBuilder::new(loc, "scf.yield")
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
        let c1 = arith::Const::operands()
            .value(Attribute::Int(1))
            .results(i32_ty)
            .build(&mut ctx, loc);
        ctx.push_op(else_block, c1.op_ref());
        let c1_val = c1.result(&ctx);
        let sum = arith::Addi::operands(param, c1_val).build(&mut ctx, loc);
        ctx.push_op(else_block, sum.op_ref());
        let sum_val = sum.result(&ctx);
        let yield_else = OperationDataBuilder::new(loc, "scf.yield")
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
        let i1_ty = TypeDataBuilder::new("core.i1").intern(&mut ctx);
        let cond = arith::Const::operands()
            .value(Attribute::Int(1))
            .results(i1_ty)
            .build(&mut ctx, loc);
        ctx.push_op(entry, cond.op_ref());
        let cond_val = cond.result(&ctx);

        // scf.if
        let if_op = OperationDataBuilder::new(loc, "scf.if")
            .operand(cond_val)
            .result(i32_ty)
            .region(then_region)
            .region(else_region)
            .build(&mut ctx);
        let if_op = ctx.create_op(if_op);
        ctx.push_op(entry, if_op);
        let if_result = ctx.op_result(if_op, 0);

        let ret = func::Return::operands([if_result]).build(&mut ctx, loc);
        ctx.push_op(entry, ret.op_ref());

        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });
        let func_ty = make_func_type(&mut ctx, &[i32_ty], i32_ty);
        let func_op = func::Func::operands()
            .sym_name("nested_fn")
            .r#type(func_ty)
            .regions(body)
            .build(&mut ctx, loc);

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
        let module_op = core::Module::operands()
            .sym_name("test")
            .regions(mod_region)
            .build(&mut ctx, loc);
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
        let const_a = arith::Const::operands()
            .value(Attribute::Int(42))
            .results(i32_ty)
            .build(&mut ctx, loc);
        ctx.push_op(entry_a, const_a.op_ref());
        let value_from_a = const_a.result(&ctx);
        let ret_a = func::Return::operands([value_from_a]).build(&mut ctx, loc);
        ctx.push_op(entry_a, ret_a.op_ref());
        let body_a = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry_a],
            parent_op: None,
        });
        let func_ty = make_func_type(&mut ctx, &[], i32_ty);
        let func_a = func::Func::operands()
            .sym_name("func_a")
            .r#type(func_ty)
            .regions(body_a)
            .build(&mut ctx, loc);

        // func_b uses value_from_a (stale!)
        let entry_b = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let local = arith::Const::operands()
            .value(Attribute::Int(1))
            .results(i32_ty)
            .build(&mut ctx, loc);
        ctx.push_op(entry_b, local.op_ref());
        let local_val = local.result(&ctx);
        let add_op = arith::Addi::operands(value_from_a, local_val).build(&mut ctx, loc);
        ctx.push_op(entry_b, add_op.op_ref());
        let ret_b = func::Return::operands([value_from_a]).build(&mut ctx, loc);
        ctx.push_op(entry_b, ret_b.op_ref());
        let body_b = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry_b],
            parent_op: None,
        });
        let func_b = func::Func::operands()
            .sym_name("func_b")
            .r#type(func_ty)
            .regions(body_b)
            .build(&mut ctx, loc);

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
        let module_op = core::Module::operands()
            .sym_name("test")
            .regions(mod_region)
            .build(&mut ctx, loc);
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
        let const_a = arith::Const::operands()
            .value(Attribute::Int(99))
            .results(i32_ty)
            .build(&mut ctx, loc);
        ctx.push_op(entry_a, const_a.op_ref());
        let stale_value = const_a.result(&ctx);
        let ret_a = func::Return::operands([stale_value]).build(&mut ctx, loc);
        ctx.push_op(entry_a, ret_a.op_ref());
        let body_a = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry_a],
            parent_op: None,
        });
        let func_ty = make_func_type(&mut ctx, &[], i32_ty);
        let func_a = func::Func::operands()
            .sym_name("func_a")
            .r#type(func_ty)
            .regions(body_a)
            .build(&mut ctx, loc);

        // func_b (wasm.func) uses stale_value from func_a
        let entry_b = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let ret_b = func::Return::operands([stale_value]).build(&mut ctx, loc);
        ctx.push_op(entry_b, ret_b.op_ref());
        let body_b = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry_b],
            parent_op: None,
        });
        // Build wasm.func manually
        let wasm_func_data = OperationDataBuilder::new(loc, "wasm.func")
            .attr("sym_name", Attribute::String(ctx.intern_str("func_b")))
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
        let module_op = core::Module::operands()
            .sym_name("test")
            .regions(mod_region)
            .build(&mut ctx, loc);
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
        let i1_ty = TypeDataBuilder::new("core.i1").intern(&mut ctx);

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
        let inner_const = arith::Const::operands()
            .value(Attribute::Int(42))
            .results(i32_ty)
            .build(&mut ctx, loc);
        ctx.push_op(then_block, inner_const.op_ref());
        let inner_val = inner_const.result(&ctx);
        let yield_then = OperationDataBuilder::new(loc, "scf.yield")
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
        let else_const = arith::Const::operands()
            .value(Attribute::Int(0))
            .results(i32_ty)
            .build(&mut ctx, loc);
        ctx.push_op(else_block, else_const.op_ref());
        let else_val = else_const.result(&ctx);
        let yield_else = OperationDataBuilder::new(loc, "scf.yield")
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
        let cond_op = arith::Const::operands()
            .value(Attribute::Int(1))
            .results(i1_ty)
            .build(&mut ctx, loc);
        ctx.push_op(entry, cond_op.op_ref());
        let cond = cond_op.result(&ctx);

        // scf.if
        let if_data = OperationDataBuilder::new(loc, "scf.if")
            .operand(cond)
            .result(i32_ty)
            .region(then_region)
            .region(else_region)
            .build(&mut ctx);
        let if_op = ctx.create_op(if_data);
        ctx.push_op(entry, if_op);

        // BUG: outer block uses %inner_val which is defined only inside the then region
        let ret = func::Return::operands([inner_val]).build(&mut ctx, loc);
        ctx.push_op(entry, ret.op_ref());

        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });
        let func_ty = make_func_type(&mut ctx, &[], i32_ty);
        let func_op = func::Func::operands()
            .sym_name("bad_scope")
            .r#type(func_ty)
            .regions(body)
            .build(&mut ctx, loc);

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
        let module_op = core::Module::operands()
            .sym_name("test")
            .regions(mod_region)
            .build(&mut ctx, loc);
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
    fn use_chain_detects_missing_and_mismatched_operand_entries() {
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @f(%x: core.i32, %y: core.i32) {
    func.call %x {callee = @consume}
    func.call %y {callee = @consume}
    func.return
  }
  func.func @consume(%value: core.i32)
}"#,
        );
        let function = module.ops(&ctx)[0];
        let block = ctx.region(ctx.op_region(function, 0).unwrap()).blocks[0];
        let first = ctx.block(block).ops[0];
        let second = ctx.block(block).ops[1];
        assert!(validate_use_chains(&ctx, module).is_ok());
        // Swap operand lists without updating either recorded use: each call
        // now has one missing use-chain entry and one mismatched recorded use.
        let first_args = std::mem::take(&mut ctx.op_mut(first).operands);
        let second_args = std::mem::replace(&mut ctx.op_mut(second).operands, first_args);
        ctx.op_mut(first).operands = second_args;
        let result = validate_use_chains(&ctx, module);
        assert_eq!(result.errors.len(), 4, "{result}");
        let message = result.to_string();
        assert!(message.contains("no use-chain entry exists"), "{message}");
        assert!(message.contains("no such operand exists"), "{message}");
    }

    #[test]
    fn use_chain_checks_unused_results_and_nested_block_arguments() {
        for use_argument in [false, true] {
            for placement in ["detached", "other_module", "missing_operand"] {
                let mut ctx = IrContext::new();
                let module = crate::parser::parse_test_module(
                    &mut ctx,
                    r#"core.module @test {
  func.func @f(%x: core.i32) {
    %unused = arith.const {value = 0} : core.i32
    func.return
  }
}"#,
                );
                let function = module.ops(&ctx)[0];
                let block = ctx.region(ctx.op_region(function, 0).unwrap()).blocks[0];
                let value = if use_argument {
                    ctx.block_args(block)[0]
                } else {
                    ctx.op_results(ctx.block(block).ops[0])[0]
                };
                assert!(validate_use_chains(&ctx, module).is_ok());
                let loc = test_location(&mut ctx);
                let user = func::Return::operands([value])
                    .build(&mut ctx, loc)
                    .op_ref();
                match placement {
                    "other_module" => {
                        let other = crate::parser::parse_test_module(
                            &mut ctx,
                            "core.module @other { func.func @g() { func.return } }",
                        );
                        let other_block = ctx.region(other.body(&ctx).unwrap()).blocks[0];
                        ctx.push_op(other_block, user);
                    }
                    "missing_operand" => {
                        ctx.push_op(block, user);
                        let _operands = std::mem::take(&mut ctx.op_mut(user).operands);
                    }
                    _ => {}
                }
                let result = validate_use_chains(&ctx, module);
                assert_eq!(result.errors.len(), 1, "{placement}: {result}");
                assert!(
                    result.to_string().contains("no such operand exists"),
                    "{result}"
                );
            }
        }
    }

    // ========================================================================
    // Call arity validation tests (textual IR)
    // ========================================================================

    #[test]
    fn tail_call_indirect_typed_cps_transfer_passes() {
        let input = r#"core.module @test {
  func.func @main(%k: closure.closure<func.func_sig<(core.i32) -> core.never>>, %value: core.i32) -> core.never {
    func.tail_call_indirect %k, %value {signature = func.func_sig<(core.i32) -> core.never>}
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
  func.func @main(%k: closure.closure<func.func_sig<(core.i32) -> core.i32>>, %value: core.i1) -> core.never {
    func.tail_call_indirect %k, %value {signature = func.func_sig<(core.i32) -> core.i32>}
  }
}"#;
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(&mut ctx, input);
        let result = validate_all(&ctx, module, &mut Default::default());
        let text = result.to_string();
        assert!(text.contains("caller/callee result lists differ"), "{text}");
        assert!(
            text.contains("operands `args`: expected S::Inputs = (core.i32), found (core.i1)"),
            "{text}"
        );
    }

    #[test]
    fn tail_call_indirect_rejects_malformed_shapes_and_accepts_bare_function() {
        let input = r#"core.module @test {
  !never = func.func_sig<() -> core.never>
  !unary = func.func_sig<(core.i32) -> core.never>
  func.func @result(%k: closure.closure<!never>) -> core.never {
    %bad = func.tail_call_indirect %k {signature = !never} : core.i32
  }
  func.func @not_last(%k: !never) -> core.never {
    func.tail_call_indirect %k {signature = !never}
    func.unreachable
  }
  func.func @missing() -> core.never {
    func.tail_call_indirect {signature = !never}
  }
  func.func @unsigned(%k: !never) -> core.never {
    func.tail_call_indirect %k
  }
  func.func @bad_closure(%k: closure.closure) -> core.never {
    func.tail_call_indirect %k {signature = !never}
  }
  func.func @mismatched(%k: !unary) -> core.never {
    func.tail_call_indirect %k {signature = !never}
  }
  func.func @erased(%k: core.ptr) -> core.never {
    func.tail_call_indirect %k {signature = !never}
  }
  func.func @direct_function(%k: !never) -> core.never {
    func.tail_call_indirect %k {signature = !never}
  }
  func.func @arity(%k: closure.closure<!unary>) -> core.never {
    func.tail_call_indirect %k {signature = !unary}
  }
}"#;
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(&mut ctx, input);
        let result = validate_operation_verifiers(&ctx, module);
        let text = result.to_string();
        for expected in [
            "expected 0 result(s), found 1",
            "must terminate its block",
            "expected at least 1 operand(s), found 0",
            "missing required attribute `signature`",
            "closure type must contain exactly one function type",
            "exact indirect signature differs from typed callee",
            "operands `args`: expected S::Inputs = (core.i32), found ()",
        ] {
            assert!(text.contains(expected), "missing `{expected}`:\n{text}");
        }
        assert_eq!(result.errors.len(), 7, "{text}");
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
    fn check_passes_accept_valid_ir_and_report_arity_as_a_warning() {
        use crate::pass::PassManager;
        use std::cell::RefCell;
        use std::rc::Rc;

        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @add(%0: core.i32, %1: core.i32) -> core.i32 {
    %2 = arith.addi %0, %1 : core.i32
    func.return %2
  }
  func.func @main() -> core.i32 {
    %0 = arith.const {value = 1} : core.i32
    %1 = func.call %0 {callee = @add} : core.i32
    func.return %1
  }
}"#,
        );
        let root = core::Module::from_op(&ctx, module.op()).expect("core.module");
        let passes = Rc::new(RefCell::new(Vec::new()));
        let seen = Rc::clone(&passes);
        let mut pm = PassManager::new();
        pm.add_pass(CheckValueIntegrity { boundary: "test" })
            .add_pass(CheckCallArity);
        pm.with_instrumentation(move |_ctx, name, _op| seen.borrow_mut().push(name.to_owned()));

        pm.run(&mut ctx, root, &mut AnalysisCache::new())
            .expect("valid IR passes both checks; an arity mismatch is only a warning");

        assert_eq!(
            *passes.borrow(),
            ["check-value-integrity", "check-call-arity"]
        );
        let diagnostics = ctx.diagnostics();
        assert_eq!(diagnostics.len(), 1);
        assert!(
            diagnostics[0].message.contains("1 argument(s), expected 2"),
            "{}",
            diagnostics[0].message
        );
    }

    #[test]
    fn call_arity_is_checked_against_module_callees() {
        // (case, input, fragments of the single expected diagnostic; empty
        // when the module must pass)
        let cases: [(&str, &str, &[&str]); 6] = [
            (
                "too few args: add expects 2 params, caller passes 1",
                r#"core.module @test {
  func.func @add(%0: core.i32, %1: core.i32) -> core.i32 {
    %2 = arith.addi %0, %1 : core.i32
    func.return %2
  }
  func.func @main() -> core.i32 {
    %0 = arith.const {value = 1} : core.i32
    %1 = func.call %0 {callee = @add} : core.i32
    func.return %1
  }
}"#,
                &["main", "add", "1 argument(s), expected 2"],
            ),
            (
                "too many args: add expects 1 param, caller passes 3",
                r#"core.module @test {
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
}"#,
                &["3 argument(s), expected 1"],
            ),
            (
                "tail_call passes 1 arg to a 2-param callee",
                r#"core.module @test {
  func.func @add(%0: core.i32, %1: core.i32) -> core.i32 {
    %2 = arith.addi %0, %1 : core.i32
    func.return %2
  }
  func.func @main() -> core.i32 {
    %0 = arith.const {value = 1} : core.i32
    func.tail_call %0 {callee = @add}
  }
}"#,
                &["add", "1 argument(s), expected 2"],
            ),
            (
                "args passed to a zero-param function",
                r#"core.module @test {
  func.func @unit() -> core.i32 {
    %0 = arith.const {value = 0} : core.i32
    func.return %0
  }
  func.func @main() -> core.i32 {
    %0 = arith.const {value = 1} : core.i32
    %1 = func.call %0 {callee = @unit} : core.i32
    func.return %1
  }
}"#,
                &["1 argument(s), expected 0"],
            ),
            (
                "correct arity passes",
                r#"core.module @test {
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
}"#,
                &[],
            ),
            (
                "callee not defined in this module is skipped",
                r#"core.module @test {
  func.func @main() -> core.i32 {
    %0 = arith.const {value = 1} : core.i32
    %1 = func.call %0 {callee = @extern_fn} : core.i32
    func.return %1
  }
}"#,
                &[],
            ),
        ];

        for (case, input, fragments) in cases {
            let mut ctx = IrContext::new();
            let module = crate::parser::parse_test_module(&mut ctx, input);

            validate_call_arity(&ctx, module);
            let diagnostics = ctx.diagnostics();
            if fragments.is_empty() {
                assert!(diagnostics.is_empty(), "{case}: {:?}", &*diagnostics);
                continue;
            }
            let [diagnostic] = &*diagnostics else {
                panic!(
                    "{case}: expected exactly one diagnostic, got {:?}",
                    &*diagnostics
                );
            };
            for fragment in fragments {
                assert!(
                    diagnostic.message.contains(fragment),
                    "{case}: missing {fragment:?} in {:?}",
                    diagnostic.message
                );
            }
        }
    }

    #[test]
    fn cmpf_subset_predicate_passes() {
        let input = r#"core.module @test {
  func.func @main(%0: core.f64, %1: core.f64) -> core.i1 {
    %2 = arith.cmpf %0, %1 {predicate = "une"} : core.i1
    func.return %2
  }
}"#;
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(&mut ctx, input);

        let result = validate_operation_verifiers(&ctx, module);
        assert!(result.is_ok(), "{result}");
    }

    #[test]
    fn textual_operation_verifier_rejections() {
        // (op, input, fragments of the single expected operation error)
        let cases: [(&str, &str, &[&str]); 4] = [
            (
                "arith.cmpf",
                r#"core.module @test {
  func.func @main(%0: core.f64, %1: core.f64) -> core.i1 {
    %2 = arith.cmpf %0, %1 {predicate = "ueq"} : core.i1
    func.return %2
  }
}"#,
                &["unsupported predicate 'ueq'"],
            ),
            (
                "arith.cmpf",
                r#"core.module @test {
  func.func @main(%0: core.f64, %1: core.f64) -> core.i1 {
    %2 = arith.cmpf %0, %1 : core.i1
    func.return %2
  }
}"#,
                &["missing required attribute `predicate`"],
            ),
            (
                "scf.if",
                r#"core.module @test {
  func.func @main(%cond: core.i1, %x: core.i32) -> core.i32 {
    %r = scf.if %cond : core.i32 {
      scf.yield
    } {
      scf.yield %x
    }
    func.return %r
  }
}"#,
                &["forwards 0 value(s)", "successor expects 1 input(s)"],
            ),
            (
                "scf.if",
                r#"core.module @test {
  func.func @main(%cond: core.i1, %x: core.i32) -> core.i32 {
    %r = scf.if %cond : core.i32 {
      %zero = arith.const {value = 0} : core.i32
    } {
      scf.yield %x
    }
    func.return %r
  }
}"#,
                &["then_region must terminate with scf.yield"],
            ),
        ];

        for (op, input, fragments) in cases {
            let mut ctx = IrContext::new();
            let module = crate::parser::parse_test_module(&mut ctx, input);

            let result = validate_operation_verifiers(&ctx, module);
            let operation_errors = operation_error_messages(&result);
            let [error] = operation_errors.as_slice() else {
                panic!(
                    "{op} {fragments:?}: expected one operation error, got {operation_errors:?}"
                );
            };
            let header = format!("operation verifier failed for {op}");
            for fragment in std::iter::once(header.as_str()).chain(fragments.iter().copied()) {
                assert!(
                    error.contains(fragment),
                    "{op} {fragments:?}: missing {fragment:?} in {error}"
                );
            }
        }
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
        let branches = operations_named(&ctx, module, crate::dialect::cf::Br::NAME);
        assert_eq!(branches.len(), 1);
        let branch = branches[0];
        let interface = DynBranch::get(&ctx, branch).expect("cf.br must implement Branch");
        let successors = interface.successors(&ctx).unwrap();
        assert_eq!(successors.as_slice().len(), 1);
        let edge = &successors.as_slice()[0];
        assert_eq!(edge.forwarded.as_slice(), ctx.op_operands(branch));
        assert_eq!(ctx.block_args(edge.block).len(), 1);
    }

    #[test]
    fn cf_switch_interface_reports_the_default_and_every_target() {
        let module_text = |cases: &str, successors: &str| {
            format!(
                r#"core.module @test {{
  func.func @main(%choice: core.i32) {{
    ^entry:
      cf.switch %choice [{successors}] {{cases = {cases}}}
    ^fallback:
      func.return
    ^first:
      func.return
    ^second:
      func.return
  }}
}}"#
            )
        };
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(
            &mut ctx,
            &module_text("[4, -1]", "^fallback, ^first, ^second"),
        );

        let result = validate_operation_verifiers(&ctx, module);
        assert!(result.is_ok(), "{result}");
        let branch = operations_named(&ctx, module, crate::dialect::cf::Switch::NAME)[0];
        let successors = DynBranch::get(&ctx, branch)
            .expect("cf.switch must implement Branch")
            .successors(&ctx)
            .unwrap();
        let blocks: crate::BlockList = ctx.op_successors(branch).collect();
        assert_eq!(blocks.len(), 3);
        assert!(
            successors
                .as_slice()
                .iter()
                .zip(&blocks)
                .all(|(edge, &block)| edge.block == block && edge.forwarded.is_empty())
        );

        for (cases, successors, expected) in [
            ("[4, -1]", "^fallback, ^first", "2 case(s) but 1 target(s)"),
            ("[4, 4]", "^fallback, ^first, ^second", "duplicate case 4"),
            (
                "[-1, 4294967295]",
                "^fallback, ^first, ^second",
                "duplicate case 4294967295",
            ),
            (
                "[4, 4294967296]",
                "^fallback, ^first, ^second",
                "case 4294967296 is not a value of a 32-bit integer",
            ),
        ] {
            let mut ctx = IrContext::new();
            let module =
                crate::parser::parse_test_module(&mut ctx, &module_text(cases, successors));
            let result = validate_operation_verifiers(&ctx, module);
            let errors = operation_error_messages(&result);
            assert!(
                errors.iter().any(|error| error.contains(expected)),
                "{errors:?}"
            );
        }
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
        let branches = operations_named(&ctx, module, crate::dialect::cf::CondBr::NAME);
        assert_eq!(branches.len(), 1);
        let branch = branches[0];
        let successors = DynBranch::get(&ctx, branch)
            .unwrap()
            .successors(&ctx)
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
        let ifs = operations_named(&ctx, module, crate::dialect::scf::If::NAME);
        assert_eq!(ifs.len(), 2);
        for if_op in ifs {
            let interface = DynRegionBranch::get(&ctx, if_op).unwrap();
            let successors = interface
                .successors(&ctx, RegionBranchPoint::Parent)
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
        let loops = operations_named(&ctx, module, crate::dialect::scf::Loop::NAME);
        assert_eq!(loops.len(), 1);
        let loop_op = loops[0];
        let interface = DynRegionBranch::get(&ctx, loop_op).unwrap();
        let entry = interface
            .successors(&ctx, RegionBranchPoint::Parent)
            .unwrap();
        assert_eq!(entry.as_slice().len(), 1);
        let RegionSuccessor::Region(body) = entry.as_slice()[0] else {
            unreachable!()
        };
        let continues = operations_named(&ctx, module, crate::dialect::scf::Continue::NAME);
        assert_eq!(continues.len(), 1);
        let continue_op = continues[0];
        let backedge = DynRegionBranch::value_transfer(
            &ctx,
            loop_op,
            RegionBranchPoint::Terminator(continue_op),
            RegionSuccessor::Region(body),
        )
        .unwrap();
        assert_eq!(backedge.forwarded.as_slice().len(), 1);
        assert_eq!(backedge.inputs.as_slice().len(), 1);

        let breaks = operations_named(&ctx, module, crate::dialect::scf::Break::NAME);
        assert_eq!(breaks.len(), 1);
        let break_op = breaks[0];
        let exit = DynRegionBranch::value_transfer(
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
    %unit = core.nil_value : core.nil
    func.return %unit
  }
}"#;
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(&mut ctx, input);

        let result = validate_operation_verifiers(&ctx, module);
        assert!(result.is_ok(), "{result}");
        let switches = operations_named(&ctx, module, crate::dialect::scf::Switch::NAME);
        assert_eq!(switches.len(), 1);
        let switch = switches[0];
        let interface = DynRegionBranch::get(&ctx, switch).unwrap();
        let successors = interface
            .successors(&ctx, RegionBranchPoint::Parent)
            .unwrap();
        assert_eq!(successors.as_slice().len(), 2);
        assert!(
            successors
                .as_slice()
                .iter()
                .all(|successor| matches!(successor, RegionSuccessor::Region(_)))
        );
        for wrapper in operations_named(&ctx, module, crate::dialect::scf::Case::NAME)
            .into_iter()
            .chain(operations_named(
                &ctx,
                module,
                crate::dialect::scf::Default::NAME,
            ))
        {
            assert!(DynRegionBranch::get(&ctx, wrapper).is_some());
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

        let if_op = operations_named(&ctx, module, crate::dialect::scf::If::NAME)[0];
        let switch = operations_named(&ctx, module, crate::dialect::scf::Switch::NAME)[0];
        let case = operations_named(&ctx, module, crate::dialect::scf::Case::NAME)[0];
        let default = operations_named(&ctx, module, crate::dialect::scf::Default::NAME)[0];
        let yields = operations_named(&ctx, module, crate::dialect::scf::Yield::NAME);
        let if_yield = *yields
            .iter()
            .find(|&&op| direct_owner(op) == Some(if_op))
            .unwrap();
        let case_yield = *yields
            .iter()
            .find(|&&op| direct_owner(op) == Some(case))
            .unwrap();

        let if_interface = DynRegionBranch::get(&ctx, if_op).unwrap();
        let error = if_interface
            .successors(&ctx, RegionBranchPoint::Terminator(case_yield))
            .unwrap_err();
        assert!(error.is_not_applicable());
        assert!(error.to_string().contains("not an scf.if region yield"));
        assert!(
            if_interface
                .entry_successor_operands(&ctx, RegionSuccessor::Parent)
                .is_err()
        );

        let switch_interface = DynRegionBranch::get(&ctx, switch).unwrap();
        assert_eq!(
            switch_interface
                .successors(&ctx, RegionBranchPoint::Terminator(case_yield),)
                .unwrap()
                .as_slice(),
            &[RegionSuccessor::Parent]
        );
        assert!(
            switch_interface
                .successors(&ctx, RegionBranchPoint::Terminator(if_yield),)
                .unwrap_err()
                .is_not_applicable()
        );
        assert!(
            switch_interface
                .entry_successor_operands(&ctx, RegionSuccessor::Parent)
                .is_err()
        );

        for wrapper in [case, default] {
            let interface = DynRegionBranch::get(&ctx, wrapper).unwrap();
            assert!(
                interface
                    .successors(&ctx, RegionBranchPoint::Terminator(if_yield),)
                    .unwrap_err()
                    .is_not_applicable()
            );
            assert!(
                interface
                    .entry_successor_operands(&ctx, RegionSuccessor::Parent)
                    .is_err()
            );
        }

        let loops = operations_named(&ctx, module, crate::dialect::scf::Loop::NAME);
        let result_loop = *loops
            .iter()
            .find(|&&op| !ctx.op_results(op).is_empty())
            .unwrap();
        let result_loop_interface = DynRegionBranch::get(&ctx, result_loop).unwrap();
        assert!(
            result_loop_interface
                .successors(&ctx, RegionBranchPoint::Terminator(if_yield),)
                .unwrap_err()
                .is_not_applicable()
        );
        assert!(
            result_loop_interface
                .entry_successor_operands(&ctx, RegionSuccessor::Parent)
                .is_err()
        );
        let loop_entry = result_loop_interface
            .successors(&ctx, RegionBranchPoint::Parent)
            .unwrap();
        assert_eq!(loop_entry.as_slice().len(), 1);
        let RegionSuccessor::Region(loop_body) = loop_entry.as_slice()[0] else {
            unreachable!()
        };

        let continues = operations_named(&ctx, module, crate::dialect::scf::Continue::NAME);
        let continue_op = *continues
            .iter()
            .find(|&&op| direct_owner(op) == Some(result_loop))
            .unwrap();
        let stray_continue = *continues
            .iter()
            .find(|&&op| {
                direct_owner(op)
                    .is_some_and(|owner| ctx.op(owner).name.dialect(&ctx) == func::DIALECT_NAME)
            })
            .unwrap();
        assert!(
            DynRegionBranchTerminator::get(&ctx, continue_op)
                .unwrap()
                .successor_operands(&ctx, RegionSuccessor::Parent)
                .is_err()
        );
        let unrelated_region = ctx.op_region(case, 0).unwrap();
        assert!(
            DynRegionBranchTerminator::get(&ctx, continue_op)
                .unwrap()
                .successor_operands(&ctx, RegionSuccessor::Region(unrelated_region),)
                .is_err()
        );
        assert!(
            DynRegionBranchTerminator::get(&ctx, stray_continue)
                .unwrap()
                .successor_operands(&ctx, RegionSuccessor::Region(loop_body),)
                .unwrap_err()
                .to_string()
                .contains("no enclosing scf.loop")
        );
        assert!(
            DynRegionBranchTerminator::get(&ctx, case_yield)
                .unwrap()
                .successor_operands(&ctx, RegionSuccessor::Region(loop_body))
                .is_err()
        );

        let breaks = operations_named(&ctx, module, crate::dialect::scf::Break::NAME);
        let result_break = *breaks
            .iter()
            .find(|&&op| direct_owner(op) == Some(result_loop))
            .unwrap();
        assert!(
            DynRegionBranchTerminator::get(&ctx, result_break)
                .unwrap()
                .successor_operands(&ctx, RegionSuccessor::Region(loop_body))
                .is_err()
        );
        let resultless_break = *breaks
            .iter()
            .find(|&&op| {
                direct_owner(op).is_some_and(|owner| {
                    ctx.op(owner).name.is(&ctx, "scf.loop") && ctx.op_results(owner).is_empty()
                })
            })
            .unwrap();
        assert!(
            DynRegionBranchTerminator::get(&ctx, resultless_break)
                .unwrap()
                .successor_operands(&ctx, RegionSuccessor::Parent)
                .unwrap()
                .is_empty()
        );
        let stray_break = *breaks
            .iter()
            .find(|&&op| {
                direct_owner(op)
                    .is_some_and(|owner| ctx.op(owner).name.dialect(&ctx) == func::DIALECT_NAME)
            })
            .unwrap();
        assert!(
            DynRegionBranchTerminator::get(&ctx, stray_break)
                .unwrap()
                .successor_operands(&ctx, RegionSuccessor::Parent)
                .unwrap_err()
                .to_string()
                .contains("no enclosing scf.loop")
        );

        let constant = operations_named(&ctx, module, crate::dialect::arith::Const::NAME)[0];
        assert!(
            DynRegionBranch::value_transfer(
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
            DynRegionBranch::value_transfer(
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

        let branch = operations_named(&ctx, module, crate::dialect::cf::Br::NAME)[0];
        let cond_branch = operations_named(&ctx, module, crate::dialect::cf::CondBr::NAME)[0];
        let switches = operations_named(&ctx, module, crate::dialect::scf::Switch::NAME);
        let loop_op = operations_named(&ctx, module, crate::dialect::scf::Loop::NAME)[0];
        let case = operations_named(&ctx, module, crate::dialect::scf::Case::NAME)[0];
        let break_op = operations_named(&ctx, module, crate::dialect::scf::Break::NAME)[0];

        ctx.truncate_op_successors(branch, 0);
        assert!(
            DynBranch::get(&ctx, branch)
                .unwrap()
                .successors(&ctx)
                .is_err()
        );
        let count = ctx.op_successor_count(cond_branch);
        ctx.truncate_op_successors(cond_branch, count - 1);
        assert!(
            DynBranch::get(&ctx, cond_branch)
                .unwrap()
                .successors(&ctx)
                .is_err()
        );
        ctx.clear_op_regions(loop_op);
        assert!(
            DynRegionBranch::get(&ctx, loop_op)
                .unwrap()
                .successors(&ctx, RegionBranchPoint::Parent)
                .is_err()
        );
        ctx.remove_op_operand(switches[0], 0);
        assert!(
            DynRegionBranch::get(&ctx, switches[0])
                .unwrap()
                .successors(&ctx, RegionBranchPoint::Parent)
                .is_err()
        );
        let second_switch_body = ctx.op_region(switches[1], 0).unwrap();
        ctx.region_mut(second_switch_body).blocks.clear();
        assert!(
            DynRegionBranch::get(&ctx, switches[1])
                .unwrap()
                .successors(&ctx, RegionBranchPoint::Parent)
                .is_err()
        );
        ctx.clear_op_regions(case);
        assert!(
            DynRegionBranch::get(&ctx, case)
                .unwrap()
                .successors(&ctx, RegionBranchPoint::Parent)
                .is_err()
        );
        ctx.remove_op_operand(break_op, 0);
        assert!(
            DynRegionBranchTerminator::get(&ctx, break_op)
                .unwrap()
                .successor_operands(&ctx, RegionSuccessor::Parent)
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
    fn scf_if_malformed_structure_is_rejected() {
        type RegionBuilder = fn(&mut IrContext, Location) -> RegionRef;
        fn yield_region(ctx: &mut IrContext, loc: Location) -> RegionRef {
            single_block_yield_region(ctx, loc, [])
        }
        fn multiblock_yield_region(ctx: &mut IrContext, loc: Location) -> RegionRef {
            let empty_block = |ctx: &mut IrContext| {
                ctx.create_block(BlockData {
                    location: loc,
                    args: vec![],
                    ops: smallvec![],
                    parent_region: None,
                })
            };
            let first = empty_block(ctx);
            let second = empty_block(ctx);
            let yield_op = OperationDataBuilder::new(loc, "scf.yield").build(ctx);
            let yield_op = ctx.create_op(yield_op);
            ctx.push_op(first, yield_op);
            ctx.create_region(RegionData {
                location: loc,
                blocks: smallvec![first, second],
                parent_op: None,
            })
        }

        // (case, condition operand count, regions, expected error fragment)
        let cases: [(&str, i64, &[RegionBuilder], &str); 3] = [
            (
                "two condition operands",
                2,
                &[yield_region, yield_region],
                "expected 1 operand(s), found 2",
            ),
            (
                "one region",
                1,
                &[yield_region],
                "expected 2 region(s), found 1",
            ),
            (
                "multi-block then region",
                1,
                &[multiblock_yield_region, yield_region],
                "then_region expects 1 block, found 2",
            ),
        ];

        for (case, condition_count, regions, expected) in cases {
            let mut ctx = IrContext::new();
            let loc = test_location(&mut ctx);
            let i32_ty = make_i32_type(&mut ctx);
            let i1_ty = TypeDataBuilder::new("core.i1").intern(&mut ctx);

            let entry = ctx.create_block(BlockData {
                location: loc,
                args: vec![],
                ops: smallvec![],
                parent_region: None,
            });
            let mut if_op = OperationDataBuilder::new(loc, "scf.if");
            for index in 0..condition_count {
                let cond = arith::Const::operands()
                    .value(Attribute::Int(i128::from(1 - index)))
                    .results(i1_ty)
                    .build(&mut ctx, loc);
                ctx.push_op(entry, cond.op_ref());
                if_op = if_op.operand(cond.result(&ctx));
            }
            for region in regions {
                if_op = if_op.region(region(&mut ctx, loc));
            }
            let if_op = if_op.build(&mut ctx);
            let if_op = ctx.create_op(if_op);
            let module = wrap_if_in_module(&mut ctx, loc, i32_ty, entry, if_op);

            let result = validate_operation_verifiers(&ctx, module);
            let operation_errors = operation_error_messages(&result);
            let [error] = operation_errors.as_slice() else {
                panic!("{case}: expected one operation error, got {operation_errors:?}");
            };
            assert!(
                error.contains("operation verifier failed for scf.if"),
                "{case}: {error}"
            );
            assert!(
                error.contains(expected),
                "{case}: missing {expected:?} in {error}"
            );
        }
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
        let loop_data = OperationDataBuilder::new(loc, "scf.loop")
            .result(i32_ty)
            .result(i32_ty)
            .region(loop_region)
            .build(&mut ctx);
        let loop_op = ctx.create_op(loop_data);
        ctx.push_op(entry, loop_op);

        let discriminant = arith::Const::operands()
            .value(Attribute::Int(0))
            .results(i32_ty)
            .build(&mut ctx, loc);
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
        let switch_data = OperationDataBuilder::new(loc, "scf.switch")
            .operand(discriminant.result(&ctx))
            .result(i32_ty)
            .region(switch_region)
            .build(&mut ctx);
        let switch_op = ctx.create_op(switch_data);
        ctx.push_op(entry, switch_op);

        let ret = func::Return::operands(std::iter::empty()).build(&mut ctx, loc);
        ctx.push_op(entry, ret.op_ref());
        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });
        let func_ty = make_func_type(&mut ctx, &[], i32_ty);
        let func_op = func::Func::operands()
            .sym_name("malformed")
            .r#type(func_ty)
            .regions(body)
            .build(&mut ctx, loc);
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
        let module_op = core::Module::operands()
            .sym_name("test")
            .regions(module_region)
            .build(&mut ctx, loc);
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
    %2 = arith.cmpf %0, %1 {predicate = "one"} : core.i1
    func.return %2
  }
}"#;
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(&mut ctx, input);

        let result = validate_all(&ctx, module, &mut Default::default());
        assert!(!result.is_ok());
        assert_eq!(operation_error_messages(&result).len(), 1);
    }

    #[test]
    fn validate_op_schemas_skips_verify_hooks() {
        // The declared constraints hold, but the `#[verify]` hook rejects a
        // callee whose function type differs from `signature`.
        let input = r#"core.module @test {
  func.func @main(%f: func.func_sig<(core.i32) -> core.i32>, %x: core.i64) -> core.i64 {
    %r = func.call_indirect %f, %x {signature = func.func_sig<(core.i64) -> core.i64>} : core.i64
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(&mut ctx, input);

        let schemas = validate_op_schemas(&ctx, module.op());
        assert!(schemas.is_ok(), "{:?}", schemas.errors);

        let verifiers = validate_operation_verifiers(&ctx, module);
        assert!(
            operation_error_messages(&verifiers)
                .iter()
                .any(|message| message.contains("func.call_indirect")),
            "{:?}",
            verifiers.errors
        );
    }
    #[test]
    fn operation_verifiers_reject_a_symbol_reference_to_the_root_module() {
        let input = r#"core.module @test {
  func.func @main() -> core.i32 {
    %r = func.call {callee = @main} : core.i32
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(&mut ctx, input);
        assert!(validate_operation_verifiers(&ctx, module).is_ok());

        let attribute = Attribute::List(vec![Attribute::SymbolRef(SymbolPath::root())]);
        ctx.op_attrs_mut(module.op())
            .insert("referenced", attribute);
        let verifiers = validate_operation_verifiers(&ctx, module);
        assert_eq!(
            operation_error_messages(&verifiers)
                .iter()
                .filter(|message| message.contains("a symbol reference has no component"))
                .count(),
            1,
            "{:?}",
            verifiers.errors
        );
    }

    #[test]
    fn function_contracts_resolve_callees_in_the_nearest_module() {
        let errors = |callee: &str| {
            let mut ctx = IrContext::new();
            let module = crate::parser::parse_test_module(
                &mut ctx,
                &format!(
                    r#"core.module @test {{
  func.func @target(%value: core.i32) -> core.i32 {{
    func.return %value
  }}
  core.module @inner {{
    func.func @target(%value: core.i64) -> core.i64 {{
      func.return %value
    }}
    func.func @caller(%value: core.i32) -> core.i32 {{
      %result = func.call %value {{callee = {callee}}} : core.i32
      func.return %result
    }}
  }}
}}"#
                ),
            );
            validate_function_contracts(&ctx, module).errors
        };
        // The bare name names the sibling in `inner`, whose signature does not
        // match; the root definition is not visible from `inner`.
        assert!(!errors("@target").is_empty());
        // `@inner::@target` is looked up in `inner`, which has no `inner`, so
        // it names an undeclared target.
        assert!(errors("@inner::@target").is_empty());
    }
}
