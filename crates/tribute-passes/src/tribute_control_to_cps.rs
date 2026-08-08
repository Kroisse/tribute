//! Atomic legalization of verified `tribute_control` callable/control IR.
//!
//! The pass deliberately stops at the physical `func`/`closure` and logical
//! `ability` surface. It does not run closure extraction, evidence lowering,
//! or target-specific conversion.

use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt;

use tribute_core::{
    CALLING_CONVENTION_ATTR, CallableAbi, CallingConvention, cps_closure_function_type,
    cps_completion_type, cps_dispatch_type, cps_done_type, cps_parent_layout_type,
    cps_parent_ref_type, cps_resume_exact_type, cps_resume_type, physical_closure_type,
    set_calling_convention,
};
use tribute_ir::dialect::{ability, closure, effect, tribute_control, tribute_rt};
use trunk_ir::context::{BlockArgData, BlockData, IrContext, RegionData};
use trunk_ir::dialect::{adt, arith, core, func, scf};
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::pass::{Pass, PassRunResult};
use trunk_ir::refs::{BlockRef, OpRef, RegionRef, TypeRef, ValueRef};
use trunk_ir::rewrite::{ConversionMode, ConversionTarget, Module};
use trunk_ir::types::{Attribute, AttributeMap, Location, TypeDataBuilder};
use trunk_ir::{OperationDataBuilder, Symbol};

pub const PRE_CPS_BOUNDARY: &str = "tribute-control-pre-cps";
pub const POST_CPS_BOUNDARY: &str = "tribute-control-post-cps";

/// One source-located failure at a named callable/control boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoundaryFailure {
    pub op: Option<OpRef>,
    pub location: Option<Location>,
    pub message: String,
}

/// Failure returned without claiming that the requested named boundary holds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TributeControlToCpsError {
    pub boundary: &'static str,
    pub failures: Vec<BoundaryFailure>,
}

impl TributeControlToCpsError {
    fn one(
        boundary: &'static str,
        op: Option<OpRef>,
        location: Option<Location>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            boundary,
            failures: vec![BoundaryFailure {
                op,
                location,
                message: message.into(),
            }],
        }
    }
}

impl fmt::Display for TributeControlToCpsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "{} failed with {} error(s):",
            self.boundary,
            self.failures.len()
        )?;
        for failure in &self.failures {
            if let Some(op) = failure.op {
                write!(f, "  - {op}: ")?;
            } else {
                write!(f, "  - ")?;
            }
            writeln!(f, "{}", failure.message)?;
        }
        Ok(())
    }
}

impl Error for TributeControlToCpsError {}

/// Full frontend-side operation target for verified direct-style control IR.
pub fn tribute_control_pre_cps_target() -> ConversionTarget {
    ConversionTarget::new()
        .legal_dialect("core")
        .legal_dialect("tribute_control")
        .legal_dialect("scf")
        .legal_dialect("arith")
        .legal_dialect("adt")
        .legal_dialect("list")
        .legal_dialect("tribute_rt")
        .legal_dialect("tribute_io")
}

/// Partial post-legalization target. Unknown operations belong to later passes.
pub fn tribute_control_post_cps_target() -> ConversionTarget {
    ConversionTarget::new().illegal_dialect("tribute_control")
}

#[derive(Clone, Copy)]
enum TypeBoundary {
    Pre,
    Post,
}

fn type_is(ctx: &IrContext, ty: TypeRef, dialect: &str, name: &str) -> bool {
    let data = ctx.types.get(ty);
    data.dialect == Symbol::from_dynamic(dialect) && data.name == Symbol::from_dynamic(name)
}

fn walk_attribute_types(
    ctx: &IrContext,
    attribute: &Attribute,
    boundary: TypeBoundary,
    seen: &mut HashSet<TypeRef>,
    errors: &mut Vec<String>,
) {
    match attribute {
        Attribute::Type(ty) => walk_type(ctx, *ty, boundary, seen, errors),
        Attribute::List(values) => {
            for value in values {
                walk_attribute_types(ctx, value, boundary, seen, errors);
            }
        }
        _ => {}
    }
}

fn walk_type(
    ctx: &IrContext,
    ty: TypeRef,
    boundary: TypeBoundary,
    seen: &mut HashSet<TypeRef>,
    errors: &mut Vec<String>,
) {
    if !seen.insert(ty) {
        return;
    }
    let data = ctx.types.get(ty);
    let forbidden = match boundary {
        TypeBoundary::Pre => {
            type_is(ctx, ty, "core", "func")
                || type_is(ctx, ty, "closure", "closure")
                || data.dialect == Symbol::new("ability")
                || data.dialect == Symbol::new("effect")
        }
        TypeBoundary::Post => data.dialect == Symbol::new("tribute_control"),
    };
    if forbidden {
        errors.push(format!(
            "forbidden type {ty} is reachable at the named boundary"
        ));
    }
    for param in data.params.iter().copied() {
        walk_type(ctx, param, boundary, seen, errors);
    }
    for attribute in data.attrs.values() {
        walk_attribute_types(ctx, attribute, boundary, seen, errors);
    }
}

fn check_op_types(
    ctx: &IrContext,
    op: OpRef,
    boundary: TypeBoundary,
    failures: &mut Vec<BoundaryFailure>,
) {
    let mut errors = Vec::new();
    let mut seen = HashSet::new();
    for operand in ctx.op_operands(op) {
        walk_type(
            ctx,
            ctx.value_ty(*operand),
            boundary,
            &mut seen,
            &mut errors,
        );
    }
    for ty in ctx.op_result_types(op) {
        walk_type(ctx, *ty, boundary, &mut seen, &mut errors);
    }
    for attribute in ctx.op(op).attributes.values() {
        walk_attribute_types(ctx, attribute, boundary, &mut seen, &mut errors);
    }
    for region in ctx.op(op).regions.iter().copied() {
        for block in ctx.region(region).blocks.iter().copied() {
            for arg in ctx.block_args(block) {
                walk_type(ctx, ctx.value_ty(*arg), boundary, &mut seen, &mut errors);
            }
            for child in ctx.block(block).ops.iter().copied() {
                check_op_types(ctx, child, boundary, failures);
            }
        }
    }
    failures.extend(errors.into_iter().map(|message| BoundaryFailure {
        op: Some(op),
        location: Some(ctx.op(op).location),
        message,
    }));
}

fn verify_type_boundary(
    ctx: &IrContext,
    module: Module,
    boundary: TypeBoundary,
) -> Vec<BoundaryFailure> {
    let mut failures = Vec::new();
    check_op_types(ctx, module.op(), boundary, &mut failures);
    let mut seen = HashSet::new();
    let mut alias_errors = Vec::new();
    for (_, ty) in ctx.type_aliases() {
        walk_type(ctx, *ty, boundary, &mut seen, &mut alias_errors);
    }
    failures.extend(alias_errors.into_iter().map(|message| BoundaryFailure {
        op: Some(module.op()),
        location: Some(ctx.op(module.op()).location),
        message: format!("type alias contains {message}"),
    }));
    failures
}

fn verify_final_handle_dispatch_types(ctx: &IrContext, module: Module) -> Vec<BoundaryFailure> {
    fn check_dispatcher(
        ctx: &IrContext,
        owner: OpRef,
        value: ValueRef,
        evidence: TypeRef,
        general: bool,
        failures: &mut Vec<BoundaryFailure>,
    ) {
        let closure_ty = ctx.types.get(ctx.value_ty(value));
        let valid = if closure_ty.dialect == Symbol::new("closure")
            && closure_ty.name == Symbol::new("closure")
            && closure_ty.params.len() == 1
        {
            let func_ty = ctx.types.get(closure_ty.params[0]);
            let params = &func_ty.params;
            if general {
                func_ty.dialect == Symbol::new("core")
                    && func_ty.name == Symbol::new("func")
                    && params.len() == 5
                    && type_is(ctx, params[0], "core", "never")
                    && params[1] == evidence
                    && ctx.types.get(params[2]).dialect == Symbol::new("adt")
                    && ctx.types.get(params[2]).name == Symbol::new("typeref")
                    && type_is(ctx, params[3], "core", "i32")
                    && type_is(ctx, params[4], "tribute_rt", "anyref")
            } else {
                func_ty.dialect == Symbol::new("core")
                    && func_ty.name == Symbol::new("func")
                    && params.len() == 4
                    && type_is(ctx, params[0], "tribute_rt", "anyref")
                    && params[1] == evidence
                    && type_is(ctx, params[2], "core", "i32")
                    && type_is(ctx, params[3], "tribute_rt", "anyref")
            }
        } else {
            false
        };
        if !valid {
            failures.push(BoundaryFailure {
                op: Some(owner),
                location: Some(ctx.op(owner).location),
                message: format!(
                    "{} dispatcher has the wrong typed closure ABI",
                    if general {
                        "general"
                    } else {
                        "tail-resumptive"
                    }
                ),
            });
        }
        let expected = if general {
            CallingConvention::Cps as i64
        } else {
            CallingConvention::EvidenceDirect as i64
        };
        let trunk_ir::ValueDef::OpResult(def, _) = ctx.value_def(value) else {
            failures.push(BoundaryFailure {
                op: Some(owner),
                location: Some(ctx.op(owner).location),
                message: "dispatcher must be a physical closure result with exact calling convention metadata"
                    .into(),
            });
            return;
        };
        if ctx.op(def).attributes.get_i64(CALLING_CONVENTION_ATTR) != Ok(Some(expected)) {
            failures.push(BoundaryFailure {
                op: Some(def),
                location: Some(ctx.op(def).location),
                message: format!("dispatcher must preserve calling convention metadata {expected}"),
            });
        }
    }

    fn visit(ctx: &IrContext, op: OpRef, failures: &mut Vec<BoundaryFailure>) {
        if ability::HandleDispatch::matches(ctx, op) {
            let operands = ctx.op_operands(op);
            if let Some((evidence, operands)) = operands.split_first()
                && let Some((_, dispatchers)) = operands.split_first()
            {
                let evidence_ty = ctx.value_ty(*evidence);
                for pair in dispatchers.chunks_exact(2) {
                    check_dispatcher(ctx, op, pair[0], evidence_ty, false, failures);
                    check_dispatcher(ctx, op, pair[1], evidence_ty, true, failures);
                }
            }
        }
        for region in ctx.op(op).regions.iter().copied() {
            for block in ctx.region(region).blocks.iter().copied() {
                for child in ctx.block(block).ops.iter().copied() {
                    visit(ctx, child, failures);
                }
            }
        }
    }

    let mut failures = Vec::new();
    visit(ctx, module.op(), &mut failures);
    failures
}

fn verify_physical_callable_graph(ctx: &IrContext, module: Module) -> Vec<BoundaryFailure> {
    fn collect(
        ctx: &IrContext,
        op: OpRef,
        signatures: &mut HashMap<Symbol, (TypeRef, Option<i64>)>,
    ) {
        let data = ctx.op(op);
        if data.dialect == Symbol::new("core") && data.name == Symbol::new("module") {
            return;
        }
        if func::Func::matches(ctx, op)
            && let (Some(symbol), Some(ty)) = (
                data.attributes.get_symbol("sym_name"),
                data.attributes.get_type("type"),
            )
        {
            let convention = data
                .attributes
                .get_i64(CALLING_CONVENTION_ATTR)
                .ok()
                .flatten();
            signatures.insert(symbol, (ty, convention));
        }
        for region in ctx.op(op).regions.iter().copied() {
            for block in ctx.region(region).blocks.iter().copied() {
                for child in ctx.block(block).ops.iter().copied() {
                    collect(ctx, child, signatures);
                }
            }
        }
    }

    fn visit(
        ctx: &IrContext,
        op: OpRef,
        signatures: &HashMap<Symbol, (TypeRef, Option<i64>)>,
        failures: &mut Vec<BoundaryFailure>,
    ) {
        let data = ctx.op(op);
        if data.dialect == Symbol::new("core") && data.name == Symbol::new("module") {
            verify_module(ctx, op, failures);
            return;
        }
        let op_convention = data
            .attributes
            .get_i64(CALLING_CONVENTION_ATTR)
            .ok()
            .flatten();
        let requires_convention = (data.dialect == Symbol::new("func")
            && matches!(
                data.name.with_str(|name| name.to_owned()).as_str(),
                "func" | "call" | "call_indirect" | "tail_call" | "tail_call_indirect"
            ))
            || (data.dialect == Symbol::new("closure")
                && matches!(
                    data.name.with_str(|name| name.to_owned()).as_str(),
                    "lambda" | "new"
                ));
        if requires_convention && !matches!(op_convention, Some(0..=2)) {
            failures.push(BoundaryFailure {
                op: Some(op),
                location: Some(data.location),
                message: format!(
                    "{}.{} must carry exact Direct, EvidenceDirect, or Cps metadata",
                    data.dialect, data.name
                ),
            });
        }
        if data.dialect == Symbol::new("func") && data.name == Symbol::new("tail_call") {
            let Some(callee) = data.attributes.get_symbol("callee") else {
                failures.push(BoundaryFailure {
                    op: Some(op),
                    location: Some(data.location),
                    message: "func.tail_call requires a resolved callee symbol".into(),
                });
                return;
            };
            let Some((func_ty, convention)) = signatures.get(&callee) else {
                failures.push(BoundaryFailure {
                    op: Some(op),
                    location: Some(data.location),
                    message: format!("func.tail_call references unresolved callee @{callee}"),
                });
                return;
            };
            let ty = ctx.types.get(*func_ty);
            let valid_never = ty.dialect == Symbol::new("core")
                && ty.name == Symbol::new("func")
                && ty
                    .params
                    .first()
                    .is_some_and(|result| type_is(ctx, *result, "core", "never"));
            if !valid_never {
                failures.push(BoundaryFailure {
                    op: Some(op),
                    location: Some(data.location),
                    message: "func.tail_call target must have core.never result".into(),
                });
            }
            let expected = ty.params.get(1..).unwrap_or_default();
            let operands = ctx.op_operands(op);
            if operands.len() != expected.len()
                || operands
                    .iter()
                    .zip(expected)
                    .any(|(value, ty)| ctx.value_ty(*value) != *ty)
            {
                failures.push(BoundaryFailure {
                    op: Some(op),
                    location: Some(data.location),
                    message: "func.tail_call operands do not match the target signature".into(),
                });
            }
            if *convention != Some(CallingConvention::Cps as i64)
                || op_convention != Some(CallingConvention::Cps as i64)
            {
                failures.push(BoundaryFailure {
                    op: Some(op),
                    location: Some(data.location),
                    message: "func.tail_call must preserve exact Cps metadata".into(),
                });
            }
        } else if data.dialect == Symbol::new("func")
            && data.name == Symbol::new("tail_call_indirect")
            && op_convention != Some(CallingConvention::Cps as i64)
        {
            failures.push(BoundaryFailure {
                op: Some(op),
                location: Some(data.location),
                message: "func.tail_call_indirect must carry exact Cps metadata".into(),
            });
        } else if data.dialect == Symbol::new("func") && data.name == Symbol::new("call") {
            let callee = data.attributes.get_symbol("callee");
            if let Some(callee) = callee {
                match signatures.get(&callee) {
                    Some((_, target_convention)) if *target_convention == op_convention => {}
                    Some(_) => failures.push(BoundaryFailure {
                        op: Some(op),
                        location: Some(data.location),
                        message: "func.call metadata does not match its target".into(),
                    }),
                    None => failures.push(BoundaryFailure {
                        op: Some(op),
                        location: Some(data.location),
                        message: format!("func.call references unresolved callee @{callee}"),
                    }),
                }
            }
        } else if data.dialect == Symbol::new("func")
            && data.name == Symbol::new("call_indirect")
            && op_convention == Some(CallingConvention::Cps as i64)
        {
            failures.push(BoundaryFailure {
                op: Some(op),
                location: Some(data.location),
                message: "dynamic Cps transfers must use func.tail_call_indirect".into(),
            });
        }
        for region in data.regions.iter().copied() {
            for block in ctx.region(region).blocks.iter().copied() {
                for child in ctx.block(block).ops.iter().copied() {
                    visit(ctx, child, signatures, failures);
                }
            }
        }
    }

    fn verify_module(ctx: &IrContext, module_op: OpRef, failures: &mut Vec<BoundaryFailure>) {
        let mut signatures = HashMap::new();
        let regions = ctx.op(module_op).regions.to_vec();
        for region in regions.iter().copied() {
            for block in ctx.region(region).blocks.iter().copied() {
                for op in ctx.block(block).ops.iter().copied() {
                    collect(ctx, op, &mut signatures);
                }
            }
        }
        for region in regions {
            for block in ctx.region(region).blocks.iter().copied() {
                for op in ctx.block(block).ops.iter().copied() {
                    visit(ctx, op, &signatures, failures);
                }
            }
        }
    }

    let mut failures = Vec::new();
    verify_module(ctx, module.op(), &mut failures);
    failures
}

fn verify_source_conversion_shapes(ctx: &IrContext, module: Module) -> Vec<BoundaryFailure> {
    fn failure(ctx: &IrContext, op: OpRef, message: impl Into<String>) -> BoundaryFailure {
        BoundaryFailure {
            op: Some(op),
            location: Some(ctx.op(op).location),
            message: message.into(),
        }
    }

    fn visit(ctx: &IrContext, op: OpRef, failures: &mut Vec<BoundaryFailure>) {
        let data = ctx.op(op);
        if !data.successors.is_empty() {
            failures.push(failure(
                ctx,
                op,
                "tribute-control-pre-cps is structured and forbids block successors",
            ));
        }
        if data.dialect == Symbol::new("scf") && data.name == Symbol::new("switch") {
            if ctx.op_operands(op).len() != 1
                || !ctx.op_result_types(op).is_empty()
                || data.regions.len() != 1
            {
                failures.push(failure(
                    ctx,
                    op,
                    "scf.switch requires one discriminant, no results, and one body region",
                ));
            } else {
                let blocks = &ctx.region(data.regions[0]).blocks;
                if let [body] = blocks.as_slice() {
                    for arm in ctx.block(*body).ops.iter().copied() {
                        let arm_data = ctx.op(arm);
                        let is_case = arm_data.dialect == Symbol::new("scf")
                            && arm_data.name == Symbol::new("case");
                        let is_default = arm_data.dialect == Symbol::new("scf")
                            && arm_data.name == Symbol::new("default");
                        if !is_case && !is_default {
                            failures.push(failure(
                                ctx,
                                arm,
                                "scf.switch body may contain only scf.case and scf.default",
                            ));
                            continue;
                        }
                        if is_case && !arm_data.attributes.contains_key("value") {
                            failures.push(failure(ctx, arm, "scf.case requires a value attribute"));
                        }
                        if let [region] = arm_data.regions.as_slice() {
                            if ctx.region(*region).blocks.len() != 1 {
                                failures.push(failure(
                                    ctx,
                                    arm,
                                    "scf switch arm region requires exactly one block",
                                ));
                            }
                        } else {
                            failures.push(failure(
                                ctx,
                                arm,
                                "scf switch arm requires exactly one region",
                            ));
                        }
                    }
                } else {
                    failures.push(failure(
                        ctx,
                        op,
                        "scf.switch body region requires exactly one block",
                    ));
                }
            }
        }
        for region in data.regions.iter().copied() {
            for block in ctx.region(region).blocks.iter().copied() {
                for child in ctx.block(block).ops.iter().copied() {
                    visit(ctx, child, failures);
                }
            }
        }
    }

    let mut failures = Vec::new();
    visit(ctx, module.op(), &mut failures);
    failures
}

/// Verify the complete named pre-CPS boundary.
pub fn verify_tribute_control_pre_cps(
    ctx: &IrContext,
    module: Module,
    declarations: &[tribute_control::OperationDeclaration],
) -> Result<(), TributeControlToCpsError> {
    let mut failures = Vec::new();
    let validation = tribute_control::validate(ctx, module, declarations);
    failures.extend(validation.errors.into_iter().map(|error| BoundaryFailure {
        op: error.op,
        location: error.location,
        message: error.message,
    }));

    if let Some(body) = module.body(ctx) {
        failures.extend(
            tribute_control_pre_cps_target()
                .verify_mode(ctx, body, ConversionMode::Full)
                .into_iter()
                .map(|illegal| BoundaryFailure {
                    op: Some(illegal.op),
                    location: Some(ctx.op(illegal.op).location),
                    message: format!(
                        "{} operation {}.{} is not legal",
                        match illegal.legality {
                            trunk_ir::rewrite::LegalityCheck::Illegal => "illegal",
                            trunk_ir::rewrite::LegalityCheck::Unknown => "unknown",
                            trunk_ir::rewrite::LegalityCheck::Legal => "unexpected legal",
                        },
                        ctx.op(illegal.op).dialect,
                        ctx.op(illegal.op).name
                    ),
                }),
        );
    }
    failures.extend(verify_type_boundary(ctx, module, TypeBoundary::Pre));
    failures.extend(verify_source_conversion_shapes(ctx, module));
    failures.extend(
        trunk_ir::validation::validate_all(ctx, module)
            .errors
            .into_iter()
            .map(|error| BoundaryFailure {
                op: None,
                location: None,
                message: error.to_string(),
            }),
    );

    if failures.is_empty() {
        Ok(())
    } else {
        Err(TributeControlToCpsError {
            boundary: PRE_CPS_BOUNDARY,
            failures,
        })
    }
}

/// Verify the complete named post-CPS boundary.
pub fn verify_tribute_control_post_cps(
    ctx: &IrContext,
    module: Module,
) -> Result<(), TributeControlToCpsError> {
    let mut failures = Vec::new();
    if let Some(body) = module.body(ctx) {
        failures.extend(
            tribute_control_post_cps_target()
                .verify_mode(ctx, body, ConversionMode::Partial)
                .into_iter()
                .map(|illegal| BoundaryFailure {
                    op: Some(illegal.op),
                    location: Some(ctx.op(illegal.op).location),
                    message: format!(
                        "residual {}.{} operation",
                        ctx.op(illegal.op).dialect,
                        ctx.op(illegal.op).name
                    ),
                }),
        );
    }
    failures.extend(verify_type_boundary(ctx, module, TypeBoundary::Post));
    if let Err(error) = crate::resolve_evidence::validate_final_handle_dispatches(ctx, module) {
        let op = error.op();
        failures.push(BoundaryFailure {
            op: Some(op),
            location: Some(ctx.op(op).location),
            message: error.to_string(),
        });
    }
    failures.extend(verify_final_handle_dispatch_types(ctx, module));
    failures.extend(verify_physical_callable_graph(ctx, module));
    failures.extend(
        trunk_ir::validation::validate_all(ctx, module)
            .errors
            .into_iter()
            .map(|error| BoundaryFailure {
                op: None,
                location: None,
                message: error.to_string(),
            }),
    );
    if failures.is_empty() {
        Ok(())
    } else {
        Err(TributeControlToCpsError {
            boundary: POST_CPS_BOUNDARY,
            failures,
        })
    }
}

fn convert_convention(convention: tribute_control::CallingConvention) -> CallingConvention {
    match convention {
        tribute_control::CallingConvention::Direct => CallingConvention::Direct,
        tribute_control::CallingConvention::EvidenceDirect => CallingConvention::EvidenceDirect,
        tribute_control::CallingConvention::Cps => CallingConvention::Cps,
    }
}

#[derive(Clone)]
struct CallableInfo {
    symbol: Symbol,
    convention: CallingConvention,
    source_result: TypeRef,
    source_params: Vec<TypeRef>,
}

#[derive(Clone, Copy)]
struct ParentTypes {
    reference: TypeRef,
    layout: TypeRef,
    done: TypeRef,
    dispatch: TypeRef,
}

#[derive(Clone)]
struct HandlerArmInfo {
    op: OpRef,
    value: ValueRef,
    ability_ref: TypeRef,
    op_name: Symbol,
    kind: Symbol,
    operation_result: TypeRef,
    params: Vec<TypeRef>,
    has_resume_token: bool,
    parent_type: Option<TypeRef>,
}

/// Immutable recipe for rebuilding one exact handler token. Grouping these
/// values makes it explicit that a token receives the dynamic Parent only at
/// invocation; the factory metadata itself is never overwritten by re-entry.
#[derive(Clone)]
struct HandlerTokenFactory {
    resume_body: ValueRef,
    completion: ValueRef,
    input_type: TypeRef,
    body_type: TypeRef,
    answer_type: TypeRef,
    dispatch_factory: Symbol,
    dispatch_factory_args: Vec<ValueRef>,
}

/// The exact dispatcher to install in a resumed `Parent<X>`.
enum ResumeDispatchRebuild {
    Adapter { factory: Symbol },
    LocalFactory(LocalDispatchFactory),
}

/// Immutable inputs for rebuilding the lexical dispatcher after a foreign
/// resumption. The dynamic Parent is supplied by `build_rebound_resume`; this
/// recipe contains only the lexical factory and captures it needs to rebuild.
struct LocalDispatchFactory {
    factory: Symbol,
    args: Vec<ValueRef>,
}

struct Converter<'a> {
    ctx: &'a mut IrContext,
    module_block: BlockRef,
    funcs_by_module: HashMap<OpRef, HashMap<Symbol, CallableInfo>>,
    current_module: OpRef,
    converted_types: HashMap<TypeRef, TypeRef>,
    parents: HashMap<TypeRef, ParentTypes>,
    parent_layout_aliases: Vec<(Symbol, TypeRef)>,
    helper_index: u32,
}

#[derive(Clone)]
struct Flow {
    convention: CallingConvention,
    /// Evidence used to evaluate the current lexical computation.
    evidence: Option<ValueRef>,
    /// Evidence carried by an explicit callback context. A resumed suffix uses
    /// this dynamic value when it performs again, while a handler arm may keep
    /// evaluating under its outer lexical evidence.
    resume_evidence: Option<ValueRef>,
    done: Option<ValueRef>,
    dispatch: Option<ValueRef>,
    boundary_result: TypeRef,
    /// A handler arm routes resumed suffixes through this completion using the
    /// dynamic outer `Done` supplied to the suffix frame.
    resume_after: Option<ValueRef>,
    /// The enclosing `Done<O>` used to invoke a resumed arm suffix. This is
    /// distinct from the arm's normal `Done<A>` exit.
    resume_done: Option<ValueRef>,
    /// Resultless structured control uses this exact zero-argument suffix.
    void_done: Option<ValueRef>,
    preserve_scf_yield: bool,
}

impl<'a> Converter<'a> {
    fn new(
        ctx: &'a mut IrContext,
        module_block: BlockRef,
        funcs_by_module: HashMap<OpRef, HashMap<Symbol, CallableInfo>>,
        current_module: OpRef,
    ) -> Self {
        Self {
            ctx,
            module_block,
            funcs_by_module,
            current_module,
            converted_types: HashMap::new(),
            parents: HashMap::new(),
            parent_layout_aliases: Vec::new(),
            helper_index: 0,
        }
    }

    fn current_func(&self, symbol: Symbol) -> Option<CallableInfo> {
        self.funcs_by_module
            .get(&self.current_module)
            .and_then(|funcs| funcs.get(&symbol))
            .cloned()
    }

    fn current_direct_call_target(&self, _source: OpRef, symbol: Symbol) -> Option<CallableInfo> {
        self.current_func(symbol)
    }

    fn malformed_source(
        &self,
        source: OpRef,
        message: impl Into<String>,
    ) -> TributeControlToCpsError {
        TributeControlToCpsError::one(
            PRE_CPS_BOUNDARY,
            Some(source),
            Some(self.ctx.op(source).location),
            message,
        )
    }

    fn never_type(&mut self) -> TypeRef {
        core::never(self.ctx).as_type_ref()
    }

    fn evidence_type(&mut self) -> TypeRef {
        ability::evidence_adt_type_ref(self.ctx)
    }

    fn anyref_type(&mut self) -> TypeRef {
        tribute_rt::anyref(self.ctx).as_type_ref()
    }

    fn i32_type(&mut self) -> TypeRef {
        self.ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build())
    }

    fn done_k_type(&mut self, result: TypeRef) -> TypeRef {
        cps_done_type(self.ctx, result)
    }

    fn completion_type(&mut self, value: TypeRef, boundary: TypeRef) -> TypeRef {
        let evidence = self.evidence_type();
        let parent = self.parent_types(boundary).reference;
        cps_completion_type(self.ctx, evidence, value, parent)
    }

    fn resumption_type(&mut self, input: TypeRef, boundary: TypeRef) -> TypeRef {
        let evidence = self.evidence_type();
        let parent = self.parent_types(boundary).reference;
        cps_resume_exact_type(self.ctx, evidence, input, parent)
    }

    fn erased_resumption_type(&mut self, boundary: TypeRef) -> TypeRef {
        let evidence = self.evidence_type();
        let anyref = self.anyref_type();
        let parent = self.parent_types(boundary).reference;
        cps_resume_type(self.ctx, evidence, parent, anyref)
    }

    fn dispatch_type(&mut self, boundary: TypeRef) -> TypeRef {
        self.parent_types(boundary).dispatch
    }

    fn parent_types(&mut self, result: TypeRef) -> ParentTypes {
        if let Some(parent) = self.parents.get(&result).copied() {
            return parent;
        }
        let name = Symbol::from_dynamic(&format!("__tribute_parent_{result:?}"));
        let reference = cps_parent_ref_type(self.ctx, name, result);
        let done = self.done_k_type(result);
        let evidence = self.evidence_type();
        let anyref = self.anyref_type();
        let i32 = self.i32_type();
        let dispatch = cps_dispatch_type(self.ctx, evidence, reference, anyref, i32);
        let layout = cps_parent_layout_type(self.ctx, name, result, done, dispatch);
        self.parent_layout_aliases.push((name, layout));
        let parent = ParentTypes {
            reference,
            layout,
            done,
            dispatch,
        };
        self.parents.insert(result, parent);
        parent
    }

    fn pack_parent(
        &mut self,
        block: BlockRef,
        location: Location,
        result: TypeRef,
        done: ValueRef,
        dispatch: ValueRef,
    ) -> ValueRef {
        let parent = self.parent_types(result);
        assert_eq!(
            self.ctx.value_ty(done),
            parent.done,
            "Parent done type mismatch"
        );
        assert_eq!(
            self.ctx.value_ty(dispatch),
            parent.dispatch,
            "Parent dispatch type mismatch"
        );
        let pack = adt::struct_new(
            self.ctx,
            location,
            [done, dispatch],
            parent.reference,
            parent.layout,
        );
        self.ctx.push_op(block, pack.op_ref());
        pack.result(self.ctx)
    }

    fn unpack_parent(
        &mut self,
        block: BlockRef,
        location: Location,
        result: TypeRef,
        parent_value: ValueRef,
    ) -> (ValueRef, ValueRef) {
        let parent = self.parent_types(result);
        assert_eq!(
            self.ctx.value_ty(parent_value),
            parent.reference,
            "Parent provenance mismatch"
        );
        let done = adt::struct_get(
            self.ctx,
            location,
            parent_value,
            parent.done,
            parent.layout,
            0,
        );
        self.ctx.push_op(block, done.op_ref());
        let dispatch = adt::struct_get(
            self.ctx,
            location,
            parent_value,
            parent.dispatch,
            parent.layout,
            1,
        );
        self.ctx.push_op(block, dispatch.op_ref());
        (done.result(self.ctx), dispatch.result(self.ctx))
    }

    fn convert_attribute(&mut self, attribute: &Attribute) -> Attribute {
        match attribute {
            Attribute::Type(ty) => Attribute::Type(self.convert_type(*ty)),
            Attribute::List(values) => Attribute::List(
                values
                    .iter()
                    .map(|value| self.convert_attribute(value))
                    .collect(),
            ),
            value => value.clone(),
        }
    }

    fn convert_attrs(&mut self, attrs: &AttributeMap) -> Vec<(Symbol, Attribute)> {
        attrs
            .iter()
            .map(|(key, value)| (*key, self.convert_attribute(value)))
            .collect()
    }

    fn convert_type(&mut self, ty: TypeRef) -> TypeRef {
        if let Some(converted) = self.converted_types.get(&ty) {
            return *converted;
        }
        if let Some(callable) = tribute_control::Callable::from_type_ref(self.ctx, ty) {
            let convention = convert_convention(
                tribute_control::callable_convention(self.ctx, ty)
                    .expect("pre-CPS validation checked callable convention"),
            );
            let result = self.convert_type(callable.result(self.ctx));
            let source_params = callable.params(self.ctx).to_vec();
            let params: Vec<_> = source_params
                .into_iter()
                .map(|param| self.convert_type(param))
                .collect();
            let abi = CallableAbi::new(convention, params, result);
            let evidence = self.evidence_type();
            let done_k = self.done_k_type(result);
            let dispatch = self.dispatch_type(result);
            let lowered_params = abi.lowered_params_with_dispatch(evidence, done_k, dispatch);
            let lowered_result = if convention == CallingConvention::Cps {
                self.never_type()
            } else {
                result
            };
            let function = core::func(self.ctx, lowered_result, lowered_params).as_type_ref();
            let converted = physical_closure_type(self.ctx, function, convention);
            self.converted_types.insert(ty, converted);
            return converted;
        }
        let data = self.ctx.types.get(ty).clone();
        if data.dialect == Symbol::new("tribute_control")
            && data.name == Symbol::new("resume_token")
            && data.params.len() == 2
        {
            let input = self.convert_type(data.params[0]);
            let answer = self.convert_type(data.params[1]);
            let converted = self.resumption_type(input, answer);
            self.converted_types.insert(ty, converted);
            return converted;
        }

        let params: Vec<_> = data
            .params
            .iter()
            .copied()
            .map(|param| self.convert_type(param))
            .collect();
        let attrs: Vec<_> = data
            .attrs
            .iter()
            .map(|(key, value)| (*key, self.convert_attribute(value)))
            .collect();
        if params == data.params.as_slice()
            && attrs
                .iter()
                .all(|(key, value)| data.attrs.get(key) == Some(value))
        {
            self.converted_types.insert(ty, ty);
            return ty;
        }
        let mut builder = TypeDataBuilder::new(data.dialect, data.name).params(params);
        for (key, value) in attrs {
            builder = builder.attr(key, value);
        }
        let converted = self.ctx.types.intern(builder.build());
        self.converted_types.insert(ty, converted);
        converted
    }

    fn physical_function_type(&mut self, logical: TypeRef) -> TypeRef {
        let callable = tribute_control::Callable::from_type_ref(self.ctx, logical)
            .expect("pre-CPS validation checked function type");
        let convention = convert_convention(
            tribute_control::callable_convention(self.ctx, logical)
                .expect("pre-CPS validation checked function convention"),
        );
        let result = self.convert_type(callable.result(self.ctx));
        let source_params = callable.params(self.ctx).to_vec();
        let params: Vec<_> = source_params
            .into_iter()
            .map(|param| self.convert_type(param))
            .collect();
        let evidence = self.evidence_type();
        let done_k = self.done_k_type(result);
        let dispatch = self.dispatch_type(result);
        let abi = CallableAbi::new(convention, params, result);
        let params = abi.lowered_params_with_dispatch(evidence, done_k, dispatch);
        let result = if convention == CallingConvention::Cps {
            self.never_type()
        } else {
            result
        };
        core::func(self.ctx, result, params).as_type_ref()
    }

    fn copy_extra_attrs(&mut self, source: OpRef, target: OpRef, excluded: &[&str]) {
        let excluded: HashSet<Symbol> = excluded
            .iter()
            .map(|name| Symbol::from_dynamic(name))
            .collect();
        let attrs = self.convert_attrs(&self.ctx.op(source).attributes.clone());
        for (key, value) in attrs {
            if !excluded.contains(&key) {
                self.ctx.op_mut(target).attributes.insert(key, value);
            }
        }
    }

    fn fresh_helper(&mut self, prefix: &str) -> Symbol {
        let index = self.helper_index;
        self.helper_index += 1;
        Symbol::from_dynamic(&format!("__tribute_{prefix}_{index}"))
    }

    fn make_block(&mut self, location: Location, types: &[TypeRef]) -> BlockRef {
        self.ctx.create_block(BlockData {
            location,
            args: types
                .iter()
                .copied()
                .map(|ty| BlockArgData {
                    ty,
                    attrs: AttributeMap::new(),
                })
                .collect(),
            ops: Default::default(),
            parent_region: None,
        })
    }

    fn single_block_region(&mut self, location: Location, block: BlockRef) -> RegionRef {
        self.ctx.create_region(RegionData {
            location,
            blocks: trunk_ir::smallvec::smallvec![block],
            parent_op: None,
        })
    }

    fn clone_plain_op(
        &mut self,
        source: OpRef,
        mapping: &mut HashMap<ValueRef, ValueRef>,
    ) -> Result<OpRef, TributeControlToCpsError> {
        let data = self.ctx.op(source);
        if data.dialect == Symbol::new("tribute_control") {
            return Err(TributeControlToCpsError::one(
                POST_CPS_BOUNDARY,
                Some(source),
                Some(data.location),
                format!(
                    "unsupported tribute_control operation {} reached plain cloning",
                    data.name
                ),
            ));
        }
        let location = data.location;
        let dialect = data.dialect;
        let name = data.name;
        let operands: Vec<_> = self
            .ctx
            .op_operands(source)
            .iter()
            .map(|value| mapping.get(value).copied().unwrap_or(*value))
            .collect();
        let result_types: Vec<_> = self.ctx.op_result_types(source).to_vec();
        let attrs = data.attributes.clone();
        let regions = data.regions.to_vec();
        let successors = data.successors.to_vec();
        if !successors.is_empty() {
            return Err(self.malformed_source(
                source,
                "plain operation cloning does not support block successors",
            ));
        }

        let mut builder = OperationDataBuilder::new(location, dialect, name);
        for operand in operands {
            builder = builder.operand(operand);
        }
        for ty in result_types {
            let converted = self.convert_type(ty);
            builder = builder.result(converted);
        }
        for (key, value) in self.convert_attrs(&attrs) {
            builder = builder.attr(key, value);
        }
        for region in regions {
            let converted = if dialect == Symbol::new("core") && name == Symbol::new("module") {
                let previous_module = self.current_module;
                self.current_module = source;
                let converted = self.clone_module_region(region);
                self.current_module = previous_module;
                converted?
            } else {
                self.clone_plain_region(region, mapping)?
            };
            builder = builder.region(converted);
        }
        let data = builder.build(self.ctx);
        let cloned = self.ctx.create_op(data);
        for (old, new) in self
            .ctx
            .op_results(source)
            .to_vec()
            .into_iter()
            .zip(self.ctx.op_results(cloned).to_vec())
        {
            mapping.insert(old, new);
        }
        Ok(cloned)
    }

    fn clone_module_region(
        &mut self,
        source: RegionRef,
    ) -> Result<RegionRef, TributeControlToCpsError> {
        let location = self.ctx.region(source).location;
        let source_blocks = self.ctx.region(source).blocks.to_vec();
        let mut blocks = Vec::with_capacity(source_blocks.len());
        for source_block in source_blocks {
            let logical_arg_types = self
                .ctx
                .block_args(source_block)
                .iter()
                .map(|arg| self.ctx.value_ty(*arg))
                .collect::<Vec<_>>();
            let source_arg_types = logical_arg_types
                .into_iter()
                .map(|ty| self.convert_type(ty))
                .collect::<Vec<_>>();
            let block = self.make_block(self.ctx.block(source_block).location, &source_arg_types);
            let previous_module_block = self.module_block;
            self.module_block = block;
            let conversion = (|| {
                let mut mapping = HashMap::new();
                for (old, new) in self
                    .ctx
                    .block_args(source_block)
                    .iter()
                    .copied()
                    .zip(self.ctx.block_args(block).iter().copied())
                {
                    mapping.insert(old, new);
                }
                let source_ops = self.ctx.block(source_block).ops.to_vec();
                for source_op in source_ops {
                    let converted = if tribute_control::Func::matches(self.ctx, source_op) {
                        self.convert_func(source_op)?
                    } else if self.ctx.op(source_op).dialect == Symbol::new("tribute_control") {
                        return Err(TributeControlToCpsError::one(
                            PRE_CPS_BOUNDARY,
                            Some(source_op),
                            Some(self.ctx.op(source_op).location),
                            "only tribute_control.func may appear directly in a module block",
                        ));
                    } else {
                        self.clone_plain_op(source_op, &mut mapping)?
                    };
                    self.ctx.push_op(block, converted);
                }
                Ok(())
            })();
            self.module_block = previous_module_block;
            conversion?;
            blocks.push(block);
        }
        Ok(self.ctx.create_region(RegionData {
            location,
            blocks: blocks.into(),
            parent_op: None,
        }))
    }

    fn clone_plain_region(
        &mut self,
        source: RegionRef,
        mapping: &mut HashMap<ValueRef, ValueRef>,
    ) -> Result<RegionRef, TributeControlToCpsError> {
        let location = self.ctx.region(source).location;
        let source_blocks = self.ctx.region(source).blocks.to_vec();
        let mut blocks = Vec::with_capacity(source_blocks.len());
        for source_block in source_blocks {
            let source_arg_types: Vec<_> = self
                .ctx
                .block_args(source_block)
                .iter()
                .map(|arg| self.ctx.value_ty(*arg))
                .collect();
            let arg_types: Vec<_> = source_arg_types
                .into_iter()
                .map(|ty| self.convert_type(ty))
                .collect();
            let block = self.make_block(self.ctx.block(source_block).location, &arg_types);
            for (old, new) in self
                .ctx
                .block_args(source_block)
                .to_vec()
                .into_iter()
                .zip(self.ctx.block_args(block).to_vec())
            {
                mapping.insert(old, new);
            }
            let source_ops = self.ctx.block(source_block).ops.to_vec();
            for op in source_ops {
                let cloned = self.clone_plain_op(op, mapping)?;
                self.ctx.push_op(block, cloned);
            }
            blocks.push(block);
        }
        Ok(self.ctx.create_region(RegionData {
            location,
            blocks: blocks.into(),
            parent_op: None,
        }))
    }

    fn contains_tribute_control(&self, op: OpRef) -> bool {
        self.ctx.op(op).regions.iter().copied().any(|region| {
            self.ctx.region(region).blocks.iter().copied().any(|block| {
                self.ctx.block(block).ops.iter().copied().any(|child| {
                    self.ctx.op(child).dialect == Symbol::new("tribute_control")
                        || self.contains_tribute_control(child)
                })
            })
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_structured_if(
        &mut self,
        source: OpRef,
        source_ops: &[OpRef],
        index: usize,
        block: BlockRef,
        mapping: &mut HashMap<ValueRef, ValueRef>,
        flow: &Flow,
    ) -> Result<(), TributeControlToCpsError> {
        let result_types = self.ctx.op_result_types(source).to_vec();
        if result_types.len() > 1 {
            return Err(TributeControlToCpsError::one(
                POST_CPS_BOUNDARY,
                Some(source),
                Some(self.ctx.op(source).location),
                "effectful scf.if requires zero or one result",
            ));
        }
        if flow.convention != CallingConvention::Cps {
            return self
                .lower_direct_structured_if(source, source_ops, index, block, mapping, flow);
        }
        let location = self.ctx.op(source).location;
        // `core.never` is the structured proper-tail marker, not an SSA value
        // boundary. In particular, an `op -> Never` branch keeps the active
        // `Flow<R>` so its rejecting `Resume<R>` matches `Dispatch<R>`.
        let value_result_type = result_types
            .first()
            .copied()
            .filter(|ty| !type_is(self.ctx, *ty, "core", "never"));
        let is_tail_never_marker = value_result_type.is_none() && result_types.len() == 1;
        let (branch_done, branch_dispatch, branch_void_done) = if is_tail_never_marker {
            (flow.done, flow.dispatch, flow.void_done)
        } else {
            let continuation = if let Some(source_result_type) = value_result_type {
                self.build_suffix_continuation(
                    source_ops,
                    index + 1,
                    self.ctx.op_result(source, 0),
                    source_result_type,
                    mapping,
                    flow,
                    location,
                )?
            } else {
                self.build_void_suffix_continuation(source_ops, index + 1, mapping, flow, location)?
            };
            let continuation_op = match self.ctx.value_def(continuation) {
                trunk_ir::ValueDef::OpResult(op, _) => op,
                _ => unreachable!("structured continuation is a closure.lambda"),
            };
            self.ctx.push_op(block, continuation_op);
            if let Some(source_result_type) = value_result_type {
                let value_type = self.convert_type(source_result_type);
                let (ops, done, dispatch) =
                    self.build_cps_call_adapters(value_type, continuation, flow, location)?;
                for op in ops {
                    self.ctx.push_op(block, op);
                }
                (Some(done), Some(dispatch), flow.void_done)
            } else {
                (flow.done, flow.dispatch, Some(continuation))
            }
        };

        let mut converted_regions = Vec::new();
        let source_regions = self.ctx.op(source).regions.to_vec();
        for source_region in source_regions {
            let source_blocks = self.ctx.region(source_region).blocks.to_vec();
            let [source_block] = source_blocks.as_slice() else {
                return Err(TributeControlToCpsError::one(
                    POST_CPS_BOUNDARY,
                    Some(source),
                    Some(location),
                    "effectful scf.if regions must contain one block",
                ));
            };
            let source_arg_types = self
                .ctx
                .block_args(*source_block)
                .iter()
                .map(|arg| self.ctx.value_ty(*arg))
                .collect::<Vec<_>>();
            let arg_types = source_arg_types
                .into_iter()
                .map(|ty| self.convert_type(ty))
                .collect::<Vec<_>>();
            let converted_block =
                self.make_block(self.ctx.block(*source_block).location, &arg_types);
            let mut branch_mapping = mapping.clone();
            for (old, new) in self
                .ctx
                .block_args(*source_block)
                .iter()
                .copied()
                .zip(self.ctx.block_args(converted_block).iter().copied())
            {
                branch_mapping.insert(old, new);
            }
            let branch_flow = Flow {
                done: branch_done,
                dispatch: branch_dispatch,
                void_done: branch_void_done,
                ..flow.clone()
            };
            self.convert_sequence(
                self.ctx.block(*source_block).ops.to_vec(),
                0,
                converted_block,
                &mut branch_mapping,
                &branch_flow,
            )?;
            converted_regions.push(self.single_block_region(location, converted_block));
        }
        let [then_region, else_region] = converted_regions.as_slice() else {
            return Err(TributeControlToCpsError::one(
                POST_CPS_BOUNDARY,
                Some(source),
                Some(location),
                "scf.if requires exactly two regions",
            ));
        };
        let never = self.never_type();
        let condition = self.ctx.op_operands(source)[0];
        let condition = mapping.get(&condition).copied().unwrap_or(condition);
        let lowered = scf::r#if(
            self.ctx,
            location,
            condition,
            never,
            *then_region,
            *else_region,
        );
        self.copy_extra_attrs(source, lowered.op_ref(), &[]);
        self.ctx.push_op(block, lowered.op_ref());
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_direct_structured_if(
        &mut self,
        source: OpRef,
        source_ops: &[OpRef],
        index: usize,
        block: BlockRef,
        mapping: &mut HashMap<ValueRef, ValueRef>,
        flow: &Flow,
    ) -> Result<(), TributeControlToCpsError> {
        let location = self.ctx.op(source).location;
        let mut converted_regions = Vec::new();
        let source_regions = self.ctx.op(source).regions.clone();
        for source_region in source_regions {
            let source_blocks = self.ctx.region(source_region).blocks.to_vec();
            let [source_block] = source_blocks.as_slice() else {
                return Err(self
                    .malformed_source(source, "effectful scf.if regions must contain one block"));
            };
            let source_arg_types = self
                .ctx
                .block_args(*source_block)
                .iter()
                .map(|arg| self.ctx.value_ty(*arg))
                .collect::<Vec<_>>();
            let arg_types = source_arg_types
                .into_iter()
                .map(|ty| self.convert_type(ty))
                .collect::<Vec<_>>();
            let converted_block =
                self.make_block(self.ctx.block(*source_block).location, &arg_types);
            let mut branch_mapping = mapping.clone();
            for (old, new) in self
                .ctx
                .block_args(*source_block)
                .iter()
                .copied()
                .zip(self.ctx.block_args(converted_block).iter().copied())
            {
                branch_mapping.insert(old, new);
            }
            let branch_flow = Flow {
                preserve_scf_yield: true,
                ..flow.clone()
            };
            self.convert_sequence(
                self.ctx.block(*source_block).ops.to_vec(),
                0,
                converted_block,
                &mut branch_mapping,
                &branch_flow,
            )?;
            converted_regions.push(self.single_block_region(location, converted_block));
        }
        let [then_region, else_region] = converted_regions.as_slice() else {
            return Err(self.malformed_source(source, "scf.if requires exactly two regions"));
        };
        let condition = self.ctx.op_operands(source)[0];
        let condition = mapping.get(&condition).copied().unwrap_or(condition);
        let mut builder =
            OperationDataBuilder::new(location, Symbol::new("scf"), Symbol::new("if"))
                .operand(condition);
        for result_type in self.ctx.op_result_types(source).to_vec() {
            builder = builder.result(self.convert_type(result_type));
        }
        builder = builder.region(*then_region).region(*else_region);
        let data = builder.build(self.ctx);
        let lowered = self.ctx.create_op(data);
        self.copy_extra_attrs(source, lowered, &[]);
        self.ctx.push_op(block, lowered);
        for (old, new) in self
            .ctx
            .op_results(source)
            .to_vec()
            .into_iter()
            .zip(self.ctx.op_results(lowered).to_vec())
        {
            mapping.insert(old, new);
        }
        self.convert_sequence(source_ops.to_vec(), index + 1, block, mapping, flow)
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_structured_switch(
        &mut self,
        source: OpRef,
        source_ops: &[OpRef],
        index: usize,
        block: BlockRef,
        mapping: &HashMap<ValueRef, ValueRef>,
        flow: &Flow,
    ) -> Result<(), TributeControlToCpsError> {
        if flow.convention != CallingConvention::Cps || !self.ctx.op_result_types(source).is_empty()
        {
            return Err(TributeControlToCpsError::one(
                POST_CPS_BOUNDARY,
                Some(source),
                Some(self.ctx.op(source).location),
                "effectful scf.switch must be resultless inside a CPS callable",
            ));
        }
        let location = self.ctx.op(source).location;
        let continuation =
            self.build_void_suffix_continuation(source_ops, index + 1, mapping, flow, location)?;
        let continuation_op = match self.ctx.value_def(continuation) {
            trunk_ir::ValueDef::OpResult(op, _) => op,
            _ => unreachable!("structured continuation is a closure.lambda"),
        };
        self.ctx.push_op(block, continuation_op);

        let [source_body] = self.ctx.op(source).regions.as_slice() else {
            return Err(TributeControlToCpsError::one(
                POST_CPS_BOUNDARY,
                Some(source),
                Some(location),
                "scf.switch requires exactly one body region",
            ));
        };
        let source_body_blocks = self.ctx.region(*source_body).blocks.to_vec();
        let [source_body_block] = source_body_blocks.as_slice() else {
            return Err(TributeControlToCpsError::one(
                POST_CPS_BOUNDARY,
                Some(source),
                Some(location),
                "scf.switch body requires exactly one block",
            ));
        };
        let switch_block = self.make_block(location, &[]);
        let source_cases = self.ctx.block(*source_body_block).ops.to_vec();
        for case in source_cases {
            let case_data = self.ctx.op(case);
            let case_location = case_data.location;
            let is_case =
                case_data.dialect == Symbol::new("scf") && case_data.name == Symbol::new("case");
            let is_default =
                case_data.dialect == Symbol::new("scf") && case_data.name == Symbol::new("default");
            if !is_case && !is_default {
                return Err(TributeControlToCpsError::one(
                    POST_CPS_BOUNDARY,
                    Some(case),
                    Some(case_location),
                    "scf.switch body may contain only scf.case and scf.default",
                ));
            }
            let case_regions = case_data.regions.to_vec();
            let case_value = case_data.attributes.get("value").cloned();
            let [source_region] = case_regions.as_slice() else {
                return Err(TributeControlToCpsError::one(
                    POST_CPS_BOUNDARY,
                    Some(case),
                    Some(case_location),
                    "scf switch arm requires exactly one region",
                ));
            };
            let source_case_blocks = self.ctx.region(*source_region).blocks.to_vec();
            let [source_case_block] = source_case_blocks.as_slice() else {
                return Err(TributeControlToCpsError::one(
                    POST_CPS_BOUNDARY,
                    Some(case),
                    Some(case_location),
                    "scf switch arm requires exactly one block",
                ));
            };
            let source_case_ops = self.ctx.block(*source_case_block).ops.to_vec();
            let converted_block = self.make_block(case_location, &[]);
            let mut case_mapping = mapping.clone();
            let case_flow = Flow {
                done: Some(continuation),
                void_done: Some(continuation),
                ..flow.clone()
            };
            self.convert_sequence(
                source_case_ops,
                0,
                converted_block,
                &mut case_mapping,
                &case_flow,
            )?;
            let converted_region = self.single_block_region(case_location, converted_block);
            let converted = if is_case {
                let Some(case_value) = case_value else {
                    return Err(self.malformed_source(case, "scf.case requires a value attribute"));
                };
                scf::case(self.ctx, case_location, case_value, converted_region).op_ref()
            } else {
                scf::default(self.ctx, case_location, converted_region).op_ref()
            };
            self.ctx.push_op(switch_block, converted);
        }
        let switch_region = self.single_block_region(location, switch_block);
        let discriminant = self.ctx.op_operands(source)[0];
        let discriminant = mapping.get(&discriminant).copied().unwrap_or(discriminant);
        let lowered = scf::switch(self.ctx, location, discriminant, switch_region);
        self.copy_extra_attrs(source, lowered.op_ref(), &[]);
        self.ctx.push_op(block, lowered.op_ref());
        Ok(())
    }

    fn emit_exit(
        &mut self,
        block: BlockRef,
        location: Location,
        value: ValueRef,
        flow: &Flow,
    ) -> Result<(), TributeControlToCpsError> {
        if flow.convention == CallingConvention::Cps {
            let done = flow.done.ok_or_else(|| {
                TributeControlToCpsError::one(
                    POST_CPS_BOUNDARY,
                    None,
                    Some(location),
                    "CPS region has no verified Done boundary",
                )
            })?;
            let tail = func::tail_call_indirect(self.ctx, location, done, [value]);
            set_calling_convention(self.ctx, tail.op_ref(), CallingConvention::Cps);
            self.ctx.push_op(block, tail.op_ref());
        } else {
            let ret = func::r#return(self.ctx, location, [value]);
            self.ctx.push_op(block, ret.op_ref());
        }
        Ok(())
    }

    fn emit_void_exit(
        &mut self,
        block: BlockRef,
        location: Location,
        flow: &Flow,
    ) -> Result<(), TributeControlToCpsError> {
        let done = flow.done.ok_or_else(|| {
            TributeControlToCpsError::one(
                POST_CPS_BOUNDARY,
                None,
                Some(location),
                "structured region has no verified exit continuation",
            )
        })?;
        let tail =
            func::tail_call_indirect(self.ctx, location, done, std::iter::empty::<ValueRef>());
        set_calling_convention(self.ctx, tail.op_ref(), CallingConvention::Cps);
        self.ctx.push_op(block, tail.op_ref());
        Ok(())
    }

    fn build_void_suffix_continuation(
        &mut self,
        source_ops: &[OpRef],
        start: usize,
        mapping: &HashMap<ValueRef, ValueRef>,
        flow: &Flow,
        location: Location,
    ) -> Result<ValueRef, TributeControlToCpsError> {
        let block = self.make_block(location, &[]);
        let mut suffix_mapping = mapping.clone();
        self.convert_sequence(source_ops.to_vec(), start, block, &mut suffix_mapping, flow)?;
        let region = self.single_block_region(location, block);
        let captures = self.continuation_captures(region, flow);
        let never = self.never_type();
        let function = core::func(self.ctx, never, std::iter::empty::<TypeRef>()).as_type_ref();
        let closure_type = physical_closure_type(self.ctx, function, CallingConvention::Cps);
        let lambda = closure::lambda(self.ctx, location, captures, closure_type, region);
        set_calling_convention(self.ctx, lambda.op_ref(), CallingConvention::Cps);
        Ok(lambda.result(self.ctx))
    }

    fn lower_lambda(
        &mut self,
        source: OpRef,
        mapping: &mut HashMap<ValueRef, ValueRef>,
    ) -> Result<OpRef, TributeControlToCpsError> {
        let location = self.ctx.op(source).location;
        let logical_ty = self.ctx.op_result_types(source)[0];
        let callable = tribute_control::Callable::from_type_ref(self.ctx, logical_ty).unwrap();
        let convention =
            convert_convention(tribute_control::callable_convention(self.ctx, logical_ty).unwrap());
        let source_result = callable.result(self.ctx);
        let source_params = callable.params(self.ctx).to_vec();
        let result = self.convert_type(source_result);
        let source_param_types: Vec<_> = source_params
            .iter()
            .copied()
            .map(|ty| self.convert_type(ty))
            .collect();
        let evidence = self.evidence_type();
        let done_k = self.done_k_type(result);
        let dispatch = self.dispatch_type(result);
        let abi = CallableAbi::new(convention, source_param_types.clone(), result);
        let params = abi.lowered_params_with_dispatch(evidence, done_k, dispatch);
        let body_source = self.ctx.op(source).regions[0];
        let entry_source = self.ctx.region(body_source).blocks[0];
        let block = self.make_block(location, &params);
        let mut body_mapping = mapping.clone();
        let offset = abi.source_param_offset_with_dispatch();
        for (old, new) in self
            .ctx
            .block_args(entry_source)
            .to_vec()
            .into_iter()
            .zip(self.ctx.block_args(block)[offset..].iter().copied())
        {
            body_mapping.insert(old, new);
        }
        let evidence_value = convention
            .needs_evidence()
            .then(|| self.ctx.block_args(block)[0]);
        let done_index = usize::from(convention.needs_evidence());
        let done = convention
            .needs_done_k()
            .then(|| self.ctx.block_args(block)[done_index]);
        let dispatch = convention
            .needs_done_k()
            .then(|| self.ctx.block_args(block)[done_index + 1]);
        let flow = Flow {
            convention,
            evidence: evidence_value,
            resume_evidence: evidence_value,
            done,
            dispatch,
            boundary_result: result,
            resume_after: None,
            resume_done: None,
            void_done: None,
            preserve_scf_yield: false,
        };
        self.convert_sequence(
            self.ctx.block(entry_source).ops.to_vec(),
            0,
            block,
            &mut body_mapping,
            &flow,
        )?;
        let body = self.single_block_region(location, block);
        let captures: Vec<_> = self
            .ctx
            .op_operands(source)
            .iter()
            .map(|capture| mapping.get(capture).copied().unwrap_or(*capture))
            .collect();
        let physical_ty = self.convert_type(logical_ty);
        let lambda = closure::lambda(self.ctx, location, captures, physical_ty, body);
        self.copy_extra_attrs(source, lambda.op_ref(), &[CALLING_CONVENTION_ATTR]);
        set_calling_convention(self.ctx, lambda.op_ref(), convention);
        Ok(lambda.op_ref())
    }

    fn current_evidence(
        &self,
        source: OpRef,
        flow: &Flow,
    ) -> Result<ValueRef, TributeControlToCpsError> {
        flow.evidence.ok_or_else(|| {
            TributeControlToCpsError::one(
                POST_CPS_BOUNDARY,
                Some(source),
                Some(self.ctx.op(source).location),
                "operation requires evidence but the enclosing callable convention is Direct",
            )
        })
    }

    fn current_resume_evidence(
        &self,
        source: OpRef,
        flow: &Flow,
    ) -> Result<ValueRef, TributeControlToCpsError> {
        flow.resume_evidence.ok_or_else(|| {
            TributeControlToCpsError::one(
                POST_CPS_BOUNDARY,
                Some(source),
                Some(self.ctx.op(source).location),
                "CPS call has no verified dynamic evidence context",
            )
        })
    }

    fn lower_direct_call(
        &mut self,
        source: OpRef,
        target: &CallableInfo,
        mapping: &HashMap<ValueRef, ValueRef>,
        flow: &Flow,
    ) -> Result<func::Call, TributeControlToCpsError> {
        let location = self.ctx.op(source).location;
        let mut args = Vec::new();
        if target.convention.needs_evidence() {
            args.push(self.current_evidence(source, flow)?);
        }
        args.extend(
            self.ctx
                .op_operands(source)
                .iter()
                .map(|value| mapping.get(value).copied().unwrap_or(*value)),
        );
        let result_ty = self.convert_type(target.source_result);
        let call = func::call(self.ctx, location, args, result_ty, target.symbol);
        set_calling_convention(self.ctx, call.op_ref(), target.convention);
        Ok(call)
    }

    #[allow(clippy::too_many_arguments)]
    fn build_suffix_continuation(
        &mut self,
        source_ops: &[OpRef],
        start: usize,
        source_result: ValueRef,
        result_type: TypeRef,
        mapping: &HashMap<ValueRef, ValueRef>,
        flow: &Flow,
        location: Location,
    ) -> Result<ValueRef, TributeControlToCpsError> {
        let result_type = self.convert_type(result_type);
        let evidence_type = self.evidence_type();
        let parent_type = self.parent_types(flow.boundary_result).reference;
        let block = self.make_block(location, &[evidence_type, parent_type, result_type]);
        let context_evidence = self.ctx.block_args(block)[0];
        let parent_value = self.ctx.block_args(block)[1];
        let (parent_done, parent_dispatch) =
            self.unpack_parent(block, location, flow.boundary_result, parent_value);
        let mut body_mapping = mapping.clone();
        body_mapping.insert(source_result, self.ctx.block_args(block)[2]);
        let dynamic_done = if let Some(after) = flow.resume_after {
            let (op, done) = self.build_done_adapter(
                result_type,
                after,
                context_evidence,
                parent_value,
                location,
            );
            self.ctx.push_op(block, op);
            done
        } else {
            parent_done
        };
        let suffix_flow = Flow {
            evidence: Some(context_evidence),
            resume_evidence: Some(context_evidence),
            done: Some(dynamic_done),
            dispatch: Some(parent_dispatch),
            ..flow.clone()
        };
        self.convert_sequence(
            source_ops.to_vec(),
            start,
            block,
            &mut body_mapping,
            &suffix_flow,
        )?;
        let region = self.single_block_region(location, block);
        let captures = self.continuation_captures(region, flow);
        let closure_ty = self.completion_type(result_type, flow.boundary_result);
        let lambda = closure::lambda(self.ctx, location, captures, closure_ty, region);
        set_calling_convention(self.ctx, lambda.op_ref(), CallingConvention::Cps);
        Ok(lambda.result(self.ctx))
    }

    /// Adapt a call returning `value_type` to the caller's result-indexed
    /// `Done`/`Dispatch` boundary. Both adapters are immutable closure frames;
    /// the callee never needs to inspect or clone an opaque caller continuation.
    fn build_cps_call_adapters(
        &mut self,
        value_type: TypeRef,
        completion: ValueRef,
        flow: &Flow,
        location: Location,
    ) -> Result<(Vec<OpRef>, ValueRef, ValueRef), TributeControlToCpsError> {
        let boundary = flow.boundary_result;
        let current_done = flow.done.ok_or_else(|| {
            TributeControlToCpsError::one(
                POST_CPS_BOUNDARY,
                None,
                Some(location),
                "CPS call has no verified Done boundary",
            )
        })?;
        let current_dispatch = flow.dispatch.ok_or_else(|| {
            TributeControlToCpsError::one(
                POST_CPS_BOUNDARY,
                None,
                Some(location),
                "CPS call has no verified Dispatch boundary",
            )
        })?;
        let context_evidence = flow.resume_evidence.ok_or_else(|| {
            TributeControlToCpsError::one(
                POST_CPS_BOUNDARY,
                None,
                Some(location),
                "CPS call has no verified dynamic evidence context",
            )
        })?;

        let done_block = self.make_block(location, &[value_type]);
        let done_value = self.ctx.block_args(done_block)[0];
        let current_parent = self.pack_parent(
            done_block,
            location,
            boundary,
            current_done,
            current_dispatch,
        );
        let done_tail = func::tail_call_indirect(
            self.ctx,
            location,
            completion,
            [context_evidence, current_parent, done_value],
        );
        set_calling_convention(self.ctx, done_tail.op_ref(), CallingConvention::Cps);
        self.ctx.push_op(done_block, done_tail.op_ref());
        let done_region = self.single_block_region(location, done_block);
        let done_captures = ordered_external_values(self.ctx, done_region);
        let done_ty = self.done_k_type(value_type);
        let done_lambda = closure::lambda(self.ctx, location, done_captures, done_ty, done_region);
        set_calling_convention(self.ctx, done_lambda.op_ref(), CallingConvention::Cps);

        let factory = self.build_dispatch_adapter_factory(location, value_type, boundary);
        let dispatch_ty = self.dispatch_type(value_type);
        let dispatch_call = func::call(
            self.ctx,
            location,
            [completion, current_dispatch],
            dispatch_ty,
            factory,
        );
        set_calling_convention(self.ctx, dispatch_call.op_ref(), CallingConvention::Direct);

        Ok((
            vec![done_lambda.op_ref(), dispatch_call.op_ref()],
            done_lambda.result(self.ctx),
            dispatch_call.result(self.ctx),
        ))
    }

    /// Rebuild an exact `Resume<R>` from the dynamic `Parent<R>` supplied by
    /// a later handler arm. Both ordinary call adapters and lexical handler
    /// dispatchers use this same immutable continuation frame.
    fn build_rebound_resume(
        &mut self,
        location: Location,
        value_type: TypeRef,
        boundary: TypeRef,
        resume_body: ValueRef,
        completion: ValueRef,
        dispatch_rebuild: ResumeDispatchRebuild,
    ) -> (OpRef, ValueRef) {
        let evidence_type = self.evidence_type();
        let anyref = self.anyref_type();
        let parent_type = self.parent_types(boundary).reference;
        let block = self.make_block(location, &[evidence_type, parent_type, anyref]);
        let args = self.ctx.block_args(block).to_vec();
        let (next_done_op, next_done) =
            self.build_done_adapter(value_type, completion, args[0], args[1], location);
        self.ctx.push_op(block, next_done_op);
        let dispatch_type = self.dispatch_type(value_type);
        let rebuilt_dispatch = match dispatch_rebuild {
            ResumeDispatchRebuild::Adapter { factory } => {
                let (_, outer_dispatch) = self.unpack_parent(block, location, boundary, args[1]);
                func::call(
                    self.ctx,
                    location,
                    [completion, outer_dispatch],
                    dispatch_type,
                    factory,
                )
            }
            ResumeDispatchRebuild::LocalFactory(LocalDispatchFactory {
                factory,
                args: mut factory_args,
            }) => {
                factory_args.insert(0, args[1]);
                func::call(self.ctx, location, factory_args, dispatch_type, factory)
            }
        };
        set_calling_convention(
            self.ctx,
            rebuilt_dispatch.op_ref(),
            CallingConvention::Direct,
        );
        self.ctx.push_op(block, rebuilt_dispatch.op_ref());
        let parent = self.pack_parent(
            block,
            location,
            value_type,
            next_done,
            rebuilt_dispatch.result(self.ctx),
        );
        let tail =
            func::tail_call_indirect(self.ctx, location, resume_body, [args[0], parent, args[2]]);
        set_calling_convention(self.ctx, tail.op_ref(), CallingConvention::Cps);
        self.ctx.push_op(block, tail.op_ref());
        let region = self.single_block_region(location, block);
        let closure_type = self.erased_resumption_type(boundary);
        let resume = closure::lambda(
            self.ctx,
            location,
            ordered_external_values(self.ctx, region),
            closure_type,
            region,
        );
        set_calling_convention(self.ctx, resume.op_ref(), CallingConvention::Cps);
        (resume.op_ref(), resume.result(self.ctx))
    }

    /// Emit a direct factory for the `Dispatch<X>` adapter used at a CPS call
    /// inside a `Flow<R>`.  The factory is recursive only in the construction
    /// sense: every resumed invocation allocates a fresh dispatcher which
    /// captures the dynamically supplied `Parent<R>`.  The direct factory
    /// call builds a closure; it is never a control-flow fallback.
    fn build_dispatch_adapter_factory(
        &mut self,
        location: Location,
        value_type: TypeRef,
        boundary: TypeRef,
    ) -> Symbol {
        let symbol = self.fresh_helper("make_dispatch_adapter");
        let completion_ty = self.completion_type(value_type, boundary);
        let outer_dispatch_ty = self.dispatch_type(boundary);
        let dispatch_ty = self.dispatch_type(value_type);
        let factory_ty =
            core::func(self.ctx, dispatch_ty, [completion_ty, outer_dispatch_ty]).as_type_ref();
        let factory_block = self.make_block(location, &[completion_ty, outer_dispatch_ty]);
        let factory_args = self.ctx.block_args(factory_block).to_vec();
        let completion = factory_args[0];
        let outer_dispatch = factory_args[1];

        let anyref = self.anyref_type();
        let evidence_type = self.evidence_type();
        let i32_type = self.i32_type();
        let erased_resume_value_ty = self.erased_resumption_type(value_type);
        let dispatch_block = self.make_block(
            location,
            &[
                evidence_type,
                erased_resume_value_ty,
                i32_type,
                i32_type,
                i32_type,
                anyref,
            ],
        );
        let dispatch_args = self.ctx.block_args(dispatch_block).to_vec();
        let context_evidence = dispatch_args[0];
        let resume_value = dispatch_args[1];
        let prompt = dispatch_args[2];
        let ability_id = dispatch_args[3];
        let op_id = dispatch_args[4];
        let payload = dispatch_args[5];

        let (resume_op, resume_lambda) = self.build_rebound_resume(
            location,
            value_type,
            boundary,
            resume_value,
            completion,
            ResumeDispatchRebuild::Adapter { factory: symbol },
        );
        self.ctx.push_op(dispatch_block, resume_op);
        let dispatch_tail = func::tail_call_indirect(
            self.ctx,
            location,
            outer_dispatch,
            [
                context_evidence,
                resume_lambda,
                prompt,
                ability_id,
                op_id,
                payload,
            ],
        );
        set_calling_convention(self.ctx, dispatch_tail.op_ref(), CallingConvention::Cps);
        self.ctx.push_op(dispatch_block, dispatch_tail.op_ref());
        let dispatch_region = self.single_block_region(location, dispatch_block);
        let dispatch_lambda = closure::lambda(
            self.ctx,
            location,
            ordered_external_values(self.ctx, dispatch_region),
            dispatch_ty,
            dispatch_region,
        );
        set_calling_convention(self.ctx, dispatch_lambda.op_ref(), CallingConvention::Cps);
        self.ctx.push_op(factory_block, dispatch_lambda.op_ref());
        let ret = func::r#return(self.ctx, location, [dispatch_lambda.result(self.ctx)]);
        self.ctx.push_op(factory_block, ret.op_ref());
        let region = self.single_block_region(location, factory_block);
        let factory = func::func(self.ctx, location, symbol, factory_ty, region);
        set_calling_convention(self.ctx, factory.op_ref(), CallingConvention::Direct);
        self.ctx.push_op(self.module_block, factory.op_ref());
        symbol
    }

    /// Build only the `Done<X>` half of a result-indexed CPS boundary.
    ///
    /// This is used by a handler's normal completion: the handler answer
    /// crosses `after_handle`, while the handler's own dispatch remains the
    /// enclosing dispatch. Keeping the frame immutable is what makes nested
    /// resumes compose rather than overwrite an ambient root continuation.
    fn build_done_adapter(
        &mut self,
        value_type: TypeRef,
        completion: ValueRef,
        context_evidence: ValueRef,
        current_parent: ValueRef,
        location: Location,
    ) -> (OpRef, ValueRef) {
        let block = self.make_block(location, &[value_type]);
        let value = self.ctx.block_args(block)[0];
        let tail = func::tail_call_indirect(
            self.ctx,
            location,
            completion,
            [context_evidence, current_parent, value],
        );
        set_calling_convention(self.ctx, tail.op_ref(), CallingConvention::Cps);
        self.ctx.push_op(block, tail.op_ref());
        let region = self.single_block_region(location, block);
        let captures = ordered_external_values(self.ctx, region);
        let done_type = self.done_k_type(value_type);
        let lambda = closure::lambda(self.ctx, location, captures, done_type, region);
        set_calling_convention(self.ctx, lambda.op_ref(), CallingConvention::Cps);
        (lambda.op_ref(), lambda.result(self.ctx))
    }

    fn lower_func_ref(
        &mut self,
        source: OpRef,
    ) -> Result<(Vec<OpRef>, ValueRef), TributeControlToCpsError> {
        let location = self.ctx.op(source).location;
        let target_symbol = self
            .ctx
            .op(source)
            .attributes
            .get_symbol("func_ref")
            .expect("pre-CPS validation checked func_ref target");
        let target = self
            .current_func(target_symbol)
            .expect("pre-CPS validation resolved func_ref target in this module");
        let result_logical_ty = self.ctx.op_result_types(source)[0];
        let result_callable = tribute_control::Callable::from_type_ref(self.ctx, result_logical_ty)
            .expect("pre-CPS validation checked func_ref result type");
        let result_convention = tribute_control::callable_convention(self.ctx, result_logical_ty)
            .map(convert_convention)
            .expect("pre-CPS validation checked func_ref convention");
        debug_assert!(
            !target.convention.needs_done_k() || result_convention.needs_done_k(),
            "pre-CPS validation rejects a weaker func_ref result convention"
        );
        let result = self.convert_type(result_callable.result(self.ctx));
        let source_params: Vec<_> = result_callable
            .params(self.ctx)
            .to_vec()
            .into_iter()
            .map(|ty| self.convert_type(ty))
            .collect();
        let evidence_ty = self.evidence_type();
        let done_k_ty = self.done_k_type(result);
        let dispatch_ty = self.dispatch_type(result);
        let abi = CallableAbi::new(result_convention, source_params.clone(), result);
        let logical_params = abi.lowered_params_with_dispatch(evidence_ty, done_k_ty, dispatch_ty);
        let env_ty = self.anyref_type();
        let physical_params = abi.interpose_environment(&logical_params, env_ty);
        let physical_result = if result_convention == CallingConvention::Cps {
            self.never_type()
        } else {
            result
        };
        let adapter_ty =
            core::func(self.ctx, physical_result, physical_params.clone()).as_type_ref();
        let evidence_offset = usize::from(result_convention.needs_evidence());
        let block = self.make_block(location, &physical_params);
        self.ctx.block_mut(block).args[evidence_offset]
            .attrs
            .insert(
                Symbol::new("bind_name"),
                Attribute::Symbol(Symbol::new("__env")),
            );
        let args = self.ctx.block_args(block).to_vec();
        // The environment is interposed immediately after optional evidence,
        // so a present done continuation is the following slot.
        let done_offset = evidence_offset + 1;
        let dispatch_offset = done_offset + 1;
        let source_offset = usize::from(result_convention.needs_evidence())
            + 1
            + usize::from(result_convention.needs_done_k()) * 2;
        let mut target_args = Vec::new();
        if target.convention.needs_evidence() {
            target_args.push(args[0]);
        }
        if target.convention.needs_done_k() {
            target_args.push(args[done_offset]);
            target_args.push(args[dispatch_offset]);
        }
        target_args.extend_from_slice(&args[source_offset..]);
        if target.convention == CallingConvention::Cps {
            let tail = func::tail_call(self.ctx, location, target_args, target.symbol);
            set_calling_convention(self.ctx, tail.op_ref(), CallingConvention::Cps);
            self.ctx.push_op(block, tail.op_ref());
        } else {
            let target_result = self.convert_type(target.source_result);
            let call = func::call(
                self.ctx,
                location,
                target_args,
                target_result,
                target.symbol,
            );
            set_calling_convention(self.ctx, call.op_ref(), target.convention);
            self.ctx.push_op(block, call.op_ref());
            if result_convention == CallingConvention::Cps {
                let done_k = args[done_offset];
                let tail =
                    func::tail_call_indirect(self.ctx, location, done_k, [call.result(self.ctx)]);
                set_calling_convention(self.ctx, tail.op_ref(), CallingConvention::Cps);
                self.ctx.push_op(block, tail.op_ref());
            } else {
                let ret = func::r#return(self.ctx, location, [call.result(self.ctx)]);
                self.ctx.push_op(block, ret.op_ref());
            }
        }
        let region = self.single_block_region(location, block);
        let adapter_symbol = self.fresh_helper("func_ref_adapter");
        let adapter = func::func(self.ctx, location, adapter_symbol, adapter_ty, region);
        set_calling_convention(self.ctx, adapter.op_ref(), result_convention);
        self.ctx.push_op(self.module_block, adapter.op_ref());

        let empty_env_ty = self.ctx.types.intern(
            TypeDataBuilder::new(Symbol::new("adt"), Symbol::new("struct"))
                .attr(
                    "name",
                    Attribute::Symbol(Symbol::from_dynamic(&format!("{adapter_symbol}::env"))),
                )
                .attr("fields", Attribute::List(vec![]))
                .build(),
        );
        let empty_env = adt::struct_new(
            self.ctx,
            location,
            std::iter::empty::<ValueRef>(),
            empty_env_ty,
            empty_env_ty,
        );
        let closure_ty = self.convert_type(result_logical_ty);
        let closure_new = closure::new(
            self.ctx,
            location,
            empty_env.result(self.ctx),
            closure_ty,
            adapter_symbol,
        );
        set_calling_convention(self.ctx, closure_new.op_ref(), result_convention);
        Ok((
            vec![empty_env.op_ref(), closure_new.op_ref()],
            closure_new.result(self.ctx),
        ))
    }

    fn build_completion_continuation(
        &mut self,
        source_region: RegionRef,
        mapping: &HashMap<ValueRef, ValueRef>,
        flow: &Flow,
        location: Location,
    ) -> Result<(OpRef, ValueRef), TributeControlToCpsError> {
        let source_block = self.ctx.region(source_region).blocks[0];
        let source_arg = self.ctx.block_args(source_block)[0];
        let arg_type = self.convert_type(self.ctx.value_ty(source_arg));
        let evidence_type = self.evidence_type();
        let parent_type = self.parent_types(flow.boundary_result).reference;
        let block = self.make_block(location, &[evidence_type, parent_type, arg_type]);
        let context_evidence = self.ctx.block_args(block)[0];
        let parent_value = self.ctx.block_args(block)[1];
        let (parent_done, parent_dispatch) =
            self.unpack_parent(block, location, flow.boundary_result, parent_value);
        let mut body_mapping = mapping.clone();
        body_mapping.insert(source_arg, self.ctx.block_args(block)[2]);
        let completion_flow = Flow {
            convention: CallingConvention::Cps,
            evidence: Some(context_evidence),
            resume_evidence: Some(context_evidence),
            done: Some(parent_done),
            dispatch: Some(parent_dispatch),
            boundary_result: flow.boundary_result,
            resume_after: None,
            resume_done: None,
            void_done: None,
            preserve_scf_yield: false,
        };
        self.convert_sequence(
            self.ctx.block(source_block).ops.to_vec(),
            0,
            block,
            &mut body_mapping,
            &completion_flow,
        )?;
        let body = self.single_block_region(location, block);
        let captures = self.continuation_captures(body, flow);
        let closure_type = self.completion_type(arg_type, flow.boundary_result);
        let lambda = closure::lambda(self.ctx, location, captures, closure_type, body);
        set_calling_convention(self.ctx, lambda.op_ref(), CallingConvention::Cps);
        Ok((lambda.op_ref(), lambda.result(self.ctx)))
    }

    #[allow(clippy::too_many_arguments)]
    fn build_raw_resumption(
        &mut self,
        source_ops: &[OpRef],
        start: usize,
        source_result: ValueRef,
        input_type: TypeRef,
        mapping: &HashMap<ValueRef, ValueRef>,
        flow: &Flow,
        location: Location,
    ) -> Result<(OpRef, ValueRef), TributeControlToCpsError> {
        let input_type = self.convert_type(input_type);
        let evidence_type = self.evidence_type();
        let parent_type = self.parent_types(flow.boundary_result).reference;
        let block = self.make_block(location, &[evidence_type, parent_type, input_type]);
        let resume_evidence = self.ctx.block_args(block)[0];
        let parent_value = self.ctx.block_args(block)[1];
        let (resume_done, resume_dispatch) =
            self.unpack_parent(block, location, flow.boundary_result, parent_value);
        let resume_input = self.ctx.block_args(block)[2];
        let mut body_mapping = mapping.clone();
        body_mapping.insert(source_result, resume_input);
        let suffix_flow = Flow {
            evidence: Some(resume_evidence),
            resume_evidence: Some(resume_evidence),
            done: Some(resume_done),
            dispatch: Some(resume_dispatch),
            ..flow.clone()
        };
        self.convert_sequence(
            source_ops.to_vec(),
            start,
            block,
            &mut body_mapping,
            &suffix_flow,
        )?;
        let region = self.single_block_region(location, block);
        let captures = self.continuation_captures(region, &suffix_flow);
        let closure_type = self.resumption_type(input_type, flow.boundary_result);
        let lambda = closure::lambda(self.ctx, location, captures, closure_type, region);
        set_calling_convention(self.ctx, lambda.op_ref(), CallingConvention::Cps);
        Ok((lambda.op_ref(), lambda.result(self.ctx)))
    }

    fn build_one_shot_wrapper(
        &mut self,
        raw_continuation: ValueRef,
        input_type: TypeRef,
        answer_type: TypeRef,
        location: Location,
    ) -> (Vec<OpRef>, ValueRef) {
        let i1_type = self
            .ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i1")).build());
        let state_name = self.fresh_helper("one_shot_state");
        let state_type = self.ctx.types.intern(
            TypeDataBuilder::new(Symbol::new("adt"), Symbol::new("struct"))
                .attr("name", Attribute::Symbol(state_name))
                .attr(
                    "fields",
                    Attribute::List(vec![Attribute::List(vec![
                        Attribute::Symbol(Symbol::new("consumed")),
                        Attribute::Type(i1_type),
                    ])]),
                )
                .build(),
        );
        let not_consumed = arith::r#const(self.ctx, location, i1_type, Attribute::Int(0));
        let state = adt::struct_new(
            self.ctx,
            location,
            [not_consumed.result(self.ctx)],
            state_type,
            state_type,
        );

        let evidence_type = self.evidence_type();
        let parent_type = self.parent_types(answer_type).reference;
        let anyref = self.anyref_type();
        let block = self.make_block(location, &[evidence_type, parent_type, anyref]);
        let args = self.ctx.block_args(block).to_vec();
        let input = core::unrealized_conversion_cast(self.ctx, location, args[2], input_type);
        self.ctx.push_op(block, input.op_ref());
        let consumed = adt::struct_get(
            self.ctx,
            location,
            state.result(self.ctx),
            i1_type,
            state_type,
            0,
        );
        self.ctx.push_op(block, consumed.op_ref());

        let reject_block = self.make_block(location, &[]);
        let unreachable = func::unreachable(self.ctx, location);
        self.ctx.push_op(reject_block, unreachable.op_ref());
        let reject_region = self.single_block_region(location, reject_block);

        let enter_block = self.make_block(location, &[]);
        let consumed_true = arith::r#const(self.ctx, location, i1_type, Attribute::Int(1));
        self.ctx.push_op(enter_block, consumed_true.op_ref());
        let mark = adt::struct_set(
            self.ctx,
            location,
            state.result(self.ctx),
            consumed_true.result(self.ctx),
            state_type,
            0,
        );
        self.ctx.push_op(enter_block, mark.op_ref());
        let tail = func::tail_call_indirect(
            self.ctx,
            location,
            raw_continuation,
            [args[0], args[1], input.result(self.ctx)],
        );
        set_calling_convention(self.ctx, tail.op_ref(), CallingConvention::Cps);
        self.ctx.push_op(enter_block, tail.op_ref());
        let enter_region = self.single_block_region(location, enter_block);

        let never = self.never_type();
        let guard = scf::r#if(
            self.ctx,
            location,
            consumed.result(self.ctx),
            never,
            reject_region,
            enter_region,
        );
        self.ctx.push_op(block, guard.op_ref());
        let region = self.single_block_region(location, block);
        let captures = ordered_external_values(self.ctx, region);
        let closure_type = self.erased_resumption_type(answer_type);
        let wrapper = closure::lambda(self.ctx, location, captures, closure_type, region);
        set_calling_convention(self.ctx, wrapper.op_ref(), CallingConvention::Cps);
        (
            vec![not_consumed.op_ref(), state.op_ref(), wrapper.op_ref()],
            wrapper.result(self.ctx),
        )
    }

    /// Callback ABIs carry dynamic evidence explicitly. Ordinary closure
    /// captures are therefore determined solely by lexical value use.
    fn continuation_captures(&self, region: RegionRef, flow: &Flow) -> Vec<ValueRef> {
        let _ = flow;
        ordered_external_values(self.ctx, region)
    }

    fn build_reject_continuation(
        &mut self,
        _input_type: TypeRef,
        answer_type: TypeRef,
        location: Location,
    ) -> (OpRef, ValueRef) {
        let evidence_type = self.evidence_type();
        let parent_type = self.parent_types(answer_type).reference;
        let anyref = self.anyref_type();
        let block = self.make_block(location, &[evidence_type, parent_type, anyref]);
        let unreachable = func::unreachable(self.ctx, location);
        self.ctx.push_op(block, unreachable.op_ref());
        let region = self.single_block_region(location, block);
        let closure_type = self.erased_resumption_type(answer_type);
        let lambda = closure::lambda(
            self.ctx,
            location,
            std::iter::empty::<ValueRef>(),
            closure_type,
            region,
        );
        set_calling_convention(self.ctx, lambda.op_ref(), CallingConvention::Cps);
        (lambda.op_ref(), lambda.result(self.ctx))
    }

    fn lower_general_perform(
        &mut self,
        source: OpRef,
        source_ops: &[OpRef],
        index: usize,
        block: BlockRef,
        mapping: &HashMap<ValueRef, ValueRef>,
        flow: &Flow,
    ) -> Result<(), TributeControlToCpsError> {
        if flow.convention != CallingConvention::Cps {
            return Err(TributeControlToCpsError::one(
                POST_CPS_BOUNDARY,
                Some(source),
                Some(self.ctx.op(source).location),
                "general operation appears in a non-CPS callable",
            ));
        }
        let location = self.ctx.op(source).location;
        let resume_flow = flow.clone();
        let old_result = self.ctx.op_result(source, 0);
        let input_type = self.ctx.op_result_types(source)[0];
        let continuation = if type_is(self.ctx, input_type, "core", "never") {
            let (op, value) =
                self.build_reject_continuation(input_type, flow.boundary_result, location);
            self.ctx.push_op(block, op);
            value
        } else {
            let (raw_op, raw) = self.build_raw_resumption(
                source_ops,
                index + 1,
                old_result,
                input_type,
                mapping,
                &resume_flow,
                location,
            )?;
            self.ctx.push_op(block, raw_op);
            let converted_input = self.convert_type(input_type);
            let (ops, one_shot) = self.build_one_shot_wrapper(
                raw,
                converted_input,
                resume_flow.boundary_result,
                location,
            );
            for op in ops {
                self.ctx.push_op(block, op);
            }
            one_shot
        };
        let args: Vec<_> = self
            .ctx
            .op_operands(source)
            .iter()
            .map(|arg| mapping.get(arg).copied().unwrap_or(*arg))
            .collect();
        let ability_ref = self
            .ctx
            .op(source)
            .attributes
            .get_type("ability_ref")
            .expect("pre-CPS validation checked perform ability");
        let op_name = self
            .ctx
            .op(source)
            .attributes
            .get_symbol("op_name")
            .expect("pre-CPS validation checked perform operation");
        let perform = ability::perform(
            self.ctx,
            location,
            resume_flow.resume_evidence.ok_or_else(|| {
                TributeControlToCpsError::one(
                    POST_CPS_BOUNDARY,
                    Some(source),
                    Some(location),
                    "general operation has no verified dynamic evidence context",
                )
            })?,
            resume_flow.dispatch.ok_or_else(|| {
                TributeControlToCpsError::one(
                    POST_CPS_BOUNDARY,
                    Some(source),
                    Some(location),
                    "general operation has no verified Dispatch boundary",
                )
            })?,
            continuation,
            args,
            ability_ref,
            op_name,
        );
        self.ctx.push_op(block, perform.op_ref());
        Ok(())
    }

    fn lower_resume(
        &mut self,
        source: OpRef,
        source_ops: &[OpRef],
        index: usize,
        block: BlockRef,
        mapping: &HashMap<ValueRef, ValueRef>,
        flow: &Flow,
    ) -> Result<(), TributeControlToCpsError> {
        if flow.convention != CallingConvention::Cps {
            return Err(TributeControlToCpsError::one(
                POST_CPS_BOUNDARY,
                Some(source),
                Some(self.ctx.op(source).location),
                "resume appears outside a CPS handler arm",
            ));
        }
        let location = self.ctx.op(source).location;
        let resume_flow = flow.clone();
        let token_source = self.ctx.op_operands(source)[0];
        let value_source = self.ctx.op_operands(source)[1];
        let token = mapping.get(&token_source).copied().unwrap_or(token_source);
        let value = mapping.get(&value_source).copied().unwrap_or(value_source);
        let old_result = self.ctx.op_result(source, 0);
        let result_type = self.ctx.op_result_types(source)[0];
        let suffix = self.build_suffix_continuation(
            source_ops,
            index + 1,
            old_result,
            result_type,
            mapping,
            &resume_flow,
            location,
        )?;
        let suffix_op = match self.ctx.value_def(suffix) {
            trunk_ir::ValueDef::OpResult(op, _) => op,
            _ => unreachable!("suffix is produced by closure.lambda"),
        };
        self.ctx.push_op(block, suffix_op);
        let answer_type = self.convert_type(result_type);
        let (adapter_ops, arm_done, arm_dispatch) =
            self.build_cps_call_adapters(answer_type, suffix, &resume_flow, location)?;
        for op in adapter_ops {
            self.ctx.push_op(block, op);
        }
        let arm_parent = self.pack_parent(
            block,
            location,
            resume_flow.boundary_result,
            arm_done,
            arm_dispatch,
        );
        let context_evidence = resume_flow.resume_evidence.ok_or_else(|| {
            TributeControlToCpsError::one(
                POST_CPS_BOUNDARY,
                Some(source),
                Some(location),
                "resume has no verified dynamic evidence context",
            )
        })?;
        let tail = func::tail_call_indirect(
            self.ctx,
            location,
            token,
            [context_evidence, arm_parent, value],
        );
        set_calling_convention(self.ctx, tail.op_ref(), CallingConvention::Cps);
        self.ctx.push_op(block, tail.op_ref());
        Ok(())
    }

    fn lower_handler_arm(
        &mut self,
        source: OpRef,
        outer_mapping: &HashMap<ValueRef, ValueRef>,
        outer_flow: &Flow,
    ) -> Result<HandlerArmInfo, TributeControlToCpsError> {
        let location = self.ctx.op(source).location;
        let ability_ref = self
            .ctx
            .op(source)
            .attributes
            .get_type("ability_ref")
            .unwrap();
        let op_name = self
            .ctx
            .op(source)
            .attributes
            .get_symbol("op_name")
            .unwrap();
        let kind = self.ctx.op(source).attributes.get_symbol("kind").unwrap();
        let operation_result = self
            .ctx
            .op(source)
            .attributes
            .get_type("operation_result_type")
            .unwrap();
        let source_region = self.ctx.op(source).regions[0];
        let source_block = self.ctx.region(source_region).blocks[0];
        let source_args = self.ctx.block_args(source_block).to_vec();
        let has_resume_token = source_args.last().is_some_and(|arg| {
            type_is(
                self.ctx,
                self.ctx.value_ty(*arg),
                "tribute_control",
                "resume_token",
            )
        });
        let evidence_type = self.evidence_type();
        let converted_args: Vec<_> = source_args
            .iter()
            .map(|arg| self.convert_type(self.ctx.value_ty(*arg)))
            .collect();
        let mut params = vec![evidence_type];
        let parent_type = (kind == Symbol::new("op"))
            .then(|| self.parent_types(outer_flow.boundary_result).reference);
        if let Some(parent_type) = parent_type {
            params.push(parent_type);
        }
        // A general handler arm evaluates under its outer lexical evidence,
        // but receives the effect-point evidence separately for any resume
        // context it constructs.
        if kind == Symbol::new("op") {
            params.push(evidence_type);
        }
        params.extend_from_slice(&converted_args);
        let block = self.make_block(location, &params);
        let mut mapping = outer_mapping.clone();
        let source_offset =
            1 + usize::from(parent_type.is_some()) + usize::from(kind == Symbol::new("op"));
        for (old, new) in source_args
            .into_iter()
            .zip(self.ctx.block_args(block)[source_offset..].iter().copied())
        {
            mapping.insert(old, new);
        }
        let evidence = self.ctx.block_args(block)[0];
        let resume_evidence = (kind == Symbol::new("op")).then(|| self.ctx.block_args(block)[2]);
        let convention = if kind == Symbol::new("fn") {
            CallingConvention::EvidenceDirect
        } else {
            CallingConvention::Cps
        };
        let parent_values = parent_type.map(|_| {
            self.unpack_parent(
                block,
                location,
                outer_flow.boundary_result,
                self.ctx.block_args(block)[1],
            )
        });
        let flow = Flow {
            convention,
            evidence: Some(evidence),
            resume_evidence: resume_evidence.or(Some(evidence)),
            done: parent_values.map(|(done, _)| done),
            dispatch: parent_values.map(|(_, dispatch)| dispatch),
            boundary_result: outer_flow.boundary_result,
            resume_after: outer_flow.resume_after,
            resume_done: outer_flow.resume_done,
            void_done: outer_flow.void_done,
            preserve_scf_yield: false,
        };
        self.convert_sequence(
            self.ctx.block(source_block).ops.to_vec(),
            0,
            block,
            &mut mapping,
            &flow,
        )?;
        let region = self.single_block_region(location, block);
        let captures = ordered_external_values(self.ctx, region);
        let result = if convention == CallingConvention::Cps {
            self.never_type()
        } else {
            self.convert_type(operation_result)
        };
        let function = core::func(self.ctx, result, params).as_type_ref();
        let closure_type = physical_closure_type(self.ctx, function, convention);
        let lambda = closure::lambda(self.ctx, location, captures, closure_type, region);
        set_calling_convention(self.ctx, lambda.op_ref(), convention);
        Ok(HandlerArmInfo {
            op: lambda.op_ref(),
            value: lambda.result(self.ctx),
            ability_ref,
            op_name,
            kind,
            operation_result: self.convert_type(operation_result),
            params: converted_args,
            has_resume_token,
            parent_type,
        })
    }

    fn unpack_handler_payload(
        &mut self,
        block: BlockRef,
        location: Location,
        payload: ValueRef,
        arm: &HandlerArmInfo,
    ) -> Vec<ValueRef> {
        let value_params = if arm.has_resume_token {
            &arm.params[..arm.params.len() - 1]
        } else {
            arm.params.as_slice()
        };
        let payload_type = ability::operation_payload_type_ref(
            self.ctx,
            arm.ability_ref,
            arm.op_name,
            value_params.iter().copied(),
        );
        let cast = core::unrealized_conversion_cast(self.ctx, location, payload, payload_type);
        self.ctx.push_op(block, cast.op_ref());
        value_params
            .iter()
            .copied()
            .enumerate()
            .map(|(index, ty)| {
                let get = adt::struct_get(
                    self.ctx,
                    location,
                    cast.result(self.ctx),
                    ty,
                    payload_type,
                    index as u32,
                );
                self.ctx.push_op(block, get.op_ref());
                get.result(self.ctx)
            })
            .collect()
    }

    fn build_handler_dispatcher(
        &mut self,
        location: Location,
        arms: &[HandlerArmInfo],
        general: bool,
    ) -> (OpRef, ValueRef) {
        // The marker's compatibility handler slot is deliberately not a final
        // control path.  Final `effect.dispatch_cps` reaches the exact local
        // dispatcher factory above, where a `Parent<A>` is available.  Keep a
        // typed rejecting placeholder here instead of casting the old marker
        // continuation into the new `ResumeExact<I, A>` ABI.
        if general {
            let evidence_type = self.evidence_type();
            let parent_type = arms
                .iter()
                .find(|arm| arm.kind == Symbol::new("op"))
                .and_then(|arm| arm.parent_type)
                .unwrap_or_else(|| self.parent_types(arms[0].operation_result).reference);
            let i32_type = self.i32_type();
            let anyref = self.anyref_type();
            let params = [evidence_type, parent_type, i32_type, anyref];
            let block = self.make_block(location, &params);
            let unreachable = func::unreachable(self.ctx, location);
            self.ctx.push_op(block, unreachable.op_ref());
            let region = self.single_block_region(location, block);
            let never = self.never_type();
            let function = core::func(self.ctx, never, params).as_type_ref();
            let closure_type = physical_closure_type(self.ctx, function, CallingConvention::Cps);
            let lambda = closure::lambda(
                self.ctx,
                location,
                std::iter::empty::<ValueRef>(),
                closure_type,
                region,
            );
            set_calling_convention(self.ctx, lambda.op_ref(), CallingConvention::Cps);
            return (lambda.op_ref(), lambda.result(self.ctx));
        }
        let evidence_type = self.evidence_type();
        let anyref = self.anyref_type();
        let i32_type = self
            .ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());
        let params = if general {
            vec![evidence_type, anyref, i32_type, anyref]
        } else {
            vec![evidence_type, i32_type, anyref]
        };
        let result = if general { self.never_type() } else { anyref };
        let block = self.make_block(location, &params);
        let args = self.ctx.block_args(block).to_vec();
        let evidence = args[0];
        let (continuation, op_idx, payload) = if general {
            (Some(args[1]), args[2], args[3])
        } else {
            (None, args[1], args[2])
        };

        let switch_block = self.make_block(location, &[]);
        let selected: Vec<_> = arms
            .iter()
            .filter(|arm| (arm.kind == Symbol::new("op")) == general)
            .collect();
        for arm in selected {
            let case_block = self.make_block(location, &[]);
            let mut call_args = vec![evidence];
            call_args.extend(self.unpack_handler_payload(case_block, location, payload, arm));
            if arm.has_resume_token {
                let token_type = *arm.params.last().unwrap();
                let token = core::unrealized_conversion_cast(
                    self.ctx,
                    location,
                    continuation.expect("general dispatcher has a continuation"),
                    token_type,
                );
                self.ctx.push_op(case_block, token.op_ref());
                call_args.push(token.result(self.ctx));
            }
            if general {
                let tail = func::tail_call_indirect(self.ctx, location, arm.value, call_args);
                set_calling_convention(self.ctx, tail.op_ref(), CallingConvention::Cps);
                self.ctx.push_op(case_block, tail.op_ref());
            } else {
                let call = func::call_indirect(
                    self.ctx,
                    location,
                    arm.value,
                    call_args,
                    arm.operation_result,
                );
                set_calling_convention(self.ctx, call.op_ref(), CallingConvention::EvidenceDirect);
                self.ctx.push_op(case_block, call.op_ref());
                let erased = core::unrealized_conversion_cast(
                    self.ctx,
                    location,
                    call.result(self.ctx),
                    anyref,
                );
                self.ctx.push_op(case_block, erased.op_ref());
                let ret = func::r#return(self.ctx, location, [erased.result(self.ctx)]);
                self.ctx.push_op(case_block, ret.op_ref());
            }
            let case_region = self.single_block_region(location, case_block);
            let op_index = ability::compute_op_idx(
                self.ctx.types.get(arm.ability_ref).attrs.get_symbol("name"),
                Some(arm.op_name),
            );
            let case = scf::case(
                self.ctx,
                location,
                Attribute::Int(op_index as i128),
                case_region,
            );
            self.ctx.push_op(switch_block, case.op_ref());
        }
        let reject_block = self.make_block(location, &[]);
        let unreachable = func::unreachable(self.ctx, location);
        self.ctx.push_op(reject_block, unreachable.op_ref());
        let reject_region = self.single_block_region(location, reject_block);
        let default = scf::default(self.ctx, location, reject_region);
        self.ctx.push_op(switch_block, default.op_ref());
        let switch_region = self.single_block_region(location, switch_block);
        let switch = scf::switch(self.ctx, location, op_idx, switch_region);
        self.ctx.push_op(block, switch.op_ref());

        let region = self.single_block_region(location, block);
        let captures = ordered_external_values(self.ctx, region);
        let convention = if general {
            CallingConvention::Cps
        } else {
            CallingConvention::EvidenceDirect
        };
        let function = core::func(self.ctx, result, params).as_type_ref();
        let closure_type = physical_closure_type(self.ctx, function, convention);
        let lambda = closure::lambda(self.ctx, location, captures, closure_type, region);
        set_calling_convention(self.ctx, lambda.op_ref(), convention);
        (lambda.op_ref(), lambda.result(self.ctx))
    }

    /// Emit an immutable recursive factory for a lexical handle dispatcher.
    /// A factory invocation captures one exact `Parent<A>` in the returned
    /// dispatcher.  Resuming a body invokes the same factory with a new
    /// parent, so a nested resume cannot overwrite an outer arm suffix.
    fn build_local_cps_dispatch_factory(
        &mut self,
        location: Location,
        arms: &[HandlerArmInfo],
        body_type: TypeRef,
        answer_type: TypeRef,
    ) -> Symbol {
        let symbol = self.fresh_helper("make_local_dispatch");
        let parent_type = self.parent_types(answer_type).reference;
        let evidence_type = self.evidence_type();
        let i32_type = self.i32_type();
        let completion_type = self.completion_type(body_type, answer_type);
        let mut params = vec![parent_type, evidence_type, i32_type, completion_type];
        params.extend(arms.iter().map(|arm| self.ctx.value_ty(arm.value)));
        let dispatch_type = self.dispatch_type(body_type);
        let factory_type = core::func(self.ctx, dispatch_type, params.clone()).as_type_ref();
        let factory_block = self.make_block(location, &params);
        let args = self.ctx.block_args(factory_block).to_vec();
        let mut factory_arms = arms.to_vec();
        for (arm, value) in factory_arms.iter_mut().zip(args[4..].iter().copied()) {
            arm.value = value;
        }
        let (_, parent_dispatch) =
            self.unpack_parent(factory_block, location, answer_type, args[0]);
        let foreign_factory = LocalDispatchFactory {
            factory: symbol,
            args: args[1..].to_vec(),
        };
        let (foreign_op, foreign_dispatch) = self.build_local_foreign_dispatch(
            location,
            body_type,
            answer_type,
            parent_dispatch,
            args[3],
            foreign_factory,
        );
        self.ctx.push_op(factory_block, foreign_op);
        let dispatcher = self.build_local_cps_dispatcher_instance(
            location,
            &factory_arms,
            body_type,
            answer_type,
            args[0],
            args[1],
            args[2],
            args[3],
            foreign_dispatch,
            symbol,
            args[1..].to_vec(),
        );
        self.ctx.push_op(factory_block, dispatcher.0);
        let ret = func::r#return(self.ctx, location, [dispatcher.1]);
        self.ctx.push_op(factory_block, ret.op_ref());
        let region = self.single_block_region(location, factory_block);
        let factory = func::func(self.ctx, location, symbol, factory_type, region);
        set_calling_convention(self.ctx, factory.op_ref(), CallingConvention::Direct);
        self.ctx.push_op(self.module_block, factory.op_ref());
        symbol
    }

    /// Forward a foreign operation through the enclosing dispatcher while
    /// retaining this lexical handle's factory for a later resumption.
    fn build_local_foreign_dispatch(
        &mut self,
        location: Location,
        body_type: TypeRef,
        answer_type: TypeRef,
        parent_dispatch: ValueRef,
        completion: ValueRef,
        local_factory: LocalDispatchFactory,
    ) -> (OpRef, ValueRef) {
        let evidence_type = self.evidence_type();
        let i32_type = self.i32_type();
        let anyref = self.anyref_type();
        let body_resume_type = self.erased_resumption_type(body_type);
        let block = self.make_block(
            location,
            &[
                evidence_type,
                body_resume_type,
                i32_type,
                i32_type,
                i32_type,
                anyref,
            ],
        );
        let args = self.ctx.block_args(block).to_vec();
        let (resume_op, resume) = self.build_rebound_resume(
            location,
            body_type,
            answer_type,
            args[1],
            completion,
            ResumeDispatchRebuild::LocalFactory(local_factory),
        );
        self.ctx.push_op(block, resume_op);
        let foreign_tail = func::tail_call_indirect(
            self.ctx,
            location,
            parent_dispatch,
            [args[0], resume, args[2], args[3], args[4], args[5]],
        );
        set_calling_convention(self.ctx, foreign_tail.op_ref(), CallingConvention::Cps);
        self.ctx.push_op(block, foreign_tail.op_ref());
        let region = self.single_block_region(location, block);
        let dispatch_type = self.dispatch_type(body_type);
        let dispatch = closure::lambda(
            self.ctx,
            location,
            ordered_external_values(self.ctx, region),
            dispatch_type,
            region,
        );
        set_calling_convention(self.ctx, dispatch.op_ref(), CallingConvention::Cps);
        (dispatch.op_ref(), dispatch.result(self.ctx))
    }

    #[allow(clippy::too_many_arguments)]
    fn build_local_cps_dispatcher_instance(
        &mut self,
        location: Location,
        arms: &[HandlerArmInfo],
        body_type: TypeRef,
        answer_type: TypeRef,
        parent: ValueRef,
        outer_evidence: ValueRef,
        prompt_tag: ValueRef,
        completion: ValueRef,
        foreign_dispatch: ValueRef,
        factory: Symbol,
        factory_static_args: Vec<ValueRef>,
    ) -> (OpRef, ValueRef) {
        let anyref = self.anyref_type();
        let i32_type = self.i32_type();
        let i1_type = self
            .ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i1")).build());
        let evidence_type = self.evidence_type();
        let params = [
            evidence_type,
            self.erased_resumption_type(body_type),
            i32_type,
            i32_type,
            i32_type,
            anyref,
        ];
        let block = self.make_block(location, &params);
        let args = self.ctx.block_args(block).to_vec();
        let effect_evidence = args[0];
        let resume = args[1];
        let prompt = args[2];
        let ability_id = args[3];
        let op_id = args[4];
        let payload = args[5];

        let foreign_block = self.make_block(location, &[]);
        let foreign_tail = func::tail_call_indirect(
            self.ctx,
            location,
            foreign_dispatch,
            [effect_evidence, resume, prompt, ability_id, op_id, payload],
        );
        set_calling_convention(self.ctx, foreign_tail.op_ref(), CallingConvention::Cps);
        self.ctx.push_op(foreign_block, foreign_tail.op_ref());
        let foreign_region = self.single_block_region(location, foreign_block);

        let selected_block = self.make_block(location, &[]);
        for arm in arms.iter().filter(|arm| arm.kind == Symbol::new("op")) {
            let case_block = self.make_block(location, &[]);
            let arm_ability =
                ability::ability_id_const(self.ctx, location, i32_type, arm.ability_ref);
            self.ctx.push_op(case_block, arm_ability.op_ref());
            let same_ability = arith::cmpi(
                self.ctx,
                location,
                ability_id,
                arm_ability.result(self.ctx),
                i1_type,
                Symbol::new("eq"),
            );
            self.ctx.push_op(case_block, same_ability.op_ref());

            let local_block = self.make_block(location, &[]);
            let mut call_args = vec![outer_evidence, parent, effect_evidence];
            call_args.extend(self.unpack_handler_payload(local_block, location, payload, arm));
            if arm.has_resume_token {
                let token_type = *arm.params.last().expect("resume arm has token type");
                let token_function = cps_closure_function_type(self.ctx, token_type)
                    .expect("pre-CPS validation proved a CPS resume token closure");
                let input_type =
                    *self.ctx.types.get(token_function).params.get(3).expect(
                        "resume token has Evidence, Parent and exact source-input parameters",
                    );
                let (token_op, token) = self.build_exact_handler_token(
                    location,
                    HandlerTokenFactory {
                        resume_body: resume,
                        completion,
                        input_type,
                        body_type,
                        answer_type,
                        dispatch_factory: factory,
                        dispatch_factory_args: factory_static_args.clone(),
                    },
                );
                self.ctx.push_op(local_block, token_op);
                call_args.push(token);
            }
            let local_tail = func::tail_call_indirect(self.ctx, location, arm.value, call_args);
            set_calling_convention(self.ctx, local_tail.op_ref(), CallingConvention::Cps);
            self.ctx.push_op(local_block, local_tail.op_ref());
            let local_region = self.single_block_region(location, local_block);

            let fallback_block = self.make_block(location, &[]);
            let fallback_tail = func::tail_call_indirect(
                self.ctx,
                location,
                foreign_dispatch,
                [effect_evidence, resume, prompt, ability_id, op_id, payload],
            );
            set_calling_convention(self.ctx, fallback_tail.op_ref(), CallingConvention::Cps);
            self.ctx.push_op(fallback_block, fallback_tail.op_ref());
            let fallback_region = self.single_block_region(location, fallback_block);
            let never = self.never_type();
            let choice = scf::r#if(
                self.ctx,
                location,
                same_ability.result(self.ctx),
                never,
                local_region,
                fallback_region,
            );
            self.ctx.push_op(case_block, choice.op_ref());
            let case_region = self.single_block_region(location, case_block);
            let op_index = ability::compute_op_idx(
                self.ctx.types.get(arm.ability_ref).attrs.get_symbol("name"),
                Some(arm.op_name),
            );
            let case = scf::case(
                self.ctx,
                location,
                Attribute::Int(op_index as i128),
                case_region,
            );
            self.ctx.push_op(selected_block, case.op_ref());
        }
        let default_block = self.make_block(location, &[]);
        let default_tail = func::tail_call_indirect(
            self.ctx,
            location,
            foreign_dispatch,
            [effect_evidence, resume, prompt, ability_id, op_id, payload],
        );
        set_calling_convention(self.ctx, default_tail.op_ref(), CallingConvention::Cps);
        self.ctx.push_op(default_block, default_tail.op_ref());
        let default_region = self.single_block_region(location, default_block);
        let default = scf::default(self.ctx, location, default_region);
        self.ctx.push_op(selected_block, default.op_ref());
        let selected_region = self.single_block_region(location, selected_block);
        let selected = scf::switch(self.ctx, location, op_id, selected_region);

        let same_prompt = arith::cmpi(
            self.ctx,
            location,
            prompt,
            prompt_tag,
            i1_type,
            Symbol::new("eq"),
        );
        self.ctx.push_op(block, same_prompt.op_ref());
        let local_switch_block = self.make_block(location, &[]);
        self.ctx.push_op(local_switch_block, selected.op_ref());
        let local_switch_region = self.single_block_region(location, local_switch_block);
        let never = self.never_type();
        let choose = scf::r#if(
            self.ctx,
            location,
            same_prompt.result(self.ctx),
            never,
            local_switch_region,
            foreign_region,
        );
        self.ctx.push_op(block, choose.op_ref());
        let region = self.single_block_region(location, block);
        let captures = ordered_external_values(self.ctx, region);
        let dispatch_type = self.dispatch_type(body_type);
        let lambda = closure::lambda(self.ctx, location, captures, dispatch_type, region);
        set_calling_convention(self.ctx, lambda.op_ref(), CallingConvention::Cps);
        (lambda.op_ref(), lambda.result(self.ctx))
    }

    /// Adapt a typed handler token without erasing a continuation. Only the
    /// source operation input crosses the exact `I -> anyref` boundary.
    fn build_exact_handler_token(
        &mut self,
        location: Location,
        factory: HandlerTokenFactory,
    ) -> (OpRef, ValueRef) {
        let HandlerTokenFactory {
            resume_body,
            completion,
            input_type,
            body_type,
            answer_type,
            dispatch_factory,
            dispatch_factory_args,
        } = factory;
        let anyref = self.anyref_type();
        let evidence_type = self.evidence_type();
        let answer_parent_type = self.parent_types(answer_type).reference;
        let exact_block =
            self.make_block(location, &[evidence_type, answer_parent_type, input_type]);
        let exact_args = self.ctx.block_args(exact_block).to_vec();
        let resume_block = self.make_block(location, &[evidence_type, answer_parent_type, anyref]);
        let resume_args = self.ctx.block_args(resume_block).to_vec();
        let next_block = self.make_block(location, &[body_type]);
        let next_value = self.ctx.block_args(next_block)[0];
        let next_tail = func::tail_call_indirect(
            self.ctx,
            location,
            completion,
            [resume_args[0], resume_args[1], next_value],
        );
        set_calling_convention(self.ctx, next_tail.op_ref(), CallingConvention::Cps);
        self.ctx.push_op(next_block, next_tail.op_ref());
        let next_region = self.single_block_region(location, next_block);
        let next_captures = ordered_external_values(self.ctx, next_region);
        let body_done_type = self.done_k_type(body_type);
        let next_done = closure::lambda(
            self.ctx,
            location,
            next_captures,
            body_done_type,
            next_region,
        );
        set_calling_convention(self.ctx, next_done.op_ref(), CallingConvention::Cps);
        self.ctx.push_op(resume_block, next_done.op_ref());
        let mut factory_args = vec![resume_args[1]];
        factory_args.extend(dispatch_factory_args);
        let body_dispatch_type = self.dispatch_type(body_type);
        let dispatch = func::call(
            self.ctx,
            location,
            factory_args,
            body_dispatch_type,
            dispatch_factory,
        );
        set_calling_convention(self.ctx, dispatch.op_ref(), CallingConvention::Direct);
        self.ctx.push_op(resume_block, dispatch.op_ref());
        let parent = self.pack_parent(
            resume_block,
            location,
            body_type,
            next_done.result(self.ctx),
            dispatch.result(self.ctx),
        );
        let resume_tail = func::tail_call_indirect(
            self.ctx,
            location,
            resume_body,
            [resume_args[0], parent, resume_args[2]],
        );
        set_calling_convention(self.ctx, resume_tail.op_ref(), CallingConvention::Cps);
        self.ctx.push_op(resume_block, resume_tail.op_ref());
        let resume_region = self.single_block_region(location, resume_block);
        let resume_captures = ordered_external_values(self.ctx, resume_region);
        let erased_resume_type = self.erased_resumption_type(answer_type);
        let resume = closure::lambda(
            self.ctx,
            location,
            resume_captures,
            erased_resume_type,
            resume_region,
        );
        set_calling_convention(self.ctx, resume.op_ref(), CallingConvention::Cps);
        self.ctx.push_op(exact_block, resume.op_ref());
        let erased = core::unrealized_conversion_cast(self.ctx, location, exact_args[2], anyref);
        self.ctx.push_op(exact_block, erased.op_ref());
        let tail = func::tail_call_indirect(
            self.ctx,
            location,
            resume.result(self.ctx),
            [exact_args[0], exact_args[1], erased.result(self.ctx)],
        );
        set_calling_convention(self.ctx, tail.op_ref(), CallingConvention::Cps);
        self.ctx.push_op(exact_block, tail.op_ref());
        let exact_region = self.single_block_region(location, exact_block);
        let exact_captures = ordered_external_values(self.ctx, exact_region);
        let token_type = self.resumption_type(input_type, answer_type);
        let token = closure::lambda(self.ctx, location, exact_captures, token_type, exact_region);
        set_calling_convention(self.ctx, token.op_ref(), CallingConvention::Cps);
        (token.op_ref(), token.result(self.ctx))
    }

    fn lower_handle(
        &mut self,
        source: OpRef,
        source_ops: &[OpRef],
        index: usize,
        block: BlockRef,
        mapping: &HashMap<ValueRef, ValueRef>,
        flow: &Flow,
    ) -> Result<(), TributeControlToCpsError> {
        if flow.convention != CallingConvention::Cps {
            return Err(TributeControlToCpsError::one(
                POST_CPS_BOUNDARY,
                Some(source),
                Some(self.ctx.op(source).location),
                "handle appears in a non-CPS callable",
            ));
        }
        let location = self.ctx.op(source).location;
        let handle_result_source = self.ctx.op_result(source, 0);
        let handle_answer_source = self.ctx.op_result_types(source)[0];
        let handle_answer = self.convert_type(handle_answer_source);
        let after_handle = self.build_suffix_continuation(
            source_ops,
            index + 1,
            handle_result_source,
            handle_answer_source,
            mapping,
            flow,
            location,
        )?;
        let after_handle_op = match self.ctx.value_def(after_handle) {
            trunk_ir::ValueDef::OpResult(op, _) => op,
            _ => unreachable!("handle continuation is produced by closure.lambda"),
        };
        self.ctx.push_op(block, after_handle_op);
        let i32_type = self.i32_type();
        let prompt_tag = effect::fresh_prompt_tag(self.ctx, location, i32_type);
        let prompt_tag_value = prompt_tag.result(self.ctx);
        self.ctx.push_op(block, prompt_tag.op_ref());

        // A normal handler-arm yield has answer type `A`, but general
        // operations inside the arm still cross the enclosing `O` dispatch
        // boundary. `after_done` is the sole immutable bridge between them.
        let (after_adapter_ops, after_done, after_dispatch) =
            self.build_cps_call_adapters(handle_answer, after_handle, flow, location)?;
        for op in after_adapter_ops {
            self.ctx.push_op(block, op);
        }

        let regions = self.ctx.op(source).regions.to_vec();
        let [body_source, completion_source, handlers_region] = regions.as_slice() else {
            unreachable!("pre-CPS validation checked handle regions");
        };
        let (body_source, completion_source, handlers_region) =
            (*body_source, *completion_source, *handlers_region);
        let handlers_block = self.ctx.region(handlers_region).blocks[0];
        let mut handler_arms = Vec::new();
        let source_handlers = self.ctx.block(handlers_block).ops.to_vec();
        let handler_flow = Flow {
            convention: CallingConvention::Cps,
            evidence: flow.evidence,
            resume_evidence: flow.resume_evidence,
            done: None,
            dispatch: None,
            boundary_result: handle_answer,
            resume_after: None,
            resume_done: None,
            void_done: None,
            preserve_scf_yield: false,
        };
        for handler in source_handlers {
            let arm = self.lower_handler_arm(handler, mapping, &handler_flow)?;
            self.ctx.push_op(block, arm.op);
            handler_arms.push(arm);
        }

        let mut ability_refs = Vec::new();
        for arm in &handler_arms {
            if !ability_refs.contains(&arm.ability_ref) {
                ability_refs.push(arm.ability_ref);
            }
        }
        let mut dispatchers = Vec::new();
        for ability_ref in ability_refs.iter().copied() {
            let ability_arms = handler_arms
                .iter()
                .filter(|arm| arm.ability_ref == ability_ref)
                .cloned()
                .collect::<Vec<_>>();
            let (tr_op, tr_value) = self.build_handler_dispatcher(location, &ability_arms, false);
            self.ctx.push_op(block, tr_op);
            dispatchers.push(tr_value);
            let (handler_op, handler_value) =
                self.build_handler_dispatcher(location, &ability_arms, true);
            self.ctx.push_op(block, handler_op);
            dispatchers.push(handler_value);
        }

        let evidence_type = self.evidence_type();
        let body_block = self.make_block(location, &[evidence_type]);
        let extended_evidence = self.ctx.block_args(body_block)[0];
        let mut body_mapping = mapping.clone();
        let body_flow_base = Flow {
            convention: CallingConvention::Cps,
            evidence: Some(extended_evidence),
            resume_evidence: Some(extended_evidence),
            done: None,
            dispatch: Some(after_dispatch),
            boundary_result: handle_answer,
            resume_after: None,
            resume_done: None,
            void_done: None,
            preserve_scf_yield: false,
        };
        let (completion_op, completion_k) = self.build_completion_continuation(
            completion_source,
            &body_mapping,
            &body_flow_base,
            location,
        )?;
        self.ctx.push_op(body_block, completion_op);
        let body_type = self.convert_type(
            self.ctx.value_ty(
                self.ctx
                    .block_args(self.ctx.region(completion_source).blocks[0])[0],
            ),
        );
        let after_parent = self.pack_parent(
            body_block,
            location,
            handle_answer,
            after_done,
            after_dispatch,
        );
        let (body_done_op, body_done) = self.build_done_adapter(
            body_type,
            completion_k,
            extended_evidence,
            after_parent,
            location,
        );
        self.ctx.push_op(body_block, body_done_op);
        let outer_evidence = self.current_evidence(source, flow)?;
        let local_dispatch_factory = self.build_local_cps_dispatch_factory(
            location,
            &handler_arms,
            body_type,
            handle_answer,
        );
        let mut local_factory_args =
            vec![after_parent, outer_evidence, prompt_tag_value, completion_k];
        local_factory_args.extend(handler_arms.iter().map(|arm| arm.value));
        let body_dispatch_type = self.dispatch_type(body_type);
        let local_dispatch_call = func::call(
            self.ctx,
            location,
            local_factory_args,
            body_dispatch_type,
            local_dispatch_factory,
        );
        set_calling_convention(
            self.ctx,
            local_dispatch_call.op_ref(),
            CallingConvention::Direct,
        );
        self.ctx.push_op(body_block, local_dispatch_call.op_ref());
        let body_flow = Flow {
            done: Some(body_done),
            dispatch: Some(local_dispatch_call.result(self.ctx)),
            boundary_result: body_type,
            ..body_flow_base
        };
        let source_body_block = self.ctx.region(body_source).blocks[0];
        self.convert_sequence(
            self.ctx.block(source_body_block).ops.to_vec(),
            0,
            body_block,
            &mut body_mapping,
            &body_flow,
        )?;
        let body_region = self.single_block_region(location, body_block);
        let current_evidence = self.current_evidence(source, flow)?;
        let dispatch = ability::handle_dispatch(
            self.ctx,
            location,
            current_evidence,
            prompt_tag_value,
            dispatchers,
            Attribute::List(ability_refs.into_iter().map(Attribute::Type).collect()),
            body_region,
        );
        self.ctx.push_op(block, dispatch.op_ref());
        Ok(())
    }

    fn convert_sequence(
        &mut self,
        source_ops: Vec<OpRef>,
        mut index: usize,
        block: BlockRef,
        mapping: &mut HashMap<ValueRef, ValueRef>,
        flow: &Flow,
    ) -> Result<(), TributeControlToCpsError> {
        while index < source_ops.len() {
            let source = source_ops[index];
            let location = self.ctx.op(source).location;
            let dialect = self.ctx.op(source).dialect;
            let name = self.ctx.op(source).name;
            if dialect == Symbol::new("scf") && name == Symbol::new("yield") {
                if flow.preserve_scf_yield {
                    let cloned = self.clone_plain_op(source, mapping)?;
                    self.ctx.push_op(block, cloned);
                    return Ok(());
                }
                let values = self.ctx.op_operands(source);
                if values.is_empty() {
                    if let Some(void_done) = flow.void_done {
                        let tail = func::tail_call_indirect(
                            self.ctx,
                            location,
                            void_done,
                            std::iter::empty::<ValueRef>(),
                        );
                        set_calling_convention(self.ctx, tail.op_ref(), CallingConvention::Cps);
                        self.ctx.push_op(block, tail.op_ref());
                        return Ok(());
                    }
                    self.emit_void_exit(block, location, flow)?;
                    return Ok(());
                }
                if values.len() != 1 {
                    return Err(TributeControlToCpsError::one(
                        POST_CPS_BOUNDARY,
                        Some(source),
                        Some(location),
                        "CPS structured exit requires exactly one scf.yield value",
                    ));
                }
                let value = mapping.get(&values[0]).copied().unwrap_or(values[0]);
                self.emit_exit(block, location, value, flow)?;
                return Ok(());
            }
            if dialect == Symbol::new("scf")
                && name == Symbol::new("switch")
                && self.contains_tribute_control(source)
            {
                self.lower_structured_switch(source, &source_ops, index, block, mapping, flow)?;
                return Ok(());
            }
            if dialect == Symbol::new("scf")
                && name == Symbol::new("if")
                && self.contains_tribute_control(source)
            {
                self.lower_structured_if(source, &source_ops, index, block, mapping, flow)?;
                return Ok(());
            }
            // Source case lowering can use an unrealized cast into the
            // logical bottom solely to join an `op -> Never` arm with a
            // normal arm. At a proven CPS structured exit, that cast carries
            // no runtime value: route the represented input directly to the
            // active `Flow<R>` instead. Any other Never cast remains for the
            // named boundary verifier to reject.
            if flow.convention == CallingConvention::Cps
                && dialect == Symbol::new("core")
                && name == Symbol::new("unrealized_conversion_cast")
            {
                let results = self.ctx.op_results(source);
                let operands = self.ctx.op_operands(source);
                if let ([result], [input]) = (results, operands)
                    && type_is(self.ctx, self.ctx.value_ty(*result), "core", "never")
                    && !self.ctx.uses(*result).is_empty()
                    && self.ctx.uses(*result).iter().all(|use_| {
                        self.ctx.op(use_.user).dialect == Symbol::new("scf")
                            && self.ctx.op(use_.user).name == Symbol::new("yield")
                    })
                {
                    let input = mapping.get(input).copied().unwrap_or(*input);
                    if self.ctx.value_ty(input) == flow.boundary_result {
                        mapping.insert(*result, input);
                        index += 1;
                        continue;
                    }
                }
            }
            if dialect != Symbol::new("tribute_control") {
                let cloned = self.clone_plain_op(source, mapping)?;
                self.ctx.push_op(block, cloned);
                index += 1;
                continue;
            }

            match name.with_str(|value| value.to_owned()).as_str() {
                "unreachable" => {
                    let unreachable = func::unreachable(self.ctx, location);
                    self.ctx.push_op(block, unreachable.op_ref());
                    return Ok(());
                }
                "return" | "yield" => {
                    let value = self.ctx.op_operands(source)[0];
                    let value = mapping.get(&value).copied().unwrap_or(value);
                    self.emit_exit(block, location, value, flow)?;
                    return Ok(());
                }
                "lambda" => {
                    let lambda = self.lower_lambda(source, mapping)?;
                    self.ctx.push_op(block, lambda);
                    mapping.insert(self.ctx.op_result(source, 0), self.ctx.op_result(lambda, 0));
                    index += 1;
                }
                "func_ref" => {
                    let (ops, value) = self.lower_func_ref(source)?;
                    for op in ops {
                        self.ctx.push_op(block, op);
                    }
                    mapping.insert(self.ctx.op_result(source, 0), value);
                    index += 1;
                }
                "call" => {
                    let target_symbol = self
                        .ctx
                        .op(source)
                        .attributes
                        .get_symbol("callee")
                        .expect("pre-CPS validation checked direct callee");
                    let target = self
                        .current_direct_call_target(source, target_symbol)
                        .expect("pre-CPS validation resolved direct callee in this module");
                    if target.convention == CallingConvention::Cps {
                        if flow.convention != CallingConvention::Cps {
                            return Err(TributeControlToCpsError::one(
                                POST_CPS_BOUNDARY,
                                Some(source),
                                Some(location),
                                format!(
                                    "a {:?} callable cannot call CPS target @{}",
                                    flow.convention, target_symbol
                                ),
                            ));
                        }
                        let old_result = self.ctx.op_result(source, 0);
                        let result_type = self.ctx.op_result_types(source)[0];
                        let completion = self.build_suffix_continuation(
                            &source_ops,
                            index + 1,
                            old_result,
                            result_type,
                            mapping,
                            flow,
                            location,
                        )?;
                        let completion_op = match self.ctx.value_def(completion) {
                            trunk_ir::ValueDef::OpResult(op, _) => op,
                            _ => unreachable!("continuation is produced by closure.lambda"),
                        };
                        self.ctx.push_op(block, completion_op);
                        let value_type = self.convert_type(result_type);
                        let (adapter_ops, done, dispatch) =
                            self.build_cps_call_adapters(value_type, completion, flow, location)?;
                        for op in adapter_ops {
                            self.ctx.push_op(block, op);
                        }
                        let mut args =
                            vec![self.current_resume_evidence(source, flow)?, done, dispatch];
                        args.extend(
                            self.ctx
                                .op_operands(source)
                                .iter()
                                .map(|arg| mapping.get(arg).copied().unwrap_or(*arg)),
                        );
                        let tail = func::tail_call(self.ctx, location, args, target_symbol);
                        set_calling_convention(self.ctx, tail.op_ref(), CallingConvention::Cps);
                        self.ctx.push_op(block, tail.op_ref());
                        return Ok(());
                    }
                    let call = self.lower_direct_call(source, &target, mapping, flow)?;
                    self.ctx.push_op(block, call.op_ref());
                    mapping.insert(self.ctx.op_result(source, 0), call.result(self.ctx));
                    index += 1;
                }
                "call_indirect" => {
                    let source_callee = self.ctx.op_operands(source)[0];
                    let logical_type = self.ctx.value_ty(source_callee);
                    let convention = tribute_control::callable_convention(self.ctx, logical_type)
                        .map(convert_convention)
                        .expect("pre-CPS validation checked indirect callee convention");
                    let callee = mapping
                        .get(&source_callee)
                        .copied()
                        .unwrap_or(source_callee);
                    let source_args = self.ctx.op_operands(source)[1..].to_vec();
                    if convention == CallingConvention::Cps {
                        if flow.convention != CallingConvention::Cps {
                            return Err(TributeControlToCpsError::one(
                                POST_CPS_BOUNDARY,
                                Some(source),
                                Some(location),
                                "a non-CPS callable cannot make a CPS indirect call",
                            ));
                        }
                        let old_result = self.ctx.op_result(source, 0);
                        let result_type = self.ctx.op_result_types(source)[0];
                        let completion = self.build_suffix_continuation(
                            &source_ops,
                            index + 1,
                            old_result,
                            result_type,
                            mapping,
                            flow,
                            location,
                        )?;
                        let completion_op = match self.ctx.value_def(completion) {
                            trunk_ir::ValueDef::OpResult(op, _) => op,
                            _ => unreachable!("continuation is produced by closure.lambda"),
                        };
                        self.ctx.push_op(block, completion_op);
                        let value_type = self.convert_type(result_type);
                        let (adapter_ops, done, dispatch) =
                            self.build_cps_call_adapters(value_type, completion, flow, location)?;
                        for op in adapter_ops {
                            self.ctx.push_op(block, op);
                        }
                        let mut args =
                            vec![self.current_resume_evidence(source, flow)?, done, dispatch];
                        args.extend(
                            source_args
                                .iter()
                                .map(|arg| mapping.get(arg).copied().unwrap_or(*arg)),
                        );
                        let tail = func::tail_call_indirect(self.ctx, location, callee, args);
                        set_calling_convention(self.ctx, tail.op_ref(), CallingConvention::Cps);
                        self.ctx.push_op(block, tail.op_ref());
                        return Ok(());
                    }
                    let mut args = Vec::new();
                    if convention.needs_evidence() {
                        args.push(self.current_evidence(source, flow)?);
                    }
                    args.extend(
                        source_args
                            .iter()
                            .map(|arg| mapping.get(arg).copied().unwrap_or(*arg)),
                    );
                    let result_type = self.convert_type(self.ctx.op_result_types(source)[0]);
                    let call = func::call_indirect(self.ctx, location, callee, args, result_type);
                    set_calling_convention(self.ctx, call.op_ref(), convention);
                    self.ctx.push_op(block, call.op_ref());
                    mapping.insert(self.ctx.op_result(source, 0), call.result(self.ctx));
                    index += 1;
                }
                "perform" => {
                    let kind = self
                        .ctx
                        .op(source)
                        .attributes
                        .get_symbol("operation_kind")
                        .expect("pre-CPS validation checked operation kind");
                    if kind == Symbol::new("fn") {
                        let evidence = self.current_evidence(source, flow)?;
                        let _ = evidence;
                        let args: Vec<_> = self
                            .ctx
                            .op_operands(source)
                            .iter()
                            .map(|arg| mapping.get(arg).copied().unwrap_or(*arg))
                            .collect();
                        let result_type = self.convert_type(self.ctx.op_result_types(source)[0]);
                        let ability_ref = self
                            .ctx
                            .op(source)
                            .attributes
                            .get_type("ability_ref")
                            .expect("pre-CPS validation checked perform ability");
                        let op_name = self
                            .ctx
                            .op(source)
                            .attributes
                            .get_symbol("op_name")
                            .expect("pre-CPS validation checked perform operation");
                        let call = ability::call(
                            self.ctx,
                            location,
                            args,
                            result_type,
                            ability_ref,
                            op_name,
                        );
                        self.ctx.push_op(block, call.op_ref());
                        mapping.insert(self.ctx.op_result(source, 0), call.result(self.ctx));
                        index += 1;
                    } else {
                        self.lower_general_perform(
                            source,
                            &source_ops,
                            index,
                            block,
                            mapping,
                            flow,
                        )?;
                        return Ok(());
                    }
                }
                "resume" => {
                    self.lower_resume(source, &source_ops, index, block, mapping, flow)?;
                    return Ok(());
                }
                "handle" => {
                    self.lower_handle(source, &source_ops, index, block, mapping, flow)?;
                    return Ok(());
                }
                other => {
                    return Err(TributeControlToCpsError::one(
                        POST_CPS_BOUNDARY,
                        Some(source),
                        Some(location),
                        format!("unsupported tribute_control operation '{other}'"),
                    ));
                }
            }
        }
        Ok(())
    }

    fn convert_func(&mut self, source: OpRef) -> Result<OpRef, TributeControlToCpsError> {
        let location = self.ctx.op(source).location;
        let symbol = self
            .ctx
            .op(source)
            .attributes
            .get_symbol("sym_name")
            .expect("pre-CPS validation checked function symbol");
        let logical_type = self
            .ctx
            .op(source)
            .attributes
            .get_type("type")
            .expect("pre-CPS validation checked function type");
        let info = self
            .current_func(symbol)
            .expect("validated function is present in its module callable graph");
        let physical_type = self.physical_function_type(logical_type);
        if self.ctx.op(source).regions.is_empty() {
            let mut builder =
                OperationDataBuilder::new(location, Symbol::new("func"), Symbol::new("func"))
                    .attr("sym_name", Attribute::Symbol(symbol))
                    .attr("type", Attribute::Type(physical_type));
            for (key, value) in self.convert_attrs(&self.ctx.op(source).attributes.clone()) {
                if key != Symbol::new("sym_name") && key != Symbol::new("type") {
                    builder = builder.attr(key, value);
                }
            }
            let data = builder.build(self.ctx);
            let declaration = self.ctx.create_op(data);
            set_calling_convention(self.ctx, declaration, info.convention);
            return Ok(declaration);
        }

        let source_region = self.ctx.op(source).regions[0];
        let source_block = self.ctx.region(source_region).blocks[0];
        let source_result = self.convert_type(info.source_result);
        let source_params: Vec<_> = info
            .source_params
            .iter()
            .copied()
            .map(|ty| self.convert_type(ty))
            .collect();
        let abi = CallableAbi::new(info.convention, source_params, source_result);
        let evidence_ty = self.evidence_type();
        let done_k_ty = self.done_k_type(source_result);
        let dispatch_ty = self.dispatch_type(source_result);
        let params = abi.lowered_params_with_dispatch(evidence_ty, done_k_ty, dispatch_ty);
        let block = self.make_block(location, &params);
        let mut mapping = HashMap::new();
        for (old, new) in self.ctx.block_args(source_block).to_vec().into_iter().zip(
            self.ctx.block_args(block)[abi.source_param_offset_with_dispatch()..]
                .iter()
                .copied(),
        ) {
            mapping.insert(old, new);
        }
        let evidence = info
            .convention
            .needs_evidence()
            .then(|| self.ctx.block_args(block)[0]);
        let done_index = usize::from(info.convention.needs_evidence());
        let done = info
            .convention
            .needs_done_k()
            .then(|| self.ctx.block_args(block)[done_index]);
        let dispatch = info
            .convention
            .needs_done_k()
            .then(|| self.ctx.block_args(block)[done_index + 1]);
        let flow = Flow {
            convention: info.convention,
            evidence,
            resume_evidence: evidence,
            done,
            dispatch,
            boundary_result: source_result,
            resume_after: None,
            resume_done: None,
            void_done: None,
            preserve_scf_yield: false,
        };
        self.convert_sequence(
            self.ctx.block(source_block).ops.to_vec(),
            0,
            block,
            &mut mapping,
            &flow,
        )?;
        let region = self.single_block_region(location, block);
        let function = func::func(self.ctx, location, symbol, physical_type, region);
        self.copy_extra_attrs(
            source,
            function.op_ref(),
            &["sym_name", "type", CALLING_CONVENTION_ATTR],
        );
        set_calling_convention(self.ctx, function.op_ref(), info.convention);
        Ok(function.op_ref())
    }
}

fn collect_defined_values(ctx: &IrContext, region: RegionRef, defined: &mut HashSet<ValueRef>) {
    for block in ctx.region(region).blocks.iter().copied() {
        defined.extend(ctx.block_args(block).iter().copied());
        for op in ctx.block(block).ops.iter().copied() {
            defined.extend(ctx.op_results(op).iter().copied());
            for nested in ctx.op(op).regions.iter().copied() {
                collect_defined_values(ctx, nested, defined);
            }
        }
    }
}

fn collect_external_in_order(
    ctx: &IrContext,
    region: RegionRef,
    defined: &HashSet<ValueRef>,
    seen: &mut HashSet<ValueRef>,
    external: &mut Vec<ValueRef>,
) {
    for block in ctx.region(region).blocks.iter().copied() {
        for op in ctx.block(block).ops.iter().copied() {
            for operand in ctx.op_operands(op).iter().copied() {
                if !defined.contains(&operand) && seen.insert(operand) {
                    external.push(operand);
                }
            }
            for nested in ctx.op(op).regions.iter().copied() {
                collect_external_in_order(ctx, nested, defined, seen, external);
            }
        }
    }
}

fn ordered_external_values(ctx: &IrContext, region: RegionRef) -> Vec<ValueRef> {
    let mut defined = HashSet::new();
    collect_defined_values(ctx, region, &mut defined);
    let mut seen = HashSet::new();
    let mut external = Vec::new();
    collect_external_in_order(ctx, region, &defined, &mut seen, &mut external);
    external
}

fn collect_callable_graph(
    ctx: &IrContext,
    module: Module,
) -> HashMap<OpRef, HashMap<Symbol, CallableInfo>> {
    fn collect_scope(
        ctx: &IrContext,
        region: RegionRef,
        funcs: &mut HashMap<Symbol, CallableInfo>,
        nested_modules: &mut Vec<OpRef>,
    ) {
        for block in ctx.region(region).blocks.iter().copied() {
            for op in ctx.block(block).ops.iter().copied() {
                let data = ctx.op(op);
                if data.dialect == Symbol::new("core") && data.name == Symbol::new("module") {
                    nested_modules.push(op);
                    continue;
                }
                if tribute_control::Func::matches(ctx, op) {
                    let symbol = data
                        .attributes
                        .get_symbol("sym_name")
                        .expect("pre-CPS validation checked function symbol");
                    let logical_type = data
                        .attributes
                        .get_type("type")
                        .expect("pre-CPS validation checked function type");
                    let callable = tribute_control::Callable::from_type_ref(ctx, logical_type)
                        .expect("pre-CPS validation checked callable type");
                    let convention = tribute_control::callable_convention(ctx, logical_type)
                        .expect("pre-CPS validation checked callable convention");
                    funcs.insert(
                        symbol,
                        CallableInfo {
                            symbol,
                            convention: convert_convention(convention),
                            source_result: callable.result(ctx),
                            source_params: callable.params(ctx).to_vec(),
                        },
                    );
                }
                for nested in data.regions.iter().copied() {
                    collect_scope(ctx, nested, funcs, nested_modules);
                }
            }
        }
    }

    fn visit_module(
        ctx: &IrContext,
        module_op: OpRef,
        funcs_by_module: &mut HashMap<OpRef, HashMap<Symbol, CallableInfo>>,
    ) {
        let mut funcs = HashMap::new();
        let mut nested_modules = Vec::new();
        for region in ctx.op(module_op).regions.iter().copied() {
            collect_scope(ctx, region, &mut funcs, &mut nested_modules);
        }
        funcs_by_module.insert(module_op, funcs);
        for nested in nested_modules {
            visit_module(ctx, nested, funcs_by_module);
        }
    }

    let mut funcs_by_module = HashMap::new();
    visit_module(ctx, module.op(), &mut funcs_by_module);
    funcs_by_module
}

fn verify_candidate_or_restore_aliases(
    ctx: &mut IrContext,
    candidate: Module,
    source_aliases: &[(Symbol, TypeRef)],
    generated_aliases: &[(Symbol, TypeRef)],
) -> Result<(), TributeControlToCpsError> {
    if let Err(error) = verify_tribute_control_post_cps(ctx, candidate) {
        for (name, _) in generated_aliases {
            ctx.remove_type_alias(*name);
        }
        for (name, ty) in source_aliases {
            ctx.register_type_alias(*name, *ty);
        }
        ctx.remove_op(candidate.op());
        return Err(error);
    }
    Ok(())
}

/// Atomically convert the complete logical callable/control graph.
///
/// The existing module region is not detached until a separately built module
/// has passed the post-CPS operation and recursive type boundary.
pub fn tribute_control_to_cps(
    ctx: &mut IrContext,
    module: Module,
    declarations: &[tribute_control::OperationDeclaration],
) -> Result<(), TributeControlToCpsError> {
    verify_tribute_control_pre_cps(ctx, module, declarations)?;
    let funcs_by_module = collect_callable_graph(ctx, module);
    let source_region = module.body(ctx).ok_or_else(|| {
        TributeControlToCpsError::one(
            PRE_CPS_BOUNDARY,
            Some(module.op()),
            Some(ctx.op(module.op()).location),
            "core.module has no body region",
        )
    })?;
    let source_blocks = ctx.region(source_region).blocks.to_vec();
    if source_blocks.len() != 1 {
        return Err(TributeControlToCpsError::one(
            PRE_CPS_BOUNDARY,
            Some(module.op()),
            Some(ctx.op(module.op()).location),
            "tribute_control_to_cps currently requires a single module block",
        ));
    }
    let module_location = ctx.op(module.op()).location;
    let source_aliases = ctx.type_aliases().to_vec();
    let mut converted_aliases = Vec::with_capacity(source_aliases.len());
    let generated_parent_aliases;
    let new_block = ctx.create_block(BlockData {
        location: ctx.block(source_blocks[0]).location,
        args: vec![],
        ops: Default::default(),
        parent_region: None,
    });
    {
        let mut converter = Converter::new(ctx, new_block, funcs_by_module, module.op());
        let mut mapping = HashMap::new();
        let source_ops = converter.ctx.block(source_blocks[0]).ops.to_vec();
        for source in source_ops {
            if tribute_control::Func::matches(converter.ctx, source) {
                let function = converter.convert_func(source)?;
                converter.ctx.push_op(new_block, function);
            } else if converter.ctx.op(source).dialect == Symbol::new("tribute_control") {
                return Err(TributeControlToCpsError::one(
                    PRE_CPS_BOUNDARY,
                    Some(source),
                    Some(converter.ctx.op(source).location),
                    "only tribute_control.func may appear directly in a module block",
                ));
            } else {
                let cloned = converter.clone_plain_op(source, &mut mapping)?;
                converter.ctx.push_op(new_block, cloned);
            }
        }
        converted_aliases.extend(
            source_aliases
                .iter()
                .map(|(name, ty)| (*name, converter.convert_type(*ty))),
        );
        generated_parent_aliases = converter.parent_layout_aliases.clone();
    }
    converted_aliases.extend(generated_parent_aliases.iter().copied());
    let new_region = ctx.create_region(RegionData {
        location: ctx.region(source_region).location,
        blocks: trunk_ir::smallvec::smallvec![new_block],
        parent_op: None,
    });
    let temp_symbol = Symbol::new("__tribute_control_to_cps_candidate");
    let temp_module = core::module(ctx, module_location, temp_symbol, new_region);
    let candidate: Module = temp_module.into();
    for (name, ty) in &converted_aliases {
        ctx.register_type_alias(*name, *ty);
    }
    verify_candidate_or_restore_aliases(
        ctx,
        candidate,
        &source_aliases,
        &generated_parent_aliases,
    )?;

    ctx.detach_region(new_region);
    ctx.remove_op(candidate.op());
    ctx.detach_region(source_region);
    ctx.op_mut(module.op()).regions.push(new_region);
    ctx.region_mut(new_region).parent_op = Some(module.op());
    if let Err(error) = verify_tribute_control_post_cps(ctx, module) {
        ctx.detach_region(new_region);
        ctx.op_mut(module.op()).regions.push(source_region);
        ctx.region_mut(source_region).parent_op = Some(module.op());
        for (name, _) in &generated_parent_aliases {
            ctx.remove_type_alias(*name);
        }
        for (name, ty) in &source_aliases {
            ctx.register_type_alias(*name, *ty);
        }
        return Err(error);
    }
    Ok(())
}

/// Pass-manager wrapper carrying the verified source operation declarations.
pub struct TributeControlToCps {
    declarations: Vec<tribute_control::OperationDeclaration>,
}

impl TributeControlToCps {
    pub fn new(
        declarations: impl IntoIterator<Item = tribute_control::OperationDeclaration>,
    ) -> Self {
        Self {
            declarations: declarations.into_iter().collect(),
        }
    }
}

impl Pass for TributeControlToCps {
    type Target = core::Module;

    fn name(&self) -> &'static str {
        "tribute-control-to-cps"
    }

    fn run(&mut self, ctx: &mut IrContext, target: core::Module) -> PassRunResult {
        tribute_control_to_cps(ctx, target.into(), &self.declarations)
            .map_err(|error| Box::new(error) as _)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;

    fn parse(input: &str) -> (IrContext, Module) {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);
        (ctx, module)
    }

    #[test]
    fn textual_callable_graph_converts_and_reparses() {
        let input = r#"core.module @test {
  tribute_control.func @decl(%left: core.i32, %right: core.i32) -> core.i32 convention(direct)
  tribute_control.func @identity(%value: core.i32) -> core.i32 convention(evidence_direct) {
    tribute_control.return %value
  }
  tribute_control.func @cps_identity(%value: core.i32) -> core.i32 convention(cps) {
    tribute_control.return %value
  }
  tribute_control.func @outer(%first: core.i32) -> core.i32 convention(direct) {
    %captured = tribute_control.lambda(%value: core.i32) -> core.i32 convention(direct) captures [%first] {
      tribute_control.return %first
    }
    tribute_control.return %first
  }
}"#;
        let (mut ctx, module) = parse(input);
        tribute_control_to_cps(&mut ctx, module, &[]).unwrap();
        verify_tribute_control_post_cps(&ctx, module).unwrap();
        let printed = print_module(&ctx, module.op());
        assert!(!printed.contains("tribute_control.func "));
        assert!(!printed.contains("tribute_control.func_ref "));
        assert!(!printed.contains("tribute_control.call_indirect "));
        assert!(printed.contains("func.func @decl"));
        assert!(printed.contains("closure.lambda"));
        assert!(printed.contains("func.tail_call_indirect"));
        assert!(printed.contains("tribute.calling_convention = 2"));

        let mut reparsed = IrContext::new();
        let reparsed_module = parse_test_module(&mut reparsed, &printed);
        verify_tribute_control_post_cps(&reparsed, reparsed_module).unwrap();
    }

    #[test]
    fn pass_wrapper_runs_the_verified_conversion() {
        let input = r#"core.module @test {
  tribute_control.func @identity(%value: core.i32) -> core.i32 convention(direct) {
    tribute_control.return %value
  }
}"#;
        let (mut ctx, module) = parse(input);
        let target = core::Module::from_op(&ctx, module.op()).unwrap();
        let mut pass = TributeControlToCps::new([]);
        assert_eq!(pass.name(), "tribute-control-to-cps");
        pass.run(&mut ctx, target).unwrap();
        verify_tribute_control_post_cps(&ctx, module).unwrap();
    }

    #[test]
    fn textual_direct_evidence_and_cps_transfers_preserve_exact_abis() {
        let input = r#"core.module @test {
  !direct = tribute_control.callable(core.i32, core.i32) {tribute.calling_convention = 0}
  !evidence = tribute_control.callable(core.i32, core.i32) {tribute.calling_convention = 1}
  !cps = tribute_control.callable(core.i32, core.i32) {tribute.calling_convention = 2}
  tribute_control.func @direct(%value: core.i32) -> core.i32 convention(direct) {
    tribute_control.return %value
  }
  tribute_control.func @evidence(%value: core.i32) -> core.i32 convention(evidence_direct) {
    tribute_control.return %value
  }
  tribute_control.func @cps(%value: core.i32) -> core.i32 convention(cps) {
    tribute_control.return %value
  }
  tribute_control.func @exercise(%value: core.i32) -> core.i32 convention(cps) {
    %direct_result = tribute_control.call %value {callee = @direct} : core.i32
    %evidence_result = tribute_control.call %direct_result {callee = @evidence} : core.i32
    %direct_ref = tribute_control.func_ref {func_ref = @direct} : !direct
    %direct_indirect = tribute_control.call_indirect %direct_ref, %evidence_result : core.i32
    %evidence_ref = tribute_control.func_ref {func_ref = @evidence} : !evidence
    %evidence_indirect = tribute_control.call_indirect %evidence_ref, %direct_indirect : core.i32
    %cps_ref = tribute_control.func_ref {func_ref = @cps} : !cps
    %cps_indirect = tribute_control.call_indirect %cps_ref, %evidence_indirect : core.i32
    tribute_control.return %cps_indirect
  }
  tribute_control.func @known_cps(%value: core.i32) -> core.i32 convention(cps) {
    %result = tribute_control.call %value {callee = @cps} : core.i32
    tribute_control.return %result
  }
}"#;
        let (mut ctx, module) = parse(input);
        tribute_control_to_cps(&mut ctx, module, &[]).unwrap();
        let printed = print_module(&ctx, module.op());
        assert!(printed.contains("func.call "));
        assert!(printed.contains("func.call_indirect"));
        assert!(printed.contains("func.tail_call "));
        assert!(printed.contains("func.tail_call_indirect"));
        assert!(printed.contains("tribute.calling_convention = 0"));
        assert!(printed.contains("tribute.calling_convention = 1"));
        assert!(printed.contains("tribute.calling_convention = 2"));
        assert!(!printed.contains("tribute_control."), "{printed}");

        let mut reparsed = IrContext::new();
        let reparsed_module = parse_test_module(&mut reparsed, &printed);
        verify_tribute_control_post_cps(&reparsed, reparsed_module).unwrap();
    }

    #[test]
    fn nested_textual_module_converts_its_callable_graph_atomically() {
        let input = r#"core.module @outer {
  core.module @inner {
    tribute_control.func @nested(%value: core.i32) -> core.i32 convention(cps) {
      tribute_control.return %value
    }
  }
}"#;
        let (mut ctx, module) = parse(input);
        tribute_control_to_cps(&mut ctx, module, &[]).unwrap();
        let printed = print_module(&ctx, module.op());
        assert!(printed.contains("core.module @inner"));
        assert!(printed.contains("func.func @nested"));
        assert!(printed.contains("func.tail_call_indirect"));
        assert!(!printed.contains("tribute_control."));

        let mut reparsed = IrContext::new();
        let reparsed_module = parse_test_module(&mut reparsed, &printed);
        verify_tribute_control_post_cps(&reparsed, reparsed_module).unwrap();
    }

    #[test]
    fn nested_modules_resolve_same_named_callables_in_their_own_scope() {
        let input = r#"core.module @outer {
  tribute_control.func @same(%value: core.i32) -> core.i32 convention(direct) {
    tribute_control.return %value
  }
  tribute_control.func @outer_call(%value: core.i32) -> core.i32 convention(direct) {
    %result = tribute_control.call %value {callee = @same} : core.i32
    tribute_control.return %result
  }
  core.module @inner {
    tribute_control.func @same(%value: core.i1) -> core.i1 convention(evidence_direct) {
      tribute_control.return %value
    }
    tribute_control.func @inner_call(%value: core.i1) -> core.i1 convention(evidence_direct) {
      %result = tribute_control.call %value {callee = @same} : core.i1
      tribute_control.return %result
    }
  }
}"#;
        let (mut ctx, module) = parse(input);
        tribute_control_to_cps(&mut ctx, module, &[]).unwrap();
        verify_tribute_control_post_cps(&ctx, module).unwrap();
        let printed = print_module(&ctx, module.op());
        assert_eq!(printed.matches("func.func @same").count(), 2, "{printed}");
        assert!(printed.contains("func.func @outer_call"), "{printed}");
        assert!(printed.contains("func.func @inner_call"), "{printed}");
        assert!(printed.contains("tribute.calling_convention = 0"));
        assert!(printed.contains("tribute.calling_convention = 1"));

        let mut reparsed = IrContext::new();
        let reparsed_module = parse_test_module(&mut reparsed, &printed);
        verify_tribute_control_post_cps(&reparsed, reparsed_module).unwrap();
    }

    #[test]
    fn textual_nested_attribute_types_convert_atomically() {
        let input = r#"core.module @test {
  !callback = tribute_control.callable(core.i32, core.i32) {tribute.calling_convention = 0}
  !record = adt.struct() {fields = [[@callback, !callback]], name = @CallbackRecord}
  tribute_control.func @identity(%value: !record) -> !record convention(direct) {
    tribute_control.return %value
  }
}"#;
        let (mut ctx, module) = parse(input);
        tribute_control_to_cps(&mut ctx, module, &[]).unwrap();
        let printed = print_module(&ctx, module.op());
        assert!(printed.contains("name = @CallbackRecord"));
        assert!(printed.contains("closure.closure(core.func(core.i32, core.i32))"));
        assert!(!printed.contains("tribute_control."));

        let mut reparsed = IrContext::new();
        let reparsed_module = parse_test_module(&mut reparsed, &printed);
        verify_tribute_control_post_cps(&reparsed, reparsed_module).unwrap();
    }

    #[test]
    fn malformed_pre_boundary_is_atomic() {
        let input = r#"core.module @test {
  tribute_control.func @broken(%value: core.i32) -> core.i32 convention(direct) {
    %illegal = func.call %value {callee = @broken} : core.i32
    tribute_control.return %illegal
  }
}"#;
        let (mut ctx, module) = parse(input);
        let before = print_module(&ctx, module.op());
        let error = tribute_control_to_cps(&mut ctx, module, &[]).unwrap_err();
        assert_eq!(error.boundary, PRE_CPS_BOUNDARY);
        assert!(error.to_string().contains("func.call"));
        assert_eq!(print_module(&ctx, module.op()), before);
    }

    #[test]
    fn malformed_lookup_inputs_fail_before_conversion_and_remain_unchanged() {
        let malformed = [
            (
                r#"core.module @test {
  tribute_control.func @broken(%value: core.i32) -> core.i32 convention(direct) {
    %result = tribute_control.call %value : core.i32
    tribute_control.return %result
  }
}"#,
                "requires 'callee' attribute",
            ),
            (
                r#"core.module @test {
  tribute_control.func @broken(%value: core.i32) -> core.i32 convention(direct) {
    %result = tribute_control.call %value {callee = @missing} : core.i32
    tribute_control.return %result
  }
}"#,
                "unresolved callee @missing",
            ),
            (
                r#"core.module @test {
  tribute_control.func @broken(%value: core.i32) -> core.i32 convention(direct) {
    %result = tribute_control.call_indirect %value, %value : core.i32
    tribute_control.return %result
  }
}"#,
                "callee operand must have tribute_control.callable type",
            ),
            (
                r#"core.module @test {
  tribute_control.func @broken(%value: core.i32) -> core.i32 convention(cps) {
    %result = tribute_control.perform %value : core.i32
    tribute_control.return %result
  }
}"#,
                "requires 'ability_ref' attribute",
            ),
            (
                r#"core.module @test {
  tribute_control.func @broken(%value: core.i32) -> core.i32 convention(cps) {
    %result = tribute_control.handle : core.i32
    tribute_control.return %result
  }
}"#,
                "expects 3 region(s)",
            ),
        ];
        for (input, expected) in malformed {
            let (mut ctx, module) = parse(input);
            let before = print_module(&ctx, module.op());
            let error = tribute_control_to_cps(&mut ctx, module, &[]).unwrap_err();
            assert_eq!(error.boundary, PRE_CPS_BOUNDARY);
            assert!(error.to_string().contains(expected), "{error}");
            assert_eq!(print_module(&ctx, module.op()), before);
        }
    }

    #[test]
    fn source_successors_fail_before_conversion_and_leave_ir_unchanged() {
        let input = r#"core.module @test {
  ^entry:
    scf.br [^exit]
  ^exit:
}"#;
        let (mut ctx, module) = parse(input);
        let before = print_module(&ctx, module.op());
        let error = tribute_control_to_cps(&mut ctx, module, &[]).unwrap_err();
        assert_eq!(error.boundary, PRE_CPS_BOUNDARY);
        assert!(
            error.to_string().contains("forbids block successors"),
            "{error}"
        );
        assert_eq!(print_module(&ctx, module.op()), before);
    }

    #[test]
    fn malformed_switch_case_fails_before_conversion_and_is_atomic() {
        let input = r#"core.module @test {
  tribute_control.func @broken(%value: core.i32) -> core.i32 convention(cps) {
    scf.switch %value {
      scf.case {
        %identity = tribute_control.lambda(%nested: core.i32) -> core.i32 convention(direct) captures [] {
          tribute_control.return %nested
        }
        scf.yield
      }
    }
    tribute_control.return %value
  }
}"#;
        let (mut ctx, module) = parse(input);
        let before = print_module(&ctx, module.op());
        let error = tribute_control_to_cps(&mut ctx, module, &[]).unwrap_err();
        assert_eq!(error.boundary, PRE_CPS_BOUNDARY);
        assert!(
            error.to_string().contains("scf.case requires a value"),
            "{error}"
        );
        assert_eq!(print_module(&ctx, module.op()), before);
    }

    #[test]
    fn source_shape_validation_rejects_every_malformed_switch_component() {
        let malformed = [
            (
                r#"core.module @test {
  %value = arith.const {value = 0} : core.i32
  scf.switch {
    scf.default {
      scf.yield
    }
  }
}"#,
                "one discriminant, no results, and one body region",
            ),
            (
                r#"core.module @test {
  %value = arith.const {value = 0} : core.i32
  %result = scf.switch %value : core.i32 {
    scf.default {
      scf.yield
    }
  }
}"#,
                "one discriminant, no results, and one body region",
            ),
            (
                r#"core.module @test {
  %value = arith.const {value = 0} : core.i32
  scf.switch %value {
    ^first:
      scf.default {
        scf.yield
      }
    ^second:
  }
}"#,
                "body region requires exactly one block",
            ),
            (
                r#"core.module @test {
  %value = arith.const {value = 0} : core.i32
  scf.switch %value {
    %invalid = arith.const {value = 1} : core.i32
  }
}"#,
                "body may contain only scf.case and scf.default",
            ),
            (
                r#"core.module @test {
  %value = arith.const {value = 0} : core.i32
  scf.switch %value {
    scf.case {value = 0}
  }
}"#,
                "arm requires exactly one region",
            ),
            (
                r#"core.module @test {
  %value = arith.const {value = 0} : core.i32
  scf.switch %value {
    scf.default {
      ^first:
        scf.yield
      ^second:
    }
  }
}"#,
                "arm region requires exactly one block",
            ),
        ];

        for (input, expected) in malformed {
            let (ctx, module) = parse(input);
            let failures = verify_source_conversion_shapes(&ctx, module);
            assert!(
                failures
                    .iter()
                    .any(|failure| failure.message.contains(expected)),
                "{failures:?}"
            );
        }
    }

    #[test]
    fn named_boundaries_report_local_and_core_validation_errors() {
        let local_input = r#"core.module @test {
  tribute_control.func @broken(%value: core.i32) -> core.i32 convention(direct) {
    tribute_control.return
  }
}"#;
        let (ctx, module) = parse(local_input);
        let error = verify_tribute_control_pre_cps(&ctx, module, &[]).unwrap_err();
        assert!(error.to_string().contains("expects 1 operand"), "{error}");

        let core_input = r#"core.module @test {
  tribute_control.func @broken(%value: core.i32) -> core.i32 convention(direct) {
    func.tail_call_indirect
    tribute_control.return %value
  }
}"#;
        let (ctx, module) = parse(core_input);
        let error = verify_tribute_control_pre_cps(&ctx, module, &[]).unwrap_err();
        assert!(error.to_string().contains("requires a callee"), "{error}");

        let post_input = r#"core.module @test {
  func.func @broken() -> core.never attributes {tribute.calling_convention = 2} {
    func.tail_call_indirect
  }
}"#;
        let (ctx, module) = parse(post_input);
        let error = verify_tribute_control_post_cps(&ctx, module).unwrap_err();
        assert!(error.to_string().contains("requires a callee"), "{error}");

        let malformed_delimiter = r#"core.module @test {
  !evidence = core.array(adt.struct() {fields = [[@ability_id, core.i32], [@prompt_tag, core.i32], [@tr_dispatch_fn, core.ptr], [@handler_dispatch, core.ptr]], name = @_Marker})
  func.func @broken() -> core.never attributes {tribute.calling_convention = 2} {
    ability.handle_dispatch {ability_refs = []} {
      ^body(%inner: !evidence):
        func.unreachable
    }
  }
}"#;
        let (ctx, module) = parse(malformed_delimiter);
        let error = verify_tribute_control_post_cps(&ctx, module).unwrap_err();
        assert!(error.to_string().contains("requires an evidence operand"));
    }

    #[test]
    fn post_boundary_rejects_recursively_nested_control_types() {
        let input = r#"core.module @test {
  !nested = core.tuple(closure.closure(core.func(core.i32, tribute_control.resume_token(core.i32, core.i32))))
}"#;
        let (ctx, module) = parse(input);
        let error = verify_tribute_control_post_cps(&ctx, module).unwrap_err();
        assert!(error.to_string().contains("forbidden type"), "{error}");
    }

    #[test]
    fn post_boundary_rejects_malformed_physical_callable_transfers() {
        let malformed = [
            (
                r#"core.module @test {
  func.func @callee(%value: core.i32) -> core.never attributes {tribute.calling_convention = 2} {
    func.unreachable
  }
  func.func @caller(%value: core.i32) -> core.never attributes {tribute.calling_convention = 2} {
    func.tail_call %value {tribute.calling_convention = 2}
  }
}"#,
                "requires a resolved callee symbol",
            ),
            (
                r#"core.module @test {
  func.func @callee(%value: core.i32) -> core.never attributes {tribute.calling_convention = 2} {
    func.unreachable
  }
  func.func @caller(%value: core.i32) -> core.never attributes {tribute.calling_convention = 2} {
    func.tail_call %value {callee = @callee}
  }
}"#,
                "func.tail_call must carry exact Direct, EvidenceDirect, or Cps metadata",
            ),
            (
                r#"core.module @test {
  func.func @caller(%value: core.i32) -> core.never attributes {tribute.calling_convention = 2} {
    func.tail_call %value {callee = @missing, tribute.calling_convention = 2}
  }
}"#,
                "unresolved callee @missing",
            ),
            (
                r#"core.module @test {
  func.func @callee(%value: core.i64) -> core.never attributes {tribute.calling_convention = 2} {
    func.unreachable
  }
  func.func @caller(%value: core.i32) -> core.never attributes {tribute.calling_convention = 2} {
    func.tail_call %value {callee = @callee, tribute.calling_convention = 2}
  }
}"#,
                "operands do not match the target signature",
            ),
            (
                r#"core.module @test {
  func.func @callee(%value: core.i32) -> core.i32 attributes {tribute.calling_convention = 2} {
    func.return %value
  }
  func.func @caller(%value: core.i32) -> core.never attributes {tribute.calling_convention = 2} {
    func.tail_call %value {callee = @callee, tribute.calling_convention = 2}
  }
}"#,
                "target must have core.never result",
            ),
            (
                r#"core.module @test {
  func.func @callee(%value: core.i32) -> core.never attributes {tribute.calling_convention = 1} {
    func.unreachable
  }
  func.func @caller(%value: core.i32) -> core.never attributes {tribute.calling_convention = 2} {
    func.tail_call %value {callee = @callee, tribute.calling_convention = 2}
  }
}"#,
                "must preserve exact Cps metadata",
            ),
            (
                r#"core.module @test {
  func.func @caller(%callee: closure.closure(core.func(core.never, core.i32)), %value: core.i32) -> core.never attributes {tribute.calling_convention = 2} {
    func.tail_call_indirect %callee, %value
  }
}"#,
                "tail_call_indirect must carry exact Cps metadata",
            ),
            (
                r#"core.module @test {
  func.func @callee(%value: core.i32) -> core.i32 attributes {tribute.calling_convention = 0} {
    func.return %value
  }
  func.func @caller(%value: core.i32) -> core.i32 attributes {tribute.calling_convention = 0} {
    %result = func.call %value {callee = @callee, tribute.calling_convention = 1} : core.i32
    func.return %result
  }
}"#,
                "func.call metadata does not match its target",
            ),
            (
                r#"core.module @test {
  func.func @caller(%value: core.i32) -> core.i32 attributes {tribute.calling_convention = 0} {
    %result = func.call %value {callee = @missing, tribute.calling_convention = 0} : core.i32
    func.return %result
  }
}"#,
                "func.call references unresolved callee @missing",
            ),
            (
                r#"core.module @test {
  func.func @caller(%callee: closure.closure(core.func(core.never, core.i32)), %value: core.i32) -> core.never attributes {tribute.calling_convention = 2} {
    %result = func.call_indirect %callee, %value {tribute.calling_convention = 2} : core.never
    func.unreachable
  }
}"#,
                "dynamic Cps transfers must use func.tail_call_indirect",
            ),
        ];
        for (input, expected) in malformed {
            let (ctx, module) = parse(input);
            let error = verify_tribute_control_post_cps(&ctx, module).unwrap_err();
            assert!(error.to_string().contains(expected), "{error}");
        }
    }

    #[test]
    fn post_boundary_rejects_nonphysical_dispatchers_and_residual_control_ops() {
        let dispatcher_input = r#"core.module @test {
  !evidence = core.array(adt.struct() {fields = [[@ability_id, core.i32], [@prompt_tag, core.i32], [@tr_dispatch_fn, core.ptr], [@handler_dispatch, core.ptr]], name = @_Marker})
  !tr = closure.closure(core.func(tribute_rt.anyref, !evidence, core.i32, tribute_rt.anyref))
  !general = closure.closure(core.func(core.never, !evidence, tribute_rt.anyref, core.i32, tribute_rt.anyref))
  func.func @caller(%ev: !evidence, %tr: !tr, %general: !general) -> core.never attributes {tribute.calling_convention = 2} {
    %prompt = arith.const {value = 0} : core.i32
    ability.handle_dispatch %ev, %prompt, %tr, %general {ability_refs = [core.ability_ref() {name = @State}]} {
      ^body(%inner: !evidence):
        func.unreachable
    }
  }
}"#;
        let (ctx, module) = parse(dispatcher_input);
        let error = verify_tribute_control_post_cps(&ctx, module).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("dispatcher must be a physical closure result"),
            "{error}"
        );

        let wrong_abi_input = r#"core.module @test {
  !evidence = core.array(adt.struct() {fields = [[@ability_id, core.i32], [@prompt_tag, core.i32], [@tr_dispatch_fn, core.ptr], [@handler_dispatch, core.ptr]], name = @_Marker})
  func.func @caller(%ev: !evidence) -> core.never attributes {tribute.calling_convention = 2} {
    %prompt = arith.const {value = 0} : core.i32
    %tr = closure.lambda(%inner: !evidence) -> tribute_rt.anyref {tribute.calling_convention = 1} {
      func.unreachable
    }
    %general = closure.lambda(%inner: !evidence) -> core.never {tribute.calling_convention = 2} {
      func.unreachable
    }
    ability.handle_dispatch %ev, %prompt, %tr, %general {ability_refs = [core.ability_ref() {name = @State}]} {
      ^body(%inner: !evidence):
        func.unreachable
    }
  }
}"#;
        let (ctx, module) = parse(wrong_abi_input);
        let error = verify_tribute_control_post_cps(&ctx, module).unwrap_err();
        let text = error.to_string();
        assert!(
            text.contains("tail-resumptive dispatcher has the wrong"),
            "{text}"
        );
        assert!(text.contains("general dispatcher has the wrong"), "{text}");

        let wrong_metadata_input = r#"core.module @test {
  !evidence = core.array(adt.struct() {fields = [[@ability_id, core.i32], [@prompt_tag, core.i32], [@tr_dispatch_fn, core.ptr], [@handler_dispatch, core.ptr]], name = @_Marker})
  func.func @caller(%ev: !evidence) -> core.never attributes {tribute.calling_convention = 2} {
    %prompt = arith.const {value = 0} : core.i32
    %tr = closure.lambda(%inner: !evidence, %op_idx: core.i32, %payload: tribute_rt.anyref) -> tribute_rt.anyref {tribute.calling_convention = 0} {
      func.unreachable
    }
    %general = closure.lambda(%inner: !evidence, %continuation: tribute_rt.anyref, %op_idx: core.i32, %payload: tribute_rt.anyref) -> core.never {tribute.calling_convention = 1} {
      func.unreachable
    }
    ability.handle_dispatch %ev, %prompt, %tr, %general {ability_refs = [core.ability_ref() {name = @State}]} {
      ^body(%inner: !evidence):
        func.unreachable
    }
  }
}"#;
        let (ctx, module) = parse(wrong_metadata_input);
        let error = verify_tribute_control_post_cps(&ctx, module).unwrap_err();
        let text = error.to_string();
        assert!(text.contains("calling convention metadata 1"), "{text}");
        assert!(text.contains("calling convention metadata 2"), "{text}");

        let residual_input = r#"core.module @test {
  tribute_control.func @residual(%value: core.i32) -> core.i32 convention(direct) {
    tribute_control.return %value
  }
}"#;
        let (ctx, module) = parse(residual_input);
        let error = verify_tribute_control_post_cps(&ctx, module).unwrap_err();
        assert!(error.to_string().contains("residual tribute_control.func"));
    }

    #[test]
    fn conversion_error_display_handles_diagnostics_without_an_operation() {
        let error =
            TributeControlToCpsError::one(PRE_CPS_BOUNDARY, None, None, "synthetic failure");
        let text = error.to_string();
        assert!(text.contains("1 error(s)"));
        assert!(text.contains("  - synthetic failure"));
    }

    #[test]
    fn textual_resumptive_handle_emits_one_resultless_delimiter() {
        let input = r#"core.module @test {
  tribute_control.func @run(%input: core.i32) -> core.i32 convention(cps) {
    %handled = tribute_control.handle : core.i32 {
      %performed = tribute_control.perform %input {ability_ref = core.ability_ref() {name = @State}, op_name = @get, operation_kind = @op} : core.i32
      tribute_control.yield %performed
    } {
      ^completion(%value: core.i32):
        tribute_control.yield %value
    } {
      tribute_control.handler {ability_ref = core.ability_ref() {name = @State}, kind = @op, op_name = @get, operation_result_type = core.i32} {
        ^arm(%argument: core.i32, %token: tribute_control.resume_token(core.i32, core.i32)):
          %resumed = tribute_control.resume %token, %argument : core.i32
          tribute_control.yield %resumed
      }
    }
    tribute_control.return %handled
  }
}"#;
        let (mut ctx, module) = parse(input);
        let mut ability_ref = None;
        fn find_ability(ctx: &IrContext, region: RegionRef, found: &mut Option<TypeRef>) {
            for block in ctx.region(region).blocks.iter().copied() {
                for op in ctx.block(block).ops.iter().copied() {
                    if tribute_control::Handler::matches(ctx, op) {
                        *found = ctx.op(op).attributes.get_type("ability_ref");
                    }
                    for nested in ctx.op(op).regions.iter().copied() {
                        find_ability(ctx, nested, found);
                    }
                }
            }
        }
        find_ability(&ctx, module.body(&ctx).unwrap(), &mut ability_ref);
        let i32_type = ctx
            .types
            .iter()
            .find_map(|(ty, data)| {
                (data.dialect == Symbol::new("core") && data.name == Symbol::new("i32"))
                    .then_some(ty)
            })
            .unwrap();
        let declarations = [tribute_control::OperationDeclaration::new(
            ability_ref.unwrap(),
            Symbol::new("get"),
            Symbol::new("op"),
            vec![i32_type],
            i32_type,
        )];
        tribute_control_to_cps(&mut ctx, module, &declarations).unwrap();
        let mut consumed_get = None;
        let mut consumed_set = None;
        fn find_one_shot_state_ops(
            ctx: &IrContext,
            op: OpRef,
            consumed_get: &mut Option<ValueRef>,
            consumed_set: &mut Option<ValueRef>,
        ) {
            let is_one_shot_type = |ty: TypeRef| {
                ctx.types
                    .get(ty)
                    .attrs
                    .get_symbol("name")
                    .is_some_and(|name| {
                        name.with_str(|text| text.starts_with("__tribute_one_shot_state"))
                    })
            };
            if let Ok(get) = adt::StructGet::from_op(ctx, op)
                && is_one_shot_type(get.r#type(ctx))
            {
                *consumed_get = Some(get.r#ref(ctx));
            }
            if let Ok(set) = adt::StructSet::from_op(ctx, op)
                && is_one_shot_type(set.r#type(ctx))
            {
                assert!(ctx.op_results(op).is_empty(), "struct_set mutates in place");
                *consumed_set = Some(set.r#ref(ctx));
            }
            for region in ctx.op(op).regions.iter().copied() {
                for block in ctx.region(region).blocks.iter().copied() {
                    for child in ctx.block(block).ops.iter().copied() {
                        find_one_shot_state_ops(ctx, child, consumed_get, consumed_set);
                    }
                }
            }
        }
        find_one_shot_state_ops(&ctx, module.op(), &mut consumed_get, &mut consumed_set);
        assert_eq!(consumed_get, consumed_set);
        assert!(consumed_get.is_some(), "one-shot state was not emitted");
        let printed = print_module(&ctx, module.op());
        assert_eq!(printed.matches("ability.handle_dispatch").count(), 1);
        assert!(printed.contains("effect.fresh_prompt_tag"));
        assert!(printed.contains("ability_refs = [core.ability_ref"));
        assert!(printed.contains("func.tail_call_indirect"));
        assert!(printed.contains("adt.struct_set"));
        assert!(!printed.contains("tribute_control."));
        assert!(!printed.contains("handler_metadata"));
        assert!(!printed.contains("adt.ref_null"));
        assert!(!printed.contains("__tribute_cps_control"));
    }

    #[test]
    fn multiple_arms_for_one_ability_emit_one_dispatcher_pair() {
        let input = r#"core.module @test {
  tribute_control.func @run(%input: core.i32) -> core.i32 convention(cps) {
    %handled = tribute_control.handle : core.i32 {
      %performed = tribute_control.perform %input {ability_ref = core.ability_ref() {name = @State}, op_name = @get, operation_kind = @op} : core.i32
      tribute_control.yield %performed
    } {
      ^completion(%value: core.i32):
        tribute_control.yield %value
    } {
      tribute_control.handler {ability_ref = core.ability_ref() {name = @State}, kind = @op, op_name = @get, operation_result_type = core.i32} {
        ^get(%argument: core.i32, %token: tribute_control.resume_token(core.i32, core.i32)):
          %resumed = tribute_control.resume %token, %argument : core.i32
          tribute_control.yield %resumed
      }
      tribute_control.handler {ability_ref = core.ability_ref() {name = @State}, kind = @op, op_name = @set, operation_result_type = core.i32} {
        ^set(%argument: core.i32, %token: tribute_control.resume_token(core.i32, core.i32)):
          %fallback = arith.const {value = 9} : core.i32
          tribute_control.yield %fallback
      }
    }
    tribute_control.return %handled
  }
}"#;
        let (mut ctx, module) = parse(input);
        let ability_ref = ctx
            .types
            .iter()
            .find_map(|(ty, data)| {
                (data.dialect == Symbol::new("core") && data.name == Symbol::new("ability_ref"))
                    .then_some(ty)
            })
            .unwrap();
        let i32_type = ctx
            .types
            .iter()
            .find_map(|(ty, data)| {
                (data.dialect == Symbol::new("core") && data.name == Symbol::new("i32"))
                    .then_some(ty)
            })
            .unwrap();
        let declarations = [
            tribute_control::OperationDeclaration::new(
                ability_ref,
                Symbol::new("get"),
                Symbol::new("op"),
                [i32_type],
                i32_type,
            ),
            tribute_control::OperationDeclaration::new(
                ability_ref,
                Symbol::new("set"),
                Symbol::new("op"),
                [i32_type],
                i32_type,
            ),
        ];
        tribute_control_to_cps(&mut ctx, module, &declarations).unwrap();

        let mut delimiters = Vec::new();
        fn collect_delimiters(ctx: &IrContext, op: OpRef, found: &mut Vec<OpRef>) {
            if ability::HandleDispatch::matches(ctx, op) {
                found.push(op);
            }
            for region in ctx.op(op).regions.iter().copied() {
                for block in ctx.region(region).blocks.iter().copied() {
                    for child in ctx.block(block).ops.iter().copied() {
                        collect_delimiters(ctx, child, found);
                    }
                }
            }
        }
        collect_delimiters(&ctx, module.op(), &mut delimiters);
        let [delimiter] = delimiters.as_slice() else {
            panic!("expected one final delimiter");
        };
        assert_eq!(ctx.op_operands(*delimiter).len(), 4);
        let Some(Attribute::List(ability_refs)) = ctx.op(*delimiter).attributes.get("ability_refs")
        else {
            panic!("final delimiter must have ability_refs");
        };
        assert_eq!(ability_refs.len(), 1);
        let printed = print_module(&ctx, module.op());
        assert!(printed.contains("value = 9"));
        assert!(printed.contains("func.tail_call_indirect"));
        assert!(printed.contains("effect.fresh_prompt_tag"));
    }

    #[test]
    fn textual_scf_branch_captures_only_the_selected_suffix() {
        let input = r#"core.module @test {
  tribute_control.func @branch(%input: core.i32) -> core.i32 convention(cps) {
    %condition = arith.const {value = true} : core.i1
    %selected = scf.if %condition : core.i32 {
      %performed = tribute_control.perform %input {ability_ref = core.ability_ref() {name = @State}, op_name = @get, operation_kind = @op} : core.i32
      scf.yield %performed
    } {
      %fallback = arith.const {value = 7} : core.i32
      scf.yield %fallback
    }
    %one = arith.const {value = 1} : core.i32
    %sum = arith.addi %selected, %one : core.i32
    tribute_control.return %sum
  }
}"#;
        let (mut ctx, module) = parse(input);
        let ability_ref = ctx
            .types
            .iter()
            .find_map(|(ty, data)| {
                (data.dialect == Symbol::new("core") && data.name == Symbol::new("ability_ref"))
                    .then_some(ty)
            })
            .unwrap();
        let i32_type = ctx
            .types
            .iter()
            .find_map(|(ty, data)| {
                (data.dialect == Symbol::new("core") && data.name == Symbol::new("i32"))
                    .then_some(ty)
            })
            .unwrap();
        let declarations = [tribute_control::OperationDeclaration::new(
            ability_ref,
            Symbol::new("get"),
            Symbol::new("op"),
            vec![i32_type],
            i32_type,
        )];
        tribute_control_to_cps(&mut ctx, module, &declarations).unwrap();
        let printed = print_module(&ctx, module.op());
        assert!(printed.contains("scf.if"));
        assert!(printed.contains(" : core.never"));
        assert!(printed.contains("ability.perform"));
        assert!(printed.contains("func.tail_call_indirect"));
        assert!(printed.contains("arith.addi"));
        assert!(!printed.contains("tribute_control."));
    }

    #[test]
    fn textual_zero_result_cps_and_direct_scf_branches_lower() {
        let input = r#"core.module @test {
  tribute_control.func @identity(%value: core.i32) -> core.i32 convention(direct) {
    tribute_control.return %value
  }
  tribute_control.func @branch(%input: core.i32, %condition: core.i1) -> core.i32 convention(cps) {
    %direct = tribute_control.lambda(%value: core.i32) -> core.i32 convention(direct) captures [%condition] {
      %selected = scf.if %condition : core.i32 {
        %called = tribute_control.call %value {callee = @identity} : core.i32
        scf.yield %called
      } {
        scf.yield %value
      }
      tribute_control.return %selected
    }
    scf.if %condition {
      %performed = tribute_control.perform %input {ability_ref = core.ability_ref() {name = @State}, op_name = @get, operation_kind = @op} : core.i32
      scf.yield
    } {
      %performed = tribute_control.perform %input {ability_ref = core.ability_ref() {name = @State}, op_name = @set, operation_kind = @op} : core.i32
      scf.yield
    }
    tribute_control.return %input
  }
}"#;
        let (mut ctx, module) = parse(input);
        let ability_ref = ctx
            .types
            .iter()
            .find_map(|(ty, data)| {
                (data.dialect == Symbol::new("core") && data.name == Symbol::new("ability_ref"))
                    .then_some(ty)
            })
            .unwrap();
        let i32_type = ctx
            .types
            .iter()
            .find_map(|(ty, data)| {
                (data.dialect == Symbol::new("core") && data.name == Symbol::new("i32"))
                    .then_some(ty)
            })
            .unwrap();
        let declarations = [
            tribute_control::OperationDeclaration::new(
                ability_ref,
                Symbol::new("get"),
                Symbol::new("op"),
                [i32_type],
                i32_type,
            ),
            tribute_control::OperationDeclaration::new(
                ability_ref,
                Symbol::new("set"),
                Symbol::new("op"),
                [i32_type],
                i32_type,
            ),
        ];
        tribute_control_to_cps(&mut ctx, module, &declarations).unwrap();
        let printed = print_module(&ctx, module.op());
        let scf_ifs: Vec<_> = printed
            .lines()
            .filter(|line| line.contains("scf.if"))
            .collect();
        let value_if_count = scf_ifs
            .iter()
            .filter(|line| line.contains(": core.i32"))
            .count();
        assert_eq!(value_if_count, 1, "{printed}");
        assert!(
            scf_ifs.iter().any(|line| line.contains(": core.never")),
            "{printed}"
        );
        assert_eq!(printed.matches("ability.perform").count(), 2, "{printed}");
        assert!(printed.contains("func.call") && printed.contains("func.tail_call_indirect"));
        assert!(!printed.contains("tribute_control."));

        let mut reparsed = IrContext::new();
        let reparsed_module = parse_test_module(&mut reparsed, &printed);
        verify_tribute_control_post_cps(&reparsed, reparsed_module).unwrap();
    }

    #[test]
    fn malformed_multi_result_effectful_scf_if_remains_unchanged() {
        let input = r#"core.module @test {
  tribute_control.func @broken(%input: core.i32, %condition: core.i1) -> core.i32 convention(cps) {
    %left, %right = scf.if %condition : core.i32, core.i32 {
      %performed = tribute_control.perform %input {ability_ref = core.ability_ref() {name = @State}, op_name = @get, operation_kind = @op} : core.i32
      scf.yield %performed, %input
    } {
      scf.yield %input, %input
    }
    tribute_control.return %left
  }
}"#;
        let (mut ctx, module) = parse(input);
        let before = print_module(&ctx, module.op());
        let ability_ref = ctx
            .types
            .iter()
            .find_map(|(ty, data)| {
                (data.dialect == Symbol::new("core") && data.name == Symbol::new("ability_ref"))
                    .then_some(ty)
            })
            .unwrap();
        let i32_type = ctx
            .types
            .iter()
            .find_map(|(ty, data)| {
                (data.dialect == Symbol::new("core") && data.name == Symbol::new("i32"))
                    .then_some(ty)
            })
            .unwrap();
        let declarations = [tribute_control::OperationDeclaration::new(
            ability_ref,
            Symbol::new("get"),
            Symbol::new("op"),
            [i32_type],
            i32_type,
        )];
        let error = tribute_control_to_cps(&mut ctx, module, &declarations).unwrap_err();
        assert_eq!(error.boundary, POST_CPS_BOUNDARY);
        assert!(
            error.to_string().contains("requires zero or one result"),
            "{error}"
        );
        assert_eq!(print_module(&ctx, module.op()), before);
    }

    #[test]
    fn stronger_func_ref_builds_a_cps_adapter_without_a_null_environment() {
        let input = r#"core.module @test {
  !cps = tribute_control.callable(core.i32, core.i32) {tribute.calling_convention = 2}
  tribute_control.func @id(%value: core.i32) -> core.i32 convention(direct) {
    tribute_control.return %value
  }
  tribute_control.func @run(%value: core.i32) -> core.i32 convention(cps) {
    %callee = tribute_control.func_ref {func_ref = @id} : !cps
    %result = tribute_control.call_indirect %callee, %value : core.i32
    tribute_control.return %result
  }
}"#;
        let (mut ctx, module) = parse(input);
        tribute_control_to_cps(&mut ctx, module, &[]).unwrap();
        let printed = print_module(&ctx, module.op());
        assert!(printed.contains("__tribute_func_ref_adapter"));
        assert!(printed.contains("closure.new"));
        assert!(printed.contains("func.tail_call_indirect"));
        assert!(printed.contains("tribute.calling_convention = 2"));
        assert!(!printed.contains("adt.ref_null"));
        assert!(!printed.contains("tribute_control.func "));
        assert!(!printed.contains("tribute_control.func_ref "));
        assert!(!printed.contains("tribute_control.call_indirect "));
    }

    #[test]
    fn func_ref_adapters_cover_every_legal_convention_strengthening() {
        let input = r#"core.module @test {
  !direct = tribute_control.callable(core.i32, core.i32) {tribute.calling_convention = 0}
  !evidence = tribute_control.callable(core.i32, core.i32) {tribute.calling_convention = 1}
  !cps = tribute_control.callable(core.i32, core.i32) {tribute.calling_convention = 2}
  tribute_control.func @direct(%value: core.i32) -> core.i32 convention(direct) {
    tribute_control.return %value
  }
  tribute_control.func @evidence(%value: core.i32) -> core.i32 convention(evidence_direct) {
    tribute_control.return %value
  }
  tribute_control.func @cps(%value: core.i32) -> core.i32 convention(cps) {
    tribute_control.return %value
  }
  tribute_control.func @refs(%value: core.i32) -> core.i32 convention(cps) {
    %direct_direct = tribute_control.func_ref {func_ref = @direct} : !direct
    %direct_evidence = tribute_control.func_ref {func_ref = @direct} : !evidence
    %direct_cps = tribute_control.func_ref {func_ref = @direct} : !cps
    %evidence_evidence = tribute_control.func_ref {func_ref = @evidence} : !evidence
    %evidence_cps = tribute_control.func_ref {func_ref = @evidence} : !cps
    %cps_cps = tribute_control.func_ref {func_ref = @cps} : !cps
    tribute_control.return %value
  }
}"#;
        let (mut ctx, module) = parse(input);
        tribute_control_to_cps(&mut ctx, module, &[]).unwrap();
        verify_tribute_control_post_cps(&ctx, module).unwrap();
        let printed = print_module(&ctx, module.op());
        assert_eq!(
            printed
                .matches("func.func @__tribute_func_ref_adapter")
                .count(),
            6,
            "{printed}"
        );
        assert!(!printed.contains("tribute_control."));
    }

    #[test]
    fn weaker_func_ref_adapter_is_rejected_before_mutation() {
        let input = r#"core.module @test {
  !evidence = tribute_control.callable(core.i32, core.i32) {tribute.calling_convention = 1}
  tribute_control.func @cps(%value: core.i32) -> core.i32 convention(cps) {
    tribute_control.return %value
  }
  tribute_control.func @broken(%value: core.i32) -> core.i32 convention(evidence_direct) {
    %callee = tribute_control.func_ref {func_ref = @cps} : !evidence
    tribute_control.return %value
  }
}"#;
        let (mut ctx, module) = parse(input);
        let before = print_module(&ctx, module.op());
        let error = tribute_control_to_cps(&mut ctx, module, &[]).unwrap_err();
        assert_eq!(error.boundary, PRE_CPS_BOUNDARY);
        assert!(
            error
                .to_string()
                .contains("result convention must be at least as strong"),
            "{error}"
        );
        assert_eq!(print_module(&ctx, module.op()), before);
    }

    #[test]
    fn nested_textual_handles_keep_distinct_delimiters() {
        let input = r#"core.module @test {
  tribute_control.func @nested(%input: core.i32) -> core.i32 convention(cps) {
    %outer = tribute_control.handle : core.i32 {
      %inner = tribute_control.handle : core.i32 {
        tribute_control.yield %input
      } {
        ^inner_completion(%value: core.i32):
          tribute_control.yield %value
      } {
        ^inner_handlers:
      }
      tribute_control.yield %inner
    } {
      ^outer_completion(%value: core.i32):
        tribute_control.yield %value
    } {
      ^outer_handlers:
    }
    tribute_control.return %outer
  }
}"#;
        let (mut ctx, module) = parse(input);
        tribute_control_to_cps(&mut ctx, module, &[]).unwrap();
        let printed = print_module(&ctx, module.op());
        assert_eq!(printed.matches("ability.handle_dispatch").count(), 2);
        assert!(printed.contains("effect.fresh_prompt_tag"));
        assert!(printed.matches("func.tail_call_indirect").count() >= 3);
        assert!(!printed.contains("__tribute_cps_control"));
    }

    #[test]
    fn op_to_never_uses_a_typed_zero_capture_reject_continuation() {
        let input = r#"core.module @test {
  tribute_control.func @abortable(%input: core.i32) -> core.i32 convention(cps) {
    %handled = tribute_control.handle : core.i32 {
      %never = tribute_control.perform %input {ability_ref = core.ability_ref() {name = @Abort}, op_name = @abort, operation_kind = @op} : core.never
      %unreachable_suffix = arith.const {value = 99} : core.i32
      tribute_control.yield %unreachable_suffix
    } {
      ^completion(%value: core.i32):
        tribute_control.yield %value
    } {
      tribute_control.handler {ability_ref = core.ability_ref() {name = @Abort}, kind = @op, op_name = @abort, operation_result_type = core.never} {
        ^arm(%argument: core.i32):
          %fallback = arith.const {value = 7} : core.i32
          tribute_control.yield %fallback
      }
    }
    tribute_control.return %handled
  }
}"#;
        let (mut ctx, module) = parse(input);
        let ability_ref = ctx
            .types
            .iter()
            .find_map(|(ty, data)| {
                (data.dialect == Symbol::new("core") && data.name == Symbol::new("ability_ref"))
                    .then_some(ty)
            })
            .unwrap();
        let i32_type = ctx
            .types
            .iter()
            .find_map(|(ty, data)| {
                (data.dialect == Symbol::new("core") && data.name == Symbol::new("i32"))
                    .then_some(ty)
            })
            .unwrap();
        let never_type = ctx
            .types
            .iter()
            .find_map(|(ty, data)| {
                (data.dialect == Symbol::new("core") && data.name == Symbol::new("never"))
                    .then_some(ty)
            })
            .unwrap();
        let declarations = [tribute_control::OperationDeclaration::new(
            ability_ref,
            Symbol::new("abort"),
            Symbol::new("op"),
            vec![i32_type],
            never_type,
        )];
        tribute_control_to_cps(&mut ctx, module, &declarations).unwrap();
        let printed = print_module(&ctx, module.op());
        assert!(printed.contains("ability.perform"));
        assert!(printed.contains("func.unreachable"));
        assert!(!printed.contains("value = 99"));
        assert!(!printed.contains("@consumed"));
        assert!(!printed.contains("adt.struct_set"));
        assert!(!printed.contains("adt.ref_null"));
    }

    #[test]
    fn fn_operation_stays_evidence_direct_without_continuation_capture() {
        let input = r#"core.module @test {
  tribute_control.func @read(%input: core.i32) -> core.i32 convention(cps) {
    %handled = tribute_control.handle : core.i32 {
      %value = tribute_control.perform %input {ability_ref = core.ability_ref() {name = @Reader}, op_name = @read, operation_kind = @fn} : core.i32
      tribute_control.yield %value
    } {
      ^completion(%value: core.i32):
        tribute_control.yield %value
    } {
      tribute_control.handler {ability_ref = core.ability_ref() {name = @Reader}, kind = @fn, op_name = @read, operation_result_type = core.i32} {
        ^arm(%argument: core.i32):
          tribute_control.yield %argument
      }
    }
    tribute_control.return %handled
  }
}"#;
        let (mut ctx, module) = parse(input);
        let ability_ref = ctx
            .types
            .iter()
            .find_map(|(ty, data)| {
                (data.dialect == Symbol::new("core") && data.name == Symbol::new("ability_ref"))
                    .then_some(ty)
            })
            .unwrap();
        let i32_type = ctx
            .types
            .iter()
            .find_map(|(ty, data)| {
                (data.dialect == Symbol::new("core") && data.name == Symbol::new("i32"))
                    .then_some(ty)
            })
            .unwrap();
        let declarations = [tribute_control::OperationDeclaration::new(
            ability_ref,
            Symbol::new("read"),
            Symbol::new("fn"),
            vec![i32_type],
            i32_type,
        )];
        tribute_control_to_cps(&mut ctx, module, &declarations).unwrap();
        let printed = print_module(&ctx, module.op());
        assert!(printed.contains("ability.call"));
        assert!(!printed.contains("ability.perform"));
        assert!(!printed.contains("@consumed"));
        assert!(!printed.contains("adt.struct_set"));
        assert!(printed.contains("tribute.calling_convention = 1"));
    }

    #[test]
    fn textual_scf_switch_reenters_the_shared_suffix() {
        let input = r#"core.module @test {
  tribute_control.func @switching(%input: core.i32) -> core.i32 convention(cps) {
    %choice = arith.const {value = 0} : core.i32
    scf.switch %choice {
      scf.case {value = 0} {
        %performed = tribute_control.perform %input {ability_ref = core.ability_ref() {name = @State}, op_name = @get, operation_kind = @op} : core.i32
        scf.yield
      }
      scf.default {
        scf.yield
      }
    }
    tribute_control.return %input
  }
}"#;
        let (mut ctx, module) = parse(input);
        let ability_ref = ctx
            .types
            .iter()
            .find_map(|(ty, data)| {
                (data.dialect == Symbol::new("core") && data.name == Symbol::new("ability_ref"))
                    .then_some(ty)
            })
            .unwrap();
        let i32_type = ctx
            .types
            .iter()
            .find_map(|(ty, data)| {
                (data.dialect == Symbol::new("core") && data.name == Symbol::new("i32"))
                    .then_some(ty)
            })
            .unwrap();
        let declarations = [tribute_control::OperationDeclaration::new(
            ability_ref,
            Symbol::new("get"),
            Symbol::new("op"),
            vec![i32_type],
            i32_type,
        )];
        tribute_control_to_cps(&mut ctx, module, &declarations).unwrap();
        let printed = print_module(&ctx, module.op());
        assert!(printed.contains("scf.switch"));
        assert!(printed.contains("scf.case"));
        assert!(printed.contains("ability.perform"));
        assert!(printed.matches("func.tail_call_indirect").count() >= 2);
        assert!(!printed.contains("tribute_control."));
    }
}
