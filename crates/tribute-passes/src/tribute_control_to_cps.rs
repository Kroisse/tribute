//! Atomic legalization of verified `tribute_control` callable/control IR.
//!
//! The pass deliberately stops at the physical `func`/`closure` and logical
//! `ability` surface. It does not run closure extraction, evidence lowering,
//! or target-specific conversion.

use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt;

use tribute_core::calling_convention::{
    cps_closure_function_type, cps_completion_type, cps_continuation_frame_layout_type,
    cps_continuation_frame_ref_type, cps_done_type, cps_resume_exact_type,
    physical_closure_function_type, physical_closure_type_with_environment_index,
};
use tribute_core::{
    CALLING_CONVENTION_ATTR, CallableAbi, CallingConvention, physical_closure_type,
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
                    && type_is(ctx, params[2], "tribute_rt", "anyref")
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
            if let Some((evidence, rest)) = operands.split_first()
                && let Some((_, dispatchers)) = rest.split_first()
            {
                let evidence_ty = ctx.value_ty(*evidence);
                for pair in dispatchers.as_chunks::<2>().0 {
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
    compiler_intrinsics: &[tribute_control::CompilerIntrinsicDeclaration],
) -> Result<(), TributeControlToCpsError> {
    let mut failures = Vec::new();
    let validation = tribute_control::validate(ctx, module, declarations, compiler_intrinsics);
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
}

struct Converter<'a> {
    ctx: &'a mut IrContext,
    module_block: BlockRef,
    funcs_by_module: HashMap<OpRef, HashMap<Symbol, CallableInfo>>,
    current_module: OpRef,
    converted_types: HashMap<TypeRef, TypeRef>,
    frames: HashMap<TypeRef, FrameTypes>,
    frame_layout_aliases: Vec<(Symbol, TypeRef)>,
    helper_index: u32,
}

#[derive(Clone, Copy)]
struct FrameTypes {
    reference: TypeRef,
    layout: TypeRef,
    done: TypeRef,
    dispatch: TypeRef,
}

#[derive(Clone, Copy)]
struct DispatchFactoryArgs<'a> {
    prefix: &'a [ValueRef],
    suffix: &'a [ValueRef],
}

#[derive(Clone)]
struct Flow {
    convention: CallingConvention,
    evidence: Option<ValueRef>,
    exit_k: Option<ValueRef>,
    root_exit_k: Option<ValueRef>,
    /// The exact lexical dispatcher installed by the nearest handle. General
    /// operations carry it through the effect ABI without erasing it.
    dispatch: Option<ValueRef>,
    /// Zero-result structured control uses a private suffix closure whose
    /// arguments are `(Evidence, ContinuationFrame<R>)` rather than `Done<R>`.
    void_exit_k: Option<ValueRef>,
    answer_type: TypeRef,
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
            frames: HashMap::new(),
            frame_layout_aliases: Vec::new(),
            helper_index: 0,
        }
    }

    fn current_func(&self, symbol: Symbol) -> Option<CallableInfo> {
        self.funcs_by_module
            .get(&self.current_module)
            .and_then(|funcs| funcs.get(&symbol))
            .cloned()
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

    /// Emit a final CPS tail transfer from the callee's exact typed closure
    /// contract. This must not reconstruct a signature from physical operands:
    /// closure extraction interposes an environment later and preserves this
    /// contract on the resulting indirect transfer.
    fn emit_cps_tail_call_indirect(
        &mut self,
        block: BlockRef,
        location: Location,
        callee: ValueRef,
        args: impl IntoIterator<Item = ValueRef>,
    ) -> Result<OpRef, TributeControlToCpsError> {
        let args = args.into_iter().collect::<Vec<_>>();
        let closure_type = self.ctx.value_ty(callee);
        let signature = cps_closure_function_type(self.ctx, closure_type).ok_or_else(|| {
            TributeControlToCpsError::one(
                POST_CPS_BOUNDARY,
                None,
                Some(location),
                "CPS indirect tail callee has no exact provenance-bearing closure contract",
            )
        })?;
        let callable = core::Func::from_type_ref(self.ctx, signature).ok_or_else(|| {
            TributeControlToCpsError::one(
                POST_CPS_BOUNDARY,
                None,
                Some(location),
                "CPS indirect tail callee contract is not core.func",
            )
        })?;
        if callable.r#return(self.ctx) != self.never_type()
            || callable.params(self.ctx).len() != args.len()
            || callable
                .params(self.ctx)
                .iter()
                .zip(&args)
                .any(|(expected, actual)| *expected != self.ctx.value_ty(*actual))
        {
            return Err(TributeControlToCpsError::one(
                POST_CPS_BOUNDARY,
                None,
                Some(location),
                "CPS indirect tail operands differ from the exact closure contract",
            ));
        }
        let tail = func::tail_call_indirect(self.ctx, location, callee, args, Some(signature));
        set_calling_convention(self.ctx, tail.op_ref(), CallingConvention::Cps);
        self.ctx.push_op(block, tail.op_ref());
        Ok(tail.op_ref())
    }

    fn evidence_type(&mut self) -> TypeRef {
        ability::evidence_adt_type_ref(self.ctx)
    }

    fn anyref_type(&mut self) -> TypeRef {
        tribute_rt::anyref(self.ctx).as_type_ref()
    }

    fn done_k_type(&mut self, answer: TypeRef) -> TypeRef {
        cps_done_type(self.ctx, answer)
    }

    fn resumption_type(&mut self, input: TypeRef, answer: TypeRef) -> TypeRef {
        let evidence = self.evidence_type();
        let frame = self.frame_types(answer).reference;
        cps_resume_exact_type(self.ctx, evidence, input, frame)
    }

    fn completion_type(&mut self, value: TypeRef, answer: TypeRef) -> TypeRef {
        let evidence = self.evidence_type();
        let frame = self.frame_types(answer).reference;
        cps_completion_type(self.ctx, evidence, value, frame)
    }

    fn i32_type(&mut self) -> TypeRef {
        self.ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build())
    }

    fn frame_types(&mut self, answer: TypeRef) -> FrameTypes {
        if let Some(frame) = self.frames.get(&answer).copied() {
            return frame;
        }
        let name = Symbol::from_dynamic(&format!("__tribute_continuation_frame_{answer:?}"));
        let reference = cps_continuation_frame_ref_type(self.ctx, name, answer);
        let done = self.done_k_type(answer);
        let evidence = self.evidence_type();
        let anyref = self.anyref_type();
        let i32 = self.i32_type();
        let dispatch = tribute_core::calling_convention::cps_dispatch_type(
            self.ctx, evidence, reference, anyref, i32,
        );
        let layout = cps_continuation_frame_layout_type(self.ctx, name, answer, done, dispatch);
        let frame = FrameTypes {
            reference,
            layout,
            done,
            dispatch,
        };
        self.frames.insert(answer, frame);
        self.frame_layout_aliases.push((name, layout));
        frame
    }

    fn unpack_frame(
        &mut self,
        block: BlockRef,
        location: Location,
        answer: TypeRef,
        frame_value: ValueRef,
    ) -> (ValueRef, ValueRef) {
        let frame = self.frame_types(answer);
        let done = adt::struct_get(self.ctx, location, frame_value, frame.done, frame.layout, 0);
        self.ctx.push_op(block, done.op_ref());
        let dispatch = adt::struct_get(
            self.ctx,
            location,
            frame_value,
            frame.dispatch,
            frame.layout,
            1,
        );
        self.ctx.push_op(block, dispatch.op_ref());
        (done.result(self.ctx), dispatch.result(self.ctx))
    }

    fn pack_frame(
        &mut self,
        block: BlockRef,
        location: Location,
        answer: TypeRef,
        done: ValueRef,
        dispatch: ValueRef,
    ) -> ValueRef {
        let frame = self.frame_types(answer);
        let packed = adt::struct_new(
            self.ctx,
            location,
            [done, dispatch],
            frame.reference,
            frame.layout,
        );
        self.ctx.push_op(block, packed.op_ref());
        packed.result(self.ctx)
    }

    fn build_done_adapter(
        &mut self,
        value_type: TypeRef,
        completion: ValueRef,
        evidence: ValueRef,
        outer_frame: ValueRef,
        location: Location,
    ) -> Result<(OpRef, ValueRef), TributeControlToCpsError> {
        let done_block = self.make_block(location, &[value_type]);
        let value = self.ctx.block_args(done_block)[0];
        self.emit_cps_tail_call_indirect(
            done_block,
            location,
            completion,
            [evidence, outer_frame, value],
        )?;
        let region = self.single_block_region(location, done_block);
        let done_type = self.done_k_type(value_type);
        let done = closure::lambda(
            self.ctx,
            location,
            ordered_external_values(self.ctx, region),
            done_type,
            region,
        );
        set_calling_convention(self.ctx, done.op_ref(), CallingConvention::Cps);
        Ok((done.op_ref(), done.result(self.ctx)))
    }

    #[allow(clippy::too_many_arguments)]
    fn build_rebound_resume(
        &mut self,
        location: Location,
        value_type: TypeRef,
        boundary: TypeRef,
        resume_body: ValueRef,
        completion: ValueRef,
        dispatch_factory: Symbol,
        dispatch_factory_args: DispatchFactoryArgs<'_>,
    ) -> Result<(OpRef, ValueRef), TributeControlToCpsError> {
        let evidence_type = self.evidence_type();
        let anyref = self.anyref_type();
        let boundary_frame = self.frame_types(boundary).reference;
        let block = self.make_block(location, &[evidence_type, boundary_frame, anyref]);
        let args = self.ctx.block_args(block).to_vec();
        let (done_op, done) =
            self.build_done_adapter(value_type, completion, args[0], args[1], location)?;
        self.ctx.push_op(block, done_op);
        let (_, outer_dispatch) = self.unpack_frame(block, location, boundary, args[1]);
        let dispatch_type = self.frame_types(value_type).dispatch;
        let mut factory_args = dispatch_factory_args.prefix.to_vec();
        factory_args.extend([completion, outer_dispatch]);
        factory_args.extend_from_slice(dispatch_factory_args.suffix);
        let dispatch = func::call(
            self.ctx,
            location,
            factory_args,
            dispatch_type,
            dispatch_factory,
        );
        set_calling_convention(self.ctx, dispatch.op_ref(), CallingConvention::Direct);
        self.ctx.push_op(block, dispatch.op_ref());
        let frame = self.pack_frame(block, location, value_type, done, dispatch.result(self.ctx));
        self.emit_cps_tail_call_indirect(block, location, resume_body, [args[0], frame, args[2]])?;
        let region = self.single_block_region(location, block);
        let resume_type = tribute_core::calling_convention::cps_resume_type(
            self.ctx,
            evidence_type,
            boundary_frame,
            anyref,
        );
        let resume = closure::lambda(
            self.ctx,
            location,
            ordered_external_values(self.ctx, region),
            resume_type,
            region,
        );
        set_calling_convention(self.ctx, resume.op_ref(), CallingConvention::Cps);
        Ok((resume.op_ref(), resume.result(self.ctx)))
    }

    fn build_dispatch_adapter_factory(
        &mut self,
        location: Location,
        value_type: TypeRef,
        boundary: TypeRef,
    ) -> Result<Symbol, TributeControlToCpsError> {
        let symbol = self.fresh_helper("make_dispatch_adapter");
        let evidence_type = self.evidence_type();
        let anyref = self.anyref_type();
        let i32_type = self.i32_type();
        let completion_type = self.completion_type(value_type, boundary);
        let outer_dispatch_type = self.frame_types(boundary).dispatch;
        let dispatch_type = self.frame_types(value_type).dispatch;
        let factory_type = core::func(
            self.ctx,
            dispatch_type,
            [completion_type, outer_dispatch_type],
        )
        .as_type_ref();
        let factory_block = self.make_block(location, &[completion_type, outer_dispatch_type]);
        let factory_args = self.ctx.block_args(factory_block).to_vec();
        let value_frame = self.frame_types(value_type).reference;
        let resume_type = tribute_core::calling_convention::cps_resume_type(
            self.ctx,
            evidence_type,
            value_frame,
            anyref,
        );
        let dispatch_block = self.make_block(
            location,
            &[
                evidence_type,
                resume_type,
                i32_type,
                i32_type,
                i32_type,
                anyref,
            ],
        );
        let dispatch_args = self.ctx.block_args(dispatch_block).to_vec();
        let (resume_op, rebound_resume) = self.build_rebound_resume(
            location,
            value_type,
            boundary,
            dispatch_args[1],
            factory_args[0],
            symbol,
            DispatchFactoryArgs {
                prefix: &[],
                suffix: &[],
            },
        )?;
        self.ctx.push_op(dispatch_block, resume_op);
        self.emit_cps_tail_call_indirect(
            dispatch_block,
            location,
            factory_args[1],
            [
                dispatch_args[0],
                rebound_resume,
                dispatch_args[2],
                dispatch_args[3],
                dispatch_args[4],
                dispatch_args[5],
            ],
        )?;
        let dispatch_region = self.single_block_region(location, dispatch_block);
        let dispatch = closure::lambda(
            self.ctx,
            location,
            ordered_external_values(self.ctx, dispatch_region),
            dispatch_type,
            dispatch_region,
        );
        set_calling_convention(self.ctx, dispatch.op_ref(), CallingConvention::Cps);
        self.ctx.push_op(factory_block, dispatch.op_ref());
        let ret = func::r#return(self.ctx, location, [dispatch.result(self.ctx)]);
        self.ctx.push_op(factory_block, ret.op_ref());
        let factory_region = self.single_block_region(location, factory_block);
        let factory = func::func(self.ctx, location, symbol, factory_type, factory_region);
        set_calling_convention(self.ctx, factory.op_ref(), CallingConvention::Direct);
        self.ctx.push_op(self.module_block, factory.op_ref());
        Ok(symbol)
    }

    fn frame_for_suffix(
        &mut self,
        block: BlockRef,
        location: Location,
        value_type: TypeRef,
        flow: &Flow,
        suffix: ValueRef,
    ) -> Result<ValueRef, TributeControlToCpsError> {
        let outer = flow.exit_k.ok_or_else(|| {
            TributeControlToCpsError::one(
                POST_CPS_BOUNDARY,
                None,
                Some(location),
                "CPS transfer has no verified ContinuationFrame",
            )
        })?;
        let evidence = flow.evidence.ok_or_else(|| {
            TributeControlToCpsError::one(
                POST_CPS_BOUNDARY,
                None,
                Some(location),
                "CPS transfer has no verified evidence",
            )
        })?;
        let (done_op, done) =
            self.build_done_adapter(value_type, suffix, evidence, outer, location)?;
        self.ctx.push_op(block, done_op);
        let outer_dispatch = match flow.dispatch {
            Some(dispatch) => dispatch,
            None => {
                self.unpack_frame(block, location, flow.answer_type, outer)
                    .1
            }
        };
        let dispatch_factory =
            self.build_dispatch_adapter_factory(location, value_type, flow.answer_type)?;
        let completion_type = self.completion_type(value_type, flow.answer_type);
        let typed_completion =
            core::unrealized_conversion_cast(self.ctx, location, suffix, completion_type);
        self.ctx.push_op(block, typed_completion.op_ref());
        let dispatch_type = self.frame_types(value_type).dispatch;
        let dispatch = func::call(
            self.ctx,
            location,
            [typed_completion.result(self.ctx), outer_dispatch],
            dispatch_type,
            dispatch_factory,
        );
        set_calling_convention(self.ctx, dispatch.op_ref(), CallingConvention::Direct);
        self.ctx.push_op(block, dispatch.op_ref());
        Ok(self.pack_frame(block, location, value_type, done, dispatch.result(self.ctx)))
    }

    fn generated_continuation_type(&mut self, function: TypeRef) -> TypeRef {
        physical_closure_type_with_environment_index(self.ctx, function, CallingConvention::Cps, 0)
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
            let frame = self.frame_types(result).reference;
            let lowered_params = abi.lowered_params(evidence, frame);
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
        let frame = self.frame_types(result).reference;
        let abi = CallableAbi::new(convention, params, result);
        let params = abi.lowered_params(evidence, frame);
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
        let is_cps = flow.convention == CallingConvention::Cps;
        if is_cps && result_types.len() > 1 {
            return Err(TributeControlToCpsError::one(
                POST_CPS_BOUNDARY,
                Some(source),
                Some(self.ctx.op(source).location),
                "effectful scf.if requires zero or one result inside a CPS callable",
            ));
        }
        let location = self.ctx.op(source).location;
        let continuation = if is_cps {
            Some(if let [source_result_type] = result_types.as_slice() {
                self.build_suffix_continuation(
                    source_ops,
                    index + 1,
                    self.ctx.op_result(source, 0),
                    *source_result_type,
                    mapping,
                    flow,
                    location,
                )?
            } else {
                self.build_void_suffix_continuation(source_ops, index + 1, mapping, flow, location)?
            })
        } else {
            None
        };
        if let Some(continuation) = continuation {
            let continuation_op = match self.ctx.value_def(continuation) {
                trunk_ir::ValueDef::OpResult(op, _) => op,
                _ => unreachable!("structured continuation is a closure.lambda"),
            };
            self.ctx.push_op(block, continuation_op);
        }

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
            let branch_flow = match continuation {
                Some(continuation) if result_types.is_empty() => Flow {
                    void_exit_k: Some(continuation),
                    ..flow.clone()
                },
                Some(continuation) => {
                    let [source_result_type] = result_types.as_slice() else {
                        unreachable!("non-void CPS branch has one result")
                    };
                    let result_type = self.convert_type(*source_result_type);
                    let frame =
                        self.frame_for_suffix(block, location, result_type, flow, continuation)?;
                    Flow {
                        exit_k: Some(frame),
                        root_exit_k: Some(frame),
                        answer_type: result_type,
                        ..flow.clone()
                    }
                }
                None => Flow {
                    preserve_scf_yield: true,
                    ..flow.clone()
                },
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
        let condition = self.ctx.op_operands(source)[0];
        let condition = mapping.get(&condition).copied().unwrap_or(condition);
        if continuation.is_some() {
            let never = self.never_type();
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
            return Ok(());
        }

        let mut builder =
            OperationDataBuilder::new(location, Symbol::new("scf"), Symbol::new("if"))
                .operand(condition);
        for result_type in result_types {
            builder = builder.result(self.convert_type(result_type));
        }
        let data = builder
            .region(*then_region)
            .region(*else_region)
            .build(self.ctx);
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
                void_exit_k: Some(continuation),
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
            let exit_k = flow.exit_k.ok_or_else(|| {
                TributeControlToCpsError::one(
                    POST_CPS_BOUNDARY,
                    None,
                    Some(location),
                    "CPS region has no verified exit continuation",
                )
            })?;
            let (done, _) = self.unpack_frame(block, location, flow.answer_type, exit_k);
            self.emit_cps_tail_call_indirect(block, location, done, [value])?;
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
        if let Some(void_exit) = flow.void_exit_k {
            let evidence = flow.evidence.ok_or_else(|| {
                TributeControlToCpsError::one(
                    POST_CPS_BOUNDARY,
                    None,
                    Some(location),
                    "zero-result suffix has no verified evidence",
                )
            })?;
            let frame = flow.exit_k.ok_or_else(|| {
                TributeControlToCpsError::one(
                    POST_CPS_BOUNDARY,
                    None,
                    Some(location),
                    "zero-result suffix has no verified ContinuationFrame",
                )
            })?;
            self.emit_cps_tail_call_indirect(block, location, void_exit, [evidence, frame])?;
            return Ok(());
        }
        let exit_k = flow.exit_k.ok_or_else(|| {
            TributeControlToCpsError::one(
                POST_CPS_BOUNDARY,
                None,
                Some(location),
                "structured region has no verified exit continuation",
            )
        })?;
        let (done, _) = self.unpack_frame(block, location, flow.answer_type, exit_k);
        self.emit_cps_tail_call_indirect(block, location, done, std::iter::empty::<ValueRef>())?;
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
        let evidence_type = self.evidence_type();
        let frame_type = self.frame_types(flow.answer_type).reference;
        let block = self.make_block(location, &[evidence_type, frame_type]);
        let mut suffix_mapping = mapping.clone();
        let suffix_flow = Flow {
            evidence: Some(self.ctx.block_args(block)[0]),
            exit_k: Some(self.ctx.block_args(block)[1]),
            root_exit_k: Some(self.ctx.block_args(block)[1]),
            void_exit_k: None,
            ..flow.clone()
        };
        self.convert_sequence(
            source_ops.to_vec(),
            start,
            block,
            &mut suffix_mapping,
            &suffix_flow,
        )?;
        let region = self.single_block_region(location, block);
        let captures = ordered_external_values(self.ctx, region);
        let never = self.never_type();
        let function = core::func(self.ctx, never, [evidence_type, frame_type]).as_type_ref();
        let closure_type = self.generated_continuation_type(function);
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
        let frame = self.frame_types(result).reference;
        let abi = CallableAbi::new(convention, source_param_types.clone(), result);
        let params = abi.lowered_params(evidence, frame);
        let body_source = self.ctx.op(source).regions[0];
        let entry_source = self.ctx.region(body_source).blocks[0];
        let block = self.make_block(location, &params);
        let mut body_mapping = mapping.clone();
        let offset = abi.source_param_offset();
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
        let exit_k = convention
            .needs_done_k()
            .then(|| self.ctx.block_args(block)[usize::from(convention.needs_evidence())]);
        let flow = Flow {
            convention,
            evidence: evidence_value,
            exit_k,
            root_exit_k: exit_k,
            void_exit_k: None,
            dispatch: None,
            answer_type: result,
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
        let frame_type = self.frame_types(flow.answer_type).reference;
        let block = self.make_block(location, &[evidence_type, frame_type, result_type]);
        let mut body_mapping = mapping.clone();
        body_mapping.insert(source_result, self.ctx.block_args(block)[2]);
        let suffix_flow = Flow {
            evidence: Some(self.ctx.block_args(block)[0]),
            exit_k: Some(self.ctx.block_args(block)[1]),
            root_exit_k: Some(self.ctx.block_args(block)[1]),
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
        let captures = ordered_external_values(self.ctx, region);
        let closure_ty = self.completion_type(result_type, flow.answer_type);
        let lambda = closure::lambda(self.ctx, location, captures, closure_ty, region);
        set_calling_convention(self.ctx, lambda.op_ref(), CallingConvention::Cps);
        Ok(lambda.result(self.ctx))
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
        let frame_ty = self.frame_types(result).reference;
        let abi = CallableAbi::new(result_convention, source_params.clone(), result);
        let logical_params = abi.lowered_params(evidence_ty, frame_ty);
        let env_ty = self.anyref_type();
        let physical_params = abi.interpose_environment(&logical_params, env_ty);
        let physical_result = if result_convention == CallingConvention::Cps {
            self.never_type()
        } else {
            result
        };
        let adapter_ty =
            core::func(self.ctx, physical_result, physical_params.clone()).as_type_ref();
        let block = self.make_block(location, &physical_params);
        let args = self.ctx.block_args(block).to_vec();
        let evidence_offset = usize::from(result_convention.needs_evidence());
        // The environment is interposed immediately after optional evidence,
        // so a present ContinuationFrame is the following slot.
        let frame_offset = evidence_offset + 1;
        let source_offset = usize::from(result_convention.needs_evidence())
            + 1
            + usize::from(result_convention.needs_done_k());
        let mut target_args = Vec::new();
        if target.convention.needs_evidence() {
            target_args.push(args[0]);
        }
        if target.convention.needs_done_k() {
            target_args.push(args[frame_offset]);
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
                let frame = args[frame_offset];
                let (done_k, _) = self.unpack_frame(block, location, result, frame);
                self.emit_cps_tail_call_indirect(block, location, done_k, [call.result(self.ctx)])?;
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
        let frame_type = self.frame_types(flow.answer_type).reference;
        let block = self.make_block(location, &[evidence_type, frame_type, arg_type]);
        let mut body_mapping = mapping.clone();
        body_mapping.insert(source_arg, self.ctx.block_args(block)[2]);
        let completion_flow = Flow {
            convention: CallingConvention::Cps,
            evidence: Some(self.ctx.block_args(block)[0]),
            exit_k: Some(self.ctx.block_args(block)[1]),
            root_exit_k: Some(self.ctx.block_args(block)[1]),
            void_exit_k: None,
            dispatch: None,
            answer_type: flow.answer_type,
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
        let captures = ordered_external_values(self.ctx, body);
        let closure_type = self.completion_type(arg_type, flow.answer_type);
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
        let frame_type = self.frame_types(flow.answer_type).reference;
        let block = self.make_block(location, &[evidence_type, frame_type, input_type]);
        let resume_evidence = self.ctx.block_args(block)[0];
        let resume_frame = self.ctx.block_args(block)[1];
        let resume_input = self.ctx.block_args(block)[2];
        let mut body_mapping = mapping.clone();
        body_mapping.insert(source_result, resume_input);
        let mut suffix_flow = flow.clone();
        // A resumed suffix receives the dynamic frame directly. It carries
        // both the completion and lexical-dispatch provenance, so no closure
        // graph cloning or ambient-evidence reconstruction is permitted.
        suffix_flow.evidence = Some(resume_evidence);
        suffix_flow.exit_k = Some(resume_frame);
        suffix_flow.root_exit_k = Some(resume_frame);
        self.convert_sequence(
            source_ops.to_vec(),
            start,
            block,
            &mut body_mapping,
            &suffix_flow,
        )?;
        let region = self.single_block_region(location, block);
        let captures = ordered_external_values(self.ctx, region);
        let closure_type = self.resumption_type(input_type, flow.answer_type);
        let lambda = closure::lambda(self.ctx, location, captures, closure_type, region);
        set_calling_convention(self.ctx, lambda.op_ref(), CallingConvention::Cps);
        Ok((lambda.op_ref(), lambda.result(self.ctx)))
    }

    #[allow(clippy::too_many_arguments)]
    fn build_exact_handler_token(
        &mut self,
        location: Location,
        raw_resume: ValueRef,
        input_type: TypeRef,
        body_type: TypeRef,
        answer_type: TypeRef,
        completion: ValueRef,
        dispatch_factory: Symbol,
        dispatch_factory_args: DispatchFactoryArgs<'_>,
    ) -> Result<(OpRef, ValueRef), TributeControlToCpsError> {
        let evidence_type = self.evidence_type();
        let frame_type = self.frame_types(answer_type).reference;
        let block = self.make_block(location, &[evidence_type, frame_type, input_type]);
        let args = self.ctx.block_args(block).to_vec();
        let anyref = self.anyref_type();
        let erased_input = core::unrealized_conversion_cast(self.ctx, location, args[2], anyref);
        self.ctx.push_op(block, erased_input.op_ref());
        let (rebound_op, rebound) = self.build_rebound_resume(
            location,
            body_type,
            answer_type,
            raw_resume,
            completion,
            dispatch_factory,
            dispatch_factory_args,
        )?;
        self.ctx.push_op(block, rebound_op);
        self.emit_cps_tail_call_indirect(
            block,
            location,
            rebound,
            [args[0], args[1], erased_input.result(self.ctx)],
        )?;
        let region = self.single_block_region(location, block);
        let token_type = self.resumption_type(input_type, answer_type);
        let lambda = closure::lambda(
            self.ctx,
            location,
            ordered_external_values(self.ctx, region),
            token_type,
            region,
        );
        set_calling_convention(self.ctx, lambda.op_ref(), CallingConvention::Cps);
        Ok((lambda.op_ref(), lambda.result(self.ctx)))
    }

    /// Build an immutable factory for one lexical handle dispatcher. Each
    /// resumption calls this factory again with its current outer dispatcher,
    /// so the resumed body retains this handle's arm-selection boundary.
    fn build_local_dispatcher_factory(
        &mut self,
        location: Location,
        arms: &[HandlerArmInfo],
        body_type: TypeRef,
        answer_type: TypeRef,
    ) -> Result<Symbol, TributeControlToCpsError> {
        let symbol = self.fresh_helper("make_local_dispatch");
        let completion_type = self.completion_type(body_type, answer_type);
        let i32_type = self.i32_type();
        let parent_dispatch_type = self.frame_types(answer_type).dispatch;
        let dispatch_type = self.frame_types(body_type).dispatch;
        let mut params = vec![completion_type, parent_dispatch_type, i32_type];
        params.extend(arms.iter().map(|arm| self.ctx.value_ty(arm.value)));
        let factory_type = core::func(self.ctx, dispatch_type, params.clone()).as_type_ref();
        let factory_block = self.make_block(location, &params);
        let factory_args = self.ctx.block_args(factory_block).to_vec();
        let mut factory_arms = arms.to_vec();
        for (arm, value) in factory_arms
            .iter_mut()
            .zip(factory_args[3..].iter().copied())
        {
            arm.value = value;
        }
        let (dispatcher_op, dispatcher) = self.build_local_dispatcher_instance(
            location,
            &factory_arms,
            body_type,
            answer_type,
            factory_args[0],
            factory_args[1],
            symbol,
            factory_args[2],
            factory_args[3..].to_vec(),
        )?;
        self.ctx.push_op(factory_block, dispatcher_op);
        let ret = func::r#return(self.ctx, location, [dispatcher]);
        self.ctx.push_op(factory_block, ret.op_ref());
        let region = self.single_block_region(location, factory_block);
        let factory = func::func(self.ctx, location, symbol, factory_type, region);
        set_calling_convention(self.ctx, factory.op_ref(), CallingConvention::Direct);
        self.ctx.push_op(self.module_block, factory.op_ref());
        Ok(symbol)
    }

    #[allow(clippy::too_many_arguments)]
    fn build_local_foreign_dispatch(
        &mut self,
        location: Location,
        body_type: TypeRef,
        answer_type: TypeRef,
        completion: ValueRef,
        parent_dispatch: ValueRef,
        factory: Symbol,
        factory_prefix_args: &[ValueRef],
        factory_suffix_args: &[ValueRef],
    ) -> Result<(OpRef, ValueRef), TributeControlToCpsError> {
        let evidence_type = self.evidence_type();
        let anyref = self.anyref_type();
        let i32_type = self.i32_type();
        let body_frame = self.frame_types(body_type).reference;
        let resume_type = tribute_core::calling_convention::cps_resume_type(
            self.ctx,
            evidence_type,
            body_frame,
            anyref,
        );
        let block = self.make_block(
            location,
            &[
                evidence_type,
                resume_type,
                i32_type,
                i32_type,
                i32_type,
                anyref,
            ],
        );
        let args = self.ctx.block_args(block).to_vec();
        let (rebound_op, rebound) = self.build_rebound_resume(
            location,
            body_type,
            answer_type,
            args[1],
            completion,
            factory,
            DispatchFactoryArgs {
                prefix: factory_prefix_args,
                suffix: factory_suffix_args,
            },
        )?;
        self.ctx.push_op(block, rebound_op);
        self.emit_cps_tail_call_indirect(
            block,
            location,
            parent_dispatch,
            [args[0], rebound, args[2], args[3], args[4], args[5]],
        )?;
        let region = self.single_block_region(location, block);
        let dispatch_type = self.frame_types(body_type).dispatch;
        let lambda = closure::lambda(
            self.ctx,
            location,
            ordered_external_values(self.ctx, region),
            dispatch_type,
            region,
        );
        set_calling_convention(self.ctx, lambda.op_ref(), CallingConvention::Cps);
        Ok((lambda.op_ref(), lambda.result(self.ctx)))
    }

    #[allow(clippy::too_many_arguments)]
    fn build_local_dispatcher_instance(
        &mut self,
        location: Location,
        arms: &[HandlerArmInfo],
        body_type: TypeRef,
        answer_type: TypeRef,
        completion: ValueRef,
        parent_dispatch: ValueRef,
        factory: Symbol,
        local_prompt: ValueRef,
        factory_args: Vec<ValueRef>,
    ) -> Result<(OpRef, ValueRef), TributeControlToCpsError> {
        let evidence_type = self.evidence_type();
        let anyref = self.anyref_type();
        let i32_type = self.i32_type();
        let body_frame = self.frame_types(body_type).reference;
        let resume_type = tribute_core::calling_convention::cps_resume_type(
            self.ctx,
            evidence_type,
            body_frame,
            anyref,
        );
        let block = self.make_block(
            location,
            &[
                evidence_type,
                resume_type,
                i32_type,
                i32_type,
                i32_type,
                anyref,
            ],
        );
        let args = self.ctx.block_args(block).to_vec();
        let mut factory_suffix_args = vec![local_prompt];
        factory_suffix_args.extend(factory_args.iter().copied());
        let (foreign_op, foreign_dispatch) = self.build_local_foreign_dispatch(
            location,
            body_type,
            answer_type,
            completion,
            parent_dispatch,
            factory,
            &[],
            &factory_suffix_args,
        )?;
        self.ctx.push_op(block, foreign_op);
        let switch_block = self.make_block(location, &[]);
        let i1_type = self
            .ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i1")).build());
        for arm in arms.iter().filter(|arm| arm.kind == Symbol::new("op")) {
            let case_block = self.make_block(location, &[]);
            let same_prompt = arith::cmpi(
                self.ctx,
                location,
                args[2],
                local_prompt,
                i1_type,
                Symbol::new("eq"),
            );
            self.ctx.push_op(case_block, same_prompt.op_ref());

            let local_block = self.make_block(location, &[]);
            let mut call_args = vec![args[0]];
            call_args.extend(self.unpack_handler_payload(local_block, location, args[5], arm));
            if arm.has_resume_token {
                let input_type = *arm.params.last().expect("resumptive arm has a token");
                let token_input = cps_closure_function_type(self.ctx, input_type)
                    .and_then(|function| core::Func::from_type_ref(self.ctx, function))
                    .and_then(|function| function.params(self.ctx).get(2).copied())
                    .ok_or_else(|| {
                        TributeControlToCpsError::one(
                            POST_CPS_BOUNDARY,
                            None,
                            Some(location),
                            "handler resume token lacks an exact callable input",
                        )
                    })?;
                let (token_op, token) = self.build_exact_handler_token(
                    location,
                    args[1],
                    token_input,
                    body_type,
                    answer_type,
                    completion,
                    factory,
                    DispatchFactoryArgs {
                        prefix: &[],
                        suffix: &factory_suffix_args,
                    },
                )?;
                self.ctx.push_op(local_block, token_op);
                call_args.push(token);
            }
            self.emit_cps_tail_call_indirect(local_block, location, arm.value, call_args)?;
            let local_region = self.single_block_region(location, local_block);
            let fallback_block = self.make_block(location, &[]);
            self.emit_cps_tail_call_indirect(
                fallback_block,
                location,
                foreign_dispatch,
                [args[0], args[1], args[2], args[3], args[4], args[5]],
            )?;
            let fallback_region = self.single_block_region(location, fallback_block);
            let never = self.never_type();
            let choose = scf::r#if(
                self.ctx,
                location,
                same_prompt.result(self.ctx),
                never,
                local_region,
                fallback_region,
            );
            self.ctx.push_op(case_block, choose.op_ref());
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
        let default_block = self.make_block(location, &[]);
        self.emit_cps_tail_call_indirect(
            default_block,
            location,
            foreign_dispatch,
            [args[0], args[1], args[2], args[3], args[4], args[5]],
        )?;
        let foreign_region = self.single_block_region(location, default_block);
        let default = scf::default(self.ctx, location, foreign_region);
        self.ctx.push_op(switch_block, default.op_ref());
        let switch_region = self.single_block_region(location, switch_block);
        let switch = scf::switch(self.ctx, location, args[4], switch_region);
        self.ctx.push_op(block, switch.op_ref());
        let region = self.single_block_region(location, block);
        let dispatch_type = self.frame_types(body_type).dispatch;
        let lambda = closure::lambda(
            self.ctx,
            location,
            ordered_external_values(self.ctx, region),
            dispatch_type,
            region,
        );
        set_calling_convention(self.ctx, lambda.op_ref(), CallingConvention::Cps);
        Ok((lambda.op_ref(), lambda.result(self.ctx)))
    }

    fn build_one_shot_wrapper(
        &mut self,
        raw_continuation: ValueRef,
        input_type: TypeRef,
        answer_type: TypeRef,
        location: Location,
    ) -> Result<(Vec<OpRef>, ValueRef), TributeControlToCpsError> {
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
        let frame_type = self.frame_types(answer_type).reference;
        let anyref = self.anyref_type();
        // The dispatcher ABI is existential only at this boundary. Keep the
        // captured continuation exact, recover this operation's declared input,
        // then transfer in proper tail position.
        let block = self.make_block(location, &[evidence_type, frame_type, anyref]);
        let args = self.ctx.block_args(block).to_vec();
        let input = if type_is(self.ctx, input_type, "core", "nil") {
            // Nil has no physical payload: its exact resumption receives the
            // canonical unit instead of an erased runtime value.
            let unit = arith::r#const(self.ctx, location, input_type, Attribute::Unit);
            self.ctx.push_op(block, unit.op_ref());
            unit.result(self.ctx)
        } else if type_is(self.ctx, input_type, "adt", "typeref") {
            // Dynamic effect values recover nominal references only through
            // their declared type, preserving the typed ownership boundary.
            let recovered = adt::ref_cast(self.ctx, location, args[2], input_type, input_type);
            self.ctx.push_op(block, recovered.op_ref());
            recovered.result(self.ctx)
        } else {
            let recovered =
                core::unrealized_conversion_cast(self.ctx, location, args[2], input_type);
            self.ctx.push_op(block, recovered.op_ref());
            recovered.result(self.ctx)
        };
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
        self.emit_cps_tail_call_indirect(
            enter_block,
            location,
            raw_continuation,
            [args[0], args[1], input],
        )?;
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
        let closure_type = tribute_core::calling_convention::cps_resume_type(
            self.ctx,
            evidence_type,
            frame_type,
            anyref,
        );
        let wrapper = closure::lambda(self.ctx, location, captures, closure_type, region);
        set_calling_convention(self.ctx, wrapper.op_ref(), CallingConvention::Cps);
        Ok((
            vec![not_consumed.op_ref(), state.op_ref(), wrapper.op_ref()],
            wrapper.result(self.ctx),
        ))
    }

    fn build_reject_continuation(
        &mut self,
        input_type: TypeRef,
        answer_type: TypeRef,
        location: Location,
    ) -> (OpRef, ValueRef) {
        let input_type = self.convert_type(input_type);
        let evidence_type = self.evidence_type();
        let frame_type = self.frame_types(answer_type).reference;
        let block = self.make_block(location, &[evidence_type, frame_type, input_type]);
        let unreachable = func::unreachable(self.ctx, location);
        self.ctx.push_op(block, unreachable.op_ref());
        let region = self.single_block_region(location, block);
        let closure_type = self.resumption_type(input_type, answer_type);
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
        let old_result = self.ctx.op_result(source, 0);
        let input_type = self.ctx.op_result_types(source)[0];
        let continuation = if type_is(self.ctx, input_type, "core", "never") {
            let (op, value) =
                self.build_reject_continuation(input_type, flow.answer_type, location);
            self.ctx.push_op(block, op);
            value
        } else {
            let (raw_op, raw) = self.build_raw_resumption(
                source_ops,
                index + 1,
                old_result,
                input_type,
                mapping,
                flow,
                location,
            )?;
            self.ctx.push_op(block, raw_op);
            let converted_input = self.convert_type(input_type);
            let (ops, one_shot) =
                self.build_one_shot_wrapper(raw, converted_input, flow.answer_type, location)?;
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
        let evidence = self.current_evidence(source, flow)?;
        let frame = flow.exit_k.ok_or_else(|| {
            TributeControlToCpsError::one(
                POST_CPS_BOUNDARY,
                Some(source),
                Some(location),
                "general operation has no verified Dispatch boundary",
            )
        })?;
        let dispatch = match flow.dispatch {
            Some(dispatch) => dispatch,
            None => {
                self.unpack_frame(block, location, flow.answer_type, frame)
                    .1
            }
        };
        let perform = ability::perform(
            self.ctx,
            location,
            evidence,
            dispatch,
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
            flow,
            location,
        )?;
        let suffix_op = match self.ctx.value_def(suffix) {
            trunk_ir::ValueDef::OpResult(op, _) => op,
            _ => unreachable!("suffix is produced by closure.lambda"),
        };
        self.ctx.push_op(block, suffix_op);
        let resume_result = self.convert_type(result_type);
        let resume_frame = self.frame_for_suffix(block, location, resume_result, flow, suffix)?;
        self.emit_cps_tail_call_indirect(
            block,
            location,
            token,
            [self.current_evidence(source, flow)?, resume_frame, value],
        )?;
        Ok(())
    }

    fn lower_handler_arm(
        &mut self,
        source: OpRef,
        outer_mapping: &HashMap<ValueRef, ValueRef>,
        handle_exit: ValueRef,
        handle_answer: TypeRef,
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
        params.extend_from_slice(&converted_args);
        let block = self.make_block(location, &params);
        let mut mapping = outer_mapping.clone();
        for (old, new) in source_args
            .into_iter()
            .zip(self.ctx.block_args(block)[1..].iter().copied())
        {
            mapping.insert(old, new);
        }
        let evidence = self.ctx.block_args(block)[0];
        let convention = if kind == Symbol::new("fn") {
            CallingConvention::EvidenceDirect
        } else {
            CallingConvention::Cps
        };
        let flow = Flow {
            convention,
            evidence: Some(evidence),
            exit_k: (convention == CallingConvention::Cps).then_some(handle_exit),
            root_exit_k: (convention == CallingConvention::Cps).then_some(handle_exit),
            void_exit_k: None,
            dispatch: None,
            answer_type: handle_answer,
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
        let anyref = self.anyref_type();
        let payload_type = ability::operation_payload_type_ref(
            self.ctx,
            arm.ability_ref,
            arm.op_name,
            value_params.iter().map(|_| anyref),
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
                    anyref,
                    payload_type,
                    index as u32,
                );
                self.ctx.push_op(block, get.op_ref());
                let recovered =
                    core::unrealized_conversion_cast(self.ctx, location, get.result(self.ctx), ty);
                self.ctx.push_op(block, recovered.op_ref());
                recovered.result(self.ctx)
            })
            .collect()
    }

    fn build_handler_dispatcher(
        &mut self,
        location: Location,
        arms: &[HandlerArmInfo],
        general: bool,
    ) -> Result<(OpRef, ValueRef), TributeControlToCpsError> {
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
                self.emit_cps_tail_call_indirect(case_block, location, arm.value, call_args)?;
            } else {
                let signature = physical_closure_function_type(
                    self.ctx,
                    self.ctx.value_ty(arm.value),
                    CallingConvention::EvidenceDirect,
                )
                .ok_or_else(|| {
                    TributeControlToCpsError::one(
                        POST_CPS_BOUNDARY,
                        Some(arm.op),
                        Some(location),
                        "fn handler indirect callee has no exact provenance-bearing closure contract",
                    )
                })?;
                let call = func::call_indirect(
                    self.ctx,
                    location,
                    arm.value,
                    call_args,
                    arm.operation_result,
                    Some(signature),
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
        let function = core::func(self.ctx, result, params).as_type_ref();
        let convention = if general {
            CallingConvention::Cps
        } else {
            CallingConvention::EvidenceDirect
        };
        let closure_type = physical_closure_type(self.ctx, function, convention);
        let lambda = closure::lambda(self.ctx, location, captures, closure_type, region);
        set_calling_convention(self.ctx, lambda.op_ref(), convention);
        Ok((lambda.op_ref(), lambda.result(self.ctx)))
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
        let handle_frame =
            self.frame_for_suffix(block, location, handle_answer, flow, after_handle)?;

        let regions = self.ctx.op(source).regions.to_vec();
        let [body_source, completion_source, handlers_region] = regions.as_slice() else {
            unreachable!("pre-CPS validation checked handle regions");
        };
        let (body_source, completion_source, handlers_region) =
            (*body_source, *completion_source, *handlers_region);
        let handlers_block = self.ctx.region(handlers_region).blocks[0];
        let mut handler_arms = Vec::new();
        let source_handlers = self.ctx.block(handlers_block).ops.to_vec();
        for handler in source_handlers {
            let arm = self.lower_handler_arm(handler, mapping, handle_frame, handle_answer)?;
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
            let (tr_op, tr_value) =
                self.build_handler_dispatcher(location, &ability_arms, false)?;
            self.ctx.push_op(block, tr_op);
            dispatchers.push(tr_value);
            let (handler_op, handler_value) =
                self.build_handler_dispatcher(location, &ability_arms, true)?;
            self.ctx.push_op(block, handler_op);
            dispatchers.push(handler_value);
        }

        let evidence_type = self.evidence_type();
        let i32_type = self.i32_type();
        let prompt = effect::fresh_prompt_tag(self.ctx, location, i32_type);
        self.ctx.push_op(block, prompt.op_ref());
        let body_block = self.make_block(location, &[evidence_type]);
        let extended_evidence = self.ctx.block_args(body_block)[0];
        let mut body_mapping = mapping.clone();
        let body_flow_base = Flow {
            convention: CallingConvention::Cps,
            evidence: Some(extended_evidence),
            exit_k: Some(handle_frame),
            root_exit_k: Some(handle_frame),
            void_exit_k: None,
            dispatch: None,
            answer_type: handle_answer,
            preserve_scf_yield: false,
        };
        let (completion_op, completion_k) = self.build_completion_continuation(
            completion_source,
            &body_mapping,
            &body_flow_base,
            location,
        )?;
        self.ctx.push_op(body_block, completion_op);
        let completion_input = self.convert_type(
            self.ctx.value_ty(
                self.ctx
                    .block_args(self.ctx.region(completion_source).blocks[0])[0],
            ),
        );
        let completion_frame = self.frame_for_suffix(
            body_block,
            location,
            completion_input,
            &body_flow_base,
            completion_k,
        )?;
        let (_, parent_dispatch) =
            self.unpack_frame(body_block, location, handle_answer, handle_frame);
        let local_dispatch_factory = self.build_local_dispatcher_factory(
            location,
            &handler_arms,
            completion_input,
            handle_answer,
        )?;
        let mut local_dispatch_args = vec![completion_k, parent_dispatch, prompt.result(self.ctx)];
        local_dispatch_args.extend(handler_arms.iter().map(|arm| arm.value));
        let local_dispatch_type = self.frame_types(completion_input).dispatch;
        let local_dispatch_op = func::call(
            self.ctx,
            location,
            local_dispatch_args,
            local_dispatch_type,
            local_dispatch_factory,
        );
        set_calling_convention(
            self.ctx,
            local_dispatch_op.op_ref(),
            CallingConvention::Direct,
        );
        let local_dispatch = local_dispatch_op.result(self.ctx);
        self.ctx.push_op(body_block, local_dispatch_op.op_ref());
        let body_flow = Flow {
            exit_k: Some(completion_frame),
            root_exit_k: Some(completion_frame),
            dispatch: Some(local_dispatch),
            answer_type: completion_input,
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
            prompt.result(self.ctx),
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
            if dialect != Symbol::new("tribute_control") {
                let cloned = self.clone_plain_op(source, mapping)?;
                self.ctx.push_op(block, cloned);
                index += 1;
                continue;
            }

            match name.with_str(|value| value.to_owned()).as_str() {
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
                        .current_func(target_symbol)
                        .expect("pre-CPS validation resolved direct callee in this module");
                    if target.convention == CallingConvention::Cps {
                        if flow.convention != CallingConvention::Cps {
                            return Err(TributeControlToCpsError::one(
                                POST_CPS_BOUNDARY,
                                Some(source),
                                Some(location),
                                "a non-CPS callable cannot call a CPS target",
                            ));
                        }
                        let old_result = self.ctx.op_result(source, 0);
                        let result_type = self.ctx.op_result_types(source)[0];
                        let continuation = self.build_suffix_continuation(
                            &source_ops,
                            index + 1,
                            old_result,
                            result_type,
                            mapping,
                            flow,
                            location,
                        )?;
                        let continuation_op = match self.ctx.value_def(continuation) {
                            trunk_ir::ValueDef::OpResult(op, _) => op,
                            _ => unreachable!("continuation is produced by closure.lambda"),
                        };
                        self.ctx.push_op(block, continuation_op);
                        let converted_result = self.convert_type(result_type);
                        let frame = self.frame_for_suffix(
                            block,
                            location,
                            converted_result,
                            flow,
                            continuation,
                        )?;
                        let mut args = vec![self.current_evidence(source, flow)?, frame];
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
                        let continuation = self.build_suffix_continuation(
                            &source_ops,
                            index + 1,
                            old_result,
                            result_type,
                            mapping,
                            flow,
                            location,
                        )?;
                        let continuation_op = match self.ctx.value_def(continuation) {
                            trunk_ir::ValueDef::OpResult(op, _) => op,
                            _ => unreachable!("continuation is produced by closure.lambda"),
                        };
                        self.ctx.push_op(block, continuation_op);
                        let converted_result = self.convert_type(result_type);
                        let frame = self.frame_for_suffix(
                            block,
                            location,
                            converted_result,
                            flow,
                            continuation,
                        )?;
                        let mut args = vec![self.current_evidence(source, flow)?, frame];
                        args.extend(
                            source_args
                                .iter()
                                .map(|arg| mapping.get(arg).copied().unwrap_or(*arg)),
                        );
                        self.emit_cps_tail_call_indirect(block, location, callee, args)?;
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
                    let call =
                        func::call_indirect(self.ctx, location, callee, args, result_type, None);
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
        let frame_ty = self.frame_types(source_result).reference;
        let params = abi.lowered_params(evidence_ty, frame_ty);
        let block = self.make_block(location, &params);
        let mut mapping = HashMap::new();
        for (old, new) in self.ctx.block_args(source_block).to_vec().into_iter().zip(
            self.ctx.block_args(block)[abi.source_param_offset()..]
                .iter()
                .copied(),
        ) {
            mapping.insert(old, new);
        }
        let evidence = info
            .convention
            .needs_evidence()
            .then(|| self.ctx.block_args(block)[0]);
        let exit_k = info
            .convention
            .needs_done_k()
            .then(|| self.ctx.block_args(block)[usize::from(info.convention.needs_evidence())]);
        let flow = Flow {
            convention: info.convention,
            evidence,
            exit_k,
            root_exit_k: exit_k,
            void_exit_k: None,
            dispatch: None,
            answer_type: source_result,
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
) -> Result<(), TributeControlToCpsError> {
    if let Err(error) = verify_tribute_control_post_cps(ctx, candidate) {
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
    compiler_intrinsics: &[tribute_control::CompilerIntrinsicDeclaration],
) -> Result<(), TributeControlToCpsError> {
    verify_tribute_control_pre_cps(ctx, module, declarations, compiler_intrinsics)?;
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
        converted_aliases.extend(converter.frame_layout_aliases.iter().copied());
    }
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
    verify_candidate_or_restore_aliases(ctx, candidate, &source_aliases)?;

    ctx.detach_region(new_region);
    ctx.remove_op(candidate.op());
    ctx.detach_region(source_region);
    ctx.op_mut(module.op()).regions.push(new_region);
    ctx.region_mut(new_region).parent_op = Some(module.op());
    if let Err(error) = verify_tribute_control_post_cps(ctx, module) {
        ctx.detach_region(new_region);
        ctx.op_mut(module.op()).regions.push(source_region);
        ctx.region_mut(source_region).parent_op = Some(module.op());
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
    compiler_intrinsics: Vec<tribute_control::CompilerIntrinsicDeclaration>,
}

impl TributeControlToCps {
    pub fn new(
        declarations: impl IntoIterator<Item = tribute_control::OperationDeclaration>,
    ) -> Self {
        Self {
            declarations: declarations.into_iter().collect(),
            compiler_intrinsics: Vec::new(),
        }
    }

    pub fn with_compiler_intrinsics(
        mut self,
        declarations: impl IntoIterator<Item = tribute_control::CompilerIntrinsicDeclaration>,
    ) -> Self {
        self.compiler_intrinsics = declarations.into_iter().collect();
        self
    }
}

impl Pass for TributeControlToCps {
    type Target = core::Module;

    fn name(&self) -> &'static str {
        "tribute-control-to-cps"
    }

    fn run(&mut self, ctx: &mut IrContext, target: core::Module) -> PassRunResult {
        tribute_control_to_cps(
            ctx,
            target.into(),
            &self.declarations,
            &self.compiler_intrinsics,
        )
        .map_err(|error| Box::new(error) as _)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::ops::DialectType;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;

    fn parse(input: &str) -> (IrContext, Module) {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);
        (ctx, module)
    }

    fn operation_declarations(
        ctx: &IrContext,
        operations: &[(&str, &str)],
    ) -> Vec<tribute_control::OperationDeclaration> {
        let i32_type = ctx
            .types
            .iter()
            .find_map(|(ty, data)| {
                (data.dialect == Symbol::new("core") && data.name == Symbol::new("i32"))
                    .then_some(ty)
            })
            .expect("test module declares core.i32");
        operations
            .iter()
            .map(|&(ability_name, op_name)| {
                let ability_ref = ctx
                    .types
                    .iter()
                    .find_map(|(ty, data)| {
                        (data.dialect == Symbol::new("core")
                            && data.name == Symbol::new("ability_ref")
                            && data.attrs.get_symbol("name")
                                == Some(Symbol::from_dynamic(ability_name)))
                        .then_some(ty)
                    })
                    .unwrap_or_else(|| panic!("test module declares {ability_name}"));
                tribute_control::OperationDeclaration::new(
                    ability_ref,
                    Symbol::from_dynamic(op_name),
                    Symbol::new("op"),
                    [i32_type],
                    i32_type,
                )
            })
            .collect()
    }

    fn assert_nested_resume_frames(printed: &str) {
        assert_eq!(
            printed.matches("ability.handle_dispatch").count(),
            2,
            "{printed}"
        );
        assert_eq!(printed.matches("ability.perform").count(), 2, "{printed}");
        assert!(
            printed
                .matches("tribute.cps_continuation_frame_result")
                .count()
                >= 2,
            "{printed}"
        );
        assert!(
            printed.matches("adt.struct_get").count() >= 6,
            "dynamic frames must be unpacked to recover done and dispatch: {printed}"
        );
        assert!(
            printed.matches("adt.struct_new").count() >= 4,
            "suffixes and resumes must repack the lexical dispatcher with a new done target: {printed}"
        );
        assert!(
            printed
                .lines()
                .filter(|line| line.contains("ability.perform"))
                .all(|line| !line.contains(": core.i32")),
            "final ability.perform must be resultless: {printed}"
        );
        assert!(!printed.contains("tribute_control."), "{printed}");
    }

    fn collect_lambdas(ctx: &IrContext, op: OpRef, lambdas: &mut Vec<OpRef>) {
        if closure::Lambda::matches(ctx, op) {
            lambdas.push(op);
            return;
        }
        for &region in &ctx.op(op).regions {
            for &block in &ctx.region(region).blocks {
                for &child in &ctx.block(block).ops {
                    collect_lambdas(ctx, child, lambdas);
                }
            }
        }
    }

    fn collect_cps_tail_calls(ctx: &IrContext, op: OpRef, tails: &mut Vec<OpRef>) {
        if func::TailCallIndirect::matches(ctx, op)
            && tribute_core::get_calling_convention(ctx, op) == Some(CallingConvention::Cps)
        {
            tails.push(op);
        }
        for &region in &ctx.op(op).regions {
            for &block in &ctx.region(region).blocks {
                for &child in &ctx.block(block).ops {
                    collect_cps_tail_calls(ctx, child, tails);
                }
            }
        }
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
        tribute_control_to_cps(&mut ctx, module, &[], &[]).unwrap();
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
        tribute_control_to_cps(&mut ctx, module, &[], &[]).unwrap();
        let printed = print_module(&ctx, module.op());
        assert!(printed.contains("func.call "));
        assert!(printed.contains("func.call_indirect"));
        assert!(printed.contains("func.tail_call "));
        assert!(printed.contains("func.tail_call_indirect"));
        assert!(printed.contains("tribute.calling_convention = 0"));
        assert!(printed.contains("tribute.calling_convention = 1"));
        assert!(printed.contains("tribute.calling_convention = 2"));
        assert!(!printed.contains("tribute_control."), "{printed}");

        // `closure.lambda` assembly only preserves its callable shape. The
        // exact outer closure convention remains canonical type provenance
        // and is verified above before this printer/parser boundary.
        let mut lambdas = Vec::new();
        collect_lambdas(&ctx, module.op(), &mut lambdas);
        let one_parameter_cps = lambdas.into_iter().find(|lambda| {
            let closure_ty = ctx.op_result_types(*lambda)[0];
            tribute_core::get_calling_convention(&ctx, *lambda) == Some(CallingConvention::Cps)
                && closure::Closure::from_type_ref(&ctx, closure_ty)
                    .and_then(|closure| core::Func::from_type_ref(&ctx, closure.func_type(&ctx)))
                    .is_some_and(|callable| callable.params(&ctx).len() == 1)
        });
        let lambda =
            one_parameter_cps.expect("conversion should produce a one-parameter CPS continuation");
        let closure_ty = ctx.op_result_types(lambda)[0];
        assert_eq!(
            tribute_core::calling_convention::get_physical_closure_environment_index(
                &ctx, closure_ty
            ),
            Some(0)
        );
        crate::lower_closure_lambda::lower_closure_lambda(&mut ctx, module);
        crate::closure_lower::lower_closures(&mut ctx, module);
        let lowered = print_module(&ctx, module.op());
        assert!(
            !lowered.contains("closure.lambda") && !lowered.contains("closure.new"),
            "{lowered}"
        );
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
        tribute_control_to_cps(&mut ctx, module, &[], &[]).unwrap();
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
        tribute_control_to_cps(&mut ctx, module, &[], &[]).unwrap();
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
        tribute_control_to_cps(&mut ctx, module, &[], &[]).unwrap();
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
        let error = tribute_control_to_cps(&mut ctx, module, &[], &[]).unwrap_err();
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
            let error = tribute_control_to_cps(&mut ctx, module, &[], &[]).unwrap_err();
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
        let error = tribute_control_to_cps(&mut ctx, module, &[], &[]).unwrap_err();
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
        let error = tribute_control_to_cps(&mut ctx, module, &[], &[]).unwrap_err();
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
    fn post_candidate_failure_restores_alias_maps_and_canonical_ir() {
        let input = r#"core.module @candidate {
  !callback = tribute_control.callable(core.i32, core.i32) {tribute.calling_convention = 0}
  func.func @broken(%value: core.i32) -> core.i32 {
    func.return %value
  }
}"#;
        let (mut ctx, candidate) = parse(input);
        let before = print_module(&ctx, candidate.op());
        let source_aliases = ctx.type_aliases().to_vec();
        let (alias_name, source_type) = source_aliases[0];
        let converted_type = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());
        ctx.register_type_alias(alias_name, converted_type);

        let error =
            verify_candidate_or_restore_aliases(&mut ctx, candidate, &source_aliases).unwrap_err();
        assert_eq!(error.boundary, POST_CPS_BOUNDARY);
        assert_eq!(ctx.type_alias_by_name(alias_name), Some(source_type));
        assert_eq!(ctx.type_alias_by_type(source_type), Some(alias_name));
        assert_eq!(ctx.type_alias_by_type(converted_type), None);
        assert_eq!(print_module(&ctx, candidate.op()), before);
    }

    #[test]
    fn named_boundaries_report_local_and_core_validation_errors() {
        let local_input = r#"core.module @test {
  tribute_control.func @broken(%value: core.i32) -> core.i32 convention(direct) {
    tribute_control.return
  }
}"#;
        let (ctx, module) = parse(local_input);
        let error = verify_tribute_control_pre_cps(&ctx, module, &[], &[]).unwrap_err();
        assert!(error.to_string().contains("expects 1 operand"), "{error}");

        let core_input = r#"core.module @test {
  tribute_control.func @broken(%value: core.i32) -> core.i32 convention(direct) {
    func.tail_call_indirect
    tribute_control.return %value
  }
}"#;
        let (ctx, module) = parse(core_input);
        let error = verify_tribute_control_pre_cps(&ctx, module, &[], &[]).unwrap_err();
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
  func.func @caller(%ev: !evidence, %prompt: core.i32, %tr: !tr, %general: !general) -> core.never attributes {tribute.calling_convention = 2} {
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
  func.func @caller(%ev: !evidence, %prompt: core.i32) -> core.never attributes {tribute.calling_convention = 2} {
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
  func.func @caller(%ev: !evidence, %prompt: core.i32) -> core.never attributes {tribute.calling_convention = 2} {
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
        tribute_control_to_cps(&mut ctx, module, &declarations, &[]).unwrap();
        let mut perform_resume = None;
        fn find_perform_resume(ctx: &IrContext, op: OpRef, found: &mut Option<ValueRef>) {
            if let Ok(perform) = ability::Perform::from_op(ctx, op) {
                *found = Some(perform.resume(ctx));
            }
            for region in ctx.op(op).regions.iter().copied() {
                for block in ctx.region(region).blocks.iter().copied() {
                    for child in ctx.block(block).ops.iter().copied() {
                        find_perform_resume(ctx, child, found);
                    }
                }
            }
        }
        find_perform_resume(&ctx, module.op(), &mut perform_resume);
        let perform_resume = perform_resume.expect("resumptive handler emits ability.perform");
        let erased_function = cps_closure_function_type(&ctx, ctx.value_ty(perform_resume))
            .expect("ability.perform resume has CPS callable provenance");
        let erased = core::Func::from_type_ref(&ctx, erased_function)
            .expect("ability.perform resume has a function signature");
        assert_eq!(erased.params(&ctx).len(), 3);
        assert!(
            type_is(&ctx, erased.params(&ctx)[2], "tribute_rt", "anyref"),
            "ability.perform must receive Resume<anyref, R>, not the exact continuation: {}",
            print_module(&ctx, module.op())
        );
        let trunk_ir::ValueDef::OpResult(wrapper_op, _) = ctx.value_def(perform_resume) else {
            panic!("ability.perform resume must be the one-shot adapter closure");
        };
        assert!(closure::Lambda::matches(&ctx, wrapper_op));
        assert!(
            ctx.op_operands(wrapper_op).iter().copied().any(|capture| {
                cps_closure_function_type(&ctx, ctx.value_ty(capture))
                    .and_then(|function| core::Func::from_type_ref(&ctx, function))
                    .is_some_and(|function| {
                        function.params(&ctx).len() == 3
                            && type_is(&ctx, function.params(&ctx)[2], "core", "i32")
                    })
            }),
            "the erased adapter must capture the declared ResumeExact<I, R>: {}",
            print_module(&ctx, module.op())
        );
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
        tribute_control_to_cps(&mut ctx, module, &declarations, &[]).unwrap();

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
        assert!(
            printed.contains("@__tribute_make_dispatch_adapter_"),
            "the dispatch-adapter factory must be present: {printed}"
        );
        let mut tails = Vec::new();
        collect_cps_tail_calls(&ctx, module.op(), &mut tails);
        assert!(
            tails.len() >= 3,
            "the dispatch adapter and separate handler paths must tail-transfer: {printed}"
        );
        for tail in tails {
            let signature =
                trunk_ir::op_interface::IndirectCallLikeOps::exact_signature(&ctx, tail)
                    .expect("every emitted CPS indirect tail has its exact closure contract");
            let callable = core::Func::from_type_ref(&ctx, signature)
                .expect("the indirect tail signature is a core.func");
            assert_eq!(callable.r#return(&ctx), core::never(&mut ctx).as_type_ref());
            assert_eq!(
                callable.params(&ctx).len() + 1,
                ctx.op_operands(tail).len(),
                "the exact signature covers every tail operand: {printed}"
            );
        }
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
        tribute_control_to_cps(&mut ctx, module, &declarations, &[]).unwrap();
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
        tribute_control_to_cps(&mut ctx, module, &declarations, &[]).unwrap();
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

        let mut tails = Vec::new();
        collect_cps_tail_calls(&ctx, module.op(), &mut tails);
        assert!(
            tails.len() >= 3,
            "the dispatch adapter and independent CPS exits must both tail-transfer: {printed}"
        );
        for tail in tails {
            let signature =
                trunk_ir::op_interface::IndirectCallLikeOps::exact_signature(&ctx, tail)
                    .expect("every emitted CPS indirect tail has its exact closure contract");
            let callable = core::Func::from_type_ref(&ctx, signature)
                .expect("the indirect tail signature is a core.func");
            assert_eq!(callable.r#return(&ctx), core::never(&mut ctx).as_type_ref());
            assert_eq!(
                callable.params(&ctx).len() + 1,
                ctx.op_operands(tail).len(),
                "the exact signature covers every tail operand: {printed}"
            );
            assert!(
                callable
                    .params(&ctx)
                    .iter()
                    .zip(&ctx.op_operands(tail)[1..])
                    .all(|(expected, actual)| *expected == ctx.value_ty(*actual)),
                "the exact signature must not be inferred from erased operands: {printed}"
            );
        }

        let mut lambdas = Vec::new();
        collect_lambdas(&ctx, module.op(), &mut lambdas);
        let mut generated_param_counts = Vec::new();
        for lambda in lambdas {
            if tribute_core::get_calling_convention(&ctx, lambda) != Some(CallingConvention::Cps) {
                continue;
            }
            let closure_ty = ctx.op_result_types(lambda)[0];
            assert_eq!(
                tribute_core::get_physical_closure_convention(&ctx, closure_ty),
                Some(CallingConvention::Cps)
            );
            let callable = closure::Closure::from_type_ref(&ctx, closure_ty)
                .and_then(|closure| core::Func::from_type_ref(&ctx, closure.func_type(&ctx)))
                .unwrap();
            generated_param_counts.push(callable.params(&ctx).len());
        }
        assert!(generated_param_counts.contains(&2), "{printed}");

        crate::lower_closure_lambda::lower_closure_lambda(&mut ctx, module);
        let lifted = print_module(&ctx, module.op());
        assert!(!lifted.contains("closure.lambda"), "{lifted}");
        assert!(
            lifted.contains("tribute.closure_environment_index = 0"),
            "{lifted}"
        );
        crate::closure_lower::lower_closures(&mut ctx, module);
        let lowered = print_module(&ctx, module.op());
        assert!(!lowered.contains("closure.new"), "{lowered}");
        assert!(lowered.contains("signature"), "{lowered}");
    }

    #[test]
    fn cps_indirect_tail_without_a_provenance_bearing_closure_fails_before_insertion() {
        let (mut ctx, module) = parse(
            r#"core.module @test {
  func.func @caller() -> core.never {
    func.unreachable
  }
}"#,
        );
        let module_block = ctx.region(module.body(&ctx).unwrap()).blocks[0];
        let location = ctx.op(module.op()).location;
        let never = core::never(&mut ctx).as_type_ref();
        let raw_type = core::func(&mut ctx, never, []).as_type_ref();
        let raw = func::constant(&mut ctx, location, raw_type, Symbol::new("raw"));
        let before = ctx.block(module_block).ops.to_vec();
        let mut converter = Converter::new(&mut ctx, module_block, HashMap::new(), module.op());

        let error = converter
            .emit_cps_tail_call_indirect(
                module_block,
                location,
                raw.result(converter.ctx),
                std::iter::empty(),
            )
            .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("no exact provenance-bearing closure contract"),
            "{error}"
        );
        assert_eq!(
            converter.ctx.block(module_block).ops.as_slice(),
            before.as_slice()
        );
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
        let error = tribute_control_to_cps(&mut ctx, module, &declarations, &[]).unwrap_err();
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
        tribute_control_to_cps(&mut ctx, module, &[], &[]).unwrap();
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
        tribute_control_to_cps(&mut ctx, module, &[], &[]).unwrap();
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
        let error = tribute_control_to_cps(&mut ctx, module, &[], &[]).unwrap_err();
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
    fn raw_pointer_managed_masquerade_is_rejected_before_mutation() {
        let input = r#"core.module @test {
  !S = adt.struct() {name = @S, fields = []}
  !R = adt.typeref() {name = @S}
  tribute_control.func @broken(%raw: core.ptr) -> !R convention(direct) {
    %middle = arith.cast %raw : core.i64
    %managed = core.unrealized_conversion_cast %middle : !R
    tribute_control.return %managed
  }
}"#;
        let (mut ctx, module) = parse(input);
        let before = print_module(&ctx, module.op());
        let error = tribute_control_to_cps(&mut ctx, module, &[], &[]).unwrap_err();
        assert_eq!(error.boundary, PRE_CPS_BOUNDARY);
        assert!(error.to_string().contains("core.ptr cast chain"), "{error}");
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
        tribute_control_to_cps(&mut ctx, module, &[], &[]).unwrap();
        let printed = print_module(&ctx, module.op());
        assert_eq!(printed.matches("ability.handle_dispatch").count(), 2);
        assert!(printed.contains("effect.fresh_prompt_tag"));
        assert!(printed.matches("func.tail_call_indirect").count() >= 3);
        assert!(!printed.contains("__tribute_cps_control"));
    }

    #[test]
    fn nested_same_ability_resumes_rebuild_the_dynamic_frame_dispatcher() {
        let input = r#"core.module @test {
  tribute_control.func @nested_same(%input: core.i32) -> core.i32 convention(cps) {
    %outer = tribute_control.handle : core.i32 {
      %inner = tribute_control.handle : core.i32 {
        %performed = tribute_control.perform %input {ability_ref = core.ability_ref() {name = @State}, op_name = @get, operation_kind = @op} : core.i32
        tribute_control.yield %performed
      } {
        ^inner_completion(%value: core.i32):
          tribute_control.yield %value
      } {
        tribute_control.handler {ability_ref = core.ability_ref() {name = @State}, kind = @op, op_name = @get, operation_result_type = core.i32} {
          ^inner_arm(%argument: core.i32, %token: tribute_control.resume_token(core.i32, core.i32)):
            %resumed = tribute_control.resume %token, %argument : core.i32
            tribute_control.yield %resumed
        }
      }
      %performed = tribute_control.perform %inner {ability_ref = core.ability_ref() {name = @State}, op_name = @get, operation_kind = @op} : core.i32
      tribute_control.yield %performed
    } {
      ^outer_completion(%value: core.i32):
        tribute_control.yield %value
    } {
      tribute_control.handler {ability_ref = core.ability_ref() {name = @State}, kind = @op, op_name = @get, operation_result_type = core.i32} {
        ^outer_arm(%argument: core.i32, %token: tribute_control.resume_token(core.i32, core.i32)):
          %resumed = tribute_control.resume %token, %argument : core.i32
          tribute_control.yield %resumed
      }
    }
    tribute_control.return %outer
  }
}"#;
        let (mut ctx, module) = parse(input);
        let declarations = operation_declarations(&ctx, &[("State", "get")]);
        tribute_control_to_cps(&mut ctx, module, &declarations, &[]).unwrap();
        assert_nested_resume_frames(&print_module(&ctx, module.op()));
    }

    #[test]
    fn nested_cross_ability_resumes_rebuild_the_dynamic_frame_dispatcher() {
        let input = r#"core.module @test {
  tribute_control.func @nested_cross(%input: core.i32) -> core.i32 convention(cps) {
    %outer = tribute_control.handle : core.i32 {
      %inner = tribute_control.handle : core.i32 {
        %performed = tribute_control.perform %input {ability_ref = core.ability_ref() {name = @Console}, op_name = @read, operation_kind = @op} : core.i32
        tribute_control.yield %performed
      } {
        ^inner_completion(%value: core.i32):
          tribute_control.yield %value
      } {
        tribute_control.handler {ability_ref = core.ability_ref() {name = @Console}, kind = @op, op_name = @read, operation_result_type = core.i32} {
          ^inner_arm(%argument: core.i32, %token: tribute_control.resume_token(core.i32, core.i32)):
            %resumed = tribute_control.resume %token, %argument : core.i32
            tribute_control.yield %resumed
        }
      }
      %performed = tribute_control.perform %inner {ability_ref = core.ability_ref() {name = @State}, op_name = @get, operation_kind = @op} : core.i32
      tribute_control.yield %performed
    } {
      ^outer_completion(%value: core.i32):
        tribute_control.yield %value
    } {
      tribute_control.handler {ability_ref = core.ability_ref() {name = @State}, kind = @op, op_name = @get, operation_result_type = core.i32} {
        ^outer_arm(%argument: core.i32, %token: tribute_control.resume_token(core.i32, core.i32)):
          %resumed = tribute_control.resume %token, %argument : core.i32
          tribute_control.yield %resumed
      }
    }
    tribute_control.return %outer
  }
}"#;
        let (mut ctx, module) = parse(input);
        let declarations = operation_declarations(&ctx, &[("State", "get"), ("Console", "read")]);
        tribute_control_to_cps(&mut ctx, module, &declarations, &[]).unwrap();
        assert_nested_resume_frames(&print_module(&ctx, module.op()));
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
        tribute_control_to_cps(&mut ctx, module, &declarations, &[]).unwrap();
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
        tribute_control_to_cps(&mut ctx, module, &declarations, &[]).unwrap();
        let printed = print_module(&ctx, module.op());
        assert!(printed.contains("ability.call"));
        assert!(!printed.contains("ability.perform"));
        assert!(!printed.contains("@consumed"));
        assert!(!printed.contains("adt.struct_set"));
        assert!(printed.contains("tribute.calling_convention = 1"));
        let mut indirect_calls = Vec::new();
        fn collect_indirect_calls(ctx: &IrContext, op: OpRef, calls: &mut Vec<OpRef>) {
            if func::CallIndirect::matches(ctx, op) {
                calls.push(op);
            }
            for region in ctx.op(op).regions.iter().copied() {
                for block in ctx.region(region).blocks.iter().copied() {
                    for child in ctx.block(block).ops.iter().copied() {
                        collect_indirect_calls(ctx, child, calls);
                    }
                }
            }
        }
        collect_indirect_calls(&ctx, module.op(), &mut indirect_calls);
        let call = indirect_calls
            .into_iter()
            .find(|op| {
                tribute_core::get_calling_convention(&ctx, *op)
                    == Some(CallingConvention::EvidenceDirect)
            })
            .expect("fn handler dispatcher call");
        let call = func::CallIndirect::from_op(&ctx, call).unwrap();
        let expected = physical_closure_function_type(
            &ctx,
            ctx.value_ty(call.callee(&ctx)),
            CallingConvention::EvidenceDirect,
        )
        .expect("fn arm retains an exact closure contract");
        assert_eq!(call.signature(&ctx), Some(expected));
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
        tribute_control_to_cps(&mut ctx, module, &declarations, &[]).unwrap();
        let printed = print_module(&ctx, module.op());
        assert!(printed.contains("scf.switch"));
        assert!(printed.contains("scf.case"));
        assert!(printed.contains("ability.perform"));
        assert!(printed.matches("func.tail_call_indirect").count() >= 2);
        assert!(!printed.contains("tribute_control."));
    }
}
