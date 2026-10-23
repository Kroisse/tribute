//! Named pre- and post-CPS boundaries: their conversion targets, type walks,
//! and whole-graph verification.

use super::*;
use tribute_ir::calling_convention::DynConventionBearing;

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
    pub(super) fn one(
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

    /// A failure of the post-CPS boundary at `location` that no single
    /// operation carries.
    pub(crate) fn post_at(location: Location, message: impl Into<String>) -> Self {
        Self::one(POST_CPS_BOUNDARY, None, Some(location), message)
    }

    /// A failure of the post-CPS boundary caused by `op`.
    pub(super) fn post_op(op: OpRef, location: Location, message: impl Into<String>) -> Self {
        Self::one(POST_CPS_BOUNDARY, Some(op), Some(location), message)
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
        .legal_dialect(trunk_ir::dialect::core::DIALECT_NAME)
        .legal_dialect(tribute_ir::dialect::tribute_control::DIALECT_NAME)
        .legal_dialect(trunk_ir::dialect::scf::DIALECT_NAME)
        .legal_dialect(trunk_ir::dialect::arith::DIALECT_NAME)
        .legal_dialect(tribute_ir::dialect::adt::DIALECT_NAME)
        .legal_dialect(tribute_ir::dialect::list::DIALECT_NAME)
        .legal_dialect(tribute_ir::dialect::tribute_rt::DIALECT_NAME)
        .legal_dialect(tribute_ir::dialect::tribute_io::DIALECT_NAME)
        .legal_op("ability.decl")
        .recursive_legal_op("ability.decl")
}

/// Partial post-legalization target. Unknown operations belong to later passes.
pub fn tribute_control_post_cps_target() -> ConversionTarget {
    ConversionTarget::new().illegal_dialect(tribute_ir::dialect::tribute_control::DIALECT_NAME)
}

#[derive(Clone, Copy)]
enum TypeBoundary {
    Pre,
    Post,
}

fn walk_attribute_types(
    ctx: &IrContext,
    attribute: &Attribute,
    boundary: TypeBoundary,
    seen: &mut HashSet<TypeRef>,
    errors: &mut Vec<String>,
) {
    attribute.visit_types(&mut |ty| walk_type(ctx, ty, boundary, seen, errors));
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
    let data = ctx.get_type(ty);
    let forbidden = match boundary {
        TypeBoundary::Pre => {
            trunk_ir::dialect::func::FuncSig::matches(ctx, ty)
                || tribute_ir::dialect::closure::Closure::matches(ctx, ty)
                || [ability::DIALECT_NAME, effect::DIALECT_NAME].contains(&data.name.dialect(ctx))
        }
        TypeBoundary::Post => {
            data.name.dialect(ctx) == tribute_ir::dialect::tribute_control::DIALECT_NAME
        }
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
    let mut seen = HashSet::default();
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
    for region in ctx.op_regions(op) {
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
    let mut seen = HashSet::default();
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
        failures: &mut Vec<BoundaryFailure>,
    ) {
        let closure_ty = ctx.get_type(ctx.value_ty(value));
        let valid = if closure_ty.name.is(ctx, "closure.closure") && closure_ty.params.len() == 1 {
            func::FuncSig::from_type_ref(ctx, closure_ty.params[0]).is_some_and(|function| {
                let params = function.inputs(ctx);
                let result = function.single_result(ctx);
                result.is_some_and(|result| {
                    tribute_ir::dialect::tribute_rt::Anyref::matches(ctx, result)
                }) && params.len() == 3
                    && params[0] == evidence
                    && trunk_ir::dialect::core::I32::matches(ctx, params[1])
                    && tribute_ir::dialect::tribute_rt::Anyref::matches(ctx, params[2])
            })
        } else {
            false
        };
        if !valid {
            failures.push(BoundaryFailure {
                op: Some(owner),
                location: Some(ctx.op(owner).location),
                message: "tail-resumptive dispatcher has the wrong typed closure ABI".into(),
            });
        }
        let expected = CallingConvention::EvidenceDirect;
        let trunk_ir::ValueDef::OpResult(def, _) = ctx.value_def(value) else {
            failures.push(BoundaryFailure {
                op: Some(owner),
                location: Some(ctx.op(owner).location),
                message: "dispatcher must be a physical closure result with exact calling convention metadata"
                    .into(),
            });
            return;
        };
        if DynConventionBearing::of(ctx, def) != Some(expected) {
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
                for &dispatcher in dispatchers {
                    check_dispatcher(ctx, op, dispatcher, evidence_ty, failures);
                }
            }
        }
        for region in ctx.op_regions(op) {
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

/// The result of the `func.func` whose body directly holds `op`, if a
/// structured operation is all that lies between them.
fn enclosing_func_result(ctx: &IrContext, op: OpRef) -> Option<TypeRef> {
    let mut current = op;
    loop {
        let block = ctx.op(current).parent_block?;
        let region = ctx.block(block).parent_region?;
        let owner = ctx.region(region).parent_op?;
        if let Ok(function) = func::Func::from_op(ctx, owner) {
            let ty = function.r#type(ctx);
            return func::FuncSig::from_type_ref(ctx, ty)?.single_result(ctx);
        }
        if ctx.op(owner).name.dialect(ctx) != trunk_ir::dialect::scf::DIALECT_NAME {
            return None;
        }
        current = owner;
    }
}

fn verify_physical_callable_graph(
    ctx: &IrContext,
    module: Module,
    symbols: &SymbolTable,
) -> Vec<BoundaryFailure> {
    fn visit(
        ctx: &IrContext,
        op: OpRef,
        symbols: &SymbolTable,
        signatures: &HashMap<OpRef, (TypeRef, Option<CallingConvention>)>,
        failures: &mut Vec<BoundaryFailure>,
    ) {
        let data = ctx.op(op);
        let op_convention = DynConventionBearing::of(ctx, op);
        if DynConventionBearing::is_implemented(ctx, op) && op_convention.is_none() {
            failures.push(BoundaryFailure {
                op: Some(op),
                location: Some(data.location),
                message: format!(
                    "{} must carry exact Direct, EvidenceDirect, or Cps metadata",
                    data.name.display(ctx)
                ),
            });
        }
        if data.name.is(ctx, "func.tail_call") {
            let Some(callee) = ctx.op_attrs(op).get_symbol_ref("callee") else {
                failures.push(BoundaryFailure {
                    op: Some(op),
                    location: Some(data.location),
                    message: "func.tail_call requires a resolved callee symbol".into(),
                });
                return;
            };
            let Some((func_ty, convention)) = symbols
                .resolve(ctx, op, callee)
                .and_then(|target| signatures.get(&target))
            else {
                failures.push(BoundaryFailure {
                    op: Some(op),
                    location: Some(data.location),
                    message: format!("func.tail_call references unresolved callee @{callee}"),
                });
                return;
            };
            let function = func::FuncSig::from_type_ref(ctx, *func_ty);
            let is_cps = op_convention == Some(CallingConvention::Cps);
            let valid_never = function
                .and_then(|function| function.single_result(ctx))
                .is_some_and(|result| trunk_ir::dialect::core::Never::matches(ctx, result));
            if is_cps && !valid_never {
                failures.push(BoundaryFailure {
                    op: Some(op),
                    location: Some(data.location),
                    message: "func.tail_call target must have core.never result".into(),
                });
            }
            let expected = function
                .map(|function| function.inputs(ctx))
                .unwrap_or_default();
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
            if is_cps && *convention != op_convention {
                failures.push(BoundaryFailure {
                    op: Some(op),
                    location: Some(data.location),
                    message: "func.tail_call must preserve exact Cps metadata".into(),
                });
            } else if !is_cps {
                // A source `become` between value-returning callables keeps
                // the target's convention and returns the target's result as
                // the caller's.
                if *convention != op_convention {
                    failures.push(BoundaryFailure {
                        op: Some(op),
                        location: Some(data.location),
                        message: "func.tail_call metadata does not match its target".into(),
                    });
                }
                let caller_result = enclosing_func_result(ctx, op);
                let target_result = function.and_then(|function| function.single_result(ctx));
                if caller_result.is_some() && caller_result != target_result {
                    failures.push(BoundaryFailure {
                        op: Some(op),
                        location: Some(data.location),
                        message: "func.tail_call target result differs from the caller's".into(),
                    });
                }
            }
        } else if data.name.is(ctx, "func.call") {
            let callee = ctx.op_attrs(op).get_symbol_ref("callee");
            if let Some(callee) = callee {
                match symbols
                    .resolve(ctx, op, callee)
                    .and_then(|target| signatures.get(&target))
                {
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
        } else if data.name.is(ctx, "func.call_indirect")
            && op_convention == Some(CallingConvention::Cps)
        {
            failures.push(BoundaryFailure {
                op: Some(op),
                location: Some(data.location),
                message: "dynamic Cps transfers must use func.tail_call_indirect".into(),
            });
        }
        for region in ctx.op_regions(op) {
            for block in ctx.region(region).blocks.iter().copied() {
                for child in ctx.block(block).ops.iter().copied() {
                    visit(ctx, child, symbols, signatures, failures);
                }
            }
        }
    }

    // A callee resolves in the module nearest to the call.
    let signatures: HashMap<OpRef, (TypeRef, Option<CallingConvention>)> = symbols
        .symbols()
        .iter()
        .filter_map(|&op| {
            if !func::Func::matches(ctx, op) {
                return None;
            }
            let convention = DynConventionBearing::of(ctx, op);
            Some((op, (ctx.op_attrs(op).get_type("type")?, convention)))
        })
        .collect();
    let mut failures = Vec::new();
    for &op in module.ops(ctx) {
        visit(ctx, op, symbols, &signatures, &mut failures);
    }
    failures
}

pub(super) fn verify_source_conversion_shapes(
    ctx: &IrContext,
    module: Module,
) -> Vec<BoundaryFailure> {
    fn failure(ctx: &IrContext, op: OpRef, message: impl Into<String>) -> BoundaryFailure {
        BoundaryFailure {
            op: Some(op),
            location: Some(ctx.op(op).location),
            message: message.into(),
        }
    }

    fn visit(ctx: &IrContext, op: OpRef, failures: &mut Vec<BoundaryFailure>) {
        let data = ctx.op(op);
        if ctx.op_has_successors(op) {
            failures.push(failure(
                ctx,
                op,
                "tribute-control-pre-cps is structured and forbids block successors",
            ));
        }
        if data.name.is(ctx, "scf.switch") {
            let body_region =
                ctx.op_regions(op).exactly_one().ok().filter(|_| {
                    ctx.op_operands(op).len() == 1 && ctx.op_result_types(op).is_empty()
                });
            if let Some(body_region) = body_region {
                let blocks = &ctx.region(body_region).blocks;
                if let [body] = blocks.as_slice() {
                    for arm in ctx.block(*body).ops.iter().copied() {
                        let arm_data = ctx.op(arm);
                        let is_case = arm_data.name.is(ctx, "scf.case");
                        let is_default = arm_data.name.is(ctx, "scf.default");
                        if !is_case && !is_default {
                            failures.push(failure(
                                ctx,
                                arm,
                                "scf.switch body may contain only scf.case and scf.default",
                            ));
                            continue;
                        }
                        if is_case && !ctx.op_attrs(arm).contains_key("value") {
                            failures.push(failure(ctx, arm, "scf.case requires a value attribute"));
                        }
                        if let Ok(region) = ctx.op_regions(arm).exactly_one() {
                            if ctx.region(region).blocks.len() != 1 {
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
            } else {
                failures.push(failure(
                    ctx,
                    op,
                    "scf.switch requires one discriminant, no results, and one body region",
                ));
            }
        }
        for region in ctx.op_regions(op) {
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
    compiler_intrinsics: &[tribute_control::CompilerIntrinsicDeclaration],
    analyses: &mut AnalysisCache,
) -> Result<(), TributeControlToCpsError> {
    let mut failures = Vec::new();
    if tribute_control::conventions_assigned(ctx, module) == Some(false) {
        failures.push(BoundaryFailure {
            op: None,
            location: None,
            message: "source signatures have no calling convention; class specialization must run before legalization".into(),
        });
    }
    let validation = tribute_control::validate(ctx, module, compiler_intrinsics, analyses);
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
                        "{} operation {} is not legal",
                        match illegal.legality {
                            trunk_ir::rewrite::LegalityCheck::Illegal => "illegal",
                            trunk_ir::rewrite::LegalityCheck::Unknown => "unknown",
                            trunk_ir::rewrite::LegalityCheck::Legal => "unexpected legal",
                        },
                        ctx.op(illegal.op).name.display(ctx)
                    ),
                }),
        );
    }
    failures.extend(verify_type_boundary(ctx, module, TypeBoundary::Pre));
    failures.extend(verify_source_conversion_shapes(ctx, module));
    failures.extend(
        trunk_ir::validation::validate_all(ctx, module, analyses)
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
    analyses: &mut AnalysisCache,
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
                        "residual {} operation",
                        ctx.op(illegal.op).name.display(ctx)
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
    failures.extend(
        tribute_ir::calling_convention::verify_conventions(ctx, module.op())
            .into_iter()
            .map(|violation| BoundaryFailure {
                op: Some(violation.op),
                location: Some(ctx.op(violation.op).location),
                message: format!(
                    "{}: {}",
                    ctx.op(violation.op).name.display(ctx),
                    violation.error
                ),
            }),
    );
    let symbols = analyses.require::<SymbolTable>(ctx, module.op());
    failures.extend(verify_physical_callable_graph(ctx, module, &symbols));
    failures.extend(
        trunk_ir::validation::validate_all(ctx, module, analyses)
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
