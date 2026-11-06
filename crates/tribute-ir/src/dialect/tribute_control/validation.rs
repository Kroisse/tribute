//! Validation errors, shared helpers, and the validation entry points.

use super::*;

/// A focused `tribute_control` validation failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidationError {
    pub op: Option<OpRef>,
    pub location: Option<Location>,
    pub message: String,
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(op) = self.op {
            write!(f, "{op}: ")?;
        }
        write!(f, "{}", self.message)
    }
}

/// Result returned by the explicit Tribute validator.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
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
            return write!(f, "tribute_control validation passed");
        }
        writeln!(
            f,
            "{} tribute_control validation error(s):",
            self.errors.len()
        )?;
        for error in &self.errors {
            writeln!(f, "  - {error}")?;
        }
        Ok(())
    }
}

pub(super) fn push_op_error(
    ctx: &IrContext,
    op: OpRef,
    errors: &mut Vec<ValidationError>,
    message: impl Into<String>,
) {
    errors.push(ValidationError {
        op: Some(op),
        location: Some(ctx.op(op).location),
        message: message.into(),
    });
}

pub(super) fn push_type_error(errors: &mut Vec<ValidationError>, message: impl Into<String>) {
    errors.push(ValidationError {
        op: None,
        location: None,
        message: message.into(),
    });
}

pub(super) fn parent_op(ctx: &IrContext, op: OpRef) -> Option<OpRef> {
    let block = ctx.op(op).parent_block?;
    let region = ctx.block(block).parent_region?;
    ctx.region(region).parent_op
}

pub(super) fn parent_region(ctx: &IrContext, op: OpRef) -> Option<RegionRef> {
    let block = ctx.op(op).parent_block?;
    ctx.block(block).parent_region
}

fn is_unresolved_type(ctx: &IrContext, ty: TypeRef) -> bool {
    let data = ctx.get_type(ty);
    let unresolved_name = matches!(data.name.strip_dialect(ctx), "var" | "infer" | "unresolved");
    (matches!(data.name.dialect(ctx), "type" | "tribute")) && unresolved_name
}

pub(super) fn contains_unresolved_type(
    ctx: &IrContext,
    ty: TypeRef,
    visiting: &mut HashSet<TypeRef>,
) -> bool {
    if !visiting.insert(ty) {
        return false;
    }
    let data = ctx.get_type(ty);
    let unresolved = is_unresolved_type(ctx, ty)
        || data
            .params
            .iter()
            .copied()
            .any(|param| contains_unresolved_type(ctx, param, visiting));
    visiting.remove(&ty);
    unresolved
}

fn contains_forbidden_logical_component(
    ctx: &IrContext,
    ty: TypeRef,
    visiting: &mut HashSet<TypeRef>,
) -> bool {
    if !visiting.insert(ty) {
        return true;
    }
    let data = ctx.get_type(ty);
    let forbidden = (data.name.is(ctx, "func.func_sig"))
        || (data.name.is(ctx, "closure.closure"))
        || ResumeToken::matches(ctx, ty)
        || is_unresolved_type(ctx, ty)
        || data
            .params
            .iter()
            .copied()
            .any(|param| contains_forbidden_logical_component(ctx, param, visiting));
    visiting.remove(&ty);
    forbidden
}

pub(super) fn validate_control_types(ctx: &IrContext, errors: &mut Vec<ValidationError>) {
    for (ty, data) in ctx.types().iter() {
        if data.name.dialect(ctx) != DIALECT_NAME {
            continue;
        }
        if FuncSig::matches(ctx, ty) {
            if let Err(error) = <FuncSig as trunk_ir::type_verifier::VerifyType>::verify(ctx, ty) {
                push_type_error(
                    errors,
                    format!("{ty}: malformed tribute_control.func_sig: {error:?}"),
                );
                continue;
            }
            for component in data.params.iter().copied() {
                if contains_forbidden_logical_component(ctx, component, &mut HashSet::default()) {
                    push_type_error(
                        errors,
                        format!(
                            "{ty}: callable component {component} is unresolved, recursive, physical, or contains a resume token"
                        ),
                    );
                }
            }
        } else if ResumeToken::matches(ctx, ty) {
            if data.params.len() != 2 {
                push_type_error(
                    errors,
                    format!(
                        "{ty}: tribute_control.resume_token requires exactly input and answer types"
                    ),
                );
            } else {
                for component in data.params.iter().copied() {
                    if contains_forbidden_logical_component(ctx, component, &mut HashSet::default())
                    {
                        push_type_error(
                            errors,
                            format!(
                                "{ty}: resume token component {component} is not a resolved logical value type"
                            ),
                        );
                    }
                }
            }
        } else {
            push_type_error(
                errors,
                format!(
                    "{ty}: unsupported tribute_control type '{}'",
                    data.name.strip_dialect(ctx)
                ),
            );
        }
    }
}

pub(super) fn validate_attr_keys(
    ctx: &IrContext,
    op: OpRef,
    schema: &OpSchema,
    allowed_extra: bool,
    errors: &mut Vec<ValidationError>,
) {
    let data = ctx.op(op);
    if ctx.op_attrs(op).get(CALLING_CONVENTION_ATTR).is_some() {
        push_op_error(
            ctx,
            op,
            errors,
            format!(
                "{CALLING_CONVENTION_ATTR} belongs only on physical operations; a tribute_cps operation declares `convention`"
            ),
        );
    }
    if ctx.op_attrs(op).contains_key(ABILITY_ROW_ATTR) {
        push_op_error(
            ctx,
            op,
            errors,
            format!(
                "{ABILITY_ROW_ATTR} belongs only on tribute_control.func_sig, not on the operation"
            ),
        );
    }
    if allowed_extra {
        return;
    }
    for key in data.attributes.keys() {
        let key = ctx.str(key);
        let declared = schema.attributes.iter().any(|attr| attr.name == key);
        if !declared {
            push_op_error(
                ctx,
                op,
                errors,
                format!("has unsupported attribute '{key}'"),
            );
        }
    }
}

pub(super) fn single_block(
    ctx: &IrContext,
    owner: OpRef,
    region: RegionRef,
    name: &str,
    errors: &mut Vec<ValidationError>,
) -> Option<BlockRef> {
    let blocks = &ctx.region(region).blocks;
    if let [block] = blocks.as_slice() {
        Some(*block)
    } else {
        push_op_error(
            ctx,
            owner,
            errors,
            format!(
                "{name} region expects exactly one block, found {}",
                blocks.len()
            ),
        );
        None
    }
}

pub(super) fn terminator(
    ctx: &IrContext,
    owner: OpRef,
    block: BlockRef,
    expected: &'static OpDef,
    errors: &mut Vec<ValidationError>,
) -> Option<OpRef> {
    let Some(&op) = ctx.block(block).ops.last() else {
        push_op_error(
            ctx,
            owner,
            errors,
            format!("region must terminate with {}", expected.name.as_str()),
        );
        return None;
    };
    let data = ctx.op(op);
    if data.name != expected {
        push_op_error(
            ctx,
            owner,
            errors,
            format!("region must terminate with {}", expected.name.as_str()),
        );
        None
    } else {
        Some(op)
    }
}

pub(super) fn check_types_equal(
    ctx: &IrContext,
    owner: OpRef,
    actual: &[TypeRef],
    expected: &[TypeRef],
    what: &str,
    errors: &mut Vec<ValidationError>,
) {
    if actual != expected {
        push_op_error(
            ctx,
            owner,
            errors,
            format!("{what} types do not match the logical signature"),
        );
    }
}

pub(super) fn block_arg_types(ctx: &IrContext, block: BlockRef) -> Vec<TypeRef> {
    ctx.block_args(block)
        .iter()
        .map(|value| ctx.value_ty(*value))
        .collect()
}

pub(super) fn value_types(ctx: &IrContext, values: &[ValueRef]) -> Vec<TypeRef> {
    values.iter().map(|value| ctx.value_ty(*value)).collect()
}

pub(super) fn validate_callable_body(
    ctx: &IrContext,
    owner: OpRef,
    callable_ty: TypeRef,
    body: RegionRef,
    errors: &mut Vec<ValidationError>,
) {
    let (result_ty, params) =
        func_sig_parts(ctx, callable_ty).expect("schema-verified tribute_control.func_sig");
    let Some(block) = single_block(ctx, owner, body, "body", errors) else {
        return;
    };
    check_types_equal(
        ctx,
        owner,
        &block_arg_types(ctx, block),
        &params,
        "entry block argument",
        errors,
    );
    if ctx
        .block(block)
        .ops
        .last()
        .is_some_and(|&last| is_tail_call(ctx, last))
    {
        // The tail call checks its own result against this callable.
        return;
    }
    if let Some(ret) = terminator(ctx, owner, block, Return::DEF, errors)
        && ctx.op_operands(ret).len() == 1
        && ctx.value_ty(ctx.op_operands(ret)[0]) != result_ty
    {
        push_op_error(
            ctx,
            owner,
            errors,
            "return operand type does not match callable result type",
        );
    }
}

pub(super) fn collect_region_values(
    ctx: &IrContext,
    region: RegionRef,
    values: &mut HashSet<ValueRef>,
) {
    for block in ctx.region(region).blocks.iter().copied() {
        values.extend(ctx.block_args(block).iter().copied());
        for op in ctx.block(block).ops.iter().copied() {
            values.extend(ctx.op_results(op).iter().copied());
            for nested in ctx.op_regions(op) {
                collect_region_values(ctx, nested, values);
            }
        }
    }
}

pub(super) fn walk_region_ops(
    ctx: &IrContext,
    region: RegionRef,
    callback: &mut impl FnMut(OpRef),
) {
    for block in ctx.region(region).blocks.iter().copied() {
        for op in ctx.block(block).ops.iter().copied() {
            callback(op);
            for nested in ctx.op_regions(op) {
                walk_region_ops(ctx, nested, callback);
            }
        }
    }
}

pub(super) fn validate_func_isolation(
    ctx: &IrContext,
    op: OpRef,
    body: RegionRef,
    errors: &mut Vec<ValidationError>,
) {
    let mut defined = HashSet::default();
    collect_region_values(ctx, body, &mut defined);
    let mut invalid = Vec::new();
    walk_region_ops(ctx, body, &mut |nested| {
        for operand in ctx.op_operands(nested).iter().copied() {
            if !defined.contains(&operand) {
                invalid.push((nested, operand));
            }
        }
    });
    for (consumer, value) in invalid {
        push_op_error(
            ctx,
            op,
            errors,
            format!("isolated func body operation {consumer} references external value {value}"),
        );
    }
}

pub(super) fn validate_return_or_yield_shape(
    ctx: &IrContext,
    op: OpRef,
    errors: &mut Vec<ValidationError>,
) {
    let Some(block) = ctx.op(op).parent_block else {
        push_op_error(ctx, op, errors, "must be attached to a block");
        return;
    };
    if ctx.block(block).ops.last().copied() != Some(op) {
        push_op_error(ctx, op, errors, "must be the final operation in its block");
    }
}

/// Validate graph-wide `tribute_control` invariants.
pub fn validate_whole_ir(
    ctx: &IrContext,
    module: Module,
    compiler_intrinsics: &[CompilerIntrinsicDeclaration],
) -> ValidationResult {
    whole_ir(
        ctx,
        module,
        &SymbolTable::collect(ctx, module),
        compiler_intrinsics,
    )
}

fn whole_ir(
    ctx: &IrContext,
    module: Module,
    symbols: &SymbolTable,
    compiler_intrinsics: &[CompilerIntrinsicDeclaration],
) -> ValidationResult {
    let mut errors = Vec::new();
    let Some(body) = module.body(ctx) else {
        return ValidationResult { errors };
    };
    let declarations = &declared_operations(ctx, module, &mut errors);
    validate_module_symbols(ctx, module, symbols, &mut errors);
    validate_row_scopes(ctx, module, &declared_instances(ctx, module), &mut errors);
    validate_local_quantification(ctx, module, &mut errors);
    validate_evidence_tails(ctx, body, symbols, &mut errors);
    validate_lambda_captures(ctx, body, &mut errors);
    let declarations = declaration_map(ctx, declarations, &mut errors);
    if specialized(ctx, module).is_none() {
        push_type_error(
            &mut errors,
            "class specialization replaces every tribute_control func, lambda, func_ref, \
             call_indirect, and tail_call_indirect with its tribute_cps operation",
        );
    }
    let reachable_types = collect_reachable_ir_types(ctx, module.op());
    let nominal_layouts = canonical_nominal_layouts(ctx, &reachable_types, &mut errors);
    validate_managed_reference_boundaries(
        ctx,
        body,
        &reachable_types,
        &nominal_layouts,
        &mut errors,
    );
    validate_callable_origins(
        ctx,
        body,
        symbols,
        compiler_intrinsics,
        &declarations,
        &nominal_layouts,
        &mut errors,
    );
    validate_return_contracts(ctx, body, &mut errors);
    validate_declaration_uses(ctx, body, &declarations, &mut errors);
    validate_resume_ownership(ctx, body, &mut errors);
    validate_token_placements(ctx, body, &mut errors);
    ValidationResult { errors }
}

/// Run the complete local and whole-IR `tribute_control` validation.
///
/// Ability operations are checked against the module's `ability.decl`
/// definitions. The [`SymbolTable`] is queried through `analyses`.
pub fn validate(
    ctx: &IrContext,
    module: Module,
    compiler_intrinsics: &[CompilerIntrinsicDeclaration],
    analyses: &mut AnalysisCache,
) -> ValidationResult {
    let symbols = analyses.require::<SymbolTable>(ctx, module.op());
    let mut local = validate_local(ctx, module);
    local
        .errors
        .extend(whole_ir(ctx, module, &symbols, compiler_intrinsics).errors);
    local
}
