//! Policy-neutral native ownership flow facts.
//!
//! These facts describe the typed, post-`scf_to_cf`, pre-`func_to_clif`
//! boundary. They never depend on [`NativeOwnershipPlanOptions`]: borrow
//! elision and entry ownership are policy choices applied by the planner that
//! consumes the facts, so conservative and field-borrow-enabled planning share
//! one cached computation per target.

use super::*;
use trunk_ir::analysis::{Analysis, AnalysisContext, AnalysisError};

/// Module-scope, policy-neutral facts shared by every defined function.
///
/// The target is a `core.module` op. Function discovery, definition identity,
/// and managed nominal layouts are validated once per module, so every
/// function-scope analysis and every planner consumer reuses them.
pub struct NativeOwnershipModuleFacts {
    function_ops: Vec<OpRef>,
    definitions: HashMap<Symbol, OpRef>,
    managed_layouts: HashSet<TypeRef>,
}

impl NativeOwnershipModuleFacts {
    /// Every `func.func` in document order, declarations included.
    pub fn function_ops(&self) -> &[OpRef] {
        &self.function_ops
    }

    /// Unique function symbol to its operation.
    pub fn definitions(&self) -> &HashMap<Symbol, OpRef> {
        &self.definitions
    }

    /// Validated managed nominal layouts reachable from the module.
    pub fn managed_layouts(&self) -> &HashSet<TypeRef> {
        &self.managed_layouts
    }
}

impl Analysis for NativeOwnershipModuleFacts {
    fn compute(ctx: &mut AnalysisContext<'_>, target: OpRef) -> Result<Self, AnalysisError> {
        let ir = ctx.ir();
        let module = Module::new(ir, target)
            .expect("native ownership module facts require a `core.module` target");
        compute_module_facts(ir, module).map_err(|error| AnalysisError::new::<Self>(target, error))
    }
}

/// Function-scope, policy-neutral ownership flow facts.
///
/// The target is a defined `func.func` op; declarations have no flow facts and
/// are reported as an unsupported analysis boundary. Computing this analysis
/// resolves the outermost [`NativeOwnershipModuleFacts`] through the same cache.
pub struct NativeOwnershipFunctionFacts {
    cfg: ValidatedFlatCfg,
    managed: HashSet<ValueRef>,
    aliases: HashMap<ValueRef, ValueRef>,
    projection_owners: HashMap<ValueRef, ValueRef>,
    block_flow: HashMap<BlockRef, BlockFlowFacts>,
}

impl NativeOwnershipFunctionFacts {
    /// Validated flat control-flow facts for the function body.
    pub(super) fn cfg(&self) -> &ValidatedFlatCfg {
        &self.cfg
    }

    /// Managed values before type erasure, after exact aliases are folded.
    pub fn managed_values(&self) -> &HashSet<ValueRef> {
        &self.managed
    }

    /// Exact managed alias roots: alias result to its managed root.
    ///
    /// Roots are derived from the typed cast contract only, never from
    /// `core.ptr` or physical shape.
    pub fn aliases(&self) -> &HashMap<ValueRef, ValueRef> {
        &self.aliases
    }

    /// Validated managed projection to its owning managed root.
    pub fn projection_owners(&self) -> &HashMap<ValueRef, ValueRef> {
        &self.projection_owners
    }

    /// Policy-neutral liveness inputs for one control-flow block.
    pub(super) fn block_flow(&self, block: BlockRef) -> &BlockFlowFacts {
        &self.block_flow[&block]
    }
}

impl Analysis for NativeOwnershipFunctionFacts {
    fn compute(ctx: &mut AnalysisContext<'_>, target: OpRef) -> Result<Self, AnalysisError> {
        let module_op = owning_module_op(ctx.ir(), target)
            .expect("native ownership function facts require a func.func inside a module");
        let module = ctx.get::<NativeOwnershipModuleFacts>(module_op)?;
        let ir = ctx.ir();
        compute_function_facts(ir, target, &module)
            .map_err(|error| AnalysisError::new::<Self>(target, error))
    }
}

/// Policy-neutral liveness inputs for one block.
///
/// The liveness scan visits block arguments, then each operation's operands
/// followed by its results, treating a managed root as a use only while it is
/// not already defined in the block. Recording the same ordered events keeps
/// that policy-neutral part out of the policy-applied fixed point.
pub(super) struct BlockFlowFacts {
    events: Vec<FlowEvent>,
}

impl BlockFlowFacts {
    pub(super) fn events(&self) -> &[FlowEvent] {
        &self.events
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum FlowKind {
    Use,
    Def,
}

#[derive(Clone, Copy)]
pub(super) struct FlowEvent {
    pub(super) root: ValueRef,
    pub(super) kind: FlowKind,
}

fn compute_module_facts(
    ctx: &IrContext,
    module: Module,
) -> Result<NativeOwnershipModuleFacts, OwnershipPlanError> {
    module
        .first_block(ctx)
        .ok_or_else(|| OwnershipPlanError::new("module has no body block"))?;
    let mut function_ops = Vec::new();
    walk_module(ctx, module, |op| {
        if func::Func::matches(ctx, op) {
            function_ops.push(op);
        }
    });
    // Reject malformed topology before entry-contract or call-graph analysis.
    for &op in &function_ops {
        ownership_callable_body(ctx, op)?;
    }
    let definitions = collect_function_definitions(ctx, &function_ops)?;
    let managed_layouts = collect_and_validate_managed_layouts(ctx, module)?;
    Ok(NativeOwnershipModuleFacts {
        function_ops,
        definitions,
        managed_layouts,
    })
}

fn compute_function_facts(
    ctx: &IrContext,
    op: OpRef,
    module: &NativeOwnershipModuleFacts,
) -> Result<NativeOwnershipFunctionFacts, OwnershipPlanError> {
    let managed_layouts = module.managed_layouts();
    let body = match ownership_callable_body(ctx, op)? {
        CallableBody::Definition { region, .. } => region,
        CallableBody::Declaration => {
            return Err(OwnershipPlanError::new(
                "native ownership function facts require a defined function",
            ));
        }
    };
    let cfg = ValidatedFlatCfg::build(ctx, body)?;
    validate_function_contract(ctx, op, &cfg, managed_layouts)?;
    let mut managed = collect_managed_values(ctx, cfg.blocks(), managed_layouts);
    let aliases = build_aliases(ctx, cfg.blocks(), &mut managed, managed_layouts)?;
    let projection_owners = collect_borrowed_loads(ctx, cfg.blocks(), managed_layouts, &aliases)?;
    let block_flow = collect_block_flow(ctx, &cfg, &managed, &aliases);
    Ok(NativeOwnershipFunctionFacts {
        cfg,
        managed,
        aliases,
        projection_owners,
        block_flow,
    })
}

/// Walk parents to the outermost enclosing `core.module`.
///
/// Ownership planning always targets the compilation's shared module, whose
/// facts cover every nested module subtree. Resolving the outermost module here
/// keys function facts to exactly the module facts the planner consumes, so a
/// nested function never observes a narrower managed-layout contract.
fn owning_module_op(ctx: &IrContext, mut op: OpRef) -> Option<OpRef> {
    let mut outermost = None;
    loop {
        if Module::new(ctx, op).is_some() {
            outermost = Some(op);
        }
        let Some(block) = ctx.op(op).parent_block else {
            break;
        };
        let Some(region) = ctx.block(block).parent_region else {
            break;
        };
        let Some(parent) = ctx.region(region).parent_op else {
            break;
        };
        op = parent;
    }
    outermost
}

fn collect_managed_values(
    ctx: &IrContext,
    blocks: &[BlockRef],
    managed_layouts: &HashSet<TypeRef>,
) -> HashSet<ValueRef> {
    let mut values = HashSet::new();
    for &block in blocks {
        for &value in ctx.block_args(block) {
            if is_managed_value(ctx, value, managed_layouts) {
                values.insert(value);
            }
        }
        for &op in &ctx.block(block).ops {
            for &value in ctx.op_results(op) {
                if is_managed_value(ctx, value, managed_layouts) {
                    values.insert(value);
                }
            }
        }
    }
    values
}

fn build_aliases(
    ctx: &IrContext,
    blocks: &[BlockRef],
    managed: &mut HashSet<ValueRef>,
    managed_layouts: &HashSet<TypeRef>,
) -> Result<HashMap<ValueRef, ValueRef>, OwnershipPlanError> {
    let mut aliases = HashMap::new();
    for &block in blocks {
        for &op in &ctx.block(block).ops {
            if !(adt::RefCast::matches(ctx, op)
                || adt::VariantCast::matches(ctx, op)
                || core::UnrealizedConversionCast::matches(ctx, op))
            {
                continue;
            }
            let ([input], [output]) = (ctx.op_operands(op), ctx.op_results(op)) else {
                return Err(OwnershipPlanError::new("managed alias has malformed arity"));
            };
            let input_managed = is_managed_value(ctx, *input, managed_layouts);
            let output_managed = is_managed_value(ctx, *output, managed_layouts);
            let input_data = ctx.types.get(ctx.value_ty(*input));
            let output_data = ctx.types.get(ctx.value_ty(*output));
            if input_data.dialect == Symbol::new("core")
                && input_data.name == Symbol::new("ptr")
                && output_data.dialect == Symbol::new("adt")
                && output_data.name == Symbol::new("typeref")
            {
                return Err(OwnershipPlanError::new(format!(
                    "raw pointer alias {op:?} masquerades as managed adt.typeref: {} -> {}",
                    ctx.value_ty(*input),
                    ctx.value_ty(*output)
                )));
            }
            if let Ok(cast) = adt::RefCast::from_op(ctx, op) {
                let target = cast.r#type(ctx);
                if ctx.value_ty(*output) != target
                    || !input_managed
                    || !output_managed
                    || (!is_anyref_type(ctx, ctx.value_ty(*input))
                        && !is_anyref_type(ctx, target)
                        && !nominal_types_compatible(ctx, ctx.value_ty(*input), target))
                {
                    return Err(OwnershipPlanError::new(
                        "adt.ref_cast does not preserve a compatible managed reference",
                    ));
                }
            }
            // A compiler-generated conversion of a managed closure value to
            // its exact callable representation keeps the source ownership
            // unit live through the callable use. This is a typed pre-erasure
            // handoff, not a `core.ptr` provenance rule.
            let callable_handoff = core::UnrealizedConversionCast::matches(ctx, op)
                && func::FuncSig::from_type_ref(ctx, ctx.value_ty(*output)).is_some();
            if input_managed
                && is_internal_closure_layout(ctx, ctx.value_ty(*input), managed_layouts)
                && is_core_ptr_type(ctx, ctx.value_ty(*output))
            {
                return Err(OwnershipPlanError::new(
                    "internal _closure to core.ptr handoff requires tribute_rt.into_raw",
                ));
            }
            if input_managed && (output_managed || callable_handoff) {
                let root = aliases.get(input).copied().unwrap_or(*input);
                aliases.insert(*output, root);
                if output_managed {
                    managed.remove(output);
                }
            }
        }
    }
    Ok(aliases)
}

pub(super) fn is_internal_closure_layout(
    ctx: &IrContext,
    ty: TypeRef,
    managed_layouts: &HashSet<TypeRef>,
) -> bool {
    if !managed_layouts.contains(&ty) || !crate::closure_lower::is_closure_struct_type_ref(ctx, ty)
    {
        return false;
    }
    let Some(fields) = get_struct_fields(ctx, ty) else {
        return false;
    };
    matches!(
        fields.as_slice(),
        [(code_name, code_ty), (environment_name, environment_ty)]
            if *code_name == Symbol::new("func_ptr")
                && *environment_name == Symbol::new("env")
                && is_core_i32_type(ctx, *code_ty)
                && is_anyref_type(ctx, *environment_ty)
    )
}

pub(super) fn is_core_ptr_type(ctx: &IrContext, ty: TypeRef) -> bool {
    let data = ctx.types.get(ty);
    data.dialect == Symbol::new("core") && data.name == Symbol::new("ptr")
}

fn is_core_i32_type(ctx: &IrContext, ty: TypeRef) -> bool {
    let data = ctx.types.get(ty);
    data.dialect == Symbol::new("core") && data.name == Symbol::new("i32")
}

pub(super) fn root_value(aliases: &HashMap<ValueRef, ValueRef>, value: ValueRef) -> ValueRef {
    aliases.get(&value).copied().unwrap_or(value)
}

pub(super) fn borrowed_owner(
    borrowed: &HashMap<ValueRef, ValueRef>,
    aliases: &HashMap<ValueRef, ValueRef>,
    value: ValueRef,
) -> Option<ValueRef> {
    let mut owner = root_value(aliases, value);
    let mut found = false;
    while let Some(next) = borrowed.get(&owner) {
        let next = root_value(aliases, *next);
        if next == owner {
            break;
        }
        owner = next;
        found = true;
    }
    found.then_some(owner)
}

fn collect_borrowed_loads(
    ctx: &IrContext,
    blocks: &[BlockRef],
    managed_layouts: &HashSet<TypeRef>,
    aliases: &HashMap<ValueRef, ValueRef>,
) -> Result<HashMap<ValueRef, ValueRef>, OwnershipPlanError> {
    let mut borrowed = HashMap::new();
    for &block in blocks {
        for &op in &ctx.block(block).ops {
            let source = if let Ok(get) = adt::StructGet::from_op(ctx, op) {
                Some(get.r#ref(ctx))
            } else if let Ok(get) = adt::VariantGet::from_op(ctx, op) {
                Some(get.r#ref(ctx))
            } else {
                None
            };
            let Some(source) = source else { continue };
            let [result] = ctx.op_results(op) else {
                return Err(OwnershipPlanError::new(
                    "ADT projection must have exactly one result",
                ));
            };
            validate_projection_contract(ctx, op, source, *result, managed_layouts)?;
            if is_managed_value(ctx, *result, managed_layouts) {
                borrowed.insert(*result, root_value(aliases, source));
            }
        }
    }
    Ok(borrowed)
}

fn validate_projection_contract(
    ctx: &IrContext,
    op: OpRef,
    source: ValueRef,
    result: ValueRef,
    managed_layouts: &HashSet<TypeRef>,
) -> Result<(), OwnershipPlanError> {
    let (layout, field_ty) = if let Ok(get) = adt::StructGet::from_op(ctx, op) {
        let fields = get_struct_fields(ctx, get.r#type(ctx))
            .ok_or_else(|| OwnershipPlanError::new("struct_get has invalid layout"))?;
        let (_, field_ty) = fields
            .get(get.field(ctx) as usize)
            .ok_or_else(|| OwnershipPlanError::new("struct_get field is stale"))?;
        (get.r#type(ctx), *field_ty)
    } else {
        let get = adt::VariantGet::from_op(ctx, op)
            .map_err(|_| OwnershipPlanError::new("unsupported ADT projection"))?;
        let variants = get_enum_variants(ctx, get.r#type(ctx))
            .ok_or_else(|| OwnershipPlanError::new("variant_get has invalid layout"))?;
        let fields = variants
            .iter()
            .find(|(tag, _)| *tag == get.tag(ctx))
            .map(|(_, fields)| fields)
            .ok_or_else(|| OwnershipPlanError::new("variant_get tag is stale"))?;
        let field_ty = fields
            .get(get.field(ctx) as usize)
            .ok_or_else(|| OwnershipPlanError::new("variant_get field is stale"))?;
        (get.r#type(ctx), *field_ty)
    };
    let source_ty = ctx.value_ty(source);
    let source_data = ctx.types.get(source_ty);
    let raw_source =
        source_data.dialect == Symbol::new("core") && source_data.name == Symbol::new("ptr");
    let result_managed = is_managed_value(ctx, result, managed_layouts);
    if (!raw_source && !types_compatible(ctx, source_ty, layout, managed_layouts))
        || (result_managed
            && !types_compatible(ctx, ctx.value_ty(result), field_ty, managed_layouts))
    {
        return Err(OwnershipPlanError::new(
            "ADT projection managed type contract is malformed",
        ));
    }
    Ok(())
}

/// Record the policy-neutral use/definition events liveness replays.
fn collect_block_flow(
    ctx: &IrContext,
    cfg: &ValidatedFlatCfg,
    managed: &HashSet<ValueRef>,
    aliases: &HashMap<ValueRef, ValueRef>,
) -> HashMap<BlockRef, BlockFlowFacts> {
    let mut flow = HashMap::new();
    for &block in cfg.blocks() {
        let mut events = Vec::new();
        for &argument in ctx.block_args(block) {
            if managed.contains(&argument) {
                events.push(FlowEvent {
                    root: argument,
                    kind: FlowKind::Def,
                });
            }
        }
        for &op in &ctx.block(block).ops {
            for &operand in ctx.op_operands(op) {
                events.push(FlowEvent {
                    root: root_value(aliases, operand),
                    kind: FlowKind::Use,
                });
            }
            for &result in ctx.op_results(op) {
                events.push(FlowEvent {
                    root: root_value(aliases, result),
                    kind: FlowKind::Def,
                });
            }
        }
        flow.insert(block, BlockFlowFacts { events });
    }
    flow
}
