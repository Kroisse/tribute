//! Typed native ownership and RTTI planning.
//!
//! The plan is built after structured control has been normalized to `cf` and
//! before `func_to_clif` erases semantic reference types.  Building and
//! validating it never mutates the input IR.

use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;
use std::fmt;
use std::ops::ControlFlow;

use tribute_ir::dialect::adt;
use tribute_ir::dialect::adt::layout::{get_enum_variants, get_struct_fields};
use trunk_ir::analysis::AnalysisCache;
use trunk_ir::callable::{CallableBody, classify_callable_body};
use trunk_ir::context::IrContext;
use trunk_ir::dialect::{core, func};
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::rewrite::Module;
use trunk_ir::symbol_table::qualified_name;
use trunk_ir::transforms::call_graph::{CallGraph, recursive_functions};
use trunk_ir::walk::{WalkAction, walk_op};

use crate::target_abi::{CONSUMED, OWNERSHIP_ATTR};
use trunk_ir::{
    Attribute, BlockRef, OpRef, RegionRef, StringRef, Symbol, SymbolPath, TypeRef, ValueDef,
    ValueRef,
};

mod actions;
mod cfg;
mod facts;
mod liveness;
use actions::{
    ActionInputs, exact_into_raw_transfers, plan_function_actions, validate_result_contract,
};
use cfg::ValidatedFlatCfg;
pub use facts::{NativeOwnershipFunctionFacts, NativeOwnershipModuleFacts};
use liveness::BlockLiveness;
pub use liveness::NativeManagedLiveness;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwnershipPlanError(String);

impl OwnershipPlanError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for OwnershipPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "typed native ownership plan: {}", self.0)
    }
}

impl std::error::Error for OwnershipPlanError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EntryOwnership {
    Plain,
    Borrowed,
    Retained,
    Consumed,
}

/// Policy choices captured by the typed native ownership plan.  These are
/// deliberately decisions made while semantic managed types still exist; the
/// materializer never re-discovers either class of borrow after erasure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NativeOwnershipPlanOptions {
    pub elide_proven_borrowed_parameters: bool,
    pub elide_proven_field_borrows: bool,
}

impl NativeOwnershipPlanOptions {
    pub const fn production() -> Self {
        Self {
            elide_proven_borrowed_parameters: true,
            elide_proven_field_borrows: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActionKind {
    EntryAcquire,
    CallBorrow,
    CallRetain,
    CallAcquire,
    StoreAcquire,
    CopyAcquire,
    BorrowLoad,
    ReleaseReplacedField,
    FinalRelease,
    ReturnTransfer,
    TailTransfer,
    IntoRawTransfer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActionAnchor {
    BlockStart(BlockRef),
    Before(OpRef),
    After(OpRef),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OwnershipAction {
    pub kind: ActionKind,
    pub value: ValueRef,
    pub anchor: ActionAnchor,
    /// Distinguishes independently owned destinations at the same operation.
    pub destination: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionOwnershipPlan {
    symbol: SymbolPath,
    operation: OpRef,
    entries: Vec<EntryOwnership>,
    actions: Vec<OwnershipAction>,
}

impl FunctionOwnershipPlan {
    pub fn symbol(&self) -> &SymbolPath {
        &self.symbol
    }

    pub fn operation(&self) -> OpRef {
        self.operation
    }

    pub fn entries(&self) -> &[EntryOwnership] {
        &self.entries
    }

    pub fn actions(&self) -> &[OwnershipAction] {
        &self.actions
    }
}

pub use tribute_ir::dialect::tribute_rtti::FieldKind;
use tribute_ir::dialect::tribute_rtti::{allocation_descriptor, descriptor_field_types};

/// One runtime type descriptor: a struct layout, or one variant of an enum
/// layout, with how the runtime reads each of its fields.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RttiTypePlan {
    pub ty: TypeRef,
    pub tag: Option<StringRef>,
    pub fields: Vec<FieldKind>,
}

impl RttiTypePlan {
    pub fn key(&self) -> (TypeRef, Option<StringRef>) {
        (self.ty, self.tag)
    }
}

#[derive(Debug, Clone)]
pub struct NativeOwnershipPlan {
    module: OpRef,
    managed_layouts: HashSet<TypeRef>,
    /// The compiler-generated native layout for semantic closure values. This
    /// exact `TypeRef` is collected before type erasure, never inferred from a
    /// physical pointer.
    closure_layout: Option<TypeRef>,
    functions: Vec<FunctionOwnershipPlan>,
    rtti_types: Vec<RttiTypePlan>,
}

impl NativeOwnershipPlan {
    pub fn functions(&self) -> &[FunctionOwnershipPlan] {
        &self.functions
    }

    pub fn rtti_types(&self) -> &[RttiTypePlan] {
        &self.rtti_types
    }

    pub fn function(&self, symbol: &Symbol) -> Option<&FunctionOwnershipPlan> {
        self.functions
            .iter()
            .find(|function| function.symbol == *symbol)
    }

    pub fn is_managed_type(&self, ctx: &IrContext, ty: TypeRef) -> bool {
        is_typed_managed_reference(ctx, ty, &self.managed_layouts)
    }

    /// Resolve an exact semantic managed reference to the unique aggregate
    /// layout that owns its native allocation.  This is intentionally a
    /// plan-owned nominal lookup: physical pointers never participate.
    pub(crate) fn allocation_layout_for_type(
        &self,
        ctx: &IrContext,
        ty: TypeRef,
    ) -> Result<TypeRef, OwnershipPlanError> {
        if self.managed_layouts.contains(&ty) {
            return Ok(ty);
        }
        let data = ctx.get_type(ty);
        if data.dialect != "adt" || data.name != "typeref" {
            return Err(OwnershipPlanError::new(format!(
                "managed release type {ty} ({data:?}) has no exact nominal allocation layout"
            )));
        }
        let name = data.attrs.get_string_ref("name").ok_or_else(|| {
            OwnershipPlanError::new("managed release typeref lacks nominal allocation identity")
        })?;
        let mut layouts = self
            .managed_layouts
            .iter()
            .copied()
            .filter(|layout| ctx.get_type(*layout).attrs.get_string_ref("name") == Some(name));
        let layout = layouts.next().ok_or_else(|| {
            OwnershipPlanError::new("managed release typeref has no planned allocation layout")
        })?;
        if layouts.next().is_some() {
            return Err(OwnershipPlanError::new(
                "managed release typeref has ambiguous planned allocation layouts",
            ));
        }
        Ok(layout)
    }

    /// Check that the planned RTTI layouts are exactly the allocation layouts
    /// of `module`. No dialect/name/layout matching is permitted here.
    fn validate_rtti_types(
        &self,
        ctx: &IrContext,
        module: Module,
    ) -> Result<(), OwnershipPlanError> {
        let planned = self
            .rtti_types
            .iter()
            .map(RttiTypePlan::key)
            .collect::<HashSet<_>>();
        if planned.len() != self.rtti_types.len() {
            return Err(OwnershipPlanError::new(
                "RTTI plan has duplicate descriptor identities",
            ));
        }

        let mut current = HashSet::default();
        walk_module(ctx, module, |op| {
            if let Some(descriptor) = allocation_descriptor(ctx, op) {
                current.insert(descriptor);
            }
        });
        if current != planned {
            return Err(OwnershipPlanError::new(
                "RTTI allocation descriptors differ from the plan",
            ));
        }
        Ok(())
    }

    /// Revalidate the stable pre-erasure identities before a future consumer
    /// materializes this plan.
    pub fn validate_against(
        &self,
        ctx: &IrContext,
        module: Module,
    ) -> Result<(), OwnershipPlanError> {
        if self.module != module.op() {
            return Err(OwnershipPlanError::new("module identity is stale"));
        }
        let mut function_ops = Vec::new();
        walk_module(ctx, module, |op| {
            if func::Func::matches(ctx, op) {
                function_ops.push(op);
            }
        });
        let mut current_functions = HashSet::default();
        for op in function_ops {
            if matches!(
                ownership_callable_body(ctx, op)?,
                CallableBody::Definition { .. }
            ) {
                current_functions.insert(op);
            }
        }
        let planned_functions = self
            .functions
            .iter()
            .map(|function| function.operation)
            .collect::<HashSet<_>>();
        if planned_functions.len() != self.functions.len() || current_functions != planned_functions
        {
            return Err(OwnershipPlanError::new(
                "reachable function identities differ from the ownership plan",
            ));
        }
        validate_plan(ctx, self)?;
        let current_closure_layout = collect_closure_layout(ctx, module, &self.managed_layouts)?;
        if current_closure_layout != self.closure_layout {
            return Err(OwnershipPlanError::new(
                "semantic closure allocation layouts differ from the ownership plan",
            ));
        }
        self.validate_rtti_types(ctx, module)?;
        Ok(())
    }
}

/// The single semantic managed-reference predicate used by ownership and RTTI
/// planning. `core.ptr` and other physical pointer-shaped types are never
/// managed here.
fn is_typed_managed_reference(
    ctx: &IrContext,
    ty: TypeRef,
    managed_layouts: &HashSet<TypeRef>,
) -> bool {
    if managed_layouts.contains(&ty) {
        return true;
    }
    let data = ctx.get_type(ty);
    (data.dialect == "adt" && data.name == "typeref")
        || (data.dialect == "tribute_rt" && (matches!(data.name.as_str(), "anyref" | "intref")))
}

fn is_managed_value(ctx: &IrContext, value: ValueRef, managed_layouts: &HashSet<TypeRef>) -> bool {
    is_typed_managed_reference(ctx, ctx.value_ty(value), managed_layouts)
}

fn is_anyref_type(ctx: &IrContext, ty: TypeRef) -> bool {
    let data = ctx.get_type(ty);
    data.dialect == "tribute_rt" && data.name == "anyref"
}

/// Build the plan while reusing cached policy-neutral ownership flow facts.
///
/// Callers inside one pipeline phase share an [`AnalysisCache`] so the module
/// and function facts are computed once per target. The facts never depend on
/// `options`; borrow elision and entry ownership stay policy decisions made
/// here.
pub fn build_native_ownership_plan(
    ctx: &IrContext,
    module: Module,
    options: NativeOwnershipPlanOptions,
    analyses: &mut AnalysisCache,
) -> Result<NativeOwnershipPlan, OwnershipPlanError> {
    let module_facts = analyses
        .get::<NativeOwnershipModuleFacts>(ctx, module.op())
        .map_err(|error| OwnershipPlanError::new(error.to_string()))?;
    let definitions = module_facts.definitions();
    let managed_layouts = module_facts.managed_layouts().clone();
    let closure_layout = collect_closure_layout(ctx, module, &managed_layouts)?;
    let rtti_types = build_rtti_plan(ctx, module, &managed_layouts)?;
    let entry_contracts = compute_entry_contracts(
        ctx,
        &analyses.require::<CallGraph>(ctx, module.op()),
        definitions,
        &managed_layouts,
        options.elide_proven_borrowed_parameters,
    )?;

    let mut functions = Vec::new();
    for &op in module_facts.function_ops() {
        let Ok(function) = func::Func::from_op(ctx, op) else {
            continue;
        };
        let symbol =
            qualified_name(ctx, op).unwrap_or_else(|| SymbolPath::from(function.sym_name(ctx)));
        if let CallableBody::Declaration = ownership_callable_body(ctx, op)? {
            validate_bodyless_signature(ctx, op, &managed_layouts)?;
            continue;
        }
        let facts = analyses
            .get::<NativeOwnershipFunctionFacts>(ctx, op)
            .map_err(|error| OwnershipPlanError::new(error.to_string()))?;
        let liveness = analyses
            .get::<NativeManagedLiveness>(ctx, op)
            .map_err(|error| OwnershipPlanError::new(error.to_string()))?;
        let entries = entry_contracts
            .get(&symbol)
            .cloned()
            .ok_or_else(|| OwnershipPlanError::new("defined function has no entry contract"))?;
        let actions = plan_function_actions(
            ctx,
            ActionInputs {
                facts: &facts,
                liveness: liveness.view(options.elide_proven_field_borrows),
            },
            &entries,
            &entry_contracts,
            definitions,
            &managed_layouts,
            options.elide_proven_field_borrows,
        )?;
        functions.push(FunctionOwnershipPlan {
            symbol,
            operation: op,
            entries,
            actions,
        });
    }

    let plan = NativeOwnershipPlan {
        module: module.op(),
        managed_layouts,
        closure_layout,
        functions,
        rtti_types,
    };
    validate_plan(ctx, &plan)?;
    Ok(plan)
}

fn ownership_callable_body(ctx: &IrContext, op: OpRef) -> Result<CallableBody, OwnershipPlanError> {
    classify_callable_body(ctx, op).map_err(|error| {
        let symbol = ctx
            .op(op)
            .attributes
            .get_str(ctx, "sym_name")
            .map(Symbol::new)
            .map(|name| format!("@{name}"))
            .unwrap_or_else(|| "<unnamed>".into());
        OwnershipPlanError::new(format!("func.func {symbol}: {error}"))
    })
}

fn collect_closure_layout(
    ctx: &IrContext,
    module: Module,
    managed_layouts: &HashSet<TypeRef>,
) -> Result<Option<TypeRef>, OwnershipPlanError> {
    let mut closure_layout = None;
    let mut ambiguous = false;
    walk_module(ctx, module, |op| {
        let Ok(new) = adt::StructNew::from_op(ctx, op) else {
            return;
        };
        let layout = new.r#type(ctx);
        if !crate::closure_lower::is_closure_struct_type_ref(ctx, layout) {
            return;
        }
        if !managed_layouts.contains(&layout) {
            return;
        }
        match closure_layout {
            Some(existing) if existing != layout => {
                // The generated closure layout is an exact compiler ABI
                // identity; more than one identity is stale plan input.
                ambiguous = true;
            }
            None => closure_layout = Some(layout),
            Some(_) => {}
        }
    });
    if ambiguous {
        return Err(OwnershipPlanError::new(
            "semantic closure has ambiguous compiler-owned allocation layouts",
        ));
    }
    Ok(closure_layout)
}

fn collect_function_definitions(
    ctx: &IrContext,
    module_ops: &[OpRef],
) -> Result<HashMap<SymbolPath, OpRef>, OwnershipPlanError> {
    let mut definitions = HashMap::default();
    for &op in module_ops {
        let Ok(function) = func::Func::from_op(ctx, op) else {
            continue;
        };
        // Direct callees name their targets by root-qualified path.
        let symbol =
            qualified_name(ctx, op).unwrap_or_else(|| SymbolPath::from(function.sym_name(ctx)));
        if definitions.insert(symbol.clone(), op).is_some() {
            return Err(OwnershipPlanError::new(format!(
                "duplicate function identity @{symbol}"
            )));
        }
    }
    Ok(definitions)
}

fn collect_and_validate_managed_layouts(
    ctx: &IrContext,
    module: Module,
) -> Result<HashSet<TypeRef>, OwnershipPlanError> {
    let mut layouts = HashSet::default();
    let mut nominal_layouts: HashMap<StringRef, Vec<TypeRef>> = HashMap::default();
    let mut typerefs = HashSet::default();
    let mut pending_typerefs = Vec::new();
    for &(_, ty) in ctx.type_aliases() {
        index_nominal_layout(ctx, ty, &mut nominal_layouts);
    }
    let mut visited_types = HashSet::default();
    walk_module(ctx, module, |op| {
        for &ty in ctx.op_result_types(op) {
            collect_reachable_type_contract(
                ctx,
                ty,
                &mut typerefs,
                &mut pending_typerefs,
                &mut nominal_layouts,
                &mut layouts,
                &mut visited_types,
            );
        }
        for &operand in ctx.op_operands(op) {
            collect_reachable_type_contract(
                ctx,
                ctx.value_ty(operand),
                &mut typerefs,
                &mut pending_typerefs,
                &mut nominal_layouts,
                &mut layouts,
                &mut visited_types,
            );
        }
        for attribute in ctx.op(op).attributes.values() {
            collect_reachable_attribute_type_contract(
                ctx,
                attribute,
                &mut typerefs,
                &mut pending_typerefs,
                &mut nominal_layouts,
                &mut layouts,
                &mut visited_types,
            );
        }
        for region in ctx.op_regions(op) {
            for block in ctx.region(region).blocks.iter().copied() {
                for &argument in ctx.block_args(block) {
                    collect_reachable_type_contract(
                        ctx,
                        ctx.value_ty(argument),
                        &mut typerefs,
                        &mut pending_typerefs,
                        &mut nominal_layouts,
                        &mut layouts,
                        &mut visited_types,
                    );
                }
            }
        }
        if let Ok(new) = adt::StructNew::from_op(ctx, op) {
            let layout = new.r#type(ctx);
            if get_struct_fields(ctx, layout).is_some() {
                layouts.insert(layout);
            }
        } else if let Ok(new) = adt::VariantNew::from_op(ctx, op) {
            let layout = new.r#type(ctx);
            if get_enum_variants(ctx, layout).is_some() {
                layouts.insert(layout);
            }
        }
    });

    while let Some(typeref) = pending_typerefs.pop() {
        let Some(name) = ctx.get_type(typeref).attrs.get_string_ref("name") else {
            return Err(OwnershipPlanError::new(
                "adt.typeref lacks nominal identity",
            ));
        };
        if !ctx.get_type(typeref).params.is_empty() {
            return Err(OwnershipPlanError::new(format!(
                "adt.typeref {:?} has unexpected parameters",
                ctx.str(name)
            )));
        }
        if nominal_layouts
            .get(&name)
            .is_none_or(|layouts| layouts.len() != 1)
        {
            return Err(OwnershipPlanError::new(format!(
                "adt.typeref {:?} has no unique native layout",
                ctx.str(name)
            )));
        }
        let layout = nominal_layouts[&name][0];
        layouts.insert(layout);
        collect_reachable_type_contract(
            ctx,
            layout,
            &mut typerefs,
            &mut pending_typerefs,
            &mut nominal_layouts,
            &mut layouts,
            &mut visited_types,
        );
    }
    Ok(layouts)
}

fn index_nominal_layout(
    ctx: &IrContext,
    ty: TypeRef,
    nominal_layouts: &mut HashMap<StringRef, Vec<TypeRef>>,
) {
    let data = ctx.get_type(ty);
    if data.dialect == "adt"
        && (matches!(data.name.as_str(), "struct" | "enum"))
        && let Some(name) = data.attrs.get_string_ref("name")
    {
        let layouts = nominal_layouts.entry(name).or_default();
        if !layouts.contains(&ty) {
            layouts.push(ty);
        }
    }
}

fn collect_reachable_type_contract(
    ctx: &IrContext,
    ty: TypeRef,
    typerefs: &mut HashSet<TypeRef>,
    pending_typerefs: &mut Vec<TypeRef>,
    nominal_layouts: &mut HashMap<StringRef, Vec<TypeRef>>,
    layouts: &mut HashSet<TypeRef>,
    visited_types: &mut HashSet<TypeRef>,
) {
    if !visited_types.insert(ty) {
        return;
    }
    let data = ctx.get_type(ty);
    index_nominal_layout(ctx, ty, nominal_layouts);
    if (data.dialect == "adt") && (matches!(data.name.as_str(), "struct" | "enum")) {
        layouts.insert(ty);
    }
    if data.dialect == "adt" && data.name == "typeref" && typerefs.insert(ty) {
        pending_typerefs.push(ty);
    }
    for &parameter in &data.params {
        collect_reachable_type_contract(
            ctx,
            parameter,
            typerefs,
            pending_typerefs,
            nominal_layouts,
            layouts,
            visited_types,
        );
    }
    for attribute in data.attrs.values() {
        collect_reachable_attribute_type_contract(
            ctx,
            attribute,
            typerefs,
            pending_typerefs,
            nominal_layouts,
            layouts,
            visited_types,
        );
    }
}

fn collect_reachable_attribute_type_contract(
    ctx: &IrContext,
    attribute: &trunk_ir::Attribute,
    typerefs: &mut HashSet<TypeRef>,
    pending_typerefs: &mut Vec<TypeRef>,
    nominal_layouts: &mut HashMap<StringRef, Vec<TypeRef>>,
    layouts: &mut HashSet<TypeRef>,
    visited_types: &mut HashSet<TypeRef>,
) {
    attribute.visit_types(&mut |ty| {
        collect_reachable_type_contract(
            ctx,
            ty,
            typerefs,
            pending_typerefs,
            nominal_layouts,
            layouts,
            visited_types,
        );
    });
}

fn nominal_types_compatible(ctx: &IrContext, left: TypeRef, right: TypeRef) -> bool {
    let identity = |ty| {
        let data = ctx.get_type(ty);
        (data.dialect == "adt").then(|| data.attrs.get_string_ref("name"))?
    };
    identity(left).is_some() && identity(left) == identity(right)
}

fn types_compatible(
    ctx: &IrContext,
    actual: TypeRef,
    expected: TypeRef,
    managed_layouts: &HashSet<TypeRef>,
) -> bool {
    actual == expected
        || (is_typed_managed_reference(ctx, actual, managed_layouts)
            && is_typed_managed_reference(ctx, expected, managed_layouts)
            && (is_anyref_type(ctx, actual)
                || is_anyref_type(ctx, expected)
                || nominal_types_compatible(ctx, actual, expected)))
}

fn build_rtti_plan(
    ctx: &IrContext,
    module: Module,
    managed_layouts: &HashSet<TypeRef>,
) -> Result<Vec<RttiTypePlan>, OwnershipPlanError> {
    let mut order = Vec::new();
    let mut seen = HashSet::default();
    walk_module(ctx, module, |op| {
        if let Some(descriptor) = allocation_descriptor(ctx, op)
            && managed_layouts.contains(&descriptor.0)
            && seen.insert(descriptor)
        {
            order.push(descriptor);
        }
    });

    order
        .into_iter()
        .map(|(ty, tag)| {
            let fields = build_field_kinds(ctx, ty, tag, managed_layouts)?;
            Ok(RttiTypePlan { ty, tag, fields })
        })
        .collect()
}

fn build_field_kinds(
    ctx: &IrContext,
    ty: TypeRef,
    tag: Option<StringRef>,
    managed_layouts: &HashSet<TypeRef>,
) -> Result<Vec<FieldKind>, OwnershipPlanError> {
    let fields = descriptor_field_types(ctx, ty, tag)
        .map_err(|error| OwnershipPlanError::new(format!("RTTI descriptor: {error}")))?;
    Ok(fields
        .into_iter()
        .map(|field| field_kind(ctx, field, managed_layouts))
        .collect())
}

/// How the runtime reads a field of semantic type `ty`. Managed-ness follows
/// [`is_typed_managed_reference`], so release and ownership agree.
fn field_kind(ctx: &IrContext, ty: TypeRef, managed_layouts: &HashSet<TypeRef>) -> FieldKind {
    if is_typed_managed_reference(ctx, ty, managed_layouts) {
        let data = ctx.get_type(ty);
        let dynamic =
            data.dialect == "tribute_rt" && (matches!(data.name.as_str(), "anyref" | "intref"));
        return if dynamic {
            FieldKind::Dynamic
        } else {
            FieldKind::Managed
        };
    }
    // Pointers, code references, and runtime buffers that ownership does not
    // manage are raw: neither released nor followed.
    FieldKind::scalar(ctx, ty).unwrap_or(FieldKind::Raw)
}

fn compute_entry_contracts(
    ctx: &IrContext,
    call_graph: &CallGraph,
    definitions: &HashMap<SymbolPath, OpRef>,
    managed_layouts: &HashSet<TypeRef>,
    elide_proven_borrowed_parameters: bool,
) -> Result<HashMap<SymbolPath, Vec<EntryOwnership>>, OwnershipPlanError> {
    let recursive = recursive_functions(call_graph);
    let mut summaries = HashMap::default();
    for (symbol, &op) in definitions {
        let symbol = symbol.clone();
        let entry = match ownership_callable_body(ctx, op)? {
            CallableBody::Declaration => {
                let signature = validate_bodyless_signature(ctx, op, managed_layouts)?;
                let entries = bodyless_c_entry_contract(ctx, op, managed_layouts)?
                    .unwrap_or_else(|| vec![EntryOwnership::Plain; signature.inputs(ctx).len()]);
                summaries.insert(symbol, entries);
                continue;
            }
            CallableBody::Definition { entry, .. } => entry,
        };
        let ineligible = recursive.contains(&symbol) || ctx.op(op).attributes.contains_key("abi");
        let signature = ctx
            .op(op)
            .attributes
            .get_type("type")
            .and_then(|ty| func::FuncSig::from_type_ref(ctx, ty))
            .ok_or_else(|| OwnershipPlanError::new("function definition lacks exact signature"))?;
        let consumed = consumed_inputs(ctx, signature)?;
        if consumed.len() != ctx.block_args(entry).len() {
            return Err(OwnershipPlanError::new(
                "function entry arity differs from its exact signature",
            ));
        }
        summaries.insert(
            symbol,
            ctx.block_args(entry)
                .iter()
                .zip(consumed)
                .map(|(&parameter, consumed)| {
                    if !is_managed_value(ctx, parameter, managed_layouts) {
                        EntryOwnership::Plain
                    } else if consumed {
                        EntryOwnership::Consumed
                    } else if ineligible || !elide_proven_borrowed_parameters {
                        EntryOwnership::Retained
                    } else {
                        EntryOwnership::Borrowed
                    }
                })
                .collect::<Vec<_>>(),
        );
    }

    loop {
        let mut changed = false;
        for (symbol, &op) in definitions {
            let symbol = symbol.clone();
            let CallableBody::Definition {
                region: body,
                entry,
            } = ownership_callable_body(ctx, op)?
            else {
                continue;
            };
            for (index, &parameter) in ctx.block_args(entry).iter().enumerate() {
                if summaries[&symbol][index] == EntryOwnership::Borrowed
                    && !value_is_borrowed(ctx, body, parameter, &summaries, &mut HashSet::default())
                {
                    summaries.get_mut(&symbol).unwrap()[index] = EntryOwnership::Retained;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    Ok(summaries)
}

/// Native `extern "C"` declarations are the one trusted bodyless managed
/// boundary. Their logical managed arguments are borrowed for the call; a
/// logical managed result is a fresh owned value, as for every call result.
fn bodyless_c_entry_contract(
    ctx: &IrContext,
    op: OpRef,
    managed_layouts: &HashSet<TypeRef>,
) -> Result<Option<Vec<EntryOwnership>>, OwnershipPlanError> {
    if ctx.op(op).attributes.get_str(ctx, "abi") != Some("C") {
        return Ok(None);
    }
    let signature = ctx
        .op(op)
        .attributes
        .get_type("type")
        .and_then(|ty| func::FuncSig::from_type_ref(ctx, ty))
        .ok_or_else(|| OwnershipPlanError::new("bodyless function lacks exact signature"))?;
    Ok(Some(
        signature
            .inputs(ctx)
            .iter()
            .map(|&ty| {
                if is_typed_managed_reference(ctx, ty, managed_layouts) {
                    EntryOwnership::Borrowed
                } else {
                    EntryOwnership::Plain
                }
            })
            .collect(),
    ))
}

/// Which inputs of `signature` carry the `consumed` entry contract that the
/// representation/ABI boundary records in the exact physical signature.
///
/// The marker is inert on unmanaged inputs; callers combine it with the typed
/// managed-reference contract. Any other ownership value is rejected.
fn consumed_inputs(
    ctx: &IrContext,
    signature: func::FuncSig,
) -> Result<Vec<bool>, OwnershipPlanError> {
    signature
        .input_attrs(ctx)
        .map(|attrs| match attrs.get(OWNERSHIP_ATTR) {
            None => Ok(false),
            Some(Attribute::String(mode)) if ctx.str(*mode) == CONSUMED => Ok(true),
            Some(_) => Err(OwnershipPlanError::new(format!(
                "unknown {OWNERSHIP_ATTR} parameter contract"
            ))),
        })
        .collect()
}

fn value_is_borrowed(
    ctx: &IrContext,
    body: RegionRef,
    value: ValueRef,
    summaries: &HashMap<SymbolPath, Vec<EntryOwnership>>,
    visiting: &mut HashSet<ValueRef>,
) -> bool {
    if !visiting.insert(value) {
        return true;
    }
    ctx.uses(value).iter().all(|use_| {
        let op = use_.user;
        if ctx
            .op(op)
            .parent_block
            .is_none_or(|block| ctx.block(block).parent_region != Some(body))
        {
            return false;
        }
        let index = use_.operand_index as usize;
        if (adt::StructGet::matches(ctx, op)
            || adt::VariantGet::matches(ctx, op)
            || adt::VariantIs::matches(ctx, op)
            || adt::RefIsNull::matches(ctx, op))
            && index == 0
        {
            return true;
        }
        if (adt::RefCast::matches(ctx, op) || core::UnrealizedConversionCast::matches(ctx, op))
            && index == 0
            && ctx.op_results(op).len() == 1
        {
            return value_is_borrowed(ctx, body, ctx.op_result(op, 0), summaries, visiting);
        }
        if let Ok(call) = func::Call::from_op(ctx, op) {
            return summaries
                .get(call.callee(ctx))
                .and_then(|entries| entries.get(index))
                == Some(&EntryOwnership::Borrowed);
        }
        false
    })
}

fn validate_bodyless_signature(
    ctx: &IrContext,
    op: OpRef,
    managed_layouts: &HashSet<TypeRef>,
) -> Result<func::FuncSig, OwnershipPlanError> {
    let signature = ctx
        .op(op)
        .attributes
        .get_type("type")
        .and_then(|ty| func::FuncSig::from_type_ref(ctx, ty))
        .ok_or_else(|| OwnershipPlanError::new("bodyless function lacks exact signature"))?;
    if ctx.op(op).attributes.get_str(ctx, "abi") == Some("C") {
        return Ok(signature);
    }
    if signature
        .inputs(ctx)
        .iter()
        .chain(signature.results(ctx))
        .any(|&ty| is_typed_managed_reference(ctx, ty, managed_layouts))
    {
        return Err(OwnershipPlanError::new(
            "bodyless native declaration exposes a managed reference",
        ));
    }
    Ok(signature)
}

fn validate_function_contract(
    ctx: &IrContext,
    function_op: OpRef,
    cfg: &ValidatedFlatCfg,
    managed_layouts: &HashSet<TypeRef>,
) -> Result<(), OwnershipPlanError> {
    let signature = ctx
        .op(function_op)
        .attributes
        .get_type("type")
        .and_then(|ty| func::FuncSig::from_type_ref(ctx, ty))
        .ok_or_else(|| OwnershipPlanError::new("defined function lacks exact signature"))?;
    let entry_args = ctx.block_args(cfg.entry());
    if entry_args.len() != signature.inputs(ctx).len()
        || entry_args
            .iter()
            .zip(signature.inputs(ctx))
            .any(|(&argument, &expected)| {
                let actual = ctx.value_ty(argument);
                (is_typed_managed_reference(ctx, actual, managed_layouts)
                    || is_typed_managed_reference(ctx, expected, managed_layouts))
                    && !types_compatible(ctx, actual, expected, managed_layouts)
            })
    {
        return Err(OwnershipPlanError::new(
            "function entry arguments differ from its exact signature",
        ));
    }
    for &block in cfg.blocks() {
        let terminator = cfg.terminator(block);
        if func::Return::matches(ctx, terminator) {
            validate_result_contract(
                ctx,
                ctx.op_operands(terminator),
                signature.results(ctx),
                managed_layouts,
                "function return",
            )?;
        }
    }
    Ok(())
}

fn validate_plan(ctx: &IrContext, plan: &NativeOwnershipPlan) -> Result<(), OwnershipPlanError> {
    let mut function_symbols = HashSet::default();
    for function in &plan.functions {
        if !function_symbols.insert(function.symbol.clone()) {
            return Err(OwnershipPlanError::new(
                "plan has duplicate function identity",
            ));
        }
        let Ok(_) = func::Func::from_op(ctx, function.operation) else {
            return Err(OwnershipPlanError::new(
                "planned function identity is stale",
            ));
        };
        let CallableBody::Definition {
            region: body,
            entry,
        } = ownership_callable_body(ctx, function.operation)?
        else {
            return Err(OwnershipPlanError::new("planned function body is stale"));
        };
        if function.entries.len() != ctx.block_args(entry).len()
            || function
                .entries
                .iter()
                .zip(ctx.block_args(entry))
                .any(|(mode, &value)| {
                    (*mode != EntryOwnership::Plain)
                        != is_managed_value(ctx, value, &plan.managed_layouts)
                })
        {
            return Err(OwnershipPlanError::new("planned entry contract is stale"));
        }
        let mut action_keys = HashSet::default();
        let mut action_targets = HashMap::default();
        for action in &function.actions {
            if !action_keys.insert((action.anchor, action.kind, action.value, action.destination)) {
                return Err(OwnershipPlanError::new(
                    "plan has duplicate ownership action",
                ));
            }
            let target = (action.anchor, action.value, action.destination);
            if let Some(previous) = action_targets.insert(target, action.kind)
                && !compatible_action_pair(previous, action.kind)
            {
                return Err(OwnershipPlanError::new(
                    "plan has conflicting ownership actions",
                ));
            }
            if action.kind != ActionKind::ReleaseReplacedField
                && !is_managed_value(ctx, action.value, &plan.managed_layouts)
            {
                return Err(OwnershipPlanError::new(format!(
                    "ownership action {:?} targets unmanaged or stale value {:?} of type {}",
                    action.kind,
                    action.value,
                    ctx.value_ty(action.value)
                )));
            }
            if action.kind == ActionKind::IntoRawTransfer
                && !validate_into_raw_transfer_action(
                    ctx,
                    action,
                    plan.closure_layout,
                    &plan.managed_layouts,
                )
            {
                return Err(OwnershipPlanError::new("into_raw transfer action is stale"));
            }
            let valid_anchor = match action.anchor {
                ActionAnchor::BlockStart(block) => ctx.block(block).parent_region == Some(body),
                ActionAnchor::Before(op) | ActionAnchor::After(op) => ctx
                    .op(op)
                    .parent_block
                    .is_some_and(|block| ctx.block(block).parent_region == Some(body)),
            };
            if !valid_anchor {
                return Err(OwnershipPlanError::new("ownership action anchor is stale"));
            }
            if value_region(ctx, action.value) != Some(body) {
                return Err(OwnershipPlanError::new("ownership action value is stale"));
            }
            if matches!(action.anchor, ActionAnchor::After(op) if func::TailCall::matches(ctx, op) || func::TailCallIndirect::matches(ctx, op))
            {
                return Err(OwnershipPlanError::new(
                    "ownership action follows a proper-tail terminator",
                ));
            }
        }
        validate_into_raw_transfer_groups(ctx, function, body, plan)?;
    }
    let mut rtti_types = HashSet::default();
    for entry in &plan.rtti_types {
        if !rtti_types.insert(entry.key()) || !plan.managed_layouts.contains(&entry.ty) {
            return Err(OwnershipPlanError::new(
                "RTTI plan has duplicate or stale descriptor",
            ));
        }
        if build_field_kinds(ctx, entry.ty, entry.tag, &plan.managed_layouts)? != entry.fields {
            return Err(OwnershipPlanError::new("RTTI field kinds are stale"));
        }
    }
    Ok(())
}

fn validate_into_raw_transfer_action(
    ctx: &IrContext,
    action: &OwnershipAction,
    closure_layout: Option<TypeRef>,
    managed_layouts: &HashSet<TypeRef>,
) -> bool {
    let ActionAnchor::Before(into_raw) = action.anchor else {
        return false;
    };
    if action.destination != 0 || !tribute_ir::dialect::tribute_rt::IntoRaw::matches(ctx, into_raw)
    {
        return false;
    }
    let ([source], [result]) = (ctx.op_operands(into_raw), ctx.op_results(into_raw)) else {
        return false;
    };
    let Some(closure_layout) = closure_layout else {
        return false;
    };
    let result_ty = ctx.get_type(ctx.value_ty(*result));
    *source == action.value
        && ctx.value_ty(*source) == closure_layout
        && is_managed_value(ctx, *source, managed_layouts)
        && result_ty.dialect == "core"
        && result_ty.name == "ptr"
}

fn validate_into_raw_transfer_groups(
    ctx: &IrContext,
    function: &FunctionOwnershipPlan,
    body: RegionRef,
    plan: &NativeOwnershipPlan,
) -> Result<(), OwnershipPlanError> {
    let mut sources = HashSet::default();
    for &block in &ctx.region(body).blocks {
        for &op in &ctx.block(block).ops {
            if !tribute_ir::dialect::tribute_rt::IntoRaw::matches(ctx, op) {
                continue;
            }
            let [source] = ctx.op_operands(op) else {
                return Err(OwnershipPlanError::new(
                    "tribute_rt.into_raw has malformed arity",
                ));
            };
            sources.insert(*source);
        }
    }

    for source in &sources {
        let (_, operations) = exact_into_raw_transfers(
            ctx,
            *source,
            plan.closure_layout.ok_or_else(|| {
                OwnershipPlanError::new(
                    "tribute_rt.into_raw requires the current compiler closure layout",
                )
            })?,
            &plan.managed_layouts,
        )?;
        let transfer_actions = function
            .actions
            .iter()
            .enumerate()
            .filter(|(_, action)| {
                action.kind == ActionKind::IntoRawTransfer && action.value == *source
            })
            .collect::<Vec<_>>();
        if transfer_actions.len() != operations.len()
            || transfer_actions
                .iter()
                .zip(&operations)
                .any(|((_, action), &op)| action.anchor != ActionAnchor::Before(op))
        {
            return Err(OwnershipPlanError::new(
                "into_raw transfer actions do not cover the exact grouped transfers",
            ));
        }

        let copy_actions = function
            .actions
            .iter()
            .enumerate()
            .filter(|(_, action)| {
                action.kind == ActionKind::CopyAcquire
                    && action.value == *source
                    && action.anchor == ActionAnchor::Before(operations[0])
            })
            .collect::<Vec<_>>();
        if copy_actions.len() != operations.len().saturating_sub(1)
            || copy_actions.iter().enumerate().any(|(index, (_, action))| {
                action.anchor != ActionAnchor::Before(operations[0])
                    || action.destination != (index + 1) as u32
            })
        {
            return Err(OwnershipPlanError::new(
                "into_raw grouped transfers have stale copy-acquire actions",
            ));
        }
        let first_transfer = transfer_actions[0].0;
        if copy_actions
            .iter()
            .any(|(index, _)| *index >= first_transfer)
        {
            return Err(OwnershipPlanError::new(
                "into_raw copy-acquire actions must precede the first transfer",
            ));
        }
    }

    if function.actions.iter().any(|action| {
        action.kind == ActionKind::IntoRawTransfer && !sources.contains(&action.value)
    }) {
        return Err(OwnershipPlanError::new("into_raw transfer action is stale"));
    }
    Ok(())
}

fn value_region(ctx: &IrContext, value: ValueRef) -> Option<RegionRef> {
    match ctx.value_def(value) {
        ValueDef::OpResult(op, _) => ctx
            .op(op)
            .parent_block
            .and_then(|block| ctx.block(block).parent_region),
        ValueDef::BlockArg(block, _) => ctx.block(block).parent_region,
    }
}

fn compatible_action_pair(left: ActionKind, right: ActionKind) -> bool {
    matches!(
        (left, right),
        (ActionKind::CopyAcquire, ActionKind::ReturnTransfer)
            | (ActionKind::CopyAcquire, ActionKind::TailTransfer)
            | (ActionKind::EntryAcquire, ActionKind::FinalRelease)
            | (ActionKind::StoreAcquire, ActionKind::ReleaseReplacedField)
    )
}

fn walk_module(ctx: &IrContext, module: Module, mut visit: impl FnMut(OpRef)) {
    let _ = walk_op::<()>(ctx, module.op(), &mut |op| {
        visit(op);
        ControlFlow::Continue(WalkAction::Advance)
    });
}

#[cfg(test)]
mod facts_tests;

#[cfg(test)]
mod liveness_tests;

#[cfg(test)]
mod tests;
