//! Typed native ownership and RTTI planning.
//!
//! The plan is built after structured control has been normalized to `cf` and
//! before `func_to_clif` erases semantic reference types.  Building and
//! validating it never mutates the input IR.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::ops::ControlFlow;

use tribute_core::{CallingConvention, get_calling_convention, get_indirect_call_signature};
use tribute_ir::dialect::closure;
use trunk_ir::adt_layout::{get_enum_variants, get_struct_fields};
use trunk_ir::context::IrContext;
use trunk_ir::dialect::{adt, core, func};
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::rewrite::Module;
use trunk_ir::transforms::call_graph::{build_call_graph, recursive_functions};
use trunk_ir::walk::{WalkAction, walk_op};
use trunk_ir::{BlockRef, OpRef, RegionRef, Symbol, TypeRef, ValueDef, ValueRef};
use trunk_ir_cranelift_backend::passes::func_to_clif::TypeRewrite;

mod actions;
mod cfg;
use actions::{exact_into_raw_transfers, plan_function_actions, validate_result_contract};
use cfg::ValidatedFlatCfg;

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
    symbol: Symbol,
    operation: OpRef,
    entries: Vec<EntryOwnership>,
    actions: Vec<OwnershipAction>,
}

impl FunctionOwnershipPlan {
    pub fn symbol(&self) -> Symbol {
        self.symbol
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ManagedFieldBitmap {
    Struct(Vec<bool>),
    Enum(Vec<Vec<bool>>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RttiTypePlan {
    pub ty: TypeRef,
    pub fields: ManagedFieldBitmap,
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

    pub fn function(&self, symbol: Symbol) -> Option<&FunctionOwnershipPlan> {
        self.functions
            .iter()
            .find(|function| function.symbol == symbol)
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
        if closure::Closure::matches(ctx, ty) {
            return self.closure_layout.ok_or_else(|| {
                OwnershipPlanError::new(
                    "semantic closure release has no compiler-generated allocation layout",
                )
            });
        }
        let data = ctx.types.get(ty);
        if data.dialect != Symbol::new("adt") || data.name != Symbol::new("typeref") {
            return Err(OwnershipPlanError::new(format!(
                "managed release type {ty} ({data:?}) has no exact nominal allocation layout"
            )));
        }
        let name = data.attrs.get_symbol("name").ok_or_else(|| {
            OwnershipPlanError::new("managed release typeref lacks nominal allocation identity")
        })?;
        let mut layouts = self
            .managed_layouts
            .iter()
            .copied()
            .filter(|layout| ctx.types.get(*layout).attrs.get_symbol("name") == Some(name));
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

    /// Carry the typed RTTI bitmap through exact compiler-reported type
    /// rewrites. No dialect/name/layout matching is permitted here.
    pub fn remap_rtti_types(
        &self,
        ctx: &IrContext,
        module: Module,
        rewrites: &[TypeRewrite],
    ) -> Result<Vec<RttiTypePlan>, OwnershipPlanError> {
        let planned = self
            .rtti_types
            .iter()
            .map(|entry| entry.ty)
            .collect::<HashSet<_>>();
        let mut rewrite_map = HashMap::new();
        for rewrite in rewrites {
            if !planned.contains(&rewrite.source) {
                return Err(OwnershipPlanError::new(
                    "RTTI type rewrite has a stale source identity",
                ));
            }
            if rewrite_map.insert(rewrite.source, rewrite.target).is_some() {
                return Err(OwnershipPlanError::new(
                    "RTTI type rewrite has a duplicate source identity",
                ));
            }
        }

        let remapped = self
            .rtti_types
            .iter()
            .map(|entry| RttiTypePlan {
                ty: rewrite_map.get(&entry.ty).copied().unwrap_or(entry.ty),
                fields: entry.fields.clone(),
            })
            .collect::<Vec<_>>();
        let remapped_set = remapped
            .iter()
            .map(|entry| entry.ty)
            .collect::<HashSet<_>>();
        if remapped_set.len() != remapped.len() {
            return Err(OwnershipPlanError::new(
                "RTTI type rewrites have ambiguous target identities",
            ));
        }

        let mut current = HashSet::new();
        walk_module(ctx, module, |op| {
            let ty = adt::StructNew::from_op(ctx, op)
                .ok()
                .map(|new| new.r#type(ctx))
                .or_else(|| {
                    adt::VariantNew::from_op(ctx, op)
                        .ok()
                        .map(|new| new.r#type(ctx))
                });
            if let Some(ty) = ty {
                current.insert(ty);
            }
        });
        if current != remapped_set {
            return Err(OwnershipPlanError::new(
                "RTTI allocation layout identity changed without an exact rewrite",
            ));
        }
        Ok(remapped)
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
        let mut current_functions = HashSet::new();
        walk_module(ctx, module, |op| {
            if let Ok(function) = func::Func::from_op(ctx, op)
                && function
                    .body_if_present(ctx)
                    .is_some_and(|body| !ctx.region(body).blocks.is_empty())
            {
                current_functions.insert(op);
            }
        });
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
        self.remap_rtti_types(ctx, module, &[])?;
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
    let data = ctx.types.get(ty);
    (data.dialect == Symbol::new("adt") && data.name == Symbol::new("typeref"))
        || (data.dialect == Symbol::new("closure") && data.name == Symbol::new("closure"))
        || (data.dialect == Symbol::new("tribute_rt")
            && (data.name == Symbol::new("anyref") || data.name == Symbol::new("intref")))
}

fn is_managed_value(ctx: &IrContext, value: ValueRef, managed_layouts: &HashSet<TypeRef>) -> bool {
    is_typed_managed_reference(ctx, ctx.value_ty(value), managed_layouts)
}

fn is_anyref_type(ctx: &IrContext, ty: TypeRef) -> bool {
    let data = ctx.types.get(ty);
    data.dialect == Symbol::new("tribute_rt") && data.name == Symbol::new("anyref")
}

pub fn build_native_ownership_plan(
    ctx: &IrContext,
    module: Module,
) -> Result<NativeOwnershipPlan, OwnershipPlanError> {
    build_native_ownership_plan_with_options(ctx, module, NativeOwnershipPlanOptions::production())
}

pub fn build_native_ownership_plan_with_options(
    ctx: &IrContext,
    module: Module,
    options: NativeOwnershipPlanOptions,
) -> Result<NativeOwnershipPlan, OwnershipPlanError> {
    let _module_block = module
        .first_block(ctx)
        .ok_or_else(|| OwnershipPlanError::new("module has no body block"))?;
    let mut function_ops = Vec::new();
    walk_module(ctx, module, |op| {
        if func::Func::matches(ctx, op) {
            function_ops.push(op);
        }
    });
    let definitions = collect_function_definitions(ctx, &function_ops)?;
    let managed_layouts = collect_and_validate_managed_layouts(ctx, module)?;
    let closure_layout = collect_closure_layout(ctx, module, &managed_layouts)?;
    let rtti_types = build_rtti_plan(ctx, module, &managed_layouts)?;
    let entry_contracts = compute_entry_contracts(
        ctx,
        module,
        &definitions,
        &managed_layouts,
        options.elide_proven_borrowed_parameters,
    )?;

    let mut functions = Vec::new();
    for &op in &function_ops {
        let Ok(function) = func::Func::from_op(ctx, op) else {
            continue;
        };
        let symbol = function.sym_name(ctx);
        let Some(body) = function.body_if_present(ctx) else {
            validate_bodyless_signature(ctx, op, &managed_layouts)?;
            continue;
        };
        if ctx.region(body).blocks.is_empty() {
            validate_bodyless_signature(ctx, op, &managed_layouts)?;
            continue;
        }
        let cfg = ValidatedFlatCfg::build(ctx, body)?;
        validate_function_contract(ctx, op, &cfg, &managed_layouts)?;
        let entries = entry_contracts
            .get(&symbol)
            .cloned()
            .ok_or_else(|| OwnershipPlanError::new("defined function has no entry contract"))?;
        let actions = plan_function_actions(
            ctx,
            &cfg,
            &entries,
            &entry_contracts,
            &definitions,
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
) -> Result<HashMap<Symbol, OpRef>, OwnershipPlanError> {
    let mut definitions = HashMap::new();
    for &op in module_ops {
        let Ok(function) = func::Func::from_op(ctx, op) else {
            continue;
        };
        let symbol = function.sym_name(ctx);
        if definitions.insert(symbol, op).is_some() {
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
    let mut layouts = HashSet::new();
    let mut nominal_layouts: HashMap<Symbol, Vec<TypeRef>> = HashMap::new();
    let mut typerefs = HashSet::new();
    for &(_, ty) in ctx.type_aliases() {
        collect_type_contract(ctx, ty, &mut typerefs, &mut nominal_layouts);
    }
    walk_module(ctx, module, |op| {
        for &ty in ctx.op_result_types(op) {
            collect_type_contract(ctx, ty, &mut typerefs, &mut nominal_layouts);
        }
        for &operand in ctx.op_operands(op) {
            collect_type_contract(
                ctx,
                ctx.value_ty(operand),
                &mut typerefs,
                &mut nominal_layouts,
            );
        }
        for attribute in ctx.op(op).attributes.values() {
            collect_attribute_type_contract(ctx, attribute, &mut typerefs, &mut nominal_layouts);
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

    for typeref in typerefs {
        let Some(name) = ctx.types.get(typeref).attrs.get_symbol("name") else {
            return Err(OwnershipPlanError::new(
                "adt.typeref lacks nominal identity",
            ));
        };
        if !ctx.types.get(typeref).params.is_empty() {
            return Err(OwnershipPlanError::new(format!(
                "adt.typeref @{name} has unexpected parameters"
            )));
        }
        if nominal_layouts
            .get(&name)
            .is_none_or(|layouts| layouts.len() != 1)
        {
            return Err(OwnershipPlanError::new(format!(
                "adt.typeref @{name} has no unique native layout"
            )));
        }
    }
    for nominal in nominal_layouts.values() {
        for &layout in nominal {
            layouts.insert(layout);
        }
    }
    Ok(layouts)
}

fn collect_type_contract(
    ctx: &IrContext,
    ty: TypeRef,
    typerefs: &mut HashSet<TypeRef>,
    nominal_layouts: &mut HashMap<Symbol, Vec<TypeRef>>,
) {
    let data = ctx.types.get(ty);
    if data.dialect == Symbol::new("adt") && data.name == Symbol::new("typeref") {
        typerefs.insert(ty);
    } else if data.dialect == Symbol::new("adt")
        && (data.name == Symbol::new("struct") || data.name == Symbol::new("enum"))
        && let Some(name) = data.attrs.get_symbol("name")
    {
        let layouts = nominal_layouts.entry(name).or_default();
        if !layouts.contains(&ty) {
            layouts.push(ty);
        }
    }
    for &parameter in &data.params {
        collect_type_contract(ctx, parameter, typerefs, nominal_layouts);
    }
    for attribute in data.attrs.values() {
        collect_attribute_type_contract(ctx, attribute, typerefs, nominal_layouts);
    }
}

fn collect_attribute_type_contract(
    ctx: &IrContext,
    attribute: &trunk_ir::Attribute,
    typerefs: &mut HashSet<TypeRef>,
    nominal_layouts: &mut HashMap<Symbol, Vec<TypeRef>>,
) {
    match attribute {
        trunk_ir::Attribute::Type(ty) => {
            collect_type_contract(ctx, *ty, typerefs, nominal_layouts);
        }
        trunk_ir::Attribute::List(values) => {
            for value in values {
                collect_attribute_type_contract(ctx, value, typerefs, nominal_layouts);
            }
        }
        _ => {}
    }
}

fn nominal_types_compatible(ctx: &IrContext, left: TypeRef, right: TypeRef) -> bool {
    let identity = |ty| {
        let data = ctx.types.get(ty);
        (data.dialect == Symbol::new("adt")).then(|| data.attrs.get_symbol("name"))?
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
        || closure_layout_compatible(ctx, actual, expected, managed_layouts)
        || closure_layout_compatible(ctx, expected, actual, managed_layouts)
        || (is_typed_managed_reference(ctx, actual, managed_layouts)
            && is_typed_managed_reference(ctx, expected, managed_layouts)
            && (is_anyref_type(ctx, actual)
                || is_anyref_type(ctx, expected)
                || nominal_types_compatible(ctx, actual, expected)))
}

fn closure_layout_compatible(
    ctx: &IrContext,
    actual: TypeRef,
    expected: TypeRef,
    managed_layouts: &HashSet<TypeRef>,
) -> bool {
    let actual_data = ctx.types.get(actual);
    if actual_data.dialect != Symbol::new("closure")
        || actual_data.name != Symbol::new("closure")
        || !managed_layouts.contains(&expected)
    {
        return false;
    }
    let Some(fields) = get_struct_fields(ctx, expected) else {
        return false;
    };
    matches!(fields.as_slice(), [(_, code), (_, env)] if {
        let code = ctx.types.get(*code);
        code.dialect == Symbol::new("core")
            && code.name == Symbol::new("i32")
            && is_anyref_type(ctx, *env)
    })
}

fn build_rtti_plan(
    ctx: &IrContext,
    module: Module,
    managed_layouts: &HashSet<TypeRef>,
) -> Result<Vec<RttiTypePlan>, OwnershipPlanError> {
    let mut order = Vec::new();
    let mut seen = HashSet::new();
    walk_module(ctx, module, |op| {
        let layout = adt::StructNew::from_op(ctx, op)
            .ok()
            .map(|new| new.r#type(ctx))
            .or_else(|| {
                adt::VariantNew::from_op(ctx, op)
                    .ok()
                    .map(|new| new.r#type(ctx))
            });
        if let Some(layout) = layout
            && managed_layouts.contains(&layout)
            && seen.insert(layout)
        {
            order.push(layout);
        }
    });

    order
        .into_iter()
        .map(|ty| {
            let fields = build_managed_field_bitmap(ctx, ty, managed_layouts)?;
            Ok(RttiTypePlan { ty, fields })
        })
        .collect()
}

fn build_managed_field_bitmap(
    ctx: &IrContext,
    ty: TypeRef,
    managed_layouts: &HashSet<TypeRef>,
) -> Result<ManagedFieldBitmap, OwnershipPlanError> {
    if let Some(fields) = get_struct_fields(ctx, ty) {
        Ok(ManagedFieldBitmap::Struct(
            fields
                .iter()
                .map(|(_, ty)| is_typed_managed_reference(ctx, *ty, managed_layouts))
                .collect(),
        ))
    } else if let Some(variants) = get_enum_variants(ctx, ty) {
        Ok(ManagedFieldBitmap::Enum(
            variants
                .iter()
                .map(|(_, fields)| {
                    fields
                        .iter()
                        .map(|ty| is_typed_managed_reference(ctx, *ty, managed_layouts))
                        .collect()
                })
                .collect(),
        ))
    } else {
        Err(OwnershipPlanError::new("RTTI type has no aggregate layout"))
    }
}

fn compute_entry_contracts(
    ctx: &IrContext,
    module: Module,
    definitions: &HashMap<Symbol, OpRef>,
    managed_layouts: &HashSet<TypeRef>,
    elide_proven_borrowed_parameters: bool,
) -> Result<HashMap<Symbol, Vec<EntryOwnership>>, OwnershipPlanError> {
    let recursive = recursive_functions(&build_call_graph(ctx, module));
    let mut summaries = HashMap::new();
    for (&symbol, &op) in definitions {
        let Some(body) = ctx.op(op).regions.first().copied() else {
            continue;
        };
        let Some(&entry) = ctx.region(body).blocks.first() else {
            continue;
        };
        let ineligible = recursive.contains(&symbol) || ctx.op(op).attributes.contains_key("abi");
        summaries.insert(
            symbol,
            ctx.block_args(entry)
                .iter()
                .map(|&parameter| {
                    if !is_managed_value(ctx, parameter, managed_layouts) {
                        EntryOwnership::Plain
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
        for (&symbol, &op) in definitions {
            let Some(body) = ctx.op(op).regions.first().copied() else {
                continue;
            };
            let Some(&entry) = ctx.region(body).blocks.first() else {
                continue;
            };
            for (index, &parameter) in ctx.block_args(entry).iter().enumerate() {
                if summaries[&symbol][index] == EntryOwnership::Borrowed
                    && !value_is_borrowed(ctx, body, parameter, &summaries, &mut HashSet::new())
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

    for (&symbol, &op) in definitions {
        if is_defined_physical_cps_function(ctx, op)
            && let Some(entries) = summaries.get_mut(&symbol)
        {
            for entry in entries {
                if *entry != EntryOwnership::Plain {
                    *entry = EntryOwnership::Consumed;
                }
            }
        }
    }
    Ok(summaries)
}

fn is_defined_physical_cps_function(ctx: &IrContext, op: OpRef) -> bool {
    if get_calling_convention(ctx, op) != Some(CallingConvention::Cps)
        || ctx.op(op).attributes.contains_key("abi")
    {
        return false;
    }
    let Some(signature) = ctx.op(op).attributes.get_type("type") else {
        return false;
    };
    core::Func::from_type_ref(ctx, signature).is_some_and(|callable| {
        let result = ctx.types.get(callable.r#return(ctx));
        result.dialect == Symbol::new("core")
            && (result.name == Symbol::new("nil") || result.name == Symbol::new("never"))
    })
}

fn value_is_borrowed(
    ctx: &IrContext,
    body: RegionRef,
    value: ValueRef,
    summaries: &HashMap<Symbol, Vec<EntryOwnership>>,
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
                .get(&call.callee(ctx))
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
) -> Result<(), OwnershipPlanError> {
    let signature = ctx
        .op(op)
        .attributes
        .get_type("type")
        .and_then(|ty| core::Func::from_type_ref(ctx, ty))
        .ok_or_else(|| OwnershipPlanError::new("bodyless function lacks exact signature"))?;
    if signature
        .params(ctx)
        .iter()
        .chain(std::iter::once(&signature.r#return(ctx)))
        .any(|&ty| is_typed_managed_reference(ctx, ty, managed_layouts))
    {
        return Err(OwnershipPlanError::new(
            "bodyless native declaration exposes a managed reference",
        ));
    }
    Ok(())
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
        .and_then(|ty| core::Func::from_type_ref(ctx, ty))
        .ok_or_else(|| OwnershipPlanError::new("defined function lacks exact signature"))?;
    let entry_args = ctx.block_args(cfg.entry());
    if entry_args.len() != signature.params(ctx).len()
        || entry_args
            .iter()
            .zip(signature.params(ctx))
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
                signature.r#return(ctx),
                managed_layouts,
                "function return",
            )?;
        }
    }
    Ok(())
}

fn validate_plan(ctx: &IrContext, plan: &NativeOwnershipPlan) -> Result<(), OwnershipPlanError> {
    let mut function_symbols = HashSet::new();
    for function in &plan.functions {
        if !function_symbols.insert(function.symbol) {
            return Err(OwnershipPlanError::new(
                "plan has duplicate function identity",
            ));
        }
        let Ok(function_op) = func::Func::from_op(ctx, function.operation) else {
            return Err(OwnershipPlanError::new(
                "planned function identity is stale",
            ));
        };
        let body = function_op
            .body_if_present(ctx)
            .ok_or_else(|| OwnershipPlanError::new("planned function body is stale"))?;
        let entry = ctx
            .region(body)
            .blocks
            .first()
            .copied()
            .ok_or_else(|| OwnershipPlanError::new("planned function entry is stale"))?;
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
        let mut action_keys = HashSet::new();
        let mut action_targets = HashMap::new();
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
    let mut rtti_types = HashSet::new();
    for entry in &plan.rtti_types {
        if !rtti_types.insert(entry.ty) || !plan.managed_layouts.contains(&entry.ty) {
            return Err(OwnershipPlanError::new(
                "RTTI plan has duplicate or stale type",
            ));
        }
        if build_managed_field_bitmap(ctx, entry.ty, &plan.managed_layouts)? != entry.fields {
            return Err(OwnershipPlanError::new(
                "RTTI managed-field bitmap is stale",
            ));
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
    let result_ty = ctx.types.get(ctx.value_ty(*result));
    *source == action.value
        && ctx.value_ty(*source) == closure_layout
        && is_managed_value(ctx, *source, managed_layouts)
        && result_ty.dialect == Symbol::new("core")
        && result_ty.name == Symbol::new("ptr")
}

fn validate_into_raw_transfer_groups(
    ctx: &IrContext,
    function: &FunctionOwnershipPlan,
    body: RegionRef,
    plan: &NativeOwnershipPlan,
) -> Result<(), OwnershipPlanError> {
    let mut sources = HashSet::new();
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
            .filter(|(_, action)| action.kind == ActionKind::CopyAcquire && action.value == *source)
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
#[path = "ownership_plan/tests.rs"]
mod tests;
