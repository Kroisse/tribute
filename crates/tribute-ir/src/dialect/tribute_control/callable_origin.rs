//! Which values may be called indirectly: callables whose source signature
//! comes from a declaration, decided by the rule of the operation that makes
//! each value.

use itertools::Itertools;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use trunk_ir::op_interface::CallLike;
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::refs::{OpRef, RegionRef, TypeRef, ValueDef, ValueRef};
use trunk_ir::symbol_table::{SymbolTable, qualified_name};
use trunk_ir::types::StringRef;
use trunk_ir::{IrContext, SymbolPath};

use super::{
    COMPILER_INTRINSIC_ATTR, Call, CallIndirect, CallingConvention, CompilerIntrinsicDeclaration,
    Func, FuncRef, FuncSig, Handle, Handler, Instantiate, Lambda, OperationDeclaration,
    OperationDeclarationUse, Perform, Resume, ValidationError, Yield, compiler_intrinsic_map,
    contains_adt_typeref, func_sig_parts, is_call_indirect, is_func, is_tail_call_indirect,
    nominal_identity, parent_region, push_op_error, push_type_error, walk_region_ops,
};
use crate::dialect::{adt, list, tribute_cps};

fn adt_projection_has_exact_callable_type(
    ctx: &IrContext,
    producer: OpRef,
    result_type: TypeRef,
    nominal_layouts: &HashMap<StringRef, TypeRef>,
) -> bool {
    if ctx.op_result_types(producer) != [result_type] {
        return false;
    }
    if let Ok(get) = adt::StructGet::from_op(ctx, producer) {
        let layout = get.r#type(ctx);
        return projection_source_matches_layout(ctx, get.r#ref(ctx), layout, nominal_layouts)
            && struct_field_type(ctx, layout, get.field(ctx)) == Some(result_type);
    }
    if let Ok(get) = adt::VariantGet::from_op(ctx, producer) {
        let layout = get.r#type(ctx);
        return projection_source_matches_layout(ctx, get.r#ref(ctx), layout, nominal_layouts)
            && variant_field_type(ctx, layout, get.tag_ref(ctx), get.field(ctx))
                == Some(result_type);
    }
    false
}

fn projection_source_matches_layout(
    ctx: &IrContext,
    source: ValueRef,
    layout: TypeRef,
    nominal_layouts: &HashMap<StringRef, TypeRef>,
) -> bool {
    let layout_identity = nominal_identity(ctx, layout);
    layout_identity.is_some_and(|identity| nominal_layouts.get(&identity) == Some(&layout))
        && nominal_identity(ctx, ctx.value_ty(source)) == layout_identity
}

fn struct_field_type(ctx: &IrContext, layout: TypeRef, field: u32) -> Option<TypeRef> {
    adt::Struct::from_type_ref(ctx, layout)?.field_type(ctx, usize::try_from(field).ok()?)
}

fn variant_field_type(
    ctx: &IrContext,
    layout: TypeRef,
    tag: StringRef,
    field: u32,
) -> Option<TypeRef> {
    adt::Enum::from_type_ref(ctx, layout)?
        .variant_fields(ctx, tag)?
        .get(usize::try_from(field).ok()?)
        .copied()
}

/// What an indirect callee's provenance is checked against, and the origin
/// of every value a rule has asked about.
pub struct CallableQuery<'a> {
    functions: &'a SymbolTable,
    registered: &'a HashMap<SymbolPath, &'a CompilerIntrinsicDeclaration>,
    declarations: &'a HashMap<(&'a SymbolPath, StringRef), &'a OperationDeclaration>,
    nominal_layouts: &'a HashMap<StringRef, TypeRef>,
    /// Whether each value reached so far is still taken to be an origin.
    origins: HashMap<ValueRef, bool>,
    /// The values whose rules asked about each value.
    dependents: HashMap<ValueRef, Vec<ValueRef>>,
    /// Values whose rule is to be evaluated against the current table.
    pending: Vec<ValueRef>,
    /// The value whose rule is being evaluated.
    evaluating: Option<ValueRef>,
    #[cfg(test)]
    pub(super) evaluations: usize,
}

impl<'a> CallableQuery<'a> {
    pub(super) fn new(
        functions: &'a SymbolTable,
        registered: &'a HashMap<SymbolPath, &'a CompilerIntrinsicDeclaration>,
        declarations: &'a HashMap<(&'a SymbolPath, StringRef), &'a OperationDeclaration>,
        nominal_layouts: &'a HashMap<StringRef, TypeRef>,
    ) -> Self {
        Self {
            functions,
            registered,
            declarations,
            nominal_layouts,
            origins: HashMap::default(),
            dependents: HashMap::default(),
            pending: Vec::new(),
            evaluating: None,
            #[cfg(test)]
            evaluations: 0,
        }
    }
}

impl CallableQuery<'_> {
    /// Whether `value` is taken to be a callable whose source signature comes
    /// from a declaration. A rule asks this about the values it depends on.
    ///
    /// A value not yet evaluated is taken to be one until its rule is, so a
    /// rule may read an answer that [`Self::resolve`] later withdraws; the
    /// rule is then evaluated again.
    pub(crate) fn accepts(&mut self, _ctx: &IrContext, value: ValueRef) -> bool {
        if let Some(dependent) = self.evaluating {
            let dependents = self.dependents.entry(value).or_default();
            if !dependents.contains(&dependent) {
                dependents.push(dependent);
            }
        }
        *self.origins.entry(value).or_insert_with(|| {
            self.pending.push(value);
            true
        })
    }

    /// Settle the origin of every value asked about, evaluating each rule
    /// until no answer changes.
    ///
    /// Every rule combines the values it asks about by conjunction, so the
    /// origins are the greatest fixed point: a value is an origin when its
    /// rule holds and every value it reaches is one. A cycle, such as one
    /// from a handle's handlers back to its result, adds no source of its
    /// own. A rule is evaluated once, and again only when a value it depends
    /// on is withdrawn, which happens at most once per value.
    pub(super) fn resolve(&mut self, ctx: &IrContext) {
        while let Some(value) = self.pending.pop() {
            if self.origins.get(&value) == Some(&false) {
                continue;
            }
            self.evaluating = Some(value);
            let holds = self.evaluate(ctx, value);
            self.evaluating = None;
            if !holds {
                self.origins.insert(value, false);
                let dependents = self.dependents.remove(&value).unwrap_or_default();
                self.pending.extend(dependents);
            }
        }
    }

    /// The [`CallableResultOrigin`] rule of `value`'s producer, or the
    /// [`CallableArgumentOrigin`] rule of the operation that owns its block,
    /// against the current answers.
    fn evaluate(&mut self, ctx: &IrContext, value: ValueRef) -> bool {
        #[cfg(test)]
        {
            self.evaluations += 1;
        }
        FuncSig::from_type_ref(ctx, ctx.value_ty(value)).is_some()
            && match ctx.value_def(value) {
                ValueDef::OpResult(producer, _) => DynCallableResultOrigin::get(ctx, producer)
                    .is_some_and(|origin| origin.result_is_callable_origin(ctx, value, self)),
                ValueDef::BlockArg(block, index) => {
                    let region = ctx.block(block).parent_region;
                    let owner = region.and_then(|region| ctx.region(region).parent_op);
                    match (
                        region,
                        owner.and_then(|owner| DynCallableArgumentOrigin::get(ctx, owner)),
                    ) {
                        (Some(region), Some(origin)) => {
                            origin.argument_is_callable_origin(ctx, region, index, value, self)
                        }
                        _ => false,
                    }
                }
            }
    }

    /// Whether `value`, settled by [`Self::resolve`], is an origin.
    pub(super) fn is_origin(&self, value: ValueRef) -> bool {
        self.origins.get(&value) == Some(&true)
    }

    /// The declaration `declared` names, when it claims that declaration's kind.
    fn claimed_declaration(
        &self,
        ctx: &IrContext,
        declared: &impl OperationDeclarationUse,
    ) -> Option<&OperationDeclaration> {
        let claim = declared.declaration_use(ctx)?;
        self.declarations
            .get(&(claim.ability, claim.op_name))
            .copied()
            .filter(|declaration| claim.kind == Some(declaration.kind))
    }

    /// Whether `symbol`, referenced from `op`, names a defined function or a
    /// registered intrinsic.
    pub(crate) fn names_verified_function(
        &self,
        ctx: &IrContext,
        op: OpRef,
        symbol: &SymbolPath,
    ) -> bool {
        self.functions
            .resolve(ctx, op, symbol)
            .filter(|&function| is_func(ctx, function))
            .is_some_and(|function| verified_callable_declaration(ctx, function, self.registered))
    }
}

/// An operation whose results can be callables whose source signature comes
/// from a declaration, and the rule that proves one.
///
/// A rule asks about other values only through [`CallableQuery`] and
/// combines the answers by conjunction. Whole-IR validation also queries
/// operations the local checks rejected, so a rule reads its operation
/// fallibly.
#[trunk_ir::op_interface]
pub trait CallableResultOrigin: DialectOp {
    /// Whether `result`, a result of this operation, is such a callable.
    fn result_is_callable_origin<'q>(
        &self,
        ctx: &IrContext,
        result: ValueRef,
        query: &mut CallableQuery<'q>,
    ) -> bool;
}

/// An operation whose region arguments can be such callables, under the
/// same contract as [`CallableResultOrigin`].
#[trunk_ir::op_interface]
pub trait CallableArgumentOrigin: DialectOp {
    /// Whether `argument`, argument `index` of a block in `region` of this
    /// operation, is such a callable.
    fn argument_is_callable_origin<'q>(
        &self,
        ctx: &IrContext,
        region: RegionRef,
        index: u32,
        argument: ValueRef,
        query: &mut CallableQuery<'q>,
    ) -> bool;
}

/// A parameter of the callable whose only region is `region`, whose
/// signature is `signature`.
pub(crate) fn is_signature_parameter(
    ctx: &IrContext,
    op: OpRef,
    signature: Option<TypeRef>,
    region: RegionRef,
    index: u32,
    argument: ValueRef,
) -> bool {
    ctx.op_regions(op).eq([region])
        && signature
            .and_then(|ty| func_sig_parts(ctx, ty))
            .is_some_and(|(_, parameters)| {
                parameters.get(index as usize) == Some(&ctx.value_ty(argument))
            })
}

impl CallableArgumentOrigin for Func {
    fn argument_is_callable_origin<'q>(
        &self,
        ctx: &IrContext,
        region: RegionRef,
        index: u32,
        argument: ValueRef,
        _query: &mut CallableQuery<'q>,
    ) -> bool {
        let signature = ctx.op_attrs(self.op_ref()).get_type("type");
        is_signature_parameter(ctx, self.op_ref(), signature, region, index, argument)
    }
}

impl CallableResultOrigin for Lambda {
    fn result_is_callable_origin<'q>(
        &self,
        _ctx: &IrContext,
        _result: ValueRef,
        _query: &mut CallableQuery<'q>,
    ) -> bool {
        true
    }
}

impl CallableArgumentOrigin for Lambda {
    fn argument_is_callable_origin<'q>(
        &self,
        ctx: &IrContext,
        region: RegionRef,
        index: u32,
        argument: ValueRef,
        _query: &mut CallableQuery<'q>,
    ) -> bool {
        let signature = ctx.op_result_types(self.op_ref()).first().copied();
        is_signature_parameter(ctx, self.op_ref(), signature, region, index, argument)
    }
}

impl CallableResultOrigin for Instantiate {
    fn result_is_callable_origin<'q>(
        &self,
        ctx: &IrContext,
        _result: ValueRef,
        query: &mut CallableQuery<'q>,
    ) -> bool {
        ctx.op_operands(self.op_ref())
            .first()
            .is_some_and(|&lambda| query.accepts(ctx, lambda))
    }
}

impl CallableResultOrigin for FuncRef {
    fn result_is_callable_origin<'q>(
        &self,
        ctx: &IrContext,
        _result: ValueRef,
        query: &mut CallableQuery<'q>,
    ) -> bool {
        ctx.op_attrs(self.op_ref())
            .get_symbol_ref("func_ref")
            .is_some_and(|symbol| query.names_verified_function(ctx, self.op_ref(), symbol))
    }
}

impl CallableResultOrigin for Call {
    // The callee is read through `CallLike` once the call's structure
    // satisfies its schema.
    fn result_is_callable_origin<'q>(
        &self,
        ctx: &IrContext,
        _result: ValueRef,
        query: &mut CallableQuery<'q>,
    ) -> bool {
        Self::DEF
            .schema
            .verify_structure(ctx, self.op_ref())
            .is_empty()
            && query.names_verified_function(ctx, self.op_ref(), self.direct_callee(ctx))
    }
}

impl CallableResultOrigin for CallIndirect {
    fn result_is_callable_origin<'q>(
        &self,
        ctx: &IrContext,
        _result: ValueRef,
        query: &mut CallableQuery<'q>,
    ) -> bool {
        ctx.op_operands(self.op_ref())
            .first()
            .is_some_and(|&callee| query.accepts(ctx, callee))
    }
}

impl CallableResultOrigin for Perform {
    fn result_is_callable_origin<'q>(
        &self,
        ctx: &IrContext,
        result: ValueRef,
        query: &mut CallableQuery<'q>,
    ) -> bool {
        let result_type = ctx.value_ty(result);
        ctx.op_result_types(self.op_ref()) == [result_type]
            && self.declared_result_type(ctx) == Some(result_type)
            && query
                .claimed_declaration(ctx, self)
                .is_some_and(|declaration| declaration.result_type == result_type)
    }
}

impl CallableArgumentOrigin for Handler {
    fn argument_is_callable_origin<'q>(
        &self,
        ctx: &IrContext,
        _region: RegionRef,
        _index: u32,
        argument: ValueRef,
        query: &mut CallableQuery<'q>,
    ) -> bool {
        self.declared_arguments(ctx)
            .and_then(|arguments| arguments.iter().position(|&value| value == argument))
            .zip(query.claimed_declaration(ctx, self))
            .is_some_and(|(position, declaration)| {
                declaration.parameter_types.get(position) == Some(&ctx.value_ty(argument))
            })
    }
}

impl CallableResultOrigin for Handle {
    fn result_is_callable_origin<'q>(
        &self,
        ctx: &IrContext,
        result: ValueRef,
        query: &mut CallableQuery<'q>,
    ) -> bool {
        handle_result_sources(ctx, self.op_ref()).is_some_and(|sources| {
            sources.into_iter().all(|source| {
                ctx.value_ty(source) == ctx.value_ty(result) && query.accepts(ctx, source)
            })
        })
    }
}

impl CallableArgumentOrigin for Handle {
    fn argument_is_callable_origin<'q>(
        &self,
        ctx: &IrContext,
        region: RegionRef,
        index: u32,
        argument: ValueRef,
        query: &mut CallableQuery<'q>,
    ) -> bool {
        ctx.op_regions(self.op_ref())
            .collect_tuple()
            .is_some_and(|(body, completion, _)| {
                completion == region
                    && index == 0
                    && region_yield_value(ctx, body).is_some_and(|value| {
                        ctx.value_ty(value) == ctx.value_ty(argument) && query.accepts(ctx, value)
                    })
            })
    }
}

impl CallableResultOrigin for Resume {
    fn result_is_callable_origin<'q>(
        &self,
        ctx: &IrContext,
        result: ValueRef,
        query: &mut CallableQuery<'q>,
    ) -> bool {
        resumed_handle_result(ctx, self.op_ref()).is_some_and(|handled| {
            ctx.value_ty(handled) == ctx.value_ty(result) && query.accepts(ctx, handled)
        })
    }
}

/// A list's head is a callable origin when the declared element type is
/// exactly the result's type.
impl CallableResultOrigin for list::Head {
    fn result_is_callable_origin<'q>(
        &self,
        ctx: &IrContext,
        result: ValueRef,
        _query: &mut CallableQuery<'q>,
    ) -> bool {
        let result_type = ctx.value_ty(result);
        self.element_type(ctx) == result_type && ctx.op_result_types(self.op_ref()) == [result_type]
    }
}

/// An ADT field projection is a callable origin when the canonical layout of
/// its source declares the field with exactly the result's type.
impl CallableResultOrigin for adt::StructGet {
    fn result_is_callable_origin<'q>(
        &self,
        ctx: &IrContext,
        result: ValueRef,
        query: &mut CallableQuery<'q>,
    ) -> bool {
        adt_projection_has_exact_callable_type(
            ctx,
            self.op_ref(),
            ctx.value_ty(result),
            query.nominal_layouts,
        )
    }
}

/// A variant field projection, under the rule of [`adt::StructGet`].
impl CallableResultOrigin for adt::VariantGet {
    fn result_is_callable_origin<'q>(
        &self,
        ctx: &IrContext,
        result: ValueRef,
        query: &mut CallableQuery<'q>,
    ) -> bool {
        adt_projection_has_exact_callable_type(
            ctx,
            self.op_ref(),
            ctx.value_ty(result),
            query.nominal_layouts,
        )
    }
}

trunk_ir::register_interfaces!(
    list::Head: CallableResultOrigin;
    adt::StructGet: CallableResultOrigin;
    adt::VariantGet: CallableResultOrigin
);

/// The value a single-block region yields to its parent.
fn region_yield_value(ctx: &IrContext, region: RegionRef) -> Option<ValueRef> {
    let block = ctx.region(region).blocks.iter().exactly_one().ok()?;
    let terminator = ctx.block(*block).ops.last().copied()?;
    let value = ctx.op_operands(terminator).iter().exactly_one().ok()?;
    Yield::matches(ctx, terminator).then_some(*value)
}

/// The values that become a `handle`'s result: the completion region's
/// yield and the yield of each general (`op`) handler. A `fn` handler yields
/// to its `perform` instead.
fn handle_result_sources(ctx: &IrContext, handle: OpRef) -> Option<Vec<ValueRef>> {
    let (_, completion, handlers) = ctx.op_regions(handle).collect_tuple()?;
    let mut sources = vec![region_yield_value(ctx, completion)?];
    for &block in &ctx.region(handlers).blocks {
        for &handler in &ctx.block(block).ops {
            if Handler::matches(ctx, handler) && ctx.op_attrs(handler).get_str("kind") == Some("op")
            {
                sources.push(region_yield_value(ctx, ctx.op_region(handler, 0)?)?);
            }
        }
    }
    Some(sources)
}

/// The result of the `handle` whose handler holds the resume token that
/// `resume` consumes.
fn resumed_handle_result(ctx: &IrContext, resume: OpRef) -> Option<ValueRef> {
    let token = *ctx.op_operands(resume).first()?;
    let ValueDef::BlockArg(block, _) = ctx.value_def(token) else {
        return None;
    };
    let handler = ctx.region(ctx.block(block).parent_region?).parent_op?;
    let handle = ctx.region(parent_region(ctx, handler)?).parent_op?;
    if !Handler::matches(ctx, handler) || !Handle::matches(ctx, handle) {
        return None;
    }
    ctx.op_results(handle).first().copied()
}

fn verified_callable_declaration(
    ctx: &IrContext,
    function: OpRef,
    registered: &HashMap<SymbolPath, &CompilerIntrinsicDeclaration>,
) -> bool {
    if ctx.op_has_regions(function) {
        return true;
    }
    let (Some(symbol), Some(identity)) = (
        qualified_name(ctx, function),
        ctx.op_attrs(function).get_str(COMPILER_INTRINSIC_ATTR),
    ) else {
        return false;
    };
    registered
        .get(&symbol)
        .is_some_and(|declaration| declaration.identity == identity)
}

/// The callee of `op` when it is a source-logical indirect call.
fn indirect_callee(ctx: &IrContext, op: OpRef) -> Option<ValueRef> {
    (is_call_indirect(ctx, op) || is_tail_call_indirect(ctx, op))
        .then(|| ctx.op_operands(op).first().copied())
        .flatten()
}

/// Every indirect callee in `body`.
pub(super) fn indirect_callees(ctx: &IrContext, body: RegionRef) -> Vec<ValueRef> {
    let mut callees = Vec::new();
    walk_region_ops(ctx, body, &mut |op| {
        callees.extend(indirect_callee(ctx, op))
    });
    callees
}

pub(super) fn validate_callable_origins(
    ctx: &IrContext,
    body: RegionRef,
    functions: &SymbolTable,
    declarations: &[CompilerIntrinsicDeclaration],
    operation_declarations: &HashMap<(&SymbolPath, StringRef), &OperationDeclaration>,
    nominal_layouts: &HashMap<StringRef, TypeRef>,
    errors: &mut Vec<ValidationError>,
) {
    let registered = compiler_intrinsic_map(declarations, errors);
    let mut query = CallableQuery::new(
        functions,
        &registered,
        operation_declarations,
        nominal_layouts,
    );
    for callee in indirect_callees(ctx, body) {
        query.accepts(ctx, callee);
    }
    query.resolve(ctx);
    let mut seen = HashSet::default();
    walk_region_ops(ctx, body, &mut |op| {
        if is_func(ctx, op) {
            let (Some(symbol), Some(func_sig_type)) =
                (qualified_name(ctx, op), ctx.op_attrs(op).get_type("type"))
            else {
                return;
            };
            let intrinsic_identity = ctx.op_attrs(op).get_str(COMPILER_INTRINSIC_ATTR);
            let bodyless = !ctx.op_has_regions(op);
            if !bodyless && intrinsic_identity.is_some() {
                push_op_error(
                    ctx,
                    op,
                    errors,
                    "defined Tribute function cannot claim compiler intrinsic identity",
                );
            }
            if bodyless {
                let exact_intrinsic = match (registered.get(&symbol), intrinsic_identity) {
                    (Some(expected), Some(actual)) if expected.identity == actual => {
                        if tribute_cps::declared_convention(ctx, op)
                            .is_some_and(|convention| convention != CallingConvention::Direct)
                        {
                            push_op_error(
                                ctx,
                                op,
                                errors,
                                format!(
                                    "registered compiler intrinsic @{symbol} must use Direct calling convention"
                                ),
                            );
                        }
                        seen.insert(symbol);
                        true
                    }
                    (Some(_), _) => {
                        push_op_error(
                            ctx,
                            op,
                            errors,
                            "compiler intrinsic identity does not match the registered declaration",
                        );
                        false
                    }
                    (None, Some(_)) => {
                        push_op_error(
                            ctx,
                            op,
                            errors,
                            "unregistered declaration cannot claim compiler intrinsic identity",
                        );
                        false
                    }
                    (None, None) => false,
                };
                let trusted_c_ffi = ctx.op_attrs(op).get_str("abi") == Some("C");
                if !exact_intrinsic
                    && !trusted_c_ffi
                    && contains_adt_typeref(ctx, func_sig_type, &mut HashSet::default())
                {
                    push_op_error(
                        ctx,
                        op,
                        errors,
                        "unclassified bodyless external declaration cannot use managed adt.typeref",
                    );
                }
            }
        } else if let Some(callee) = indirect_callee(ctx, op)
            && !query.is_origin(callee)
        {
            push_op_error(
                ctx,
                op,
                errors,
                "indirect callee lacks verified source-logical callable provenance",
            );
        }
    });
    for symbol in registered.keys() {
        if !seen.contains(symbol) {
            push_type_error(
                errors,
                format!(
                    "registered compiler intrinsic @{symbol} has no exact bodyless declaration"
                ),
            );
        }
    }
}
