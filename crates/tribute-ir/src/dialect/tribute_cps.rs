//! Source-logical callables whose calling convention the CPS strategy has
//! assigned (`new-plans/ir.md`, "Convention과 class 특수화").
//!
//! Class specialization replaces the five `tribute_control` operations whose
//! meaning depends on a convention with the operation of the same name here,
//! which adds that convention, and adds the `tribute_cps.func_table` that a
//! function reference names. The other `tribute_control` operations stay
//! and run inside these callables. Rows, logical types, and the
//! `tribute_control.func_sig` callable type are unchanged, and no hidden
//! parameter exists yet. Whole-IR verification is
//! [`tribute_control::validate`], which
//! reads both dialects.

use trunk_ir::IrContext;
use trunk_ir::attr_kind::SymbolRef;
use trunk_ir::op_flag::{IsolatedFromAbove, Pure};
use trunk_ir::op_interface::CallableExit;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{RegionRef, ValueRef};

use crate::calling_convention::CallingConvention;
use crate::dialect::tribute_control::{
    self, CallableArgumentOrigin, CallableQuery, CallableResultOrigin, EvidencePlanSite,
    EvidenceStep, FuncSig, LocalValidation, Row, RowBinder, RowUnion, ValidationError,
};

/// The attribute that holds a `tribute_cps` operation's convention.
pub const CONVENTION_ATTR: &str = "convention";

/// The attribute that holds the conventions of a `tribute_cps.lambda`'s
/// bodies, in body order.
pub const CONVENTIONS_ATTR: &str = "conventions";

#[trunk_ir::dialect]
mod tribute_cps {
    /// A named source function instance with its convention.
    ///
    /// ```text
    /// tribute_cps.func @id(%x: T) -> R convention(direct) {
    ///   ...
    ///   tribute_control.return %value
    /// }
    /// ```
    ///
    /// - **Shape:** `tribute_control.func` with a required
    ///   `convention: String`, one of `"direct"`, `"evidence_direct"`, and
    ///   `"cps"`. The concise form always writes it.
    /// - **Meaning:** one class instance of a source function. Class
    ///   specialization computes the convention from the row with the
    ///   instance's classes (`new-plans/cps-effects.md`, "Row와 호출 규약").
    ///   The signature, row, and body keep the meaning they have on
    ///   `tribute_control.func`.
    /// - **Verification:** the local verifier checks the convention keyword.
    ///   Otherwise as `tribute_control.func`.
    /// - **Ownership and value flow:** as `tribute_control.func`.
    /// - **Symbol:** as `tribute_control.func`.
    /// - **Location:** the source function's span.
    #[verify]
    #[asm_format]
    #[flags(IsolatedFromAbove)]
    #[implements(
        CallableArgumentOrigin,
        LocalValidation,
        trunk_ir::op_interface::SymbolDefinition
    )]
    fn func<S: FuncSig>(
        sym_name: Attr<String>,
        r#type: Attr<TypeOf<S>>,
        convention: Attr<CallingConvention>,
        row_binders: Option<Attr<[RowBinder]>>,
        row_unions: Option<Attr<[RowUnion]>>,
        inferred_row: Option<Attr<bool>>,
    ) {
        #[region(body?)]
        {}
    }

    /// A source lambda with a body per convention entry its uses select.
    ///
    /// ```text
    /// %f = tribute_cps.lambda(%x: T) -> R captures [%capture0, ...] entries {
    ///   direct {
    ///     ...
    ///     tribute_control.return %value
    ///   }
    ///   cps {
    ///     ...
    ///   }
    /// }
    /// ```
    ///
    /// - **Shape:** `tribute_control.lambda` with a required
    ///   `conventions` list and one body region per element, in the same
    ///   order. Each body takes the source parameters. Each body is written
    ///   under its convention's keyword in `entries`.
    /// - **Meaning:** one lambda value with a body for each class
    ///   `new-plans/ir.md` ("Convention entry") gives it: each class its uses
    ///   select, or each class from its row's lower bound to its upper bound
    ///   when that row reads as an upper bound. Its body of a convention
    ///   reads each binder it quantifies with the class its uses of that
    ///   class pass, as an upper bound when they pass different ones, and
    ///   its own row's tail, when that tail reads as an upper bound, as an
    ///   upper bound narrowed to that convention. Calling the value with a
    ///   convention that has a body runs that body; legalization fills the
    ///   other convention entries.
    /// - **Verification:** the verifier checks that the conventions are
    ///   strictly increasing and that there is a body for each one.
    ///   Otherwise as `tribute_control.lambda`, for each body.
    /// - **Ownership and value flow:** as `tribute_control.lambda`; the
    ///   captures are shared by every body.
    /// - **Location:** the source lambda's span.
    #[verify]
    #[asm_format]
    #[flags(Pure)]
    #[implements(CallableResultOrigin, CallableArgumentOrigin, LocalValidation)]
    fn lambda(
        conventions: Attr<[CallingConvention]>,
        quantified_rows: Option<Attr<[u32]>>,
        captures: Variadic<_>,
    ) -> Value<impl FuncSig> {
        #[regions(bodies)]
        {}
    }

    /// A named function as a value, entered through the convention table
    /// of its definition and class arguments.
    ///
    /// ```text
    /// %f = tribute_cps.func_ref {func_ref = @"id$9e$8"} : !Callable
    /// ```
    ///
    /// - **Shape:** `tribute_control.func_ref` whose `func_ref` names a
    ///   `tribute_cps.func_table` instead of a function.
    /// - **Meaning:** the callable value whose convention entries are the
    ///   table's. References that share a table share its entries.
    /// - **Verification:** the local verifier checks the attributes.
    ///   Whole-IR verification checks that `func_ref` resolves to a
    ///   `tribute_cps.func_table`, and checks the source signature against
    ///   each of its targets as for `tribute_control.func_ref`.
    /// - **Ownership and value flow:** the result is an ordinary SSA value.
    /// - **Location:** the source reference span.
    #[verify]
    #[flags(Pure)]
    #[implements(CallableResultOrigin)]
    fn func_ref(func_ref: Attr<SymbolRef>, row_args: Option<Attr<[Row]>>) -> Value<impl FuncSig> {}

    /// The convention table of a named function for one set of class
    /// arguments, with a target instance per convention its row reads.
    ///
    /// ```text
    /// tribute_cps.func_table @"id$9e$8" {conventions = ["direct", "evidence_direct"], targets = [@"id$9D", @"id$9e"]}
    /// ```
    ///
    /// - **Shape:** no operands, results, or regions. `sym_name: String`,
    ///   and `conventions` and `targets` lists of the same length.
    /// - **Placement:** directly in a module, as a symbol definition. Class
    ///   specialization makes one for each definition and class arguments
    ///   that a `tribute_cps.func_ref` refers to, named as
    ///   `new-plans/cps-effects.md` ("Class 특수화 pass") gives.
    /// - **Meaning:** the convention entry of each element of `conventions`
    ///   runs the `tribute_cps.func` instance at the same position of
    ///   `targets`. Each instance is the one the class arguments select
    ///   once the argument of the class variable that is the definition
    ///   row's tail is narrowed to that convention, as a lambda body narrows
    ///   a tail. The conventions are the classes `new-plans/ir.md`
    ///   ("Convention entry", 함수 참조) gives the definition's row read with
    ///   the class arguments. They depend only on the definition and the
    ///   class arguments, never on the row a reference passes, so every
    ///   reference with those arguments can share the table. When an instance's convention is weaker, legalization
    ///   fills the entry with an adapter. When a classless tail makes the
    ///   instance `Cps` and its convention is weaker, the entry is the
    ///   `func_ref` adapter that calls the worker through a value delimiter
    ///   (`new-plans/cps-effects.md`, 값 delimiter). Legalization fills the
    ///   other entries by the same rule as a lambda's, once for the table.
    /// - **Verification:** the local verifier checks the attributes, that
    ///   the lists have the same length, and that the conventions are
    ///   strictly increasing. Whole-IR verification checks that each target
    ///   resolves to a `tribute_cps.func` of the same definition, and that
    ///   each convention is not weaker than its target's unless that target
    ///   is `Cps`.
    /// - **Ownership and value flow:** defines no value.
    /// - **Symbol:** a discardable `SymbolDefinition` that is not a
    ///   declaration. Its `targets` are its uses, so it keeps them alive
    ///   only while a live `tribute_cps.func_ref` uses it, and global DCE
    ///   removes it once none does (`new-plans/ir.md`, "Core Invariants").
    /// - **Location:** the span of the referenced definition.
    #[verify]
    #[implements(trunk_ir::op_interface::SymbolDefinition)]
    fn func_table(
        sym_name: Attr<String>,
        conventions: Attr<[CallingConvention]>,
        targets: Attr<[SymbolRef]>,
    ) {
    }

    /// An indirect call through one convention.
    ///
    /// ```text
    /// %result = tribute_cps.call_indirect %callee, %arg0, ... {convention = "direct"} : Result
    /// ```
    ///
    /// - **Shape:** `tribute_control.call_indirect` with a required
    ///   `convention: String`.
    /// - **Meaning:** calls the callee's convention entry of `convention`,
    ///   which class specialization computed from the callee type's row
    ///   with the enclosing instance's classes.
    /// - **Verification:** the local verifier checks the convention keyword.
    ///   Otherwise as `tribute_control.call_indirect`.
    /// - **Ownership and value flow:** as `tribute_control.call_indirect`.
    /// - **Location:** the source indirect-call span.
    #[verify]
    #[implements(CallableResultOrigin)]
    fn call_indirect<S: FuncSig>(
        convention: Attr<CallingConvention>,
        evidence_plan: Option<Attr<[EvidenceStep]>>,
        callee: Value<S>,
        args: Values<S::Inputs>,
    ) -> Value<S::Result> {
    }

    /// A proper tail call of a callable value through one convention.
    ///
    /// ```text
    /// tribute_cps.tail_call_indirect %callee, %arg0, ... {convention = "cps"}
    /// ```
    ///
    /// - **Shape:** `tribute_control.tail_call_indirect` with a required
    ///   `convention: String`.
    /// - **Placement:** as `tribute_control.tail_call`.
    /// - **Meaning:** a proper tail call of the callee's convention entry of
    ///   `convention`.
    /// - **Verification:** the local verifier checks the convention keyword,
    ///   and that `convention` is not stronger than the enclosing callable's.
    ///   Otherwise as `tribute_control.tail_call_indirect`.
    /// - **Ownership and value flow:** as
    ///   `tribute_control.tail_call_indirect`.
    /// - **Location:** the source span including the `become` keyword.
    #[verify]
    #[implements(CallableExit, LocalValidation)]
    fn tail_call_indirect<S: FuncSig>(
        convention: Attr<CallingConvention>,
        evidence_plan: Option<Attr<[EvidenceStep]>>,
        callee: Value<S>,
        args: Values<S::Inputs>,
    ) {
    }
}

/// Why a `tribute_cps` operation's convention is not one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, derive_more::Display, derive_more::Error)]
#[display("`convention` must be one of \"direct\", \"evidence_direct\", and \"cps\"")]
pub struct MalformedConvention;

/// The convention `op` declares, when it is a `tribute_cps` operation with a
/// valid one. A `tribute_cps.lambda` declares one per body instead; see
/// [`body_conventions`] and [`convention_at`].
pub fn declared_convention(
    ctx: &IrContext,
    op: trunk_ir::refs::OpRef,
) -> Option<CallingConvention> {
    if ctx.op(op).name.dialect(ctx) != DIALECT_NAME {
        return None;
    }
    CallingConvention::from_attribute(ctx, ctx.op_attrs(op).get(CONVENTION_ATTR)?)
}

/// Why a `tribute_cps.lambda`'s conventions do not name its bodies.
#[derive(Clone, Copy, Debug, PartialEq, Eq, derive_more::Display, derive_more::Error)]
#[display("`conventions` must be strictly increasing, with one body for each")]
pub struct MalformedBodies;

/// The conventions of a lambda's bodies, when it is a `tribute_cps.lambda`
/// whose list is well formed, in body order.
pub fn body_conventions(
    ctx: &IrContext,
    op: trunk_ir::refs::OpRef,
) -> Option<Vec<CallingConvention>> {
    if !Lambda::matches(ctx, op) {
        return None;
    }
    let trunk_ir::types::Attribute::List(conventions) = ctx.op_attrs(op).get(CONVENTIONS_ATTR)?
    else {
        return None;
    };
    let conventions = conventions
        .iter()
        .map(|convention| CallingConvention::from_attribute(ctx, convention))
        .collect::<Option<Vec<_>>>()?;
    (!conventions.is_empty()
        && conventions.is_sorted_by(|a, b| a < b)
        && conventions.len() == ctx.op_region_count(op))
    .then_some(conventions)
}

/// Why a `tribute_cps.func_table`'s conventions do not name its targets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, derive_more::Display, derive_more::Error)]
#[display("`conventions` must be strictly increasing, with one target for each")]
pub struct MalformedTable;

/// The conventions and targets of `op`, when it is a
/// `tribute_cps.func_table` whose lists are well formed, in convention order.
pub fn table_entries(
    ctx: &IrContext,
    op: trunk_ir::refs::OpRef,
) -> Option<Vec<(CallingConvention, trunk_ir::SymbolPath)>> {
    let table = FuncTable::from_op(ctx, op).ok()?;
    let trunk_ir::types::Attribute::List(conventions) = ctx.op_attrs(op).get(CONVENTIONS_ATTR)?
    else {
        return None;
    };
    let conventions = conventions
        .iter()
        .map(|convention| CallingConvention::from_attribute(ctx, convention))
        .collect::<Option<Vec<_>>>()?;
    let targets: Vec<_> = table.targets(ctx).cloned().collect();
    (!conventions.is_empty()
        && conventions.is_sorted_by(|a, b| a < b)
        && conventions.len() == targets.len())
    .then(|| conventions.into_iter().zip(targets).collect())
}

/// The `tribute_cps.func_table` that `op`, a `tribute_cps.func_ref`, names:
/// the table of that name directly in the nearest enclosing module.
pub fn func_ref_table(ctx: &IrContext, op: trunk_ir::refs::OpRef) -> Option<trunk_ir::refs::OpRef> {
    let name = FuncRef::from_op(ctx, op)
        .ok()?
        .func_ref(ctx)
        .as_simple()?
        .clone();
    let mut current = op;
    let module = loop {
        let block = ctx.op(current).parent_block?;
        current = ctx.region(ctx.block(block).parent_region?).parent_op?;
        if trunk_ir::dialect::core::Module::matches(ctx, current) {
            break trunk_ir::rewrite::Module::from(
                trunk_ir::dialect::core::Module::from_op(ctx, current).ok()?,
            );
        }
    };
    module.ops(ctx).iter().copied().find(|&table| {
        FuncTable::matches(ctx, table)
            && ctx.op_attrs(table).get_str("sym_name") == Some(name.as_str())
    })
}

/// The convention of the body of `owner`, a callable, that holds `op`: the
/// convention of a `tribute_cps.func`, or of the `tribute_cps.lambda` body
/// `op` is in.
pub fn convention_at(
    ctx: &IrContext,
    owner: trunk_ir::refs::OpRef,
    op: trunk_ir::refs::OpRef,
) -> Option<CallingConvention> {
    let Some(conventions) = body_conventions(ctx, owner) else {
        return declared_convention(ctx, owner);
    };
    let mut current = op;
    let body = loop {
        let region = ctx
            .op(current)
            .parent_block
            .and_then(|block| ctx.block(block).parent_region)?;
        let parent = ctx.region(region).parent_op?;
        if parent == owner {
            break region;
        }
        current = parent;
    };
    let index = ctx.op_regions(owner).position(|region| region == body)?;
    conventions.get(index).copied()
}

/// Replace `op`, a `tribute_control` func, lambda, func_ref, call_indirect,
/// or tail_call_indirect, with its operation here, which declares
/// `convention`; a lambda's one body has that convention, and a func_ref
/// declares none, since its table holds its conventions. The replacement
/// takes the operation's place, operands,
/// attributes, and regions, and the uses of its results.
///
/// # Panics
///
/// If `op` is none of those operations.
pub fn replace_control_op(
    ctx: &mut IrContext,
    op: trunk_ir::refs::OpRef,
    convention: CallingConvention,
) -> trunk_ir::refs::OpRef {
    let name = if tribute_control::Func::matches(ctx, op) {
        Func::NAME
    } else if tribute_control::Lambda::matches(ctx, op) {
        Lambda::NAME
    } else if tribute_control::FuncRef::matches(ctx, op) {
        FuncRef::NAME
    } else if tribute_control::CallIndirect::matches(ctx, op) {
        CallIndirect::NAME
    } else if tribute_control::TailCallIndirect::matches(ctx, op) {
        TailCallIndirect::NAME
    } else {
        panic!("only a convention-dependent tribute_control operation has a tribute_cps form")
    };
    let data = ctx.op(op);
    let mut builder = trunk_ir::context::OperationDataBuilder::new(data.location, name)
        .operands(ctx.op_operands(op).iter().copied())
        .results(ctx.op_result_types(op).iter().copied());
    for (key, value) in data.attributes.iter() {
        builder = builder.attr(key, value.clone());
    }
    let regions: Vec<_> = ctx.op_regions(op).collect();
    for &region in &regions {
        ctx.detach_region(region);
        builder = builder.region(region);
    }
    let convention = <CallingConvention as trunk_ir::attr_kind::AttrKind>::write(ctx, convention);
    let builder = if name == Lambda::NAME {
        builder.attr(
            CONVENTIONS_ATTR,
            trunk_ir::types::Attribute::List(vec![convention]),
        )
    } else if name == FuncRef::NAME {
        builder
    } else {
        builder.attr(CONVENTION_ATTR, convention)
    };
    let data = builder.build(ctx);
    let replacement = ctx.create_op(data);
    if let Some(block) = ctx.op(op).parent_block {
        ctx.insert_op_before(block, op, replacement);
        ctx.remove_op_from_block(block, op);
    }
    let results = ctx.op_results(op).to_vec();
    for (index, old) in results.into_iter().enumerate() {
        let new = ctx.op_result(replacement, index as u32);
        ctx.replace_all_uses(old, new);
    }
    ctx.remove_op(op);
    replacement
}

fn verify_convention(
    ctx: &IrContext,
    op: trunk_ir::refs::OpRef,
) -> Result<(), trunk_ir::ops::VerificationError> {
    match declared_convention(ctx, op) {
        Some(_) => Ok(()),
        None => Err(MalformedConvention.into()),
    }
}

impl trunk_ir::ops::Verify for Func {
    fn verify(&self, ctx: &IrContext) -> Result<(), trunk_ir::ops::VerificationError> {
        verify_convention(ctx, self.op_ref())?;
        Ok(tribute_control::verify_row_declarations(
            ctx,
            self.op_ref(),
        )?)
    }
}

impl trunk_ir::ops::Verify for Lambda {
    fn verify(&self, ctx: &IrContext) -> Result<(), trunk_ir::ops::VerificationError> {
        match body_conventions(ctx, self.op_ref()) {
            Some(_) => Ok(()),
            None => Err(MalformedBodies.into()),
        }
    }
}

impl trunk_ir::ops::Verify for FuncRef {
    fn verify(&self, ctx: &IrContext) -> Result<(), trunk_ir::ops::VerificationError> {
        Ok(tribute_control::row_args(ctx, self.op_ref()).map(drop)?)
    }
}

impl trunk_ir::ops::Verify for FuncTable {
    fn verify(&self, ctx: &IrContext) -> Result<(), trunk_ir::ops::VerificationError> {
        match table_entries(ctx, self.op_ref()) {
            Some(_) => Ok(()),
            None => Err(MalformedTable.into()),
        }
    }
}

impl trunk_ir::ops::Verify for CallIndirect {
    fn verify(&self, ctx: &IrContext) -> Result<(), trunk_ir::ops::VerificationError> {
        verify_convention(ctx, self.op_ref())?;
        Ok(tribute_control::verify_evidence_plan(
            ctx,
            self.op_ref(),
            EvidencePlanSite::Source,
        )?)
    }
}

impl trunk_ir::ops::Verify for TailCallIndirect {
    fn verify(&self, ctx: &IrContext) -> Result<(), trunk_ir::ops::VerificationError> {
        verify_convention(ctx, self.op_ref())?;
        tribute_control::verify_resolved_types(ctx, self.op_ref())?;
        Ok(tribute_control::verify_evidence_plan(
            ctx,
            self.op_ref(),
            EvidencePlanSite::Source,
        )?)
    }
}

impl trunk_ir::asm_format::OpAsmFormat for Func {
    fn print(&self, h: &mut trunk_ir::printer::OpPrintHelper<'_, '_>) -> std::fmt::Result {
        match declared_convention(h.ctx(), self.op_ref()) {
            Some(convention) => {
                tribute_control::print_func(h, self.op_ref(), "tribute_cps.func", Some(convention))
            }
            None => h.print_generic(self.op_ref()),
        }
    }

    fn parse<'a>(
        input: &mut &'a str,
        results: Vec<&'a str>,
        sym_name: Option<String>,
    ) -> winnow::ModalResult<trunk_ir::parser::raw::RawOperation<'a>> {
        tribute_control::parse_func(input, results, sym_name, "tribute_cps.func", true)
    }
}

impl trunk_ir::asm_format::OpAsmFormat for Lambda {
    fn print(&self, h: &mut trunk_ir::printer::OpPrintHelper<'_, '_>) -> std::fmt::Result {
        match body_conventions(h.ctx(), self.op_ref()) {
            Some(conventions) => tribute_control::print_lambda(
                h,
                self.op_ref(),
                "tribute_cps.lambda",
                Some(&conventions),
            ),
            None => h.print_generic(self.op_ref()),
        }
    }

    fn parse<'a>(
        input: &mut &'a str,
        results: Vec<&'a str>,
        sym_name: Option<String>,
    ) -> winnow::ModalResult<trunk_ir::parser::raw::RawOperation<'a>> {
        tribute_control::parse_lambda(input, results, sym_name, "tribute_cps.lambda", true)
    }
}

impl CallableExit for TailCallIndirect {}

/// As `tribute_control.func`.
impl trunk_ir::op_interface::SymbolDefinition for Func {
    fn is_discardable(&self, ctx: &IrContext) -> bool {
        !(ctx.op_attrs(self.op_ref()).contains_key("abi") && ctx.op_has_regions(self.op_ref()))
    }
}

/// A table has no body, but it defines its entries rather than declaring
/// a symbol defined elsewhere.
impl trunk_ir::op_interface::SymbolDefinition for FuncTable {
    fn is_declaration(&self, _ctx: &IrContext) -> bool {
        false
    }
}

impl LocalValidation for Func {
    fn validate(&self, ctx: &IrContext, errors: &mut Vec<ValidationError>) {
        tribute_control::validate_func(ctx, self.op_ref(), errors);
    }
}

impl LocalValidation for Lambda {
    fn validate(&self, ctx: &IrContext, errors: &mut Vec<ValidationError>) {
        tribute_control::validate_lambda(ctx, self.op_ref(), errors);
    }
}

impl LocalValidation for TailCallIndirect {
    fn validate(&self, ctx: &IrContext, errors: &mut Vec<ValidationError>) {
        tribute_control::validate_tail_call(ctx, self.op_ref(), errors);
    }
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
        tribute_control::is_signature_parameter(
            ctx,
            self.op_ref(),
            signature,
            region,
            index,
            argument,
        )
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
        tribute_control::is_signature_parameter(
            ctx,
            self.op_ref(),
            signature,
            region,
            index,
            argument,
        )
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
            .is_some_and(|symbol| query.names_verified_table(ctx, self.op_ref(), symbol))
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

#[cfg(test)]
mod tests;
