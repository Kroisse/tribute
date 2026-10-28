//! Source-logical callables whose calling convention the CPS strategy has
//! assigned (`new-plans/ir.md`, "Convention과 class 특수화").
//!
//! Class specialization replaces the five `tribute_control` operations whose
//! meaning depends on a convention with the operation of the same name here,
//! which adds that convention. The other `tribute_control` operations stay
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

use crate::calling_convention::CallingConvention;
use crate::dialect::tribute_control::{
    self, CallableArgumentOrigin, CallableResultOrigin, EvidencePlanSite, EvidenceStep, FuncSig,
    Row, RowBinder, RowUnion,
};

/// The attribute that holds a `tribute_cps` operation's convention.
pub const CONVENTION_ATTR: &str = "convention";

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
    /// - **Location:** the source function's span.
    #[verify]
    #[asm_format]
    #[flags(IsolatedFromAbove)]
    #[implements(CallableArgumentOrigin)]
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

    /// A source lambda with its convention entry.
    ///
    /// ```text
    /// %f = tribute_cps.lambda(%x: T) -> R captures [%capture0, ...] entries {
    ///   cps {
    ///     ...
    ///     tribute_control.return %value
    ///   }
    /// }
    /// ```
    ///
    /// - **Shape:** `tribute_control.lambda` with a required
    ///   `convention: String`. The body is the convention entry of that
    ///   convention, written under its keyword in `entries`.
    /// - **Meaning:** the lambda of one class vector of its uses, whose body
    ///   reads the row of each binder with the class its uses pass
    ///   (`new-plans/ir.md`, "Convention entry"). Calling the value with its
    ///   convention runs the body.
    /// - **Verification:** the local verifier checks the convention keyword.
    ///   Otherwise as `tribute_control.lambda`.
    /// - **Ownership and value flow:** as `tribute_control.lambda`.
    /// - **Location:** the source lambda's span.
    #[verify]
    #[asm_format]
    #[flags(Pure)]
    #[implements(CallableResultOrigin, CallableArgumentOrigin)]
    fn lambda(
        convention: Attr<CallingConvention>,
        quantified_rows: Option<Attr<[u32]>>,
        captures: Variadic<_>,
    ) -> Value<impl FuncSig> {
        #[region(body)]
        {}
    }

    /// A named function instance as a value, entered with one convention.
    ///
    /// ```text
    /// %f = tribute_cps.func_ref {convention = "cps", func_ref = @id} : !Callable
    /// ```
    ///
    /// - **Shape:** `tribute_control.func_ref` with a required
    ///   `convention: String`.
    /// - **Meaning:** the callable value whose convention entry `convention`
    ///   runs the instance `func_ref` names, which the reference's
    ///   `row_args` selected. When the instance's convention is weaker,
    ///   legalization fills the entry with an adapter (`new-plans/ir.md`,
    ///   "Convention entry").
    /// - **Verification:** the local verifier checks the attributes. Whole-IR
    ///   verification checks the source signature as for
    ///   `tribute_control.func_ref`, and that `convention` is not weaker than
    ///   the instance's.
    /// - **Ownership and value flow:** the result is an ordinary SSA value.
    /// - **Location:** the source reference span.
    #[verify]
    #[flags(Pure)]
    #[implements(CallableResultOrigin)]
    fn func_ref(
        func_ref: Attr<SymbolRef>,
        convention: Attr<CallingConvention>,
        row_args: Option<Attr<[Row]>>,
    ) -> Value<impl FuncSig> {
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
    #[implements(CallableExit)]
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
/// valid one.
pub fn declared_convention(
    ctx: &IrContext,
    op: trunk_ir::refs::OpRef,
) -> Option<CallingConvention> {
    if ctx.op(op).name.dialect(ctx) != DIALECT_NAME {
        return None;
    }
    CallingConvention::from_attribute(ctx, ctx.op_attrs(op).get(CONVENTION_ATTR)?)
}

/// Replace `op`, a `tribute_control` func, lambda, func_ref, call_indirect,
/// or tail_call_indirect, with its operation here, which declares
/// `convention`. The replacement takes the operation's place, operands,
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
    let builder = builder.attr(
        CONVENTION_ATTR,
        <CallingConvention as trunk_ir::attr_kind::AttrKind>::write(ctx, convention),
    );
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
        verify_convention(ctx, self.op_ref())
    }
}

impl trunk_ir::ops::Verify for Lambda {
    fn verify(&self, ctx: &IrContext) -> Result<(), trunk_ir::ops::VerificationError> {
        verify_convention(ctx, self.op_ref())
    }
}

impl trunk_ir::ops::Verify for FuncRef {
    fn verify(&self, ctx: &IrContext) -> Result<(), trunk_ir::ops::VerificationError> {
        verify_convention(ctx, self.op_ref())?;
        Ok(tribute_control::row_args(ctx, self.op_ref()).map(drop)?)
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
        match declared_convention(h.ctx(), self.op_ref()) {
            Some(convention) => tribute_control::print_lambda(
                h,
                self.op_ref(),
                "tribute_cps.lambda",
                Some(convention),
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

#[cfg(test)]
mod tests;
