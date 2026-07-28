//! Opaque control carriers used by compositional CPS lowering.
//!
//! The marker types are lowering-internal answer domains. They do not change
//! the physical TrunkIR ABI, which remains `anyref` until #774.

use std::marker::PhantomData;

use trunk_ir::dialect::{func, scf};
use trunk_ir::refs::{TypeRef, ValueRef};
use trunk_ir::types::Location;

use super::IrBuilder;

mod private {
    pub trait Sealed {}
}

pub(super) trait ControlDomain: private::Sealed {}

pub(super) enum Ambient {}
pub(super) enum HandleAnswer {}
pub(super) enum TailResume {}

impl private::Sealed for Ambient {}
impl private::Sealed for HandleAnswer {}
impl private::Sealed for TailResume {}
impl ControlDomain for Ambient {}
impl ControlDomain for HandleAnswer {}
impl ControlDomain for TailResume {}

/// A source-result continuation whose opaque answer belongs to `D`.
pub(super) struct ContinuationRef<D: ControlDomain>(ValueRef, PhantomData<fn() -> D>);

/// The current logical control result in answer domain `D`.
pub(super) struct ControlResultRef<D: ControlDomain>(ValueRef, PhantomData<fn() -> D>);

/// The affine suspended continuation bound by an `op` handler arm.
pub(super) struct ResumeRef(ContinuationRef<HandleAnswer>);
impl<D: ControlDomain> Clone for ContinuationRef<D> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<D: ControlDomain> Copy for ContinuationRef<D> {}

impl<D: ControlDomain> Clone for ControlResultRef<D> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<D: ControlDomain> Copy for ControlResultRef<D> {}

impl Clone for ResumeRef {
    fn clone(&self) -> Self {
        *self
    }
}

impl Copy for ResumeRef {}

pub(super) fn continuation_from_abi<D: ControlDomain>(value: ValueRef) -> ContinuationRef<D> {
    ContinuationRef(value, PhantomData)
}

pub(super) fn control_from_abi<D: ControlDomain>(value: ValueRef) -> ControlResultRef<D> {
    ControlResultRef(value, PhantomData)
}

/// Expose a continuation only for a CPS ABI argument.
pub(super) fn continuation_abi<D: ControlDomain>(continuation: ContinuationRef<D>) -> ValueRef {
    continuation.0
}

pub(super) fn resume_from_abi(value: ValueRef) -> ResumeRef {
    ResumeRef(continuation_from_abi(value))
}

/// Invoke a source-result continuation and keep its compatibility carrier
/// opaque in the same answer domain.
pub(super) fn invoke_continuation<D: ControlDomain>(
    builder: &mut IrBuilder<'_, '_>,
    location: Location,
    continuation: ContinuationRef<D>,
    value: ValueRef,
) -> ControlResultRef<D> {
    let anyref_ty = builder.ctx.anyref_type(builder.ir);
    let closure_func_ty = builder.ctx.func_type(builder.ir, &[anyref_ty], anyref_ty);
    let closure_ty = builder.ctx.closure_type(builder.ir, closure_func_ty);
    let continuation = builder.cast_if_needed(location, continuation.0, closure_ty);
    let value = builder.cast_if_needed(location, value, anyref_ty);
    let call = func::call_indirect(builder.ir, location, continuation, vec![value], anyref_ty);
    builder.ir.push_op(builder.block, call.op_ref());
    control_from_abi(call.result(builder.ir))
}

pub(super) fn invoke_resume(
    builder: &mut IrBuilder<'_, '_>,
    location: Location,
    resume: ResumeRef,
    value: ValueRef,
) -> ControlResultRef<HandleAnswer> {
    invoke_continuation(builder, location, resume.0, value)
}

/// Emit a control-only function return. This is one of the legal carrier
/// unwrap sites.
pub(super) fn emit_control_return<D: ControlDomain>(
    builder: &mut IrBuilder<'_, '_>,
    location: Location,
    result: ControlResultRef<D>,
) {
    let ret = func::r#return(builder.ir, location, [result.0]);
    builder.ir.push_op(builder.block, ret.op_ref());
}

/// Emit a structured-control yield. This is the other general legal carrier
/// unwrap site.
pub(super) fn emit_control_yield<D: ControlDomain>(
    builder: &mut IrBuilder<'_, '_>,
    location: Location,
    result: ControlResultRef<D>,
) {
    let yield_op = scf::r#yield(builder.ir, location, [result.0]);
    builder.ir.push_op(builder.block, yield_op.op_ref());
}

/// Complete a tail-resumptive handler arm through the `ability.yield`
/// compatibility region.
pub(super) fn emit_tail_resume_yield(
    builder: &mut IrBuilder<'_, '_>,
    location: Location,
    result: ControlResultRef<TailResume>,
) {
    emit_control_yield(builder, location, result);
}

/// The sanctioned delimiter conversion for a completed handle answer.
pub(super) fn handle_answer_to_source(
    builder: &mut IrBuilder<'_, '_>,
    location: Location,
    answer: ControlResultRef<HandleAnswer>,
    logical_ty: TypeRef,
) -> ValueRef {
    builder.cast_if_needed(location, answer.0, logical_ty)
}

/// The sanctioned conversion for the answer returned to a `resume` call.
pub(super) fn resume_answer_to_source(
    builder: &mut IrBuilder<'_, '_>,
    location: Location,
    answer: ControlResultRef<HandleAnswer>,
    logical_ty: TypeRef,
) -> ValueRef {
    builder.cast_if_needed(location, answer.0, logical_ty)
}

/// The sanctioned root-entry conversion for a computation closed by the
/// frontend's identity `done_k`. Root `main` is the only Ambient delimiter;
/// ordinary source lowering must not recover an Ambient control carrier.
pub(super) fn root_main_answer_to_source(
    builder: &mut IrBuilder<'_, '_>,
    location: Location,
    answer: ControlResultRef<Ambient>,
    logical_ty: TypeRef,
) -> ValueRef {
    // Source root main returns Nil. Its completion value is deliberately
    // ignored after the CPS chain has run; a native/Wasm ABI cannot represent
    // an anyref-to-nil cast as a value conversion.
    if logical_ty == builder.ctx.nil_type(builder.ir) {
        let _ = answer;
        return builder.emit_nil(location);
    }
    builder.cast_if_needed(location, answer.0, logical_ty)
}

pub(super) fn identity_continuation<D: ControlDomain>(
    builder: &mut IrBuilder<'_, '_>,
    location: Location,
) -> ContinuationRef<D> {
    continuation_from_abi(super::create_identity_done_k(builder, location))
}
