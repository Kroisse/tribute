//! Opaque control carriers used by compositional CPS lowering.
//!
//! The marker types are lowering-internal answer domains. They do not change
//! the physical TrunkIR ABI, which remains `anyref` until #774.

use std::marker::PhantomData;

use trunk_ir::context::{BlockData, RegionData};
use trunk_ir::dialect::{adt, func, scf};
use trunk_ir::refs::{TypeRef, ValueRef};
use trunk_ir::types::Location;

use tribute_ir::dialect::ability;

use super::IrBuilder;

mod private {
    pub trait Sealed {}
}

pub(super) trait ControlDomain: private::Sealed {
    /// Whether a nested handle must retain the private control carrier for
    /// the current continuation instead of closing it as a source delimiter.
    const PROPAGATES_HANDLES: bool;
}

pub(super) enum Ambient {}
pub(super) enum HandleAnswer {}
pub(super) enum TailResume {}

impl private::Sealed for Ambient {}
impl private::Sealed for HandleAnswer {}
impl private::Sealed for TailResume {}
impl ControlDomain for Ambient {
    const PROPAGATES_HANDLES: bool = true;
}
impl ControlDomain for HandleAnswer {
    const PROPAGATES_HANDLES: bool = true;
}
impl ControlDomain for TailResume {
    const PROPAGATES_HANDLES: bool = false;
}

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

/// Continue an `op` arm after a resumed body completion.
///
/// The captured resume continuation and the arm-local continuation are both
/// private handle boundaries. A resumed `Normal` may run the arm's strict
/// suffix. The arm's own Escape is retagged to `Normal` so the enclosing `do`
/// still runs; a foreign Escape bypasses that suffix unchanged.
pub(super) fn resume_into_current<D: ControlDomain>(
    builder: &mut IrBuilder<'_, '_>,
    location: Location,
    resume: ResumeRef,
    value: ValueRef,
    logical_ty: TypeRef,
    current_k: ContinuationRef<D>,
    current_owner: ValueRef,
) -> ControlResultRef<D> {
    let answer = invoke_resume(builder, location, resume, value);
    let anyref_ty = builder.ctx.anyref_type(builder.ir);
    let bool_ty = builder.ctx.bool_type(builder.ir);
    let control_ty = super::cps_control_type(builder);
    let normal = adt::variant_is(
        builder.ir,
        location,
        answer.0,
        bool_ty,
        control_ty,
        trunk_ir::Symbol::new(ability::CPS_CONTROL_NORMAL_VARIANT),
    );
    builder.ir.push_op(builder.block, normal.op_ref());

    let normal_block = builder.ir.create_block(BlockData {
        location,
        args: vec![],
        ops: Default::default(),
        parent_region: None,
    });
    {
        let mut normal_builder = IrBuilder::new(builder.ctx, builder.ir, normal_block);
        let i32_ty = normal_builder.ctx.i32_type(normal_builder.ir);
        let bool_ty = normal_builder.ctx.bool_type(normal_builder.ir);
        let cast = adt::variant_cast(
            normal_builder.ir,
            location,
            answer.0,
            anyref_ty,
            control_ty,
            trunk_ir::Symbol::new(ability::CPS_CONTROL_NORMAL_VARIANT),
        );
        normal_builder.ir.push_op(normal_block, cast.op_ref());
        let payload = adt::variant_get(
            normal_builder.ir,
            location,
            cast.result(normal_builder.ir),
            anyref_ty,
            control_ty,
            trunk_ir::Symbol::new(ability::CPS_CONTROL_NORMAL_VARIANT),
            0,
        );
        normal_builder.ir.push_op(normal_block, payload.op_ref());
        let payload =
            normal_builder.cast_if_needed(location, payload.result(normal_builder.ir), logical_ty);
        let after_arm = invoke_continuation(&mut normal_builder, location, current_k, payload);
        // `current_k` is the arm-local completion chain, so its result is a
        // proven private Escape. A suffix may have forwarded a foreign Escape;
        // only the current dynamic owner can become Normal here.
        let escape = adt::variant_cast(
            normal_builder.ir,
            location,
            after_arm.0,
            anyref_ty,
            control_ty,
            trunk_ir::Symbol::new(ability::CPS_CONTROL_ESCAPE_VARIANT),
        );
        normal_builder.ir.push_op(normal_block, escape.op_ref());
        let owner = adt::variant_get(
            normal_builder.ir,
            location,
            escape.result(normal_builder.ir),
            i32_ty,
            control_ty,
            trunk_ir::Symbol::new(ability::CPS_CONTROL_ESCAPE_VARIANT),
            0,
        );
        normal_builder.ir.push_op(normal_block, owner.op_ref());
        let payload = adt::variant_get(
            normal_builder.ir,
            location,
            escape.result(normal_builder.ir),
            anyref_ty,
            control_ty,
            trunk_ir::Symbol::new(ability::CPS_CONTROL_ESCAPE_VARIANT),
            1,
        );
        normal_builder.ir.push_op(normal_block, payload.op_ref());
        let owner_matches = trunk_ir::dialect::arith::cmpi(
            normal_builder.ir,
            location,
            owner.result(normal_builder.ir),
            current_owner,
            bool_ty,
            trunk_ir::Symbol::new("eq"),
        );
        normal_builder
            .ir
            .push_op(normal_block, owner_matches.op_ref());
        let own_block = normal_builder.ir.create_block(BlockData {
            location,
            args: vec![],
            ops: Default::default(),
            parent_region: None,
        });
        let retagged = adt::variant_new(
            normal_builder.ir,
            location,
            vec![payload.result(normal_builder.ir)],
            anyref_ty,
            control_ty,
            trunk_ir::Symbol::new(ability::CPS_CONTROL_NORMAL_VARIANT),
        );
        normal_builder.ir.push_op(own_block, retagged.op_ref());
        let own_yield = scf::r#yield(
            normal_builder.ir,
            location,
            [retagged.result(normal_builder.ir)],
        );
        normal_builder.ir.push_op(own_block, own_yield.op_ref());
        let own_region = normal_builder.ir.create_region(RegionData {
            location,
            blocks: trunk_ir::smallvec::smallvec![own_block],
            parent_op: None,
        });
        let foreign_block = normal_builder.ir.create_block(BlockData {
            location,
            args: vec![],
            ops: Default::default(),
            parent_region: None,
        });
        let foreign_yield = scf::r#yield(normal_builder.ir, location, [after_arm.0]);
        normal_builder
            .ir
            .push_op(foreign_block, foreign_yield.op_ref());
        let foreign_region = normal_builder.ir.create_region(RegionData {
            location,
            blocks: trunk_ir::smallvec::smallvec![foreign_block],
            parent_op: None,
        });
        let selected = scf::r#if(
            normal_builder.ir,
            location,
            owner_matches.result(normal_builder.ir),
            anyref_ty,
            own_region,
            foreign_region,
        );
        normal_builder.ir.push_op(normal_block, selected.op_ref());
        let selected = selected.result(normal_builder.ir);
        emit_control_yield(
            &mut normal_builder,
            location,
            control_from_abi::<D>(selected),
        );
    }
    let normal_region = builder.ir.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![normal_block],
        parent_op: None,
    });

    let escape_block = builder.ir.create_block(BlockData {
        location,
        args: vec![],
        ops: Default::default(),
        parent_region: None,
    });
    {
        let mut escape_builder = IrBuilder::new(builder.ctx, builder.ir, escape_block);
        emit_control_yield(
            &mut escape_builder,
            location,
            control_from_abi::<D>(answer.0),
        );
    }
    let escape_region = builder.ir.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![escape_block],
        parent_op: None,
    });

    let branch = scf::r#if(
        builder.ir,
        location,
        normal.result(builder.ir),
        anyref_ty,
        normal_region,
        escape_region,
    );
    builder.ir.push_op(builder.block, branch.op_ref());
    control_from_abi(branch.result(builder.ir))
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

/// Close a completed private handle carrier at a source delimiter. The
/// carrier has already been consumed by its dynamic owner before this point,
/// so no variant probe is needed or permitted here.
pub(super) fn handle_answer_to_source(
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

/// A continuation whose result is proven to be the private #815 completion
/// carrier. It is used only at handle-body and general-handler boundaries.
pub(super) fn cps_control_continuation(
    builder: &mut IrBuilder<'_, '_>,
    location: Location,
) -> ContinuationRef<HandleAnswer> {
    continuation_from_abi(super::create_cps_control_done_k(builder, location))
}
