//! Plan-driven native RC materialization before type erasure.
//!
//! This pass consumes only a validated [`NativeOwnershipPlan`].  It does not
//! discover ownership from physical types, pointer provenance, aliases, or
//! liveness after semantic references have been erased.

use std::collections::HashMap;

use tribute_ir::dialect::tribute_rt::{self, RC_HEADER_SIZE};
use trunk_ir::adt_layout::{compute_enum_layout, compute_struct_layout, get_struct_fields};
use trunk_ir::context::IrContext;
use trunk_ir::dialect::adt;
use trunk_ir::ops::DialectOp;
use trunk_ir::rewrite::{Module, TypeConverter};
use trunk_ir::{BlockRef, OpRef, Symbol, TypeRef, ValueRef};

use super::ownership_plan::{ActionAnchor, ActionKind, NativeOwnershipPlan, OwnershipPlanError};

pub type RcMaterializationError = OwnershipPlanError;

#[derive(Clone, Copy)]
enum Placement {
    BlockStart(BlockRef),
    Before(OpRef),
    After(OpRef),
}

enum ScheduledAction {
    Retain {
        placement: Placement,
        value: ValueRef,
    },
    Release {
        placement: Placement,
        value: ValueRef,
        alloc_size: u64,
    },
    ReleaseReplacedField {
        placement: Placement,
        object: ValueRef,
        layout: TypeRef,
        field: u32,
        field_ty: TypeRef,
        alloc_size: u64,
    },
}

/// Validate every requested insertion before materializing the plan's exact RC
/// operations.  On error, no operation is attached to the module.
pub fn materialize(
    ctx: &mut IrContext,
    module: Module,
    plan: &NativeOwnershipPlan,
) -> Result<(), RcMaterializationError> {
    // Constructing the native type converter may intern representation types,
    // but the complete schedule is still validated before any IR mutation.
    let (type_converter, _) = super::type_converter::native_type_converter(ctx);
    let schedule = build_schedule(ctx, module, plan, &type_converter)?;
    let mut at_start = HashMap::<BlockRef, Vec<OpRef>>::new();
    let mut before = HashMap::<OpRef, Vec<OpRef>>::new();
    let mut after = HashMap::<OpRef, Vec<OpRef>>::new();

    for action in schedule {
        let (placement, operations) = build_operations(ctx, action);
        match placement {
            Placement::BlockStart(block) => at_start.entry(block).or_default().extend(operations),
            Placement::Before(op) => before.entry(op).or_default().extend(operations),
            Placement::After(op) => after.entry(op).or_default().extend(operations),
        }
    }

    for (block, operations) in at_start {
        let first = ctx
            .block(block)
            .ops
            .first()
            .copied()
            .expect("validated materialization block must have an insertion anchor");
        insert_before(ctx, block, first, operations);
    }
    for (op, operations) in before {
        let block = ctx.op(op).parent_block.expect("validated action anchor");
        insert_before(ctx, block, op, operations);
    }
    for (op, operations) in after {
        let block = ctx.op(op).parent_block.expect("validated action anchor");
        insert_after(ctx, block, op, operations);
    }
    Ok(())
}

fn build_schedule(
    ctx: &IrContext,
    module: Module,
    plan: &NativeOwnershipPlan,
    type_converter: &TypeConverter,
) -> Result<Vec<ScheduledAction>, RcMaterializationError> {
    plan.validate_against(ctx, module)?;
    let mut schedule = Vec::new();
    for function in plan.functions() {
        for action in function.actions() {
            let placement = placement_for(ctx, action.anchor)?;
            match action.kind {
                ActionKind::EntryAcquire
                | ActionKind::CallRetain
                | ActionKind::CallAcquire
                | ActionKind::StoreAcquire
                | ActionKind::CopyAcquire => schedule.push(ScheduledAction::Retain {
                    placement,
                    value: action.value,
                }),
                ActionKind::FinalRelease => schedule.push(ScheduledAction::Release {
                    placement,
                    value: action.value,
                    alloc_size: allocation_size_for_value(ctx, plan, action.value, type_converter)?,
                }),
                ActionKind::ReleaseReplacedField => {
                    let (object, layout, field, field_ty) = validate_replaced_field(
                        ctx,
                        action.anchor,
                        action.value,
                        action.destination,
                        plan,
                    )?;
                    schedule.push(ScheduledAction::ReleaseReplacedField {
                        placement,
                        object,
                        layout,
                        field,
                        field_ty,
                        alloc_size: allocation_size_for_type(ctx, plan, field_ty, type_converter)?,
                    });
                }
                // These actions are ownership facts used by the planner. They
                // deliberately have no standalone RC operation to emit.
                ActionKind::CallBorrow
                | ActionKind::BorrowLoad
                | ActionKind::ReturnTransfer
                | ActionKind::TailTransfer
                | ActionKind::IntoRawTransfer => {}
            }
        }
    }
    Ok(schedule)
}

/// Native allocation size is the aggregate payload layout plus the runtime RC
/// header.  This runs only on an already-planned semantic value before
/// `func_to_clif` erases it to `core.ptr`.
fn allocation_size_for_value(
    ctx: &IrContext,
    plan: &NativeOwnershipPlan,
    value: ValueRef,
    type_converter: &TypeConverter,
) -> Result<u64, RcMaterializationError> {
    allocation_size_for_type(ctx, plan, ctx.value_ty(value), type_converter)
}

fn allocation_size_for_type(
    ctx: &IrContext,
    plan: &NativeOwnershipPlan,
    ty: TypeRef,
    type_converter: &TypeConverter,
) -> Result<u64, RcMaterializationError> {
    // `anyref`/`intref` deliberately carry no nominal allocation identity.
    // Zero is a dispatch-only dynamic-size signal: header RTTI must resolve it
    // to an exact release function before deallocation. It is never a shallow
    // fallback and is not inferred from physical definitions.
    let data = ctx.types.get(ty);
    if data.dialect == Symbol::new("tribute_rt")
        && matches!(data.name, name if name == Symbol::new("anyref") || name == Symbol::new("intref"))
    {
        return Ok(0);
    }
    let layout = plan.allocation_layout_for_type(ctx, ty)?;
    let payload_size = compute_struct_layout(ctx, layout, type_converter)
        .map(|layout| layout.total_size)
        .or_else(|| {
            compute_enum_layout(ctx, layout, type_converter).map(|layout| layout.total_size)
        })
        .ok_or_else(|| OwnershipPlanError::new("planned release layout has no native size"))?;
    Ok(u64::from(payload_size) + RC_HEADER_SIZE)
}

fn placement_for(
    ctx: &IrContext,
    anchor: ActionAnchor,
) -> Result<Placement, RcMaterializationError> {
    match anchor {
        ActionAnchor::BlockStart(block) if !ctx.block(block).ops.is_empty() => {
            Ok(Placement::BlockStart(block))
        }
        ActionAnchor::BlockStart(_) => Err(OwnershipPlanError::new(
            "ownership action has no block-start insertion anchor",
        )),
        ActionAnchor::Before(op) => Ok(Placement::Before(op)),
        ActionAnchor::After(op) => Ok(Placement::After(op)),
    }
}

fn validate_replaced_field(
    ctx: &IrContext,
    anchor: ActionAnchor,
    object: ValueRef,
    destination: u32,
    plan: &NativeOwnershipPlan,
) -> Result<(ValueRef, TypeRef, u32, TypeRef), RcMaterializationError> {
    let ActionAnchor::Before(op) = anchor else {
        return Err(OwnershipPlanError::new(
            "replaced-field release must be anchored before its struct_set",
        ));
    };
    let set = adt::StructSet::from_op(ctx, op).map_err(|_| {
        OwnershipPlanError::new("replaced-field release anchor is not an adt.struct_set")
    })?;
    let field = set.field(ctx);
    if set.r#ref(ctx) != object || field != destination {
        return Err(OwnershipPlanError::new(
            "replaced-field release does not match its planned object or field",
        ));
    }
    let fields = get_struct_fields(ctx, set.r#type(ctx))
        .ok_or_else(|| OwnershipPlanError::new("replaced-field layout is stale"))?;
    let (_, field_ty) = fields
        .get(field as usize)
        .ok_or_else(|| OwnershipPlanError::new("replaced-field index is stale"))?;
    if !plan.is_managed_type(ctx, *field_ty) {
        return Err(OwnershipPlanError::new(
            "replaced-field release targets an unmanaged field",
        ));
    }
    Ok((object, set.r#type(ctx), field, *field_ty))
}

fn build_operations(ctx: &mut IrContext, action: ScheduledAction) -> (Placement, Vec<OpRef>) {
    match action {
        ScheduledAction::Retain { placement, value } => {
            let ty = ctx.value_ty(value);
            let retain = tribute_rt::retain(ctx, location_for(ctx, placement), value, ty);
            (placement, vec![retain.op_ref()])
        }
        ScheduledAction::Release {
            placement,
            value,
            alloc_size,
        } => {
            let release = tribute_rt::release(ctx, location_for(ctx, placement), value, alloc_size);
            (placement, vec![release.op_ref()])
        }
        ScheduledAction::ReleaseReplacedField {
            placement,
            object,
            layout,
            field,
            field_ty,
            alloc_size,
        } => {
            let location = location_for(ctx, placement);
            let get = adt::struct_get(ctx, location, object, field_ty, layout, field);
            let release = tribute_rt::release(ctx, location, get.result(ctx), alloc_size);
            (placement, vec![get.op_ref(), release.op_ref()])
        }
    }
}

fn location_for(ctx: &IrContext, placement: Placement) -> trunk_ir::Location {
    match placement {
        Placement::BlockStart(block) => ctx.block(block).location,
        Placement::Before(op) | Placement::After(op) => ctx.op(op).location,
    }
}

fn insert_before(ctx: &mut IrContext, block: BlockRef, anchor: OpRef, operations: Vec<OpRef>) {
    for operation in operations {
        ctx.insert_op_before(block, anchor, operation);
    }
}

fn insert_after(ctx: &mut IrContext, block: BlockRef, anchor: OpRef, operations: Vec<OpRef>) {
    let next = ctx
        .block(block)
        .ops
        .iter()
        .position(|&operation| operation == anchor)
        .and_then(|index| ctx.block(block).ops.get(index + 1).copied());
    for operation in operations {
        if let Some(next) = next {
            ctx.insert_op_before(block, next, operation);
        } else {
            ctx.push_op(block, operation);
        }
    }
}
