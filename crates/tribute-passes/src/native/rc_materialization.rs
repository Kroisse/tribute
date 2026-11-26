//! Plan-driven native RC materialization before type erasure.
//!
//! This pass consumes only a validated [`NativeOwnershipPlan`].  It does not
//! discover ownership from physical types, pointer provenance, aliases, or
//! liveness after semantic references have been erased.

use rustc_hash::FxHashMap as HashMap;

use tribute_ir::dialect::adt::layout::{compute_struct_layout, get_struct_fields, natural_layout};
use tribute_ir::dialect::tribute_rt::{self, RC_HEADER_SIZE};
use tribute_ir::dialect::tribute_rtti::{DynOperandOwnership, OperandRole};
use tribute_ir::dialect::{adt, heap};
use trunk_ir::context::IrContext;
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::rewrite::{Module, TypeConverter};
use trunk_ir::{BlockRef, OpRef, TypeRef, ValueRef};

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
    /// Release the value that the assigning operation `store` replaces,
    /// read by its
    /// [`build_replaced_read`](tribute_ir::dialect::tribute_rtti::OperandOwnership::build_replaced_read).
    ReleaseReplaced {
        placement: Placement,
        store: OpRef,
        alloc_size: u64,
    },
    ReleaseMoving {
        placement: Placement,
        owner: ValueRef,
        moved: Vec<ValueRef>,
        alloc_size: u64,
        release_offsets: Vec<i32>,
    },
}

/// Validate every requested insertion before materializing the plan's exact RC
/// operations.  On error, no operation is attached to the module.
pub(crate) fn materialize(
    ctx: &mut IrContext,
    module: Module,
    plan: &NativeOwnershipPlan,
) -> Result<(), RcMaterializationError> {
    // Constructing the native type converter may intern representation types,
    // but the complete schedule is still validated before any IR mutation.
    let (type_converter, _) = super::type_converter::native_type_converter(ctx);
    let schedule = build_schedule(ctx, module, plan, &type_converter)?;
    let mut at_start = HashMap::<BlockRef, Vec<OpRef>>::default();
    let mut before = HashMap::<OpRef, Vec<OpRef>>::default();
    let mut after = HashMap::<OpRef, Vec<OpRef>>::default();

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
        // Owners dying after the same operation share an anchor, so each
        // moved field is keyed by the owner its projection reads too.
        let mut field_moves = HashMap::<(ActionAnchor, ValueRef), Vec<(ValueRef, u32)>>::default();
        for action in function.actions() {
            if action.kind == ActionKind::FieldMove {
                field_moves
                    .entry((action.anchor, projection_owner(ctx, action.value)?))
                    .or_default()
                    .push((action.value, action.destination));
            }
        }
        for action in function.actions() {
            let placement = placement_for(ctx, action.anchor)?;
            match action.kind {
                ActionKind::EntryAcquire
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
                    let (store, field_ty) = validate_replaced_field(
                        ctx,
                        action.anchor,
                        action.value,
                        action.destination,
                        plan,
                    )?;
                    schedule.push(ScheduledAction::ReleaseReplaced {
                        placement,
                        store,
                        alloc_size: allocation_size_for_type(ctx, plan, field_ty, type_converter)?,
                    });
                }
                ActionKind::ReleaseMoving => {
                    let moved = field_moves
                        .remove(&(action.anchor, action.value))
                        .ok_or_else(|| {
                            OwnershipPlanError::new("moving release has no moved field")
                        })?;
                    schedule.push(release_moving(
                        ctx,
                        plan,
                        placement,
                        action.value,
                        moved,
                        type_converter,
                    )?);
                }
                // These actions are ownership facts used by the planner. They
                // deliberately have no standalone RC operation to emit. A
                // retained parameter's callee acquires its own unit at entry,
                // so the caller only keeps its value alive across the call.
                ActionKind::CallBorrow
                | ActionKind::CallRetain
                | ActionKind::BorrowLoad
                | ActionKind::CallTransfer
                | ActionKind::StoreTransfer
                | ActionKind::ReturnTransfer
                | ActionKind::TailTransfer
                | ActionKind::IntoRawTransfer
                | ActionKind::FieldMove => {}
            }
        }
        if !field_moves.is_empty() {
            return Err(OwnershipPlanError::new(
                "moved field has no moving release of its owner",
            ));
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
    let ty = crate::native::ownership_plan::boxed_object_type(ctx, ty);
    // `anyref`/`intref` deliberately carry no nominal allocation identity.
    // Zero is a dispatch-only dynamic-size signal: header RTTI must resolve it
    // to an exact release function before deallocation. It is never a shallow
    // fallback and is not inferred from physical definitions.
    if tribute_ir::dialect::tribute_rt::Anyref::matches(ctx, ty)
        || tribute_ir::dialect::tribute_rt::Intref::matches(ctx, ty)
    {
        return Ok(0);
    }
    // An array's size depends on its length.
    if tribute_ir::dialect::adt::Array::matches(ctx, ty) {
        return Ok(0);
    }
    let layout = plan.allocation_layout_for_type(ctx, ty)?;
    // A cell's object holds its value as its only field.
    if let Some(cell) = heap::Cell::from_type_ref(ctx, layout) {
        let contents = type_converter.convert_type_or_identity(ctx, cell.contents(ctx));
        return Ok(u64::from(natural_layout(ctx, [contents]).total_size) + RC_HEADER_SIZE);
    }
    // Each variant of an enum has its own allocation size, so an enum value's
    // size is also resolved by its header RTTI. So is a closure's: a value of
    // the closure base may be any capture layout, and the native layouts are
    // chosen only after this step.
    if adt::Enum::from_type_ref(ctx, layout).is_some()
        || crate::closure_lower::is_closure_layout(ctx, layout)
    {
        return Ok(0);
    }
    let payload_size = compute_struct_layout(ctx, layout, type_converter)
        .map(|layout| layout.total_size)
        .ok_or_else(|| OwnershipPlanError::new("planned release layout has no native size"))?;
    Ok(u64::from(payload_size) + RC_HEADER_SIZE)
}

/// Schedule the release of `owner` that moves the units of the fields its
/// `moved` projections read. The unique path releases the owner's other
/// managed fields at their offsets in the exact struct layout.
fn release_moving(
    ctx: &IrContext,
    plan: &NativeOwnershipPlan,
    placement: Placement,
    owner: ValueRef,
    moved: Vec<(ValueRef, u32)>,
    type_converter: &TypeConverter,
) -> Result<ScheduledAction, RcMaterializationError> {
    let owner_ty = ctx.value_ty(owner);
    let layout = plan.allocation_layout_for_type(ctx, owner_ty)?;
    let fields = get_struct_fields(ctx, layout)
        .filter(|_| adt::Struct::from_type_ref(ctx, layout).is_some())
        .ok_or_else(|| OwnershipPlanError::new("moving release owner is not an exact struct"))?;
    let offsets = compute_struct_layout(ctx, layout, type_converter)
        .ok_or_else(|| OwnershipPlanError::new("moving release owner has no native layout"))?
        .field_offsets;
    for &(projection, field) in &moved {
        let trunk_ir::ValueDef::OpResult(op, _) = ctx.value_def(projection) else {
            return Err(OwnershipPlanError::new("moved field is not a projection"));
        };
        let get = adt::StructGet::from_op(ctx, op)
            .map_err(|_| OwnershipPlanError::new("moved field is not an adt.struct_get"))?;
        if get.field(ctx) != field || alias_root(ctx, get.r#ref(ctx)) != owner {
            return Err(OwnershipPlanError::new(
                "moved field does not read its planned owner and field",
            ));
        }
    }
    let alloc_size = allocation_size_for_type(ctx, plan, owner_ty, type_converter)?;
    if alloc_size == 0 {
        return Err(OwnershipPlanError::new(
            "moving release owner has no static allocation size",
        ));
    }
    let release_offsets = fields
        .iter()
        .zip(&offsets)
        .enumerate()
        .filter(|&(index, (ty, _))| {
            plan.is_managed_type(ctx, *ty)
                && !moved.iter().any(|&(_, field)| field as usize == index)
        })
        .map(|(_, (_, &offset))| offset as i32)
        .collect();
    Ok(ScheduledAction::ReleaseMoving {
        placement,
        owner,
        moved: moved
            .into_iter()
            .map(|(projection, _)| projection)
            .collect(),
        alloc_size,
        release_offsets,
    })
}

/// The owner a moved projection reads, through exact alias casts.
fn projection_owner(
    ctx: &IrContext,
    projection: ValueRef,
) -> Result<ValueRef, RcMaterializationError> {
    let trunk_ir::ValueDef::OpResult(op, _) = ctx.value_def(projection) else {
        return Err(OwnershipPlanError::new("moved field is not a projection"));
    };
    let get = adt::StructGet::from_op(ctx, op)
        .map_err(|_| OwnershipPlanError::new("moved field is not an adt.struct_get"))?;
    Ok(alias_root(ctx, get.r#ref(ctx)))
}

/// The value a chain of forwarding casts ([`OperandRole::Forward`]) starts
/// from.
fn alias_root(ctx: &IrContext, mut value: ValueRef) -> ValueRef {
    while let trunk_ir::ValueDef::OpResult(op, _) = ctx.value_def(value)
        && DynOperandOwnership::role(ctx, op, 0) == Some(OperandRole::Forward)
    {
        value = ctx.op_operands(op)[0];
    }
    value
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

/// The assigning operation a replaced-field release is anchored before, and
/// the type of the field it assigns in `object` at `destination`.
fn validate_replaced_field(
    ctx: &IrContext,
    anchor: ActionAnchor,
    object: ValueRef,
    destination: u32,
    plan: &NativeOwnershipPlan,
) -> Result<(OpRef, TypeRef), RcMaterializationError> {
    let ActionAnchor::Before(op) = anchor else {
        return Err(OwnershipPlanError::new(
            "replaced-field release must be anchored before its assignment",
        ));
    };
    let roles = DynOperandOwnership::get(ctx, op)
        .ok_or_else(|| OwnershipPlanError::new("replaced-field release anchor assigns nothing"))?
        .operand_roles(ctx)?;
    let operands = ctx.op_operands(op);
    let field_ty = roles
        .into_iter()
        .find_map(|role| match role {
            OperandRole::Assign {
                target,
                slot,
                field,
                ..
            } if operands.get(target) == Some(&object) && slot == destination => Some(field),
            _ => None,
        })
        .ok_or_else(|| {
            OwnershipPlanError::new(
                "replaced-field release does not match its planned object or field",
            )
        })?;
    if !plan.is_managed_type(ctx, field_ty) {
        return Err(OwnershipPlanError::new(
            "replaced-field release targets an unmanaged field",
        ));
    }
    Ok((op, field_ty))
}

fn build_operations(ctx: &mut IrContext, action: ScheduledAction) -> (Placement, Vec<OpRef>) {
    match action {
        ScheduledAction::Retain { placement, value } => {
            let ty = ctx.value_ty(value);
            let retain = tribute_rt::Retain::operands(value)
                .results(ty)
                .build(ctx, location_for(ctx, placement));
            (placement, vec![retain.op_ref()])
        }
        ScheduledAction::Release {
            placement,
            value,
            alloc_size,
        } => {
            let release = tribute_rt::Release::operands(value)
                .alloc_size(alloc_size)
                .build(ctx, location_for(ctx, placement));
            (placement, vec![release.op_ref()])
        }
        ScheduledAction::ReleaseReplaced {
            placement,
            store,
            alloc_size,
        } => {
            let location = location_for(ctx, placement);
            let read = DynOperandOwnership::get(ctx, store)
                .and_then(|ownership| ownership.build_replaced_read(ctx, location))
                .expect("a validated assignment reads the value it replaces");
            let release = tribute_rt::Release::operands(ctx.op_result(read, 0))
                .alloc_size(alloc_size)
                .build(ctx, location);
            (placement, vec![read, release.op_ref()])
        }
        ScheduledAction::ReleaseMoving {
            placement,
            owner,
            moved,
            alloc_size,
            release_offsets,
        } => {
            let release = tribute_rt::ReleaseMoving::operands(owner, moved)
                .alloc_size(alloc_size)
                .release_offsets(release_offsets)
                .build(ctx, location_for(ctx, placement));
            (placement, vec![release.op_ref()])
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
