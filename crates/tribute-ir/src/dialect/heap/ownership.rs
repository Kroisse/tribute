//! Operand ownership roles of the `heap` operations.

use trunk_ir::context::IrContext;
use trunk_ir::ops::DialectType;
use trunk_ir::types::Location;
use trunk_ir::{OpRef, TypeRef, ValueRef};

use super::{
    ArrayGet, ArrayLen, ArrayNew, ArrayOf, ArraySet, ArraySplice, Box, Cast, Cell, CellGet,
    CellNew, CellSet, IsNull, StructGet, StructNew, StructUpdate, VariantGet, VariantIs,
    VariantNew, VariantTag,
};
use crate::dialect::adt::Array;
use crate::dialect::tribute_rtti::{OperandOwnership, OperandOwnershipError, OperandRole};

type Result<T, E = OperandOwnershipError> = std::result::Result<T, E>;

/// Every operand of `op` is borrowed.
fn borrow_all(ctx: &IrContext, op: OpRef) -> Result<Vec<OperandRole>> {
    Ok(vec![OperandRole::Borrow; ctx.op_operands(op).len()])
}

/// The box type of `value`.
fn box_of(ctx: &IrContext, value: ValueRef) -> Result<Box> {
    Box::from_type_ref(ctx, ctx.value_ty(value)).ok_or(OperandOwnershipError::new(
        "heap operation has a non-box type",
    ))
}

/// The element type of the array objects of the box `value`.
fn element_of(ctx: &IrContext, value: ValueRef) -> Result<TypeRef> {
    Array::from_type_ref(ctx, box_of(ctx, value)?.layout(ctx))
        .map(|array| array.element(ctx))
        .ok_or(OperandOwnershipError::new(
            "array operation has a non-array box",
        ))
}

/// The operands of `op` from position `first` on initialize one field of
/// type `field(index)` each; the operands before them are borrowed.
fn initialize_from(
    ctx: &IrContext,
    op: OpRef,
    first: usize,
    field: impl Fn(u32) -> Result<TypeRef>,
) -> Result<Vec<OperandRole>> {
    let count = ctx.op_operands(op).len().saturating_sub(first);
    std::iter::repeat_n(Ok(OperandRole::Borrow), first)
        .chain(
            (0u32..)
                .take(count)
                .map(|index| field(index).map(|field| OperandRole::Initialize { field })),
        )
        .collect()
}

fn stale(_: super::BoxError) -> OperandOwnershipError {
    OperandOwnershipError::new("heap operation does not fit its layout")
}

impl OperandOwnership for StructNew {
    fn operand_roles(&self, ctx: &IrContext) -> Result<Vec<OperandRole>> {
        let boxed = box_of(ctx, self.result(ctx))?;
        initialize_from(ctx, self.op_ref(), 0, |field| {
            super::struct_field(ctx, boxed, field).map_err(stale)
        })
    }
}

impl OperandOwnership for VariantNew {
    fn operand_roles(&self, ctx: &IrContext) -> Result<Vec<OperandRole>> {
        let boxed = box_of(ctx, self.result(ctx))?;
        let tag = self.tag(ctx);
        initialize_from(ctx, self.op_ref(), 0, |field| {
            super::variant_field(ctx, boxed, tag, field).map_err(stale)
        })
    }
}

impl OperandOwnership for StructGet {
    fn operand_roles(&self, ctx: &IrContext) -> Result<Vec<OperandRole>> {
        let boxed = box_of(ctx, self.r#ref(ctx))?;
        let field = super::struct_field(ctx, boxed, self.field(ctx)).map_err(stale)?;
        let object = boxed.layout(ctx);
        Ok(vec![OperandRole::Project { object, field }])
    }
}

impl OperandOwnership for VariantGet {
    fn operand_roles(&self, ctx: &IrContext) -> Result<Vec<OperandRole>> {
        let boxed = box_of(ctx, self.r#ref(ctx))?;
        let field =
            super::variant_field(ctx, boxed, self.tag(ctx), self.field(ctx)).map_err(stale)?;
        let object = boxed.layout(ctx);
        Ok(vec![OperandRole::Project { object, field }])
    }
}

impl OperandOwnership for StructUpdate {
    fn operand_roles(&self, ctx: &IrContext) -> Result<Vec<OperandRole>> {
        let boxed = box_of(ctx, self.source(ctx))?;
        let field = super::struct_field(ctx, boxed, self.field(ctx)).map_err(stale)?;
        Ok(vec![
            OperandRole::Consume,
            OperandRole::Initialize { field },
        ])
    }
}

impl OperandOwnership for ArrayOf {
    fn operand_roles(&self, ctx: &IrContext) -> Result<Vec<OperandRole>> {
        let element = element_of(ctx, self.result(ctx))?;
        initialize_from(ctx, self.op_ref(), 0, |_| Ok(element))
    }
}

impl OperandOwnership for ArraySplice {
    fn operand_roles(&self, ctx: &IrContext) -> Result<Vec<OperandRole>> {
        // `source`, `start`, and `end` come before the new elements.
        const FIRST_ELEMENT: usize = 3;
        let element = element_of(ctx, self.result(ctx))?;
        initialize_from(ctx, self.op_ref(), FIRST_ELEMENT, |_| Ok(element))
    }
}

impl OperandOwnership for ArrayGet {
    fn operand_roles(&self, ctx: &IrContext) -> Result<Vec<OperandRole>> {
        let object = box_of(ctx, self.r#ref(ctx))?.layout(ctx);
        let field = element_of(ctx, self.r#ref(ctx))?;
        Ok(vec![
            OperandRole::Project { object, field },
            OperandRole::Borrow,
        ])
    }
}

impl OperandOwnership for ArraySet {
    fn operand_roles(&self, ctx: &IrContext) -> Result<Vec<OperandRole>> {
        let field = element_of(ctx, self.source(ctx))?;
        Ok(vec![
            OperandRole::Consume,
            OperandRole::Borrow,
            OperandRole::Initialize { field },
        ])
    }
}

/// Implement [`OperandOwnership`] for operations that borrow every operand.
macro_rules! borrows_every_operand {
    ($($op:ident),+) => {$(
        impl OperandOwnership for $op {
            fn operand_roles(&self, ctx: &IrContext) -> Result<Vec<OperandRole>> {
                borrow_all(ctx, self.op_ref())
            }
        }
    )+};
}

borrows_every_operand!(ArrayNew, ArrayLen, VariantIs, VariantTag, IsNull);

impl OperandOwnership for Cast {
    fn operand_roles(&self, _ctx: &IrContext) -> Result<Vec<OperandRole>> {
        Ok(vec![OperandRole::Forward])
    }
}

/// The value type of the cell type `cell`.
fn contents_of(ctx: &IrContext, cell: TypeRef) -> Result<TypeRef> {
    Cell::from_type_ref(ctx, cell)
        .map(|cell| cell.contents(ctx))
        .ok_or(OperandOwnershipError::new(
            "cell operation has a non-cell type",
        ))
}

impl OperandOwnership for CellNew {
    fn operand_roles(&self, ctx: &IrContext) -> Result<Vec<OperandRole>> {
        let field = contents_of(ctx, ctx.value_ty(self.result(ctx)))?;
        Ok(vec![OperandRole::Initialize { field }])
    }
}

impl OperandOwnership for CellGet {
    fn operand_roles(&self, ctx: &IrContext) -> Result<Vec<OperandRole>> {
        let object = ctx.value_ty(self.cell(ctx));
        let field = contents_of(ctx, object)?;
        Ok(vec![OperandRole::Copy { object, field }])
    }
}

impl OperandOwnership for CellSet {
    fn operand_roles(&self, ctx: &IrContext) -> Result<Vec<OperandRole>> {
        let object = ctx.value_ty(self.cell(ctx));
        let field = contents_of(ctx, object)?;
        Ok(vec![
            OperandRole::Borrow,
            OperandRole::Assign {
                target: 0,
                object,
                slot: 0,
                field,
            },
        ])
    }

    fn build_replaced_read(&self, ctx: &mut IrContext, location: Location) -> Option<OpRef> {
        let get = CellGet::operands(self.cell(ctx)).build(ctx, location);
        Some(get.op_ref())
    }
}
