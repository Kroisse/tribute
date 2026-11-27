//! Operand ownership roles of the `heap` operations.

use trunk_ir::context::IrContext;
use trunk_ir::ops::DialectType;
use trunk_ir::types::Location;
use trunk_ir::{OpRef, TypeRef};

use super::{BoxValue, Cast, Cell, CellGet, CellNew, CellSet, IsNull, UnboxValue};
use crate::dialect::tribute_rtti::{OperandOwnership, OperandOwnershipError, OperandRole};

type Result<T, E = OperandOwnershipError> = std::result::Result<T, E>;

// Until representation selection records the struct types it leaves
// unboxed, every `adt` value is the object its box refers to, so both
// boundary conversions are the boxed case: the result shares the operand's
// unit.

impl OperandOwnership for BoxValue {
    fn operand_roles(&self, _ctx: &IrContext) -> Result<Vec<OperandRole>> {
        Ok(vec![OperandRole::Forward])
    }
}

impl OperandOwnership for UnboxValue {
    fn operand_roles(&self, _ctx: &IrContext) -> Result<Vec<OperandRole>> {
        Ok(vec![OperandRole::Forward])
    }
}

impl OperandOwnership for IsNull {
    fn operand_roles(&self, _ctx: &IrContext) -> Result<Vec<OperandRole>> {
        Ok(vec![OperandRole::Borrow])
    }
}

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
