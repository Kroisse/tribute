//! Heap dialect — objects and references to them.
//!
//! `adt` operates on values; `heap` operates on the objects that hold them.
//! Both targets share it, and it assumes no reclamation scheme. See
//! `design/tribute-dialects.md` § "`heap`".

use crate::dialect::tribute_rtti::{
    DescriptorFieldsError, DescriptorLayoutType, ManagedRefType, OperandOwnership,
    OperandOwnershipError, OperandRole,
};
use trunk_ir::context::IrContext;
use trunk_ir::ops::DialectType;
use trunk_ir::refs::{OpRef, TypeRef};
use trunk_ir::types::Location;

#[trunk_ir::dialect]
mod heap {
    /// A managed reference to a mutable object holding one value of type
    /// `Contents`, like ML's `ref`.
    ///
    /// - **Meaning:** every reference to the same cell sees the value the
    ///   last `heap.cell_set` wrote. A cell is always an object, so
    ///   representation selection never unboxes it.
    /// - **Contents:** the storage representation of the held value. A
    ///   managed value is one unit the cell owns.
    /// - **Descriptor:** a cell allocation's descriptor is the cell type
    ///   without a tag, whose only field is `Contents`.
    /// - **Lowering:** each target stores the value as the only field of an
    ///   object, as it stores a field of a one-field struct.
    ///
    /// ```text
    /// heap.cell<core.i1>
    /// ```
    #[implements(ManagedRefType, DescriptorLayoutType)]
    struct Cell<Contents>;

    /// A new cell holding `value`.
    ///
    /// - **Ownership:** a managed `value` moves its unit into the cell
    ///   (`Initialize`).
    /// - **Effects:** each `cell_new` allocates a distinct cell, so two of
    ///   them are never merged; the operation is not `Pure`.
    ///
    /// ```text
    /// %c = heap.cell_new %v : heap.cell<core.i1>
    /// ```
    #[implements(OperandOwnership)]
    #[asm_format(value attr_dict_bare ":" type(result))]
    fn cell_new<C: Cell>(value: Value<C::Contents>) -> Value<C> {}

    /// The value `cell` holds now.
    ///
    /// - **Ownership:** the cell is borrowed for the read. A managed result
    ///   is never a borrow of the cell: it acquires its own unit right after
    ///   the read, because a later `heap.cell_set` reached through another
    ///   reference or a called function would release the value it replaces
    ///   (`Copy`; `design/rc.md` § "3. RC optimization pass").
    /// - **Effects:** a `heap.cell_set` between two reads may change the
    ///   value, so the operation is not `Pure`.
    ///
    /// ```text
    /// %v = heap.cell_get %c : core.i1
    /// ```
    #[implements(OperandOwnership)]
    #[asm_format(cell attr_dict_bare ":" type(result))]
    fn cell_get<C: Cell>(cell: Value<C>) -> Value<C::Contents> {}

    /// Replace the value `cell` holds with `value`.
    ///
    /// - **Ownership:** the cell is borrowed for the write. A managed
    ///   `value` moves its unit into the cell, and the unit of the replaced
    ///   value is released (`Assign`). A parameter written through is
    ///   retained, not borrowed (`design/rc.md` § "Borrowed parameter
    ///   elision").
    ///
    /// ```text
    /// heap.cell_set %c, %v
    /// ```
    #[implements(OperandOwnership)]
    #[asm_format(cell "," value attr_dict_bare)]
    fn cell_set<C: Cell>(cell: Value<C>, value: Value<C::Contents>) {}
}

impl ManagedRefType for Cell {}

impl DescriptorLayoutType for Cell {
    fn descriptor_fields<'a>(
        &self,
        ctx: &'a IrContext,
        tag: Option<u32>,
    ) -> Result<&'a [TypeRef], DescriptorFieldsError> {
        match tag {
            // A cell's only parameter is the type of the value it holds.
            None => Ok(&ctx.get_type(self.as_type_ref()).params),
            Some(_) => Err(DescriptorFieldsError::UnexpectedTag {
                layout: "heap.cell",
            }),
        }
    }
}

/// The value type of the cell type `cell`.
fn contents_of(ctx: &IrContext, cell: TypeRef) -> Result<TypeRef, OperandOwnershipError> {
    Cell::from_type_ref(ctx, cell)
        .map(|cell| cell.contents(ctx))
        .ok_or(OperandOwnershipError::new(
            "cell operation has a non-cell type",
        ))
}

impl OperandOwnership for CellNew {
    fn operand_roles(&self, ctx: &IrContext) -> Result<Vec<OperandRole>, OperandOwnershipError> {
        let field = contents_of(ctx, ctx.value_ty(self.result(ctx)))?;
        Ok(vec![OperandRole::Initialize { field }])
    }
}

impl OperandOwnership for CellGet {
    fn operand_roles(&self, ctx: &IrContext) -> Result<Vec<OperandRole>, OperandOwnershipError> {
        let object = ctx.value_ty(self.cell(ctx));
        let field = contents_of(ctx, object)?;
        Ok(vec![OperandRole::Copy { object, field }])
    }
}

impl OperandOwnership for CellSet {
    fn operand_roles(&self, ctx: &IrContext) -> Result<Vec<OperandRole>, OperandOwnershipError> {
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

#[cfg(test)]
mod tests;
