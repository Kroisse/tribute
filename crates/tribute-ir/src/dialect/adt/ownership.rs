//! Operand ownership roles of the `adt` operations.

use trunk_ir::context::IrContext;
use trunk_ir::ops::DialectType;
use trunk_ir::types::Location;
use trunk_ir::{OpRef, TypeRef};

use super::layout::get_struct_fields;
use super::{
    Array, ArrayGet, ArrayLen, ArrayNew, ArrayOf, ArraySet, ArraySplice, Enum, RefCast, RefIsNull,
    StructGet, StructNew, StructSet, VariantGet, VariantIs, VariantNew, VariantTag,
};
use crate::dialect::tribute_rtti::{OperandOwnership, OperandOwnershipError, OperandRole};

type Roles = Result<Vec<OperandRole>, OperandOwnershipError>;

/// Every operand of `op` is borrowed.
fn borrow_all(ctx: &IrContext, op: OpRef) -> Roles {
    Ok(vec![OperandRole::Borrow; ctx.op_operands(op).len()])
}

/// The operands of `op` from position `first` on initialize one field of
/// `fields` each; the operands before them are borrowed.
fn initialize_from(ctx: &IrContext, op: OpRef, first: usize, fields: &[TypeRef]) -> Roles {
    if ctx.op_operands(op).len() != first + fields.len() {
        return Err(OperandOwnershipError::new(
            "aggregate constructor arity is malformed",
        ));
    }
    Ok(std::iter::repeat_n(OperandRole::Borrow, first)
        .chain(
            fields
                .iter()
                .map(|&field| OperandRole::Initialize { field }),
        )
        .collect())
}

fn array_element(ctx: &IrContext, array: TypeRef) -> Result<TypeRef, OperandOwnershipError> {
    Array::from_type_ref(ctx, array)
        .map(|array| array.element(ctx))
        .ok_or(OperandOwnershipError::new(
            "array operation has a non-array type",
        ))
}

impl OperandOwnership for StructNew {
    fn operand_roles(&self, ctx: &IrContext) -> Roles {
        let fields = get_struct_fields(ctx, self.layout(ctx))
            .ok_or(OperandOwnershipError::new("struct_new has invalid layout"))?;
        initialize_from(ctx, self.op_ref(), 0, &fields)
    }
}

impl OperandOwnership for VariantNew {
    fn operand_roles(&self, ctx: &IrContext) -> Roles {
        let fields = Enum::from_type_ref(ctx, self.layout(ctx))
            .ok_or(OperandOwnershipError::new("variant_new has invalid layout"))?
            .variant_fields(ctx, self.variant(ctx))
            .ok_or(OperandOwnershipError::new("variant_new tag is stale"))?;
        initialize_from(ctx, self.op_ref(), 0, fields)
    }
}

impl OperandOwnership for ArrayOf {
    fn operand_roles(&self, ctx: &IrContext) -> Roles {
        let element = array_element(ctx, self.r#type(ctx))?;
        let count = ctx.op_operands(self.op_ref()).len();
        initialize_from(ctx, self.op_ref(), 0, &vec![element; count])
    }
}

impl OperandOwnership for ArraySplice {
    fn operand_roles(&self, ctx: &IrContext) -> Roles {
        // `source`, `start`, and `end` come before the new elements.
        const FIRST_ELEMENT: usize = 3;
        let element = array_element(ctx, self.r#type(ctx))?;
        let count = ctx
            .op_operands(self.op_ref())
            .len()
            .saturating_sub(FIRST_ELEMENT);
        initialize_from(ctx, self.op_ref(), FIRST_ELEMENT, &vec![element; count])
    }
}

impl OperandOwnership for ArrayNew {
    fn operand_roles(&self, ctx: &IrContext) -> Roles {
        borrow_all(ctx, self.op_ref())
    }
}

impl OperandOwnership for StructGet {
    fn operand_roles(&self, ctx: &IrContext) -> Roles {
        let object = self.r#type(ctx);
        let fields = get_struct_fields(ctx, object)
            .ok_or(OperandOwnershipError::new("struct_get has invalid layout"))?;
        let field = *fields
            .get(self.field(ctx) as usize)
            .ok_or(OperandOwnershipError::new("struct_get field is stale"))?;
        Ok(vec![OperandRole::Project { object, field }])
    }
}

impl OperandOwnership for VariantGet {
    fn operand_roles(&self, ctx: &IrContext) -> Roles {
        let object = self.r#type(ctx);
        let fields = Enum::from_type_ref(ctx, object)
            .ok_or(OperandOwnershipError::new("variant_get has invalid layout"))?
            .variant_fields(ctx, self.variant(ctx))
            .ok_or(OperandOwnershipError::new("variant_get tag is stale"))?;
        let field = *fields
            .get(self.field(ctx) as usize)
            .ok_or(OperandOwnershipError::new("variant_get field is stale"))?;
        Ok(vec![OperandRole::Project { object, field }])
    }
}

impl OperandOwnership for ArrayGet {
    fn operand_roles(&self, ctx: &IrContext) -> Roles {
        let object = ctx.value_ty(self.r#ref(ctx));
        let field = array_element(ctx, object)
            .map_err(|_| OperandOwnershipError::new("array_get reads a non-array value"))?;
        Ok(vec![
            OperandRole::Project { object, field },
            OperandRole::Borrow,
        ])
    }
}

impl OperandOwnership for StructSet {
    fn operand_roles(&self, ctx: &IrContext) -> Roles {
        let object = self.r#type(ctx);
        let fields = get_struct_fields(ctx, object)
            .ok_or(OperandOwnershipError::new("struct_set has invalid layout"))?;
        let slot = self.field(ctx);
        let field = *fields
            .get(slot as usize)
            .ok_or(OperandOwnershipError::new("struct_set field is stale"))?;
        Ok(vec![
            OperandRole::Borrow,
            OperandRole::Assign {
                target: 0,
                object,
                slot,
                field,
            },
        ])
    }

    fn build_replaced_read(&self, ctx: &mut IrContext, location: Location) -> Option<OpRef> {
        let (object, slot) = (self.r#type(ctx), self.field(ctx));
        let field = *get_struct_fields(ctx, object)?.get(slot as usize)?;
        let get = StructGet::operands(self.r#ref(ctx))
            .r#type(object)
            .field(slot)
            .results(field)
            .build(ctx, location);
        Some(get.op_ref())
    }
}

impl OperandOwnership for ArraySet {
    fn operand_roles(&self, ctx: &IrContext) -> Roles {
        let object = ctx.value_ty(self.r#ref(ctx));
        let field = array_element(ctx, object)?;
        Ok(vec![
            OperandRole::Borrow,
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
        let get = ArrayGet::operands(self.r#ref(ctx), self.index(ctx)).build(ctx, location);
        Some(get.op_ref())
    }
}

impl OperandOwnership for VariantIs {
    fn operand_roles(&self, ctx: &IrContext) -> Roles {
        borrow_all(ctx, self.op_ref())
    }
}

impl OperandOwnership for VariantTag {
    fn operand_roles(&self, ctx: &IrContext) -> Roles {
        borrow_all(ctx, self.op_ref())
    }
}

impl OperandOwnership for ArrayLen {
    fn operand_roles(&self, ctx: &IrContext) -> Roles {
        borrow_all(ctx, self.op_ref())
    }
}

impl OperandOwnership for RefIsNull {
    fn operand_roles(&self, ctx: &IrContext) -> Roles {
        borrow_all(ctx, self.op_ref())
    }
}

impl OperandOwnership for RefCast {
    fn operand_roles(&self, _ctx: &IrContext) -> Roles {
        Ok(vec![OperandRole::Forward])
    }
}
