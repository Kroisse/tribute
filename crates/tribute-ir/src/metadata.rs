use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::refs::{OpRef, TypeRef};
use trunk_ir::types::Attribute;

const STRING_TYPE_ATTR: &str = "tribute.type.string";
const LIST_TYPE_ATTR: &str = "tribute.type.list";

/// Tribute semantic type identities attached to a root IR module.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WellKnownTypes {
    pub string: Option<TypeRef>,
    pub list: Option<TypeRef>,
}

impl WellKnownTypes {
    pub fn from_module(ctx: &IrContext, module: OpRef) -> Self {
        Self {
            string: ctx.op(module).attributes.get_type(STRING_TYPE_ATTR),
            list: ctx.op(module).attributes.get_type(LIST_TYPE_ATTR),
        }
    }

    pub fn attach(self, ctx: &mut IrContext, module: OpRef) {
        if let Some(string) = self.string {
            ctx.op_mut(module)
                .attributes
                .insert(Symbol::new(STRING_TYPE_ATTR), Attribute::Type(string));
        }
        if let Some(list) = self.list {
            ctx.op_mut(module)
                .attributes
                .insert(Symbol::new(LIST_TYPE_ATTR), Attribute::Type(list));
        }
    }
}
