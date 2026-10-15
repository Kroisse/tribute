use trunk_ir::context::IrContext;
use trunk_ir::refs::{OpRef, TypeRef};
use trunk_ir::types::Attribute;

const STRING_TYPE_ATTR: &str = "tribute.type.string";

/// Tribute semantic type identities attached to a root IR module.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WellKnownTypes {
    pub string: Option<TypeRef>,
}

impl WellKnownTypes {
    pub fn from_module(ctx: &IrContext, module: OpRef) -> Self {
        Self {
            string: ctx.op(module).attributes.get_type(ctx, STRING_TYPE_ATTR),
        }
    }

    pub fn attach(self, ctx: &mut IrContext, module: OpRef) {
        if let Some(string) = self.string {
            ctx.set_attr(module, STRING_TYPE_ATTR, Attribute::Type(string));
        }
    }
}
