use trunk_ir::SymbolPath;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::core;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{OpRef, TypeRef};
use trunk_ir::types::{Attribute, AttributeMap};

/// The root-module dictionary of prelude type identities, keyed by
/// canonical declaration path (`new-plans/ir.md`).
pub const PRELUDE_TYPES_ATTR: &str = "tribute.types";

const STRING_PATH: &str = "std::String";

/// The prelude type published under the canonical path `path`.
pub fn prelude_type(ctx: &IrContext, module: OpRef, path: &str) -> Option<TypeRef> {
    match ctx.op_attrs(module).get(PRELUDE_TYPES_ATTR)? {
        Attribute::Dict(types) => types.view(ctx).get_type(path),
        _ => None,
    }
}

/// Publish `ty` as the prelude type of the canonical path `path`.
pub fn publish_prelude_type(ctx: &mut IrContext, module: OpRef, path: &str, ty: TypeRef) {
    let mut types = match ctx.op_attrs(module).get(PRELUDE_TYPES_ATTR) {
        Some(Attribute::Dict(types)) => types.clone(),
        _ => AttributeMap::new(),
    };
    types
        .view_mut(ctx)
        .insert(path.to_owned(), Attribute::Type(ty));
    ctx.op_attrs_mut(module)
        .insert(PRELUDE_TYPES_ATTR, Attribute::Dict(types));
}

/// The root-module reference to the function the root package of an
/// executable build declares as its entry point (`new-plans/ir.md`).
pub const ROOT_ENTRY_ATTR: &str = "tribute.entry";

/// The entry function the root module `module` names, if any.
pub fn root_entry(ctx: &IrContext, module: OpRef) -> Option<&SymbolPath> {
    ctx.op_attrs(module).get_symbol_ref(ROOT_ENTRY_ATTR)
}

/// Name `entry` as the entry function of the root module `module`.
pub fn set_root_entry(ctx: &mut IrContext, module: OpRef, entry: SymbolPath) {
    ctx.op_attrs_mut(module)
        .insert(ROOT_ENTRY_ATTR, Attribute::SymbolRef(entry));
}

/// Remove the entry reference of the root module `module`, once the root
/// entry bridge has consumed it.
pub fn take_root_entry(ctx: &mut IrContext, module: OpRef) -> Option<SymbolPath> {
    match ctx.op_attrs_mut(module).remove(ROOT_ENTRY_ATTR) {
        Some(Attribute::SymbolRef(entry)) => Some(entry),
        _ => None,
    }
}

/// Whether `op` is the definition its root module names as the entry
/// function.
pub fn is_root_entry(ctx: &IrContext, op: OpRef) -> bool {
    let Some(module) = ctx
        .op(op)
        .parent_block
        .and_then(|block| ctx.block(block).parent_region)
        .and_then(|region| ctx.region(region).parent_op)
    else {
        return false;
    };
    let is_root = core::Module::matches(ctx, module) && ctx.op(module).parent_block.is_none();
    is_root
        && root_entry(ctx, module)
            .and_then(SymbolPath::as_simple)
            .zip(ctx.op_attrs(op).get_str("sym_name"))
            .is_some_and(|(entry, name)| *entry == *name)
}

/// Tribute semantic type identities attached to a root IR module.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WellKnownTypes {
    pub string: Option<TypeRef>,
}

impl WellKnownTypes {
    pub fn from_module(ctx: &IrContext, module: OpRef) -> Self {
        Self {
            string: prelude_type(ctx, module, STRING_PATH),
        }
    }

    pub fn attach(self, ctx: &mut IrContext, module: OpRef) {
        if let Some(string) = self.string {
            publish_prelude_type(ctx, module, STRING_PATH, string);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::types::TypeDataBuilder;

    #[test]
    fn publishes_entries_in_one_dictionary() {
        let mut ctx = IrContext::new();
        let module = trunk_ir::parser::parse_test_module(
            &mut ctx,
            r#"core.module @main {tribute.types = {"std::runtime::Cell" = core.i64}} {
}"#,
        );
        let i32_ty = TypeDataBuilder::new("core.i32").intern(&mut ctx);
        let i64_ty = TypeDataBuilder::new("core.i64").intern(&mut ctx);
        WellKnownTypes {
            string: Some(i32_ty),
        }
        .attach(&mut ctx, module.op());

        assert_eq!(
            WellKnownTypes::from_module(&ctx, module.op()).string,
            Some(i32_ty)
        );
        assert_eq!(
            prelude_type(&ctx, module.op(), "std::runtime::Cell"),
            Some(i64_ty)
        );
        assert_eq!(prelude_type(&ctx, module.op(), "std::Option"), None);
        assert_eq!(ctx.op_attrs(module.op()).len(), 2);
    }
}
