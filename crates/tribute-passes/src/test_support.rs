//! Assertions and fixtures shared by this crate's unit tests.

use trunk_ir::{IrContext, Module, TypeRef};

/// Publish a field-less evidence marker in `module`'s `tribute.types`, as
/// the frontend publishes the prelude's, and return the evidence type.
pub(crate) fn publish_evidence_marker(ctx: &mut IrContext, module: Module) -> TypeRef {
    let path =
        tribute_ir::runtime_items::runtime_symbol(tribute_ir::runtime_items::evidence::MARKER);
    let marker = tribute_ir::dialect::adt::Struct::from_fields(
        ctx,
        path.to_string(),
        [],
        trunk_ir::types::AttributeMap::new(),
    )
    .as_type_ref();
    tribute_ir::metadata::publish_prelude_type(ctx, module.op(), path.as_str(), marker);
    tribute_ir::dialect::ability::evidence_type(ctx, module)
}

/// Parse the module `text` and publish the evidence marker in it, as the
/// frontend does for every module.
pub(crate) fn parse_with_evidence_marker(ctx: &mut IrContext, text: &str) -> Module {
    let module = trunk_ir::parser::parse_test_module(ctx, text);
    publish_evidence_marker(ctx, module);
    module
}

/// Run `transform` on `module`, require it to fail, and require the printed
/// module to be unchanged by the failed attempt. Returns the error so callers
/// can assert on its contents.
#[track_caller]
pub(crate) fn assert_unchanged_on_error<T, E>(
    ctx: &mut IrContext,
    module: Module,
    transform: impl FnOnce(&mut IrContext, Module) -> Result<T, E>,
) -> E {
    let before = module.op().display(ctx).to_string();
    let Err(error) = transform(ctx, module) else {
        panic!("expected the transform to fail on:\n{before}");
    };
    assert_eq!(
        module.op().display(ctx).to_string(),
        before,
        "a failed transform must leave the IR unchanged"
    );
    error
}
