//! Assertions shared by this crate's unit tests.

use trunk_ir::{IrContext, Module};

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
