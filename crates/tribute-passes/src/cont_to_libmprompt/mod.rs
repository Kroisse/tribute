//! Lower cont dialect operations to libmprompt-based FFI calls.
//!
//! This pass transforms continuation operations for the native (Cranelift) backend
//! using libmprompt for real stack-switching delimited continuations:
//!
//! - `cont.shift` → `func.call @__tribute_yield`
//! - `cont.resume` → `func.call @__tribute_resume`
//! - `cont.drop` → `func.call @__tribute_resume_drop`
//! - `cont.push_prompt` → body outlining + `func.call @__tribute_prompt`
//! - `cont.handler_dispatch` → `scf.loop` with TLS-based yield dispatch
//!
//! Unlike the trampoline backend (`cont_to_trampoline`), libmprompt provides
//! real stack switching, so:
//! - No Step type wrapping/unwrapping
//! - No effectful function analysis
//! - No return type changes
//! - Shift reaches the correct prompt directly (no tag matching)

pub(crate) mod ffi;
pub(crate) mod handler_dispatch;
pub(crate) mod patterns;
pub(crate) mod push_prompt;

#[cfg(test)]
mod tests;

use std::cell::RefCell;
use std::rc::Rc;

use trunk_ir::DialectType;
use trunk_ir::dialect::core::Module;
use trunk_ir::dialect::{cont, core};
use trunk_ir::rewrite::{ConversionError, ConversionTarget, PatternApplicator, TypeConverter};

use ffi::ensure_libmprompt_ffi;
use handler_dispatch::LowerHandlerDispatchPattern;
use patterns::{LowerDropPattern, LowerResumePattern, LowerShiftPattern};
use push_prompt::{LowerPushPromptPattern, add_outlined_bodies};

/// Lower cont dialect operations to libmprompt-based FFI calls.
///
/// This is the main entry point for the native backend continuation lowering.
///
/// Returns an error if any `cont.*` operations (except `cont.drop`, `cont.done`,
/// `cont.suspend`) remain after conversion.
pub fn lower_cont_to_libmprompt<'db>(
    db: &'db dyn salsa::Database,
    module: Module<'db>,
) -> Result<Module<'db>, ConversionError> {
    // Step 1: Ensure FFI function declarations are present
    let module = ensure_libmprompt_ffi(db, module);

    // Step 2: Apply lowering patterns
    let outlined_bodies = Rc::new(RefCell::new(Vec::new()));
    let body_counter = Rc::new(RefCell::new(0u32));

    let type_converter = TypeConverter::new().add_conversion(|db, ty| {
        cont::PromptTag::from_type(db, ty).map(|_| core::I32::new(db).as_type())
    });

    let applicator = PatternApplicator::new(type_converter)
        .add_pattern(LowerShiftPattern)
        .add_pattern(LowerResumePattern)
        .add_pattern(LowerDropPattern)
        .add_pattern(LowerHandlerDispatchPattern)
        .add_pattern(LowerPushPromptPattern {
            outlined_bodies: Rc::clone(&outlined_bodies),
            body_counter: Rc::clone(&body_counter),
        });

    let empty_target = ConversionTarget::new();
    let result = applicator.apply_partial(db, module, empty_target);

    // Step 3: Add outlined body functions to the module
    let module = add_outlined_bodies(db, result.module, &outlined_bodies.borrow());

    // Step 4: Verify all cont.* ops are converted
    // cont.done and cont.suspend are child ops consumed by handler_dispatch
    let target = ConversionTarget::new()
        .illegal_dialect("cont")
        .legal_op("cont", "drop")
        .legal_op("cont", "done")
        .legal_op("cont", "suspend");

    target.verify(db, &module)?;
    Ok(module)
}
