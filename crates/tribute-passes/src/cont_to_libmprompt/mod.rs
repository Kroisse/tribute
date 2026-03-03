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

use trunk_ir::Symbol;
use trunk_ir::arena::context::IrContext;
use trunk_ir::arena::refs::{BlockRef, RegionRef, TypeRef};
use trunk_ir::arena::rewrite::{
    ArenaConversionTarget, ArenaModule, ArenaTypeConverter,
    PatternApplicator as ArenaPatternApplicator,
};
use trunk_ir::arena::types::TypeDataBuilder;

use ffi::ensure_libmprompt_ffi;
use handler_dispatch::LowerHandlerDispatchPattern;
use patterns::{LowerDropPattern, LowerResumePattern, LowerShiftPattern};
use push_prompt::LowerPushPromptPattern;

/// Lower cont dialect operations to libmprompt-based FFI calls.
///
/// This is the main entry point for the native backend continuation lowering.
pub fn lower_cont_to_libmprompt(ctx: &mut IrContext, module: ArenaModule) {
    // Step 1: Ensure FFI function declarations are present
    ensure_libmprompt_ffi(ctx, module);

    // Step 2: Apply lowering patterns with type conversion
    // Pre-intern types before the closure (closure receives &IrContext, not &mut)
    let prompt_tag_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("cont"), Symbol::new("prompt_tag")).build());
    let i32_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());

    let mut type_converter = ArenaTypeConverter::new();
    type_converter.add_conversion(move |_ctx, ty| {
        if ty == prompt_tag_ty {
            Some(i32_ty)
        } else {
            None
        }
    });

    let applicator = ArenaPatternApplicator::new(type_converter)
        .with_auto_type_conversion(true)
        .add_pattern(LowerShiftPattern)
        .add_pattern(LowerResumePattern)
        .add_pattern(LowerDropPattern)
        .add_pattern(LowerHandlerDispatchPattern)
        .add_pattern(LowerPushPromptPattern::new());

    applicator.apply_partial(ctx, module);

    // Step 3: Verify all cont.* ops are converted
    let mut target = ArenaConversionTarget::new();
    target.add_illegal_dialect("cont");
    target.add_legal_op("cont", "drop");
    target.add_legal_op("cont", "done");
    target.add_legal_op("cont", "suspend");

    if let Some(body) = module.body(ctx) {
        let illegal = target.verify(ctx, body);
        assert!(
            illegal.is_empty(),
            "lower_cont_to_libmprompt: unconverted cont.* ops remain: {:?}",
            illegal,
        );
    }

    // Step 4: Convert remaining prompt_tag result types to core.i32
    //
    // auto_type_conversion handles block argument types and operand casts,
    // but does NOT convert operation result types. Non-cont ops (e.g.,
    // arith.const, adt.struct_get) may still have prompt_tag result types.
    if let Some(body) = module.body(ctx) {
        convert_result_types_in_region(ctx, body, prompt_tag_ty, i32_ty);
    }
}

/// Recursively walk a region and convert operation result types from `from` to `to`.
///
/// Block argument types and operand casts are handled by `auto_type_conversion`
/// in the pattern applicator; this function only fixes up result types on
/// operations that were not matched by any pattern.
fn convert_result_types_in_region(
    ctx: &mut IrContext,
    region: RegionRef,
    from: TypeRef,
    to: TypeRef,
) {
    let blocks: Vec<BlockRef> = ctx.region(region).blocks.to_vec();
    for block in blocks {
        let ops: Vec<_> = ctx.block(block).ops.to_vec();
        for op in ops {
            // Convert result types
            let result_types: Vec<_> = ctx.op_result_types(op).to_vec();
            for (i, &ty) in result_types.iter().enumerate() {
                if ty == from {
                    ctx.set_op_result_type(op, i as u32, to);
                }
            }
            // Recurse into nested regions
            let regions: Vec<_> = ctx.op(op).regions.to_vec();
            for r in regions {
                convert_result_types_in_region(ctx, r, from, to);
            }
        }
    }
}
