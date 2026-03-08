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
use trunk_ir::context::IrContext;
use trunk_ir::dialect::cont as arena_cont;
use trunk_ir::refs::{BlockRef, RegionRef, TypeRef};
use trunk_ir::rewrite::{ConversionTarget, Module, PatternApplicator, TypeConverter};
use trunk_ir::types::TypeDataBuilder;

use ffi::ensure_libmprompt_ffi;
use handler_dispatch::LowerHandlerDispatchPattern;
use patterns::{LowerDropPattern, LowerResumePattern, LowerShiftPattern};
use push_prompt::LowerPushPromptPattern;

/// Lower cont dialect operations to libmprompt-based FFI calls.
///
/// This is the main entry point for the native backend continuation lowering.
pub fn lower_cont_to_libmprompt(ctx: &mut IrContext, module: Module) {
    // Step 1: Ensure FFI function declarations are present
    ensure_libmprompt_ffi(ctx, module);

    // Step 2: Apply lowering patterns with type conversion
    // Pre-intern types before the closure (closure receives &IrContext, not &mut)
    let prompt_tag_ty = arena_cont::prompt_tag(ctx).as_type_ref();
    let i32_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());

    let mut type_converter = TypeConverter::new();
    type_converter.add_conversion(move |_ctx, ty| {
        if ty == prompt_tag_ty {
            Some(i32_ty)
        } else {
            None
        }
    });

    let applicator = PatternApplicator::new(type_converter)
        .with_auto_type_conversion(true)
        .add_pattern(LowerShiftPattern)
        .add_pattern(LowerResumePattern)
        .add_pattern(LowerDropPattern)
        .add_pattern(LowerHandlerDispatchPattern)
        .add_pattern(LowerPushPromptPattern::new());

    applicator.apply_partial(ctx, module);

    // Step 3: Verify all cont.* ops are converted
    let mut target = ConversionTarget::new();
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

    // Step 4: Convert remaining prompt_tag types to core.i32
    //
    // auto_type_conversion handles block argument types and operand casts
    // during pattern application, but does NOT convert operation result
    // types. Additionally, patterns create new ops (outlined functions,
    // scf.loop regions) whose block args may retain the old type.
    // This post-pass ensures all prompt_tag types are fully converted.
    if let Some(body) = module.body(ctx) {
        convert_types_in_region(ctx, body, prompt_tag_ty, i32_ty);
    }
}

/// Recursively walk a region and convert types from `from` to `to`.
///
/// Converts both operation result types and block argument types.
/// `auto_type_conversion` handles block args and operand casts during
/// pattern application, but newly created ops (e.g., outlined functions,
/// scf.loop regions) may have block args or result types that still use
/// the old type. This post-pass ensures complete conversion.
fn convert_types_in_region(ctx: &mut IrContext, region: RegionRef, from: TypeRef, to: TypeRef) {
    let blocks: Vec<BlockRef> = ctx.region(region).blocks.to_vec();
    for block in blocks {
        // Convert block argument types
        let num_args = ctx.block(block).args.len();
        for i in 0..num_args {
            if ctx.block(block).args[i].ty == from {
                ctx.set_block_arg_type(block, i as u32, to);
            }
        }

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
                convert_types_in_region(ctx, r, from, to);
            }
        }
    }
}
