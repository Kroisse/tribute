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

use trunk_ir::DialectType;
use trunk_ir::Symbol;
use trunk_ir::arena::context::IrContext;
use trunk_ir::arena::refs::{BlockRef, RegionRef, TypeRef};
use trunk_ir::arena::rewrite::{
    ArenaConversionTarget, ArenaModule, ArenaTypeConverter,
    PatternApplicator as ArenaPatternApplicator,
};
use trunk_ir::arena::types::TypeDataBuilder;
use trunk_ir::dialect::core::Module;
use trunk_ir::dialect::{cont, core};
use trunk_ir::rewrite::{ConversionError, ConversionTarget, PatternApplicator, TypeConverter};

use ffi::{ensure_libmprompt_ffi, ensure_libmprompt_ffi_arena};
use handler_dispatch::{ArenaLowerHandlerDispatchPattern, LowerHandlerDispatchPattern};
use patterns::{
    ArenaLowerDropPattern, ArenaLowerResumePattern, ArenaLowerShiftPattern, LowerDropPattern,
    LowerResumePattern, LowerShiftPattern,
};
use push_prompt::{ArenaLowerPushPromptPattern, LowerPushPromptPattern};

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
    // Outlined body functions are added via `rewriter.add_module_op()` during
    // pattern application, so no side-channel is needed.
    let type_converter = TypeConverter::new().add_conversion(|db, ty| {
        cont::PromptTag::from_type(db, ty).map(|_| core::I32::new(db).as_type())
    });

    let applicator = PatternApplicator::new(type_converter)
        .add_pattern(LowerShiftPattern)
        .add_pattern(LowerResumePattern)
        .add_pattern(LowerDropPattern)
        .add_pattern(LowerHandlerDispatchPattern)
        .add_pattern(LowerPushPromptPattern::new());

    let empty_target = ConversionTarget::new();
    let result = applicator.apply_partial(db, module, empty_target);
    let module = result.module;

    // Step 3: Verify all cont.* ops are converted
    // cont.done and cont.suspend are child ops consumed by handler_dispatch
    let target = ConversionTarget::new()
        .illegal_dialect("cont")
        .legal_op("cont", "drop")
        .legal_op("cont", "done")
        .legal_op("cont", "suspend");

    target.verify(db, &module)?;
    Ok(module)
}

/// Lower cont dialect operations to libmprompt-based FFI calls (arena version).
///
/// This is the arena-based entry point for native backend continuation lowering.
pub fn lower_cont_to_libmprompt_arena(ctx: &mut IrContext, module: ArenaModule) {
    // Step 1: Ensure FFI function declarations are present
    ensure_libmprompt_ffi_arena(ctx, module);

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
        .add_pattern(ArenaLowerShiftPattern)
        .add_pattern(ArenaLowerResumePattern)
        .add_pattern(ArenaLowerDropPattern)
        .add_pattern(ArenaLowerHandlerDispatchPattern)
        .add_pattern(ArenaLowerPushPromptPattern::new());

    applicator.apply_partial(ctx, module);

    // Step 3: Verify all cont.* ops are converted (matching Salsa path)
    let mut target = ArenaConversionTarget::new();
    target.add_illegal_dialect("cont");
    target.add_legal_op("cont", "drop");
    target.add_legal_op("cont", "done");
    target.add_legal_op("cont", "suspend");

    if let Some(body) = module.body(ctx) {
        let illegal = target.verify(ctx, body);
        assert!(
            illegal.is_empty(),
            "lower_cont_to_libmprompt_arena: unconverted cont.* ops remain: {:?}",
            illegal,
        );
    }

    // Step 4: Convert remaining cont.prompt_tag types to core.i32
    //
    // The arena PatternApplicator doesn't yet support automatic type conversion
    // (the Salsa version does this via virtual type mapping in RewriteContext).
    // As a workaround, we walk the module and convert types directly.
    if let Some(body) = module.body(ctx) {
        convert_types_in_region(ctx, body, prompt_tag_ty, i32_ty);
    }
}

/// Recursively walk a region and convert all values with `from` type to `to` type.
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

        // Walk ops
        let ops: Vec<_> = ctx.block(block).ops.to_vec();
        for op in ops {
            // Convert result types
            let num_results = ctx.op_result_types(op).len();
            for i in 0..num_results {
                if ctx.op_result_types(op)[i] == from {
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
