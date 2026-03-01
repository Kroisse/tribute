//! Normalize `tribute_rt` primitive types to core/wasm types.
//!
//! This pass converts high-level `tribute_rt` primitive types to their
//! corresponding `core` and `wasm` types early in the WASM pipeline, ensuring
//! that downstream passes and the emit phase don't need to handle
//! tribute-specific types.
//!
//! ## Type Conversions
//!
//! | Source Type          | Target Type   |
//! |----------------------|---------------|
//! | `tribute_rt.int`     | `core.i32`    |
//! | `tribute_rt.nat`     | `core.i32`    |
//! | `tribute_rt.bool`    | `core.i32`    |
//! | `tribute_rt.float`   | `core.f64`    |
//! | `tribute_rt.any`     | `wasm.anyref` |
//! | `tribute_rt.intref`  | `wasm.i31ref` |
//!
//! ## What this pass normalizes
//!
//! - Function signatures (parameter and return types)
//! - Operation result types
//! - Block argument types
//!
//! The pass uses `wasm_type_converter()` which defines these conversions
//! and their materializations (boxing/unboxing operations).

use tracing::debug;
use trunk_ir::arena::context::IrContext;
use trunk_ir::arena::dialect::func as arena_func;
use trunk_ir::arena::ops::ArenaDialectOp;
use trunk_ir::arena::refs::{OpRef, TypeRef};
use trunk_ir::arena::rewrite::{
    ArenaModule, ArenaRewritePattern, FuncSignatureConversionPattern, PatternApplicator,
    PatternRewriter, WasmFuncSignatureConversionPattern,
};
use trunk_ir::ir::Symbol;

/// Normalize tribute_rt primitive types to core types.
///
/// This pass should run early in the WASM pipeline, before `trampoline_to_wasm`.
pub fn lower(ctx: &mut IrContext, module: ArenaModule) {
    let type_converter = crate::wasm::type_converter::wasm_type_converter(ctx);

    let applicator = PatternApplicator::new(type_converter)
        .add_pattern(FuncSignatureConversionPattern)
        .add_pattern(WasmFuncSignatureConversionPattern)
        .add_pattern(NormalizeCallPattern)
        .add_pattern(NormalizeCallIndirectPattern)
        .add_pattern(NormalizeOpResultPattern);
    applicator.apply_partial(ctx, module);
}

/// Check if a type is a tribute_rt primitive type or closure type.
fn is_type(ctx: &IrContext, ty: TypeRef, dialect: &'static str, name: &'static str) -> bool {
    let data = ctx.types.get(ty);
    data.dialect == Symbol::new(dialect) && data.name == Symbol::new(name)
}

/// Convert a primitive type to its core/wasm equivalent.
/// Returns None if the type doesn't need conversion.
fn convert_primitive_type(ctx: &mut IrContext, ty: TypeRef) -> Option<TypeRef> {
    // tribute_rt.int, tribute_rt.nat, tribute_rt.bool -> core.i32
    if is_type(ctx, ty, "tribute_rt", "int")
        || is_type(ctx, ty, "tribute_rt", "nat")
        || is_type(ctx, ty, "tribute_rt", "bool")
    {
        let i32_ty = ctx.types.intern(
            trunk_ir::arena::types::TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32"))
                .build(),
        );
        return Some(i32_ty);
    }

    // tribute_rt.float -> core.f64
    if is_type(ctx, ty, "tribute_rt", "float") {
        let f64_ty = ctx.types.intern(
            trunk_ir::arena::types::TypeDataBuilder::new(Symbol::new("core"), Symbol::new("f64"))
                .build(),
        );
        return Some(f64_ty);
    }

    // tribute_rt.any -> wasm.anyref
    if is_type(ctx, ty, "tribute_rt", "any") {
        let anyref_ty = ctx.types.intern(
            trunk_ir::arena::types::TypeDataBuilder::new(
                Symbol::new("wasm"),
                Symbol::new("anyref"),
            )
            .build(),
        );
        return Some(anyref_ty);
    }

    // tribute_rt.intref -> wasm.i31ref
    if is_type(ctx, ty, "tribute_rt", "intref") {
        let i31ref_ty = ctx.types.intern(
            trunk_ir::arena::types::TypeDataBuilder::new(
                Symbol::new("wasm"),
                Symbol::new("i31ref"),
            )
            .build(),
        );
        return Some(i31ref_ty);
    }

    // closure.closure -> adt.struct(name="_closure")
    if is_type(ctx, ty, "closure", "closure") {
        return Some(closure_adt_type(ctx));
    }

    None
}

/// Create the canonical Closure ADT type in arena IR.
fn closure_adt_type(ctx: &mut IrContext) -> TypeRef {
    let i32_ty = ctx.types.intern(
        trunk_ir::arena::types::TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32"))
            .build(),
    );
    let anyref_ty = ctx.types.intern(
        trunk_ir::arena::types::TypeDataBuilder::new(Symbol::new("wasm"), Symbol::new("anyref"))
            .build(),
    );

    ctx.types.intern(
        trunk_ir::arena::types::TypeDataBuilder::new(Symbol::new("adt"), Symbol::new("struct"))
            .param(i32_ty)
            .param(anyref_ty)
            .attr(
                "name",
                trunk_ir::arena::types::Attribute::Symbol(Symbol::new("_closure")),
            )
            .build(),
    )
}

// ============================================================================
// Patterns
// ============================================================================

/// Normalize func.call operation result types.
struct NormalizeCallPattern;

impl ArenaRewritePattern for NormalizeCallPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(call_op) = arena_func::Call::from_op(ctx, op) else {
            return false;
        };

        let result_types = ctx.op_result_types(op).to_vec();
        if result_types.is_empty() {
            return false;
        }

        let result_ty = result_types[0];
        let Some(new_result_ty) = convert_primitive_type(ctx, result_ty) else {
            return false;
        };

        debug!(
            "normalize_primitive_types: func.call {} result type normalized",
            call_op.callee(ctx),
        );

        let loc = ctx.op(op).location;
        let callee = call_op.callee(ctx);
        let args: Vec<_> = call_op.args(ctx).to_vec();

        let new_op = arena_func::call(ctx, loc, args, new_result_ty, callee);
        rewriter.replace_op(new_op.op_ref());
        true
    }

    fn name(&self) -> &'static str {
        "NormalizeCallPattern"
    }
}

/// Normalize func.call_indirect operation result types.
struct NormalizeCallIndirectPattern;

impl ArenaRewritePattern for NormalizeCallIndirectPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(_call_op) = arena_func::CallIndirect::from_op(ctx, op) else {
            return false;
        };

        let result_types = ctx.op_result_types(op).to_vec();
        if result_types.is_empty() {
            return false;
        }

        let result_ty = result_types[0];
        let Some(new_result_ty) = convert_primitive_type(ctx, result_ty) else {
            return false;
        };

        debug!("normalize_primitive_types: func.call_indirect result type normalized");

        let loc = ctx.op(op).location;
        let operands = ctx.op_operands(op).to_vec();
        let callee = operands[0];
        let args: Vec<_> = operands[1..].to_vec();

        let new_op = arena_func::call_indirect(ctx, loc, callee, args, new_result_ty);
        rewriter.replace_op(new_op.op_ref());
        true
    }

    fn name(&self) -> &'static str {
        "NormalizeCallIndirectPattern"
    }
}

/// Normalize general operation result types.
///
/// This is a catch-all pattern for operations with primitive result types
/// that weren't handled by more specific patterns.
struct NormalizeOpResultPattern;

impl ArenaRewritePattern for NormalizeOpResultPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let result_types = ctx.op_result_types(op).to_vec();
        if result_types.is_empty() {
            return false;
        }

        // Check if any result types need normalization
        let mut any_changed = false;
        let new_results: Vec<TypeRef> = result_types
            .iter()
            .map(|&result_ty| {
                if let Some(converted) = convert_primitive_type(ctx, result_ty) {
                    any_changed = true;
                    converted
                } else {
                    result_ty
                }
            })
            .collect();

        if !any_changed {
            return false;
        }

        // Skip operations already handled by other patterns
        let data = ctx.op(op);
        let dialect = data.dialect;
        let name = data.name;
        if dialect == Symbol::new("func")
            && (name == Symbol::new("func")
                || name == Symbol::new("call")
                || name == Symbol::new("call_indirect"))
        {
            return false;
        }
        if dialect == Symbol::new("wasm") && name == Symbol::new("func") {
            return false;
        }

        debug!(
            "normalize_primitive_types: {}.{} result type normalized",
            dialect, name
        );

        // Create a replacement op with updated result types.
        let loc = ctx.op(op).location;
        let operands: Vec<_> = ctx.op_operands(op).to_vec();
        let regions: Vec<_> = ctx.op(op).regions.to_vec();
        let successors: Vec<_> = ctx.op(op).successors.to_vec();
        let attributes = ctx.op(op).attributes.clone();

        // Detach regions so they can be reused
        for &region in &regions {
            ctx.detach_region(region);
        }

        let mut builder = trunk_ir::arena::context::OperationDataBuilder::new(loc, dialect, name)
            .operands(operands)
            .results(new_results);

        for (key, val) in attributes {
            builder = builder.attr(key, val);
        }
        for region in regions {
            builder = builder.region(region);
        }
        for succ in successors {
            builder = builder.successor(succ);
        }

        let op_data = builder.build(ctx);
        let new_op = ctx.create_op(op_data);
        rewriter.replace_op(new_op);
        true
    }

    fn name(&self) -> &'static str {
        "NormalizeOpResultPattern"
    }
}
