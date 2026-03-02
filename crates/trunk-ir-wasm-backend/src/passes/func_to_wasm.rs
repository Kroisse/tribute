//! Lower func dialect operations to wasm dialect (arena IR).
//!
//! This pass converts function-level operations to wasm operations:
//! - `func.func` -> `wasm.func`
//! - `func.call` -> `wasm.call`
//! - `func.call_indirect` -> `wasm.call_indirect`
//! - `func.return` -> `wasm.return`
//! - `func.tail_call` -> `wasm.return_call`
//! - `func.unreachable` -> `wasm.unreachable`
//! - `func.constant` -> `wasm.i32_const` (function table index)
//!
//! For closures, this pass also:
//! - Collects all functions referenced by `func.constant`
//! - Creates a function table with those functions
//! - Generates `wasm.table` and `wasm.elem` operations

use std::collections::HashMap;

use trunk_ir::arena::context::IrContext;
use trunk_ir::arena::dialect::func as arena_func;
use trunk_ir::arena::dialect::wasm as arena_wasm;
use trunk_ir::arena::ops::ArenaDialectOp;
use trunk_ir::arena::refs::{OpRef, RegionRef, TypeRef};
use trunk_ir::arena::rewrite::{
    ArenaModule, ArenaRewritePattern, ArenaTypeConverter, PatternApplicator, PatternRewriter,
};
use trunk_ir::arena::types::TypeDataBuilder;
use trunk_ir::arena::{BlockData, RegionData};
use trunk_ir::ir::Symbol;

use trunk_ir::smallvec::smallvec;

/// Lower func dialect to wasm dialect using arena IR.
///
/// The `type_converter` parameter allows language-specific backends to provide
/// their own type conversion rules.
pub fn lower(ctx: &mut IrContext, module: ArenaModule, type_converter: ArenaTypeConverter) {
    // 1. Collect all functions referenced by func.constant operations
    let func_refs = collect_func_constant_refs(ctx, module);

    if func_refs.is_empty() {
        // No func.constant operations - just apply patterns without table generation
        let applicator = PatternApplicator::new(type_converter)
            .add_pattern(FuncFuncPattern)
            .add_pattern(FuncCallPattern)
            .add_pattern(FuncCallIndirectPattern)
            .add_pattern(FuncReturnPattern)
            .add_pattern(FuncTailCallPattern)
            .add_pattern(FuncUnreachablePattern)
            .add_pattern(FuncConstantPattern {
                table_indices: HashMap::new(),
            });
        applicator.apply_partial(ctx, module);
        return;
    }

    // 2. Assign table indices (sorted for deterministic ordering)
    let mut sorted_funcs: Vec<_> = func_refs.into_iter().collect();
    sorted_funcs.sort_by(|a, b| a.with_str(|a_str| b.with_str(|b_str| a_str.cmp(b_str))));

    let table_indices: HashMap<Symbol, u32> = sorted_funcs
        .iter()
        .enumerate()
        .map(|(idx, sym)| (*sym, idx as u32))
        .collect();

    let table_size = sorted_funcs.len() as u32;

    // 3. Apply patterns to transform operations
    let applicator = PatternApplicator::new(type_converter)
        .add_pattern(FuncFuncPattern)
        .add_pattern(FuncCallPattern)
        .add_pattern(FuncCallIndirectPattern)
        .add_pattern(FuncReturnPattern)
        .add_pattern(FuncTailCallPattern)
        .add_pattern(FuncUnreachablePattern)
        .add_pattern(FuncConstantPattern {
            table_indices: table_indices.clone(),
        });
    applicator.apply_partial(ctx, module);

    // 4. Add wasm.table and wasm.elem to the module
    add_function_table(ctx, module, &sorted_funcs, table_size);
}

/// Collect all function symbols referenced by func.constant operations.
fn collect_func_constant_refs(ctx: &IrContext, module: ArenaModule) -> Vec<Symbol> {
    let mut funcs = Vec::new();
    if let Some(body) = module.body(ctx) {
        collect_refs_in_region(ctx, body, &mut funcs);
    }

    // Deduplicate while preserving order
    let mut seen = std::collections::HashSet::new();
    funcs.retain(|sym| seen.insert(*sym));

    funcs
}

fn collect_refs_in_region(ctx: &IrContext, region: RegionRef, refs: &mut Vec<Symbol>) {
    for &block in ctx.region(region).blocks.iter() {
        for &op in ctx.block(block).ops.iter() {
            // Check for func.constant
            if let Ok(const_op) = arena_func::Constant::from_op(ctx, op) {
                refs.push(const_op.func_ref(ctx));
            }

            // Recurse into nested regions
            for &nested in ctx.op(op).regions.iter() {
                collect_refs_in_region(ctx, nested, refs);
            }
        }
    }
}

/// Add wasm.table and wasm.elem operations to the module for the function table.
fn add_function_table(ctx: &mut IrContext, module: ArenaModule, funcs: &[Symbol], table_size: u32) {
    let Some(first_block) = module.first_block(ctx) else {
        return;
    };

    // Use the location from the module op
    let location = ctx.op(module.op()).location;

    // Create wasm.table for closure functions
    let table_op = arena_wasm::table(
        ctx,
        location,
        Symbol::new("funcref"),
        table_size,
        Some(table_size),
    );

    // Create wasm.ref_func operations for each function in the element segment
    let funcref_ty = intern_funcref_type(ctx);
    let func_ref_ops: Vec<OpRef> = funcs
        .iter()
        .map(|func_sym| arena_wasm::ref_func(ctx, location, funcref_ty, *func_sym).op_ref())
        .collect();

    // Create the funcs region for wasm.elem
    let funcs_block = ctx.create_block(BlockData {
        location,
        args: vec![],
        ops: smallvec![],
        parent_region: None,
    });
    for ref_op in &func_ref_ops {
        ctx.push_op(funcs_block, *ref_op);
    }
    let funcs_region = ctx.create_region(RegionData {
        location,
        blocks: smallvec![funcs_block],
        parent_op: None,
    });

    // Create wasm.elem with table 0 and offset 0
    let elem_op = arena_wasm::elem(ctx, location, Some(0), Some(0), funcs_region);

    // Prepend table and elem operations to the module body.
    // We insert before the first op in the block (if any), or push at the end.
    let existing_ops: Vec<OpRef> = ctx.block(first_block).ops.to_vec();
    if let Some(&first_op) = existing_ops.first() {
        ctx.insert_op_before(first_block, first_op, table_op.op_ref());
        ctx.insert_op_before(first_block, first_op, elem_op.op_ref());
    } else {
        ctx.push_op(first_block, table_op.op_ref());
        ctx.push_op(first_block, elem_op.op_ref());
    }
}

/// Pattern for `func.func` -> `wasm.func`
struct FuncFuncPattern;

impl ArenaRewritePattern for FuncFuncPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(func_op) = arena_func::Func::from_op(ctx, op) else {
            return false;
        };

        let loc = ctx.op(op).location;
        let sym_name = func_op.sym_name(ctx);
        let func_type = func_op.r#type(ctx);
        let body = func_op.body(ctx);

        // Detach body region so it can be reused in the new wasm.func
        ctx.detach_region(body);

        let new_op = arena_wasm::func(ctx, loc, sym_name, func_type, body);
        rewriter.replace_op(new_op.op_ref());
        true
    }
}

/// Pattern for `func.call` -> `wasm.call`
struct FuncCallPattern;

impl ArenaRewritePattern for FuncCallPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(call_op) = arena_func::Call::from_op(ctx, op) else {
            return false;
        };

        let loc = ctx.op(op).location;
        let callee = call_op.callee(ctx);
        let args: Vec<_> = ctx.op_operands(op).to_vec();
        let result_types: Vec<TypeRef> = ctx.op_result_types(op).to_vec();

        let new_op = arena_wasm::call(ctx, loc, args, result_types, callee);
        rewriter.replace_op(new_op.op_ref());
        true
    }
}

/// Pattern for `func.call_indirect` -> `wasm.call_indirect`
///
/// Transforms indirect function calls for closures.
/// The callee (i32 table index) is the first operand, followed by arguments.
struct FuncCallIndirectPattern;

impl ArenaRewritePattern for FuncCallIndirectPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(_call_indirect) = arena_func::CallIndirect::from_op(ctx, op) else {
            return false;
        };

        let loc = ctx.op(op).location;
        let all_operands: Vec<_> = ctx.op_operands(op).to_vec();
        let result_types: Vec<TypeRef> = ctx.op_result_types(op).to_vec();

        // Build wasm.call_indirect with same operands
        // The emit phase will resolve the type_idx and table attributes
        let new_op = arena_wasm::call_indirect(ctx, loc, all_operands, result_types, 0, 0);
        rewriter.replace_op(new_op.op_ref());
        true
    }
}

/// Pattern for `func.return` -> `wasm.return`
struct FuncReturnPattern;

impl ArenaRewritePattern for FuncReturnPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(_return_op) = arena_func::Return::from_op(ctx, op) else {
            return false;
        };

        let loc = ctx.op(op).location;
        let values: Vec<_> = ctx.op_operands(op).to_vec();

        let new_op = arena_wasm::r#return(ctx, loc, values);
        rewriter.replace_op(new_op.op_ref());
        true
    }
}

/// Pattern for `func.tail_call` -> `wasm.return_call`
struct FuncTailCallPattern;

impl ArenaRewritePattern for FuncTailCallPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(tail_call_op) = arena_func::TailCall::from_op(ctx, op) else {
            return false;
        };

        let loc = ctx.op(op).location;
        let callee = tail_call_op.callee(ctx);
        let args: Vec<_> = ctx.op_operands(op).to_vec();

        let new_op = arena_wasm::return_call(ctx, loc, args, callee);
        rewriter.replace_op(new_op.op_ref());
        true
    }
}

/// Pattern for `func.unreachable` -> `wasm.unreachable`
struct FuncUnreachablePattern;

impl ArenaRewritePattern for FuncUnreachablePattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(_unreachable_op) = arena_func::Unreachable::from_op(ctx, op) else {
            return false;
        };

        let loc = ctx.op(op).location;

        let new_op = arena_wasm::unreachable(ctx, loc);
        rewriter.replace_op(new_op.op_ref());
        true
    }
}

/// Pattern for `func.constant` -> `wasm.i32_const` (i32 table index)
///
/// Transforms function constant references to i32 table indices.
/// Used for closures where lifted functions are stored via function table.
struct FuncConstantPattern {
    table_indices: HashMap<Symbol, u32>,
}

impl ArenaRewritePattern for FuncConstantPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(const_op) = arena_func::Constant::from_op(ctx, op) else {
            return false;
        };

        let func_ref = const_op.func_ref(ctx);

        // All func.constant operations must be registered in the function table.
        // They are collected by collect_func_constant_refs before pattern application.
        let table_idx = self
            .table_indices
            .get(&func_ref)
            .copied()
            .expect("All func.constant must be registered in table");

        let loc = ctx.op(op).location;
        let i32_ty = intern_i32_type(ctx);
        let new_op = arena_wasm::i32_const(ctx, loc, i32_ty, table_idx as i32);

        rewriter.replace_op(new_op.op_ref());
        true
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Intern a core.i32 type.
fn intern_i32_type(ctx: &mut IrContext) -> TypeRef {
    ctx.types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build())
}

/// Intern a wasm.funcref type.
fn intern_funcref_type(ctx: &mut IrContext) -> TypeRef {
    ctx.types
        .intern(TypeDataBuilder::new(Symbol::new("wasm"), Symbol::new("funcref")).build())
}
