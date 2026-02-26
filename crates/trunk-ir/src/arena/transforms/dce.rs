//! Dead Code Elimination (DCE) pass for arena IR.
//!
//! Removes operations whose results are never used and which have no side effects.
//! Uses the arena's built-in use-chains for O(1) dead-op detection, making this
//! significantly simpler than the Salsa-based version which requires backward
//! liveness analysis.

use crate::arena::context::IrContext;
use crate::arena::refs::{BlockRef, OpRef, RegionRef};
use crate::arena::rewrite::ArenaModule;
use crate::op_interface::PureOps;

/// Configuration for dead code elimination.
#[derive(Debug, Clone)]
pub struct DceConfig {
    /// Maximum fixpoint iterations before giving up. Default: 100.
    pub max_iterations: usize,
    /// Whether to recursively process nested regions. Default: true.
    pub recursive: bool,
}

impl Default for DceConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            recursive: true,
        }
    }
}

/// Result of running dead code elimination.
pub struct DceResult {
    /// Total number of operations removed.
    pub removed_count: usize,
    /// Number of fixpoint iterations performed.
    pub iterations: usize,
    /// Whether fixpoint was reached (no more changes possible).
    pub reached_fixpoint: bool,
}

/// Eliminate dead code from a module using default configuration.
pub fn eliminate_dead_code(ctx: &mut IrContext, module: ArenaModule) -> DceResult {
    eliminate_dead_code_with_config(ctx, module, DceConfig::default())
}

/// Eliminate dead code with custom configuration.
pub fn eliminate_dead_code_with_config(
    ctx: &mut IrContext,
    module: ArenaModule,
    config: DceConfig,
) -> DceResult {
    let max_iterations = if config.max_iterations == 0 {
        100
    } else {
        config.max_iterations
    };

    let mut total_removed = 0;

    for iteration in 0..max_iterations {
        let removed = sweep_module(ctx, module, &config);

        if removed == 0 {
            return DceResult {
                removed_count: total_removed,
                iterations: iteration + 1,
                reached_fixpoint: true,
            };
        }

        total_removed += removed;
    }

    DceResult {
        removed_count: total_removed,
        iterations: max_iterations,
        reached_fixpoint: false,
    }
}

/// Sweep all top-level functions in a module, removing dead ops.
/// Returns the number of ops removed in this sweep.
fn sweep_module(ctx: &mut IrContext, module: ArenaModule, config: &DceConfig) -> usize {
    let body = match module.body(ctx) {
        Some(r) => r,
        None => return 0,
    };
    sweep_region(ctx, body, config)
}

/// Sweep all blocks in a region. Returns the number of ops removed.
fn sweep_region(ctx: &mut IrContext, region: RegionRef, config: &DceConfig) -> usize {
    let blocks: Vec<BlockRef> = ctx.region(region).blocks.to_vec();
    let mut removed = 0;
    for block in blocks {
        removed += sweep_block(ctx, block, config);
    }
    removed
}

/// Sweep a single block in reverse order, removing dead ops.
///
/// Iterating in reverse maximizes cascade removal: if op C uses op B's
/// result and op B uses op A's result, removing C first frees B, then A.
fn sweep_block(ctx: &mut IrContext, block: BlockRef, config: &DceConfig) -> usize {
    let mut removed = 0;

    // First, recursively process nested regions of all ops
    if config.recursive {
        let ops: Vec<OpRef> = ctx.block(block).ops.to_vec();
        for op in ops {
            let regions: Vec<RegionRef> = ctx.op(op).regions.to_vec();
            for region in regions {
                removed += sweep_region(ctx, region, config);
            }
        }
    }

    // Now sweep this block's ops in reverse
    let ops: Vec<OpRef> = ctx.block(block).ops.to_vec();
    for &op in ops.iter().rev() {
        if is_dead(ctx, op) {
            ctx.remove_op_from_block(block, op);
            ctx.remove_op(op);
            removed += 1;
        }
    }

    removed
}

/// Check if an operation is dead (pure + all results unused).
fn is_dead(ctx: &IrContext, op: OpRef) -> bool {
    // Non-pure operations have side effects — must keep
    if !PureOps::is_pure_arena(ctx, op) {
        return false;
    }

    // Operations with no results: pure ops without results are trivially dead
    let results = ctx.op_results(op);
    if results.is_empty() {
        return true;
    }

    // Check if any result is still used
    for &result in results {
        if ctx.has_uses(result) {
            return false;
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::dialect::{arith, func};
    use crate::arena::*;
    use crate::ir::Symbol;
    use crate::location::Span;
    use smallvec::smallvec;
    fn test_ctx() -> (IrContext, Location) {
        let mut ctx = IrContext::new();
        let path = ctx.paths.intern("test.trb".to_owned());
        let loc = Location::new(path, Span::new(0, 0));
        (ctx, loc)
    }

    fn i32_type(ctx: &mut IrContext) -> TypeRef {
        ctx.types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build())
    }

    fn fn_type(ctx: &mut IrContext) -> TypeRef {
        ctx.types
            .intern(TypeDataBuilder::new(Symbol::new("func"), Symbol::new("fn")).build())
    }

    /// Build a minimal module wrapping the given function ops.
    fn build_module(ctx: &mut IrContext, loc: Location, func_ops: Vec<OpRef>) -> ArenaModule {
        let block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        for op in func_ops {
            ctx.push_op(block, op);
        }
        let region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![block],
            parent_op: None,
        });
        let module_data =
            OperationDataBuilder::new(loc, Symbol::new("core"), Symbol::new("module"))
                .attr("sym_name", Attribute::Symbol(Symbol::new("test")))
                .region(region)
                .build(ctx);
        let module_op = ctx.create_op(module_data);
        ArenaModule::new(ctx, module_op).unwrap()
    }

    /// Build a func.func with a body built by the callback.
    fn build_func(
        ctx: &mut IrContext,
        loc: Location,
        name: &str,
        build_body: impl FnOnce(&mut IrContext, Location, BlockRef),
    ) -> OpRef {
        let fn_ty = fn_type(ctx);
        let sym_name = Symbol::from_dynamic(name);
        let entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        build_body(ctx, loc, entry);
        let body_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });
        func::func(ctx, loc, sym_name, fn_ty, body_region).op_ref()
    }

    #[test]
    fn removes_unused_pure_op() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);

        let func_op = build_func(&mut ctx, loc, "main", |ctx, loc, entry| {
            // dead: result is never used
            let _dead = arith::r#const(ctx, loc, i32_ty, Attribute::IntBits(42));
            ctx.push_op(entry, _dead.op_ref());

            // alive: used by return
            let alive = arith::r#const(ctx, loc, i32_ty, Attribute::IntBits(1));
            ctx.push_op(entry, alive.op_ref());
            let ret = func::r#return(ctx, loc, [alive.result(ctx)]);
            ctx.push_op(entry, ret.op_ref());
        });
        let module = build_module(&mut ctx, loc, vec![func_op]);

        let result = eliminate_dead_code(&mut ctx, module);

        assert_eq!(result.removed_count, 1);
        assert!(result.reached_fixpoint);
    }

    #[test]
    fn keeps_non_pure_ops() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);

        let func_op = build_func(&mut ctx, loc, "main", |ctx, loc, entry| {
            // func.call is not pure — must keep even if result unused
            let call_data =
                OperationDataBuilder::new(loc, Symbol::new("func"), Symbol::new("call"))
                    .result(i32_ty)
                    .attr("callee", Attribute::Symbol(Symbol::new("foo")))
                    .build(ctx);
            let call_op = ctx.create_op(call_data);
            ctx.push_op(entry, call_op);

            let ret = func::r#return(ctx, loc, std::iter::empty());
            ctx.push_op(entry, ret.op_ref());
        });
        let module = build_module(&mut ctx, loc, vec![func_op]);

        let result = eliminate_dead_code(&mut ctx, module);

        assert_eq!(result.removed_count, 0);
    }

    #[test]
    fn cascade_removal() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);

        let func_op = build_func(&mut ctx, loc, "main", |ctx, loc, entry| {
            // Chain: a -> b -> c (all unused)
            let a = arith::r#const(ctx, loc, i32_ty, Attribute::IntBits(1));
            ctx.push_op(entry, a.op_ref());

            let b = arith::add(ctx, loc, a.result(ctx), a.result(ctx), i32_ty);
            ctx.push_op(entry, b.op_ref());

            let c = arith::add(ctx, loc, b.result(ctx), a.result(ctx), i32_ty);
            ctx.push_op(entry, c.op_ref());

            let ret = func::r#return(ctx, loc, std::iter::empty());
            ctx.push_op(entry, ret.op_ref());
        });
        let module = build_module(&mut ctx, loc, vec![func_op]);

        let result = eliminate_dead_code(&mut ctx, module);

        // All 3 pure ops should be removed (reverse iteration cascades in one pass)
        assert_eq!(result.removed_count, 3);
        assert!(result.reached_fixpoint);
    }

    #[test]
    fn keeps_used_ops() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);

        let func_op = build_func(&mut ctx, loc, "main", |ctx, loc, entry| {
            let a = arith::r#const(ctx, loc, i32_ty, Attribute::IntBits(1));
            ctx.push_op(entry, a.op_ref());
            let b = arith::r#const(ctx, loc, i32_ty, Attribute::IntBits(2));
            ctx.push_op(entry, b.op_ref());
            let c = arith::add(ctx, loc, a.result(ctx), b.result(ctx), i32_ty);
            ctx.push_op(entry, c.op_ref());
            let ret = func::r#return(ctx, loc, [c.result(ctx)]);
            ctx.push_op(entry, ret.op_ref());
        });
        let module = build_module(&mut ctx, loc, vec![func_op]);

        let result = eliminate_dead_code(&mut ctx, module);

        assert_eq!(result.removed_count, 0);
    }

    #[test]
    fn empty_module() {
        let (mut ctx, loc) = test_ctx();
        let module = build_module(&mut ctx, loc, vec![]);

        let result = eliminate_dead_code(&mut ctx, module);

        assert_eq!(result.removed_count, 0);
        assert!(result.reached_fixpoint);
        assert_eq!(result.iterations, 1);
    }

    #[test]
    fn nested_region_dce() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);

        // Build a func that contains another region (simulated with a generic op)
        let func_op = build_func(&mut ctx, loc, "main", |ctx, loc, entry| {
            // Inner region with a dead const
            let inner_dead = arith::r#const(ctx, loc, i32_ty, Attribute::IntBits(99));
            let inner_block = ctx.create_block(BlockData {
                location: loc,
                args: vec![],
                ops: smallvec![],
                parent_region: None,
            });
            ctx.push_op(inner_block, inner_dead.op_ref());
            let inner_ret = func::r#return(ctx, loc, std::iter::empty());
            ctx.push_op(inner_block, inner_ret.op_ref());

            let inner_region = ctx.create_region(RegionData {
                location: loc,
                blocks: smallvec![inner_block],
                parent_op: None,
            });

            // Outer op that owns the inner region (use func.func as container)
            let fn_ty = ctx
                .types
                .intern(TypeDataBuilder::new(Symbol::new("func"), Symbol::new("fn")).build());
            let nested_func = func::func(ctx, loc, Symbol::new("nested"), fn_ty, inner_region);
            ctx.push_op(entry, nested_func.op_ref());

            let ret = func::r#return(ctx, loc, std::iter::empty());
            ctx.push_op(entry, ret.op_ref());
        });
        let module = build_module(&mut ctx, loc, vec![func_op]);

        let result = eliminate_dead_code(&mut ctx, module);

        // The dead const in the nested region should be removed
        assert_eq!(result.removed_count, 1);
    }

    #[test]
    fn fixpoint_iteration() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);

        // This scenario doesn't actually need multiple iterations since
        // reverse sweep handles cascades. But test the config path.
        let func_op = build_func(&mut ctx, loc, "main", |ctx, loc, entry| {
            let a = arith::r#const(ctx, loc, i32_ty, Attribute::IntBits(1));
            ctx.push_op(entry, a.op_ref());
            let ret = func::r#return(ctx, loc, std::iter::empty());
            ctx.push_op(entry, ret.op_ref());
        });
        let module = build_module(&mut ctx, loc, vec![func_op]);

        let config = DceConfig {
            max_iterations: 1,
            recursive: true,
        };
        let result = eliminate_dead_code_with_config(&mut ctx, module, config);

        assert_eq!(result.removed_count, 1);
    }

    #[test]
    fn non_recursive_config() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);

        let func_op = build_func(&mut ctx, loc, "main", |ctx, loc, entry| {
            // Inner region with a dead const
            let inner_dead = arith::r#const(ctx, loc, i32_ty, Attribute::IntBits(99));
            let inner_block = ctx.create_block(BlockData {
                location: loc,
                args: vec![],
                ops: smallvec![],
                parent_region: None,
            });
            ctx.push_op(inner_block, inner_dead.op_ref());
            let inner_ret = func::r#return(ctx, loc, std::iter::empty());
            ctx.push_op(inner_block, inner_ret.op_ref());

            let inner_region = ctx.create_region(RegionData {
                location: loc,
                blocks: smallvec![inner_block],
                parent_op: None,
            });

            let fn_ty = ctx
                .types
                .intern(TypeDataBuilder::new(Symbol::new("func"), Symbol::new("fn")).build());
            let nested_func = func::func(ctx, loc, Symbol::new("nested"), fn_ty, inner_region);
            ctx.push_op(entry, nested_func.op_ref());

            let ret = func::r#return(ctx, loc, std::iter::empty());
            ctx.push_op(entry, ret.op_ref());
        });
        let module = build_module(&mut ctx, loc, vec![func_op]);

        let config = DceConfig {
            max_iterations: 100,
            recursive: false,
        };
        let result = eliminate_dead_code_with_config(&mut ctx, module, config);

        // With recursive=false, the inner dead const should NOT be removed
        assert_eq!(result.removed_count, 0);
    }
}
