//! Cast resolution pass for eliminating `unrealized_conversion_cast` operations.
//!
//! After dialect conversion, `unrealized_conversion_cast` operations may remain
//! in the IR as placeholders for type mismatches. This pass resolves them by:
//! 1. Finding all `unrealized_conversion_cast` operations
//! 2. Using the TypeConverter's materialization functions to generate actual conversion code
//! 3. Replacing the casts with the materialized operations
//!
//! If any casts cannot be resolved, the pass returns the unresolved list.

use crate::arena::context::IrContext;
use crate::arena::dialect::core as arena_core;
use crate::arena::ops::DialectOp;
use crate::arena::refs::{BlockRef, OpRef, RegionRef, TypeRef};
use crate::arena::rewrite::{Module, TypeConverter};
use crate::arena::types::Location as ArenaLocation;

/// Information about an unresolved cast (arena version).
#[derive(Debug, Clone)]
pub struct UnresolvedCast {
    /// Location of the cast operation.
    pub location: ArenaLocation,
    /// Source type.
    pub from_type: TypeRef,
    /// Target type.
    pub to_type: TypeRef,
}

/// Result of resolving casts in an arena module.
#[derive(Debug)]
pub struct ResolveResult {
    /// Number of casts that were resolved.
    pub resolved_count: usize,
    /// Casts that could not be resolved.
    pub unresolved: Vec<UnresolvedCast>,
}

/// Resolve all `unrealized_conversion_cast` operations in an arena module.
///
/// Uses the provided TypeConverter's materialization functions to generate
/// actual conversion operations. Mutates the module in place.
pub fn resolve_unrealized_casts_arena(
    ctx: &mut IrContext,
    module: Module,
    tc: &TypeConverter,
) -> ResolveResult {
    tracing::debug!("resolve_unrealized_casts_arena: starting resolution");
    let mut resolver = ArenaCastResolver::new();

    let body = match module.body(ctx) {
        Some(r) => r,
        None => {
            return ResolveResult {
                resolved_count: 0,
                unresolved: vec![],
            };
        }
    };

    resolver.resolve_region(ctx, tc, body);

    tracing::debug!(
        "resolve_unrealized_casts_arena: resolved {} casts, {} unresolved",
        resolver.resolved_count,
        resolver.unresolved.len()
    );

    ResolveResult {
        resolved_count: resolver.resolved_count,
        unresolved: resolver.unresolved,
    }
}

/// Internal resolver state for arena IR.
struct ArenaCastResolver {
    unresolved: Vec<UnresolvedCast>,
    resolved_count: usize,
}

impl ArenaCastResolver {
    fn new() -> Self {
        Self {
            unresolved: Vec::new(),
            resolved_count: 0,
        }
    }

    fn resolve_region(&mut self, ctx: &mut IrContext, tc: &TypeConverter, region: RegionRef) {
        let blocks: Vec<BlockRef> = ctx.region(region).blocks.to_vec();
        for block in blocks {
            self.resolve_block(ctx, tc, block);
        }
    }

    fn resolve_block(&mut self, ctx: &mut IrContext, tc: &TypeConverter, block: BlockRef) {
        // Snapshot ops to iterate over (block may be mutated)
        let ops: Vec<OpRef> = ctx.block(block).ops.to_vec();

        for op in ops {
            // Skip ops removed from block during processing
            if ctx.op(op).parent_block != Some(block) {
                continue;
            }

            // Check if this is an unrealized_conversion_cast
            if arena_core::UnrealizedConversionCast::matches(ctx, op) {
                self.try_resolve_cast(ctx, tc, block, op);
                continue;
            }

            // Recurse into nested regions
            let regions: Vec<RegionRef> = ctx.op(op).regions.to_vec();
            for region in regions {
                self.resolve_region(ctx, tc, region);
            }
        }
    }

    fn try_resolve_cast(
        &mut self,
        ctx: &mut IrContext,
        tc: &TypeConverter,
        block: BlockRef,
        op: OpRef,
    ) {
        let operands = ctx.op_operands(op).to_vec();
        let result_types = ctx.op_result_types(op).to_vec();
        let location = ctx.op(op).location;

        let Some(&input_value) = operands.first() else {
            return;
        };
        let Some(&original_to_type) = result_types.first() else {
            return;
        };

        // Convert the target type if needed
        let to_type = tc
            .convert_type(ctx, original_to_type)
            .unwrap_or(original_to_type);

        // Get the source type from the input value
        let from_type = ctx.value_ty(input_value);

        // Get the cast result value
        let cast_result = ctx.op_result(op, 0);

        // If types are the same, just RAUW and remove the cast
        if from_type == to_type {
            ctx.replace_all_uses(cast_result, input_value);
            ctx.remove_op_from_block(block, op);
            self.resolved_count += 1;
            return;
        }

        // Try to materialize the conversion
        let mat_result = tc.materialize(ctx, location, input_value, from_type, to_type);

        match mat_result {
            Some(mat) => {
                // Insert materialized ops before the cast
                for &mat_op in &mat.ops {
                    ctx.insert_op_before(block, op, mat_op);
                }
                // RAUW the cast result with the materialized value
                ctx.replace_all_uses(cast_result, mat.value);
                ctx.remove_op_from_block(block, op);
                self.resolved_count += 1;
            }
            None => {
                // Could not materialize - keep the cast and mark as unresolved
                tracing::debug!("resolve_unrealized_casts_arena: FAILED to materialize cast");
                self.unresolved.push(UnresolvedCast {
                    location,
                    from_type,
                    to_type,
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    mod arena_tests {
        use crate::arena::OperationDataBuilder;
        use crate::arena::context::{BlockData, IrContext, RegionData};
        use crate::arena::dialect::arith as arena_arith;
        use crate::arena::dialect::core as arena_core;
        use crate::arena::refs::{OpRef, TypeRef};
        use crate::arena::rewrite::{Module, TypeConverter};
        use crate::arena::types::{Attribute, Location, TypeDataBuilder};
        use crate::conversion::resolve_unrealized_casts_arena;
        use crate::location::Span;
        use crate::symbol::Symbol;
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

        fn i64_type(ctx: &mut IrContext) -> TypeRef {
            ctx.types
                .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i64")).build())
        }

        fn make_module(ctx: &mut IrContext, loc: Location, ops: Vec<OpRef>) -> Module {
            let block = ctx.create_block(BlockData {
                location: loc,
                args: vec![],
                ops: smallvec![],
                parent_region: None,
            });
            for op in ops {
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
            Module::new(ctx, module_op).expect("test module should be valid")
        }

        #[test]
        fn arena_resolve_no_materializer() {
            let (mut ctx, loc) = test_ctx();
            let i32_ty = i32_type(&mut ctx);
            let i64_ty = i64_type(&mut ctx);

            // arith.const -> i32
            let const_op = arena_arith::r#const(&mut ctx, loc, i32_ty, Attribute::IntBits(42));
            let const_result = const_op.result(&ctx);

            // unrealized_conversion_cast(const_result) -> i64
            let cast_op =
                arena_core::unrealized_conversion_cast(&mut ctx, loc, const_result, i64_ty);

            let module = make_module(&mut ctx, loc, vec![const_op.op_ref(), cast_op.op_ref()]);

            let tc = TypeConverter::new();
            let result = resolve_unrealized_casts_arena(&mut ctx, module, &tc);

            // Should fail — no materializer registered
            assert_eq!(result.unresolved.len(), 1);
            assert_eq!(result.resolved_count, 0);
        }

        #[test]
        fn arena_resolve_same_type_noop() {
            let (mut ctx, loc) = test_ctx();
            let i32_ty = i32_type(&mut ctx);

            // arith.const -> i32
            let const_op = arena_arith::r#const(&mut ctx, loc, i32_ty, Attribute::IntBits(42));
            let const_result = const_op.result(&ctx);

            // unrealized_conversion_cast(const_result) -> i32 (same type)
            let cast_op =
                arena_core::unrealized_conversion_cast(&mut ctx, loc, const_result, i32_ty);

            let module = make_module(&mut ctx, loc, vec![const_op.op_ref(), cast_op.op_ref()]);

            let tc = TypeConverter::new();
            let result = resolve_unrealized_casts_arena(&mut ctx, module, &tc);

            // Same-type casts should be resolved even without a materializer
            assert!(result.unresolved.is_empty());
            assert_eq!(result.resolved_count, 1);

            // Cast should be removed; only const remains
            let ops = module.ops(&ctx);
            assert_eq!(ops.len(), 1);
            assert_eq!(ctx.op(ops[0]).dialect, Symbol::new("arith"));
        }

        #[test]
        fn arena_resolve_with_materializer() {
            let (mut ctx, loc) = test_ctx();
            let i32_ty = i32_type(&mut ctx);
            let i64_ty = i64_type(&mut ctx);

            // arith.const -> i32
            let const_op = arena_arith::r#const(&mut ctx, loc, i32_ty, Attribute::IntBits(42));
            let const_result = const_op.result(&ctx);

            // unrealized_conversion_cast(const_result) -> i64
            let cast_op =
                arena_core::unrealized_conversion_cast(&mut ctx, loc, const_result, i64_ty);
            let cast_result = cast_op.result(&ctx);

            // A user of the cast result
            let user_op_data =
                OperationDataBuilder::new(loc, Symbol::new("test"), Symbol::new("use"))
                    .operand(cast_result)
                    .build(&mut ctx);
            let user_op = ctx.create_op(user_op_data);

            let module = make_module(
                &mut ctx,
                loc,
                vec![const_op.op_ref(), cast_op.op_ref(), user_op],
            );

            // Set up a materializer that creates a sextend op
            let mut tc = TypeConverter::new();
            tc.set_materializer(move |ctx, loc, value, _from_ty, to_ty| {
                use crate::arena::rewrite::type_converter::MaterializeResult;
                let op_data =
                    OperationDataBuilder::new(loc, Symbol::new("clif"), Symbol::new("sextend"))
                        .operand(value)
                        .result(to_ty)
                        .build(ctx);
                let mat_op = ctx.create_op(op_data);
                let result_val = ctx.op_result(mat_op, 0);
                Some(MaterializeResult {
                    value: result_val,
                    ops: vec![mat_op],
                })
            });

            let result = resolve_unrealized_casts_arena(&mut ctx, module, &tc);

            assert!(result.unresolved.is_empty());
            assert_eq!(result.resolved_count, 1);

            // Module should now have: const, sextend, use
            let ops = module.ops(&ctx);
            assert_eq!(ops.len(), 3);
            assert_eq!(ctx.op(ops[0]).name, Symbol::new("const"));
            assert_eq!(ctx.op(ops[1]).name, Symbol::new("sextend"));
            assert_eq!(ctx.op(ops[2]).name, Symbol::new("use"));

            // The user op should reference the sextend result
            let user_operands = ctx.op_operands(ops[2]);
            let sextend_result = ctx.op_result(ops[1], 0);
            assert_eq!(user_operands[0], sextend_result);
        }
    }
}
