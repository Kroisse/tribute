//! Evidence insertion pass for ability system.
//!
//! This pass transforms effectful functions to receive an evidence pointer parameter
//! and transforms ability operations to evidence lookups.
//!
//! ## Transformations
//!
//! 1. **Effectful function signatures**: Add evidence parameter as first argument
//!    ```text
//!    // Before
//!    fn foo(x: Int) ->{State(Int)} Int
//!
//!    // After
//!    fn foo(ev: Evidence, x: Int) -> Int
//!    ```
//!
//! 2. **Call sites**: Pass evidence through call chains
//!    ```text
//!    // Before
//!    func.call @effectful_fn(%arg)
//!
//!    // After
//!    func.call @effectful_fn(%ev, %arg)
//!    ```
//!
//! 3. **Ability operations**: Transform to evidence lookups (placeholder for now)
//!    ```text
//!    // Before
//!    %result = ability.perform { ability_ref: State, op: "get" }
//!
//!    // After (conceptual - actual lowering in handler_lower pass)
//!    %marker = evidence.lookup %ev, STATE_ID
//!    %result = evidence.dispatch %marker, ...
//!    ```
//!
//! ## Design
//!
//! Per `new-plans/implementation.md`:
//! - Evidence is passed as a pointer (8 bytes) to all effectful functions
//! - Pure functions (empty effect row `{}`) don't receive evidence
//! - Handler installation creates new Evidence (handled in handler_lower pass)

use std::collections::HashSet;

use tribute_ir::dialect::ability;
use trunk_ir::arena::bridge::{export_to_salsa, import_salsa_module};
use trunk_ir::dialect::{core, func};
use trunk_ir::{DialectOp, DialectType, Symbol, Type};

/// Phase 1: Add evidence parameters to effectful function signatures.
///
/// This pass adds an evidence pointer parameter as the first argument to all
/// effectful functions. This must run BEFORE lambda lifting so that:
/// 1. Lambdas can capture the evidence parameter from their enclosing function
/// 2. Lifted lambdas will have evidence in scope
///
/// Transformation:
/// ```text
/// fn foo(x: Int) ->{State(Int)} Int
///   =>
/// fn foo(ev: Evidence, x: Int) -> Int
/// ```
#[salsa::tracked]
pub fn add_evidence_params<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
) -> core::Module<'db> {
    // Early exit: check on Salsa side before paying bridge cost
    let effectful_fns = collect_effectful_functions(db, &module);
    if effectful_fns.is_empty() {
        return module;
    }

    // Bridge to arena, run arena pass, bridge back
    let (mut ctx, arena_module) = import_salsa_module(db, module.as_operation());
    arena::arena_add_evidence_params(&mut ctx, arena_module);
    let exported = export_to_salsa(db, &ctx, arena_module);
    core::Module::from_operation(db, exported).unwrap()
}

/// Phase 2: Transform calls to pass evidence through.
///
/// This pass transforms call sites to pass evidence:
/// - Calls inside effectful functions pass the evidence parameter
///
/// This must run AFTER lambda lifting and closure lowering so that:
/// 1. Lifted lambdas already have evidence parameters
/// 2. Closure calls can also receive evidence
///
/// Transformation:
/// ```text
/// func.call @effectful_fn(%arg)
///   =>
/// func.call @effectful_fn(%ev, %arg)
/// ```
#[salsa::tracked]
pub fn transform_evidence_calls<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
) -> core::Module<'db> {
    // Early exit: check on Salsa side before paying bridge cost
    let effectful_fns = collect_effectful_functions(db, &module);
    let fns_with_evidence = collect_functions_with_evidence_param(db, &module);

    if effectful_fns.is_empty() && fns_with_evidence.is_empty() {
        return module;
    }

    // Bridge to arena, run arena pass, bridge back
    let (mut ctx, arena_module) = import_salsa_module(db, module.as_operation());
    arena::arena_transform_evidence_calls(&mut ctx, arena_module);
    let exported = export_to_salsa(db, &ctx, arena_module);
    core::Module::from_operation(db, exported).unwrap()
}

/// Collect all function names that have `ability.evidence_ptr` as their first parameter.
fn collect_functions_with_evidence_param<'db>(
    db: &'db dyn salsa::Database,
    module: &core::Module<'db>,
) -> HashSet<Symbol> {
    let mut fns_with_evidence = HashSet::new();

    let body = module.body(db);
    for block in body.blocks(db).iter() {
        for op in block.operations(db).iter() {
            if let Ok(func_op) = func::Func::from_operation(db, *op) {
                let func_ty = func_op.r#type(db);
                if let Some(core_func) = core::Func::from_type(db, func_ty) {
                    let params = core_func.params(db);
                    if !params.is_empty() && ability::is_evidence_type(db, params[0]) {
                        fns_with_evidence.insert(func_op.sym_name(db));
                    }
                }
            }
        }
    }

    fns_with_evidence
}

/// Insert evidence parameters for effectful functions.
///
/// This is the combined entry point that runs both phases sequentially.
/// For the new pipeline that separates lambda lifting, use `add_evidence_params`
/// and `transform_evidence_calls` separately.
///
/// The pass works in two phases:
/// 1. Add evidence parameter to effectful function signatures
/// 2. Transform calls inside effectful functions to pass evidence
#[salsa::tracked]
pub fn insert_evidence<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
) -> core::Module<'db> {
    let module = add_evidence_params(db, module);
    transform_evidence_calls(db, module)
}

/// Collect all function names that are effectful.
pub fn collect_effectful_functions<'db>(
    db: &'db dyn salsa::Database,
    module: &core::Module<'db>,
) -> HashSet<Symbol> {
    let mut effectful = HashSet::new();

    let body = module.body(db);
    for block in body.blocks(db).iter() {
        for op in block.operations(db).iter() {
            if let Ok(func_op) = func::Func::from_operation(db, *op) {
                let func_ty = func_op.r#type(db);
                if is_effectful_type(db, func_ty) {
                    effectful.insert(func_op.sym_name(db));
                }
            }
        }
    }

    effectful
}

/// Check if a function type has concrete abilities in its effect row.
///
/// A function is considered effectful if its effect row contains actual abilities.
/// A row with only a tail variable (polymorphic row) but no concrete abilities
/// is considered pure, since at this point no effects were inferred for it.
pub fn is_effectful_type<'db>(db: &'db dyn salsa::Database, ty: Type<'db>) -> bool {
    let Some(func_ty) = core::Func::from_type(db, ty) else {
        return false;
    };

    let Some(effect) = func_ty.effect(db) else {
        return false;
    };

    let Some(row) = core::EffectRowType::from_type(db, effect) else {
        return false;
    };

    // Check if there are actual abilities in the row.
    // A row with only a tail variable and no concrete abilities is considered pure.
    !row.abilities(db).is_empty()
}

// ============================================================================
// Arena-based evidence pass implementation
// ============================================================================

pub mod arena {
    use std::collections::HashSet;

    use tribute_ir::arena::dialect::ability as arena_ability;
    use trunk_ir::arena::context::{BlockArgData, IrContext};
    use trunk_ir::arena::dialect::func as arena_func;
    use trunk_ir::arena::ops::ArenaDialectOp;
    use trunk_ir::arena::refs::{BlockRef, OpRef, RegionRef, TypeRef, ValueRef};
    use trunk_ir::arena::rewrite::ArenaModule;
    use trunk_ir::arena::types::{Attribute, TypeDataBuilder};
    use trunk_ir::ir::Symbol;

    use std::collections::BTreeMap;

    /// Check if an arena `core.func` type has concrete abilities in its effect row.
    pub fn is_effectful_type_arena(ctx: &IrContext, ty: TypeRef) -> bool {
        let data = ctx.types.get(ty);
        if data.dialect != Symbol::new("core") || data.name != Symbol::new("func") {
            return false;
        }

        let effect_ty = match data.attrs.get(&Symbol::new("effect")) {
            Some(Attribute::Type(ty)) => *ty,
            _ => return false,
        };

        let effect_data = ctx.types.get(effect_ty);
        if effect_data.dialect != Symbol::new("core")
            || effect_data.name != Symbol::new("effect_row")
        {
            return false;
        }

        // Effect row params are the concrete abilities; non-empty means effectful
        !effect_data.params.is_empty()
    }

    /// Check if a `core.func` type has evidence as its first parameter.
    fn has_evidence_first_param(ctx: &IrContext, func_ty: TypeRef) -> bool {
        let data = ctx.types.get(func_ty);
        if data.dialect != Symbol::new("core") || data.name != Symbol::new("func") {
            return false;
        }
        // params[0] = return, params[1..] = param types
        if data.params.len() < 2 {
            return false;
        }
        arena_ability::is_evidence_type_ref(ctx, data.params[1])
    }

    /// Collect names of all effectful functions in the module.
    pub fn collect_effectful_functions_arena(
        ctx: &IrContext,
        module: ArenaModule,
    ) -> HashSet<Symbol> {
        let mut effectful = HashSet::new();
        for op in module.ops(ctx) {
            if let Ok(func_op) = arena_func::Func::from_op(ctx, op) {
                let func_ty = func_op.r#type(ctx);
                if is_effectful_type_arena(ctx, func_ty) {
                    effectful.insert(func_op.sym_name(ctx));
                }
            }
        }
        effectful
    }

    /// Collect names of functions whose first parameter is evidence type.
    fn collect_functions_with_evidence_param_arena(
        ctx: &IrContext,
        module: ArenaModule,
    ) -> HashSet<Symbol> {
        let mut fns_with_evidence = HashSet::new();
        for op in module.ops(ctx) {
            if let Ok(func_op) = arena_func::Func::from_op(ctx, op) {
                let func_ty = func_op.r#type(ctx);
                if has_evidence_first_param(ctx, func_ty) {
                    fns_with_evidence.insert(func_op.sym_name(ctx));
                }
            }
        }
        fns_with_evidence
    }

    /// Build a new `core.func` TypeRef with evidence prepended to params.
    fn build_func_type_with_evidence(
        ctx: &mut IrContext,
        old_func_ty: TypeRef,
        ev_ty: TypeRef,
    ) -> TypeRef {
        let data = ctx.types.get(old_func_ty);
        // params[0] = return, params[1..] = param types
        let result_ty = data.params[0];
        let old_params = &data.params[1..];

        let mut builder = TypeDataBuilder::new(Symbol::new("core"), Symbol::new("func"))
            .param(result_ty)
            .param(ev_ty)
            .params(old_params.iter().copied());

        // Preserve effect attribute
        if let Some(eff) = data.attrs.get(&Symbol::new("effect")) {
            builder = builder.attr("effect", eff.clone());
        }

        ctx.types.intern(builder.build())
    }

    /// Phase 1 (arena): Add evidence parameters to effectful function signatures.
    pub fn arena_add_evidence_params(ctx: &mut IrContext, module: ArenaModule) {
        let effectful_fns = collect_effectful_functions_arena(ctx, module);
        if effectful_fns.is_empty() {
            return;
        }

        let ev_ty = arena_ability::evidence_adt_type_ref(ctx);

        // Snapshot the module ops before mutating
        let ops: Vec<OpRef> = module.ops(ctx);

        for op in ops {
            let Ok(func_op) = arena_func::Func::from_op(ctx, op) else {
                continue;
            };

            let func_name = func_op.sym_name(ctx);
            if !effectful_fns.contains(&func_name) {
                continue;
            }

            let func_ty = func_op.r#type(ctx);

            // Skip if first param is already evidence type
            if has_evidence_first_param(ctx, func_ty) {
                continue;
            }

            // Build new function type with evidence prepended
            let new_func_ty = build_func_type_with_evidence(ctx, func_ty, ev_ty);

            // Prepend evidence arg to entry block
            let body = func_op.body(ctx);
            let blocks = &ctx.region(body).blocks;
            if blocks.is_empty() {
                continue;
            }
            let entry_block = blocks[0];

            ctx.prepend_block_arg(
                entry_block,
                BlockArgData {
                    ty: ev_ty,
                    attrs: BTreeMap::new(),
                },
            );

            // Update the func type attribute in-place
            ctx.op_mut(op)
                .attributes
                .insert(Symbol::new("type"), Attribute::Type(new_func_ty));
        }
    }

    /// Phase 2 (arena): Transform calls to pass evidence through call sites.
    pub fn arena_transform_evidence_calls(ctx: &mut IrContext, module: ArenaModule) {
        let effectful_fns = collect_effectful_functions_arena(ctx, module);
        let fns_with_evidence = collect_functions_with_evidence_param_arena(ctx, module);

        if effectful_fns.is_empty() && fns_with_evidence.is_empty() {
            return;
        }

        // All functions that are callee targets requiring evidence
        let all_effectful: HashSet<Symbol> =
            effectful_fns.union(&fns_with_evidence).copied().collect();

        let ops: Vec<OpRef> = module.ops(ctx);

        for op in ops {
            let Ok(func_op) = arena_func::Func::from_op(ctx, op) else {
                continue;
            };

            let func_name = func_op.sym_name(ctx);
            let func_ty = func_op.r#type(ctx);

            // Only process functions that have evidence as first parameter
            let is_explicitly_effectful = all_effectful.contains(&func_name);
            let has_ev_param = has_evidence_first_param(ctx, func_ty);

            if !is_explicitly_effectful && !has_ev_param {
                continue;
            }

            // Get the evidence value from first block arg
            let body = func_op.body(ctx);
            let blocks = &ctx.region(body).blocks;
            if blocks.is_empty() {
                continue;
            }
            let entry_block = blocks[0];
            let block_args = ctx.block_args(entry_block);
            if block_args.is_empty() {
                continue;
            }
            let ev_value = ctx.block_arg(entry_block, 0);

            // Transform calls in all blocks of this function's body
            transform_calls_in_region(ctx, body, ev_value, &all_effectful);
        }
    }

    /// Recursively transform calls in a region.
    fn transform_calls_in_region(
        ctx: &mut IrContext,
        region: RegionRef,
        ev_value: ValueRef,
        effectful_fns: &HashSet<Symbol>,
    ) {
        let blocks: Vec<BlockRef> = ctx.region(region).blocks.to_vec();
        for block in blocks {
            transform_calls_in_block(ctx, block, ev_value, effectful_fns);
        }
    }

    /// Transform calls in a single block, adding evidence to calls to effectful functions.
    fn transform_calls_in_block(
        ctx: &mut IrContext,
        block: BlockRef,
        ev_value: ValueRef,
        effectful_fns: &HashSet<Symbol>,
    ) {
        // Snapshot ops before mutation
        let ops: Vec<OpRef> = ctx.block(block).ops.to_vec();

        for op in ops {
            // First, recurse into nested regions (e.g., scf.if bodies)
            let regions: Vec<RegionRef> = ctx.op(op).regions.to_vec();
            for region in regions {
                transform_calls_in_region(ctx, region, ev_value, effectful_fns);
            }

            // Check if this is a call to an effectful function
            let Ok(call_op) = arena_func::Call::from_op(ctx, op) else {
                continue;
            };

            let callee = call_op.callee(ctx);
            if !effectful_fns.contains(&callee) {
                continue;
            }

            // Check if evidence is already the first argument
            let operands = ctx.op_operands(op);
            if !operands.is_empty() && operands[0] == ev_value {
                continue;
            }

            // Build new call with evidence prepended to args
            let old_args: Vec<ValueRef> = operands.to_vec();
            let result_types: Vec<TypeRef> = ctx.op_result_types(op).to_vec();
            let loc = ctx.op(op).location;

            let mut new_args = vec![ev_value];
            new_args.extend(old_args.iter().copied());

            let result_ty = result_types.first().copied().unwrap_or_else(|| {
                ctx.types
                    .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("nil")).build())
            });

            // Create new call op
            let new_call = arena_func::call(ctx, loc, new_args, result_ty, callee);
            let new_op = new_call.op_ref();

            // Insert new call before old one
            ctx.insert_op_before(block, op, new_op);

            // RAUW: replace all uses of old result with new result
            let old_results = ctx.op_results(op);
            let new_results = ctx.op_results(new_op);
            if !old_results.is_empty() && !new_results.is_empty() {
                let old_result = old_results[0];
                let new_result = new_results[0];
                ctx.replace_all_uses(old_result, new_result);
            }

            // Remove old call
            ctx.remove_op_from_block(block, op);
            ctx.remove_op(op);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa::{Database, DatabaseImpl};
    use trunk_ir::{Symbol, idvec};

    fn make_db() -> DatabaseImpl {
        DatabaseImpl::default()
    }

    #[test]
    fn test_is_effectful_type() {
        let db = make_db();
        db.attach(|db| {
            // Pure function
            let pure_ty = core::Func::new(db, idvec![], *core::I32::new(db));
            assert!(!is_effectful_type(db, *pure_ty));

            // Effectful function with State
            let state_ability = core::AbilityRefType::with_params(
                db,
                Symbol::new("State"),
                idvec![*core::I32::new(db)],
            );
            let effect_row = core::EffectRowType::concrete(db, idvec![*state_ability]);
            let effectful_ty =
                core::Func::with_effect(db, idvec![], *core::I32::new(db), Some(*effect_row));
            assert!(is_effectful_type(db, *effectful_ty));

            // Function with empty effect row
            let empty_row = core::EffectRowType::empty(db);
            let empty_effect_ty =
                core::Func::with_effect(db, idvec![], *core::I32::new(db), Some(*empty_row));
            assert!(!is_effectful_type(db, *empty_effect_ty));
        });
    }
}

#[cfg(test)]
mod arena_tests {
    use super::arena::*;
    use std::collections::BTreeMap;
    use tribute_ir::arena::dialect::ability as arena_ability;
    use trunk_ir::arena::context::{
        BlockArgData, BlockData, IrContext, OperationDataBuilder, RegionData,
    };
    use trunk_ir::arena::dialect::func as arena_func;
    use trunk_ir::arena::ops::ArenaDialectOp;
    use trunk_ir::arena::rewrite::ArenaModule;
    use trunk_ir::arena::types::{Attribute, Location, TypeDataBuilder};
    use trunk_ir::ir::Symbol;
    use trunk_ir::location::Span;
    use trunk_ir::smallvec::smallvec;

    fn test_ctx() -> (IrContext, Location) {
        let mut ctx = IrContext::new();
        let path = ctx.paths.intern("test.trb".to_owned());
        let loc = Location::new(path, Span::new(0, 0));
        (ctx, loc)
    }

    fn i32_type(ctx: &mut IrContext) -> trunk_ir::arena::refs::TypeRef {
        ctx.types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build())
    }

    fn make_func_type(
        ctx: &mut IrContext,
        params: &[trunk_ir::arena::refs::TypeRef],
        ret: trunk_ir::arena::refs::TypeRef,
    ) -> trunk_ir::arena::refs::TypeRef {
        ctx.types.intern(
            TypeDataBuilder::new(Symbol::new("core"), Symbol::new("func"))
                .param(ret)
                .params(params.iter().copied())
                .build(),
        )
    }

    fn make_effectful_func_type(
        ctx: &mut IrContext,
        params: &[trunk_ir::arena::refs::TypeRef],
        ret: trunk_ir::arena::refs::TypeRef,
    ) -> trunk_ir::arena::refs::TypeRef {
        // Create an effect row with a concrete ability
        let i32_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());
        let state_ability = ctx.types.intern(
            TypeDataBuilder::new(Symbol::new("core"), Symbol::new("ability_ref"))
                .param(i32_ty)
                .attr("name", Attribute::Symbol(Symbol::new("State")))
                .build(),
        );
        let effect_row = ctx.types.intern(
            TypeDataBuilder::new(Symbol::new("core"), Symbol::new("effect_row"))
                .param(state_ability)
                .build(),
        );
        ctx.types.intern(
            TypeDataBuilder::new(Symbol::new("core"), Symbol::new("func"))
                .param(ret)
                .params(params.iter().copied())
                .attr("effect", Attribute::Type(effect_row))
                .build(),
        )
    }

    fn make_func_op(
        ctx: &mut IrContext,
        loc: Location,
        name: &'static str,
        func_type: trunk_ir::arena::refs::TypeRef,
        param_types: &[trunk_ir::arena::refs::TypeRef],
    ) -> trunk_ir::arena::refs::OpRef {
        let entry_block = ctx.create_block(BlockData {
            location: loc,
            args: param_types
                .iter()
                .map(|&ty| BlockArgData {
                    ty,
                    attrs: BTreeMap::new(),
                })
                .collect(),
            ops: smallvec![],
            parent_region: None,
        });
        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry_block],
            parent_op: None,
        });
        let f = arena_func::func(ctx, loc, Symbol::new(name), func_type, body);
        f.op_ref()
    }

    fn make_module(
        ctx: &mut IrContext,
        loc: Location,
        ops: Vec<trunk_ir::arena::refs::OpRef>,
    ) -> ArenaModule {
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
        ArenaModule::new(ctx, module_op).expect("test module should be valid")
    }

    // === is_effectful_type_arena tests ===

    #[test]
    fn test_is_effectful_type_arena_pure() {
        let (mut ctx, _) = test_ctx();
        let i32_ty = i32_type(&mut ctx);
        let pure_ty = make_func_type(&mut ctx, &[i32_ty], i32_ty);
        assert!(!is_effectful_type_arena(&ctx, pure_ty));
    }

    #[test]
    fn test_is_effectful_type_arena_effectful() {
        let (mut ctx, _) = test_ctx();
        let i32_ty = i32_type(&mut ctx);
        let effectful_ty = make_effectful_func_type(&mut ctx, &[i32_ty], i32_ty);
        assert!(is_effectful_type_arena(&ctx, effectful_ty));
    }

    #[test]
    fn test_is_effectful_type_arena_empty_effect_row() {
        let (mut ctx, _) = test_ctx();
        let i32_ty = i32_type(&mut ctx);
        // Empty effect row (no abilities)
        let empty_row = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("effect_row")).build());
        let ty = ctx.types.intern(
            TypeDataBuilder::new(Symbol::new("core"), Symbol::new("func"))
                .param(i32_ty)
                .attr("effect", Attribute::Type(empty_row))
                .build(),
        );
        assert!(!is_effectful_type_arena(&ctx, ty));
    }

    #[test]
    fn test_is_effectful_type_arena_non_func() {
        let (mut ctx, _) = test_ctx();
        let i32_ty = i32_type(&mut ctx);
        assert!(!is_effectful_type_arena(&ctx, i32_ty));
    }

    // === arena_add_evidence_params tests ===

    #[test]
    fn test_arena_add_evidence_params() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);

        // Create an effectful function: fn foo(x: i32) ->{State(i32)} i32
        let effectful_ty = make_effectful_func_type(&mut ctx, &[i32_ty], i32_ty);
        let func_op = make_func_op(&mut ctx, loc, "foo", effectful_ty, &[i32_ty]);

        // Create a pure function: fn bar(x: i32) -> i32
        let pure_ty = make_func_type(&mut ctx, &[i32_ty], i32_ty);
        let pure_op = make_func_op(&mut ctx, loc, "bar", pure_ty, &[i32_ty]);

        let module = make_module(&mut ctx, loc, vec![func_op, pure_op]);

        // Run add_evidence_params
        arena_add_evidence_params(&mut ctx, module);

        // Verify: effectful function should now have evidence as first param
        let ops = module.ops(&ctx);
        assert_eq!(ops.len(), 2);

        let foo = arena_func::Func::from_op(&ctx, ops[0]).unwrap();
        let foo_ty = foo.r#type(&ctx);
        let foo_data = ctx.types.get(foo_ty);
        // params[0] = return(i32), params[1] = evidence, params[2] = i32
        assert_eq!(foo_data.params.len(), 3);
        assert!(arena_ability::is_evidence_type_ref(
            &ctx,
            foo_data.params[1]
        ));
        assert_eq!(foo_data.params[2], i32_ty);

        // Verify entry block has 2 args now (evidence + i32)
        let foo_body = foo.body(&ctx);
        let foo_entry = ctx.region(foo_body).blocks[0];
        assert_eq!(ctx.block_args(foo_entry).len(), 2);
        let ev_arg = ctx.block_arg(foo_entry, 0);
        assert!(arena_ability::is_evidence_type_ref(
            &ctx,
            ctx.value_ty(ev_arg)
        ));

        // Verify: pure function should be unchanged
        let bar = arena_func::Func::from_op(&ctx, ops[1]).unwrap();
        let bar_ty = bar.r#type(&ctx);
        assert_eq!(bar_ty, pure_ty);
    }

    #[test]
    fn test_arena_add_evidence_params_idempotent() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);
        let effectful_ty = make_effectful_func_type(&mut ctx, &[i32_ty], i32_ty);
        let func_op = make_func_op(&mut ctx, loc, "foo", effectful_ty, &[i32_ty]);
        let module = make_module(&mut ctx, loc, vec![func_op]);

        // Run twice
        arena_add_evidence_params(&mut ctx, module);
        arena_add_evidence_params(&mut ctx, module);

        // Should still have only one evidence param
        let ops = module.ops(&ctx);
        let foo = arena_func::Func::from_op(&ctx, ops[0]).unwrap();
        let foo_ty = foo.r#type(&ctx);
        let foo_data = ctx.types.get(foo_ty);
        // params[0] = return, params[1] = evidence, params[2] = i32
        assert_eq!(foo_data.params.len(), 3);
    }

    #[test]
    fn test_arena_add_evidence_params_no_effectful() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);
        let pure_ty = make_func_type(&mut ctx, &[i32_ty], i32_ty);
        let func_op = make_func_op(&mut ctx, loc, "bar", pure_ty, &[i32_ty]);
        let module = make_module(&mut ctx, loc, vec![func_op]);

        arena_add_evidence_params(&mut ctx, module);

        // Pure function should be unchanged
        let ops = module.ops(&ctx);
        let bar = arena_func::Func::from_op(&ctx, ops[0]).unwrap();
        assert_eq!(bar.r#type(&ctx), pure_ty);
    }

    // === arena_transform_evidence_calls tests ===

    #[test]
    fn test_arena_transform_evidence_calls() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);

        // Create effectful callee: fn callee(x: i32) ->{State} i32
        let callee_ty = make_effectful_func_type(&mut ctx, &[i32_ty], i32_ty);
        let callee_op = make_func_op(&mut ctx, loc, "callee", callee_ty, &[i32_ty]);

        // Create caller with evidence param that calls callee:
        // fn caller(ev: Evidence, x: i32) ->{State} i32
        let ev_ty = arena_ability::evidence_adt_type_ref(&mut ctx);
        let caller_ty = make_effectful_func_type(&mut ctx, &[ev_ty, i32_ty], i32_ty);

        // Build caller body with a call to callee
        let entry_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![
                BlockArgData {
                    ty: ev_ty,
                    attrs: BTreeMap::new(),
                },
                BlockArgData {
                    ty: i32_ty,
                    attrs: BTreeMap::new(),
                },
            ],
            ops: smallvec![],
            parent_region: None,
        });
        let x_arg = ctx.block_arg(entry_block, 1);
        // func.call @callee(%x) : i32
        let call = arena_func::call(&mut ctx, loc, vec![x_arg], i32_ty, Symbol::new("callee"));
        ctx.push_op(entry_block, call.op_ref());
        // func.return %call_result
        let call_result = ctx.op_result(call.op_ref(), 0);
        let ret = arena_func::r#return(&mut ctx, loc, vec![call_result]);
        ctx.push_op(entry_block, ret.op_ref());

        let caller_body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry_block],
            parent_op: None,
        });
        let caller_func =
            arena_func::func(&mut ctx, loc, Symbol::new("caller"), caller_ty, caller_body);

        let module = make_module(&mut ctx, loc, vec![callee_op, caller_func.op_ref()]);

        // Run phase 1 first to add evidence to callee
        arena_add_evidence_params(&mut ctx, module);
        // Run phase 2
        arena_transform_evidence_calls(&mut ctx, module);

        // Verify: the call inside caller should now have evidence as first arg
        let ops = module.ops(&ctx);
        // Find caller function
        let caller = arena_func::Func::from_op(&ctx, ops[1]).unwrap();
        let body = caller.body(&ctx);
        let entry = ctx.region(body).blocks[0];
        let block_ops = ctx.block(entry).ops.to_vec();

        // First op should be the transformed call
        let first_op = block_ops[0];
        let transformed_call = arena_func::Call::from_op(&ctx, first_op).unwrap();
        let call_args = ctx.op_operands(first_op);

        // Should have 2 args: evidence + x
        assert_eq!(call_args.len(), 2);
        // First arg should be the evidence value (block arg 0 of caller)
        let ev_val = ctx.block_arg(entry, 0);
        assert_eq!(call_args[0], ev_val);
        assert_eq!(transformed_call.callee(&ctx), Symbol::new("callee"));
    }

    #[test]
    fn test_arena_evidence_no_calls_to_transform() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);

        // Pure caller calling pure function — nothing to transform
        let pure_ty = make_func_type(&mut ctx, &[i32_ty], i32_ty);
        let func_op = make_func_op(&mut ctx, loc, "pure_fn", pure_ty, &[i32_ty]);
        let module = make_module(&mut ctx, loc, vec![func_op]);

        // Should be no-op
        arena_transform_evidence_calls(&mut ctx, module);

        let ops = module.ops(&ctx);
        let f = arena_func::Func::from_op(&ctx, ops[0]).unwrap();
        assert_eq!(f.r#type(&ctx), pure_ty);
    }

    // === Integration: both phases ===

    #[test]
    fn test_arena_evidence_full_pipeline() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);

        // Create effectful function with a return op
        let effectful_ty = make_effectful_func_type(&mut ctx, &[i32_ty], i32_ty);
        let entry_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![BlockArgData {
                ty: i32_ty,
                attrs: BTreeMap::new(),
            }],
            ops: smallvec![],
            parent_region: None,
        });
        let x_arg = ctx.block_arg(entry_block, 0);
        let ret = arena_func::r#return(&mut ctx, loc, vec![x_arg]);
        ctx.push_op(entry_block, ret.op_ref());
        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry_block],
            parent_op: None,
        });
        let func_op = arena_func::func(&mut ctx, loc, Symbol::new("id"), effectful_ty, body);

        let module = make_module(&mut ctx, loc, vec![func_op.op_ref()]);

        // Run both phases
        arena_add_evidence_params(&mut ctx, module);
        arena_transform_evidence_calls(&mut ctx, module);

        // Verify function signature has evidence param
        let ops = module.ops(&ctx);
        let f = arena_func::Func::from_op(&ctx, ops[0]).unwrap();
        let f_ty = f.r#type(&ctx);
        let f_data = ctx.types.get(f_ty);
        assert_eq!(f_data.params.len(), 3); // return + evidence + i32

        // Verify entry block has 2 args
        let body = f.body(&ctx);
        let entry = ctx.region(body).blocks[0];
        assert_eq!(ctx.block_args(entry).len(), 2);
    }
}
