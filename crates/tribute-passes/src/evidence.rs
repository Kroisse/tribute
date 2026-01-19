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

use crate::type_converter::generic_type_converter;
use tribute_ir::dialect::{ability, adt, tribute};
use trunk_ir::dialect::{core, func, wasm};
use trunk_ir::rewrite::{
    ConversionTarget, OpAdaptor, PatternApplicator, RewritePattern, RewriteResult,
};
use trunk_ir::{
    Block, BlockArg, DialectOp, DialectType, IdVec, Operation, Region, Symbol, Type, Value,
};

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
    let effectful_fns = collect_effectful_functions(db, &module);

    if effectful_fns.is_empty() {
        return module;
    }

    let converter = generic_type_converter();
    let target = ConversionTarget::new();
    PatternApplicator::new(converter)
        .add_pattern(AddEvidenceParamPattern::new(effectful_fns))
        .apply_partial(db, module, target)
        .module
}

/// Phase 2: Transform calls to pass evidence through.
///
/// This pass transforms call sites to pass evidence:
/// 1. Calls inside effectful functions pass the evidence parameter
/// 2. Calls inside tribute.handle bodies pass null evidence
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
    // Collect functions that are explicitly effectful (have effect row with abilities)
    let effectful_fns = collect_effectful_functions(db, &module);

    // Also collect functions that have evidence parameter (added by add_evidence_params or lambda_lift)
    // These functions need their internal calls transformed even if their type no longer shows effects
    let fns_with_evidence = collect_functions_with_evidence_param(db, &module);

    if effectful_fns.is_empty() && fns_with_evidence.is_empty() {
        return module;
    }

    // Combine both sets for the callee check - we need to pass evidence to calls to either type
    let all_effectful: HashSet<Symbol> = effectful_fns.union(&fns_with_evidence).copied().collect();

    let converter = generic_type_converter();
    let target = ConversionTarget::new();
    PatternApplicator::new(converter)
        .add_pattern(TransformCallsPattern::new(all_effectful.clone()))
        .add_pattern(TransformHandlerCallsPattern::new(all_effectful))
        .apply_partial(db, module, target)
        .module
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
                    if !params.is_empty()
                        && ability::EvidencePtr::from_type(db, params[0]).is_some()
                    {
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
/// The pass works in two phases via `PatternApplicator`:
/// 1. `AddEvidenceParamPattern`: Add evidence parameter to effectful function signatures
/// 2. `TransformCallsPattern`: Transform calls inside effectful functions to pass evidence
#[salsa::tracked]
pub fn insert_evidence<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
) -> core::Module<'db> {
    let module = add_evidence_params(db, module);
    transform_evidence_calls(db, module)
}

/// Transform calls in a block, returning the new block and whether any changes were made.
fn transform_calls_in_block<'db>(
    db: &'db dyn salsa::Database,
    block: &Block<'db>,
    ev_value: Value<'db>,
    effectful_fns: &HashSet<Symbol>,
) -> (Block<'db>, bool) {
    use std::collections::HashMap;

    let mut new_ops = Vec::new();
    let mut changed = false;
    let mut value_map: HashMap<Value<'db>, Value<'db>> = HashMap::new();

    for op in block.operations(db).iter() {
        // First, remap operands using the value map
        let remapped_operands: Vec<Value<'db>> = op
            .operands(db)
            .iter()
            .map(|v| *value_map.get(v).unwrap_or(v))
            .collect();

        // Check if this is a call to an effectful function
        if let Ok(call_op) = func::Call::from_operation(db, *op) {
            let callee = call_op.callee(db);
            if effectful_fns.contains(&callee) {
                // Check if evidence is already the first argument (to prevent re-adding)
                let first_arg = remapped_operands.first().copied();
                if first_arg == Some(ev_value) {
                    // Already has evidence as first arg, skip transformation
                    new_ops.push(*op);
                    continue;
                }

                // Add evidence as first argument
                let mut new_args: Vec<Value<'db>> = vec![ev_value];
                // Use remapped operands (which are the call args)
                new_args.extend(remapped_operands.iter().copied());

                let location = op.location(db);
                let result_ty = op
                    .results(db)
                    .first()
                    .copied()
                    .unwrap_or_else(|| *core::Nil::new(db));

                let new_call = func::call(db, location, new_args, result_ty, callee);
                let new_call_op = new_call.as_operation();

                // Map old result to new result
                if !op.results(db).is_empty() {
                    let old_result = op.result(db, 0);
                    let new_result = new_call_op.result(db, 0);
                    value_map.insert(old_result, new_result);
                }

                new_ops.push(new_call_op);
                changed = true;
                continue;
            }
        }

        // NOTE: Indirect calls (closure calls) are handled by closure_lower pass,
        // which has access to closure type information before lowering.
        // We don't handle them here because closure types have already been lowered
        // to adt.struct by the time this pass runs.

        // Recursively transform nested regions (e.g., in scf.if, tribute.case)
        let regions = op.regions(db);
        if !regions.is_empty() {
            let mut region_changed = false;
            let new_regions: IdVec<Region<'db>> = regions
                .iter()
                .map(|region| {
                    let (new_region, r_changed) =
                        transform_calls_in_region(db, region, ev_value, effectful_fns);
                    if r_changed {
                        region_changed = true;
                    }
                    new_region
                })
                .collect();

            if region_changed {
                changed = true;
                // Rebuild op with new regions and remapped operands
                let new_op = op
                    .modify(db)
                    .operands(IdVec::from(remapped_operands))
                    .regions(new_regions)
                    .build();
                // Map old results to new results
                for i in 0..op.results(db).len() {
                    let old_result = op.result(db, i);
                    let new_result = new_op.result(db, i);
                    value_map.insert(old_result, new_result);
                }
                new_ops.push(new_op);
                continue;
            }
        }

        // If operands were remapped or we have changes, rebuild the operation
        let operands_changed = op
            .operands(db)
            .iter()
            .zip(remapped_operands.iter())
            .any(|(old, new)| old != new);

        if operands_changed {
            let new_op = op
                .modify(db)
                .operands(IdVec::from(remapped_operands))
                .build();
            // Map old results to new results
            for i in 0..op.results(db).len() {
                let old_result = op.result(db, i);
                let new_result = new_op.result(db, i);
                value_map.insert(old_result, new_result);
            }
            new_ops.push(new_op);
        } else {
            new_ops.push(*op);
        }
    }

    let new_block = Block::new(
        db,
        block.id(db),
        block.location(db),
        block.args(db).clone(),
        new_ops.into_iter().collect(),
    );

    (new_block, changed)
}

/// Transform calls in a region, returning the new region and whether any changes were made.
fn transform_calls_in_region<'db>(
    db: &'db dyn salsa::Database,
    region: &Region<'db>,
    ev_value: Value<'db>,
    effectful_fns: &HashSet<Symbol>,
) -> (Region<'db>, bool) {
    let mut changed = false;
    let new_blocks: IdVec<Block<'db>> = region
        .blocks(db)
        .iter()
        .map(|block| {
            let (new_block, block_changed) =
                transform_calls_in_block(db, block, ev_value, effectful_fns);
            if block_changed {
                changed = true;
            }
            new_block
        })
        .collect();

    let new_region = Region::new(db, region.location(db), new_blocks);
    (new_region, changed)
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

/// Pattern: Add evidence parameter to effectful function signatures.
struct AddEvidenceParamPattern {
    effectful_fns: HashSet<Symbol>,
}

impl AddEvidenceParamPattern {
    fn new(effectful_fns: HashSet<Symbol>) -> Self {
        Self { effectful_fns }
    }
}

impl<'db> RewritePattern<'db> for AddEvidenceParamPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        // Match: func.func
        let func_op = match func::Func::from_operation(db, *op) {
            Ok(f) => f,
            Err(_) => return RewriteResult::Unchanged,
        };

        let func_name = func_op.sym_name(db);

        // Check if this function is effectful
        if !self.effectful_fns.contains(&func_name) {
            return RewriteResult::Unchanged;
        }

        // Already has evidence parameter? (check for re-application)
        let func_ty = func_op.r#type(db);
        let Some(core_func) = core::Func::from_type(db, func_ty) else {
            return RewriteResult::Unchanged;
        };

        let params = core_func.params(db);
        if !params.is_empty() {
            // Check if first param is already evidence_ptr
            if ability::EvidencePtr::from_type(db, params[0]).is_some() {
                return RewriteResult::Unchanged;
            }
        }

        // Create evidence type
        let ev_ty = *ability::EvidencePtr::new(db);

        // Create new function type with evidence as first parameter
        let mut new_params = IdVec::with_capacity(params.len() + 1);
        new_params.push(ev_ty);
        new_params.extend(params.iter().copied());

        let result_ty = core_func.result(db);
        // Preserve effect annotation - cont_to_trampoline needs it to identify effectful functions
        let effect = core_func.effect(db);
        let new_func_ty = core::Func::with_effect(db, new_params.clone(), result_ty, effect);

        // Rebuild body region with evidence as first block argument
        let old_body = func_op.body(db);
        let old_blocks = old_body.blocks(db);

        let new_blocks: IdVec<Block<'db>> = old_blocks
            .iter()
            .enumerate()
            .map(|(i, block)| {
                if i == 0 {
                    // Entry block: add evidence as first argument
                    let old_args = block.args(db);
                    let mut new_args = IdVec::with_capacity(old_args.len() + 1);
                    new_args.push(BlockArg::of_type(db, ev_ty));
                    new_args.extend(old_args.iter().copied());

                    // Preserve original BlockId to maintain any inter-block references
                    Block::new(
                        db,
                        block.id(db),
                        block.location(db),
                        new_args,
                        block.operations(db).clone(),
                    )
                } else {
                    *block
                }
            })
            .collect();

        let new_body = Region::new(db, old_body.location(db), new_blocks);
        let location = op.location(db);

        // Build new func.func operation
        let new_func = func::func(db, location, func_name, *new_func_ty, new_body);

        RewriteResult::Replace(new_func.as_operation())
    }
}

/// Pattern: Transform calls to effectful functions inside effectful function bodies.
///
/// This pattern matches `func.func` operations for effectful functions and
/// transforms all calls to effectful functions within their bodies to pass
/// the evidence parameter.
struct TransformCallsPattern {
    effectful_fns: HashSet<Symbol>,
}

impl TransformCallsPattern {
    fn new(effectful_fns: HashSet<Symbol>) -> Self {
        Self { effectful_fns }
    }
}

impl<'db> RewritePattern<'db> for TransformCallsPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        // Match: func.func
        let func_op = match func::Func::from_operation(db, *op) {
            Ok(f) => f,
            Err(_) => return RewriteResult::Unchanged,
        };

        let func_name = func_op.sym_name(db);

        // Get evidence value from first block's first argument
        let body = func_op.body(db);
        let blocks = body.blocks(db);
        let Some(entry_block) = blocks.first() else {
            return RewriteResult::Unchanged;
        };

        let args = entry_block.args(db);
        if args.is_empty() {
            return RewriteResult::Unchanged;
        }

        // Check if this function has an evidence parameter.
        // Process if:
        // 1. It's in effectful_fns (explicitly effectful), OR
        // 2. Its first parameter is ability.evidence_ptr (e.g., lifted lambdas)
        let func_ty = func_op.r#type(db);
        let first_param_type =
            core::Func::from_type(db, func_ty).and_then(|ft| ft.params(db).first().copied());
        let has_evidence_param =
            first_param_type.is_some_and(|ty| ability::EvidencePtr::from_type(db, ty).is_some());

        if !self.effectful_fns.contains(&func_name) && !has_evidence_param {
            return RewriteResult::Unchanged;
        }

        // First arg should be evidence_ptr after Phase 1 transformation
        let ev_value = entry_block.arg(db, 0);

        // Transform calls in all blocks
        let mut changed = false;
        let new_blocks: IdVec<Block<'db>> = blocks
            .iter()
            .map(|block| {
                let (new_block, block_changed) =
                    transform_calls_in_block(db, block, ev_value, &self.effectful_fns);
                if block_changed {
                    changed = true;
                }
                new_block
            })
            .collect();

        if !changed {
            return RewriteResult::Unchanged;
        }

        // Rebuild function with transformed body
        let location = op.location(db);
        let new_body = Region::new(db, location, new_blocks);
        let func_ty = func_op.r#type(db);

        let new_func = func::func(db, location, func_name, func_ty, new_body);

        RewriteResult::Replace(new_func.as_operation())
    }
}

/// Pattern: Transform calls to effectful functions inside tribute.handle bodies.
///
/// This pattern matches `tribute.handle` operations and transforms all calls to
/// effectful functions within their body regions to pass a null evidence pointer.
/// The handler will provide the actual evidence at runtime via evidence lookup.
struct TransformHandlerCallsPattern {
    effectful_fns: HashSet<Symbol>,
}

impl TransformHandlerCallsPattern {
    fn new(effectful_fns: HashSet<Symbol>) -> Self {
        Self { effectful_fns }
    }
}

impl<'db> RewritePattern<'db> for TransformHandlerCallsPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        // Match: tribute.handle
        let handle_op = match tribute::Handle::from_operation(db, *op) {
            Ok(h) => h,
            Err(_) => return RewriteResult::Unchanged,
        };

        let location = op.location(db);
        let body_region = handle_op.body(db);
        let arms_region = handle_op.arms(db);
        let result_type = op
            .results(db)
            .first()
            .copied()
            .unwrap_or_else(|| core::Nil::new(db).as_type());

        // Create null evidence pointer for handler body calls
        let ev_ty = ability::EvidencePtr::new(db).as_type();
        let anyref_ty = wasm::Anyref::new(db).as_type();
        let null_ev_op = adt::ref_null(db, location, ev_ty, anyref_ty);
        let null_ev_value = null_ev_op.as_operation().result(db, 0);

        // Transform calls in body region
        let (new_body, body_changed) =
            transform_calls_in_region(db, &body_region, null_ev_value, &self.effectful_fns);

        if !body_changed {
            return RewriteResult::Unchanged;
        }

        // Rebuild tribute.handle with transformed body
        // We need to prepend the null_ev_op to the operations
        let mut result_ops = vec![null_ev_op.as_operation()];

        let new_handle = tribute::handle(db, location, result_type, new_body, arms_region);
        result_ops.push(new_handle.as_operation());

        RewriteResult::expand(result_ops)
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
