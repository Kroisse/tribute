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

use trunk_ir::dialect::{ability, core, func};
use trunk_ir::rewrite::{PatternApplicator, RewritePattern, RewriteResult};
use trunk_ir::{
    Block, BlockId, DialectOp, DialectType, IdVec, Operation, QualifiedName, Region, Type, Value,
    ValueDef,
};

/// Insert evidence parameters for effectful functions.
///
/// This is the main entry point for the evidence insertion pass.
#[salsa::tracked]
pub fn insert_evidence<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
) -> core::Module<'db> {
    // First pass: collect all effectful functions
    let effectful_fns = collect_effectful_functions(db, &module);

    if effectful_fns.is_empty() {
        return module;
    }

    // Apply patterns with knowledge of effectful functions
    let applicator = PatternApplicator::new()
        .add_pattern(AddEvidenceParamPattern::new(effectful_fns.clone()))
        .add_pattern(PassEvidenceToCallsPattern::new(effectful_fns));

    applicator.apply(db, module).module
}

/// Collect all function names that are effectful.
fn collect_effectful_functions<'db>(
    db: &'db dyn salsa::Database,
    module: &core::Module<'db>,
) -> HashSet<QualifiedName> {
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

/// Check if a function type has a non-empty effect row.
fn is_effectful_type<'db>(db: &'db dyn salsa::Database, ty: Type<'db>) -> bool {
    let Some(func_ty) = core::Func::from_type(db, ty) else {
        return false;
    };

    let Some(effect) = func_ty.effect(db) else {
        return false;
    };

    let Some(row) = core::EffectRowType::from_type(db, effect) else {
        return false;
    };

    !row.is_empty(db)
}

/// Pattern: Add evidence parameter to effectful function signatures.
struct AddEvidenceParamPattern {
    effectful_fns: HashSet<QualifiedName>,
}

impl AddEvidenceParamPattern {
    fn new(effectful_fns: HashSet<QualifiedName>) -> Self {
        Self { effectful_fns }
    }
}

impl RewritePattern for AddEvidenceParamPattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
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
        // Create pure function type (effects are now explicit via evidence)
        let new_func_ty = core::Func::new(db, new_params.clone(), result_ty);

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
                    new_args.push(ev_ty);
                    new_args.extend(old_args.iter().copied());

                    Block::new(
                        db,
                        BlockId::fresh(),
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
        let new_op = Operation::of_name(db, location, "func.func")
            .attr("sym_name", trunk_ir::Attribute::QualifiedName(func_name))
            .attr("type", trunk_ir::Attribute::Type(*new_func_ty))
            .region(new_body)
            .build();

        RewriteResult::Replace(new_op)
    }
}

/// Pattern: Pass evidence through call sites.
struct PassEvidenceToCallsPattern {
    effectful_fns: HashSet<QualifiedName>,
}

impl PassEvidenceToCallsPattern {
    fn new(effectful_fns: HashSet<QualifiedName>) -> Self {
        Self { effectful_fns }
    }
}

impl RewritePattern for PassEvidenceToCallsPattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
    ) -> RewriteResult<'db> {
        // Match: func.call
        let call_op = match func::Call::from_operation(db, *op) {
            Ok(c) => c,
            Err(_) => return RewriteResult::Unchanged,
        };

        let callee = call_op.callee(db);

        // Check if callee is effectful
        if !self.effectful_fns.contains(&callee) {
            return RewriteResult::Unchanged;
        }

        // Already has evidence argument? (check first arg type)
        let args = call_op.args(db);
        if !args.is_empty()
            && let Some(first_arg_ty) = get_value_type(db, args[0])
            && ability::EvidencePtr::from_type(db, first_arg_ty).is_some()
        {
            return RewriteResult::Unchanged;
        }

        // Find the evidence value from the enclosing function's first block arg
        let Some(ev_value) = find_evidence_value(db, op) else {
            // No evidence available - this call is in a pure context
            // This shouldn't happen for valid programs, but we handle it gracefully
            return RewriteResult::Unchanged;
        };

        // Create new call with evidence as first argument
        let mut new_args: Vec<Value<'db>> = vec![ev_value];
        new_args.extend(args.iter().copied());

        let location = op.location(db);
        let result_ty = op
            .results(db)
            .first()
            .copied()
            .unwrap_or_else(|| *core::Nil::new(db));

        let new_call = func::call(db, location, new_args, result_ty, callee);

        RewriteResult::Replace(new_call.as_operation())
    }
}

/// Get the type of a value from its definition site.
fn get_value_type<'db>(db: &'db dyn salsa::Database, value: Value<'db>) -> Option<Type<'db>> {
    match value.def(db) {
        ValueDef::OpResult(op) => op.results(db).get(value.index(db)).copied(),
        ValueDef::BlockArg(_block_id) => {
            // For block args, we would need to find the Block from BlockId
            // and then look up the arg types. For now, return None.
            // TODO: Implement proper block arg type lookup
            None
        }
    }
}

/// Find the evidence value from the enclosing function.
///
/// This looks for the first block argument of type `core.evidence_ptr`
/// in the function containing this operation.
fn find_evidence_value<'db>(
    _db: &'db dyn salsa::Database,
    _op: &Operation<'db>,
) -> Option<Value<'db>> {
    // Walk up to find the enclosing func.func
    // For now, we use a heuristic: the operation should be inside a block
    // whose first argument is evidence_ptr

    // In a real implementation, we'd track this during the rewrite.
    // For now, we return None and handle it during a second pass.
    //
    // TODO: Implement proper evidence tracking through the rewrite context
    None
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
