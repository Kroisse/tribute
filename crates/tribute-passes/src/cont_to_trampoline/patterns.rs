use std::collections::HashSet;
use std::rc::Rc;

use tribute_ir::dialect::tribute_rt;
use trunk_ir::dialect::core::{self};
use trunk_ir::dialect::func::{self};
use trunk_ir::dialect::{arith, cont, scf, trampoline};
use trunk_ir::ir::BlockBuilder;
use trunk_ir::rewrite::{PatternRewriter, RewritePattern};
use trunk_ir::{DialectOp, DialectType, IdVec, Location, Operation, Region, Symbol, Type, Value};

use super::get_region_result_value;

// ============================================================================
// Pattern: Lower cont.resume
// ============================================================================

pub(crate) struct LowerResumePattern;

impl<'db> RewritePattern<'db> for LowerResumePattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        rewriter: &mut PatternRewriter<'db, '_>,
    ) -> bool {
        let Ok(_) = cont::Resume::from_operation(db, *op) else {
            return false;
        };

        let location = op.location(db);
        let i32_ty = core::I32::new(db).as_type();
        let anyref_ty = tribute_rt::Any::new(db).as_type();

        let operands = rewriter.operands();
        let continuation = operands
            .first()
            .copied()
            .expect("resume requires continuation");
        let value = operands.get(1).copied();

        let mut ops = Vec::new();

        // === 1. Reset yield state ===
        ops.push(trampoline::reset_yield_state(db, location).as_operation());

        // === 2. Get resume_fn from continuation (i32 table index) ===
        let get_resume_fn = trampoline::continuation_get(
            db,
            location,
            continuation,
            i32_ty,
            Symbol::new("resume_fn"),
        );
        let resume_fn_val = get_resume_fn.as_operation().result(db, 0);
        ops.push(get_resume_fn.as_operation());

        // === 3. Get state from continuation ===
        let get_state = trampoline::continuation_get(
            db,
            location,
            continuation,
            anyref_ty,
            Symbol::new("state"),
        );
        let state_val = get_state.as_operation().result(db, 0);
        ops.push(get_state.as_operation());

        // === 4. Build resume wrapper ===
        let wrapper_ty = trampoline::ResumeWrapper::new(db).as_type();
        let resume_value = value.unwrap_or(state_val);

        let wrapper_op =
            trampoline::build_resume_wrapper(db, location, state_val, resume_value, wrapper_ty);
        let wrapper_val = wrapper_op.as_operation().result(db, 0);
        ops.push(wrapper_op.as_operation());

        // === 5. Call resume function ===
        // Resume functions take (evidence, wrapper) where wrapper is the resume_wrapper type.
        let evidence_ty = tribute_ir::dialect::ability::evidence_adt_type(db);
        let null_evidence =
            trunk_ir::dialect::adt::ref_null(db, location, evidence_ty, evidence_ty);
        ops.push(null_evidence.as_operation());
        let evidence_val = null_evidence.as_operation().result(db, 0);

        let step_ty = trampoline::Step::new(db).as_type();
        let call_op = func::call_indirect(
            db,
            location,
            resume_fn_val,
            IdVec::from(vec![evidence_val, wrapper_val]),
            step_ty,
        );
        ops.push(call_op.as_operation());

        let last = ops.pop().unwrap();
        for o in ops {
            rewriter.insert_op(o);
        }
        rewriter.replace_op(last);
        true
    }
}

// ============================================================================
// Pattern: Update func.call result type for effectful functions
// ============================================================================

/// Pattern that updates func.call to effectful functions to return Step type.
/// This handles calls inside scf.if/scf.loop regions that weren't processed
/// by truncate_after_shift (which only processes function entry blocks).
pub(crate) struct UpdateEffectfulCallResultTypePattern {
    pub(crate) effectful_funcs: Rc<HashSet<Symbol>>,
}

impl<'db> RewritePattern<'db> for UpdateEffectfulCallResultTypePattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        rewriter: &mut PatternRewriter<'db, '_>,
    ) -> bool {
        // Only handle func.call operations
        let Ok(call) = func::Call::from_operation(db, *op) else {
            return false;
        };

        let callee = call.callee(db);

        // Skip if not an effectful function
        if !self.effectful_funcs.contains(&callee) {
            return false;
        }

        // Skip if already returns Step
        let step_ty = trampoline::Step::new(db).as_type();
        if op
            .results(db)
            .first()
            .is_some_and(|ty| trampoline::Step::from_type(db, *ty).is_some())
        {
            return false;
        }

        // Skip if no results
        if op.results(db).is_empty() {
            return false;
        }

        let location = op.location(db);
        let original_result_ty = op.results(db).first().copied().unwrap();

        tracing::debug!(
            "UpdateEffectfulCallResultTypePattern: updating call to {} from {} to Step",
            callee,
            original_result_ty.name(db)
        );

        // Create new call with Step result type
        let new_call = Operation::new(
            db,
            location,
            op.dialect(db),
            op.name(db),
            rewriter.operands().clone(),
            IdVec::from(vec![step_ty]),
            op.attributes(db).clone(),
            op.regions(db).clone(),
            op.successors(db).clone(),
        );

        // Return the new call directly without adding a cast.
        // The Step value should propagate up through the effectful context.
        // Any downstream operations that need the original type will need to handle
        // Step unpacking appropriately.
        rewriter.replace_op(new_call);
        true
    }
}

// ============================================================================
// Pattern: Update scf.if result type when branches contain effectful calls
// ============================================================================

/// Pattern that updates scf.if result type to Step when its branches
/// contain calls that return Step (effectful calls that have been transformed).
///
/// NOTE: This pattern only changes the result type. The internal scf.yield
/// operations are updated by UpdateScfYieldToStepPattern which runs after
/// the nested regions have been processed by PatternApplicator.
pub(crate) struct UpdateScfIfResultTypePattern;

impl<'db> RewritePattern<'db> for UpdateScfIfResultTypePattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        rewriter: &mut PatternRewriter<'db, '_>,
    ) -> bool {
        // Only handle scf.if operations
        if scf::If::from_operation(db, *op).is_err() {
            return false;
        }

        // Skip if no results
        if op.results(db).is_empty() {
            return false;
        }

        // Skip if already returns Step
        let step_ty = trampoline::Step::new(db).as_type();
        if op
            .results(db)
            .first()
            .is_some_and(|ty| trampoline::Step::from_type(db, *ty).is_some())
        {
            return false;
        }

        // Check if any branch contains operations that return Step type
        // This includes func.call that have been transformed to return Step
        let branches_have_step_ops = op.regions(db).iter().any(|region| {
            region.blocks(db).iter().any(|block| {
                block.operations(db).iter().any(|branch_op| {
                    // Check if any operation produces Step type result
                    branch_op
                        .results(db)
                        .first()
                        .is_some_and(|ty| trampoline::Step::from_type(db, *ty).is_some())
                })
            })
        });

        if !branches_have_step_ops {
            return false;
        }

        let loc = op.location(db);
        tracing::debug!(
            "UpdateScfIfResultTypePattern: updating scf.if result to Step at {:?}",
            loc
        );

        // Only change the result type - let PatternApplicator handle nested regions
        // and UpdateScfYieldToStepPattern handle the yield updates
        let new_if = Operation::new(
            db,
            op.location(db),
            op.dialect(db),
            op.name(db),
            op.operands(db).clone(),
            IdVec::from(vec![step_ty]),
            op.attributes(db).clone(),
            op.regions(db).clone(), // Keep original regions - will be processed recursively
            op.successors(db).clone(),
        );

        rewriter.replace_op(new_if);
        true
    }
}

/// Pattern that updates scf.yield to yield Step value when it's inside
/// a block that contains effectful operations returning Step.
///
/// This pattern is applied after UpdateEffectfulCallResultTypePattern has
/// changed func.call results to Step, so we can find the actual Step values.
pub(crate) struct UpdateScfYieldToStepPattern;

impl<'db> RewritePattern<'db> for UpdateScfYieldToStepPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        rewriter: &mut PatternRewriter<'db, '_>,
    ) -> bool {
        // Only handle scf.yield operations
        let Ok(_yield_op) = scf::Yield::from_operation(db, *op) else {
            return false;
        };

        // Skip if already yielding Step
        if let Some(operand) = rewriter.operands().first()
            && let Some(ty) = rewriter.get_raw_value_type(db, *operand)
            && trampoline::Step::from_type(db, ty).is_some()
        {
            return false;
        }

        // Check the context to find if there's a Step value we should yield instead
        // We look at the remapped operand types to see what the current value types are
        let current_operands = rewriter.operands();
        if current_operands.is_empty() {
            return false;
        }

        // Get the actual type of the yielded value
        let yielded_value = current_operands[0];
        let Some(yielded_ty) = rewriter.get_raw_value_type(db, yielded_value) else {
            return false;
        };

        // If the yielded value is already Step, no change needed
        if trampoline::Step::from_type(db, yielded_ty).is_some() {
            return false;
        }

        // Check if we can find the Step value through cast chain
        // The yielded value might be a cast result where the input is Step
        if let Some(step_value) = find_step_source(db, yielded_value, rewriter) {
            tracing::debug!(
                "UpdateScfYieldToStepPattern: updating scf.yield to yield Step at {:?}",
                op.location(db)
            );
            let new_yield = scf::r#yield(db, op.location(db), vec![step_value]);
            rewriter.replace_op(new_yield.as_operation());
            return true;
        }

        false
    }
}

/// Find the Step source value by tracing through the value map and cast chain.
fn find_step_source<'db>(
    db: &'db dyn salsa::Database,
    value: Value<'db>,
    rewriter: &PatternRewriter<'db, '_>,
) -> Option<Value<'db>> {
    // Check if the value itself is Step (after remapping)
    if let Some(ty) = rewriter.get_raw_value_type(db, value)
        && trampoline::Step::from_type(db, ty).is_some()
    {
        return Some(value);
    }

    // Check if the value definition is a cast from Step
    if let trunk_ir::ValueDef::OpResult(defining_op) = value.def(db) {
        // If this is a cast, check the input
        if let Ok(cast) = core::UnrealizedConversionCast::from_operation(db, defining_op) {
            let input = cast.value(db);
            if let Some(input_ty) = rewriter.get_raw_value_type(db, input)
                && trampoline::Step::from_type(db, input_ty).is_some()
            {
                return Some(input);
            }
        }

        // Check if the defining op produces Step
        if let Some(ty) = defining_op.results(db).first()
            && trampoline::Step::from_type(db, *ty).is_some()
        {
            return Some(defining_op.result(db, 0));
        }
    }

    None
}

// ============================================================================
// Pattern: Lower cont.push_prompt
// ============================================================================

pub(crate) struct LowerPushPromptPattern;

impl<'db> RewritePattern<'db> for LowerPushPromptPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        rewriter: &mut PatternRewriter<'db, '_>,
    ) -> bool {
        let push_prompt = match cont::PushPrompt::from_operation(db, *op) {
            Ok(p) => p,
            Err(_) => return false,
        };

        let location = op.location(db);
        let i32_ty = core::I32::new(db).as_type();
        let step_ty = trampoline::Step::new(db).as_type();
        let tag = push_prompt.tag(db);

        // Get the body region (already recursively transformed by applicator)
        let body = push_prompt.body(db);

        // Get body result value
        let body_result = get_region_result_value(db, &body);

        let mut all_ops = Vec::new();

        // Add all body operations
        if let Some(body_block) = body.blocks(db).first() {
            for body_op in body_block.operations(db).iter() {
                all_ops.push(*body_op);
            }
        }

        // check_yield
        let check_yield = trampoline::check_yield(db, location, i32_ty);
        let is_yielding = check_yield.as_operation().result(db, 0);
        all_ops.push(check_yield.as_operation());

        // Build yield handling branches (tag is passed as u32, will be converted to constant inside region)
        let then_region = build_yield_then_branch(db, location, tag, step_ty);
        let else_region = build_yield_else_branch(db, location, body_result, step_ty);

        // scf.if for yield check
        let if_op = scf::r#if(db, location, is_yielding, step_ty, then_region, else_region);
        all_ops.push(if_op.as_operation());

        let last = all_ops.pop().unwrap();
        for o in all_ops {
            rewriter.insert_op(o);
        }
        rewriter.replace_op(last);
        true
    }
}

fn build_yield_then_branch<'db>(
    db: &'db dyn salsa::Database,
    location: Location<'db>,
    tag: u32,
    step_ty: Type<'db>,
) -> Region<'db> {
    let i32_ty = core::I32::new(db).as_type();
    let mut builder = BlockBuilder::new(db, location);

    // Create tag constant inside the region
    let tag_const = builder.op(arith::r#const(
        db,
        location,
        i32_ty,
        trunk_ir::Attribute::IntBits(tag as u64),
    ));
    let tag_val = tag_const.result(db);

    let cont_ty = trampoline::Continuation::new(db).as_type();
    let get_cont = builder.op(trampoline::get_yield_continuation(db, location, cont_ty));
    let cont_val = get_cont.result(db);

    let step_shift = builder.op(trampoline::step_shift(
        db, location, tag_val, cont_val, step_ty, 0,
    ));
    builder.op(scf::r#yield(db, location, vec![step_shift.result(db)]));

    let block = builder.build();
    Region::new(db, location, IdVec::from(vec![block]))
}

fn build_yield_else_branch<'db>(
    db: &'db dyn salsa::Database,
    location: Location<'db>,
    body_result: Option<Value<'db>>,
    step_ty: Type<'db>,
) -> Region<'db> {
    use trunk_ir::ValueDef;

    let mut builder = BlockBuilder::new(db, location);

    let step_value = if let Some(result) = body_result {
        // Check if result is already a Step (from effectful call or scf.if returning Step)
        let is_step = match result.def(db) {
            ValueDef::OpResult(def_op) => {
                // Check if this operation produces Step
                def_op
                    .results(db)
                    .first()
                    .map(|ty| trampoline::Step::from_type(db, *ty).is_some())
                    .unwrap_or(false)
            }
            ValueDef::BlockArg(_) => false,
        };

        if is_step {
            // Already a Step, return it directly
            result
        } else {
            // Wrap the value in step_done
            let step_done = builder.op(trampoline::step_done(db, location, result, step_ty));
            step_done.result(db)
        }
    } else {
        // No body result - create a step_done with zero value (unit placeholder)
        let zero = builder.op(arith::Const::i32(db, location, 0));
        let step_done = builder.op(trampoline::step_done(
            db,
            location,
            zero.result(db),
            step_ty,
        ));
        step_done.result(db)
    };

    builder.op(scf::r#yield(db, location, vec![step_value]));

    let block = builder.build();
    Region::new(db, location, IdVec::from(vec![block]))
}
