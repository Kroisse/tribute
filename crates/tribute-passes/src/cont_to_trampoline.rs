//! Lower cont dialect operations to trampoline dialect.
//!
//! This pass transforms continuation operations to trampoline operations:
//! - `cont.shift` → build_state + build_continuation + set_yield_state + step_shift
//! - `cont.resume` → reset_yield_state + continuation_get + build_resume_wrapper + call
//! - `cont.push_prompt` → trampoline yield check loop + dispatch
//! - `cont.handler_dispatch` → yield check + multi-arm dispatch
//! - `cont.get_continuation` → `trampoline.get_yield_continuation`
//! - `cont.get_shift_value` → `trampoline.get_yield_shift_value`
//! - `cont.get_done_value` → `trampoline.step_get(field="value")`
//! - `cont.drop` → pass through (handled later)
//!
//! This pass is backend-agnostic and should run after `tribute_to_cont`/`handler_lower`
//! and before `trampoline_to_adt`.

use std::sync::{LazyLock, Mutex};

use tribute_ir::dialect::adt;
use trunk_ir::dialect::core::{self, Module};
use trunk_ir::dialect::func::{self, Func};
use trunk_ir::dialect::{arith, cont, scf, trampoline, wasm};
use trunk_ir::ir::BlockBuilder;
use trunk_ir::rewrite::{
    OpAdaptor, PatternApplicator, RewritePattern, RewriteResult, TypeConverter,
};
use trunk_ir::{
    Block, DialectOp, DialectType, IdVec, Location, Operation, Region, Symbol, Type, Value,
};

// ============================================================================
// Public API
// ============================================================================

/// Lower cont dialect operations to trampoline dialect.
pub fn lower_cont_to_trampoline<'db>(
    db: &'db dyn salsa::Database,
    module: Module<'db>,
) -> Module<'db> {
    // Clear any previous state from prior compilations
    RESUME_FUNCTIONS.lock().unwrap().clear();
    *RESUME_COUNTER.lock().unwrap() = 0;

    let converter = TypeConverter::new();

    let applicator = PatternApplicator::new(converter)
        .add_pattern(LowerShiftPattern)
        .add_pattern(LowerResumePattern)
        .add_pattern(LowerGetContinuationPattern)
        .add_pattern(LowerGetShiftValuePattern)
        .add_pattern(LowerGetDoneValuePattern)
        .add_pattern(LowerPushPromptPattern)
        .add_pattern(LowerHandlerDispatchPattern);

    let result = applicator.apply(db, module);

    // Add generated resume functions to module
    let resume_funcs: Vec<Operation<'db>> = RESUME_FUNCTIONS.lock().unwrap().drain(..).collect();
    if resume_funcs.is_empty() {
        return result.module;
    }

    // Add resume functions to module body
    let body = result.module.body(db);
    let mut blocks: Vec<Block<'db>> = body.blocks(db).iter().copied().collect();
    if let Some(block) = blocks.first_mut() {
        let mut ops: Vec<Operation<'db>> = block.operations(db).iter().copied().collect();
        ops.extend(resume_funcs);
        *block = Block::new(
            db,
            block.id(db),
            block.location(db),
            block.args(db).clone(),
            IdVec::from(ops),
        );
    }

    let new_body = Region::new(db, body.location(db), IdVec::from(blocks));
    Module::create(
        db,
        result.module.location(db),
        result.module.name(db),
        new_body,
    )
}

// ============================================================================
// Resume Function Storage
// ============================================================================

/// Global storage for generated resume functions.
/// This is needed because RewritePattern doesn't have a way to add new top-level ops.
static RESUME_FUNCTIONS: LazyLock<Mutex<Vec<Operation<'static>>>> =
    LazyLock::new(|| Mutex::new(Vec::new()));

/// Counter for generating unique resume function names.
static RESUME_COUNTER: LazyLock<Mutex<u32>> = LazyLock::new(|| Mutex::new(0));

fn fresh_resume_name() -> Symbol {
    let mut counter = RESUME_COUNTER.lock().unwrap();
    let id = *counter;
    *counter += 1;
    Symbol::from_dynamic(&format!("__resume_{}", id))
}

/// Generate a unique state type name based on ability and operation info.
fn state_type_name(ability_name: Option<Symbol>, op_name: Option<Symbol>, tag: u32) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    if let Some(ability) = ability_name {
        ability.to_string().hash(&mut hasher);
    }
    if let Some(name) = op_name {
        name.to_string().hash(&mut hasher);
    }
    tag.hash(&mut hasher);

    let hash = hasher.finish();
    format!("__State_{:x}", hash & 0xFFFFFF)
}

// ============================================================================
// Pattern: Lower cont.shift
// ============================================================================

struct LowerShiftPattern;

impl<'db> RewritePattern<'db> for LowerShiftPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let shift_op = match cont::Shift::from_operation(db, *op) {
            Ok(s) => s,
            Err(_) => return RewriteResult::Unchanged,
        };

        let location = op.location(db);
        let tag = shift_op.tag(db);

        // Compute op_idx from ability_ref and op_name
        let ability_ref_type = shift_op.ability_ref(db);
        let ability_name =
            core::AbilityRefType::from_type(db, ability_ref_type).and_then(|ar| ar.name(db));
        let op_name = Some(shift_op.op_name(db));
        let op_idx = compute_op_idx(ability_name, op_name);

        let mut ops = Vec::new();

        // === 1. Build State Struct ===
        // For now, capture no locals (simplified implementation)
        // Create an empty ADT struct type with a unique name for the state.
        // When we capture locals, we'll add fields here.
        let state_name = Symbol::from_dynamic(&state_type_name(ability_name, op_name, tag));
        let state_adt_ty = adt::struct_type(db, state_name, vec![]);
        let state_op = trampoline::build_state(db, location, vec![], state_adt_ty, state_adt_ty);
        let state_val = state_op.as_operation().result(db, 0);
        ops.push(state_op.as_operation());

        // === 2. Get resume function reference ===
        let funcref_ty = wasm::Funcref::new(db).as_type();
        let resume_name = fresh_resume_name();

        // Generate resume function
        generate_resume_function(db, resume_name, location);

        let const_op = func::constant(db, location, funcref_ty, resume_name);
        let resume_fn_val = const_op.as_operation().result(db, 0);
        ops.push(const_op.as_operation());

        // === 3. Get shift value (the value passed to the effect operation) ===
        let shift_value_val = adaptor.operands().first().copied().unwrap_or(state_val);

        // === 4. Build Continuation ===
        let cont_ty = trampoline::Continuation::new(db).as_type();
        let cont_op = trampoline::build_continuation(
            db,
            location,
            resume_fn_val,
            state_val,
            shift_value_val,
            cont_ty,
            tag,
            op_idx,
        );
        let cont_val = cont_op.as_operation().result(db, 0);
        ops.push(cont_op.as_operation());

        // === 5. Set Yield State ===
        let set_yield_op = trampoline::set_yield_state(db, location, cont_val, tag, op_idx);
        ops.push(set_yield_op.as_operation());

        // === 6. Return Step::Shift ===
        let step_ty = trampoline::Step::new(db).as_type();
        let step_op = trampoline::step_shift(db, location, cont_val, step_ty, tag, op_idx);
        ops.push(step_op.as_operation());

        RewriteResult::expand(ops)
    }
}

/// Generate a resume function that just returns the resume value.
/// This is a simplified implementation - full implementation would capture/restore state.
fn generate_resume_function<'db>(
    db: &'db dyn salsa::Database,
    name: Symbol,
    location: Location<'db>,
) {
    let wrapper_ty = trampoline::ResumeWrapper::new(db).as_type();
    let step_ty = trampoline::Step::new(db).as_type();
    let anyref_ty = wasm::Anyref::new(db).as_type();

    let func_op = Func::build(
        db,
        location,
        name,
        IdVec::from(vec![wrapper_ty]),
        step_ty,
        |builder| {
            let wrapper_arg = builder.block_arg(db, 0);

            // Extract resume_value from wrapper (anyref type)
            let get_resume_value = builder.op(trampoline::resume_wrapper_get(
                db,
                location,
                wrapper_arg,
                anyref_ty,
                Symbol::new("resume_value"),
            ));
            let resume_value = get_resume_value.result(db);

            // Return Step.Done with resume_value
            let step_done = builder.op(trampoline::step_done(db, location, resume_value, step_ty));
            builder.op(func::r#return(db, location, Some(step_done.result(db))));
        },
    );

    // SAFETY: We're storing Operation<'db> as Operation<'static>, but this is safe
    // because we only access it within the same compilation unit where 'db is still valid.
    let func_static: Operation<'static> = unsafe { std::mem::transmute(func_op.as_operation()) };
    RESUME_FUNCTIONS.lock().unwrap().push(func_static);
}

// ============================================================================
// Pattern: Lower cont.resume
// ============================================================================

struct LowerResumePattern;

impl<'db> RewritePattern<'db> for LowerResumePattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(_) = cont::Resume::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let location = op.location(db);
        let funcref_ty = wasm::Funcref::new(db).as_type();
        let anyref_ty = wasm::Anyref::new(db).as_type();

        let operands = adaptor.operands();
        let continuation = operands
            .first()
            .copied()
            .expect("resume requires continuation");
        let value = operands.get(1).copied();

        let mut ops = Vec::new();

        // === 1. Reset yield state ===
        ops.push(trampoline::reset_yield_state(db, location).as_operation());

        // === 2. Get resume_fn from continuation ===
        let get_resume_fn = trampoline::continuation_get(
            db,
            location,
            continuation,
            funcref_ty,
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
        let step_ty = trampoline::Step::new(db).as_type();
        let call_op = func::call_indirect(
            db,
            location,
            resume_fn_val,
            IdVec::from(vec![wrapper_val]),
            step_ty,
        );
        ops.push(call_op.as_operation());

        RewriteResult::expand(ops)
    }
}

// ============================================================================
// Pattern: Lower cont.get_continuation
// ============================================================================

struct LowerGetContinuationPattern;

impl<'db> RewritePattern<'db> for LowerGetContinuationPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(_) = cont::GetContinuation::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let location = op.location(db);
        let result_type = op
            .results(db)
            .first()
            .copied()
            .unwrap_or_else(|| trampoline::Continuation::new(db).as_type());

        let trampoline_op = trampoline::get_yield_continuation(db, location, result_type);
        RewriteResult::Replace(trampoline_op.as_operation())
    }
}

// ============================================================================
// Pattern: Lower cont.get_shift_value
// ============================================================================

struct LowerGetShiftValuePattern;

impl<'db> RewritePattern<'db> for LowerGetShiftValuePattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(_) = cont::GetShiftValue::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let location = op.location(db);
        let result_type = op
            .results(db)
            .first()
            .copied()
            .unwrap_or_else(|| core::Ptr::new(db).as_type());

        let trampoline_op = trampoline::get_yield_shift_value(db, location, result_type);
        RewriteResult::Replace(trampoline_op.as_operation())
    }
}

// ============================================================================
// Pattern: Lower cont.get_done_value
// ============================================================================

struct LowerGetDoneValuePattern;

impl<'db> RewritePattern<'db> for LowerGetDoneValuePattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(_) = cont::GetDoneValue::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let location = op.location(db);
        let result_type = op
            .results(db)
            .first()
            .copied()
            .unwrap_or_else(|| core::Ptr::new(db).as_type());

        let step_value = adaptor
            .operands()
            .first()
            .copied()
            .expect("get_done_value requires step operand");

        let trampoline_op =
            trampoline::step_get(db, location, step_value, result_type, Symbol::new("value"));
        RewriteResult::Replace(trampoline_op.as_operation())
    }
}

// ============================================================================
// Pattern: Lower cont.push_prompt
// ============================================================================

struct LowerPushPromptPattern;

impl<'db> RewritePattern<'db> for LowerPushPromptPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let push_prompt = match cont::PushPrompt::from_operation(db, *op) {
            Ok(p) => p,
            Err(_) => return RewriteResult::Unchanged,
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

        // Build yield handling branches
        let then_region = build_yield_then_branch(db, location, tag, step_ty);
        let else_region = build_yield_else_branch(db, location, body_result, step_ty);

        // scf.if for yield check
        let if_op = scf::r#if(db, location, is_yielding, step_ty, then_region, else_region);
        all_ops.push(if_op.as_operation());

        RewriteResult::expand(all_ops)
    }
}

fn build_yield_then_branch<'db>(
    db: &'db dyn salsa::Database,
    location: Location<'db>,
    our_tag: u32,
    step_ty: Type<'db>,
) -> Region<'db> {
    let mut builder = BlockBuilder::new(db, location);

    let cont_ty = trampoline::Continuation::new(db).as_type();
    let get_cont = builder.op(trampoline::get_yield_continuation(db, location, cont_ty));
    let cont_val = get_cont.result(db);

    let step_shift = builder.op(trampoline::step_shift(
        db, location, cont_val, step_ty, our_tag, 0,
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
    let mut builder = BlockBuilder::new(db, location);

    let step_value = if let Some(result) = body_result {
        let step_done = builder.op(trampoline::step_done(db, location, result, step_ty));
        step_done.result(db)
    } else {
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

// ============================================================================
// Pattern: Lower cont.handler_dispatch
// ============================================================================

struct LowerHandlerDispatchPattern;

impl<'db> RewritePattern<'db> for LowerHandlerDispatchPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(_) = cont::HandlerDispatch::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let location = op.location(db);
        let i32_ty = core::I32::new(db).as_type();
        let step_ty = trampoline::Step::new(db).as_type();

        // Get the body region with multiple blocks
        let body_region = op
            .regions(db)
            .first()
            .cloned()
            .unwrap_or_else(|| Region::new(db, location, IdVec::new()));
        let blocks = body_region.blocks(db);

        // Block 0 = done case
        let done_region = if let Some(done_block) = blocks.first() {
            Region::new(db, location, IdVec::from(vec![*done_block]))
        } else {
            Region::new(db, location, IdVec::new())
        };

        // Block 1+ = suspend cases
        let suspend_region = if blocks.len() > 1 {
            Region::new(db, location, IdVec::from(vec![blocks[1]]))
        } else {
            let mut builder = BlockBuilder::new(db, location);
            let zero = builder.op(arith::Const::i32(db, location, 0));
            let step_done = builder.op(trampoline::step_done(
                db,
                location,
                zero.result(db),
                step_ty,
            ));
            builder.op(scf::r#yield(db, location, vec![step_done.result(db)]));
            Region::new(db, location, IdVec::from(vec![builder.build()]))
        };

        // check_yield
        let check_yield = trampoline::check_yield(db, location, i32_ty);
        let is_yielding = check_yield.as_operation().result(db, 0);

        // scf.if
        let if_op = scf::r#if(
            db,
            location,
            is_yielding,
            step_ty,
            suspend_region,
            done_region,
        );

        RewriteResult::expand(vec![check_yield.as_operation(), if_op.as_operation()])
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn get_region_result_value<'db>(
    db: &'db dyn salsa::Database,
    region: &Region<'db>,
) -> Option<Value<'db>> {
    let blocks = region.blocks(db);
    let last_block = blocks.last()?;
    let ops = last_block.operations(db);
    let last_op = ops.last()?;
    last_op.results(db).first().map(|_| last_op.result(db, 0))
}

fn compute_op_idx(ability_ref: Option<Symbol>, op_name: Option<Symbol>) -> u32 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();

    if let Some(ability) = ability_ref {
        ability.to_string().hash(&mut hasher);
    }
    if let Some(name) = op_name {
        name.to_string().hash(&mut hasher);
    }

    (hasher.finish() % 0x7FFFFFFF) as u32
}
