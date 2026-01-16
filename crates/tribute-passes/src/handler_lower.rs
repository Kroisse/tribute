//! Handler lowering pass for ability system.
//!
//! This pass transforms ability dialect operations to continuation dialect operations:
//!
//! ## Transformations
//!
//! 1. `tribute.handle` -> `cont.push_prompt`
//! 2. `ability.perform` -> `cont.shift` (with evidence lookup)
//! 3. `ability.resume` -> `cont.resume`
//! 4. `ability.abort` -> `cont.drop`
//!
//! ## Design
//!
//! Per `new-plans/implementation.md`:
//!
//! ```text
//! tribute.handle  → push_prompt(tag, body)
//! ability.perform → shift(tag, |k| handler(k))
//! ```
//!
//! The prompt tag connects `push_prompt` with corresponding `shift` operations.
//! Handler dispatch uses evidence to find the correct handler at runtime.

use std::sync::atomic::{AtomicU32, Ordering};

use tribute_ir::dialect::{ability, tribute, tribute_rt};
#[cfg(test)]
use trunk_ir::Attribute;
use trunk_ir::DialectType;
#[cfg(test)]
use trunk_ir::dialect::arith;
use trunk_ir::dialect::{cont, core, func};
#[cfg(test)]
use trunk_ir::rewrite::RewriteContext;
use trunk_ir::rewrite::{
    ConversionError, ConversionTarget, OpAdaptor, PatternApplicator, RewritePattern, RewriteResult,
    TypeConverter,
};
use trunk_ir::{Block, BlockId, DialectOp, IdVec, Operation, Region};

/// Lower handler operations from ability dialect to cont dialect.
///
/// This is the main entry point for the handler lowering pass.
/// Returns an error if any `ability.*` operations remain after conversion.
#[salsa::tracked]
pub fn lower_handlers<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
) -> Result<core::Module<'db>, ConversionError> {
    let converter = TypeConverter::new()
        .add_conversion(|db, ty| {
            tribute_rt::Int::from_type(db, ty).map(|_| core::I32::new(db).as_type())
        })
        .add_conversion(|db, ty| {
            tribute_rt::Nat::from_type(db, ty).map(|_| core::I32::new(db).as_type())
        })
        .add_conversion(|db, ty| {
            tribute_rt::Bool::from_type(db, ty).map(|_| core::I::<1>::new(db).as_type())
        })
        .add_conversion(|db, ty| {
            tribute_rt::Float::from_type(db, ty).map(|_| core::F64::new(db).as_type())
        });

    // Note: tribute.handle → cont.push_prompt is now handled by tribute_to_scf.
    // This pass only handles ability.* → cont.* transformations.
    // Verify all ability.* ops are converted after the pass
    let target = ConversionTarget::new().illegal_dialect("ability");

    let applicator = PatternApplicator::new(converter)
        .add_pattern(LowerPerformPattern::new())
        .add_pattern(LowerContinuationCallPattern) // Must come before LowerResumePattern
        .add_pattern(LowerResumePattern)
        .add_pattern(LowerAbortPattern);

    Ok(applicator.apply(db, module, target)?.module)
}

// === Prompt Tag Generation ===
//
// IMPORTANT: This is a placeholder implementation.
//
// The current approach of generating independent tags for push_prompt and shift
// is INCORRECT for actual runtime semantics - they need matching tags to work.
//
// In the full implementation, tags will come from evidence lookup:
//   1. `tribute.handle` installs a handler and creates an evidence marker
//   2. `ability.perform` looks up the evidence to find the matching prompt tag
//
// The global tag generator here is temporary scaffolding that will be replaced
// once evidence-based handler dispatch is implemented. It allows the pass to
// produce valid IR structure for testing the lowering logic.

/// Generator for unique prompt tags.
///
/// NOTE: This is placeholder infrastructure. In the final implementation,
/// prompt tags will be derived from evidence markers, not generated independently.
struct PromptTagGenerator {
    next_id: AtomicU32,
}

impl PromptTagGenerator {
    fn new() -> Self {
        Self {
            next_id: AtomicU32::new(0),
        }
    }

    fn fresh(&self) -> u32 {
        // Relaxed ordering is sufficient - we only need uniqueness, not ordering
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }
}

// Global prompt tag generator (placeholder - see note above)
static PROMPT_TAG_GEN: std::sync::LazyLock<PromptTagGenerator> =
    std::sync::LazyLock::new(PromptTagGenerator::new);

fn fresh_prompt_tag() -> u32 {
    PROMPT_TAG_GEN.fresh()
}

// === Pattern: Lower tribute.handle to cont.push_prompt ===

#[allow(dead_code)]
struct LowerPromptPattern;

#[allow(dead_code)]
impl LowerPromptPattern {
    fn new() -> Self {
        Self
    }
}

impl<'db> RewritePattern<'db> for LowerPromptPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        // Match: tribute.handle (has body and arms regions)
        let handle_op = match tribute::Handle::from_operation(db, *op) {
            Ok(h) => h,
            Err(_) => return RewriteResult::Unchanged,
        };

        // Generate fresh prompt tag
        let tag = fresh_prompt_tag();

        // Get the body region
        let body = handle_op.body(db);

        // Get the arms region and convert to handlers region
        let arms = handle_op.arms(db);
        let handlers = lower_handler_arms(db, arms);

        // Create cont.push_prompt with body and handlers
        // Note: The body may contain ability.perform ops that will be
        // transformed by LowerPerformPattern in a subsequent pass iteration
        let location = op.location(db);
        let result_ty = op
            .results(db)
            .first()
            .copied()
            .unwrap_or_else(|| *core::Nil::new(db));

        let new_op = cont::push_prompt(db, location, result_ty, tag, body, handlers);

        RewriteResult::Replace(new_op.as_operation())
    }
}

/// Lower tribute.arm operations from arms region to handlers region.
///
/// Each tribute.arm in the arms region becomes a block in the handlers region.
/// The pattern region determines whether it's a "done" or "suspend" handler:
/// - handler_done pattern -> first handler (done handler)
/// - handler_suspend pattern -> subsequent handlers (suspend handlers)
///
/// Handler arm bodies may contain references to pattern variables (result, args, k).
/// These are transformed to use cont.get_continuation and cont.get_shift_value.
#[allow(dead_code)]
fn lower_handler_arms<'db>(db: &'db dyn salsa::Database, arms: Region<'db>) -> Region<'db> {
    let location = arms.location(db);

    // Collect handler blocks from tribute.arm operations
    let mut handler_blocks = Vec::new();

    for block in arms.blocks(db).iter() {
        for arm_op in block.operations(db).iter() {
            if let Ok(arm) = tribute::Arm::from_operation(db, *arm_op) {
                // Get the arm's body region - this becomes a handler block
                let arm_body = arm.body(db);

                // Copy the body blocks to handlers
                // The body contains the handler code with tribute.yield at the end
                for body_block in arm_body.blocks(db).iter() {
                    handler_blocks.push(*body_block);
                }
            }
        }
    }

    // If no handlers found, create empty region
    if handler_blocks.is_empty() {
        let empty_block = Block::new(db, BlockId::fresh(), location, IdVec::new(), IdVec::new());
        return Region::new(db, location, IdVec::from(vec![empty_block]));
    }

    Region::new(db, location, IdVec::from(handler_blocks))
}

// === Pattern: Lower ability.perform to cont.shift ===

struct LowerPerformPattern;

impl LowerPerformPattern {
    fn new() -> Self {
        Self
    }
}

impl<'db> RewritePattern<'db> for LowerPerformPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        // Match: ability.perform
        let perform_op = match ability::Perform::from_operation(db, *op) {
            Ok(p) => p,
            Err(_) => return RewriteResult::Unchanged,
        };

        // Get operation attributes
        let ability_ref = perform_op.ability_ref(db);
        let op_name = perform_op.op(db);

        // Get args from adaptor (remapped values)
        let args: Vec<_> = adaptor.operands().iter().copied().collect();

        // For now, we create a cont.shift with a placeholder tag.
        // In the full implementation, the tag would come from evidence lookup:
        //   let marker = ev.get(ABILITY_ID)
        //   let tag = marker.prompt()
        //
        // For the initial implementation, we use a static tag based on
        // the ability reference. The handler matching will be resolved
        // at the push_prompt level.
        let tag = fresh_prompt_tag();

        let location = op.location(db);

        // Get the result type from ability.perform to pass to cont.shift.
        // This is the type that will be returned when the continuation is resumed.
        let result_ty = op
            .results(db)
            .first()
            .copied()
            .unwrap_or_else(|| *core::Nil::new(db));

        // TODO: Create handler region with actual handler logic.
        // The handler region should contain the code that runs when the continuation
        // is captured. In the full implementation, this will be populated based on
        // the handler patterns from tribute.case matching on the ability operation.
        // For now, we create an empty placeholder region.
        let empty_block = Block::new(db, BlockId::fresh(), location, IdVec::new(), IdVec::new());
        let handler_region = Region::new(db, location, IdVec::from(vec![empty_block]));

        // Create cont.shift with all attributes
        // Note: op_idx is not stored in IR; it will be computed from op_name
        // during WASM lowering.
        let shift_op = cont::shift(
            db,
            location,
            args,
            result_ty,
            tag,
            ability_ref,
            op_name,
            handler_region,
        );

        RewriteResult::Replace(shift_op.as_operation())
    }
}

// === Pattern: Lower ability.resume to cont.resume ===

struct LowerResumePattern;

impl<'db> RewritePattern<'db> for LowerResumePattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        // Match: ability.resume
        if ability::Resume::from_operation(db, *op).is_err() {
            return RewriteResult::Unchanged;
        }

        // Get operands from adaptor (remapped values)
        let continuation = adaptor.operand(0).expect("resume requires continuation");
        let value = adaptor.operand(1).expect("resume requires value");

        let location = op.location(db);
        let result_ty = op
            .results(db)
            .first()
            .copied()
            .unwrap_or_else(|| *core::Nil::new(db));

        // Create cont.resume with remapped operands
        let new_op = cont::resume(db, location, continuation, value, result_ty);

        RewriteResult::Replace(new_op.as_operation())
    }
}

// === Pattern: Lower ability.abort to cont.drop ===

struct LowerAbortPattern;

impl<'db> RewritePattern<'db> for LowerAbortPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        // Match: ability.abort
        if ability::Abort::from_operation(db, *op).is_err() {
            return RewriteResult::Unchanged;
        }

        // Get operand from adaptor (remapped value)
        let continuation = adaptor.operand(0).expect("abort requires continuation");
        let location = op.location(db);

        // Create cont.drop with remapped continuation
        let new_op = cont::drop(db, location, continuation);

        RewriteResult::Replace(new_op.as_operation())
    }
}

// === Pattern: Lower func.call_indirect with continuation callee to ability.resume ===

struct LowerContinuationCallPattern;

impl<'db> RewritePattern<'db> for LowerContinuationCallPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        // Match: func.call_indirect
        if func::CallIndirect::from_operation(db, *op).is_err() {
            return RewriteResult::Unchanged;
        }

        tracing::debug!(
            "LowerContinuationCallPattern: found func.call_indirect, operand_type(0) = {:?}",
            adaptor
                .operand_type(0)
                .map(|ty| format!("{}.{}", ty.dialect(db), ty.name(db)))
        );

        // Check if the callee (operand 0) has a continuation type
        let callee_ty = match adaptor.operand_type(0) {
            Some(ty) => ty,
            None => return RewriteResult::Unchanged,
        };

        // Try to match cont.continuation type
        if cont::Continuation::from_type(db, callee_ty).is_none() {
            tracing::debug!(
                "LowerContinuationCallPattern: callee type is not continuation: {}.{}",
                callee_ty.dialect(db),
                callee_ty.name(db)
            );
            return RewriteResult::Unchanged;
        }

        // Get the continuation (callee) and value (first argument) from adaptor
        let continuation = adaptor.operand(0).expect("call_indirect requires callee");
        let value = adaptor
            .operand(1)
            .unwrap_or_else(|| panic!("continuation call requires value argument"));

        let location = op.location(db);
        let result_ty = op
            .results(db)
            .first()
            .copied()
            .unwrap_or_else(|| *core::Nil::new(db));

        // Create ability.resume which will be further lowered to cont.resume
        let new_op = ability::resume(db, location, continuation, value, result_ty);

        RewriteResult::Replace(new_op.as_operation())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use trunk_ir::{Location, PathId, Span, Symbol, Value, ValueDef, idvec};

    fn test_location(db: &dyn salsa::Database) -> Location<'_> {
        let path = PathId::new(db, "test.trb".to_owned());
        Location::new(path, Span::new(0, 0))
    }

    /// Test helper: builds prompt op and applies the lowering pattern.
    /// Returns (dialect, name) of the result.
    #[salsa::tracked]
    fn lower_prompt_test(db: &dyn salsa::Database) -> (Symbol, Symbol) {
        let location = test_location(db);

        // Create empty body region
        let body_block = Block::new(db, BlockId::fresh(), location, IdVec::new(), idvec![]);
        let body = Region::new(db, location, idvec![body_block]);

        // Create empty arms region (fused handler syntax)
        let arms_block = Block::new(db, BlockId::fresh(), location, IdVec::new(), idvec![]);
        let arms = Region::new(db, location, idvec![arms_block]);

        let result_ty = *core::Nil::new(db);
        let handle_op = tribute::handle(db, location, result_ty, body, arms).as_operation();

        // Apply the pattern
        let pattern = LowerPromptPattern::new();
        let ctx = RewriteContext::new();
        let type_converter = TypeConverter::new();
        let adaptor = OpAdaptor::new(
            handle_op,
            handle_op.operands(db).clone(),
            vec![],
            &ctx,
            &type_converter,
        );
        let result = pattern.match_and_rewrite(db, &handle_op, &adaptor);

        match result {
            RewriteResult::Replace(new_op) => (new_op.dialect(db), new_op.name(db)),
            _ => panic!("Expected Replace result"),
        }
    }

    /// Test helper: builds resume op and applies the lowering pattern.
    /// Returns (dialect, name) of the result.
    #[salsa::tracked]
    fn lower_resume_test(db: &dyn salsa::Database) -> (Symbol, Symbol) {
        let location = test_location(db);

        // Create a dummy continuation value and value to resume with
        let cont_ty = *core::Nil::new(db);
        let val_ty = *core::I32::new(db);

        // Use arith.const for test values
        let cont_const = arith::r#const(db, location, cont_ty, Attribute::IntBits(0));
        let val_const = arith::r#const(db, location, val_ty, Attribute::IntBits(42));

        let cont_val = Value::new(db, ValueDef::OpResult(cont_const.as_operation()), 0);
        let val_val = Value::new(db, ValueDef::OpResult(val_const.as_operation()), 0);

        // Create ability.resume
        let resume_op = ability::resume(db, location, cont_val, val_val, val_ty);

        // Apply the pattern
        let op = resume_op.as_operation();
        let ctx = RewriteContext::new();
        let type_converter = TypeConverter::new();
        let adaptor = OpAdaptor::new(op, op.operands(db).clone(), vec![], &ctx, &type_converter);
        let result = LowerResumePattern.match_and_rewrite(db, &op, &adaptor);

        match result {
            RewriteResult::Replace(new_op) => (new_op.dialect(db), new_op.name(db)),
            _ => panic!("Expected Replace result"),
        }
    }

    /// Test helper: builds abort op and applies the lowering pattern.
    /// Returns (dialect, name) of the result.
    #[salsa::tracked]
    fn lower_abort_test(db: &dyn salsa::Database) -> (Symbol, Symbol) {
        let location = test_location(db);

        // Create a dummy continuation value
        let cont_ty = *core::Nil::new(db);
        let cont_const = arith::r#const(db, location, cont_ty, Attribute::IntBits(0));
        let cont_val = Value::new(db, ValueDef::OpResult(cont_const.as_operation()), 0);

        // Create ability.abort
        let abort_op = ability::abort(db, location, cont_val);

        // Apply the pattern
        let op = abort_op.as_operation();
        let ctx = RewriteContext::new();
        let type_converter = TypeConverter::new();
        let adaptor = OpAdaptor::new(op, op.operands(db).clone(), vec![], &ctx, &type_converter);
        let result = LowerAbortPattern.match_and_rewrite(db, &op, &adaptor);

        match result {
            RewriteResult::Replace(new_op) => (new_op.dialect(db), new_op.name(db)),
            _ => panic!("Expected Replace result"),
        }
    }

    /// Test helper: builds perform op and applies the lowering pattern.
    /// Returns (dialect, name) of the result.
    #[salsa::tracked]
    fn lower_perform_test(db: &dyn salsa::Database) -> (Symbol, Symbol) {
        let location = test_location(db);

        // Create ability.perform with an ability reference
        let ability_ref = *core::AbilityRefType::simple(db, Symbol::new("State"));
        let op_name = Symbol::new("get");

        let perform_op = ability::perform(
            db,
            location,
            std::iter::empty::<Value>(),
            *core::I32::new(db),
            ability_ref,
            op_name,
        )
        .as_operation();

        // Apply the pattern
        let pattern = LowerPerformPattern::new();
        let ctx = RewriteContext::new();
        let type_converter = TypeConverter::new();
        let adaptor = OpAdaptor::new(
            perform_op,
            perform_op.operands(db).clone(),
            vec![],
            &ctx,
            &type_converter,
        );
        let result = pattern.match_and_rewrite(db, &perform_op, &adaptor);

        match result {
            RewriteResult::Replace(new_op) => (new_op.dialect(db), new_op.name(db)),
            _ => panic!("Expected Replace result"),
        }
    }

    #[salsa_test]
    fn test_lower_prompt(db: &salsa::DatabaseImpl) {
        let (dialect, name) = lower_prompt_test(db);
        assert_eq!(dialect, Symbol::new("cont"));
        assert_eq!(name, Symbol::new("push_prompt"));
    }

    #[salsa_test]
    fn test_lower_perform(db: &salsa::DatabaseImpl) {
        let (dialect, name) = lower_perform_test(db);
        assert_eq!(dialect, Symbol::new("cont"));
        assert_eq!(name, Symbol::new("shift"));
    }

    #[salsa_test]
    fn test_lower_resume(db: &salsa::DatabaseImpl) {
        let (dialect, name) = lower_resume_test(db);
        assert_eq!(dialect, Symbol::new("cont"));
        assert_eq!(name, Symbol::new("resume"));
    }

    #[salsa_test]
    fn test_lower_abort(db: &salsa::DatabaseImpl) {
        let (dialect, name) = lower_abort_test(db);
        assert_eq!(dialect, Symbol::new("cont"));
        assert_eq!(name, Symbol::new("drop"));
    }
}
