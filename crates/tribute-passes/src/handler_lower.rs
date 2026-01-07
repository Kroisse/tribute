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

use tribute_ir::dialect::{ability, tribute};
use trunk_ir::dialect::{cont, core};
#[cfg(test)]
use trunk_ir::rewrite::RewriteContext;
use trunk_ir::rewrite::{OpAdaptor, PatternApplicator, RewritePattern, RewriteResult};
use trunk_ir::{Attribute, Block, BlockId, DialectOp, IdVec, Operation, Region};

/// Lower handler operations from ability dialect to cont dialect.
///
/// This is the main entry point for the handler lowering pass.
#[salsa::tracked]
pub fn lower_handlers<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
) -> core::Module<'db> {
    let applicator = PatternApplicator::new()
        .add_pattern(LowerPromptPattern::new())
        .add_pattern(LowerPerformPattern::new())
        .add_pattern(LowerResumePattern)
        .add_pattern(LowerAbortPattern);

    applicator.apply(db, module).module
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

struct LowerPromptPattern;

impl LowerPromptPattern {
    fn new() -> Self {
        Self
    }
}

impl RewritePattern for LowerPromptPattern {
    fn match_and_rewrite<'db>(
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

        // Generate fresh prompt tag
        let tag = fresh_prompt_tag();

        // Get the body region
        let body = handle_op.body(db);

        // Create cont.push_prompt with the same body
        // Note: The body may contain ability.perform ops that will be
        // transformed by LowerPerformPattern in a subsequent pass iteration
        let location = op.location(db);
        let result_ty = op
            .results(db)
            .first()
            .copied()
            .unwrap_or_else(|| *core::Nil::new(db));

        let new_op = cont::push_prompt(db, location, result_ty, tag, body);

        RewriteResult::Replace(new_op.as_operation())
    }
}

// === Pattern: Lower ability.perform to cont.shift ===

struct LowerPerformPattern;

impl LowerPerformPattern {
    fn new() -> Self {
        Self
    }
}

impl RewritePattern for LowerPerformPattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        // Match: ability.perform
        let perform_op = match ability::Perform::from_operation(db, *op) {
            Ok(p) => p,
            Err(_) => return RewriteResult::Unchanged,
        };

        // Get operation attributes
        let _ability_ref = perform_op.ability_ref(db);
        let op_name = perform_op.op(db);
        let _args = perform_op.args(db);

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

        // Create cont.shift with typed helper function.
        // The op_idx attribute is set to 0 as a placeholder here; it will be resolved
        // during handler dispatch based on the order of handler arms.
        let shift_op = cont::shift(db, location, vec![], result_ty, tag, 0, handler_region);

        // Add the op_name attribute for handler dispatch
        // (not part of the dialect definition but needed for current implementation)
        // TODO: In full implementation, op_idx would be pre-computed from ability definition.
        let op_with_name = Operation::of(db, location, shift_op.dialect(db), shift_op.name(db))
            .operands(shift_op.operands(db).clone())
            .results(shift_op.results(db).clone())
            .attr("tag", Attribute::IntBits(tag as u64))
            .attr("op_idx", Attribute::IntBits(0))
            .attr("op_name", Attribute::Symbol(op_name))
            .region(handler_region)
            .build();

        RewriteResult::Replace(op_with_name)
    }
}

// === Pattern: Lower ability.resume to cont.resume ===

struct LowerResumePattern;

impl RewritePattern for LowerResumePattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        // Match: ability.resume
        let resume_op = match ability::Resume::from_operation(db, *op) {
            Ok(r) => r,
            Err(_) => return RewriteResult::Unchanged,
        };

        let continuation = resume_op.continuation(db);
        let value = resume_op.value(db);

        let location = op.location(db);
        let result_ty = op
            .results(db)
            .first()
            .copied()
            .unwrap_or_else(|| *core::Nil::new(db));

        // Create cont.resume with the same operands
        let new_op = cont::resume(db, location, continuation, value, result_ty);

        RewriteResult::Replace(new_op.as_operation())
    }
}

// === Pattern: Lower ability.abort to cont.drop ===

struct LowerAbortPattern;

impl RewritePattern for LowerAbortPattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        // Match: ability.abort
        let abort_op = match ability::Abort::from_operation(db, *op) {
            Ok(a) => a,
            Err(_) => return RewriteResult::Unchanged,
        };

        let continuation = abort_op.continuation(db);
        let location = op.location(db);

        // Create cont.drop with the continuation
        let new_op = cont::drop(db, location, continuation);

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

        let handle_op = Operation::of_name(db, location, "tribute.handle")
            .result(*core::Nil::new(db))
            .region(body)
            .build();

        // Apply the pattern
        let pattern = LowerPromptPattern::new();
        let ctx = RewriteContext::new();
        let adaptor = OpAdaptor::new(handle_op, handle_op.operands(db).clone(), &ctx);
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

        // Create dummy operations to get values from
        let cont_op = Operation::of_name(db, location, "test.cont")
            .result(cont_ty)
            .build();
        let val_op = Operation::of_name(db, location, "test.val")
            .result(val_ty)
            .build();

        let cont_val = Value::new(db, ValueDef::OpResult(cont_op), 0);
        let val_val = Value::new(db, ValueDef::OpResult(val_op), 0);

        // Create ability.resume
        let resume_op = ability::resume(db, location, cont_val, val_val, val_ty);

        // Apply the pattern
        let op = resume_op.as_operation();
        let ctx = RewriteContext::new();
        let adaptor = OpAdaptor::new(op, op.operands(db).clone(), &ctx);
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
        let cont_op = Operation::of_name(db, location, "test.cont")
            .result(cont_ty)
            .build();
        let cont_val = Value::new(db, ValueDef::OpResult(cont_op), 0);

        // Create ability.abort
        let abort_op = ability::abort(db, location, cont_val);

        // Apply the pattern
        let op = abort_op.as_operation();
        let ctx = RewriteContext::new();
        let adaptor = OpAdaptor::new(op, op.operands(db).clone(), &ctx);
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

        let perform_op = Operation::of_name(db, location, "ability.perform")
            .attr("ability_ref", Attribute::Type(ability_ref))
            .attr("op", Attribute::Symbol(op_name))
            .result(*core::I32::new(db))
            .build();

        // Apply the pattern
        let pattern = LowerPerformPattern::new();
        let ctx = RewriteContext::new();
        let adaptor = OpAdaptor::new(perform_op, perform_op.operands(db).clone(), &ctx);
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
