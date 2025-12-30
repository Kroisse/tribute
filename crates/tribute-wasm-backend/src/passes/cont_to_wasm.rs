//! Lower continuation dialect to wasm dialect.
//!
//! This pass converts delimited continuation operations to WebAssembly using
//! yield bubbling (for WasmGC without native stack switching support).
//!
//! ## Yield Bubbling Strategy
//!
//! Per `new-plans/implementation.md`, WasmGC without stack switching uses
//! yield bubbling: each effectful call checks for yield state and propagates
//! upward.
//!
//! ```text
//! fn effectful_call(ev: *const Evidence) -> Result<T, Yield> {
//!     let result = do_operation(ev)
//!     if is_yielding() {
//!         return Yield(build_continuation())
//!     }
//!     Ok(result)
//! }
//! ```
//!
//! ## Transformations
//!
//! 1. `cont.push_prompt` -> yield-checking block wrapper
//! 2. `cont.shift` -> set yield globals, build continuation, return
//! 3. `cont.resume` -> reset yield state, invoke continuation
//! 4. `cont.drop` -> deallocate continuation (GC handles this)
//!
//! ## Global State
//!
//! The runtime uses global variables to track yield state:
//! - `$yield_state`: i32 (0 = normal, 1 = yielding)
//! - `$yield_tag`: i32 (prompt tag being yielded to)
//! - `$yield_cont`: anyref (captured continuation, GC-managed)
//! - `$yield_value`: anyref (value passed with shift)

use trunk_ir::dialect::cont;
use trunk_ir::dialect::core::Module;
use trunk_ir::rewrite::{PatternApplicator, RewritePattern, RewriteResult};
use trunk_ir::{Attribute, DialectOp, Operation, Symbol};

/// Global variable indices for yield state.
/// These will be allocated in the module preamble.
pub struct YieldGlobals {
    pub state_idx: u32,
    pub tag_idx: u32,
    pub cont_idx: u32,
    pub value_idx: u32,
}

impl Default for YieldGlobals {
    fn default() -> Self {
        Self::new()
    }
}

impl YieldGlobals {
    /// Create yield globals starting at the given base index.
    ///
    /// The base should be set to the number of existing globals in the module
    /// to avoid index conflicts.
    pub fn with_base(base: u32) -> Self {
        Self {
            state_idx: base,
            tag_idx: base + 1,
            cont_idx: base + 2,
            value_idx: base + 3,
        }
    }

    /// Create yield globals starting at index 0.
    ///
    /// Use `with_base()` if the module has existing globals.
    pub fn new() -> Self {
        Self::with_base(0)
    }
}

/// Analysis result for continuation lowering.
#[salsa::tracked]
pub struct ContAnalysis<'db> {
    /// Whether the module uses any continuation operations
    pub has_continuations: bool,
    /// Number of prompt tags used
    pub prompt_count: u32,
}

/// Analyze module for continuation operations.
#[salsa::tracked]
pub fn analyze_continuations<'db>(
    db: &'db dyn salsa::Database,
    module: Module<'db>,
) -> ContAnalysis<'db> {
    let mut has_continuations = false;
    let mut prompt_count = 0u32;

    // Scan all operations recursively, including those inside function bodies
    fn scan_region<'db>(
        db: &'db dyn salsa::Database,
        region: &trunk_ir::Region<'db>,
        has_cont: &mut bool,
        prompt_count: &mut u32,
    ) {
        for block in region.blocks(db).iter() {
            for op in block.operations(db).iter() {
                let dialect = op.dialect(db);
                if dialect == cont::DIALECT_NAME() {
                    *has_cont = true;
                    // Count unique prompt tags
                    if op.name(db) == cont::PUSH_PROMPT() {
                        *prompt_count = prompt_count.saturating_add(1);
                    }
                }
                // Recursively scan nested regions (e.g., function bodies)
                for nested_region in op.regions(db).iter() {
                    scan_region(db, nested_region, has_cont, prompt_count);
                }
            }
        }
    }

    let body = module.body(db);
    scan_region(db, &body, &mut has_continuations, &mut prompt_count);

    ContAnalysis::new(db, has_continuations, prompt_count)
}

/// Lower continuation dialect to wasm dialect.
pub fn lower<'db>(db: &'db dyn salsa::Database, module: Module<'db>) -> Module<'db> {
    let analysis = analyze_continuations(db, module);

    if !analysis.has_continuations(db) {
        return module;
    }

    PatternApplicator::new()
        .add_pattern(PushPromptPattern)
        .add_pattern(ShiftPattern)
        .add_pattern(ResumePattern)
        .add_pattern(DropPattern)
        .apply(db, module)
        .module
}

/// Pattern for `cont.push_prompt` -> yield-checking block
struct PushPromptPattern;

impl RewritePattern for PushPromptPattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
    ) -> RewriteResult<'db> {
        let Ok(_push_prompt) = cont::PushPrompt::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let location = op.location(db);

        // Get the prompt tag
        let tag = op
            .attributes(db)
            .get(&Symbol::new("tag"))
            .cloned()
            .unwrap_or(Attribute::IntBits(0));

        // For now, create a wasm.block that wraps the body
        // Full implementation will add yield state checking
        //
        // TODO: Full yield bubbling implementation:
        // 1. Execute body in a loop
        // 2. After body, check $yield_state global
        // 3. If yielding and tag matches, invoke handler
        // 4. If yielding and tag doesn't match, propagate
        // 5. If not yielding, return result
        let new_op = Operation::of_name(db, location, "wasm.block")
            .attr("label", tag)
            .results(op.results(db).clone())
            .regions(op.regions(db).clone())
            .build();

        RewriteResult::Replace(new_op)
    }
}

/// Pattern for `cont.shift` -> set yield state and build continuation
struct ShiftPattern;

impl RewritePattern for ShiftPattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
    ) -> RewriteResult<'db> {
        let Ok(_shift) = cont::Shift::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let location = op.location(db);

        // Get the prompt tag
        let _tag = op
            .attributes(db)
            .get(&Symbol::new("tag"))
            .cloned()
            .unwrap_or(Attribute::IntBits(0));

        // For now, create wasm.unreachable as placeholder
        // Full implementation will:
        // 1. Capture current frame into continuation struct
        // 2. Set $yield_state = 1
        // 3. Set $yield_tag = tag
        // 4. Set $yield_cont = captured_continuation
        // 5. Return/unwind to nearest push_prompt
        //
        // TODO: Continuation capture requires:
        // - Identifying live locals at this point
        // - Creating a reentry point (resume dispatch)
        // - Building the continuation struct
        let new_op = Operation::of_name(db, location, "wasm.unreachable").build();

        RewriteResult::Replace(new_op)
    }
}

/// Pattern for `cont.resume` -> invoke continuation
struct ResumePattern;

impl RewritePattern for ResumePattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
    ) -> RewriteResult<'db> {
        let Ok(resume) = cont::Resume::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let location = op.location(db);
        let _continuation = resume.continuation(db);
        let _value = resume.value(db);

        // For now, create wasm.unreachable as placeholder
        // Full implementation will:
        // 1. Reset $yield_state = 0
        // 2. Extract resume function from continuation
        // 3. Call resume function with value and captured env
        let new_op = Operation::of_name(db, location, "wasm.unreachable").build();

        RewriteResult::Replace(new_op)
    }
}

/// Pattern for `cont.drop` -> deallocate continuation
struct DropPattern;

impl RewritePattern for DropPattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
    ) -> RewriteResult<'db> {
        let Ok(drop_op) = cont::Drop::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let location = op.location(db);
        let continuation = drop_op.continuation(db);

        // In WasmGC, continuations are GC-managed, so "drop" just
        // removes the reference. We use wasm.drop to pop the value.
        let new_op = Operation::of_name(db, location, "wasm.drop")
            .operand(continuation)
            .build();

        RewriteResult::Replace(new_op)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use trunk_ir::DialectType;
    use trunk_ir::dialect::core;
    use trunk_ir::{Block, BlockId, IdVec, Location, PathId, Region, Span, idvec};

    fn test_location(db: &dyn salsa::Database) -> Location<'_> {
        let path = PathId::new(db, "file:///test.trb".to_owned());
        Location::new(path, Span::new(0, 0))
    }

    #[salsa::tracked]
    fn make_module_with_push_prompt(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();

        // Create empty body for push_prompt
        let body_block = Block::new(db, BlockId::fresh(), location, IdVec::new(), idvec![]);
        let body_region = Region::new(db, location, idvec![body_block]);

        let push_prompt = Operation::of_name(db, location, "cont.push_prompt")
            .attr("tag", Attribute::IntBits(42))
            .result(i32_ty)
            .region(body_region)
            .build();

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![push_prompt],
        );
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa::tracked]
    fn lower_and_check_dialect(db: &dyn salsa::Database, module: Module<'_>) -> String {
        let lowered = lower(db, module);
        let body = lowered.body(db);
        let ops = body.blocks(db)[0].operations(db);
        ops.first().map(|op| op.full_name(db)).unwrap_or_default()
    }

    #[salsa_test]
    fn test_push_prompt_to_wasm_block(db: &salsa::DatabaseImpl) {
        let module = make_module_with_push_prompt(db);
        let op_name = lower_and_check_dialect(db, module);
        assert_eq!(op_name, "wasm.block");
    }

    #[salsa_test]
    fn test_analyze_continuations(db: &salsa::DatabaseImpl) {
        let module = make_module_with_push_prompt(db);
        let analysis = analyze_continuations(db, module);
        assert!(analysis.has_continuations(db));
        assert_eq!(analysis.prompt_count(db), 1);
    }

    #[salsa::tracked]
    fn make_empty_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let block = Block::new(db, BlockId::fresh(), location, idvec![], idvec![]);
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa_test]
    fn test_empty_module_no_continuations(db: &salsa::DatabaseImpl) {
        let module = make_empty_module(db);
        let analysis = analyze_continuations(db, module);
        assert!(!analysis.has_continuations(db));
    }
}
