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

use std::collections::BTreeMap;

use trunk_ir::DialectType;
use trunk_ir::dialect::cont;
use trunk_ir::dialect::core::{self, Module};
use trunk_ir::dialect::wasm;
use trunk_ir::rewrite::{PatternApplicator, RewritePattern, RewriteResult};
use trunk_ir::{Attribute, DialectOp, Location, Operation, QualifiedName, Symbol, Type};

/// Global variable indices for yield state.
/// These will be allocated in the module preamble.
#[allow(dead_code)] // Fields will be used when implementing global get/set
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
    /// Information about each shift point, keyed by location
    pub shift_points: ShiftPointMap<'db>,
}

/// Map from shift location to its info.
/// Uses BTreeMap for deterministic ordering.
pub type ShiftPointMap<'db> = BTreeMap<LocationKey, ShiftPointInfo<'db>>;

/// Serializable location key for shift points.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, salsa::Update)]
pub struct LocationKey {
    /// Start byte of the span.
    pub start: usize,
    /// End byte of the span.
    pub end: usize,
}

impl LocationKey {
    fn from_location(loc: Location<'_>) -> Self {
        Self {
            start: loc.span.start,
            end: loc.span.end,
        }
    }
}

/// Information about a shift point for continuation capture.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub struct ShiftPointInfo<'db> {
    /// The prompt tag this shift corresponds to.
    pub tag: u64,
    /// The containing function's name (for generating resume function name).
    pub containing_func: Option<QualifiedName>,
    /// Index of this shift within its containing function (for unique naming).
    pub shift_index: u32,
    /// Live local variables at this shift point that need to be captured.
    /// Each entry is (name, type).
    pub live_locals: Vec<LiveLocal<'db>>,
    /// Generated state struct type name.
    pub state_type_name: Option<QualifiedName>,
}

/// A live local variable that needs to be captured at a shift point.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub struct LiveLocal<'db> {
    /// Variable name (if known from src.var).
    pub name: Option<Symbol>,
    /// Variable type.
    pub ty: Type<'db>,
}

/// Analyze module for continuation operations.
#[salsa::tracked]
pub fn analyze_continuations<'db>(
    db: &'db dyn salsa::Database,
    module: Module<'db>,
) -> ContAnalysis<'db> {
    let mut collector = ContinuationAnalyzer::new(db);
    collector.analyze_module(module);

    ContAnalysis::new(
        db,
        collector.has_continuations,
        collector.prompt_count,
        collector.shift_points,
    )
}

/// Collector for continuation analysis.
struct ContinuationAnalyzer<'db> {
    db: &'db dyn salsa::Database,
    has_continuations: bool,
    prompt_count: u32,
    shift_points: ShiftPointMap<'db>,
    /// Current containing function name (if inside a function).
    current_func: Option<QualifiedName>,
    /// Counter for shifts within current function.
    shift_counter: u32,
}

impl<'db> ContinuationAnalyzer<'db> {
    fn new(db: &'db dyn salsa::Database) -> Self {
        Self {
            db,
            has_continuations: false,
            prompt_count: 0,
            shift_points: BTreeMap::new(),
            current_func: None,
            shift_counter: 0,
        }
    }

    fn analyze_module(&mut self, module: Module<'db>) {
        let body = module.body(self.db);
        self.analyze_region(&body);
    }

    fn analyze_region(&mut self, region: &trunk_ir::Region<'db>) {
        for block in region.blocks(self.db).iter() {
            self.analyze_block(block);
        }
    }

    fn analyze_block(&mut self, block: &trunk_ir::Block<'db>) {
        for op in block.operations(self.db).iter() {
            self.analyze_operation(op);
        }
    }

    fn analyze_operation(&mut self, op: &Operation<'db>) {
        let dialect = op.dialect(self.db);
        let name = op.name(self.db);

        // Track function context
        if dialect == trunk_ir::dialect::func::DIALECT_NAME()
            && name == trunk_ir::dialect::func::FUNC()
        {
            if let Ok(func_op) = trunk_ir::dialect::func::Func::from_operation(self.db, *op) {
                let prev_func = self.current_func.take();
                let prev_counter = self.shift_counter;

                self.current_func = Some(func_op.sym_name(self.db));
                self.shift_counter = 0;

                // Analyze function body
                for region in op.regions(self.db).iter() {
                    self.analyze_region(region);
                }

                self.current_func = prev_func;
                self.shift_counter = prev_counter;
                return;
            }
        }

        // Check for continuation operations
        if dialect == cont::DIALECT_NAME() {
            self.has_continuations = true;

            if name == cont::PUSH_PROMPT() {
                self.prompt_count = self.prompt_count.saturating_add(1);
            } else if name == cont::SHIFT() {
                self.collect_shift_info(op);
            }
        }

        // Recursively analyze nested regions
        for region in op.regions(self.db).iter() {
            self.analyze_region(region);
        }
    }

    fn collect_shift_info(&mut self, op: &Operation<'db>) {
        let location = op.location(self.db);
        let location_key = LocationKey::from_location(location);

        // Extract tag from operation
        let tag = op
            .attributes(self.db)
            .get(&Symbol::new("tag"))
            .and_then(|attr| {
                if let Attribute::IntBits(v) = attr {
                    Some(*v)
                } else {
                    None
                }
            })
            .unwrap_or(0);

        // Generate state type name based on function and shift index
        let state_type_name = self.current_func.as_ref().map(|func_name| {
            let state_name = Symbol::from_dynamic(&format!(
                "{}_state_{}",
                func_name.name(),
                self.shift_counter
            ));
            QualifiedName::new(func_name.as_parent(), state_name)
        });

        let info = ShiftPointInfo {
            tag,
            containing_func: self.current_func.clone(),
            shift_index: self.shift_counter,
            live_locals: Vec::new(), // TODO: Implement live local analysis
            state_type_name,
        };

        self.shift_points.insert(location_key, info);
        self.shift_counter += 1;
    }
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
        let i32_ty = core::I32::new(db).as_type();
        let anyref_ty = wasm::Anyref::new(db).as_type();

        // Get the prompt tag
        let tag = op
            .attributes(db)
            .get(&Symbol::new("tag"))
            .cloned()
            .unwrap_or(Attribute::IntBits(0));

        let tag_value = match tag {
            Attribute::IntBits(v) => v,
            _ => 0,
        };

        // NOTE: The shift operation has a handler region that contains the code
        // to run when the continuation is captured. Currently this region is dropped
        // because full continuation capture is not yet implemented.
        //
        // TODO: Full continuation capture requires:
        // - Identifying live locals at this point (Phase 2)
        // - Creating state struct to capture locals
        // - Building continuation struct with resume function
        // - Setting $yield_cont to the actual continuation

        let mut ops = Vec::new();

        // 1. Set $yield_state = 1
        let const_1 = wasm::i32_const(db, location, i32_ty, Attribute::IntBits(1));
        let const_1_val = const_1.as_operation().result(db, 0);
        ops.push(const_1.as_operation());
        ops.push(wasm::global_set(db, location, const_1_val, YIELD_STATE_IDX).as_operation());

        // 2. Set $yield_tag = tag
        let const_tag = wasm::i32_const(db, location, i32_ty, Attribute::IntBits(tag_value));
        let const_tag_val = const_tag.as_operation().result(db, 0);
        ops.push(const_tag.as_operation());
        ops.push(wasm::global_set(db, location, const_tag_val, YIELD_TAG_IDX).as_operation());

        // 3. Set $yield_cont = null (placeholder - full impl will capture actual continuation)
        let null_cont = wasm::ref_null(
            db,
            location,
            anyref_ty,
            Attribute::Symbol(Symbol::new("any")),
        );
        let null_cont_val = null_cont.as_operation().result(db, 0);
        ops.push(null_cont.as_operation());
        ops.push(wasm::global_set(db, location, null_cont_val, YIELD_CONT_IDX).as_operation());

        // 4. Set $yield_value = null (placeholder - full impl will pass shift value)
        let null_value = wasm::ref_null(
            db,
            location,
            anyref_ty,
            Attribute::Symbol(Symbol::new("any")),
        );
        let null_value_val = null_value.as_operation().result(db, 0);
        ops.push(null_value.as_operation());
        ops.push(wasm::global_set(db, location, null_value_val, YIELD_VALUE_IDX).as_operation());

        // 5. Return to unwind stack (will bubble up to push_prompt)
        ops.push(wasm::r#return(db, location, None).as_operation());

        RewriteResult::Expand(ops)
    }
}

// Yield global indices (must match order in lower_wasm.rs module_preamble_ops)
const YIELD_STATE_IDX: u32 = 0;
const YIELD_TAG_IDX: u32 = 1;
const YIELD_CONT_IDX: u32 = 2;
const YIELD_VALUE_IDX: u32 = 3;

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
    use trunk_ir::{Block, BlockId, IdVec, Location, PathId, Region, Span, Value, ValueDef, idvec};

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

    // === Tests for other patterns ===

    #[salsa::tracked]
    fn make_module_with_shift(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);

        // Create empty handler region for shift
        let handler_block = Block::new(db, BlockId::fresh(), location, IdVec::new(), idvec![]);
        let handler_region = Region::new(db, location, idvec![handler_block]);

        let shift = Operation::of_name(db, location, "cont.shift")
            .attr("tag", Attribute::IntBits(42))
            .region(handler_region)
            .build();

        let block = Block::new(db, BlockId::fresh(), location, idvec![], idvec![shift]);
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa::tracked]
    fn lower_shift_and_check(db: &dyn salsa::Database, module: Module<'_>) -> Vec<String> {
        let lowered = lower(db, module);
        let body = lowered.body(db);
        let ops = body.blocks(db)[0].operations(db);
        ops.iter().map(|op| op.full_name(db)).collect()
    }

    #[salsa_test]
    fn test_shift_expands_to_yield_sequence(db: &salsa::DatabaseImpl) {
        let module = make_module_with_shift(db);
        let op_names = lower_shift_and_check(db, module);

        // Shift should expand to: i32_const, global_set (x4), ref_null (x2), return
        // First op: i32_const (for yield_state = 1)
        assert_eq!(op_names.first(), Some(&"wasm.i32_const".to_string()));
        // Last op: return (to unwind stack)
        assert_eq!(op_names.last(), Some(&"wasm.return".to_string()));
        // Should have multiple operations (9 total: 2 const + 2 global_set + 2 ref_null + 2 global_set + 1 return)
        assert_eq!(op_names.len(), 9);
    }

    #[salsa::tracked]
    fn make_module_with_resume(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();

        // Create dummy continuation and value operands
        let cont_op = Operation::of_name(db, location, "test.dummy_cont")
            .result(i32_ty)
            .build();
        let cont_val = Value::new(db, ValueDef::OpResult(cont_op), 0);

        let val_op = Operation::of_name(db, location, "test.dummy_val")
            .result(i32_ty)
            .build();
        let value = Value::new(db, ValueDef::OpResult(val_op), 0);

        let resume = Operation::of_name(db, location, "cont.resume")
            .operand(cont_val)
            .operand(value)
            .result(i32_ty)
            .build();

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![cont_op, val_op, resume],
        );
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa::tracked]
    fn lower_resume_and_check(db: &dyn salsa::Database, module: Module<'_>) -> String {
        let lowered = lower(db, module);
        let body = lowered.body(db);
        let ops = body.blocks(db)[0].operations(db);
        // The resume op is the third operation (after two dummy ops)
        ops.get(2).map(|op| op.full_name(db)).unwrap_or_default()
    }

    #[salsa_test]
    fn test_resume_to_wasm_unreachable(db: &salsa::DatabaseImpl) {
        let module = make_module_with_resume(db);
        let op_name = lower_resume_and_check(db, module);
        assert_eq!(op_name, "wasm.unreachable");
    }

    #[salsa::tracked]
    fn make_module_with_drop(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();

        // Create dummy continuation operand
        let cont_op = Operation::of_name(db, location, "test.dummy_cont")
            .result(i32_ty)
            .build();
        let cont_val = Value::new(db, ValueDef::OpResult(cont_op), 0);

        let drop_op = Operation::of_name(db, location, "cont.drop")
            .operand(cont_val)
            .build();

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![cont_op, drop_op],
        );
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa::tracked]
    fn lower_drop_and_check(db: &dyn salsa::Database, module: Module<'_>) -> String {
        let lowered = lower(db, module);
        let body = lowered.body(db);
        let ops = body.blocks(db)[0].operations(db);
        // The drop op is the second operation (after dummy op)
        ops.get(1).map(|op| op.full_name(db)).unwrap_or_default()
    }

    #[salsa_test]
    fn test_drop_to_wasm_drop(db: &salsa::DatabaseImpl) {
        let module = make_module_with_drop(db);
        let op_name = lower_drop_and_check(db, module);
        assert_eq!(op_name, "wasm.drop");
    }

    // === Tests for shift point analysis ===

    fn shift_location(db: &dyn salsa::Database) -> Location<'_> {
        let path = PathId::new(db, "file:///test.trb".to_owned());
        Location::new(path, Span::new(100, 120)) // Distinct location for shift
    }

    #[salsa::tracked]
    fn make_module_with_shift_in_function(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let shift_loc = shift_location(db);
        let i32_ty = core::I32::new(db).as_type();

        // Create shift inside a function
        let handler_block = Block::new(db, BlockId::fresh(), shift_loc, IdVec::new(), idvec![]);
        let handler_region = Region::new(db, shift_loc, idvec![handler_block]);

        let shift = Operation::of_name(db, shift_loc, "cont.shift")
            .attr("tag", Attribute::IntBits(99))
            .region(handler_region)
            .build();

        // Create function body with shift
        let func_body_block = Block::new(db, BlockId::fresh(), location, idvec![], idvec![shift]);
        let func_body = Region::new(db, location, idvec![func_body_block]);

        // Create func.func operation
        let func_ty = core::Func::new(db, idvec![], i32_ty).as_type();
        let func_op = Operation::of_name(db, location, "func.func")
            .attr(
                "sym_name",
                Attribute::QualifiedName(QualifiedName::simple(Symbol::new("my_func"))),
            )
            .attr("type", Attribute::Type(func_ty))
            .region(func_body)
            .build();

        let block = Block::new(db, BlockId::fresh(), location, idvec![], idvec![func_op]);
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa::tracked]
    fn analyze_shift_points_test(db: &dyn salsa::Database) -> (usize, Option<u64>, Option<String>) {
        let module = make_module_with_shift_in_function(db);
        let analysis = analyze_continuations(db, module);
        let shift_points = analysis.shift_points(db);

        let count = shift_points.len();
        let first_info = shift_points.values().next();
        let tag = first_info.map(|info| info.tag);
        let func_name = first_info
            .and_then(|info| info.containing_func.as_ref())
            .map(|name| name.name().to_string());

        (count, tag, func_name)
    }

    #[salsa_test]
    fn test_shift_point_analysis(db: &salsa::DatabaseImpl) {
        let (count, tag, func_name) = analyze_shift_points_test(db);

        // Should find one shift point
        assert_eq!(count, 1);
        // Should have correct tag
        assert_eq!(tag, Some(99));
        // Should have correct containing function
        assert_eq!(func_name, Some("my_func".to_string()));
    }
}
