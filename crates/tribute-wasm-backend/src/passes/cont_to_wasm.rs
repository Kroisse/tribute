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
use trunk_ir::{
    Attribute, DialectOp, IdVec, Location, Operation, QualifiedName, Symbol, Type, Value, ValueDef,
};

/// Continuation struct layout:
/// - Field 0: resume_fn (funcref) - function to call when resuming
/// - Field 1: state (anyref) - captured local state
/// - Field 2: tag (i32) - prompt tag this continuation belongs to
///
/// The continuation type is a WasmGC struct with these fixed fields.
/// Each shift point generates a unique State struct type to capture live locals.
pub mod cont_types {
    use super::*;

    /// Create a continuation struct type.
    ///
    /// Layout: (resume_fn: funcref, state: anyref, tag: i32)
    ///
    /// The actual GC struct type is inferred from struct_new operands at emit time.
    /// We use `wasm.structref` as a placeholder type that represents any struct.
    pub fn continuation_type(db: &dyn salsa::Database) -> Type<'_> {
        // Use structref as the continuation type
        // The actual struct layout is determined by struct_new operands
        wasm::Structref::new(db).as_type()
    }

    /// Create a state struct type for a specific shift point.
    ///
    /// The state struct captures all live locals at the shift point.
    /// Currently returns anyref as live local analysis is a placeholder.
    ///
    /// When live local analysis is fully implemented, this will create
    /// a unique struct type per shift point with the captured locals' types.
    pub fn state_type<'db>(
        db: &'db dyn salsa::Database,
        _live_locals: &[LiveLocal<'db>],
    ) -> Type<'db> {
        // Use anyref as a placeholder for the state type
        // The actual struct layout will be determined when live local capture is implemented
        wasm::Anyref::new(db).as_type()
    }
}

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
///
/// Transforms `cont.push_prompt` into a wasm block that:
/// 1. Executes the body
/// 2. Checks yield_state after body completes
/// 3. If yielding and tag matches: handles the continuation
/// 4. If yielding and tag doesn't match: propagates (returns)
/// 5. If not yielding: returns body result normally
struct PushPromptPattern;

impl RewritePattern for PushPromptPattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
    ) -> RewriteResult<'db> {
        let Ok(push_prompt) = cont::PushPrompt::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let location = op.location(db);
        let i32_ty = core::I32::new(db).as_type();

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

        // Get the original body region
        let original_body = push_prompt.body(db);
        let result_types = op.results(db).clone();

        // Build yield checking operations to append after body
        let mut yield_check_ops: Vec<Operation<'db>> = Vec::new();

        // Check: if (global.get $yield_state)
        let get_yield_state = wasm::global_get(db, location, i32_ty, YIELD_STATE_IDX);
        let yield_state_val = get_yield_state.as_operation().result(db, 0);
        yield_check_ops.push(get_yield_state.as_operation());

        // Create the if condition body: check tag match
        // if yield_state != 0:
        //   if yield_tag == our_tag:
        //     reset and handle
        //   else:
        //     propagate (return)

        // Get yield_tag for comparison
        let get_yield_tag = wasm::global_get(db, location, i32_ty, YIELD_TAG_IDX);
        let yield_tag_val = get_yield_tag.as_operation().result(db, 0);

        // Create our tag constant
        let our_tag = wasm::i32_const(db, location, i32_ty, Attribute::IntBits(tag_value));
        let our_tag_val = our_tag.as_operation().result(db, 0);

        // Compare: yield_tag == our_tag
        let tag_cmp = wasm::i32_eq(db, location, yield_tag_val, our_tag_val, i32_ty);
        let tag_match_val = tag_cmp.as_operation().result(db, 0);

        // If tag matches: reset yield_state and handle
        // For now, we just reset yield_state. Handler invocation requires more infrastructure.
        let const_0 = wasm::i32_const(db, location, i32_ty, Attribute::IntBits(0));
        let const_0_val = const_0.as_operation().result(db, 0);
        let reset_yield = wasm::global_set(db, location, const_0_val, YIELD_STATE_IDX);

        // Build the "then" block for tag match (handle)
        // TODO: Actually invoke handler with continuation
        // For now, just reset yield_state
        let then_block = trunk_ir::Block::new(
            db,
            trunk_ir::BlockId::fresh(),
            location,
            IdVec::new(),
            IdVec::from(vec![const_0.as_operation(), reset_yield.as_operation()]),
        );
        let then_region = trunk_ir::Region::new(db, location, IdVec::from(vec![then_block]));

        // Build the "else" block for tag mismatch (propagate)
        // Return to continue bubbling up
        let return_op = wasm::r#return(db, location, None);
        let else_block = trunk_ir::Block::new(
            db,
            trunk_ir::BlockId::fresh(),
            location,
            IdVec::new(),
            IdVec::from(vec![return_op.as_operation()]),
        );
        let else_region = trunk_ir::Region::new(db, location, IdVec::from(vec![else_block]));

        // Inner if: check tag match
        let inner_if = Operation::of_name(db, location, "wasm.if")
            .operand(tag_match_val)
            .region(then_region)
            .region(else_region)
            .build();

        // Build then block for outer if (yield_state check)
        let outer_then_block = trunk_ir::Block::new(
            db,
            trunk_ir::BlockId::fresh(),
            location,
            IdVec::new(),
            IdVec::from(vec![
                get_yield_tag.as_operation(),
                our_tag.as_operation(),
                tag_cmp.as_operation(),
                inner_if,
            ]),
        );
        let outer_then_region =
            trunk_ir::Region::new(db, location, IdVec::from(vec![outer_then_block]));

        // Build empty else block (no-op when not yielding)
        let empty_else_block = trunk_ir::Block::new(
            db,
            trunk_ir::BlockId::fresh(),
            location,
            IdVec::new(),
            IdVec::new(),
        );
        let outer_else_region =
            trunk_ir::Region::new(db, location, IdVec::from(vec![empty_else_block]));

        // Outer if: check yield_state
        let outer_if = Operation::of_name(db, location, "wasm.if")
            .operand(yield_state_val)
            .region(outer_then_region)
            .region(outer_else_region)
            .build();

        yield_check_ops.push(outer_if);

        // Append yield_check_ops to the original body's last block
        let new_body = append_ops_to_region(db, location, &original_body, &yield_check_ops);

        // Create the wrapper block with the modified body
        let new_op = Operation::of_name(db, location, "wasm.block")
            .attr("label", Attribute::IntBits(tag_value))
            .results(result_types)
            .region(new_body)
            .build();

        RewriteResult::Replace(new_op)
    }
}

/// Append operations to the last block of a region.
fn append_ops_to_region<'db>(
    db: &'db dyn salsa::Database,
    location: Location<'db>,
    region: &trunk_ir::Region<'db>,
    ops_to_append: &[Operation<'db>],
) -> trunk_ir::Region<'db> {
    let blocks = region.blocks(db);
    if blocks.is_empty() {
        // Create a new block with just the appended ops
        let new_block = trunk_ir::Block::new(
            db,
            trunk_ir::BlockId::fresh(),
            location,
            IdVec::new(),
            ops_to_append.iter().copied().collect(),
        );
        return trunk_ir::Region::new(db, location, IdVec::from(vec![new_block]));
    }

    // Clone all blocks, modifying only the last one
    let mut new_blocks: Vec<trunk_ir::Block<'db>> = Vec::with_capacity(blocks.len());
    for (i, block) in blocks.iter().enumerate() {
        if i == blocks.len() - 1 {
            // Last block: append operations
            let existing_ops = block.operations(db);
            let mut all_ops: Vec<Operation<'db>> = existing_ops.iter().copied().collect();
            all_ops.extend(ops_to_append.iter().copied());

            let new_block = trunk_ir::Block::new(
                db,
                block.id(db),
                block.location(db),
                block.args(db).clone(),
                all_ops.into_iter().collect(),
            );
            new_blocks.push(new_block);
        } else {
            // Other blocks: keep as-is
            new_blocks.push(*block);
        }
    }

    trunk_ir::Region::new(db, location, new_blocks.into_iter().collect())
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
        let funcref_ty = wasm::Funcref::new(db).as_type();

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
        // to run when the continuation is captured. The handler region is executed
        // when the continuation is invoked via cont.resume.
        //
        // Current implementation:
        // - Creates an empty state struct (live local capture not yet implemented)
        // - Creates continuation struct with (null resume_fn, state, tag)
        // - Resume function generation is deferred to Phase 4

        let mut ops = Vec::new();

        // === Build Continuation Struct ===

        // 1. Create empty state struct (placeholder for captured locals)
        // Type: core.nil (empty state, will be anyref at runtime)
        let state_ty = cont_types::state_type(db, &[]);
        let state_struct = wasm::nop(db, location, state_ty);
        let state_val = state_struct.as_operation().result(db, 0);
        ops.push(state_struct.as_operation());

        // 2. Create continuation struct: (resume_fn, state, tag)
        // Field 0: resume_fn (ref.null func) - placeholder until resume fn generation
        let null_resume_fn = wasm::ref_null(
            db,
            location,
            funcref_ty,
            Attribute::Symbol(Symbol::new("func")),
        );
        let resume_fn_val = null_resume_fn.as_operation().result(db, 0);
        ops.push(null_resume_fn.as_operation());

        // Field 1: state (the state struct we just created)
        // Already have state_val

        // Field 2: tag (i32)
        let tag_const = wasm::i32_const(db, location, i32_ty, Attribute::IntBits(tag_value));
        let tag_val = tag_const.as_operation().result(db, 0);
        ops.push(tag_const.as_operation());

        // Build the continuation struct
        let cont_ty = cont_types::continuation_type(db);
        let cont_struct = Operation::of_name(db, location, "wasm.struct_new")
            .operands(IdVec::from(vec![resume_fn_val, state_val, tag_val]))
            .attr("type", Attribute::Type(cont_ty))
            .results(IdVec::from(vec![cont_ty]))
            .build();
        let cont_val = Value::new(db, ValueDef::OpResult(cont_struct), 0);
        ops.push(cont_struct);

        // === Set Yield Globals ===

        // 3. Set $yield_state = 1 (yielding)
        let const_1 = wasm::i32_const(db, location, i32_ty, Attribute::IntBits(1));
        let const_1_val = const_1.as_operation().result(db, 0);
        ops.push(const_1.as_operation());
        ops.push(wasm::global_set(db, location, const_1_val, YIELD_STATE_IDX).as_operation());

        // 4. Set $yield_tag = tag
        let tag_const2 = wasm::i32_const(db, location, i32_ty, Attribute::IntBits(tag_value));
        let tag_const2_val = tag_const2.as_operation().result(db, 0);
        ops.push(tag_const2.as_operation());
        ops.push(wasm::global_set(db, location, tag_const2_val, YIELD_TAG_IDX).as_operation());

        // 5. Set $yield_cont = continuation struct
        ops.push(wasm::global_set(db, location, cont_val, YIELD_CONT_IDX).as_operation());

        // 6. Set $yield_value = null (placeholder - full impl will pass shift value)
        let null_value = wasm::ref_null(
            db,
            location,
            anyref_ty,
            Attribute::Symbol(Symbol::new("any")),
        );
        let null_value_val = null_value.as_operation().result(db, 0);
        ops.push(null_value.as_operation());
        ops.push(wasm::global_set(db, location, null_value_val, YIELD_VALUE_IDX).as_operation());

        // 7. Return to unwind stack (will bubble up to push_prompt)
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
///
/// Continuation struct layout:
/// - Field 0: resume_fn (funcref)
/// - Field 1: state (anyref)
/// - Field 2: tag (i32)
///
/// Resume operation:
/// 1. Reset $yield_state = 0
/// 2. Extract resume_fn from continuation
/// 3. Extract state from continuation
/// 4. Call resume_fn(state, value) via call_indirect
struct ResumePattern;

/// Continuation struct field indices
const CONT_FIELD_RESUME_FN: u32 = 0;
const CONT_FIELD_STATE: u32 = 1;
#[allow(dead_code)]
const CONT_FIELD_TAG: u32 = 2;

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
        let continuation = resume.continuation(db);
        let value = resume.value(db);

        let i32_ty = core::I32::new(db).as_type();
        let funcref_ty = wasm::Funcref::new(db).as_type();
        let anyref_ty = wasm::Anyref::new(db).as_type();
        let cont_ty = cont_types::continuation_type(db);

        let mut ops = Vec::new();

        // 1. Reset $yield_state = 0 (not yielding anymore)
        let const_0 = wasm::i32_const(db, location, i32_ty, Attribute::IntBits(0));
        let const_0_val = const_0.as_operation().result(db, 0);
        ops.push(const_0.as_operation());
        ops.push(wasm::global_set(db, location, const_0_val, YIELD_STATE_IDX).as_operation());

        // 2. Extract resume_fn from continuation (field 0)
        let get_resume_fn = Operation::of_name(db, location, "wasm.struct_get")
            .operand(continuation)
            .attr("type", Attribute::Type(cont_ty))
            .attr("field_idx", Attribute::IntBits(CONT_FIELD_RESUME_FN as u64))
            .results(IdVec::from(vec![funcref_ty]))
            .build();
        let resume_fn_val = Value::new(db, ValueDef::OpResult(get_resume_fn), 0);
        ops.push(get_resume_fn);

        // 3. Extract state from continuation (field 1)
        let get_state = Operation::of_name(db, location, "wasm.struct_get")
            .operand(continuation)
            .attr("type", Attribute::Type(cont_ty))
            .attr("field_idx", Attribute::IntBits(CONT_FIELD_STATE as u64))
            .results(IdVec::from(vec![anyref_ty]))
            .build();
        let state_val = Value::new(db, ValueDef::OpResult(get_state), 0);
        ops.push(get_state);

        // 4. Call resume_fn(state, value) via call_indirect
        // Resume function signature: (state: anyref, value: anyref) -> result
        // For now, we use call_indirect with type_idx 0 (placeholder)
        // The actual type will be resolved at emit time
        let result_ty = op.results(db).first().copied().unwrap_or(anyref_ty);
        let call_indirect = Operation::of_name(db, location, "wasm.call_indirect")
            .operands(IdVec::from(vec![state_val, value, resume_fn_val]))
            .attr("type_idx", Attribute::IntBits(0)) // Placeholder - resolved at emit
            .attr("table", Attribute::IntBits(0)) // Default table
            .results(IdVec::from(vec![result_ty]))
            .build();
        ops.push(call_indirect);

        RewriteResult::Expand(ops)
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

        // Shift should expand to:
        // 1. wasm.nop (state placeholder)
        // 2. wasm.ref_null (resume_fn = null funcref)
        // 3. wasm.i32_const (tag value for continuation struct)
        // 4. wasm.struct_new (continuation struct)
        // 5. wasm.i32_const (yield_state = 1)
        // 6. wasm.global_set (yield_state)
        // 7. wasm.i32_const (yield_tag)
        // 8. wasm.global_set (yield_tag)
        // 9. wasm.global_set (yield_cont)
        // 10. wasm.ref_null (yield_value = null)
        // 11. wasm.global_set (yield_value)
        // 12. wasm.return

        // First op: nop (state placeholder)
        assert_eq!(op_names.first(), Some(&"wasm.nop".to_string()));
        // Last op: return (to unwind stack)
        assert_eq!(op_names.last(), Some(&"wasm.return".to_string()));
        // Should contain struct_new for continuation
        assert!(op_names.contains(&"wasm.struct_new".to_string()));
        // Total: 12 operations
        assert_eq!(op_names.len(), 12);
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
    fn lower_resume_and_check(db: &dyn salsa::Database, module: Module<'_>) -> Vec<String> {
        let lowered = lower(db, module);
        let body = lowered.body(db);
        let ops = body.blocks(db)[0].operations(db);
        // Skip the first two dummy ops and collect the rest
        ops.iter().skip(2).map(|op| op.full_name(db)).collect()
    }

    #[salsa_test]
    fn test_resume_invokes_continuation(db: &salsa::DatabaseImpl) {
        let module = make_module_with_resume(db);
        let op_names = lower_resume_and_check(db, module);

        // Resume should expand to:
        // 1. wasm.i32_const (yield_state = 0)
        // 2. wasm.global_set (reset yield_state)
        // 3. wasm.struct_get (extract resume_fn)
        // 4. wasm.struct_get (extract state)
        // 5. wasm.call_indirect (invoke resume_fn)

        // First op: i32_const for resetting yield_state
        assert_eq!(op_names.first(), Some(&"wasm.i32_const".to_string()));
        // Last op: call_indirect to invoke resume function
        assert_eq!(op_names.last(), Some(&"wasm.call_indirect".to_string()));
        // Should contain struct_get operations
        assert!(op_names.contains(&"wasm.struct_get".to_string()));
        // Total: 5 operations
        assert_eq!(op_names.len(), 5);
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
