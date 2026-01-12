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
//! - `$yield_op_idx`: i32 (operation index for multi-op abilities)
//!
//! Note: The value passed with shift is stored in the continuation struct's
//! `shift_value` field instead of a global variable.

use std::collections::{BTreeMap, HashMap};

use tribute_ir::ModulePathExt;
use tribute_ir::dialect::{tribute, tribute_rt};
use trunk_ir::DialectType;
use trunk_ir::dialect::cont;
use trunk_ir::dialect::core::{self, Module};
use trunk_ir::dialect::wasm;
use trunk_ir::rewrite::{OpAdaptor, PatternApplicator, RewritePattern, RewriteResult};
use trunk_ir::{
    Attribute, Block, BlockId, DialectOp, IdVec, Location, Operation, Region, Symbol, Type, Value,
    ValueDef,
};

use crate::type_converter::wasm_type_converter;

/// Continuation struct layout:
/// - Field 0: resume_fn (funcref) - function to call when resuming
/// - Field 1: state (anyref) - captured local state
/// - Field 2: tag (i32) - prompt tag this continuation belongs to
/// - Field 3: shift_value (anyref) - value yielded by shift (for handler)
///
/// The continuation type is a WasmGC struct with these fixed fields.
/// Each shift point generates a unique State struct type to capture live locals.
///
/// ## Resume Wrapper
///
/// When resuming a continuation, a wrapper struct is created:
/// - Field 0: state (anyref) - the original state from continuation
/// - Field 1: resume_value (anyref) - value passed by handler to continuation
///
/// This allows the resume function to receive both state and value in a single
/// parameter, eliminating the need for globals.
pub mod cont_types {
    use super::*;

    /// Create a continuation struct type.
    ///
    /// Layout: (resume_fn: funcref, state: anyref, tag: i32, shift_value: anyref)
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
    /// We return `structref` as the type - the actual struct layout is determined
    /// at emit time from the `struct_new` operands.
    ///
    /// For empty live locals (pure shift with no captured state), we still use
    /// structref for consistency - an empty struct will be created.
    pub fn state_type<'db>(
        db: &'db dyn salsa::Database,
        _live_locals: &[LiveLocal<'db>],
    ) -> Type<'db> {
        // Use structref - the actual struct type is determined at emit time
        // from the types of the operands passed to struct_new
        wasm::Structref::new(db).as_type()
    }

    /// Create the resume wrapper struct type.
    ///
    /// Layout: (state: anyref, resume_value: anyref)
    ///
    /// This wrapper is created at resume time to pass both state and value
    /// to the resume function in a single parameter.
    pub fn resume_wrapper_type(db: &dyn salsa::Database) -> Type<'_> {
        wasm::Structref::new(db).as_type()
    }

    /// Create the resume function type.
    ///
    /// Signature: (wrapper: anyref) -> (ref $Step)
    ///
    /// The wrapper contains both state and resume_value.
    /// Returns Step because resume functions can also shift
    /// (if the resumed continuation shifts again).
    pub fn resume_fn_type(db: &dyn salsa::Database) -> Type<'_> {
        let anyref = wasm::Anyref::new(db).as_type();
        let step_ty = step_type(db);
        core::Func::new(db, IdVec::from(vec![anyref]), step_ty).as_type()
    }

    /// Create the Step struct type.
    ///
    /// Layout: (tag: i32, value: anyref, prompt: i32, op_idx: i32)
    /// - tag = 0: Done (value is the boxed result, prompt/op_idx ignored)
    /// - tag = 1: Shift (value is the continuation struct, prompt/op_idx identify handler)
    ///
    /// This type unifies all effectful function returns for the trampoline-based
    /// effect system, eliminating cascading type changes and enabling centralized
    /// handler dispatch.
    pub fn step_type(db: &dyn salsa::Database) -> Type<'_> {
        // Use the unique Step marker type from gc_types
        crate::gc_types::step_marker_type(db)
    }

    /// Tag value for Done (successful completion).
    pub const STEP_TAG_DONE: i32 = crate::gc_types::STEP_TAG_DONE;

    /// Tag value for Shift (suspended with continuation, needs handler dispatch).
    pub const STEP_TAG_SHIFT: i32 = crate::gc_types::STEP_TAG_SHIFT;
}

/// Resume function generation.
///
/// For each shift point, we generate a resume function that:
/// 1. Takes (state: anyref, value: anyref) as parameters
/// 2. Restores captured locals from state
/// 3. Continues execution from after the shift point
pub mod resume_gen {
    use std::collections::HashMap;

    use super::*;
    use trunk_ir::{Block, BlockArg, BlockId, Region};

    /// Information about a generated resume function.
    #[derive(Clone, Debug)]
    pub struct ResumeFunctionInfo<'db> {
        /// The qualified name of the resume function.
        pub name: Symbol,
        /// The location for the function.
        pub location: Location<'db>,
        /// The shift point this resume function corresponds to.
        pub shift_key: LocationKey,
    }

    /// Generate resume functions for all shift points in the module.
    ///
    /// Returns a list of resume function operations to add to the module,
    /// and a map from shift location to resume function name.
    #[allow(dead_code)]
    pub fn generate_resume_functions<'db>(
        db: &'db dyn salsa::Database,
        analysis: &ContAnalysis<'db>,
        location: Location<'db>,
    ) -> (Vec<Operation<'db>>, BTreeMap<LocationKey, Symbol>) {
        let mut functions = Vec::new();
        let mut resume_fn_names = BTreeMap::new();

        for (key, info) in analysis.shift_points(db).iter() {
            let resume_fn_name = generate_resume_fn_name(info);
            let resume_fn = generate_resume_function(db, location, resume_fn_name, info);

            functions.push(resume_fn);
            resume_fn_names.insert(key.clone(), resume_fn_name);
        }

        (functions, resume_fn_names)
    }

    /// Generate only resume function names (not the actual functions).
    ///
    /// This is used for inline resume function generation where we need
    /// to know the names upfront for ref.func, but generate the functions
    /// on-the-fly during shift transformation.
    pub fn generate_resume_fn_names<'db>(
        db: &'db dyn salsa::Database,
        analysis: &ContAnalysis<'db>,
    ) -> BTreeMap<LocationKey, Symbol> {
        let mut resume_fn_names = BTreeMap::new();

        for (key, info) in analysis.shift_points(db).iter() {
            let resume_fn_name = generate_resume_fn_name(info);
            resume_fn_names.insert(key.clone(), resume_fn_name);
        }

        resume_fn_names
    }

    /// Make generate_resume_fn_name public for inline generation.
    pub fn make_resume_fn_name(info: &ShiftPointInfo<'_>) -> Symbol {
        generate_resume_fn_name(info)
    }

    /// Generate the resume function name for a shift point.
    fn generate_resume_fn_name(info: &ShiftPointInfo<'_>) -> Symbol {
        let base_name = info
            .containing_func
            .as_ref()
            .map(|n| n.last_segment().to_string())
            .unwrap_or_else(|| "anon".to_string());

        let resume_name =
            Symbol::from_dynamic(&format!("{}$resume${}", base_name, info.shift_index));

        if let Some(ref containing) = info.containing_func {
            // Build qualified name: parent::resume_name
            if let Some(parent) = containing.parent_path() {
                parent.join_path(resume_name)
            } else {
                resume_name
            }
        } else {
            resume_name
        }
    }

    /// Generate a resume function for a shift point.
    ///
    /// The resume function has signature: (state: anyref, value: anyref) -> anyref
    ///
    /// This function:
    /// 1. Casts state to the concrete state struct type
    /// 2. Extracts captured locals from state using struct_get
    /// 3. Remaps the continuation operations to use extracted values
    /// 4. Executes the continuation body (code after shift)
    fn generate_resume_function<'db>(
        db: &'db dyn salsa::Database,
        location: Location<'db>,
        name: Symbol,
        info: &ShiftPointInfo<'db>,
    ) -> Operation<'db> {
        let anyref = wasm::Anyref::new(db).as_type();
        let func_ty = cont_types::resume_fn_type(db);

        // Block arguments are typed (state: anyref, value: anyref)
        let block_args = IdVec::from(vec![
            BlockArg::of_type(db, anyref),
            BlockArg::of_type(db, anyref),
        ]);
        let block_id = BlockId::fresh();

        // Get block argument values
        let state_param = Value::new(db, ValueDef::BlockArg(block_id), 0);
        let value_param = Value::new(db, ValueDef::BlockArg(block_id), 1);

        let mut ops: Vec<Operation<'db>> = Vec::new();
        let mut value_mapping: HashMap<Value<'db>, Value<'db>> = HashMap::new();

        // Extract live locals from state struct
        // Use structref as placeholder type - emit will resolve via placeholder map
        let structref_ty = wasm::Structref::new(db).as_type();
        let field_count = info.live_locals.len() as u64;

        for (field_idx, live_local) in info.live_locals.iter().enumerate() {
            // Generate struct_get to extract this field
            // Add type and field_count attributes for proper type resolution at emit time
            // live_local.ty is already converted by compute_live_locals
            let struct_get = wasm::struct_get(
                db,
                location,
                state_param,
                live_local.ty,
                0, // type_idx - Placeholder, resolved at emit time
                field_idx as u32,
            )
            .as_operation()
            .modify(db)
            .attr("type", Attribute::Type(structref_ty))
            .attr("field_count", Attribute::IntBits(field_count))
            .build();

            let extracted_value = Value::new(db, ValueDef::OpResult(struct_get), 0);
            ops.push(struct_get);

            // Map the original value to the extracted value
            value_mapping.insert(live_local.value, extracted_value);
        }

        // Remap continuation operations and add them to the body
        if !info.continuation_ops.is_empty() {
            let remapped_ops = remap_operations(db, &info.continuation_ops, &mut value_mapping);
            ops.extend(remapped_ops);
        } else {
            // No continuation body - just return the value parameter
            ops.push(wasm::r#return(db, location, Some(value_param)).as_operation());
        }

        // Create function body block with typed arguments
        let body_block = Block::new(
            db,
            block_id,
            location,
            block_args,
            ops.into_iter().collect(),
        );
        let body_region = Region::new(db, location, IdVec::from(vec![body_block]));

        // Create the wasm.func operation
        wasm::func(db, location, name, func_ty, body_region).as_operation()
    }

    /// Remap operations to use new values from the value mapping.
    ///
    /// For each operation:
    /// 1. Remap operands using the value mapping
    /// 2. Create a new operation with remapped operands
    /// 3. Add the new operation's results to the mapping
    pub fn remap_operations<'db>(
        db: &'db dyn salsa::Database,
        ops: &[Operation<'db>],
        value_mapping: &mut HashMap<Value<'db>, Value<'db>>,
    ) -> Vec<Operation<'db>> {
        use tribute_ir::dialect::tribute;
        let mut result = Vec::with_capacity(ops.len());

        for &op in ops {
            // Skip tribute.var operations - kept for LSP, no runtime effect
            // Their results should already be mapped by earlier passes
            if op.dialect(db) == tribute::DIALECT_NAME() && op.name(db) == tribute::VAR() {
                continue;
            }

            // Remap operands
            let new_operands: IdVec<Value<'db>> = op
                .operands(db)
                .iter()
                .map(|&v| *value_mapping.get(&v).unwrap_or(&v))
                .collect();

            // Recursively remap nested regions
            let new_regions: IdVec<Region<'db>> = op
                .regions(db)
                .iter()
                .map(|region| remap_region(db, region, value_mapping))
                .collect();

            // Special handling for wasm.return in resume functions:
            // If the return value is a primitive type (Int/Nat), we need to box it
            // since the resume function returns anyref.
            if op.dialect(db) == wasm::DIALECT_NAME()
                && op.name(db) == wasm::RETURN()
                && let Some(&return_value) = new_operands.first()
            {
                // Look up value type from its definition
                let value_ty = match return_value.def(db) {
                    ValueDef::OpResult(defining_op) => {
                        defining_op.results(db).get(return_value.index(db)).copied()
                    }
                    ValueDef::BlockArg(_) => None, // Block args in resume should not be primitives
                };

                if let Some(value_ty) = value_ty
                    && (core::I64::from_type(db, value_ty).is_some()
                        || core::I32::from_type(db, value_ty).is_some()
                        || tribute_rt::is_int(db, value_ty)
                        || tribute_rt::is_nat(db, value_ty))
                {
                    // Box the primitive value to anyref before returning
                    // Resume functions return anyref, so we need consistent anyref type
                    let mut boxing_ops = Vec::new();
                    let boxed = super::box_value_to_anyref(
                        db,
                        op.location(db),
                        return_value,
                        value_ty,
                        &mut boxing_ops,
                    );
                    result.extend(boxing_ops);

                    // Create new return with boxed value
                    let new_return = wasm::r#return(db, op.location(db), Some(boxed));
                    result.push(new_return.as_operation());
                    continue;
                }
            }

            // Create new operation with remapped operands and regions
            let new_op = op
                .modify(db)
                .operands(new_operands)
                .regions(new_regions)
                .build();

            // Map old results to new results
            let result_types = op.results(db);
            for i in 0..result_types.len() {
                let old_result = op.result(db, i);
                let new_result = new_op.result(db, i);
                value_mapping.insert(old_result, new_result);
            }

            result.push(new_op);
        }

        result
    }

    /// Remap values in a region.
    fn remap_region<'db>(
        db: &'db dyn salsa::Database,
        region: &Region<'db>,
        value_mapping: &mut HashMap<Value<'db>, Value<'db>>,
    ) -> Region<'db> {
        let new_blocks: IdVec<Block<'db>> = region
            .blocks(db)
            .iter()
            .map(|block| remap_block(db, block, value_mapping))
            .collect();

        Region::new(db, region.location(db), new_blocks)
    }

    /// Remap values in a block.
    fn remap_block<'db>(
        db: &'db dyn salsa::Database,
        block: &Block<'db>,
        value_mapping: &mut HashMap<Value<'db>, Value<'db>>,
    ) -> Block<'db> {
        let remapped_ops = remap_operations(
            db,
            &block.operations(db).iter().copied().collect::<Vec<_>>(),
            value_mapping,
        );

        Block::new(
            db,
            block.id(db),
            block.location(db),
            block.args(db).clone(),
            remapped_ops.into_iter().collect(),
        )
    }
}

// Yield global indices - these are hardcoded since they're always at the
// start of the module's global section (indices 0-3).
// See lower_wasm.rs::module_preamble_ops for where these are emitted.

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

/// Location key for shift points that can be used in 'static contexts.
///
/// Unlike `Location<'db>`, this doesn't contain a salsa-interned PathId,
/// so it can be stored in patterns that require 'static lifetime.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, salsa::Update)]
pub struct LocationKey {
    /// File path URI (e.g., "file:///path/to/file.trb")
    pub path: String,
    /// Start byte of the span.
    pub start: usize,
    /// End byte of the span.
    pub end: usize,
}

impl LocationKey {
    fn from_location(db: &dyn salsa::Database, loc: Location<'_>) -> Self {
        Self {
            path: loc.path.uri(db).to_owned(),
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
    pub containing_func: Option<Symbol>,
    /// Index of this shift within its containing function (for unique naming).
    pub shift_index: u32,
    /// Live local variables at this shift point that need to be captured.
    /// Each entry is (name, type).
    pub live_locals: Vec<LiveLocal<'db>>,
    /// Generated state struct type name.
    pub state_type_name: Option<Symbol>,
    /// Operations that come after the shift (continuation body).
    /// These will be executed when the continuation is resumed.
    pub continuation_ops: Vec<Operation<'db>>,
}

/// A live local variable that needs to be captured at a shift point.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub struct LiveLocal<'db> {
    /// The SSA value that needs to be captured.
    pub value: Value<'db>,
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
    current_func: Option<Symbol>,
    /// Counter for shifts within current function.
    shift_counter: u32,
    /// Current function's block arguments (entry point parameters).
    current_func_args: Vec<Value<'db>>,
    /// Types of current function's block arguments.
    current_func_arg_types: Vec<Type<'db>>,
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
            current_func_args: Vec::new(),
            current_func_arg_types: Vec::new(),
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

        // Track function context - check both func.func and wasm.func
        // (wasm.func is present after func_to_wasm lowering)
        if let Ok(func_op) = trunk_ir::dialect::func::Func::from_operation(self.db, *op) {
            self.analyze_function_with_live_locals(
                func_op.sym_name(self.db),
                func_op.body(self.db),
            );
            return;
        } else if let Ok(wasm_func_op) = wasm::Func::from_operation(self.db, *op) {
            self.analyze_function_with_live_locals(
                wasm_func_op.sym_name(self.db),
                wasm_func_op.body(self.db),
            );
            return;
        }

        // Check for continuation operations (for non-function contexts)
        if dialect == cont::DIALECT_NAME() {
            self.has_continuations = true;

            if name == cont::PUSH_PROMPT() {
                self.prompt_count = self.prompt_count.saturating_add(1);
            } else if name == cont::SHIFT() {
                // In non-function context, we don't have live locals or continuation ops
                self.collect_shift_info(op, &[], Vec::new());
            }
        }

        // Recursively analyze nested regions
        for region in op.regions(self.db).iter() {
            self.analyze_region(region);
        }
    }

    /// Analyze a function body with live local analysis for shift points.
    fn analyze_function_with_live_locals(
        &mut self,
        func_name: Symbol,
        body: trunk_ir::Region<'db>,
    ) {
        let prev_func = self.current_func.take();
        let prev_counter = self.shift_counter;
        let prev_args = std::mem::take(&mut self.current_func_args);
        let prev_arg_types = std::mem::take(&mut self.current_func_arg_types);

        self.current_func = Some(func_name);
        self.shift_counter = 0;

        // Collect entry block arguments (function parameters) and their types
        if let Some(entry_block) = body.blocks(self.db).first() {
            let block_args = entry_block.args(self.db);
            for (i, arg) in block_args.iter().enumerate() {
                self.current_func_args.push(entry_block.arg(self.db, i));
                self.current_func_arg_types.push(arg.ty(self.db));
            }
        }

        // Flatten all operations in the function body
        let flat_ops = self.flatten_region(&body);

        // Find shift points and compute live locals for each
        for (op_idx, op) in flat_ops.iter().enumerate() {
            let dialect = op.dialect(self.db);
            let name = op.name(self.db);

            if dialect == cont::DIALECT_NAME() {
                self.has_continuations = true;

                if name == cont::PUSH_PROMPT() {
                    self.prompt_count = self.prompt_count.saturating_add(1);
                } else if name == cont::SHIFT() {
                    // Compute live locals: values defined before shift, used after shift
                    let live_locals = self.compute_live_locals(&flat_ops, op_idx);
                    // Collect operations after shift (continuation body)
                    let continuation_ops: Vec<Operation<'db>> =
                        flat_ops.iter().skip(op_idx + 1).copied().collect();
                    self.collect_shift_info(op, &live_locals, continuation_ops);
                }
            }
        }

        self.current_func = prev_func;
        self.shift_counter = prev_counter;
        self.current_func_args = prev_args;
        self.current_func_arg_types = prev_arg_types;
    }

    /// Flatten a region into a linear sequence of operations.
    fn flatten_region(&self, region: &trunk_ir::Region<'db>) -> Vec<Operation<'db>> {
        let mut ops = Vec::new();
        for block in region.blocks(self.db).iter() {
            self.flatten_block(block, &mut ops);
        }
        ops
    }

    /// Flatten a block and its nested regions into a linear sequence.
    fn flatten_block(&self, block: &trunk_ir::Block<'db>, ops: &mut Vec<Operation<'db>>) {
        for op in block.operations(self.db).iter() {
            ops.push(*op);
            // Recursively flatten nested regions
            for region in op.regions(self.db).iter() {
                for nested_block in region.blocks(self.db).iter() {
                    self.flatten_block(nested_block, ops);
                }
            }
        }
    }

    /// Compute live locals at a shift point.
    ///
    /// Live locals are values that:
    /// 1. Are defined before the shift (including function parameters)
    /// 2. Are used after the shift
    fn compute_live_locals(
        &self,
        flat_ops: &[Operation<'db>],
        shift_idx: usize,
    ) -> Vec<LiveLocal<'db>> {
        use std::collections::HashSet;

        let converter = wasm_type_converter();

        // Collect values defined before shift (including function args)
        let mut defined_before: HashSet<Value<'db>> = HashSet::new();

        // Function parameters are defined before any operation
        for arg in &self.current_func_args {
            defined_before.insert(*arg);
        }

        // Values from operations before shift
        for op in flat_ops.iter().take(shift_idx) {
            let result_types = op.results(self.db);
            for i in 0..result_types.len() {
                defined_before.insert(op.result(self.db, i));
            }
        }

        // Collect values used after shift
        let mut used_after: HashSet<Value<'db>> = HashSet::new();
        for op in flat_ops.iter().skip(shift_idx + 1) {
            for operand in op.operands(self.db).iter() {
                used_after.insert(*operand);
            }
            // Also check nested regions
            for region in op.regions(self.db).iter() {
                self.collect_uses_in_region(region, &mut used_after);
            }
        }

        // Live locals = defined before ∩ used after
        let mut live_locals = Vec::new();
        for value in defined_before.intersection(&used_after) {
            // Get the type of this value and convert using TypeConverter
            let ty = self.get_value_type(*value);
            if let Some(ty) = ty {
                // Convert high-level types to WASM types
                let converted_ty = converter.convert_type(self.db, ty).unwrap_or(ty);

                // Get operation info for better debugging
                let (op_dialect, op_name, op_result_ty) = match value.def(self.db) {
                    ValueDef::OpResult(op) => {
                        let result_ty = op.results(self.db).get(value.index(self.db)).copied();
                        (
                            op.dialect(self.db).to_string(),
                            op.name(self.db).to_string(),
                            result_ty,
                        )
                    }
                    ValueDef::BlockArg(_) => ("block_arg".to_string(), "".to_string(), None),
                };
                tracing::debug!(
                    "compute_live_locals: value from {}.{} type={}.{} result_ty={:?}",
                    op_dialect,
                    op_name,
                    converted_ty.dialect(self.db),
                    converted_ty.name(self.db),
                    op_result_ty.map(|t| format!("{}.{}", t.dialect(self.db), t.name(self.db)))
                );
                live_locals.push(LiveLocal {
                    value: *value,
                    name: None, // TODO: Could extract from tribute.var if available
                    ty: converted_ty,
                });
            } else {
                // Warn about values that should be captured but have unknown types
                // This can happen with nested block arguments (e.g., loop variables)
                tracing::warn!(
                    ?value,
                    "Live value has unknown type and will not be captured in continuation state. \
                     This may cause incorrect behavior if the value is actually used after resume."
                );
            }
        }

        // Sort for deterministic ordering
        live_locals.sort_by_key(|l| {
            // Sort by value definition for determinism
            match l.value.def(self.db) {
                ValueDef::BlockArg(block_id) => (0, block_id.0, l.value.index(self.db) as u64),
                ValueDef::OpResult(op) => (
                    1,
                    op.location(self.db).span.start as u64,
                    l.value.index(self.db) as u64,
                ),
            }
        });

        live_locals
    }

    /// Collect all value uses in a region recursively.
    fn collect_uses_in_region(
        &self,
        region: &trunk_ir::Region<'db>,
        uses: &mut std::collections::HashSet<Value<'db>>,
    ) {
        for block in region.blocks(self.db).iter() {
            for op in block.operations(self.db).iter() {
                for operand in op.operands(self.db).iter() {
                    uses.insert(*operand);
                }
                for nested in op.regions(self.db).iter() {
                    self.collect_uses_in_region(nested, uses);
                }
            }
        }
    }

    /// Get the type of a value.
    fn get_value_type(&self, value: Value<'db>) -> Option<Type<'db>> {
        match value.def(self.db) {
            ValueDef::OpResult(op) => {
                let results = op.results(self.db);
                results.get(value.index(self.db)).copied()
            }
            ValueDef::BlockArg(_block_id) => {
                // Check if this is a function argument
                if let Some(idx) = self.current_func_args.iter().position(|v| *v == value) {
                    self.current_func_arg_types.get(idx).copied()
                } else {
                    // For other block args (e.g., loop variables), we'd need more context
                    None
                }
            }
        }
    }

    fn collect_shift_info(
        &mut self,
        op: &Operation<'db>,
        live_locals: &[LiveLocal<'db>],
        continuation_ops: Vec<Operation<'db>>,
    ) {
        let location = op.location(self.db);
        let location_key = LocationKey::from_location(self.db, location);

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
                func_name.last_segment(),
                self.shift_counter
            ));
            // Build qualified name: parent::state_name
            if let Some(parent) = func_name.parent_path() {
                parent.join_path(state_name)
            } else {
                state_name
            }
        });

        let info = ShiftPointInfo {
            tag,
            containing_func: self.current_func,
            shift_index: self.shift_counter,
            live_locals: live_locals.to_vec(),
            state_type_name,
            continuation_ops,
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

    // Generate only resume function names upfront (for ref.func references).
    // The actual resume functions are generated inline during transform_shifts.
    let resume_fn_names = resume_gen::generate_resume_fn_names(db, &analysis);

    // Apply pattern transformations for non-shift operations.
    let module = PatternApplicator::new(wasm_type_converter())
        .add_pattern(PushPromptPattern)
        .add_pattern(HandlerDispatchPattern)
        .add_pattern(GetContinuationPattern)
        .add_pattern(GetShiftValuePattern)
        // NOTE: GetDoneValuePattern is disabled - done values are handled
        // inline in wrap_yields_in_done_step to ensure proper boxing
        // .add_pattern(GetDoneValuePattern)
        .add_pattern(ResumePattern)
        .add_pattern(DropPattern)
        .apply(db, module)
        .module;

    // Transform shift operations with inline resume function generation.
    // This is done in a loop because expanding shifts inside resume functions
    // creates new shifts that need to be processed.
    let module = transform_shifts(db, module, &analysis, &resume_fn_names);

    // Post-process: update function return types for functions that can yield.
    // The shift expansion adds `wasm.return ref.null any` which requires the
    // function's return type to be anyref instead of its original type.
    let module = update_yielding_function_types(db, module);

    // Wrap return values in Done Step for effectful functions.
    // After updating function return types to Step, we need to wrap the actual
    // return values (which may be i32, etc.) in Done Step structs.
    //
    // Only wrap returns in functions that have Step return type.
    wrap_returns_for_step_functions(db, module)
}

/// Update function return types for functions that can yield (directly or transitively).
///
/// This function:
/// 1. Identifies direct yielders (functions containing `wasm.return ref.null any`)
/// 2. Builds a call graph and propagates "effectful" mark transitively
/// 3. Updates all effectful function return types to anyref
///
/// When a function contains a shift/yield, the shift expansion adds:
/// ```text
/// ref.null any
/// wasm.return
/// ```
/// This requires the function's return type to be anyref. Additionally, any
/// function that CALLS an effectful function must also return anyref because
/// it may need to propagate the yield.
fn update_yielding_function_types<'db>(
    db: &'db dyn salsa::Database,
    module: Module<'db>,
) -> Module<'db> {
    use std::collections::{HashMap, HashSet};

    let body = module.body(db);

    // Phase 1: Build call graph and identify direct yielders
    let mut call_graph: HashMap<Symbol, HashSet<Symbol>> = HashMap::new();
    let mut direct_yielders: HashSet<Symbol> = HashSet::new();
    let mut func_ops: HashMap<Symbol, Operation<'db>> = HashMap::new();

    for block in body.blocks(db).iter() {
        for op in block.operations(db).iter() {
            if op.dialect(db) == wasm::DIALECT_NAME()
                && op.name(db) == wasm::FUNC()
                && let Ok(func_op) = wasm::Func::from_operation(db, *op)
            {
                let func_name = func_op.sym_name(db);
                let func_body = func_op.body(db);

                // Track this function
                func_ops.insert(func_name, *op);

                // Check if direct yielder (contains shift/yield)
                if function_body_can_yield(db, &func_body) {
                    direct_yielders.insert(func_name);
                    tracing::debug!(
                        "update_yielding_function_types: {} is a direct yielder",
                        func_name
                    );
                }

                // Check if function calls continuation via call_indirect
                // Handler arm lambdas do this - they're effectful because
                // the continuation may yield
                if function_body_has_call_indirect(db, &func_body) {
                    direct_yielders.insert(func_name);
                    tracing::debug!(
                        "update_yielding_function_types: {} is effectful (has call_indirect)",
                        func_name
                    );
                }

                // Collect callees for call graph
                let callees = collect_callees(db, &func_body);
                call_graph.insert(func_name, callees);
            }
        }
    }

    // Phase 2: Propagate effectful mark transitively (fixed-point iteration)
    let mut effectful_funcs = direct_yielders.clone();
    let mut changed = true;
    while changed {
        changed = false;
        for (caller, callees) in &call_graph {
            if !effectful_funcs.contains(caller) {
                // Check if any callee is effectful
                for callee in callees {
                    if effectful_funcs.contains(callee) {
                        tracing::debug!(
                            "update_yielding_function_types: {} is effectful (calls {})",
                            caller,
                            callee
                        );
                        effectful_funcs.insert(*caller);
                        changed = true;
                        break;
                    }
                }
            }
        }
    }

    // Phase 3: Update effectful function return types to anyref
    let mut modified = false;
    let mut new_ops: Vec<Operation<'db>> = Vec::new();

    for block in body.blocks(db).iter() {
        for op in block.operations(db).iter() {
            if op.dialect(db) == wasm::DIALECT_NAME()
                && op.name(db) == wasm::FUNC()
                && let Ok(func_op) = wasm::Func::from_operation(db, *op)
            {
                let func_name = func_op.sym_name(db);

                if effectful_funcs.contains(&func_name) {
                    let func_ty = func_op.r#type(db);
                    if let Some(core_func) = core::Func::from_type(db, func_ty) {
                        let current_result = core_func.result(db);
                        let step_ty = cont_types::step_type(db);

                        // Skip if already Step (structref)
                        if wasm::Structref::from_type(db, current_result).is_some() {
                            new_ops.push(*op);
                            continue;
                        }

                        let params = core_func.params(db);
                        let new_func_ty = core::Func::new(db, params, step_ty).as_type();

                        tracing::debug!(
                            "update_yielding_function_types: {} return type {} -> structref (Step)",
                            func_name,
                            current_result.name(db)
                        );

                        let new_op = op
                            .modify(db)
                            .attr(Symbol::new("type"), Attribute::Type(new_func_ty))
                            // Store original return type for trampoline unwrapping
                            .attr(
                                Symbol::new("original_result_type"),
                                Attribute::Type(current_result),
                            )
                            .build();
                        new_ops.push(new_op);
                        modified = true;
                        continue;
                    }
                }
                new_ops.push(*op);
                continue;
            }
            new_ops.push(*op);
        }
    }

    if !modified {
        return module;
    }

    // Rebuild module with updated operations
    let location = body.location(db);
    let first_block = body.blocks(db).first().cloned().unwrap();
    let new_block = trunk_ir::Block::new(
        db,
        first_block.id(db),
        first_block.location(db),
        first_block.args(db).clone(),
        new_ops.into_iter().collect(),
    );
    let new_body = trunk_ir::Region::new(db, location, IdVec::from(vec![new_block]));
    Module::create(db, module.location(db), module.name(db), new_body)
}

/// Wrap return values in Done Step for functions that return Step type.
///
/// This pass iterates through all functions and wraps return values in Done Step
/// only for functions whose return type is Step. This avoids wrapping returns
/// in non-effectful functions.
fn wrap_returns_for_step_functions<'db>(
    db: &'db dyn salsa::Database,
    module: Module<'db>,
) -> Module<'db> {
    let body = module.body(db);
    let blocks = body.blocks(db);

    let new_blocks: IdVec<Block<'db>> = blocks
        .iter()
        .map(|block| {
            let new_ops: IdVec<Operation<'db>> = block
                .operations(db)
                .iter()
                .map(|op| {
                    // Check if this is a wasm.func (use dialect constant)
                    if op.dialect(db) != wasm::DIALECT_NAME() || op.name(db) != wasm::FUNC() {
                        return *op;
                    }

                    // Check if function returns Step type
                    let func_ty_attr = op.attributes(db).get(&Symbol::new("type"));
                    let Some(Attribute::Type(func_ty)) = func_ty_attr else {
                        return *op;
                    };

                    // Get return type from function type using typed helper
                    let Some(func_type) = core::Func::from_type(db, *func_ty) else {
                        return *op;
                    };
                    let return_ty = func_type.result(db);

                    // Check if return type is Step (use step_type helper for consistency)
                    let step_ty = cont_types::step_type(db);
                    let is_step_return = return_ty == step_ty;

                    if !is_step_return {
                        return *op;
                    }

                    // This function returns Step - wrap its return values
                    wrap_returns_in_func(db, *op)
                })
                .collect();

            Block::new(
                db,
                block.id(db),
                block.location(db),
                block.args(db).clone(),
                new_ops,
            )
        })
        .collect();

    let new_body = Region::new(db, body.location(db), new_blocks);
    Module::create(db, module.location(db), module.name(db), new_body)
}

/// Wrap return values in a single function.
fn wrap_returns_in_func<'db>(
    db: &'db dyn salsa::Database,
    func_op: Operation<'db>,
) -> Operation<'db> {
    let regions = func_op.regions(db);
    if regions.is_empty() {
        return func_op;
    }

    let body = regions[0];
    let new_body = wrap_returns_in_region(db, body);

    func_op
        .modify(db)
        .regions(IdVec::from(vec![new_body]))
        .build()
}

/// Wrap return values in a region.
fn wrap_returns_in_region<'db>(db: &'db dyn salsa::Database, region: Region<'db>) -> Region<'db> {
    wrap_returns_in_region_with_map(db, region, &HashMap::new())
}

/// Wrap return values in a region, with an initial value map for remapping.
fn wrap_returns_in_region_with_map<'db>(
    db: &'db dyn salsa::Database,
    region: Region<'db>,
    initial_map: &HashMap<Value<'db>, Value<'db>>,
) -> Region<'db> {
    let blocks = region.blocks(db);

    let new_blocks: IdVec<Block<'db>> = blocks
        .iter()
        .map(|block| wrap_returns_in_block_with_map(db, *block, region, initial_map))
        .collect();

    Region::new(db, region.location(db), new_blocks)
}

/// Wrap return values in a block.
#[allow(dead_code)]
fn wrap_returns_in_block<'db>(
    db: &'db dyn salsa::Database,
    block: Block<'db>,
    region: Region<'db>,
) -> Block<'db> {
    wrap_returns_in_block_with_map(db, block, region, &HashMap::new())
}

/// Wrap return values in a block, with an initial value map for remapping.
fn wrap_returns_in_block_with_map<'db>(
    db: &'db dyn salsa::Database,
    block: Block<'db>,
    region: Region<'db>,
    initial_map: &HashMap<Value<'db>, Value<'db>>,
) -> Block<'db> {
    let mut new_ops = Vec::new();
    let mut value_map: HashMap<Value<'db>, Value<'db>> = initial_map.clone();
    let _step_ty = cont_types::step_type(db);

    for original_op in block.operations(db).iter() {
        // Remap operands first using current value map
        let mut op = remap_op_operands(db, *original_op, &value_map);
        let mut op_was_modified = op != *original_op;

        // Track the intermediate op for result mapping if operands were remapped
        let op_after_remap = op;

        // Recurse into nested regions, passing the current value_map
        if !op.regions(db).is_empty() {
            let new_regions: IdVec<Region<'db>> = op
                .regions(db)
                .iter()
                .map(|r| wrap_returns_in_region_with_map(db, *r, &value_map))
                .collect();

            // Just update the regions - don't modify result types.
            // The emit code will infer the correct block type from the branches.
            op = op.modify(db).regions(new_regions).build();
            op_was_modified = true;
        }

        // Map results from original op to final op if modified
        // This handles both operand remapping and region modification
        if op_was_modified {
            let old_results = original_op.results(db);
            let new_results = op.results(db);
            for i in 0..old_results.len().min(new_results.len()) {
                value_map.insert(original_op.result(db, i), op.result(db, i));
            }

            // Also map from intermediate op if it was different from original
            if op_after_remap != *original_op && op_after_remap != op {
                let intermediate_results = op_after_remap.results(db);
                for i in 0..intermediate_results.len().min(new_results.len()) {
                    value_map.insert(op_after_remap.result(db, i), op.result(db, i));
                }
            }
        }

        // Check if this is a wasm.return
        #[allow(clippy::collapsible_if)]
        if op.dialect(db) == wasm::DIALECT_NAME() && op.name(db) == wasm::RETURN() {
            if let Some(return_val) = op.operands(db).first().copied() {
                // Check if already a Step
                if !is_value_already_step(db, return_val) && !is_ref_null_any(db, return_val) {
                    // Wrap in Done Step
                    let location = op.location(db);

                    // Infer type from the return value's definition
                    let type_hint = match return_val.def(db) {
                        ValueDef::OpResult(def_op) => {
                            def_op.results(db).get(return_val.index(db)).copied()
                        }
                        ValueDef::BlockArg(block_id) => region
                            .blocks(db)
                            .iter()
                            .find(|b| b.id(db) == block_id)
                            .and_then(|block| block.args(db).get(return_val.index(db)))
                            .map(|arg| arg.ty(db)),
                    };

                    let (step_ops, step_val) =
                        create_done_step_ops(db, location, return_val, region, type_hint);
                    new_ops.extend(step_ops);

                    let new_return = wasm::r#return(db, location, Some(step_val));
                    new_ops.push(new_return.as_operation());
                    continue;
                }
            }
        }

        new_ops.push(op);
    }

    Block::new(
        db,
        block.id(db),
        block.location(db),
        block.args(db).clone(),
        IdVec::from(new_ops),
    )
}

/// Remap operands of an operation using the value map.
fn remap_op_operands<'db>(
    db: &'db dyn salsa::Database,
    op: Operation<'db>,
    value_map: &HashMap<Value<'db>, Value<'db>>,
) -> Operation<'db> {
    if value_map.is_empty() {
        return op;
    }

    let operands = op.operands(db);
    let mut new_operands = IdVec::new();
    let mut changed = false;

    for &operand in operands.iter() {
        if let Some(&new_val) = value_map.get(&operand) {
            new_operands.push(new_val);
            changed = true;
        } else {
            new_operands.push(operand);
        }
    }

    if changed {
        op.modify(db).operands(new_operands).build()
    } else {
        op
    }
}

#[allow(dead_code)]
/// Check if a region ends with a return statement (terminates without producing a value).
fn region_ends_with_return<'db>(db: &'db dyn salsa::Database, region: Region<'db>) -> bool {
    let blocks = region.blocks(db);
    let Some(last_block) = blocks.last() else {
        return false;
    };

    let ops = last_block.operations(db);
    let Some(last_op) = ops.last() else {
        return false;
    };

    last_op.dialect(db) == wasm::DIALECT_NAME() && last_op.name(db) == wasm::RETURN()
}

/// Check if a value is ref.null any (used for yield path returns).
fn is_ref_null_any<'db>(db: &'db dyn salsa::Database, value: Value<'db>) -> bool {
    let def = value.def(db);
    let ValueDef::OpResult(op) = def else {
        return false;
    };

    // Check operation name using dialect constant pattern
    if op.dialect(db) != wasm::DIALECT_NAME() || op.name(db) != Symbol::new("ref_null") {
        return false;
    }

    // Tighten filtering: verify it's actually anyref type
    let result_ty = op
        .results(db)
        .first()
        .copied()
        .unwrap_or_else(|| wasm::Anyref::new(db).as_type());
    wasm::Anyref::from_type(db, result_ty).is_some()
}

/// Create operations to wrap a value in Done Step (without return/yield).
/// Returns (operations, step_value).
///
/// The `value_type_hint` parameter can be provided when the caller knows the value's type,
/// which helps resolve BlockArgs from outer regions that aren't in the immediate `region`.
fn create_done_step_ops<'db>(
    db: &'db dyn salsa::Database,
    location: Location<'db>,
    value: Value<'db>,
    region: Region<'db>,
    value_type_hint: Option<Type<'db>>,
) -> (Vec<Operation<'db>>, Value<'db>) {
    let mut ops = Vec::new();
    let i32_ty = core::I32::new(db).as_type();
    let step_ty = cont_types::step_type(db);

    // Create Done tag (0)
    let done_tag = wasm::i32_const(db, location, i32_ty, cont_types::STEP_TAG_DONE);
    let done_tag_val = done_tag.as_operation().result(db, 0);
    ops.push(done_tag.as_operation());

    // Box the value to anyref for Step.value field
    // Try type hint first, then fall back to inference
    let value_ty = value_type_hint.or_else(|| match value.def(db) {
        ValueDef::OpResult(def_op) => def_op.results(db).get(value.index(db)).copied(),
        ValueDef::BlockArg(block_id) => {
            // Find the block in the region and get the argument type
            // Note: This only searches the immediate region. BlockArgs from outer regions
            // or different control flow contexts won't be found.
            let found_block = region.blocks(db).iter().find(|b| b.id(db) == block_id);

            if found_block.is_none() {
                tracing::debug!(
                    "create_done_step_ops: BlockArg block not found in immediate region (block_id={:?}). \
                     This may occur for BlockArgs from outer regions. Consider passing value_type_hint.",
                    block_id
                );
            }

            found_block
                .and_then(|block| block.args(db).get(value.index(db)))
                .map(|arg| arg.ty(db))
        }
    });
    let boxed_val = if let Some(ty) = value_ty {
        box_value_to_anyref(db, location, value, ty, &mut ops)
    } else {
        // CRITICAL: No type info available - this may produce invalid Step payloads
        // if the value is an unboxed primitive (i32, f64, etc.).
        // The caller should provide value_type_hint when possible to avoid this.
        tracing::warn!(
            "create_done_step_ops: No type info for value (def={:?}, index={}). \
             Assuming already boxed as anyref, but this may be incorrect if value is primitive. \
             This can cause invalid Step payloads and runtime failures. \
             Consider providing value_type_hint parameter.",
            value.def(db),
            value.index(db)
        );
        // Assume anyref - if this is wrong (e.g., unboxed i32), it will fail at runtime
        value
    };

    // Create zero for unused prompt and op_idx fields
    let zero_prompt_op = wasm::i32_const(db, location, i32_ty, 0);
    let zero_prompt = zero_prompt_op.as_operation().result(db, 0);
    ops.push(zero_prompt_op.as_operation());
    let zero_op_idx_op = wasm::i32_const(db, location, i32_ty, 0);
    let zero_op_idx = zero_op_idx_op.as_operation().result(db, 0);
    ops.push(zero_op_idx_op.as_operation());

    // Create Step struct: (tag=0, value=boxed_val, prompt=0, op_idx=0)
    let step = wasm::struct_new(
        db,
        location,
        IdVec::from(vec![done_tag_val, boxed_val, zero_prompt, zero_op_idx]),
        step_ty,
        crate::gc_types::STEP_IDX,
    );
    let step_val = step.as_operation().result(db, 0);
    ops.push(step.as_operation());

    (ops, step_val)
}

#[allow(dead_code)]
/// Create operations to wrap a value in Done Step and return it.
fn create_done_step_return<'db>(
    db: &'db dyn salsa::Database,
    location: Location<'db>,
    return_val: Value<'db>,
    region: Region<'db>,
) -> Vec<Operation<'db>> {
    // Infer type from return_val's definition
    let type_hint = match return_val.def(db) {
        ValueDef::OpResult(def_op) => def_op.results(db).get(return_val.index(db)).copied(),
        ValueDef::BlockArg(block_id) => region
            .blocks(db)
            .iter()
            .find(|b| b.id(db) == block_id)
            .and_then(|block| block.args(db).get(return_val.index(db)))
            .map(|arg| arg.ty(db)),
    };

    let (mut ops, step_val) = create_done_step_ops(db, location, return_val, region, type_hint);

    // Return the Step
    let return_op = wasm::r#return(db, location, Some(step_val));
    ops.push(return_op.as_operation());

    ops
}

/// Collect all function callees from a region (for call graph construction).
///
/// Note: This only collects direct `wasm.call` targets, not `wasm.call_indirect`.
/// Indirect calls go through funcref values (closures, continuations) whose targets
/// are resolved at runtime - there is no static callee symbol to extract.
///
/// Functions containing `call_indirect` are handled separately by
/// `function_body_has_call_indirect`, which conservatively marks them as effectful.
/// This is correct because indirect calls in this codebase are primarily for:
/// - Closures (which may capture effectful computations)
/// - Continuations (which always potentially yield)
fn collect_callees<'db>(
    db: &'db dyn salsa::Database,
    region: &Region<'db>,
) -> std::collections::HashSet<Symbol> {
    let mut callees = std::collections::HashSet::new();

    for block in region.blocks(db).iter() {
        for op in block.operations(db).iter() {
            // Check for wasm.call
            if op.dialect(db) == wasm::DIALECT_NAME()
                && op.name(db) == Symbol::new("call")
                && let Some(Attribute::Symbol(callee)) =
                    op.attributes(db).get(&Symbol::new("callee"))
            {
                callees.insert(*callee);
            }

            // Recursively check nested regions
            for nested_region in op.regions(db).iter() {
                callees.extend(collect_callees(db, nested_region));
            }
        }
    }

    callees
}

/// Check if a function body contains any call_indirect operations.
///
/// Functions with call_indirect (like handler arm lambdas calling continuations)
/// are considered potentially effectful because the target function may yield.
fn function_body_has_call_indirect<'db>(
    db: &'db dyn salsa::Database,
    region: &Region<'db>,
) -> bool {
    for block in region.blocks(db).iter() {
        for op in block.operations(db).iter() {
            if op.dialect(db) == wasm::DIALECT_NAME() && op.name(db) == Symbol::new("call_indirect")
            {
                return true;
            }
            // Check nested regions
            for nested_region in op.regions(db).iter() {
                if function_body_has_call_indirect(db, nested_region) {
                    return true;
                }
            }
        }
    }
    false
}

/// Check if a function body contains yield code.
///
/// Detects the Step pattern from shift expansion:
/// - wasm.return whose operand is from a wasm.struct_new with Step tag
fn function_body_can_yield<'db>(db: &'db dyn salsa::Database, region: &Region<'db>) -> bool {
    for block in region.blocks(db).iter() {
        for op in block.operations(db).iter() {
            // Check for wasm.return
            if op.dialect(db) == wasm::DIALECT_NAME()
                && op.name(db) == wasm::RETURN()
                && let Some(operand) = op.operands(db).first()
                && let ValueDef::OpResult(def_op) = operand.def(db)
            {
                // Check if it's a Step struct_new with Shift tag
                // Step struct has 4 fields: (tag, value, prompt, op_idx)
                if def_op.dialect(db) == wasm::DIALECT_NAME()
                    && def_op.name(db) == Symbol::new("struct_new")
                {
                    let operands = def_op.operands(db);
                    // Check for 4-field Step struct
                    if operands.len() == 4
                        && let Some(tag_val) = operands.first()
                        && let ValueDef::OpResult(tag_op) = tag_val.def(db)
                        && tag_op.dialect(db) == wasm::DIALECT_NAME()
                        && tag_op.name(db) == Symbol::new("i32_const")
                        && let Some(Attribute::IntBits(v)) =
                            tag_op.attributes(db).get(&Symbol::new("value"))
                        && *v as i32 == cont_types::STEP_TAG_SHIFT
                    {
                        return true;
                    }
                }
                // Also check for legacy pattern (ref.null any) for backwards compatibility
                if def_op.dialect(db) == wasm::DIALECT_NAME()
                    && def_op.name(db) == Symbol::new("ref_null")
                {
                    return true;
                }
            }
            // Check nested regions
            for nested_region in op.regions(db).iter() {
                if function_body_can_yield(db, nested_region) {
                    return true;
                }
            }
        }
    }
    false
}

/// Prepend operations to the first block of a module's body region.
fn prepend_operations_to_module<'db>(
    db: &'db dyn salsa::Database,
    module: Module<'db>,
    ops_to_prepend: &[Operation<'db>],
) -> Module<'db> {
    if ops_to_prepend.is_empty() {
        return module;
    }

    let body = module.body(db);
    let blocks = body.blocks(db);

    if blocks.is_empty() {
        // No blocks to prepend to - create a new block with just the ops
        let location = module.location(db);
        let new_block = trunk_ir::Block::new(
            db,
            trunk_ir::BlockId::fresh(),
            location,
            IdVec::new(),
            ops_to_prepend.iter().copied().collect(),
        );
        let new_body = trunk_ir::Region::new(db, location, IdVec::from(vec![new_block]));
        return Module::create(db, location, module.name(db), new_body);
    }

    // Prepend to the first block
    let first_block = &blocks[0];
    let existing_ops = first_block.operations(db);
    let mut all_ops: Vec<Operation<'db>> = ops_to_prepend.to_vec();
    all_ops.extend(existing_ops.iter().copied());

    let new_first_block = trunk_ir::Block::new(
        db,
        first_block.id(db),
        first_block.location(db),
        first_block.args(db).clone(),
        all_ops.into_iter().collect(),
    );

    // Rebuild the blocks list with the modified first block
    let mut new_blocks: Vec<trunk_ir::Block<'db>> = vec![new_first_block];
    new_blocks.extend(blocks.iter().skip(1).copied());

    let location = module.location(db);
    let new_body = trunk_ir::Region::new(db, location, new_blocks.into_iter().collect());
    Module::create(db, location, module.name(db), new_body)
}

/// Transform shift operations with inline resume function generation.
///
/// This function walks the module and transforms each `cont.shift` operation
/// into the yield bubbling sequence, capturing live locals in the state struct.
/// Resume functions are generated inline during transformation to avoid stale
/// SSA references when functions contain multiple shifts.
///
/// The transformation is done in a loop because expanding shifts inside resume
/// functions creates new shifts that need to be processed.
fn transform_shifts<'db>(
    db: &'db dyn salsa::Database,
    mut module: Module<'db>,
    analysis: &ContAnalysis<'db>,
    resume_fn_names: &BTreeMap<LocationKey, Symbol>,
) -> Module<'db> {
    let shift_points = analysis.shift_points(db);
    let location = module.location(db);

    tracing::debug!(
        "transform_shifts: {} shift points in analysis",
        shift_points.len()
    );

    // Track inline resume function counter for unique naming
    let mut inline_resume_counter: usize = 0;

    // Loop until no more shifts are found.
    // This handles nested shifts created when expanding shifts inside resume functions.
    const MAX_ITERATIONS: usize = 10;
    for iteration in 0..MAX_ITERATIONS {
        let body = module.body(db);
        let mut generated_resume_fns: Vec<Operation<'db>> = Vec::new();

        let (new_body, changed) = transform_shifts_in_region(
            db,
            &body,
            &shift_points,
            resume_fn_names,
            &mut generated_resume_fns,
            &mut inline_resume_counter,
        );

        tracing::debug!(
            "transform_shifts iteration {}: changed={}, generated {} resume fns",
            iteration,
            changed,
            generated_resume_fns.len()
        );

        if !changed && generated_resume_fns.is_empty() {
            break;
        }

        module = Module::create(db, location, module.name(db), new_body);

        // Prepend generated resume functions to the module
        if !generated_resume_fns.is_empty() {
            module = prepend_operations_to_module(db, module, &generated_resume_fns);
        }
    }

    module
}

/// Transform shifts in a region recursively.
fn transform_shifts_in_region<'db>(
    db: &'db dyn salsa::Database,
    region: &trunk_ir::Region<'db>,
    shift_points: &ShiftPointMap<'db>,
    resume_fn_names: &BTreeMap<LocationKey, Symbol>,
    generated_resume_fns: &mut Vec<Operation<'db>>,
    inline_resume_counter: &mut usize,
) -> (trunk_ir::Region<'db>, bool) {
    let mut changed = false;
    let new_blocks: IdVec<trunk_ir::Block<'db>> = region
        .blocks(db)
        .iter()
        .map(|block| {
            let (new_block, block_changed) = transform_shifts_in_block(
                db,
                block,
                shift_points,
                resume_fn_names,
                generated_resume_fns,
                inline_resume_counter,
            );
            changed |= block_changed;
            new_block
        })
        .collect();

    let new_region = trunk_ir::Region::new(db, region.location(db), new_blocks);
    (new_region, changed)
}

/// Transform shifts in a block with inline resume function generation.
fn transform_shifts_in_block<'db>(
    db: &'db dyn salsa::Database,
    block: &trunk_ir::Block<'db>,
    shift_points: &ShiftPointMap<'db>,
    resume_fn_names: &BTreeMap<LocationKey, Symbol>,
    generated_resume_fns: &mut Vec<Operation<'db>>,
    inline_resume_counter: &mut usize,
) -> (trunk_ir::Block<'db>, bool) {
    let mut changed = false;
    let mut new_ops: IdVec<Operation<'db>> = IdVec::new();

    // Collect block arguments for live local computation
    let block_args: Vec<Value<'db>> = (0..block.args(db).len())
        .map(|i| block.arg(db, i))
        .collect();

    let ops: Vec<Operation<'db>> = block.operations(db).iter().copied().collect();

    for (op_idx, &op) in ops.iter().enumerate() {
        // First, recursively process nested regions
        let op_with_processed_regions = if !op.regions(db).is_empty() {
            let mut region_changed = false;
            let new_regions: IdVec<trunk_ir::Region<'db>> = op
                .regions(db)
                .iter()
                .map(|region| {
                    let (new_region, rc) = transform_shifts_in_region(
                        db,
                        region,
                        shift_points,
                        resume_fn_names,
                        generated_resume_fns,
                        inline_resume_counter,
                    );
                    region_changed |= rc;
                    new_region
                })
                .collect();

            if region_changed {
                changed = true;
                op.modify(db).regions(new_regions).build()
            } else {
                op
            }
        } else {
            op
        };

        // Check if this is a shift operation
        if let Ok(_shift) = cont::Shift::from_operation(db, op_with_processed_regions) {
            let location = op_with_processed_regions.location(db);
            let location_key = LocationKey::from_location(db, location);

            tracing::debug!(
                "transform_shifts_in_block: found cont.shift at {:?}",
                location_key
            );

            // Compute live locals on-the-fly based on current block context
            let live_locals = compute_current_live_locals(db, block, &block_args, op_idx);

            // Collect continuation operations (operations after this shift)
            // Note: tribute.var operations are kept for LSP support and will be
            // skipped during emission in emit.rs
            let continuation_ops: Vec<Operation<'db>> =
                ops.iter().skip(op_idx + 1).copied().collect();

            // Get the shift's result value and type (used by continuation ops)
            let shift_result = op_with_processed_regions
                .results(db)
                .first()
                .map(|_| op_with_processed_regions.result(db, 0));
            let shift_result_type = op_with_processed_regions.results(db).first().copied();

            // Determine if we need a resume function
            let needs_resume_fn = !continuation_ops.is_empty() || !live_locals.is_empty();

            // Generate the resume function name if needed
            let resume_fn_name = if needs_resume_fn {
                let name = resume_fn_names
                    .get(&location_key)
                    .copied()
                    .unwrap_or_else(|| {
                        // Generate a unique name for inline resume functions
                        let name = Symbol::from_dynamic(&format!(
                            "__inline_resume${}",
                            *inline_resume_counter
                        ));
                        *inline_resume_counter += 1;
                        name
                    });

                // Generate the resume function inline
                let resume_fn = generate_inline_resume_function(
                    db,
                    location,
                    name,
                    &live_locals,
                    &continuation_ops,
                    shift_result,
                    shift_result_type,
                );
                generated_resume_fns.push(resume_fn);

                Some(name)
            } else {
                None
            };

            // Generate the shift expansion with computed live locals
            let expanded_ops = expand_shift_operation(
                db,
                &op_with_processed_regions,
                &live_locals,
                resume_fn_name,
            );

            new_ops.extend(expanded_ops);
            changed = true;

            // Skip remaining ops - they're now in the resume function
            break;
        } else {
            new_ops.push(op_with_processed_regions);
        }
    }

    let new_block = trunk_ir::Block::new(
        db,
        block.id(db),
        block.location(db),
        block.args(db).clone(),
        new_ops,
    );
    (new_block, changed)
}

/// Compute live locals on-the-fly based on current block context.
///
/// This computes live locals at the shift point by analyzing the current block
/// rather than relying on pre-computed analysis. This prevents stale SSA references
/// when shifts appear inside resume functions after transformation.
fn compute_current_live_locals<'db>(
    db: &'db dyn salsa::Database,
    block: &trunk_ir::Block<'db>,
    block_args: &[Value<'db>],
    shift_op_idx: usize,
) -> Vec<LiveLocal<'db>> {
    use std::collections::{HashMap, HashSet};

    let converter = wasm_type_converter();

    // Helper to convert types using the TypeConverter
    let convert_ty = |ty: Type<'db>| converter.convert_type(db, ty).unwrap_or(ty);

    // Collect values defined before the shift along with their (converted) types
    let mut defined_before: HashMap<Value<'db>, Type<'db>> = HashMap::new();

    // Block arguments are defined before any operation
    for (i, &arg) in block_args.iter().enumerate() {
        if let Some(block_arg) = block.args(db).get(i) {
            defined_before.insert(arg, convert_ty(block_arg.ty(db)));
        }
    }

    // Results of operations before the shift
    let ops: Vec<Operation<'db>> = block.operations(db).iter().copied().collect();
    for op in ops.iter().take(shift_op_idx) {
        let result_types = op.results(db);
        for i in 0..result_types.len() {
            let ty = convert_ty(result_types[i]);
            defined_before.insert(op.result(db, i), ty);
        }
    }

    // Collect values used after the shift
    let mut used_after: HashSet<Value<'db>> = HashSet::new();
    for op in ops.iter().skip(shift_op_idx + 1) {
        for operand in op.operands(db).iter() {
            used_after.insert(*operand);
        }
        for region in op.regions(db).iter() {
            collect_uses_in_region_recursive(db, region, &mut used_after);
        }
    }

    // Live locals = defined before ∩ used after
    let mut live_locals = Vec::new();
    for (value, ty) in defined_before {
        if used_after.contains(&value) {
            live_locals.push(LiveLocal {
                value,
                name: None,
                ty,
            });
        }
    }

    // Sort for deterministic ordering
    live_locals.sort_by_key(|l| match l.value.def(db) {
        ValueDef::BlockArg(block_id) => (0, block_id.0, l.value.index(db) as u64),
        ValueDef::OpResult(op) => (
            1,
            op.location(db).span.start as u64,
            l.value.index(db) as u64,
        ),
    });

    live_locals
}

/// Collect all value uses in a region recursively.
fn collect_uses_in_region_recursive<'db>(
    db: &'db dyn salsa::Database,
    region: &Region<'db>,
    uses: &mut std::collections::HashSet<Value<'db>>,
) {
    for block in region.blocks(db).iter() {
        for op in block.operations(db).iter() {
            for operand in op.operands(db).iter() {
                uses.insert(*operand);
            }
            for nested in op.regions(db).iter() {
                collect_uses_in_region_recursive(db, nested, uses);
            }
        }
    }
}

/// Box a primitive value and return it as anyref type.
///
/// This is needed for continuation struct fields and Step.value where all values
/// must be anyref to ensure consistent struct types across different shift points.
fn box_value_to_anyref<'db>(
    db: &'db dyn salsa::Database,
    location: Location<'db>,
    value: Value<'db>,
    value_ty: Type<'db>,
    ops: &mut Vec<Operation<'db>>,
) -> Value<'db> {
    let anyref_ty = wasm::Anyref::new(db).as_type();

    if tribute_rt::is_int(db, value_ty)
        || tribute_rt::is_nat(db, value_ty)
        || core::I32::from_type(db, value_ty).is_some()
    {
        // Box Int/Nat/I32 to i31ref, then mark as anyref
        // The actual boxing uses ref.i31, and we annotate the result as anyref
        // since anyref is a supertype of i31ref
        let ref_i31 = wasm::ref_i31(db, location, value, anyref_ty);
        ops.push(ref_i31.as_operation());
        ref_i31.result(db)
    } else if core::I64::from_type(db, value_ty).is_some() {
        // Box I64: i64 → i32 → anyref (truncate and wrap)
        let i32_ty = core::I32::new(db).as_type();
        let wrap = wasm::i32_wrap_i64(db, location, value, i32_ty);
        ops.push(wrap.as_operation());

        let ref_i31 = wasm::ref_i31(db, location, wrap.result(db), anyref_ty);
        ops.push(ref_i31.as_operation());
        ref_i31.result(db)
    } else if tribute_rt::is_float(db, value_ty) || tribute_rt::is_bool(db, value_ty) {
        panic!("box_value_to_anyref: float/bool boxing not yet implemented for continuations");
    } else {
        // Reference types are already compatible with anyref
        value
    }
}

/// Generate a resume function inline during shift transformation.
///
/// This creates a resume function with the current live locals and continuation ops,
/// avoiding stale SSA references that would occur if generated upfront.
///
/// The `shift_result` is the result value of the cont.shift operation. If the
/// continuation ops use this value, we map it to the resume_value extracted from
/// the wrapper struct (the value passed when the continuation is resumed).
///
/// The `shift_result_type` is the expected type of the shift result. If it's a
/// primitive type (Int, Nat), we need to unbox the resume_value (anyref → i32).
///
/// Resume function signature: (wrapper: anyref) -> anyref
/// Wrapper layout: (state: anyref, resume_value: anyref)
fn generate_inline_resume_function<'db>(
    db: &'db dyn salsa::Database,
    location: Location<'db>,
    name: Symbol,
    live_locals: &[LiveLocal<'db>],
    continuation_ops: &[Operation<'db>],
    shift_result: Option<Value<'db>>,
    shift_result_type: Option<Type<'db>>,
) -> Operation<'db> {
    use std::collections::HashMap;
    use trunk_ir::{BlockArg, BlockId};

    let anyref = wasm::Anyref::new(db).as_type();
    let structref_ty = wasm::Structref::new(db).as_type();
    let func_ty = cont_types::resume_fn_type(db);

    // Block argument is wrapper: anyref
    let block_args = IdVec::from(vec![BlockArg::of_type(db, anyref)]);
    let block_id = BlockId::fresh();

    // Get wrapper parameter
    let wrapper_param = Value::new(db, ValueDef::BlockArg(block_id), 0);

    use trunk_ir::rewrite::MaterializeResult;

    let mut ops: Vec<Operation<'db>> = Vec::new();
    let mut value_mapping: HashMap<Value<'db>, Value<'db>> = HashMap::new();

    // Extract state and resume_value from wrapper struct
    // Wrapper layout: (state: anyref, resume_value: anyref)
    const WRAPPER_FIELD_COUNT: u64 = 2;

    // Cast wrapper from anyref to structref
    let wrapper_cast_op = wasm::ref_cast(db, location, wrapper_param, structref_ty, structref_ty)
        .as_operation()
        .modify(db)
        .attr("field_count", Attribute::IntBits(WRAPPER_FIELD_COUNT))
        .build();
    let wrapper_cast = Value::new(db, ValueDef::OpResult(wrapper_cast_op), 0);
    ops.push(wrapper_cast_op);

    // Extract state from wrapper (field 0)
    let get_state = wasm::struct_get(
        db,
        location,
        wrapper_cast,
        anyref,
        0, // type_idx - Placeholder
        0, // field_idx for state
    )
    .as_operation()
    .modify(db)
    .attr("type", Attribute::Type(structref_ty))
    .attr("field_count", Attribute::IntBits(WRAPPER_FIELD_COUNT))
    .build();
    let state_val = Value::new(db, ValueDef::OpResult(get_state), 0);
    ops.push(get_state);

    // Extract resume_value from wrapper (field 1)
    let get_resume_value = wasm::struct_get(
        db,
        location,
        wrapper_cast,
        anyref,
        0, // type_idx - Placeholder
        1, // field_idx for resume_value
    )
    .as_operation()
    .modify(db)
    .attr("type", Attribute::Type(structref_ty))
    .attr("field_count", Attribute::IntBits(WRAPPER_FIELD_COUNT))
    .build();
    let resume_value = Value::new(db, ValueDef::OpResult(get_resume_value), 0);
    ops.push(get_resume_value);

    // Map the shift's result to the resume_value (possibly after unboxing)
    // The resume_value is anyref, but the shift result may expect a primitive type
    if let Some(shift_result_value) = shift_result {
        let mapped_value = if let Some(result_ty) = shift_result_type {
            // Use TypeConverter to materialize anyref → target_ty conversion
            match wasm_type_converter().materialize(db, location, resume_value, anyref, result_ty) {
                Some(MaterializeResult::Ops(generated_ops)) => {
                    let last_op = *generated_ops.last().unwrap();
                    ops.extend(generated_ops);
                    last_op.result(db, 0)
                }
                _ => resume_value,
            }
        } else {
            resume_value
        };
        value_mapping.insert(shift_result_value, mapped_value);
    }

    // Extract live locals from state struct
    // Cast state from anyref to structref for struct_get operations
    let field_count = live_locals.len();
    let state_cast = if !live_locals.is_empty() {
        // Add field_count attribute so emit can look up concrete type index from placeholder map
        let ref_cast_op = wasm::ref_cast(db, location, state_val, structref_ty, structref_ty)
            .as_operation()
            .modify(db)
            .attr("field_count", Attribute::IntBits(field_count as u64))
            .build();
        let cast_result = Value::new(db, ValueDef::OpResult(ref_cast_op), 0);
        ops.push(ref_cast_op);
        cast_result
    } else {
        state_val
    };

    for (field_idx, live_local) in live_locals.iter().enumerate() {
        // Generate struct_get to extract this field
        // Use structref as the type attribute - emit will resolve via placeholder map
        // live_local.ty is already converted by compute_current_live_locals
        let struct_get = wasm::struct_get(
            db,
            location,
            state_cast,
            live_local.ty,
            0, // type_idx - Will be resolved via placeholder map
            field_idx as u32,
        )
        .as_operation()
        .modify(db)
        .attr("type", Attribute::Type(structref_ty))
        .attr("field_count", Attribute::IntBits(field_count as u64))
        .build();

        let extracted_value = Value::new(db, ValueDef::OpResult(struct_get), 0);
        ops.push(struct_get);

        // Map the original value to the extracted value
        value_mapping.insert(live_local.value, extracted_value);
    }

    // Remap continuation operations and add them to the body
    let i32_ty = core::I32::new(db).as_type();
    let step_ty = cont_types::step_type(db);

    if !continuation_ops.is_empty() {
        let remapped_ops = resume_gen::remap_operations(db, continuation_ops, &mut value_mapping);

        // Post-process: wrap return values in Done Step
        // Find return operations and wrap their operands (unless already Step)
        for remapped_op in remapped_ops {
            if remapped_op.dialect(db) == wasm::DIALECT_NAME()
                && remapped_op.name(db) == wasm::RETURN()
            {
                // This is a wasm.return - wrap its operand in Done Step if not already Step
                if let Some(return_val) = remapped_op.operands(db).first().copied() {
                    // Check if the return value is already a Step to prevent double-wrapping
                    if is_value_already_step(db, return_val) {
                        // Already a Step - just return it directly
                        ops.push(wasm::r#return(db, location, Some(return_val)).as_operation());
                    } else {
                        // Not a Step - wrap in Done Step
                        // Create Done tag (0)
                        let done_tag =
                            wasm::i32_const(db, location, i32_ty, cont_types::STEP_TAG_DONE);
                        let done_tag_val = done_tag.as_operation().result(db, 0);
                        ops.push(done_tag.as_operation());

                        // Box the return value to anyref for Step.value field
                        let return_val_ty = match return_val.def(db) {
                            ValueDef::OpResult(def_op) => {
                                def_op.results(db).get(return_val.index(db)).copied()
                            }
                            ValueDef::BlockArg(_) => None,
                        };
                        let boxed_return_val = if let Some(ty) = return_val_ty {
                            box_value_to_anyref(db, location, return_val, ty, &mut ops)
                        } else {
                            return_val
                        };

                        // Create zero for unused prompt and op_idx fields
                        let zero_prompt_op = wasm::i32_const(db, location, i32_ty, 0);
                        let zero_prompt = zero_prompt_op.as_operation().result(db, 0);
                        ops.push(zero_prompt_op.as_operation());
                        let zero_op_idx_op = wasm::i32_const(db, location, i32_ty, 0);
                        let zero_op_idx = zero_op_idx_op.as_operation().result(db, 0);
                        ops.push(zero_op_idx_op.as_operation());

                        // Create Step struct: (tag=0, value=boxed_return_val, prompt=0, op_idx=0)
                        let step = wasm::struct_new(
                            db,
                            location,
                            IdVec::from(vec![
                                done_tag_val,
                                boxed_return_val,
                                zero_prompt,
                                zero_op_idx,
                            ]),
                            step_ty,
                            crate::gc_types::STEP_IDX,
                        );
                        let step_val = step.as_operation().result(db, 0);
                        ops.push(step.as_operation());

                        ops.push(wasm::r#return(db, location, Some(step_val)).as_operation());
                    }
                } else {
                    // No operand - create nil Step
                    let done_tag = wasm::i32_const(db, location, i32_ty, cont_types::STEP_TAG_DONE);
                    let done_tag_val = done_tag.as_operation().result(db, 0);
                    ops.push(done_tag.as_operation());

                    let anyref_ty = wasm::Anyref::new(db).as_type();
                    let null_ref = wasm::ref_null(db, location, anyref_ty, anyref_ty);
                    let null_val = null_ref.as_operation().result(db, 0);
                    ops.push(null_ref.as_operation());

                    // Create zero for unused prompt and op_idx fields
                    let zero_prompt_op = wasm::i32_const(db, location, i32_ty, 0);
                    let zero_prompt = zero_prompt_op.as_operation().result(db, 0);
                    ops.push(zero_prompt_op.as_operation());
                    let zero_op_idx_op = wasm::i32_const(db, location, i32_ty, 0);
                    let zero_op_idx = zero_op_idx_op.as_operation().result(db, 0);
                    ops.push(zero_op_idx_op.as_operation());

                    let step = wasm::struct_new(
                        db,
                        location,
                        IdVec::from(vec![done_tag_val, null_val, zero_prompt, zero_op_idx]),
                        step_ty,
                        crate::gc_types::STEP_IDX,
                    );
                    let step_val = step.as_operation().result(db, 0);
                    ops.push(step.as_operation());

                    ops.push(wasm::r#return(db, location, Some(step_val)).as_operation());
                }
            } else {
                // Not a return - keep as-is
                ops.push(remapped_op);
            }
        }
    } else {
        // No continuation body - wrap resume_value in Done Step and return

        // Create Done tag (0)
        let done_tag = wasm::i32_const(db, location, i32_ty, cont_types::STEP_TAG_DONE);
        let done_tag_val = done_tag.as_operation().result(db, 0);
        ops.push(done_tag.as_operation());

        // Create zero for unused prompt and op_idx fields
        let zero_prompt_op = wasm::i32_const(db, location, i32_ty, 0);
        let zero_prompt = zero_prompt_op.as_operation().result(db, 0);
        ops.push(zero_prompt_op.as_operation());
        let zero_op_idx_op = wasm::i32_const(db, location, i32_ty, 0);
        let zero_op_idx = zero_op_idx_op.as_operation().result(db, 0);
        ops.push(zero_op_idx_op.as_operation());

        // Create Step struct: (tag=0, value=resume_value, prompt=0, op_idx=0)
        let step = wasm::struct_new(
            db,
            location,
            IdVec::from(vec![done_tag_val, resume_value, zero_prompt, zero_op_idx]),
            step_ty,
            crate::gc_types::STEP_IDX,
        );
        let step_val = step.as_operation().result(db, 0);
        ops.push(step.as_operation());

        ops.push(wasm::r#return(db, location, Some(step_val)).as_operation());
    }

    // Create function body block with typed arguments
    let body_block = Block::new(
        db,
        block_id,
        location,
        block_args,
        ops.into_iter().collect(),
    );
    let body_region = Region::new(db, location, IdVec::from(vec![body_block]));

    // Create the wasm.func operation
    wasm::func(db, location, name, func_ty, body_region).as_operation()
}

/// Expand a shift operation into the yield bubbling sequence.
///
/// This generates:
/// 1. State struct with captured live locals
/// 2. Continuation struct (resume_fn, state, tag)
/// 3. Set yield globals
/// 4. Return to unwind
fn expand_shift_operation<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
    live_locals: &[LiveLocal<'db>],
    resume_fn_name: Option<Symbol>,
) -> Vec<Operation<'db>> {
    let location = op.location(db);
    let i32_ty = core::I32::new(db).as_type();
    let funcref_ty = wasm::Funcref::new(db).as_type();
    let anyref_ty = wasm::Anyref::new(db).as_type();

    // Get shift operation attributes using typed helper
    let shift_op = cont::Shift::from_operation(db, *op)
        .expect("expand_shift_operation called on non-shift operation");

    let tag_value = shift_op.tag(db) as u64;

    // Compute op_idx deterministically from op_name
    let op_name = shift_op.op_name(db);
    let op_idx_value = op_name.with_str(compute_op_idx_hash);

    // Also available for debugging/validation:
    // let ability_ref = shift_op.ability_ref(db);

    let mut ops = Vec::new();

    // === 1. Build State Struct (capturing live locals) ===
    let state_ty = cont_types::state_type(db, live_locals);

    // Collect the values to capture
    let live_values: IdVec<Value<'db>> = live_locals.iter().map(|ll| ll.value).collect();

    // Debug: log value types with more details
    for (idx, ll) in live_locals.iter().enumerate() {
        let (value_ty, op_info) = match ll.value.def(db) {
            ValueDef::OpResult(op) => {
                let ty = op.results(db).get(ll.value.index(db)).copied();
                let info = format!("{}.{}", op.dialect(db), op.name(db));
                (ty, info)
            }
            ValueDef::BlockArg(_) => (None, "block_arg".to_string()),
        };
        tracing::debug!(
            "build_continuation_struct: live_local[{}] from {} recorded_ty={}.{} actual_value_ty={:?}",
            idx,
            op_info,
            ll.ty.dialect(db),
            ll.ty.name(db),
            value_ty.map(|t| format!("{}.{}", t.dialect(db), t.name(db)))
        );
    }

    let state_struct = wasm::struct_new(
        db,
        location,
        live_values,
        state_ty,
        0, // type_idx - Placeholder, resolved at emit time
    );
    let state_val = Value::new(db, ValueDef::OpResult(state_struct.as_operation()), 0);
    ops.push(state_struct.as_operation());

    // === 2. Create continuation struct: (resume_fn, state, tag) ===
    // Field 0: resume_fn - reference to the generated resume function
    let resume_fn_val = if let Some(resume_fn_name) = resume_fn_name {
        // Use ref.func to get function reference
        let ref_func = wasm::ref_func(db, location, funcref_ty, resume_fn_name);
        let val = ref_func.as_operation().result(db, 0);
        ops.push(ref_func.as_operation());
        val
    } else {
        // Fallback: use null funcref if no resume function was generated
        let null_resume_fn = wasm::ref_null(db, location, funcref_ty, funcref_ty);
        let val = null_resume_fn.as_operation().result(db, 0);
        ops.push(null_resume_fn.as_operation());
        val
    };

    // Field 1: state (the state struct we just created)
    // Already have state_val

    // Field 2: tag (i32)
    let tag_const = wasm::i32_const(db, location, i32_ty, tag_value as i32);
    let tag_val = tag_const.as_operation().result(db, 0);
    ops.push(tag_const.as_operation());

    // Field 3: shift_value (anyref) - the value yielded by shift
    // Primitives need to be boxed since this field is anyref
    let shift_operands = op.operands(db);
    let shift_value_val = if let Some(&first_value) = shift_operands.first() {
        // Get the type of the value to check if boxing is needed
        let value_ty = match first_value.def(db) {
            ValueDef::OpResult(def_op) => def_op.results(db).get(first_value.index(db)).copied(),
            ValueDef::BlockArg(_) => None,
        };
        // Box primitive values before storing in anyref field
        // Use box_value_to_anyref which ensures consistent anyref type
        if let Some(ty) = value_ty {
            box_value_to_anyref(db, location, first_value, ty, &mut ops)
        } else {
            first_value
        }
    } else {
        // No value provided - use null
        let null_value = wasm::ref_null(db, location, anyref_ty, anyref_ty);
        let val = null_value.as_operation().result(db, 0);
        ops.push(null_value.as_operation());
        val
    };

    // Build the continuation struct with 4 fields: (resume_fn, state, tag, shift_value)
    let cont_ty = cont_types::continuation_type(db);
    let cont_struct = wasm::struct_new(
        db,
        location,
        IdVec::from(vec![resume_fn_val, state_val, tag_val, shift_value_val]),
        cont_ty,
        0, // type_idx - Placeholder, resolved at emit time
    );
    let cont_val = Value::new(db, ValueDef::OpResult(cont_struct.as_operation()), 0);
    ops.push(cont_struct.as_operation());

    // === 3. Set Yield Globals ===
    // These are still needed for multi-prompt dispatch (which tag to handle)

    // Set $yield_state = 1 (yielding)
    let const_1 = wasm::i32_const(db, location, i32_ty, 1);
    let const_1_val = const_1.as_operation().result(db, 0);
    ops.push(const_1.as_operation());
    ops.push(wasm::global_set(db, location, const_1_val, YIELD_STATE_IDX).as_operation());

    // Set $yield_tag = tag
    let tag_const2 = wasm::i32_const(db, location, i32_ty, tag_value as i32);
    let tag_const2_val = tag_const2.as_operation().result(db, 0);
    ops.push(tag_const2.as_operation());
    ops.push(wasm::global_set(db, location, tag_const2_val, YIELD_TAG_IDX).as_operation());

    // Set $yield_cont = continuation struct
    ops.push(wasm::global_set(db, location, cont_val, YIELD_CONT_IDX).as_operation());

    // Set $yield_op_idx = op_idx (for multi-op ability dispatch)
    let op_idx_const = wasm::i32_const(db, location, i32_ty, op_idx_value as i32);
    let op_idx_val = op_idx_const.as_operation().result(db, 0);
    ops.push(op_idx_const.as_operation());
    ops.push(wasm::global_set(db, location, op_idx_val, YIELD_OP_IDX).as_operation());

    // === 4. Return Step with tag=1 (Shift) ===
    // Create Step struct: (tag: i32, value: anyref, prompt: i32, op_idx: i32)
    // - tag = 1 (STEP_TAG_SHIFT)
    // - value = continuation struct
    // - prompt = tag_value (handler's prompt tag)
    // - op_idx = op_idx_value (ability operation index)
    let step_ty = cont_types::step_type(db);
    let shift_tag_const = wasm::i32_const(db, location, i32_ty, cont_types::STEP_TAG_SHIFT);
    let shift_tag_val = shift_tag_const.as_operation().result(db, 0);
    ops.push(shift_tag_const.as_operation());

    // Prompt tag for handler dispatch
    let prompt_const = wasm::i32_const(db, location, i32_ty, tag_value as i32);
    let prompt_val = prompt_const.as_operation().result(db, 0);
    ops.push(prompt_const.as_operation());

    // Op index for multi-op ability dispatch
    let op_idx_const2 = wasm::i32_const(db, location, i32_ty, op_idx_value as i32);
    let op_idx_val2 = op_idx_const2.as_operation().result(db, 0);
    ops.push(op_idx_const2.as_operation());

    let step = wasm::struct_new(
        db,
        location,
        IdVec::from(vec![shift_tag_val, cont_val, prompt_val, op_idx_val2]),
        step_ty,
        crate::gc_types::STEP_IDX, // Fixed type index for Step
    );
    let step_val = step.as_operation().result(db, 0);
    ops.push(step.as_operation());

    ops.push(wasm::r#return(db, location, Some(step_val)).as_operation());

    ops
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
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(push_prompt) = cont::PushPrompt::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let location = op.location(db);
        let i32_ty = core::I32::new(db).as_type();

        // Get the prompt tag using typed helper
        let tag_value = push_prompt.tag(db) as u64;

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
        let our_tag = wasm::i32_const(db, location, i32_ty, tag_value as i32);
        let our_tag_val = our_tag.as_operation().result(db, 0);

        // Compare: yield_tag == our_tag
        let tag_cmp = wasm::i32_eq(db, location, yield_tag_val, our_tag_val, i32_ty);
        let tag_match_val = tag_cmp.as_operation().result(db, 0);

        // If tag matches: handle the yielded effect
        //
        // Handler invocation flow:
        // 1. Reset yield_state to 0 (we're handling this yield)
        // 2. Load the continuation from $yield_cont
        // 3. Extract shift_value from continuation struct (field 3)
        // 4. Call the handler with (continuation, shift_value)
        //
        // Currently, step 4 requires upstream changes:
        // - handler_lower.rs needs to populate shift's handler region with actual code
        // - The handler function would be passed via a global or stored in the continuation
        //
        // For now, we reset yield_state and load the values as preparation.

        let const_0 = wasm::i32_const(db, location, i32_ty, 0);
        let const_0_val = const_0.as_operation().result(db, 0);
        let reset_yield = wasm::global_set(db, location, const_0_val, YIELD_STATE_IDX);

        // Load continuation from $yield_cont for handler invocation
        // Global $yield_cont has type anyref, so we need to cast to structref
        let anyref_ty = wasm::Anyref::new(db).as_type();
        let structref_ty = wasm::Structref::new(db).as_type();
        let get_cont = wasm::global_get(db, location, anyref_ty, YIELD_CONT_IDX);
        let cont_anyref = get_cont.as_operation().result(db, 0);

        // Extract shift_value from continuation struct (field 3)
        // Continuation layout: (resume_fn, state, tag, shift_value)
        const SHIFT_VALUE_FIELD_IDX: u32 = 3;
        const CONT_FIELD_COUNT: u64 = 4;

        // Cast anyref to structref for struct.get
        // Include field_count so emit.rs can find the correct struct type
        let cont_cast_op = wasm::ref_cast(db, location, cont_anyref, structref_ty, structref_ty)
            .as_operation()
            .modify(db)
            .attr("field_count", Attribute::IntBits(CONT_FIELD_COUNT))
            .build();
        let cont_val = Value::new(db, ValueDef::OpResult(cont_cast_op), 0);
        let get_value = wasm::struct_get(
            db,
            location,
            cont_val,
            anyref_ty,
            0, // type_idx - Placeholder, resolved at emit time
            SHIFT_VALUE_FIELD_IDX,
        )
        .as_operation()
        .modify(db)
        .attr("type", Attribute::Type(structref_ty))
        .attr("field_count", Attribute::IntBits(CONT_FIELD_COUNT))
        .build();
        let value_val = Value::new(db, ValueDef::OpResult(get_value), 0);

        // Drop the loaded values until handler invocation is implemented
        // TODO(#100): Replace these drops with actual handler call
        let drop_cont = wasm::drop(db, location, cont_val);
        let drop_value = wasm::drop(db, location, value_val);

        // Build the "then" block for tag match (handle)
        // TODO(#100): Call handler function with (continuation, value)
        // The handler function reference needs to be provided by handler_lower.rs
        // via the shift's handler region or a separate mechanism.
        let then_block = trunk_ir::Block::new(
            db,
            trunk_ir::BlockId::fresh(),
            location,
            IdVec::new(),
            IdVec::from(vec![
                const_0.as_operation(),
                reset_yield.as_operation(),
                get_cont.as_operation(),
                cont_cast_op,
                get_value,
                drop_cont.as_operation(),
                drop_value.as_operation(),
            ]),
        );
        let then_region = trunk_ir::Region::new(db, location, IdVec::from(vec![then_block]));

        // Build the "else" block for tag mismatch (propagate)
        // Return the body's result (the Step from inner call) to continue bubbling up
        // We need to find the body's result value - it's the last operation's result in the body
        let body_result_value = get_body_result_value(db, &original_body);
        let return_op = wasm::r#return(db, location, body_result_value);
        let else_block = trunk_ir::Block::new(
            db,
            trunk_ir::BlockId::fresh(),
            location,
            IdVec::new(),
            IdVec::from(vec![return_op.as_operation()]),
        );
        let else_region = trunk_ir::Region::new(db, location, IdVec::from(vec![else_block]));

        // Inner if: check tag match - no result type needed
        let nil_ty = core::Nil::new(db).as_type();
        let inner_if = wasm::r#if(
            db,
            location,
            tag_match_val,
            nil_ty,
            then_region,
            else_region,
        );

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
                inner_if.as_operation(),
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
        let outer_if = wasm::r#if(
            db,
            location,
            yield_state_val,
            nil_ty,
            outer_then_region,
            outer_else_region,
        );

        yield_check_ops.push(outer_if.as_operation());

        // Wrap body result in Done Step and add yield check.
        // This ensures push_prompt always returns Step, which handler_dispatch expects.
        let new_body = if result_types.is_empty()
            || result_types
                .first()
                .map(|ty| is_nil_type(db, *ty))
                .unwrap_or(true)
        {
            // No result - just append yield check
            append_ops_to_region(db, location, &original_body, &yield_check_ops)
        } else {
            // Has result - wrap in Done Step, then add yield check
            let body_result = get_body_result_value(db, &original_body);

            if let Some(result_val) = body_result {
                // Check if result is already a Step - avoid double wrapping (Done(Step))
                if is_value_already_step(db, result_val) {
                    // Already a Step - just add yield check without wrapping
                    insert_ops_before_last(db, location, &original_body, &yield_check_ops)
                } else {
                    // Not a Step - wrap in Done Step
                    // Pass type hint from result_types to prevent invalid Step payloads
                    let type_hint = result_types.first().copied();
                    let (step_ops, step_val) =
                        create_done_step_ops(db, location, result_val, original_body, type_hint);

                    let blocks = original_body.blocks(db);
                    if let Some(last_block) = blocks.last() {
                        let ops = last_block.operations(db);
                        let mut new_ops: Vec<Operation<'db>> = Vec::new();

                        // Keep all ops (including the last result-producing one)
                        new_ops.extend(ops.iter().copied());

                        // Add Step wrapping
                        new_ops.extend(step_ops);

                        // Add yield check
                        new_ops.extend(yield_check_ops.iter().copied());

                        // Add wasm.yield with Step value
                        let yield_op = wasm::r#yield(db, location, step_val);
                        new_ops.push(yield_op.as_operation());

                        let new_block = trunk_ir::Block::new(
                            db,
                            last_block.id(db),
                            last_block.location(db),
                            last_block.args(db).clone(),
                            IdVec::from(new_ops),
                        );

                        let mut new_blocks: Vec<trunk_ir::Block<'db>> =
                            blocks.iter().take(blocks.len() - 1).copied().collect();
                        new_blocks.push(new_block);

                        trunk_ir::Region::new(db, location, IdVec::from(new_blocks))
                    } else {
                        append_ops_to_region(db, location, &original_body, &yield_check_ops)
                    }
                }
            } else {
                insert_ops_before_last(db, location, &original_body, &yield_check_ops)
            }
        };

        // Create the wrapper block with Step result type.
        // Push_prompt always returns Step (either from yield or Done Step wrapping).
        let step_ty = cont_types::step_type(db);
        let new_op = wasm::block(db, location, step_ty, new_body).as_operation();

        RewriteResult::Replace(new_op)
    }
}

/// Get the result value from a body region.
///
/// This finds the last operation's result in the last block of the region.
/// Used for bubbling: when a handler's tag doesn't match, we need to
/// return the inner call's result (which contains the Step).
fn get_body_result_value<'db>(
    db: &'db dyn salsa::Database,
    body: &trunk_ir::Region<'db>,
) -> Option<Value<'db>> {
    let blocks = body.blocks(db);
    let last_block = blocks.last()?;
    let ops = last_block.operations(db);
    let last_op = ops.last().copied()?;

    // Special case: wasm.yield has a value operand but no SSA result.
    // The value operand is the body's result.
    if let Ok(yield_op) = wasm::Yield::from_operation(db, last_op) {
        return Some(yield_op.value(db));
    }

    let results = last_op.results(db);
    if results.is_empty() {
        None
    } else {
        Some(Value::new(db, ValueDef::OpResult(last_op), 0))
    }
}

/// Check if a type is nil type
fn is_nil_type(db: &dyn salsa::Database, ty: Type<'_>) -> bool {
    ty.dialect(db) == core::DIALECT_NAME() && ty.name(db) == core::NIL()
}

/// Check if a value is already a Step struct.
///
/// This is used to prevent double-wrapping in resume functions.
/// When the continuation body already returns a Step (e.g., from another
/// effectful call or handler dispatch), we should not wrap it again.
fn is_value_already_step<'db>(db: &'db dyn salsa::Database, value: Value<'db>) -> bool {
    let def = value.def(db);
    let ValueDef::OpResult(op) = def else {
        return false;
    };

    // Check if it's a struct.new operation with STEP_IDX
    if op.dialect(db) == wasm::DIALECT_NAME() && op.name(db) == Symbol::new("struct_new") {
        // Check if the type_idx attribute matches STEP_IDX
        if let Some(Attribute::IntBits(type_idx)) = op.attributes(db).get(&Symbol::new("type_idx"))
        {
            return *type_idx == crate::gc_types::STEP_IDX as u64;
        }
    }

    // Also check the result type - if it's the Step marker type
    if let Some(result_ty) = op.results(db).first() {
        return result_ty.dialect(db) == wasm::DIALECT_NAME()
            && result_ty.name(db) == Symbol::new("step");
    }

    false
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

/// Insert operations before the last operation in the region's last block.
/// This preserves the last operation (typically the result-producing one) at the end.
fn insert_ops_before_last<'db>(
    db: &'db dyn salsa::Database,
    location: Location<'db>,
    region: &trunk_ir::Region<'db>,
    ops_to_insert: &[Operation<'db>],
) -> trunk_ir::Region<'db> {
    let blocks = region.blocks(db);
    if blocks.is_empty() {
        // Create a new block with just the inserted ops
        let new_block = trunk_ir::Block::new(
            db,
            trunk_ir::BlockId::fresh(),
            location,
            IdVec::new(),
            ops_to_insert.iter().copied().collect(),
        );
        return trunk_ir::Region::new(db, location, IdVec::from(vec![new_block]));
    }

    // Clone all blocks, modifying only the last one
    let mut new_blocks: Vec<trunk_ir::Block<'db>> = Vec::with_capacity(blocks.len());
    for (i, block) in blocks.iter().enumerate() {
        if i == blocks.len() - 1 {
            // Last block: insert operations before the last op
            let existing_ops = block.operations(db);
            let mut all_ops: Vec<Operation<'db>> =
                Vec::with_capacity(existing_ops.len() + ops_to_insert.len());

            if existing_ops.is_empty() {
                // No existing ops, just add the inserted ones
                all_ops.extend(ops_to_insert.iter().copied());
            } else {
                // Add all ops except last
                for op in existing_ops.iter().take(existing_ops.len() - 1) {
                    all_ops.push(*op);
                }
                // Insert new ops
                all_ops.extend(ops_to_insert.iter().copied());
                // Add last op at the end
                if let Some(last_op) = existing_ops.last() {
                    all_ops.push(*last_op);
                }
            }

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

// Yield global indices (must match order in lower_wasm.rs module_preamble_ops)
// Note: $yield_value is no longer used - shift_value is stored in continuation struct
const YIELD_STATE_IDX: u32 = 0;
const YIELD_TAG_IDX: u32 = 1;
const YIELD_CONT_IDX: u32 = 2;
const YIELD_OP_IDX: u32 = 3;

/// Compute a deterministic hash-based index from an operation name.
/// This is used by both shift expansion and handler dispatch to ensure
/// they agree on the same index for the same operation.
fn compute_op_idx_hash(name: &str) -> u64 {
    name.bytes()
        .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64))
        & 0xFFFF // Keep lower 16 bits
}

/// Pattern for `cont.handler_dispatch` -> yield-checking if/else with multi-arm dispatch
///
/// This operation is generated by case_lowering for handler patterns.
/// It dispatches based on the yield state and op_idx:
/// - If yield_state == 0 (not yielding): execute done_body
/// - If yield_state != 0 (yielding): check op_idx and dispatch to correct suspend_body
///
/// Layout of cont.handler_dispatch:
/// - Region 0: done_body
/// - Region 1+: suspend_bodies (one per handler arm)
/// - Attribute `op_idx_N`: hash-based index for suspend arm N
/// - Attribute `num_suspend_arms`: number of suspend arms
struct HandlerDispatchPattern;

impl RewritePattern for HandlerDispatchPattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(_dispatch) = cont::HandlerDispatch::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let location = op.location(db);
        let i32_ty = core::I32::new(db).as_type();

        // Get all regions: first is done_body, rest are suspend_bodies
        let regions = op.regions(db);
        let done_body = regions
            .first()
            .cloned()
            .unwrap_or_else(|| Region::new(db, location, IdVec::new()));

        // Get the number of suspend arms
        let num_suspend_arms = op
            .attributes(db)
            .get(&Symbol::new("num_suspend_arms"))
            .and_then(|attr| {
                if let Attribute::IntBits(v) = attr {
                    Some(*v as usize)
                } else {
                    None
                }
            })
            .unwrap_or(0);

        // Collect suspend bodies with their op_idx values (computed from op_name)
        let mut suspend_arms: Vec<(u64, Region<'db>)> = Vec::new();
        for i in 0..num_suspend_arms {
            // Get the suspend body region (offset by 1 for done_body)
            let body = regions
                .get(i + 1)
                .cloned()
                .unwrap_or_else(|| Region::new(db, location, IdVec::new()));

            // Get the op_name for this arm and compute hash-based op_idx
            let attr_name = format!("op_name_{}", i);
            let op_idx = op
                .attributes(db)
                .get(&Symbol::from_dynamic(&attr_name))
                .and_then(|attr| {
                    if let Attribute::Symbol(s) = attr {
                        Some(s.with_str(compute_op_idx_hash))
                    } else {
                        None
                    }
                })
                .unwrap_or(0);

            suspend_arms.push((op_idx, body));
        }

        // Use Step type for wasm.if result since both branches yield Step
        let step_ty = crate::gc_types::step_marker_type(db);

        // Check yield_state: global.get $yield_state
        let get_yield_state = wasm::global_get(db, location, i32_ty, YIELD_STATE_IDX);
        let yield_state_val = get_yield_state.as_operation().result(db, 0);

        // Wrap done_body yields in Done Step (so both branches produce Step)
        let wrapped_done_body = wrap_yields_in_done_step(db, done_body);

        // Wrap suspend arms' yields in Done Step (so both branches produce Step).
        // Suspend arms may also return values directly (not through yield),
        // which need to be wrapped in Step structures.
        let wrapped_suspend_arms: Vec<(u64, Region<'db>)> = suspend_arms
            .iter()
            .map(|(op_idx, body)| (*op_idx, wrap_yields_in_done_step(db, *body)))
            .collect();

        // Build the suspend dispatch logic
        // If we have multiple suspend arms, build a nested if-chain based on op_idx
        let suspend_body = if wrapped_suspend_arms.len() <= 1 {
            // Single or no suspend arm - use directly
            wrapped_suspend_arms
                .first()
                .map(|(_, body)| *body)
                .unwrap_or_else(|| Region::new(db, location, IdVec::new()))
        } else {
            // Multiple suspend arms - build dispatch chain
            // if (op_idx == arm0_idx) { arm0_body }
            // else if (op_idx == arm1_idx) { arm1_body }
            // else { last_arm_body }
            build_multi_arm_dispatch(db, location, i32_ty, step_ty, &wrapped_suspend_arms)
        };

        // Build wasm.if with:
        // - condition: yield_state (non-zero means yielding)
        // - then: suspend_body (yielding case with dispatch - already yields Step)
        // - else: done_body (normal completion case - now wrapped to yield Step)
        let if_op = wasm::r#if(
            db,
            location,
            yield_state_val,
            step_ty,
            suspend_body,
            wrapped_done_body,
        );

        // Expand to: [global_get, if]
        RewriteResult::Expand(vec![get_yield_state.as_operation(), if_op.as_operation()])
    }
}

/// Handles implicit yield for the last block if needed.
///
/// If the last block doesn't end with a yield, wraps the last operation's
/// result in Done Step and yields it.
fn handle_last_block_implicit_yield<'db>(
    db: &'db dyn salsa::Database,
    new_ops: &mut Vec<Operation<'db>>,
    region: Region<'db>,
) {
    let Some(&last_op_ref) = new_ops.last() else {
        return;
    };

    let last_op = last_op_ref;

    // Check if last operation is already a yield
    let is_yield = wasm::Yield::from_operation(db, last_op).is_ok();

    // Check if last operation is cont.get_done_value (not yet expanded)
    let is_get_done_value =
        last_op.dialect(db) == cont::DIALECT_NAME() && last_op.name(db) == cont::GET_DONE_VALUE();

    tracing::debug!(
        "handle_last_block_implicit_yield: last_op={}.{}, is_yield={}, is_get_done_value={}, has_results={}",
        last_op.dialect(db),
        last_op.name(db),
        is_yield,
        is_get_done_value,
        !last_op.results(db).is_empty()
    );

    if is_get_done_value {
        // Remove the cont.get_done_value operation
        new_ops.pop();

        // Expand cont.get_done_value and wrap result in Done Step
        // Keep as anyref to avoid unbox/rebox cycle (we'll wrap in Step which needs anyref)
        let get_done_value = cont::GetDoneValue::from_operation(db, last_op).unwrap();
        let location = last_op.location(db);

        let (expansion_ops, boxed_val, _needs_unbox) =
            expand_get_done_value_inline(db, get_done_value, location, true);
        new_ops.extend(expansion_ops);

        // Create Done Step: (tag=0, value=boxed_val, prompt=0, op_idx=0)
        let i32_ty = core::I32::new(db).as_type();
        let done_tag_op = wasm::i32_const(db, location, i32_ty, cont_types::STEP_TAG_DONE);
        let done_tag_val = done_tag_op.as_operation().result(db, 0);
        new_ops.push(done_tag_op.as_operation());

        let zero_prompt_op = wasm::i32_const(db, location, i32_ty, 0);
        let zero_prompt = zero_prompt_op.as_operation().result(db, 0);
        new_ops.push(zero_prompt_op.as_operation());

        let zero_op_idx_op = wasm::i32_const(db, location, i32_ty, 0);
        let zero_op_idx = zero_op_idx_op.as_operation().result(db, 0);
        new_ops.push(zero_op_idx_op.as_operation());

        let step_marker_ty = crate::gc_types::step_marker_type(db);
        let step_op = wasm::struct_new(
            db,
            location,
            IdVec::from(vec![done_tag_val, boxed_val, zero_prompt, zero_op_idx]),
            step_marker_ty,
            crate::gc_types::STEP_IDX,
        );
        let step_val = step_op.as_operation().result(db, 0);
        new_ops.push(step_op.as_operation());

        tracing::debug!(
            "handle_last_block_implicit_yield: wrapped cont.get_done_value result in Done Step"
        );

        let new_yield = wasm::r#yield(db, location, step_val);
        new_ops.push(new_yield.as_operation());
    } else if !is_yield && !last_op.results(db).is_empty() {
        // Last operation has a result and isn't a yield
        // Wrap it in Done Step and yield
        let result_val = Value::new(db, ValueDef::OpResult(last_op), 0);
        let location = last_op.location(db);

        tracing::debug!(
            "handle_last_block_implicit_yield: wrapping last_op result, is_already_step={}",
            is_value_already_step(db, result_val)
        );

        // Check if already a Step
        if !is_value_already_step(db, result_val) {
            // Get type hint from last_op's result type
            let type_hint = last_op.results(db).first().copied();
            let (step_ops, step_val) =
                create_done_step_ops(db, location, result_val, region, type_hint);
            tracing::debug!(
                "handle_last_block_implicit_yield: created {} step_ops for wrapping",
                step_ops.len()
            );
            new_ops.extend(step_ops);
            let new_yield = wasm::r#yield(db, location, step_val);
            new_ops.push(new_yield.as_operation());
        }
    }
}

/// Processes a wasm.yield operation, wrapping its value in Done Step if needed.
///
/// Returns operations to add (either just the yield with remapped value,
/// or step creation ops + new yield).
fn process_yield_operation<'db>(
    db: &'db dyn salsa::Database,
    yield_val: Value<'db>,
    location: Location<'db>,
    value_mapping: &std::collections::HashMap<Value<'db>, Value<'db>>,
    region: Region<'db>,
) -> Vec<Operation<'db>> {
    // Remap yield value if needed
    let remapped_yield_val = value_mapping.get(&yield_val).copied().unwrap_or(yield_val);

    // Check if already a Step
    if is_value_already_step(db, remapped_yield_val) {
        // Return yield operation with remapped value
        let new_yield = wasm::r#yield(db, location, remapped_yield_val);
        return vec![new_yield.as_operation()];
    }

    // Wrap in Done Step and yield that
    // Infer type from the value's definition
    let type_hint = match remapped_yield_val.def(db) {
        ValueDef::OpResult(def_op) => def_op
            .results(db)
            .get(remapped_yield_val.index(db))
            .copied(),
        ValueDef::BlockArg(block_id) => region
            .blocks(db)
            .iter()
            .find(|b| b.id(db) == block_id)
            .and_then(|block| block.args(db).get(remapped_yield_val.index(db)))
            .map(|arg| arg.ty(db)),
    };

    let (step_ops, step_val) =
        create_done_step_ops(db, location, remapped_yield_val, region, type_hint);
    let mut ops = step_ops;
    let new_yield = wasm::r#yield(db, location, step_val);
    ops.push(new_yield.as_operation());
    ops
}

/// Expands cont.get_done_value inline into WASM operations.
///
/// Returns:
/// - Vec<Operation>: operations to add (ref_cast, struct_get, optional i31 unbox)
/// - Value: the final extracted value
/// - bool: whether the value was unboxed (true) or kept as anyref (false)
///
/// If `keep_as_anyref` is true, skips the unboxing step and returns the value as anyref.
/// This is useful when the caller will immediately re-box the value (e.g., for Step wrapping).
fn expand_get_done_value_inline<'db>(
    db: &'db dyn salsa::Database,
    get_done_value_op: cont::GetDoneValue<'db>,
    location: Location<'db>,
    keep_as_anyref: bool,
) -> (Vec<Operation<'db>>, Value<'db>, bool) {
    let anyref_ty = wasm::Anyref::new(db).as_type();
    let i32_ty = core::I32::new(db).as_type();
    let i31ref_ty = wasm::I31ref::new(db).as_type();

    let mut ops = Vec::new();

    // Get the Step operand
    let step_anyref = get_done_value_op.step(db);

    // Cast anyref to Step structref and extract value
    let step_ty = crate::gc_types::step_marker_type(db);
    let step_cast_op = wasm::ref_cast(db, location, step_anyref, step_ty, step_ty);
    let step_val = step_cast_op.as_operation().result(db, 0);
    ops.push(step_cast_op.as_operation());

    // Extract value from Step struct (field 1)
    const STEP_VALUE_FIELD: u32 = 1;
    let get_value = wasm::struct_get(
        db,
        location,
        step_val,
        anyref_ty,
        crate::gc_types::STEP_IDX,
        STEP_VALUE_FIELD,
    )
    .as_operation();
    let value_anyref = Value::new(db, ValueDef::OpResult(get_value), 0);
    ops.push(get_value);

    // If caller wants anyref, skip unboxing
    if keep_as_anyref {
        return (ops, value_anyref, false);
    }

    // Check if we need to unbox (get result type from original op)
    let result_ty = get_done_value_op
        .as_operation()
        .results(db)
        .first()
        .copied()
        .unwrap_or(anyref_ty);
    let needs_unbox = tribute_rt::is_int(db, result_ty)
        || tribute_rt::is_nat(db, result_ty)
        || core::I32::from_type(db, result_ty).is_some()
        || tribute::is_type_var(db, result_ty);

    let final_value = if needs_unbox {
        // Cast anyref to i31ref
        let ref_cast_i31 = wasm::ref_cast(db, location, value_anyref, i31ref_ty, i31ref_ty);
        let i31_val = ref_cast_i31.as_operation().result(db, 0);
        ops.push(ref_cast_i31.as_operation());

        // Unbox i31ref to i32
        let unbox_op = wasm::i31_get_s(db, location, i31_val, i32_ty);
        let unboxed_val = unbox_op.as_operation().result(db, 0);
        ops.push(unbox_op.as_operation());
        unboxed_val
    } else {
        value_anyref
    };

    (ops, final_value, needs_unbox)
}

/// Wrap wasm.yield values in Done Step within a region.
/// This is used for done_body in handler dispatch so that both branches of
/// wasm.if produce Step values.
///
/// Note: This only processes top-level yields, not yields inside nested
/// control flow (wasm.block, wasm.if, etc.). Nested control flow operations
/// that produce results must be handled elsewhere.
fn wrap_yields_in_done_step<'db>(db: &'db dyn salsa::Database, region: Region<'db>) -> Region<'db> {
    use std::collections::HashMap;

    let mut new_blocks: Vec<Block<'db>> = Vec::new();

    let blocks = region.blocks(db);
    for (block_idx, block) in blocks.iter().enumerate() {
        let mut new_ops: Vec<Operation<'db>> = Vec::new();
        let is_last_block = block_idx == blocks.len() - 1;

        // Track SSA value remapping for cont.get_done_value expansion
        let mut value_mapping: HashMap<Value<'db>, Value<'db>> = HashMap::new();

        for op in block.operations(db).iter() {
            // Note: We DON'T recursively process nested regions here to avoid
            // breaking SSA value references. Nested control flow must be handled
            // differently (e.g., the inner operation already produces Step).

            // Check for cont.get_done_value (not yet expanded) anywhere in the block
            let is_get_done_value =
                op.dialect(db) == cont::DIALECT_NAME() && op.name(db) == cont::GET_DONE_VALUE();

            if is_get_done_value {
                // Expand cont.get_done_value inline and track SSA value remapping
                let get_done_value = cont::GetDoneValue::from_operation(db, *op).unwrap();
                let location = op.location(db);

                let (expansion_ops, final_value, _needs_unbox) =
                    expand_get_done_value_inline(db, get_done_value, location, false);

                new_ops.extend(expansion_ops);

                // Map the original cont.get_done_value result to the final expanded value
                if !op.results(db).is_empty() {
                    let original_val = Value::new(db, ValueDef::OpResult(*op), 0);
                    value_mapping.insert(original_val, final_value);
                }

                continue;
            }

            // Check for wasm.yield
            if let Ok(yield_op) = wasm::Yield::from_operation(db, *op) {
                let yield_val = yield_op.value(db);
                let location = op.location(db);

                let yield_ops =
                    process_yield_operation(db, yield_val, location, &value_mapping, region);
                new_ops.extend(yield_ops);
                continue;
            }

            // Remap operands for all other operations
            let operands = op.operands(db);
            let has_remapped = operands.iter().any(|v| value_mapping.contains_key(v));

            if has_remapped {
                let remapped_operands: IdVec<Value<'db>> = operands
                    .iter()
                    .map(|v| value_mapping.get(v).copied().unwrap_or(*v))
                    .collect();
                let new_op = op.modify(db).operands(remapped_operands).build();
                new_ops.push(new_op);
            } else {
                new_ops.push(*op);
            }
        }

        // If this is the last block and doesn't end with a yield,
        // wrap the last operation's result in Done Step and yield it.
        if is_last_block {
            handle_last_block_implicit_yield(db, &mut new_ops, region);
        }

        let new_block = Block::new(
            db,
            block.id(db),
            block.location(db),
            block.args(db).clone(),
            IdVec::from(new_ops),
        );
        new_blocks.push(new_block);
    }

    Region::new(db, region.location(db), IdVec::from(new_blocks))
}

/// Build a nested if-chain for multi-arm dispatch based on op_idx.
///
/// Generates:
/// ```text
/// global.get $yield_op_idx
/// if (op_idx == arm0_idx) {
///     arm0_body
/// } else {
///     if (op_idx == arm1_idx) {
///         arm1_body
///     } else {
///         ... (last arm as fallback)
///     }
/// }
/// ```
fn build_multi_arm_dispatch<'db>(
    db: &'db dyn salsa::Database,
    location: Location<'db>,
    i32_ty: Type<'db>,
    result_ty: Type<'db>,
    suspend_arms: &[(u64, Region<'db>)],
) -> Region<'db> {
    if suspend_arms.is_empty() {
        return Region::new(db, location, IdVec::new());
    }

    // Start with the last arm as the base (fallback)
    let mut current_else = suspend_arms.last().map(|(_, body)| *body).unwrap();

    // Build from back to front, creating nested if-else
    for (op_idx, body) in suspend_arms.iter().rev().skip(1) {
        // global.get $yield_op_idx
        let get_op_idx = wasm::global_get(db, location, i32_ty, YIELD_OP_IDX);
        let op_idx_val = get_op_idx.as_operation().result(db, 0);

        // i32.const <op_idx>
        let expected_idx = wasm::i32_const(db, location, i32_ty, *op_idx as i32);
        let expected_idx_val = expected_idx.as_operation().result(db, 0);

        // i32.eq
        let cmp = wasm::i32_eq(db, location, op_idx_val, expected_idx_val, i32_ty);
        let cmp_val = cmp.as_operation().result(db, 0);

        // Build the condition block with the comparison operations
        let _cond_block = Block::new(
            db,
            BlockId::fresh(),
            location,
            IdVec::new(),
            IdVec::from(vec![
                get_op_idx.as_operation(),
                expected_idx.as_operation(),
                cmp.as_operation(),
            ]),
        );
        // Note: cond_block is not used directly; we inline the operations below

        // Build if with the comparison result
        // Note: We need to wrap this in a region that first evaluates the condition
        // then branches on the result
        let if_op = wasm::r#if(db, location, cmp_val, result_ty, *body, current_else);

        // The new else block becomes this if wrapped in the condition
        let new_block = Block::new(
            db,
            BlockId::fresh(),
            location,
            IdVec::new(),
            IdVec::from(vec![
                get_op_idx.as_operation(),
                expected_idx.as_operation(),
                cmp.as_operation(),
                if_op.as_operation(),
            ]),
        );
        current_else = Region::new(db, location, IdVec::from(vec![new_block]));
    }

    current_else
}

/// Pattern for `cont.resume` -> invoke continuation
///
/// Continuation struct layout:
/// - Field 0: resume_fn (funcref)
/// - Field 1: state (anyref)
/// - Field 2: tag (i32)
/// - Field 3: shift_value (anyref)
///
/// Resume wrapper struct layout:
/// - Field 0: state (anyref)
/// - Field 1: resume_value (anyref)
///
/// Resume operation:
/// 1. Reset $yield_state = 0
/// 2. Extract resume_fn from continuation
/// 3. Extract state from continuation
/// 4. Create wrapper struct with (state, boxed_value)
/// 5. Call resume_fn(wrapper) via call_indirect
struct ResumePattern;

/// Continuation struct field indices
const CONT_FIELD_RESUME_FN: u32 = 0;
const CONT_FIELD_STATE: u32 = 1;
#[allow(dead_code)]
const CONT_FIELD_TAG: u32 = 2;
#[allow(dead_code)]
const CONT_FIELD_SHIFT_VALUE: u32 = 3;

/// Resume wrapper struct field indices
#[allow(dead_code)]
const WRAPPER_FIELD_STATE: u32 = 0;
#[allow(dead_code)]
const WRAPPER_FIELD_RESUME_VALUE: u32 = 1;

impl RewritePattern for ResumePattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
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
        let structref_ty = wasm::Structref::new(db).as_type();

        let mut ops = Vec::new();

        // 1. Reset $yield_state = 0 (not yielding anymore)
        let const_0 = wasm::i32_const(db, location, i32_ty, 0);
        let const_0_val = const_0.as_operation().result(db, 0);
        ops.push(const_0.as_operation());
        ops.push(wasm::global_set(db, location, const_0_val, YIELD_STATE_IDX).as_operation());

        // 2. Cast continuation to structref if it's anyref
        // This is needed when the continuation was captured by a closure (closures store values as anyref)
        const CONT_FIELD_COUNT: u64 = 4;

        let cast_cont_op = wasm::ref_cast(db, location, continuation, structref_ty, structref_ty)
            .as_operation()
            .modify(db)
            .attr("field_count", Attribute::IntBits(CONT_FIELD_COUNT))
            .build();
        let cont_structref = Value::new(db, ValueDef::OpResult(cast_cont_op), 0);
        ops.push(cast_cont_op);

        // 3. Extract resume_fn from continuation (field 0)
        // Continuation layout: (resume_fn, state, tag, shift_value)
        let get_resume_fn = wasm::struct_get(
            db,
            location,
            cont_structref,
            funcref_ty,
            0, // type_idx - Placeholder, resolved at emit time
            CONT_FIELD_RESUME_FN,
        )
        .as_operation()
        .modify(db)
        .attr("type", Attribute::Type(structref_ty))
        .attr("field_count", Attribute::IntBits(CONT_FIELD_COUNT))
        .build();
        let resume_fn_val = Value::new(db, ValueDef::OpResult(get_resume_fn), 0);
        ops.push(get_resume_fn);

        // 4. Extract state from continuation (field 1)
        let get_state = wasm::struct_get(
            db,
            location,
            cont_structref,
            anyref_ty,
            0, // type_idx - Placeholder, resolved at emit time
            CONT_FIELD_STATE,
        )
        .as_operation()
        .modify(db)
        .attr("type", Attribute::Type(structref_ty))
        .attr("field_count", Attribute::IntBits(CONT_FIELD_COUNT))
        .build();
        let state_val = Value::new(db, ValueDef::OpResult(get_state), 0);
        ops.push(get_state);

        // 5. Box the value if it's a primitive type
        // Resume wrapper expects (state: anyref, resume_value: anyref), so primitives need boxing
        // Use box_value_to_anyref for consistent anyref type in struct fields
        let value_ty = match value.def(db) {
            ValueDef::OpResult(def_op) => def_op.results(db).get(value.index(db)).copied(),
            ValueDef::BlockArg(_) => None,
        };
        let boxed_value = if let Some(ty) = value_ty {
            box_value_to_anyref(db, location, value, ty, &mut ops)
        } else {
            value
        };

        // 6. Create wrapper struct with (state, boxed_value)
        // Resume wrapper layout: (state: anyref, resume_value: anyref)
        let wrapper_ty = cont_types::resume_wrapper_type(db);
        let wrapper_struct = wasm::struct_new(
            db,
            location,
            IdVec::from(vec![state_val, boxed_value]),
            wrapper_ty,
            0, // type_idx - Placeholder, resolved at emit time
        );
        let wrapper_val = Value::new(db, ValueDef::OpResult(wrapper_struct.as_operation()), 0);
        ops.push(wrapper_struct.as_operation());

        // 7. Call resume_fn(wrapper) via call_indirect
        // Resume function signature: (wrapper: anyref) -> Step
        // Resume functions always return Step for trampoline dispatch
        let result_ty = cont_types::step_type(db);
        let call_indirect = wasm::call_indirect(
            db,
            location,
            IdVec::from(vec![wrapper_val, resume_fn_val]),
            IdVec::from(vec![result_ty]),
            0, // type_idx - Placeholder - resolved at emit
            0, // table - Default table
        );
        ops.push(call_indirect.as_operation());

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
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(drop_op) = cont::Drop::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let location = op.location(db);
        let continuation = drop_op.continuation(db);

        // In WasmGC, continuations are GC-managed, so "drop" just
        // removes the reference. We use wasm.drop to pop the value.
        let new_op = wasm::drop(db, location, continuation);

        RewriteResult::Replace(new_op.as_operation())
    }
}

/// Pattern for `cont.get_continuation` -> load continuation from yield global
///
/// This operation is used in handler arm bodies to get the captured continuation.
/// It expands to: global.get $yield_cont -> ref_cast structref
struct GetContinuationPattern;

impl RewritePattern for GetContinuationPattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(_get_cont) = cont::GetContinuation::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let location = op.location(db);
        let anyref_ty = wasm::Anyref::new(db).as_type();
        let structref_ty = wasm::Structref::new(db).as_type();

        let mut ops = Vec::new();

        // Load continuation from $yield_cont global
        let get_cont = wasm::global_get(db, location, anyref_ty, YIELD_CONT_IDX);
        let cont_anyref = get_cont.as_operation().result(db, 0);
        ops.push(get_cont.as_operation());

        // Cast anyref to structref for proper typing
        // Include field_count attribute so emit.rs can find the correct struct type
        const CONT_FIELD_COUNT: u64 = 4;
        let cont_cast_op = wasm::ref_cast(db, location, cont_anyref, structref_ty, structref_ty)
            .as_operation()
            .modify(db)
            .attr("field_count", Attribute::IntBits(CONT_FIELD_COUNT))
            .build();
        ops.push(cont_cast_op);

        RewriteResult::Expand(ops)
    }
}

/// Pattern for `cont.get_shift_value` -> extract shift_value from continuation struct
///
/// This operation is used in handler arm bodies to get the value passed to the effect.
/// For example, in `State::set!(n)`, this extracts the value `n`.
///
/// It expands to:
/// 1. global.get $yield_cont
/// 2. ref_cast structref
/// 3. struct.get field 3 (shift_value) -> anyref
///
/// The result is kept as anyref. Unboxing to concrete types (e.g., i32 for Int)
/// happens at the use site via TypeConverter.materialize() when the expected type
/// is known.
struct GetShiftValuePattern;

impl RewritePattern for GetShiftValuePattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(_get_value) = cont::GetShiftValue::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let location = op.location(db);
        let anyref_ty = wasm::Anyref::new(db).as_type();
        let structref_ty = wasm::Structref::new(db).as_type();

        let mut ops = Vec::new();

        // Load continuation from $yield_cont global
        let get_cont = wasm::global_get(db, location, anyref_ty, YIELD_CONT_IDX);
        let cont_anyref = get_cont.as_operation().result(db, 0);
        ops.push(get_cont.as_operation());

        // Cast anyref to structref
        const CONT_FIELD_COUNT: u64 = 4;
        let cont_cast_op = wasm::ref_cast(db, location, cont_anyref, structref_ty, structref_ty)
            .as_operation()
            .modify(db)
            .attr("field_count", Attribute::IntBits(CONT_FIELD_COUNT))
            .build();
        let cont_val = Value::new(db, ValueDef::OpResult(cont_cast_op), 0);
        ops.push(cont_cast_op);

        // Extract shift_value from continuation struct (field 3)
        // Continuation layout: (resume_fn, state, tag, shift_value)
        // Result type is anyref - unboxing happens at use site when concrete type is known
        let get_value = wasm::struct_get(
            db,
            location,
            cont_val,
            anyref_ty,
            0, // type_idx - Placeholder, resolved at emit time
            CONT_FIELD_SHIFT_VALUE,
        )
        .as_operation()
        .modify(db)
        .attr("type", Attribute::Type(structref_ty))
        .attr("field_count", Attribute::IntBits(CONT_FIELD_COUNT))
        .build();
        ops.push(get_value);

        RewriteResult::Expand(ops)
    }
}

/// Pattern for `cont.get_done_value` -> extract value from Done Step struct
///
/// This operation is used in handler "done" arm bodies to extract the result
/// value from a Step struct returned by push_prompt.
///
/// It expands to:
/// 1. ref_cast structref (cast Step anyref to structref)
/// 2. struct.get field 1 (value) -> anyref
/// 3. If result type is primitive (Int/Nat/i32):
///    a. ref_cast anyref -> i31ref
#[allow(dead_code)]
///    b. i31_get_s i31ref -> i32
///
/// For primitive types, the final result is i32. For reference types, it's anyref.
struct GetDoneValuePattern;

impl RewritePattern for GetDoneValuePattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        // Debug: Check if we're seeing cont.get_done_value operations
        if op.dialect(db) == cont::DIALECT_NAME() && op.name(db) == cont::GET_DONE_VALUE() {
            tracing::debug!(
                "GetDoneValuePattern: found cont.get_done_value, results={:?}",
                op.results(db)
            );
        }

        let Ok(get_done_value) = cont::GetDoneValue::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        tracing::debug!("GetDoneValuePattern: matched, processing...");

        let location = op.location(db);
        let anyref_ty = wasm::Anyref::new(db).as_type();
        let i32_ty = core::I32::new(db).as_type();
        let i31ref_ty = wasm::I31ref::new(db).as_type();

        // Get the expected result type from the operation
        let result_ty = op.results(db).first().copied().unwrap_or(anyref_ty);

        tracing::debug!(
            "GetDoneValuePattern: result_ty = {}.{}, attrs={:?}",
            result_ty.dialect(db),
            result_ty.name(db),
            result_ty.attrs(db)
        );

        // Check if we need to unbox to i32
        let is_int = tribute_rt::is_int(db, result_ty);
        let is_nat = tribute_rt::is_nat(db, result_ty);
        let is_i32 = core::I32::from_type(db, result_ty).is_some();
        // IMPORTANT: tribute.type_var appears when type substitution hasn't been applied.
        // In the context of handler dispatch, we know the function returns Int,
        // but the IR still has type_var. Check for it and unbox.
        let is_type_var = tribute::is_type_var(db, result_ty);

        tracing::debug!(
            "GetDoneValuePattern: is_int={}, is_nat={}, is_i32={}, is_type_var={}",
            is_int,
            is_nat,
            is_i32,
            is_type_var
        );

        // For type_var, we conservatively unbox since we're in a handler context
        // where primitive return types are common.
        let needs_unbox = is_int || is_nat || is_i32 || is_type_var;

        tracing::debug!("GetDoneValuePattern: needs_unbox = {}", needs_unbox);

        let mut ops = Vec::new();

        // Get the Step operand (the push_prompt result)
        let step_anyref = get_done_value.step(db);

        // Cast anyref to Step structref for struct.get
        // Step layout: (tag: i32, value: anyref, prompt: i32, op_idx: i32)
        // Use step_marker_type so emit.rs resolves to builtin STEP_IDX (3)
        let step_ty = crate::gc_types::step_marker_type(db);
        let step_cast_op = wasm::ref_cast(db, location, step_anyref, step_ty, step_ty);
        let step_val = step_cast_op.as_operation().result(db, 0);
        ops.push(step_cast_op.as_operation());

        // Extract value from Step struct (field 1)
        const STEP_VALUE_FIELD: u32 = 1;
        let get_value = wasm::struct_get(
            db,
            location,
            step_val,
            anyref_ty,
            crate::gc_types::STEP_IDX, // type_idx for Step struct
            STEP_VALUE_FIELD,
        )
        .as_operation();
        let value_anyref = Value::new(db, ValueDef::OpResult(get_value), 0);
        ops.push(get_value);

        // If primitive type, add unboxing: anyref -> i31ref -> i32
        if needs_unbox {
            // Cast anyref to i31ref
            let ref_cast_i31 = wasm::ref_cast(db, location, value_anyref, i31ref_ty, i31ref_ty);
            let i31_val = ref_cast_i31.as_operation().result(db, 0);
            ops.push(ref_cast_i31.as_operation());

            // Unbox i31ref to i32
            let unbox_op = wasm::i31_get_s(db, location, i31_val, i32_ty);
            ops.push(unbox_op.as_operation());
        }

        RewriteResult::Expand(ops)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use trunk_ir::DialectType;
    use trunk_ir::dialect::{arith, core, func};
    use trunk_ir::{
        Attribute, Block, BlockId, IdVec, Location, PathId, Region, Span, Value, ValueDef, idvec,
    };

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

        // Create empty handlers region
        let handlers_block = Block::new(db, BlockId::fresh(), location, IdVec::new(), idvec![]);
        let handlers_region = Region::new(db, location, idvec![handlers_block]);

        // Use typed helper
        let push_prompt = cont::push_prompt(db, location, i32_ty, 42, body_region, handlers_region)
            .as_operation();

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
        let i32_ty = core::I32::new(db).as_type();

        // Create empty handler region for shift
        let handler_block = Block::new(db, BlockId::fresh(), location, IdVec::new(), idvec![]);
        let handler_region = Region::new(db, location, idvec![handler_block]);

        // cont.shift now has a result (the value passed when continuation is resumed)
        // For test purposes, use dummy ability_ref and op_name
        let test_ability_ref = *core::AbilityRefType::simple(db, Symbol::new("TestAbility"));
        let test_op_name = Symbol::new("test_op");

        let shift = cont::shift(
            db,
            location,
            std::iter::empty::<Value>(),
            i32_ty,
            42,
            test_ability_ref,
            test_op_name,
            handler_region,
        )
        .as_operation();

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

        // This test has a bare shift with no continuation ops and no live locals,
        // so no resume function is generated. The shift expansion is:
        // 0. wasm.struct_new (empty state struct)
        // 1. wasm.ref_null (null funcref for resume_fn since no resume function)
        // 2. wasm.i32_const (tag value for continuation struct)
        // 3. wasm.ref_null (shift_value = null since no value operand)
        // 4. wasm.struct_new (continuation struct with 4 fields)
        // 5. wasm.i32_const (yield_state = 1)
        // 6. wasm.global_set (yield_state)
        // 7. wasm.i32_const (yield_tag)
        // 8. wasm.global_set (yield_tag)
        // 9. wasm.global_set (yield_cont)
        // 10. wasm.i32_const (yield_op_idx = 0)
        // 11. wasm.global_set (yield_op_idx)
        // 12. wasm.i32_const (Step tag = 1, Shift)
        // 13. wasm.i32_const (prompt = 0)
        // 14. wasm.i32_const (op_idx = 0)
        // 15. wasm.struct_new (Step struct with 4 fields: tag, value, prompt, op_idx)
        // 16. wasm.return

        // First op: struct_new (empty state struct)
        assert_eq!(op_names.first(), Some(&"wasm.struct_new".to_string()));
        // Last op: return (to unwind stack)
        assert_eq!(op_names.last(), Some(&"wasm.return".to_string()));
        // Should NOT contain ref_func (no resume function when no continuation)
        assert!(!op_names.contains(&"wasm.ref_func".to_string()));
        // Should contain ref_null for resume_fn and shift_value (not return value anymore)
        let ref_null_count = op_names.iter().filter(|n| *n == "wasm.ref_null").count();
        assert!(
            ref_null_count >= 2,
            "expected at least 2 ref_null (resume_fn + shift_value), got {}",
            ref_null_count
        );
        // Should contain struct_new for state, continuation, and Step
        let struct_new_count = op_names.iter().filter(|n| *n == "wasm.struct_new").count();
        assert!(
            struct_new_count >= 3,
            "expected at least 3 struct_new (state + continuation + Step), got {}",
            struct_new_count
        );
        // Total: 17 operations (no resume function, shift expansion with Step having 4 fields)
        assert_eq!(op_names.len(), 17);
    }

    #[salsa::tracked]
    fn make_module_with_resume(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();

        // Create dummy continuation and value operands using arith.const
        let cont_op = arith::r#const(db, location, i32_ty, Attribute::IntBits(0));
        let cont_val = cont_op.result(db);

        let val_op = arith::r#const(db, location, i32_ty, Attribute::IntBits(42));
        let value = val_op.result(db);

        let resume = cont::resume(db, location, cont_val, value, i32_ty).as_operation();

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![cont_op.as_operation(), val_op.as_operation(), resume],
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
        // 3. wasm.ref_cast (cast continuation to structref for struct_get)
        // 4. wasm.struct_get (extract resume_fn from continuation)
        // 5. wasm.struct_get (extract state from continuation)
        // 6. wasm.ref_i31 (box the i32 value to anyref for wrapper)
        // 7. wasm.struct_new (create wrapper struct with state + boxed_value)
        // 8. wasm.call_indirect (invoke resume_fn with wrapper)
        // First op: i32_const for resetting yield_state
        assert_eq!(op_names.first(), Some(&"wasm.i32_const".to_string()));
        // Last op: call_indirect to invoke resume function
        assert_eq!(op_names.last(), Some(&"wasm.call_indirect".to_string()));
        // Should contain ref_cast for continuation (added for closure capture case)
        assert!(op_names.contains(&"wasm.ref_cast".to_string()));
        // Should contain struct_get operations
        assert!(op_names.contains(&"wasm.struct_get".to_string()));
        // Should contain ref_i31 for boxing i32 to anyref
        assert!(op_names.contains(&"wasm.ref_i31".to_string()));
        // Should contain struct_new for wrapper
        assert!(op_names.contains(&"wasm.struct_new".to_string()));
        // Total: 8 operations (added ref_cast for continuation casting)
        assert_eq!(op_names.len(), 8);
    }

    #[salsa::tracked]
    fn make_module_with_drop(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();

        // Create dummy continuation operand using arith.const
        let cont_op = arith::r#const(db, location, i32_ty, Attribute::IntBits(0));
        let cont_val = cont_op.result(db);

        let drop_op = cont::drop(db, location, cont_val).as_operation();

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![cont_op.as_operation(), drop_op],
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

        // cont.shift now has a result (the value passed when continuation is resumed)
        // For test purposes, use dummy ability_ref and op_name
        let test_ability_ref = *core::AbilityRefType::simple(db, Symbol::new("TestAbility"));
        let test_op_name = Symbol::new("test_op");

        let shift = cont::shift(
            db,
            shift_loc,
            std::iter::empty::<Value>(),
            i32_ty,
            99,
            test_ability_ref,
            test_op_name,
            handler_region,
        )
        .as_operation();

        // Create function body with shift
        let func_body_block = Block::new(db, BlockId::fresh(), location, idvec![], idvec![shift]);
        let func_body = Region::new(db, location, idvec![func_body_block]);

        // Create func.func operation
        let func_ty = core::Func::new(db, idvec![], i32_ty).as_type();
        let func_op =
            func::func(db, location, Symbol::new("my_func"), func_ty, func_body).as_operation();

        let block = Block::new(db, BlockId::fresh(), location, idvec![], idvec![func_op]);
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa::tracked]
    fn analyze_shift_points_test(db: &dyn salsa::Database) -> (usize, Option<u64>, Option<Symbol>) {
        let module = make_module_with_shift_in_function(db);
        let analysis = analyze_continuations(db, module);
        let shift_points = analysis.shift_points(db);

        let count = shift_points.len();
        let first_info = shift_points.values().next();
        let tag = first_info.map(|info| info.tag);
        let func_name = first_info
            .and_then(|info| info.containing_func.as_ref())
            .map(|name| name.last_segment());

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
        assert_eq!(func_name, Some(Symbol::new("my_func")));
    }

    // === Tests for live local capture ===

    fn shift_location_for_live(db: &dyn salsa::Database) -> Location<'_> {
        let path = PathId::new(db, "file:///test.trb".to_owned());
        Location::new(path, Span::new(200, 220)) // Distinct location
    }

    #[salsa::tracked]
    fn make_module_with_live_locals(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let shift_loc = shift_location_for_live(db);
        let i32_ty = core::I32::new(db).as_type();

        // Create a function body with:
        // 1. A value defined before shift (local_val)
        // 2. A shift point
        // 3. An operation that uses local_val after shift

        // Define a value before shift
        let local_op = wasm::i32_const(db, location, i32_ty, 42).as_operation();
        let local_val = Value::new(db, ValueDef::OpResult(local_op), 0);

        // Create shift
        let handler_block = Block::new(db, BlockId::fresh(), shift_loc, IdVec::new(), idvec![]);
        let handler_region = Region::new(db, shift_loc, idvec![handler_block]);

        // cont.shift now has a result (the value passed when continuation is resumed)
        // For test purposes, use dummy ability_ref and op_name
        let test_ability_ref = *core::AbilityRefType::simple(db, Symbol::new("TestAbility"));
        let test_op_name = Symbol::new("test_op");

        let shift = cont::shift(
            db,
            shift_loc,
            std::iter::empty::<Value>(),
            i32_ty,
            1,
            test_ability_ref,
            test_op_name,
            handler_region,
        )
        .as_operation();

        // Use local_val after shift
        let use_after = wasm::drop(db, location, local_val).as_operation();

        // Return
        let return_op = wasm::r#return(db, location, None).as_operation();

        // Create function body
        let func_body_block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![local_op, shift, use_after, return_op],
        );
        let func_body = Region::new(db, location, idvec![func_body_block]);

        // Create func.func operation
        let func_ty = core::Func::new(db, idvec![], core::Nil::new(db).as_type()).as_type();
        let func_op = func::func(
            db,
            location,
            Symbol::new("fn_with_live"),
            func_ty,
            func_body,
        )
        .as_operation();

        let block = Block::new(db, BlockId::fresh(), location, idvec![], idvec![func_op]);
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa::tracked]
    fn analyze_live_locals_test(db: &dyn salsa::Database) -> (usize, usize) {
        let module = make_module_with_live_locals(db);
        let analysis = analyze_continuations(db, module);
        let shift_points = analysis.shift_points(db);

        let count = shift_points.len();
        let live_count = shift_points
            .values()
            .next()
            .map(|info| info.live_locals.len())
            .unwrap_or(0);

        (count, live_count)
    }

    #[salsa_test]
    fn test_live_local_analysis(db: &salsa::DatabaseImpl) {
        let (count, live_count) = analyze_live_locals_test(db);

        // Should find one shift point
        assert_eq!(count, 1);
        // Should have one live local (local_val is defined before and used after shift)
        assert_eq!(live_count, 1, "Expected 1 live local, got {}", live_count);
    }

    #[salsa::tracked]
    fn check_resume_function_has_struct_get(db: &dyn salsa::Database) -> bool {
        let module = make_module_with_live_locals(db);
        let lowered = lower(db, module);
        let body = lowered.body(db);
        let ops = body.blocks(db)[0].operations(db);

        // The first operation should be the resume function (wasm.func)
        if let Some(resume_fn) = ops.first()
            && resume_fn.full_name(db) == "wasm.func"
        {
            // Check if the resume function body contains struct_get
            let fn_body = resume_fn.regions(db);
            if let Some(region) = fn_body.first()
                && let Some(block) = region.blocks(db).first()
            {
                return block
                    .operations(db)
                    .iter()
                    .any(|op| op.full_name(db) == "wasm.struct_get");
            }
        }
        false
    }

    #[salsa_test]
    fn test_resume_function_extracts_state(db: &salsa::DatabaseImpl) {
        let has_struct_get = check_resume_function_has_struct_get(db);
        assert!(
            has_struct_get,
            "Resume function should contain struct_get to extract live locals from state"
        );
    }

    #[salsa::tracked]
    fn check_continuation_ops_captured(db: &dyn salsa::Database) -> usize {
        let module = make_module_with_live_locals(db);
        let analysis = analyze_continuations(db, module);
        let shift_points = analysis.shift_points(db);

        shift_points
            .values()
            .next()
            .map(|info| info.continuation_ops.len())
            .unwrap_or(0)
    }

    #[salsa_test]
    fn test_continuation_ops_captured(db: &salsa::DatabaseImpl) {
        let cont_op_count = check_continuation_ops_captured(db);

        // Should have captured 2 continuation ops: wasm.drop and wasm.return
        assert_eq!(
            cont_op_count, 2,
            "Expected 2 continuation ops (drop + return), got {}",
            cont_op_count
        );
    }
}
