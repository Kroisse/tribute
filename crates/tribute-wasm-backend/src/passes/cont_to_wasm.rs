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

use tribute_ir::ModulePathExt;
use trunk_ir::DialectType;
use trunk_ir::dialect::cont;
use trunk_ir::dialect::core::{self, Module};
use trunk_ir::dialect::wasm;
use trunk_ir::rewrite::{PatternApplicator, RewritePattern, RewriteResult};
use trunk_ir::{
    Attribute, Block, BlockId, DialectOp, IdVec, Location, Operation, Region, Symbol, Type, Value,
    ValueDef,
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

    /// Create the resume function type.
    ///
    /// Signature: (state: anyref, value: anyref) -> anyref
    pub fn resume_fn_type(db: &dyn salsa::Database) -> Type<'_> {
        let anyref = wasm::Anyref::new(db).as_type();
        core::Func::new(db, IdVec::from(vec![anyref, anyref]), anyref).as_type()
    }
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
        let state_ty = cont_types::state_type(db, &info.live_locals);

        for (field_idx, live_local) in info.live_locals.iter().enumerate() {
            // Generate struct_get to extract this field
            let struct_get = Operation::of_name(db, location, "wasm.struct_get")
                .operand(state_param)
                .attr("type", Attribute::Type(state_ty))
                .attr("field_idx", Attribute::IntBits(field_idx as u64))
                .results(IdVec::from(vec![live_local.ty]))
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
        let mut result = Vec::with_capacity(ops.len());

        for &op in ops {
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
            // Get the type of this value
            let ty = self.get_value_type(*value);
            if let Some(ty) = ty {
                live_locals.push(LiveLocal {
                    value: *value,
                    name: None, // TODO: Could extract from tribute.var if available
                    ty,
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

    // Apply pattern transformations for non-shift operations
    let module = PatternApplicator::new()
        .add_pattern(PushPromptPattern)
        .add_pattern(HandlerDispatchPattern)
        .add_pattern(ResumePattern)
        .add_pattern(DropPattern)
        .apply(db, module)
        .module;

    // Transform shift operations with inline resume function generation.
    // This is done in a loop because expanding shifts inside resume functions
    // creates new shifts that need to be processed.
    transform_shifts(db, module, &analysis, &resume_fn_names)
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
            let continuation_ops: Vec<Operation<'db>> =
                ops.iter().skip(op_idx + 1).copied().collect();

            // Get the shift's result value (used by continuation ops)
            let shift_result = op_with_processed_regions
                .results(db)
                .first()
                .map(|_| op_with_processed_regions.result(db, 0));

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

    // Collect values defined before the shift along with their types
    let mut defined_before: HashMap<Value<'db>, Type<'db>> = HashMap::new();

    // Block arguments are defined before any operation
    for (i, &arg) in block_args.iter().enumerate() {
        if let Some(block_arg) = block.args(db).get(i) {
            defined_before.insert(arg, block_arg.ty(db));
        }
    }

    // Results of operations before the shift
    let ops: Vec<Operation<'db>> = block.operations(db).iter().copied().collect();
    for op in ops.iter().take(shift_op_idx) {
        let result_types = op.results(db);
        for i in 0..result_types.len() {
            let ty = result_types[i];
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

/// Generate a resume function inline during shift transformation.
///
/// This creates a resume function with the current live locals and continuation ops,
/// avoiding stale SSA references that would occur if generated upfront.
///
/// The `shift_result` is the result value of the cont.shift operation. If the
/// continuation ops use this value, we map it to the `value` parameter of the
/// resume function (the value passed when the continuation is resumed).
fn generate_inline_resume_function<'db>(
    db: &'db dyn salsa::Database,
    location: Location<'db>,
    name: Symbol,
    live_locals: &[LiveLocal<'db>],
    continuation_ops: &[Operation<'db>],
    shift_result: Option<Value<'db>>,
) -> Operation<'db> {
    use std::collections::HashMap;
    use trunk_ir::{BlockArg, BlockId};

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

    // Map the shift's result to the value parameter
    // This is the value passed when the continuation is resumed
    if let Some(shift_result_value) = shift_result {
        value_mapping.insert(shift_result_value, value_param);
    }

    // Extract live locals from state struct
    let state_ty = cont_types::state_type(db, live_locals);

    for (field_idx, live_local) in live_locals.iter().enumerate() {
        // Generate struct_get to extract this field
        let struct_get = Operation::of_name(db, location, "wasm.struct_get")
            .operand(state_param)
            .attr("type", Attribute::Type(state_ty))
            .attr("field_idx", Attribute::IntBits(field_idx as u64))
            .results(IdVec::from(vec![live_local.ty]))
            .build();

        let extracted_value = Value::new(db, ValueDef::OpResult(struct_get), 0);
        ops.push(struct_get);

        // Map the original value to the extracted value
        value_mapping.insert(live_local.value, extracted_value);
    }

    // Remap continuation operations and add them to the body
    if !continuation_ops.is_empty() {
        let remapped_ops = resume_gen::remap_operations(db, continuation_ops, &mut value_mapping);
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

    // Get the operation index (for multi-op ability dispatch)
    //
    // For now, we compute a deterministic index from the operation name.
    // This allows both shift sites and handler dispatch to agree on indices.
    // The index is computed as a hash of the operation name modulo a reasonable max.
    //
    // TODO: In full implementation, indices would be looked up from ability definitions.
    let op_idx_value = op
        .attributes(db)
        .get(&Symbol::new("op_name"))
        .and_then(|attr| {
            if let Attribute::Symbol(name) = attr {
                // Use a simple hash of the operation name as the index
                // This ensures consistent indices across shift and handler dispatch
                Some(name.with_str(compute_op_idx_hash))
            } else {
                None
            }
        })
        .unwrap_or(0);

    let mut ops = Vec::new();

    // === 1. Build State Struct (capturing live locals) ===
    let state_ty = cont_types::state_type(db, live_locals);

    // Collect the values to capture
    let live_values: IdVec<Value<'db>> = live_locals.iter().map(|ll| ll.value).collect();

    let state_struct = Operation::of_name(db, location, "wasm.struct_new")
        .operands(live_values)
        .attr("type", Attribute::Type(state_ty))
        .results(IdVec::from(vec![state_ty]))
        .build();
    let state_val = Value::new(db, ValueDef::OpResult(state_struct), 0);
    ops.push(state_struct);

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
        let null_resume_fn = wasm::ref_null(
            db,
            location,
            funcref_ty,
            Attribute::Symbol(Symbol::new("func")),
        );
        let val = null_resume_fn.as_operation().result(db, 0);
        ops.push(null_resume_fn.as_operation());
        val
    };

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

    // === 3. Set Yield Globals ===

    // Set $yield_state = 1 (yielding)
    let const_1 = wasm::i32_const(db, location, i32_ty, Attribute::IntBits(1));
    let const_1_val = const_1.as_operation().result(db, 0);
    ops.push(const_1.as_operation());
    ops.push(wasm::global_set(db, location, const_1_val, YIELD_STATE_IDX).as_operation());

    // Set $yield_tag = tag
    let tag_const2 = wasm::i32_const(db, location, i32_ty, Attribute::IntBits(tag_value));
    let tag_const2_val = tag_const2.as_operation().result(db, 0);
    ops.push(tag_const2.as_operation());
    ops.push(wasm::global_set(db, location, tag_const2_val, YIELD_TAG_IDX).as_operation());

    // Set $yield_cont = continuation struct
    ops.push(wasm::global_set(db, location, cont_val, YIELD_CONT_IDX).as_operation());

    // Set $yield_value from shift's operand (or null if no value provided)
    let shift_operands = op.operands(db);
    let yield_value_val = if let Some(&first_value) = shift_operands.first() {
        // Use the first operand as the yield value
        first_value
    } else {
        // No value provided - use null
        let null_value = wasm::ref_null(
            db,
            location,
            anyref_ty,
            Attribute::Symbol(Symbol::new("any")),
        );
        let val = null_value.as_operation().result(db, 0);
        ops.push(null_value.as_operation());
        val
    };
    ops.push(wasm::global_set(db, location, yield_value_val, YIELD_VALUE_IDX).as_operation());

    // Set $yield_op_idx = op_idx (for multi-op ability dispatch)
    let op_idx_const = wasm::i32_const(db, location, i32_ty, Attribute::IntBits(op_idx_value));
    let op_idx_val = op_idx_const.as_operation().result(db, 0);
    ops.push(op_idx_const.as_operation());
    ops.push(wasm::global_set(db, location, op_idx_val, YIELD_OP_IDX).as_operation());

    // === 4. Return to unwind stack ===
    ops.push(wasm::r#return(db, location, None).as_operation());

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

        // If tag matches: handle the yielded effect
        //
        // Handler invocation flow:
        // 1. Reset yield_state to 0 (we're handling this yield)
        // 2. Load the continuation from $yield_cont
        // 3. Load the yield value from $yield_value
        // 4. Call the handler with (continuation, value)
        //
        // Currently, step 4 requires upstream changes:
        // - handler_lower.rs needs to populate shift's handler region with actual code
        // - The handler function would be passed via a global or stored in the continuation
        //
        // For now, we reset yield_state and load the values as preparation.

        let const_0 = wasm::i32_const(db, location, i32_ty, Attribute::IntBits(0));
        let const_0_val = const_0.as_operation().result(db, 0);
        let reset_yield = wasm::global_set(db, location, const_0_val, YIELD_STATE_IDX);

        // Load continuation from $yield_cont for handler invocation
        let cont_ty = cont_types::continuation_type(db);
        let get_cont = wasm::global_get(db, location, cont_ty, YIELD_CONT_IDX);
        let cont_val = get_cont.as_operation().result(db, 0);

        // Load yield value from $yield_value for handler invocation
        let anyref_ty = wasm::Anyref::new(db).as_type();
        let get_value = wasm::global_get(db, location, anyref_ty, YIELD_VALUE_IDX);
        let value_val = get_value.as_operation().result(db, 0);

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
                get_value.as_operation(),
                drop_cont.as_operation(),
                drop_value.as_operation(),
            ]),
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

// Yield global indices (must match order in lower_wasm.rs module_preamble_ops)
const YIELD_STATE_IDX: u32 = 0;
const YIELD_TAG_IDX: u32 = 1;
const YIELD_CONT_IDX: u32 = 2;
const YIELD_VALUE_IDX: u32 = 3;
const YIELD_OP_IDX: u32 = 4;

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

        // Collect suspend bodies with their op_idx values
        let mut suspend_arms: Vec<(u64, Region<'db>)> = Vec::new();
        for i in 0..num_suspend_arms {
            // Get the suspend body region (offset by 1 for done_body)
            let body = regions
                .get(i + 1)
                .cloned()
                .unwrap_or_else(|| Region::new(db, location, IdVec::new()));

            // Get the op_idx for this arm
            let attr_name = format!("op_idx_{}", i);
            let op_idx = op
                .attributes(db)
                .get(&Symbol::from_dynamic(&attr_name))
                .and_then(|attr| {
                    if let Attribute::IntBits(v) = attr {
                        Some(*v)
                    } else {
                        None
                    }
                })
                .unwrap_or(0);

            suspend_arms.push((op_idx, body));
        }

        // Get result type (if any)
        let result_ty = op
            .results(db)
            .first()
            .copied()
            .unwrap_or_else(|| core::Nil::new(db).as_type());

        // Check yield_state: global.get $yield_state
        let get_yield_state = wasm::global_get(db, location, i32_ty, YIELD_STATE_IDX);
        let yield_state_val = get_yield_state.as_operation().result(db, 0);

        // Build the suspend dispatch logic
        // If we have multiple suspend arms, build a nested if-chain based on op_idx
        let suspend_body = if suspend_arms.len() <= 1 {
            // Single or no suspend arm - use directly
            suspend_arms
                .first()
                .map(|(_, body)| *body)
                .unwrap_or_else(|| Region::new(db, location, IdVec::new()))
        } else {
            // Multiple suspend arms - build dispatch chain
            // if (op_idx == arm0_idx) { arm0_body }
            // else if (op_idx == arm1_idx) { arm1_body }
            // else { last_arm_body }
            build_multi_arm_dispatch(db, location, i32_ty, result_ty, &suspend_arms)
        };

        // Build wasm.if with:
        // - condition: yield_state (non-zero means yielding)
        // - then: suspend_body (yielding case with dispatch)
        // - else: done_body (normal completion case)
        let if_op = wasm::r#if(
            db,
            location,
            yield_state_val,
            result_ty,
            suspend_body,
            done_body,
        );

        // Expand to: [global_get, if]
        RewriteResult::Expand(vec![get_yield_state.as_operation(), if_op.as_operation()])
    }
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
        let expected_idx = wasm::i32_const(db, location, i32_ty, Attribute::IntBits(*op_idx));
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
        let i32_ty = core::I32::new(db).as_type();

        // Create empty handler region for shift
        let handler_block = Block::new(db, BlockId::fresh(), location, IdVec::new(), idvec![]);
        let handler_region = Region::new(db, location, idvec![handler_block]);

        // cont.shift now has a result (the value passed when continuation is resumed)
        let shift = Operation::of_name(db, location, "cont.shift")
            .attr("tag", Attribute::IntBits(42))
            .attr("op_idx", Attribute::IntBits(0)) // Required attribute for multi-op dispatch
            .result(i32_ty)
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

        // This test has a bare shift with no continuation ops and no live locals,
        // so no resume function is generated. The shift expansion is:
        // 0. wasm.struct_new (empty state struct)
        // 1. wasm.ref_null (null funcref for resume_fn since no resume function)
        // 2. wasm.i32_const (tag value for continuation struct)
        // 3. wasm.struct_new (continuation struct)
        // 4. wasm.i32_const (yield_state = 1)
        // 5. wasm.global_set (yield_state)
        // 6. wasm.i32_const (yield_tag)
        // 7. wasm.global_set (yield_tag)
        // 8. wasm.global_set (yield_cont)
        // 9. wasm.ref_null (yield_value = null)
        // 10. wasm.global_set (yield_value)
        // 11. wasm.i32_const (yield_op_idx = 0)
        // 12. wasm.global_set (yield_op_idx)
        // 13. wasm.return

        // First op: struct_new (empty state struct)
        assert_eq!(op_names.first(), Some(&"wasm.struct_new".to_string()));
        // Last op: return (to unwind stack)
        assert_eq!(op_names.last(), Some(&"wasm.return".to_string()));
        // Should NOT contain ref_func (no resume function when no continuation)
        assert!(!op_names.contains(&"wasm.ref_func".to_string()));
        // Should contain ref_null for resume_fn
        assert!(op_names.contains(&"wasm.ref_null".to_string()));
        // Should contain struct_new for continuation (and state)
        let struct_new_count = op_names.iter().filter(|n| *n == "wasm.struct_new").count();
        assert!(
            struct_new_count >= 2,
            "expected at least 2 struct_new (state + continuation), got {}",
            struct_new_count
        );
        // Total: 14 operations (no resume function, just shift expansion)
        assert_eq!(op_names.len(), 14);
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

        // cont.shift now has a result (the value passed when continuation is resumed)
        let shift = Operation::of_name(db, shift_loc, "cont.shift")
            .attr("tag", Attribute::IntBits(99))
            .result(i32_ty)
            .region(handler_region)
            .build();

        // Create function body with shift
        let func_body_block = Block::new(db, BlockId::fresh(), location, idvec![], idvec![shift]);
        let func_body = Region::new(db, location, idvec![func_body_block]);

        // Create func.func operation
        let func_ty = core::Func::new(db, idvec![], i32_ty).as_type();
        let func_op = Operation::of_name(db, location, "func.func")
            .attr("sym_name", Attribute::Symbol(Symbol::new("my_func")))
            .attr("type", Attribute::Type(func_ty))
            .region(func_body)
            .build();

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
        let local_op = Operation::of_name(db, location, "wasm.i32_const")
            .attr("value", Attribute::IntBits(42))
            .result(i32_ty)
            .build();
        let local_val = Value::new(db, ValueDef::OpResult(local_op), 0);

        // Create shift
        let handler_block = Block::new(db, BlockId::fresh(), shift_loc, IdVec::new(), idvec![]);
        let handler_region = Region::new(db, shift_loc, idvec![handler_block]);
        // cont.shift now has a result (the value passed when continuation is resumed)
        let shift = Operation::of_name(db, shift_loc, "cont.shift")
            .attr("tag", Attribute::IntBits(1))
            .attr("op_idx", Attribute::IntBits(0))
            .result(i32_ty)
            .region(handler_region)
            .build();

        // Use local_val after shift
        let use_after = Operation::of_name(db, location, "wasm.drop")
            .operand(local_val)
            .build();

        // Return
        let return_op = Operation::of_name(db, location, "wasm.return").build();

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
        let func_op = Operation::of_name(db, location, "func.func")
            .attr("sym_name", Attribute::Symbol(Symbol::new("fn_with_live")))
            .attr("type", Attribute::Type(func_ty))
            .region(func_body)
            .build();

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
