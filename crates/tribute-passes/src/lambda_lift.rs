//! Lambda lifting pass.
//!
//! Transforms `tribute.lambda` operations into:
//! 1. A lifted top-level `func.func` operation
//! 2. A `closure.new` operation at the original lambda location
//!
//! This pass runs after type checking (to know captured variable types)
//! and before TDNR (so closures are visible for method resolution).
//!
//! Architecture:
//! - Phase 1 (collect_lambda_info): Traverse module to collect captured variables
//!   for each lambda, keyed by Location. This is cached by Salsa.
//! - Phase 2 (lift_lambdas): Stateless transformation using the pre-collected info.

use std::collections::{BTreeMap, HashMap, HashSet};

use tribute_ir::ModulePathExt;
use tribute_ir::dialect::{adt, closure, tribute, tribute_pat};
use trunk_ir::dialect::{core, func};
use trunk_ir::rewrite::RewriteContext;
use trunk_ir::{
    Block, BlockId, DialectOp, DialectType, IdVec, Location, Operation, Region, Symbol, Type,
    Value, ValueDef,
};

// ============================================================================
// Phase 1: Collect lambda capture information (cached by Salsa)
// ============================================================================

/// Information about a captured variable.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub struct CaptureInfo<'db> {
    /// Variable name.
    pub name: Symbol,
    /// Variable type.
    pub ty: Type<'db>,
}

/// Information about a lambda, keyed by its location.
#[derive(Clone, Debug, PartialEq, Eq, salsa::Update)]
pub struct LambdaInfo<'db> {
    /// Captured variables (name + type).
    pub captures: Vec<CaptureInfo<'db>>,
    /// Generated function name.
    pub lifted_name: Symbol,
}

/// Map from lambda Location to its capture info.
/// Uses BTreeMap for deterministic ordering.
pub type LambdaInfoMap<'db> = BTreeMap<LocationKey, LambdaInfo<'db>>;

/// Serializable location key.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, salsa::Update)]
pub struct LocationKey {
    /// Start byte of the span.
    pub start: usize,
    /// End byte of the span.
    pub end: usize,
}

impl LocationKey {
    fn from_location(_db: &dyn salsa::Database, loc: Location<'_>) -> Self {
        // Use span bytes as key - each lambda should be at a unique location
        Self {
            start: loc.span.start,
            end: loc.span.end,
        }
    }
}

/// Collector for lambda information.
struct LambdaInfoCollector<'db> {
    db: &'db dyn salsa::Database,

    /// Stack of local scopes: Symbol -> Type.
    local_scopes: Vec<HashMap<Symbol, Type<'db>>>,

    /// Block argument types for resolving BlockArg values.
    block_args: HashMap<BlockId, IdVec<Type<'db>>>,

    /// Counter for generating unique lambda names.
    lambda_counter: u64,

    /// Module name for qualified lambda names.
    module_name: Symbol,
}

impl<'db> LambdaInfoCollector<'db> {
    fn new(db: &'db dyn salsa::Database, module_name: Symbol) -> Self {
        Self {
            db,
            local_scopes: Vec::new(),
            block_args: HashMap::new(),
            lambda_counter: 0,
            module_name,
        }
    }

    fn push_scope(&mut self) {
        self.local_scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.local_scopes.pop();
    }

    fn add_local(&mut self, name: Symbol, ty: Type<'db>) {
        if let Some(scope) = self.local_scopes.last_mut() {
            scope.insert(name, ty);
        }
    }

    fn lookup_local(&self, name: Symbol) -> Option<Type<'db>> {
        for scope in self.local_scopes.iter().rev() {
            if let Some(ty) = scope.get(&name) {
                return Some(*ty);
            }
        }
        None
    }

    fn gen_lambda_name(&mut self) -> Symbol {
        let name = Symbol::from_dynamic(&format!("__lambda_{}", self.lambda_counter));
        self.lambda_counter += 1;
        // Build qualified name as Symbol using module_name::lambda_name format
        self.module_name.join_path(name)
    }

    /// Extract parameter names from tribute.var ops at the start of a block.
    /// Function/lambda parameters are declared as tribute.var ops in the body.
    fn extract_param_names_from_body(
        &mut self,
        block: &Block<'db>,
        param_types: &IdVec<Type<'db>>,
    ) {
        let ops = block.operations(self.db);
        let mut param_idx = 0;

        for op in ops.iter() {
            if param_idx >= param_types.len() {
                break;
            }

            // Check if this is a tribute.var (parameter declaration)
            if let Ok(var_op) = tribute::Var::from_operation(self.db, *op) {
                let name = var_op.name(self.db);
                let ty = param_types[param_idx];
                self.add_local(name, ty);
                param_idx += 1;
            } else {
                // Non-var op means we're past parameter declarations
                break;
            }
        }
    }

    /// Collect info for all lambdas in the module.
    fn collect_module(&mut self, module: core::Module<'db>) -> LambdaInfoMap<'db> {
        let body = module.body(self.db);
        self.collect_region(&body)
    }

    fn collect_region(&mut self, region: &Region<'db>) -> LambdaInfoMap<'db> {
        let mut lambda_info = BTreeMap::new();
        for block in region.blocks(self.db).iter() {
            lambda_info.extend(self.collect_block(block));
        }
        lambda_info
    }

    fn collect_block(&mut self, block: &Block<'db>) -> LambdaInfoMap<'db> {
        // Register block argument types for BlockArg value lookups
        self.block_args
            .insert(block.id(self.db), block.arg_types(self.db));

        let mut lambda_info = BTreeMap::new();
        for op in block.operations(self.db).iter() {
            lambda_info.extend(self.collect_operation(op));
        }
        lambda_info
    }

    fn collect_operation(&mut self, op: &Operation<'db>) -> LambdaInfoMap<'db> {
        // Handle function definitions - track their scope
        if let Ok(func_op) = func::Func::from_operation(self.db, *op) {
            return self.collect_in_function(func_op);
        }

        // Handle lambda expressions - collect capture info
        if let Ok(lambda_op) = tribute::Lambda::from_operation(self.db, *op) {
            return self.collect_lambda(lambda_op);
        }

        // Handle let bindings - track bound variables
        if let Ok(let_op) = tribute::Let::from_operation(self.db, *op) {
            return self.collect_in_let(let_op);
        }

        // Handle case arm pattern bindings (including handler arm continuations)
        if let Ok(arm_op) = tribute::Arm::from_operation(self.db, *op) {
            return self.collect_in_arm(arm_op);
        }

        // Default: recurse into regions
        let mut lambda_info = BTreeMap::new();
        for region in op.regions(self.db).iter() {
            lambda_info.extend(self.collect_region(region));
        }
        lambda_info
    }

    fn collect_in_function(&mut self, op: func::Func<'db>) -> LambdaInfoMap<'db> {
        self.push_scope();

        // Add function parameters to scope
        for block in op.body(self.db).blocks(self.db) {
            let func_type = op.r#type(self.db);
            if let Some(func_ty) = core::Func::from_type(self.db, func_type) {
                let params = func_ty.params(self.db);
                for (i, &param_ty) in params.iter().enumerate() {
                    // Add synthetic param name for now
                    // Real param names will be extracted from tribute.var ops in the body
                    let param_name = Symbol::from_dynamic(&format!("__param_{}", i));
                    self.add_local(param_name, param_ty);
                }

                // Also extract parameter names from tribute.var ops at start of body
                self.extract_param_names_from_body(block, &params);
            }
        }

        // Process function body
        let lambda_info_map = self.collect_region(&op.body(self.db));

        self.pop_scope();
        lambda_info_map
    }

    fn collect_in_let(&mut self, op: tribute::Let<'db>) -> LambdaInfoMap<'db> {
        // Get the bound value type
        let bound_ty = self.get_value_type(op.value(self.db));

        // Extract bindings from pattern region
        let pattern_region = op.pattern(self.db);
        self.collect_pattern_bindings(&pattern_region, bound_ty);

        // Process nested regions
        self.collect_region(&pattern_region)
    }

    fn collect_in_arm(&mut self, op: tribute::Arm<'db>) -> LambdaInfoMap<'db> {
        self.push_scope();

        // Extract pattern bindings (recursively includes handler_suspend.continuation)
        let pattern_region = op.pattern(self.db);
        // Use a type variable as the scrutinee type - actual type comes from the bind op's result
        let scrutinee_ty = tribute::new_type_var(self.db, std::collections::BTreeMap::new());
        self.collect_pattern_bindings(&pattern_region, scrutinee_ty);

        // Process arm body (lambdas inside will now see pattern bindings including continuations)
        let result = self.collect_region(&op.body(self.db));

        self.pop_scope();
        result
    }

    fn get_value_type(&self, value: Value<'db>) -> Type<'db> {
        match value.def(self.db) {
            ValueDef::OpResult(op) => {
                if let Some(&ty) = op.results(self.db).get(value.index(self.db)) {
                    return ty;
                }
            }
            ValueDef::BlockArg(block_id) => {
                if let Some(&ty) = self
                    .block_args
                    .get(&block_id)
                    .and_then(|args| args.get(value.index(self.db)))
                {
                    return ty;
                }
            }
        }
        tribute::new_type_var(self.db, std::collections::BTreeMap::new())
    }

    /// Get the continuation type from a continuation region if it has a pat.bind with a typed result.
    fn get_continuation_type_from_region(&self, region: &Region<'db>) -> Option<Type<'db>> {
        for block in region.blocks(self.db) {
            for op in block.operations(self.db) {
                if tribute_pat::Bind::from_operation(self.db, *op).is_ok() {
                    // Get the result type of the bind operation
                    if let Some(&ty) = op.results(self.db).first() {
                        // Only return if it's not a type variable
                        if !tribute::is_type_var(self.db, ty) {
                            return Some(ty);
                        }
                    }
                }
            }
        }
        None
    }

    fn collect_pattern_bindings(&mut self, region: &Region<'db>, ty: Type<'db>) {
        for block in region.blocks(self.db) {
            for op in block.operations(self.db) {
                if let Ok(bind_op) = tribute_pat::Bind::from_operation(self.db, *op) {
                    let name = bind_op.name(self.db);
                    let binding_ty = op.results(self.db).first().copied().unwrap_or(ty);
                    self.add_local(name, binding_ty);
                }

                // Handle handler_suspend specially: the continuation region needs a continuation type
                if let Ok(handler_suspend) =
                    tribute_pat::HandlerSuspend::from_operation(self.db, *op)
                {
                    // Process args region with parent type
                    self.collect_pattern_bindings(&handler_suspend.args(self.db), ty);

                    // Process continuation region.
                    // The continuation type should already be set by typeck (cont.continuation).
                    // We use the type from the pat.bind result if available, otherwise fall back
                    // to a closure type for backwards compatibility.
                    let cont_region = handler_suspend.continuation(self.db);
                    let cont_ty = self
                        .get_continuation_type_from_region(&cont_region)
                        .unwrap_or_else(|| {
                            // Fallback: create a closure type if typeck didn't set the type
                            let func_ty = core::Func::new(
                                self.db,
                                IdVec::from(vec![ty]), // param: effect result type
                                ty,                    // return: same as effect result (simplified)
                            )
                            .as_type();
                            closure::Closure::new(self.db, func_ty).as_type()
                        });
                    self.collect_pattern_bindings(&cont_region, cont_ty);
                } else {
                    // Recurse into nested regions (for variant fields, etc.)
                    for nested_region in op.regions(self.db).iter() {
                        self.collect_pattern_bindings(nested_region, ty);
                    }
                }
            }
        }
    }

    fn collect_lambda(&mut self, op: tribute::Lambda<'db>) -> LambdaInfoMap<'db> {
        let location = op.location(self.db);
        let body = op.body(self.db);

        // Analyze captures
        let captures = self.analyze_captures(&body);

        // Generate unique name
        let lifted_name = self.gen_lambda_name();

        // Store info
        let location_key = LocationKey::from_location(self.db, location);

        let mut lambda_info_map = LambdaInfoMap::new();
        lambda_info_map.insert(
            location_key,
            LambdaInfo {
                captures,
                lifted_name,
            },
        );

        // Also collect from lambda body (for nested lambdas)
        // But first, add lambda params to scope
        self.push_scope();

        let lambda_type = op.r#type(self.db);
        if let Some(func_ty) = core::Func::from_type(self.db, lambda_type) {
            for block in body.blocks(self.db) {
                let params = func_ty.params(self.db);
                // Extract parameter names from tribute.var ops
                self.extract_param_names_from_body(block, &params);
            }
        }

        self.collect_region(&body);
        self.pop_scope();
        lambda_info_map
    }

    fn analyze_captures(&self, body: &Region<'db>) -> Vec<CaptureInfo<'db>> {
        let mut captures = Vec::new();
        let mut seen_names = HashSet::new();
        self.find_captures_in_region(body, &mut captures, &mut seen_names);
        captures
    }

    fn find_captures_in_region(
        &self,
        region: &Region<'db>,
        captures: &mut Vec<CaptureInfo<'db>>,
        seen_names: &mut HashSet<Symbol>,
    ) {
        for block in region.blocks(self.db).iter() {
            for op in block.operations(self.db).iter() {
                self.find_captures_in_op(op, captures, seen_names);
            }
        }
    }

    fn find_captures_in_op(
        &self,
        op: &Operation<'db>,
        captures: &mut Vec<CaptureInfo<'db>>,
        seen_names: &mut HashSet<Symbol>,
    ) {
        let dialect = op.dialect(self.db);
        let op_name = op.name(self.db);

        // Check for variable references
        if dialect == tribute::DIALECT_NAME() && op_name == tribute::VAR() {
            if let Ok(var_op) = tribute::Var::from_operation(self.db, *op) {
                let name = var_op.name(self.db);

                if !seen_names.contains(&name)
                    && let Some(ty) = self.lookup_local(name)
                {
                    seen_names.insert(name);
                    captures.push(CaptureInfo { name, ty });
                }
            }
            return;
        }

        // Skip nested lambdas - they handle their own captures
        if dialect == tribute::DIALECT_NAME() && op_name == tribute::LAMBDA() {
            return;
        }

        // Recurse into nested regions
        for region in op.regions(self.db).iter() {
            self.find_captures_in_region(region, captures, seen_names);
        }
    }
}

/// Collect capture information for all lambdas in a module.
/// Cached by salsa for incremental compilation.
#[salsa::tracked]
pub fn collect_lambda_info<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
) -> LambdaInfoMap<'db> {
    let mut collector = LambdaInfoCollector::new(db, module.name(db));
    collector.collect_module(module)
}

// ============================================================================
// Phase 2: Transform lambdas using pre-collected info
// ============================================================================

/// Lambda lifter that uses pre-collected capture info.
struct LambdaTransformer<'db, 'a> {
    db: &'db dyn salsa::Database,

    /// Rewrite context for value mapping.
    ctx: RewriteContext<'db>,

    /// Pre-collected lambda info.
    lambda_info: &'a LambdaInfoMap<'db>,

    /// Stack of local scopes for value lookup: Symbol -> Value.
    local_scopes: Vec<HashMap<Symbol, Value<'db>>>,

    /// Collected lifted functions to be added to module.
    lifted_functions: Vec<Operation<'db>>,
}

impl<'db, 'a> LambdaTransformer<'db, 'a> {
    fn new(db: &'db dyn salsa::Database, lambda_info: &'a LambdaInfoMap<'db>) -> Self {
        Self {
            db,
            ctx: RewriteContext::new(),
            lambda_info,
            local_scopes: Vec::new(),
            lifted_functions: Vec::new(),
        }
    }

    fn push_scope(&mut self) {
        self.local_scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.local_scopes.pop();
    }

    fn add_local(&mut self, name: Symbol, value: Value<'db>) {
        if let Some(scope) = self.local_scopes.last_mut() {
            scope.insert(name, value);
        }
    }

    fn lookup_local(&self, name: Symbol) -> Option<Value<'db>> {
        for scope in self.local_scopes.iter().rev() {
            if let Some(value) = scope.get(&name) {
                return Some(*value);
            }
        }
        None
    }

    /// Extract parameter names from tribute.var ops at the start of a block
    /// and map them to block argument values.
    fn extract_param_bindings(&mut self, block: &Block<'db>) {
        let ops = block.operations(self.db);
        let block_id = block.id(self.db);
        let mut param_idx = 0;

        for op in ops.iter() {
            // Check if this is a tribute.var (parameter declaration)
            if op.dialect(self.db) == tribute::DIALECT_NAME() && op.name(self.db) == tribute::VAR()
            {
                if let Ok(var_op) = tribute::Var::from_operation(self.db, *op) {
                    let name = var_op.name(self.db);
                    let block_arg = Value::new(self.db, ValueDef::BlockArg(block_id), param_idx);
                    self.add_local(name, block_arg);
                    param_idx += 1;
                }
            } else {
                // Non-var op means we're past parameter declarations
                break;
            }
        }
    }

    /// Transform the module.
    fn transform_module(&mut self, module: &core::Module<'db>) -> core::Module<'db> {
        let body = module.body(self.db);
        let new_body = self.transform_region(&body);

        // Combine lifted functions with transformed body
        let blocks = new_body.blocks(self.db);
        if blocks.is_empty() {
            return *module;
        }

        let first_block = &blocks[0];
        let mut all_ops: IdVec<Operation<'db>> = IdVec::new();

        // Add lifted functions first
        for lifted_fn in &self.lifted_functions {
            all_ops.push(*lifted_fn);
        }

        // Add original operations
        for op in first_block.operations(self.db).iter() {
            all_ops.push(*op);
        }

        let new_block = Block::new(
            self.db,
            first_block.id(self.db),
            first_block.location(self.db),
            first_block.args(self.db).clone(),
            all_ops,
        );

        let new_region = Region::new(
            self.db,
            body.location(self.db),
            IdVec::from(vec![new_block]),
        );

        core::Module::create(
            self.db,
            module.location(self.db),
            module.name(self.db),
            new_region,
        )
    }

    fn transform_region(&mut self, region: &Region<'db>) -> Region<'db> {
        let new_blocks: IdVec<Block<'db>> = region
            .blocks(self.db)
            .iter()
            .map(|block| self.transform_block(block))
            .collect();
        Region::new(self.db, region.location(self.db), new_blocks)
    }

    fn transform_block(&mut self, block: &Block<'db>) -> Block<'db> {
        let new_ops: IdVec<Operation<'db>> = block
            .operations(self.db)
            .iter()
            .flat_map(|op| self.transform_operation(op))
            .collect();

        Block::new(
            self.db,
            block.id(self.db),
            block.location(self.db),
            block.args(self.db).clone(),
            new_ops,
        )
    }

    fn transform_operation(&mut self, op: &Operation<'db>) -> Vec<Operation<'db>> {
        // Remap operands from previous transformations
        let remapped_op = self.ctx.remap_operands(self.db, op);

        let dialect = remapped_op.dialect(self.db);
        let op_name = remapped_op.name(self.db);

        // Handle function definitions - track their scope
        if dialect == func::DIALECT_NAME() && op_name == func::FUNC() {
            let final_op = self.transform_in_function(&remapped_op);
            // Map results from original op to final op
            if final_op != *op {
                self.ctx.map_results(self.db, op, &final_op);
            }
            return vec![final_op];
        }

        // Handle lambda expressions - transform to closure.new
        if dialect == tribute::DIALECT_NAME() && op_name == tribute::LAMBDA() {
            let result = self.transform_lambda(&remapped_op);
            // Lambda transform returns [env_op, closure_op]; the closure result replaces lambda
            // Note: transform_lambda already maps old_result -> new_result internally
            return result;
        }

        // Handle let bindings - track bound variables
        if dialect == tribute::DIALECT_NAME() && op_name == tribute::LET() {
            let final_op = self.transform_in_let(&remapped_op);
            if final_op != *op {
                self.ctx.map_results(self.db, op, &final_op);
            }
            return vec![final_op];
        }

        // Default: recursively process regions
        let final_op = self.transform_op_regions(&remapped_op);
        // Always map results when the operation changed (operands or regions)
        if final_op != *op {
            self.ctx.map_results(self.db, op, &final_op);
        }
        vec![final_op]
    }

    fn transform_in_function(&mut self, op: &Operation<'db>) -> Operation<'db> {
        self.push_scope();

        // Add function parameters to scope
        let regions = op.regions(self.db);
        if let Some(body) = regions.first() {
            let blocks = body.blocks(self.db);
            if let Some(entry_block) = blocks.first() {
                // Extract parameter names from tribute.var ops and map to block args
                self.extract_param_bindings(entry_block);
            }
        }

        // Process function body
        let result = self.transform_op_regions(op);

        self.pop_scope();

        result
    }

    fn transform_in_let(&mut self, op: &Operation<'db>) -> Operation<'db> {
        // Get the bound value
        let operands = op.operands(self.db);
        if let Some(&bound_value) = operands.first() {
            // Extract bindings from pattern region
            let regions = op.regions(self.db);
            if let Some(pattern_region) = regions.first() {
                self.collect_pattern_bindings(pattern_region, bound_value);
            }
        }

        // Process nested regions
        self.transform_op_regions(op)
    }

    fn collect_pattern_bindings(&mut self, region: &Region<'db>, value: Value<'db>) {
        for block in region.blocks(self.db).iter() {
            for op in block.operations(self.db).iter() {
                if let Ok(bind_op) = tribute_pat::Bind::from_operation(self.db, *op) {
                    let name = bind_op.name(self.db);
                    self.add_local(name, value);
                }
            }
        }
    }

    fn transform_lambda(&mut self, op: &Operation<'db>) -> Vec<Operation<'db>> {
        let location = op.location(self.db);
        let location_key = LocationKey::from_location(self.db, location);

        // Look up pre-collected info
        let info = match self.lambda_info.get(&location_key) {
            Some(info) => info,
            None => {
                // No captures - just return original op for now
                // This shouldn't happen if collection was done correctly
                return vec![*op];
            }
        };

        let lambda_op = tribute::Lambda::from_operation(self.db, *op)
            .expect("already checked this is tribute.lambda");

        let lambda_type = lambda_op.r#type(self.db);
        let body = lambda_op.body(self.db);

        // Parse the function type
        let (param_types, result_type, effect_type) =
            if let Some(func_ty) = core::Func::from_type(self.db, lambda_type) {
                (
                    func_ty.params(self.db).clone(),
                    func_ty.result(self.db),
                    func_ty.effect(self.db),
                )
            } else {
                let infer_ty = tribute::new_type_var(self.db, std::collections::BTreeMap::new());
                (IdVec::new(), infer_ty, None)
            };

        // Get capture values from current scope
        let mut capture_values: Vec<Value<'db>> = Vec::new();
        for capture_info in &info.captures {
            if let Some(value) = self.lookup_local(capture_info.name) {
                capture_values.push(value);
            }
        }

        // Create env struct type for captures
        let env_type = self.create_env_type(&info.captures, info.lifted_name);

        // Build lifted function (takes env as first param)
        let lifted_func = self.build_lifted_function(
            location,
            info.lifted_name,
            &info.captures,
            env_type,
            &param_types,
            result_type,
            effect_type,
            &body,
        );

        self.lifted_functions.push(lifted_func);

        // Create env struct with captured values
        // IMPORTANT: This operation must be included in the output!
        let env_op = if capture_values.is_empty() {
            // No captures - create unit/nil value for env
            let nil_ty = core::Nil::new(self.db);
            adt::struct_new(self.db, location, capture_values, *nil_ty, *nil_ty).as_operation()
        } else {
            adt::struct_new(self.db, location, capture_values, env_type, env_type).as_operation()
        };
        let env_value = env_op.result(self.db, 0);

        // Create closure.new with closure type (not raw function type)
        // This allows us to distinguish closures from bare function refs
        let closure_ty = closure::Closure::new(self.db, lambda_type);
        let closure_op = closure::new(self.db, location, env_value, *closure_ty, info.lifted_name);

        // Map old lambda result to new closure result
        let old_result = op.result(self.db, 0);
        let new_result = closure_op.as_operation().result(self.db, 0);
        self.ctx.map_value(old_result, new_result);

        // Return both env struct and closure operations
        // The env struct must come first since closure.new references its result
        vec![env_op, closure_op.as_operation()]
    }

    /// Create an env struct type for the captured variables.
    fn create_env_type(&self, captures: &[CaptureInfo<'db>], lambda_name: Symbol) -> Type<'db> {
        if captures.is_empty() {
            return *core::Nil::new(self.db);
        }

        // Create anonymous struct type with captured variable types as fields
        let fields: Vec<(Symbol, Type<'db>)> = captures
            .iter()
            .enumerate()
            .map(|(i, cap)| (Symbol::from_dynamic(&format!("_{}", i)), cap.ty))
            .collect();

        // Use lambda name + "_env" for the struct name
        // e.g., "module::__lambda_0" -> "module::__lambda_0_env"
        let env_name = if let Some(parent) = lambda_name.parent_path() {
            parent.join_path(Symbol::from_dynamic(&format!(
                "{}_env",
                lambda_name.last_segment()
            )))
        } else {
            Symbol::from_dynamic(&format!("{}_env", lambda_name))
        };

        adt::struct_type(self.db, env_name, fields)
    }

    #[allow(clippy::too_many_arguments)]
    fn build_lifted_function(
        &self,
        location: Location<'db>,
        name: Symbol,
        captures: &[CaptureInfo<'db>],
        env_type: Type<'db>,
        param_types: &IdVec<Type<'db>>,
        result_type: Type<'db>,
        effect_type: Option<Type<'db>>,
        body: &Region<'db>,
    ) -> Operation<'db> {
        // Build parameter list: env first, then original params
        let mut all_params: IdVec<Type<'db>> = IdVec::new();
        all_params.push(env_type);
        all_params.extend(param_types.iter().copied());

        let db = self.db;
        let captures_vec: Vec<_> = captures.to_vec();
        let param_count = param_types.len();

        // Build the function
        func::Func::build_with_effect(
            self.db,
            location,
            name,
            all_params,
            result_type,
            effect_type,
            |entry| {
                // Get env parameter (first block argument)
                let env_param = entry.block_arg(db, 0);

                // Build mapping: capture name -> extracted value
                let mut capture_values: HashMap<Symbol, Value<'db>> = HashMap::new();
                for (i, capture) in captures_vec.iter().enumerate() {
                    let extracted = entry.op(adt::struct_get(
                        db,
                        location,
                        env_param,
                        capture.ty,
                        env_type,
                        trunk_ir::Attribute::IntBits(i as u64),
                    ));
                    capture_values.insert(capture.name, extracted.result(db));
                }

                // Transform the lambda body
                let body_blocks = body.blocks(db);
                if let Some(orig_block) = body_blocks.first() {
                    let orig_block_id = orig_block.id(db);

                    // Build value remapping context
                    let mut value_map: HashMap<Value<'db>, Value<'db>> = HashMap::new();

                    // Map original block args (params 0..n) to new block args (1..n+1)
                    for i in 0..param_count {
                        let orig_arg = Value::new(db, ValueDef::BlockArg(orig_block_id), i);
                        let new_arg = entry.block_arg(db, i + 1);
                        value_map.insert(orig_arg, new_arg);
                    }

                    // Process operations from original body
                    let ops = orig_block.operations(db);
                    let mut param_decl_count = 0;

                    for op in ops.iter() {
                        // Handle tribute.var ops - either parameter declarations or captured refs
                        if let Ok(var_op) = tribute::Var::from_operation(db, *op) {
                            let var_name = var_op.name(db);

                            // Check if this is a parameter declaration (first N tribute.var ops)
                            if param_decl_count < param_count {
                                // This is a parameter declaration
                                // Map its result to the shifted block arg
                                let orig_result = op.result(db, 0);
                                let new_arg = entry.block_arg(db, param_decl_count + 1);
                                value_map.insert(orig_result, new_arg);
                                param_decl_count += 1;
                                // Don't emit the tribute.var op - params are block args now
                                continue;
                            }

                            // Check if this is a captured variable reference
                            if let Some(&extracted_val) = capture_values.get(&var_name) {
                                // Map the tribute.var result to the extracted value
                                let orig_result = op.result(db, 0);
                                value_map.insert(orig_result, extracted_val);
                                // Don't emit the tribute.var op - we use the extracted value
                                continue;
                            }
                        }

                        // Handle tribute.yield -> func.return conversion
                        if op.dialect(db) == tribute::DIALECT_NAME()
                            && op.name(db) == tribute::YIELD()
                        {
                            let return_vals: Vec<_> = op
                                .operands(db)
                                .iter()
                                .map(|&v| *value_map.get(&v).unwrap_or(&v))
                                .collect();
                            entry.op(func::r#return(db, op.location(db), return_vals));
                            continue;
                        }

                        // For other ops, remap operands and transform nested regions
                        let remapped_operands: IdVec<Value<'db>> = op
                            .operands(db)
                            .iter()
                            .map(|&v| *value_map.get(&v).unwrap_or(&v))
                            .collect();

                        // Transform nested regions (for tribute.block, etc.)
                        let new_regions =
                            transform_nested_regions(db, op, &capture_values, &mut value_map);

                        let mut builder = op.modify(db);
                        if remapped_operands != *op.operands(db) {
                            builder = builder.operands(remapped_operands);
                        }
                        if !new_regions.is_empty() {
                            builder = builder.regions(new_regions);
                        }
                        let new_op = builder.build();

                        entry.op(new_op);

                        // Map old results to new results
                        for (i, _) in op.results(db).iter().enumerate() {
                            let old_result = op.result(db, i);
                            let new_result = new_op.result(db, i);
                            if old_result != new_result {
                                value_map.insert(old_result, new_result);
                            }
                        }
                    }
                }
            },
        )
        .as_operation()
    }
}

/// Recursively transform nested regions, replacing tribute.var references
/// to captured variables with the extracted values.
fn transform_nested_regions<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
    capture_values: &HashMap<Symbol, Value<'db>>,
    value_map: &mut HashMap<Value<'db>, Value<'db>>,
) -> IdVec<Region<'db>> {
    let regions = op.regions(db);
    if regions.is_empty() {
        return IdVec::new();
    }

    regions
        .iter()
        .map(|region| transform_region_for_captures(db, region, capture_values, value_map))
        .collect()
}

fn transform_region_for_captures<'db>(
    db: &'db dyn salsa::Database,
    region: &Region<'db>,
    capture_values: &HashMap<Symbol, Value<'db>>,
    value_map: &mut HashMap<Value<'db>, Value<'db>>,
) -> Region<'db> {
    let new_blocks: IdVec<Block<'db>> = region
        .blocks(db)
        .iter()
        .map(|block| transform_block_for_captures(db, block, capture_values, value_map))
        .collect();
    Region::new(db, region.location(db), new_blocks)
}

fn transform_block_for_captures<'db>(
    db: &'db dyn salsa::Database,
    block: &Block<'db>,
    capture_values: &HashMap<Symbol, Value<'db>>,
    value_map: &mut HashMap<Value<'db>, Value<'db>>,
) -> Block<'db> {
    let mut new_ops: IdVec<Operation<'db>> = IdVec::new();

    for op in block.operations(db).iter() {
        // Handle tribute.var - map captured variables
        if let Ok(var_op) = tribute::Var::from_operation(db, *op) {
            let var_name = var_op.name(db);

            if let Some(&extracted_val) = capture_values.get(&var_name) {
                // Map this var's result to the captured value
                let orig_result = op.result(db, 0);
                value_map.insert(orig_result, extracted_val);
                // Don't emit the tribute.var - it's now the captured value
                continue;
            }
        }

        // Remap operands
        let remapped_operands: IdVec<Value<'db>> = op
            .operands(db)
            .iter()
            .map(|&v| *value_map.get(&v).unwrap_or(&v))
            .collect();

        // Recursively transform nested regions
        let new_regions = transform_nested_regions(db, op, capture_values, value_map);

        let mut builder = op.modify(db);
        if remapped_operands != *op.operands(db) {
            builder = builder.operands(remapped_operands);
        }
        if !new_regions.is_empty() {
            builder = builder.regions(new_regions);
        }
        let new_op = builder.build();

        new_ops.push(new_op);

        // Map old results to new results
        for (i, _) in op.results(db).iter().enumerate() {
            let old_result = op.result(db, i);
            let new_result = new_op.result(db, i);
            if old_result != new_result {
                value_map.insert(old_result, new_result);
            }
        }
    }

    Block::new(
        db,
        block.id(db),
        block.location(db),
        block.args(db).clone(),
        new_ops,
    )
}

impl<'db, 'a> LambdaTransformer<'db, 'a> {
    fn transform_op_regions(&mut self, op: &Operation<'db>) -> Operation<'db> {
        let regions = op.regions(self.db);
        if regions.is_empty() {
            return *op;
        }

        let new_regions: IdVec<Region<'db>> =
            regions.iter().map(|r| self.transform_region(r)).collect();

        // Use regions() to replace all regions at once (not region() which appends)
        op.modify(self.db).regions(new_regions).build()
    }
}

// ============================================================================
// Public API
// ============================================================================

/// Entry point: lift all lambdas in a module.
///
/// This is a two-phase process:
/// 1. Collect capture information for all lambdas (cacheable)
/// 2. Transform lambdas to closure.new + lifted functions
#[salsa::tracked]
pub fn lift_lambdas<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
) -> core::Module<'db> {
    // Phase 1: Collect lambda info
    let lambda_info = collect_lambda_info(db, module);

    // If no lambdas, return module unchanged
    if lambda_info.is_empty() {
        return module;
    }

    // Phase 2: Transform using collected info
    let mut transformer = LambdaTransformer::new(db, &lambda_info);
    transformer.transform_module(&module)
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use tribute_ir::dialect::{closure, tribute};
    use trunk_ir::dialect::{arith, core, func};
    use trunk_ir::{BlockArg, BlockId, Location, PathId, Span, idvec};

    fn test_location(db: &dyn salsa::Database) -> Location<'_> {
        let path = PathId::new(db, "file:///test.trb".to_owned());
        Location::new(path, Span::new(0, 100))
    }

    fn lambda_location(db: &dyn salsa::Database, start: usize) -> Location<'_> {
        let path = PathId::new(db, "file:///test.trb".to_owned());
        Location::new(path, Span::new(start, start + 10))
    }

    /// Build a module without lambdas.
    #[salsa::tracked]
    fn build_module_no_lambdas(db: &dyn salsa::Database) -> core::Module<'_> {
        let location = test_location(db);

        core::Module::build(db, location, Symbol::new("test"), |top| {
            top.op(func::Func::build(
                db,
                location,
                "main",
                idvec![],
                *core::I64::new(db),
                |entry| {
                    let val = entry.op(arith::Const::i64(db, location, 42));
                    entry.op(func::Return::value(db, location, val.result(db)));
                },
            ));
        })
    }

    /// Test that a module without lambdas is unchanged.
    #[salsa_test]
    fn test_no_lambdas(db: &salsa::DatabaseImpl) {
        let module = build_module_no_lambdas(db);
        let lifted = lift_lambdas(db, module);

        // Module should be unchanged
        assert_eq!(lifted, module);
    }

    /// Build a module with a lambda that captures 'x'.
    #[salsa::tracked]
    fn build_module_with_capture(db: &dyn salsa::Database) -> core::Module<'_> {
        let location = test_location(db);
        let lambda_loc = lambda_location(db, 50);

        core::Module::build(db, location, Symbol::new("test"), |top| {
            top.op(func::Func::build(
                db,
                location,
                "outer",
                idvec![*core::I64::new(db)],
                *core::I64::new(db),
                |entry| {
                    // Parameter x
                    let x_var = entry.op(tribute::var(
                        db,
                        location,
                        *core::I64::new(db),
                        Symbol::new("x"),
                    ));

                    // Lambda that captures x: fn(y) { x + y }
                    let lambda_type =
                        core::Func::new(db, idvec![*core::I64::new(db)], *core::I64::new(db))
                            .as_type();

                    let lambda_body = Region::new(
                        db,
                        lambda_loc,
                        idvec![Block::new(
                            db,
                            BlockId::fresh(),
                            lambda_loc,
                            idvec![BlockArg::of_type(db, *core::I64::new(db))],
                            {
                                let mut ops = IdVec::new();
                                // Parameter y declaration
                                let y_var = tribute::var(
                                    db,
                                    lambda_loc,
                                    *core::I64::new(db),
                                    Symbol::new("y"),
                                );
                                ops.push(y_var.as_operation());

                                // Reference to captured x
                                let x_ref = tribute::var(
                                    db,
                                    lambda_loc,
                                    *core::I64::new(db),
                                    Symbol::new("x"),
                                );
                                ops.push(x_ref.as_operation());

                                // x + y
                                let add_op = arith::add(
                                    db,
                                    lambda_loc,
                                    x_ref.result(db),
                                    y_var.result(db),
                                    *core::I64::new(db),
                                );
                                ops.push(add_op.as_operation());

                                // yield result
                                let yield_op = tribute::r#yield(db, lambda_loc, add_op.result(db));
                                ops.push(yield_op.as_operation());
                                ops
                            },
                        )],
                    );

                    let lambda = entry.op(tribute::lambda(
                        db,
                        lambda_loc,
                        lambda_type,
                        lambda_type,
                        lambda_body,
                    ));

                    let _ = x_var;
                    entry.op(func::Return::value(db, location, lambda.result(db)));
                },
            ));
        })
    }

    /// Test collecting capture info from a lambda.
    #[salsa_test]
    fn test_collect_captures(db: &salsa::DatabaseImpl) {
        let module = build_module_with_capture(db);

        // Collect lambda info
        let info = collect_lambda_info(db, module);

        // Should have one lambda
        assert_eq!(info.len(), 1);

        // The lambda should have captured 'x'
        let (_key, lambda_info) = info.iter().next().unwrap();
        assert_eq!(lambda_info.captures.len(), 1);
        assert_eq!(lambda_info.captures[0].name, Symbol::new("x"));
    }

    /// Build a simple module with an identity lambda.
    #[salsa::tracked]
    fn build_module_with_simple_lambda(db: &dyn salsa::Database) -> core::Module<'_> {
        let location = test_location(db);
        let lambda_loc = lambda_location(db, 50);

        core::Module::build(db, location, Symbol::new("test"), |top| {
            top.op(func::Func::build(
                db,
                location,
                "main",
                idvec![],
                *core::I64::new(db),
                |entry| {
                    // Simple lambda: fn(x) { x }
                    let lambda_type =
                        core::Func::new(db, idvec![*core::I64::new(db)], *core::I64::new(db))
                            .as_type();

                    let lambda_body = Region::new(
                        db,
                        lambda_loc,
                        idvec![Block::new(
                            db,
                            BlockId::fresh(),
                            lambda_loc,
                            idvec![BlockArg::of_type(db, *core::I64::new(db))],
                            {
                                let mut ops = IdVec::new();
                                let x_var = tribute::var(
                                    db,
                                    lambda_loc,
                                    *core::I64::new(db),
                                    Symbol::new("x"),
                                );
                                ops.push(x_var.as_operation());
                                let yield_op = tribute::r#yield(db, lambda_loc, x_var.result(db));
                                ops.push(yield_op.as_operation());
                                ops
                            },
                        )],
                    );

                    let lambda = entry.op(tribute::lambda(
                        db,
                        lambda_loc,
                        lambda_type,
                        lambda_type,
                        lambda_body,
                    ));

                    entry.op(func::Return::value(db, location, lambda.result(db)));
                },
            ));
        })
    }

    /// Lift lambdas in the simple lambda module.
    #[salsa::tracked]
    fn lift_simple_lambda_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let module = build_module_with_simple_lambda(db);
        lift_lambdas(db, module)
    }

    /// Test that lifted functions are added to the module.
    #[salsa_test]
    fn test_lifted_function_created(db: &salsa::DatabaseImpl) {
        let lifted = lift_simple_lambda_module(db);

        // Count operations in the module
        let body = lifted.body(db);
        let blocks = body.blocks(db);
        let ops = blocks[0].operations(db);

        // Should have:
        // 1. Lifted lambda function (__lambda_0)
        // 2. Original main function (with closure.new instead of tribute.lambda)
        assert!(ops.len() >= 2, "Expected at least 2 ops, got {}", ops.len());

        // Check that first op is a func.func (the lifted lambda)
        let first_op = &ops[0];
        assert_eq!(first_op.dialect(db), func::DIALECT_NAME());
        assert_eq!(first_op.name(db), func::FUNC());

        // Check that it has the expected name pattern
        if let Ok(lifted_func) = func::Func::from_operation(db, *first_op) {
            let name = lifted_func.sym_name(db);
            let starts_with_lambda = name.last_segment().with_str(|s| s.starts_with("__lambda_"));
            assert!(
                starts_with_lambda,
                "Expected lifted function name to start with __lambda_, got {:?}",
                name
            );
        }
    }

    /// Test that closure.new is created at the lambda location.
    #[salsa_test]
    fn test_closure_new_created(db: &salsa::DatabaseImpl) {
        let lifted = lift_simple_lambda_module(db);

        // Find the main function and check for closure.new
        let body = lifted.body(db);
        let blocks = body.blocks(db);
        let ops = blocks[0].operations(db);

        // Find the main function
        let main_func = ops.iter().find(|op| {
            if op.dialect(db) == func::DIALECT_NAME()
                && op.name(db) == func::FUNC()
                && let Ok(f) = func::Func::from_operation(db, **op)
            {
                return f.sym_name(db).last_segment() == "main";
            }
            false
        });

        assert!(main_func.is_some(), "Main function not found");

        let main_func = main_func.unwrap();
        let main_body = main_func.regions(db);
        let main_blocks = main_body[0].blocks(db);
        let main_ops = main_blocks[0].operations(db);

        // Check for closure.new in main function
        let has_closure_new = main_ops
            .iter()
            .any(|op| op.dialect(db) == closure::DIALECT_NAME() && op.name(db) == closure::NEW());

        assert!(
            has_closure_new,
            "Expected closure.new in main function body"
        );
    }
}
