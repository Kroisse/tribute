//! Bidirectional type checker for TrunkIR.
//!
//! This module implements bidirectional type checking with the following judgments:
//!
//! ```text
//! Γ ⊢ e ⇒ A ; E    -- Infer: infer type A and effect E for expression e
//! Γ ⊢ e ⇐ A ; E    -- Check: check expression e against type A with effect E
//! ```
//!
//! The checker walks TrunkIR operations and generates constraints, which are
//! then solved by the [`TypeSolver`].

use std::collections::HashMap;
use std::sync::Arc;

use crate::diagnostic::{CompilationPhase, Diagnostic, DiagnosticSeverity};
use salsa::Accumulator;
use tracing::trace;
use tribute_ir::dialect::{ability, closure, tribute, tribute_pat};
use trunk_ir::dialect::adt;
use trunk_ir::{
    Attribute, Block, DialectOp, DialectType, IdVec, Operation, Region, Symbol, Type, Value,
    dialect::{arith, cont, core, func},
};

use super::constraint::ConstraintSet;
use super::effect_row::{AbilityRef, EffectRow, RowVar};
use super::solver::{SolveResult, TypeSolver};
use super::subst::has_type_vars;

/// Type checking mode: infer or check.
#[allow(dead_code)] // Part of public API, will be used in future
#[derive(Clone, Copy, Debug)]
pub enum Mode<'db> {
    /// Infer mode: synthesize the type of an expression.
    Infer,
    /// Check mode: check against an expected type.
    Check(Type<'db>),
}

/// Signature of an ability operation.
///
/// Used for looking up operation signatures when type-checking handler patterns.
/// The continuation type in a handler arm is derived from this signature:
/// `k: (return_ty) ->{remaining_effects} handler_result_ty`
#[derive(Clone, Debug)]
pub struct AbilityOpSignature<'db> {
    /// The ability name (e.g., "State").
    pub ability_name: Symbol,
    /// The operation name (e.g., "get", "set").
    pub op_name: Symbol,
    /// Parameter types of the operation.
    pub params: IdVec<Type<'db>>,
    /// Return type of the operation.
    pub return_ty: Type<'db>,
}

/// Key for looking up ability operation signatures.
pub type AbilityOpKey = (Symbol, Symbol); // (ability_name, op_name)

/// Type checking context.
///
/// Tracks the current environment (bindings) and generates constraints.
pub struct TypeChecker<'db> {
    db: &'db dyn salsa::Database,
    /// Map from SSA values to their types.
    value_types: HashMap<Value<'db>, Type<'db>>,
    /// Constraint set being built.
    constraints: ConstraintSet<'db>,
    /// Counter for fresh type variables.
    next_type_var: u64,
    /// Counter for fresh row variables.
    next_row_var: u64,
    /// Current effect row (effects performed by current expression).
    current_effect: EffectRow<'db>,
    /// Entry block argument types for the current function being checked.
    /// Used as fallback for block argument lookups when the resolver creates
    /// stale block references (issue #43).
    entry_block_arg_types: Vec<Type<'db>>,
    /// Map from function names to their types for looking up callees.
    /// This enables proper type inference for generic function calls.
    /// Wrapped in Arc for cheap cloning across function checks.
    function_types: Arc<HashMap<Symbol, Type<'db>>>,
    /// Map from (ability_name, op_name) to operation signatures.
    /// Used for looking up ability operation types when checking handler patterns.
    /// Wrapped in Arc for cheap cloning across function checks.
    ability_op_types: Arc<HashMap<AbilityOpKey, AbilityOpSignature<'db>>>,
    /// Map from type names to their ADT types (enum or struct).
    /// Used for resolving user-defined type references during type checking.
    /// Wrapped in Arc for cheap cloning across function checks.
    type_defs: Arc<HashMap<Symbol, Type<'db>>>,
}

impl<'db> TypeChecker<'db> {
    /// Create a new type checker.
    pub fn new(db: &'db dyn salsa::Database) -> Self {
        Self {
            db,
            value_types: HashMap::new(),
            constraints: ConstraintSet::new(),
            next_type_var: 0,
            next_row_var: 0,
            current_effect: EffectRow::empty(),
            entry_block_arg_types: Vec::new(),
            function_types: Arc::new(HashMap::new()),
            ability_op_types: Arc::new(HashMap::new()),
            type_defs: Arc::new(HashMap::new()),
        }
    }

    /// Create a new type checker with a pre-built function type map.
    pub fn with_function_types(
        db: &'db dyn salsa::Database,
        function_types: Arc<HashMap<Symbol, Type<'db>>>,
    ) -> Self {
        Self {
            db,
            value_types: HashMap::new(),
            constraints: ConstraintSet::new(),
            next_type_var: 0,
            next_row_var: 0,
            current_effect: EffectRow::empty(),
            entry_block_arg_types: Vec::new(),
            function_types,
            ability_op_types: Arc::new(HashMap::new()),
            type_defs: Arc::new(HashMap::new()),
        }
    }

    /// Create a new type checker with pre-built function, ability operation, and type definition maps.
    pub fn with_type_maps(
        db: &'db dyn salsa::Database,
        function_types: Arc<HashMap<Symbol, Type<'db>>>,
        ability_op_types: Arc<HashMap<AbilityOpKey, AbilityOpSignature<'db>>>,
        type_defs: Arc<HashMap<Symbol, Type<'db>>>,
    ) -> Self {
        Self {
            db,
            value_types: HashMap::new(),
            constraints: ConstraintSet::new(),
            next_type_var: 0,
            next_row_var: 0,
            current_effect: EffectRow::empty(),
            entry_block_arg_types: Vec::new(),
            function_types,
            ability_op_types,
            type_defs,
        }
    }

    /// Generate a fresh type variable.
    pub fn fresh_type_var(&mut self) -> Type<'db> {
        let id = self.next_type_var;
        self.next_type_var += 1;
        tribute::type_var_with_id(self.db, id)
    }

    /// Generate a fresh row variable.
    pub fn fresh_row_var(&mut self) -> RowVar {
        let id = self.next_row_var;
        self.next_row_var += 1;
        RowVar::new(id)
    }

    /// Record the type of a value.
    pub fn record_type(&mut self, value: Value<'db>, ty: Type<'db>) {
        self.value_types.insert(value, ty);
    }

    /// Get the type of a value.
    ///
    /// For block arguments, this also tries a fallback lookup by argument index
    /// to handle stale block references created by the resolver (issue #43).
    pub fn get_type(&self, value: Value<'db>) -> Option<Type<'db>> {
        // Direct lookup first
        if let Some(ty) = self.value_types.get(&value).copied() {
            return Some(ty);
        }

        // For BlockArg, try entry block arg types as fallback.
        // This handles the case where the resolver creates block arguments
        // that reference old blocks, but the actual types are recorded for
        // the new blocks created during resolution.
        if let trunk_ir::ValueDef::BlockArg(_) = value.def(self.db) {
            let index = value.index(self.db);
            if index < self.entry_block_arg_types.len() {
                return Some(self.entry_block_arg_types[index]);
            }
        }

        None
    }

    /// Add a type equality constraint.
    pub fn constrain_eq(&mut self, t1: Type<'db>, t2: Type<'db>) {
        trace!(?t1, ?t2, "adding type constraint");
        self.constraints.add_type_eq(t1, t2);
    }

    /// Add a row equality constraint.
    pub fn constrain_row_eq(&mut self, r1: EffectRow<'db>, r2: EffectRow<'db>) {
        self.constraints.add_row_eq(r1, r2);
    }

    /// Get the accumulated constraints.
    pub fn constraints(&self) -> &ConstraintSet<'db> {
        &self.constraints
    }

    /// Take the constraints, consuming them.
    pub fn take_constraints(&mut self) -> ConstraintSet<'db> {
        std::mem::take(&mut self.constraints)
    }

    /// Get the current effect row.
    pub fn current_effect(&self) -> &EffectRow<'db> {
        &self.current_effect
    }

    /// Merge effects from a sub-expression.
    pub fn merge_effect(&mut self, effect: EffectRow<'db>) {
        let fresh_var = || {
            let id = self.next_row_var;
            self.next_row_var += 1;
            RowVar::new(id)
        };
        self.current_effect = self.current_effect.union(&effect, fresh_var);
    }

    /// Check for conflicting abilities in the current effect row and report an error.
    ///
    /// Conflicting abilities are abilities with the same name but different type parameters
    /// (e.g., `State(Int)` and `State(Text)`). Handler patterns match by name only, so having
    /// multiple parameterizations of the same ability would be ambiguous.
    ///
    /// This restriction may be lifted in the future with named effects.
    fn check_ability_conflicts(&self, span: trunk_ir::Span) {
        if let Some((name, abilities)) = self.current_effect.find_conflicting_abilities() {
            // Filter to only fully concrete abilities (non-empty params with no type variables)
            let concrete: Vec<_> = abilities
                .iter()
                .filter(|ability| {
                    !ability.params.is_empty()
                        && !ability
                            .params
                            .iter()
                            .any(|ty| tribute::is_type_var(self.db, *ty))
                })
                .collect();
            // Skip if fewer than 2 concrete abilities (no conflict to report yet)
            if concrete.len() < 2 {
                return;
            }
            let ability_strs: Vec<String> = concrete
                .iter()
                .map(|a| {
                    if a.params.is_empty() {
                        a.name.to_string()
                    } else {
                        let params: Vec<String> =
                            a.params.iter().map(|ty| format!("{:?}", ty)).collect();
                        format!("{}({})", a.name, params.join(", "))
                    }
                })
                .collect();

            Diagnostic {
                message: format!(
                    "conflicting ability parameterizations for `{}`: {} are all in scope. \
                     Handler patterns match abilities by name only, so mixing different \
                     parameterizations is not allowed.",
                    name,
                    ability_strs.join(" and ")
                ),
                span,
                severity: DiagnosticSeverity::Error,
                phase: CompilationPhase::TypeChecking,
            }
            .accumulate(self.db);
        }
    }

    /// Check a module.
    pub fn check_module(&mut self, module: &core::Module<'db>) {
        self.seed_var_counters(module);

        // Collect all function types, ability operation types, and type definitions before checking
        // This enables proper type inference for generic function calls, handler patterns,
        // and user-defined type references (enum/struct)
        let body = module.body(self.db);
        if let Some(block) = body.blocks(self.db).first() {
            self.collect_function_types_from_block(block);
            self.collect_ability_op_types_from_block(block);
            self.collect_type_definitions_from_block(block);
        }

        self.check_region(&body);
    }

    /// Collect function types from a block.
    fn collect_function_types_from_block(&mut self, block: &Block<'db>) {
        for op in block.operations(self.db).iter() {
            if let Ok(func_op) = func::Func::from_operation(self.db, *op) {
                let name = func_op.name(self.db);
                let func_type = func_op.r#type(self.db);
                Arc::make_mut(&mut self.function_types).insert(name, func_type);
            }
        }
    }

    /// Collect ability operation types from a block.
    ///
    /// Walks `tribute.ability_def` operations and extracts operation signatures
    /// from their `operations` regions. This builds a map from (ability_name, op_name)
    /// to `AbilityOpSignature` for use in handler pattern type checking.
    fn collect_ability_op_types_from_block(&mut self, block: &Block<'db>) {
        for op in block.operations(self.db).iter() {
            // Check for tribute.ability_def
            if let Ok(ability_def) = tribute::AbilityDef::from_operation(self.db, *op) {
                let ability_name = ability_def.sym_name(self.db);
                let operations_region = ability_def.operations(self.db);

                // Walk the operations region to find tribute.op_def
                for inner_block in operations_region.blocks(self.db).iter() {
                    for inner_op in inner_block.operations(self.db).iter() {
                        if let Ok(op_def) = tribute::OpDef::from_operation(self.db, *inner_op) {
                            let op_name = op_def.sym_name(self.db);
                            let op_type = op_def.r#type(self.db);

                            // Extract params and return_ty from the function type
                            if let Some(func_ty) = core::Func::from_type(self.db, op_type) {
                                let params = func_ty.params(self.db);
                                let return_ty = func_ty.result(self.db);

                                let signature = AbilityOpSignature {
                                    ability_name,
                                    op_name,
                                    params: params.clone(),
                                    return_ty,
                                };

                                trace!(
                                    "collect_ability_op_types: {:?}::{:?} -> params={:?}, return={:?}",
                                    ability_name, op_name, params, return_ty
                                );

                                let key = (ability_name, op_name);
                                Arc::make_mut(&mut self.ability_op_types).insert(key, signature);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Collect type definitions (struct and enum) from a block.
    ///
    /// Walks `tribute.struct_def` and `tribute.enum_def` operations and builds
    /// a map from type names to their ADT types for use in resolving user-defined
    /// type references during type checking.
    fn collect_type_definitions_from_block(&mut self, block: &Block<'db>) {
        let collected = collect_type_defs(self.db, block);
        Arc::make_mut(&mut self.type_defs).extend(collected.iter().map(|(k, v)| (*k, *v)));
    }

    /// Look up an ability operation's signature.
    ///
    /// Returns `None` if the operation is not found or if type information is incomplete.
    pub fn lookup_ability_op(
        &self,
        ability_name: Symbol,
        op_name: Symbol,
    ) -> Option<&AbilityOpSignature<'db>> {
        self.ability_op_types.get(&(ability_name, op_name))
    }

    fn seed_var_counters(&mut self, module: &core::Module<'db>) {
        let body = module.body(self.db);
        let mut max_type_var_id: Option<u64> = None;
        let mut max_row_var_id: Option<u64> = None;
        self.collect_vars_in_region(&body, &mut max_type_var_id, &mut max_row_var_id);

        if let Some(id) = max_type_var_id {
            self.next_type_var = self.next_type_var.max(id + 1);
        }
        if let Some(id) = max_row_var_id {
            self.next_row_var = self.next_row_var.max(id + 1);
        }
    }

    fn collect_vars_in_region(
        &self,
        region: &Region<'db>,
        max_type_var_id: &mut Option<u64>,
        max_row_var_id: &mut Option<u64>,
    ) {
        for block in region.blocks(self.db).iter() {
            for arg in block.args(self.db).iter() {
                self.collect_vars_in_type(arg.ty(self.db), max_type_var_id, max_row_var_id);
            }
            for op in block.operations(self.db).iter() {
                self.collect_vars_in_operation(op, max_type_var_id, max_row_var_id);
            }
        }
    }

    fn collect_vars_in_operation(
        &self,
        op: &Operation<'db>,
        max_type_var_id: &mut Option<u64>,
        max_row_var_id: &mut Option<u64>,
    ) {
        for &ty in op.results(self.db).iter() {
            self.collect_vars_in_type(ty, max_type_var_id, max_row_var_id);
        }
        for attr in op.attributes(self.db).values() {
            self.collect_vars_in_attr(attr, max_type_var_id, max_row_var_id);
        }
        for region in op.regions(self.db).iter() {
            self.collect_vars_in_region(region, max_type_var_id, max_row_var_id);
        }
    }

    fn collect_vars_in_type(
        &self,
        ty: Type<'db>,
        max_type_var_id: &mut Option<u64>,
        max_row_var_id: &mut Option<u64>,
    ) {
        // Check for type variable
        if tribute::is_type_var(self.db, ty)
            && let Some(Attribute::IntBits(id)) = ty.get_attr(self.db, Symbol::new("id"))
        {
            *max_type_var_id = Some(max_type_var_id.map_or(*id, |current| current.max(*id)));
        }

        // Check for effect row with tail variable
        if let Some(effect_row) = core::EffectRowType::from_type(self.db, ty)
            && let Some(tail) = effect_row.tail_var(self.db)
        {
            *max_row_var_id = Some(max_row_var_id.map_or(tail, |current| current.max(tail)));
        }

        for &param in ty.params(self.db).iter() {
            self.collect_vars_in_type(param, max_type_var_id, max_row_var_id);
        }
        for attr in ty.attrs(self.db).values() {
            self.collect_vars_in_attr(attr, max_type_var_id, max_row_var_id);
        }
    }

    fn collect_vars_in_attr(
        &self,
        attr: &Attribute<'db>,
        max_type_var_id: &mut Option<u64>,
        max_row_var_id: &mut Option<u64>,
    ) {
        match attr {
            Attribute::Type(ty) => self.collect_vars_in_type(*ty, max_type_var_id, max_row_var_id),
            Attribute::List(items) => {
                for item in items {
                    self.collect_vars_in_attr(item, max_type_var_id, max_row_var_id);
                }
            }
            _ => {}
        }
    }

    /// Check a region (sequence of blocks).
    pub fn check_region(&mut self, region: &Region<'db>) {
        for block in region.blocks(self.db).iter() {
            // Record block argument types (e.g., function parameters)
            let num_args = block.args(self.db).len();
            if num_args > 0 {
                trace!(num_args, "recording block arguments");
            }
            for (i, arg) in block.args(self.db).iter().enumerate() {
                let arg_value = block.arg(self.db, i);
                let arg_ty = arg.ty(self.db);
                trace!(?arg_value, ?arg_ty, "recording block arg type");
                self.record_type(arg_value, arg_ty);
            }

            for op in block.operations(self.db).iter() {
                self.check_operation(op);
            }
        }
    }

    /// Check a single operation.
    ///
    /// Uses cached symbols for efficient O(1) dispatch via symbol ID comparison.
    pub fn check_operation(&mut self, op: &Operation<'db>) {
        let dialect = op.dialect(self.db);
        let name = op.name(self.db);
        trace!(%dialect, %name, "checking operation");

        // Dispatch by dialect first, then by operation name
        // Use cached static symbols to avoid interner write locks
        if dialect == func::DIALECT_NAME() {
            if name == func::FUNC() {
                self.check_func_def(op);
            } else if name == func::RETURN() {
                self.check_return(op);
            } else if name == func::CALL() {
                self.check_func_call(op);
            } else if name == func::CALL_INDIRECT() {
                self.check_func_call_indirect(op);
            } else if name == func::CONSTANT() {
                self.check_func_constant(op);
            } else {
                self.check_unknown_op(op);
            }
        } else if dialect == arith::DIALECT_NAME() {
            if name == arith::CONST() {
                self.check_arith_const(op);
            } else if name == arith::ADD()
                || name == arith::SUB()
                || name == arith::MUL()
                || name == arith::DIV()
            {
                self.check_arith_binop(op);
            } else if name == arith::NEG() {
                self.check_arith_neg(op);
            } else if name == arith::CMP_EQ()
                || name == arith::CMP_NE()
                || name == arith::CMP_LT()
                || name == arith::CMP_LE()
                || name == arith::CMP_GT()
                || name == arith::CMP_GE()
            {
                self.check_arith_cmp(op);
            } else {
                self.check_unknown_op(op);
            }
        } else if dialect == tribute::DIALECT_NAME() {
            if name == tribute::TUPLE() {
                self.check_src_tuple(op);
            } else if name == tribute::HANDLE() {
                self.check_ability_prompt(op);
            } else if name == tribute::STRUCT_DEF()
                || name == tribute::ENUM_DEF()
                || name == tribute::ABILITY_DEF()
            {
                // Type declarations don't need type checking - no-op
            } else {
                self.check_unknown_op(op);
            }
        } else if dialect == adt::DIALECT_NAME() {
            if name == adt::STRING_CONST() {
                self.check_string_const(op);
            } else if name == adt::BYTES_CONST() {
                self.check_bytes_const(op);
            } else if name == adt::STRUCT_NEW() || name == adt::VARIANT_NEW() {
                // For struct/variant construction, the result type is already set correctly
                // to the struct/enum type. Just record this type for the result value.
                self.check_adt_new(op);
            } else if name == adt::VARIANT_GET() {
                self.check_variant_get(op);
            } else if name == adt::STRUCT_GET() {
                self.check_struct_get(op);
            } else {
                self.check_unknown_op(op);
            }
        } else if dialect == ability::DIALECT_NAME() {
            if name == ability::PERFORM() {
                self.check_ability_perform(op);
            } else if name == ability::RESUME() {
                self.check_ability_resume(op);
            } else if name == ability::ABORT() {
                self.check_ability_abort(op);
            } else {
                self.check_unknown_op(op);
            }
        } else if dialect == core::DIALECT_NAME() {
            if name == core::MODULE() {
                // Module is checked via check_module
            } else if name == core::UNREALIZED_CONVERSION_CAST() {
                // Pass through - assign fresh type var to result
                self.check_unknown_op(op);
            } else {
                self.check_unknown_op(op);
            }
        } else {
            self.check_unknown_op(op);
        }
    }

    /// Handle unknown operations by assigning fresh type vars to results.
    fn check_unknown_op(&mut self, op: &Operation<'db>) {
        let results = op.results(self.db);
        for (i, _result_ty) in results.iter().enumerate() {
            let value = op.result(self.db, i);
            let ty = self.fresh_type_var();
            self.record_type(value, ty);
        }
    }

    /// Handle adt.struct_new and adt.variant_new operations.
    ///
    /// The result type is already set to the struct/enum type from the resolve pass.
    /// We just need to record this type for the result value so type inference
    /// can use it when checking method calls.
    fn check_adt_new(&mut self, op: &Operation<'db>) {
        let results = op.results(self.db);
        if let Some(&result_ty) = results.first() {
            let value = op.result(self.db, 0);
            self.record_type(value, result_ty);
        }
    }

    // === func dialect checking ===

    fn check_func_def(&mut self, op: &Operation<'db>) {
        // Get function type from the "type" attribute (not from results)
        let func_type = func::Func::from_operation(self.db, *op)
            .ok()
            .map(|f| f.r#type(self.db));
        tracing::debug!(?func_type, "check_func_def: entered");

        // Save current entry block arg types (for nested functions)
        let saved_entry_args = std::mem::take(&mut self.entry_block_arg_types);

        // Extract the return type from the function signature for later unification
        let return_type = func_type.and_then(|ft| {
            let result = core::Func::from_type(self.db, ft).map(|f| f.result(self.db));
            tracing::debug!(?ft, ?result, "check_func_def: extracting return type");
            result
        });

        // Check the body
        let regions = op.regions(self.db);
        if let Some(body) = regions.first() {
            // Set up entry block argument types from function signature or entry block
            if let Some(entry_block) = body.blocks(self.db).first() {
                // Get parameter types from function type if available
                if let Some(func_ty_value) = func_type
                    && let Some(func_ty) = core::Func::from_type(self.db, func_ty_value)
                {
                    self.entry_block_arg_types = func_ty.params(self.db).iter().copied().collect();
                } else {
                    // Fallback to entry block's declared arg types
                    self.entry_block_arg_types = entry_block.arg_types(self.db).to_vec();
                }

                // Record types for the actual entry block arguments (for direct lookups)
                for (i, arg) in entry_block.args(self.db).iter().enumerate() {
                    let arg_value = entry_block.arg(self.db, i);
                    self.record_type(arg_value, arg.ty(self.db));
                }
            }

            self.check_region(body);

            // Unify the body's yield type with the function's declared return type
            if let Some(ret_ty) = return_type {
                let yield_ty = self.get_region_yield_type(body);
                tracing::debug!(
                    ?ret_ty,
                    ?yield_ty,
                    "check_func_def: trying to unify body yield with return type"
                );
                if let Some(yield_ty) = yield_ty {
                    self.constraints.add_type_eq(ret_ty, yield_ty);
                }
            }
        }

        // Restore entry block arg types
        self.entry_block_arg_types = saved_entry_args;

        // Note: func.func is a declaration without result values (unlike func.call),
        // so we don't record a result type here.
    }

    fn check_return(&mut self, _op: &Operation<'db>) {
        // Return doesn't have a result value
        // The operand should be checked against the function return type
        // (This requires function context tracking, simplified for now)
    }

    fn check_func_call(&mut self, op: &Operation<'db>) {
        // func.call: direct call to a function symbol
        // The effect of the call is the effect declared in the function type
        let mut effect_handled = false;
        let operands = op.operands(self.db);
        let results = op.results(self.db);
        let result_type = results
            .first()
            .copied()
            .unwrap_or_else(|| self.fresh_type_var());

        // Try to get callee info from the func.call operation
        if let Ok(call_op) = func::Call::from_operation(self.db, *op) {
            let callee_name = call_op.callee(self.db);
            trace!(
                "check_func_call: callee={:?}, function_types_count={}",
                callee_name,
                self.function_types.len()
            );

            // Look up the callee's function type
            if let Some(&callee_type) = self.function_types.get(&callee_name) {
                trace!("check_func_call: found callee type");
                if let Some(func_type) = core::Func::from_type(self.db, callee_type) {
                    let args: Vec<_> = operands.iter().copied().collect();
                    effect_handled = self.constrain_call_types(&func_type, &args, result_type);
                }
            }
        }

        // Record the result type
        let value = op.result(self.db, 0);
        self.record_type(value, result_type);

        // Default effect if not handled above
        if !effect_handled {
            let call_effect = EffectRow::var(self.fresh_row_var());
            self.merge_effect(call_effect);
        }
    }

    /// Instantiate a function type by replacing type variables with fresh ones.
    ///
    /// This is needed for generic function calls: `identity(42)` where `identity: (a) -> a`
    /// should create fresh type var `?0` and return params `[?0]` and result `?0`.
    fn instantiate_function_type(
        &mut self,
        func_type: &core::Func<'db>,
    ) -> (Vec<Type<'db>>, Type<'db>) {
        let params = func_type.params(self.db);
        let result = func_type.result(self.db);

        // Mapping from original type var ids to fresh type vars
        let mut var_mapping: HashMap<u64, Type<'db>> = HashMap::new();

        // Instantiate parameters
        let instantiated_params: Vec<Type<'db>> = params
            .iter()
            .map(|&ty| self.instantiate_type(ty, &mut var_mapping))
            .collect();

        // Instantiate result
        let instantiated_result = self.instantiate_type(result, &mut var_mapping);

        (instantiated_params, instantiated_result)
    }

    /// Common logic for constraining call types (used by both direct and indirect calls).
    ///
    /// Instantiates generic parameters, constrains argument and result types,
    /// and propagates effects. Returns true if the function's effect was handled.
    fn constrain_call_types(
        &mut self,
        func_type: &core::Func<'db>,
        args: &[Value<'db>],
        result_type: Type<'db>,
    ) -> bool {
        // Instantiate fresh type variables for generic parameters
        let (instantiated_params, instantiated_result) = self.instantiate_function_type(func_type);

        // Constrain result type with instantiated return type
        self.constrain_eq(result_type, instantiated_result);

        // Constrain argument types with instantiated param types
        for (i, &param_ty) in instantiated_params.iter().enumerate() {
            if let Some(&arg) = args.get(i)
                && let Some(arg_ty) = self.get_type(arg)
            {
                self.constrain_eq(arg_ty, param_ty);
            }
        }

        // Propagate the function's effect
        if let Some(effect_ty) = func_type.effect(self.db)
            && let Some(effect_row) = EffectRow::from_type(self.db, effect_ty)
        {
            self.merge_effect(effect_row);
            true
        } else {
            false
        }
    }

    /// Instantiate a type by replacing type variables with fresh ones from the mapping.
    fn instantiate_type(
        &mut self,
        ty: Type<'db>,
        var_mapping: &mut HashMap<u64, Type<'db>>,
    ) -> Type<'db> {
        if tribute::is_type_var(self.db, ty) {
            // Extract the var id and map it
            if let Some(Attribute::IntBits(var_id)) = ty.attrs(self.db).get(&Symbol::new("id")) {
                *var_mapping
                    .entry(*var_id)
                    .or_insert_with(|| self.fresh_type_var())
            } else {
                // No id attribute, create a new fresh var
                self.fresh_type_var()
            }
        } else if let Some(func_ty) = core::Func::from_type(self.db, ty) {
            // Recursively instantiate function types
            let new_params: IdVec<Type<'db>> = func_ty
                .params(self.db)
                .iter()
                .map(|&t| self.instantiate_type(t, var_mapping))
                .collect();
            let new_result = self.instantiate_type(func_ty.result(self.db), var_mapping);

            core::Func::with_effect(self.db, new_params, new_result, func_ty.effect(self.db))
                .as_type()
        } else {
            // Recursively instantiate type parameters for composite types (e.g., List<a>)
            let params = ty.params(self.db);
            if params.is_empty() {
                ty
            } else {
                let new_params: IdVec<Type<'db>> = params
                    .iter()
                    .map(|&t| self.instantiate_type(t, var_mapping))
                    .collect();
                Type::new(
                    self.db,
                    ty.dialect(self.db),
                    ty.name(self.db),
                    new_params,
                    ty.attrs(self.db).clone(),
                )
            }
        }
    }

    fn check_func_call_indirect(&mut self, op: &Operation<'db>) {
        // func.call_indirect: indirect call via function value
        // The callee is the first operand (function value)
        let mut effect_handled = false;
        let operands = op.operands(self.db);
        let results = op.results(self.db);
        let result_type = results
            .first()
            .copied()
            .unwrap_or_else(|| self.fresh_type_var());

        // Get the callee type and extract its effect
        if let Some(&callee) = operands.first() {
            let callee_type = self.get_type(callee);
            trace!(
                "check_func_call_indirect: callee={:?}, callee_type={:?}",
                callee,
                callee_type.map(|t| format!("{}.{}", t.dialect(self.db), t.name(self.db)))
            );

            if let Some(callee_type) = callee_type {
                // Try to extract function type from various callee types:
                // 1. Direct function type (core.func)
                // 2. Closure type (closure.closure<func_type>)
                // 3. Continuation type (cont.continuation<arg, result, effect>)
                let func_type_opt = if let Some(func_type) =
                    core::Func::from_type(self.db, callee_type)
                {
                    Some(func_type)
                } else if let Some(closure_type) = closure::Closure::from_type(self.db, callee_type)
                {
                    // Extract the wrapped function type from the closure
                    core::Func::from_type(self.db, closure_type.func_type(self.db))
                } else if let Some(cont_type) = cont::Continuation::from_type(self.db, callee_type)
                {
                    // For continuations, create a synthetic function type:
                    // (arg_ty) -> result_ty
                    // arg_ty is what we pass to resume, result_ty is what resume returns
                    let arg_ty = cont_type.arg(self.db);
                    let return_ty = cont_type.result(self.db);
                    let effect_ty = cont_type.effect(self.db);
                    Some(core::Func::with_effect(
                        self.db,
                        IdVec::from(vec![arg_ty]),
                        return_ty,
                        Some(effect_ty),
                    ))
                } else {
                    None
                };

                if let Some(func_type) = func_type_opt {
                    // Arguments start after the callee (index 0)
                    let args: Vec<_> = operands.iter().skip(1).copied().collect();
                    effect_handled = self.constrain_call_types(&func_type, &args, result_type);
                }
            }
        }

        let value = op.result(self.db, 0);
        self.record_type(value, result_type);

        // Default effect if not handled above
        if !effect_handled {
            let call_effect = EffectRow::var(self.fresh_row_var());
            self.merge_effect(call_effect);
        }
    }

    fn check_func_constant(&mut self, op: &Operation<'db>) {
        // func.constant: creates a function value from a symbol reference
        // The result type should be a function type
        let results = op.results(self.db);
        let result_type = results
            .first()
            .copied()
            .unwrap_or_else(|| self.fresh_type_var());

        let value = op.result(self.db, 0);
        self.record_type(value, result_type);
    }

    // === arith dialect checking ===

    fn check_arith_const(&mut self, op: &Operation<'db>) {
        let results = op.results(self.db);
        if let Some(&result_type) = results.first() {
            let value = op.result(self.db, 0);
            self.record_type(value, result_type);
        }
    }

    fn check_arith_binop(&mut self, op: &Operation<'db>) {
        let operands = op.operands(self.db);
        let results = op.results(self.db);
        let result_type = results
            .first()
            .copied()
            .unwrap_or_else(|| self.fresh_type_var());

        // Both operands should have the same type as the result
        for (i, operand) in operands.iter().enumerate() {
            if let Some(op_type) = self.get_type(*operand) {
                trace!(
                    operand_index = i,
                    ?op_type,
                    ?result_type,
                    "constraining arith binop operand"
                );
                self.constrain_eq(op_type, result_type);
            } else {
                trace!(
                    operand_index = i,
                    ?operand,
                    entry_block_arg_count = self.entry_block_arg_types.len(),
                    "operand type not found - fallback may have failed"
                );
            }
        }

        let value = op.result(self.db, 0);
        self.record_type(value, result_type);
    }

    fn check_arith_neg(&mut self, op: &Operation<'db>) {
        let results = op.results(self.db);
        let result_type = results
            .first()
            .copied()
            .unwrap_or_else(|| self.fresh_type_var());

        if let Some(operand) = op.operands(self.db).first()
            && let Some(op_type) = self.get_type(*operand)
        {
            self.constrain_eq(op_type, result_type);
        }

        let value = op.result(self.db, 0);
        self.record_type(value, result_type);
    }

    fn check_arith_cmp(&mut self, op: &Operation<'db>) {
        // Comparison returns I1 (bool)
        let bool_type = core::I1::new(self.db);
        let value = op.result(self.db, 0);
        self.record_type(value, *bool_type);
    }

    /// Get the type of the value yielded by a region.
    ///
    /// This looks for a terminator operation (`tribute.yield` or `func.return`)
    /// in the last block of the region and returns the type of its operand.
    fn get_region_yield_type(&self, region: &Region<'db>) -> Option<Type<'db>> {
        let blocks = region.blocks(self.db);
        let last_block = blocks.last()?;
        let ops = last_block.operations(self.db);

        // Find func.return in the block
        for op in ops.iter().rev() {
            let dialect = op.dialect(self.db);
            let name = op.name(self.db);

            // Check for func.return
            if dialect == func::DIALECT_NAME() && name == func::RETURN() {
                let operands = op.operands(self.db);
                if let Some(value) = operands.first() {
                    let ty = self.get_type(*value);
                    tracing::debug!(?value, ?ty, "get_region_yield_type: found return operand");
                    return ty;
                }
                // func.return with no operand means Nil return
                return Some(core::Nil::new(self.db).as_type());
            }
        }
        None
    }

    fn check_src_tuple(&mut self, op: &Operation<'db>) {
        let results = op.results(self.db);
        let result_type = results
            .first()
            .copied()
            .unwrap_or_else(|| self.fresh_type_var());
        let value = op.result(self.db, 0);
        self.record_type(value, result_type);
    }

    // === adt dialect checking ===

    fn check_string_const(&mut self, op: &Operation<'db>) {
        let string_type = core::String::new(self.db);
        let value = op.result(self.db, 0);
        self.record_type(value, *string_type);
    }

    fn check_bytes_const(&mut self, op: &Operation<'db>) {
        let bytes_type = core::Bytes::new(self.db);
        let value = op.result(self.db, 0);
        self.record_type(value, *bytes_type);
    }

    /// Check `adt.variant_get` operation.
    ///
    /// This operation extracts a field from a variant. The operand can be:
    /// 1. A variant instance type (from variant_cast) - field types are directly available
    /// 2. An enum type (raw scrutinee) - we look up field types from enum definition
    ///
    /// The `field` attribute specifies which field to extract (by index or name).
    fn check_variant_get(&mut self, op: &Operation<'db>) {
        let results = op.results(self.db);
        let result_type = results
            .first()
            .copied()
            .unwrap_or_else(|| self.fresh_type_var());

        // Get the operand (variant reference) type
        let operands = op.operands(self.db);
        if let Some(&ref_operand) = operands.first()
            && let Some(ref_type) = self.get_type(ref_operand)
            && let Some(field_type) = self.get_variant_field_type(op, ref_type)
        {
            trace!(
                ?ref_type,
                ?field_type,
                ?result_type,
                "check_variant_get: constraining result type"
            );
            self.constrain_eq(result_type, field_type);
        }

        let value = op.result(self.db, 0);
        self.record_type(value, result_type);
    }

    /// Get the field type from a variant_get operation's operand type.
    ///
    /// Handles both variant instance types and enum types.
    fn get_variant_field_type(
        &self,
        op: &Operation<'db>,
        operand_type: Type<'db>,
    ) -> Option<Type<'db>> {
        // Get field index from the operation's `field` attribute
        let field_attr = op.attributes(self.db).get(&Symbol::new("field"))?;
        let field_index = match field_attr {
            Attribute::IntBits(idx) => *idx as usize,
            _ => return None,
        };

        // Case 1: Variant instance type (has variant_fields attribute)
        if let Some(field_types) = adt::get_variant_field_types(self.db, operand_type) {
            return field_types.get(field_index).copied();
        }

        // Case 2: Enum type - we need to find which variant this is for
        // This happens in pattern matching where the scrutinee is still the enum type.
        // We can't determine the field type here without additional context.
        // The type will be constrained when we check the arm in check_case_arm_with_variant.
        None
    }

    /// Check `adt.struct_get` operation.
    ///
    /// This operation extracts a field from a struct. The `field` attribute
    /// specifies which field (by index or name), and `type` attribute provides
    /// the struct type.
    fn check_struct_get(&mut self, op: &Operation<'db>) {
        let results = op.results(self.db);
        let result_type = results
            .first()
            .copied()
            .unwrap_or_else(|| self.fresh_type_var());

        // Try to get field type from struct type attribute
        if let Some(Attribute::Type(struct_type)) = op.attributes(self.db).get(&Symbol::new("type"))
            && let Some(field_type) = self.get_struct_field_type(op, *struct_type)
        {
            trace!(
                ?struct_type,
                ?field_type,
                ?result_type,
                "check_struct_get: constraining result type"
            );
            self.constrain_eq(result_type, field_type);
        }

        let value = op.result(self.db, 0);
        self.record_type(value, result_type);
    }

    /// Get the field type from a struct_get operation.
    fn get_struct_field_type(
        &self,
        op: &Operation<'db>,
        struct_type: Type<'db>,
    ) -> Option<Type<'db>> {
        let field_attr = op.attributes(self.db).get(&Symbol::new("field"))?;

        // Get struct fields
        let fields = adt::get_struct_fields(self.db, struct_type)?;

        match field_attr {
            // Field by index (u64)
            Attribute::IntBits(idx) => fields.get(*idx as usize).map(|(_, ty)| *ty),
            _ => None,
        }
    }

    /// Check handler patterns and bind continuation types.
    ///
    /// For each `pat.handler_suspend` pattern:
    /// 1. Extract the handled ability and add to the list
    /// 2. Look up the operation's return type from ability definition
    /// 3. Create continuation type: `fn(return_ty) ->{remaining_effects} handler_result_ty`
    /// 4. Bind the continuation variable (if present) to this type
    ///
    /// This ensures that continuation variables like `k` in `{ State::get() -> k }`
    /// have the correct function type for resuming the computation.
    fn check_handler_pattern_continuations(
        &mut self,
        pattern_region: &Region<'db>,
        handled: &mut Vec<AbilityRef<'db>>,
        handler_result_ty: Type<'db>,
    ) {
        let ability_ref_sym = Symbol::new("ability_ref");
        let op_sym = Symbol::new("op");

        for block in pattern_region.blocks(self.db).iter() {
            for op in block.operations(self.db).iter() {
                tracing::debug!(
                    "check_handler_pattern_continuations: op = {}.{}",
                    op.dialect(self.db),
                    op.name(self.db)
                );

                // Check for pat.handler_done - bind result variable type
                if op.dialect(self.db) == tribute_pat::DIALECT_NAME()
                    && op.name(self.db) == tribute_pat::HANDLER_DONE()
                {
                    // handler_done has a result region containing the pattern for the result value
                    // The result variable should have the same type as the handler's body result.
                    // handler_result_ty is already a fresh type var created for this handler's result.
                    let result_regions = op.regions(self.db);
                    if let Some(result_region) = result_regions.first() {
                        // Look for tribute_pat.bind in the result pattern
                        for result_block in result_region.blocks(self.db).iter() {
                            for pat_op in result_block.operations(self.db).iter() {
                                if pat_op.dialect(self.db) == tribute_pat::DIALECT_NAME()
                                    && pat_op.name(self.db) == tribute_pat::BIND()
                                {
                                    // Get the bind operation's result (the bound variable)
                                    // and set its type to handler_result_ty (the handler's result type var)
                                    let bound_value = pat_op.result(self.db, 0);
                                    // Use handler_result_ty as the type for the done result
                                    // The body yield will unify with this
                                    self.record_type(bound_value, handler_result_ty);
                                    tracing::debug!(
                                        "check_handler_pattern_continuations: bound done result to {:?}",
                                        handler_result_ty
                                    );
                                }
                            }
                        }
                    }
                }
                // Check for pat.handler_suspend
                else if op.dialect(self.db) == tribute_pat::DIALECT_NAME()
                    && op.name(self.db) == tribute_pat::HANDLER_SUSPEND()
                {
                    let attrs = op.attributes(self.db);

                    // Extract ability reference
                    if let Some(Attribute::Type(ability_ty)) = attrs.get(&ability_ref_sym)
                        && let Some(pattern_ability) = AbilityRef::from_type(self.db, *ability_ty)
                    {
                        // Find the fully parameterized ability from the effect row
                        let matching_abilities = if !pattern_ability.params.is_empty() {
                            vec![pattern_ability.clone()]
                        } else {
                            self.current_effect.find_by_name(pattern_ability.name)
                        };

                        // Add to handled list
                        for ability in &matching_abilities {
                            if !handled.contains(ability) {
                                handled.push(ability.clone());
                            }
                        }

                        // Get operation name for looking up signature
                        let op_name = attrs.get(&op_sym).and_then(|a| {
                            if let Attribute::Symbol(s) = a {
                                Some(*s)
                            } else {
                                None
                            }
                        });

                        tracing::debug!(
                            "check_handler_pattern_continuations: matching_abilities = {:?}, op_name = {:?}",
                            matching_abilities
                                .iter()
                                .map(|a| a.name)
                                .collect::<Vec<_>>(),
                            op_name
                        );

                        // Look up operation signature and bind continuation type
                        if let Some(op_name) = op_name {
                            // Find the first matching ability to get the operation signature
                            for ability in &matching_abilities {
                                tracing::debug!(
                                    "check_handler_pattern_continuations: looking up {}.{}",
                                    ability.name,
                                    op_name
                                );
                                if let Some(sig) = self.lookup_ability_op(ability.name, op_name) {
                                    // Create continuation type:
                                    // cont.continuation<arg=op_return_ty, result=handler_result_ty, effect=remaining>
                                    let remaining_effect = self.current_effect.to_type(self.db);
                                    let continuation_ty = cont::Continuation::new(
                                        self.db,
                                        sig.return_ty, // arg_ty: value passed to resume
                                        handler_result_ty, // result_ty: what resume returns
                                        remaining_effect,
                                    )
                                    .as_type();

                                    // Find and bind the continuation variable
                                    self.bind_continuation_in_pattern(op, continuation_ty);

                                    trace!(
                                        "check_handler_pattern_continuations: {:?}::{:?} -> cont_ty={:?}",
                                        ability.name, op_name, continuation_ty
                                    );
                                    break; // Only need to bind once
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Bind the continuation variable in a handler_suspend pattern to a type.
    ///
    /// This reads the `continuation_type` attribute from handler_suspend (a type variable
    /// created by tirgen) and constrains it to the computed continuation type.
    /// The TypeSubst will later resolve this type variable in the attribute.
    fn bind_continuation_in_pattern(
        &mut self,
        handler_suspend_op: &Operation<'db>,
        continuation_ty: Type<'db>,
    ) {
        use tribute_pat::handler_suspend_attrs::CONTINUATION_TYPE;

        // Check if handler_suspend has a continuation_type attribute with a type variable
        if let Some(Attribute::Type(attr_type_var)) = handler_suspend_op
            .attributes(self.db)
            .get(&CONTINUATION_TYPE())
        {
            // Constrain the type variable to the computed continuation type
            // TypeSubst will resolve this during type substitution
            self.constrain_eq(*attr_type_var, continuation_ty);

            trace!(
                "bind_continuation_in_pattern: constrained {:?} = {:?}",
                attr_type_var, continuation_ty
            );
        } else {
            // Fallback for IR without the continuation_type attribute
            // (e.g., from older code or tests)
            trace!("bind_continuation_in_pattern: no continuation_type attribute found, skipping");
        }
    }

    // === ability dialect checking ===

    fn check_ability_perform(&mut self, op: &Operation<'db>) {
        // ability.perform: performs an ability operation
        // This adds the ability to the current effect row

        let results = op.results(self.db);
        let result_type = results
            .first()
            .copied()
            .unwrap_or_else(|| self.fresh_type_var());

        // Get the ability reference from attributes
        // attr: ability_ref: Type (core.ability_ref) - supports parameterized abilities
        // attr: op: Symbol (operation name)
        if let Some(Attribute::Type(ability_ty)) =
            op.attributes(self.db).get(&Symbol::new("ability_ref"))
        {
            // Convert the Type to AbilityRef.
            // Note: The resolve phase guarantees that ability_ref attributes contain valid
            // core.ability_ref types. Conversion failures indicate corrupted IR rather than
            // semantic errors, so we log and skip rather than reporting a user-facing error.
            if let Some(ability) = AbilityRef::from_type(self.db, *ability_ty) {
                // Create an effect row with this ability and merge it
                let effect = EffectRow::concrete([ability]);
                self.merge_effect(effect);

                // Check for conflicting abilities (same name with different type parameters)
                self.check_ability_conflicts(op.location(self.db).span);
            } else {
                trace!(
                    ?ability_ty,
                    "failed to convert ability_ref Type to AbilityRef"
                );
            }
        }

        let value = op.result(self.db, 0);
        self.record_type(value, result_type);
    }

    fn check_ability_prompt(&mut self, op: &Operation<'db>) {
        // tribute.handle: runs body in a delimited context with handler arms
        //
        // The body's effects are captured by the prompt. Handler arms
        // (tribute.arm with tribute_pat.handler_suspend patterns) bind continuation
        // variables that need proper typing.

        let results = op.results(self.db);
        let result_type = results
            .first()
            .copied()
            .unwrap_or_else(|| self.fresh_type_var());

        // Save current effect
        let outer_effect = std::mem::take(&mut self.current_effect);

        // Check the body region
        let regions = op.regions(self.db);
        if let Some(body) = regions.first() {
            self.check_region(body);
        }

        // The body's effects are "captured" by the prompt.
        // We propagate body's effects to outer context first.
        let body_effect = std::mem::replace(&mut self.current_effect, outer_effect);
        self.merge_effect(body_effect);

        // Check for conflicting abilities after merging body effects
        self.check_ability_conflicts(op.location(self.db).span);

        // Process handler arms (second region) to bind continuation types
        let mut handled_abilities = Vec::new();
        if let Some(arms) = regions.get(1) {
            tracing::debug!("check_ability_prompt: processing arms region");
            for block in arms.blocks(self.db).iter() {
                for arm_op in block.operations(self.db).iter() {
                    tracing::debug!(
                        "check_ability_prompt: arm op = {}.{}",
                        arm_op.dialect(self.db),
                        arm_op.name(self.db)
                    );
                    if arm_op.dialect(self.db) == tribute::DIALECT_NAME()
                        && arm_op.name(self.db) == tribute::ARM()
                    {
                        // Extract pattern region and check for handler patterns
                        let arm_regions = arm_op.regions(self.db);
                        if let Some(pattern_region) = arm_regions.first() {
                            self.check_handler_pattern_continuations(
                                pattern_region,
                                &mut handled_abilities,
                                result_type,
                            );
                        }

                        // Check the body region and unify its result with handler result
                        if let Some(body_region) = arm_regions.get(1) {
                            self.check_region(body_region);

                            // Find the yield value in the body and unify with result_type
                            if let Some(yield_value_type) = self.get_region_yield_type(body_region)
                            {
                                tracing::debug!(
                                    ?result_type,
                                    ?yield_value_type,
                                    "check_ability_prompt: unifying handler result with arm yield"
                                );
                                self.constraints.add_type_eq(result_type, yield_value_type);
                            } else {
                                tracing::debug!("check_ability_prompt: no yield found in arm body");
                            }
                        }
                    }
                }
            }
        }

        // Remove handled abilities from the current effect row
        for ability in handled_abilities {
            self.current_effect.remove_ability(&ability);
        }

        let value = op.result(self.db, 0);
        self.record_type(value, result_type);
    }

    fn check_ability_resume(&mut self, op: &Operation<'db>) {
        // ability.resume: resumes a captured continuation
        // The result type depends on what the continuation expects

        let results = op.results(self.db);
        let result_type = results
            .first()
            .copied()
            .unwrap_or_else(|| self.fresh_type_var());

        // The continuation operand should have a continuation type
        // For now, just record the result type
        let value = op.result(self.db, 0);
        self.record_type(value, result_type);
    }

    fn check_ability_abort(&mut self, _op: &Operation<'db>) {
        // ability.abort: discards a continuation without resuming
        // No result value, just consumes the continuation
    }

    /// Solve all accumulated constraints and return the solved substitution.
    pub fn solve(&mut self) -> SolveResult<'db, TypeSolver<'db>> {
        let constraints = self.take_constraints();
        let mut solver = TypeSolver::new(self.db);
        solver.solve(constraints)?;
        Ok(solver)
    }
}

/// Type check a module and solve constraints.
pub fn typecheck_module<'db>(
    db: &'db dyn salsa::Database,
    module: &core::Module<'db>,
) -> SolveResult<'db, TypeSolver<'db>> {
    let mut checker = TypeChecker::new(db);
    checker.check_module(module);
    checker.solve()
}

// =============================================================================
// Per-function type checking (salsa::tracked)
// =============================================================================

/// Result of type checking a single function.
///
/// This is a salsa::tracked struct, enabling incremental caching of
/// per-function type checking results.
#[salsa::tracked]
pub struct FunctionTypeResult<'db> {
    /// The function operation with types resolved (or original if failed).
    pub operation: Operation<'db>,
    /// Whether type checking succeeded.
    pub success: bool,
}

/// Type check a single function operation.
///
/// This is a salsa::tracked function, so results are cached per-function.
/// Each function is type-checked independently with its own constraint set.
///
/// Returns a `FunctionTypeResult` containing the typed operation and success status.
/// Type errors are also reported via salsa accumulators (Diagnostic).
#[salsa::tracked]
pub fn typecheck_function<'db>(
    db: &'db dyn salsa::Database,
    func_op: Operation<'db>,
) -> FunctionTypeResult<'db> {
    let mut checker = TypeChecker::new(db);

    // Check the function definition
    checker.check_func_def(&func_op);

    // Solve constraints for this function
    match checker.solve() {
        Ok(solver) => {
            // Apply substitution to the function's regions
            let subst = solver.type_subst();
            let new_regions: trunk_ir::IdVec<_> = func_op
                .regions(db)
                .iter()
                .map(|r| super::subst::apply_subst_to_region(db, r, subst))
                .collect();

            // Create new operation with resolved types
            let new_results: trunk_ir::IdVec<_> = func_op
                .results(db)
                .iter()
                .map(|ty| subst.apply(db, *ty))
                .collect();

            let new_op = Operation::new(
                db,
                func_op.location(db),
                func_op.dialect(db),
                func_op.name(db),
                func_op.operands(db).clone(),
                new_results,
                func_op.attributes(db).clone(),
                new_regions,
                func_op.successors(db).clone(),
            );

            FunctionTypeResult::new(db, new_op, true)
        }
        Err(_err) => {
            // TODO: Emit type error via accumulator
            // Return the original operation with failure status
            FunctionTypeResult::new(db, func_op, false)
        }
    }
}

/// Validate that a top-level function has explicit type annotations.
///
/// Top-level functions (those directly in the module body) must have
/// explicit type annotations for all parameters and the return type.
/// This is required because type inference scope is per-function.
fn validate_toplevel_function_types<'db>(db: &'db dyn salsa::Database, func_op: &func::Func<'db>) {
    let func_type_attr = func_op.r#type(db);

    // Try to get the function type from the attribute
    let Some(func_ty) = core::Func::from_type(db, func_type_attr) else {
        return;
    };

    let location = func_op.operation().location(db);
    let func_name = func_op.name(db);

    // Check return type
    let result_ty = func_ty.result(db);
    if has_type_vars(db, result_ty) {
        Diagnostic {
            message: format!(
                "top-level function `{}` must have an explicit return type annotation",
                func_name
            ),
            span: location.span,
            severity: DiagnosticSeverity::Error,
            phase: CompilationPhase::TypeChecking,
        }
        .accumulate(db);
    }

    // Check parameter types
    let params = func_ty.params(db);
    for (i, param_ty) in params.iter().enumerate() {
        if has_type_vars(db, *param_ty) {
            Diagnostic {
                message: format!(
                    "parameter {} of top-level function `{}` must have an explicit type annotation",
                    i + 1,
                    func_name
                ),
                span: location.span,
                severity: DiagnosticSeverity::Error,
                phase: CompilationPhase::TypeChecking,
            }
            .accumulate(db);
        }
    }
}

/// Type check a module using per-function approach.
///
/// Each function is type-checked independently, but with access to a pre-built
/// map of all function types. This enables proper type inference for generic
/// function calls across function boundaries.
///
/// This also validates that top-level functions have explicit type annotations.
pub fn typecheck_module_per_function<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
) -> core::Module<'db> {
    use trunk_ir::{Block, IdVec, Region};

    let body = module.body(db);
    let blocks = body.blocks(db);

    if blocks.is_empty() {
        return module;
    }

    let block = &blocks[0];

    // First pass: collect all function types, ability operation types, and type definitions
    let function_types = collect_function_types(db, block);
    let ability_op_types = collect_ability_op_types(db, block);
    let type_defs = collect_type_defs(db, block);

    // Second pass: type check each function with the type maps
    let mut new_ops: IdVec<Operation<'db>> = IdVec::new();
    for op in block.operations(db).iter() {
        if op.dialect(db) == func::DIALECT_NAME() && op.name(db) == func::FUNC() {
            // Validate type annotations for top-level functions
            if let Ok(func_op) = func::Func::from_operation(db, *op) {
                validate_toplevel_function_types(db, &func_op);
            }

            // Type check this function with access to all function, ability, and type maps
            let typed_op = typecheck_function_with_context(
                db,
                *op,
                function_types.clone(),
                ability_op_types.clone(),
                type_defs.clone(),
            );
            new_ops.push(typed_op);
        } else {
            // Non-function operations pass through
            new_ops.push(*op);
        }
    }

    let new_block = Block::new(
        db,
        block.id(db),
        block.location(db),
        block.args(db).clone(),
        new_ops,
    );
    let new_body = Region::new(db, body.location(db), IdVec::from(vec![new_block]));

    core::Module::create(db, module.location(db), module.name(db), new_body)
}

/// Collect function types from all function definitions in a block.
fn collect_function_types<'db>(
    db: &'db dyn salsa::Database,
    block: &Block<'db>,
) -> Arc<HashMap<Symbol, Type<'db>>> {
    let mut function_types = HashMap::new();

    for op in block.operations(db).iter() {
        if let Ok(func_op) = func::Func::from_operation(db, *op) {
            let name = func_op.name(db);
            let func_type = func_op.r#type(db);
            trace!(
                "collect_function_types: found {:?} with type {:?}",
                name, func_type
            );
            function_types.insert(name, func_type);
        }
    }

    trace!(
        "collect_function_types: collected {} functions",
        function_types.len()
    );
    Arc::new(function_types)
}

/// Collect ability operation types from all ability definitions in a block.
///
/// Walks `tribute.ability_def` operations and extracts operation signatures
/// from their `operations` regions.
fn collect_ability_op_types<'db>(
    db: &'db dyn salsa::Database,
    block: &Block<'db>,
) -> Arc<HashMap<AbilityOpKey, AbilityOpSignature<'db>>> {
    let mut ability_op_types = HashMap::new();

    for op in block.operations(db).iter() {
        if let Ok(ability_def) = tribute::AbilityDef::from_operation(db, *op) {
            let ability_name = ability_def.sym_name(db);
            let operations_region = ability_def.operations(db);

            for inner_block in operations_region.blocks(db).iter() {
                for inner_op in inner_block.operations(db).iter() {
                    if let Ok(op_def) = tribute::OpDef::from_operation(db, *inner_op) {
                        let op_name = op_def.sym_name(db);
                        let op_type = op_def.r#type(db);

                        if let Some(func_ty) = core::Func::from_type(db, op_type) {
                            let params = func_ty.params(db);
                            let return_ty = func_ty.result(db);

                            let signature = AbilityOpSignature {
                                ability_name,
                                op_name,
                                params: params.clone(),
                                return_ty,
                            };

                            trace!(
                                "collect_ability_op_types: {:?}::{:?} -> params={:?}, return={:?}",
                                ability_name, op_name, params, return_ty
                            );

                            let key = (ability_name, op_name);
                            ability_op_types.insert(key, signature);
                        }
                    }
                }
            }
        }
    }

    trace!(
        "collect_ability_op_types: collected {} ability operations",
        ability_op_types.len()
    );
    Arc::new(ability_op_types)
}

/// Collect type definitions (struct and enum) from a block.
///
/// Walks `tribute.struct_def` and `tribute.enum_def` operations and builds
/// a map from type names to their ADT types for use in resolving user-defined
/// type references during type checking.
fn collect_type_defs<'db>(
    db: &'db dyn salsa::Database,
    block: &Block<'db>,
) -> Arc<HashMap<Symbol, Type<'db>>> {
    let mut type_defs = HashMap::new();

    for op in block.operations(db).iter() {
        // Collect struct definitions
        if let Ok(struct_def) = tribute::StructDef::from_operation(db, *op) {
            let name = struct_def.sym_name(db);
            if let Some(&struct_ty) = op.results(db).first() {
                trace!("collect_type_defs: found struct {:?}", name);
                type_defs.insert(name, struct_ty);
            }
        }

        // Collect enum definitions
        if let Ok(enum_def) = tribute::EnumDef::from_operation(db, *op) {
            let name = enum_def.sym_name(db);
            if let Some(&enum_ty) = op.results(db).first() {
                trace!("collect_type_defs: found enum {:?}", name);
                type_defs.insert(name, enum_ty);
            }
        }
    }

    trace!(
        "collect_type_defs: collected {} type definitions",
        type_defs.len()
    );
    Arc::new(type_defs)
}

/// Type check a function with access to all function, ability, and type definition maps.
///
/// This is similar to `typecheck_function` but accepts type maps
/// for proper type inference of generic function calls and handler patterns.
fn typecheck_function_with_context<'db>(
    db: &'db dyn salsa::Database,
    func_op: Operation<'db>,
    function_types: Arc<HashMap<Symbol, Type<'db>>>,
    ability_op_types: Arc<HashMap<AbilityOpKey, AbilityOpSignature<'db>>>,
    type_defs: Arc<HashMap<Symbol, Type<'db>>>,
) -> Operation<'db> {
    let mut checker = TypeChecker::with_type_maps(db, function_types, ability_op_types, type_defs);

    // Check the function definition
    checker.check_func_def(&func_op);

    // Solve constraints for this function
    match checker.solve() {
        Ok(solver) => {
            // Apply substitution to the function's regions
            let subst = solver.type_subst();
            let new_regions: trunk_ir::IdVec<_> = func_op
                .regions(db)
                .iter()
                .map(|r| super::subst::apply_subst_to_region(db, r, subst))
                .collect();

            // Create new operation with resolved types
            let new_results: trunk_ir::IdVec<_> = func_op
                .results(db)
                .iter()
                .map(|ty| subst.apply(db, *ty))
                .collect();

            Operation::new(
                db,
                func_op.location(db),
                func_op.dialect(db),
                func_op.name(db),
                func_op.operands(db).clone(),
                new_results,
                func_op.attributes(db).clone(),
                new_regions,
                func_op.successors(db).clone(),
            )
        }
        Err(_err) => {
            // TODO: Emit type error via accumulator
            // Return the original operation with failure status
            func_op
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use trunk_ir::dialect::arith;
    use trunk_ir::idvec;
    use trunk_ir::{PathId, Span};

    #[salsa::tracked]
    fn build_simple_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let path = PathId::new(db, "file:///test.trb".to_owned());
        let location = trunk_ir::Location::new(path, Span::new(0, 0));

        core::Module::build(db, location, Symbol::new("main"), |entry| {
            let _ = entry.op(arith::Const::i64(db, location, 42));
        })
    }

    #[salsa::tracked]
    fn build_arith_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let path = PathId::new(db, "file:///test.trb".to_owned());
        let location = trunk_ir::Location::new(path, Span::new(0, 0));
        let i64_ty = core::I64::new(db);

        core::Module::build(db, location, Symbol::new("main"), |entry| {
            let a = entry.op(arith::Const::i64(db, location, 1));
            let b = entry.op(arith::Const::i64(db, location, 2));
            let _ = entry.op(arith::add(
                db,
                location,
                a.result(db),
                b.result(db),
                *i64_ty,
            ));
        })
    }

    #[salsa_test]
    fn test_check_simple_module(db: &salsa::DatabaseImpl) {
        let module = build_simple_module(db);
        let result = typecheck_module(db, &module);
        assert!(result.is_ok());
    }

    #[salsa_test]
    fn test_check_arith_binop(db: &salsa::DatabaseImpl) {
        let module = build_arith_module(db);
        let result = typecheck_module(db, &module);
        assert!(result.is_ok());
    }

    #[salsa::tracked]
    fn run_per_function_typecheck(db: &dyn salsa::Database) -> core::Module<'_> {
        let module = build_simple_module(db);
        typecheck_module_per_function(db, module)
    }

    #[salsa_test]
    fn test_typecheck_function_per_function(db: &salsa::DatabaseImpl) {
        let typed_module = run_per_function_typecheck(db);
        assert_eq!(typed_module.name(db), "main");
    }

    #[salsa_test]
    fn test_has_type_vars_detection(db: &salsa::DatabaseImpl) {
        use tribute_ir::dialect::tribute;

        // Type variable should be detected
        let type_var = tribute::type_var_with_id(db, 42);
        assert!(
            has_type_vars(db, type_var),
            "Type variable should be detected"
        );

        // Concrete type should not have type vars
        let i64_ty = core::I64::new(db);
        assert!(
            !has_type_vars(db, *i64_ty),
            "Concrete type should not have type vars"
        );

        // Function type with type var in return should be detected
        let func_with_var = core::Func::new(db, idvec![*i64_ty], type_var).as_type();
        assert!(
            has_type_vars(db, func_with_var),
            "Function with type var return should be detected"
        );

        // Function type with concrete types should not have type vars
        let func_concrete = core::Func::new(db, idvec![*i64_ty], *i64_ty).as_type();
        assert!(
            !has_type_vars(db, func_concrete),
            "Function with concrete types should not have type vars"
        );
    }

    // =========================================================================
    // Pattern Matching Tests
    // =========================================================================

    /// Build a module with struct field access using adt.struct_get
    #[salsa::tracked]
    fn build_struct_get_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let path = PathId::new(db, "file:///test.trb".to_owned());
        let location = trunk_ir::Location::new(path, Span::new(0, 0));
        let i64_ty = *core::I64::new(db);

        // Create struct type: struct Point { x: i64, y: i64 }
        let point_ty = adt::struct_type(
            db,
            Symbol::new("Point"),
            vec![(Symbol::new("x"), i64_ty), (Symbol::new("y"), i64_ty)],
        );

        let func = func::Func::build(db, location, "test_struct_get", idvec![], i64_ty, |entry| {
            // Create a struct: Point { x: 1, y: 2 }
            let x_val = entry.op(arith::Const::i64(db, location, 1));
            let y_val = entry.op(arith::Const::i64(db, location, 2));
            let point = entry.op(adt::struct_new(
                db,
                location,
                vec![x_val.result(db), y_val.result(db)],
                point_ty,
                point_ty,
            ));

            // Access field x (index 0)
            let field_val = entry.op(adt::struct_get(
                db,
                location,
                point.result(db),
                i64_ty,
                point_ty,
                0,
            ));
            entry.op(func::Return::value(db, location, field_val.result(db)));
        });

        core::Module::build(db, location, Symbol::new("main"), |top| {
            top.op(func);
        })
    }

    #[salsa_test]
    fn test_struct_get_type_inference(db: &salsa::DatabaseImpl) {
        let module = build_struct_get_module(db);
        let result = typecheck_module(db, &module);
        assert!(result.is_ok(), "Struct field access should typecheck");
    }

    /// Build a module with enum variant creation using adt.variant_new
    #[salsa::tracked]
    fn build_variant_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let path = PathId::new(db, "file:///test.trb".to_owned());
        let location = trunk_ir::Location::new(path, Span::new(0, 0));
        let i64_ty = *core::I64::new(db);

        // Create enum type: enum Option { Some(i64), None }
        let option_ty = adt::enum_type(
            db,
            Symbol::new("Option"),
            vec![
                (Symbol::new("Some"), vec![i64_ty]),
                (Symbol::new("None"), vec![]),
            ],
        );

        let func = func::Func::build(
            db,
            location,
            "test_variant_new",
            idvec![],
            option_ty,
            |entry| {
                // Create variant: Some(42)
                let value = entry.op(arith::Const::i64(db, location, 42));
                let variant = entry.op(adt::variant_new(
                    db,
                    location,
                    vec![value.result(db)],
                    option_ty,
                    option_ty,
                    Symbol::new("Some"),
                ));
                entry.op(func::Return::value(db, location, variant.result(db)));
            },
        );

        core::Module::build(db, location, Symbol::new("main"), |top| {
            top.op(func);
        })
    }

    #[salsa_test]
    fn test_variant_new_type_inference(db: &salsa::DatabaseImpl) {
        let module = build_variant_module(db);
        let result = typecheck_module(db, &module);
        assert!(result.is_ok(), "Variant creation should typecheck");
    }

    /// Build a module with nested struct types
    #[salsa::tracked]
    fn build_nested_struct_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let path = PathId::new(db, "file:///test.trb".to_owned());
        let location = trunk_ir::Location::new(path, Span::new(0, 0));
        let i64_ty = *core::I64::new(db);

        // struct Point { x: i64, y: i64 }
        let point_ty = adt::struct_type(
            db,
            Symbol::new("Point"),
            vec![(Symbol::new("x"), i64_ty), (Symbol::new("y"), i64_ty)],
        );

        // struct Line { start: Point, end: Point }
        let line_ty = adt::struct_type(
            db,
            Symbol::new("Line"),
            vec![
                (Symbol::new("start"), point_ty),
                (Symbol::new("end"), point_ty),
            ],
        );

        let func = func::Func::build(
            db,
            location,
            "test_nested_struct",
            idvec![],
            i64_ty,
            |entry| {
                // Create points
                let x1 = entry.op(arith::Const::i64(db, location, 0));
                let y1 = entry.op(arith::Const::i64(db, location, 0));
                let start = entry.op(adt::struct_new(
                    db,
                    location,
                    vec![x1.result(db), y1.result(db)],
                    point_ty,
                    point_ty,
                ));

                let x2 = entry.op(arith::Const::i64(db, location, 10));
                let y2 = entry.op(arith::Const::i64(db, location, 10));
                let end = entry.op(adt::struct_new(
                    db,
                    location,
                    vec![x2.result(db), y2.result(db)],
                    point_ty,
                    point_ty,
                ));

                // Create line
                let line = entry.op(adt::struct_new(
                    db,
                    location,
                    vec![start.result(db), end.result(db)],
                    line_ty,
                    line_ty,
                ));

                // Access nested field: line.start (returns Point)
                let line_start = entry.op(adt::struct_get(
                    db,
                    location,
                    line.result(db),
                    point_ty,
                    line_ty,
                    0,
                ));

                // Access deeper: line.start.x (returns i64)
                let x_val = entry.op(adt::struct_get(
                    db,
                    location,
                    line_start.result(db),
                    i64_ty,
                    point_ty,
                    0,
                ));
                entry.op(func::Return::value(db, location, x_val.result(db)));
            },
        );

        core::Module::build(db, location, Symbol::new("main"), |top| {
            top.op(func);
        })
    }

    #[salsa_test]
    fn test_nested_struct_type_inference(db: &salsa::DatabaseImpl) {
        let module = build_nested_struct_module(db);
        let result = typecheck_module(db, &module);
        assert!(result.is_ok(), "Nested struct access should typecheck");
    }

    /// Test that struct field type inference works correctly
    #[salsa_test]
    fn test_get_struct_field_type(db: &salsa::DatabaseImpl) {
        let i64_ty = *core::I64::new(db);
        let f64_ty = *core::F64::new(db);

        // struct User { score: f64, age: i64 }
        let user_ty = adt::struct_type(
            db,
            Symbol::new("User"),
            vec![(Symbol::new("score"), f64_ty), (Symbol::new("age"), i64_ty)],
        );

        // Verify we can get field types from the struct
        let fields = adt::get_struct_fields(db, user_ty);
        assert!(fields.is_some(), "Should be able to get struct fields");

        let fields = fields.unwrap();
        assert_eq!(fields.len(), 2, "User struct should have 2 fields");

        // Check field types
        assert_eq!(fields[0].0, Symbol::new("score"));
        assert_eq!(fields[0].1, f64_ty);
        assert_eq!(fields[1].0, Symbol::new("age"));
        assert_eq!(fields[1].1, i64_ty);
    }

    /// Test that enum variant field types can be retrieved
    #[salsa_test]
    fn test_get_enum_variant_types(db: &salsa::DatabaseImpl) {
        let i64_ty = *core::I64::new(db);
        let f64_ty = *core::F64::new(db);

        // enum Result { Ok(i64), Err(f64) }
        let result_ty = adt::enum_type(
            db,
            Symbol::new("Result"),
            vec![
                (Symbol::new("Ok"), vec![i64_ty]),
                (Symbol::new("Err"), vec![f64_ty]),
            ],
        );

        // Verify we can get variant info
        let variants = adt::get_enum_variants(db, result_ty);
        assert!(variants.is_some(), "Should be able to get enum variants");

        let variants = variants.unwrap();
        assert_eq!(variants.len(), 2, "Result enum should have 2 variants");

        // Check variant types
        let (ok_name, ok_fields) = &variants[0];
        assert_eq!(*ok_name, Symbol::new("Ok"));
        assert_eq!(ok_fields.len(), 1);
        assert_eq!(ok_fields[0], i64_ty);

        let (err_name, err_fields) = &variants[1];
        assert_eq!(*err_name, Symbol::new("Err"));
        assert_eq!(err_fields.len(), 1);
        assert_eq!(err_fields[0], f64_ty);
    }
}
