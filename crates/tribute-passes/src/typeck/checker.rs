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
use trunk_ir::{
    Attribute, Block, DialectOp, DialectType, IdVec, Operation, QualifiedName, Region, Symbol,
    Type, Value,
    dialect::{ability, adt, arith, case, core, func, list, pat, src, ty},
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
    function_types: Arc<HashMap<QualifiedName, Type<'db>>>,
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
        }
    }

    /// Create a new type checker with a pre-built function type map.
    pub fn with_function_types(
        db: &'db dyn salsa::Database,
        function_types: Arc<HashMap<QualifiedName, Type<'db>>>,
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
        }
    }

    /// Generate a fresh type variable.
    pub fn fresh_type_var(&mut self) -> Type<'db> {
        let id = self.next_type_var;
        self.next_type_var += 1;
        ty::var_with_id(self.db, id)
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

    /// Check a module.
    pub fn check_module(&mut self, module: &core::Module<'db>) {
        self.seed_var_counters(module);

        // Collect all function types before checking
        // This enables proper type inference for generic function calls
        let body = module.body(self.db);
        if let Some(block) = body.blocks(self.db).first() {
            self.collect_function_types_from_block(block);
        }

        self.check_region(&body);
    }

    /// Collect function types from a block.
    fn collect_function_types_from_block(&mut self, block: &Block<'db>) {
        for op in block.operations(self.db).iter() {
            if let Ok(func_op) = func::Func::from_operation(self.db, *op) {
                let name = func_op.qualified_name(self.db);
                let func_type = func_op.r#type(self.db);
                trace!(
                    "collect_function_types_from_block: {:?} -> {:?}",
                    name, func_type
                );
                Arc::make_mut(&mut self.function_types).insert(name, func_type);
            }
        }
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
            for &arg in block.args(self.db).iter() {
                self.collect_vars_in_type(arg, max_type_var_id, max_row_var_id);
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
        if ty::is_var(self.db, ty)
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
            for (i, &arg_ty) in block.args(self.db).iter().enumerate() {
                let arg_value = block.arg(self.db, i);
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
        } else if dialect == src::DIALECT_NAME() {
            if name == src::VAR() {
                self.check_src_var(op);
            } else if name == src::CALL() || name == src::CONS() {
                self.check_src_call(op);
            } else if name == src::BINOP() {
                self.check_src_binop(op);
            } else if name == src::LAMBDA() {
                self.check_src_lambda(op);
            } else if name == src::BLOCK() {
                self.check_src_block(op);
            } else if name == src::YIELD() {
                self.check_src_yield(op);
            } else if name == src::TUPLE() {
                self.check_src_tuple(op);
            } else if name == src::CONST() {
                self.check_src_const(op);
            } else {
                self.check_unknown_op(op);
            }
        } else if dialect == adt::DIALECT_NAME() {
            if name == adt::STRING_CONST() {
                self.check_string_const(op);
            } else if name == adt::STRUCT_NEW() || name == adt::VARIANT_NEW() {
                // For struct/variant construction, the result type is already set correctly
                // to the struct/enum type. Just record this type for the result value.
                self.check_adt_new(op);
            } else {
                self.check_unknown_op(op);
            }
        } else if dialect == list::DIALECT_NAME() {
            if name == list::NEW() {
                self.check_list_new(op);
            } else {
                self.check_unknown_op(op);
            }
        } else if dialect == case::DIALECT_NAME() {
            if name == case::CASE() {
                self.check_case(op);
            } else {
                self.check_unknown_op(op);
            }
        } else if dialect == ability::DIALECT_NAME() {
            if name == ability::PERFORM() {
                self.check_ability_perform(op);
            } else if name == ability::PROMPT() {
                self.check_ability_prompt(op);
            } else if name == ability::RESUME() {
                self.check_ability_resume(op);
            } else if name == ability::ABORT() {
                self.check_ability_abort(op);
            } else {
                self.check_unknown_op(op);
            }
        } else if dialect == ty::DIALECT_NAME() {
            // Type declarations (struct, enum, ability) don't need type checking
            if name == ty::STRUCT() || name == ty::ENUM() || name == ty::ABILITY() {
                // No-op
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
        // Get function result type (first result type is the function type itself)
        let results = op.results(self.db);
        let func_type = results.first().copied();

        // Save current entry block arg types (for nested functions)
        let saved_entry_args = std::mem::take(&mut self.entry_block_arg_types);

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
                    self.entry_block_arg_types =
                        entry_block.args(self.db).iter().copied().collect();
                }

                // Record types for the actual entry block arguments (for direct lookups)
                for (i, &arg_ty) in entry_block.args(self.db).iter().enumerate() {
                    let arg_value = entry_block.arg(self.db, i);
                    self.record_type(arg_value, arg_ty);
                }
            }

            self.check_region(body);
        }

        // Restore entry block arg types
        self.entry_block_arg_types = saved_entry_args;

        // Record result type
        if let Some(ty) = func_type {
            let value = op.result(self.db, 0);
            self.record_type(value, ty);
        }
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
        if ty::is_var(self.db, ty) {
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
        if let Some(&callee) = operands.first()
            && let Some(callee_type) = self.get_type(callee)
            && let Some(func_type) = core::Func::from_type(self.db, callee_type)
        {
            // Arguments start after the callee (index 0)
            let args: Vec<_> = operands.iter().skip(1).copied().collect();
            effect_handled = self.constrain_call_types(&func_type, &args, result_type);
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

    // === src dialect checking ===

    fn check_src_var(&mut self, op: &Operation<'db>) {
        // Variable reference - type is determined by binding
        let results = op.results(self.db);
        let result_type = results
            .first()
            .copied()
            .unwrap_or_else(|| self.fresh_type_var());
        let value = op.result(self.db, 0);
        self.record_type(value, result_type);
    }

    fn check_src_call(&mut self, op: &Operation<'db>) {
        // Call expression - need to check callee type and argument types
        let results = op.results(self.db);
        let result_type = results
            .first()
            .copied()
            .unwrap_or_else(|| self.fresh_type_var());

        // Create fresh type var for result
        let actual_result = self.fresh_type_var();
        self.constrain_eq(actual_result, result_type);

        let value = op.result(self.db, 0);
        self.record_type(value, actual_result);

        // TODO: Once name resolution is done, constrain callee function type
        // with argument types
    }

    fn check_src_binop(&mut self, op: &Operation<'db>) {
        let operands = op.operands(self.db);
        let results = op.results(self.db);
        let result_type = results
            .first()
            .copied()
            .unwrap_or_else(|| self.fresh_type_var());

        // For most binops, operands should have compatible types
        if operands.len() >= 2 {
            let lhs = operands[0];
            let rhs = operands[1];

            if let (Some(lhs_ty), Some(rhs_ty)) = (self.get_type(lhs), self.get_type(rhs)) {
                // For now, require same type (will be refined with type classes)
                self.constrain_eq(lhs_ty, rhs_ty);
            }
        }

        let value = op.result(self.db, 0);
        self.record_type(value, result_type);
    }

    fn check_src_lambda(&mut self, op: &Operation<'db>) {
        // Lambda expression
        let results = op.results(self.db);
        let lambda_type = results
            .first()
            .copied()
            .unwrap_or_else(|| self.fresh_type_var());

        // Check the body
        let regions = op.regions(self.db);
        if let Some(body) = regions.first() {
            self.check_region(body);
        }

        let value = op.result(self.db, 0);
        self.record_type(value, lambda_type);
    }

    fn check_src_block(&mut self, op: &Operation<'db>) {
        let results = op.results(self.db);
        let result_type = results
            .first()
            .copied()
            .unwrap_or_else(|| self.fresh_type_var());

        // Check the body
        let regions = op.regions(self.db);
        if let Some(body) = regions.first() {
            self.check_region(body);
        }

        let value = op.result(self.db, 0);
        self.record_type(value, result_type);
    }

    fn check_src_yield(&mut self, _op: &Operation<'db>) {
        // Yield doesn't have a result
        // The value should match the enclosing block's result type
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

    fn check_src_const(&mut self, op: &Operation<'db>) {
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

    // === list dialect checking ===

    fn check_list_new(&mut self, op: &Operation<'db>) {
        let results = op.results(self.db);
        let result_type = results
            .first()
            .copied()
            .unwrap_or_else(|| self.fresh_type_var());
        let value = op.result(self.db, 0);
        self.record_type(value, result_type);
    }

    // === case dialect checking ===

    fn check_case(&mut self, op: &Operation<'db>) {
        let results = op.results(self.db);
        let result_type = results
            .first()
            .copied()
            .unwrap_or_else(|| self.fresh_type_var());

        // Check each branch region and collect handled abilities from handler patterns
        let regions = op.regions(self.db);
        let mut handled_abilities = Vec::new();

        for region in regions.iter() {
            // Look for case.arm operations in the region
            for block in region.blocks(self.db).iter() {
                for arm_op in block.operations(self.db).iter() {
                    if arm_op.dialect(self.db) == case::DIALECT_NAME()
                        && arm_op.name(self.db) == case::ARM()
                    {
                        // Extract handled abilities from the pattern region
                        let arm_regions = arm_op.regions(self.db);
                        if let Some(pattern_region) = arm_regions.first() {
                            self.extract_handled_abilities(pattern_region, &mut handled_abilities);
                        }

                        // Check the body region (second region)
                        if let Some(body_region) = arm_regions.get(1) {
                            self.check_region(body_region);
                        }
                    } else {
                        // Regular operation in case body
                        self.check_operation(arm_op);
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

    /// Extract abilities being handled from a pattern region.
    fn extract_handled_abilities(
        &self,
        pattern_region: &Region<'db>,
        handled: &mut Vec<AbilityRef<'db>>,
    ) {
        let ability_ref_sym = Symbol::new("ability_ref");
        for block in pattern_region.blocks(self.db).iter() {
            for op in block.operations(self.db).iter() {
                // Check for pat.handler_suspend
                if op.dialect(self.db) == pat::DIALECT_NAME()
                    && op.name(self.db) == pat::HANDLER_SUSPEND()
                {
                    // Extract ability reference from attributes
                    // The ability_ref is now a Type (core.ability_ref) instead of QualifiedName
                    let attrs = op.attributes(self.db);
                    if let Some(Attribute::Type(ability_ty)) = attrs.get(&ability_ref_sym)
                        && let Some(ability) = AbilityRef::from_type(self.db, *ability_ty)
                        && !handled.contains(&ability)
                    {
                        handled.push(ability);
                    }
                }
            }
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
        // ability.prompt: runs body in a delimited context, returns Request
        //
        // The body's effects are captured by the prompt. The resulting Request
        // will be pattern-matched by case.case, which handles effect elimination.

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
        // We propagate body's effects to outer context first, then case.case
        // will eliminate handled abilities based on handler patterns
        // (pat.handler_suspend in pattern regions).
        let body_effect = std::mem::replace(&mut self.current_effect, outer_effect);
        self.merge_effect(body_effect);

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

    // First pass: collect all function types
    let function_types = collect_function_types(db, block);

    // Second pass: type check each function with the function types map
    let mut new_ops: IdVec<Operation<'db>> = IdVec::new();
    for op in block.operations(db).iter() {
        if op.dialect(db) == func::DIALECT_NAME() && op.name(db) == func::FUNC() {
            // Validate type annotations for top-level functions
            if let Ok(func_op) = func::Func::from_operation(db, *op) {
                validate_toplevel_function_types(db, &func_op);
            }

            // Type check this function with access to all function types
            let typed_op = typecheck_function_with_context(db, *op, function_types.clone());
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
) -> Arc<HashMap<QualifiedName, Type<'db>>> {
    let mut function_types = HashMap::new();

    for op in block.operations(db).iter() {
        if let Ok(func_op) = func::Func::from_operation(db, *op) {
            let name = func_op.qualified_name(db);
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

/// Type check a function with access to all function types.
///
/// This is similar to `typecheck_function` but accepts a function types map
/// for proper type inference of generic function calls.
fn typecheck_function_with_context<'db>(
    db: &'db dyn salsa::Database,
    func_op: Operation<'db>,
    function_types: Arc<HashMap<QualifiedName, Type<'db>>>,
) -> Operation<'db> {
    let mut checker = TypeChecker::with_function_types(db, function_types);

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
        use trunk_ir::dialect::ty;

        // Type variable should be detected
        let type_var = ty::var_with_id(db, 42);
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
}
