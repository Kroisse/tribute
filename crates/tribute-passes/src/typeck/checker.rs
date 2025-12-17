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

use tribute_trunk_ir::{
    Operation, Region, Type, Value,
    dialect::{core, ty},
};

use super::constraint::ConstraintSet;
use super::effect_row::{EffectRow, RowVar};
use super::solver::{SolveResult, TypeSolver};

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
    pub fn get_type(&self, value: Value<'db>) -> Option<Type<'db>> {
        self.value_types.get(&value).copied()
    }

    /// Add a type equality constraint.
    pub fn constrain_eq(&mut self, t1: Type<'db>, t2: Type<'db>) {
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
        let body = module.body(self.db);
        self.check_region(&body);
    }

    /// Check a region (sequence of blocks).
    pub fn check_region(&mut self, region: &Region<'db>) {
        for block in region.blocks(self.db).iter() {
            for op in block.operations(self.db).iter() {
                self.check_operation(op);
            }
        }
    }

    /// Check a single operation.
    pub fn check_operation(&mut self, op: &Operation<'db>) {
        let dialect = op.dialect(self.db).text(self.db);
        let name = op.name(self.db).text(self.db);

        match (dialect, name) {
            // === func dialect ===
            ("func", "func") => self.check_func_def(op),
            ("func", "return") => self.check_return(op),

            // === arith dialect ===
            ("arith", "const") => self.check_arith_const(op),
            ("arith", "add") | ("arith", "sub") | ("arith", "mul") | ("arith", "div") => {
                self.check_arith_binop(op)
            }
            ("arith", "neg") => self.check_arith_neg(op),
            ("arith", "cmp") => self.check_arith_cmp(op),

            // === src dialect (unresolved) ===
            ("src", "var") => self.check_src_var(op),
            ("src", "call") => self.check_src_call(op),
            ("src", "binop") => self.check_src_binop(op),
            ("src", "lambda") => self.check_src_lambda(op),
            ("src", "block") => self.check_src_block(op),
            ("src", "yield") => self.check_src_yield(op),
            ("src", "tuple") => self.check_src_tuple(op),
            ("src", "const") => self.check_src_const(op),

            // === adt dialect ===
            ("adt", "string_const") => self.check_string_const(op),

            // === list dialect ===
            ("list", "new") => self.check_list_new(op),

            // === case dialect ===
            ("case", "case") => self.check_case(op),

            // === type dialect ===
            ("type", "struct") | ("type", "enum") | ("type", "ability") => {
                // Type declarations don't need type checking
            }

            // === core dialect ===
            ("core", "module") => {
                // Module is checked via check_module
            }
            ("core", "unrealized_conversion_cast") => {
                // Pass through for now - assign fresh type var to result
                let results = op.results(self.db);
                for (i, _result_ty) in results.iter().enumerate() {
                    let value = op.result(self.db, i);
                    let ty = self.fresh_type_var();
                    self.record_type(value, ty);
                }
            }

            _ => {
                // Unknown operation - assign fresh type vars to results
                let results = op.results(self.db);
                for (i, _result_ty) in results.iter().enumerate() {
                    let value = op.result(self.db, i);
                    let ty = self.fresh_type_var();
                    self.record_type(value, ty);
                }
            }
        }
    }

    // === func dialect checking ===

    fn check_func_def(&mut self, op: &Operation<'db>) {
        // Get function result type (first result type is the function type itself)
        let results = op.results(self.db);
        let func_type = results.first().copied();

        // Check the body
        let regions = op.regions(self.db);
        if let Some(body) = regions.first() {
            self.check_region(body);
        }

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
        for operand in operands.iter() {
            if let Some(op_type) = self.get_type(*operand) {
                self.constrain_eq(op_type, result_type);
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
        let bool_type = *core::I1::new(self.db);
        let value = op.result(self.db, 0);
        self.record_type(value, bool_type);
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
        let string_type = *core::String::new(self.db);
        let value = op.result(self.db, 0);
        self.record_type(value, string_type);
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

        // Check each branch region
        let regions = op.regions(self.db);
        for region in regions.iter() {
            self.check_region(region);
        }

        let value = op.result(self.db, 0);
        self.record_type(value, result_type);
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

#[cfg(test)]
mod tests {
    use super::*;
    use salsa::Database;
    use std::path::PathBuf;
    use tribute_core::{PathId, Span, TributeDatabaseImpl};
    use tribute_trunk_ir::dialect::arith;

    #[salsa::tracked]
    fn build_simple_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let path = PathId::new(db, PathBuf::from("test.tr"));
        let location = tribute_core::Location::new(path, Span::new(0, 0));

        core::Module::build(db, location, "test", |entry| {
            let _ = entry.op(arith::Const::i64(db, location, 42));
        })
    }

    #[salsa::tracked]
    fn build_arith_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let path = PathId::new(db, PathBuf::from("test.tr"));
        let location = tribute_core::Location::new(path, Span::new(0, 0));
        let i64_ty = *core::I64::new(db);

        core::Module::build(db, location, "test", |entry| {
            let a = entry.op(arith::Const::i64(db, location, 1));
            let b = entry.op(arith::Const::i64(db, location, 2));
            let _ = entry.op(arith::add(db, location, a.result(db), b.result(db), i64_ty));
        })
    }

    #[test]
    fn test_check_simple_module() {
        TributeDatabaseImpl::default().attach(|db| {
            let module = build_simple_module(db);
            let result = typecheck_module(db, &module);
            assert!(result.is_ok());
        });
    }

    #[test]
    fn test_check_arith_binop() {
        TributeDatabaseImpl::default().attach(|db| {
            let module = build_arith_module(db);
            let result = typecheck_module(db, &module);
            assert!(result.is_ok());
        });
    }
}
