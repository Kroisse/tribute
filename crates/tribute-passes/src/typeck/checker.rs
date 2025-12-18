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
    Attribute, DialectType, Operation, Region, Symbol, Type, Value,
    dialect::{core, ty},
};

use super::constraint::ConstraintSet;
use super::effect_row::{AbilityRef, EffectRow, RowVar};
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

/// Cached symbols for efficient operation dispatch.
///
/// Pre-interned symbols avoid repeated `Symbol::new` calls during type checking.
struct OpSymbols<'db> {
    // Dialects
    func: Symbol<'db>,
    arith: Symbol<'db>,
    src: Symbol<'db>,
    adt: Symbol<'db>,
    list: Symbol<'db>,
    case: Symbol<'db>,
    ability: Symbol<'db>,
    r#type: Symbol<'db>,
    core: Symbol<'db>,

    // func dialect ops
    func_func: Symbol<'db>,
    func_return: Symbol<'db>,
    func_call: Symbol<'db>,
    func_call_indirect: Symbol<'db>,
    func_constant: Symbol<'db>,

    // arith dialect ops
    arith_const: Symbol<'db>,
    arith_add: Symbol<'db>,
    arith_sub: Symbol<'db>,
    arith_mul: Symbol<'db>,
    arith_div: Symbol<'db>,
    arith_neg: Symbol<'db>,
    arith_cmp: Symbol<'db>,

    // src dialect ops
    src_var: Symbol<'db>,
    src_call: Symbol<'db>,
    src_binop: Symbol<'db>,
    src_lambda: Symbol<'db>,
    src_block: Symbol<'db>,
    src_yield: Symbol<'db>,
    src_tuple: Symbol<'db>,
    src_const: Symbol<'db>,

    // adt dialect ops
    adt_string_const: Symbol<'db>,

    // list dialect ops
    list_new: Symbol<'db>,

    // case dialect ops
    case_case: Symbol<'db>,

    // ability dialect ops
    ability_perform: Symbol<'db>,
    ability_prompt: Symbol<'db>,
    ability_resume: Symbol<'db>,
    ability_abort: Symbol<'db>,

    // type dialect ops
    type_struct: Symbol<'db>,
    type_enum: Symbol<'db>,
    type_ability: Symbol<'db>,

    // core dialect ops
    core_module: Symbol<'db>,
    core_unrealized_conversion_cast: Symbol<'db>,

    // pat dialect
    pat: Symbol<'db>,
    pat_handler_suspend: Symbol<'db>,
    #[allow(dead_code)] // Will be used for Done pattern handling
    pat_handler_done: Symbol<'db>,

    // case dialect arm
    case_arm: Symbol<'db>,

    // Attribute keys
    attr_ability_ref: Symbol<'db>,
    #[allow(dead_code)] // Will be used for operation name extraction
    attr_op: Symbol<'db>,
}

impl<'db> OpSymbols<'db> {
    fn new(db: &'db dyn salsa::Database) -> Self {
        Self {
            // Dialects
            func: Symbol::new(db, "func"),
            arith: Symbol::new(db, "arith"),
            src: Symbol::new(db, "src"),
            adt: Symbol::new(db, "adt"),
            list: Symbol::new(db, "list"),
            case: Symbol::new(db, "case"),
            ability: Symbol::new(db, "ability"),
            r#type: Symbol::new(db, "type"),
            core: Symbol::new(db, "core"),

            // func dialect ops
            func_func: Symbol::new(db, "func"),
            func_return: Symbol::new(db, "return"),
            func_call: Symbol::new(db, "call"),
            func_call_indirect: Symbol::new(db, "call_indirect"),
            func_constant: Symbol::new(db, "constant"),

            // arith dialect ops
            arith_const: Symbol::new(db, "const"),
            arith_add: Symbol::new(db, "add"),
            arith_sub: Symbol::new(db, "sub"),
            arith_mul: Symbol::new(db, "mul"),
            arith_div: Symbol::new(db, "div"),
            arith_neg: Symbol::new(db, "neg"),
            arith_cmp: Symbol::new(db, "cmp"),

            // src dialect ops
            src_var: Symbol::new(db, "var"),
            src_call: Symbol::new(db, "call"),
            src_binop: Symbol::new(db, "binop"),
            src_lambda: Symbol::new(db, "lambda"),
            src_block: Symbol::new(db, "block"),
            src_yield: Symbol::new(db, "yield"),
            src_tuple: Symbol::new(db, "tuple"),
            src_const: Symbol::new(db, "const"),

            // adt dialect ops
            adt_string_const: Symbol::new(db, "string_const"),

            // list dialect ops
            list_new: Symbol::new(db, "new"),

            // case dialect ops
            case_case: Symbol::new(db, "case"),

            // ability dialect ops
            ability_perform: Symbol::new(db, "perform"),
            ability_prompt: Symbol::new(db, "prompt"),
            ability_resume: Symbol::new(db, "resume"),
            ability_abort: Symbol::new(db, "abort"),

            // type dialect ops
            type_struct: Symbol::new(db, "struct"),
            type_enum: Symbol::new(db, "enum"),
            type_ability: Symbol::new(db, "ability"),

            // core dialect ops
            core_module: Symbol::new(db, "module"),
            core_unrealized_conversion_cast: Symbol::new(db, "unrealized_conversion_cast"),

            // pat dialect
            pat: Symbol::new(db, "pat"),
            pat_handler_suspend: Symbol::new(db, "handler_suspend"),
            pat_handler_done: Symbol::new(db, "handler_done"),

            // case dialect arm
            case_arm: Symbol::new(db, "arm"),

            // Attribute keys
            attr_ability_ref: Symbol::new(db, "ability_ref"),
            attr_op: Symbol::new(db, "op"),
        }
    }
}

/// Type checking context.
///
/// Tracks the current environment (bindings) and generates constraints.
pub struct TypeChecker<'db> {
    db: &'db dyn salsa::Database,
    /// Cached symbols for efficient dispatch.
    syms: OpSymbols<'db>,
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
            syms: OpSymbols::new(db),
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
    ///
    /// Uses cached symbols for efficient O(1) dispatch via symbol ID comparison.
    pub fn check_operation(&mut self, op: &Operation<'db>) {
        let dialect = op.dialect(self.db);
        let name = op.name(self.db);
        let s = &self.syms;

        // Dispatch by dialect first, then by operation name
        if dialect == s.func {
            if name == s.func_func {
                self.check_func_def(op);
            } else if name == s.func_return {
                self.check_return(op);
            } else if name == s.func_call {
                self.check_func_call(op);
            } else if name == s.func_call_indirect {
                self.check_func_call_indirect(op);
            } else if name == s.func_constant {
                self.check_func_constant(op);
            } else {
                self.check_unknown_op(op);
            }
        } else if dialect == s.arith {
            if name == s.arith_const {
                self.check_arith_const(op);
            } else if name == s.arith_add
                || name == s.arith_sub
                || name == s.arith_mul
                || name == s.arith_div
            {
                self.check_arith_binop(op);
            } else if name == s.arith_neg {
                self.check_arith_neg(op);
            } else if name == s.arith_cmp {
                self.check_arith_cmp(op);
            } else {
                self.check_unknown_op(op);
            }
        } else if dialect == s.src {
            if name == s.src_var {
                self.check_src_var(op);
            } else if name == s.src_call {
                self.check_src_call(op);
            } else if name == s.src_binop {
                self.check_src_binop(op);
            } else if name == s.src_lambda {
                self.check_src_lambda(op);
            } else if name == s.src_block {
                self.check_src_block(op);
            } else if name == s.src_yield {
                self.check_src_yield(op);
            } else if name == s.src_tuple {
                self.check_src_tuple(op);
            } else if name == s.src_const {
                self.check_src_const(op);
            } else {
                self.check_unknown_op(op);
            }
        } else if dialect == s.adt {
            if name == s.adt_string_const {
                self.check_string_const(op);
            } else {
                self.check_unknown_op(op);
            }
        } else if dialect == s.list {
            if name == s.list_new {
                self.check_list_new(op);
            } else {
                self.check_unknown_op(op);
            }
        } else if dialect == s.case {
            if name == s.case_case {
                self.check_case(op);
            } else {
                self.check_unknown_op(op);
            }
        } else if dialect == s.ability {
            if name == s.ability_perform {
                self.check_ability_perform(op);
            } else if name == s.ability_prompt {
                self.check_ability_prompt(op);
            } else if name == s.ability_resume {
                self.check_ability_resume(op);
            } else if name == s.ability_abort {
                self.check_ability_abort(op);
            } else {
                self.check_unknown_op(op);
            }
        } else if dialect == s.r#type {
            // Type declarations (struct, enum, ability) don't need type checking
            if name == s.type_struct || name == s.type_enum || name == s.type_ability {
                // No-op
            } else {
                self.check_unknown_op(op);
            }
        } else if dialect == s.core {
            if name == s.core_module {
                // Module is checked via check_module
            } else if name == s.core_unrealized_conversion_cast {
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

    fn check_func_call(&mut self, op: &Operation<'db>) {
        // func.call: direct call to a function symbol
        // The effect of the call is the effect declared in the function type
        let results = op.results(self.db);
        let result_type = results
            .first()
            .copied()
            .unwrap_or_else(|| self.fresh_type_var());

        // Record the result type
        let value = op.result(self.db, 0);
        self.record_type(value, result_type);

        // TODO: Look up the function by its callee symbol to get the function type
        // and propagate its effect. For now, we assign a fresh effect row variable
        // to represent the potential effects of the call.
        let call_effect = EffectRow::var(self.fresh_row_var());
        self.merge_effect(call_effect);
    }

    fn check_func_call_indirect(&mut self, op: &Operation<'db>) {
        // func.call_indirect: indirect call via function value
        // The callee is the first operand (function value)
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
            // Constrain result type
            let func_result = func_type.result(self.db);
            self.constrain_eq(result_type, func_result);

            // Propagate the function's effect
            if let Some(effect_ty) = func_type.effect(self.db)
                && let Some(effect_row) = EffectRow::from_type(self.db, effect_ty)
            {
                self.merge_effect(effect_row);
            }

            // Constrain argument types
            let param_types = func_type.params(self.db);
            for (i, &param_ty) in param_types.iter().enumerate() {
                // Arguments start after the callee (index 0)
                if let Some(&arg) = operands.get(i + 1)
                    && let Some(arg_ty) = self.get_type(arg)
                {
                    self.constrain_eq(arg_ty, param_ty);
                }
            }
        }

        let value = op.result(self.db, 0);
        self.record_type(value, result_type);
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

        // Check each branch region and collect handled abilities from handler patterns
        let regions = op.regions(self.db);
        let mut handled_abilities = Vec::new();

        for region in regions.iter() {
            // Look for case.arm operations in the region
            for block in region.blocks(self.db).iter() {
                for arm_op in block.operations(self.db).iter() {
                    if arm_op.dialect(self.db) == self.syms.case
                        && arm_op.name(self.db) == self.syms.case_arm
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
        for block in pattern_region.blocks(self.db).iter() {
            for op in block.operations(self.db).iter() {
                // Check for pat.handler_suspend
                if op.dialect(self.db) == self.syms.pat
                    && op.name(self.db) == self.syms.pat_handler_suspend
                {
                    // Extract ability reference from attributes
                    let attrs = op.attributes(self.db);
                    if let Some(Attribute::SymbolRef(ability_path)) =
                        attrs.get(&self.syms.attr_ability_ref)
                    {
                        // Extract ability name from path
                        let ability_name = ability_path
                            .last()
                            .copied()
                            .unwrap_or_else(|| Symbol::new(self.db, "unknown"));

                        let ability = AbilityRef::simple(ability_name);
                        if !handled.contains(&ability) {
                            handled.push(ability);
                        }
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
        // attr: ability_ref: SymbolRef (path to ability)
        // attr: op: Symbol (operation name)
        if let Some(Attribute::SymbolRef(ability_path)) =
            op.attributes(self.db).get(&self.syms.attr_ability_ref)
        {
            // Extract ability name from the path (last component)
            let ability_name = ability_path
                .last()
                .copied()
                .unwrap_or_else(|| Symbol::new(self.db, "unknown"));

            // Create ability reference (with no type params for now)
            // TODO: Extract type parameters from the ability definition
            let ability = AbilityRef::simple(ability_name);

            // Create an effect row with this ability and merge it
            let effect = EffectRow::concrete([ability]);
            self.merge_effect(effect);
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
