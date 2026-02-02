//! Function-level type inference context.
//!
//! `FunctionInferenceContext` provides isolated state for type-checking a single
//! function body. Each function gets its own:
//! - Local variable bindings
//! - Type constraints
//! - Type variable counters (scoped to the function)
//!
//! UniVar IDs include the function name, making them globally unique across all
//! functions without needing a global counter.

use std::collections::HashMap;

use trunk_ir::Symbol;

use crate::ast::{
    CtorId, EffectRow, EffectVar, FuncDefId, LocalId, NodeId, Type, TypeKind, TypeScheme, UniVarId,
    UniVarSource,
};

use super::constraint::ConstraintSet;
use super::context::ModuleTypeEnv;
use super::subst;

/// Function-level type inference context.
///
/// Each function body is type-checked with its own `FunctionInferenceContext`,
/// which provides:
/// - Local variable type bindings (with scoping for lambdas and case arms)
/// - Fresh type/row variable generation (scoped to this function)
/// - Constraint collection (solved per-function)
/// - Node type recording (for TypedRef construction)
///
/// UniVar IDs are unique because they include the function name.
/// Module-level type information (function signatures, constructors, type defs)
/// is accessed via a reference to `ModuleTypeEnv`.
///
/// Scoping: Bindings are stored in a stack of scopes. When entering a new
/// scope (e.g., lambda body, case arm), push_scope() creates a new scope.
/// When exiting, pop_scope() removes it. Lookups search from innermost to
/// outermost scope.
pub struct FunctionInferenceContext<'a, 'db> {
    db: &'db dyn salsa::Database,

    /// Module-level type information (read-only).
    env: &'a ModuleTypeEnv<'db>,

    /// Function definition ID (used in UniVarSource for globally unique IDs).
    func_id: FuncDefId<'db>,

    /// Types of local variables (by LocalId), organized as a stack of scopes.
    /// The last element is the innermost (current) scope.
    local_scopes: Vec<HashMap<LocalId, Type<'db>>>,

    /// Types of local variables by name, organized as a stack of scopes.
    /// The last element is the innermost (current) scope.
    name_scopes: Vec<HashMap<Symbol, Type<'db>>>,

    /// Types of AST nodes (for TypedRef construction).
    node_types: HashMap<NodeId, Type<'db>>,

    /// Generated constraints for this function.
    constraints: ConstraintSet<'db>,

    /// Counter for fresh type variables (local to this function).
    next_type_var: u64,

    /// Counter for fresh effect row variables (local to this function).
    next_row_var: u64,

    /// Current accumulated effects.
    current_effect: EffectRow<'db>,
}

impl<'a, 'db> FunctionInferenceContext<'a, 'db> {
    /// Create a new function inference context.
    ///
    /// `func_id` is the function definition ID, which provides globally unique
    /// identification for UniVar IDs.
    pub fn new(
        db: &'db dyn salsa::Database,
        env: &'a ModuleTypeEnv<'db>,
        func_id: FuncDefId<'db>,
    ) -> Self {
        Self {
            db,
            env,
            func_id,
            // Start with one scope (the function's top-level scope)
            local_scopes: vec![HashMap::new()],
            name_scopes: vec![HashMap::new()],
            node_types: HashMap::new(),
            constraints: ConstraintSet::new(),
            next_type_var: 0,
            next_row_var: 0,
            current_effect: EffectRow::pure(db),
        }
    }

    /// Get the database.
    pub fn db(&self) -> &'db dyn salsa::Database {
        self.db
    }

    /// Get the module type environment.
    pub fn env(&self) -> &'a ModuleTypeEnv<'db> {
        self.env
    }

    // =========================================================================
    // Fresh variable generation
    // =========================================================================

    /// Generate a fresh type variable.
    ///
    /// UniVar IDs are globally unique because they include the function ID
    /// and a local index. This prevents collisions across different functions.
    pub fn fresh_type_var(&mut self) -> Type<'db> {
        let index = self.next_type_var;
        self.next_type_var += 1;
        let source = UniVarSource::FunctionLocal {
            func_id: self.func_id,
            index,
        };
        let id = UniVarId::new(self.db, source, 0);
        Type::new(self.db, TypeKind::UniVar { id })
    }

    /// Generate a fresh effect row variable.
    pub fn fresh_row_var(&mut self) -> EffectVar {
        let id = self.next_row_var;
        self.next_row_var += 1;
        EffectVar { id }
    }

    /// Generate a fresh effect row with only a variable.
    pub fn fresh_effect_row(&mut self) -> EffectRow<'db> {
        let var = self.fresh_row_var();
        EffectRow::open(self.db, var)
    }

    // =========================================================================
    // Scope management
    // =========================================================================

    /// Push a new scope. Call this when entering a lambda body or case arm.
    pub fn push_scope(&mut self) {
        self.local_scopes.push(HashMap::new());
        self.name_scopes.push(HashMap::new());
    }

    /// Pop the current scope. Call this when exiting a lambda body or case arm.
    pub fn pop_scope(&mut self) {
        // Never pop the last scope (function's top-level scope)
        if self.local_scopes.len() > 1 {
            self.local_scopes.pop();
            self.name_scopes.pop();
        }
    }

    /// Execute a closure within a new scope.
    ///
    /// This is a convenience method that ensures scope is properly pushed and popped.
    pub fn with_scope<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(&mut Self) -> R,
    {
        self.push_scope();
        let result = f(self);
        self.pop_scope();
        result
    }

    // =========================================================================
    // Local variable bindings
    // =========================================================================

    /// Bind a local variable to a type in the current scope.
    pub fn bind_local(&mut self, local: LocalId, ty: Type<'db>) {
        if let Some(scope) = self.local_scopes.last_mut() {
            scope.insert(local, ty);
        }
    }

    /// Look up the type of a local variable.
    ///
    /// Searches from innermost to outermost scope.
    pub fn lookup_local(&self, local: LocalId) -> Option<Type<'db>> {
        for scope in self.local_scopes.iter().rev() {
            if let Some(ty) = scope.get(&local) {
                return Some(*ty);
            }
        }
        None
    }

    /// Bind a local variable by name in the current scope.
    pub fn bind_local_by_name(&mut self, name: Symbol, ty: Type<'db>) {
        if let Some(scope) = self.name_scopes.last_mut() {
            scope.insert(name, ty);
        }
    }

    /// Look up a local variable by name.
    ///
    /// Searches from innermost to outermost scope.
    pub fn lookup_local_by_name(&self, name: Symbol) -> Option<Type<'db>> {
        for scope in self.name_scopes.iter().rev() {
            if let Some(ty) = scope.get(&name) {
                return Some(*ty);
            }
        }
        None
    }

    // =========================================================================
    // Node types (for TypedRef construction)
    // =========================================================================

    /// Record the type of an AST node.
    pub fn record_node_type(&mut self, node: NodeId, ty: Type<'db>) {
        self.node_types.insert(node, ty);
    }

    /// Get the type of an AST node.
    pub fn get_node_type(&self, node: NodeId) -> Option<Type<'db>> {
        self.node_types.get(&node).copied()
    }

    // =========================================================================
    // Module-level lookups (delegated to ModuleTypeEnv)
    // =========================================================================

    /// Look up a function's type scheme.
    pub fn lookup_function(&self, id: FuncDefId<'db>) -> Option<TypeScheme<'db>> {
        self.env.lookup_function(id)
    }

    /// Look up a constructor's type scheme.
    pub fn lookup_constructor(&self, id: CtorId<'db>) -> Option<TypeScheme<'db>> {
        self.env.lookup_constructor(id)
    }

    /// Look up a type definition.
    pub fn lookup_type_def(&self, name: Symbol) -> Option<TypeScheme<'db>> {
        self.env.lookup_type_def(name)
    }

    /// Instantiate a function's type scheme with fresh type variables.
    pub fn instantiate_function(&mut self, id: FuncDefId<'db>) -> Option<Type<'db>> {
        let scheme = self.lookup_function(id)?;
        Some(self.instantiate_scheme(scheme))
    }

    /// Instantiate a constructor's type scheme with fresh type variables.
    pub fn instantiate_constructor(&mut self, id: CtorId<'db>) -> Option<Type<'db>> {
        let scheme = self.lookup_constructor(id)?;
        Some(self.instantiate_scheme(scheme))
    }

    /// Instantiate a type scheme with fresh type variables.
    ///
    /// Replaces each `BoundVar { index: i }` with a fresh `UniVar`.
    pub fn instantiate_scheme(&mut self, scheme: TypeScheme<'db>) -> Type<'db> {
        let params = scheme.type_params(self.db);
        if params.is_empty() {
            return scheme.body(self.db);
        }

        // Generate fresh variables for each parameter
        let fresh_vars: Vec<Type<'db>> = (0..params.len()).map(|_| self.fresh_type_var()).collect();

        // Substitute bound variables with fresh variables
        self.substitute_bound_vars(scheme.body(self.db), &fresh_vars)
    }

    /// Substitute bound variables with given types.
    ///
    /// Panics if a BoundVar index is out of bounds.
    fn substitute_bound_vars(&self, ty: Type<'db>, args: &[Type<'db>]) -> Type<'db> {
        subst::substitute_bound_vars(self.db, ty, args).unwrap_or_else(|index, max| {
            panic!(
                "BoundVar index out of range: index={}, subst.len()={}",
                index, max
            )
        })
    }

    // =========================================================================
    // Constraints
    // =========================================================================

    /// Add a type equality constraint.
    pub fn constrain_eq(&mut self, t1: Type<'db>, t2: Type<'db>) {
        self.constraints.add_type_eq(t1, t2);
    }

    /// Add an effect row equality constraint.
    pub fn constrain_row_eq(&mut self, r1: EffectRow<'db>, r2: EffectRow<'db>) {
        self.constraints.add_row_eq(r1, r2);
    }

    /// Take the constraint set, leaving an empty set.
    pub fn take_constraints(&mut self) -> ConstraintSet<'db> {
        std::mem::take(&mut self.constraints)
    }

    // =========================================================================
    // Effect tracking
    // =========================================================================

    /// Get the current effect row.
    pub fn current_effect(&self) -> EffectRow<'db> {
        self.current_effect
    }

    /// Set the current effect row.
    pub fn set_current_effect(&mut self, effect: EffectRow<'db>) {
        self.current_effect = effect;
    }

    /// Merge an effect into the current effect row.
    ///
    /// Implements row-polymorphic effect union:
    /// - `{} ∪ ρ = ρ` (pure is identity)
    /// - `{A} ∪ {B} = {A, B}` (closed rows merge directly)
    /// - `{A | e1} ∪ {B | e2} = {A, B | e3}` with constraints:
    ///   - `e1 = {B | e3}`
    ///   - `e2 = {A | e3}`
    pub fn merge_effect(&mut self, effect: EffectRow<'db>) {
        use super::effect_row;

        let db = self.db;
        let current = self.current_effect;

        // Pure effect doesn't change anything
        if effect.is_pure(db) {
            return;
        }

        // If current is pure, just use the new effect
        if current.is_pure(db) {
            self.current_effect = effect;
            return;
        }

        match (current.rest(db), effect.rest(db)) {
            (None, None) | (Some(_), None) | (None, Some(_)) => {
                // At most one row is open: simple union
                self.current_effect = effect_row::union(db, current, effect, || {
                    unreachable!("fresh var not needed when at most one row is open")
                });
            }
            (Some(e1), Some(e2)) if e1 == e2 => {
                // Both open with the same rest variable: simple union, no constraints needed
                let mut effects = current.effects(db).clone();
                for e in effect.effects(db) {
                    if !effects.contains(e) {
                        effects.push(e.clone());
                    }
                }
                self.current_effect = EffectRow::new(db, effects, Some(e1));
            }
            (Some(e1), Some(e2)) => {
                // Both open with different rest variables: generate constraints
                let e3 = self.fresh_row_var();

                // e1 = {effect's effects | e3}
                let row_for_e1 = EffectRow::new(db, effect.effects(db).clone(), Some(e3));
                self.constrain_row_eq(EffectRow::open(db, e1), row_for_e1);

                // e2 = {current's effects | e3}
                let row_for_e2 = EffectRow::new(db, current.effects(db).clone(), Some(e3));
                self.constrain_row_eq(EffectRow::open(db, e2), row_for_e2);

                // Result: {current effects ∪ effect effects | e3}
                let mut effects = current.effects(db).clone();
                for e in effect.effects(db) {
                    if !effects.contains(e) {
                        effects.push(e.clone());
                    }
                }
                self.current_effect = EffectRow::new(db, effects, Some(e3));
            }
        }
    }

    // =========================================================================
    // Primitive types (convenience methods - delegate to env)
    // =========================================================================

    /// Create the Int type.
    pub fn int_type(&self) -> Type<'db> {
        self.env.int_type()
    }

    /// Create the Nat type.
    pub fn nat_type(&self) -> Type<'db> {
        self.env.nat_type()
    }

    /// Create the Float type.
    pub fn float_type(&self) -> Type<'db> {
        self.env.float_type()
    }

    /// Create the Bool type.
    pub fn bool_type(&self) -> Type<'db> {
        self.env.bool_type()
    }

    /// Create the String type.
    pub fn string_type(&self) -> Type<'db> {
        self.env.string_type()
    }

    /// Create the Bytes type.
    pub fn bytes_type(&self) -> Type<'db> {
        self.env.bytes_type()
    }

    /// Create the Rune type (Unicode code point).
    pub fn rune_type(&self) -> Type<'db> {
        self.env.rune_type()
    }

    /// Create the Nil (unit) type.
    pub fn nil_type(&self) -> Type<'db> {
        self.env.nil_type()
    }

    /// Create an error type.
    pub fn error_type(&self) -> Type<'db> {
        self.env.error_type()
    }

    /// Create a tuple type.
    pub fn tuple_type(&self, elements: Vec<Type<'db>>) -> Type<'db> {
        self.env.tuple_type(elements)
    }

    /// Create a function type.
    pub fn func_type(
        &self,
        params: Vec<Type<'db>>,
        result: Type<'db>,
        effect: EffectRow<'db>,
    ) -> Type<'db> {
        self.env.func_type(params, result, effect)
    }

    /// Create a named type.
    pub fn named_type(&self, name: Symbol, args: Vec<Type<'db>>) -> Type<'db> {
        self.env.named_type(name, args)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::TypeParam;
    use salsa_test_macros::salsa_test;
    use trunk_ir::SymbolVec;

    /// Create a type parameter with just a name.
    fn type_param(name: Symbol) -> TypeParam {
        TypeParam {
            name: Some(name),
            kind: None,
        }
    }

    #[salsa::tracked]
    fn test_fresh_type_var_per_function_inner<'db>(db: &'db dyn salsa::Database) -> bool {
        let env = ModuleTypeEnv::new(db);

        // Two separate FunctionInferenceContexts with different function IDs
        // should produce globally unique UniVar IDs
        let func_id1 = FuncDefId::new(db, SymbolVec::new(), Symbol::new("func1"));
        let func_id2 = FuncDefId::new(db, SymbolVec::new(), Symbol::new("func2"));
        let mut ctx1 = FunctionInferenceContext::new(db, &env, func_id1);
        let mut ctx2 = FunctionInferenceContext::new(db, &env, func_id2);

        let var1_1 = ctx1.fresh_type_var();
        let var1_2 = ctx1.fresh_type_var();
        let var2_1 = ctx2.fresh_type_var();
        let var2_2 = ctx2.fresh_type_var();

        // All variables should be different (globally unique due to function IDs)
        if var1_1 == var1_2 || var1_1 == var2_1 || var1_1 == var2_2 {
            return false;
        }
        if var1_2 == var2_1 || var1_2 == var2_2 {
            return false;
        }
        if var2_1 == var2_2 {
            return false;
        }

        true
    }

    #[salsa_test]
    fn test_fresh_type_var_per_function(db: &dyn salsa::Database) {
        assert!(
            test_fresh_type_var_per_function_inner(db),
            "Type variables should be globally unique across all function contexts"
        );
    }

    #[salsa::tracked]
    fn test_instantiate_scheme_inner<'db>(db: &'db dyn salsa::Database) -> bool {
        let mut env = ModuleTypeEnv::new(db);

        // Create a polymorphic function type: forall a. a -> a
        let name = Symbol::new("identity");
        let func_id = FuncDefId::new(db, SymbolVec::new(), name);

        let bound_var = Type::new(db, TypeKind::BoundVar { index: 0 });
        let effect = EffectRow::pure(db);
        let func_ty = Type::new(
            db,
            TypeKind::Func {
                params: vec![bound_var],
                result: bound_var,
                effect,
            },
        );
        let scheme = TypeScheme::new(db, vec![type_param(Symbol::new("a"))], func_ty);
        env.register_function(func_id, scheme);

        // Create a FunctionInferenceContext and instantiate the function
        let test_func_id = FuncDefId::new(db, SymbolVec::new(), Symbol::new("test_func"));
        let mut ctx = FunctionInferenceContext::new(db, &env, test_func_id);

        let ty1 = ctx.instantiate_function(func_id).unwrap();
        let ty2 = ctx.instantiate_function(func_id).unwrap();

        // Two instantiations should get different UniVars
        if ty1 == ty2 {
            return false;
        }

        // Both should be function types with UniVar params
        if let (
            TypeKind::Func {
                params: p1,
                result: r1,
                ..
            },
            TypeKind::Func {
                params: p2,
                result: r2,
                ..
            },
        ) = (ty1.kind(db), ty2.kind(db))
        {
            p1.len() == 1
                && p1[0] == *r1
                && p2[0] == *r2
                && p1[0] != p2[0] // Different UniVars
                && matches!(p1[0].kind(db), TypeKind::UniVar { .. })
        } else {
            false
        }
    }

    #[salsa_test]
    fn test_instantiate_scheme(db: &dyn salsa::Database) {
        assert!(
            test_instantiate_scheme_inner(db),
            "Each instantiation should produce fresh type variables"
        );
    }

    #[salsa_test]
    fn test_local_binding(db: &dyn salsa::Database) {
        let env = ModuleTypeEnv::new(db);
        let test_func_id = FuncDefId::new(db, SymbolVec::new(), Symbol::new("test_func"));
        let mut ctx = FunctionInferenceContext::new(db, &env, test_func_id);

        // Bind a local by LocalId
        let local_id = LocalId::new(1);
        let ty = ctx.int_type();
        ctx.bind_local(local_id, ty);

        assert_eq!(ctx.lookup_local(local_id), Some(ty));

        // Bind by name
        let name = Symbol::new("x");
        let ty2 = ctx.bool_type();
        ctx.bind_local_by_name(name, ty2);

        assert_eq!(ctx.lookup_local_by_name(name), Some(ty2));
    }

    #[salsa_test]
    fn test_constraints(db: &dyn salsa::Database) {
        let env = ModuleTypeEnv::new(db);
        let test_func_id = FuncDefId::new(db, SymbolVec::new(), Symbol::new("test_func"));
        let mut ctx = FunctionInferenceContext::new(db, &env, test_func_id);

        let var = ctx.fresh_type_var();
        let int_ty = ctx.int_type();

        ctx.constrain_eq(var, int_ty);

        let constraints = ctx.take_constraints();
        assert_eq!(constraints.len(), 1);

        // After taking, constraints should be empty
        let empty = ctx.take_constraints();
        assert!(empty.is_empty());
    }

    #[salsa::tracked]
    fn test_instantiate_constructor_inner<'db>(db: &'db dyn salsa::Database) -> bool {
        let mut env = ModuleTypeEnv::new(db);

        // Create a polymorphic constructor: forall a. a -> Option(a)
        let type_name = Symbol::new("Option");
        let ctor_id = CtorId::new(db, SymbolVec::new(), type_name);

        let bound_var = Type::new(db, TypeKind::BoundVar { index: 0 });
        let result_ty = Type::new(
            db,
            TypeKind::Named {
                name: type_name,
                args: vec![bound_var],
            },
        );
        let effect = EffectRow::pure(db);
        let func_ty = Type::new(
            db,
            TypeKind::Func {
                params: vec![bound_var],
                result: result_ty,
                effect,
            },
        );
        let scheme = TypeScheme::new(db, vec![type_param(Symbol::new("a"))], func_ty);
        env.register_constructor(ctor_id, scheme);

        // Create a FunctionInferenceContext and instantiate the constructor
        let test_func_id = FuncDefId::new(db, SymbolVec::new(), Symbol::new("test_func"));
        let mut ctx = FunctionInferenceContext::new(db, &env, test_func_id);

        let ty1 = ctx.instantiate_constructor(ctor_id).unwrap();
        let ty2 = ctx.instantiate_constructor(ctor_id).unwrap();

        // Two instantiations should get different UniVars
        if ty1 == ty2 {
            return false;
        }

        // Both should be function types with UniVar params
        if let (
            TypeKind::Func {
                params: p1,
                result: r1,
                ..
            },
            TypeKind::Func {
                params: p2,
                result: r2,
                ..
            },
        ) = (ty1.kind(db), ty2.kind(db))
        {
            // Verify param is UniVar and same as result's type arg
            if !matches!(p1[0].kind(db), TypeKind::UniVar { .. }) {
                return false;
            }
            if !matches!(p2[0].kind(db), TypeKind::UniVar { .. }) {
                return false;
            }

            // Results should be Named types with the UniVar as argument
            let r1_ok = if let TypeKind::Named { name, args } = r1.kind(db) {
                *name == type_name && args.len() == 1 && args[0] == p1[0]
            } else {
                false
            };
            let r2_ok = if let TypeKind::Named { name, args } = r2.kind(db) {
                *name == type_name && args.len() == 1 && args[0] == p2[0]
            } else {
                false
            };

            // Different UniVars for different instantiations
            r1_ok && r2_ok && p1[0] != p2[0]
        } else {
            false
        }
    }

    #[salsa_test]
    fn test_instantiate_constructor(db: &dyn salsa::Database) {
        assert!(
            test_instantiate_constructor_inner(db),
            "Each constructor instantiation should produce fresh type variables"
        );
    }
}

#[cfg(test)]
mod merge_effect_tests {
    use super::*;
    use crate::ast::AbilityId;
    use crate::typeck::constraint::Constraint;
    use crate::typeck::effect_row::simple_effect;
    use salsa_test_macros::salsa_test;
    use trunk_ir::SymbolVec;

    /// Helper to create a simple AbilityId with empty module path
    fn test_ability_id<'db>(db: &'db dyn salsa::Database, name: &str) -> AbilityId<'db> {
        AbilityId::new(db, SymbolVec::new(), Symbol::from_dynamic(name))
    }

    #[salsa_test]
    fn merge_pure_rows(db: &dyn salsa::Database) {
        let env = ModuleTypeEnv::new(db);
        let func_id = FuncDefId::new(db, SymbolVec::new(), Symbol::new("test"));
        let mut ctx = FunctionInferenceContext::new(db, &env, func_id);

        let pure = EffectRow::pure(db);

        // Pure + Pure = Pure
        ctx.set_current_effect(pure);
        ctx.merge_effect(pure);
        assert!(ctx.current_effect().is_pure(db));

        // No constraints generated
        assert!(ctx.take_constraints().is_empty());
    }

    #[salsa_test]
    fn merge_concrete_effects(db: &dyn salsa::Database) {
        let env = ModuleTypeEnv::new(db);
        let func_id = FuncDefId::new(db, SymbolVec::new(), Symbol::new("test"));
        let mut ctx = FunctionInferenceContext::new(db, &env, func_id);

        let console = simple_effect(db, test_ability_id(db, "Console"));
        let state = simple_effect(db, test_ability_id(db, "State"));

        let row1 = EffectRow::single(db, console.clone());
        let row2 = EffectRow::single(db, state.clone());

        // {Console} + {State} = {Console, State}
        ctx.set_current_effect(row1);
        ctx.merge_effect(row2);

        let result = ctx.current_effect();
        let effects = result.effects(db);
        assert_eq!(effects.len(), 2);
        assert!(effects.contains(&console));
        assert!(effects.contains(&state));
        assert!(result.rest(db).is_none()); // Still closed

        // No constraints for closed rows
        assert!(ctx.take_constraints().is_empty());
    }

    #[salsa_test]
    fn merge_open_rows(db: &dyn salsa::Database) {
        let env = ModuleTypeEnv::new(db);
        let func_id = FuncDefId::new(db, SymbolVec::new(), Symbol::new("test"));
        let mut ctx = FunctionInferenceContext::new(db, &env, func_id);

        let console = simple_effect(db, test_ability_id(db, "Console"));
        let state = simple_effect(db, test_ability_id(db, "State"));

        let e1 = EffectVar { id: 1 };
        let e2 = EffectVar { id: 2 };

        let row1 = EffectRow::new(db, vec![console.clone()], Some(e1));
        let row2 = EffectRow::new(db, vec![state.clone()], Some(e2));

        // {Console | e1} + {State | e2} = {Console, State | e3}
        ctx.set_current_effect(row1);
        ctx.merge_effect(row2);

        let result = ctx.current_effect();
        let effects = result.effects(db);
        assert_eq!(effects.len(), 2);
        assert!(effects.contains(&console));
        assert!(effects.contains(&state));

        // Result should have a fresh row variable (e3)
        let e3 = result.rest(db).expect("Should have a row variable");
        assert_ne!(e3, e1);
        assert_ne!(e3, e2);

        // Should have generated 2 constraints
        let constraints = ctx.take_constraints();
        assert_eq!(constraints.len(), 2);

        // Verify constraint structure
        for constraint in constraints.constraints() {
            match constraint {
                Constraint::RowEq(lhs, rhs) => {
                    // LHS should be an open row with e1 or e2
                    let lhs_rest = lhs.rest(db);
                    assert!(lhs_rest == Some(e1) || lhs_rest == Some(e2));
                    // RHS should contain the other effect with e3 as rest
                    assert_eq!(rhs.rest(db), Some(e3));
                }
                _ => panic!("Expected RowEq constraint"),
            }
        }
    }

    #[salsa_test]
    fn merge_deduplicates(db: &dyn salsa::Database) {
        let env = ModuleTypeEnv::new(db);
        let func_id = FuncDefId::new(db, SymbolVec::new(), Symbol::new("test"));
        let mut ctx = FunctionInferenceContext::new(db, &env, func_id);

        let console = simple_effect(db, test_ability_id(db, "Console"));

        let row1 = EffectRow::single(db, console.clone());
        let row2 = EffectRow::single(db, console.clone());

        // {Console} + {Console} = {Console}
        ctx.set_current_effect(row1);
        ctx.merge_effect(row2);

        let result = ctx.current_effect();
        let effects = result.effects(db);
        assert_eq!(effects.len(), 1);
        assert!(effects.contains(&console));
    }

    #[salsa_test]
    fn merge_closed_with_open(db: &dyn salsa::Database) {
        let env = ModuleTypeEnv::new(db);
        let func_id = FuncDefId::new(db, SymbolVec::new(), Symbol::new("test"));
        let mut ctx = FunctionInferenceContext::new(db, &env, func_id);

        let console = simple_effect(db, test_ability_id(db, "Console"));
        let state = simple_effect(db, test_ability_id(db, "State"));

        let e1 = EffectVar { id: 1 };

        let closed_row = EffectRow::single(db, console.clone());
        let open_row = EffectRow::new(db, vec![state.clone()], Some(e1));

        // {Console} + {State | e1} = {Console, State | e1}
        ctx.set_current_effect(closed_row);
        ctx.merge_effect(open_row);

        let result = ctx.current_effect();
        let effects = result.effects(db);
        assert_eq!(effects.len(), 2);
        assert!(effects.contains(&console));
        assert!(effects.contains(&state));
        assert_eq!(result.rest(db), Some(e1)); // Preserves the row variable

        // No constraints needed when only one is open
        assert!(ctx.take_constraints().is_empty());
    }

    #[salsa_test]
    fn merge_open_rows_same_rest_var(db: &dyn salsa::Database) {
        let env = ModuleTypeEnv::new(db);
        let func_id = FuncDefId::new(db, SymbolVec::new(), Symbol::new("test"));
        let mut ctx = FunctionInferenceContext::new(db, &env, func_id);

        let console = simple_effect(db, test_ability_id(db, "Console"));
        let state = simple_effect(db, test_ability_id(db, "State"));

        // Both rows share the same rest variable
        let shared_var = EffectVar { id: 1 };

        let row1 = EffectRow::new(db, vec![console.clone()], Some(shared_var));
        let row2 = EffectRow::new(db, vec![state.clone()], Some(shared_var));

        // {Console | e1} + {State | e1} = {Console, State | e1}
        // When rest vars are the same, we optimize by skipping constraint generation
        ctx.set_current_effect(row1);
        ctx.merge_effect(row2);

        let result = ctx.current_effect();
        let effects = result.effects(db);
        assert_eq!(effects.len(), 2);
        assert!(effects.contains(&console));
        assert!(effects.contains(&state));

        // Result should preserve the shared row variable (no fresh var created)
        assert_eq!(result.rest(db), Some(shared_var));

        // No constraints should be generated for same rest var optimization
        assert!(ctx.take_constraints().is_empty());
    }

    #[salsa_test]
    fn merge_open_rows_same_rest_var_with_overlap(db: &dyn salsa::Database) {
        let env = ModuleTypeEnv::new(db);
        let func_id = FuncDefId::new(db, SymbolVec::new(), Symbol::new("test"));
        let mut ctx = FunctionInferenceContext::new(db, &env, func_id);

        let console = simple_effect(db, test_ability_id(db, "Console"));
        let state = simple_effect(db, test_ability_id(db, "State"));

        // Both rows share the same rest variable and have overlapping effects
        let shared_var = EffectVar { id: 1 };

        let row1 = EffectRow::new(db, vec![console.clone(), state.clone()], Some(shared_var));
        let row2 = EffectRow::new(db, vec![state.clone()], Some(shared_var));

        // {Console, State | e1} + {State | e1} = {Console, State | e1}
        ctx.set_current_effect(row1);
        ctx.merge_effect(row2);

        let result = ctx.current_effect();
        let effects = result.effects(db);
        assert_eq!(effects.len(), 2); // Deduplication should work
        assert!(effects.contains(&console));
        assert!(effects.contains(&state));
        assert_eq!(result.rest(db), Some(shared_var));

        // No constraints for same rest var
        assert!(ctx.take_constraints().is_empty());
    }
}
