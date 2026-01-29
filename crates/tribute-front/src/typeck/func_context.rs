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
    CtorId, Effect, EffectRow, EffectVar, FuncDefId, LocalId, NodeId, Type, TypeKind, TypeScheme,
    UniVarId, UniVarSource,
};

use super::constraint::ConstraintSet;
use super::context::ModuleTypeEnv;

/// Function-level type inference context.
///
/// Each function body is type-checked with its own `FunctionInferenceContext`,
/// which provides:
/// - Local variable type bindings
/// - Fresh type/row variable generation (scoped to this function)
/// - Constraint collection (solved per-function)
/// - Node type recording (for TypedRef construction)
///
/// UniVar IDs are unique because they include the function name.
/// Module-level type information (function signatures, constructors, type defs)
/// is accessed via a reference to `ModuleTypeEnv`.
pub struct FunctionInferenceContext<'a, 'db> {
    db: &'db dyn salsa::Database,

    /// Module-level type information (read-only).
    env: &'a ModuleTypeEnv<'db>,

    /// Qualified function name (used in UniVarSource for globally unique IDs).
    func_name: Symbol,

    /// Types of local variables (by LocalId).
    local_types: HashMap<LocalId, Type<'db>>,

    /// Types of local variables by name (for parameters without LocalId).
    local_types_by_name: HashMap<Symbol, Type<'db>>,

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
    /// `func_name` should be the qualified function name (e.g., "module::func")
    /// to ensure UniVar IDs are globally unique.
    pub fn new(
        db: &'db dyn salsa::Database,
        env: &'a ModuleTypeEnv<'db>,
        func_name: Symbol,
    ) -> Self {
        Self {
            db,
            env,
            func_name,
            local_types: HashMap::new(),
            local_types_by_name: HashMap::new(),
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
    /// UniVar IDs are globally unique because they include the function name
    /// and a local index. This prevents collisions across different functions.
    pub fn fresh_type_var(&mut self) -> Type<'db> {
        let index = self.next_type_var;
        self.next_type_var += 1;
        let source = UniVarSource::FunctionLocal {
            func_name: self.func_name,
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
    // Local variable bindings
    // =========================================================================

    /// Bind a local variable to a type.
    pub fn bind_local(&mut self, local: LocalId, ty: Type<'db>) {
        self.local_types.insert(local, ty);
    }

    /// Look up the type of a local variable.
    pub fn lookup_local(&self, local: LocalId) -> Option<Type<'db>> {
        self.local_types.get(&local).copied()
    }

    /// Bind a local variable by name (for parameters without LocalId).
    pub fn bind_local_by_name(&mut self, name: Symbol, ty: Type<'db>) {
        self.local_types_by_name.insert(name, ty);
    }

    /// Look up a local variable by name.
    pub fn lookup_local_by_name(&self, name: Symbol) -> Option<Type<'db>> {
        self.local_types_by_name.get(&name).copied()
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
    fn substitute_bound_vars(&self, ty: Type<'db>, subst: &[Type<'db>]) -> Type<'db> {
        match ty.kind(self.db) {
            TypeKind::BoundVar { index } => subst.get(*index as usize).copied().unwrap_or(ty),
            TypeKind::Named { name, args } => {
                let args = args
                    .iter()
                    .map(|a| self.substitute_bound_vars(*a, subst))
                    .collect();
                Type::new(self.db, TypeKind::Named { name: *name, args })
            }
            TypeKind::Func {
                params,
                result,
                effect,
            } => {
                let params = params
                    .iter()
                    .map(|p| self.substitute_bound_vars(*p, subst))
                    .collect();
                let result = self.substitute_bound_vars(*result, subst);
                let effect = self.substitute_effect_row(*effect, subst);
                Type::new(
                    self.db,
                    TypeKind::Func {
                        params,
                        result,
                        effect,
                    },
                )
            }
            TypeKind::Tuple(elements) => {
                let elements = elements
                    .iter()
                    .map(|e| self.substitute_bound_vars(*e, subst))
                    .collect();
                Type::new(self.db, TypeKind::Tuple(elements))
            }
            TypeKind::App { ctor, args } => {
                let ctor = self.substitute_bound_vars(*ctor, subst);
                let args = args
                    .iter()
                    .map(|a| self.substitute_bound_vars(*a, subst))
                    .collect();
                Type::new(self.db, TypeKind::App { ctor, args })
            }
            // Primitive types and other type variables are unchanged
            _ => ty,
        }
    }

    /// Substitute bound variables within an effect row.
    fn substitute_effect_row(&self, row: EffectRow<'db>, subst: &[Type<'db>]) -> EffectRow<'db> {
        let effects = row.effects(self.db);
        let mut changed = false;

        let new_effects: Vec<_> = effects
            .iter()
            .map(|effect| {
                let new_args: Vec<_> = effect
                    .args
                    .iter()
                    .map(|a| self.substitute_bound_vars(*a, subst))
                    .collect();
                if new_args != effect.args {
                    changed = true;
                }
                Effect {
                    name: effect.name,
                    args: new_args,
                }
            })
            .collect();

        if changed {
            EffectRow::new(self.db, new_effects, row.rest(self.db))
        } else {
            row
        }
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
    pub fn merge_effect(&mut self, effect: EffectRow<'db>) {
        // TODO: Implement effect row union
        // For now, just replace
        if !effect.is_pure(self.db) {
            self.current_effect = effect;
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

        // Two separate FunctionInferenceContexts with different function names
        // should produce globally unique UniVar IDs
        let mut ctx1 = FunctionInferenceContext::new(db, &env, Symbol::new("func1"));
        let mut ctx2 = FunctionInferenceContext::new(db, &env, Symbol::new("func2"));

        let var1_1 = ctx1.fresh_type_var();
        let var1_2 = ctx1.fresh_type_var();
        let var2_1 = ctx2.fresh_type_var();
        let var2_2 = ctx2.fresh_type_var();

        // All variables should be different (globally unique due to function names)
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
        let func_id = FuncDefId::new(db, name);

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
        let mut ctx = FunctionInferenceContext::new(db, &env, Symbol::new("test_func"));

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
        let mut ctx = FunctionInferenceContext::new(db, &env, Symbol::new("test_func"));

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
        let mut ctx = FunctionInferenceContext::new(db, &env, Symbol::new("test_func"));

        let var = ctx.fresh_type_var();
        let int_ty = ctx.int_type();

        ctx.constrain_eq(var, int_ty);

        let constraints = ctx.take_constraints();
        assert_eq!(constraints.len(), 1);

        // After taking, constraints should be empty
        let empty = ctx.take_constraints();
        assert!(empty.is_empty());
    }
}
