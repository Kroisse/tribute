//! Type checking context.
//!
//! The context tracks type bindings, function signatures, and other information
//! needed during type checking.

use std::collections::HashMap;

use trunk_ir::Symbol;

use crate::ast::{
    CtorId, EffectRow, EffectVar, FuncDefId, LocalId, NodeId, Type, TypeKind, TypeScheme, UniVarId,
    UniVarSource,
};

use super::constraint::ConstraintSet;

/// Type checking context.
///
/// Tracks type information during type checking:
/// - Local variable types
/// - Function signatures
/// - Type definitions (structs, enums)
/// - Current effect accumulation
pub struct TypeContext<'db> {
    db: &'db dyn salsa::Database,

    /// Types of local variables.
    local_types: HashMap<LocalId, Type<'db>>,

    /// Types of local variables by name (temporary workaround until ParamDecl has LocalId).
    local_types_by_name: HashMap<Symbol, Type<'db>>,

    /// Types of AST nodes (for TypedRef construction).
    node_types: HashMap<NodeId, Type<'db>>,

    /// Function signatures (polymorphic).
    function_types: HashMap<FuncDefId<'db>, TypeScheme<'db>>,

    /// Constructor types.
    constructor_types: HashMap<CtorId<'db>, TypeScheme<'db>>,

    /// Type definitions (struct/enum names to their types).
    type_defs: HashMap<Symbol, TypeScheme<'db>>,

    /// Generated constraints.
    constraints: ConstraintSet<'db>,

    /// Counter for fresh type variables.
    next_type_var: u64,

    /// Counter for fresh effect row variables.
    next_row_var: u64,

    /// Current accumulated effects.
    current_effect: EffectRow<'db>,
}

impl<'db> TypeContext<'db> {
    /// Create a new type context.
    pub fn new(db: &'db dyn salsa::Database) -> Self {
        Self {
            db,
            local_types: HashMap::new(),
            local_types_by_name: HashMap::new(),
            node_types: HashMap::new(),
            function_types: HashMap::new(),
            constructor_types: HashMap::new(),
            type_defs: HashMap::new(),
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

    // =========================================================================
    // Fresh variables
    // =========================================================================

    /// Generate a fresh type variable.
    ///
    /// This creates an anonymous type variable for local inference (lambdas, let bindings, etc.).
    /// For polymorphic instantiation, use `instantiate_function` or `instantiate_constructor`.
    pub fn fresh_type_var(&mut self) -> Type<'db> {
        let counter = self.next_type_var;
        self.next_type_var += 1;
        let source = UniVarSource::Anonymous(counter);
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
    // Local variable types
    // =========================================================================

    /// Bind a local variable to a type.
    pub fn bind_local(&mut self, local: LocalId, ty: Type<'db>) {
        self.local_types.insert(local, ty);
    }

    /// Look up the type of a local variable.
    pub fn lookup_local(&self, local: LocalId) -> Option<Type<'db>> {
        self.local_types.get(&local).copied()
    }

    /// Bind a local variable by name (temporary workaround until ParamDecl has LocalId).
    ///
    /// This is used for function parameters where we only have the name available.
    /// Once ParamDecl includes LocalId, this method should be removed in favor of `bind_local`.
    pub fn bind_local_by_name(&mut self, name: Symbol, ty: Type<'db>) {
        self.local_types_by_name.insert(name, ty);
    }

    /// Look up a local variable by name.
    ///
    /// This is used for function parameters where we only have the name available.
    pub fn lookup_local_by_name(&self, name: Symbol) -> Option<Type<'db>> {
        self.local_types_by_name.get(&name).copied()
    }

    // =========================================================================
    // Node types (for TypedRef)
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
    // Function signatures
    // =========================================================================

    /// Register a function's type scheme.
    pub fn register_function(&mut self, id: FuncDefId<'db>, scheme: TypeScheme<'db>) {
        self.function_types.insert(id, scheme);
    }

    /// Look up a function's type scheme.
    pub fn lookup_function(&self, id: FuncDefId<'db>) -> Option<TypeScheme<'db>> {
        self.function_types.get(&id).copied()
    }

    /// Instantiate a function's type scheme with fresh type variables.
    ///
    /// Uses deterministic UniVar IDs based on the function ID, so repeated calls
    /// for the same function return the same instantiated type.
    pub fn instantiate_function(&self, id: FuncDefId<'db>) -> Option<Type<'db>> {
        let scheme = self.lookup_function(id)?;
        let ty = self.instantiate_scheme_with_source(scheme, UniVarSource::Function(id));
        Some(ty)
    }

    // =========================================================================
    // Constructor types
    // =========================================================================

    /// Register a constructor's type scheme.
    pub fn register_constructor(&mut self, id: CtorId<'db>, scheme: TypeScheme<'db>) {
        self.constructor_types.insert(id, scheme);
    }

    /// Look up a constructor's type scheme.
    pub fn lookup_constructor(&self, id: CtorId<'db>) -> Option<TypeScheme<'db>> {
        self.constructor_types.get(&id).copied()
    }

    /// Instantiate a constructor's type scheme with fresh type variables.
    ///
    /// Uses deterministic UniVar IDs based on the constructor ID, so repeated calls
    /// for the same constructor return the same instantiated type.
    pub fn instantiate_constructor(&self, id: CtorId<'db>) -> Option<Type<'db>> {
        let scheme = self.lookup_constructor(id)?;
        let ty = self.instantiate_scheme_with_source(scheme, UniVarSource::Constructor(id));
        Some(ty)
    }

    // =========================================================================
    // Type definitions
    // =========================================================================

    /// Register a type definition.
    pub fn register_type_def(&mut self, name: Symbol, scheme: TypeScheme<'db>) {
        self.type_defs.insert(name, scheme);
    }

    /// Look up a type definition.
    pub fn lookup_type_def(&self, name: Symbol) -> Option<TypeScheme<'db>> {
        self.type_defs.get(&name).copied()
    }

    // =========================================================================
    // Type scheme instantiation
    // =========================================================================

    /// Instantiate a type scheme with fresh type variables.
    ///
    /// Replaces each `BoundVar { index: i }` with a fresh `UniVar`.
    /// Uses anonymous UniVar source - for caching, use `instantiate_function` or
    /// `instantiate_constructor` instead.
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

    /// Instantiate a type scheme with deterministic type variables.
    ///
    /// Creates UniVars with the given source, ensuring repeated calls with the same
    /// source produce identical types (enabling Salsa caching via interning).
    fn instantiate_scheme_with_source(
        &self,
        scheme: TypeScheme<'db>,
        source: UniVarSource<'db>,
    ) -> Type<'db> {
        let params = scheme.type_params(self.db);
        if params.is_empty() {
            return scheme.body(self.db);
        }

        // Generate deterministic UniVars based on source and index
        let fresh_vars: Vec<Type<'db>> = (0..params.len())
            .map(|i| {
                let id = UniVarId::new(self.db, source, i as u32);
                Type::new(self.db, TypeKind::UniVar { id })
            })
            .collect();

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
                crate::ast::Effect {
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

    /// Take the constraint set.
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
    // Primitive types
    // =========================================================================

    /// Create the Int type.
    pub fn int_type(&self) -> Type<'db> {
        Type::new(self.db, TypeKind::Int)
    }

    /// Create the Nat type.
    pub fn nat_type(&self) -> Type<'db> {
        Type::new(self.db, TypeKind::Nat)
    }

    /// Create the Float type.
    pub fn float_type(&self) -> Type<'db> {
        Type::new(self.db, TypeKind::Float)
    }

    /// Create the Bool type.
    pub fn bool_type(&self) -> Type<'db> {
        Type::new(self.db, TypeKind::Bool)
    }

    /// Create the String type.
    pub fn string_type(&self) -> Type<'db> {
        Type::new(self.db, TypeKind::String)
    }

    /// Create the Bytes type.
    pub fn bytes_type(&self) -> Type<'db> {
        Type::new(self.db, TypeKind::Bytes)
    }

    /// Create the Rune type (Unicode code point).
    pub fn rune_type(&self) -> Type<'db> {
        Type::new(self.db, TypeKind::Rune)
    }

    /// Create the Nil (unit) type.
    pub fn nil_type(&self) -> Type<'db> {
        Type::new(self.db, TypeKind::Nil)
    }

    /// Create an error type.
    pub fn error_type(&self) -> Type<'db> {
        Type::new(self.db, TypeKind::Error)
    }

    /// Create a tuple type.
    pub fn tuple_type(&self, elements: Vec<Type<'db>>) -> Type<'db> {
        Type::new(self.db, TypeKind::Tuple(elements))
    }

    /// Create a function type.
    pub fn func_type(
        &self,
        params: Vec<Type<'db>>,
        result: Type<'db>,
        effect: EffectRow<'db>,
    ) -> Type<'db> {
        Type::new(
            self.db,
            TypeKind::Func {
                params,
                result,
                effect,
            },
        )
    }

    /// Create a named type.
    pub fn named_type(&self, name: Symbol, args: Vec<Type<'db>>) -> Type<'db> {
        Type::new(self.db, TypeKind::Named { name, args })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Effect, TypeParam, TypeScheme};
    use salsa_test_macros::salsa_test;

    /// Create a type parameter with just a name.
    fn type_param(name: Symbol) -> TypeParam {
        TypeParam {
            name: Some(name),
            kind: None,
        }
    }

    /// Tracked helper for function caching test.
    #[salsa::tracked]
    fn test_function_caching_inner<'db>(db: &'db dyn salsa::Database) -> bool {
        let mut ctx = TypeContext::new(db);

        // Create a polymorphic function type: forall a. a -> a
        let name = Symbol::new("identity");
        let func_id = FuncDefId::new(db, name);

        // BoundVar(0) -> BoundVar(0)
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
        ctx.register_function(func_id, scheme);

        // First instantiation
        let ty1 = ctx.instantiate_function(func_id).unwrap();

        // Second instantiation should return the same type (cached)
        let ty2 = ctx.instantiate_function(func_id).unwrap();

        // Both should be the same type
        if ty1 != ty2 {
            return false;
        }

        // Verify it's a function type with UniVar
        if let TypeKind::Func { params, result, .. } = ty1.kind(db) {
            params.len() == 1
                && params[0] == *result
                && matches!(params[0].kind(db), TypeKind::UniVar { .. })
        } else {
            false
        }
    }

    #[salsa_test]
    fn test_instantiate_function_caching(db: &dyn salsa::Database) {
        assert!(
            test_function_caching_inner(db),
            "Function instantiation should return cached type on second call"
        );
    }

    /// Tracked helper for constructor caching test.
    #[salsa::tracked]
    fn test_constructor_caching_inner<'db>(db: &'db dyn salsa::Database) -> bool {
        let mut ctx = TypeContext::new(db);

        // Create a polymorphic constructor type: forall a. a -> Option a
        let type_name = Symbol::new("Option");
        let ctor_id = CtorId::new(db, type_name);

        // a -> Option(a)
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
        ctx.register_constructor(ctor_id, scheme);

        // First instantiation
        let ty1 = ctx.instantiate_constructor(ctor_id).unwrap();

        // Second instantiation should return the same type (cached)
        let ty2 = ctx.instantiate_constructor(ctor_id).unwrap();

        // Both should be the same type
        if ty1 != ty2 {
            return false;
        }

        // Verify it's a function type
        if let TypeKind::Func { params, result, .. } = ty1.kind(db) {
            if params.len() != 1 {
                return false;
            }
            // Result should be Named type with the same UniVar as param
            if let TypeKind::Named { name, args } = result.kind(db) {
                *name == type_name && args.len() == 1 && args[0] == params[0]
            } else {
                false
            }
        } else {
            false
        }
    }

    #[salsa_test]
    fn test_instantiate_constructor_caching(db: &dyn salsa::Database) {
        assert!(
            test_constructor_caching_inner(db),
            "Constructor instantiation should return cached type on second call"
        );
    }

    /// Tracked helper for different functions test.
    #[salsa::tracked]
    fn test_different_functions_inner<'db>(db: &'db dyn salsa::Database) -> bool {
        let mut ctx = TypeContext::new(db);

        // Create two different functions with the same scheme
        let name1 = Symbol::new("f1");
        let name2 = Symbol::new("f2");
        let func_id1 = FuncDefId::new(db, name1);
        let func_id2 = FuncDefId::new(db, name2);

        // forall a. a -> a
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
        ctx.register_function(func_id1, scheme);
        ctx.register_function(func_id2, scheme);

        // Instantiate both
        let ty1 = ctx.instantiate_function(func_id1).unwrap();
        let ty2 = ctx.instantiate_function(func_id2).unwrap();

        // They should have different UniVars
        if let (
            TypeKind::Func {
                params: params1, ..
            },
            TypeKind::Func {
                params: params2, ..
            },
        ) = (ty1.kind(db), ty2.kind(db))
        {
            // The UniVar IDs should be different
            params1[0] != params2[0]
        } else {
            false
        }
    }

    #[salsa_test]
    fn test_different_functions_get_different_types(db: &dyn salsa::Database) {
        assert!(
            test_different_functions_inner(db),
            "Different functions should get different type variable instantiations"
        );
    }

    #[salsa_test]
    fn test_lookup_local_fallback_to_name(db: &dyn salsa::Database) {
        let mut ctx = TypeContext::new(db);

        // Bind a local by name only
        let name = Symbol::new("x");
        let ty = ctx.int_type();
        ctx.bind_local_by_name(name, ty);

        // Lookup by LocalId should fail, but by name should succeed
        let local_id = LocalId::new(1);
        assert!(ctx.lookup_local(local_id).is_none());
        assert_eq!(ctx.lookup_local_by_name(name), Some(ty));
    }

    #[salsa_test]
    fn test_lookup_local_prefers_local_id(db: &dyn salsa::Database) {
        let mut ctx = TypeContext::new(db);

        // Bind both by LocalId and by name with different types
        let name = Symbol::new("x");
        let local_id = LocalId::new(1);
        let ty_by_id = ctx.int_type();
        let ty_by_name = ctx.string_type();

        ctx.bind_local(local_id, ty_by_id);
        ctx.bind_local_by_name(name, ty_by_name);

        // lookup_local should return the type bound by LocalId
        assert_eq!(ctx.lookup_local(local_id), Some(ty_by_id));
        // lookup_local_by_name should return the type bound by name
        assert_eq!(ctx.lookup_local_by_name(name), Some(ty_by_name));
    }

    // =========================================================================
    // Effect row substitution tests
    // =========================================================================

    #[salsa_test]
    fn test_substitute_effect_row_with_bound_var(db: &dyn salsa::Database) {
        // forall a. fn() ->{State(a)} a
        // After instantiation, the BoundVar(0) inside the effect row should be
        // replaced with a fresh UniVar, same as the result type.
        let mut ctx = TypeContext::new(db);

        let bound_var = Type::new(db, TypeKind::BoundVar { index: 0 });

        // Effect row: {State(BoundVar(0))}
        let effect = EffectRow::new(
            db,
            vec![Effect {
                name: Symbol::new("State"),
                args: vec![bound_var],
            }],
            None,
        );

        // fn() ->{State(a)} a
        let func_ty = Type::new(
            db,
            TypeKind::Func {
                params: vec![],
                result: bound_var,
                effect,
            },
        );

        let scheme = TypeScheme::new(db, vec![type_param(Symbol::new("a"))], func_ty);
        let instantiated = ctx.instantiate_scheme(scheme);

        if let TypeKind::Func { result, effect, .. } = instantiated.kind(db) {
            // result should be a UniVar
            assert!(
                matches!(result.kind(db), TypeKind::UniVar { .. }),
                "Expected UniVar for result, got {:?}",
                result.kind(db)
            );

            // The effect row's State arg should be the same UniVar as result
            let effects = effect.effects(db);
            assert_eq!(effects.len(), 1);
            assert_eq!(effects[0].name, Symbol::new("State"));
            assert_eq!(effects[0].args.len(), 1);
            assert_eq!(
                effects[0].args[0], *result,
                "Effect arg should be the same UniVar as result type"
            );
        } else {
            panic!("Expected Func type, got {:?}", instantiated.kind(db));
        }
    }

    #[salsa_test]
    fn test_substitute_effect_row_no_bound_vars(db: &dyn salsa::Database) {
        // forall a. fn(a) ->{IO} a
        // The effect row has no BoundVars, so it should be unchanged after instantiation.
        let mut ctx = TypeContext::new(db);

        let bound_var = Type::new(db, TypeKind::BoundVar { index: 0 });

        let effect = EffectRow::new(
            db,
            vec![Effect {
                name: Symbol::new("IO"),
                args: vec![],
            }],
            None,
        );

        let func_ty = Type::new(
            db,
            TypeKind::Func {
                params: vec![bound_var],
                result: bound_var,
                effect,
            },
        );

        let scheme = TypeScheme::new(db, vec![type_param(Symbol::new("a"))], func_ty);
        let instantiated = ctx.instantiate_scheme(scheme);

        if let TypeKind::Func {
            effect: inst_effect,
            ..
        } = instantiated.kind(db)
        {
            // Effect row should be identical (no substitution needed)
            assert_eq!(
                *inst_effect, effect,
                "Effect row without BoundVars should be unchanged"
            );
        } else {
            panic!("Expected Func type, got {:?}", instantiated.kind(db));
        }
    }
}
