//! Type checking context.
//!
//! The context tracks type bindings, function signatures, and other information
//! needed during type checking.

use std::collections::HashMap;

use trunk_ir::Symbol;

use crate::ast::{
    CtorId, EffectRow, EffectVar, FuncDefId, LocalId, NodeId, Type, TypeKind, TypeScheme,
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
    pub fn fresh_type_var(&mut self) -> Type<'db> {
        let id = self.next_type_var;
        self.next_type_var += 1;
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
    pub fn instantiate_function(&mut self, id: FuncDefId<'db>) -> Option<Type<'db>> {
        let scheme = self.lookup_function(id)?;
        Some(self.instantiate_scheme(scheme))
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
    pub fn instantiate_constructor(&mut self, id: CtorId<'db>) -> Option<Type<'db>> {
        let scheme = self.lookup_constructor(id)?;
        Some(self.instantiate_scheme(scheme))
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
                Type::new(
                    self.db,
                    TypeKind::Func {
                        params,
                        result,
                        effect: *effect,
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
