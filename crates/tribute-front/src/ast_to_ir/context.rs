//! IR lowering context.
//!
//! Manages state during AST-to-IR transformation.

use std::collections::HashMap;

use tribute_ir::dialect::tribute_rt;
use trunk_ir::dialect::core;
use trunk_ir::{DialectType, Location, Operation, PathId, Symbol, SymbolVec, Type, Value};

use crate::ast::{CtorId, LocalId, NodeId, SpanMap, TypeKind, TypeScheme};

/// Information about a captured variable.
#[derive(Clone, Debug)]
pub struct CaptureInfo<'db> {
    /// Variable name.
    pub name: Symbol,
    /// Variable's LocalId.
    pub local_id: LocalId,
    /// Variable type (IR type).
    pub ty: Type<'db>,
    /// The SSA value in the outer scope.
    pub value: Value<'db>,
}

/// Context for lowering AST to TrunkIR.
pub struct IrLoweringCtx<'db> {
    pub db: &'db dyn salsa::Database,
    pub path: PathId<'db>,
    /// Span map for looking up source locations.
    span_map: SpanMap,
    /// Stack of scopes, each mapping LocalId to (name, SSA value).
    scopes: Vec<HashMap<LocalId, (Symbol, Value<'db>)>>,
    /// Function type schemes from type checking, keyed by function name.
    function_types: HashMap<Symbol, TypeScheme<'db>>,
    /// Module path as a vector of segments (e.g., ["std", "Option"]).
    module_path: SymbolVec,
    /// Counter for generating unique lambda names.
    lambda_counter: u64,
    /// Counter for generating unique local IDs (for synthetic bindings like continuations).
    local_id_counter: u32,
    /// Lifted lambda functions to be added at module level.
    lifted_functions: Vec<Operation<'db>>,
    /// Struct field order: CtorId → [field_names in definition order].
    /// Used for lowering Record expressions to adt.struct_new.
    struct_fields: HashMap<CtorId<'db>, Vec<Symbol>>,
    /// Struct IR types: CtorId → adt.struct type with full field info.
    /// Used to set the correct type attribute on adt.struct_new operations.
    struct_ir_types: HashMap<CtorId<'db>, Type<'db>>,
    /// Counter for generating unique prompt tags (per-module deterministic).
    prompt_tag_counter: u32,
    /// Stack of active prompt tags for nested handlers.
    /// The top of the stack is the currently active prompt tag.
    active_prompt_tag_stack: Vec<u32>,
    /// Track IR types of SSA values for cast insertion.
    /// Maps each generated Value to its IR type.
    value_types: HashMap<Value<'db>, Type<'db>>,
    /// Node types from type checking, keyed by NodeId.
    /// Used to get the effect type of lambda expressions.
    node_types: HashMap<NodeId, crate::ast::Type<'db>>,
}

impl<'db> IrLoweringCtx<'db> {
    /// Create a new IR lowering context.
    pub fn new(
        db: &'db dyn salsa::Database,
        path: PathId<'db>,
        span_map: SpanMap,
        function_types: HashMap<Symbol, TypeScheme<'db>>,
        module_path: SymbolVec,
        node_types: HashMap<NodeId, crate::ast::Type<'db>>,
    ) -> Self {
        Self {
            db,
            path,
            span_map,
            scopes: vec![HashMap::new()],
            function_types,
            module_path,
            lambda_counter: 0,
            local_id_counter: 0x8000_0000, // Start high to avoid collisions with parsed LocalIds
            lifted_functions: Vec::new(),
            struct_fields: HashMap::new(),
            struct_ir_types: HashMap::new(),
            prompt_tag_counter: 0,
            active_prompt_tag_stack: Vec::new(),
            value_types: HashMap::new(),
            node_types,
        }
    }

    /// Get the current module path.
    pub fn module_path(&self) -> &SymbolVec {
        &self.module_path
    }

    /// Enter a nested module, updating the module path.
    pub fn enter_module(&mut self, name: Symbol) {
        self.module_path.push(name);
    }

    /// Exit a nested module, restoring the parent module path.
    pub fn exit_module(&mut self) {
        self.module_path.pop();
    }

    /// Create a location for a node.
    pub fn location(&self, node_id: NodeId) -> Location<'db> {
        let span = self.span_map.get_or_default(node_id);
        Location::new(self.path, span)
    }

    /// Enter a new scope.
    pub fn enter_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    /// Exit the current scope.
    pub fn exit_scope(&mut self) {
        self.scopes.pop();
    }

    /// Execute a closure with a new scope, automatically entering and exiting.
    pub fn scoped<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(&mut Self) -> R,
    {
        self.enter_scope();
        let result = f(self);
        self.exit_scope();
        result
    }

    /// Bind a local variable to an SSA value.
    pub fn bind(&mut self, local_id: LocalId, name: Symbol, value: Value<'db>) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(local_id, (name, value));
        }
    }

    /// Look up a function's type scheme by name.
    pub fn lookup_function_type(&self, name: Symbol) -> Option<&TypeScheme<'db>> {
        self.function_types.get(&name)
    }

    /// Look up a local variable.
    pub fn lookup(&self, local_id: LocalId) -> Option<Value<'db>> {
        for scope in self.scopes.iter().rev() {
            if let Some((_, value)) = scope.get(&local_id) {
                return Some(*value);
            }
        }
        None
    }

    /// Generate a unique LocalId for synthetic bindings (e.g., continuations).
    ///
    /// # Panics
    /// Panics if the counter would overflow into the UNRESOLVED sentinel value.
    pub fn next_local_id(&mut self) -> LocalId {
        let id = self.local_id_counter;
        // Ensure we don't hit LocalId::UNRESOLVED (u32::MAX)
        if id == u32::MAX {
            panic!("ICE: local_id_counter overflow - would produce UNRESOLVED sentinel");
        }
        self.local_id_counter = self
            .local_id_counter
            .checked_add(1)
            .expect("ICE: local_id_counter overflow");
        LocalId::new(id)
    }

    /// Generate a fresh prompt tag and push it onto the active stack.
    ///
    /// This should be called when entering a `handle` expression.
    /// The tag is used by both `cont.push_prompt` and `cont.shift` to ensure
    /// they reference the same prompt.
    pub fn push_prompt_tag(&mut self) -> u32 {
        let tag = self.prompt_tag_counter;
        self.prompt_tag_counter = self
            .prompt_tag_counter
            .checked_add(1)
            .expect("ICE: prompt_tag_counter overflow");
        self.active_prompt_tag_stack.push(tag);
        tag
    }

    /// Pop the current active prompt tag from the stack.
    ///
    /// This should be called when exiting a `handle` expression.
    pub fn pop_prompt_tag(&mut self) {
        self.active_prompt_tag_stack.pop();
    }

    /// Get the currently active prompt tag.
    ///
    /// Returns `None` if not inside a handler context.
    pub fn active_prompt_tag(&self) -> Option<u32> {
        self.active_prompt_tag_stack.last().copied()
    }

    /// Generate a unique lambda name qualified with module path.
    pub fn gen_lambda_name(&mut self) -> Symbol {
        let lambda_name = format!("__lambda_{}", self.lambda_counter);
        self.lambda_counter += 1;

        if self.module_path.is_empty() {
            Symbol::from_dynamic(&lambda_name)
        } else {
            let path_str = self
                .module_path
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join("::");
            Symbol::from_dynamic(&format!("{}::{}", path_str, lambda_name))
        }
    }

    /// Add a lifted function to be included in the module.
    pub fn add_lifted_function(&mut self, func: Operation<'db>) {
        self.lifted_functions.push(func);
    }

    /// Take all lifted functions (consumes them).
    pub fn take_lifted_functions(&mut self) -> Vec<Operation<'db>> {
        std::mem::take(&mut self.lifted_functions)
    }

    /// Register struct field order for lowering Record expressions.
    pub fn register_struct_fields(&mut self, ctor_id: CtorId<'db>, field_names: Vec<Symbol>) {
        self.struct_fields.insert(ctor_id, field_names);
    }

    /// Get struct field order (for lowering Record → adt.struct_new).
    pub fn get_struct_field_order(&self, ctor_id: CtorId<'db>) -> Option<&Vec<Symbol>> {
        self.struct_fields.get(&ctor_id)
    }

    /// Register the full IR type for a struct (adt.struct with field names and types).
    pub fn register_struct_ir_type(&mut self, ctor_id: CtorId<'db>, ir_type: Type<'db>) {
        self.struct_ir_types.insert(ctor_id, ir_type);
    }

    /// Get the registered IR type for a struct.
    pub fn get_struct_ir_type(&self, ctor_id: CtorId<'db>) -> Option<Type<'db>> {
        self.struct_ir_types.get(&ctor_id).copied()
    }

    /// Track the IR type of a generated SSA value.
    ///
    /// This is used by `cast_if_needed` to determine if a cast is required.
    pub fn track_value_type(&mut self, value: Value<'db>, ty: Type<'db>) {
        self.value_types.insert(value, ty);
    }

    /// Get the tracked IR type of a value.
    ///
    /// Returns `None` if the value's type was not tracked.
    pub fn get_value_type(&self, value: Value<'db>) -> Option<Type<'db>> {
        self.value_types.get(&value).copied()
    }

    /// Get the type of an AST node by NodeId.
    ///
    /// Returns the type assigned during type checking.
    /// Used to get the effect type of lambda expressions.
    pub fn get_node_type(&self, node_id: NodeId) -> Option<&crate::ast::Type<'db>> {
        self.node_types.get(&node_id)
    }

    /// Get all bindings visible in the current scope (for capture analysis).
    /// Returns bindings from all scopes, innermost first.
    pub fn all_bindings(&self) -> impl Iterator<Item = (LocalId, Symbol, Value<'db>)> + '_ {
        self.scopes
            .iter()
            .rev()
            .flat_map(|scope| scope.iter().map(|(&id, &(name, value))| (id, name, value)))
    }

    /// Convert an AST type to a TrunkIR type.
    pub fn convert_type(&self, ty: crate::ast::Type<'db>) -> Type<'db> {
        match ty.kind(self.db) {
            TypeKind::Int => core::I32::new(self.db).as_type(),
            TypeKind::Nat => core::I32::new(self.db).as_type(),
            TypeKind::Float => core::F64::new(self.db).as_type(),
            TypeKind::Bool => core::I1::new(self.db).as_type(),
            TypeKind::String => core::String::new(self.db).as_type(),
            TypeKind::Bytes => core::Bytes::new(self.db).as_type(),
            TypeKind::Rune => core::I32::new(self.db).as_type(),
            TypeKind::Nil | TypeKind::Error => core::Nil::new(self.db).as_type(),
            TypeKind::BoundVar { .. } => {
                // Quantified type variable in TypeScheme body → type-erased any
                tribute_rt::any_type(self.db)
            }
            TypeKind::UniVar { id } => {
                // UniVar surviving substitution indicates incomplete constraint solving.
                // This can happen with complex effect row polymorphism (e.g., handler
                // arms with continuation calls in recursive functions).
                // Type-erase to any since the runtime uses a type-erased representation.
                tracing::debug!(
                    "UniVar({:?}) survived substitution — type-erasing to any",
                    id
                );
                tribute_rt::any_type(self.db)
            }
            TypeKind::Named { .. } => {
                // Type erasure: struct/enum → tribute_rt.any
                tribute_rt::any_type(self.db)
            }
            TypeKind::Func { params, result, .. } => {
                let params: Vec<Type<'db>> = params.iter().map(|p| self.convert_type(*p)).collect();
                let result = self.convert_type(*result);
                core::Func::new(self.db, params.into(), result).as_type()
            }
            TypeKind::Tuple(_) => {
                // Type erasure: tuple → tribute_rt.any
                tribute_rt::any_type(self.db)
            }
            TypeKind::App { ctor, .. } => self.convert_type(*ctor),
            TypeKind::Continuation { arg, result, .. } => {
                use trunk_ir::dialect::cont;
                let ir_arg = self.convert_type(*arg);
                let ir_result = self.convert_type(*result);
                let effect = tribute_rt::any_type(self.db);
                cont::Continuation::new(self.db, ir_arg, ir_result, effect).as_type()
            }
        }
    }

    /// Get the nil type.
    pub fn nil_type(&self) -> Type<'db> {
        core::Nil::new(self.db).as_type()
    }

    /// Get the int type.
    pub fn int_type(&self) -> Type<'db> {
        core::I32::new(self.db).as_type()
    }

    /// Get the bool type.
    pub fn bool_type(&self) -> Type<'db> {
        core::I1::new(self.db).as_type()
    }

    /// Convert an AST EffectRow to a TrunkIR effect type.
    ///
    /// The effect row contains abilities (effects) and possibly an open tail variable.
    /// We convert this to a `core.effect_row` type for the IR.
    pub fn convert_effect_row(&self, row: crate::ast::EffectRow<'db>) -> Type<'db> {
        let effects = row.effects(self.db);
        let rest = row.rest(self.db);

        // Convert each Effect to an AbilityRefType
        let ability_types: trunk_ir::IdVec<Type<'db>> = effects
            .iter()
            .map(|effect| {
                // Build qualified name for the ability
                let ability_name = effect.ability_id.qualified_name(self.db).to_string();
                let ability_sym = Symbol::from_dynamic(&ability_name);

                // Convert type arguments
                let params: trunk_ir::IdVec<Type<'db>> = effect
                    .args
                    .iter()
                    .map(|ty| self.convert_type(*ty))
                    .collect();

                if params.is_empty() {
                    core::AbilityRefType::simple(self.db, ability_sym).as_type()
                } else {
                    core::AbilityRefType::with_params(self.db, ability_sym, params).as_type()
                }
            })
            .collect();

        // Create EffectRowType with tail variable if present
        let tail_var_id = rest.map(|v| v.id).unwrap_or(0);
        core::EffectRowType::new(self.db, ability_types, tail_var_id).as_type()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{EffectRow, Type as AstType, TypeKind, TypeScheme};
    use salsa_test_macros::salsa_test;

    // =========================================================================
    // convert_type tests
    // =========================================================================

    #[salsa_test]
    fn test_convert_type_bound_var_to_any(db: &salsa::DatabaseImpl) {
        let path = PathId::new(db, "test.trb".to_owned());
        let ctx = IrLoweringCtx::new(
            db,
            path,
            crate::ast::SpanMap::default(),
            HashMap::new(),
            smallvec::smallvec![Symbol::new("test")],
            HashMap::new(),
        );

        let ty = AstType::new(db, TypeKind::BoundVar { index: 0 });
        let ir_ty = ctx.convert_type(ty);
        assert_eq!(ir_ty, tribute_rt::any_type(db));
    }

    #[salsa_test]
    fn test_convert_type_named_to_any(db: &salsa::DatabaseImpl) {
        let path = PathId::new(db, "test.trb".to_owned());
        let ctx = IrLoweringCtx::new(
            db,
            path,
            crate::ast::SpanMap::default(),
            HashMap::new(),
            smallvec::smallvec![Symbol::new("test")],
            HashMap::new(),
        );

        let int_ty = AstType::new(db, TypeKind::Int);
        let ty = AstType::new(
            db,
            TypeKind::Named {
                name: Symbol::new("List"),
                args: vec![int_ty],
            },
        );
        let ir_ty = ctx.convert_type(ty);
        assert_eq!(ir_ty, tribute_rt::any_type(db));
    }

    #[salsa_test]
    fn test_convert_type_tuple_to_any(db: &salsa::DatabaseImpl) {
        let path = PathId::new(db, "test.trb".to_owned());
        let ctx = IrLoweringCtx::new(
            db,
            path,
            crate::ast::SpanMap::default(),
            HashMap::new(),
            smallvec::smallvec![Symbol::new("test")],
            HashMap::new(),
        );

        let int_ty = AstType::new(db, TypeKind::Int);
        let bool_ty = AstType::new(db, TypeKind::Bool);
        let ty = AstType::new(db, TypeKind::Tuple(vec![int_ty, bool_ty]));
        let ir_ty = ctx.convert_type(ty);
        assert_eq!(ir_ty, tribute_rt::any_type(db));
    }

    #[salsa_test]
    fn test_convert_type_func_with_bound_vars(db: &salsa::DatabaseImpl) {
        let path = PathId::new(db, "test.trb".to_owned());
        let ctx = IrLoweringCtx::new(
            db,
            path,
            crate::ast::SpanMap::default(),
            HashMap::new(),
            smallvec::smallvec![Symbol::new("test")],
            HashMap::new(),
        );

        let bound_var = AstType::new(db, TypeKind::BoundVar { index: 0 });
        let effect = EffectRow::pure(db);
        let ty = AstType::new(
            db,
            TypeKind::Func {
                params: vec![bound_var],
                result: bound_var,
                effect,
            },
        );
        let ir_ty = ctx.convert_type(ty);

        // BoundVar params/result → tribute_rt.any, wrapped in core.func
        let any_ty = tribute_rt::any_type(db);
        let expected = core::Func::new(db, vec![any_ty].into(), any_ty).as_type();
        assert_eq!(ir_ty, expected);
    }

    #[salsa_test]
    fn test_convert_type_primitives_unchanged(db: &salsa::DatabaseImpl) {
        let path = PathId::new(db, "test.trb".to_owned());
        let ctx = IrLoweringCtx::new(
            db,
            path,
            crate::ast::SpanMap::default(),
            HashMap::new(),
            smallvec::smallvec![Symbol::new("test")],
            HashMap::new(),
        );

        // Int → I32
        let int_ty = AstType::new(db, TypeKind::Int);
        assert_eq!(ctx.convert_type(int_ty), core::I32::new(db).as_type());

        // Bool → I1
        let bool_ty = AstType::new(db, TypeKind::Bool);
        assert_eq!(ctx.convert_type(bool_ty), core::I1::new(db).as_type());

        // String → string
        let str_ty = AstType::new(db, TypeKind::String);
        assert_eq!(ctx.convert_type(str_ty), core::String::new(db).as_type());

        // Float → F64
        let float_ty = AstType::new(db, TypeKind::Float);
        assert_eq!(ctx.convert_type(float_ty), core::F64::new(db).as_type());

        // Nil → Nil
        let nil_ty = AstType::new(db, TypeKind::Nil);
        assert_eq!(ctx.convert_type(nil_ty), core::Nil::new(db).as_type());
    }

    // =========================================================================
    // lookup_function_type tests
    // =========================================================================

    #[salsa_test]
    fn test_lookup_function_type_found(db: &salsa::DatabaseImpl) {
        let path = PathId::new(db, "test.trb".to_owned());
        let name = Symbol::new("foo");
        let body = AstType::new(db, TypeKind::Int);
        let scheme = TypeScheme::new(db, vec![], body);

        let mut ft = HashMap::new();
        ft.insert(name, scheme);

        let ctx = IrLoweringCtx::new(
            db,
            path,
            crate::ast::SpanMap::default(),
            ft,
            smallvec::smallvec![Symbol::new("test")],
            HashMap::new(),
        );
        assert_eq!(ctx.lookup_function_type(name), Some(&scheme));
    }

    #[salsa_test]
    fn test_lookup_function_type_not_found(db: &salsa::DatabaseImpl) {
        let path = PathId::new(db, "test.trb".to_owned());
        let ctx = IrLoweringCtx::new(
            db,
            path,
            crate::ast::SpanMap::default(),
            HashMap::new(),
            smallvec::smallvec![Symbol::new("test")],
            HashMap::new(),
        );
        assert_eq!(ctx.lookup_function_type(Symbol::new("missing")), None);
    }
}
