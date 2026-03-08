//! IR lowering context.
//!
//! Manages state during AST-to-IR transformation.
//! Emits arena IR (`IrContext` / `TypeRef` / `ValueRef`) directly.

use std::collections::HashMap;

use tribute_ir::arena::dialect::closure as arena_closure;
use tribute_ir::arena::dialect::tribute_rt as arena_tribute_rt;
use trunk_ir::Symbol;
use trunk_ir::SymbolVec;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::cont as arena_cont;
use trunk_ir::dialect::core as arena_core;
use trunk_ir::refs::{BlockRef, PathRef, TypeRef, ValueRef};
use trunk_ir::types::{Attribute, Location, TypeDataBuilder};

use crate::ast::{CtorId, LocalId, NodeId, SpanMap, TypeKind, TypeScheme};

/// Information about a captured variable.
#[derive(Clone, Debug)]
pub struct CaptureInfo {
    /// Variable name.
    pub name: Symbol,
    /// Variable's LocalId.
    pub local_id: LocalId,
    /// Variable type (arena IR type).
    pub ty: TypeRef,
    /// The SSA value in the outer scope.
    pub value: ValueRef,
}

/// Context for lowering AST to arena TrunkIR.
pub struct IrLoweringCtx<'db> {
    pub db: &'db dyn salsa::Database,
    pub path: PathRef,
    /// Span map for looking up source locations.
    span_map: SpanMap,
    /// Stack of scopes, each mapping LocalId to (name, SSA value).
    scopes: Vec<HashMap<LocalId, (Symbol, ValueRef)>>,
    /// Function type schemes from type checking, keyed by function name.
    function_types: HashMap<Symbol, TypeScheme<'db>>,
    /// Module path as a vector of segments (e.g., ["std", "Option"]).
    module_path: SymbolVec,
    /// Counter for generating unique lambda names.
    lambda_counter: u64,
    /// Counter for generating unique local IDs (for synthetic bindings like continuations).
    local_id_counter: u32,
    /// Module's top-level block, used for in-place insertion of lifted lambdas.
    module_block: Option<BlockRef>,
    /// Struct field order: CtorId → [field_names in definition order].
    /// Used for lowering Record expressions to adt.struct_new.
    struct_fields: HashMap<CtorId<'db>, Vec<Symbol>>,
    /// Type map: type name → arena TypeRef for adt.struct / adt.enum.
    /// Used for named structs, tuples, and (future) enum variants.
    type_map: im::HashMap<Symbol, TypeRef>,
    /// Counter for generating unique prompt tags (per-module deterministic).
    prompt_tag_counter: u32,
    /// Stack of active prompt tags for nested handlers.
    /// The top of the stack is the currently active prompt tag.
    active_prompt_tag_stack: Vec<u32>,

    /// Node types from type checking, keyed by NodeId.
    /// Used to get the effect type of lambda expressions.
    node_types: HashMap<NodeId, crate::ast::Type<'db>>,
}

impl<'db> IrLoweringCtx<'db> {
    /// Create a new IR lowering context.
    pub fn new(
        db: &'db dyn salsa::Database,
        path: PathRef,
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
            module_block: None,
            struct_fields: HashMap::new(),
            type_map: im::HashMap::new(),
            prompt_tag_counter: 0,
            active_prompt_tag_stack: Vec::new(),

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

    /// Set the module-level block for in-place insertion of lifted functions.
    pub fn set_module_block(&mut self, block: BlockRef) {
        self.module_block = Some(block);
    }

    /// Get the module-level block.
    pub fn module_block(&self) -> Option<BlockRef> {
        self.module_block
    }

    /// Create an arena Location for a node.
    pub fn location(&self, node_id: NodeId) -> Location {
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
    pub fn bind(&mut self, local_id: LocalId, name: Symbol, value: ValueRef) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(local_id, (name, value));
        }
    }

    /// Look up a function's type scheme by name.
    pub fn lookup_function_type(&self, name: Symbol) -> Option<&TypeScheme<'db>> {
        self.function_types.get(&name)
    }

    /// Look up a local variable.
    pub fn lookup(&self, local_id: LocalId) -> Option<ValueRef> {
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

    /// Register struct field order for lowering Record expressions.
    pub fn register_struct_fields(&mut self, ctor_id: CtorId<'db>, field_names: Vec<Symbol>) {
        self.struct_fields.insert(ctor_id, field_names);
    }

    /// Get struct field order (for lowering Record → adt.struct_new).
    pub fn get_struct_field_order(&self, ctor_id: CtorId<'db>) -> Option<&Vec<Symbol>> {
        self.struct_fields.get(&ctor_id)
    }

    /// Register a type (named struct, tuple, etc.) in the type map.
    pub fn register_type(&mut self, name: Symbol, ty: TypeRef) {
        self.type_map.insert(name, ty);
    }

    /// Get a registered type by name.
    pub fn get_type(&self, name: Symbol) -> Option<TypeRef> {
        self.type_map.get(&name).copied()
    }

    /// Resolve an AST type to its registered ADT IR type (struct or enum).
    ///
    /// For `Named { name, .. }` types, looks up the type_map by name.
    /// Returns `None` if the type is not a registered ADT type.
    pub fn resolve_adt_type(&self, ty: crate::ast::Type<'db>) -> Option<TypeRef> {
        match ty.kind(self.db) {
            TypeKind::Named { name, .. } => self.get_type(*name),
            _ => None,
        }
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
    pub fn all_bindings(&self) -> impl Iterator<Item = (LocalId, Symbol, ValueRef)> + '_ {
        self.scopes
            .iter()
            .rev()
            .flat_map(|scope| scope.iter().map(|(&id, &(name, value))| (id, name, value)))
    }

    // =========================================================================
    // Arena type conversion
    // =========================================================================

    /// Convert an AST type to an arena TypeRef.
    pub fn convert_type(&self, ir: &mut IrContext, ty: crate::ast::Type<'db>) -> TypeRef {
        match ty.kind(self.db) {
            TypeKind::Int | TypeKind::Nat | TypeKind::Rune => self.i32_type(ir),
            TypeKind::Float => self.f64_type(ir),
            TypeKind::Bool => self.bool_type(ir),
            TypeKind::String => self.string_type(ir),
            TypeKind::Bytes => self.bytes_type(ir),
            TypeKind::Nil | TypeKind::Error => self.nil_type(ir),
            TypeKind::BoundVar { .. } => {
                // Quantified type variable in TypeScheme body → type-erased any
                self.any_type(ir)
            }
            TypeKind::UniVar { id } => {
                // UniVar surviving substitution indicates incomplete constraint solving.
                tracing::debug!(
                    "UniVar({:?}) survived substitution — type-erasing to any",
                    id
                );
                self.any_type(ir)
            }
            TypeKind::Named { .. } => {
                // Type erasure: struct/enum → tribute_rt.any
                self.any_type(ir)
            }
            TypeKind::Func {
                params,
                result,
                effect,
            } => {
                let param_refs: Vec<TypeRef> =
                    params.iter().map(|p| self.convert_type(ir, *p)).collect();
                let result_ref = self.convert_type(ir, *result);
                let effect_ref = if effect.is_pure(self.db) {
                    None
                } else {
                    Some(self.convert_effect_row(ir, *effect))
                };
                self.func_type_with_effect(ir, &param_refs, result_ref, effect_ref)
            }
            TypeKind::Tuple(_) => {
                // Type erasure: tuple → tribute_rt.any
                self.any_type(ir)
            }
            TypeKind::App { ctor, .. } => self.convert_type(ir, *ctor),
            TypeKind::Continuation { arg, result, .. } => {
                let ir_arg = self.convert_type(ir, *arg);
                let ir_result = self.convert_type(ir, *result);
                let effect = self.any_type(ir);
                self.continuation_type(ir, ir_arg, ir_result, effect)
            }
        }
    }

    /// Convert an AST EffectRow to an arena TypeRef.
    pub fn convert_effect_row(
        &self,
        ir: &mut IrContext,
        row: crate::ast::EffectRow<'db>,
    ) -> TypeRef {
        let effects = row.effects(self.db);
        let rest = row.rest(self.db);

        // Convert each Effect to an AbilityRefType
        let ability_types: Vec<TypeRef> = effects
            .iter()
            .map(|effect| {
                let ability_name = effect.ability_id.qualified_name(self.db).to_string();
                let ability_sym = Symbol::from_dynamic(&ability_name);

                // Convert type arguments
                let params: Vec<TypeRef> = effect
                    .args
                    .iter()
                    .map(|ty| self.convert_type(ir, *ty))
                    .collect();

                self.ability_ref_type(ir, ability_sym, &params)
            })
            .collect();

        // Create EffectRowType with tail variable if present
        let tail_var_id = rest.map(|v| v.id as u32).unwrap_or(0);
        self.effect_row_type(ir, &ability_types, tail_var_id)
    }

    // =========================================================================
    // Arena type helpers
    // =========================================================================

    /// Get the `core.i32` type.
    pub fn i32_type(&self, ir: &mut IrContext) -> TypeRef {
        ir.types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build())
    }

    /// Get the `core.nil` type.
    pub fn nil_type(&self, ir: &mut IrContext) -> TypeRef {
        arena_core::nil(ir).as_type_ref()
    }

    /// Get the `core.i1` (bool) type.
    pub fn bool_type(&self, ir: &mut IrContext) -> TypeRef {
        ir.types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i1")).build())
    }

    /// Get the `core.f64` type.
    pub fn f64_type(&self, ir: &mut IrContext) -> TypeRef {
        ir.types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("f64")).build())
    }

    /// Get the `core.string` type.
    pub fn string_type(&self, ir: &mut IrContext) -> TypeRef {
        arena_core::string(ir).as_type_ref()
    }

    /// Get the `core.bytes` type.
    pub fn bytes_type(&self, ir: &mut IrContext) -> TypeRef {
        arena_core::bytes(ir).as_type_ref()
    }

    /// Get the `tribute_rt.any` type.
    pub fn any_type(&self, ir: &mut IrContext) -> TypeRef {
        arena_tribute_rt::any(ir).as_type_ref()
    }

    /// Get the `cont.prompt_tag` type.
    pub fn prompt_tag_type(&self, ir: &mut IrContext) -> TypeRef {
        arena_cont::prompt_tag(ir).as_type_ref()
    }

    /// Create a `core.func` type with params and result.
    ///
    /// Layout follows Salsa `core::Func`: `params[0] = result, params[1..] = param_types`.
    pub fn func_type(&self, ir: &mut IrContext, params: &[TypeRef], result: TypeRef) -> TypeRef {
        arena_core::func(ir, result, params.iter().copied(), None).as_type_ref()
    }

    /// Create a `core.func` type with params, result, and effect.
    ///
    /// Layout follows Salsa `core::Func`: `params[0] = result, params[1..] = param_types`.
    pub fn func_type_with_effect(
        &self,
        ir: &mut IrContext,
        params: &[TypeRef],
        result: TypeRef,
        effect: Option<TypeRef>,
    ) -> TypeRef {
        arena_core::func(ir, result, params.iter().copied(), effect).as_type_ref()
    }

    /// Create a `cont.continuation` type.
    pub fn continuation_type(
        &self,
        ir: &mut IrContext,
        arg: TypeRef,
        result: TypeRef,
        effect: TypeRef,
    ) -> TypeRef {
        arena_cont::continuation(ir, arg, result, effect).as_type_ref()
    }

    /// Create a `core.ability_ref` type.
    pub fn ability_ref_type(
        &self,
        ir: &mut IrContext,
        ability_name: Symbol,
        params: &[TypeRef],
    ) -> TypeRef {
        let mut builder = TypeDataBuilder::new(Symbol::new("core"), Symbol::new("ability_ref"))
            .attr("name", Attribute::Symbol(ability_name));
        for &p in params {
            builder = builder.param(p);
        }
        ir.types.intern(builder.build())
    }

    /// Create a `core.effect_row` type.
    pub fn effect_row_type(
        &self,
        ir: &mut IrContext,
        abilities: &[TypeRef],
        tail_var_id: u32,
    ) -> TypeRef {
        let mut builder = TypeDataBuilder::new(Symbol::new("core"), Symbol::new("effect_row"));
        for &a in abilities {
            builder = builder.param(a);
        }
        builder = builder.attr("tail_var_id", Attribute::IntBits(tail_var_id as u64));
        ir.types.intern(builder.build())
    }

    /// Create an `adt.struct` type with name and fields.
    pub fn adt_struct_type(
        &self,
        ir: &mut IrContext,
        name: Symbol,
        fields: &[(Symbol, TypeRef)],
    ) -> TypeRef {
        let fields_attr: Vec<Attribute> = fields
            .iter()
            .map(|(field_name, field_type)| {
                Attribute::List(vec![
                    Attribute::Symbol(*field_name),
                    Attribute::Type(*field_type),
                ])
            })
            .collect();

        ir.types.intern(
            TypeDataBuilder::new(Symbol::new("adt"), Symbol::new("struct"))
                .attr("name", Attribute::Symbol(name))
                .attr("fields", Attribute::List(fields_attr))
                .build(),
        )
    }

    /// Create an `adt.enum` type with name and variants.
    pub fn adt_enum_type(
        &self,
        ir: &mut IrContext,
        name: Symbol,
        variants: &[(Symbol, Vec<TypeRef>)],
    ) -> TypeRef {
        let variants_attr: Vec<Attribute> = variants
            .iter()
            .map(|(variant_name, field_types)| {
                let field_attrs: Vec<Attribute> =
                    field_types.iter().map(|t| Attribute::Type(*t)).collect();
                Attribute::List(vec![
                    Attribute::Symbol(*variant_name),
                    Attribute::List(field_attrs),
                ])
            })
            .collect();

        ir.types.intern(
            TypeDataBuilder::new(Symbol::new("adt"), Symbol::new("enum"))
                .attr("name", Attribute::Symbol(name))
                .attr("variants", Attribute::List(variants_attr))
                .build(),
        )
    }

    /// Create an `adt.typeref` type — a reference to a named type.
    pub fn adt_typeref(&self, ir: &mut IrContext, name: Symbol) -> TypeRef {
        ir.types.intern(
            TypeDataBuilder::new(Symbol::new("adt"), Symbol::new("typeref"))
                .attr("name", Attribute::Symbol(name))
                .build(),
        )
    }

    /// Create the `closure.closure` type wrapping a function type.
    pub fn closure_type(&self, ir: &mut IrContext, func_type: TypeRef) -> TypeRef {
        arena_closure::closure(ir, func_type).as_type_ref()
    }

    /// Check if a type is a `closure.closure` type.
    pub fn is_closure_type(&self, ir: &IrContext, ty: TypeRef) -> bool {
        ir.types
            .is_dialect(ty, Symbol::new("closure"), Symbol::new("closure"))
    }

    /// Check if a type is a `core.func` type.
    pub fn is_func_type(&self, ir: &IrContext, ty: TypeRef) -> bool {
        ir.types
            .is_dialect(ty, Symbol::new("core"), Symbol::new("func"))
    }

    /// Get the param count from a core.func type.
    /// core.func stores params as: [result, param1, param2, ...] (result first).
    pub fn func_type_param_count(&self, ir: &IrContext, ty: TypeRef) -> usize {
        let td = ir.types.get(ty);
        if !td.params.is_empty() {
            td.params.len() - 1 // first param is the result type, rest are param types
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Type as AstType, TypeKind};

    fn test_db() -> salsa::DatabaseImpl {
        salsa::DatabaseImpl::new()
    }

    #[test]
    fn test_convert_type_bound_var_to_any() {
        let db = test_db();
        let mut ir = IrContext::new();
        let path = ir.paths.intern("test.trb".to_owned());
        let ctx = IrLoweringCtx::new(
            &db,
            path,
            crate::ast::SpanMap::default(),
            HashMap::new(),
            smallvec::smallvec![Symbol::new("test")],
            HashMap::new(),
        );

        let ty = AstType::new(&db, TypeKind::BoundVar { index: 0 });
        let ir_ty = ctx.convert_type(&mut ir, ty);
        let expected = ctx.any_type(&mut ir);
        assert_eq!(ir_ty, expected);
    }

    #[test]
    fn test_convert_type_named_to_any() {
        let db = test_db();
        let mut ir = IrContext::new();
        let path = ir.paths.intern("test.trb".to_owned());
        let ctx = IrLoweringCtx::new(
            &db,
            path,
            crate::ast::SpanMap::default(),
            HashMap::new(),
            smallvec::smallvec![Symbol::new("test")],
            HashMap::new(),
        );

        let int_ty = AstType::new(&db, TypeKind::Int);
        let ty = AstType::new(
            &db,
            TypeKind::Named {
                name: Symbol::new("List"),
                args: vec![int_ty],
            },
        );
        let ir_ty = ctx.convert_type(&mut ir, ty);
        let expected = ctx.any_type(&mut ir);
        assert_eq!(ir_ty, expected);
    }

    #[test]
    fn test_convert_type_tuple_to_any() {
        let db = test_db();
        let mut ir = IrContext::new();
        let path = ir.paths.intern("test.trb".to_owned());
        let ctx = IrLoweringCtx::new(
            &db,
            path,
            crate::ast::SpanMap::default(),
            HashMap::new(),
            smallvec::smallvec![Symbol::new("test")],
            HashMap::new(),
        );

        let int_ty = AstType::new(&db, TypeKind::Int);
        let bool_ty = AstType::new(&db, TypeKind::Bool);
        let ty = AstType::new(&db, TypeKind::Tuple(vec![int_ty, bool_ty]));
        let ir_ty = ctx.convert_type(&mut ir, ty);
        let expected = ctx.any_type(&mut ir);
        assert_eq!(ir_ty, expected);
    }

    #[test]
    fn test_convert_type_func_with_bound_vars() {
        let db = test_db();
        let mut ir = IrContext::new();
        let path = ir.paths.intern("test.trb".to_owned());
        let ctx = IrLoweringCtx::new(
            &db,
            path,
            crate::ast::SpanMap::default(),
            HashMap::new(),
            smallvec::smallvec![Symbol::new("test")],
            HashMap::new(),
        );

        let bound_var = AstType::new(&db, TypeKind::BoundVar { index: 0 });
        let effect = crate::ast::EffectRow::pure(&db);
        let ty = AstType::new(
            &db,
            TypeKind::Func {
                params: vec![bound_var],
                result: bound_var,
                effect,
            },
        );
        let ir_ty = ctx.convert_type(&mut ir, ty);

        // BoundVar params/result → tribute_rt.any, wrapped in core.func
        let any_ty = ctx.any_type(&mut ir);
        let expected = ctx.func_type(&mut ir, &[any_ty], any_ty);
        assert_eq!(ir_ty, expected);
    }

    #[test]
    fn test_convert_type_primitives() {
        let db = test_db();
        let mut ir = IrContext::new();
        let path = ir.paths.intern("test.trb".to_owned());
        let ctx = IrLoweringCtx::new(
            &db,
            path,
            crate::ast::SpanMap::default(),
            HashMap::new(),
            smallvec::smallvec![Symbol::new("test")],
            HashMap::new(),
        );

        // Int → I32
        let int_ty = AstType::new(&db, TypeKind::Int);
        assert_eq!(ctx.convert_type(&mut ir, int_ty), ctx.i32_type(&mut ir));

        // Bool → I1
        let bool_ty = AstType::new(&db, TypeKind::Bool);
        assert_eq!(ctx.convert_type(&mut ir, bool_ty), ctx.bool_type(&mut ir));

        // String → string
        let str_ty = AstType::new(&db, TypeKind::String);
        assert_eq!(ctx.convert_type(&mut ir, str_ty), ctx.string_type(&mut ir));

        // Float → F64
        let float_ty = AstType::new(&db, TypeKind::Float);
        assert_eq!(ctx.convert_type(&mut ir, float_ty), ctx.f64_type(&mut ir));

        // Nil → Nil
        let nil_ty = AstType::new(&db, TypeKind::Nil);
        assert_eq!(ctx.convert_type(&mut ir, nil_ty), ctx.nil_type(&mut ir));
    }

    #[test]
    fn test_lookup_function_type() {
        let db = test_db();
        let mut ir = IrContext::new();
        let path = ir.paths.intern("test.trb".to_owned());
        let name = Symbol::new("foo");
        let body = AstType::new(&db, TypeKind::Int);
        let scheme = TypeScheme::new(&db, vec![], body);

        let mut ft = HashMap::new();
        ft.insert(name, scheme);

        let ctx = IrLoweringCtx::new(
            &db,
            path,
            crate::ast::SpanMap::default(),
            ft,
            smallvec::smallvec![Symbol::new("test")],
            HashMap::new(),
        );
        assert_eq!(ctx.lookup_function_type(name), Some(&scheme));
        assert_eq!(ctx.lookup_function_type(Symbol::new("missing")), None);
    }
}
