//! IR lowering context.
//!
//! Manages state during AST-to-IR transformation.
//! Emits arena IR (`IrContext` / `TypeRef` / `ValueRef`) directly.

use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;
use std::ops::{Deref, DerefMut};

use tribute_ir::dialect::tribute_control::{Row, RowBinder, RowUnion};
use trunk_ir::Symbol;
use trunk_ir::SymbolPath;
use trunk_ir::context::IrContext;
use trunk_ir::refs::{BlockRef, OpRef, PathRef, TypeRef, ValueRef};
use trunk_ir::types::Location;

use super::nominal::NominalTypes;
use super::row_binders::RowBinders;
use crate::SortedMap;
use crate::ast::{CtorId, LocalId, NodeId, SpanMaps, TypeScheme};

/// A source-visible callable synthesized by the logical frontend, such as a
/// struct-field accessor. These signatures are semantic lowering metadata,
/// not reconstructed by inspecting emitted operations.
#[derive(Clone)]
pub(crate) struct LogicalGeneratedSignature {
    pub param_types: Vec<TypeRef>,
    pub return_type: TypeRef,
    pub row: Row,
    pub binders: Vec<RowBinder>,
}

/// The definition scope [`IrLoweringCtx::enter_definition`] replaced.
pub(crate) struct DefinitionScope {
    binders: RowBinders,
}

/// Context for lowering AST to arena TrunkIR.
pub struct IrLoweringCtx<'db> {
    pub db: &'db dyn salsa::Database,
    pub path: PathRef,
    /// Paths of merged sources, keyed by the source hash of their nodes.
    source_paths: HashMap<u64, PathRef>,
    /// Span maps of the merged sources, for looking up source locations.
    span_map: SpanMaps<'db>,
    /// Stack of scopes, each mapping LocalId to (name, SSA value).
    scopes: Vec<HashMap<LocalId, (Symbol, ValueRef)>>,
    local_callable_values: Vec<HashMap<(NodeId, crate::ast::Type<'db>, TypeRef), ValueRef>>,
    /// Scoped tags identifying locals whose SSA value is a suspended handler
    /// continuation rather than a source value.
    resume_scopes: Vec<HashSet<LocalId>>,
    /// Function type schemes from type checking, keyed by function name.
    function_types: HashMap<SymbolPath, TypeScheme<'db>>,
    /// Nominal layouts, row binders, and the conversion of source types.
    nominal: NominalTypes<'db>,
    /// Source-visible generated callables that have no source TypeScheme.
    logical_generated_signatures: HashMap<SymbolPath, LogicalGeneratedSignature>,
    /// Source declarations collected before logical lowering begins.
    logical_source_functions: HashSet<SymbolPath>,
    /// The function each `func_ref` operation of the module names. The IR
    /// names a definition by its joined path as one component, so the path
    /// cannot be read back from the operation.
    function_ref_paths: HashMap<OpRef, SymbolPath>,
    /// Extern declarations synthesized for referenced prelude functions.
    logical_emitted_externs: HashSet<SymbolPath>,
    /// Module path as a vector of segments (e.g., ["std", "Option"]).
    module_path: SymbolPath,
    /// Module's top-level block, used for in-place insertion of lifted lambdas.
    module_block: Option<BlockRef>,
    /// Exact intrinsic-directive declaration ID to canonical identity.
    compiler_intrinsics: HashMap<NodeId, Symbol>,
    /// The runtime structs to publish as prelude types.
    runtime_types: super::runtime_items::RuntimeTypes,
    /// The `==` functions that `String` and `Bytes` literal patterns call.
    literal_equalities: LiteralEqualities,
    /// Node types from type checking, keyed by NodeId.
    /// Used to get the effect type of lambda expressions.
    node_types: SortedMap<NodeId, crate::ast::Type<'db>>,
}

/// The functions that compare a scrutinee with a `String` or `Bytes` literal
/// pattern, by their qualified names.
#[derive(Clone, Debug, Default)]
pub(crate) struct LiteralEqualities {
    pub string: Option<SymbolPath>,
    pub bytes: Option<SymbolPath>,
}

impl<'db> IrLoweringCtx<'db> {
    pub(crate) fn db(&self) -> &'db dyn salsa::Database {
        self.db
    }

    /// Create a new IR lowering context.
    pub fn new(
        db: &'db dyn salsa::Database,
        path: PathRef,
        span_map: SpanMaps<'db>,
        function_types: HashMap<SymbolPath, TypeScheme<'db>>,
        node_types: SortedMap<NodeId, crate::ast::Type<'db>>,
    ) -> Self {
        Self {
            db,
            path,
            source_paths: HashMap::default(),
            span_map,
            scopes: vec![HashMap::default()],
            local_callable_values: vec![HashMap::default()],
            resume_scopes: vec![HashSet::default()],
            function_types,
            nominal: NominalTypes::new(db),
            logical_generated_signatures: HashMap::default(),
            logical_source_functions: HashSet::default(),
            function_ref_paths: HashMap::default(),
            logical_emitted_externs: HashSet::default(),
            module_path: SymbolPath::root(),
            module_block: None,
            compiler_intrinsics: HashMap::default(),
            runtime_types: Default::default(),
            literal_equalities: LiteralEqualities::default(),

            node_types,
        }
    }

    pub(crate) fn with_compiler_intrinsics(
        mut self,
        compiler_intrinsics: HashMap<NodeId, Symbol>,
    ) -> Self {
        self.compiler_intrinsics = compiler_intrinsics;
        self
    }

    pub(crate) fn with_runtime_types(
        mut self,
        runtime_types: super::runtime_items::RuntimeTypes,
    ) -> Self {
        self.runtime_types = runtime_types;
        self
    }

    /// Read types as the definition `name` sees them, and return the scope
    /// to restore with [`Self::leave_definition`].
    pub(crate) fn enter_definition(&mut self, name: &[Symbol]) -> DefinitionScope {
        let binders = self.row_binders_of(name);
        DefinitionScope {
            binders: self.nominal.replace_current_row_binders(binders),
        }
    }

    pub(crate) fn leave_definition(&mut self, outer: DefinitionScope) {
        self.nominal.replace_current_row_binders(outer.binders);
    }

    /// Record the row binders of the function definition `name`.
    pub(crate) fn register_row_binders(&mut self, name: SymbolPath, binders: RowBinders) {
        self.nominal.register_row_binders(name, binders)
    }

    /// The row binders of the function `name`; those its scheme quantifies
    /// when it has no definition in the module.
    pub(crate) fn row_binders_of(&self, name: &[Symbol]) -> RowBinders {
        self.nominal.declared_row_binders(name).unwrap_or_else(|| {
            self.lookup_function_type(name)
                .map(|scheme| super::row_binders::of_scheme(self.db, *scheme))
                .unwrap_or_default()
        })
    }

    /// The row binders of the definition whose types are being read.
    pub(crate) fn current_row_binders(&self) -> &RowBinders {
        self.nominal.current_row_binders()
    }

    /// The IR row of `row` in the current definition.
    ///
    /// # Panics
    ///
    /// If the row's tail is not a binder of the current definition.
    pub(crate) fn ability_row(&self, row: crate::ast::AbilityRow<'db>) -> Row {
        self.nominal.ability_row(row)
    }

    /// The tail unions `scheme` declares, numbered by the current binders.
    /// Only a union of several open sources selects evidence by tail.
    pub(crate) fn row_unions(&self, scheme: TypeScheme<'db>) -> Vec<RowUnion> {
        self.nominal.row_unions(scheme)
    }

    pub(crate) fn with_source_paths(mut self, source_paths: HashMap<u64, PathRef>) -> Self {
        self.source_paths = source_paths;
        self
    }

    pub(crate) fn compiler_intrinsic(&self, declaration: NodeId) -> Option<Symbol> {
        self.compiler_intrinsics.get(&declaration).cloned()
    }

    /// The canonical path a runtime struct declaration is published under.
    pub(crate) fn runtime_type_path(&self, declaration: NodeId) -> Option<&SymbolPath> {
        self.runtime_types.0.get(&declaration)
    }

    pub(crate) fn with_literal_equalities(mut self, literal_equalities: LiteralEqualities) -> Self {
        self.literal_equalities = literal_equalities;
        self
    }

    pub(crate) fn literal_equalities(&self) -> LiteralEqualities {
        self.literal_equalities.clone()
    }

    /// Get the current module path.
    pub fn module_path(&self) -> &SymbolPath {
        &self.module_path
    }

    /// Lower the declarations of `module` next.
    pub fn set_module(&mut self, module: SymbolPath) {
        self.module_path = module;
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
        let path = self
            .source_paths
            .get(&node_id.source())
            .copied()
            .unwrap_or(self.path);
        Location::new(path, span)
    }

    /// Enter a new scope, returning a guard that exits on drop.
    ///
    /// The guard dereferences to `IrLoweringCtx`, so callers can use it
    /// in place of `self`/`ctx`. The scope is automatically exited when
    /// the guard is dropped, even on early returns or `?`.
    #[must_use]
    pub fn scope(&mut self) -> ScopeGuard<'_, 'db> {
        self.enter_scope();
        ScopeGuard { ctx: self }
    }

    /// Enter a new scope (internal — use `scope()` guard instead).
    fn enter_scope(&mut self) {
        self.scopes.push(HashMap::default());
        self.local_callable_values.push(HashMap::default());
        self.resume_scopes.push(HashSet::default());
    }

    /// Exit the current scope (internal — use `scope()` guard instead).
    fn exit_scope(&mut self) {
        self.scopes.pop();
        self.local_callable_values.pop();
        self.resume_scopes.pop();
    }

    /// Bind a local variable to an SSA value.
    pub fn bind(&mut self, local_id: LocalId, name: Symbol, value: ValueRef) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(local_id, (name, value));
        }
    }

    pub(crate) fn bind_local_callable(
        &mut self,
        key: (NodeId, crate::ast::Type<'db>, TypeRef),
        value: ValueRef,
    ) {
        self.local_callable_values
            .last_mut()
            .expect("active scope")
            .insert(key, value);
    }

    pub(crate) fn lookup_local_callable(
        &self,
        key: (NodeId, crate::ast::Type<'db>, TypeRef),
    ) -> Option<ValueRef> {
        self.local_callable_values
            .iter()
            .rev()
            .find_map(|scope| scope.get(&key).copied())
    }

    pub(crate) fn bind_resume(&mut self, local_id: LocalId, name: Symbol, value: ValueRef) {
        self.bind(local_id, name, value);
        if let Some(scope) = self.resume_scopes.last_mut() {
            scope.insert(local_id);
        }
    }

    pub(crate) fn lookup_resume(&self, local_id: LocalId) -> Option<ValueRef> {
        self.resume_scopes
            .iter()
            .rev()
            .any(|scope| scope.contains(&local_id))
            .then(|| self.lookup(local_id))
            .flatten()
    }

    /// Look up a function's type scheme by name.
    pub fn lookup_function_type(&self, name: &[Symbol]) -> Option<&TypeScheme<'db>> {
        self.function_types.get(name)
    }

    pub(crate) fn register_logical_generated_signature(
        &mut self,
        name: &SymbolPath,
        signature: LogicalGeneratedSignature,
    ) {
        if let Some(existing) = self
            .logical_generated_signatures
            .insert(name.clone(), signature.clone())
        {
            assert!(
                existing.param_types == signature.param_types
                    && existing.return_type == signature.return_type
                    && existing.row == signature.row
                    && existing.binders == signature.binders,
                "conflicting logical generated signature for {name}"
            );
        }
    }

    pub(crate) fn lookup_logical_generated_signature(
        &self,
        name: &[Symbol],
    ) -> Option<&LogicalGeneratedSignature> {
        self.logical_generated_signatures.get(name)
    }

    pub(crate) fn register_logical_source_function(&mut self, name: SymbolPath) {
        self.logical_source_functions.insert(name);
    }

    /// The IR symbol of the function at `name`: its path joined with `::`.
    /// An `extern "C"` function links under its declared name instead, which
    /// its `link_name` carries.
    pub(crate) fn function_symbol(&self, name: &[Symbol]) -> Symbol {
        SymbolPath::new(name.iter().cloned()).to_symbol()
    }

    /// Record that the `func_ref` operation `op` names the function `path`.
    pub(crate) fn record_function_ref(&mut self, op: OpRef, path: SymbolPath) {
        self.function_ref_paths.insert(op, path);
    }

    /// The function the `func_ref` operation `op` names.
    pub(crate) fn function_ref_path(&self, op: OpRef) -> Option<&SymbolPath> {
        self.function_ref_paths.get(&op)
    }

    pub(crate) fn is_logical_source_function(&self, name: &[Symbol]) -> bool {
        self.logical_source_functions.contains(name)
    }

    /// Returns true exactly once for each prelude declaration that must be
    /// materialized in the logical module.
    pub(crate) fn mark_logical_extern_emitted(&mut self, name: SymbolPath) -> bool {
        self.logical_emitted_externs.insert(name)
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

    /// The path of `name` in the current module: the declaration's entry
    /// module, which holds only inline modules.
    ///
    /// - `["String"]` + `"to_bytes"` → `["String", "to_bytes"]`
    /// - `[]` + `"print_line"` → `["print_line"]`
    pub fn qualify_name(&self, name: &Symbol) -> SymbolPath {
        self.module_path.child(name.clone())
    }

    /// Register struct field order for lowering Record expressions.
    pub fn register_struct_fields(&mut self, ctor_id: CtorId<'db>, field_names: Vec<Symbol>) {
        self.nominal.register_struct_fields(ctor_id, field_names)
    }

    /// Register the field names of a named-field enum variant.
    pub fn register_variant_field_names(
        &mut self,
        layout: TypeRef,
        tag: Symbol,
        field_names: Vec<Symbol>,
    ) {
        self.nominal
            .register_variant_field_names(layout, tag, field_names)
    }

    /// Field names of a named-field enum variant in declaration order.
    pub fn variant_field_names(&self, layout: TypeRef, tag: Symbol) -> Option<Vec<Symbol>> {
        self.nominal.variant_field_names(layout, tag)
    }

    /// Get struct field order (for lowering Record → adt.struct_new).
    pub fn get_struct_field_order(&self, ctor_id: CtorId<'db>) -> Option<&Vec<Symbol>> {
        self.nominal.get_struct_field_order(ctor_id)
    }

    /// Register a type (named struct, tuple, etc.) in the type map.
    pub fn register_type(&mut self, name: Symbol, ty: TypeRef) {
        self.nominal.register_type(name, ty)
    }

    /// Get a registered type by name.
    pub fn get_type(&self, name: &Symbol) -> Option<TypeRef> {
        self.nominal.get_type(name)
    }

    /// Mark a nominal source type as known before its layout is complete.
    pub fn declare_logical_nominal(&mut self, name: Symbol) {
        self.nominal.declare_logical_nominal(name)
    }

    /// Get the type of an AST node by NodeId.
    ///
    /// Returns the type assigned during type checking.
    /// Used to get the effect type of lambda expressions.
    pub fn resolve_adt_type(&self, ty: crate::ast::Type<'db>) -> Option<TypeRef> {
        self.nominal.resolve_adt_type(ty)
    }

    /// The type of every node, from type checking.
    pub(crate) fn node_types(&self) -> &SortedMap<NodeId, crate::ast::Type<'db>> {
        &self.node_types
    }

    pub fn get_node_type(&self, node_id: NodeId) -> Option<&crate::ast::Type<'db>> {
        self.node_types.get(&node_id)
    }

    /// Get all bindings visible in the current scope (for capture analysis).
    ///
    /// The stable local-ID order keeps generated closure capture lists and IR
    /// snapshots deterministic; each local ID is unique even across scopes.
    pub fn all_bindings(&self) -> impl Iterator<Item = (LocalId, Symbol, ValueRef)> + '_ {
        let mut bindings: Vec<_> = self
            .scopes
            .iter()
            .rev()
            .flat_map(|scope| {
                scope
                    .iter()
                    .map(|(&id, (name, value))| (id, name.clone(), *value))
            })
            .collect();
        bindings.sort_unstable_by_key(|(local_id, _, _)| local_id.raw());
        bindings.into_iter()
    }

    // =========================================================================
    // Arena type conversion
    // =========================================================================

    /// Convert a source type at the source-logical `tribute_control` boundary.
    pub fn convert_logical_type(&self, ir: &mut IrContext, ty: crate::ast::Type<'db>) -> TypeRef {
        self.nominal.convert_logical_type(ir, ty)
    }

    // =========================================================================
    // Arena type helpers
    // =========================================================================

    /// Get the `core.i32` type.
    pub fn i32_type(&self, ir: &mut IrContext) -> TypeRef {
        self.nominal.i32_type(ir)
    }

    /// Get the `core.nil` type.
    pub fn nil_type(&self, ir: &mut IrContext) -> TypeRef {
        self.nominal.nil_type(ir)
    }

    /// Get the `core.i1` (bool) type.
    pub fn bool_type(&self, ir: &mut IrContext) -> TypeRef {
        self.nominal.bool_type(ir)
    }

    /// Get the `core.f64` type.
    pub fn f64_type(&self, ir: &mut IrContext) -> TypeRef {
        self.nominal.f64_type(ir)
    }

    /// Get the `core.bytes` type.
    pub fn bytes_type(&self, ir: &mut IrContext) -> TypeRef {
        self.nominal.bytes_type(ir)
    }

    /// Get the `tribute_rt.anyref` type.
    pub fn anyref_type(&self, ir: &mut IrContext) -> TypeRef {
        self.nominal.anyref_type(ir)
    }

    /// The symbol of the `ability.decl` of one ability instance.
    ///
    /// It is the instance key, derived from the checked source arguments, so
    /// arguments sharing a lowered representation stay distinct instances.
    pub fn ability_symbol(
        &self,
        ability_name: &Symbol,
        arguments: &[crate::ast::Type<'db>],
    ) -> SymbolPath {
        self.nominal.ability_symbol(ability_name, arguments)
    }

    /// Create an `adt.struct` type with name and fields.
    pub fn adt_struct_type(
        &self,
        ir: &mut IrContext,
        name: &Symbol,
        fields: &[(Symbol, TypeRef)],
    ) -> TypeRef {
        self.nominal.adt_struct_type(ir, name, fields)
    }

    /// Create an `adt.enum` type with name and variants.
    pub fn adt_enum_type(
        &self,
        ir: &mut IrContext,
        name: &Symbol,
        variants: &[(Symbol, Vec<TypeRef>)],
    ) -> TypeRef {
        self.nominal.adt_enum_type(ir, name, variants)
    }

    /// Create an `adt.enum` type that records where it was declared.
    pub fn adt_enum_type_with_definition(
        &self,
        ir: &mut IrContext,
        name: &Symbol,
        variants: &[(Symbol, Vec<TypeRef>)],
        definition: Location,
    ) -> TypeRef {
        self.nominal
            .adt_enum_type_with_definition(ir, name, variants, definition)
    }

    /// Create an `adt.typeref` type — a reference to a named type.
    pub fn adt_typeref(&self, ir: &mut IrContext, name: &Symbol) -> TypeRef {
        self.nominal.adt_typeref(ir, name)
    }
}

/// RAII guard that exits a scope on drop.
///
/// Created by [`IrLoweringCtx::scope()`]. Dereferences to `IrLoweringCtx`
/// so it can be used as a drop-in replacement for `&mut ctx`.
pub struct ScopeGuard<'a, 'db> {
    ctx: &'a mut IrLoweringCtx<'db>,
}

impl<'db> Deref for ScopeGuard<'_, 'db> {
    type Target = IrLoweringCtx<'db>;
    fn deref(&self) -> &Self::Target {
        self.ctx
    }
}

impl<'db> DerefMut for ScopeGuard<'_, 'db> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.ctx
    }
}

impl Drop for ScopeGuard<'_, '_> {
    fn drop(&mut self) {
        self.ctx.exit_scope();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Type as AstType, TypeKind};
    use trunk_ir::dialect::core;
    use trunk_ir::ops::DialectType;

    fn test_db() -> salsa::DatabaseImpl {
        salsa::DatabaseImpl::new()
    }

    #[test]
    fn test_convert_logical_bound_var_to_any() {
        let db = test_db();
        let mut ir = IrContext::new();
        let path = ir.intern_path("test.trb");
        let ctx = IrLoweringCtx::new(
            &db,
            path,
            crate::ast::SpanMaps::default(),
            HashMap::default(),
            SortedMap::default(),
        );

        let ty = AstType::new(&db, TypeKind::BoundVar { index: 0 });
        let ir_ty = ctx.convert_logical_type(&mut ir, ty);
        let expected = ctx.anyref_type(&mut ir);
        assert_eq!(ir_ty, expected);
    }

    /// The logical frontend boundary has its own recursive conversion: source
    /// callables, bottom values, resumptions, tuple layouts, and forward
    /// nominals must never fall back to the physical `func.func_sig` carrier.
    #[test]
    fn test_convert_logical_types_preserves_recursive_control_shapes() {
        let db = test_db();
        let mut ir = IrContext::new();
        let path = ir.intern_path("test.trb");
        let mut ctx = IrLoweringCtx::new(
            &db,
            path,
            crate::ast::SpanMaps::default(),
            HashMap::default(),
            SortedMap::default(),
        );
        let int = AstType::new(&db, TypeKind::Int);
        let effect = crate::ast::AbilityRow::pure(&db);
        let callable = AstType::new(
            &db,
            TypeKind::Func {
                params: vec![int],
                result: int,
                effect,
            },
        );
        let logical_callable = ctx.convert_logical_type(&mut ir, callable);
        let i32_ty = ctx.i32_type(&mut ir);
        let signature =
            tribute_ir::dialect::tribute_control::FuncSig::from_type_ref(&ir, logical_callable)
                .expect("logical function type must be a control callable");
        assert_eq!(signature.result(&ir), i32_ty);
        assert_eq!(signature.inputs(&ir), &[i32_ty]);

        let never = AstType::new(&db, TypeKind::Never);
        assert_eq!(
            ctx.convert_logical_type(&mut ir, never),
            core::Never::params().intern(&mut ir).as_type_ref()
        );

        let resume = AstType::new(
            &db,
            TypeKind::Continuation {
                arg: int,
                result: int,
                effect,
            },
        );
        let resume = ctx.convert_logical_type(&mut ir, resume);
        let resume_data = ir.get_type(resume);
        assert!(resume_data.name.is(&ir, "tribute_control.resume_token"));

        let tuple = AstType::new(&db, TypeKind::Tuple(vec![callable, never]));
        let tuple_ir = ctx.convert_logical_type(&mut ir, tuple);
        let callable_ir = ctx.convert_logical_type(&mut ir, callable);
        let never_ir = ctx.convert_logical_type(&mut ir, never);
        assert_eq!(
            tribute_ir::dialect::tribute_control::logical_tuple_elements(&ir, tuple_ir),
            Some(&[callable_ir, never_ir][..])
        );

        let nominal_name = Symbol::new("Nested::Forward");
        ctx.declare_logical_nominal(nominal_name.clone());
        let forward = AstType::new(
            &db,
            TypeKind::Named {
                id: crate::ast::TypeDefId::synthetic(
                    &db,
                    trunk_ir::SymbolPath::new([nominal_name.clone()]),
                ),
                name: Symbol::new("Forward"),
                args: vec![],
            },
        );
        assert_eq!(
            ctx.convert_logical_type(&mut ir, forward),
            ctx.adt_typeref(&mut ir, &nominal_name)
        );

        // A source declaration named List must not capture the builtin type,
        // including when it occurs recursively inside a callable or tuple.
        let list_name = Symbol::new("List");
        ctx.declare_logical_nominal(list_name.clone());
        let source_list = AstType::new(
            &db,
            TypeKind::Named {
                id: crate::ast::TypeDefId::synthetic(
                    &db,
                    trunk_ir::SymbolPath::new([list_name.clone()]),
                ),
                name: list_name.clone(),
                args: vec![],
            },
        );
        let builtin_list = AstType::new(
            &db,
            TypeKind::Named {
                id: crate::ast::TypeDefId::builtin_list(&db),
                name: list_name.clone(),
                args: vec![int],
            },
        );
        let list_callable = AstType::new(
            &db,
            TypeKind::Func {
                params: vec![builtin_list, source_list],
                result: builtin_list,
                effect,
            },
        );
        let callable_ir = ctx.convert_logical_type(&mut ir, list_callable);
        let list_signature =
            tribute_ir::dialect::tribute_control::FuncSig::from_type_ref(&ir, callable_ir).unwrap();
        let anyref = ctx.anyref_type(&mut ir);
        let nominal_list = ctx.adt_typeref(&mut ir, &list_name);
        assert_eq!(list_signature.inputs(&ir), &[anyref, nominal_list]);
        assert_eq!(list_signature.result(&ir), anyref);
        let tuple = AstType::new(&db, TypeKind::Tuple(vec![builtin_list, source_list]));
        let tuple_ir = ctx.convert_logical_type(&mut ir, tuple);
        assert_eq!(
            tribute_ir::dialect::tribute_control::logical_tuple_elements(&ir, tuple_ir),
            Some(&[anyref, nominal_list][..])
        );

        let generated = SymbolPath::new(["Forward", "value"]);
        ctx.register_logical_generated_signature(
            &generated,
            LogicalGeneratedSignature {
                param_types: vec![i32_ty],
                return_type: i32_ty,
                row: Row::default(),
                binders: vec![],
            },
        );
        // Re-registering the exact semantic signature is a deterministic
        // no-op; a conflicting signature remains fail-closed in the lowering
        // context rather than being silently replaced.
        ctx.register_logical_generated_signature(
            &generated,
            LogicalGeneratedSignature {
                param_types: vec![i32_ty],
                return_type: i32_ty,
                row: Row::default(),
                binders: vec![],
            },
        );
        let generated_signature = ctx
            .lookup_logical_generated_signature(&generated)
            .expect("generated logical signature must be retained");
        assert_eq!(generated_signature.param_types, vec![i32_ty]);
        assert_eq!(generated_signature.return_type, i32_ty);

        let source_function = SymbolPath::new(["Forward", "run"]);
        ctx.register_logical_source_function(source_function.clone());
        assert!(ctx.is_logical_source_function(&source_function));
        assert!(ctx.mark_logical_extern_emitted(SymbolPath::new(["prelude", "id"])));
        assert!(!ctx.mark_logical_extern_emitted(SymbolPath::new(["prelude", "id"])));

        ctx.set_module(SymbolPath::new(["Nested"]));
        assert_eq!(ctx.qualify_name(&Symbol::new("run")), ["Nested", "run"]);
        ctx.set_module(SymbolPath::root());

        let erased_cases = [
            AstType::new(&db, TypeKind::BoundVar { index: 0 }),
            AstType::new(
                &db,
                TypeKind::UniVar {
                    id: crate::ast::UniVarId::new(&db, crate::ast::UniVarSource::Anonymous(0), 0),
                },
            ),
            AstType::new(
                &db,
                TypeKind::Named {
                    id: crate::ast::TypeDefId::builtin_list(&db),
                    name: Symbol::new("Undeclared"),
                    args: vec![],
                },
            ),
        ];
        let anyref = ctx.anyref_type(&mut ir);
        for source_ty in erased_cases {
            assert_eq!(ctx.convert_logical_type(&mut ir, source_ty), anyref);
        }
        assert_eq!(
            ctx.convert_logical_type(&mut ir, AstType::new(&db, TypeKind::Error)),
            ctx.nil_type(&mut ir)
        );
    }

    #[test]
    fn test_convert_logical_primitives() {
        let db = test_db();
        let mut ir = IrContext::new();
        let path = ir.intern_path("test.trb");
        let ctx = IrLoweringCtx::new(
            &db,
            path,
            crate::ast::SpanMaps::default(),
            HashMap::default(),
            SortedMap::default(),
        );

        // Int → I32
        let int_ty = AstType::new(&db, TypeKind::Int);
        assert_eq!(
            ctx.convert_logical_type(&mut ir, int_ty),
            ctx.i32_type(&mut ir)
        );

        // Bool → I1
        let bool_ty = AstType::new(&db, TypeKind::Bool);
        assert_eq!(
            ctx.convert_logical_type(&mut ir, bool_ty),
            ctx.bool_type(&mut ir)
        );

        // Float → F64
        let float_ty = AstType::new(&db, TypeKind::Float);
        assert_eq!(
            ctx.convert_logical_type(&mut ir, float_ty),
            ctx.f64_type(&mut ir)
        );

        // Nil → Nil
        let nil_ty = AstType::new(&db, TypeKind::Nil);
        assert_eq!(
            ctx.convert_logical_type(&mut ir, nil_ty),
            ctx.nil_type(&mut ir)
        );
    }

    #[test]
    fn test_lookup_function_type() {
        let db = test_db();
        let mut ir = IrContext::new();
        let path = ir.intern_path("test.trb");
        let name = SymbolPath::new(["foo"]);
        let body = AstType::new(&db, TypeKind::Int);
        let scheme = TypeScheme::new(&db, vec![], vec![], body);

        let mut ft = HashMap::default();
        ft.insert(name.clone(), scheme);

        let ctx = IrLoweringCtx::new(
            &db,
            path,
            crate::ast::SpanMaps::default(),
            ft,
            SortedMap::default(),
        );
        assert_eq!(ctx.lookup_function_type(&name), Some(&scheme));
        assert_eq!(ctx.lookup_function_type(&[Symbol::new("missing")]), None);
    }
}
